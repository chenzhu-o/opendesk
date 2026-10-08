"""Training-side handoff — turn exported trajectories into dataset rows.

:mod:`opendesk.learning.trajectories` and :mod:`opendesk.learning.preference`
produce the environment half of RL: reward-labeled episodes and chosen/rejected
pairs.  This module is the handoff to a trainer — TRL, verl, OpenRLHF, or a
bespoke loop.  It reads those artefacts back, renders each episode as the text
a policy would have emitted, and writes rows in the shape those trainers want.

The rule that matters
---------------------
**A completion holds the agent's actions and nothing else.**

Everything else a trajectory carries — per-step reward, discounted return,
effect, whether the attempt succeeded — is a *label*.  Labels belong in the row's
metadata, where a loss can use them for credit assignment.  They must not go
into the token stream, because a model trained on them learns to emit them: put
``reward="1"`` at the end of every successful completion and DPO will discover
that the cheapest way to raise its log-probability difference is to print a
reward, not to solve the task.  Same for per-step annotations: they are the
grader's answer key, and a policy that can read the answer key learns the key.

So :func:`render_completion` emits action calls only, and
:func:`describe_trajectory` — which *does* include the labels — is for humans
reading a log, never for training.

What a completion is not
------------------------
The audit log records structured actions, not the policy's tokens: there is no
``completion`` field to recover.  What this module renders is a *canonical
serialization* of what the agent did, so rows carry
``"completion_source": "opendesk.canonical/1"`` to say so.  Training on it
teaches the format as much as the behaviour.  If you logged your model's raw
output alongside the episode, pass a ``resolver`` to :func:`build_dataset` and
the row will carry your text with ``"completion_source": "caller"`` instead.

Example::

    from opendesk.learning import training

    episodes = training.load_trajectories("runs/")
    pairs = training.load_pairs("runs/preferences.jsonl")
    job = training.build_dataset(episodes, pairs=pairs, format="dpo")
    training.export_jsonl(job.rows, "runs/dpo.jsonl")
    print(job.summary_text())
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

SCHEMA_VERSION = "opendesk.dataset/1"
#: Identifies how a completion was produced.  ``opendesk.canonical/1`` means it
#: was re-rendered from the action log — see "What a completion is not" above.
RENDER_VERSION = "opendesk.canonical/1"

#: How far an episode's dense signal can be trusted, best first.  ``ui`` is read
#: off the accessibility tree; ``screen`` groups states by layout and so cannot
#: see a content edit inside unchanged layout; ``action`` infers effect from the
#: action itself rather than observing it.  ``unknown`` is a record that predates
#: the field.  This mirrors ``process.signal_source``.
TRUST_ORDER = {"ui": 3, "screen": 2, "action": 1, "unknown": 0}

#: Default trust floor.  ``None`` means *do not filter* — every row carries its
#: ``trust`` and the stats tally them, so a weak signal is visible instead of
#: silently excluded or silently included.  Trust only describes the quality of
#: the *step-level* signal; an SFT row's label is the outcome reward and a DPO
#: pair's label comes from the verifiable reward, so neither needs it.  A GRPO
#: caller should set ``min_trust="screen"``, because there the per-step returns
#: *are* the training signal.
DEFAULT_MIN_TRUST: Optional[str] = None

#: Trust below this is reported as a weak signal even when it is kept.
WEAK_BELOW = "screen"


# ---------------------------------------------------------------------------
# Reading what the producers wrote
# ---------------------------------------------------------------------------

def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{lineno} is not valid JSON ({exc.msg}). A trajectory "
                    f"is written one JSON object per line; a truncated final line "
                    f"usually means the export was interrupted."
                ) from exc


def _collect_paths(source: Any) -> list[str]:
    """Accept a file, a directory, or an explicit list of either."""
    items = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    out: list[str] = []
    for item in items:
        p = os.path.expanduser(str(item))
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".jsonl"):
                    out.append(os.path.join(p, name))
        else:
            out.append(p)
    return out


def load_trajectories(source: Any) -> list[dict[str, Any]]:
    """Read trajectory JSONL into whole-episode records.

    *source* may be a ``.jsonl`` file, a directory of them, or a list of either.
    A file may hold several episodes appended one after another, which is how a
    run directory accumulates: each ``type: "episode"`` line opens a new record
    and the ``type: "step"`` lines after it belong to it.

    Embedded base64 screenshots are dropped.  A single inlined PNG is tens of
    thousands of tokens and several steps of them will silently exceed any
    context window — a failure that shows up as an OOM deep in a training run
    rather than as an error here.
    """
    records: list[dict[str, Any]] = []
    for path in _collect_paths(source):
        current: Optional[dict[str, Any]] = None
        for obj in _iter_jsonl(path):
            kind = obj.get("type")
            if kind == "episode":
                current = {**obj, "steps": [], "path": path}
                records.append(current)
            elif kind == "step":
                if current is None:
                    # A step with no header: treat the file as one episode and
                    # synthesise the header, rather than dropping the data.
                    current = {
                        "schema": None, "type": "episode", "steps": [],
                        "path": path, "episode_id": None,
                    }
                    records.append(current)
                current["steps"].append(obj)
            # Anything else is a record kind this module does not know about;
            # skipping is correct, guessing is not.
    return records


def load_pairs(path: Any) -> list[dict[str, Any]]:
    """Read a preference-pair JSONL written by ``preference.export_pairs``."""
    out: list[dict[str, Any]] = []
    for p in _collect_paths(path):
        for obj in _iter_jsonl(p):
            if obj.get("type") == "preference" or "chosen" in obj:
                out.append(obj)
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    """One deterministic literal.

    Strings are quoted so that ``type(text="exit")`` cannot be confused with
    ``type(text=exit)``, and mappings are key-sorted so the same params always
    render byte-identically — otherwise two rollouts that did the same thing
    could produce different completions, and a preference pair built from them
    would carry noise instead of signal.
    """
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_action(action: Any, *, omit_none: bool = True) -> str:
    """Render one audit action as a single call expression.

    ``{"type": "click", "params": {"name": "Export", "role": "button"}}``
    becomes ``click(name="Export", role="button")``.

    *omit_none* drops params recorded as ``None``.  The audit log stores the
    tool's *parsed* params, which is its whole signature — so an action that
    set one field arrives with every other field present and null.  Rendering
    them costs most of the transcript and teaches a model to recite
    ``argv=null, command=null, timeout=null``, none of which is part of the
    task.  A param whose value merely repeats the action's own name is dropped
    too; a *different* one (``system``'s ``file_write(action="write_file")``) is
    kept, because there the sub-action is something the agent chose.
    """
    if isinstance(action, dict):
        name = action.get("type") or action.get("action") or ""
        params = action.get("params") or {}
    else:
        name = getattr(action, "action_type", "")
        name = getattr(name, "value", name)
        params = getattr(action, "params", None) or {}
    if not isinstance(params, dict):
        params = {"value": params}
    args: list[str] = []
    for key in sorted(params):
        value = params[key]
        if omit_none and value is None:
            continue
        if key == "action" and value == name:
            continue  # already the call's name
        args.append(f"{key}={_fmt(value)}")
    return f"{name}({', '.join(args)})" if args else f"{name}()"


def render_completion(episode: dict[str, Any]) -> str:
    """The action sequence, one call per line.  No labels — see module docstring."""
    return "\n".join(render_action(s.get("action")) for s in episode.get("steps") or [])


def render_prompt(episode: dict[str, Any]) -> str:
    """The task text shared by every attempt at it.

    opendesk stores no screen content, so this is the task and goal only.  The
    per-step observation the policy actually conditioned on lives in the
    caller's harness — fold it in there if your trainer needs it.
    """
    lines = [f"Task: {episode.get('task') or 'unnamed task'}"]
    if episode.get("goal"):
        lines.append(f"Goal: {episode['goal']}")
    return "\n".join(lines)


def describe_trajectory(episode: dict[str, Any]) -> str:
    """A human-readable episode dump — **including labels**.

    This is for reading a log or debugging a reward.  Do not train on its
    output: it prints the per-step reward and the outcome, which is the answer
    key.
    """
    outcome = episode.get("outcome") or {}
    metrics = ((episode.get("process") or {}).get("metrics") or {})
    out = [
        f"episode {episode.get('episode_id')}  task={episode.get('task')!r}",
        f"  reward={outcome.get('reward')}  steps={len(episode.get('steps') or [])}"
        f"  efficiency={metrics.get('efficiency')}"
        f"  signal_source={trust_of(episode)}",
    ]
    for step in episode.get("steps") or []:
        mark = {"changed": "->", "same": "==", "none": ".."}.get(
            str(step.get("effect")), "??"
        )
        out.append(
            f"  [{step.get('step'):>3}] {mark} {render_action(step.get('action'))}"
            f"   reward={step.get('reward')} return={step.get('return')}"
        )
    return "\n".join(out)


def trust_of(trajectory: dict[str, Any]) -> str:
    """Which signal the step effects were read from (see :data:`TRUST_ORDER`)."""
    metrics = ((trajectory.get("process") or {}).get("metrics") or {})
    src = metrics.get("signal_source")
    return src if src in TRUST_ORDER else "unknown"


def _returns(episode: dict[str, Any]) -> list[Optional[float]]:
    return [s.get("return") for s in episode.get("steps") or []]


def _steps_of(episode: dict[str, Any]) -> int:
    return len(episode.get("steps") or [])


def _outcome_reward(episode: dict[str, Any]) -> float:
    outcome = episode.get("outcome") or {}
    try:
        return float(outcome.get("reward") or 0.0)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """Rows plus the accounting that says what was dropped and why."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    format: str = "sft"
    stats: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "format": self.format,
            "stats": self.stats,
            "rows": self.rows,
        }

    def summary_text(self) -> str:
        s = self.stats
        lines = [
            f"{s.get('rows', 0)} {self.format} row(s) from "
            f"{s.get('episodes', 0)} episode(s)"
        ]
        if s.get("pairs") is not None:
            lines.append(f"  preference pairs read: {s.get('pairs')}")
        lines.append(
            f"  tasks {s.get('tasks', 0)}  groups {s.get('groups', 0)}"
            f"  successes {s.get('successes', 0)}"
        )
        dropped = [
            (n, s.get(key)) for n, key in (
                ("identical chosen/rejected", "dropped_identical"),
                ("below trust floor", "dropped_low_trust"),
                ("no verifiable success", "dropped_unsuccessful"),
                ("pair without trajectory", "dropped_unmatched"),
                ("pair not sharing a prompt", "dropped_prompt_mismatch"),
            ) if s.get(key)
        ]
        if dropped:
            lines.append("  dropped: " + ", ".join(f"{v} {n}" for n, v in dropped))
        by_trust = s.get("by_trust") or {}
        if by_trust:
            lines.append(
                "  effect signal: "
                + ", ".join(f"{k}={v}" for k, v in sorted(by_trust.items()))
            )
        weak = s.get("weak_signal")
        if weak and s.get("min_trust") is None:
            # Kept, not dropped — but say so.  A GRPO run whose gradient comes
            # from inferred effects is a different experiment from one whose
            # effects were observed, and it should not look the same.
            lines.append(
                f"  note: {weak} row(s) have an inferred or unrecorded effect "
                f"signal; set min_trust='screen' to require observed effects."
            )
        flat = s.get("groups_no_spread")
        if flat:
            lines.append(
                f"  note: {flat} group(s) have no reward spread, so their "
                f"group-relative advantages are all zero. Keep them only if "
                f"your objective uses the reward directly."
            )
        return "\n".join(lines)


def _bucket(key: str, seed: int) -> int:
    """Stable bucket in [0, 1000) — hashing, so it does not depend on order."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1000


def split_rows(
    rows: Sequence[dict[str, Any]],
    *,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 0,
    key: str = "task",
) -> dict[str, list[dict[str, Any]]]:
    """Split **by task**, not by row.

    Attempts at one task are near-duplicates: same prompt, overlapping actions.
    Splitting them across train and eval leaks the evaluation set into training
    and reports a number that means nothing.  Grouping by the task (or by
    ``group_id`` for GRPO rows) keeps every attempt of a task on one side.
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio > 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to <= 1")
    val_cut = int(round(val_ratio * 1000))
    test_cut = val_cut + int(round(test_ratio * 1000))

    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        group = str(row.get("group_id") or row.get(key) or "")
        b = _bucket(group, seed)
        # Ordered so that a zero ratio produces an empty band rather than
        # swallowing everything: with val_ratio=test_ratio=0 both cuts are 0 and
        # every row is >= them, which is train.
        where = "val" if b < val_cut else "test" if b < test_cut else "train"
        out[where].append(row)
    return {k: v for k, v in out.items() if v}


def build_dataset(
    trajectories: Sequence[dict[str, Any]],
    *,
    pairs: Optional[Sequence[dict[str, Any]]] = None,
    format: str = "sft",
    min_reward: float = 1.0,
    min_trust: str = DEFAULT_MIN_TRUST,
    drop_degenerate: bool = True,
    resolver: Optional[Callable[[dict[str, Any]], Optional[str]]] = None,
    prompt_resolver: Optional[Callable[[dict[str, Any]], Optional[str]]] = None,
    tree_resolver: Optional[Callable[[dict[str, Any]], Optional[str]]] = None,
    canonical_actions: bool = False,
    sanitize: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
) -> Dataset:
    """Assemble training rows from trajectories (and pairs, for ``dpo``).

    Parameters
    ----------
    trajectories:
        Episode records from :func:`load_trajectories`.
    pairs:
        Preference pairs.  Required for ``format="dpo"``.
    format:
        ``"sft"`` — prompt → completion for attempts that satisfied the checks.
        ``"dpo"`` — prompt → chosen/rejected, joined from the pairs.
        ``"grpo"`` — one row per task holding every attempt, its verifiable
        reward, and its per-step returns.  This is the on-policy shape: the
        trainer samples from the prompt and re-verifies through opendesk, so the
        reward stays outside the model path.
    min_reward:
        For ``sft``: an attempt qualifies at or above this outcome reward.
    min_trust:
        Optional floor on the signal the step effects were read from.  Default
        is *no floor*: rows keep a ``trust`` field and the manifest tallies
        them, so a downgrade is visible rather than acted on silently.  Set
        ``min_trust="screen"`` for GRPO, where the per-step returns carry the
        gradient and the ``action`` fallback — which infers effect from the
        action instead of observing it — would be training on a guess.
    drop_degenerate:
        For ``dpo``: drop pairs whose two completions render identically.  Such
        a pair has no gradient to give — DPO's loss is a function of the
        difference — and a run of them is indistinguishable from a hung trainer.
    resolver:
        Optional ``episode -> str | None`` returning the policy's *own* output
        for that episode.  When it returns text, the row is marked
        ``completion_source="caller"``; when it returns ``None`` the canonical
        rendering is used.
    prompt_resolver:
        Optional ``episode -> str | None`` returning the *observation* the
        policy conditioned on — the accessibility tree, the screenshot
        description, the steps taken so far.  Without it a prompt is the task
        text alone, which every attempt at that task shares; each row then says
        "do this task" with no state, so nothing in the dataset distinguishes a
        good action from a bad one and only the output format is learnable.
        opendesk stores no screen content, so the observation has to come from
        the caller.  This is the prompt-side counterpart of *resolver*, and the
        two are needed together for per-step GUI data: one supplies the state,
        the other the response the policy actually produced in it.
    tree_resolver:
        Optional ``episode -> str | None`` returning the accessibility tree
        used for :mod:`action_space` mapping.  Defaults to ``_tree`` /
        ``observation_tree`` on the episode record.
    canonical_actions:
        When ``True``, map legacy ``Agent.click(coordinates=…)`` to semantic
        ``click(name=…, role=…)`` when the tree uniquely supports it; otherwise
        keep coordinates and set ``sample_weight`` below 1.  Rows gain
        ``completion_legacy``, ``action_tier``, ``mapping_status``.
    sanitize:
        Optional ``episode -> episode`` applied before rendering, for scrubbing
        what should not be memorised.  The audit log holds absolute paths, so a
        run in a temporary directory renders a completion containing that
        directory's *random* name: the same task then renders differently on
        every run, which both teaches the model a path that will never exist
        again and quietly defeats the identical-completion check.  Rewrite those
        params here against your sandbox root.
    """
    if sanitize is not None:
        trajectories = [sanitize(e) for e in trajectories]
    by_id: dict[str, dict[str, Any]] = {
        str(e.get("episode_id")): e for e in trajectories if e.get("episode_id")
    }
    floor = TRUST_ORDER.get(min_trust) if min_trust else None
    stats: dict[str, Any] = {
        "episodes": len(trajectories),
        "format": format,
        "rows": 0,
        "pairs": None if pairs is None else len(pairs),
        "min_trust": min_trust,
        "dropped_identical": 0,
        "dropped_low_trust": 0,
        "dropped_unsuccessful": 0,
        "dropped_unmatched": 0,
        "dropped_prompt_mismatch": 0,
        "by_trust": {},
        "weak_signal": 0,
        "groups_no_spread": 0,
        "tasks": 0,
        "groups": 0,
        "successes": 0,
        "canonical_actions": canonical_actions,
        "action_mapping": {},
    }
    trust_tally: dict[str, int] = {}
    weak_floor = TRUST_ORDER[WEAK_BELOW]

    def _tree_text(episode: dict[str, Any]) -> str:
        if tree_resolver is not None:
            t = tree_resolver(episode)
            if t:
                return t
        from opendesk.learning.action_space import observation_tree_from_episode
        return observation_tree_from_episode(episode)

    def _action_space_fields(episode: dict[str, Any], text: str) -> tuple[str, dict[str, Any]]:
        if not canonical_actions or not text:
            return text, {}
        from opendesk.learning.action_space import map_legacy_completion
        mapped = map_legacy_completion(text, _tree_text(episode))
        key = mapped.status
        stats["action_mapping"][key] = stats["action_mapping"].get(key, 0) + 1
        extra: dict[str, Any] = {
            "sample_weight": mapped.sample_weight,
            "action_tier": mapped.tier,
            "mapping_status": mapped.status,
        }
        if mapped.changed:
            extra["completion_legacy"] = mapped.legacy
        return mapped.primary, extra

    def completion(episode: dict[str, Any]) -> tuple[str, str]:
        if resolver is not None:
            text = resolver(episode)
            if text:
                return text, "caller"
        return render_completion(episode), RENDER_VERSION

    def prompt_of(episode: dict[str, Any]) -> str:
        if prompt_resolver is not None:
            text = prompt_resolver(episode)
            if text:
                return text
        return render_prompt(episode)

    def trusted(episode: dict[str, Any]) -> bool:
        t = trust_of(episode)
        trust_tally[t] = trust_tally.get(t, 0) + 1
        if TRUST_ORDER.get(t, 0) < weak_floor:
            stats["weak_signal"] += 1
        return floor is None or TRUST_ORDER.get(t, 0) >= floor

    rows: list[dict[str, Any]] = []

    if format == "sft":
        for episode in trajectories:
            if _outcome_reward(episode) < min_reward:
                stats["dropped_unsuccessful"] += 1
                continue
            if not trusted(episode):
                stats["dropped_low_trust"] += 1
                continue
            text, source = completion(episode)
            text, space_extra = _action_space_fields(episode, text)
            row = {
                "prompt": prompt_of(episode),
                "completion": text,
                "task": episode.get("task"),
                "group_id": episode.get("task"),
                "episode_id": episode.get("episode_id"),
                "reward": _outcome_reward(episode),
                "steps": _steps_of(episode),
                "returns": _returns(episode),
                "trust": trust_of(episode),
                "completion_source": source,
                "schema": SCHEMA_VERSION,
            }
            row.update(space_extra)
            rows.append(row)

    elif format == "dpo":
        if pairs is None:
            raise ValueError("format='dpo' needs pairs= (see load_pairs())")
        for pair in pairs:
            chosen = by_id.get(str((pair.get("chosen") or {}).get("episode_id")))
            rejected = by_id.get(str((pair.get("rejected") or {}).get("episode_id")))
            if chosen is None or rejected is None:
                stats["dropped_unmatched"] += 1
                continue
            if not (trusted(chosen) and trusted(rejected)):
                stats["dropped_low_trust"] += 1
                continue
            chosen_prompt = prompt_of(chosen)
            rejected_prompt = prompt_of(rejected)
            if chosen_prompt != rejected_prompt:
                # A preference pair only means something if both completions
                # answer the *same* prompt.  With a caller-supplied
                # prompt_resolver the two episodes can be different states of
                # the task, and an action taken in one state compared against an
                # action taken in another teaches nothing about either.
                stats["dropped_prompt_mismatch"] += 1
                continue
            chosen_text, chosen_src = completion(chosen)
            rejected_text, rejected_src = completion(rejected)
            chosen_text, chosen_extra = _action_space_fields(chosen, chosen_text)
            rejected_text, rejected_extra = _action_space_fields(rejected, rejected_text)
            if chosen_text == rejected_text:
                # No gradient to give: DPO's loss is a function of the
                # difference, and an identical pair makes that difference zero
                # for every token.  A file of them trains nothing and looks
                # exactly like a hung trainer.
                stats["dropped_identical"] += 1
                if drop_degenerate:
                    continue
            dpo_row = {
                "prompt": chosen_prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "task": chosen.get("task"),
                "group_id": pair.get("task") or chosen.get("task"),
                "criterion": pair.get("criterion"),
                "margin": pair.get("margin"),
                "chosen_reward": _outcome_reward(chosen),
                "rejected_reward": _outcome_reward(rejected),
                "chosen_id": chosen.get("episode_id"),
                "rejected_id": rejected.get("episode_id"),
                "chosen_returns": _returns(chosen),
                "rejected_returns": _returns(rejected),
                "trust": trust_of(chosen),
                "completion_source": (
                    chosen_src if chosen_src == rejected_src else RENDER_VERSION
                ),
                "schema": SCHEMA_VERSION,
            }
            if chosen_extra.get("sample_weight") is not None:
                dpo_row["chosen_sample_weight"] = chosen_extra["sample_weight"]
            if rejected_extra.get("sample_weight") is not None:
                dpo_row["rejected_sample_weight"] = rejected_extra["sample_weight"]
            dpo_row["chosen_mapping_status"] = chosen_extra.get("mapping_status")
            dpo_row["rejected_mapping_status"] = rejected_extra.get("mapping_status")
            rows.append(dpo_row)

    elif format == "grpo":
        groups: dict[str, list[dict[str, Any]]] = {}
        for episode in trajectories:
            if not trusted(episode):
                stats["dropped_low_trust"] += 1
                continue
            groups.setdefault(str(episode.get("task") or ""), []).append(episode)
        for task, members in groups.items():
            completions = []
            for episode in members:
                text, source = completion(episode)
                completions.append({
                    "text": text,
                    "reward": _outcome_reward(episode),
                    "episode_id": episode.get("episode_id"),
                    "steps": _steps_of(episode),
                    "returns": _returns(episode),
                    "trust": trust_of(episode),
                    "completion_source": source,
                })
            if len(completions) < 2:
                # A single sample carries no group-relative signal: every
                # advantage computed against itself is zero.
                continue
            rewards = [c["reward"] for c in completions]
            spread = max(rewards) - min(rewards)
            if spread == 0.0:
                # Every attempt scored the same, so a group-relative advantage
                # is zero for all of them and the row trains nothing.  Counted
                # rather than dropped: a constant group is still a negative
                # example for any objective that uses the reward directly, and
                # only the caller knows which loss it is running.
                stats["groups_no_spread"] += 1
            rows.append({
                "prompt": prompt_of(members[0]),
                "group_id": task,
                "task": task,
                "reward_spec": members[0].get("reward_spec"),
                "completions": completions,
                "rewards": rewards,
                "reward_mean": sum(rewards) / len(rewards),
                "reward_spread": spread,
                "trust": trust_of(members[0]),
                "schema": SCHEMA_VERSION,
            })

    else:
        raise ValueError(
            f"unknown format {format!r}; expected 'sft', 'dpo' or 'grpo'"
        )

    stats["rows"] = len(rows)
    stats["by_trust"] = trust_tally
    stats["tasks"] = len({r.get("task") for r in rows})
    stats["groups"] = len({r.get("group_id") for r in rows})
    stats["successes"] = sum(
        1 for e in trajectories if _outcome_reward(e) >= 1.0
    )
    return Dataset(rows=rows, format=format, stats=stats)


def export_jsonl(
    rows: Sequence[dict[str, Any]],
    path: str,
    *,
    manifest: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write rows as JSONL and return a manifest.

    The manifest is also written next to the data as ``<path>.manifest.json`` so
    a dataset can be identified later from the file alone — a directory of
    ``.jsonl`` with no record of which renderer or trust floor produced it is
    not reproducible.
    """
    dest = os.path.expanduser(path)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    out = {
        "path": dest,
        "rows": len(rows),
        "schema": SCHEMA_VERSION,
        "render": RENDER_VERSION,
        **(manifest or {}),
    }
    if parent:
        with open(dest + ".manifest.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
    return out
