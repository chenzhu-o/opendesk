"""Best-of-N rollouts and preference pairs.

Running the same task several times and keeping the better attempt is the
simplest form of learning available to an agent that cannot update its
weights — and it doubles as preference-data generation for pipelines that can.

Two signals are used to say which attempt is better:

* **outcome** — the verifiable reward.  A run that satisfies the checks beats
  one that does not.
* **efficiency** — when several runs succeed, the shorter one is better.  A
  reward that is uniformly 1.0 carries no gradient; trajectory length restores
  one, so a pair can still be formed from all-successful attempts.

Pairing is pure data manipulation, so it is testable without a model or a
machine.  Driving the N attempts is a separate concern: pass a ``runner``
coroutine to :func:`best_of_n` and this module will call it N times, or feed in
episodes collected elsewhere.

Example::

    from opendesk.learning.preference import best_of_n, build_pairs, export_pairs

    rollouts = await best_of_n("open the invoice and export it", n=4, runner=run_once)
    pairs = build_pairs(rollouts)
    export_pairs(pairs, path="runs/prefs.jsonl")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence


@dataclass
class Rollout:
    """One attempt at a task, with the signals needed to rank it."""

    episode_id: str
    task: str
    reward: float = 0.0
    score: float = 0.0
    steps: int = 0
    errors: int = 0
    no_effect: int = 0
    total_reward: float = 0.0
    outcome: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def efficiency(self) -> float:
        total = self.steps or 1
        return max(0.0, (total - self.no_effect) / total)

    def rank_key(self) -> tuple:
        """Sort key — **higher is better**, so negate where smaller wins."""
        return (self.reward, self.score, -self.steps, -self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "reward": round(self.reward, 4),
            "score": round(self.score, 4),
            "steps": self.steps,
            "errors": self.errors,
            "no_effect": self.no_effect,
            "efficiency": round(self.efficiency, 4),
            "total_reward": round(self.total_reward, 4),
            "error": self.error,
        }

    @classmethod
    def from_episode(cls, episode: Any) -> "Rollout":
        outcome = episode.outcome or {}
        process = episode.process or {}
        metrics = process.get("metrics") or {}
        reward = float(outcome.get("reward") or 0.0)
        return cls(
            episode_id=episode.id,
            task=episode.task,
            reward=reward,
            score=float(outcome.get("score") or 0.0),
            steps=int(metrics.get("total_steps") or 0),
            errors=int(metrics.get("error_steps") or 0),
            no_effect=int(metrics.get("no_effect_steps") or 0),
            total_reward=float(process.get("total_reward") or 0.0),
            outcome=dict(outcome) if isinstance(outcome, dict) else {},
        )


@dataclass
class Pair:
    """A chosen/rejected preference pair, plus the reason they differ."""

    task: str
    chosen: Rollout
    rejected: Rollout
    criterion: str  # "outcome" | "efficiency"
    margin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "criterion": self.criterion,
            "margin": round(self.margin, 4),
            "chosen": self.chosen.to_dict(),
            "rejected": self.rejected.to_dict(),
        }


@dataclass
class PreferenceDataset:
    pairs: list[Pair] = field(default_factory=list)
    rollouts: list[Rollout] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def outcome_pairs(self) -> int:
        return sum(1 for p in self.pairs if p.criterion == "outcome")

    @property
    def efficiency_pairs(self) -> int:
        return sum(1 for p in self.pairs if p.criterion == "efficiency")

    def best(self) -> Optional[Rollout]:
        return max(self.rollouts, key=lambda r: r.rank_key()) if self.rollouts else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollouts": [r.to_dict() for r in self.rollouts],
            "pairs": [p.to_dict() for p in self.pairs],
            "stats": {
                "rollouts": len(self.rollouts),
                "pairs": len(self.pairs),
                "outcome_pairs": self.outcome_pairs,
                "efficiency_pairs": self.efficiency_pairs,
                "success_rate": (
                    sum(1 for r in self.rollouts if r.reward >= 1.0) / len(self.rollouts)
                    if self.rollouts else 0.0
                ),
            },
        }

    def summary_text(self) -> str:
        lines = [
            f"Best-of-N rollouts — {len(self.rollouts)} attempt(s), "
            f"{len(self.pairs)} preference pair(s)",
        ]
        for r in sorted(self.rollouts, key=lambda r: r.rank_key(), reverse=True):
            lines.append(
                f"  reward={r.reward:.0f}  score={r.score:.0%}  steps={r.steps}"
                f"  errors={r.errors}  no_effect={r.no_effect}   [{r.episode_id}]"
            )
        if self.pairs:
            lines.append("")
            lines.append(
                f"Pairs: {self.outcome_pairs} from differing outcome, "
                f"{self.efficiency_pairs} from differing efficiency"
            )
        elif len(self.rollouts) > 1 and len({r.rank_key() for r in self.rollouts}) == 1:
            lines.append("")
            lines.append(
                "All attempts are equivalent — no preference signal. Increase N, "
                "or add checks that discriminate between runs."
            )
        best = self.best()
        if best:
            lines.append(f"Best attempt: {best.episode_id}")
        return "\n".join(lines)


def build_pairs(
    rollouts: Sequence[Rollout],
    *,
    min_margin: float = 0.0,
    max_pairs: Optional[int] = None,
    strategy: str = "best_vs_worst",
) -> PreferenceDataset:
    """Pair up rollouts into chosen/rejected examples.

    Parameters
    ----------
    rollouts:
        Attempts to pair. At least two are needed for any pair.
    min_margin:
        Minimum outcome-reward difference to form an ``outcome`` pair. Pairs
        formed purely on efficiency must differ in step count instead.
    max_pairs:
        Cap on the number of pairs returned.
    strategy:
        ``best_vs_worst`` (default) pairs the strongest attempt with the weakest;
        ``all_vs_best`` pairs the best against each weaker attempt; ``adjacent``
        pairs neighbours after ranking, which yields the finest-grained contrast.
    """
    items = list(rollouts)
    ranked = sorted(items, key=lambda r: r.rank_key(), reverse=True)
    pairs: list[Pair] = []

    def make(a: Rollout, b: Rollout) -> Optional[Pair]:
        if a is b:
            return None
        margin = a.reward - b.reward
        if margin > min_margin:
            return Pair(a.task, a, b, "outcome", margin)
        # Same outcome — fall back to efficiency, which is the only signal left
        # when every attempt succeeds.
        if a.reward == b.reward and a.steps != b.steps and a.steps < b.steps:
            return Pair(a.task, a, b, "efficiency", 0.0)
        if a.score != b.score and a.score > b.score:
            return Pair(a.task, a, b, "outcome", 0.0)
        return None

    if strategy == "adjacent":
        for a, b in zip(ranked, ranked[1:]):
            pair = make(a, b)
            if pair:
                pairs.append(pair)
    elif strategy == "all_vs_best":
        if ranked:
            for b in ranked[1:]:
                pair = make(ranked[0], b)
                if pair:
                    pairs.append(pair)
    else:  # best_vs_worst
        if len(ranked) >= 2:
            pair = make(ranked[0], ranked[-1])
            if pair:
                pairs.append(pair)

    if max_pairs is not None and max_pairs > 0:
        pairs = pairs[:max_pairs]
    return PreferenceDataset(pairs=pairs, rollouts=items)


def export_pairs(
    dataset: PreferenceDataset,
    *,
    path: str,
) -> dict[str, Any]:
    """Write preference pairs as JSONL (one pair per line)."""
    dest = os.path.expanduser(path)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(dest, "w", encoding="utf-8") as fh:
        for pair in dataset.pairs:
            fh.write(json.dumps({"type": "preference", **pair.to_dict()},
                                ensure_ascii=False) + "\n")

    return {
        "path": dest,
        "pairs": len(dataset.pairs),
        "rollouts": len(dataset.rollouts),
        **(dataset.to_dict()["stats"]),
    }


async def best_of_n(
    task: str,
    *,
    n: int,
    runner: Callable[[int], Awaitable[Any]],
    min_margin: float = 0.0,
    max_pairs: Optional[int] = None,
    strategy: str = "best_vs_worst",
) -> PreferenceDataset:
    """Run *task* ``n`` times via *runner* and build preference pairs.

    ``runner`` receives the attempt index (0-based) and must return an episode
    (anything with ``id``, ``task``, ``outcome`` and ``process``).  A failure
    in one attempt does not abort the sweep: the attempt is recorded with
    ``reward=0`` and its error, because a crash is itself a negative example.

    This module does not drive an agent on its own — opendesk is a tool layer,
    not a policy.  Supply a runner that talks to your model and this becomes a
    full best-of-N loop.
    """
    rollouts: list[Rollout] = []
    for i in range(max(1, n)):
        try:
            episode = await runner(i)
        except Exception as exc:
            rollouts.append(Rollout(
                episode_id=f"attempt-{i}", task=task, reward=0.0,
                error=f"{type(exc).__name__}: {exc}",
            ))
            continue
        rollout = (
            episode if isinstance(episode, Rollout) else Rollout.from_episode(episode)
        )
        if not rollout.task:
            rollout.task = task
        rollouts.append(rollout)

    return build_pairs(
        rollouts, min_margin=min_margin, max_pairs=max_pairs, strategy=strategy
    )
