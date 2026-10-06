"""Tests for the training-side handoff — trajectories and preference pairs
rendered into dataset rows for SFT / DPO / GRPO.

The tests that matter here are the ones guarding against *silently wrong*
training data: a label leaked into the completion, a pair that cannot produce a
gradient, a split that leaks the eval set.  Those failures do not raise, so they
have to be asserted.
"""

from __future__ import annotations

import json
from typing import Optional

import pytest

from opendesk.learning import training


# ---------------------------------------------------------------------------
# Fixtures — shaped exactly like trajectories.export() writes them
# ---------------------------------------------------------------------------

def step(n: int, action: str, params: dict, *, reward: float = 0.0,
         ret: float = 0.0, effect: str = "changed") -> dict:
    return {
        "type": "step",
        "step": n,
        "observation": {"screen": f"fp{n}", "observation_index": n - 1,
                        "app": "Files", "window": "Invoices",
                        "size": "800x600", "screenshot": f"img/step{n}.png"},
        "action": {"type": action, "params": params},
        "result": None,
        "error": None,
        "effect": effect,
        "next_state": f"s{n}",
        "reward": reward,
        "return": ret,
        "assertions": [],
        "done": False,
    }


def episode(episode_id: str, *, task: str = "Export the invoice",
            reward: float = 1.0, signal_source: str = "ui",
            actions=None, goal: Optional[str] = None,
            spec: Optional[dict] = None) -> dict:
    actions = actions if actions is not None else [
        ("click", {"role": "button", "name": "Export"}),
        ("type", {"text": "~/invoice.pdf"}),
        ("press_key", {"key": "Enter"}),
    ]
    steps = [
        step(i + 1, a, p, reward=0.7, ret=float(len(actions) - i))
        for i, (a, p) in enumerate(actions)
    ]
    return {
        "schema": "opendesk.trajectory/1",
        "type": "episode",
        "episode_id": episode_id,
        "task": task,
        "session_id": "default",
        "started_at": 1000.0,
        "ended_at": 1012.0,
        "duration_s": 12.0,
        "goal": goal,
        "reward_spec": spec or {"task": task, "mode": "all", "checks": [
            {"kind": "file_exists", "path": "/tmp/invoice.pdf", "required": True,
             "weight": 1.0},
        ]},
        "meta": {},
        "outcome": {"reward": reward, "score": 1.0 if reward else 0.4,
                    "checks": []},
        "assertions": None,
        "process": {
            "gamma": 0.95,
            "weights": {"step_cost": -0.1},
            "metrics": {
                "total_steps": len(actions), "effective_steps": len(actions) - 1,
                "observable_steps": len(actions) - 1, "no_effect_steps": 0,
                "error_steps": 0, "loop_steps": 0, "efficiency": 0.67,
                "passive_skipped": 0, "signal_source": signal_source,
            },
        },
        "steps": steps,
        "step_count": len(actions),
        "path": "runs/ep.jsonl",
    }


def pair(chosen_actions, rejected_actions, *, task: str = "Export the invoice",
         criterion: str = "outcome", margin: float = 1.0,
         chosen="c1", rejected="r1") -> dict:
    return {
        "type": "preference",
        "task": task,
        "criterion": criterion,
        "margin": margin,
        "chosen": {"episode_id": chosen, "task": task, "reward": 1.0,
                   "score": 1.0, "steps": len(chosen_actions), "errors": 0,
                   "no_effect": 0, "efficiency": 1.0, "total_reward": 1.0,
                   "error": None},
        "rejected": {"episode_id": rejected, "task": task, "reward": 0.0,
                     "score": 0.4, "steps": len(rejected_actions), "errors": 1,
                     "no_effect": 2, "efficiency": 0.3, "total_reward": -1.0,
                     "error": None},
    }


A_OK = [("click", {"role": "button", "name": "Export"}),
        ("type", {"text": "~/invoice.pdf"})]
A_BAD = [("click", {"role": "button", "name": "Settings"}),
         ("click", {"role": "button", "name": "Settings"})]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_action_renders_as_call(self):
        assert training.render_action(
            {"type": "click", "params": {"name": "Export", "role": "button"}}
        ) == 'click(name="Export", role="button")'

    def test_params_are_key_sorted(self):
        """Otherwise two identical rollouts could render differently and a
        preference pair built from them would carry noise, not signal."""
        a = training.render_action({"type": "x", "params": {"b": 1, "a": 2}})
        b = training.render_action({"type": "x", "params": {"a": 2, "b": 1}})
        assert a == b == "x(a=2, b=1)"

    def test_string_is_quoted_so_it_cannot_be_confused_with_an_identifier(self):
        assert training.render_action(
            {"type": "type", "params": {"text": "exit"}}
        ) == 'type(text="exit")'

    def test_no_params(self):
        assert training.render_action({"type": "screenshot", "params": {}}) \
            == "screenshot()"

    def test_null_params_are_omitted(self):
        """The audit log stores the tool's whole parsed signature, so an action
        that set one field arrives with every other field null.  Rendering them
        is most of the transcript and teaches the model to recite boilerplate."""
        text = training.render_action({"type": "file_write", "params": {
            "action": "write_file", "argv": None, "command": None,
            "content": "total: 42\n", "cwd": None, "dst": None,
            "path": "report.txt", "timeout": None,
        }})
        assert text == ('file_write(action="write_file", content="total: 42\\n", '
                        'path="report.txt")')
        assert "null" not in text

    def test_a_non_null_param_that_differs_from_the_name_is_kept(self):
        """`system`'s sub-action is a real choice, not a repeat of the call."""
        text = training.render_action(
            {"type": "file_write", "params": {"action": "write_file"}}
        )
        assert text == 'file_write(action="write_file")'

    def test_null_params_can_be_kept_on_request(self):
        text = training.render_action(
            {"type": "x", "params": {"a": None}}, omit_none=False
        )
        assert text == "x(a=null)"

    def test_completion_is_actions_only(self):
        text = training.render_completion(episode("e1", actions=A_OK))
        assert text == 'click(name="Export", role="button")\ntype(text="~/invoice.pdf")'

    def test_completion_carries_no_label(self):
        """The central guarantee.  A reward or return token in the completion
        teaches the model to emit the grader's answer key."""
        text = training.render_completion(episode("e1", reward=1.0))
        for token in ("reward", "return", "outcome", "0.7", "efficiency",
                      "changed", "success"):
            assert token not in text

    def test_completion_identical_across_identical_actions(self):
        """Two episodes that did the same thing but scored differently must
        render the same text — that is what makes the pair degenerate, and the
        detector downstream depends on it."""
        good = episode("e1", reward=1.0, actions=A_OK)
        same = episode("e2", reward=0.0, actions=A_OK)
        assert training.render_completion(good) == training.render_completion(same)

    def test_prompt_is_the_task_and_goal(self):
        assert training.render_prompt(episode("e1", goal="done")) == \
            "Task: Export the invoice\nGoal: done"

    def test_prompt_without_goal(self):
        assert training.render_prompt(episode("e1")) == "Task: Export the invoice"

    def test_describe_does_include_labels(self):
        """It is for humans; the docstring says not to train on it."""
        text = training.describe_trajectory(episode("e1"))
        assert "reward=" in text and "return=" in text and "e1" in text


# ---------------------------------------------------------------------------
# Reading producers' output
# ---------------------------------------------------------------------------

class TestLoading:
    def _write(self, tmp_path, records, name="t.jsonl"):
        p = tmp_path / name
        with open(p, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        return str(p)

    def test_reads_header_and_steps(self, tmp_path):
        ep = episode("e1")
        header = {k: v for k, v in ep.items() if k != "steps"}
        path = self._write(tmp_path, [header, *ep["steps"]])
        loaded = training.load_trajectories(path)
        assert len(loaded) == 1
        assert loaded[0]["episode_id"] == "e1"
        assert len(loaded[0]["steps"]) == 3

    def test_reads_several_episodes_appended_to_one_file(self, tmp_path):
        e1, e2 = episode("e1"), episode("e2", task="Other")
        recs = []
        for ep in (e1, e2):
            recs.append({k: v for k, v in ep.items() if k != "steps"})
            recs.extend(ep["steps"])
        loaded = training.load_trajectories(self._write(tmp_path, recs))
        assert [e["episode_id"] for e in loaded] == ["e1", "e2"]

    def test_reads_a_directory_of_jsonl(self, tmp_path):
        e1, e2 = episode("e1"), episode("e2")
        for name, ep in (("a.jsonl", e1), ("b.jsonl", e2)):
            self._write(tmp_path, [
                {k: v for k, v in ep.items() if k != "steps"}, *ep["steps"],
            ], name=name)
        loaded = training.load_trajectories(str(tmp_path))
        assert sorted(e["episode_id"] for e in loaded) == ["e1", "e2"]

    def test_drops_embedded_base64_screenshots(self, tmp_path):
        """A few inlined PNGs silently exceed any context window."""
        ep = episode("e1")
        for s in ep["steps"]:
            s["observation"]["screenshot"] = "data:image/png;base64," + "A" * 5000
        header = {k: v for k, v in ep.items() if k != "steps"}
        loaded = training.load_trajectories(
            self._write(tmp_path, [header, *ep["steps"]])
        )
        rendered = training.render_completion(loaded[0])
        assert "base64" not in rendered and len(rendered) < 200

    def test_step_without_header_is_not_dropped(self, tmp_path):
        ep = episode("e1")
        loaded = training.load_trajectories(self._write(tmp_path, ep["steps"]))
        assert len(loaded) == 1 and len(loaded[0]["steps"]) == 3

    def test_unknown_record_kinds_are_skipped(self, tmp_path):
        ep = episode("e1")
        recs = [{"type": "telemetry", "x": 1},
                {k: v for k, v in ep.items() if k != "steps"}, *ep["steps"]]
        loaded = training.load_trajectories(self._write(tmp_path, recs))
        assert len(loaded) == 1

    def test_truncated_line_raises_with_the_file_and_line(self, tmp_path):
        ep = episode("e1")
        p = tmp_path / "bad.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({k: v for k, v in ep.items() if k != "steps"}) + "\n")
            fh.write('{"type": "step", "step": 1, "act')  # interrupted export
        with pytest.raises(ValueError, match="bad.jsonl:2"):
            training.load_trajectories(str(p))

    def test_load_pairs(self, tmp_path):
        path = self._write(tmp_path, [pair(A_OK, A_BAD)])
        assert len(training.load_pairs(path)) == 1


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------

class TestSFT:
    def test_only_successful_attempts_become_rows(self):
        job = training.build_dataset([
            episode("ok", reward=1.0),
            episode("bad", reward=0.0),
        ], format="sft")
        assert [r["episode_id"] for r in job.rows] == ["ok"]
        assert job.stats["dropped_unsuccessful"] == 1

    def test_row_shape(self):
        job = training.build_dataset([episode("ok")], format="sft")
        row = job.rows[0]
        assert set(row) >= {
            "prompt", "completion", "task", "episode_id", "reward", "steps",
            "returns", "trust", "completion_source", "schema",
        }
        assert row["completion_source"] == training.RENDER_VERSION

    def test_min_reward_is_configurable(self):
        job = training.build_dataset(
            [episode("a", reward=0.4), episode("b", reward=0.0)],
            format="sft", min_reward=0.3,
        )
        assert [r["episode_id"] for r in job.rows] == ["a"]

    def test_resolver_supplies_the_real_completion(self):
        """When the caller logged the policy's own tokens, they win — the
        canonical rendering is a fallback, not the point."""
        job = training.build_dataset(
            [episode("ok")], format="sft",
            resolver=lambda ep: f"MY OWN OUTPUT for {ep['episode_id']}",
        )
        row = job.rows[0]
        assert row["completion"] == "MY OWN OUTPUT for ok"
        assert row["completion_source"] == "caller"

    def test_resolver_returning_none_falls_back(self):
        job = training.build_dataset(
            [episode("ok")], format="sft", resolver=lambda ep: None
        )
        assert job.rows[0]["completion_source"] == training.RENDER_VERSION


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

class TestDPO:
    def test_joins_pairs_to_trajectories(self):
        job = training.build_dataset(
            [episode("c1", actions=A_OK), episode("r1", reward=0.0, actions=A_BAD)],
            pairs=[pair(A_OK, A_BAD)], format="dpo",
        )
        assert len(job.rows) == 1
        row = job.rows[0]
        assert row["prompt"] == "Task: Export the invoice"
        assert "Export" in row["chosen"] and "Settings" in row["rejected"]
        assert row["criterion"] == "outcome"

    def test_identical_completions_are_dropped(self):
        """Same actions, different scores: the pair cannot produce a gradient."""
        job = training.build_dataset(
            [episode("c1", actions=A_OK), episode("r1", reward=0.0, actions=A_OK)],
            pairs=[pair(A_OK, A_OK)], format="dpo",
        )
        assert job.rows == []
        assert job.stats["dropped_identical"] == 1
        assert "identical" in job.summary_text()

    def test_identical_pair_can_be_kept_on_request(self):
        job = training.build_dataset(
            [episode("c1", actions=A_OK), episode("r1", reward=0.0, actions=A_OK)],
            pairs=[pair(A_OK, A_OK)], format="dpo", drop_degenerate=False,
        )
        assert len(job.rows) == 1 and job.stats["dropped_identical"] == 1

    def test_pair_without_a_trajectory_is_dropped_not_guessed(self):
        job = training.build_dataset(
            [episode("c1", actions=A_OK)],
            pairs=[pair(A_OK, A_BAD)], format="dpo",
        )
        assert job.rows == [] and job.stats["dropped_unmatched"] == 1

    def test_dpo_without_pairs_is_an_error(self):
        with pytest.raises(ValueError, match="needs pairs"):
            training.build_dataset([episode("c1")], format="dpo")

    def test_no_label_leaks_into_either_side(self):
        job = training.build_dataset(
            [episode("c1", actions=A_OK), episode("r1", reward=0.0, actions=A_BAD)],
            pairs=[pair(A_OK, A_BAD)], format="dpo",
        )
        row = job.rows[0]
        for side in ("chosen", "rejected"):
            for token in ("reward", "return", "outcome", "margin"):
                assert token not in row[side], f"{token} leaked into {side}"

    def test_rewards_travel_in_metadata_not_in_the_text(self):
        job = training.build_dataset(
            [episode("c1", actions=A_OK), episode("r1", reward=0.0, actions=A_BAD)],
            pairs=[pair(A_OK, A_BAD)], format="dpo",
        )
        row = job.rows[0]
        assert row["chosen_reward"] == 1.0 and row["rejected_reward"] == 0.0


# ---------------------------------------------------------------------------
# Trust floor
# ---------------------------------------------------------------------------

class TestTrust:
    def test_trust_reflects_the_effect_signal(self):
        job = training.build_dataset(
            [episode("e1", signal_source="ui")], format="sft"
        )
        assert job.rows[0]["trust"] == "ui"

    def test_missing_signal_is_unknown_not_assumed_good(self):
        """Unknown is reported as unknown.  It is not silently upgraded to a
        trusted signal, and with no floor set it is not silently dropped
        either — either would hide the downgrade."""
        ep = episode("e1")
        del ep["process"]["metrics"]["signal_source"]
        assert training.trust_of(ep) == "unknown"
        job = training.build_dataset([ep], format="sft")
        assert job.rows[0]["trust"] == "unknown"
        assert job.stats["weak_signal"] == 1
        assert "inferred or unrecorded" in job.summary_text()

    def test_min_trust_excludes_the_weak_fallback(self):
        """`action` infers effect from the action instead of observing it."""
        eps = [episode("strong", signal_source="ui"),
               episode("weak", signal_source="action")]
        assert len(training.build_dataset(eps, format="sft").rows) == 2
        strict = training.build_dataset(eps, format="sft", min_trust="screen")
        assert [r["episode_id"] for r in strict.rows] == ["strong"]
        assert strict.stats["dropped_low_trust"] == 1

    def test_weak_signal_is_kept_by_default_but_flagged(self):
        job = training.build_dataset(
            [episode("w", signal_source="action")], format="sft"
        )
        assert len(job.rows) == 1
        assert job.rows[0]["trust"] == "action"
        assert job.stats["weak_signal"] == 1
        assert job.stats["min_trust"] is None

    def test_no_warning_once_a_floor_is_set(self):
        job = training.build_dataset(
            [episode("w", signal_source="action")], format="sft",
            min_trust="screen",
        )
        assert "inferred or unrecorded" not in job.summary_text()

    def test_by_trust_tally_only_counts_what_was_examined(self):
        job = training.build_dataset([
            episode("a", signal_source="ui"),
            episode("b", signal_source="action"),
        ], format="sft")
        assert job.stats["by_trust"] == {"ui": 1, "action": 1}


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------

class TestGRPO:
    def test_groups_attempts_by_task(self):
        job = training.build_dataset([
            episode("a1", task="T", reward=1.0, actions=A_OK),
            episode("a2", task="T", reward=0.0, actions=A_BAD),
            episode("b1", task="U", reward=1.0, actions=A_OK),
            episode("b2", task="U", reward=1.0, actions=A_OK),
        ], format="grpo")
        assert len(job.rows) == 2
        t = next(r for r in job.rows if r["task"] == "T")
        assert [c["episode_id"] for c in t["completions"]] == ["a1", "a2"]
        assert t["rewards"] == [1.0, 0.0]
        assert t["reward_spread"] == 1.0

    def test_lone_attempt_is_dropped(self):
        """Group-relative advantage against a single sample is always zero."""
        job = training.build_dataset([
            episode("a1", task="T"), episode("b1", task="U"),
            episode("b2", task="U"),
        ], format="grpo")
        assert [r["task"] for r in job.rows] == ["U"]

    def test_row_carries_the_reward_spec_for_re_verification(self):
        """Verification stays in opendesk: the trainer sends the spec back
        rather than scoring with a model."""
        job = training.build_dataset(
            [episode("a1", task="T"), episode("a2", task="T", reward=0.0)],
            format="grpo",
        )
        spec = job.rows[0]["reward_spec"]
        assert spec["checks"][0]["kind"] == "file_exists"

    def test_per_step_returns_travel_for_credit_assignment(self):
        job = training.build_dataset(
            [episode("a1", task="T"), episode("a2", task="T", reward=0.0)],
            format="grpo",
        )
        c = job.rows[0]["completions"][0]
        assert len(c["returns"]) == c["steps"] == 3


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

class TestSplit:
    def _rows(self, tasks, per_task=4):
        out = []
        for t in tasks:
            for i in range(per_task):
                out.append({"task": t, "group_id": t, "prompt": t, "n": i})
        return out

    def test_split_is_by_task_not_by_row(self):
        """Attempts at one task are near-duplicates; splitting them across
        train and eval reports a number that means nothing."""
        rows = self._rows([f"task-{i}" for i in range(60)])
        parts = training.split_rows(rows, val_ratio=0.2, test_ratio=0.2)
        seen: dict[str, str] = {}
        for where, part in parts.items():
            for row in part:
                assert seen.setdefault(row["task"], where) == where

    def test_split_is_deterministic(self):
        rows = self._rows([f"task-{i}" for i in range(40)])
        a = training.split_rows(rows, val_ratio=0.25, seed=7)
        b = training.split_rows(rows, val_ratio=0.25, seed=7)
        assert [r["task"] for r in a["train"]] == [r["task"] for r in b["train"]]

    def test_split_does_not_depend_on_row_order(self):
        rows = self._rows([f"task-{i}" for i in range(40)])
        shuffled = rows[::-1]
        a = training.split_rows(rows, val_ratio=0.25)
        b = training.split_rows(shuffled, val_ratio=0.25)
        assert sorted(r["task"] for r in a["train"]) == \
            sorted(r["task"] for r in b["train"])

    def test_seed_changes_the_assignment(self):
        rows = self._rows([f"task-{i}" for i in range(80)])
        a = training.split_rows(rows, val_ratio=0.3, seed=1)["train"]
        b = training.split_rows(rows, val_ratio=0.3, seed=2)["train"]
        assert [r["task"] for r in a] != [r["task"] for r in b]

    def test_nothing_is_lost(self):
        rows = self._rows([f"task-{i}" for i in range(35)])
        parts = training.split_rows(rows, val_ratio=0.2, test_ratio=0.2)
        assert sum(len(p) for p in parts.values()) == len(rows)

    def test_empty_ratios_leave_val_and_test_out_entirely(self):
        parts = training.split_rows(self._rows(["a", "b"]), val_ratio=0.0)
        assert set(parts) == {"train"}

    def test_bad_ratios_are_rejected(self):
        with pytest.raises(ValueError, match="sum to <= 1"):
            training.split_rows([], val_ratio=0.6, test_ratio=0.6)

    def test_grpo_rows_split_by_group_id(self):
        job = training.build_dataset([
            episode("a1", task="T"), episode("a2", task="T", reward=0.0),
            episode("b1", task="U"), episode("b2", task="U", reward=0.0),
            episode("c1", task="V"), episode("c2", task="V", reward=0.0),
            episode("d1", task="W"), episode("d2", task="W", reward=0.0),
        ], format="grpo")
        parts = training.split_rows(job.rows, val_ratio=0.25)
        seen: dict[str, str] = {}
        for where, rows in parts.items():
            for row in rows:
                assert seen.setdefault(row["group_id"], where) == where
        assert len(seen) == 4


# ---------------------------------------------------------------------------
# Scrubbing what should not be memorised
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_sanitize_runs_before_rendering(self):
        def strip_root(ep):
            for step in ep["steps"]:
                params = step["action"]["params"]
                params["path"] = params["path"].replace("/tmp/run-9f3/", "")
            return ep

        job = training.build_dataset(
            [episode("e1", actions=[("file_write", {"path": "/tmp/run-9f3/a.txt"})])],
            format="sft", sanitize=strip_root,
        )
        assert job.rows[0]["completion"] == 'file_write(path="a.txt")'

    def test_the_same_task_renders_differently_across_runs(self):
        """The hazard sanitize exists for: the audit log holds absolute paths, so
        a run in a temporary directory bakes that directory's *random* name into
        every completion."""
        def render(path):
            return training.build_dataset(
                [episode("e1", actions=[("file_write", {"path": path})])],
                format="sft",
            ).rows[0]["completion"]

        assert render("/tmp/run-aaa/a.txt") != render("/tmp/run-bbb/a.txt")

    def test_a_run_specific_path_makes_a_meaningless_pair_survive(self):
        """Two attempts that did the same thing in different temp dirs render as
        two *different* completions, so the pair is kept — and the difference it
        encodes is the directory name, not the task."""
        job = training.build_dataset(
            [episode("c1", actions=[("file_write", {"path": "/tmp/run-aaa/a.txt"})]),
             episode("r1", reward=0.0,
                     actions=[("file_write", {"path": "/tmp/run-bbb/a.txt"})])],
            pairs=[pair(A_OK, A_OK, chosen="c1", rejected="r1")],
            format="dpo",
        )
        assert len(job.rows) == 1
        assert job.stats["dropped_identical"] == 0


# ---------------------------------------------------------------------------
# Groups with nothing to compare
# ---------------------------------------------------------------------------

class TestGroupSpread:
    def _grpo(self, rewards):
        eps = [episode(f"e{i}", task="T", reward=r, actions=A_OK)
               for i, r in enumerate(rewards)]
        return training.build_dataset(eps, format="grpo")

    def test_all_failed_group_is_counted_not_dropped(self):
        """Group-relative advantage is zero for every member, so the row trains
        nothing under GRPO.  It is still a negative example for an objective
        that uses the reward directly, so the caller decides."""
        job = self._grpo([0.0, 0.0, 0.0])
        assert len(job.rows) == 1
        assert job.rows[0]["reward_spread"] == 0.0
        assert job.stats["groups_no_spread"] == 1
        assert "no reward spread" in job.summary_text()

    def test_a_group_with_spread_is_not_flagged(self):
        job = self._grpo([1.0, 0.0])
        assert job.stats["groups_no_spread"] == 0
        assert "no reward spread" not in job.summary_text()

    def test_all_passed_group_is_flagged_too(self):
        job = self._grpo([1.0, 1.0])
        assert job.stats["groups_no_spread"] == 1


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_writes_jsonl_and_a_manifest(self, tmp_path):
        job = training.build_dataset(
            [episode("c1", actions=A_OK), episode("r1", reward=0.0, actions=A_BAD)],
            pairs=[pair(A_OK, A_BAD)], format="dpo",
        )
        path = str(tmp_path / "out" / "dpo.jsonl")
        manifest = training.export_jsonl(job.rows, path, manifest=job.stats)

        with open(path, encoding="utf-8") as fh:
            lines = [json.loads(ln) for ln in fh if ln.strip()]
        assert len(lines) == 1 and "chosen" in lines[0]

        # The manifest makes a dataset identifiable from the file alone.
        with open(path + ".manifest.json", encoding="utf-8") as fh:
            saved = json.load(fh)
        assert saved["rows"] == 1
        assert saved["render"] == training.RENDER_VERSION
        assert saved["schema"] == training.SCHEMA_VERSION == manifest["schema"]

    def test_rows_are_round_trippable(self, tmp_path):
        job = training.build_dataset(
            [episode("c1", actions=A_OK), episode("r1", reward=0.0, actions=A_BAD)],
            pairs=[pair(A_OK, A_BAD)], format="dpo",
        )
        path = str(tmp_path / "dpo.jsonl")
        training.export_jsonl(job.rows, path)
        with open(path, encoding="utf-8") as fh:
            back = json.loads(fh.readline())
        assert back == job.rows[0]


# ---------------------------------------------------------------------------
# End-to-end through the real producers
# ---------------------------------------------------------------------------

class TestThroughRealProducers:
    """Round-trip against trajectories.export() and preference.export_pairs()
    rather than hand-built dicts, so a change to either writer breaks this."""

    def test_export_then_build_sft(self, tmp_path):
        from opendesk.learning import trajectories

        ep = trajectories.begin_episode("Export the invoice", session_id="rt")
        trajectories.end_episode(
            ep, session_id="rt",
            outcome={"reward": 1.0, "score": 1.0, "checks": []},
            process={"gamma": 0.95, "weights": {}, "steps": [], "returns": [],
                     "metrics": {"total_steps": 0, "signal_source": "ui"}},
        )
        entries = [
            {"action": "click", "screen": "fp1", "timestamp": 1.0, "params": {}},
        ]
        path = str(tmp_path / "rt.jsonl")
        trajectories.export(
            ep, entries=entries, path=path,
            process={"gamma": 0.95, "weights": {}, "steps": [], "returns": [],
                     "metrics": {"total_steps": 1, "signal_source": "ui"}},
        )

        job = training.build_dataset(training.load_trajectories(path), format="sft")
        assert len(job.rows) == 1
        assert job.rows[0]["task"] == "Export the invoice"
        assert job.rows[0]["reward"] == 1.0
        assert job.rows[0]["trust"] == "ui"
        assert job.rows[0]["completion"] == "click()"

    def test_export_without_process_still_yields_a_row(self, tmp_path):
        """A missing process block costs the step signal, not the example: the
        completion and the outcome reward are still valid training data."""
        from opendesk.learning import trajectories

        ep = trajectories.begin_episode("T", session_id="np")
        trajectories.end_episode(ep, session_id="np", outcome={"reward": 1.0})
        path = str(tmp_path / "np.jsonl")
        trajectories.export(
            ep, entries=[{"action": "click", "timestamp": 1.0, "params": {}}],
            path=path,
        )
        job = training.build_dataset(training.load_trajectories(path), format="sft")
        assert len(job.rows) == 1
        assert job.rows[0]["trust"] == "unknown"

    def test_pairs_then_build_dpo(self, tmp_path):
        from opendesk.learning import preference

        rollouts = [
            preference.Rollout(episode_id="win", task="T", reward=1.0, score=1.0,
                               steps=3),
            preference.Rollout(episode_id="lose", task="T", reward=0.0, score=0.2,
                               steps=5, errors=1),
        ]
        dataset = preference.build_pairs(rollouts)
        path = str(tmp_path / "prefs.jsonl")
        preference.export_pairs(dataset, path=path)

        eps = [
            episode("win", task="T", actions=A_OK),
            episode("lose", task="T", reward=0.0, actions=A_BAD),
        ]
        job = training.build_dataset(
            eps, pairs=training.load_pairs(path), format="dpo"
        )
        assert len(job.rows) == 1
        assert job.rows[0]["chosen_id"] == "win"
        assert job.rows[0]["rejected_id"] == "lose"


# ---------------------------------------------------------------------------
# The tool surface
# ---------------------------------------------------------------------------

class TestToolSurface:
    """Drives the real ``rollout`` tool over a real sandbox, so the wiring
    from episode → audit log → dataset row is exercised, not just the pure
    functions."""

    @staticmethod
    async def _attempt(sid: str, eid: str, task: str, reward: float,
                       actions: list[tuple[str, dict]]) -> None:
        from opendesk.computer.sandbox import ActionType, get_sandbox
        from opendesk.learning import process, trajectories

        sb = get_sandbox(sid)
        ep = trajectories.begin_episode(
            task, session_id=sid, sandbox=sb, episode_id=eid
        )
        for fp, (action, params) in enumerate(actions):
            sb.current_screen = f"fp-{eid}-{fp}"
            await sb.record_action(ActionType(action), params)
        trajectories.end_episode(
            ep, session_id=sid, sandbox=sb,
            outcome={"reward": reward, "score": reward, "checks": []},
        )
        ep.process = process.score_process(
            sb.export_audit_log(), outcome_reward=reward
        ).to_dict()

    def _ctx(self, sid: str):
        from opendesk.computer.sandbox import clear_sandbox
        from opendesk.learning import trajectories
        from opendesk.tools.base import ToolContext

        from tests._fakes import FakeComputer

        clear_sandbox(sid)
        trajectories.clear_episodes(sid)
        return ToolContext(session_id=sid, computer=FakeComputer())

    @pytest.mark.asyncio
    async def test_sft_dataset_from_attempts(self, tmp_path):
        from opendesk.tools.rollout import RolloutTool

        sid = "ds-sft"
        ctx = self._ctx(sid)
        await self._attempt(sid, "good", "Export it", 1.0,
                            [("ui_action", {"name": "Export"})])
        await self._attempt(sid, "bad", "Export it", 0.0,
                            [("ui_action", {"name": "Settings"})])

        tool = RolloutTool()
        dest = tmp_path / "sft.jsonl"
        r = await tool.execute(ctx, tool.parse_params(
            {"action": "dataset", "format": "sft", "path": str(dest)}
        ))
        assert not r.error, r.output
        assert r.metadata["rows"] == 1

        row = json.loads(dest.read_text(encoding="utf-8").strip())
        assert row["episode_id"] == "good"
        assert row["completion"] == 'ui_action(name="Export")'
        assert "reward" not in row["completion"]
        # The manifest is what makes the file identifiable later.
        assert dest.with_suffix(".jsonl.manifest.json").is_file()

    @pytest.mark.asyncio
    async def test_dpo_dataset_from_attempts(self, tmp_path):
        from opendesk.tools.rollout import RolloutTool

        sid = "ds-dpo"
        ctx = self._ctx(sid)
        await self._attempt(sid, "good", "Export it", 1.0,
                            [("ui_action", {"name": "Export"})])
        await self._attempt(sid, "bad", "Export it", 0.0,
                            [("ui_action", {"name": "Settings"})])

        tool = RolloutTool()
        dest = tmp_path / "dpo.jsonl"
        r = await tool.execute(ctx, tool.parse_params(
            {"action": "dataset", "format": "dpo", "path": str(dest)}
        ))
        assert not r.error, r.output
        row = json.loads(dest.read_text(encoding="utf-8").strip())
        assert row["chosen"] == 'ui_action(name="Export")'
        assert row["rejected"] == 'ui_action(name="Settings")'

    @pytest.mark.asyncio
    async def test_dpo_without_a_usable_pair_explains_itself(self):
        from opendesk.tools.rollout import RolloutTool

        sid = "ds-dpo-one"
        ctx = self._ctx(sid)
        await self._attempt(sid, "only", "Export it", 1.0,
                            [("ui_action", {"name": "Export"})])

        tool = RolloutTool()
        r = await tool.execute(ctx, tool.parse_params(
            {"action": "dataset", "format": "dpo"}
        ))
        assert r.error
        assert "at least two attempts" in r.output

    @pytest.mark.asyncio
    async def test_dataset_needs_episodes(self):
        from opendesk.tools.rollout import RolloutTool

        ctx = self._ctx("ds-empty")
        tool = RolloutTool()
        r = await tool.execute(ctx, tool.parse_params({"action": "dataset"}))
        assert r.error

    @pytest.mark.asyncio
    async def test_split_writes_one_file_per_part(self, tmp_path):
        from opendesk.tools.rollout import RolloutTool

        sid = "ds-split"
        ctx = self._ctx(sid)
        for i in range(12):
            await self._attempt(sid, f"e{i}", f"Task {i}", 1.0,
                                [("ui_action", {"name": f"Act{i}"})])

        tool = RolloutTool()
        dest = tmp_path / "data.jsonl"
        r = await tool.execute(ctx, tool.parse_params({
            "action": "dataset", "format": "sft", "path": str(dest),
            "val_ratio": 0.25, "test_ratio": 0.25,
        }))
        assert not r.error, r.output
        assert r.metadata["splits"]["train"] > 0
        assert (tmp_path / "data.train.jsonl").is_file()
        assert (tmp_path / "data.val.jsonl").is_file()

        # Every task lands wholly on one side.
        seen: dict[str, str] = {}
        for part in ("train", "val", "test"):
            f = tmp_path / f"data.{part}.jsonl"
            if not f.is_file():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    t = json.loads(line)["task"]
                    assert seen.setdefault(t, part) == part

    @pytest.mark.asyncio
    async def test_min_trust_filters_the_weak_fallback(self):
        from opendesk.tools.rollout import RolloutTool

        sid = "ds-trust"
        ctx = self._ctx(sid)
        await self._attempt(sid, "a", "T", 1.0, [("ui_action", {"name": "X"})])
        # Strip the signal so the episode reads as an unrecorded one.
        from opendesk.learning import trajectories
        ep = trajectories.get_episode("a", sid)
        ep.process["metrics"].pop("signal_source", None)

        tool = RolloutTool()
        r = await tool.execute(ctx, tool.parse_params(
            {"action": "dataset", "format": "sft", "min_trust": "screen"}
        ))
        assert not r.error
        assert r.metadata["rows"] == 0
        assert r.metadata["dropped_low_trust"] == 1

    @pytest.mark.asyncio
    async def test_dataset_build_is_not_a_trajectory_step(self, tmp_path):
        """The audit trail records the build, but a journal action must never
        show up as an agent step — and, not being passive, it would otherwise
        earn a process reward for doing nothing."""
        from opendesk.computer.diagnostics import JOURNAL_ACTIONS
        from opendesk.computer.sandbox import ActionType, get_sandbox
        from opendesk.learning import trajectories
        from opendesk.tools.rollout import RolloutTool

        assert "dataset_build" in JOURNAL_ACTIONS
        ActionType("dataset_build")  # a str-enum that must accept it

        sid = "ds-journal"
        ctx = self._ctx(sid)
        await self._attempt(sid, "a", "T", 1.0, [("ui_action", {"name": "X"})])

        tool = RolloutTool()
        await tool.execute(ctx, tool.parse_params({
            "action": "dataset", "format": "sft",
            "path": str(tmp_path / "sft.jsonl"),
        }))

        # A trajectory built now must not contain the build as a step.
        ep = trajectories.begin_episode("after", session_id=sid,
                                        sandbox=get_sandbox(sid))
        sb = get_sandbox(sid)
        sb.current_screen = "fp-after"
        await sb.record_action(ActionType("ui_action"), {"name": "Y"})
        trajectories.end_episode(ep, session_id=sid, sandbox=sb,
                                 outcome={"reward": 1.0})
        traj = trajectories.build_trajectory(
            ep, entries=sb.export_audit_log(),
            process={"steps": [], "returns": [], "metrics": {}},
        )
        actions = [s["action"]["type"] for s in traj["steps"]]
        assert "dataset_build" not in actions
        assert actions == ["ui_action"]

