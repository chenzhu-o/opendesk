"""Agent-declared state assertions — the agent's own claim, verified.

The behaviour worth pinning down is not "a claim was recorded" but the two things
recording it buys: a claim is checked when it is made *and* again when the episode
closes, and the gap between those two verdicts is a regression.  Plus the rule
that makes the whole feature safe to hand an agent: a claim never grants reward.
"""

from __future__ import annotations

import json

import pytest

from opendesk.computer.sandbox import ActionType, clear_sandbox, get_sandbox
from opendesk.learning import assertions, trajectories
from opendesk.tools.base import ToolContext
from opendesk.tools.reward import RewardTool
from opendesk.tools.rollout import RolloutTool
from tests._fakes import FakeComputer

FP0 = "aaaa0000bbbb1111"
FP1 = "cccc2222dddd3333"


def ctx_for(session_id: str) -> ToolContext:
    return ToolContext(session_id=session_id, computer=FakeComputer())


def fresh(session_id: str) -> ToolContext:
    clear_sandbox(session_id)
    trajectories.clear_episodes(session_id)
    assertions.clear_assertions(session_id)
    return ctx_for(session_id)


class TreeComputer(FakeComputer):
    """A fake whose accessibility tree the test can set."""

    def __init__(self, tree) -> None:
        super().__init__()
        self._tree = tree

    async def ui_tree(self, *, window_id=None, app=None, max_depth=8):
        return self._tree


def ui_tree(*children):
    from opendesk.computer.types import UIElement

    return UIElement(role="window", name="Invoices", children=list(children))


def ui_node(role: str, name: str = "", *, value=None):
    from opendesk.computer.types import UIElement

    return UIElement(role=role, name=name, value=value)


# ---------------------------------------------------------------------------
# Declaring
# ---------------------------------------------------------------------------


class TestDeclare:
    @pytest.mark.asyncio
    async def test_a_claim_is_verified_immediately(self, tmp_path):
        sid = "as-now"
        fresh(sid)
        target = tmp_path / "invoice.pdf"
        target.write_text("x", encoding="utf-8")

        a = await assertions.declare(
            "invoice downloaded", session_id=sid,
            checks=[{"kind": "file_exists", "path": str(target)}],
        )
        assert a.held

        missing = await assertions.declare(
            "receipt saved", session_id=sid,
            checks=[{"kind": "file_exists", "path": str(tmp_path / "no.pdf")}],
        )
        assert not missing.held
        assert missing.failed and missing.failed[0].check.kind == "file_exists"

    @pytest.mark.asyncio
    async def test_the_agents_own_words_are_kept_verbatim(self):
        sid = "as-words"
        fresh(sid)
        a = await assertions.declare(
            "reconciled", session_id=sid,
            checks=[{"kind": "shell", "command": "exit 0"}],
            claim="I reconciled the ledger against the bank feed",
        )
        assert a.claim == "I reconciled the ledger against the bank feed"
        assert "reconciled the ledger" in a.to_dict()["claim"]

    @pytest.mark.asyncio
    async def test_an_assertion_cannot_be_empty(self):
        sid = "as-empty"
        fresh(sid)
        with pytest.raises(ValueError):
            await assertions.declare("vibes", session_id=sid, checks=[])

    @pytest.mark.asyncio
    async def test_the_same_vocabulary_as_the_reward_spec(self):
        """A claim is checked with the same predicates a reward is.

        That is the point of the feature: the agent states its belief in terms the
        framework can already adjudicate, including accessibility content — so
        "the Total field reads 300" is a claim, not "the screen changed".
        """
        sid = "as-vocab"
        clear_sandbox(sid)
        assertions.clear_assertions(sid)
        ctx = ToolContext(
            session_id=sid,
            computer=TreeComputer(ui_tree(ui_node("AXTextField", "Total", value="300"))),
        )
        a = await assertions.declare(
            "total entered", session_id=sid, ctx=ctx,
            checks=[{"kind": "ui_element", "role": "textfield",
                     "name_regex": "Total", "value_regex": r"^300$"}],
        )
        assert a.held
        assert a.results[0].evidence["matches"][0]["value"] == "300"


# ---------------------------------------------------------------------------
# Re-checking and regressions
# ---------------------------------------------------------------------------


class TestRecheck:
    @pytest.mark.asyncio
    async def test_a_claim_that_stops_holding_is_a_regression(self, tmp_path):
        """The failure mode neither the terminal reward nor the step effects see.

        The agent saved the file (true when claimed), then a later action removed
        it. A terminal reward says "the file is missing"; it cannot say that the
        agent believed otherwise, which is the actual finding.
        """
        sid = "as-regress"
        fresh(sid)
        target = tmp_path / "export.csv"
        target.write_text("a,b\n", encoding="utf-8")

        a = await assertions.declare(
            "export written", session_id=sid,
            checks=[{"kind": "file_exists", "path": str(target)}],
        )
        assert a.held and a.held_late is None and not a.regressed

        target.unlink()
        await assertions.recheck(session_id=sid)

        assert a.held_late is False
        assert a.regressed
        assert assertions.report(sid).regressed == 1

    @pytest.mark.asyncio
    async def test_recheck_does_not_overwrite_an_earlier_verdict(self, tmp_path):
        """A re-check is a fact about the closing state, not a running poll.

        Re-checking twice would rebind the assertion to whatever the second call
        happened to see, which on a scratch file is a coin flip.
        """
        sid = "as-idem"
        fresh(sid)
        target = tmp_path / "scratch.txt"
        target.write_text("x", encoding="utf-8")
        a = await assertions.declare(
            "note written", session_id=sid,
            checks=[{"kind": "file_exists", "path": str(target)}],
        )

        await assertions.recheck(session_id=sid)
        target.unlink()
        assert await assertions.recheck(session_id=sid) == []   # nothing left to do
        assert a.held_late is True
        assert not a.regressed

    @pytest.mark.asyncio
    async def test_recheck_is_scoped_to_one_episode(self, tmp_path):
        """On best-of-N, several attempts share a session.

        Closing the second attempt must not re-check the first attempt's claims
        against the second attempt's state and invent regressions in it.
        """
        sid = "as-bon"
        fresh(sid)
        f = tmp_path / "shared.txt"
        f.write_text("x", encoding="utf-8")

        a1 = await assertions.declare(
            "first attempt saved", session_id=sid, episode_id="ep1",
            checks=[{"kind": "file_exists", "path": str(f)}],
        )
        a2 = await assertions.declare(
            "second attempt saved", session_id=sid, episode_id="ep2",
            checks=[{"kind": "file_exists", "path": str(f)}],
        )

        f.unlink()
        await assertions.recheck(session_id=sid, episode_id="ep2")

        assert a2.late_results is not None and a2.regressed
        assert a1.late_results is None and a1.held_late is None

    @pytest.mark.asyncio
    async def test_for_episode_filters_by_attempt(self):
        sid = "as-scope"
        fresh(sid)
        for ep in ("ep1", "ep1", "ep2"):
            await assertions.declare(
                f"step in {ep}", session_id=sid, episode_id=ep,
                checks=[{"kind": "shell", "command": "exit 0"}],
            )
        assert len(assertions.for_episode("ep1", sid)) == 2
        assert len(assertions.for_episode("ep2", sid)) == 1


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestReport:
    @pytest.mark.asyncio
    async def test_precision_measures_honesty_not_difficulty(self, tmp_path):
        sid = "as-precision"
        fresh(sid)
        good = tmp_path / "there.txt"
        good.write_text("x", encoding="utf-8")
        for held in (True, True, True, False):
            checks = ([{"kind": "file_exists", "path": str(good)}] if held
                      else [{"kind": "file_exists", "path": str(tmp_path / "gone.txt")}])
            await assertions.declare(f"claim-{held}", session_id=sid, checks=checks)

        rep = assertions.report(sid)
        assert rep.declared == 4 and rep.held == 3 and rep.failed_count == 1
        assert rep.precision == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_a_vacuous_session_is_not_reported_as_dishonest(self):
        sid = "as-vacuous"
        fresh(sid)
        rep = assertions.report(sid)
        assert rep.declared == 0 and rep.precision == 1.0 and rep.stability is None

    @pytest.mark.asyncio
    async def test_stability_needs_a_recheck_to_mean_anything(self, tmp_path):
        """None, not 1.0. An un-rechecked session is unknown, not perfect."""
        sid = "as-stab"
        fresh(sid)
        f = tmp_path / "f.txt"
        f.write_text("x", encoding="utf-8")
        await assertions.declare(
            "saved", session_id=sid,
            checks=[{"kind": "file_exists", "path": str(f)}],
        )
        assert assertions.report(sid).stability is None

        await assertions.recheck(session_id=sid)
        assert assertions.report(sid).stability == 1.0

    @pytest.mark.asyncio
    async def test_a_claim_false_on_arrival_cannot_regress(self, tmp_path):
        """Regression means "was true, became false".

        Counting a claim that never held as a regression would double-report one
        failure and overstate how brittle the run was.
        """
        sid = "as-nofalse"
        fresh(sid)
        await assertions.declare(
            "never true", session_id=sid,
            checks=[{"kind": "file_exists", "path": str(tmp_path / "x")}],
        )
        await assertions.recheck(session_id=sid)
        rep = assertions.report(sid)
        assert rep.failed_count == 1 and rep.regressed == 0
        assert rep.stability == 1.0   # nothing verified, so nothing broke

    @pytest.mark.asyncio
    async def test_summary_reads_as_a_report(self, tmp_path):
        sid = "as-summary"
        fresh(sid)
        f = tmp_path / "f.txt"
        f.write_text("x", encoding="utf-8")
        await assertions.declare(
            "saved", session_id=sid, claim="I saved it",
            checks=[{"kind": "file_exists", "path": str(f)}],
        )
        await assertions.declare(
            "sent", session_id=sid,
            checks=[{"kind": "file_exists", "path": str(tmp_path / "nope")}],
        )
        text = assertions.report(sid).summary_text()
        assert "2 declared" in text and "1 held" in text
        assert "I saved it" in text            # the agent's words survive
        assert "does not exist" in text        # and a failure says why


# ---------------------------------------------------------------------------
# Trajectory milestones
# ---------------------------------------------------------------------------


class TestAttachToSteps:
    def _mk(self, name: str, index: int, held: bool = True) -> assertions.Assertion:
        from opendesk.learning.rewards import Check, CheckResult

        c = Check("file_exists", {"path": "/x"}, required=True, weight=1.0)
        a = assertions.Assertion(name=name, checks=[c], index=index)
        a.results = [CheckResult(c, held, "detail")]
        return a

    def test_a_claim_binds_to_the_step_it_followed(self):
        # Steps at audit positions 3 and 9; a claim made at 10 followed step 9.
        got = assertions.attach_to_steps(
            [self._mk("after second", index=10)], [3, 9]
        )
        assert got == {9: [{"name": "after second", "held": True,
                            "claim": None, "regressed": False, "held_late": None}]}

    def test_a_claim_made_before_any_step_lands_on_the_first(self):
        """Dropping it would put a hole in the milestone track.

        A slightly loose first entry is easier to read than a claim that
        silently vanished.
        """
        got = assertions.attach_to_steps([self._mk("pre", index=1)], [5, 8])
        assert list(got) == [5]

    def test_no_steps_means_no_binding(self):
        assert assertions.attach_to_steps([self._mk("x", index=2)], []) == {}


class TestTrajectoryCarriesMilestones:
    @pytest.mark.asyncio
    async def test_milestones_land_on_steps_and_bookkeeping_stays_out(self, tmp_path):
        sid = "as-traj"
        ctx = fresh(sid)
        reward_tool, rollout_tool = RewardTool(), RolloutTool()
        sb = get_sandbox(sid)

        target = tmp_path / "report.pdf"
        await reward_tool.execute(ctx, reward_tool.parse_params(
            {"action": "begin", "task": "produce report.pdf",
             "spec": {"checks": [{"kind": "file_exists", "path": str(target)}]}}
        ))

        sb.current_screen = FP0
        await sb.record_action(ActionType.UI_ACTION, {"what": "click export"})

        a = await reward_tool.execute(ctx, reward_tool.parse_params(
            {"action": "assert", "name": "export clicked",
             "claim": "the export dialog was accepted",
             "checks": [{"kind": "shell", "command": "exit 0"}]}
        ))
        assert not a.error and "HOLDS" in a.output

        target.write_text("pdf", encoding="utf-8")
        sb.current_screen = FP1
        await sb.record_action(ActionType.UI_ACTION, {"what": "confirm"})

        await reward_tool.execute(ctx, reward_tool.parse_params({"action": "end"}))

        dest = tmp_path / "t.jsonl"
        r = await rollout_tool.execute(ctx, rollout_tool.parse_params(
            {"action": "export", "path": str(dest)}
        ))
        assert not r.error

        lines = [json.loads(ln) for ln in dest.read_text(encoding="utf-8").splitlines()]
        header, steps = lines[0], lines[1:]

        # The claim is not an agent action, so it is not a step ...
        assert len(steps) == 2
        assert all(s["action"]["type"] != "assertion" for s in steps)
        # ... but it is not lost either: it rides on the step it followed.
        assert steps[0]["assertions"] == [{
            "name": "export clicked", "held": True,
            "claim": "the export dialog was accepted",
            "regressed": False, "held_late": True,
        }]
        assert steps[1]["assertions"] == []
        assert header["assertions"]["metrics"]["declared"] == 1

    @pytest.mark.asyncio
    async def test_a_regression_reaches_the_trajectory(self, tmp_path):
        """A regression that only lives in a report is not training data."""
        sid = "as-traj-reg"
        ctx = fresh(sid)
        reward_tool, rollout_tool = RewardTool(), RolloutTool()
        target = tmp_path / "out.txt"
        target.write_text("x", encoding="utf-8")

        await reward_tool.execute(ctx, reward_tool.parse_params(
            {"action": "begin", "task": "keep out.txt"}
        ))
        await reward_tool.execute(ctx, reward_tool.parse_params(
            {"action": "assert", "name": "out.txt kept",
             "checks": [{"kind": "file_exists", "path": str(target)}]}
        ))
        target.unlink()
        await reward_tool.execute(ctx, reward_tool.parse_params({"action": "end"}))

        dest = tmp_path / "t.jsonl"
        await rollout_tool.execute(ctx, rollout_tool.parse_params(
            {"action": "export", "path": str(dest)}
        ))
        header = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
        assert header["assertions"]["metrics"]["regressed"] == 1


# ---------------------------------------------------------------------------
# The safety rule
# ---------------------------------------------------------------------------


class TestClaimsAreNotRewards:
    @pytest.mark.asyncio
    async def test_a_claim_cannot_score_a_task(self, tmp_path):
        """An agent picks its own claims, so they can never grant reward.

        Otherwise the optimal policy is to assert something trivially true and
        collect: the checks would be grading the agent's question, not its answer.
        Reward comes only from the task spec, whose checks the agent does not
        author.
        """
        sid = "as-safe"
        ctx = fresh(sid)
        tool = RewardTool()
        never = tmp_path / "never.txt"

        await tool.execute(ctx, tool.parse_params(
            {"action": "begin", "task": "create never.txt",
             "spec": {"checks": [{"kind": "file_exists", "path": str(never)}]}}
        ))

        # A claim the agent can always satisfy — and which it makes loudly.
        r = await tool.execute(ctx, tool.parse_params(
            {"action": "assert", "name": "I am definitely done",
             "checks": [{"kind": "shell", "command": "exit 0"}]}
        ))
        assert "HOLDS" in r.output

        r = await tool.execute(ctx, tool.parse_params({"action": "end"}))
        # The task's own check still fails: the claim bought nothing.
        assert r.metadata["reward"] == 0.0
        assert r.metadata["outcome"]["passed"] is False
        assert "FAIL" in r.output
        # ... and it is still reported, because it is still evidence about
        # the agent, just not about the task.
        assert r.metadata["assertions"]["metrics"]["held"] == 1


# ---------------------------------------------------------------------------
# The tool surface
# ---------------------------------------------------------------------------


class TestAssertTool:
    @pytest.mark.asyncio
    async def test_assert_needs_a_name(self):
        ctx = fresh("at-name")
        tool = RewardTool()
        r = await tool.execute(ctx, tool.parse_params(
            {"action": "assert", "checks": [{"kind": "shell", "command": "exit 0"}]}
        ))
        assert r.error and "name" in r.output

    @pytest.mark.asyncio
    async def test_assert_needs_checks(self):
        """An unfalsifiable claim is not an assertion."""
        ctx = fresh("at-checks")
        tool = RewardTool()
        r = await tool.execute(ctx, tool.parse_params(
            {"action": "assert", "name": "vibes"}
        ))
        assert r.error and "checks" in r.output

    @pytest.mark.asyncio
    async def test_assert_rejects_a_bad_check_kind(self):
        ctx = fresh("at-bad")
        tool = RewardTool()
        r = await tool.execute(ctx, tool.parse_params(
            {"action": "assert", "name": "x", "checks": [{"kind": "teleport"}]}
        ))
        assert r.error and "teleport" in r.output

    @pytest.mark.asyncio
    async def test_a_failed_claim_says_what_was_wrong(self, tmp_path):
        ctx = fresh("at-fail")
        tool = RewardTool()
        r = await tool.execute(ctx, tool.parse_params(
            {"action": "assert", "name": "file saved",
             "checks": [{"kind": "file_exists", "path": str(tmp_path / "nope")}]}
        ))
        assert not r.error                      # a false claim is not a tool error
        assert "does not hold" in r.title
        assert "DOES NOT HOLD" in r.output
        assert "does not exist" in r.output
        assert r.metadata["held"] is False

    @pytest.mark.asyncio
    async def test_assertions_lists_calibration(self, tmp_path):
        sid = "at-list"
        ctx = fresh(sid)
        tool = RewardTool()
        await tool.execute(ctx, tool.parse_params(
            {"action": "assert", "name": "ok",
             "checks": [{"kind": "shell", "command": "exit 0"}]}
        ))
        await tool.execute(ctx, tool.parse_params(
            {"action": "assert", "name": "bad",
             "checks": [{"kind": "file_exists", "path": str(tmp_path / "no")}]}
        ))
        r = await tool.execute(ctx, tool.parse_params({"action": "assertions"}))
        assert not r.error
        assert "2 declared" in r.output and "1 held" in r.output
        assert "precision 50%" in r.output
        assert r.metadata["metrics"]["precision"] == 0.5

    @pytest.mark.asyncio
    async def test_assertions_is_empty_before_anything_is_claimed(self):
        ctx = fresh("at-empty")
        tool = RewardTool()
        r = await tool.execute(ctx, tool.parse_params({"action": "assertions"}))
        assert not r.error and "No assertions declared" in r.output
