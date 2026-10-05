"""An evaluation harness for the learning layer — runnable end to end, no GUI.

What this is, and what it is not
--------------------------------

It is the **environment half** of an RL loop, exercised for real: every task here
declares machine-checkable success criteria, every attempt produces a real audit
log, real step-level process rewards, real agent declarations, and real
best-of-N preference pairs. Nothing is stubbed except the policy.

It is **not** a model benchmark. The "agent" below is a scripted list of actions,
so the numbers say nothing about how well a model drives a desktop. That is the
point: opendesk does not ship a policy, and a harness that pretended to measure
one would be measuring the script. What it demonstrates is that the reward,
process and export machinery works coherently, deterministically, and against
observable state — which is what a training pipeline actually consumes.

Why the tasks are file tasks
----------------------------

They run anywhere, on any OS, in a temp directory, with no display, which makes
this something you can run in CI and diff over time rather than a demo that only
works on the author's laptop. That also means the actions go through the `system`
tool's filesystem operations rather than a shell string: `echo` and redirection
are not portable — on Windows PowerShell they write UTF-16 and `rm -f` is an
ambiguous parameter — and a harness whose *fixtures* are platform-dependent
cannot report on anything else.

The same machinery is what the GUI tools feed. The checks are the same check
kinds, the audit log is the same audit log, and `diagnose` falls back to
inferring effects from the actions themselves when there are no screenshots —
the documented path for a CLI-only session.

Running it
----------

    python examples/learning-eval/run.py
    python examples/learning-eval/run.py --verbose
    python examples/learning-eval/run.py --export runs/
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Allow running straight from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from opendesk.computer.sandbox import clear_sandbox  # noqa: E402
from opendesk.learning import assertions, preference, trajectories  # noqa: E402
from opendesk.tools.base import allow_all_context  # noqa: E402
from opendesk.tools.reward import RewardTool  # noqa: E402
from opendesk.tools.rollout import RolloutTool  # noqa: E402
from opendesk.tools.system import SystemTool  # noqa: E402


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """One evaluable task: what "done" means, and what a plausible run does.

    ``checks`` is the verification surface and is kept entirely separate from
    ``attempt``.  The scripted attempt does not know what it is graded on, which
    is the property that makes the reward meaningful rather than circular — the
    same split a real training loop has.
    """

    name: str
    goal: str
    #: Checks evaluated at episode end. A ``path`` is relative to the attempt
    #: directory and is made absolute before evaluation.
    checks: list[dict[str, Any]]
    #: Actions the scripted agent takes. ``(tool, params, claim)`` where claim is
    #: an optional ``(name, text, checks)`` the agent declares after the action.
    attempt: Callable[[Path, int], list[tuple]]


def _w(path: Path, content: str) -> tuple:
    return ("system", {"action": "write_file", "path": str(path), "content": content})


def _claim(name: str, text: str, *checks: dict) -> tuple:
    return ("reward", {"action": "assert", "name": name, "claim": text,
                       "checks": list(checks)})


def _exists(path: Path) -> dict:
    return {"kind": "file_exists", "path": str(path)}


# --- the tasks -------------------------------------------------------------

def task_write_report(workdir: Path, n: int) -> list[tuple]:
    report = workdir / "report.txt"
    return [
        _w(report, "total: 42\n"),
        _claim("report exists", "the report file is written", _exists(report)),
        _claim("report has a total", "the report contains a total line",
               {"kind": "file_contains", "path": str(report),
                "regex": r"^total: \d+$"}),
    ]


def task_rename_log(workdir: Path, n: int) -> list[tuple]:
    log, bak = workdir / "app.log", workdir / "app.log.bak"
    return [
        _w(log, "old\n"),
        ("system", {"action": "move", "path": str(log), "dst": str(bak)}),
        _claim("log rotated", "the log was renamed to .bak",
               {"kind": "file_absent", "path": str(log)}, _exists(bak)),
    ]


def task_two_files(workdir: Path, n: int) -> list[tuple]:
    """A task whose second file only appears on some attempts.

    An intentionally unreliable scripted policy is what makes the best-of-N and
    preference path real: without a spread of outcomes every attempt ties and the
    pairing has nothing to work with.
    """
    a, b = workdir / "a.txt", workdir / "b.txt"
    actions: list[tuple] = [
        _w(a, "a\n"),
        _claim("a.txt written", "the first file is written", _exists(a)),
    ]
    if n % 3 != 0:                       # one attempt in three forgets
        actions.append(_w(b, "b\n"))
    return actions


def task_regression(workdir: Path, n: int) -> list[tuple]:
    """A run whose own claim is falsified by a later action.

    The agent says the output is in place, then a clean-up step removes it. The
    terminal reward sees a missing file; only the assertion track can say the
    agent believed otherwise — which is the finding a trainer cannot get
    elsewhere.
    """
    out = workdir / "out.dat"
    return [
        _w(out, "data\n"),
        _claim("output written", "out.dat is in place", _exists(out)),
        ("system", {"action": "delete", "path": str(out)}),
    ]


TASKS: list[Task] = [
    Task(
        name="write-report",
        goal="Write a one-line report containing a total",
        checks=[{"kind": "file_exists", "path": "report.txt"},
                {"kind": "file_contains", "path": "report.txt",
                 "regex": r"^total: \d+$"}],
        attempt=task_write_report,
    ),
    Task(
        name="rename-log",
        goal="Rotate app.log to app.log.bak",
        checks=[{"kind": "file_absent", "path": "app.log"},
                {"kind": "file_exists", "path": "app.log.bak"}],
        attempt=task_rename_log,
    ),
    Task(
        name="two-files",
        goal="Write a.txt and b.txt",
        checks=[{"kind": "file_exists", "path": "a.txt"},
                {"kind": "file_exists", "path": "b.txt"}],
        attempt=task_two_files,
    ),
    Task(
        name="regression",
        goal="Leave out.dat in place",
        checks=[{"kind": "file_exists", "path": "out.dat"}],
        attempt=task_regression,
    ),
]


# ---------------------------------------------------------------------------
# Running one attempt
# ---------------------------------------------------------------------------

@dataclass
class AttemptResult:
    task: str
    attempt: int
    episode_id: str
    reward: float
    score: float
    steps: int
    efficiency: float
    errors: int
    declared: int
    held: int
    regressions: int
    precision: float
    effects: str = ""
    note: str = ""


async def run_attempt(
    task: Task,
    attempt_no: int,
    *,
    root: Path,
    reward_tool: RewardTool,
    system_tool: SystemTool,
    verbose: bool,
) -> tuple[AttemptResult, Any]:
    session = f"eval:{task.name}:{attempt_no}"
    workdir = root / task.name / f"attempt{attempt_no}"
    workdir.mkdir(parents=True, exist_ok=True)

    clear_sandbox(session)
    trajectories.clear_episodes(session)
    assertions.clear_assertions(session)
    ctx = allow_all_context(session)

    # Relative paths in the spec resolve against this attempt's directory, so
    # one task definition works from any root.
    spec = {
        "task": task.goal,
        "checks": [
            {**c, **({"path": str(workdir / c["path"])} if "path" in c else {})}
            for c in task.checks
        ],
    }

    await reward_tool.execute(ctx, reward_tool.parse_params(
        {"action": "begin", "task": task.goal, "spec": spec}
    ))

    for tool, params in task.attempt(workdir, attempt_no):
        target = system_tool if tool == "system" else reward_tool
        await target.execute(ctx, target.parse_params(params))

    closed = await reward_tool.execute(ctx, reward_tool.parse_params({"action": "end"}))

    episode = trajectories.latest_episode(session)
    process = episode.process or {}
    metrics = process.get("metrics") or {}
    m = (closed.metadata.get("assertions") or {}).get("metrics") or {}

    # Report what actually drove the step rewards, rather than re-deriving it.
    # There are no screenshots here, so `score_process` falls back to inferring
    # effects from the actions themselves and says so via `signal_source` — the
    # documented path for a CLI-only session, and worth surfacing so a downgrade
    # to a weaker signal is never mistaken for a real observation.
    effects = ",".join(s["effect"] for s in (process.get("steps") or []))

    result = AttemptResult(
        task=task.name,
        attempt=attempt_no,
        episode_id=episode.id,
        reward=float(closed.metadata.get("reward", 0.0)),
        score=float((closed.metadata.get("outcome") or {}).get("score", 0.0)),
        steps=int(metrics.get("total_steps", 0)),
        efficiency=float(metrics.get("efficiency", 0.0)),
        errors=int(metrics.get("error_steps", 0)),
        declared=int(m.get("declared", 0)),
        held=int(m.get("held", 0)),
        regressions=int(m.get("regressed", 0)),
        precision=float(m.get("precision", 1.0)),
        effects=effects if verbose else "",
        note=f"signal={metrics.get('signal_source', '?')}" if verbose else "",
    )
    return result, episode


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

async def run_suite(
    tasks: list[Task], attempts: int, *, root: Path, verbose: bool,
) -> tuple[list[AttemptResult], list[Any]]:
    reward_tool, system_tool = RewardTool(), SystemTool()
    results: list[AttemptResult] = []
    episodes: list[Any] = []
    for task in tasks:
        for n in range(attempts):
            result, episode = await run_attempt(
                task, n, root=root, reward_tool=reward_tool,
                system_tool=system_tool, verbose=verbose,
            )
            results.append(result)
            episodes.append(episode)
            mark = "PASS" if result.reward >= 1.0 else "fail"
            extra = f"  effects={result.effects} {result.note}" if verbose else ""
            print(f"  [{mark}] {task.name:<14} attempt {n}  "
                  f"steps={result.steps} eff={result.efficiency:.0%} "
                  f"claims={result.held}/{result.declared} "
                  f"regressed={result.regressions}{extra}")
    return results, episodes


def summarize(results: list[AttemptResult]) -> str:
    by_task: dict[str, list[AttemptResult]] = {}
    for r in results:
        by_task.setdefault(r.task, []).append(r)

    header = (f"{'task':<16}{'pass':>7}{'score':>8}{'steps':>7}"
              f"{'eff':>6}{'prec':>7}{'regr':>6}")
    lines = ["", header, "-" * len(header)]
    for name, rs in by_task.items():
        passed = sum(1 for r in rs if r.reward >= 1.0)
        # Precision is pooled over claims, not averaged over attempts: an attempt
        # with three claims is more evidence than one with a single claim, and
        # averaging by attempt would let one cautious attempt outweigh it.
        declared = sum(r.declared for r in rs)
        held = sum(r.held for r in rs)
        lines.append(
            f"{name:<16}{f'{passed}/{len(rs)}':>7}"
            f"{sum(r.score for r in rs) / len(rs):>7.0%}"
            f"{sum(r.steps for r in rs) / len(rs):>7.1f}"
            f"{sum(r.efficiency for r in rs) / len(rs):>6.0%}"
            f"{(held / declared if declared else 1.0):>7.0%}"
            f"{sum(r.regressions for r in rs):>6}"
        )

    total_pass = sum(1 for r in results if r.reward >= 1.0)
    declared = sum(r.declared for r in results)
    held = sum(r.held for r in results)
    lines += [
        "-" * len(header),
        f"  {total_pass}/{len(results)} attempts passed  |  "
        f"{sum(r.steps for r in results)} step(s) total",
        f"  {declared} claim(s) declared, {held} held "
        f"({(held / declared if declared else 1.0):.0%} precision), "
        f"{sum(r.regressions for r in results)} regressed",
        "",
        "  Reward comes only from each task's own spec. Claims belong to the",
        "  agent and never score the task — they measure how well it knows its",
        "  own state, which is what 'prec' reports. 'regr' counts claims that",
        "  held when made and were false at the end.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Best-of-N
# ---------------------------------------------------------------------------

def build_preferences(episodes: list[Any]) -> str:
    """Rank the run's attempts and pair them, using the real preference code."""
    by_task: dict[str, list[Any]] = {}
    for ep in episodes:
        by_task.setdefault(ep.task, []).append(ep)

    lines: list[str] = []
    for goal, eps in by_task.items():
        if len(eps) < 2:
            continue
        rollouts = [preference.Rollout.from_episode(ep) for ep in eps]
        dataset = preference.build_pairs(rollouts, strategy="best_vs_worst")
        label = goal[:44]
        if not dataset.pairs:
            lines.append(f"  {label:<46} {len(eps)} attempt(s), all tied — "
                         f"no informative pair")
            continue
        for pair in dataset.pairs:
            lines.append(
                f"  {label:<46} 1 pair [{pair.criterion}]  "
                f"chosen reward={pair.chosen.reward:.0f}/"
                f"{pair.chosen.steps} step(s)  vs  "
                f"rejected {pair.rejected.reward:.0f}/"
                f"{pair.rejected.steps} step(s)"
            )
    return "\n".join(lines) if lines else "  (no task produced two comparable attempts)"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace, tasks: list[Task]) -> int:
    root = (Path(args.root).resolve() if args.root
            else Path(tempfile.mkdtemp(prefix="opendesk-eval-")))
    root.mkdir(parents=True, exist_ok=True)

    print("opendesk learning-layer evaluation")
    print(f"  {len(tasks)} task(s) x {args.attempts} attempt(s)")
    print(f"  working directory: {root}")
    print()

    results, episodes = await run_suite(
        tasks, args.attempts, root=root, verbose=args.verbose,
    )
    print(summarize(results))

    print()
    print("Best-of-N preference pairs (ranked by verifiable reward):")
    print(build_preferences(episodes))

    if args.export:
        dest = Path(args.export)
        dest.mkdir(parents=True, exist_ok=True)
        rollout_tool = RolloutTool()
        exported = 0
        for r, episode in zip(results, episodes):
            ctx = allow_all_context(f"eval:{r.task}:{r.attempt}")
            out = dest / f"{r.task}-{r.attempt}.jsonl"
            res = await rollout_tool.execute(ctx, rollout_tool.parse_params(
                {"action": "export", "episode_id": episode.id, "path": str(out)}
            ))
            if not res.error:
                exported += 1
        print()
        print(f"Exported {exported} trajectory file(s) to {dest}")

    if args.keep:
        print(f"\nKept working directory: {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--attempts", type=int, default=3,
                   help="attempts per task (default: 3)")
    p.add_argument("--root", default=None,
                   help="working directory for attempts (default: a temp dir)")
    p.add_argument("--keep", action="store_true",
                   help="keep the working directory instead of deleting it")
    p.add_argument("--verbose", action="store_true",
                   help="also print per-attempt effects and diagnosis signal")
    p.add_argument("--export", default=None,
                   help="write one trajectory JSONL per attempt to this directory")
    p.add_argument("--filter", default=None,
                   help="only run tasks whose name contains this substring")
    args = p.parse_args()

    tasks = TASKS
    if args.filter:
        tasks = [t for t in tasks if args.filter in t.name]
        if not tasks:
            print(f"no task matches {args.filter!r}", file=sys.stderr)
            return 1

    return asyncio.run(main_async(args, tasks))


if __name__ == "__main__":
    raise SystemExit(main())
