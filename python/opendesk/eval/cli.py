"""``opendesk eval`` sub-commands (wired into :mod:`opendesk.cli`)."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path


def add_eval_parser(sub) -> None:
    from opendesk.eval.backends import BACKENDS, DEFAULT_MAX_TURNS, DEFAULT_MODEL
    from opendesk.eval.suites import BUILTIN_SUITES

    ev = sub.add_parser("eval", help="Run agent evaluation suites and report results")
    ev_sub = ev.add_subparsers(dest="eval_cmd")

    run_p = ev_sub.add_parser("run", help="Run a suite (built-in name, task JSON file, or directory)")
    run_p.add_argument("suite", help=f"Built-in ({', '.join(BUILTIN_SUITES)}) or path to tasks")
    run_p.add_argument("--backend", choices=sorted(BACKENDS), default="claude-code")
    run_p.add_argument("--model", default=DEFAULT_MODEL, help="Model id — keep it constant across arms")
    run_p.add_argument("--repeat", type=int, default=1, help="Trials per task")
    run_p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    run_p.add_argument("--task", action="append", default=[], help="Only run these task ids (repeatable)")
    run_p.add_argument("--tag", action="append", default=[], help="Only run tasks with this tag (repeatable)")
    run_p.add_argument("--project", default=None,
                       help="learn-regression: also run every compiled program in DIR/.opendesk/learned")
    run_p.add_argument("--native-tool", default="computer_toolset_20260801",
                       help="anthropic-native-computer: computer_toolset_20260801 (default) or computer_20251124")
    run_p.add_argument("--out", default=None, help="Results JSON path (default: eval-results/<suite>-<backend>-<ts>.json)")
    run_p.add_argument("--keep-workdir", action="store_true", help="Do not delete per-trial temp dirs")
    run_p.add_argument("--dry-run", action="store_true", help="Print the plan (commands, checks); execute nothing")
    run_p.add_argument("--json", action="store_true", help="With --dry-run: print the plan as JSON")

    rep_p = ev_sub.add_parser("report", help="Markdown table from one or more results files")
    rep_p.add_argument("results", nargs="+", help="results JSON file(s); several = side-by-side arms")
    rep_p.add_argument("--out", default=None, help="Write markdown here instead of stdout")

    ev_sub.add_parser("list", help="List built-in suites and their tasks")


def cmd_eval(args) -> None:
    cmd = getattr(args, "eval_cmd", None)
    if cmd == "run":
        sys.exit(_cmd_run(args))
    if cmd == "report":
        sys.exit(_cmd_report(args))
    if cmd == "list":
        sys.exit(_cmd_list())
    print("Usage: opendesk eval run|report|list  (see --help)")


def _select(tasks, ids, tags):
    if ids:
        tasks = [t for t in tasks if t.id in ids]
    if tags:
        tasks = [t for t in tasks if set(tags) & set(t.tags)]
    return tasks


def _cmd_run(args) -> int:
    from opendesk.eval.backends import get_backend
    from opendesk.eval.harness import EvalRunner, dry_run_plan, format_plan
    from opendesk.eval.task import load_tasks

    project = Path(args.project).expanduser() if args.project else None
    try:
        suite, tasks = load_tasks(args.suite, project_dir=project)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    tasks = _select(tasks, args.task, args.tag)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 2

    kw = {"model": args.model, "max_turns": args.max_turns}
    if args.backend == "anthropic-native-computer":
        kw["tool_version"] = args.native_tool
    backend = get_backend(args.backend, **kw)

    if args.dry_run:
        plan = dry_run_plan(tasks, backend, repeat=args.repeat)
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            print(f"DRY RUN — suite '{suite}', {len(tasks)} task(s) x {args.repeat}, "
                  f"backend={backend.name}, model={backend.model}, max_turns={backend.max_turns}")
            print("Nothing will be executed.\n")
            print(format_plan(plan))
        return 0

    def _progress(r) -> None:
        mark = {"pass": "PASS", "skipped": "SKIP"}.get(r.status, r.status.upper())
        extra = f" — {r.error[:120]}" if r.error and not r.success else ""
        turns = f", {r.turns} turns" if r.turns else ""
        print(f"  [{mark}] {r.task_id} #{r.trial} ({r.wall_time_s:.1f}s{turns}){extra}", flush=True)

    print(f"Running suite '{suite}': {len(tasks)} task(s) x {args.repeat} on "
          f"{backend.name} / {backend.model}")
    runner = EvalRunner(backend, on_trial=_progress, keep_workdir=args.keep_workdir)
    result = asyncio.run(runner.run(suite, tasks, repeat=args.repeat))

    out = Path(args.out) if args.out else (
        Path("eval-results") / f"{suite}-{backend.name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    result.save(out)
    summary = result.to_dict()["summary"]
    print(f"\n{summary['passed']}/{summary['trials']} passed. Results: {out}")
    print(f"Report: opendesk eval report {out}")
    return 0


def _cmd_report(args) -> int:
    from opendesk.eval.harness import load_results
    from opendesk.eval.report import render_report

    try:
        datasets = [load_results(Path(p)) for p in args.results]
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    md = render_report(datasets)
    if args.out:
        Path(args.out).write_text(md)
        print(f"Wrote {args.out}")
    else:
        print(md)
    return 0


def _cmd_list() -> int:
    from opendesk.eval.suites import BUILTIN_SUITES, builtin_suite

    for name in BUILTIN_SUITES:
        tasks = builtin_suite(name)
        print(f"{name} ({len(tasks)} tasks)")
        for t in tasks:
            plat = f" [{t.platform}]" if t.platform else ""
            print(f"  {t.id}{plat}  — {', '.join(t.tags)}")
    return 0
