#!/usr/bin/env python3
"""Write a harness evolution report from trajectory JSONL (review only — no auto-patch)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opendesk.learning import training  # noqa: E402
from opendesk.learning.harness_evolution import (  # noqa: E402
    HarnessProfile,
    analyze_episodes,
    export_report,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trajectories", help="JSONL file or directory of episodes")
    p.add_argument("-o", "--output", default="harness_report.json")
    p.add_argument("--max-tree-chars", type=int, default=12800)
    args = p.parse_args()

    eps = training.load_trajectories(args.trajectories)

    def legacy(ep):
        if ep.get("_response"):
            return ep["_response"]
        steps = ep.get("steps") or []
        if not steps:
            return None
        act = steps[-1].get("action")
        if isinstance(act, dict) and act.get("params", {}).get("code"):
            return act["params"]["code"]
        return None

    profile = HarnessProfile(max_tree_chars=args.max_tree_chars)
    report = analyze_episodes(eps, profile=profile, legacy_resolver=legacy)
    export_report(report, args.output)
    print("Wrote %s (+ .md) — %d suggestion(s)" % (
        args.output, len(report.suggestions)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
