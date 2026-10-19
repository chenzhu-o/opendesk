"""Benchmark macOS accessibility-tree reads: AppleScript vs the native helper.

Times ``get_tree`` for each app two ways, median of N runs:

* ``osascript`` — the AppleScript ``entire contents`` path LocalComputer used
  before (one ``osascript`` process per call, capped at 200 elements);
* ``opendesk-ax`` — the persistent Swift helper with batched attribute reads
  (default limits: 1500 nodes).

Each app needs a window open.  ``--open`` opens a Finder window on
/Applications and a blank TextEdit document first, and closes them (without
saving) afterwards.

Usage (from ``python/``)::

    python scripts/bench_ax.py                 # Finder + TextEdit, 5 runs
    python scripts/bench_ax.py --open --runs 7
    python scripts/bench_ax.py --apps Finder Safari
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opendesk.computer import macos_native  # noqa: E402
from opendesk.computer.local import LocalComputer  # noqa: E402


def _count(el) -> int:
    return 1 + sum(_count(c) for c in el.children)


def _time(fn: Callable[[], int], runs: int) -> tuple[float, int]:
    times: list[float] = []
    n = 0
    for _ in range(runs):
        t = time.perf_counter()
        n = fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000, n


def _osascript(script: str) -> None:
    subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=30)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apps", nargs="+", default=["Finder", "TextEdit"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--open", action="store_true", help="open a Finder window and a TextEdit document first")
    args = ap.parse_args()

    if platform.system() != "Darwin":
        print("macOS only.")
        return 1

    t = time.perf_counter()
    if not macos_native.available():
        print("opendesk-ax helper unavailable (needs swiftc and Accessibility permission).")
        return 1
    print(f"helper ready in {time.perf_counter() - t:.2f}s (includes the one-time compile, if any)")

    if args.open:
        _osascript('tell application "Finder" to make new Finder window to (POSIX file "/Applications")')
        _osascript('tell application "TextEdit" to make new document')
        time.sleep(1.5)

    lc = LocalComputer()
    rows = []
    try:
        for app in args.apps:
            osa_ms, osa_n = _time(lambda: _count(lc._macos_ui_tree(app, 8)), args.runs)
            ax_ms, ax_n = _time(lambda: len(macos_native.tree(app=app)["nodes"]), args.runs)
            # Same element budget as the AppleScript path, for a like-for-like read.
            ax200_ms, ax200_n = _time(
                lambda: len(macos_native.tree(app=app, max_nodes=200)["nodes"]), args.runs,
            )
            rows.append((app, osa_ms, osa_n, ax_ms, ax_n, ax200_ms, ax200_n))
    finally:
        if args.open:
            _osascript('tell application "Finder" to close Finder window 1')
            _osascript('tell application "TextEdit" to close document 1 saving no')

    print(f"\nmedian of {args.runs} runs (ms; element count in parentheses)\n")
    print(f"{'app':<12} {'osascript':>16} {'opendesk-ax':>16} {'ax @200 nodes':>16} {'speedup':>9}")
    for app, osa_ms, osa_n, ax_ms, ax_n, ax200_ms, ax200_n in rows:
        speed = osa_ms / ax200_ms if ax200_ms else float("inf")
        print(
            f"{app:<12} {osa_ms:>9.0f} ({osa_n:>4}) {ax_ms:>9.0f} ({ax_n:>4}) "
            f"{ax200_ms:>9.0f} ({ax200_n:>4}) {speed:>8.0f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
