"""Markdown reports from one or more eval results files (A/B arms side by side)."""

from __future__ import annotations

import statistics
from collections import OrderedDict
from typing import Any, Iterable, Optional


def _arm(data: dict) -> str:
    return f"{data.get('backend')} · {data.get('model')}"


def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return statistics.fmean(vals) if vals else None


def _sum(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return sum(vals) if vals else None


def _fmt(x: Optional[float], spec: str = ".1f", dash: str = "–") -> str:
    return dash if x is None else format(x, spec)


def _rate(rows: list[dict]) -> tuple[int, int]:
    counted = [r for r in rows if r.get("status") != "skipped"]
    return sum(1 for r in counted if r.get("success")), len(counted)


def _stats(rows: list[dict]) -> dict[str, Any]:
    ok, n = _rate(rows)
    ran = [r for r in rows if r.get("status") != "skipped"]
    return {
        "ok": ok, "n": n,
        "rate": (ok / n) if n else None,
        "turns": _mean(r.get("turns") for r in ran),
        "wall": _mean(r.get("wall_time_s") for r in ran),
        "tok_in": _mean(r.get("input_tokens") for r in ran),
        "tok_out": _mean(r.get("output_tokens") for r in ran),
        "cost": _sum(r.get("cost_usd") for r in ran),
    }


def render_report(datasets: list[dict]) -> str:
    """Render a markdown report for one or more results dicts."""
    lines: list[str] = ["# OpenDesk eval report", ""]
    suites = sorted({d.get("suite", "") for d in datasets})
    lines.append(f"Suite(s): {', '.join(suites)}  ")
    hosts = sorted({d.get("host", "") for d in datasets if d.get("host")})
    if hosts:
        lines.append(f"Host(s): {', '.join(hosts)}")
    lines.append("")

    # --- summary per arm ------------------------------------------------
    arms: "OrderedDict[str, list[dict]]" = OrderedDict()
    for d in datasets:
        arms.setdefault(_arm(d), []).extend(d.get("results", []))

    lines += [
        "## Summary", "",
        "| Arm | Success | Mean turns | Mean wall (s) | Mean tokens in | Mean tokens out | Total cost ($) |",
        "|---|---|---|---|---|---|---|",
    ]
    arm_stats = {}
    for arm, rows in arms.items():
        s = _stats(rows)
        arm_stats[arm] = s
        rate = f"{s['ok']}/{s['n']} ({_fmt(s['rate'] * 100 if s['rate'] is not None else None, '.0f')}%)"
        lines.append(
            f"| {arm} | {rate} | {_fmt(s['turns'])} | {_fmt(s['wall'])} | "
            f"{_fmt(s['tok_in'], ',.0f')} | {_fmt(s['tok_out'], ',.0f')} | {_fmt(s['cost'], '.2f')} |"
        )
    if len(arm_stats) == 2:
        (a, sa), (b, sb) = list(arm_stats.items())
        if sa["rate"] is not None and sb["rate"] is not None:
            lines += ["", f"Δ success ({a} − {b}): {(sa['rate'] - sb['rate']) * 100:+.1f} pts"]
    lines.append("")

    # --- per task -------------------------------------------------------
    task_ids: list[str] = []
    for rows in arms.values():
        for r in rows:
            if r["task_id"] not in task_ids:
                task_ids.append(r["task_id"])
    lines += ["## Per task", ""]
    header = "| Task |" + "".join(f" {arm} |" for arm in arms)
    lines += [header, "|---|" + "---|" * len(arms)]
    for tid in task_ids:
        cells = []
        for rows in arms.values():
            trs = [r for r in rows if r["task_id"] == tid]
            if not trs:
                cells.append(" – ")
                continue
            if all(r.get("status") == "skipped" for r in trs):
                cells.append(" skipped ")
                continue
            s = _stats(trs)
            statuses = sorted({r.get("status") for r in trs if not r.get("success")} - {"skipped"})
            fail_note = f" ({', '.join(statuses)})" if statuses else ""
            cells.append(f" {s['ok']}/{s['n']} · {_fmt(s['turns'])} turns · {_fmt(s['wall'])}s{fail_note} ")
        lines.append(f"| {tid} |" + "|".join(cells) + "|")
    lines.append("")

    # --- per tag ---------------------------------------------------------
    tags = sorted({t for rows in arms.values() for r in rows for t in r.get("tags", [])})
    if tags:
        lines += ["## Per tag", "", "| Tag |" + "".join(f" {arm} |" for arm in arms),
                  "|---|" + "---|" * len(arms)]
        for tag in tags:
            cells = []
            for rows in arms.values():
                ok, n = _rate([r for r in rows if tag in r.get("tags", [])])
                cells.append(f" {ok}/{n} " if n else " – ")
            lines.append(f"| {tag} |" + "|".join(cells) + "|")
        lines.append("")

    # --- failures ----------------------------------------------------------
    failures = [
        (arm, r) for arm, rows in arms.items() for r in rows
        if not r.get("success") and r.get("status") != "skipped"
    ]
    if failures:
        lines += ["## Failures", ""]
        for arm, r in failures:
            why = r.get("error") or (f"check exited {r.get('check_returncode')}"
                                     if r.get("check_returncode") is not None else r.get("status"))
            why = " ".join(str(why).split())[:200]
            lines.append(f"- **{r['task_id']}** #{r.get('trial')} [{arm}] {r.get('status')}: {why}")
        lines.append("")
    return "\n".join(lines)
