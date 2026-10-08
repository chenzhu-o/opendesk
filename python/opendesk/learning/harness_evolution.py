"""Harness diagnosis and *suggested* profile changes — no auto-apply.

Machines collect evidence from failed or weak episodes; humans (or a gated PR)
merge ``HarnessProfile`` updates.  Execution, reward specs, and sandbox policy
are never patched silently.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from opendesk.learning.action_space import (
    MappedCompletion,
    map_legacy_completion,
    observation_tree_from_episode,
)


@dataclass
class HarnessProfile:
    """Bounded, reviewable harness knobs (YAML/JSON serialisable)."""

    max_tree_chars: int = 12800
    max_prompt_tokens: Optional[int] = None
    escalate_tier_on_no_effect: bool = True
    max_no_effect_before_som: int = 2
    pointer_fallback_loss_weight: float = 0.35
    prefer_ui_over_coordinates: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessProfile":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessSuggestion:
    """One evidence-backed change proposal for human review."""

    suggestion_id: str
    category: str
    severity: str
    evidence: dict[str, Any]
    suggested_profile: dict[str, Any]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessEvolutionReport:
    episodes_analyzed: int
    mapping_summary: dict[str, int] = field(default_factory=dict)
    process_summary: dict[str, Any] = field(default_factory=dict)
    suggestions: list[HarnessSuggestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes_analyzed": self.episodes_analyzed,
            "mapping_summary": self.mapping_summary,
            "process_summary": self.process_summary,
            "suggestions": [s.to_dict() for s in self.suggestions],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Harness evolution report",
            "",
            "Episodes analyzed: **%d**" % self.episodes_analyzed,
            "",
            "## Action-space mapping",
            "",
        ]
        for k, v in sorted(self.mapping_summary.items()):
            lines.append("- %s: %d" % (k, v))
        lines.extend(["", "## Process signals", ""])
        for k, v in self.process_summary.items():
            lines.append("- %s: %s" % (k, v))
        lines.extend(["", "## Suggested profile patches (review before merge)", ""])
        if not self.suggestions:
            lines.append("_No suggestions — insufficient evidence._")
        for s in self.suggestions:
            lines.extend([
                "",
                "### %s (%s, %s)" % (s.suggestion_id, s.category, s.severity),
                "",
                s.rationale,
                "",
                "```json",
                json.dumps(s.suggested_profile, indent=2),
                "```",
                "",
                "Evidence:",
                "```json",
                json.dumps(s.evidence, indent=2),
                "```",
            ])
        lines.append("")
        lines.append(
            "_Apply by editing HarnessProfile and opening a PR — "
            "this module does not modify runtime code._"
        )
        return "\n".join(lines)


def _process_metrics(episode: dict) -> dict[str, Any]:
    return ((episode.get("process") or {}).get("metrics") or {})


def analyze_episodes(
    episodes: Sequence[dict],
    *,
    profile: Optional[HarnessProfile] = None,
    legacy_resolver: Optional[Any] = None,
) -> HarnessEvolutionReport:
    """Aggregate mapping + process evidence and emit reviewable suggestions."""
    profile = profile or HarnessProfile()
    mapping_counts: Counter[str] = Counter()
    no_effect = 0
    loops = 0
    low_efficiency_tasks: Counter[str] = Counter()
    pointer_retained = 0
    ambiguous = 0

    for ep in episodes:
        legacy = ""
        if legacy_resolver is not None:
            legacy = (legacy_resolver(ep) or "").strip()
        else:
            steps = ep.get("steps") or []
            if steps and isinstance(steps[0], dict):
                act = steps[0].get("action")
                if isinstance(act, dict):
                    legacy = str(act.get("params", {}).get("code") or "")
        tree = observation_tree_from_episode(ep)
        if legacy:
            mapped: MappedCompletion = map_legacy_completion(legacy, tree)
            mapping_counts[mapped.status] += 1
            if mapped.status == "retained_pointer":
                pointer_retained += 1
            if mapped.detail == "ambiguous_tree_match":
                ambiguous += 1

        m = _process_metrics(ep)
        if int(m.get("no_effect_steps") or 0) > 0:
            no_effect += 1
        if int(m.get("loop_steps") or 0) > 0:
            loops += 1
        eff = m.get("efficiency")
        if eff is not None and float(eff) < 0.5:
            task = str(ep.get("task") or "unknown")
            low_efficiency_tasks[task] += 1

    n = len(episodes)
    suggestions: list[HarnessSuggestion] = []

    if n and pointer_retained / max(n, 1) > 0.25:
        suggestions.append(HarnessSuggestion(
            suggestion_id="raise_tree_budget",
            category="observation",
            severity="medium",
            evidence={
                "retained_pointer_fraction": round(pointer_retained / n, 3),
                "mapping_counts": dict(mapping_counts),
            },
            suggested_profile={
                "max_tree_chars": min(profile.max_tree_chars * 2, 25600),
            },
            rationale=(
                "Many steps stay on coordinate fallback — targets may be "
                "truncated out of the tree. Increase observation budget and "
                "re-run mapping before training."
            ),
        ))

    if ambiguous >= 3:
        suggestions.append(HarnessSuggestion(
            suggestion_id="prefer_som_on_ambiguity",
            category="tier",
            severity="low",
            evidence={"ambiguous_clicks": ambiguous},
            suggested_profile={
                "escalate_tier_on_no_effect": True,
                "max_no_effect_before_som": max(1, profile.max_no_effect_before_som),
            },
            rationale=(
                "Multiple gold clicks match more than one tree row at tolerance — "
                "enable Tier-2 grounded SoM instead of forcing a semantic label."
            ),
        ))

    if no_effect >= max(2, n // 5):
        suggestions.append(HarnessSuggestion(
            suggestion_id="tier_escalation_after_no_effect",
            category="retry",
            severity="medium",
            evidence={"episodes_with_no_effect": no_effect, "total": n},
            suggested_profile={
                "escalate_tier_on_no_effect": True,
                "max_no_effect_before_som": profile.max_no_effect_before_som,
            },
            rationale=(
                "Repeated no-effect steps — consider escalating ui → SoM → "
                "pixel after stalled actions (harness policy only)."
            ),
        ))

    if loops >= 2:
        suggestions.append(HarnessSuggestion(
            suggestion_id="loop_breaker",
            category="retry",
            severity="high",
            evidence={"episodes_with_loops": loops},
            suggested_profile={"max_no_effect_before_som": 1},
            rationale="State loops detected — tighten retry / tier escalation.",
        ))

    top_bad = low_efficiency_tasks.most_common(3)
    if top_bad:
        suggestions.append(HarnessSuggestion(
            suggestion_id="review_low_efficiency_tasks",
            category="diagnosis",
            severity="info",
            evidence={"tasks": top_bad},
            suggested_profile={},
            rationale=(
                "Tasks with efficiency < 0.5 — inspect diagnose report and "
                "reward checks manually; no automatic reward change."
            ),
        ))

    return HarnessEvolutionReport(
        episodes_analyzed=n,
        mapping_summary=dict(mapping_counts),
        process_summary={
            "episodes_with_no_effect": no_effect,
            "episodes_with_loops": loops,
            "pointer_retained": pointer_retained,
            "ambiguous_clicks": ambiguous,
        },
        suggestions=suggestions,
    )


def export_report(report: HarnessEvolutionReport, path: str) -> None:
    """Write JSON + markdown siblings for review."""
    base, ext = path.rsplit(".", 1) if "." in path else (path, "json")
    if ext.lower() == "json":
        json_path = path
        md_path = base + ".md"
    else:
        json_path = path + ".json"
        md_path = path + ".md"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(report.to_markdown())
