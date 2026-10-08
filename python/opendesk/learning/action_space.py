"""Map legacy OSWorld-style actions to opendesk-tier completions.

Policy: map to semantic UI or *Tools* when the accessibility tree supports it;
otherwise keep the original coordinate action and mark a lower ``sample_weight``.
Never invent element names or tool calls that are not justified by the tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from opendesk.learning import action_rewards as ar

ActionTier = Literal["tool", "cli", "ui", "pointer_fallback", "passthrough"]
MappingStatus = Literal[
    "mapped_ui",
    "passthrough_tool",
    "passthrough_agent",
    "retained_pointer",
    "empty",
]

# Training loss weight when the target remains a bare coordinate click.
POINTER_FALLBACK_WEIGHT = 0.35
SEMANTIC_WEIGHT = 1.0

_AGENT_CLICK = re.compile(
    r"^Agent\.click\s*\(\s*coordinates\s*=\s*\[(\d+)\s*,\s*(\d+)\]\s*\)\s*$"
)
_TOOLS = re.compile(r"^[A-Za-z_]+Tools\.[A-Za-z_][A-Za-z0-9_]*\(")


@dataclass(frozen=True)
class MappedCompletion:
    """Result of :func:`map_legacy_completion`."""

    primary: str
    legacy: str
    tier: ActionTier
    status: MappingStatus
    sample_weight: float
    detail: str = ""

    @property
    def changed(self) -> bool:
        return self.primary.strip() != self.legacy.strip()


def _tree_elements(tree: str, *, tolerance: int = 10) -> list[dict[str, str | int]]:
    """Parse tab-separated OSWorld / opendesk tree rows with centre coordinates."""
    out: list[dict[str, str | int]] = []
    if not tree:
        return out
    for line in tree.splitlines():
        line = line.strip()
        if not line or line.startswith("Task:") or "Accessibility tree" in line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        role, name = parts[0].strip(), parts[1].strip()
        xy = ar.TREE_XY.search(line)
        if not xy:
            continue
        out.append({
            "role": role,
            "name": name,
            "x": int(xy.group("x")),
            "y": int(xy.group("y")),
        })
    return out


def _match_click_target(
    x: int,
    y: int,
    tree: str,
    *,
    tolerance: int = 10,
) -> tuple[Optional[dict], str]:
    """Return (unique element, reason)."""
    hits = []
    for el in _tree_elements(tree, tolerance=tolerance):
        if abs(int(el["x"]) - x) <= tolerance and abs(int(el["y"]) - y) <= tolerance:
            hits.append(el)
    if len(hits) == 1:
        return hits[0], "unique_tree_match"
    if len(hits) > 1:
        return None, "ambiguous_tree_match"
    return None, "coord_not_in_tree"


def _ui_click(el: dict) -> str:
    name = str(el.get("name") or "").replace('"', '\\"')
    role = str(el.get("role") or "").replace('"', '\\"')
    if name and role:
        return 'click(name="%s", role="%s")' % (name, role)
    if name:
        return 'click(name="%s")' % name
    return 'click(role="%s")' % role


def map_legacy_completion(
    legacy: str,
    tree: str = "",
    *,
    coord_tolerance: int = 10,
    pointer_weight: float = POINTER_FALLBACK_WEIGHT,
    semantic_weight: float = SEMANTIC_WEIGHT,
) -> MappedCompletion:
    """Choose the training target string and sample weight for one step."""
    raw = (legacy or "").strip()
    if not raw:
        return MappedCompletion(
            primary="",
            legacy=raw,
            tier="passthrough",
            status="empty",
            sample_weight=0.0,
            detail="empty",
        )

    if _TOOLS.match(raw):
        return MappedCompletion(
            primary=raw,
            legacy=raw,
            tier="tool",
            status="passthrough_tool",
            sample_weight=semantic_weight,
            detail="app_tool",
        )

    m = _AGENT_CLICK.match(raw)
    if not m:
        return MappedCompletion(
            primary=raw,
            legacy=raw,
            tier="passthrough",
            status="passthrough_agent",
            sample_weight=semantic_weight,
            detail="non_click_agent",
        )

    x, y = int(m.group(1)), int(m.group(2))
    el, reason = _match_click_target(x, y, tree, tolerance=coord_tolerance)
    if el is not None:
        primary = _ui_click(el)
        return MappedCompletion(
            primary=primary,
            legacy=raw,
            tier="ui",
            status="mapped_ui",
            sample_weight=semantic_weight,
            detail=reason,
        )

    return MappedCompletion(
        primary=raw,
        legacy=raw,
        tier="pointer_fallback",
        status="retained_pointer",
        sample_weight=pointer_weight,
        detail=reason,
    )


def observation_tree_from_episode(episode: dict) -> str:
    """Best-effort tree text for mapping (caller / export conventions)."""
    if episode.get("_tree"):
        return str(episode["_tree"])
    if episode.get("observation_tree"):
        return str(episode["observation_tree"])
    obs = episode.get("observation")
    if isinstance(obs, dict) and obs.get("accessibility_tree"):
        return str(obs["accessibility_tree"])
    steps = episode.get("steps") or []
    if steps:
        o = steps[0].get("observation") if isinstance(steps[0], dict) else None
        if isinstance(o, dict) and o.get("accessibility_tree"):
            return str(o["accessibility_tree"])
    return ""
