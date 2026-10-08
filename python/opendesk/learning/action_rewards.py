"""Verifiable per-step action rewards for GUI policy training.

Outcome rewards (:mod:`rewards`) score whether a *task* finished.  Process
rewards (:mod:`process`) score whether each step *changed the machine*.  This
module scores whether a *proposed action string* matches what should be done
given the gold demonstration and the accessibility tree in the prompt — without
calling a model.

Use it to:
  * rank rollouts for preference pairs when several completions exist;
  * mine hard negatives for offline DPO (chosen = gold, rejected = low reward);
  * weight DPO margins by ``chosen_reward - rejected_reward``.

Multi-step tasks are handled at export time: each step is one row; episode-level
success still comes from ``rewards`` + ``process`` on live rollouts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(")
COORD = re.compile(r"coordinates=\[(\d+),\s*(\d+)\]")
# OSWorld tree rows: role, name, (x, y), (w, h) — tab or legacy formats
TREE_XY = re.compile(
    r"\((?P<x>-?\d+),\s*(?P<y>-?\d+)\)"
)
APP_TOOL = re.compile(r"^[A-Za-z_]+Tools\.[A-Za-z_][A-Za-z0-9_]*\(")

DEFAULT_WEIGHTS = {
    "parseable": 0.05,
    "call_sequence": 0.35,
    "first_call": 0.15,
    "pointer": 0.25,
    "tool_identity": 0.20,
}


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def call_names(action: str) -> list[str]:
    return CALL.findall(action or "")


def click_coordinates(action: str) -> Optional[tuple[int, int]]:
    m = COORD.search(action or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def gold_coordinates(action: str) -> Optional[tuple[int, int]]:
    return click_coordinates(action)


def coordinate_visible_in_tree(
    x: int,
    y: int,
    tree: str,
    *,
    tolerance: int = 10,
) -> bool:
    """True if some tree row centre is within tolerance of (x, y)."""
    if not tree:
        return False
    for line in tree.splitlines():
        nums = list(TREE_XY.finditer(line))
        if len(nums) < 1:
            continue
        # Prefer the centre pair: often the first (x,y) in the row
        cx, cy = int(nums[0].group("x")), int(nums[0].group("y"))
        if abs(cx - x) <= tolerance and abs(cy - y) <= tolerance:
            return True
    return False


def is_app_tool_call(action: str) -> bool:
    s = (action or "").strip()
    return bool(APP_TOOL.match(s)) or ".Tools." in s.split("(")[0]


@dataclass
class ActionRewardReport:
    """Dense, auditable score in ``[0, 1]``."""

    reward: float
    components: dict[str, float] = field(default_factory=dict)
    detail: str = ""
    gold_visible: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "reward": round(self.reward, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "detail": self.detail,
        }
        if self.gold_visible is not None:
            out["gold_coord_visible_in_tree"] = self.gold_visible
        return out


def score_step_action(
    predicted: str,
    gold: str,
    tree: str = "",
    *,
    coord_tolerance: int = 10,
    weights: Optional[dict[str, float]] = None,
) -> ActionRewardReport:
    """Score one proposed action against gold + observation.

    Components (weighted sum):
      * **parseable** — at least one well-formed call
      * **call_sequence** — same call-name sequence as gold
      * **first_call** — first verb matches (Agent.click vs ImpressTools.save)
      * **pointer** — click coordinates match within tolerance when gold has coords
      * **tool_identity** — full-string match for ``*Tools.*`` actions
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    pred, g = (predicted or "").strip(), (gold or "").strip()
    pc, gc = call_names(pred), call_names(gold)

    components: dict[str, float] = {}
    if not gc:
        return ActionRewardReport(0.0, {}, "empty gold")

    components["parseable"] = 1.0 if pc else 0.0
    components["call_sequence"] = 1.0 if pc == gc else 0.0
    components["first_call"] = 1.0 if (pc[:1] == gc[:1] if pc else False) else 0.0

    gcoord = gold_coordinates(g)
    visible = None
    if gcoord is not None:
        visible = coordinate_visible_in_tree(
            gcoord[0], gcoord[1], tree, tolerance=coord_tolerance,
        )
        pcoord = click_coordinates(pred)
        if pcoord is None:
            components["pointer"] = 0.0
        elif _norm(pred) == _norm(g):
            components["pointer"] = 1.0
        elif visible:
            gx, gy = gcoord
            px, py = pcoord
            ok = abs(px - gx) <= coord_tolerance and abs(py - gy) <= coord_tolerance
            components["pointer"] = 1.0 if ok else 0.0
        else:
            # Target not in tree — do not pretend coordinate match is learnable
            components["pointer"] = 1.0 if components["call_sequence"] else 0.0
    else:
        components["pointer"] = 1.0 if components["call_sequence"] else 0.0

    if is_app_tool_call(g):
        components["tool_identity"] = 1.0 if _norm(pred) == _norm(g) else 0.0
    else:
        components["tool_identity"] = 1.0 if components["first_call"] else 0.0

    total_w = sum(w[k] for k in components)
    reward = sum(components[k] * w[k] for k in components) / max(total_w, 1e-9)
    detail = "pred=%r gold=%r" % (pred[:80], g[:80])
    return ActionRewardReport(
        reward=reward,
        components=components,
        detail=detail,
        gold_visible=visible,
    )


def _candidate_negatives(gold: str, tree: str, modal_click: str) -> list[str]:
    g = (gold or "").strip()
    cands = [
        modal_click,
        "Agent.click(coordinates=[0, 0])",
        "Agent.exit(success=True)",
    ]
    if g.startswith("Agent.click("):
        cands.append("Agent.click(coordinates=[1307, 787])")
    elif is_app_tool_call(g):
        cands.extend([
            "Agent.click(coordinates=[1307, 787])",
            "Agent.type(text='')",
        ])
        # Wrong tool family
        if g.startswith("ImpressTools."):
            cands.append("WriterTools.save()")
        elif g.startswith("WriterTools."):
            cands.append("ImpressTools.save()")
    elif g.startswith("Agent.type("):
        cands.append("Agent.click(coordinates=[1307, 787])")
    # dedupe, drop gold
    out = []
    seen = set()
    for c in cands:
        if c != g and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def hard_negative(
    gold: str,
    tree: str = "",
    *,
    modal_click: str = "Agent.click(coordinates=[1307, 787])",
    coord_tolerance: int = 10,
) -> tuple[str, ActionRewardReport, ActionRewardReport]:
    """Pick the lowest-reward alternative among scripted mistakes.

    Returns ``(rejected_string, gold_report, rejected_report)``.
    """
    g = (gold or "").strip()
    gold_rep = score_step_action(g, g, tree, coord_tolerance=coord_tolerance)
    best_rej, best_rep, best_r = None, None, 2.0
    for cand in _candidate_negatives(g, tree, modal_click):
        rep = score_step_action(cand, g, tree, coord_tolerance=coord_tolerance)
        if rep.reward < best_r:
            best_r, best_rej, best_rep = rep.reward, cand, rep
    if best_rej is None:
        best_rej = "Agent.click(coordinates=[0, 0])"
        best_rep = score_step_action(best_rej, g, tree, coord_tolerance=coord_tolerance)
    return best_rej, gold_rep, best_rep


def dpo_margin_from_reports(
    chosen: ActionRewardReport,
    rejected: ActionRewardReport,
) -> float:
    """Preference strength for metadata / loss weighting."""
    return max(0.05, min(1.0, chosen.reward - rejected.reward))
