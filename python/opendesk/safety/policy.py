"""Policy engine — classify every tool call by risk and decide allow / confirm / block.

Policy file: ``<home>/policy.json``::

    {
      "preset": "standard",              // relaxed | standard | strict
      "rules": [                         // ordered; first match wins
        {"tool": "script", "action": "shell", "decision": "block",
         "reason": "no shell on this machine"},
        {"tool": "keyboard", "app": "Terminal", "decision": "confirm"},
        {"tool": "*", "peer": "billing-*", "decision": "confirm"}
      ],
      "kill_hotkey": "<ctrl>+<alt>+<shift>+k",   // optional, MCP server only
      "approval": {"timeout": 60, "chain": ["mcp", "native", "console"]},
      "taint": {"calls": 10, "minutes": 10}
    }

Presets:

* ``relaxed``  — allow everything not blocked by a rule.
* ``standard`` — read-only and normal input allowed; high-risk → confirm.
* ``strict``   — read-only allowed; everything else → confirm.

Classification is by tool *name* plus ``action`` so tools that don't exist in
this process (``script``, ``browser``, ``fleet``, ``memory_manage``…) are still
classified correctly the moment they are registered.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from opendesk.safety._home import read_json, resolve_home, write_json

POLICY_FILE = "policy.json"
PRESETS = ("relaxed", "standard", "strict")
DECISIONS = ("allow", "confirm", "block")
DEFAULT_PRESET = "standard"
DEFAULT_APPROVAL_TIMEOUT = 60.0
DEFAULT_TAINT_CALLS = 10
DEFAULT_TAINT_MINUTES = 10.0


class RiskLevel(str, Enum):
    """``low`` = read-only, ``medium`` = normal input / state change, ``high`` = dangerous."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]


@dataclass(frozen=True)
class Risk:
    """Risk classification of one tool call."""

    level: RiskLevel
    reason: str
    tool: str = ""
    action: Optional[str] = None

    @property
    def read_only(self) -> bool:
        return self.level is RiskLevel.LOW


@dataclass(frozen=True)
class Decision:
    """Outcome of :func:`decide`."""

    decision: str  # allow | confirm | block
    reason: str
    risk: Risk
    rule: Optional[dict[str, Any]] = None

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


# ---------------------------------------------------------------------------
# Built-in classification
# ---------------------------------------------------------------------------

_L, _M, _H = RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH

# tool -> (per-action levels, default level for unknown / missing action)
_TABLE: dict[str, tuple[dict[str, RiskLevel], RiskLevel]] = {
    "screenshot": ({}, _L),
    "ocr": ({}, _L),
    "ui": ({"get_tree": _L, "get_value": _L, "find": _L}, _M),
    "mouse": ({"cursor_position": _L}, _M),
    "keyboard": ({}, _M),
    "app": ({"list": _L}, _M),
    "clipboard": ({"read": _L}, _M),
    "audit": ({"show": _L, "verify": _L}, _M),
    "schedule": ({"list": _L}, _M),
    "learn": ({"list": _L, "show": _L}, _M),
    "script": ({"list": _L, "show": _L, "shell": _H, "applescript": _H, "jxa": _H,
                "powershell": _H, "bash": _H, "python": _H, "exec": _H}, _H),
    "browser": ({"read": _L, "tabs": _L, "screenshot": _L, "text": _L, "find": _L,
                 "eval": _H, "evaluate": _H, "js": _H}, _M),
    "memory": ({"search": _L, "show": _L, "timeline": _L, "status": _L,
                "recall": _L, "delete": _H, "clear": _H}, _M),
    "memory_manage": ({"status": _L, "list": _L, "delete": _H, "clear": _H, "export": _H}, _M),
    "fs": ({"read": _L, "list": _L, "stat": _L, "delete": _H}, _M),
    # Session admin tools exposed by the MCP server.
    "opendesk_peers": ({}, _L),
    "opendesk_discover": ({}, _L),
    "opendesk_status": ({}, _L),
    "opendesk_capabilities": ({}, _L),
    "opendesk_describe": ({}, _L),
    "opendesk_use": ({}, _M),
    "opendesk_disconnect": ({}, _M),
}

_READ_VERBS = {"list", "show", "status", "read", "get", "search", "describe", "info"}


def _action_of(params: Mapping[str, Any] | None) -> Optional[str]:
    if not params:
        return None
    a = params.get("action")
    return str(a) if isinstance(a, str) and a else None


def _fleet_inner(params: Mapping[str, Any]) -> tuple[Optional[str], dict[str, Any]]:
    inner = params.get("tool")
    inner_params = (
        params.get("params") or params.get("arguments") or params.get("args") or {}
    )
    if not isinstance(inner_params, dict):
        inner_params = {}
    return (inner if isinstance(inner, str) and inner else None), inner_params


def classify(tool: str, params: Mapping[str, Any] | None = None) -> Risk:
    """Return the built-in :class:`Risk` of calling *tool* with *params*."""
    params = params or {}
    action = _action_of(params)

    if tool == "fleet":
        inner, inner_params = _fleet_inner(params)
        if inner:
            r = classify(inner, inner_params)
            return Risk(r.level, f"fleet → {r.reason}", tool, action or r.action)
        level = _L if (action in _READ_VERBS or action is None and not params) else _M
        return Risk(level, f"fleet {action or 'call'}", tool, action)

    if tool in _TABLE:
        table, default = _TABLE[tool]
        level = table.get(action, default) if action else default
        label = f"{tool} {action}" if action else tool
        return Risk(level, f"{label} is {_describe(level)}", tool, action)

    if tool.startswith("opendesk_"):
        return Risk(_L, f"{tool} is session admin", tool, action)

    # Unknown tool: read-ish verbs are low, everything else medium.
    level = _L if action in _READ_VERBS else _M
    return Risk(level, f"{tool} {action or ''}".strip() + " (unclassified tool)", tool, action)


def _describe(level: RiskLevel) -> str:
    return {"low": "read-only", "medium": "normal input", "high": "high-risk"}[level.value]


# ---------------------------------------------------------------------------
# Policy file
# ---------------------------------------------------------------------------


@dataclass
class PolicyConfig:
    """In-memory form of ``policy.json``."""

    preset: str = DEFAULT_PRESET
    rules: list[dict[str, Any]] = field(default_factory=list)
    kill_hotkey: Optional[str] = None
    approval: dict[str, Any] = field(default_factory=dict)
    taint: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def approval_timeout(self) -> float:
        try:
            return float(self.approval.get("timeout", DEFAULT_APPROVAL_TIMEOUT))
        except (TypeError, ValueError):
            return DEFAULT_APPROVAL_TIMEOUT

    @property
    def approval_chain(self) -> Optional[list[str]]:
        chain = self.approval.get("chain")
        return [str(c) for c in chain] if isinstance(chain, list) else None

    @property
    def taint_calls(self) -> int:
        try:
            return int(self.taint.get("calls", DEFAULT_TAINT_CALLS))
        except (TypeError, ValueError):
            return DEFAULT_TAINT_CALLS

    @property
    def taint_minutes(self) -> float:
        try:
            return float(self.taint.get("minutes", DEFAULT_TAINT_MINUTES))
        except (TypeError, ValueError):
            return DEFAULT_TAINT_MINUTES

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyConfig":
        preset = str(data.get("preset") or DEFAULT_PRESET).lower()
        if preset not in PRESETS:
            preset = DEFAULT_PRESET
        rules = [dict(r) for r in (data.get("rules") or []) if isinstance(r, dict)]
        known = {"preset", "rules", "kill_hotkey", "approval", "taint"}
        return cls(
            preset=preset,
            rules=rules,
            kill_hotkey=data.get("kill_hotkey") or None,
            approval=dict(data.get("approval") or {}),
            taint=dict(data.get("taint") or {}),
            extra={k: v for k, v in data.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"preset": self.preset, "rules": self.rules}
        if self.kill_hotkey:
            d["kill_hotkey"] = self.kill_hotkey
        if self.approval:
            d["approval"] = self.approval
        if self.taint:
            d["taint"] = self.taint
        d.update(self.extra)
        return d


def policy_path(home: Any = None) -> Path:
    return resolve_home(home) / POLICY_FILE


def load_policy(home: Any = None) -> PolicyConfig:
    """Load ``<home>/policy.json`` (defaults when missing / invalid)."""
    data = read_json(policy_path(home))
    return PolicyConfig.from_dict(data) if data else PolicyConfig()


def save_policy(cfg: PolicyConfig, home: Any = None) -> Path:
    path = policy_path(home)
    write_json(path, cfg.to_dict())
    return path


def validate_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise and validate one rule; raises :class:`ValueError`."""
    if not rule.get("tool"):
        raise ValueError("rule needs a 'tool' (use '*' for any tool)")
    decision = str(rule.get("decision") or "").lower()
    if decision not in DECISIONS:
        raise ValueError(f"rule decision must be one of {DECISIONS}, got {decision!r}")
    out = {"tool": str(rule["tool"])}
    for key in ("action", "app", "peer"):
        if rule.get(key):
            out[key] = str(rule[key])
    out["decision"] = decision
    if rule.get("reason"):
        out["reason"] = str(rule["reason"])
    return out


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def _glob(pattern: str, value: Optional[str], *, substring: bool = False) -> bool:
    if value is None:
        return False
    p, v = pattern.lower(), value.lower()
    if fnmatch.fnmatchcase(v, p):
        return True
    return substring and not any(c in p for c in "*?[") and p in v


def rule_matches(rule: Mapping[str, Any], tool: str, params: Mapping[str, Any],
                 context: Mapping[str, Any]) -> bool:
    """Does *rule* apply to this call?

    ``tool`` / ``action`` / ``peer`` are case-insensitive globs; ``app`` is a
    glob or case-insensitive substring of the target app.  For remote calls
    the raw protocol method (``context["method"]``) also matches ``tool``.
    """
    rt = str(rule.get("tool") or "*")
    method = context.get("method")
    if not (_glob(rt, tool) or (method and _glob(rt, str(method)))):
        return False
    if rule.get("action"):
        action = context.get("action") or _action_of(params)
        if not _glob(str(rule["action"]), action):
            return False
    if rule.get("app"):
        app = context.get("app") or params.get("app")
        if not _glob(str(rule["app"]), str(app) if app else None, substring=True):
            return False
    if rule.get("peer"):
        if not _glob(str(rule["peer"]), context.get("peer") or "local"):
            return False
    return True


def _preset_decision(preset: str, risk: Risk) -> tuple[str, str]:
    if preset == "relaxed":
        return "allow", f"relaxed preset: {risk.reason}"
    if preset == "strict":
        if risk.read_only:
            return "allow", f"strict preset: {risk.reason}"
        return "confirm", f"strict preset: {risk.reason} — needs approval"
    if risk.level is RiskLevel.HIGH:
        return "confirm", f"{risk.reason} — needs approval"
    return "allow", risk.reason


def decide(
    tool: str,
    params: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    *,
    policy: Optional[PolicyConfig] = None,
) -> Decision:
    """Decide ``allow`` / ``confirm`` / ``block`` for one call.

    *context* keys (all optional): ``home``, ``app``, ``peer``, ``method``,
    ``action`` (override), ``tainted`` (bool), ``taint_reason``.
    """
    params = params or {}
    context = context or {}
    if policy is None:
        policy = load_policy(context.get("home"))
    risk = classify(tool, params)
    if context.get("action") and risk.action is None:
        risk = Risk(risk.level, risk.reason, risk.tool, str(context["action"]))

    decision: Optional[Decision] = None
    for rule in policy.rules:
        if rule.get("decision") not in DECISIONS:
            continue
        if rule_matches(rule, tool, params, context):
            reason = rule.get("reason") or f"policy rule {_rule_label(rule)}"
            decision = Decision(rule["decision"], reason, risk, dict(rule))
            break
    if decision is None:
        d, reason = _preset_decision(policy.preset, risk)
        decision = Decision(d, reason, risk)

    if decision.decision == "allow" and context.get("tainted") and not risk.read_only:
        why = context.get("taint_reason") or "possible prompt injection seen on screen"
        decision = Decision(
            "confirm",
            f"session is tainted ({why}); input actions need approval",
            risk, decision.rule,
        )
    return decision


def _rule_label(rule: Mapping[str, Any]) -> str:
    parts = [f"{k}={rule[k]}" for k in ("tool", "action", "app", "peer") if rule.get(k)]
    return "[" + " ".join(parts) + f" → {rule.get('decision')}]"


class PolicyEngine:
    """Convenience wrapper bound to one home directory."""

    def __init__(self, home: Any = None) -> None:
        self.home = resolve_home(home)

    @property
    def config(self) -> PolicyConfig:
        return load_policy(self.home)

    def classify(self, tool: str, params: Mapping[str, Any] | None = None) -> Risk:
        return classify(tool, params)

    def decide(self, tool: str, params: Mapping[str, Any] | None = None,
               context: Mapping[str, Any] | None = None) -> Decision:
        ctx = dict(context or {})
        ctx.setdefault("home", self.home)
        return decide(tool, params, ctx)
