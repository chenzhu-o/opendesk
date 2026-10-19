"""Tamper-evident, persistent log of every guarded tool call.

``<home>/audit/tools-YYYY-MM-DD.jsonl`` — one hash-chained line per call
(see :mod:`opendesk.safety.chain`)::

    {"ts": ..., "session": "mcp-local", "tool": "keyboard", "action": "type",
     "peer": "local", "params": {"action": "type", "text": "'hello' (5 chars)"},
     "risk": "medium", "decision": "allow", "outcome": "ok", "duration_ms": 41,
     "prev": "…", "hash": "…"}

Params are summarised and redacted: long text is truncated, keys that look
like credentials are dropped, binary blobs become sizes, and secret
placeholders stay placeholders (values are never logged).
"""

from __future__ import annotations

import datetime as _dt
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from opendesk.safety._home import resolve_home
from opendesk.safety.chain import VerifyResult, append_chained, verify_file

AUDIT_DIR = "audit"
TOOL_LOG_PREFIX = "tools-"

_SENSITIVE_KEY = re.compile(
    r"pass(word|wd|phrase)?|secret|token|api[_-]?key|credential|cookie|auth|private[_-]?key|otp",
    re.IGNORECASE,
)
_LONG_KEYS = {"command", "script", "code", "source", "url", "query"}


def _summ_value(key: str, value: Any, depth: int = 0) -> Any:
    if _SENSITIVE_KEY.search(key or ""):
        return "[redacted]"
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, str):
        limit = 160 if key in _LONG_KEYS else 40
        if key in ("image", "content", "data") and len(value) > 200:
            return f"<{len(value)} chars>"
        if len(value) > limit:
            return f"{value[:limit]!r}… ({len(value)} chars)"
        return value
    if isinstance(value, dict):
        if depth >= 2:
            return f"<object, {len(value)} keys>"
        return {str(k): _summ_value(str(k), v, depth + 1) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        if depth >= 2 or len(value) > 20:
            return f"<list, {len(value)} items>"
        return [_summ_value(key, v, depth + 1) for v in value]
    return value


def summarize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Redacted, size-bounded summary of tool params for the audit log."""
    return {str(k): _summ_value(str(k), v) for k, v in (params or {}).items()}


class ToolAuditLog:
    """Writer / reader for the chained per-day tool-call log."""

    def __init__(self, home: Any = None) -> None:
        self.directory = resolve_home(home) / AUDIT_DIR

    def path(self, date: Optional[str] = None) -> Path:
        return self.directory / f"{TOOL_LOG_PREFIX}{date or _dt.date.today().isoformat()}.jsonl"

    def record(
        self,
        *,
        tool: str,
        params: dict[str, Any],
        decision: str,
        outcome: str,
        session: str = "",
        peer: Optional[str] = None,
        risk: Optional[str] = None,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        action = params.get("action") if isinstance(params, dict) else None
        entry: dict[str, Any] = {
            "ts": round(time.time(), 6),
            "type": "tool",
            "session": session,
            "tool": tool,
            "action": action if isinstance(action, str) else None,
            "peer": peer or "local",
            "params": summarize_params(params),
            "risk": risk,
            "decision": decision,
            "outcome": outcome,
        }
        if duration_ms is not None:
            entry["duration_ms"] = int(duration_ms)
        if error:
            entry["error"] = error[:300]
        if extra:
            entry.update(extra)
        return append_chained(self.path(), entry)

    def iter_entries(self, date: Optional[str] = None) -> list[dict[str, Any]]:
        import json
        p = self.path(date)
        if not p.exists():
            return []
        out = []
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        return out


def audit_files(home: Any = None, date: Optional[str] = None) -> list[Path]:
    """All audit JSONL files (tool + server) — optionally only for *date*."""
    d = resolve_home(home) / AUDIT_DIR
    if not d.exists():
        return []
    if date:
        cands = [d / f"{TOOL_LOG_PREFIX}{date}.jsonl", d / f"{date}.jsonl"]
        return [p for p in cands if p.exists()]
    return sorted(p for p in d.glob("*.jsonl") if p.is_file())


def verify_files(paths: Iterable[Path]) -> list[VerifyResult]:
    return [verify_file(p) for p in paths]
