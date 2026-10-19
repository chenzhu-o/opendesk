"""The safety guard — one choke point every agent-initiated tool call passes.

:func:`run_tool` / :func:`guarded_call` perform, in order:

1. **control check** — refuse while ``opendesk stop`` / ``takeover`` is active;
2. **secret placement** — ``{{secret:NAME}}`` only in typed text;
3. **classify / decide** — built-in risk + ``policy.json`` (+ taint escalation,
   + secrets need approval by default);
4. **approval** — MCP elicitation → native dialog → console → deny;
5. **secret resolution** — values from the OS store, just in time;
6. ``tool.execute``;
7. **redaction** — secret values scrubbed from the result, errors and the
   in-memory sandbox audit / replay params;
8. **untrusted wrapping + injection scan** — screen-derived text is wrapped,
   hits taint the session;
9. **chained audit** — ``<home>/audit/tools-YYYY-MM-DD.jsonl``.

``OPENDESK_SAFETY=off`` turns the guard into a pass-through.  ``Tool.execute``
itself is untouched, so tools can still be called directly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

from opendesk.safety._home import resolve_home, safety_enabled
from opendesk.safety.approval import ApprovalRequest, request_approval
from opendesk.safety.audit import ToolAuditLog, summarize_params
from opendesk.safety.control import check_control
from opendesk.safety.injection import scan, taint, taint_state, untrusted_source, warning_text, wrap
from opendesk.safety.policy import Decision, PolicyConfig, decide, load_policy
from opendesk.safety.secrets import (
    Redactor,
    SecretError,
    fetch_values,
    load_index,
    resolve,
    secret_names,
)
from opendesk.tools.base import PermissionDeniedError, ToolContext, ToolResult

log = logging.getLogger("opendesk.safety.guard")

Invoker = Callable[[dict[str, Any]], Awaitable[ToolResult]]


async def run_tool(tool: Any, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
    """Run *tool* with *params* (a raw dict) through the safety guard."""

    async def _invoke(p: dict[str, Any]) -> ToolResult:
        return await tool.execute(ctx, tool.parse_params(p))

    return await guarded_call(tool.name, params, ctx, _invoke)


def _refused(title: str, message: str, decision: str, extra: Optional[dict] = None) -> ToolResult:
    meta = {"safety": {"decision": decision, **(extra or {})}}
    return ToolResult(title=title, output=f"REFUSED: {message}", error=True, metadata=meta)


async def _focused_app(ctx: Any) -> Optional[str]:
    computer = getattr(ctx, "computer", None)
    if computer is None:
        return None
    try:
        win = await asyncio.wait_for(computer.focused_window(), timeout=3.0)
    except Exception:
        return None
    return getattr(win, "app_name", None) if win else None


def _app_matches(patterns: list[str], app: Optional[str]) -> bool:
    if not app:
        return False
    a = app.lower()
    return any(p.lower() in a or a in p.lower() for p in patterns)


def _scrub_sandbox(session_id: str, start: int, redactor: Redactor) -> None:
    try:
        from opendesk.computer.sandbox import get_sandbox
    except Exception:  # pragma: no cover
        return
    sandbox = get_sandbox(session_id)
    for entry in sandbox.audit_log[min(start, len(sandbox.audit_log)):]:
        entry.params = redactor.any(entry.params)
        if entry.result is not None:
            entry.result = redactor.text(entry.result)
        if entry.error is not None:
            entry.error = redactor.text(entry.error)
        if entry.replay_params is not None:
            entry.replay_params = redactor.any(entry.replay_params)


def _sandbox_len(session_id: str) -> int:
    try:
        from opendesk.computer.sandbox import get_sandbox
        return len(get_sandbox(session_id).audit_log)
    except Exception:  # pragma: no cover
        return 0


def _redact_result(result: ToolResult, redactor: Redactor) -> ToolResult:
    result.title = redactor.text(result.title or "")
    result.output = redactor.text(result.output or "")
    result.metadata = redactor.any(result.metadata or {})
    return result


def _summary_line(tool: str, params: dict[str, Any]) -> str:
    s = summarize_params(params)
    s.pop("action", None)
    body = ", ".join(f"{k}={v!r}" if not isinstance(v, str) else f"{k}={v}" for k, v in s.items())
    return f"{tool}: {body}" if body else tool


async def guarded_call(
    tool_name: str,
    params: dict[str, Any] | None,
    ctx: Optional[ToolContext],
    invoke: Invoker,
) -> ToolResult:
    """Run *invoke(params)* under the full safety pipeline (see module docs)."""
    params = dict(params or {})
    if not safety_enabled():
        return await invoke(params)

    meta: dict[str, Any] = (getattr(ctx, "metadata", None) or {}) if ctx is not None else {}
    home = resolve_home(ctx=ctx)
    peer = meta.get("peer")
    session = getattr(ctx, "session_id", "default") if ctx is not None else "default"
    audit = ToolAuditLog(home)
    t0 = time.monotonic()
    action = params.get("action") if isinstance(params.get("action"), str) else None

    def _audit(decision: str, outcome: str, *, risk: Optional[str] = None,
               error: Optional[str] = None, extra: Optional[dict[str, Any]] = None) -> None:
        try:
            audit.record(
                tool=tool_name, params=params, decision=decision, outcome=outcome,
                session=session, peer=peer, risk=risk,
                duration_ms=int((time.monotonic() - t0) * 1000), error=error, extra=extra,
            )
        except Exception as exc:  # audit must never break a call
            log.warning("tool audit write failed: %s", exc)

    # 1. Kill switch / take-over -------------------------------------------
    msg = check_control(home)
    if msg:
        _audit("refused", "refused", error="control")
        return _refused("Refused: control", msg, "refused", {"reason": "control"})

    # 2. Secret placement ---------------------------------------------------
    try:
        names = secret_names(tool_name, params)
    except SecretError as exc:
        _audit("block", "blocked", error=str(exc))
        return _refused("Refused: secret", str(exc), "block")

    # 3. Decide -------------------------------------------------------------
    policy: PolicyConfig = load_policy(home)
    taint_reason = taint_state(consume=True)
    index = load_index(home) if names else {}
    app = params.get("app") if isinstance(params.get("app"), str) else None
    needs_app = (
        any(r.get("app") for r in policy.rules)
        or any(index.get(n) and index[n].allowed_apps for n in names)
    )
    if app is None and needs_app and tool_name in ("keyboard", "mouse", "browser", "fleet"):
        app = await _focused_app(ctx)

    decision: Decision = decide(
        tool_name, params,
        {"home": home, "peer": peer, "app": app,
         "tainted": bool(taint_reason), "taint_reason": taint_reason},
        policy=policy,
    )
    risk = decision.risk.level.value
    reason = decision.reason

    if names and decision.decision != "block":
        for n in names:
            info = index.get(n)
            if info is None:
                err = f"unknown secret {n!r} — ask the user to run `opendesk secrets set {n}`"
                _audit("block", "blocked", risk=risk, error=err, extra={"secrets": names})
                return _refused("Refused: secret", err, "block")
            if info.allowed_apps and not _app_matches(info.allowed_apps, app):
                err = (
                    f"secret {n!r} may only be typed into {', '.join(info.allowed_apps)}"
                    f" (target app: {app or 'unknown'})"
                )
                _audit("block", "blocked", risk=risk, error=err, extra={"secrets": names})
                return _refused("Refused: secret", err, "block")
        if decision.decision == "allow" and any(index[n].confirm for n in names):
            decision = Decision("confirm", f"uses stored secret(s): {', '.join(names)}",
                                decision.risk, decision.rule)
            reason = decision.reason

    extra: dict[str, Any] = {}
    if names:
        extra["secrets"] = names
    if taint_reason:
        extra["tainted"] = True

    if decision.decision == "block":
        _audit("block", "blocked", risk=risk, error=reason, extra=extra)
        return _refused("Refused: policy", f"blocked by policy — {reason}", "block",
                        {"risk": risk, "reason": reason})

    # 4. Approval -----------------------------------------------------------
    approved_via: Optional[str] = None
    if decision.decision == "confirm":
        req = ApprovalRequest(
            tool=tool_name, action=action, summary=_summary_line(tool_name, params),
            reason=reason, risk=risk, peer=peer, secrets=list(names),
            timeout=policy.approval_timeout,
        )
        outcome = await request_approval(
            req, mcp_approver=meta.get("approver"), chain=policy.approval_chain,
        )
        if not outcome.approved:
            _audit("confirm", "denied", risk=risk, error=outcome.message,
                   extra={**extra, "approval": outcome.via})
            return _refused(
                "Refused: not approved",
                f"this action needs the user's approval ({reason}) and was not approved: "
                f"{outcome.message}",
                "denied", {"risk": risk, "reason": reason},
            )
        approved_via = outcome.via
        extra["approval"] = approved_via
        msg = check_control(home)  # the user may have hit stop while the dialog was up
        if msg:
            _audit("refused", "refused", risk=risk, error="control", extra=extra)
            return _refused("Refused: control", msg, "refused", {"reason": "control"})

    # 5. Secrets ------------------------------------------------------------
    values: dict[str, str] = {}
    if names:
        try:
            values = fetch_values(names, home)
        except SecretError as exc:
            _audit(decision.decision, "error", risk=risk, error=str(exc), extra=extra)
            return ToolResult(title="Secret error", output=f"ERROR: {exc}", error=True)
    redactor = Redactor(values)
    exec_params = resolve(params, values) if values else params

    # 6. Execute ------------------------------------------------------------
    start = _sandbox_len(session) if redactor else 0
    try:
        result = await invoke(exec_params)
    except Exception as exc:
        if redactor:
            _scrub_sandbox(session, start, redactor)
            safe = redactor.text(str(exc))
            _audit(decision.decision, "exception", risk=risk, error=safe, extra=extra)
            if isinstance(exc, PermissionDeniedError):
                raise PermissionDeniedError(safe) from None
            raise RuntimeError(safe) from None
        _audit(decision.decision, "exception", risk=risk, error=str(exc), extra=extra)
        raise
    if redactor:
        _scrub_sandbox(session, start, redactor)
        result = _redact_result(result, redactor)

    # 7. Untrusted content + injection scan ---------------------------------
    src = untrusted_source(tool_name, params)
    if src and result.output and not result.error:
        findings = scan(result.output)
        wrapped = wrap(result.output, src)
        if findings:
            taint(f"injection-like text in {src}", calls=policy.taint_calls,
                  minutes=policy.taint_minutes)
            wrapped = warning_text(findings, src) + "\n" + wrapped
            extra["injection"] = sorted({f.pattern for f in findings})
        result.output = wrapped

    # 8. Audit --------------------------------------------------------------
    _audit(decision.decision, "error" if result.error else "ok", risk=risk,
           error=(result.output[:200] if result.error else None), extra=extra)
    try:
        result.metadata = dict(result.metadata or {})
        result.metadata["safety"] = {"decision": decision.decision, "risk": risk, **extra}
    except Exception:  # pragma: no cover
        pass
    return result
