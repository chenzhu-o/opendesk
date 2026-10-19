"""Human approval for ``confirm`` decisions.

An ordered chain of approvers is tried until one gives an answer:

1. ``mcp``     — MCP elicitation, when the connected client supports it.
   :class:`~opendesk.integrations.mcp.MCPDispatcher` puts this approver into
   ``ctx.metadata["approver"]``.
2. ``native``  — OS dialog: macOS ``osascript display dialog``, Windows
   ``WScript.Shell.Popup`` via PowerShell, Linux ``zenity`` / ``kdialog``.
3. ``console`` — ``[y/N]`` prompt when stdin is a TTY.
4. otherwise   — deny with a clear message.

Each approver is an ``async (ApprovalRequest) -> bool | None`` callable;
``None`` means "can't ask here — try the next one".  The whole chain shares
one timeout (``approval.timeout`` in ``policy.json``, default 60 s); timing
out denies.

``OPENDESK_APPROVAL_CHAIN`` (comma list, e.g. ``native,console`` or
``none``) or ``approval.chain`` in ``policy.json`` override the chain.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

log = logging.getLogger("opendesk.safety.approval")

DEFAULT_CHAIN = ("mcp", "native", "console")


@dataclass
class ApprovalRequest:
    """What the human is asked to approve.  Never contains secret values."""

    tool: str
    action: Optional[str]
    summary: str
    reason: str
    risk: str = "medium"
    peer: Optional[str] = None
    secrets: list[str] = field(default_factory=list)
    timeout: float = 60.0

    @property
    def title(self) -> str:
        return "OpenDesk: approve agent action?"

    def message(self) -> str:
        target = f" on {self.peer}" if self.peer and self.peer != "local" else ""
        lines = [
            f"An AI agent wants to run `{self.tool}`"
            + (f" ({self.action})" if self.action else "") + f"{target}.",
            "",
            self.summary,
            "",
            f"Why approval is needed: {self.reason}",
        ]
        if self.secrets:
            lines.append(f"Uses stored secret(s): {', '.join(self.secrets)}")
        return "\n".join(lines)


@dataclass
class ApprovalOutcome:
    approved: bool
    via: str
    message: str = ""


Approver = Callable[[ApprovalRequest], Awaitable[Optional[bool]]]


# ---------------------------------------------------------------------------
# MCP elicitation
# ---------------------------------------------------------------------------


def make_mcp_elicitation_approver(server: Any) -> Approver:
    """Approver that asks the MCP client via ``elicitation/create``.

    Must be called inside a request handler of the low-level MCP *server*
    (it reads ``server.request_context``).  Returns ``None`` when there is no
    request context or the client didn't declare the elicitation capability.
    """

    async def _approve(req: ApprovalRequest) -> Optional[bool]:
        try:
            rc = server.request_context
        except (LookupError, AttributeError):
            return None
        session = getattr(rc, "session", None)
        if session is None or not _client_supports_elicitation(session):
            return None
        schema = {
            "type": "object",
            "properties": {
                "allow": {
                    "type": "boolean",
                    "title": "Allow this action",
                    "description": "Tick to let the agent perform this action once.",
                    "default": False,
                },
            },
            "required": ["allow"],
        }
        result = await session.elicit(
            message=req.message(), requestedSchema=schema,
            related_request_id=getattr(rc, "request_id", None),
        )
        action = getattr(result, "action", None)
        if action != "accept":
            return False
        content = getattr(result, "content", None)
        if not content or "allow" not in content:
            return True  # accept-only clients
        return bool(content.get("allow"))

    return _approve


def _client_supports_elicitation(session: Any) -> bool:
    try:
        from mcp import types as mt
        return bool(session.check_client_capability(
            mt.ClientCapabilities(elicitation=mt.ElicitationCapability()),
        ))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Native OS dialog
# ---------------------------------------------------------------------------


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Run a dialog command.  Separate function so tests can replace it."""
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=False,
    )


def _applescript_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def native_dialog_argv(req: ApprovalRequest, platform: Optional[str] = None) -> Optional[list[str]]:
    """Command line for the platform's approval dialog, or ``None``."""
    platform = platform or sys.platform
    secs = max(1, int(req.timeout))
    msg = req.message()
    if platform == "darwin":
        if not shutil.which("osascript"):
            return None
        script = (
            f"display dialog {_applescript_str(msg)} with title {_applescript_str(req.title)} "
            'buttons {"Deny", "Allow"} default button "Deny" cancel button "Deny" '
            f"with icon caution giving up after {secs}"
        )
        return ["osascript", "-e", script]
    if platform.startswith("win"):
        exe = shutil.which("powershell") or shutil.which("pwsh")
        if not exe:
            return None
        ps_msg = msg.replace("'", "''")
        ps_title = req.title.replace("'", "''")
        # 4 = Yes/No, 48 = warning icon, 256 = default button 2 (No), 4096 = system modal.
        cmd = (
            f"$r = (New-Object -ComObject WScript.Shell).Popup('{ps_msg}', {secs}, "
            f"'{ps_title}', 4 + 48 + 256 + 4096); Write-Output $r"
        )
        return [exe, "-NoProfile", "-NonInteractive", "-Command", cmd]
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    if shutil.which("zenity"):
        return ["zenity", "--question", "--title", req.title, "--text", msg,
                "--ok-label", "Allow", "--cancel-label", "Deny", f"--timeout={secs}"]
    if shutil.which("kdialog"):
        return ["kdialog", "--title", req.title, "--yesno", msg,
                "--yes-label", "Allow", "--no-label", "Deny"]
    return None


def parse_native_result(argv: list[str], proc: subprocess.CompletedProcess) -> Optional[bool]:
    """Interpret a dialog's exit status / output.  ``None`` = dialog failed to show."""
    exe = os.path.basename(argv[0]).lower()
    out = (proc.stdout or "").strip()
    if exe == "osascript":
        if proc.returncode == 0:
            return "button returned:Allow" in out and "gave up:true" not in out
        err = (proc.stderr or "")
        if "-128" in err or "User canceled" in err:
            return False
        return None  # e.g. no GUI session
    if exe.startswith("powershell") or exe.startswith("pwsh"):
        if proc.returncode != 0:
            return None
        return out.splitlines()[-1].strip() == "6" if out else False
    if exe == "zenity":
        if proc.returncode in (0, 1, 5):
            return proc.returncode == 0
        return None
    if exe == "kdialog":
        if proc.returncode in (0, 1):
            return proc.returncode == 0
        return None
    return None


async def native_dialog_approver(req: ApprovalRequest) -> Optional[bool]:
    argv = native_dialog_argv(req)
    if argv is None:
        return None
    loop = asyncio.get_running_loop()
    try:
        proc = await loop.run_in_executor(None, _run, argv, req.timeout + 5)
    except subprocess.TimeoutExpired:
        return False
    except OSError as exc:
        log.debug("native dialog failed: %s", exc)
        return None
    return parse_native_result(argv, proc)


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


def _stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except (ValueError, AttributeError):
        return False


def _read_line(prompt: str) -> str:
    return input(prompt)


async def console_approver(req: ApprovalRequest) -> Optional[bool]:
    if not _stdin_is_tty():
        return None
    prompt = "\n[opendesk] " + req.message().replace("\n", "\n[opendesk] ") + "\nAllow? [y/N] "
    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(None, _read_line, prompt)
    return answer.strip().lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


def _builtin(name: str) -> Optional[Approver]:
    # Looked up at call time so tests can monkeypatch module attributes.
    mod = sys.modules[__name__]
    return {
        "native": getattr(mod, "native_dialog_approver"),
        "console": getattr(mod, "console_approver"),
    }.get(name)


def resolve_chain(
    configured: Optional[Sequence[str]] = None,
) -> list[str]:
    env = os.environ.get("OPENDESK_APPROVAL_CHAIN")
    if env is not None:
        names = [n.strip().lower() for n in env.split(",") if n.strip()]
    elif configured is not None:
        names = [str(n).lower() for n in configured]
    else:
        names = list(DEFAULT_CHAIN)
    return [n for n in names if n in ("mcp", "native", "console")]


async def request_approval(
    req: ApprovalRequest,
    *,
    mcp_approver: Optional[Approver] = None,
    chain: Optional[Sequence[str]] = None,
    timeout: Optional[float] = None,
) -> ApprovalOutcome:
    """Run the approver chain.  Never raises; denial is an outcome."""
    if timeout is not None:
        req.timeout = float(timeout)
    deadline = time.monotonic() + req.timeout
    for name in resolve_chain(chain):
        approver = mcp_approver if name == "mcp" else _builtin(name)
        if approver is None:
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ApprovalOutcome(False, name, f"approval timed out after {req.timeout:.0f}s")
        try:
            answer = await asyncio.wait_for(approver(req), timeout=remaining)
        except asyncio.TimeoutError:
            return ApprovalOutcome(False, name, f"approval timed out after {req.timeout:.0f}s")
        except Exception as exc:
            log.debug("approver %s failed: %s", name, exc)
            continue
        if answer is None:
            continue
        if answer:
            return ApprovalOutcome(True, name, f"approved via {name}")
        return ApprovalOutcome(False, name, f"the user denied it ({name})")
    return ApprovalOutcome(
        False, "none",
        "no way to ask the user for approval here (no MCP elicitation support, "
        "no desktop dialog, no TTY). Ask the user to allow it with a policy rule, "
        "e.g. `opendesk policy add-rule --tool "
        f"{req.tool}" + (f" --action {req.action}" if req.action else "")
        + " --decision allow`, or perform it themselves.",
    )
