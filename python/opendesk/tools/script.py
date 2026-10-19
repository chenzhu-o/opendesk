"""ScriptTool — run shell / AppleScript / JXA / PowerShell code on the active
:class:`~opendesk.computer.Computer`.

Scripts are the *fast path* of computer use: reading a file, flipping a
``defaults`` key, asking Finder for its selection via AppleScript, or querying
WMI from PowerShell is one call and deterministic, where the same task through
the GUI is a dozen screenshot → click round-trips.

Everything goes through ``ctx.computer.exec(argv, ...)`` (no host-side shell
interpolation), so the tool runs unchanged on a paired remote peer — the
remote's :class:`~opendesk.remote.policy.Policy` still gates ``process.exec``.
"""

from __future__ import annotations

import asyncio
import base64
import platform
import weakref
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import Field

from opendesk.tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from opendesk.computer.base import Computer


MAX_TIMEOUT = 600.0
DEFAULT_TIMEOUT = 60.0
# Per-stream output budget.  Above this we keep the head and the tail (errors
# usually live at the end) and say how much was dropped.
MAX_STREAM_CHARS = 16_000
_HEAD_CHARS = 12_000
_TAIL_CHARS = 4_000

# Normalised OS name per Computer — environment() can be slow on a remote, so
# ask once per connection.
_OS_CACHE: "weakref.WeakKeyDictionary[Computer, str]" = weakref.WeakKeyDictionary()


def normalize_os(raw: str) -> str:
    """Map an ``Environment.os`` / ``platform.system()`` value to darwin|windows|linux|unknown."""
    v = (raw or "").strip().lower()
    if v in ("darwin", "macos", "mac", "osx", "mac os x"):
        return "darwin"
    if v.startswith("win") or any(k in v for k in ("mingw", "cygwin", "msys")):
        return "windows"
    if v.startswith("linux") or v in ("freebsd", "openbsd", "netbsd"):
        return "linux"
    return "unknown"


async def target_os(computer: "Computer") -> str:
    """Return the normalised OS of *computer* (cached)."""
    from opendesk.computer.local import LocalComputer

    if isinstance(computer, LocalComputer):
        return normalize_os(platform.system())
    try:
        cached = _OS_CACHE.get(computer)
    except TypeError:
        cached = None
    if cached is not None:
        return cached
    try:
        env = await asyncio.wait_for(computer.environment(), timeout=15)
        result = normalize_os(env.os)
    except Exception:
        result = "unknown"
    try:
        _OS_CACHE[computer] = result
    except TypeError:
        pass
    return result


def build_argv(action: str, code: str, os_name: str) -> tuple[list[str], Optional[bytes]]:
    """Return ``(argv, stdin)`` for running *code* with *action* on *os_name*.

    Raises :class:`ValueError` when the action isn't available on that OS.
    """
    if action == "shell":
        if os_name == "windows":
            return ["cmd.exe", "/d", "/s", "/c", code], None
        return ["/bin/sh", "-c", code], None
    if action in ("applescript", "jxa"):
        if os_name not in ("darwin", "unknown"):
            raise ValueError(
                f"action={action!r} needs macOS (osascript); the target is {os_name}. "
                "Use action='shell'" + (" or 'powershell'" if os_name == "windows" else "")
                + " instead."
            )
        argv = ["osascript"] if action == "applescript" else ["osascript", "-l", "JavaScript"]
        # One multi-line -e works, and keeps the source in argv so a remote
        # operator approving `process.exec` (and its audit log) sees it.
        return argv + ["-e", code], None
    if action == "powershell":
        encoded = base64.b64encode(code.encode("utf-16-le")).decode("ascii")
        exe = "powershell.exe" if os_name == "windows" else "pwsh"
        return [exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], None
    raise ValueError(f"Unknown script action: {action!r}")


def truncate_stream(text: str, limit: int = MAX_STREAM_CHARS) -> tuple[str, bool]:
    """Keep head + tail of *text* when it exceeds *limit*; return ``(text, truncated)``."""
    if len(text) <= limit:
        return text, False
    head = min(_HEAD_CHARS, limit * 3 // 4)
    tail = max(0, limit - head)
    dropped = len(text) - head - tail
    body = (
        text[:head]
        + f"\n… [truncated: {dropped:,} of {len(text):,} chars omitted] …\n"
        + (text[-tail:] if tail else "")
    )
    return body, True


class ScriptTool(Tool):
    """Run a shell command, AppleScript, JXA, or PowerShell script."""

    name = "script"
    title = "Run script"
    annotations = {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True}
    description = (
        "Run code on the target machine and return exit code, stdout and stderr.\n\n"
        "ROUTING RULE — prefer this over GUI clicks whenever the task does not need "
        "visual interaction: reading/writing files, CLIs (git, curl, ffmpeg …), "
        "macOS `defaults` / `open` / `mdfind`, an app's AppleScript dictionary "
        "(Finder, Mail, Safari, Music, System Events), Windows PowerShell/WMI/registry. "
        "One script call usually replaces many screenshot → click steps and is "
        "deterministic. Fall back to `ui` (accessibility) and only then `mouse` "
        "when the task genuinely needs the visual UI.\n\n"
        "Actions:\n"
        "  shell       — platform shell (/bin/sh -c on macOS/Linux, cmd.exe on Windows)\n"
        "  applescript — osascript (macOS only), e.g. "
        "'tell application \"Finder\" to get name of every window'\n"
        "  jxa         — JavaScript for Automation, osascript -l JavaScript (macOS only)\n"
        "  powershell  — powershell.exe on Windows, pwsh elsewhere if installed\n\n"
        f"timeout defaults to {int(DEFAULT_TIMEOUT)}s (max {int(MAX_TIMEOUT)}). "
        "Long output is truncated (head and tail kept). A non-zero exit code is "
        "reported as an error."
    )

    class Params(Tool.Params):
        action: Literal["shell", "applescript", "jxa", "powershell"] = Field(
            description="Interpreter: shell, applescript, jxa, or powershell."
        )
        code: str = Field(description="The command / script source to run.")
        timeout: float = Field(
            default=DEFAULT_TIMEOUT, gt=0, le=MAX_TIMEOUT,
            description=f"Seconds before the process is killed (max {int(MAX_TIMEOUT)}).",
        )
        cwd: Optional[str] = Field(
            default=None, description="Working directory (default: the agent's own).",
        )

    async def execute(self, ctx: ToolContext, params: "ScriptTool.Params") -> ToolResult:
        from opendesk.computer.sandbox import ActionType, get_sandbox

        preview = params.code.strip().replace("\n", " ⏎ ")
        preview = preview[:200] + ("…" if len(preview) > 200 else "")
        await ctx.check_permission(
            tool="script", argument=f"{params.action}: {preview}",
            description=f"Run {params.action} script: {preview}",
        )

        sandbox = get_sandbox(ctx.session_id)
        audit_params = {
            "action": params.action,
            "code": params.code[:1000],
            "cwd": params.cwd,
            "timeout": params.timeout,
        }

        if not params.code.strip():
            return ToolResult(title="Script error", output="code must not be empty.", error=True)

        os_name = await target_os(ctx.computer)
        try:
            argv, stdin = build_argv(params.action, params.code, os_name)
        except ValueError as exc:
            await sandbox.record_action(ActionType.SCRIPT, audit_params, error=str(exc))
            return ToolResult(title="Script unsupported", output=str(exc), error=True)

        try:
            completed = await asyncio.wait_for(
                ctx.computer.exec(
                    argv, timeout=params.timeout, cwd=params.cwd, stdin=stdin,
                ),
                # Remote exec enforces `timeout` on the peer; this is the
                # backstop if the wire stalls.
                timeout=params.timeout + 15,
            )
        except asyncio.TimeoutError:
            msg = f"Timed out after {params.timeout:g}s (process killed)."
            await sandbox.record_action(ActionType.SCRIPT, audit_params, error=msg)
            return ToolResult(title="Script timeout", output=msg, error=True)
        except FileNotFoundError as exc:
            if params.cwd and getattr(exc, "filename", None) == params.cwd:
                msg = f"Working directory not found: {params.cwd}"
            else:
                msg = f"Interpreter not found for action={params.action!r} ({argv[0]}): {exc}"
            await sandbox.record_action(ActionType.SCRIPT, audit_params, error=msg)
            return ToolResult(title="Script error", output=msg, error=True)
        except Exception as exc:
            text = str(exc) or type(exc).__name__
            if "timeout" in type(exc).__name__.lower() or "timed out" in text.lower():
                text = f"Timed out after {params.timeout:g}s: {text}"
            await sandbox.record_action(ActionType.SCRIPT, audit_params, error=text)
            return ToolResult(
                title="Script error",
                output=f"{params.action} failed: {text}",
                error=True,
            )

        stdout, out_cut = truncate_stream(completed.stdout_text())
        stderr, err_cut = truncate_stream(completed.stderr_text())
        rc = completed.returncode

        lines = [f"Exit code: {rc}  ({completed.duration:.2f}s, {params.action} on {os_name})"]
        lines.append("--- stdout ---" + (" (truncated)" if out_cut else ""))
        lines.append(stdout.rstrip("\n") if stdout else "(empty)")
        if stderr or rc != 0:
            lines.append("--- stderr ---" + (" (truncated)" if err_cut else ""))
            lines.append(stderr.rstrip("\n") if stderr else "(empty)")

        await sandbox.record_action(
            ActionType.SCRIPT, audit_params,
            result=f"exit {rc}, {len(completed.stdout)}B stdout, {len(completed.stderr)}B stderr",
            error=None if rc == 0 else f"exit {rc}",
        )
        return ToolResult(
            title=f"Script ({params.action}) exit {rc}",
            output="\n".join(lines),
            error=rc != 0,
            metadata={
                "returncode": rc,
                "duration": completed.duration,
                "stdout_truncated": out_cut,
                "stderr_truncated": err_cut,
                "os": os_name,
            },
        )
