"""SystemTool — the hybrid CLI + filesystem action layer.

Modern computer-use research converges on the same finding: driving every
task through pixels is slow, expensive, and brittle. Orchestrating a *mix* of
GUI actions and native CLI / API calls ("hybrid interfaces") is faster and
more reliable whenever a task has a command-line equivalent.

The :class:`~opendesk.computer.Computer` abstraction has always exposed the
primitives for this — ``shell``, ``exec``, ``read_file``, ``write_file``,
``list_dir``, ``processes`` … — but until now no agent-facing tool surfaced
them, so agents could only click. This tool closes that gap.

Routing rule the tool description encodes, so any model that reads the tool
list follows it without extra orchestration:

    CLI / filesystem  ──(no equivalent)──►  ui (semantic AX)  ──►  SoM  ──►  pixels

Prefer ``system`` for anything expressible as a command or a file operation.
Drop to ``ui`` / ``mouse`` only for apps with no CLI surface (canvas UIs,
games, some desktop apps).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from opendesk.tools.base import Tool, ToolContext, ToolResult

#: Cap on returned text so a chatty command can't blow up the agent's context.
DEFAULT_MAX_BYTES = 20_000


class SystemTool(Tool):
    """Run commands and manipulate files on the active computer."""

    name = "system"
    description = (
        "Operate the computer through its command line and filesystem — the "
        "fast, deterministic alternative to clicking through the GUI.\n\n"
        "PREFER THIS OVER THE GUI TOOLS whenever a task can be expressed as a "
        "command or a file operation: it is faster, cheaper (no screenshots or "
        "vision-model calls), and reproducible across app versions. Fall back "
        "to `ui` (then `screenshot(marks=True)`, then `mouse`) only for apps "
        "that expose no command-line surface.\n\n"
        "Actions:\n"
        "- shell: run a command string through the platform shell "
        "(cwd/timeout supported)\n"
        "- exec: run an argv vector with no shell interpolation (safer)\n"
        "- read_file / write_file: read or overwrite a file's contents\n"
        "- list_dir / stat / mkdir / move / delete: filesystem operations\n"
        "- processes: list running processes\n"
        "- environment: OS, hostname, locale, timezone, displays\n"
        "- notifications: list recent desktop notifications"
    )

    class Params(Tool.Params):
        action: Literal[
            "shell",
            "exec",
            "read_file",
            "write_file",
            "list_dir",
            "stat",
            "mkdir",
            "move",
            "delete",
            "processes",
            "environment",
            "notifications",
        ] = Field(description="The system action to perform.")

        command: Optional[str] = Field(
            default=None,
            description="Command string to run. Required for action='shell'.",
        )
        argv: Optional[list[str]] = Field(
            default=None,
            description=(
                "Argument vector to spawn without a shell, e.g. "
                "['git', 'status']. Required for action='exec'."
            ),
        )
        path: Optional[str] = Field(
            default=None,
            description=(
                "Target path (file or directory, '~' is expanded). Used by "
                "read_file, write_file, list_dir, stat, mkdir and delete."
            ),
        )
        dst: Optional[str] = Field(
            default=None,
            description="Destination path. Required for action='move'.",
        )
        content: Optional[str] = Field(
            default=None,
            description="UTF-8 text to write. Required for action='write_file'.",
        )
        cwd: Optional[str] = Field(
            default=None,
            description="Working directory for 'shell' / 'exec'.",
        )
        timeout: Optional[float] = Field(
            default=None,
            description="Seconds before the command is killed. Default: backend-defined.",
        )
        max_bytes: int = Field(
            default=DEFAULT_MAX_BYTES,
            description=(
                "Truncate returned stdout/stderr/file text to this many bytes. "
                "Increase for large outputs you genuinely need."
            ),
        )

    async def execute(self, ctx: ToolContext, params: "SystemTool.Params") -> ToolResult:
        from opendesk.computer.sandbox import ActionType, get_sandbox

        action = params.action
        arg_desc = _describe_arg(params)
        await ctx.check_permission(
            tool="system",
            argument=f"{action} {arg_desc}".strip(),
            description=f"System: {action} {arg_desc}".strip(),
        )

        sandbox = get_sandbox(ctx.session_id)
        replay_params = _replay_params(params)

        try:
            output = await self._dispatch(ctx, params)
        except Exception as exc:  # surfaced to the agent as a retryable error
            await sandbox.record_action(
                _action_type(ActionType, action), replay_params["params"],
                error=str(exc), replay_params=replay_params,
            )
            return ToolResult(
                title=f"system: {action} failed",
                output=f"`{action}` failed: {exc}",
                error=True,
            )

        await sandbox.record_action(
            _action_type(ActionType, action), replay_params["params"],
            result=output[:200], replay_params=replay_params,
        )
        return ToolResult(title=f"system: {action}", output=output)

    # ------------------------------------------------------------------

    async def _dispatch(self, ctx: ToolContext, params: "SystemTool.Params") -> str:
        action = params.action

        if action == "shell":
            if not params.command:
                raise ValueError("command is required for action='shell'")
            completed = await ctx.computer.shell(
                params.command, timeout=params.timeout, cwd=params.cwd,
            )
            return _format_command(completed, params.max_bytes)

        if action == "exec":
            if not params.argv:
                raise ValueError("argv is required for action='exec'")
            completed = await ctx.computer.exec(
                list(params.argv), timeout=params.timeout, cwd=params.cwd,
            )
            return _format_command(completed, params.max_bytes)

        if action == "read_file":
            path = _require_path(params)
            data = await ctx.computer.read_file(path)
            text = data.decode("utf-8", errors="replace")
            return _truncate(text, params.max_bytes, f"contents of {path}")

        if action == "write_file":
            path = _require_path(params)
            if params.content is None:
                raise ValueError("content is required for action='write_file'")
            data = params.content.encode("utf-8")
            await ctx.computer.write_file(path, data)
            return f"Wrote {len(data)} bytes to {path}."

        if action == "list_dir":
            path = _require_path(params)
            entries = await ctx.computer.list_dir(path)
            if not entries:
                return f"{path} is empty."
            lines = [f"Contents of {path} ({len(entries)} entries):"]
            for e in sorted(entries, key=lambda x: (not x.is_dir, x.name)):
                kind = "dir " if e.is_dir else "file"
                size = "" if e.is_dir else f"  {e.size} B"
                lines.append(f"  [{kind}] {e.name}{size}")
            return "\n".join(lines)

        if action == "stat":
            path = _require_path(params)
            e = await ctx.computer.stat(path)
            kind = "directory" if e.is_dir else "file"
            return (
                f"{e.path}\n  type: {kind}\n  size: {e.size} B\n"
                f"  mtime: {e.mtime}\n  mode: {oct(e.mode) if e.mode is not None else '—'}"
            )

        if action == "mkdir":
            path = _require_path(params)
            await ctx.computer.mkdir(path, parents=True)
            return f"Created directory {path}."

        if action == "move":
            if not params.path or not params.dst:
                raise ValueError("both path and dst are required for action='move'")
            await ctx.computer.move(params.path, params.dst)
            return f"Moved {params.path} -> {params.dst}."

        if action == "delete":
            path = _require_path(params)
            await ctx.computer.delete(path)
            return f"Deleted {path}."

        if action == "processes":
            procs = await ctx.computer.processes()
            if not procs:
                return "No processes reported."
            lines = [f"Running processes ({len(procs)}):"]
            for p in procs:
                cmd = " ".join(p.cmdline) if p.cmdline else p.name
                lines.append(f"  {p.pid:>7}  {cmd[:120]}")
            return "\n".join(lines)

        if action == "environment":
            env = await ctx.computer.environment()
            displays = ", ".join(
                f"{d.id}{'*' if d.primary else ''} "
                f"{d.bounds.width}x{d.bounds.height}"
                for d in env.displays
            )
            return (
                f"os: {env.os} {env.os_version}\n"
                f"hostname: {env.hostname}\n"
                f"locale: {env.locale}\ntimezone: {env.timezone}\n"
                f"displays: {displays or '—'}"
            )

        if action == "notifications":
            notes = await ctx.computer.notifications()
            if not notes:
                return "No notifications."
            lines = [f"Notifications ({len(notes)}):"]
            for n in notes:
                app = f"[{n.app}] " if n.app else ""
                lines.append(f"  {app}{n.title}: {n.body[:120]}")
            return "\n".join(lines)

        raise ValueError(f"Unknown system action: {action!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_path(params: "SystemTool.Params") -> str:
    if not params.path:
        raise ValueError(f"path is required for action='{params.action}'")
    return params.path


def _truncate(text: str, max_bytes: int, what: str) -> str:
    if max_bytes <= 0:
        max_bytes = DEFAULT_MAX_BYTES
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text or f"({what} is empty)"
    kept = encoded[:max_bytes].decode("utf-8", errors="replace")
    return f"{kept}\n\n… truncated ({len(encoded)} bytes total, showing {max_bytes})."


def _format_command(completed, max_bytes: int) -> str:
    parts = [f"exit code: {completed.returncode}"]
    if completed.duration:
        parts.append(f"duration: {completed.duration:.2f}s")
    stdout = completed.stdout_text()
    stderr = completed.stderr_text()
    if stdout:
        parts.append(_truncate(stdout.rstrip(), max_bytes, "stdout"))
    if stderr:
        parts.append("stderr:\n" + _truncate(stderr.rstrip(), max_bytes, "stderr"))
    if not stdout and not stderr:
        parts.append("(no output)")
    return "\n".join(parts)


def _describe_arg(params: "SystemTool.Params") -> str:
    if params.action == "shell":
        return (params.command or "")[:80]
    if params.action == "exec":
        return " ".join(params.argv or [])[:80]
    if params.action == "move":
        return f"{params.path} -> {params.dst}"
    return params.path or ""


def _action_type(ActionType, action: str):
    return {
        "shell": ActionType.SHELL,
        "exec": ActionType.EXEC,
        "read_file": ActionType.FILE_READ,
        "write_file": ActionType.FILE_WRITE,
        "list_dir": ActionType.FILE_LIST,
        "stat": ActionType.FILE_STAT,
        "mkdir": ActionType.FILE_MKDIR,
        "move": ActionType.FILE_MOVE,
        "delete": ActionType.FILE_DELETE,
        "processes": ActionType.PROCESS_LIST,
        "environment": ActionType.ENVIRONMENT,
        "notifications": ActionType.NOTIFICATIONS,
    }[action]


def _replay_params(params: "SystemTool.Params") -> dict:
    """Platform-neutral, complete parameter set for re-execution."""
    return {
        "tool": "system",
        "params": {
            "action": params.action,
            "command": params.command,
            "argv": params.argv,
            "path": params.path,
            "dst": params.dst,
            "content": params.content,
            "cwd": params.cwd,
            "timeout": params.timeout,
        },
    }
