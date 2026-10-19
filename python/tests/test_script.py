"""Tests for the ``script`` tool (shell / AppleScript / JXA / PowerShell)."""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

import pytest

from opendesk.computer import ComputerDispatcher, CompletedCommand, Environment, RemoteComputer
from opendesk.computer.sandbox import clear_sandbox, get_sandbox
from opendesk.integrations.mcp import MCPDispatcher, PEER_AWARE_TOOLS, TextResult
from opendesk.integrations.mcp_session import MCPSession
from opendesk.protocol import LoopbackConnection, Peer
from opendesk.protocol.auth import Identity, TrustedPeers
from opendesk.registry import create_registry
from opendesk.tools.base import ToolContext
from opendesk.tools.script import (
    MAX_STREAM_CHARS,
    ScriptTool,
    build_argv,
    normalize_os,
    truncate_stream,
)

from tests._fakes import FakeComputer


class OSFakeComputer(FakeComputer):
    """FakeComputer that reports an OS and returns a configurable exec result."""

    def __init__(self, os_name: str = "Darwin", *, returncode: int = 0,
                 stdout: bytes = b"out", stderr: bytes = b"", raise_exc=None) -> None:
        super().__init__()
        self.os_name = os_name
        self.result = CompletedCommand(returncode=returncode, stdout=stdout, stderr=stderr, duration=0.01)
        self.raise_exc = raise_exc
        self.env_calls = 0

    async def environment(self) -> Environment:
        self.env_calls += 1
        return Environment(os=self.os_name)

    async def exec(self, argv, *, timeout=None, cwd=None, env=None, stdin=None) -> CompletedCommand:
        self._record("exec", argv=list(argv), timeout=timeout, cwd=cwd, stdin=stdin)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.result


def _ctx(computer, session_id: str = "script-test") -> ToolContext:
    clear_sandbox(session_id)
    return ToolContext(session_id=session_id, computer=computer)


async def _run(computer, session_id: str = "script-test", **params):
    tool = ScriptTool()
    return await tool.execute(_ctx(computer, session_id), tool.parse_params(params))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    @pytest.mark.parametrize("raw,expected", [
        ("Darwin", "darwin"), ("macOS", "darwin"), ("Windows", "windows"),
        ("win32", "windows"), ("MINGW64_NT-10.0", "windows"), ("Linux", "linux"),
        ("fake", "unknown"), ("", "unknown"),
    ])
    def test_normalize_os(self, raw, expected):
        assert normalize_os(raw) == expected

    def test_shell_posix_and_windows(self):
        assert build_argv("shell", "ls -la", "darwin") == (["/bin/sh", "-c", "ls -la"], None)
        assert build_argv("shell", "ls", "linux")[0][0] == "/bin/sh"
        argv, stdin = build_argv("shell", "dir", "windows")
        assert argv[0] == "cmd.exe" and argv[-1] == "dir" and stdin is None

    def test_applescript_and_jxa_keep_code_in_argv(self):
        argv, stdin = build_argv("applescript", 'set x to 1\nreturn "hi"', "darwin")
        assert argv == ["osascript", "-e", 'set x to 1\nreturn "hi"'] and stdin is None
        argv, stdin = build_argv("jxa", "1 + 1", "darwin")
        assert argv == ["osascript", "-l", "JavaScript", "-e", "1 + 1"] and stdin is None

    @pytest.mark.parametrize("os_name", ["linux", "windows"])
    def test_applescript_unsupported_off_macos(self, os_name):
        with pytest.raises(ValueError, match="macOS"):
            build_argv("applescript", "x", os_name)
        with pytest.raises(ValueError, match="macOS"):
            build_argv("jxa", "x", os_name)

    def test_powershell_encoded_command(self):
        argv, stdin = build_argv("powershell", "Get-Date\nWrite-Output 'é'", "windows")
        assert argv[0] == "powershell.exe" and stdin is None
        assert "-EncodedCommand" in argv
        decoded = base64.b64decode(argv[-1]).decode("utf-16-le")
        assert decoded == "Get-Date\nWrite-Output 'é'"
        assert build_argv("powershell", "x", "darwin")[0][0] == "pwsh"

    def test_truncate_keeps_head_and_tail(self):
        text = "A" * 20_000 + "TAIL"
        out, cut = truncate_stream(text)
        assert cut
        assert out.startswith("A" * 100)
        assert out.endswith("TAIL")
        assert "truncated" in out
        assert len(out) < len(text)
        same, cut2 = truncate_stream("short")
        assert same == "short" and not cut2


# ---------------------------------------------------------------------------
# Tool behaviour against a fake Computer
# ---------------------------------------------------------------------------


class TestScriptTool:
    def test_metadata_contract(self):
        assert ScriptTool.name == "script"
        assert ScriptTool.title
        assert ScriptTool.annotations == {
            "readOnlyHint": False, "destructiveHint": True, "openWorldHint": True,
        }
        # The description must teach the routing rule.
        d = ScriptTool.description.lower()
        assert "prefer" in d and "ui" in d and "mouse" in d

    async def test_routes_through_computer_exec(self):
        comp = OSFakeComputer("Darwin", stdout=b"hello\n")
        result = await _run(comp, action="shell", code="echo hello", cwd="/tmp", timeout=5)
        assert not result.error
        assert "Exit code: 0" in result.output and "hello" in result.output
        method, kw = comp.calls[-1]
        assert method == "exec"
        assert kw["argv"] == ["/bin/sh", "-c", "echo hello"]
        assert kw["cwd"] == "/tmp" and kw["timeout"] == 5

    async def test_applescript_on_macos_passes_code_with_e(self):
        comp = OSFakeComputer("Darwin", stdout=b"3\n")
        result = await _run(comp, action="applescript", code="return 1 + 2")
        assert not result.error
        _, kw = comp.calls[-1]
        assert kw["argv"] == ["osascript", "-e", "return 1 + 2"] and kw["stdin"] is None

    async def test_applescript_on_windows_is_clear_error(self):
        comp = OSFakeComputer("Windows")
        result = await _run(comp, action="applescript", code="beep")
        assert result.error
        assert "macOS" in result.output
        assert not any(c[0] == "exec" for c in comp.calls)

    async def test_nonzero_exit_is_error_with_stderr(self):
        comp = OSFakeComputer("Linux", returncode=2, stdout=b"", stderr=b"boom")
        result = await _run(comp, action="shell", code="false")
        assert result.error
        assert "Exit code: 2" in result.output and "boom" in result.output
        assert result.metadata["returncode"] == 2

    async def test_long_output_is_truncated_and_says_so(self):
        comp = OSFakeComputer("Linux", stdout=b"x" * (MAX_STREAM_CHARS * 3))
        result = await _run(comp, action="shell", code="yes")
        assert "truncated" in result.output
        assert result.metadata["stdout_truncated"] is True
        assert len(result.output) < MAX_STREAM_CHARS * 2

    async def test_timeout_reported(self):
        comp = OSFakeComputer("Linux", raise_exc=asyncio.TimeoutError())
        result = await _run(comp, action="shell", code="sleep 100", timeout=1)
        assert result.error and "Timed out" in result.output

    async def test_missing_interpreter_reported(self):
        comp = OSFakeComputer("Darwin", raise_exc=FileNotFoundError(2, "No such file", "pwsh"))
        result = await _run(comp, action="powershell", code="Get-Date")
        assert result.error and "not found" in result.output

    async def test_empty_code_rejected(self):
        result = await _run(OSFakeComputer(), action="shell", code="   ")
        assert result.error

    def test_timeout_bounds(self):
        tool = ScriptTool()
        with pytest.raises(Exception):
            tool.parse_params({"action": "shell", "code": "x", "timeout": 601})
        assert tool.parse_params({"action": "shell", "code": "x"}).timeout == 60

    async def test_records_audit_entry(self):
        comp = OSFakeComputer("Darwin")
        await _run(comp, session_id="script-audit", action="shell", code="echo hi")
        log = get_sandbox("script-audit").export_audit_log()
        assert log and log[-1]["action"] == "script"
        assert log[-1]["params"]["code"] == "echo hi"
        assert log[-1]["result"].startswith("exit 0")

    async def test_permission_handler_sees_code(self):
        seen = []

        async def handler(tool, argument, description):
            seen.append((tool, argument))

        comp = OSFakeComputer("Darwin")
        tool = ScriptTool()
        ctx = ToolContext(session_id="perm", permission_handler=handler, computer=comp)
        await tool.execute(ctx, tool.parse_params({"action": "shell", "code": "rm -rf /tmp/x"}))
        assert seen and seen[0][0] == "script" and "rm -rf /tmp/x" in seen[0][1]

    async def test_os_lookup_is_cached(self):
        comp = OSFakeComputer("Linux")
        await _run(comp, action="shell", code="a")
        await _run(comp, action="shell", code="b")
        assert comp.env_calls == 1

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
    async def test_real_local_shell(self, tmp_path: Path):
        from opendesk.computer import LocalComputer
        result = await _run(
            LocalComputer(), action="shell",
            code="echo out; echo err 1>&2; pwd", cwd=str(tmp_path),
        )
        assert not result.error
        assert "out" in result.output and "err" in result.output
        assert tmp_path.name in result.output


# ---------------------------------------------------------------------------
# Remote peers: over the wire and through MCP
# ---------------------------------------------------------------------------


class TestScriptRemote:
    async def test_runs_on_remote_peer_over_protocol(self):
        fake = OSFakeComputer("Darwin", stdout=b"remote-ok")
        a, b = LoopbackConnection.pair()
        server_peer = Peer(b, role="server", dispatcher=ComputerDispatcher(fake))

        async def _server_hello():
            await server_peer.hello(fake.capabilities().model_dump())
            server_peer.start()

        task = asyncio.create_task(_server_hello())
        remote = await RemoteComputer.connect(a)
        await task
        try:
            result = await _run(remote, action="jxa", code="'x'.repeat(2)")
            assert not result.error, result.output
            assert "remote-ok" in result.output
            method, kw = fake.calls[-1]
            assert method == "exec"
            assert kw["argv"] == ["osascript", "-l", "JavaScript", "-e", "'x'.repeat(2)"]
        finally:
            await remote.aclose()
            await server_peer.aclose()

    async def test_mcp_peer_routing(self, tmp_path: Path):
        assert "script" in PEER_AWARE_TOOLS
        local = OSFakeComputer("Darwin")
        remote = OSFakeComputer("Linux", stdout=b"from-mini")
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="mini")
        session = MCPSession(home=tmp_path, local=local)
        session._connections["mini"] = remote  # type: ignore[attr-defined]
        dispatcher = MCPDispatcher(create_registry(), session)

        entries = {e.name: e for e in await dispatcher.list_tools()}
        assert "peer" in entries["script"].schema["properties"]

        out = await dispatcher.call_tool(
            "script", {"action": "shell", "code": "uname", "peer": "mini"},
        )
        text = "\n".join(r.text for r in out if isinstance(r, TextResult))
        assert "[on mini]" in text and "from-mini" in text
        assert any(c[0] == "exec" for c in remote.calls)
        assert not any(c[0] == "exec" for c in local.calls)
