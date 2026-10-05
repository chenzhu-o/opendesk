"""Tests for the hybrid CLI / filesystem tool (``system``)."""

from __future__ import annotations

import pytest

from opendesk.computer.types import CompletedCommand
from opendesk.tools.base import ToolContext
from opendesk.tools.system import SystemTool

from tests._fakes import FakeComputer


def _ctx(computer: FakeComputer) -> ToolContext:
    return ToolContext(session_id="test-system", computer=computer)


async def _run(tool: SystemTool, computer: FakeComputer, **params):
    return await tool.execute(_ctx(computer), tool.parse_params(params))


class TestShell:
    @pytest.mark.asyncio
    async def test_shell_routes_through_computer(self):
        comp = FakeComputer()
        result = await _run(SystemTool(), comp, action="shell", command="echo hi")
        assert not result.error
        assert "shell-out" in result.output
        assert ("shell", {"command": "echo hi"}) in comp.calls

    @pytest.mark.asyncio
    async def test_shell_without_command_is_error(self):
        result = await _run(SystemTool(), FakeComputer(), action="shell")
        assert result.error
        assert "command is required" in result.output

    @pytest.mark.asyncio
    async def test_shell_truncates_output(self):
        class BigComputer(FakeComputer):
            async def shell(self, command, *, timeout=None, cwd=None, env=None):
                return CompletedCommand(returncode=0, stdout=b"x" * 5000, stderr=b"")

        result = await _run(SystemTool(), BigComputer(), action="shell", command="x", max_bytes=100)
        assert "truncated" in result.output
        assert result.output.count("x") <= 200


class TestExec:
    @pytest.mark.asyncio
    async def test_exec_routes_argv(self):
        comp = FakeComputer()
        result = await _run(SystemTool(), comp, action="exec", argv=["git", "status"])
        assert not result.error
        assert ("exec", {"argv": ["git", "status"]}) in comp.calls

    @pytest.mark.asyncio
    async def test_exec_without_argv_is_error(self):
        result = await _run(SystemTool(), FakeComputer(), action="exec")
        assert result.error


class TestFilesystem:
    @pytest.mark.asyncio
    async def test_read_file_decodes_text(self):
        comp = FakeComputer()
        result = await _run(SystemTool(), comp, action="read_file", path="/tmp/x")
        assert not result.error
        assert "file-bytes" in result.output

    @pytest.mark.asyncio
    async def test_write_file(self):
        comp = FakeComputer()
        result = await _run(SystemTool(), comp, action="write_file", path="/tmp/x", content="hello")
        assert not result.error
        assert ("write_file", {"path": "/tmp/x", "data_len": 5}) in comp.calls

    @pytest.mark.asyncio
    async def test_write_file_without_content_is_error(self):
        result = await _run(SystemTool(), FakeComputer(), action="write_file", path="/tmp/x")
        assert result.error

    @pytest.mark.asyncio
    async def test_list_dir(self):
        from opendesk.computer.types import FileEntry

        class DirComputer(FakeComputer):
            async def list_dir(self, path):
                return [
                    FileEntry(path="a", name="a", is_dir=False, size=10),
                    FileEntry(path="b", name="b", is_dir=True, size=0),
                ]

        result = await _run(SystemTool(), DirComputer(), action="list_dir", path="/tmp")
        assert not result.error
        assert "2 entries" in result.output

    @pytest.mark.asyncio
    async def test_mkdir_and_move_and_delete(self):
        comp = FakeComputer()
        r1 = await _run(SystemTool(), comp, action="mkdir", path="/tmp/new")
        r2 = await _run(SystemTool(), comp, action="move", path="/tmp/a", dst="/tmp/b")
        r3 = await _run(SystemTool(), comp, action="delete", path="/tmp/c")
        assert not (r1.error or r2.error or r3.error)
        assert ("mkdir", {"path": "/tmp/new"}) in comp.calls
        assert ("move", {"src": "/tmp/a", "dst": "/tmp/b"}) in comp.calls
        assert ("delete", {"path": "/tmp/c"}) in comp.calls

    @pytest.mark.asyncio
    async def test_move_without_dst_is_error(self):
        result = await _run(SystemTool(), FakeComputer(), action="move", path="/tmp/a")
        assert result.error


class TestIntrospection:
    @pytest.mark.asyncio
    async def test_processes(self):
        result = await _run(SystemTool(), FakeComputer(), action="processes")
        assert not result.error

    @pytest.mark.asyncio
    async def test_environment(self):
        result = await _run(SystemTool(), FakeComputer(), action="environment")
        assert not result.error
        assert "os: fake" in result.output

    @pytest.mark.asyncio
    async def test_notifications(self):
        result = await _run(SystemTool(), FakeComputer(), action="notifications")
        assert not result.error

    @pytest.mark.asyncio
    async def test_missing_path_is_error(self):
        result = await _run(SystemTool(), FakeComputer(), action="read_file")
        assert result.error
        assert "path is required" in result.output


class TestAudit:
    @pytest.mark.asyncio
    async def test_records_action_with_replay_params(self):
        from opendesk.computer.sandbox import clear_sandbox, get_sandbox

        clear_sandbox("test-system")
        comp = FakeComputer()
        await _run(SystemTool(), comp, action="shell", command="echo hi")
        entries = get_sandbox("test-system").export_audit_log()
        assert entries and entries[-1]["action"] == "shell"
        assert entries[-1]["replay_params"]["tool"] == "system"
        clear_sandbox("test-system")
