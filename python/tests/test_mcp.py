"""Tests for the MCP integration.

Exercises the library-neutral :class:`MCPDispatcher` directly so the tests
don't depend on running an MCP transport.  The dispatcher is what
:func:`create_mcp_server` wraps with ``mcp.types.*`` conversion, so testing
it covers the full agent-facing surface.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from opendesk.computer import LocalComputer
from opendesk.integrations.mcp import (
    CallResult,
    MCPDispatcher,
    PEER_AWARE_TOOLS,
    PEER_FIELD_DESCRIPTION,
    SERVER_INSTRUCTIONS,
    TextResult,
    ImageResult,
)
from opendesk.integrations.mcp_session import LOCAL, MCPSession, MCPSessionError
from opendesk.protocol.auth import Identity, TrustedPeers
from opendesk.registry import create_registry

from tests._fakes import FakeComputer


# ---------------------------------------------------------------------------
# MCPSession unit tests (no MCP package involved)
# ---------------------------------------------------------------------------


class TestMCPSession:
    @pytest.mark.asyncio
    async def test_default_resolves_to_local_with_no_peers(self, tmp_path: Path):
        local = FakeComputer()
        session = MCPSession(home=tmp_path, local=local)
        computer, name = await session.resolve()
        assert computer is local
        assert name == LOCAL

    @pytest.mark.asyncio
    async def test_explicit_local_resolves_to_local(self, tmp_path: Path):
        local = FakeComputer()
        session = MCPSession(home=tmp_path, local=local)
        computer, name = await session.resolve("local")
        assert computer is local
        assert name == LOCAL

    def test_use_peer_validates_against_trusted(self, tmp_path: Path):
        session = MCPSession(home=tmp_path)
        with pytest.raises(MCPSessionError):
            session.use_peer("nonexistent")

    def test_use_peer_accepts_trusted(self, tmp_path: Path):
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="mini")
        session = MCPSession(home=tmp_path)
        session.use_peer("mini")
        assert session.current_peer == "mini"

    def test_use_local_reverts(self, tmp_path: Path):
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="mini")
        session = MCPSession(home=tmp_path)
        session.use_peer("mini")
        session.use_peer("local")
        assert session.current_peer is None


class TestImplicitDefault:
    """Single paired peer = implicit default — the ergonomic single-machine case."""

    @pytest.mark.asyncio
    async def test_one_trusted_peer_becomes_implicit_default(self, tmp_path: Path):
        local = FakeComputer()
        remote = FakeComputer()
        session = _session_with_remote(tmp_path, local, remote, name="mini")
        # No explicit use_peer() call.
        assert session.current_peer is None
        name, source = session.effective_peer()
        assert name == "mini" and source == "implicit"
        # resolve() without an argument picks the implicit one.
        computer, resolved = await session.resolve()
        assert resolved == "mini"
        assert computer is remote

    @pytest.mark.asyncio
    async def test_zero_peers_resolves_to_local(self, tmp_path: Path):
        local = FakeComputer()
        session = MCPSession(home=tmp_path, local=local)
        name, source = session.effective_peer()
        assert name is None and source == "local"
        computer, resolved = await session.resolve()
        assert computer is local and resolved == LOCAL

    @pytest.mark.asyncio
    async def test_two_trusted_peers_no_default_raises(self, tmp_path: Path):
        """Multiple peers + no explicit default = ambiguous; must raise."""
        local = FakeComputer()
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="mini")
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="desktop")
        session = MCPSession(home=tmp_path, local=local)
        name, source = session.effective_peer()
        assert name is None and source == "ambiguous"
        with pytest.raises(MCPSessionError) as exc_info:
            await session.resolve()
        assert "mini" in str(exc_info.value)
        assert "desktop" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_two_peers_explicit_local_still_works(self, tmp_path: Path):
        """The ambiguous case is only triggered by an *omitted* peer arg."""
        local = FakeComputer()
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="mini")
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="desktop")
        session = MCPSession(home=tmp_path, local=local)
        computer, resolved = await session.resolve("local")
        assert computer is local and resolved == LOCAL

    @pytest.mark.asyncio
    async def test_two_peers_explicit_peer_arg_works(self, tmp_path: Path):
        """Per-call `peer:` resolves a specific target in the ambiguous setup."""
        local = FakeComputer()
        remote = FakeComputer()
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="desktop")
        session = _session_with_remote(tmp_path, local, remote, name="mini")
        # Now there are two trusted peers (mini + desktop) but mini is cached.
        computer, resolved = await session.resolve("mini")
        assert computer is remote and resolved == "mini"

    @pytest.mark.asyncio
    async def test_ambiguous_resolution_through_tool_call(self, tmp_path: Path):
        """Computer-tool dispatch surfaces the ambiguity as a clear error."""
        local = FakeComputer()
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="mini")
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="desktop")
        session = MCPSession(home=tmp_path, local=local)
        dispatcher = MCPDispatcher(create_registry(), session)
        result = await dispatcher.call_tool("clipboard", {"action": "read"})
        text = result[0].text  # type: ignore[union-attr]
        assert "ERROR" in text
        assert "ambiguous" in text.lower() or "multiple peers" in text.lower()

    @pytest.mark.asyncio
    async def test_admin_status_flags_ambiguous(self, tmp_path: Path):
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="mini")
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="desktop")
        session = MCPSession(home=tmp_path, local=FakeComputer())
        dispatcher = MCPDispatcher(create_registry(), session)
        out = await dispatcher.call_tool("opendesk_status", {})
        text = out[0].text  # type: ignore[union-attr]
        # Status should flag ambiguity and tell the agent how to resolve.
        assert "multiple peers" in text.lower() or "ambiguous" in text.lower()
        assert "opendesk_use" in text

    @pytest.mark.asyncio
    async def test_explicit_local_overrides_implicit_default(self, tmp_path: Path):
        local = FakeComputer()
        remote = FakeComputer()
        session = _session_with_remote(tmp_path, local, remote, name="mini")
        computer, resolved = await session.resolve("local")
        assert computer is local and resolved == LOCAL

    @pytest.mark.asyncio
    async def test_explicit_use_peer_takes_precedence_over_implicit(self, tmp_path: Path):
        """If a user `use_peer('mini')`s then later pairs a second peer,
        the explicit choice stays in effect (source = 'explicit')."""
        local = FakeComputer()
        remote = FakeComputer()
        session = _session_with_remote(tmp_path, local, remote, name="mini")
        session.use_peer("mini")
        # Pair a second peer; the implicit-default fallback no longer applies.
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="desktop")
        name, source = session.effective_peer()
        assert name == "mini" and source == "explicit"


# ---------------------------------------------------------------------------
# Tool listing
# ---------------------------------------------------------------------------


class TestListTools:
    @pytest.mark.asyncio
    async def test_lists_computer_tools_with_peer_field(self, tmp_path: Path):
        dispatcher = MCPDispatcher(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        entries = await dispatcher.list_tools()
        by_name = {e.name: e for e in entries}

        # Computer-use tool: should have `peer` in its schema's properties,
        # described once there rather than repeated in every description.
        screenshot = by_name["screenshot"]
        assert "peer" in screenshot.schema["properties"]
        assert screenshot.schema["properties"]["peer"]["description"] == PEER_FIELD_DESCRIPTION

        # Local-only tool: no `peer` field.
        assert "learn" in by_name
        assert "peer" not in by_name["learn"].schema.get("properties", {})

    @pytest.mark.asyncio
    async def test_admin_tools_listed(self, tmp_path: Path):
        dispatcher = MCPDispatcher(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        names = {e.name for e in await dispatcher.list_tools()}
        for admin in (
            "opendesk_peers", "opendesk_discover", "opendesk_use",
            "opendesk_status", "opendesk_capabilities", "opendesk_disconnect",
        ):
            assert admin in names, f"missing {admin}"


# ---------------------------------------------------------------------------
# Computer-tool routing
# ---------------------------------------------------------------------------


class TestComputerToolRouting:
    @pytest.mark.asyncio
    async def test_default_routes_to_local(self, tmp_path: Path):
        local = FakeComputer()
        dispatcher = MCPDispatcher(
            create_registry(), MCPSession(home=tmp_path, local=local),
        )
        result = await dispatcher.call_tool("clipboard", {"action": "write", "text": "hi"})
        assert any(isinstance(r, TextResult) for r in result)
        assert any(c[0] == "clipboard_write" for c in local.calls)

    @pytest.mark.asyncio
    async def test_explicit_peer_routes_to_remote(self, tmp_path: Path):
        local = FakeComputer()
        remote = FakeComputer()
        session = _session_with_remote(tmp_path, local, remote, name="mini")
        dispatcher = MCPDispatcher(create_registry(), session)

        result = await dispatcher.call_tool(
            "clipboard", {"action": "write", "text": "remote-hi", "peer": "mini"},
        )
        # Output prefixed with "[on mini]"
        text = "\n".join(r.text for r in result if isinstance(r, TextResult))
        assert "[on mini]" in text
        # Remote received the write; local did not.
        assert any(c[0] == "clipboard_write" for c in remote.calls)
        assert not any(c[0] == "clipboard_write" for c in local.calls)

    @pytest.mark.asyncio
    async def test_default_peer_routes_to_remote(self, tmp_path: Path):
        local = FakeComputer()
        remote = FakeComputer()
        session = _session_with_remote(tmp_path, local, remote, name="mini")
        session.use_peer("mini")
        dispatcher = MCPDispatcher(create_registry(), session)

        await dispatcher.call_tool("clipboard", {"action": "write", "text": "x"})
        assert any(c[0] == "clipboard_write" for c in remote.calls)
        assert not any(c[0] == "clipboard_write" for c in local.calls)

    @pytest.mark.asyncio
    async def test_explicit_local_overrides_default(self, tmp_path: Path):
        local = FakeComputer()
        remote = FakeComputer()
        session = _session_with_remote(tmp_path, local, remote, name="mini")
        session.use_peer("mini")
        dispatcher = MCPDispatcher(create_registry(), session)

        await dispatcher.call_tool(
            "clipboard", {"action": "write", "text": "back", "peer": "local"},
        )
        assert any(c[0] == "clipboard_write" for c in local.calls)
        # remote is untouched
        assert not any(c[0] == "clipboard_write" for c in remote.calls)

    @pytest.mark.asyncio
    async def test_unknown_peer_returns_error(self, tmp_path: Path):
        dispatcher = MCPDispatcher(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        result = await dispatcher.call_tool(
            "clipboard", {"action": "read", "peer": "doesnotexist"},
        )
        text = result[0].text  # type: ignore[union-attr]
        assert "ERROR" in text
        assert "doesnotexist" in text


# ---------------------------------------------------------------------------
# Admin tool dispatch
# ---------------------------------------------------------------------------


class TestAdminTools:
    @pytest.mark.asyncio
    async def test_peers_lists_local_and_trusted(self, tmp_path: Path):
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="mini")
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="desktop")
        session = MCPSession(home=tmp_path, local=FakeComputer())
        dispatcher = MCPDispatcher(create_registry(), session)
        out = await dispatcher.call_tool("opendesk_peers", {})
        text = out[0].text  # type: ignore[union-attr]
        assert "local" in text
        assert "mini" in text
        assert "desktop" in text

    @pytest.mark.asyncio
    async def test_use_then_status_reflects_default(self, tmp_path: Path):
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="mini")
        session = MCPSession(home=tmp_path, local=FakeComputer())
        dispatcher = MCPDispatcher(create_registry(), session)

        await dispatcher.call_tool("opendesk_use", {"peer": "mini"})
        out = await dispatcher.call_tool("opendesk_status", {})
        assert "mini" in out[0].text  # type: ignore[union-attr]
        assert session.current_peer == "mini"

    @pytest.mark.asyncio
    async def test_use_local_reverts(self, tmp_path: Path):
        TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name="mini")
        session = MCPSession(home=tmp_path, local=FakeComputer())
        dispatcher = MCPDispatcher(create_registry(), session)

        session.use_peer("mini")
        await dispatcher.call_tool("opendesk_use", {"peer": "local"})
        assert session.current_peer is None

    @pytest.mark.asyncio
    async def test_capabilities_for_local(self, tmp_path: Path):
        session = MCPSession(home=tmp_path, local=FakeComputer())
        dispatcher = MCPDispatcher(create_registry(), session)
        out = await dispatcher.call_tool("opendesk_capabilities", {})
        text = out[0].text  # type: ignore[union-attr]
        assert "fake" in text
        assert "display.capture" in text

    @pytest.mark.asyncio
    async def test_disconnect_specific_peer(self, tmp_path: Path):
        local = FakeComputer()
        remote = FakeComputer()
        session = _session_with_remote(tmp_path, local, remote, name="mini")
        dispatcher = MCPDispatcher(create_registry(), session)

        # Force a connection by routing a call there.
        await dispatcher.call_tool("clipboard", {"action": "read", "peer": "mini"})
        assert "mini" in session.active_peer_names()

        await dispatcher.call_tool("opendesk_disconnect", {"peer": "mini"})
        assert "mini" not in session.active_peer_names()


# ---------------------------------------------------------------------------
# Tool metadata: titles, annotations, descriptions
# ---------------------------------------------------------------------------


ANNOTATION_KEYS = {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"}


def _dispatcher(tmp_path: Path, local: FakeComputer | None = None) -> MCPDispatcher:
    return MCPDispatcher(create_registry(), MCPSession(home=tmp_path, local=local or FakeComputer()))


class TestToolMetadata:
    @pytest.mark.asyncio
    async def test_every_tool_has_title_and_annotations(self, tmp_path: Path):
        entries = await _dispatcher(tmp_path).list_tools()
        assert len(entries) >= 18
        for e in entries:
            assert e.title and e.title != e.name, f"{e.name} has no human title"
            assert set(e.annotations) <= ANNOTATION_KEYS, e.name
            assert isinstance(e.annotations.get("readOnlyHint"), bool), e.name
            if not e.annotations["readOnlyHint"]:
                assert isinstance(e.annotations.get("destructiveHint"), bool), e.name

    @pytest.mark.asyncio
    async def test_expected_hints(self, tmp_path: Path):
        by_name = {e.name: e.annotations for e in await _dispatcher(tmp_path).list_tools()}
        for ro in ("screenshot", "ocr", "memory", "opendesk_peers", "opendesk_status",
                   "opendesk_describe", "opendesk_capabilities", "opendesk_discover"):
            assert by_name[ro]["readOnlyHint"] is True, ro
        for act in ("mouse", "keyboard", "app", "ui", "learn", "schedule", "audit", "memory_manage"):
            assert by_name[act]["readOnlyHint"] is False, act
            assert by_name[act]["destructiveHint"] is True, act
        for act in ("mouse", "keyboard", "app", "ui"):
            assert by_name[act]["openWorldHint"] is True, act
        assert by_name["clipboard"]["destructiveHint"] is False
        for idem in ("opendesk_use", "opendesk_disconnect"):
            assert by_name[idem]["readOnlyHint"] is False
            assert by_name[idem]["destructiveHint"] is False
            assert by_name[idem]["idempotentHint"] is True

    @pytest.mark.asyncio
    async def test_peer_sentence_not_repeated_in_descriptions(self, tmp_path: Path):
        entries = await _dispatcher(tmp_path).list_tools()
        for e in entries:
            assert "Routes to the session's default peer" not in e.description
            assert "to target a remote machine instead" not in e.description
        peer_aware = [e for e in entries if e.name in PEER_AWARE_TOOLS]
        assert peer_aware and all("peer" in e.schema["properties"] for e in peer_aware)

    @pytest.mark.asyncio
    async def test_memory_split_into_read_and_manage(self, tmp_path: Path):
        by_name = {e.name: e for e in await _dispatcher(tmp_path).list_tools()}
        assert set(by_name["memory"].schema["properties"]["action"]["enum"]) == {
            "search", "show", "timeline", "status",
        }
        assert set(by_name["memory_manage"].schema["properties"]["action"]["enum"]) == {
            "pause", "resume", "deny", "config", "delete", "export",
        }
        assert "peer" not in by_name["memory_manage"].schema["properties"]

    def test_custom_tool_attributes_are_picked_up(self):
        from opendesk.tools.base import Tool, ToolResult

        class _Script(Tool):
            name = "my_script"
            title = "Run Script"
            annotations = {"readOnlyHint": False, "destructiveHint": True, "bogus": True}
            description = "x"

            async def execute(self, ctx, params):
                return ToolResult(title="", output="")

        class _Bare(_Script):
            name = "bare_tool"
            title = ""
            annotations = {}

        assert _Script().display_title() == "Run Script"
        assert _Script().tool_annotations() == {"readOnlyHint": False, "destructiveHint": True}
        assert _Bare().display_title() == "Bare Tool"
        assert _Bare().tool_annotations() == {"readOnlyHint": False, "destructiveHint": True}

    def test_server_instructions_cover_workflow(self):
        text = SERVER_INSTRUCTIONS
        for needle in ("`ui`", "marks=true", "`mouse`", "`screenshot`", "opendesk_peers",
                       "opendesk_use", "`peer`", "`memory`", "approval", "kill switch",
                       "opendesk check --open"):
            assert needle in text, needle


# ---------------------------------------------------------------------------
# isError on failures
# ---------------------------------------------------------------------------


class TestIsError:
    @pytest.mark.asyncio
    async def test_success_is_not_error(self, tmp_path: Path):
        out = await _dispatcher(tmp_path).call_tool("clipboard", {"action": "write", "text": "x"})
        assert isinstance(out, CallResult) and out.is_error is False

    @pytest.mark.asyncio
    async def test_unknown_tool(self, tmp_path: Path):
        out = await _dispatcher(tmp_path).call_tool("nope", {})
        assert out.is_error and "unknown tool" in out[0].text

    @pytest.mark.asyncio
    async def test_invalid_arguments(self, tmp_path: Path):
        out = await _dispatcher(tmp_path).call_tool("clipboard", {"action": "explode"})
        assert out.is_error and "invalid arguments" in out[0].text

    @pytest.mark.asyncio
    async def test_unknown_peer(self, tmp_path: Path):
        out = await _dispatcher(tmp_path).call_tool("clipboard", {"action": "read", "peer": "ghost"})
        assert out.is_error

    @pytest.mark.asyncio
    async def test_admin_error(self, tmp_path: Path):
        out = await _dispatcher(tmp_path).call_tool("opendesk_use", {"peer": "ghost"})
        assert out.is_error and out[0].text.startswith("ERROR:")

    @pytest.mark.asyncio
    async def test_admin_success(self, tmp_path: Path):
        out = await _dispatcher(tmp_path).call_tool("opendesk_status", {})
        assert not out.is_error

    @pytest.mark.asyncio
    async def test_tool_result_error_flag(self, tmp_path: Path):
        out = await _dispatcher(tmp_path).call_tool("screenshot", {"region": [1, 2, 3]})
        assert out.is_error

    @pytest.mark.asyncio
    async def test_tool_exception_becomes_error(self, tmp_path: Path):
        from opendesk.tools.base import PermissionDeniedError

        async def deny(tool, arg, desc):
            raise PermissionDeniedError("user said no")

        disp = MCPDispatcher(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
            permission_handler=deny,
        )
        out = await disp.call_tool("clipboard", {"action": "read"})
        assert out.is_error and "user said no" in out[0].text

    @pytest.mark.asyncio
    async def test_permission_errors_get_actionable_hint(self, tmp_path: Path):
        from opendesk.integrations.mcp import _convert_result
        from opendesk.tools.base import ToolResult as TR

        out = _convert_result(
            TR(title="UI error", output="macOS Accessibility permission is required.", error=True),
            LOCAL,
        )
        assert out.is_error
        assert "opendesk check --open" in out[0].text
        assert "Claude Desktop" in out[0].text

    def test_permission_hint_only_for_permission_errors(self):
        from opendesk.computer.permissions import permission_hint
        assert permission_hint("Screen Recording permission missing")
        assert permission_hint("osascript: not allowed assistive access (-1719)")
        assert permission_hint("element 'Save' not found") == ""
        assert permission_hint("Accessibility denied; run `opendesk check --open`") == ""


# ---------------------------------------------------------------------------
# mcp.types conversion (needs the `mcp` package)
# ---------------------------------------------------------------------------


class TestMCPServerWiring:
    @pytest.mark.asyncio
    async def test_in_memory_client_sees_titles_annotations_instructions_and_iserror(self, tmp_path: Path):
        pytest.importorskip("mcp")
        from mcp.shared.memory import create_connected_server_and_client_session
        from opendesk.integrations.mcp import create_mcp_server

        server = create_mcp_server(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        async with create_connected_server_and_client_session(server) as client:
            init = client._server_capabilities  # noqa: SLF001 - set during initialize
            assert init is not None
            tools = (await client.list_tools()).tools
            by_name = {t.name: t for t in tools}
            assert by_name["screenshot"].title == "Take Screenshot"
            assert by_name["screenshot"].annotations.readOnlyHint is True
            assert by_name["mouse"].annotations.destructiveHint is True
            assert all(t.title and t.annotations is not None for t in tools)

            ok = await client.call_tool("clipboard", {"action": "write", "text": "hi"})
            assert ok.isError is False
            bad = await client.call_tool("opendesk_use", {"peer": "ghost"})
            assert bad.isError is True
            unknown = await client.call_tool("does_not_exist", {})
            assert unknown.isError is True

    def test_initialize_carries_instructions(self, tmp_path: Path):
        pytest.importorskip("mcp")
        from opendesk.integrations.mcp import create_mcp_server

        server = create_mcp_server(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        opts = server.create_initialization_options()
        assert opts.instructions == SERVER_INSTRUCTIONS
        from opendesk import __version__
        assert opts.server_version == __version__


# ---------------------------------------------------------------------------
# Streamable HTTP transport + bearer auth
# ---------------------------------------------------------------------------


_INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    },
}
_HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


class TestHTTPTransport:
    def _app(self, tmp_path: Path, token: str | None):
        pytest.importorskip("mcp")
        pytest.importorskip("httpx")
        from opendesk.integrations.mcp import create_mcp_server
        from opendesk.integrations.mcp_http import build_http_app

        server = create_mcp_server(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        return build_http_app(server, token=token, host="127.0.0.1", port=8765)

    def _client(self, app):
        from starlette.testclient import TestClient
        return TestClient(app, base_url="http://127.0.0.1:8765")

    def test_rejects_missing_token(self, tmp_path: Path):
        with self._client(self._app(tmp_path, "s3cret")) as c:
            r = c.post("/mcp", json=_INIT, headers=_HEADERS)
        assert r.status_code == 401
        assert r.headers["www-authenticate"].startswith("Bearer")

    def test_rejects_wrong_token_and_scheme(self, tmp_path: Path):
        with self._client(self._app(tmp_path, "s3cret")) as c:
            r1 = c.post("/mcp", json=_INIT, headers={**_HEADERS, "authorization": "Bearer nope"})
            r2 = c.post("/mcp", json=_INIT, headers={**_HEADERS, "authorization": "Basic s3cret"})
        assert r1.status_code == 401
        assert r2.status_code == 401

    def test_accepts_correct_token(self, tmp_path: Path):
        with self._client(self._app(tmp_path, "s3cret")) as c:
            r = c.post("/mcp", json=_INIT, headers={**_HEADERS, "authorization": "Bearer s3cret"})
        assert r.status_code == 200, r.text
        assert "instructions" in r.text and "opendesk" in r.text

    def test_rejects_foreign_host_header(self, tmp_path: Path):
        with self._client(self._app(tmp_path, "s3cret")) as c:
            r = c.post(
                "/mcp/", json=_INIT,
                headers={**_HEADERS, "authorization": "Bearer s3cret", "host": "evil.example:8765"},
            )
        assert r.status_code == 421

    def test_token_file_created_0600_and_reused(self, tmp_path: Path, monkeypatch):
        import os
        import stat
        from opendesk.integrations.mcp_http import resolve_token, token_path

        monkeypatch.delenv("OPENDESK_MCP_TOKEN", raising=False)
        tok1, src1 = resolve_token(None, tmp_path)
        tok2, src2 = resolve_token(None, tmp_path)
        assert tok1 == tok2 and len(tok1) >= 32
        assert "generated" in src1 and "read from" in src2
        if os.name == "posix":
            assert stat.S_IMODE(token_path(tmp_path).stat().st_mode) == 0o600
        assert resolve_token("explicit", tmp_path) == ("explicit", "--token")
        monkeypatch.setenv("OPENDESK_MCP_TOKEN", "from-env")
        assert resolve_token(None, tmp_path)[0] == "from-env"

    def test_refuses_non_loopback_without_auth(self, tmp_path: Path):
        from opendesk.integrations.mcp_http import is_loopback, run_http_server

        assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
        assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.5")
        with pytest.raises(ValueError, match="non-loopback"):
            run_http_server(host="0.0.0.0", port=0, no_auth=True, home=tmp_path)

    def test_cli_parser_accepts_http_flags(self):
        from opendesk.integrations.mcp import build_arg_parser

        args = build_arg_parser().parse_args(
            ["--transport", "http", "--host", "127.0.0.1", "--port", "9999", "--token", "t"],
        )
        assert (args.transport, args.host, args.port, args.token) == ("http", "127.0.0.1", 9999, "t")
        assert build_arg_parser().parse_args(["--http"]).transport == "http"
        assert build_arg_parser().parse_args([]).transport == "stdio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_with_remote(
    tmp_path: Path, local: FakeComputer, remote: FakeComputer, *, name: str,
) -> MCPSession:
    """Build a session where `name` resolves to *remote* without doing real I/O."""
    TrustedPeers(tmp_path).add(Identity.generate().public_bytes, name=name)
    session = MCPSession(home=tmp_path, local=local)
    # Pre-seed the connection cache so resolve() returns the FakeComputer
    # without trying to discover + connect.
    session._connections[name] = remote  # type: ignore[attr-defined]
    return session
