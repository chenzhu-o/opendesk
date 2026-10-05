"""Tests for the MCP integration.

Exercises the library-neutral :class:`MCPDispatcher` directly so the tests
don't depend on running an MCP transport.  The dispatcher is what
:func:`create_mcp_server` wraps with ``mcp.types.*`` conversion, so testing
it covers the full agent-facing surface.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from opendesk.computer import LocalComputer
from opendesk.integrations.mcp import (
    MCPDispatcher,
    PEER_AWARE_TOOLS,
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

        # Computer-use tool: should have `peer` in its schema's properties.
        screenshot = by_name["screenshot"]
        assert "peer" in screenshot.schema["properties"]
        assert "peer" in screenshot.description.lower()

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


# ---------------------------------------------------------------------------
# The learning layer over MCP — the path a real agent actually takes
# ---------------------------------------------------------------------------
#
# Everything below goes through MCPDispatcher, i.e. exactly what Claude Code or
# Cursor sees: list_tools() for the schemas, call_tool() with parsed JSON
# arguments.  Two things can break here that unit tests on the tools cannot
# catch -- an action missing from the advertised schema, and a parameter that
# does not survive JSON round-tripping -- so both are asserted directly.


class TestLearningToolsAreAdvertised:
    """An action an agent cannot see is an action it will never take."""

    @pytest.mark.asyncio
    async def test_the_learning_tools_are_listed(self, tmp_path: Path):
        dispatcher = MCPDispatcher(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        names = {e.name for e in await dispatcher.list_tools()}
        assert {"reward", "rollout", "diagnose", "memory"} <= names

    @pytest.mark.asyncio
    async def test_every_reward_action_is_in_the_schema(self, tmp_path: Path):
        """The enum is what the model gets to choose from.

        Adding a handler without adding it to the Literal leaves a capability
        that exists in Python and is invisible over MCP -- the exact shape of
        this bug, so it is worth pinning rather than trusting.
        """
        from opendesk.tools.reward import RewardTool

        dispatcher = MCPDispatcher(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        entry = next(e for e in await dispatcher.list_tools() if e.name == "reward")
        advertised = set(entry.schema["properties"]["action"]["enum"])
        handlers = {
            name.lstrip("_") for name in vars(RewardTool)
            if name.startswith("_") and name[1:] in advertised
        }
        # Everything the tool can do is offered, and nothing else.
        assert {"begin", "check", "assert", "assertions", "end"} <= advertised
        assert "assert" in advertised and "assertions" in advertised
        assert advertised == handlers | {"goal_capture", "goal_score",
                                        "goal_list", "episodes"}
        # The new parameters are described, not just accepted.
        assert "name" in entry.schema["properties"]
        assert "claim" in entry.schema["properties"]
        assert "checks" in entry.schema["properties"]

    @pytest.mark.asyncio
    async def test_the_reward_description_names_the_assertion_actions(self, tmp_path: Path):
        dispatcher = MCPDispatcher(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        entry = next(e for e in await dispatcher.list_tools() if e.name == "reward")
        assert "action='assert'" in entry.description
        assert "value_regex" in entry.description      # the semantic predicate


class TestLearningToolsRoundTrip:
    """One episode end to end, over the MCP surface, with a real file check."""

    @staticmethod
    def _text(result) -> str:
        return "\n".join(r.text for r in result if isinstance(r, TextResult))

    @pytest.mark.asyncio
    async def test_begin_assert_end_export_over_mcp(self, tmp_path: Path):
        from opendesk.computer.sandbox import clear_sandbox
        from opendesk.learning import assertions, trajectories

        local = FakeComputer()
        session = MCPSession(home=tmp_path, local=local)
        dispatcher = MCPDispatcher(create_registry(), session)

        sid = "mcp-local"           # the session id the dispatcher derives
        clear_sandbox(sid)
        trajectories.clear_episodes(sid)
        assertions.clear_assertions(sid)

        target = tmp_path / "invoice.pdf"

        text = self._text(await dispatcher.call_tool("reward", {
            "action": "begin",
            "task": "Export the invoice",
            "spec": {"checks": [{"kind": "file_exists", "path": str(target)}]},
        }))
        assert "Started episode" in text, text

        # The agent states what it believes — and is told it is wrong.
        text = self._text(await dispatcher.call_tool("reward", {
            "action": "assert",
            "name": "invoice exported",
            "claim": "the export dialog was accepted",
            "checks": [{"kind": "file_exists", "path": str(target)}],
        }))
        assert "DOES NOT HOLD" in text, text

        # Do the work, then the claim is true.
        target.write_text("pdf", encoding="utf-8")
        text = self._text(await dispatcher.call_tool("reward", {
            "action": "assert",
            "name": "invoice on disk",
            "claim": "the PDF is written",
            "checks": [{"kind": "file_exists", "path": str(target)}],
        }))
        assert "HOLDS" in text and "DOES NOT HOLD" not in text, text

        text = self._text(await dispatcher.call_tool("reward", {"action": "assertions"}))
        assert "2 declared" in text, text

        text = self._text(await dispatcher.call_tool("reward", {"action": "end"}))
        assert "Closed episode" in text, text
        assert "reward=1.00" in text, text
        assert "claims: 2 declared, 1 held" in text, text

        text = self._text(await dispatcher.call_tool(
            "rollout", {"action": "export", "path": str(tmp_path / "t.jsonl")}
        ))
        assert "Exported" in text, text

        record = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert record["outcome"]["reward"] == 1.0
        metrics = record["assertions"]["metrics"]
        assert metrics["declared"] == 2 and metrics["held"] == 1
        assert metrics["precision"] == 0.5

    @pytest.mark.asyncio
    async def test_the_session_id_the_agent_sees_is_stable(self, tmp_path: Path):
        """`mcp-<peer>` must match what `ui`/`screenshot` write to.

        If reward and ui disagreed about the session, a reward check would read a
        different observation store and a different audit log, and the diagnosis
        would be built from a log with no actions in it.
        """
        from opendesk.computer.sandbox import get_sandbox

        local = FakeComputer()
        session = MCPSession(home=tmp_path, local=local)
        dispatcher = MCPDispatcher(create_registry(), session)

        await dispatcher.call_tool(
            "ui", {"action": "get_tree", "app": "TextEdit"}
        )
        ui_log = len(get_sandbox("mcp-local").audit_log)

        await dispatcher.call_tool("reward", {"action": "begin", "task": "t"})
        assert len(get_sandbox("mcp-local").audit_log) > ui_log

    @pytest.mark.asyncio
    async def test_diagnose_over_mcp(self, tmp_path: Path):
        from opendesk.computer.sandbox import ActionType, clear_sandbox, get_sandbox

        local = FakeComputer()
        session = MCPSession(home=tmp_path, local=local)
        dispatcher = MCPDispatcher(create_registry(), session)

        clear_sandbox("mcp-local")
        sb = get_sandbox("mcp-local")
        for screen in ("aaaa1111bbbb2222", "aaaa1111bbbb2222", "cccc3333dddd4444"):
            sb.current_screen = screen
            await sb.record_action(ActionType.UI_ACTION, {})

        text = self._text(await dispatcher.call_tool("diagnose", {}))
        assert "State-transition diagnosis" in text, text

    @pytest.mark.asyncio
    async def test_a_bad_assertion_argument_is_an_error_not_a_crash(self, tmp_path: Path):
        """Over MCP the only channel back to the model is text.

        An unhandled exception becomes a transport error the client reports as
        "server crashed", which tells the agent nothing it can act on.
        """
        dispatcher = MCPDispatcher(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        text = self._text(await dispatcher.call_tool("reward", {
            "action": "assert", "name": "x", "checks": [{"kind": "teleport"}],
        }))
        assert text.startswith("ERROR") is False
        assert "Invalid assertion checks" in text or "teleport" in text, text


class TestSchemasSurviveTheWire:
    """A tool schema is JSON that goes over a transport.

    Anything unserialisable in it fails `initialize` or `tools/list`, which a
    client reports as the server being broken — with no hint that one field type
    was wrong. Cheaper to catch here than in a user's editor.
    """

    @pytest.mark.asyncio
    async def test_every_tool_schema_is_json_serialisable(self, tmp_path: Path):
        dispatcher = MCPDispatcher(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        for entry in await dispatcher.list_tools():
            encoded = json.dumps({"name": entry.name,
                                  "description": entry.description,
                                  "inputSchema": entry.schema})
            assert json.loads(encoded)["name"] == entry.name

    @pytest.mark.asyncio
    async def test_every_schema_is_a_valid_mcp_tool(self, tmp_path: Path):
        """Round-trip through the real ``mcp.types.Tool`` model."""
        mcp_types = pytest.importorskip("mcp.types")
        dispatcher = MCPDispatcher(
            create_registry(), MCPSession(home=tmp_path, local=FakeComputer()),
        )
        for entry in await dispatcher.list_tools():
            tool = mcp_types.Tool(
                name=entry.name, description=entry.description,
                inputSchema=entry.schema,
            )
            restored = mcp_types.Tool.model_validate_json(tool.model_dump_json())
            assert restored.name == entry.name
            assert set(restored.inputSchema.get("properties", {})) == \
                set(entry.schema.get("properties", {}))


@pytest.mark.slow
@pytest.mark.filterwarnings(
    # Windows' proactor loop tears the stdio subprocess transport down after the
    # loop has closed, so the transport's __del__ warns during GC. It is an
    # artefact of running a real subprocess inside a test-scoped loop, not a
    # leak in the server; there is nothing to fix on this side.
    "ignore::pytest.PytestUnraisableExceptionWarning",
)
class TestLiveTransport:
    """The one test that runs a real server process over stdio.

    Everything else here calls the dispatcher directly, which is what
    ``create_mcp_server`` wraps — but wrapping *is* code, and the conversion to
    ``mcp.types.*`` plus the transport is the part a client actually depends on.
    """

    @pytest.mark.asyncio
    async def test_a_real_client_sees_the_assertion_actions(self, tmp_path: Path):
        pytest.importorskip("mcp")
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable, args=["-m", "opendesk.integrations.mcp"],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name: t for t in (await session.list_tools()).tools}

                assert {"reward", "rollout", "diagnose", "memory"} <= set(tools)
                actions = tools["reward"].inputSchema["properties"]["action"]["enum"]
                assert "assert" in actions and "assertions" in actions

                # And a call with agent-declared checks comes back as text.
                result = await session.call_tool("reward", {
                    "action": "assert",
                    "name": "shell works",
                    "claim": "a trivial command succeeds",
                    "checks": [{"kind": "shell", "command": "exit 0"}],
                })
                assert result.content and "HOLDS" in result.content[0].text
