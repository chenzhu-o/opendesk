"""Tests for the ``fleet`` tool — one tool call across many peers."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from opendesk.computer import ComputerDispatcher, RemoteComputer
from opendesk.integrations.mcp import (
    LOCAL_ONLY_TOOLS,
    PEER_AWARE_TOOLS,
    ImageResult,
    MCPDispatcher,
    TextResult,
)
from opendesk.integrations.mcp_session import MCPSession
from opendesk.protocol import LoopbackConnection, Peer
from opendesk.protocol.auth import Identity, TrustedPeers
from opendesk.registry import create_registry
from opendesk.tools.base import ToolContext
from opendesk.tools.fleet import FleetTool

from tests._fakes import FakeComputer


class FailingComputer(FakeComputer):
    async def clipboard_write(self, contents) -> None:
        raise RuntimeError("clipboard locked")


class SlowComputer(FakeComputer):
    """Counts how many exec calls run at once across all instances."""

    active = 0
    peak = 0

    def __init__(self, delay: float = 0.05) -> None:
        super().__init__()
        self.delay = delay

    async def exec(self, argv, *, timeout=None, cwd=None, env=None, stdin=None):
        SlowComputer.active += 1
        SlowComputer.peak = max(SlowComputer.peak, SlowComputer.active)
        try:
            await asyncio.sleep(self.delay)
            return await super().exec(argv, timeout=timeout, cwd=cwd, env=env, stdin=stdin)
        finally:
            SlowComputer.active -= 1


def _session(tmp_path: Path, peers: dict[str, FakeComputer], descriptions: dict[str, str] | None = None,
             local: FakeComputer | None = None) -> MCPSession:
    store = TrustedPeers(tmp_path)
    for name in peers:
        store.add(Identity.generate().public_bytes, name=name)
    for name, desc in (descriptions or {}).items():
        store.set_description_override(name, desc)
    session = MCPSession(home=tmp_path, local=local or FakeComputer())
    for name, comp in peers.items():
        session._connections[name] = comp  # type: ignore[attr-defined]
    return session


def _text(out) -> str:
    return "\n".join(r.text for r in out if isinstance(r, TextResult))


class TestFleetViaMCP:
    async def test_runs_on_all_peers(self, tmp_path: Path):
        a, b = FakeComputer(), FakeComputer()
        local = FakeComputer()
        session = _session(tmp_path, {"alpha": a, "beta": b}, local=local)
        dispatcher = MCPDispatcher(create_registry(), session)

        out = await dispatcher.call_tool("fleet", {
            "peers": "all", "tool": "clipboard",
            "params": {"action": "write", "text": "hello fleet"},
        })
        text = _text(out)
        assert "2 ok, 0 error" in text
        assert "== alpha: ok" in text and "== beta: ok" in text
        for comp in (a, b):
            assert ("clipboard_write", {"text": "hello fleet"}) in comp.calls
        # "all" means trusted peers — not the local machine.
        assert not any(c[0] == "clipboard_write" for c in local.calls)

    async def test_ambiguous_default_does_not_block_fleet(self, tmp_path: Path):
        """Two peers, no default: ordinary tools error, fleet still works."""
        session = _session(tmp_path, {"alpha": FakeComputer(), "beta": FakeComputer()})
        dispatcher = MCPDispatcher(create_registry(), session)
        assert "ERROR" in _text(await dispatcher.call_tool("clipboard", {"action": "read"}))
        out = await dispatcher.call_tool("fleet", {
            "peers": ["alpha", "beta"], "tool": "clipboard", "params": {"action": "read"},
        })
        assert "2 ok" in _text(out)
        assert "[on " not in _text(out)

    async def test_match_filters_by_description(self, tmp_path: Path):
        a, b, c = FakeComputer(), FakeComputer(), FakeComputer()
        session = _session(
            tmp_path, {"alpha": a, "beta": b, "gamma": c},
            descriptions={"alpha": "Build Mac mini (Xcode)", "beta": "Windows ERP terminal",
                          "gamma": "build server, linux"},
        )
        dispatcher = MCPDispatcher(create_registry(), session)
        out = await dispatcher.call_tool("fleet", {
            "peers": "all", "match": "build", "tool": "app", "params": {"action": "list"},
        })
        text = _text(out)
        assert "2 peer(s)" in text
        assert "alpha" in text and "gamma" in text and "== beta" not in text
        assert any(x[0] == "list_apps" for x in a.calls)
        assert not any(x[0] == "list_apps" for x in b.calls)

        out = await dispatcher.call_tool("fleet", {
            "peers": "all", "match": ["build", "mac"], "tool": "app", "params": {"action": "list"},
        })
        assert "1 peer(s)" in _text(out) and "alpha" in _text(out)

    async def test_per_peer_errors_and_summary(self, tmp_path: Path):
        session = _session(tmp_path, {"good": FakeComputer(), "bad": FailingComputer()})
        dispatcher = MCPDispatcher(create_registry(), session)
        out = await dispatcher.call_tool("fleet", {
            "peers": "good,bad", "tool": "clipboard", "params": {"action": "write", "text": "x"},
        })
        text = _text(out)
        assert "1 ok, 1 error" in text
        assert "== bad: ERROR" in text and "clipboard locked" in text
        assert "== good: ok" in text

    async def test_images_only_when_requested(self, tmp_path: Path):
        session = _session(tmp_path, {"alpha": FakeComputer(), "beta": FakeComputer()})
        dispatcher = MCPDispatcher(create_registry(), session)
        out = await dispatcher.call_tool("fleet", {"peers": "all", "tool": "screenshot"})
        assert not any(isinstance(r, ImageResult) for r in out)
        assert "image(s) omitted" in _text(out)

        out = await dispatcher.call_tool("fleet", {
            "peers": "all", "tool": "screenshot", "include_images": True,
        })
        assert sum(isinstance(r, ImageResult) for r in out) == 2

    async def test_fleet_and_browser_are_local_only(self, tmp_path: Path):
        assert {"fleet", "browser"} <= LOCAL_ONLY_TOOLS
        assert "fleet" not in PEER_AWARE_TOOLS and "browser" not in PEER_AWARE_TOOLS
        dispatcher = MCPDispatcher(create_registry(), _session(tmp_path, {}))
        entries = {e.name: e for e in await dispatcher.list_tools()}
        for name in ("fleet", "browser"):
            assert "peer" not in entries[name].schema.get("properties", {})

    async def test_unknown_peer_reported_as_skipped(self, tmp_path: Path):
        session = _session(tmp_path, {"alpha": FakeComputer()})
        dispatcher = MCPDispatcher(create_registry(), session)
        out = await dispatcher.call_tool("fleet", {
            "peers": ["alpha", "ghost"], "tool": "app", "params": {"action": "list"},
        })
        text = _text(out)
        assert "1 ok" in text and "Skipped (not paired): ghost" in text

    async def test_no_peers_is_error(self, tmp_path: Path):
        dispatcher = MCPDispatcher(create_registry(), _session(tmp_path, {}))
        out = await dispatcher.call_tool("fleet", {"peers": "all", "tool": "screenshot"})
        assert "No peers" in _text(out)


class TestFleetGuards:
    async def _run(self, tmp_path, **params):
        session = _session(tmp_path, {"alpha": FakeComputer()})
        tool = FleetTool(session=session, registry=create_registry())
        return await tool.execute(ToolContext(session_id="fleet-guard", computer=FakeComputer()),
                                  tool.parse_params(params))

    async def test_refuses_nested_fleet(self, tmp_path: Path):
        r = await self._run(tmp_path, peers="all", tool="fleet",
                            params={"peers": "all", "tool": "screenshot"})
        assert r.error and "nested" in r.output

    async def test_refuses_nesting_via_metadata(self, tmp_path: Path):
        session = _session(tmp_path, {"alpha": FakeComputer()})
        tool = FleetTool(session=session, registry=create_registry())
        ctx = ToolContext(session_id="x", computer=FakeComputer(), metadata={"fleet_depth": 1})
        r = await tool.execute(ctx, tool.parse_params({"peers": "all", "tool": "screenshot"}))
        assert r.error and "nested" in r.output

    async def test_refuses_local_only_tool(self, tmp_path: Path):
        r = await self._run(tmp_path, peers="all", tool="browser", params={"action": "tabs"})
        assert r.error and "cannot run on peers" in r.output

    async def test_refuses_peer_inside_params(self, tmp_path: Path):
        r = await self._run(tmp_path, peers="all", tool="clipboard",
                            params={"action": "read", "peer": "alpha"})
        assert r.error and "peer" in r.output

    async def test_invalid_inner_params(self, tmp_path: Path):
        r = await self._run(tmp_path, peers="all", tool="app", params={"action": "explode"})
        assert r.error and "Invalid params" in r.output

    async def test_metadata_contract(self):
        assert FleetTool.name == "fleet" and FleetTool.title
        assert FleetTool.annotations == {
            "readOnlyHint": False, "destructiveHint": True, "openWorldHint": True,
        }


class TestFleetExecution:
    async def test_concurrency_limit(self, tmp_path: Path):
        SlowComputer.active = SlowComputer.peak = 0
        peers = {f"p{i}": SlowComputer() for i in range(5)}
        session = _session(tmp_path, peers)
        tool = FleetTool(session=session, registry=create_registry())
        ctx = ToolContext(session_id="fleet-conc", computer=FakeComputer())
        r = await tool.execute(ctx, tool.parse_params({
            "peers": "all", "tool": "script", "concurrency": 2,
            "params": {"action": "shell", "code": "true"},
        }))
        assert "5 ok" in r.output
        assert SlowComputer.peak == 2

    async def test_per_peer_timeout(self, tmp_path: Path):
        session = _session(tmp_path, {"slow": SlowComputer(delay=2.0), "fast": FakeComputer()})
        tool = FleetTool(session=session, registry=create_registry())
        ctx = ToolContext(session_id="fleet-timeout", computer=FakeComputer())
        r = await tool.execute(ctx, tool.parse_params({
            "peers": "all", "tool": "script", "timeout": 0.2,
            "params": {"action": "shell", "code": "sleep 9"},
        }))
        assert "1 ok, 1 error" in r.output
        assert "timed out" in r.output

    async def test_over_the_wire_peer(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("OPENDESK_APPROVAL_CHAIN", "mcp")
        """One peer is a real RemoteComputer over a loopback protocol link."""
        fake = FakeComputer()
        a, b = LoopbackConnection.pair()
        server_peer = Peer(b, role="server", dispatcher=ComputerDispatcher(fake))

        async def _hello():
            await server_peer.hello(fake.capabilities().model_dump())
            server_peer.start()

        task = asyncio.create_task(_hello())
        remote = await RemoteComputer.connect(a)
        await task
        try:
            local_peer = FakeComputer()
            session = _session(tmp_path, {"wire": remote, "inproc": local_peer})  # type: ignore[dict-item]
            dispatcher = MCPDispatcher(create_registry(), session, approver=_approve_all)
            out = await dispatcher.call_tool("fleet", {
                "peers": "all", "tool": "script",
                "params": {"action": "shell", "code": "hostname"},
            })
            text = _text(out)
            assert "2 ok" in text, text
            assert any(c[0] == "exec" and c[1]["argv"][-1] == "hostname" for c in fake.calls)
            assert any(c[0] == "exec" for c in local_peer.calls)
        finally:
            await remote.aclose()
            await server_peer.aclose()

    async def test_without_mcp_session_uses_temporary_session(self, tmp_path: Path):
        local = FakeComputer()
        tool = FleetTool()
        ctx = ToolContext(session_id="fleet-standalone", computer=local,
                          metadata={"opendesk_home": str(tmp_path)})
        r = await tool.execute(ctx, tool.parse_params({
            "peers": ["local"], "tool": "app", "params": {"action": "list"},
        }))
        assert "1 ok" in r.output
        assert any(c[0] == "list_apps" for c in local.calls)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_cli_fleet_run_local(tmp_path: Path, monkeypatch, capsys):
    from opendesk import cli
    monkeypatch.setattr(sys, "argv", [
        "opendesk", "fleet", "run", "--peers", "local", "--tool", "script",
        "--params", '{"action": "shell", "code": "echo fleet-cli-ok"}',
        "--home", str(tmp_path),
    ])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "1 ok" in out and "fleet-cli-ok" in out


async def _approve_all(_request):
    """Stand-in for the user approving the high-risk shell call in the safety guard."""
    return True
