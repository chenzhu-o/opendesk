"""FleetTool — run one tool call across several paired peers in parallel.

"Take a screenshot of every build agent", "run ``git pull`` on all the lab
machines", "which of my machines has Xcode open?" — one call fans out to each
peer's :class:`~opendesk.computer.Computer` and returns one section per peer.

Peers are resolved through the MCP session (``ctx.metadata["mcp_session"]``,
set by :class:`~opendesk.integrations.mcp.MCPDispatcher`) so connections are
shared with ordinary ``peer=`` calls.  Outside MCP, a temporary
:class:`~opendesk.integrations.mcp_session.MCPSession` is created and closed
after the call.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional, Union

from pydantic import Field

from opendesk.tools.base import Attachment, Tool, ToolContext, ToolResult


MAX_SECTION_CHARS = 4_000


def _split_words(value: Union[str, list[str], None]) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else str(value).replace(",", " ").split()
    return [w.strip().lower() for w in items if str(w).strip()]


class FleetTool(Tool):
    """Fan one tool call out to many peers and collect the results."""

    name = "fleet"
    title = "Run on Many Machines"
    annotations = {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True}
    description = (
        "Run ONE tool call on several paired peers in parallel and get one result "
        "section per peer plus a summary line. Use it for fleet-wide questions and "
        "chores: 'screenshot every lab machine', 'run `git pull` on all build "
        "boxes', 'which machines have Xcode open?'.\n\n"
        "  peers   — list of peer names, a comma-separated string, or 'all' "
        "(every trusted peer; add 'local' explicitly to include this machine)\n"
        "  match   — keep only peers whose description (see `opendesk_describe`) or "
        "name contains ALL these words, e.g. 'build mac'\n"
        "  tool    — inner tool: script, screenshot, app, clipboard, ui, ocr, mouse, keyboard\n"
        "  params  — that tool's arguments (without `peer`)\n"
        "  concurrency (default 4), timeout per peer in seconds (default 120), "
        "include_images (default false — screenshots are summarised as text)."
    )

    class Params(Tool.Params):
        peers: Union[list[str], str] = Field(
            description="Peer names (list or comma-separated) or 'all'.",
        )
        tool: str = Field(description="Name of the tool to run on each peer, e.g. 'script'.")
        params: dict[str, Any] = Field(
            default_factory=dict, description="Arguments for the inner tool.",
        )
        match: Optional[Union[list[str], str]] = Field(
            default=None,
            description="Only peers whose description/name contains all these words.",
        )
        concurrency: int = Field(default=4, ge=1, le=32, description="Max peers in flight.")
        timeout: float = Field(default=120.0, gt=0, le=3600, description="Seconds per peer.")
        include_images: bool = Field(
            default=False, description="Attach image results (e.g. screenshots) per peer.",
        )

    def __init__(self, session: Any = None, registry: Any = None) -> None:
        self._session = session
        self._registry = registry

    # ------------------------------------------------------------------

    async def execute(self, ctx: ToolContext, params: "FleetTool.Params") -> ToolResult:
        from opendesk.computer.sandbox import ActionType, get_sandbox

        if params.tool == self.name or ctx.metadata.get("fleet_depth"):
            return ToolResult(
                title="Fleet error", output="fleet cannot be nested inside fleet.", error=True,
            )

        from opendesk.integrations.mcp import PEER_AWARE_TOOLS
        if params.tool not in PEER_AWARE_TOOLS:
            return ToolResult(
                title="Fleet error",
                output=(
                    f"Tool {params.tool!r} cannot run on peers. Fleet supports: "
                    f"{', '.join(sorted(PEER_AWARE_TOOLS))}."
                ),
                error=True,
            )
        if "peer" in params.params:
            return ToolResult(
                title="Fleet error",
                output="Remove `peer` from params — fleet chooses peers via `peers` / `match`.",
                error=True,
            )

        registry = self._registry or ctx.metadata.get("registry")
        if registry is None:
            from opendesk.registry import create_registry
            registry = create_registry()
        try:
            inner = registry.get(params.tool)
        except KeyError as exc:
            return ToolResult(title="Fleet error", output=str(exc), error=True)
        try:
            inner_params = inner.parse_params(dict(params.params))
        except Exception as exc:
            return ToolResult(
                title="Fleet error",
                output=f"Invalid params for {params.tool!r}: {exc}",
                error=True,
            )

        session = self._session or ctx.metadata.get("mcp_session")
        owns_session = session is None
        if owns_session:
            from opendesk.integrations.mcp_session import MCPSession
            home = ctx.metadata.get("opendesk_home")
            session = MCPSession(home=home, local=ctx.computer)

        try:
            names, skipped = self._select_peers(session, params)
            if not names:
                why = f" matching {params.match!r}" if params.match else ""
                return ToolResult(
                    title="Fleet: no peers",
                    output=f"No peers{why}. "
                           + (f"Not paired: {', '.join(skipped)}. " if skipped else "")
                           + "See `opendesk_peers`.",
                    error=True,
                )

            await ctx.check_permission(
                tool="fleet", argument=f"{params.tool} on {', '.join(names)}",
                description=f"Run {params.tool} on {len(names)} peer(s): {', '.join(names)}",
            )

            sem = asyncio.Semaphore(params.concurrency)
            start = time.monotonic()

            async def run_one(name: str) -> tuple[str, Optional[ToolResult], Optional[str], float]:
                async with sem:
                    t0 = time.monotonic()
                    try:
                        computer, resolved = await asyncio.wait_for(
                            session.resolve(name), timeout=params.timeout,
                        )
                        sub_ctx = ToolContext(
                            session_id=f"mcp-{resolved}",
                            permission_handler=ctx.permission_handler,
                            metadata={**ctx.metadata, "fleet_depth": 1},
                            computer=computer,
                        )
                        result = await asyncio.wait_for(
                            inner.execute(sub_ctx, inner_params), timeout=params.timeout,
                        )
                        return name, result, None, time.monotonic() - t0
                    except asyncio.TimeoutError:
                        return name, None, f"timed out after {params.timeout:g}s", time.monotonic() - t0
                    except Exception as exc:
                        return name, None, str(exc) or type(exc).__name__, time.monotonic() - t0

            outcomes = await asyncio.gather(*(run_one(n) for n in names))
        finally:
            if owns_session:
                await session.disconnect()

        ok = sum(1 for _, r, e, _ in outcomes if e is None and r is not None and not r.error)
        failed = len(outcomes) - ok
        elapsed = time.monotonic() - start
        summary = (
            f"Fleet: {params.tool} on {len(outcomes)} peer(s) — {ok} ok, {failed} error "
            f"({elapsed:.1f}s)"
        )
        sections = [summary]
        attachments: list[Attachment] = []
        per_peer: dict[str, dict[str, Any]] = {}
        for name, result, err, dur in outcomes:
            if err is not None or result is None:
                sections.append(f"\n== {name}: ERROR ({dur:.1f}s) ==\n{err}")
                per_peer[name] = {"ok": False, "error": err}
                continue
            status = "ERROR" if result.error else "ok"
            body = result.output or "(no output)"
            if len(body) > MAX_SECTION_CHARS:
                body = body[:MAX_SECTION_CHARS] + f"\n… [truncated: {len(result.output):,} chars]"
            images = [a for a in result.attachments if a.media_type.startswith("image/")]
            if images and not params.include_images:
                body += f"\n({len(images)} image(s) omitted; pass include_images=true)"
            sections.append(f"\n== {name}: {status} ({dur:.1f}s) ==\n{body}")
            per_peer[name] = {"ok": not result.error, "title": result.title}
            if params.include_images:
                for att in images:
                    attachments.append(Attachment(f"{name}-{att.filename}", att.content, att.media_type))
        if skipped:
            sections.append(f"\nSkipped (not paired): {', '.join(skipped)}")

        sandbox = get_sandbox(ctx.session_id)
        await sandbox.record_action(
            ActionType.FLEET,
            {"tool": params.tool, "peers": names, "params": params.params},
            result=summary,
            error=None if failed == 0 else f"{failed} peer(s) failed",
        )
        return ToolResult(
            title=summary,
            output="\n".join(sections),
            error=ok == 0,
            attachments=attachments,
            metadata={"ok": ok, "failed": failed, "peers": per_peer},
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _select_peers(session: Any, params: "FleetTool.Params") -> tuple[list[str], list[str]]:
        """Return ``(names_to_run, requested_but_unknown)``."""
        from opendesk.integrations.mcp_session import LOCAL
        from opendesk.protocol.auth import TrustedPeers

        store = TrustedPeers(session.home)
        trusted = {p.name: p for p in store.list()}

        raw = params.peers
        requested = raw if isinstance(raw, list) else [s for s in str(raw).replace(",", " ").split()]
        requested = [r.strip() for r in requested if r and r.strip()]

        names: list[str] = []
        unknown: list[str] = []
        for r in requested:
            if r.lower() == "all":
                names.extend(n for n in trusted if n not in names)
            elif r == LOCAL or r in trusted:
                if r not in names:
                    names.append(r)
            else:
                unknown.append(r)

        words = _split_words(params.match)
        if words:
            def haystack(n: str) -> str:
                peer = trusted.get(n)
                desc = peer.effective_description if peer is not None else ""
                return f"{n} {desc}".lower()
            names = [n for n in names if all(w in haystack(n) for w in words)]
        return names, unknown
