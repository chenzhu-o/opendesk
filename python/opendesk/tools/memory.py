"""MemoryTool — actively retrieve past screen observations.

The screenshot tool feeds the agent one frame at a time; once it moves on, the
previous frame is gone.  This tool exposes the session's *lossless* observation
history (see :mod:`opendesk.computer.observations`) so the agent can look back
at what it already saw instead of guessing or re-navigating.

Actions
-------
``recall``   — bring back one observation (by index, offset, or name)
``timeline`` — list recent observations, oldest first
``find``     — filter observations by app / text / time
``diff``     — compare two stored observations **without re-capturing**
``stats``    — memory accounting (held / evicted / bytes / distinct screens)
``clear``    — drop the history

Example::

    memory(action="timeline", limit=5)
    memory(action="recall", ref="-2", include_image=True)
    memory(action="diff", ref=3, ref_b="latest")
"""

from __future__ import annotations

import asyncio
from typing import Literal, Optional, Union

from pydantic import Field

from opendesk.tools.base import Attachment, Tool, ToolContext, ToolResult


class MemoryTool(Tool):
    """Query the session's lossless visual observation memory."""

    name = "memory"
    description = (
        "Recall past screen observations from this session's visual memory.\n"
        "Every screenshot you capture is stored losslessly and can be retrieved "
        "again, so you never have to guess what a screen looked like earlier.\n\n"
        "  action='timeline' — list recent observations (oldest first)\n"
        "  action='recall'   — bring back one observation, optionally with its image\n"
        "  action='find'     — filter observations by app, text, or time\n"
        "  action='diff'     — compare two stored observations (no re-capture "
        "needed; ignore_regions excludes ambient churn)\n"
        "  action='stats'    — how much history is held / evicted\n"
        "  action='clear'    — drop the history\n\n"
        "References accept: 'latest', 'first', 'previous', an absolute index "
        "(e.g. 7), or a negative offset from the end (e.g. -1 = most recent)."
    )

    class Params(Tool.Params):
        action: Literal[
            "recall", "timeline", "find", "diff", "stats", "clear"
        ] = Field(default="timeline", description="What to do.")

        ref: Optional[Union[int, str]] = Field(
            default=None,
            description=(
                "Primary observation reference. 'latest' (default), 'first', "
                "'previous', an absolute index, or a negative offset (-1 = newest)."
            ),
        )
        ref_b: Optional[Union[int, str]] = Field(
            default=None,
            description="Second reference, for action='diff'.",
        )
        include_image: bool = Field(
            default=True,
            description=(
                "For action='recall': attach the original screenshot so you can "
                "look at it again. Set false to get only the text description."
            ),
        )
        app: Optional[str] = Field(
            default=None,
            description="For action='find': only observations from this app/window.",
        )
        text: Optional[str] = Field(
            default=None,
            description="For action='find': only observations whose metadata contains this text.",
        )
        since: Optional[float] = Field(
            default=None,
            description="For action='find': only observations at/after this Unix timestamp.",
        )
        limit: Optional[int] = Field(
            default=None,
            description="Maximum number of entries to return (most recent kept).",
        )
        ignore_regions: Optional[list[list[int]]] = Field(
            default=None,
            description=(
                "For action='diff': exclude [x, y, w, h] boxes from the pixel "
                "comparison. Use for ambient churn — a live clock, a taskbar — "
                "that changes pixels without changing state."
            ),
        )
        cap: Optional[int] = Field(
            default=None,
            description=(
                "For action='stats' or 'clear': resize the memory capacity "
                "(number of observations retained)."
            ),
        )

    async def execute(self, ctx: ToolContext, params: "MemoryTool.Params") -> ToolResult:
        from opendesk.computer.observations import get_store

        store = get_store(ctx.session_id)
        if params.cap is not None:
            store.cap = max(1, int(params.cap))

        handler = {
            "recall": self._recall,
            "timeline": self._timeline,
            "find": self._find,
            "diff": self._diff,
            "stats": self._stats,
            "clear": self._clear,
        }[params.action]
        return await handler(ctx, store, params)

    # ------------------------------------------------------------------

    async def _recall(self, ctx, store, params) -> ToolResult:
        obs = store.resolve(params.ref)
        if obs is None:
            return ToolResult(
                title="Memory recall",
                output=(
                    f"No observation matches {params.ref!r}. "
                    f"{self._held_note(store)}"
                ),
                error=True,
            )

        lines = [
            f"Recalled observation #{obs.index} — {obs.describe()}",
            f"  captured {round(max(0.0, __import__('time').time() - obs.timestamp), 1)}s ago"
            f"  |  source: {obs.source}",
        ]
        if obs.marks_summary:
            lines.append(f"  Set-of-Marks at capture time:\n{obs.marks_summary}")
            lines.append(
                "  Note: mark numbers may no longer match the live screen — "
                "re-capture with marks=true before clicking by mark."
            )
        if obs.changed_region:
            lines.append(
                f"  Change since previous observation: region {obs.changed_region}"
            )

        attachments: list[Attachment] = []
        if params.include_image:
            attachments.append(
                Attachment(f"observation_{obs.index}.png", obs.png, "image/png")
            )
            lines.append("  (original screenshot attached)")
        else:
            lines.append("  (image omitted — pass include_image=true to view it)")

        await self._audit(ctx, "memory_recall", {"ref": params.ref}, f"#{obs.index}")
        return ToolResult(
            title=f"Memory recall #{obs.index}",
            output="\n".join(lines),
            attachments=attachments,
            metadata=obs.summary(),
        )

    async def _timeline(self, ctx, store, params) -> ToolResult:
        items = store.timeline(params.limit)
        if not items:
            return ToolResult(
                title="Memory timeline",
                output="Visual memory is empty — no screenshots captured yet.",
            )

        lines = [f"Visual memory — {len(items)} observation(s), oldest first:\n"]
        for obs in items:
            marker = "  "
            lines.append(f"{marker}#{obs.index:<4} {obs.describe()}")

        lines.append(
            f"\nHeld {len(store)} of {store.cap}"
            + (f" ({store.evicted} evicted)" if store.evicted else "")
            + ".  Use memory(action='recall', ref=<index>) to view one again."
        )
        return ToolResult(
            title=f"Memory timeline ({len(items)})",
            output="\n".join(lines),
            metadata=store.stats(),
        )

    async def _find(self, ctx, store, params) -> ToolResult:
        items = store.find(
            app=params.app, text=params.text, since=params.since, limit=params.limit
        )
        criteria = ", ".join(
            f"{k}={v!r}"
            for k, v in (("app", params.app), ("text", params.text), ("since", params.since))
            if v is not None
        ) or "everything"

        if not items:
            return ToolResult(
                title="Memory find",
                output=f"No observations match {criteria}. {self._held_note(store)}",
            )

        lines = [f"{len(items)} observation(s) matching {criteria}:\n"]
        for obs in items:
            lines.append(f"  #{obs.index:<4} {obs.describe()}")
        return ToolResult(
            title=f"Memory find ({len(items)})",
            output="\n".join(lines),
            metadata={"matched": len(items)},
        )

    async def _diff(self, ctx, store, params) -> ToolResult:
        a = store.resolve(params.ref if params.ref is not None else "previous")
        b = store.resolve(params.ref_b if params.ref_b is not None else "latest")
        if a is None or b is None:
            missing = params.ref if a is None else params.ref_b
            return ToolResult(
                title="Memory diff",
                output=(
                    f"Could not resolve {missing!r}. {self._held_note(store)}"
                ),
                error=True,
            )

        from functools import partial

        from opendesk.computer.capture import diff_screenshots, fingerprint_distance

        try:
            loop = asyncio.get_event_loop()
            diff = await loop.run_in_executor(
                None, partial(diff_screenshots, a.png, b.png,
                              ignore_regions=params.ignore_regions)
            )
        except Exception as exc:
            return ToolResult(
                title="Memory diff",
                output=f"Diff failed for #{a.index} vs #{b.index}: {exc}",
                error=True,
            )

        dist = fingerprint_distance(a.fingerprint, b.fingerprint)
        lines = [
            f"Comparing observation #{a.index} → #{b.index}:",
            f"  {a.describe()}",
            f"  {b.describe()}",
            "",
            f"  Pixel change: {diff['summary']}",
            f"  Fingerprint distance: {dist} bit(s)",
        ]
        if diff.get("ignored_regions"):
            lines.append(
                f"  Excluded {diff.get('suppressed_pixels') or 0} px in "
                f"{len(diff['ignored_regions'])} ignored region(s)."
            )
        if dist not in (9999, -1) and dist <= 4:
            lines.append(
                "  → Same functional screen (visually identical within tolerance) "
                "— the actions between these two points did not change the UI."
            )
        elif dist == 9999:
            lines.append("  → Fingerprint unavailable for one side (captured before memory existed).")
        elif dist == -1:
            lines.append("  → Fingerprints use different hash sizes and cannot be compared.")

        await self._audit(
            ctx, "memory_diff", {"a": a.index, "b": b.index}, diff["summary"]
        )
        return ToolResult(
            title=f"Memory diff #{a.index}→#{b.index}",
            output="\n".join(lines),
            metadata={"change_fraction": diff.get("change_fraction"),
                      "changed_pixels": diff.get("changed_pixels"),
                      "suppressed_pixels": diff.get("suppressed_pixels"),
                      "ignored_regions": diff.get("ignored_regions"),
                      "fingerprint_distance": dist},
        )

    async def _stats(self, ctx, store, params) -> ToolResult:
        s = store.stats()
        lines = [
            f"Visual memory — session {s['session_id']!r}",
            f"  held:            {s['held']} / {s['cap']} observation(s)",
            f"  evicted:         {s['evicted']}",
            f"  total size:      {s['total_mb']} MB (lossless PNG)",
            f"  oldest index:    {s['oldest_index']}",
            f"  newest index:    {s['newest_index']}",
            f"  distinct screens:{s['distinct_screens']}",
            "",
            "Observations are stored losslessly (original bytes, no re-encoding).",
        ]
        if s["evicted"]:
            lines.append(
                f"  Note: {s['evicted']} older observation(s) were evicted to stay "
                f"under the cap. Raise it with memory(action='stats', cap=N)."
            )
        return ToolResult(title="Memory stats", output="\n".join(lines), metadata=s)

    async def _clear(self, ctx, store, params) -> ToolResult:
        n = store.clear()
        await self._audit(ctx, "memory_clear", {}, f"cleared {n}")
        return ToolResult(
            title="Memory cleared",
            output=f"Dropped {n} observation(s) from visual memory.",
            metadata={"cleared": n},
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _held_note(store) -> str:
        return (
            f"Memory holds {len(store)} observation(s)"
            + (f" (indices {store.first().index}–{store.latest().index})" if len(store) else "")
            + "."
        )

    @staticmethod
    async def _audit(ctx, action: str, params: dict, result: str) -> None:
        try:
            from opendesk.computer.sandbox import ActionType, get_sandbox

            await get_sandbox(ctx.session_id).record_action(
                ActionType(action), params=params, result=result
            )
        except Exception:
            pass
