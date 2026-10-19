"""BrowserTool — drive a Chromium-family browser over the Chrome DevTools
Protocol instead of pixels.

Reading the DOM / accessibility tree and clicking by element is faster and far
more reliable than screenshot → OCR → mouse for anything inside a web page.

Local-only in v1: the browser always runs on the machine hosting this
process (the MCP server does not route ``browser`` to paired peers).  The
launched browser uses a dedicated profile under ``<home>/browser-profile`` —
never the user's everyday profile — so cookies and logins are isolated.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import Field

from opendesk.computer.cdp import (
    DEFAULT_PORT,
    CDPConnection,
    CDPError,
    DevToolsHTTP,
    find_browser,
    launch_browser,
)
from opendesk.tools.base import Attachment, Tool, ToolContext, ToolResult


PROFILE_MARKER = ".opendesk-profile"
MAX_TEXT_CHARS = 8_000
MAX_AX_NODES = 400
MAX_EVAL_CHARS = 16_000

# AX roles that add nothing to the snapshot on their own.
_SKIP_ROLES = {"none", "generic", "InlineTextBox", "LineBreak", "presentation", "ignored"}
_AX_FLAGS = ("focused", "checked", "selected", "expanded", "disabled", "required", "pressed")


# ---------------------------------------------------------------------------
# In-page JavaScript helpers
# ---------------------------------------------------------------------------

_FIND_BY_TEXT_JS = r"""
(function (needle) {
  const norm = (s) => (s || "").replace(/\s+/g, " ").trim().toLowerCase();
  const n = norm(needle);
  if (!n) return null;
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none";
  };
  const label = (el) => norm(
    el.innerText || el.value || el.getAttribute("aria-label") ||
    el.getAttribute("title") || el.getAttribute("placeholder") || el.alt || ""
  );
  const clickable = "a,button,input,select,textarea,summary,label,option," +
    "[role=button],[role=link],[role=menuitem],[role=tab],[role=checkbox]," +
    "[role=radio],[role=option],[role=switch],[onclick],[tabindex]";
  const cands = Array.from(document.querySelectorAll(clickable)).filter(visible);
  let hit = cands.find((el) => label(el) === n) || cands.find((el) => label(el).includes(n));
  if (!hit && document.body) {
    const all = Array.from(document.body.querySelectorAll("*"))
      .filter((el) => visible(el) && norm(el.innerText).includes(n));
    hit = all.find((el) => !all.some((o) => o !== el && el.contains(o))) || null;
  }
  return hit || null;
})
"""

_CLICK_PREP_FN = r"""
function () {
  const el = this.nodeType === 1 ? this : this.parentElement;
  if (!el) return { ok: false };
  el.scrollIntoView({ block: "center", inline: "center" });
  const r = el.getBoundingClientRect();
  const x = r.left + r.width / 2, y = r.top + r.height / 2;
  const top = (r.width > 0 && r.height > 0) ? document.elementFromPoint(x, y) : null;
  const hittable = !!top && (top === el || el.contains(top) || top.contains(el));
  const label = (el.innerText || el.value || el.getAttribute("aria-label") || "")
    .replace(/\s+/g, " ").trim().slice(0, 80);
  return { ok: true, x, y, hittable, tag: el.tagName.toLowerCase(), label };
}
"""

_JS_CLICK_FN = r"""
function () { const el = this.nodeType === 1 ? this : this.parentElement; el.click(); return true; }
"""

_FOCUS_FN = r"""
function (clear) {
  const el = this.nodeType === 1 ? this : this.parentElement;
  el.scrollIntoView({ block: "center" });
  el.focus();
  if (clear) {
    if ("value" in el) {
      el.value = "";
      el.dispatchEvent(new Event("input", { bubbles: true }));
    } else if (el.isContentEditable) {
      document.execCommand("selectAll", false, null);
      document.execCommand("delete", false, null);
    }
  }
  return el.tagName.toLowerCase();
}
"""

_VALUE_FN = r"""
function () {
  const el = this.nodeType === 1 ? this : this.parentElement;
  return ("value" in el) ? String(el.value) : (el.innerText || "");
}
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class BrowserSession:
    """Per-agent-session browser state (one DevTools endpoint at a time)."""

    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"
    process: Any = None  # subprocess.Popen when we launched the browser
    profile_dir: Optional[Path] = None
    current: Optional[str] = None  # target id of the current tab
    conns: dict[str, CDPConnection] = field(default_factory=dict)
    refs: dict[str, dict[str, int]] = field(default_factory=dict)  # target → ref → backendNodeId
    attached: bool = False

    @property
    def http(self) -> DevToolsHTTP:
        return DevToolsHTTP(self.port, self.host)

    async def drop_connections(self) -> None:
        for conn in list(self.conns.values()):
            with contextlib.suppress(Exception):
                await conn.close()
        self.conns.clear()
        self.refs.clear()


_SESSIONS: dict[str, BrowserSession] = {}


def get_browser_session(session_id: str) -> BrowserSession:
    """Return (creating if needed) the browser state for *session_id*."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = BrowserSession()
    return _SESSIONS[session_id]


async def close_browser_session(session_id: str, *, kill: bool = True) -> None:
    """Close DevTools connections and (if we launched it) terminate the browser."""
    sess = _SESSIONS.pop(session_id, None)
    if sess is None:
        return
    await sess.drop_connections()
    if kill and sess.process is not None:
        proc = sess.process
        with contextlib.suppress(Exception):
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 5)
            except Exception:
                proc.kill()


def _default_home(ctx: ToolContext) -> Path:
    meta_home = ctx.metadata.get("opendesk_home") if ctx.metadata else None
    if meta_home:
        return Path(str(meta_home)).expanduser()
    from opendesk.protocol.auth.identity import DEFAULT_HOME
    return DEFAULT_HOME


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------


class BrowserTool(Tool):
    """Control Chrome / Chromium / Edge / Brave through the DevTools Protocol."""

    name = "browser"
    title = "Web Browser (DevTools)"
    annotations = {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True}
    description = (
        "Drive a Chromium-family browser (Chrome, Chromium, Edge, Brave) through the "
        "Chrome DevTools Protocol — read the DOM and click elements directly instead "
        "of screenshots + mouse. Prefer this over `mouse` for anything inside a web "
        "page; use `script` for non-visual work (files, CLIs, curl).\n\n"
        "Actions:\n"
        "  launch   — start a browser with remote debugging and a dedicated profile "
        "(<home>/browser-profile, never your everyday profile). Options: browser, "
        "port, headless, url.\n"
        "  attach   — connect to a browser already started with --remote-debugging-port.\n"
        "  tabs     — list open tabs (index, id, title, url; * = current).\n"
        "  open     — open url in a new tab and make it current.\n"
        "  navigate — load url in the current (or given) tab and wait for load.\n"
        "  read     — title, url, visible text and an accessibility snapshot where each "
        "node has a short ref like e12.\n"
        "  click    — click by ref (from read), CSS selector, or visible text.\n"
        "  type     — type text into ref/selector (or the focused element); "
        "clear=true empties it first, submit=true presses Enter.\n"
        "  wait_for — wait until a selector or text appears (timeout seconds).\n"
        "  screenshot — PNG of the tab (full_page=true for the whole page).\n"
        "  eval     — evaluate a JavaScript expression and return its JSON value.\n"
        "  close_tab — close the current (or given) tab.\n\n"
        "Typical flow: launch → navigate → read → click ref=eN → read.\n"
        "Runs on this machine only (not routed to paired peers)."
    )

    class Params(Tool.Params):
        action: Literal[
            "launch", "attach", "tabs", "open", "navigate", "read", "click",
            "type", "wait_for", "screenshot", "eval", "close_tab",
        ] = Field(description="Browser action. See tool description.")
        url: Optional[str] = Field(default=None, description="URL for launch/open/navigate.")
        port: Optional[int] = Field(
            default=None, description=f"DevTools port for launch/attach (default {DEFAULT_PORT}).",
        )
        browser: Optional[str] = Field(
            default=None,
            description="launch: chrome, chromium, edge, brave, or a full binary path. "
                        "Default: first one found.",
        )
        headless: bool = Field(default=False, description="launch: run without a window.")
        profile_dir: Optional[str] = Field(
            default=None,
            description="launch: profile directory override (default <home>/browser-profile). "
                        "Must be empty or an opendesk-created profile.",
        )
        tab: Optional[Union[int, str]] = Field(
            default=None, description="Tab index (from tabs) or target id. Default: current tab.",
        )
        ref: Optional[str] = Field(default=None, description="Element ref from the last read, e.g. 'e12'.")
        selector: Optional[str] = Field(default=None, description="CSS selector.")
        text: Optional[str] = Field(
            default=None,
            description="click/wait_for: visible text to find. type: the text to type.",
        )
        expression: Optional[str] = Field(default=None, description="eval: JavaScript expression.")
        clear: bool = Field(default=False, description="type: clear the field first.")
        submit: bool = Field(default=False, description="type: press Enter afterwards.")
        full_page: bool = Field(default=False, description="screenshot: capture the full page.")
        max_chars: int = Field(
            default=MAX_TEXT_CHARS, ge=200, le=100_000,
            description="read: maximum characters of visible text to return.",
        )
        timeout: float = Field(
            default=15.0, gt=0, le=120,
            description="Seconds to wait (wait_for, page loads, launch).",
        )

    # ------------------------------------------------------------------

    async def execute(self, ctx: ToolContext, params: "BrowserTool.Params") -> ToolResult:
        from opendesk.computer.sandbox import ActionType, get_sandbox

        target_desc = (
            params.url or params.ref or params.selector or params.text
            or (params.expression or "")[:120] or ""
        )
        await ctx.check_permission(
            tool="browser", argument=f"{params.action} {target_desc}".strip(),
            description=f"Browser {params.action}" + (f": {target_desc}" if target_desc else ""),
        )

        sandbox = get_sandbox(ctx.session_id)
        audit_params: dict[str, Any] = {"action": params.action}
        for key in ("url", "ref", "selector", "tab", "port"):
            val = getattr(params, key)
            if val is not None:
                audit_params[key] = val
        if params.text is not None:
            audit_params["text"] = params.text[:200]
        if params.expression is not None:
            audit_params["expression"] = params.expression[:500]

        sess = get_browser_session(ctx.session_id)
        handler = getattr(self, f"_do_{params.action}")
        try:
            result: ToolResult = await handler(ctx, sess, params)
        except ImportError as exc:
            result = ToolResult(title="Browser error", output=str(exc), error=True)
        except CDPError as exc:
            result = ToolResult(title="Browser error", output=f"{params.action} failed: {exc}", error=True)
        except Exception as exc:  # pragma: no cover - defensive
            result = ToolResult(
                title="Browser error",
                output=f"{params.action} failed: {type(exc).__name__}: {exc}",
                error=True,
            )

        await sandbox.record_action(
            ActionType.BROWSER, audit_params,
            result=None if result.error else result.output[:200],
            error=result.output[:200] if result.error else None,
        )
        return result

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    async def _target(self, sess: BrowserSession, tab: Optional[Union[int, str]] = None) -> dict[str, Any]:
        targets = await sess.http.targets()
        if not targets:
            raise CDPError("The browser has no open tabs. Use action='open' with a url.")
        if tab is None:
            for t in targets:
                if t.get("id") == sess.current:
                    return t
            sess.current = targets[0]["id"]
            return targets[0]
        key = str(tab).strip()
        if key.isdigit():
            idx = int(key)
            if idx >= len(targets):
                raise CDPError(f"No tab at index {idx} ({len(targets)} open). Use action='tabs'.")
            return targets[idx]
        for t in targets:
            if t.get("id") == key or t.get("id", "").startswith(key):
                return t
        low = key.lower()
        for t in targets:
            if low in (t.get("title") or "").lower() or low in (t.get("url") or "").lower():
                return t
        raise CDPError(f"No tab matches {tab!r}. Use action='tabs'.")

    async def _conn(self, sess: BrowserSession, target: dict[str, Any]) -> CDPConnection:
        tid = target["id"]
        conn = sess.conns.get(tid)
        if conn is not None and not conn.closed:
            return conn
        ws_url = target.get("webSocketDebuggerUrl")
        if not ws_url:
            raise CDPError(
                f"Tab {tid} has no webSocketDebuggerUrl (another DevTools client may own it)."
            )
        conn = await CDPConnection(ws_url).connect()
        sess.conns[tid] = conn
        with contextlib.suppress(CDPError):
            await conn.send("Page.enable", timeout=5)
        return conn

    async def _current(self, sess: BrowserSession, tab: Optional[Union[int, str]] = None):
        target = await self._target(sess, tab)
        sess.current = target["id"]
        return target, await self._conn(sess, target)

    @staticmethod
    async def _evaluate(conn: CDPConnection, expression: str, *, timeout: float = 30.0) -> Any:
        res = await conn.send("Runtime.evaluate", {
            "expression": expression, "returnByValue": True, "awaitPromise": True,
        }, timeout=timeout)
        if res.get("exceptionDetails"):
            raise CDPError(_exception_text(res["exceptionDetails"]))
        return (res.get("result") or {}).get("value")

    async def _wait_loaded(self, conn: CDPConnection, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with contextlib.suppress(CDPError):
                state = await self._evaluate(conn, "document.readyState", timeout=5)
                if state == "complete":
                    return True
            await asyncio.sleep(0.2)
        return False

    async def _resolve_element(
        self, conn: CDPConnection, sess: BrowserSession, target_id: str,
        params: "BrowserTool.Params", *, allow_text: bool,
    ) -> tuple[Optional[str], str]:
        """Return ``(objectId, how)`` for ref / selector / text; ``(None, '')`` if none given."""
        if params.ref:
            ref = params.ref.strip().lstrip("[").rstrip("]")
            backend_id = sess.refs.get(target_id, {}).get(ref)
            if backend_id is None:
                raise CDPError(
                    f"Unknown ref {params.ref!r} — refs come from the latest action='read' "
                    "on this tab; call read again."
                )
            res = await conn.send("DOM.resolveNode", {"backendNodeId": backend_id})
            oid = (res.get("object") or {}).get("objectId")
            if not oid:
                raise CDPError(f"Element {ref} is no longer in the page; call read again.")
            return oid, f"ref {ref}"
        if params.selector:
            res = await conn.send("Runtime.evaluate", {
                "expression": f"document.querySelector({json.dumps(params.selector)})",
            })
            if res.get("exceptionDetails"):
                raise CDPError(f"Invalid selector {params.selector!r}: {_exception_text(res['exceptionDetails'])}")
            obj = res.get("result") or {}
            if obj.get("subtype") == "null" or not obj.get("objectId"):
                raise CDPError(f"No element matches selector {params.selector!r}.")
            return obj["objectId"], f"selector {params.selector!r}"
        if allow_text and params.text:
            res = await conn.send("Runtime.evaluate", {
                "expression": f"{_FIND_BY_TEXT_JS}({json.dumps(params.text)})",
            })
            obj = res.get("result") or {}
            if res.get("exceptionDetails") or obj.get("subtype") == "null" or not obj.get("objectId"):
                raise CDPError(f"No visible element with text {params.text!r}.")
            return obj["objectId"], f"text {params.text!r}"
        return None, ""

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def _do_launch(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        port = p.port or DEFAULT_PORT
        http = DevToolsHTTP(port)
        try:
            info = await http.request("/json/version", timeout=1.0)
        except CDPError:
            info = None  # port is free — the normal case
        if info is not None:
            if not isinstance(info, dict):
                info = {}
            owned = sess.process is not None and sess.process.poll() is None and sess.port == port
            if owned:
                return ToolResult(
                    title="Browser already running",
                    output=f"Browser already launched on port {port}: {info.get('Browser', '?')}.",
                )
            raise CDPError(
                f"Port {port} already has a DevTools endpoint ({info.get('Browser', '?')}). "
                "Use action='attach' to drive it, or launch with a different port."
            )

        binary = find_browser(p.browser)
        if binary is None:
            want = p.browser or "Chrome, Chromium, Edge or Brave"
            raise CDPError(
                f"Could not find {want}. Install one, pass browser='/path/to/binary', "
                "or set $OPENDESK_BROWSER."
            )

        profile = Path(p.profile_dir).expanduser() if p.profile_dir else _default_home(ctx) / "browser-profile"
        if profile.exists() and any(profile.iterdir()) and not (profile / PROFILE_MARKER).exists():
            raise CDPError(
                f"Refusing to use {profile} as the browser profile: it is not empty and was "
                "not created by opendesk (it may be a real browser profile). Pick an empty "
                "directory."
            )
        profile.mkdir(parents=True, exist_ok=True)
        (profile / PROFILE_MARKER).write_text("opendesk browser profile\n")

        await sess.drop_connections()
        proc = launch_browser(
            binary, port=port, profile_dir=profile, headless=p.headless,
            url=p.url or "about:blank",
        )
        sess.port, sess.process, sess.profile_dir = port, proc, profile
        sess.current, sess.attached = None, True
        try:
            info = await http.wait_ready(timeout=max(p.timeout, 20.0))
        except CDPError:
            with contextlib.suppress(Exception):
                proc.kill()
            sess.process = None
            raise
        # The first tab can take a moment to appear.
        deadline = time.monotonic() + 5
        tabs: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            tabs = await http.targets()
            if tabs:
                break
            await asyncio.sleep(0.2)
        if tabs:
            sess.current = tabs[0]["id"]
            if p.url:
                conn = await self._conn(sess, tabs[0])
                await self._wait_loaded(conn, p.timeout)
        return ToolResult(
            title="Browser launched",
            output=(
                f"Launched {info.get('Browser', Path(binary).name)} "
                f"({'headless' if p.headless else 'windowed'}) on DevTools port {port}.\n"
                f"Profile: {profile}\nPID: {proc.pid}\nTabs: {len(tabs)}"
            ),
            metadata={"port": port, "pid": proc.pid, "profile_dir": str(profile)},
        )

    async def _do_attach(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        port = p.port or DEFAULT_PORT
        http = DevToolsHTTP(port)
        info = await http.version()
        await sess.drop_connections()
        if sess.port != port:
            sess.process = None
        sess.port, sess.current, sess.attached = port, None, True
        tabs = await http.targets()
        if tabs:
            sess.current = tabs[0]["id"]
        return ToolResult(
            title="Browser attached",
            output=(
                f"Attached to {info.get('Browser', 'browser')} on port {port}. "
                f"{len(tabs)} tab(s) open."
            ),
            metadata={"port": port},
        )

    async def _do_tabs(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        tabs = await sess.http.targets()
        if not tabs:
            return ToolResult(title="Browser tabs", output="No open tabs.")
        if sess.current not in {t["id"] for t in tabs}:
            sess.current = tabs[0]["id"]
        lines = [f"{len(tabs)} tab(s):"]
        for i, t in enumerate(tabs):
            mark = "*" if t["id"] == sess.current else " "
            lines.append(f" {mark}[{i}] {t.get('title') or '(untitled)'} — {t.get('url', '')}  (id {t['id']})")
        return ToolResult(title="Browser tabs", output="\n".join(lines),
                          metadata={"tabs": [{"id": t["id"], "title": t.get("title"), "url": t.get("url")} for t in tabs]})

    async def _do_open(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        url = p.url or "about:blank"
        target = await sess.http.new_tab(url)
        if not isinstance(target, dict) or "id" not in target:
            raise CDPError(f"Browser did not return a new tab: {target!r}")
        sess.current = target["id"]
        conn = await self._conn(sess, target)
        loaded = await self._wait_loaded(conn, p.timeout) if url != "about:blank" else True
        title, cur_url = await self._title_url(conn)
        note = "" if loaded else f" (still loading after {p.timeout:g}s)"
        return ToolResult(title="Browser open", output=f"Opened new tab{note}: {title} — {cur_url}",
                          metadata={"tab": target["id"]})

    async def _do_navigate(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        if not p.url:
            raise CDPError("url is required for action='navigate'.")
        target, conn = await self._current(sess, p.tab)
        conn.events.clear()
        res = await conn.send("Page.navigate", {"url": p.url}, timeout=p.timeout + 5)
        if res.get("errorText"):
            raise CDPError(f"Navigation to {p.url} failed: {res['errorText']}")
        sess.refs.pop(target["id"], None)
        loaded = True
        if res.get("loaderId"):
            ev = await conn.wait_event("Page.loadEventFired", p.timeout)
            loaded = ev is not None or await self._wait_loaded(conn, 1.0)
        title, cur_url = await self._title_url(conn)
        note = "" if loaded else f" (load event not seen within {p.timeout:g}s)"
        return ToolResult(title="Browser navigate", output=f"Navigated{note}: {title} — {cur_url}")

    async def _title_url(self, conn: CDPConnection) -> tuple[str, str]:
        with contextlib.suppress(CDPError):
            v = await self._evaluate(conn, "({t: document.title, u: location.href})", timeout=5)
            if isinstance(v, dict):
                return str(v.get("t") or "(untitled)"), str(v.get("u") or "")
        return "(unknown)", ""

    async def _do_read(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        target, conn = await self._current(sess, p.tab)
        page = await self._evaluate(
            conn,
            "({title: document.title, url: location.href, "
            "text: document.body ? document.body.innerText : ''})",
        ) or {}
        text = str(page.get("text") or "")
        shown = text[: p.max_chars]
        cut = len(text) > p.max_chars

        with contextlib.suppress(CDPError):
            await conn.send("Accessibility.enable", timeout=5)
        ax = await conn.send("Accessibility.getFullAXTree", {}, timeout=30)
        lines, refs, total = _render_ax(ax.get("nodes") or [], MAX_AX_NODES)
        sess.refs[target["id"]] = refs

        out = [
            f"Title: {page.get('title') or '(untitled)'}",
            f"URL: {page.get('url') or target.get('url', '')}",
            "",
            f"--- visible text ({len(text):,} chars"
            + (f", showing first {p.max_chars:,}; raise max_chars for more" if cut else "") + ") ---",
            shown.strip() or "(none)",
            "",
            f"--- accessibility snapshot ({len(refs)} refs"
            + (f"; {total - len(lines)} more nodes omitted" if total > len(lines) else "")
            + "; use ref=eN with click/type) ---",
            *lines,
        ]
        return ToolResult(
            title=f"Browser read: {page.get('title') or ''}".strip(),
            output="\n".join(out),
            metadata={"url": page.get("url"), "title": page.get("title"),
                      "text_truncated": cut, "refs": len(refs)},
        )

    async def _do_click(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        target, conn = await self._current(sess, p.tab)
        oid, how = await self._resolve_element(conn, sess, target["id"], p, allow_text=True)
        if oid is None:
            raise CDPError("click needs ref, selector, or text.")
        prep = await conn.send("Runtime.callFunctionOn", {
            "objectId": oid, "functionDeclaration": _CLICK_PREP_FN, "returnByValue": True,
        })
        info = (prep.get("result") or {}).get("value") or {}
        if not info.get("ok"):
            raise CDPError(f"Element for {how} cannot be clicked (detached node).")
        method = "mouse"
        if info.get("hittable"):
            x, y = float(info["x"]), float(info["y"])
            await conn.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
            for typ in ("mousePressed", "mouseReleased"):
                await conn.send("Input.dispatchMouseEvent", {
                    "type": typ, "x": x, "y": y, "button": "left", "clickCount": 1,
                })
        else:
            # Covered or zero-size — fall back to a DOM click.
            method = "dom"
            await conn.send("Runtime.callFunctionOn", {
                "objectId": oid, "functionDeclaration": _JS_CLICK_FN, "returnByValue": True,
            })
        label = info.get("label") or ""
        return ToolResult(
            title="Browser click",
            output=f"Clicked <{info.get('tag', '?')}> {label!r} via {how} ({method} click). "
                   "Call read to see the result.",
            metadata={"how": how, "method": method},
        )

    async def _do_type(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        if p.text is None:
            raise CDPError("text is required for action='type'.")
        target, conn = await self._current(sess, p.tab)
        oid, how = await self._resolve_element(conn, sess, target["id"], p, allow_text=False)
        if oid is not None:
            await conn.send("Runtime.callFunctionOn", {
                "objectId": oid, "functionDeclaration": _FOCUS_FN,
                "arguments": [{"value": bool(p.clear)}], "returnByValue": True,
            })
        await conn.send("Input.insertText", {"text": p.text})
        if p.submit:
            for typ in ("keyDown", "keyUp"):
                ev = {"type": typ, "key": "Enter", "code": "Enter",
                      "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13}
                if typ == "keyDown":
                    ev["text"] = "\r"
                await conn.send("Input.dispatchKeyEvent", ev)
        value_note = ""
        if oid is not None and not p.submit:
            with contextlib.suppress(CDPError):
                res = await conn.send("Runtime.callFunctionOn", {
                    "objectId": oid, "functionDeclaration": _VALUE_FN, "returnByValue": True,
                })
                val = (res.get("result") or {}).get("value")
                if isinstance(val, str):
                    value_note = f" Field now contains {val[:120]!r}."
        where = how or "the focused element"
        return ToolResult(
            title="Browser type",
            output=f"Typed {len(p.text)} char(s) into {where}"
                   + (" and pressed Enter" if p.submit else "") + "." + value_note,
        )

    async def _do_wait_for(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        if not p.selector and not p.text:
            raise CDPError("wait_for needs selector or text.")
        _target, conn = await self._current(sess, p.tab)
        if p.selector:
            expr = f"!!document.querySelector({json.dumps(p.selector)})"
            what = f"selector {p.selector!r}"
        else:
            expr = f"!!(document.body && document.body.innerText.includes({json.dumps(p.text)}))"
            what = f"text {p.text!r}"
        start = time.monotonic()
        deadline = start + p.timeout
        while True:
            with contextlib.suppress(CDPError):
                if await self._evaluate(conn, expr, timeout=5):
                    return ToolResult(
                        title="Browser wait_for",
                        output=f"Found {what} after {time.monotonic() - start:.1f}s.",
                    )
            if time.monotonic() >= deadline:
                return ToolResult(
                    title="Browser wait_for timeout",
                    output=f"Timed out after {p.timeout:g}s waiting for {what}.",
                    error=True,
                )
            await asyncio.sleep(0.25)

    async def _do_screenshot(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        _target, conn = await self._current(sess, p.tab)
        args: dict[str, Any] = {"format": "png"}
        if p.full_page:
            metrics = await conn.send("Page.getLayoutMetrics")
            size = metrics.get("cssContentSize") or metrics.get("contentSize") or {}
            if size.get("width") and size.get("height"):
                args["clip"] = {"x": 0, "y": 0, "width": size["width"],
                                "height": size["height"], "scale": 1}
                args["captureBeyondViewport"] = True
        res = await conn.send("Page.captureScreenshot", args, timeout=max(p.timeout, 30))
        data = base64.b64decode(res.get("data") or "")
        if not data:
            raise CDPError("Page.captureScreenshot returned no data.")
        title, url = await self._title_url(conn)
        return ToolResult(
            title="Browser screenshot",
            output=f"Captured {'full page' if p.full_page else 'viewport'} of {title} — {url} "
                   f"({len(data):,} bytes PNG).",
            attachments=[Attachment("browser.png", data, "image/png")],
        )

    async def _do_eval(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        if not p.expression:
            raise CDPError("expression is required for action='eval'.")
        _target, conn = await self._current(sess, p.tab)
        res = await conn.send("Runtime.evaluate", {
            "expression": p.expression, "returnByValue": True,
            "awaitPromise": True, "userGesture": True,
        }, timeout=p.timeout + 5)
        if res.get("exceptionDetails"):
            return ToolResult(
                title="Browser eval error",
                output=f"JavaScript threw: {_exception_text(res['exceptionDetails'])}",
                error=True,
            )
        obj = res.get("result") or {}
        if "value" in obj:
            try:
                text = json.dumps(obj["value"], ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                text = str(obj["value"])
        elif obj.get("type") == "undefined":
            text = "undefined"
        else:
            text = obj.get("description") or json.dumps(obj)
        if len(text) > MAX_EVAL_CHARS:
            text = text[:MAX_EVAL_CHARS] + f"\n… [truncated: {len(text):,} chars total]"
        return ToolResult(title="Browser eval", output=text, metadata={"type": obj.get("type")})

    async def _do_close_tab(self, ctx, sess: BrowserSession, p: "BrowserTool.Params") -> ToolResult:
        target = await self._target(sess, p.tab)
        tid = target["id"]
        conn = sess.conns.pop(tid, None)
        if conn is not None:
            await conn.close()
        sess.refs.pop(tid, None)
        await sess.http.close(tid)
        if sess.current == tid:
            sess.current = None
        with contextlib.suppress(CDPError):
            remaining = await sess.http.targets()
            remaining = [t for t in remaining if t["id"] != tid]
            if remaining and sess.current is None:
                sess.current = remaining[0]["id"]
        return ToolResult(
            title="Browser close_tab",
            output=f"Closed tab {target.get('title') or tid} — {target.get('url', '')}.",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exception_text(details: dict[str, Any]) -> str:
    exc = details.get("exception") or {}
    return str(exc.get("description") or details.get("text") or "exception")


def _ax_value(node: dict[str, Any], key: str) -> Any:
    v = node.get(key)
    return v.get("value") if isinstance(v, dict) else None


def _render_ax(nodes: list[dict[str, Any]], max_nodes: int) -> tuple[list[str], dict[str, int], int]:
    """Render a CDP AX node list as indented lines with short refs.

    Returns ``(lines, refs, total_candidate_nodes)``.
    """
    by_id = {n.get("nodeId"): n for n in nodes}
    child_ids = {c for n in nodes for c in (n.get("childIds") or [])}
    roots = [n for n in nodes if n.get("nodeId") not in child_ids] or nodes[:1]

    lines: list[str] = []
    refs: dict[str, int] = {}
    total = 0

    def visit(node: dict[str, Any], depth: int, parent_name: str) -> None:
        nonlocal total
        role = str(_ax_value(node, "role") or "")
        name = str(_ax_value(node, "name") or "").strip()
        shown = not node.get("ignored") and role not in _SKIP_ROLES
        if shown and role == "StaticText" and (not name or name == parent_name):
            shown = False
        if shown and not name and role in ("paragraph", "group", "section", "list", "listitem", "div"):
            # Structural containers without a label: keep children, skip the line.
            shown = False
        next_depth = depth
        if shown:
            total += 1
            if len(lines) < max_nodes:
                parts = [f"{'  ' * depth}"]
                backend = node.get("backendDOMNodeId")
                if backend is not None:
                    ref = f"e{len(refs) + 1}"
                    refs[ref] = int(backend)
                    parts.append(f"[{ref}] ")
                parts.append(role)
                if name:
                    parts.append(f" {json.dumps(name[:100], ensure_ascii=False)}")
                value = _ax_value(node, "value")
                if value not in (None, ""):
                    parts.append(f" value={json.dumps(str(value)[:80], ensure_ascii=False)}")
                flags = []
                for prop in node.get("properties") or []:
                    pname = prop.get("name")
                    pval = (prop.get("value") or {}).get("value")
                    if pname in _AX_FLAGS and pval not in (None, False, "false"):
                        flags.append(pname if pval in (True, "true") else f"{pname}={pval}")
                if flags:
                    parts.append(" (" + ", ".join(flags) + ")")
                lines.append("".join(parts))
            next_depth = depth + 1
        for cid in node.get("childIds") or []:
            child = by_id.get(cid)
            if child is not None:
                visit(child, next_depth, name if shown else parent_name)

    for root in roots:
        visit(root, 0, "")
    return lines, refs, total
