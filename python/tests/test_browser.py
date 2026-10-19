"""Tests for the ``browser`` tool and the minimal CDP client.

A fake DevTools endpoint runs in-process: a tiny HTTP server for the
``/json/*`` discovery routes plus a ``websockets`` server that answers the CDP
methods the tool uses against a toy page model.  One optional test drives a
real headless Chrome/Chromium/Edge/Brave when one is installed.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import shutil
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("websockets")

from opendesk.computer.cdp import CDPConnection, CDPError, DevToolsHTTP, find_browser
from opendesk.computer.sandbox import get_sandbox
from opendesk.tools.base import ToolContext
from opendesk.tools.browser import (
    BrowserTool,
    PROFILE_MARKER,
    _render_ax,
    close_browser_session,
    get_browser_session,
)


# ---------------------------------------------------------------------------
# Fake DevTools endpoint
# ---------------------------------------------------------------------------


class FakePage:
    def __init__(self, target_id: str, url: str = "https://example.test/", title: str = "Example") -> None:
        self.id = target_id
        self.url = url
        self.title = title
        self.value = ""          # the #q input
        self.out = "idle"        # the #out paragraph
        self.focused: str | None = None
        self.mouse: list[dict] = []
        self.keys: list[dict] = []
        self.dom_clicks = 0
        self.button_covered = False

    @property
    def text(self) -> str:
        return f"Hello\nPress me\n{self.out}"

    def click_button(self) -> None:
        self.out = f"clicked {self.value}"


# backendNodeId → (role, name, tag)
_ELEMENTS = {
    10: ("heading", "Hello", "h1"),
    11: ("textbox", "Search", "input"),
    12: ("button", "Press me", "button"),
    13: ("StaticText", "idle", "p"),
}
_SELECTORS = {"#q": 11, "#b": 12, "button": 12, "h1": 10}
_TEXT = {"press me": 12, "hello": 10}


class FakeCDP:
    """HTTP discovery + WebSocket CDP endpoint over a toy page model."""

    def __init__(self, png_size: int = 2_500_000) -> None:
        self.pages: dict[str, FakePage] = {"T1": FakePage("T1")}
        self.png = b"\x89PNG\r\n\x1a\n" + b"\x00" * png_size
        self.methods: list[str] = []
        self.ws_port = 0
        self.http_port = 0
        self._counter = 1

    # -- HTTP -------------------------------------------------------------

    def _target_json(self, page: FakePage) -> dict[str, Any]:
        return {
            "id": page.id, "type": "page", "title": page.title, "url": page.url,
            "webSocketDebuggerUrl": f"ws://127.0.0.1:{self.ws_port}/devtools/page/{page.id}",
        }

    def make_handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _send(self, payload, code=200):
                body = json.dumps(payload).encode() if not isinstance(payload, str) else payload.encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _route(self, method: str):
                path = self.path
                if path == "/json/version":
                    return self._send({"Browser": "FakeChrome/1.0",
                                       "webSocketDebuggerUrl": f"ws://127.0.0.1:{fake.ws_port}/devtools/browser/x"})
                if path in ("/json", "/json/list"):
                    return self._send([fake._target_json(p) for p in fake.pages.values()]
                                      + [{"id": "SW", "type": "service_worker", "title": "sw", "url": ""}])
                if path.startswith("/json/new"):
                    if method != "PUT":
                        return self._send("Using unsafe HTTP verb GET", 405)
                    fake._counter += 1
                    tid = f"T{fake._counter}"
                    url = path.split("?", 1)[1] if "?" in path else "about:blank"
                    from urllib.parse import unquote
                    fake.pages[tid] = FakePage(tid, url=unquote(url), title="New tab")
                    return self._send(fake._target_json(fake.pages[tid]))
                if path.startswith("/json/close/"):
                    fake.pages.pop(path.rsplit("/", 1)[1], None)
                    return self._send("Target is closing")
                if path.startswith("/json/activate/"):
                    return self._send("Target activated")
                return self._send({"error": "not found"}, 404)

            def do_GET(self):
                self._route("GET")

            def do_PUT(self):
                self._route("PUT")

        return Handler

    # -- CDP ---------------------------------------------------------------

    def _obj(self, backend: int) -> dict[str, Any]:
        return {"type": "object", "subtype": "node", "objectId": f"obj-{backend}"}

    def handle(self, page: FakePage, method: str, p: dict[str, Any]) -> tuple[dict | None, dict | None, list]:
        """Return ``(result, error, events)``."""
        self.methods.append(method)
        if method in ("Page.enable", "Accessibility.enable", "Runtime.enable"):
            return {}, None, []
        if method == "Runtime.evaluate":
            expr = p.get("expression", "")
            if expr == "document.readyState":
                return {"result": {"type": "string", "value": "complete"}}, None, []
            if expr.startswith("({title: document.title"):
                return {"result": {"type": "object", "value": {
                    "title": page.title, "url": page.url, "text": page.text}}}, None, []
            if expr.startswith("({t: document.title"):
                return {"result": {"type": "object", "value": {"t": page.title, "u": page.url}}}, None, []
            if expr.startswith("!!document.querySelector("):
                sel = json.loads(expr[len("!!document.querySelector("):-1])
                return {"result": {"type": "boolean", "value": sel in _SELECTORS}}, None, []
            if expr.startswith("!!(document.body"):
                needle = json.loads(expr.split(".includes(", 1)[1][:-2])
                return {"result": {"type": "boolean", "value": needle in page.text}}, None, []
            if expr.startswith("document.querySelector("):
                sel = json.loads(expr[len("document.querySelector("):-1])
                if sel == "[[bad":
                    return {"result": {"type": "object"}, "exceptionDetails": {
                        "text": "Uncaught", "exception": {"description": "SyntaxError: bad selector"}}}, None, []
                if sel in _SELECTORS:
                    return {"result": self._obj(_SELECTORS[sel])}, None, []
                return {"result": {"type": "object", "subtype": "null", "value": None}}, None, []
            if "(function (needle)" in expr:
                needle = json.loads(expr.rsplit("(", 1)[1][:-1]).lower()
                if needle in _TEXT:
                    return {"result": self._obj(_TEXT[needle])}, None, []
                return {"result": {"type": "object", "subtype": "null", "value": None}}, None, []
            if expr == "6 * 7":
                return {"result": {"type": "number", "value": 42, "description": "42"}}, None, []
            if expr == "({a: [1, 'two']})":
                return {"result": {"type": "object", "value": {"a": [1, "two"]}}}, None, []
            if expr == "undefined":
                return {"result": {"type": "undefined"}}, None, []
            if expr.startswith("throw"):
                return {"result": {"type": "object"}, "exceptionDetails": {
                    "text": "Uncaught", "exception": {"description": "Error: boom"}}}, None, []
            return {"result": {"type": "string", "value": f"eval:{expr}"}}, None, []
        if method == "Accessibility.getFullAXTree":
            nodes = [
                {"nodeId": "1", "ignored": False, "role": {"value": "RootWebArea"},
                 "name": {"value": page.title}, "childIds": ["2", "3", "4", "5", "6"],
                 "backendDOMNodeId": 1,
                 "properties": [{"name": "focused", "value": {"type": "boolean", "value": True}}]},
                {"nodeId": "2", "ignored": False, "role": {"value": "heading"},
                 "name": {"value": "Hello"}, "childIds": ["7"], "backendDOMNodeId": 10},
                {"nodeId": "3", "ignored": False, "role": {"value": "textbox"},
                 "name": {"value": "Search"}, "value": {"value": page.value},
                 "childIds": [], "backendDOMNodeId": 11},
                {"nodeId": "4", "ignored": False, "role": {"value": "button"},
                 "name": {"value": "Press me"}, "childIds": [], "backendDOMNodeId": 12},
                {"nodeId": "5", "ignored": False, "role": {"value": "StaticText"},
                 "name": {"value": page.out}, "childIds": [], "backendDOMNodeId": 13},
                {"nodeId": "6", "ignored": True, "role": {"value": "generic"},
                 "childIds": [], "backendDOMNodeId": 14},
                {"nodeId": "7", "ignored": False, "role": {"value": "StaticText"},
                 "name": {"value": "Hello"}, "childIds": [], "backendDOMNodeId": 15},
            ]
            return {"nodes": nodes}, None, []
        if method == "DOM.resolveNode":
            backend = p.get("backendNodeId")
            if backend not in _ELEMENTS:
                return None, {"code": -32000, "message": "No node with given id found"}, []
            return {"object": self._obj(backend)}, None, []
        if method == "Runtime.callFunctionOn":
            backend = int(p["objectId"].split("-")[1])
            fn = p.get("functionDeclaration", "")
            role, name, tag = _ELEMENTS[backend]
            if "elementFromPoint" in fn:
                hittable = not (backend == 12 and page.button_covered)
                return {"result": {"type": "object", "value": {
                    "ok": True, "x": 100.0 + backend, "y": 50.0, "hittable": hittable,
                    "tag": tag, "label": name}}}, None, []
            if "el.focus()" in fn:
                page.focused = tag
                args = p.get("arguments") or []
                if args and args[0].get("value") and backend == 11:
                    page.value = ""
                return {"result": {"type": "string", "value": tag}}, None, []
            if "el.click()" in fn:
                page.dom_clicks += 1
                if backend == 12:
                    page.click_button()
                return {"result": {"type": "boolean", "value": True}}, None, []
            if '"value" in el' in fn:
                return {"result": {"type": "string", "value": page.value if backend == 11 else name}}, None, []
            return {"result": {"type": "undefined"}}, None, []
        if method == "Input.dispatchMouseEvent":
            page.mouse.append(p)
            if p.get("type") == "mouseReleased" and abs(p["x"] - 112.0) < 0.01:
                page.click_button()
            return {}, None, []
        if method == "Input.insertText":
            if page.focused == "input":
                page.value += p["text"]
            return {}, None, []
        if method == "Input.dispatchKeyEvent":
            page.keys.append(p)
            return {}, None, []
        if method == "Page.getLayoutMetrics":
            return {"cssContentSize": {"x": 0, "y": 0, "width": 800, "height": 3000}}, None, []
        if method == "Page.captureScreenshot":
            self.last_screenshot_params = p
            return {"data": base64.b64encode(self.png).decode()}, None, []
        if method == "Page.navigate":
            if p["url"].startswith("bad://"):
                return {"frameId": "F", "errorText": "net::ERR_ABORTED"}, None, []
            page.url = p["url"]
            page.title = "Navigated"
            return {"frameId": "F", "loaderId": "L1"}, None, [
                {"method": "Page.frameStartedLoading", "params": {}},
                {"method": "Page.loadEventFired", "params": {"timestamp": 1}},
            ]
        return None, {"code": -32601, "message": f"'{method}' wasn't found"}, []

    async def _ws_handler(self, ws):
        path = ws.request.path
        tid = path.rsplit("/", 1)[1]
        page = self.pages.get(tid)
        if page is None:
            await ws.close()
            return
        async for raw in ws:
            msg = json.loads(raw)
            result, error, events = self.handle(page, msg["method"], msg.get("params") or {})
            reply = {"id": msg["id"]}
            if error is not None:
                reply["error"] = error
            else:
                reply["result"] = result
            await ws.send(json.dumps(reply))
            for ev in events:
                await ws.send(json.dumps(ev))

    @contextlib.asynccontextmanager
    async def running(self):
        from websockets.asyncio.server import serve

        server = await serve(self._ws_handler, "127.0.0.1", 0, max_size=None)
        self.ws_port = list(server.sockets)[0].getsockname()[1]
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), self.make_handler())
        self.http_port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            yield self
        finally:
            httpd.shutdown()
            httpd.server_close()
            server.close()
            await server.wait_closed()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Driver:
    def __init__(self, session_id: str) -> None:
        self.tool = BrowserTool()
        self.ctx = ToolContext(session_id=session_id)

    async def __call__(self, **params):
        return await self.tool.execute(self.ctx, self.tool.parse_params(params))


@contextlib.asynccontextmanager
async def attached(session_id: str):
    fake = FakeCDP()
    async with fake.running():
        drive = Driver(session_id)
        r = await drive(action="attach", port=fake.http_port)
        assert not r.error, r.output
        try:
            yield fake, drive
        finally:
            await close_browser_session(session_id)


# ---------------------------------------------------------------------------
# Tests — CDP client
# ---------------------------------------------------------------------------


class TestCDPClient:
    async def test_http_and_ws_roundtrip(self):
        fake = FakeCDP()
        async with fake.running():
            http = DevToolsHTTP(fake.http_port)
            assert (await http.version())["Browser"] == "FakeChrome/1.0"
            targets = await http.targets()
            assert [t["id"] for t in targets] == ["T1"]  # service worker filtered
            conn = await CDPConnection(targets[0]["webSocketDebuggerUrl"]).connect()
            try:
                res = await conn.send("Runtime.evaluate", {"expression": "6 * 7"})
                assert res["result"]["value"] == 42
                with pytest.raises(CDPError, match="wasn't found"):
                    await conn.send("Bogus.method")
            finally:
                await conn.close()
            new = await http.new_tab("https://a.test/x?y=1")
            assert new["url"] == "https://a.test/x?y=1"

    async def test_missing_endpoint_is_helpful(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with pytest.raises(CDPError, match="remote-debugging-port"):
            await DevToolsHTTP(port).version()

    def test_render_ax_refs_and_skips(self):
        nodes = [
            {"nodeId": "1", "role": {"value": "RootWebArea"}, "name": {"value": "T"},
             "childIds": ["2", "3"], "backendDOMNodeId": 1},
            {"nodeId": "2", "role": {"value": "generic"}, "childIds": ["4"], "backendDOMNodeId": 2},
            {"nodeId": "3", "role": {"value": "checkbox"}, "name": {"value": "Agree"},
             "properties": [{"name": "checked", "value": {"value": "true"}}],
             "childIds": [], "backendDOMNodeId": 3},
            {"nodeId": "4", "role": {"value": "link"}, "name": {"value": "Docs"},
             "childIds": [], "backendDOMNodeId": 4},
        ]
        lines, refs, total = _render_ax(nodes, 100)
        assert lines[0] == '[e1] RootWebArea "T"'
        # generic container skipped; its child keeps the parent's depth + 1
        assert '  [e2] link "Docs"' in lines
        assert '  [e3] checkbox "Agree" (checked)' in lines
        assert refs == {"e1": 1, "e2": 4, "e3": 3}
        assert total == 3
        lines, refs, total = _render_ax(nodes, 2)
        assert len(lines) == 2 and total == 3


# ---------------------------------------------------------------------------
# Tests — BrowserTool against the fake endpoint
# ---------------------------------------------------------------------------


class TestBrowserTool:
    def test_metadata_contract(self):
        assert BrowserTool.name == "browser" and BrowserTool.title
        assert BrowserTool.annotations == {
            "readOnlyHint": False, "destructiveHint": True, "openWorldHint": True,
        }
        actions = BrowserTool.Params.model_json_schema()["properties"]["action"]["enum"]
        assert actions == ["launch", "attach", "tabs", "open", "navigate", "read", "click",
                           "type", "wait_for", "screenshot", "eval", "close_tab"]

    async def test_attach_and_tabs(self):
        async with attached("b-tabs") as (fake, drive):
            r = await drive(action="tabs")
            assert "*[0] Example — https://example.test/" in r.output
            assert get_browser_session("b-tabs").current == "T1"

    async def test_read_returns_text_and_ax_refs(self):
        async with attached("b-read") as (fake, drive):
            r = await drive(action="read")
            assert not r.error, r.output
            assert "Title: Example" in r.output and "URL: https://example.test/" in r.output
            assert "Press me" in r.output
            assert '[e4] button "Press me"' in r.output
            assert '[e3] textbox "Search"' in r.output
            assert "(focused)" in r.output
            # StaticText that repeats its parent's name is dropped; ignored nodes too
            assert r.output.count('"Hello"') == 1
            assert "generic" not in r.output
            assert get_browser_session("b-read").refs["T1"]["e4"] == 12

    async def test_read_truncates_text(self):
        async with attached("b-trunc") as (fake, drive):
            fake.pages["T1"].out = "z" * 5000
            r = await drive(action="read", max_chars=300)
            assert "showing first 300" in r.output
            assert r.metadata["text_truncated"] is True

    async def test_click_by_ref_uses_mouse_events(self):
        async with attached("b-click") as (fake, drive):
            await drive(action="read")
            r = await drive(action="click", ref="e4")
            assert not r.error, r.output
            assert "mouse click" in r.output
            page = fake.pages["T1"]
            assert [m["type"] for m in page.mouse] == ["mouseMoved", "mousePressed", "mouseReleased"]
            assert page.out.startswith("clicked")

    async def test_click_by_selector_and_text(self):
        async with attached("b-click2") as (fake, drive):
            r = await drive(action="click", selector="#b")
            assert not r.error and "selector '#b'" in r.output
            r = await drive(action="click", text="Press me")
            assert not r.error and "text 'Press me'" in r.output

    async def test_click_falls_back_to_dom_click_when_covered(self):
        async with attached("b-cover") as (fake, drive):
            fake.pages["T1"].button_covered = True
            r = await drive(action="click", selector="button")
            assert "dom click" in r.output
            assert fake.pages["T1"].dom_clicks == 1
            assert fake.pages["T1"].out.startswith("clicked")

    async def test_click_errors(self):
        async with attached("b-clickerr") as (fake, drive):
            r = await drive(action="click", ref="e99")
            assert r.error and "call read again" in r.output
            r = await drive(action="click", selector=".nope")
            assert r.error and "No element matches" in r.output
            r = await drive(action="click", selector="[[bad")
            assert r.error and "Invalid selector" in r.output
            r = await drive(action="click", text="no such words")
            assert r.error and "No visible element" in r.output
            r = await drive(action="click")
            assert r.error and "ref, selector, or text" in r.output

    async def test_type_into_selector_and_submit(self):
        async with attached("b-type") as (fake, drive):
            fake.pages["T1"].value = "old"
            r = await drive(action="type", selector="#q", text="abc", clear=True)
            assert not r.error, r.output
            assert fake.pages["T1"].value == "abc"
            assert "'abc'" in r.output
            await drive(action="read")
            r = await drive(action="type", ref="e3", text="de", submit=True)
            assert "pressed Enter" in r.output
            assert fake.pages["T1"].value == "abcde"
            assert [k["type"] for k in fake.pages["T1"].keys] == ["keyDown", "keyUp"]
            assert fake.pages["T1"].keys[0]["key"] == "Enter"

    async def test_click_then_read_reflects_typed_value(self):
        async with attached("b-flow") as (fake, drive):
            await drive(action="type", selector="#q", text="xyz")
            await drive(action="read")
            await drive(action="click", ref="e4")
            r = await drive(action="wait_for", text="clicked xyz", timeout=2)
            assert not r.error, r.output
            r = await drive(action="eval", expression="6 * 7")
            assert r.output == "42"

    async def test_wait_for_selector_and_timeout(self):
        async with attached("b-wait") as (fake, drive):
            r = await drive(action="wait_for", selector="h1", timeout=1)
            assert not r.error
            r = await drive(action="wait_for", text="never appears", timeout=0.3)
            assert r.error and "Timed out" in r.output

    async def test_eval_values_and_exceptions(self):
        async with attached("b-eval") as (fake, drive):
            r = await drive(action="eval", expression="({a: [1, 'two']})")
            assert json.loads(r.output) == {"a": [1, "two"]}
            r = await drive(action="eval", expression="undefined")
            assert r.output == "undefined"
            r = await drive(action="eval", expression="throw new Error('boom')")
            assert r.error and "Error: boom" in r.output

    async def test_screenshot_large_frame(self):
        async with attached("b-shot") as (fake, drive):
            r = await drive(action="screenshot")
            assert not r.error, r.output
            assert len(r.attachments) == 1
            att = r.attachments[0]
            assert att.media_type == "image/png" and att.content == fake.png
            assert len(att.content) > 2_000_000  # > default websocket max_size
            r = await drive(action="screenshot", full_page=True)
            assert fake.last_screenshot_params["clip"]["height"] == 3000
            assert fake.last_screenshot_params["captureBeyondViewport"] is True

    async def test_open_navigate_close(self):
        async with attached("b-nav") as (fake, drive):
            r = await drive(action="open", url="https://two.test/")
            assert not r.error, r.output
            new_id = get_browser_session("b-nav").current
            assert new_id != "T1" and fake.pages[new_id].url == "https://two.test/"
            r = await drive(action="navigate", url="https://three.test/")
            assert not r.error and "Navigated: Navigated — https://three.test/" in r.output
            r = await drive(action="navigate", url="bad://x")
            assert r.error and "ERR_ABORTED" in r.output
            r = await drive(action="close_tab")
            assert not r.error
            assert new_id not in fake.pages
            assert get_browser_session("b-nav").current == "T1"
            r = await drive(action="eval", expression="6 * 7", tab=0)
            assert r.output == "42"

    async def test_tab_selection_errors(self):
        async with attached("b-tabsel") as (fake, drive):
            r = await drive(action="eval", expression="6 * 7", tab=5)
            assert r.error and "No tab at index 5" in r.output
            r = await drive(action="eval", expression="6 * 7", tab="example.test")
            assert r.output == "42"

    async def test_records_audit(self):
        async with attached("b-audit") as (fake, drive):
            await drive(action="eval", expression="6 * 7")
            log = get_sandbox("b-audit").export_audit_log()
            assert [e["action"] for e in log][-1] == "browser"
            assert log[-1]["params"]["expression"] == "6 * 7"

    async def test_attach_without_browser_is_error(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        drive = Driver("b-none")
        r = await drive(action="attach", port=port)
        assert r.error and "No DevTools endpoint" in r.output

    async def test_launch_refuses_foreign_profile(self, tmp_path: Path, monkeypatch):
        profile = tmp_path / "real-profile"
        profile.mkdir()
        (profile / "Local State").write_text("{}")
        monkeypatch.setattr("opendesk.tools.browser.find_browser", lambda _p=None: "/bin/true")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        r = await Driver("b-prof")(action="launch", port=port, profile_dir=str(profile))
        assert r.error and "Refusing" in r.output
        assert not (profile / PROFILE_MARKER).exists()

    async def test_launch_refuses_port_owned_by_other_browser(self, monkeypatch, tmp_path: Path):
        # Never start a real browser from this test, even if the guard regresses.
        monkeypatch.setattr("opendesk.tools.browser.find_browser", lambda _p=None: None)
        fake = FakeCDP()
        async with fake.running():
            r = await Driver("b-port")(action="launch", port=fake.http_port,
                                       profile_dir=str(tmp_path / "p"))
            assert r.error and "already has a DevTools endpoint" in r.output
            assert "attach" in r.output

    async def test_launch_without_browser_binary(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr("opendesk.tools.browser.find_browser", lambda _p=None: None)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        r = await Driver("b-nobin")(action="launch", port=port, profile_dir=str(tmp_path / "p"))
        assert r.error and "Could not find" in r.output


# ---------------------------------------------------------------------------
# Real browser smoke test (skipped when no Chromium-family browser exists)
# ---------------------------------------------------------------------------

_REAL_BROWSER = find_browser()

_SMOKE_HTML = (
    "data:text/html,<title>Smoke</title><h1>Hello</h1>"
    "<input id=q placeholder=Search>"
    "<button id=b onclick=\"document.getElementById('out').textContent="
    "'clicked '+document.getElementById('q').value\">Press me</button>"
    "<p id=out>idle</p>"
)


@pytest.mark.skipif(_REAL_BROWSER is None, reason="no Chrome/Chromium/Edge/Brave installed")
async def test_real_headless_browser(tmp_path: Path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    profile = tmp_path / "browser-profile"
    drive = Driver("b-real")
    try:
        r = await drive(action="launch", headless=True, port=port,
                        profile_dir=str(profile), url=_SMOKE_HTML, timeout=30)
        assert not r.error, r.output
        assert (profile / PROFILE_MARKER).exists()

        r = await drive(action="read")
        assert not r.error, r.output
        ref = next(line.split("[")[1].split("]")[0]
                   for line in r.output.splitlines() if '] button "Press me"' in line)

        r = await drive(action="type", selector="#q", text="abc")
        assert "'abc'" in r.output
        r = await drive(action="click", ref=ref)
        assert not r.error, r.output
        r = await drive(action="wait_for", text="clicked abc", timeout=5)
        assert not r.error, r.output
        r = await drive(action="eval", expression="document.getElementById('out').textContent")
        assert json.loads(r.output) == "clicked abc"
        r = await drive(action="screenshot")
        assert r.attachments and r.attachments[0].content.startswith(b"\x89PNG")
    finally:
        await close_browser_session("b-real")
        shutil.rmtree(profile, ignore_errors=True)
        # Chrome's helper processes can touch the profile dir while exiting.
        await asyncio.sleep(1.0)
        shutil.rmtree(profile, ignore_errors=True)
    assert not profile.exists()
