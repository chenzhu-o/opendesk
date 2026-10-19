"""Tests for the native macOS accessibility helper (opendesk-ax), element
refs, background input, the Wayland backend, and the batched UIA tree.

Protocol / wrapper tests run everywhere against a fake helper written in
Python.  Real-helper tests run only on macOS with swiftc and Accessibility
permission.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from opendesk.computer import macos_native as native
from opendesk.computer import wayland
from opendesk.computer.base import CapabilityUnsupported
from opendesk.computer.local import LocalComputer, _uia_cached_tree
from opendesk.computer.types import (
    KeyAction,
    KeyEvent,
    Point,
    PointerAction,
    PointerButton,
    PointerEvent,
    Rect,
    UIElement,
)
from opendesk.tools.base import ToolContext
from opendesk.tools.mouse import MouseTool
from opendesk.tools.ui import UITool, _format_tree

from tests._fakes import FakeComputer

REPO = Path(__file__).resolve().parents[2]

FAKE_HELPER = textwrap.dedent('''
    import json, os, sys
    trusted = os.environ.get("FAKE_TRUSTED", "1") != "0"
    NODES = [
        {"ref": "w00001", "parent": None, "role": "AXWindow", "title": "Doc", "enabled": True, "focused": False,
         "frame": {"x": 0, "y": 0, "w": 400, "h": 300}},
        {"ref": "g00001", "parent": "w00001", "role": "AXGroup", "enabled": True, "focused": False},
        {"ref": "b00001", "parent": "g00001", "role": "AXButton", "title": "Save", "enabled": True,
         "focused": False, "frame": {"x": 10, "y": 20, "w": 30, "h": 40}, "actions": ["AXPress"]},
        {"ref": "s00001", "parent": "w00001", "role": "AXStaticText", "value": "Ready", "enabled": True,
         "focused": False},
        {"ref": "t00001", "parent": "w00001", "role": "AXTextArea", "value": "hello", "enabled": True,
         "focused": True},
    ]
    for line in sys.stdin:
        req = json.loads(line)
        cmd, rid = req["cmd"], req["id"]
        if cmd == "ping":
            out = {"ok": True, "result": {"version": 1, "trusted": trusted, "pid": os.getpid()}}
        elif cmd == "tree":
            out = {"ok": True, "result": {"pid": 42, "app": req.get("app") or "Fake", "window_title": "Doc",
                                         "truncated": False, "nodes": NODES}}
        elif cmd == "window_text":
            out = {"ok": True, "result": {"text": "Doc\\nhello %s" % req.get("max_chars")}}
        elif cmd == "echo":
            out = {"ok": True, "result": req}
        elif cmd == "fail":
            out = {"ok": False, "error": "boom"}
        elif cmd == "hang":
            continue
        elif cmd == "crash":
            sys.exit(3)
        else:
            out = {"ok": False, "error": "unknown command: " + cmd}
        out["id"] = rid
        sys.stdout.write(json.dumps(out) + "\\n")
        sys.stdout.flush()
''')


@pytest.fixture
def fake_argv(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_ax.py"
    script.write_text(FAKE_HELPER)
    return [sys.executable, "-u", str(script)]


@pytest.fixture
def fake_client(fake_argv: list[str]):
    client = native.AXHelper(fake_argv)
    native.set_client(client)
    yield client
    native.set_client(None)


# ---------------------------------------------------------------------------
# Source parity
# ---------------------------------------------------------------------------


def test_swift_source_identical_to_js_copy() -> None:
    ts = (REPO / "js" / "src" / "computer" / "macosNative.ts").read_text(encoding="utf-8")
    marker = "export const SWIFT_SOURCE = String.raw`"
    start = ts.index(marker) + len(marker)
    end = ts.index("`;", start)
    assert ts[start:end] == native.SWIFT_SOURCE


def test_swift_source_has_every_command() -> None:
    for cmd in ("ping", "frontmost", "tree", "get", "perform", "set_value", "focus",
                "element_at", "window_text", "click_pid", "key_pid"):
        assert f'case "{cmd}"' in native.SWIFT_SOURCE
    assert "AXUIElementCopyMultipleAttributeValues" in native.SWIFT_SOURCE
    assert "AXManualAccessibility" in native.SWIFT_SOURCE
    assert "postToPid" in native.SWIFT_SOURCE


def test_helper_path_uses_opendesk_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENDESK_HOME", str(tmp_path))
    assert native.helper_path() == tmp_path / "bin" / "opendesk-ax-v1"


def test_ensure_helper_unavailable_off_macos(monkeypatch) -> None:
    monkeypatch.setattr(native, "_PLATFORM", "Linux")
    with pytest.raises(native.NativeAXUnavailable):
        native.ensure_helper()


def test_ensure_helper_respects_kill_switch(monkeypatch) -> None:
    monkeypatch.setattr(native, "_PLATFORM", "Darwin")
    monkeypatch.setenv("OPENDESK_NATIVE_AX", "0")
    with pytest.raises(native.NativeAXUnavailable):
        native.ensure_helper()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_request_roundtrip_drops_none_args(fake_argv: list[str]) -> None:
    h = native.AXHelper(fake_argv)
    try:
        r = h.request("echo", a=1, b=None)
        assert r["cmd"] == "echo" and r["a"] == 1 and "b" not in r
        assert isinstance(r["id"], int)
    finally:
        h.close()


def test_request_error_is_raised(fake_argv: list[str]) -> None:
    h = native.AXHelper(fake_argv)
    try:
        with pytest.raises(native.NativeAXError, match="boom"):
            h.request("fail")
        assert h.request("ping")["trusted"] is True  # still usable
    finally:
        h.close()


def test_timeout_kills_and_restarts(fake_argv: list[str]) -> None:
    h = native.AXHelper(fake_argv, timeout=0.5)
    try:
        pid1 = h.request("ping")["pid"]
        with pytest.raises(native.NativeAXError, match="timed out"):
            h.request("hang")
        pid2 = h.request("ping")["pid"]
        assert pid1 != pid2
        assert h.restarts == 1
    finally:
        h.close()


def test_crash_then_restart(fake_argv: list[str]) -> None:
    h = native.AXHelper(fake_argv)
    try:
        with pytest.raises(native.NativeAXError, match="exited"):
            h.request("crash")
        assert h.request("ping")["trusted"] is True
    finally:
        h.close()


def test_thread_safe_concurrent_requests(fake_argv: list[str]) -> None:
    h = native.AXHelper(fake_argv)
    results: dict[int, Any] = {}
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            results[i] = h.request("echo", n=i)["n"]
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    h.close()
    assert not errors
    assert results == {i: i for i in range(20)}


def test_missing_binary_is_unavailable() -> None:
    h = native.AXHelper(["/nonexistent/opendesk-ax"])
    with pytest.raises(native.NativeAXUnavailable):
        h.request("ping")


# ---------------------------------------------------------------------------
# Public API through the shared client
# ---------------------------------------------------------------------------


def test_window_text_signature_and_result(fake_client) -> None:
    import inspect
    sig = inspect.signature(native.window_text)
    assert list(sig.parameters) == ["pid", "max_chars"]
    assert sig.parameters["pid"].default is None
    assert sig.parameters["max_chars"].default == 20000
    assert native.window_text(max_chars=77) == "Doc\nhello 77"


def test_window_text_none_without_permission(fake_argv, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_TRUSTED", "0")
    native.set_client(native.AXHelper(fake_argv))
    try:
        assert native.available() is False
        assert native.window_text() is None
    finally:
        native.set_client(None)


def test_window_text_none_when_helper_cannot_start() -> None:
    native.set_client(native.AXHelper(["/nonexistent/opendesk-ax"]))
    try:
        assert native.window_text() is None
        assert native.available() is False
    finally:
        native.set_client(None)


def test_window_text_none_off_macos(monkeypatch) -> None:
    native.set_client(None)
    monkeypatch.setattr(native, "_PLATFORM", "Linux")
    assert native.window_text() is None
    assert native.available() is False
    native.set_client(None)


def test_to_ui_element_builds_tree_with_refs(fake_client) -> None:
    root = native.to_ui_element(native.tree(app="Fake"))
    assert root.role == "AXWindow" and root.ref == "w00001"
    assert root.metadata["app"] == "Fake" and root.metadata["pid"] == 42
    group = root.children[0]
    button = group.children[0]
    assert button.ref == "b00001" and button.name == "Save"
    assert button.bounds == Rect(x=10, y=20, width=30, height=40)
    assert button.actions == ["AXPress"]
    static = root.children[1]
    assert static.name == "Ready"  # static text value becomes its label
    text = _format_tree(root)
    assert "Window: Doc  [w00001]" in text
    assert "[b00001] [AXButton] Save" in text
    assert '[t00001] [AXTextArea] (unnamed) = "hello"' in text
    assert "AXGroup" not in text  # unlabelled layout container hidden


# ---------------------------------------------------------------------------
# UIElement.ref wire compatibility
# ---------------------------------------------------------------------------


def test_uielement_ref_travels_in_metadata_on_the_wire() -> None:
    el = UIElement(role="w", ref="abc", children=[UIElement(role="b", ref="x1"), UIElement(role="c")])
    wire = el.model_dump()
    assert "ref" not in wire and wire["metadata"] == {"ref": "abc"}
    assert wire["children"][0]["metadata"] == {"ref": "x1"}
    assert wire["children"][1]["metadata"] == {}
    back = UIElement.model_validate(wire)
    assert back == el and back.metadata == {}
    assert UIElement.model_validate_json(el.model_dump_json()) == el


def test_uielement_without_ref_is_unchanged_on_the_wire() -> None:
    el = UIElement(role="button", name="OK", metadata={"app": "X"})
    assert el.model_dump() == {
        "role": "button", "name": "OK", "value": None, "bounds": None, "focused": False,
        "enabled": True, "actions": [], "children": [], "metadata": {"app": "X"},
    }


def test_old_peer_model_accepts_new_wire_format() -> None:
    """A peer predating ``ref`` (extra='forbid') still validates our dumps."""
    from pydantic import BaseModel, ConfigDict, Field

    class OldUIElement(BaseModel):
        model_config = ConfigDict(extra="forbid")
        role: str
        name: str = ""
        value: Optional[str] = None
        bounds: Optional[Rect] = None
        focused: bool = False
        enabled: bool = True
        actions: list[str] = Field(default_factory=list)
        children: list["OldUIElement"] = Field(default_factory=list)
        metadata: dict[str, Any] = Field(default_factory=dict)

    OldUIElement.model_rebuild()
    old = OldUIElement.model_validate(UIElement(role="w", ref="r1", children=[UIElement(role="b", ref="r2")]).model_dump())
    assert old.metadata["ref"] == "r1"
    assert UIElement.model_validate(old.model_dump()).children[0].ref == "r2"


# ---------------------------------------------------------------------------
# LocalComputer wiring (fake native module)
# ---------------------------------------------------------------------------


class FakeNative:
    NativeAXError = native.NativeAXError

    def __init__(self, hit: Optional[dict] = None, perform_fails: bool = False) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.hit = hit
        self.perform_fails = perform_fails

    def _rec(self, name: str, *a: Any, **k: Any) -> dict:
        self.calls.append((name, a, k))
        return {"pid": 42}

    def available(self) -> bool:
        return True

    def tree(self, **kw: Any) -> dict:
        self._rec("tree", **kw)
        return {"pid": 42, "app": "Fake", "window_title": "Doc", "truncated": False, "nodes": [
            {"ref": "w1", "parent": None, "role": "AXWindow", "title": "Doc"},
            {"ref": "b1", "parent": "w1", "role": "AXButton", "title": "Save", "actions": ["AXPress"],
             "frame": {"x": 1, "y": 2, "w": 3, "h": 4}},
        ]}

    to_ui_element = staticmethod(native.to_ui_element)
    node_name = staticmethod(native.node_name)

    def perform(self, ref: str, action: str = "AXPress") -> dict:
        if self.perform_fails:
            raise native.NativeAXError("no AXPress")
        return self._rec("perform", ref, action)

    def focus(self, ref: str) -> dict:
        return self._rec("focus", ref)

    def set_value(self, ref: str, value: str) -> dict:
        return self._rec("set_value", ref, value)

    def key_pid(self, **kw: Any) -> dict:
        return self._rec("key_pid", **kw)

    def click_pid(self, x: float, y: float, **kw: Any) -> dict:
        return self._rec("click_pid", x, y, **kw)

    def element_at(self, x: float, y: float) -> Optional[dict]:
        self._rec("element_at", x, y)
        return self.hit

    def frontmost(self) -> dict:
        return {"app": "Fake", "pid": 42, "bundle_id": "x.fake", "window_title": "Doc"}


@pytest.fixture
def darwin_native(monkeypatch):
    import opendesk.computer.local as local

    fake = FakeNative()
    monkeypatch.setattr(local, "_PLATFORM", "Darwin")
    monkeypatch.setattr(local, "_native", lambda: fake)
    return fake


async def test_local_ui_tree_uses_native(darwin_native) -> None:
    tree = await LocalComputer().ui_tree(app="Fake")
    assert tree.ref == "w1" and tree.children[0].ref == "b1"
    assert darwin_native.calls[0][0] == "tree"


async def test_local_focused_window_uses_native(darwin_native) -> None:
    w = await LocalComputer().focused_window()
    assert w is not None and w.app_name == "Fake" and w.title == "Doc" and w.pid == 42


async def test_local_perform_by_ref(darwin_native) -> None:
    lc = LocalComputer()
    await lc.perform_ui_action(UIElement(role="", ref="b1"), "click")
    await lc.perform_ui_action(UIElement(role="", ref="b1"), "AXShowMenu")
    await lc.perform_ui_action(UIElement(role="", ref="t1"), "focus")
    await lc.perform_ui_action(UIElement(role="", ref="t1", value="v"), "set_value")
    await lc.perform_ui_action(UIElement(role="", ref="t1", value="hi"), "type")
    await lc.perform_ui_action(
        UIElement(role="application", name="Fake", metadata={"app": "Fake", "key": "s", "modifiers": ["cmd"]}),
        "key",
    )
    names = [(c[0], c[1], c[2]) for c in darwin_native.calls]
    assert names[0] == ("perform", ("b1", "AXPress"), {})
    assert names[1] == ("perform", ("b1", "AXShowMenu"), {})
    assert names[2] == ("focus", ("t1",), {})
    assert names[3] == ("set_value", ("t1", "v"), {})
    assert names[4] == ("key_pid", (), {"text": "hi", "ref": "t1"})
    assert names[5] == ("key_pid", (), {"key": "s", "modifiers": ["cmd"], "app": "Fake"})


async def test_native_only_action_never_falls_back_to_click(monkeypatch) -> None:
    import opendesk.computer.local as local

    monkeypatch.setattr(local, "_PLATFORM", "Darwin")
    monkeypatch.setattr(local, "_native", lambda: None)
    called: list[str] = []
    monkeypatch.setattr(local, "_osascript", lambda *a, **k: called.append("osascript"))
    with pytest.raises(CapabilityUnsupported):
        await LocalComputer().perform_ui_action(UIElement(role="AXTextArea", name="x", value="t"), "type")
    assert called == []


async def test_native_only_action_unsupported_elsewhere(monkeypatch) -> None:
    import opendesk.computer.local as local

    monkeypatch.setattr(local, "_PLATFORM", "Windows")
    with pytest.raises(CapabilityUnsupported):
        await LocalComputer().perform_ui_action(UIElement(role="Edit", name="x", value="t"), "type")


async def test_background_click_prefers_axpress(darwin_native) -> None:
    darwin_native.hit = {"ref": "s1", "role": "AXStaticText", "value": "Save", "pid": 42, "app": "Fake",
                         "pressable": "b1"}
    msg = await LocalComputer().background_click(Point(x=5, y=6))
    assert "accessibility" in msg and "cursor not moved" in msg
    assert ("perform", ("b1", "AXPress"), {}) in [(c[0], c[1], c[2]) for c in darwin_native.calls]
    assert not any(c[0] == "click_pid" for c in darwin_native.calls)


async def test_background_click_posts_to_pid(darwin_native) -> None:
    darwin_native.hit = {"ref": "c1", "role": "AXGroup", "pid": 42, "app": "Fake"}
    msg = await LocalComputer().background_click(Point(x=5, y=6), button=PointerButton.RIGHT, count=1)
    assert "Posted right click" in msg
    call = [c for c in darwin_native.calls if c[0] == "click_pid"][0]
    assert call[1] == (5.0, 6.0) and call[2] == {"pid": 42, "button": "right", "count": 1}


async def test_background_click_unsupported_without_helper(monkeypatch) -> None:
    import opendesk.computer.local as local

    monkeypatch.setattr(local, "_native", lambda: None)
    with pytest.raises(CapabilityUnsupported):
        await LocalComputer().background_click(Point(x=1, y=1))


# ---------------------------------------------------------------------------
# ui / mouse tools
# ---------------------------------------------------------------------------


class RefComputer(FakeComputer):
    """FakeComputer whose tree carries refs and whose a11y action can fail."""

    def __init__(self, perform_ok: bool = True) -> None:
        super().__init__()
        self.perform_ok = perform_ok

    async def ui_tree(self, *, window_id=None, app=None, max_depth=8) -> UIElement:
        self._record("ui_tree", app=app)
        return UIElement(role="AXWindow", name="Doc", ref="w1", children=[
            UIElement(role="AXButton", name="Save", ref="b1", actions=["AXPress"],
                      bounds=Rect(x=10, y=20, width=80, height=30)),
            UIElement(role="AXTextField", name="", value="typed", ref="t1"),
        ])

    async def perform_ui_action(self, element, action="click", *, app=None) -> None:
        self._record("perform_ui_action", element=element, action=action, app=app)
        if not self.perform_ok:
            raise RuntimeError("nope")

    async def background_click(self, point, *, button=PointerButton.LEFT, count=1) -> str:
        self._record("background_click", point=point, button=button, count=count)
        return "bg"


async def test_ui_get_tree_shows_refs() -> None:
    comp = RefComputer()
    r = await UITool().execute(ToolContext(computer=comp), UITool.Params(action="get_tree", app="Doc"))
    assert "[b1] [AXButton] Save" in r.output
    assert '[t1] [AXTextField] (unnamed) = "typed"' in r.output


async def test_ui_click_by_ref_acts_directly() -> None:
    comp = RefComputer()
    r = await UITool().execute(ToolContext(computer=comp), UITool.Params(action="click", app="Doc", ref="b1"))
    assert not r.error and "b1" in r.output
    methods = [c[0] for c in comp.calls]
    assert methods == ["perform_ui_action"]  # no tree fetch
    assert comp.calls[0][1]["element"].ref == "b1"


async def test_ui_click_by_ref_falls_back_to_bounds(monkeypatch) -> None:
    monkeypatch.delenv("OPENDESK_BACKGROUND", raising=False)
    comp = RefComputer(perform_ok=False)
    r = await UITool().execute(ToolContext(computer=comp), UITool.Params(action="click", app="Doc", ref="b1"))
    assert not r.error and "bounds-center" in r.output
    assert "pointer" in [c[0] for c in comp.calls]


async def test_ui_click_background_never_moves_cursor(monkeypatch) -> None:
    monkeypatch.setenv("OPENDESK_BACKGROUND", "1")
    comp = RefComputer(perform_ok=False)
    r = await UITool().execute(ToolContext(computer=comp), UITool.Params(action="click", app="Doc", ref="b1"))
    assert not r.error and "background click" in r.output
    methods = [c[0] for c in comp.calls]
    assert "pointer" not in methods and "background_click" in methods


async def test_ui_get_value_by_ref() -> None:
    comp = RefComputer()
    r = await UITool().execute(ToolContext(computer=comp), UITool.Params(action="get_value", app="Doc", ref="t1"))
    assert r.output == "typed"
    r = await UITool().execute(ToolContext(computer=comp), UITool.Params(action="get_value", app="Doc", ref="zz"))
    assert r.error and "get_tree" in r.output


async def test_ui_type_by_ref_foreground_focuses_element(monkeypatch) -> None:
    monkeypatch.delenv("OPENDESK_BACKGROUND", raising=False)
    comp = RefComputer()
    r = await UITool().execute(
        ToolContext(computer=comp), UITool.Params(action="type", app="Doc", ref="t1", text="hi"),
    )
    assert not r.error
    focus = [c for c in comp.calls if c[0] == "perform_ui_action"][0]
    assert focus[1]["action"] == "focus" and focus[1]["element"].ref == "t1"
    assert "text" in [c[0] for c in comp.calls]


async def test_ui_type_and_keys_background(monkeypatch) -> None:
    monkeypatch.setenv("OPENDESK_BACKGROUND", "1")
    comp = RefComputer()
    await UITool().execute(ToolContext(computer=comp), UITool.Params(action="type", app="Doc", text="hi"))
    await UITool().execute(
        ToolContext(computer=comp),
        UITool.Params(action="press_key", app="Doc", ref="t1", key="return", modifiers=["shift"]),
    )
    acts = [c[1] for c in comp.calls if c[0] == "perform_ui_action"]
    assert acts[0]["action"] == "type" and acts[0]["element"].value == "hi"
    assert acts[1]["action"] == "key" and acts[1]["element"].ref == "t1"
    assert acts[1]["element"].metadata["key"] == "return"
    methods = [c[0] for c in comp.calls]
    assert "focus_app" not in methods and "text" not in methods and "key" not in methods


async def test_mouse_background_env_refuses_cursor_moves(monkeypatch) -> None:
    monkeypatch.setenv("OPENDESK_BACKGROUND", "1")
    comp = RefComputer()
    for action in ("move", "drag", "scroll", "left_down"):
        r = await MouseTool().execute(
            ToolContext(computer=comp), MouseTool.Params(action=action, x=1, y=2, end_x=3, end_y=4, settle_ms=0),
        )
        assert r.error and "ref" in r.output
    assert comp.calls == []


async def test_mouse_background_click_routes_to_computer(monkeypatch) -> None:
    monkeypatch.setenv("OPENDESK_BACKGROUND", "1")
    comp = RefComputer()
    r = await MouseTool().execute(
        ToolContext(computer=comp), MouseTool.Params(action="double_click", x=5, y=6, settle_ms=0),
    )
    assert not r.error and r.output.startswith("bg")
    call = [c for c in comp.calls if c[0] == "background_click"][0][1]
    assert call["count"] == 2 and call["point"] == Point(x=5, y=6)
    assert "pointer" not in [c[0] for c in comp.calls]


async def test_mouse_background_flag_without_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENDESK_BACKGROUND", raising=False)
    comp = RefComputer()
    r = await MouseTool().execute(
        ToolContext(computer=comp), MouseTool.Params(action="click", x=5, y=6, background=True, settle_ms=0),
    )
    assert not r.error
    r = await MouseTool().execute(
        ToolContext(computer=comp), MouseTool.Params(action="move", x=5, y=6, background=True, settle_ms=0),
    )
    assert r.error


async def test_mouse_background_unsupported_message() -> None:
    comp = FakeComputer()  # base Computer.background_click -> CapabilityUnsupported
    r = await MouseTool().execute(
        ToolContext(computer=comp), MouseTool.Params(action="click", x=5, y=6, background=True, settle_ms=0),
    )
    assert r.error and "opendesk-ax" in r.output


# ---------------------------------------------------------------------------
# Wayland
# ---------------------------------------------------------------------------


def test_is_wayland_detection() -> None:
    assert wayland.is_wayland({"WAYLAND_DISPLAY": "wayland-0"})
    assert wayland.is_wayland({"XDG_SESSION_TYPE": "wayland"})
    assert not wayland.is_wayland({"DISPLAY": ":0"})
    assert not wayland.is_wayland({"WAYLAND_DISPLAY": "wayland-0", "OPENDESK_FORCE_X11": "1"})


def test_capture_command_construction() -> None:
    assert wayland.grim_command(None, "/t.png") == ["grim", "/t.png"]
    assert wayland.grim_command((10, 20, 300, 400), "/t.png") == ["grim", "-g", "10,20 300x400", "/t.png"]
    assert wayland.gnome_screenshot_command("/t.png") == ["gnome-screenshot", "-f", "/t.png"]
    call = wayland.portal_call_command("opendesk_abc")
    assert call[:3] == ["gdbus", "call", "--session"]
    assert "org.freedesktop.portal.Screenshot.Screenshot" in call
    assert "'handle_token': <'opendesk_abc'>" in call[-1] and "'interactive': <false>" in call[-1]


def test_capture_backend_order() -> None:
    have = {"grim", "gdbus", "gnome-screenshot"}
    assert wayland.capture_backends(lambda b: b in have) == ["grim", "portal", "gnome-screenshot"]
    assert wayland.capture_backends(lambda b: b == "gdbus") == ["portal"]
    assert wayland.capture_backends(lambda b: False) == []


def test_parse_portal_response() -> None:
    line = (
        "/org/freedesktop/portal/desktop/request/1_42/opendesk_abc: "
        "org.freedesktop.portal.Request.Response (uint32 0, "
        "{'uri': <'file:///home/u/Pictures/Screenshot%20from%20now.png'>})"
    )
    code, uri = wayland.parse_portal_response(line, "opendesk_abc")
    assert code == 0 and uri == "file:///home/u/Pictures/Screenshot%20from%20now.png"
    assert wayland.uri_to_path(uri) == "/home/u/Pictures/Screenshot from now.png"
    assert wayland.parse_portal_response(line, "other_token") is None
    denied = "/x/opendesk_abc: org.freedesktop.portal.Request.Response (uint32 1, @a{sv} {})"
    assert wayland.parse_portal_response(denied, "opendesk_abc") == (1, None)


def test_capture_without_tools_raises_actionable_error(monkeypatch) -> None:
    pytest.importorskip("PIL")
    monkeypatch.setattr(wayland, "capture_backends", lambda which=None: [])
    with pytest.raises(wayland.WaylandUnsupported, match="grim"):
        wayland.capture()


def test_ydotool_pointer_and_key_commands() -> None:
    move = PointerEvent(action=PointerAction.MOVE, point=Point(x=10, y=20))
    assert wayland.pointer_commands(move) == [["ydotool", "mousemove", "--absolute", "-x", "10", "-y", "20"]]
    down = PointerEvent(action=PointerAction.DOWN, point=Point(x=1, y=2), button=PointerButton.RIGHT)
    assert wayland.pointer_commands(down)[1] == ["ydotool", "click", "0x41"]
    up = PointerEvent(action=PointerAction.UP, point=Point(x=1, y=2), button=PointerButton.LEFT)
    assert wayland.pointer_commands(up)[1] == ["ydotool", "click", "0x80"]
    scroll = PointerEvent(action=PointerAction.SCROLL, point=Point(x=1, y=2), dy=-3)
    assert wayland.pointer_commands(scroll)[1] == ["ydotool", "mousemove", "--wheel", "-x", "0", "-y", "-3"]
    assert wayland.key_command(KeyEvent(action=KeyAction.DOWN, keysym="return")) == ["ydotool", "key", "28:1"]
    assert wayland.key_command(KeyEvent(action=KeyAction.UP, keysym="ctrl")) == ["ydotool", "key", "29:0"]
    assert wayland.type_command("hé", interval_ms=5) == ["ydotool", "type", "--key-delay", "5", "--", "hé"]
    with pytest.raises(ValueError):
        wayland.evdev_code("nonsense")


def test_ydotool_missing_gives_actionable_error(monkeypatch) -> None:
    monkeypatch.setattr(wayland.shutil, "which", lambda b: None)
    with pytest.raises(wayland.WaylandUnsupported, match="ydotoold"):
        wayland.type_text("hi")


async def test_local_routes_input_through_wayland(monkeypatch) -> None:
    import opendesk.computer.local as local

    ran: list[list[str]] = []
    monkeypatch.setattr(local, "_PLATFORM", "Linux")
    monkeypatch.setattr(wayland, "is_wayland", lambda env=None: True)
    monkeypatch.setattr(wayland, "_run", lambda cmd: ran.append(cmd))
    lc = LocalComputer()
    await lc.click(Point(x=3, y=4))
    await lc.press("a")
    await lc.type_text("hey")
    assert ran[0][:2] == ["ydotool", "mousemove"]
    assert ["ydotool", "click", "0x40"] in ran and ["ydotool", "click", "0x80"] in ran
    assert ["ydotool", "key", "30:1"] in ran and ["ydotool", "key", "30:0"] in ran
    assert ran[-1][:2] == ["ydotool", "type"] and ran[-1][-1] == "hey"


def test_capture_screen_routes_through_wayland(monkeypatch) -> None:
    Image = pytest.importorskip("PIL.Image")
    import opendesk.computer.capture as capture

    monkeypatch.setattr(capture, "_PLATFORM", "Linux")
    monkeypatch.setattr(wayland, "is_wayland", lambda env=None: True)
    monkeypatch.setattr(wayland, "capture", lambda region=None: Image.new("RGB", (40, 30)))
    png, w, h = capture.capture_screen()
    assert (w, h) == (40, 30) and png[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# Windows UIA CacheRequest
# ---------------------------------------------------------------------------


class _Arr:
    def __init__(self, items: list) -> None:
        self.items = items
        self.Length = len(items)

    def GetElement(self, i: int):
        return self.items[i]


class _El:
    def __init__(self, props: dict, children: Optional[list] = None) -> None:
        self.props = props
        self.children = children or []

    def GetCachedPropertyValue(self, pid: int):
        return self.props.get(pid)

    def GetCachedChildren(self):
        return _Arr(self.children) if self.children else None


class _Req:
    def __init__(self) -> None:
        self.props: list[int] = []
        self.TreeScope = 0

    def AddProperty(self, pid: int) -> None:
        self.props.append(pid)


def test_uia_cached_tree_single_batched_request() -> None:
    rect = SimpleNamespace(left=10, top=20, right=110, bottom=50)
    button = _El({30003: 50000, 30005: "OK", 30001: rect, 30010: True, 30011: "okBtn"})
    edit = _El({30003: 50004, 30005: "Name", 30045: "Ada", 30008: True})
    pane = _El({30003: 50033, 30005: ""}, [button, edit])
    cached_root = _El({}, [pane])
    built: list[_Req] = []

    class Root:
        def BuildUpdatedCache(self, req):
            built.append(req)
            return cached_root

    uia = SimpleNamespace(CreateCacheRequest=lambda: _Req())
    out = _uia_cached_tree(uia, Root())
    assert len(built) == 1 and built[0].TreeScope == 7
    assert {30001, 30003, 30005, 30045} <= set(built[0].props)
    assert out[0].role == "Pane"
    ok, name = out[0].children
    assert ok.role == "Button" and ok.name == "OK" and ok.metadata == {"automation_id": "okBtn"}
    assert ok.bounds == Rect(x=10, y=20, width=100, height=30)
    assert name.role == "Edit" and name.value == "Ada" and name.focused


def test_uia_cached_tree_respects_limits() -> None:
    leaves = [_El({30003: 50000, 30005: f"b{i}"}) for i in range(50)]
    root = _El({}, [_El({30003: 50033}, leaves)])

    class R:
        def BuildUpdatedCache(self, req):
            return root

    out = _uia_cached_tree(SimpleNamespace(CreateCacheRequest=lambda: _Req()), R(), max_nodes=10)
    assert len(out[0].children) == 9  # pane + 9 buttons = 10 nodes
    out = _uia_cached_tree(SimpleNamespace(CreateCacheRequest=lambda: _Req()), R(), max_depth=1)
    assert out[0].children == []


# ---------------------------------------------------------------------------
# Real helper (macOS + swiftc + Accessibility permission)
# ---------------------------------------------------------------------------

_REAL = (
    platform.system() == "Darwin"
    and shutil.which("swiftc") is not None
    and os.environ.get("OPENDESK_NATIVE_AX", "") != "0"
)


@pytest.fixture
def real_helper():
    if not _REAL:
        pytest.skip("needs macOS with swiftc")
    native.set_client(None)
    if not native.available():
        pytest.skip("opendesk-ax unavailable (no Accessibility permission?)")
    yield native
    native.set_client(None)


def test_real_helper_frontmost_tree_text(real_helper) -> None:
    info = real_helper.ping()
    assert info["version"] == 1 and info["trusted"] is True
    front = real_helper.frontmost()
    assert front and front["app"] and front["pid"] > 0
    t1 = real_helper.tree(app="Finder")
    assert t1["app"] == "Finder" and t1["nodes"]
    t2 = real_helper.tree(app="Finder")
    assert [n["ref"] for n in t1["nodes"]][:30] == [n["ref"] for n in t2["nodes"]][:30]
    assert all(len(n["ref"]) >= 6 for n in t1["nodes"])
    text = real_helper.window_text()
    assert isinstance(text, str)


def test_real_helper_errors(real_helper) -> None:
    with pytest.raises(native.NativeAXError, match="unknown ref"):
        real_helper.perform("zzzzzz")
    with pytest.raises(native.NativeAXError, match="not running"):
        real_helper.tree(app="NoSuchApp-opendesk-test")
    hit = real_helper.element_at(5, 5)
    assert hit is None or ("ref" in hit and "pid" in hit)
