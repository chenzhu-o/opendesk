"""Linux Wayland backend: screen capture and input without X11.

Under Wayland, X11 tools (mss, pyautogui, xdotool) only see XWayland
windows or nothing at all, so :class:`~opendesk.computer.local.LocalComputer`
routes through this module whenever ``WAYLAND_DISPLAY`` is set (or
``XDG_SESSION_TYPE=wayland``).  Set ``OPENDESK_FORCE_X11=1`` to opt out.

Capture, first that works:

1. ``grim`` — wlroots compositors (Sway, Hyprland, river, ...).
2. xdg-desktop-portal ``Screenshot`` API via ``gdbus`` — GNOME, KDE, and any
   desktop with a portal.  Non-interactive where the portal allows it.
3. ``gnome-screenshot -f``.

Input: ``ydotool`` (v1.0+, needs the ``ydotoold`` daemon and access to
``/dev/uinput``).  Without it, input raises :class:`WaylandUnsupported`
with instructions.

Command construction is kept in small pure functions so it can be unit
tested off Linux.
"""

from __future__ import annotations

import os
import platform
import re
import select
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from opendesk.computer.types import KeyAction, KeyEvent, PointerAction, PointerButton, PointerEvent


class WaylandUnsupported(RuntimeError):
    """No usable Wayland capture / input tool is installed."""


CAPTURE_HELP = (
    "Screen capture on Wayland needs one of: grim (wlroots compositors such as "
    "Sway/Hyprland: `sudo apt install grim`), xdg-desktop-portal with gdbus "
    "(GNOME/KDE; usually preinstalled), or gnome-screenshot. "
    "Set OPENDESK_FORCE_X11=1 to use the X11 path under XWayland instead."
)

INPUT_HELP = (
    "Mouse/keyboard input on Wayland needs ydotool (v1.0+): install it "
    "(`sudo apt install ydotool` or build from github.com/ReimuNotMoe/ydotool), "
    "start the daemon (`sudo ydotoold` or `systemctl --user enable --now ydotool`), "
    "and make sure your user can write /dev/uinput (e.g. add it to the `input` group). "
    "Set OPENDESK_FORCE_X11=1 to use pyautogui under XWayland instead (X11 apps only)."
)


def is_wayland(env: Optional[dict[str, str]] = None) -> bool:
    """True on a Linux Wayland session (unless ``OPENDESK_FORCE_X11=1``)."""
    env = os.environ if env is None else env
    if platform.system() != "Linux" and env is os.environ:
        return False
    if env.get("OPENDESK_FORCE_X11", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return bool(env.get("WAYLAND_DISPLAY")) or env.get("XDG_SESSION_TYPE", "").lower() == "wayland"


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture_backends(which: Any = shutil.which) -> list[str]:
    """Capture tools present on this machine, in preference order."""
    out: list[str] = []
    if which("grim"):
        out.append("grim")
    if which("gdbus"):
        out.append("portal")
    if which("gnome-screenshot"):
        out.append("gnome-screenshot")
    return out


def grim_command(region: Optional[tuple[int, int, int, int]], out_path: str) -> list[str]:
    cmd = ["grim"]
    if region is not None:
        x, y, w, h = region
        cmd += ["-g", f"{x},{y} {w}x{h}"]
    return cmd + [out_path]


def gnome_screenshot_command(out_path: str) -> list[str]:
    return ["gnome-screenshot", "-f", out_path]


def portal_monitor_command() -> list[str]:
    return ["gdbus", "monitor", "--session", "--dest", "org.freedesktop.portal.Desktop"]


def portal_call_command(token: str) -> list[str]:
    return [
        "gdbus", "call", "--session",
        "--dest", "org.freedesktop.portal.Desktop",
        "--object-path", "/org/freedesktop/portal/desktop",
        "--method", "org.freedesktop.portal.Screenshot.Screenshot",
        "",
        f"{{'handle_token': <'{token}'>, 'interactive': <false>, 'modal': <false>}}",
    ]


_PORTAL_RESPONSE = re.compile(r"Response\s*\(uint32 (\d+),(.*)\)\s*$")
_PORTAL_URI = re.compile(r"'uri':\s*<'([^']+)'>")


def parse_portal_response(line: str, token: str) -> Optional[tuple[int, Optional[str]]]:
    """Parse one ``gdbus monitor`` line.  Returns ``(code, uri)`` for the
    Response signal of our request (code 0 = success), else None."""
    if token not in line or "org.freedesktop.portal.Request.Response" not in line:
        return None
    m = _PORTAL_RESPONSE.search(line)
    if not m:
        return None
    uri = _PORTAL_URI.search(m.group(2))
    return int(m.group(1)), (uri.group(1) if uri else None)


def uri_to_path(uri: str) -> str:
    return unquote(urlparse(uri).path)


def _portal_capture(timeout: float = 15.0) -> bytes:
    token = f"opendesk_{uuid.uuid4().hex[:12]}"
    mon = subprocess.Popen(
        portal_monitor_command(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        time.sleep(0.2)  # let the monitor subscribe before the call
        r = subprocess.run(portal_call_command(token), capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            raise RuntimeError(f"portal Screenshot call failed: {r.stderr.strip()}")
        assert mon.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([mon.stdout], [], [], 0.25)
            if not ready:
                continue
            line = mon.stdout.readline()
            if not line:
                break
            parsed = parse_portal_response(line, token)
            if parsed is None:
                continue
            code, uri = parsed
            if code != 0 or not uri:
                raise RuntimeError(f"portal Screenshot was denied or cancelled (code {code})")
            path = uri_to_path(uri)
            try:
                return Path(path).read_bytes()
            finally:
                # The portal saves into ~/Pictures; don't leave copies behind.
                try:
                    os.unlink(path)
                except OSError:
                    pass
        raise RuntimeError("portal Screenshot timed out")
    finally:
        mon.kill()


def capture(region: Optional[tuple[int, int, int, int]] = None) -> Any:
    """Capture the screen (or ``region``) and return a PIL image."""
    import io

    from PIL import Image  # type: ignore[import-not-found]

    errors: list[str] = []
    for backend in capture_backends():
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            if backend == "portal":
                data = _portal_capture()
            else:
                cmd = (
                    grim_command(region, tmp) if backend == "grim"
                    else gnome_screenshot_command(tmp)
                )
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if r.returncode != 0 or not os.path.getsize(tmp):
                    errors.append(f"{backend}: {(r.stderr or r.stdout).strip()[:200]}")
                    continue
                data = Path(tmp).read_bytes()
            img = Image.open(io.BytesIO(data))
            img.load()
            if region is not None and backend != "grim":
                x, y, w, h = region
                img = img.crop((x, y, x + w, y + h))
            return img.convert("RGB")
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    detail = f" Tried: {'; '.join(errors)}." if errors else ""
    raise WaylandUnsupported(CAPTURE_HELP + detail)


# ---------------------------------------------------------------------------
# Input (ydotool >= 1.0)
# ---------------------------------------------------------------------------

# Linux evdev key codes (linux/input-event-codes.h).
_EVDEV: dict[str, int] = {
    "esc": 1, "escape": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8,
    "8": 9, "9": 10, "0": 11, "-": 12, "minus": 12, "=": 13, "equal": 13,
    "backspace": 14, "tab": 15, "q": 16, "w": 17, "e": 18, "r": 19, "t": 20, "y": 21,
    "u": 22, "i": 23, "o": 24, "p": 25, "[": 26, "]": 27, "enter": 28, "return": 28,
    "ctrl": 29, "control": 29, "a": 30, "s": 31, "d": 32, "f": 33, "g": 34, "h": 35,
    "j": 36, "k": 37, "l": 38, ";": 39, "'": 40, "`": 41, "shift": 42, "\\": 43,
    "z": 44, "x": 45, "c": 46, "v": 47, "b": 48, "n": 49, "m": 50, ",": 51, ".": 52,
    "/": 53, "alt": 56, "option": 56, "space": 57, " ": 57, "caps_lock": 58, "capslock": 58,
    "f1": 59, "f2": 60, "f3": 61, "f4": 62, "f5": 63, "f6": 64, "f7": 65, "f8": 66,
    "f9": 67, "f10": 68, "f11": 87, "f12": 88, "home": 102, "up": 103, "pageup": 104,
    "pgup": 104, "left": 105, "right": 106, "end": 107, "down": 108, "pagedown": 109,
    "pgdn": 109, "insert": 110, "delete": 111, "del": 111, "meta": 125, "super": 125,
    "win": 125, "cmd": 125, "command": 125,
}

_BUTTON_CODE = {
    PointerButton.LEFT: 0x00, PointerButton.RIGHT: 0x01, PointerButton.MIDDLE: 0x02,
    PointerButton.X1: 0x03, PointerButton.X2: 0x04,
}
_DOWN, _UP = 0x40, 0x80


def evdev_code(keysym: str) -> int:
    k = keysym.lower()
    if k not in _EVDEV:
        raise ValueError(f"No Wayland/evdev key code for {keysym!r}")
    return _EVDEV[k]


def pointer_commands(event: PointerEvent) -> list[list[str]]:
    """ydotool argv lists that perform ``event``."""
    x, y = int(event.point.x), int(event.point.y)
    move = ["ydotool", "mousemove", "--absolute", "-x", str(x), "-y", str(y)]
    if event.action is PointerAction.MOVE:
        return [move]
    if event.action is PointerAction.SCROLL:
        # ydotool wheel: positive y scrolls up, matching PointerEvent.dy.
        return [move, ["ydotool", "mousemove", "--wheel", "-x", str(int(event.dx)), "-y", str(int(event.dy))]]
    code = _BUTTON_CODE.get(event.button or PointerButton.LEFT, 0x00)
    if event.action is PointerAction.DOWN:
        return [move, ["ydotool", "click", hex(code | _DOWN)]]
    if event.action is PointerAction.UP:
        return [move, ["ydotool", "click", hex(code | _UP)]]
    raise ValueError(f"Unknown pointer action: {event.action!r}")


def key_command(event: KeyEvent) -> list[str]:
    state = 1 if event.action is KeyAction.DOWN else 0
    return ["ydotool", "key", f"{evdev_code(event.keysym)}:{state}"]


def type_command(text: str, *, interval_ms: int = 20) -> list[str]:
    return ["ydotool", "type", "--key-delay", str(max(0, int(interval_ms))), "--", text]


def _run(cmd: list[str]) -> None:
    if not shutil.which(cmd[0]):
        raise WaylandUnsupported(INPUT_HELP)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip()
        hint = " Is ydotoold running?" if "socket" in msg.lower() or "connect" in msg.lower() else ""
        raise RuntimeError(f"{' '.join(cmd[:2])} failed: {msg}{hint}")


def pointer(event: PointerEvent) -> None:
    for cmd in pointer_commands(event):
        _run(cmd)


def key(event: KeyEvent) -> None:
    _run(key_command(event))


def type_text(text: str, *, interval_ms: int = 20) -> None:
    _run(type_command(text, interval_ms=interval_ms))
