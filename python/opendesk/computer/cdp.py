"""Minimal Chrome DevTools Protocol (CDP) client.

Just enough CDP to drive a Chromium-family browser (Chrome, Chromium, Edge,
Brave) from the :class:`~opendesk.tools.browser.BrowserTool`:

* :func:`find_browser` / :func:`launch_browser` — locate a binary and start it
  with ``--remote-debugging-port`` and a *dedicated* profile directory (never
  the user's main profile).
* :class:`DevToolsHTTP` — the ``http://127.0.0.1:<port>/json/*`` endpoints
  (version, list, new, activate, close).
* :class:`CDPConnection` — one WebSocket to one target, request/response
  matching by id plus a buffer of recent events.

``websockets`` is imported lazily so the rest of opendesk works without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Optional


DEFAULT_PORT = 9222


class CDPError(RuntimeError):
    """A CDP command failed, or the browser endpoint could not be reached."""


def _import_websockets():
    try:
        import websockets  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "The browser tool needs the 'websockets' package: "
            "pip install websockets  (or pip install 'opendesk[remote]')"
        ) from exc
    return websockets


# ---------------------------------------------------------------------------
# Browser discovery / launch
# ---------------------------------------------------------------------------

_MAC_APPS = {
    "chrome": ["Google Chrome.app/Contents/MacOS/Google Chrome"],
    "chromium": ["Chromium.app/Contents/MacOS/Chromium"],
    "edge": ["Microsoft Edge.app/Contents/MacOS/Microsoft Edge"],
    "brave": ["Brave Browser.app/Contents/MacOS/Brave Browser"],
}
_LINUX_BINS = {
    "chrome": ["google-chrome", "google-chrome-stable"],
    "chromium": ["chromium", "chromium-browser"],
    "edge": ["microsoft-edge", "microsoft-edge-stable"],
    "brave": ["brave-browser", "brave"],
}
_WIN_PATHS = {
    "chrome": [r"Google\Chrome\Application\chrome.exe"],
    "chromium": [r"Chromium\Application\chrome.exe"],
    "edge": [r"Microsoft\Edge\Application\msedge.exe"],
    "brave": [r"BraveSoftware\Brave-Browser\Application\brave.exe"],
}
BROWSER_ORDER = ("chrome", "chromium", "edge", "brave")


def find_browser(preferred: Optional[str] = None) -> Optional[str]:
    """Return the path of a Chromium-family browser binary, or ``None``.

    *preferred* may be a family name (chrome, chromium, edge, brave) or a full
    path.  ``$OPENDESK_BROWSER`` overrides discovery when *preferred* is unset.
    """
    if preferred is None:
        preferred = os.environ.get("OPENDESK_BROWSER") or None
    if preferred and (os.sep in preferred or "/" in preferred):
        return preferred if os.path.exists(preferred) else None
    families = [preferred.lower()] if preferred else list(BROWSER_ORDER)
    system = platform.system()
    for fam in families:
        if system == "Darwin":
            for root in ("/Applications", str(Path.home() / "Applications")):
                for rel in _MAC_APPS.get(fam, []):
                    p = os.path.join(root, rel)
                    if os.path.exists(p):
                        return p
        elif system == "Windows":
            roots = [os.environ.get(k) for k in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
            for root in filter(None, roots):
                for rel in _WIN_PATHS.get(fam, []):
                    p = os.path.join(root, rel)
                    if os.path.exists(p):
                        return p
        else:
            for name in _LINUX_BINS.get(fam, []):
                p = shutil.which(name)
                if p:
                    return p
    return None


def launch_browser(
    binary: str,
    *,
    port: int,
    profile_dir: Path,
    headless: bool = False,
    url: str = "about:blank",
    extra_args: Optional[list[str]] = None,
) -> subprocess.Popen:
    """Start *binary* with remote debugging on *port* and its own profile."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    args = [
        binary,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
    ]
    if headless:
        args += ["--headless=new", "--disable-gpu", "--window-size=1280,900"]
    args += list(extra_args or [])
    args.append(url)
    return subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


class DevToolsHTTP:
    """Wrapper around the DevTools ``/json`` HTTP endpoints on *host:port*."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
        self.port = port
        self.host = host

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _request_sync(self, path: str, method: str = "GET", timeout: float = 5.0) -> Any:
        req = urllib.request.Request(self.base + path, method=method)
        # Never route loopback DevTools traffic through an http_proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=timeout) as resp:  # noqa: S310 (loopback only)
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise CDPError(f"DevTools {method} {path} → HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise CDPError(
                f"No DevTools endpoint at {self.base} ({exc}). Launch a browser "
                f"with browser action='launch', or start one with "
                f"--remote-debugging-port={self.port}."
            ) from exc
        try:
            return json.loads(body)
        except ValueError:
            return body

    async def request(self, path: str, method: str = "GET", timeout: float = 5.0) -> Any:
        return await asyncio.to_thread(self._request_sync, path, method, timeout)

    async def version(self) -> dict[str, Any]:
        return await self.request("/json/version")

    async def targets(self) -> list[dict[str, Any]]:
        """All *page* targets (tabs), in the browser's order."""
        items = await self.request("/json/list")
        if not isinstance(items, list):
            raise CDPError(f"Unexpected /json/list payload: {items!r}")
        return [t for t in items if t.get("type") == "page"]

    async def new_tab(self, url: str = "about:blank") -> dict[str, Any]:
        path = "/json/new?" + urllib.parse.quote(url, safe=":/?&=%")
        try:
            # Chrome 111+ requires PUT; older builds only accept GET.
            return await self.request(path, method="PUT")
        except CDPError:
            return await self.request(path, method="GET")

    async def activate(self, target_id: str) -> None:
        await self.request(f"/json/activate/{target_id}")

    async def close(self, target_id: str) -> None:
        await self.request(f"/json/close/{target_id}")

    async def wait_ready(self, timeout: float = 20.0) -> dict[str, Any]:
        """Poll ``/json/version`` until the endpoint answers or *timeout* passes."""
        deadline = time.monotonic() + timeout
        last: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                return await self.request("/json/version", timeout=2.0)
            except CDPError as exc:
                last = exc
                await asyncio.sleep(0.25)
        raise CDPError(f"Browser did not open DevTools on port {self.port} within {timeout:g}s: {last}")


# ---------------------------------------------------------------------------
# WebSocket connection to a single target
# ---------------------------------------------------------------------------


class CDPConnection:
    """One CDP WebSocket session (normally attached to one page target)."""

    def __init__(self, ws_url: str) -> None:
        self.ws_url = ws_url
        self._ws: Any = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader: Optional[asyncio.Task] = None
        self.events: deque[dict[str, Any]] = deque(maxlen=200)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def connect(self, timeout: float = 10.0) -> "CDPConnection":
        websockets = _import_websockets()
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(self.ws_url, max_size=None, ping_interval=None),
                timeout=timeout,
            )
        except Exception as exc:
            raise CDPError(f"Could not open DevTools WebSocket {self.ws_url}: {exc}") from exc
        self._reader = asyncio.create_task(self._read_loop())
        return self

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                mid = msg.get("id")
                if mid is not None and mid in self._pending:
                    fut = self._pending.pop(mid)
                    if not fut.done():
                        fut.set_result(msg)
                elif "method" in msg:
                    self.events.append(msg)
        except Exception:
            pass
        finally:
            self._closed = True
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(CDPError("DevTools connection closed"))
            self._pending.clear()

    async def send(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Send one CDP command and return its ``result`` dict."""
        if self._ws is None or self._closed:
            raise CDPError("DevTools connection is closed")
        self._next_id += 1
        mid = self._next_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        await self._ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        try:
            msg = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(mid, None)
            raise CDPError(f"{method} timed out after {timeout:g}s") from exc
        if "error" in msg:
            err = msg["error"] or {}
            raise CDPError(f"{method}: {err.get('message', err)}")
        return msg.get("result") or {}

    async def wait_event(self, method: str, timeout: float) -> Optional[dict[str, Any]]:
        """Wait for (and consume) the next buffered event named *method*.

        Returns ``None`` on timeout.  Call ``events.clear()`` before the
        triggering command to ignore stale events.
        """
        deadline = time.monotonic() + timeout
        while True:
            for ev in list(self.events):
                if ev.get("method") == method:
                    with contextlib.suppress(ValueError):
                        self.events.remove(ev)
                    return ev
            if self._closed or time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.05)

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(BaseException):
                await self._reader
