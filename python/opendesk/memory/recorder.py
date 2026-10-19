"""Background capture loop for screen memory.

Two triggers (``config.trigger``):

* ``events`` (default) — every ``poll_seconds`` the recorder cheaply probes
  the frontmost app + window title.  It captures when that changes
  (debounced by ``min_interval_seconds``), and otherwise falls back to one
  attempt every ``interval_seconds`` which is kept only if the screen
  actually changed.  Switching apps is recorded within about a second;
  an idle screen costs one probe per poll and nothing else.
* ``interval`` — the v1 behaviour: one attempt every ``interval_seconds``.

Each capture attempt (:meth:`ScreenMemoryRecorder.tick`):

1. Checks the shared pause flag (hotkey / ``memory_manage(action=pause)`` / CLI).
2. Reads the frontmost app + window title and consults the deny list.
3. Grabs the screen through the :class:`~opendesk.computer.Computer`.
4. Skips the frame when it is visually identical to the previous one
   (except right after an app switch).
5. Reads the focused window's text from the accessibility tree when that
   is available (``ax``); falls back to local OCR (``ocr``).
6. Redacts secrets, builds a JPEG thumbnail, and writes both to the
   :class:`~opendesk.memory.store.MemoryStore` (encrypting as configured).
7. Periodically enforces retention and the storage cap.

Nothing leaves the machine: capture, OCR, and storage are all local.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import io
import logging
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union

from opendesk.memory.config import (
    MemoryConfig,
    clear_daemon_state,
    clear_pause,
    get_pause,
    load_config,
    set_pause,
    write_daemon_state,
)
from opendesk.memory.redact import redact
from opendesk.memory.store import MemoryStore

log = logging.getLogger("opendesk.memory")

OCRFunc = Callable[[bytes], str]
WindowTextFunc = Callable[..., Any]
"""``window_text(pid=None, max_chars=20000) -> str | None`` (sync or async)."""

AX_MAX_CHARS = 20000
AX_TIMEOUT_SECONDS = 5.0
CONFIG_RELOAD_SECONDS = 10.0
HEARTBEAT_SECONDS = 10.0


@dataclass
class FrontmostInfo:
    """What the frontmost probe reports."""

    app: str = ""
    title: str = ""
    pid: Optional[int] = None


FrontmostFunc = Callable[[], Union[Awaitable[Any], Any]]


# ---------------------------------------------------------------------------
# Image helpers (pure functions, unit-testable)
# ---------------------------------------------------------------------------


def make_thumbnail(png_bytes: bytes, width: int = 480, quality: int = 60) -> tuple[bytes, int, int]:
    """Downscale a PNG to a JPEG thumbnail.  Returns ``(jpeg, w, h)``."""
    from PIL import Image  # type: ignore[import-not-found]

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if img.width > width:
        ratio = width / img.width
        img = img.resize((width, max(1, int(img.height * ratio))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), img.width, img.height


def frame_signature(png_bytes: bytes, size: int = 16) -> list[int]:
    """Tiny grayscale fingerprint used for duplicate detection."""
    from PIL import Image  # type: ignore[import-not-found]

    img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((size, size), Image.BILINEAR)
    return list(img.getdata())


def signature_delta(a: Optional[list[int]], b: Optional[list[int]]) -> float:
    """Mean absolute difference between two signatures, normalised to 0–1."""
    if not a or not b or len(a) != len(b):
        return 1.0
    total = sum(abs(x - y) for x, y in zip(a, b))
    return total / (255.0 * len(a))


def _default_ocr(png_bytes: bytes) -> str:
    from opendesk.computer.ocr import ocr_image
    return ocr_image(png_bytes)


def _ocr_failed(text: str) -> bool:
    return (
        text.startswith("OCR not available")
        or text.startswith("OCR error")
        or text.startswith("pytesseract error")
    )


# ---------------------------------------------------------------------------
# Native probes (guarded — the macOS native module is optional)
# ---------------------------------------------------------------------------


def _native(name: str) -> Optional[Callable[..., Any]]:
    """``opendesk.computer.macos_native.<name>`` if that module exists."""
    if platform.system() != "Darwin":
        return None
    try:
        # import_module consults sys.modules first, so a stub installed there
        # wins over an already-imported package attribute.
        macos_native = importlib.import_module("opendesk.computer.macos_native")
    except Exception:
        return None
    fn = getattr(macos_native, name, None)
    return fn if callable(fn) else None


def _coerce_front(obj: Any) -> FrontmostInfo:
    """Accept a dict, tuple, or object from whichever probe answered."""
    if obj is None:
        return FrontmostInfo()
    if isinstance(obj, FrontmostInfo):
        return obj
    if isinstance(obj, dict):
        get = obj.get
    elif isinstance(obj, (tuple, list)):
        vals = list(obj) + [None, None, None]
        return FrontmostInfo(str(vals[0] or ""), str(vals[1] or ""), _int_or_none(vals[2]))
    else:
        def get(k: str, d: Any = None) -> Any:
            return getattr(obj, k, d)
    app = get("app") or get("app_name") or get("name") or get("application") or ""
    title = get("title") or get("window_title") or ""
    pid = get("pid") if get("pid") is not None else get("process_id")
    return FrontmostInfo(str(app).strip(), str(title).strip(), _int_or_none(pid))


def title_key(title: str) -> str:
    """Comparison key for window titles: letters only.

    Spinners (``◐ ◑``), counters (``(3) Slack``), progress (``45%``) and
    clocks change a title every second without changing what is on screen;
    they must not count as a window change.  The idle fallback still
    captures such windows when their pixels change."""
    from opendesk.memory.crypto import normalize
    letters = "".join(ch if (ch.isalpha() or ch.isspace()) else " " for ch in normalize(title or ""))
    return " ".join(letters.split())


def _int_or_none(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None and str(v).strip() != "" else None
    except (TypeError, ValueError):
        return None


_OSASCRIPT_FRONT = """tell application "System Events"
set p to first process whose frontmost is true
set n to name of p
set i to unix id of p
set t to ""
try
set t to name of front window of p
end try
return n & linefeed & i & linefeed & t
end tell"""


def _osascript_frontmost() -> Optional[FrontmostInfo]:
    """App, pid, and window title in a single ``osascript`` call (macOS)."""
    try:
        r = subprocess.run(
            ["osascript", "-e", _OSASCRIPT_FRONT], capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    parts = r.stdout.rstrip("\n").split("\n", 2) + ["", ""]
    return FrontmostInfo(parts[0].strip(), parts[2].strip(), _int_or_none(parts[1]))


async def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a sync-or-async callable without blocking the loop."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
    if inspect.isawaitable(result):
        result = await result
    return result


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class ScreenMemoryRecorder:
    """Drives capture.  :meth:`poll` is one event-trigger step, :meth:`tick`
    one capture attempt, :meth:`run` loops forever."""

    def __init__(
        self,
        home: Optional[Path] = None,
        *,
        computer: Any = None,
        config: Optional[MemoryConfig] = None,
        store: Optional[MemoryStore] = None,
        ocr: Optional[OCRFunc] = None,
        interval_override: Optional[float] = None,
        frontmost: Optional[FrontmostFunc] = None,
        window_text: Optional[WindowTextFunc] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.home = home
        self.config = config or load_config(home)
        self._interval_override = float(interval_override) if interval_override else None
        if self._interval_override:
            self.config.interval_seconds = self._interval_override
        self.store = store or MemoryStore(home, config=self.config)
        # Native probes (accessibility text, native frontmost) only make
        # sense for the real local screen — never for an injected Computer.
        self._local = computer is None or type(computer).__name__ == "LocalComputer"
        self._computer = computer
        self._ocr: OCRFunc = ocr or _default_ocr
        self._frontmost_fn = frontmost
        self._window_text_fn = window_text
        self._window_text_resolved = window_text is not None
        self._clock: Callable[[], float] = clock or time.monotonic
        self._last_sig: Optional[list[int]] = None
        self._ticks = 0
        self._captured = 0
        self._skipped = 0
        self._redactions = 0
        self._sources: dict[str, int] = {"ax": 0, "ocr": 0, "none": 0}
        self._reasons: dict[str, int] = {}
        self._last_status = "idle"
        self._last_reason = ""
        self._last_source = ""
        self._last_key: Optional[tuple[str, str]] = None
        self._last_attempt: Optional[float] = None
        self._last_reload = self._clock()
        self._last_heartbeat = 0.0
        self._ocr_warned = False
        self._stop = asyncio.Event()
        self._hotkey_listener: Any = None

    # -- accessors -----------------------------------------------------

    @property
    def computer(self) -> Any:
        if self._computer is None:
            from opendesk.computer.local import LocalComputer
            self._computer = LocalComputer()
        return self._computer

    @property
    def captured(self) -> int:
        return self._captured

    @property
    def skipped(self) -> int:
        return self._skipped

    @property
    def last_status(self) -> str:
        return self._last_status

    @property
    def text_sources(self) -> dict[str, int]:
        """Frames captured this run, per text source."""
        return dict(self._sources)

    @property
    def redactions(self) -> int:
        return self._redactions

    def reload_config(self) -> None:
        """Pick up deny-list / interval / encryption edits made by other
        processes."""
        fresh = load_config(self.home)
        # Keep a CLI interval override sticky across reloads.
        if self._interval_override:
            fresh.interval_seconds = self._interval_override
        self.config = fresh
        self.store.apply_config(fresh)
        self._last_reload = self._clock()

    # -- event trigger -------------------------------------------------

    def trigger_reason(self, front: FrontmostInfo, now: float) -> Optional[str]:
        """Why a capture should happen now, or None.  Pure decision logic:

        * ``start``         — first attempt of the run;
        * ``app_change``    — the frontmost app changed;
        * ``title_change``  — same app, different window/tab title (compared
          by :func:`title_key`, so spinners and counters don't count);
        * ``interval``      — nothing changed for ``interval_seconds``.

        Changes are debounced: nothing fires sooner than
        ``min_interval_seconds`` after the previous attempt.
        """
        if self._last_attempt is None or self._last_key is None:
            return "start"
        since = now - self._last_attempt
        key = (front.app, title_key(front.title))
        if key != self._last_key:
            if since < self.config.min_interval_seconds:
                return None
            return "app_change" if front.app != self._last_key[0] else "title_change"
        if since >= self.config.interval_seconds:
            return "interval"
        return None

    async def poll(self) -> Optional[str]:
        """One step of the ``events`` trigger.  Probes the frontmost window
        and captures if :meth:`trigger_reason` says so.  Returns the tick
        status, ``"paused"``, or None when nothing was due."""
        now = self._clock()
        if now - self._last_reload >= CONFIG_RELOAD_SECONDS:
            self.reload_config()
        if get_pause(self.home) is not None:
            self._last_status = "paused"
            self._heartbeat()
            return "paused"
        front = await self._probe()
        reason = self.trigger_reason(front, now)
        if reason is None:
            self._heartbeat()
            return None
        return await self.tick(reason=reason, front=front)

    # -- one tick --------------------------------------------------------

    async def tick(self, *, reason: str = "manual", front: Optional[FrontmostInfo] = None) -> str:
        """Perform one capture attempt.  Returns a short status string."""
        self._ticks += 1
        if self._ticks % 10 == 0:
            self.reload_config()

        pause = get_pause(self.home)
        if pause is not None:
            return self._done("paused")

        if front is None:
            front = await self._probe()
        app, title = front.app, front.title
        self._last_key = (app, title_key(title))
        self._last_attempt = self._clock()
        self._last_reason = reason
        if self.config.is_denied(app, title):
            return self._done(f"denied:{app or title}")

        try:
            pixmap = await self.computer.capture()
        except Exception as exc:
            log.warning("capture failed: %s", exc)
            return self._done(f"error:capture:{exc}")

        png = pixmap.data
        loop = asyncio.get_running_loop()
        try:
            sig = await loop.run_in_executor(None, frame_signature, png)
        except Exception as exc:
            log.warning("signature failed: %s", exc)
            sig = None
        # A different app is a different context even when few pixels moved
        # (a small window on a large display), so app switches skip the
        # visual dedupe; everything else must actually change the screen.
        if sig is not None and self._last_sig is not None and reason != "app_change":
            if signature_delta(sig, self._last_sig) < self.config.dedupe_threshold:
                return self._done("duplicate")
        self._last_sig = sig

        text, source = await self._read_text(png, front.pid)
        if self.config.redact:
            text, n_text = redact(text)
            title, n_title = redact(title)
            self._redactions += n_text + n_title

        try:
            thumb, tw, th = await loop.run_in_executor(
                None, make_thumbnail, png,
                self.config.thumbnail_width, self.config.thumbnail_quality,
            )
        except Exception as exc:
            log.warning("thumbnail failed: %s", exc)
            thumb, tw, th = None, 0, 0

        frame = await loop.run_in_executor(None, lambda: self.store.add(
            ts=time.time(), app=app, title=title, text=text,
            thumb=thumb, width=tw, height=th, text_source=source,
        ))
        self._captured += 1
        self._sources[source] = self._sources.get(source, 0) + 1
        self._reasons[reason] = self._reasons.get(reason, 0) + 1
        self._last_source = source
        log.debug(
            "captured frame %s (%s) app=%r title=%r source=%s chars=%d",
            frame.id, reason, app, title, source, len(text),
        )

        if self._captured % 20 == 1:
            await loop.run_in_executor(None, self.housekeep)
        return self._done("captured")

    async def _read_text(self, png: bytes, pid: Optional[int]) -> tuple[str, str]:
        """``(text, source)``: accessibility text when there is enough of
        it, else OCR.  ``ocr_backend: always`` appends OCR to AX text."""
        ax = await self._ax_text(pid)
        if ax and len(ax.strip()) >= max(1, int(self.config.ax_min_chars)):
            text = ax.strip()
            if str(self.config.ocr_backend).lower() == "always":
                extra = await self._run_ocr(png)
                if extra and extra not in text:
                    text = f"{text}\n{extra}"
            return text, "ax"
        text = await self._run_ocr(png)
        return text, ("ocr" if text else "none")

    async def _run_ocr(self, png: bytes) -> str:
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, self._ocr, png)
        except Exception as exc:
            text = f"OCR error: {exc}"
        text = text or ""
        if _ocr_failed(text):
            if not self._ocr_warned:
                log.warning("OCR unavailable — frames will be stored without text: %s", text.splitlines()[0])
                self._ocr_warned = True
            return ""
        if text == "(no text detected)":
            return ""
        return text

    def _resolve_window_text(self) -> Optional[WindowTextFunc]:
        if not self._window_text_resolved:
            self._window_text_resolved = True
            self._window_text_fn = _native("window_text") if self._local else None
        return self._window_text_fn

    async def _ax_text(self, pid: Optional[int]) -> Optional[str]:
        fn = self._resolve_window_text()
        if fn is None:
            return None
        try:
            result = await asyncio.wait_for(
                _call(fn, pid=pid, max_chars=AX_MAX_CHARS), timeout=AX_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            log.debug("window_text failed: %s", exc)
            return None
        return result if isinstance(result, str) else None

    def housekeep(self) -> dict[str, int]:
        """Apply retention + storage cap.  Safe to call from any thread."""
        expired = self.store.enforce_retention(self.config.retention_days)
        capped = self.store.enforce_cap(self.config.storage_cap_mb * 1024 * 1024)
        if expired or capped:
            log.info("housekeeping: removed %d expired, %d over-cap frames", expired, capped)
        return {"expired": expired, "capped": capped}

    def _state_fields(self) -> dict[str, Any]:
        return {
            "status": self._last_status,
            "captured": self._captured,
            "skipped": self._skipped,
            "interval": self.config.interval_seconds,
            "trigger": self.config.trigger,
            "poll_seconds": self.config.poll_seconds,
            "min_interval_seconds": self.config.min_interval_seconds,
            "last_reason": self._last_reason,
            "last_text_source": self._last_source,
            "text_sources": dict(self._sources),
            "reasons": dict(self._reasons),
            "redactions": self._redactions,
        }

    def _done(self, status: str) -> str:
        self._last_status = status
        if status != "captured":
            self._skipped += 1
        write_daemon_state(self.home, **self._state_fields())
        self._last_heartbeat = time.monotonic()
        return status

    def _heartbeat(self) -> None:
        """Throttled daemon.json refresh for polls that don't capture."""
        if time.monotonic() - self._last_heartbeat >= HEARTBEAT_SECONDS:
            write_daemon_state(self.home, **self._state_fields())
            self._last_heartbeat = time.monotonic()

    async def _probe(self) -> FrontmostInfo:
        """Frontmost app/title/pid: injected probe → native (macOS) →
        one ``osascript`` call (macOS) → ``computer.focused_window()``."""
        if self._frontmost_fn is not None:
            try:
                return self._clean(_coerce_front(await _call(self._frontmost_fn)))
            except Exception:
                return FrontmostInfo()
        if self._local:
            native = _native("frontmost")
            if native is not None:
                try:
                    info = _coerce_front(await _call(native))
                    if info.app or info.title:
                        return self._clean(info)
                except Exception as exc:
                    log.debug("native frontmost failed: %s", exc)
            if platform.system() == "Darwin":
                info2 = await _call(_osascript_frontmost)
                if info2 is not None and info2.app:
                    return self._clean(info2)
        try:
            win = await self.computer.focused_window()
        except Exception:
            return FrontmostInfo()
        if win is None:
            return FrontmostInfo()
        return self._clean(FrontmostInfo(
            (getattr(win, "app_name", "") or "").strip(),
            (getattr(win, "title", "") or "").strip(),
            _int_or_none(getattr(win, "pid", None)),
        ))

    @staticmethod
    def _clean(info: FrontmostInfo) -> FrontmostInfo:
        app = (info.app or "").strip()
        title = (info.title or "").strip()
        return FrontmostInfo(app, "" if title == app else title, info.pid)

    # -- loop --------------------------------------------------------------

    async def run(self) -> None:
        """Capture until :meth:`stop` is called."""
        self._last_status = "starting"
        write_daemon_state(self.home, **self._state_fields())
        self._start_hotkey()
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                events = self.config.trigger != "interval"
                try:
                    if events:
                        await self.poll()
                    else:
                        await self.tick(reason="interval")
                except Exception as exc:  # never let one bad tick kill the loop
                    log.exception("tick failed: %s", exc)
                    self._done(f"error:{exc}")
                elapsed = time.monotonic() - started
                if events:
                    delay = max(0.2, self.config.poll_seconds - elapsed)
                else:
                    delay = max(1.0, self.config.interval_seconds - elapsed)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._stop_hotkey()
            clear_daemon_state(self.home)
            self.store.close()

    def stop(self) -> None:
        self._stop.set()

    # -- pause hotkey --------------------------------------------------------

    def toggle_pause(self) -> bool:
        """Flip the pause flag.  Returns the new paused state."""
        if get_pause(self.home) is not None:
            clear_pause(self.home)
            log.info("screen memory resumed (hotkey)")
            print(f"[{_now()}] screen memory resumed", flush=True)
            return False
        set_pause(self.home, reason="hotkey")
        log.info("screen memory paused (hotkey)")
        print(f"[{_now()}] screen memory paused — press the hotkey again to resume", flush=True)
        return True

    def _start_hotkey(self) -> None:
        combo = (self.config.pause_hotkey or "").strip()
        if not combo:
            return
        try:
            from pynput import keyboard  # type: ignore[import-not-found]
        except ImportError:
            log.warning(
                "pynput not installed — pause hotkey disabled. "
                "pip install 'opendesk[memory]' to enable it."
            )
            return
        try:
            self._hotkey_listener = keyboard.GlobalHotKeys({combo: self.toggle_pause})
            self._hotkey_listener.daemon = True
            self._hotkey_listener.start()
            log.info("pause hotkey armed: %s", combo)
        except Exception as exc:
            log.warning("could not register pause hotkey %r: %s", combo, exc)
            self._hotkey_listener = None

    def _stop_hotkey(self) -> None:
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:
                pass
            self._hotkey_listener = None


# ---------------------------------------------------------------------------
# Daemon entry point
# ---------------------------------------------------------------------------


def start_daemon(home: Optional[Path] = None, *, interval: Optional[float] = None) -> None:
    """Blocking entry used by ``opendesk memory start``."""
    from opendesk.computer.ocr import available_backend, warm_up
    from opendesk.memory.crypto import crypto_available

    config = load_config(home)
    backend = available_backend()
    if backend == "macos-vision":
        print("Preparing the macOS Vision OCR helper (one-time compile, may take a minute)…", flush=True)
        warm_up()
    if backend is None:
        print(
            "WARNING: no OCR backend available — frames will be stored without text.\n"
            "  pip install pytesseract  (and install the tesseract binary)",
            file=sys.stderr,
        )
    if (config.encrypt_thumbnails or config.encrypt_text) and not crypto_available():
        print(
            "WARNING: encryption is enabled but the 'cryptography' package is missing —\n"
            "  frames will be stored unencrypted.  pip install cryptography",
            file=sys.stderr,
        )

    recorder = ScreenMemoryRecorder(home, config=config, interval_override=interval)
    config = recorder.config
    store_dir = recorder.store.dir
    ax = "accessibility text first, OCR fallback" if _native("window_text") else "OCR"
    if config.trigger == "interval":
        trigger = f"interval — every {config.interval_seconds:g}s"
    else:
        trigger = (
            f"events — app/window changes (poll {config.poll_seconds:g}s, "
            f"min {config.min_interval_seconds:g}s apart), idle fallback every {config.interval_seconds:g}s"
        )
    enc = []
    if config.encrypt_thumbnails:
        enc.append("thumbnails")
    if config.encrypt_text:
        enc.append("text")
    print("opendesk screen memory")
    print(f"  store:      {store_dir}")
    print(f"  trigger:    {trigger}")
    print(f"  text:       {ax}; redaction {'on' if config.redact else 'OFF'}")
    print(f"  encryption: {', '.join(enc) if enc else 'off'}")
    print(f"  cap:        {config.storage_cap_mb} MB, retention {config.retention_days} days")
    print(f"  deny list:  {', '.join(config.deny_apps) or '(empty)'}")
    print(f"  OCR:        {backend or 'unavailable'}")
    print(f"  pause key:  {config.pause_hotkey or '(disabled)'}")
    if get_pause(home) is not None:
        print("  state:      PAUSED (resume with `opendesk memory resume` or the hotkey)")
    print("Ctrl-C to stop.\n")

    async def _main() -> None:
        loop = asyncio.get_running_loop()

        def _request_stop(*_: Any) -> None:
            recorder.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except (NotImplementedError, RuntimeError):
                signal.signal(sig, _request_stop)
        await recorder.run()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
    print(f"\nStopped. {recorder.captured} frames captured this run.")


def _now() -> str:
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")
