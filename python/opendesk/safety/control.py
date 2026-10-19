"""Kill switch and take-over.

``<home>/control.json``::

    {"mode": "running" | "stopped" | "takeover", "since": 1760000000.0, "reason": "..."}

While ``stopped`` or ``takeover`` every guarded tool call (MCP, SDK adapters,
inbound remote calls under ``--approve policy``) is refused with an
actionable message.  The file is re-read on every call, so ``opendesk stop``
in another terminal takes effect immediately.

An optional global kill hotkey (``kill_hotkey`` in ``policy.json``, pynput
syntax such as ``"<ctrl>+<alt>+<shift>+k"``) can be armed inside the MCP
server process; pressing it switches to ``stopped``.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from opendesk.safety._home import read_json, resolve_home, write_json

log = logging.getLogger("opendesk.safety.control")

CONTROL_FILE = "control.json"
MODES = ("running", "stopped", "takeover")


@dataclass(frozen=True)
class ControlState:
    mode: str = "running"
    since: Optional[float] = None
    reason: str = ""

    @property
    def running(self) -> bool:
        return self.mode == "running"

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "since": self.since, "reason": self.reason}


def control_path(home: Any = None) -> Path:
    return resolve_home(home) / CONTROL_FILE


def read_control(home: Any = None) -> ControlState:
    """Current control state (``running`` when the file is missing)."""
    data = read_json(control_path(home))
    if not data:
        return ControlState()
    mode = str(data.get("mode") or "running").lower()
    if mode not in MODES:
        # Unknown / corrupted mode: fail closed.
        log.warning("control.json has unknown mode %r — treating as stopped", mode)
        mode = "stopped"
    since = data.get("since")
    return ControlState(
        mode=mode,
        since=float(since) if isinstance(since, (int, float)) else None,
        reason=str(data.get("reason") or ""),
    )


def set_mode(mode: str, home: Any = None, reason: str = "") -> ControlState:
    """Persist a new control *mode*."""
    mode = mode.lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    state = ControlState(mode=mode, since=time.time(), reason=reason)
    write_json(control_path(home), state.to_dict())
    return state


def _since(state: ControlState) -> str:
    if not state.since:
        return ""
    return " since " + _dt.datetime.fromtimestamp(state.since).strftime("%Y-%m-%d %H:%M:%S")


def refusal_message(state: ControlState) -> Optional[str]:
    """Actionable error for a non-running state, or ``None`` when running."""
    if state.mode == "running":
        return None
    why = f" ({state.reason})" if state.reason else ""
    if state.mode == "takeover":
        return (
            f"User has taken control of this machine{_since(state)}{why}. "
            "Do not retry. Ask them to run `opendesk handback` when they want "
            "you to continue."
        )
    return (
        f"OpenDesk is stopped — the kill switch is engaged{_since(state)}{why}. "
        "All tool calls are refused. Ask the user to run `opendesk resume` "
        "when it is safe to continue."
    )


def check_control(home: Any = None) -> Optional[str]:
    """Return the refusal message if tool calls must be refused, else ``None``."""
    return refusal_message(read_control(home))


# ---------------------------------------------------------------------------
# Kill hotkey
# ---------------------------------------------------------------------------


def start_kill_hotkey(home: Any = None, hotkey: Optional[str] = None) -> Any:
    """Arm a global hotkey that engages the kill switch.

    *hotkey* defaults to ``kill_hotkey`` from ``policy.json``.  Returns the
    pynput listener, or ``None`` when no hotkey is configured, pynput is
    missing, or the listener can't start (e.g. no Input Monitoring
    permission).  Never raises.
    """
    if hotkey is None:
        from opendesk.safety.policy import load_policy
        hotkey = load_policy(home).kill_hotkey
    if not hotkey:
        return None
    try:
        from pynput import keyboard  # type: ignore[import-not-found]
    except Exception:
        return None

    resolved = resolve_home(home)

    def _fire() -> None:
        try:
            set_mode("stopped", resolved, reason=f"kill hotkey {hotkey}")
            log.warning("kill hotkey pressed — opendesk stopped")
        except Exception as exc:  # pragma: no cover - best effort
            log.warning("kill hotkey failed: %s", exc)

    try:
        listener = keyboard.GlobalHotKeys({hotkey: _fire})
        listener.daemon = True
        listener.start()
        return listener
    except Exception as exc:
        log.debug("kill hotkey unavailable: %s", exc)
        return None
