"""opendesk.memory — screen memory: a local, searchable desktop history.

A background loop captures the screen when the frontmost app or window
changes (or on a timer), reads the window's text from the accessibility
tree or with local OCR, redacts secrets, and stores the text plus a small
encrypted thumbnail in ``~/.opendesk/memory``.  The ``memory`` tool lets an
agent search that history ("what error did I see in the terminal on
Tuesday?").  Nothing leaves the machine.

Public surface::

    from opendesk.memory import MemoryStore, MemoryConfig, ScreenMemoryRecorder
"""

from opendesk.memory.config import (
    MemoryConfig,
    clear_pause,
    daemon_alive,
    get_pause,
    is_paused,
    load_config,
    memory_dir,
    save_config,
    set_pause,
)
from opendesk.memory.crypto import MemoryCipher, load_or_create_key, read_keyinfo
from opendesk.memory.recorder import FrontmostInfo, ScreenMemoryRecorder, start_daemon
from opendesk.memory.redact import redact, redact_text
from opendesk.memory.store import TEXT_SOURCES, Frame, MemoryStore, StoreStats
from opendesk.memory.timeparse import parse_range, parse_when

__all__ = [
    "Frame",
    "FrontmostInfo",
    "MemoryCipher",
    "MemoryConfig",
    "MemoryStore",
    "ScreenMemoryRecorder",
    "StoreStats",
    "TEXT_SOURCES",
    "clear_pause",
    "daemon_alive",
    "get_pause",
    "is_paused",
    "load_config",
    "load_or_create_key",
    "memory_dir",
    "parse_range",
    "parse_when",
    "read_keyinfo",
    "redact",
    "redact_text",
    "save_config",
    "set_pause",
    "start_daemon",
]
