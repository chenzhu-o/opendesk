"""Shared helpers for the safety package: home resolution, atomic JSON I/O."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("opendesk.safety")


def resolve_home(home: Any = None, ctx: Any = None) -> Path:
    """Resolve the opendesk home directory.

    Priority: explicit *home* → ``ctx.metadata["opendesk_home"]`` →
    ``$OPENDESK_HOME`` → ``~/.opendesk``.
    """
    if home:
        return Path(str(home)).expanduser()
    meta = getattr(ctx, "metadata", None) if ctx is not None else None
    if isinstance(meta, dict) and meta.get("opendesk_home"):
        return Path(str(meta["opendesk_home"])).expanduser()
    env = os.environ.get("OPENDESK_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".opendesk"


def read_json(path: Path) -> Optional[dict[str, Any]]:
    """Read a JSON object from *path*; ``None`` when missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, data: dict[str, Any], *, mode: int = 0o600) -> None:
    """Atomically write *data* as pretty JSON to *path* (``0600``)."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=False)
            fh.write("\n")
        with contextlib.suppress(OSError):
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def safety_enabled() -> bool:
    """``False`` when ``OPENDESK_SAFETY`` is set to off / 0 / false / disabled."""
    val = os.environ.get("OPENDESK_SAFETY", "").strip().lower()
    return val not in ("off", "0", "false", "no", "disabled")
