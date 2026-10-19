"""Hash chaining for JSONL audit logs.

Every chained line carries ``prev`` (the previous line's ``hash``; 64 zeros
for the first chained line in a file) and ``hash`` = SHA-256 over the
canonical JSON of the entry *without* ``hash`` (but including ``prev``).

Canonical JSON: keys sorted, no whitespace, UTF-8 (no ASCII escaping),
integral floats written as integers.  The JS SDK produces byte-identical
canonical forms, so either SDK can verify the other's logs.

Lines without a ``hash`` are legacy (pre-chaining) entries.  They are
accepted only *before* the first chained line; an unchained line after the
chain has started is reported as tampering.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

GENESIS = "0" * 64


def _normalise(obj: Any) -> Any:
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj) and obj.is_integer() and abs(obj) < 2**53:
            return int(obj)
        return obj
    if isinstance(obj, dict):
        return {str(k): _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, (str, int)):
        return obj
    return str(obj)


def canonical_json(obj: Any) -> str:
    return json.dumps(
        _normalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def entry_hash(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def seal(entry: dict[str, Any], prev: str) -> dict[str, Any]:
    """Return a copy of *entry* with ``prev`` and ``hash`` set."""
    out = {k: v for k, v in entry.items() if k not in ("prev", "hash")}
    out["prev"] = prev
    out["hash"] = entry_hash(out)
    return out


def last_hash(path: Path) -> str:
    """Hash of the last chained line in *path* (``GENESIS`` if none)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block = 65536
            pos = size
            buf = b""
            while pos > 0:
                read = min(block, pos)
                pos -= read
                fh.seek(pos)
                buf = fh.read(read) + buf
                lines = buf.split(b"\n")
                # The first element may be partial unless we reached the start.
                candidates = lines if pos == 0 else lines[1:]
                for raw in reversed(candidates):
                    raw = raw.strip()
                    if not raw:
                        continue
                    with contextlib.suppress(ValueError):
                        obj = json.loads(raw)
                        if isinstance(obj, dict) and isinstance(obj.get("hash"), str):
                            return obj["hash"]
                if pos == 0:
                    break
    except FileNotFoundError:
        return GENESIS
    return GENESIS


@contextlib.contextmanager
def _locked(fh: Any):
    try:
        import fcntl  # POSIX only
    except ImportError:  # pragma: no cover - Windows
        yield
        return
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def append_chained(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Append *entry* to *path* with chaining; safe across processes (POSIX)."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    existed = path.exists()
    with open(path, "a", encoding="utf-8") as fh:
        if not existed:
            with contextlib.suppress(OSError):
                os.chmod(path, 0o600)
        with _locked(fh):
            sealed = seal(entry, last_hash(path))
            fh.write(json.dumps(sealed, sort_keys=True, ensure_ascii=False, default=str) + "\n")
            fh.flush()
    return sealed


@dataclass
class VerifyResult:
    path: Path
    ok: bool
    chained: int = 0
    legacy: int = 0
    error: str = ""
    line: Optional[int] = None

    def describe(self) -> str:
        if self.ok:
            extra = f", {self.legacy} legacy unchained" if self.legacy else ""
            return f"OK    {self.path.name}: {self.chained} chained entries{extra}"
        return f"FAIL  {self.path.name}: line {self.line}: {self.error}"


def verify_file(path: Path) -> VerifyResult:
    """Verify the hash chain of one JSONL file."""
    res = VerifyResult(path=path, ok=True)
    prev = GENESIS
    started = False
    try:
        fh = open(path, encoding="utf-8")
    except OSError as exc:
        return VerifyResult(path, False, error=str(exc))
    with fh:
        for n, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                return _fail(res, n, "line is not valid JSON")
            if not isinstance(obj, dict):
                return _fail(res, n, "line is not a JSON object")
            if "hash" not in obj:
                if started:
                    return _fail(res, n, "unchained entry after the chain started (inserted line?)")
                res.legacy += 1
                continue
            started = True
            if obj.get("prev") != prev:
                return _fail(res, n, "prev does not match the previous entry's hash (deleted or reordered line?)")
            if entry_hash(obj) != obj["hash"]:
                return _fail(res, n, "hash mismatch (entry was modified)")
            prev = obj["hash"]
            res.chained += 1
    return res


def _fail(res: VerifyResult, line: int, msg: str) -> VerifyResult:
    res.ok = False
    res.line = line
    res.error = msg
    return res
