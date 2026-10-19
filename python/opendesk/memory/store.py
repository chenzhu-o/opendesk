"""Local, searchable index of screen captures.

One SQLite database (``index.db``) holds a row per kept frame — timestamp,
frontmost app, window title, text (accessibility or OCR), where that text
came from, and the path of a JPEG thumbnail on disk.  Full-text search uses
SQLite's FTS5 when the interpreter's SQLite was built with it (virtually
always), with a plain ``LIKE`` fallback.

Encryption at rest (see ``crypto.py``):

* thumbnails (``encrypt_thumbnails``, default on) are written as
  ``<id>.jpg.enc`` and decrypted transparently by :meth:`MemoryStore.thumbnail`;
* text (``encrypt_text``, opt-in) is stored AES-GCM encrypted in
  ``text_enc`` while the FTS-indexed ``text`` column holds a *blind index*:
  sorted, de-duplicated HMAC tokens of the words.  Exact-term search keeps
  working (query terms are tokenised and HMAC'd the same way); prefix
  search does not.  Snippets are computed after decryption.  App names and
  window titles stay plaintext so app filters work.

Old plaintext frames and new encrypted frames can live in the same store;
:meth:`MemoryStore.reindex` converts between the two.

The store is deliberately boring: no vector embeddings, no cloud, no
background threads.  Everything an agent needs to answer "what did I see
on Tuesday" is a single indexed query away.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from opendesk.memory import crypto
from opendesk.memory.config import MemoryConfig, load_config, memory_dir
from opendesk.memory.crypto import MemoryCipher, normalize, tokenize

log = logging.getLogger("opendesk.memory")

_SCHEMA_VERSION = 2

#: Where a frame's text came from.
TEXT_SOURCES = ("ax", "ocr", "none")


@dataclass
class Frame:
    """One captured moment."""

    id: int
    ts: float
    app: str
    title: str
    text: str
    thumb_path: Optional[str]
    thumb_bytes: int = 0
    width: int = 0
    height: int = 0
    snippet: str = ""  # populated by search()
    score: float = 0.0
    text_source: str = ""
    """``ax`` (accessibility tree), ``ocr``, or ``none``."""
    encrypted: bool = False
    """True when the text is stored encrypted at rest."""

    @property
    def when(self) -> str:
        return _dt.datetime.fromtimestamp(self.ts).strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "when": self.when,
            "app": self.app,
            "title": self.title,
            "text": self.text,
            "thumb_path": self.thumb_path,
            "snippet": self.snippet,
            "text_source": self.text_source,
            "encrypted": self.encrypted,
        }


@dataclass
class StoreStats:
    frames: int = 0
    oldest: Optional[float] = None
    newest: Optional[float] = None
    thumb_bytes: int = 0
    db_bytes: int = 0
    apps: list[tuple[str, int]] = field(default_factory=list)
    text_sources: dict[str, int] = field(default_factory=dict)
    """Frame counts per text source, e.g. ``{"ax": 12, "ocr": 40, "none": 3}``."""
    encrypted_text_frames: int = 0
    """Frames whose text is encrypted at rest (the rest are plaintext)."""
    thumbnails: int = 0
    """Frames that have a thumbnail."""
    encrypted_thumbnails: int = 0
    """Thumbnails stored encrypted (``*.jpg.enc``)."""
    encrypt_text: bool = False
    """Whether *new* frames get encrypted text (current setting)."""
    encrypt_thumbnails: bool = False
    """Whether *new* thumbnails are encrypted (current setting)."""
    key_store: Optional[str] = None
    """Where the memory key lives (``keychain`` / ``keyring`` / ``file``),
    or None when no key has been created yet."""
    encryption_available: bool = True
    """False when the ``cryptography`` package is missing."""

    @property
    def total_bytes(self) -> int:
        return self.thumb_bytes + self.db_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "oldest": self.oldest,
            "newest": self.newest,
            "thumb_bytes": self.thumb_bytes,
            "db_bytes": self.db_bytes,
            "total_bytes": self.total_bytes,
            "apps": [list(a) for a in self.apps],
            "text_sources": dict(self.text_sources),
            "encrypted_text_frames": self.encrypted_text_frames,
            "thumbnails": self.thumbnails,
            "encrypted_thumbnails": self.encrypted_thumbnails,
            "encrypt_text": self.encrypt_text,
            "encrypt_thumbnails": self.encrypt_thumbnails,
            "key_store": self.key_store,
            "encryption_available": self.encryption_available,
        }


class MemoryStore:
    """SQLite-backed frame index.  Safe for use from one process at a time
    per connection; the daemon and the tool each open their own.

    ``check_same_thread`` is disabled so the recorder can write from a
    worker thread; writes are serialised through an internal lock.

    Encryption settings come from *config* (or ``config.json`` under
    *home*) unless given explicitly.  The key is loaded lazily: reads never
    create one, and a store with nothing encrypted never touches it.
    """

    def __init__(
        self,
        home: Optional[Path] = None,
        *,
        db_path: Optional[Path] = None,
        config: Optional[MemoryConfig] = None,
        encrypt_thumbnails: Optional[bool] = None,
        encrypt_text: Optional[bool] = None,
    ) -> None:
        self.home = home
        self.dir = memory_dir(home)
        cfg = config if config is not None else load_config(home)
        self.encrypt_thumbnails = (
            cfg.encrypt_thumbnails if encrypt_thumbnails is None else bool(encrypt_thumbnails)
        )
        self.encrypt_text = cfg.encrypt_text if encrypt_text is None else bool(encrypt_text)
        self._cipher_obj: Optional[MemoryCipher] = None
        self._warned: set[str] = set()
        self.thumbs_dir = self.dir / "thumbs"
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (self.dir / "index.db")
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Zero deleted content instead of leaving it in free pages.
        self._conn.execute("PRAGMA secure_delete=ON")
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass
        self._fts = self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> bool:
        c = self._conn
        with self._lock:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS frames (
                    id          INTEGER PRIMARY KEY,
                    ts          REAL    NOT NULL,
                    app         TEXT    NOT NULL DEFAULT '',
                    title       TEXT    NOT NULL DEFAULT '',
                    text        TEXT    NOT NULL DEFAULT '',
                    text_hash   TEXT    NOT NULL DEFAULT '',
                    thumb_path  TEXT,
                    thumb_bytes INTEGER NOT NULL DEFAULT 0,
                    width       INTEGER NOT NULL DEFAULT 0,
                    height      INTEGER NOT NULL DEFAULT 0,
                    text_source TEXT    NOT NULL DEFAULT '',
                    text_enc    TEXT,
                    text_encrypted INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            c.execute("CREATE INDEX IF NOT EXISTS frames_ts ON frames(ts)")
            c.execute("CREATE INDEX IF NOT EXISTS frames_app ON frames(app)")
            c.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            c.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            self._migrate()
            c.execute(
                "CREATE INDEX IF NOT EXISTS frames_enc ON frames(id) WHERE text_encrypted = 1"
            )
            try:
                c.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS frames_fts USING fts5(
                        text, app, title,
                        content='frames', content_rowid='id',
                        tokenize='unicode61'
                    )
                    """
                )
                c.executescript(
                    """
                    CREATE TRIGGER IF NOT EXISTS frames_ai AFTER INSERT ON frames BEGIN
                        INSERT INTO frames_fts(rowid, text, app, title)
                        VALUES (new.id, new.text, new.app, new.title);
                    END;
                    CREATE TRIGGER IF NOT EXISTS frames_ad AFTER DELETE ON frames BEGIN
                        INSERT INTO frames_fts(frames_fts, rowid, text, app, title)
                        VALUES ('delete', old.id, old.text, old.app, old.title);
                    END;
                    CREATE TRIGGER IF NOT EXISTS frames_au AFTER UPDATE OF text, app, title ON frames BEGIN
                        INSERT INTO frames_fts(frames_fts, rowid, text, app, title)
                        VALUES ('delete', old.id, old.text, old.app, old.title);
                        INSERT INTO frames_fts(rowid, text, app, title)
                        VALUES (new.id, new.text, new.app, new.title);
                    END;
                    """
                )
                return True
            except sqlite3.OperationalError:
                return False

    def _migrate(self) -> None:
        """Bring a v1 database (no text_source / encryption columns) up to
        the current schema.  Idempotent."""
        c = self._conn
        cols = {r["name"] for r in c.execute("PRAGMA table_info(frames)").fetchall()}
        for name, decl in (
            ("text_source", "TEXT NOT NULL DEFAULT ''"),
            ("text_enc", "TEXT"),
            ("text_encrypted", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in cols:
                c.execute(f"ALTER TABLE frames ADD COLUMN {name} {decl}")
        row = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        try:
            version = int(row["value"]) if row else _SCHEMA_VERSION
        except (TypeError, ValueError):
            version = 1
        if version < 2:
            # v1's update trigger fired on *any* column change (re-indexing
            # the text whenever a thumbnail path was written); v2 narrows it.
            c.execute("DROP TRIGGER IF EXISTS frames_au")
            # Every v1 frame's text came from OCR.
            c.execute(
                "UPDATE frames SET text_source = CASE WHEN text != '' THEN 'ocr' ELSE 'none' END "
                "WHERE text_source = ''"
            )
            c.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'", (str(_SCHEMA_VERSION),)
            )

    @property
    def has_fts(self) -> bool:
        return self._fts

    def close(self) -> None:
        with self._lock:
            with contextlib.suppress(Exception):
                self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Encryption
    # ------------------------------------------------------------------

    def apply_config(self, config: MemoryConfig) -> None:
        """Pick up encryption settings from a (re)loaded config."""
        self.encrypt_thumbnails = bool(config.encrypt_thumbnails)
        self.encrypt_text = bool(config.encrypt_text)

    def _warn_once(self, key: str, msg: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, *args)

    def _cipher(self, *, create: bool) -> Optional[MemoryCipher]:
        """The memory cipher, loading the key on first use.  With *create*
        a key is generated when none exists yet (write paths only)."""
        if self._cipher_obj is not None:
            return self._cipher_obj
        if not crypto.crypto_available():
            self._warn_once(
                "nocrypto",
                "screen-memory encryption unavailable — pip install cryptography",
            )
            return None
        try:
            key, _store = crypto.load_or_create_key(self.home, create=create)
        except Exception as exc:
            self._warn_once("nokey", "screen-memory key unavailable: %s", exc)
            return None
        if key is None:
            return None
        self._cipher_obj = MemoryCipher(key)
        return self._cipher_obj

    def _decrypt_text(self, token: Optional[str]) -> Optional[str]:
        """Plaintext for an encrypted ``text_enc`` value, None on failure."""
        if not token:
            return ""
        c = self._cipher(create=False)
        if c is None:
            self._warn_once("nodecrypt", "cannot decrypt frame text: memory key unavailable")
            return None
        try:
            return c.decrypt_text(token)
        except Exception:
            self._warn_once("baddecrypt", "frame text failed to decrypt (wrong or replaced key?)")
            return None

    def _encode_text(self, text: str) -> tuple[str, Optional[str], int, str]:
        """``(text_column, text_enc, text_encrypted, text_hash)`` for *text*
        under the current settings."""
        if self.encrypt_text:
            c = self._cipher(create=True)
            if c is not None:
                return c.index_tokens(text), c.encrypt_text(text), 1, c.text_digest(text)
            self._warn_once("plaintext", "encrypt_text is on but no key is available; storing plaintext")
        return text, None, 0, hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()

    def has_encrypted_text(self) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM frames WHERE text_encrypted = 1 LIMIT 1"
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        ts: float,
        app: str,
        title: str,
        text: str,
        thumb: Optional[bytes],
        width: int = 0,
        height: int = 0,
        text_source: Optional[str] = None,
    ) -> Frame:
        """Insert a frame.  *thumb* (JPEG bytes) is written under
        ``thumbs/YYYY-MM-DD/<id>.jpg`` (``<id>.jpg.enc`` when encrypted).

        *text* should already be redacted (the recorder does that).
        *text_source* defaults to ``ocr`` when there is text, else ``none``.
        """
        text = text or ""
        source = text_source if text_source in TEXT_SOURCES else ("ocr" if text else "none")
        stored, enc, flag, digest = self._encode_text(text)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO frames(ts, app, title, text, text_hash, width, height, "
                "text_source, text_enc, text_encrypted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, app or "", title or "", stored, digest, width, height, source, enc, flag),
            )
            frame_id = int(cur.lastrowid)
            thumb_path: Optional[str] = None
            thumb_bytes = 0
            if thumb:
                day = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                thumb_path, thumb_bytes = self._write_thumb(day, frame_id, thumb, self.encrypt_thumbnails)
                self._conn.execute(
                    "UPDATE frames SET thumb_path=?, thumb_bytes=? WHERE id=?",
                    (thumb_path, thumb_bytes, frame_id),
                )
        return Frame(
            id=frame_id, ts=ts, app=app or "", title=title or "", text=text,
            thumb_path=thumb_path, thumb_bytes=thumb_bytes, width=width, height=height,
            text_source=source, encrypted=bool(flag),
        )

    def _write_thumb(self, day: str, frame_id: int, jpeg: bytes, encrypt: bool) -> tuple[str, int]:
        day_dir = self.thumbs_dir / day
        day_dir.mkdir(parents=True, exist_ok=True)
        data, name = jpeg, f"{frame_id}.jpg"
        if encrypt:
            c = self._cipher(create=True)
            if c is not None:
                data, name = c.encrypt_blob(jpeg), f"{frame_id}.jpg.enc"
            else:
                self._warn_once("plainthumb", "encrypt_thumbnails is on but no key is available; storing plaintext")
        p = day_dir / name
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        with contextlib.suppress(OSError):
            os.chmod(p, 0o600)
        return p.relative_to(self.dir).as_posix(), len(data)

    def last_text_hash(self) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT text_hash FROM frames ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row["text_hash"] if row else None

    def delete(self, ids: list[int]) -> int:
        """Delete frames (and their thumbnails) by id."""
        if not ids:
            return 0
        n = 0
        with self._lock:
            for chunk in _chunks(ids, 500):
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT id, thumb_path FROM frames WHERE id IN ({marks})", chunk,
                ).fetchall()
                for r in rows:
                    self._unlink_thumb(r["thumb_path"])
                cur = self._conn.execute(f"DELETE FROM frames WHERE id IN ({marks})", chunk)
                n += cur.rowcount
        self._prune_empty_thumb_dirs()
        return n

    def delete_range(self, start: float = 0.0, end: Optional[float] = None) -> int:
        with self._lock:
            if end is None:
                rows = self._conn.execute(
                    "SELECT id FROM frames WHERE ts >= ?", (start,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id FROM frames WHERE ts >= ? AND ts < ?", (start, end)
                ).fetchall()
        return self.delete([r["id"] for r in rows])

    def delete_app(self, app_pattern: str) -> int:
        pat = f"%{app_pattern}%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM frames WHERE app LIKE ? OR title LIKE ?", (pat, pat)
            ).fetchall()
        return self.delete([r["id"] for r in rows])

    def clear(self) -> int:
        with self._lock:
            rows = self._conn.execute("SELECT id FROM frames").fetchall()
        n = self.delete([r["id"] for r in rows])
        self.scrub()
        return n

    def scrub(self) -> None:
        """Physically purge deleted / rewritten text from the database file:
        rebuild the FTS index from the current rows (FTS5 otherwise keeps
        superseded terms in old segments), VACUUM, and truncate the WAL.
        Called after :meth:`reindex` and :meth:`clear`; worth calling after
        large deletions.  Cost is proportional to the index size."""
        with self._lock:
            if self._fts:
                with contextlib.suppress(sqlite3.OperationalError):
                    self._conn.execute("INSERT INTO frames_fts(frames_fts) VALUES('rebuild')")
            self._checkpoint()
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute("VACUUM")
            self._checkpoint()

    # -- migration / re-encryption ------------------------------------------

    def reindex(
        self,
        *,
        redact: bool = False,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, int]:
        """Rewrite every frame under the *current* settings.

        * text is encrypted (``encrypt_text`` on) or decrypted back to
          plaintext (off), rebuilding the search index either way;
        * thumbnails are encrypted / decrypted to match ``encrypt_thumbnails``;
        * with *redact*, secrets are scrubbed from old text and titles.

        Frames that cannot be decrypted (key missing) are left untouched and
        counted as ``failed``.  Returns counters.
        """
        from opendesk.memory.redact import redact as _redact

        counts = {
            "frames": 0, "text_encrypted": 0, "text_decrypted": 0,
            "thumbs_encrypted": 0, "thumbs_decrypted": 0, "redacted": 0, "failed": 0,
        }
        with self._lock:
            ids = [r["id"] for r in self._conn.execute("SELECT id FROM frames ORDER BY id").fetchall()]
        total = len(ids)
        for done, chunk in enumerate(_chunks(ids, 200)):
            with self._lock:
                self._conn.execute("BEGIN")
                try:
                    marks = ",".join("?" * len(chunk))
                    rows = self._conn.execute(
                        f"SELECT * FROM frames WHERE id IN ({marks})", chunk,
                    ).fetchall()
                    for row in rows:
                        self._reindex_row(row, redact, _redact, counts)
                    self._conn.execute("COMMIT")
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
            if progress:
                progress(min(total, (done + 1) * 200), total)
        if counts["text_encrypted"] or counts["text_decrypted"] or counts["redacted"]:
            self.scrub()  # don't leave the old text in free pages / FTS segments
        return counts

    def _reindex_row(self, row: sqlite3.Row, redact: bool, redact_fn: Any, counts: dict[str, int]) -> None:
        counts["frames"] += 1
        was_enc = bool(row["text_encrypted"])
        plain = self._decrypt_text(row["text_enc"]) if was_enc else (row["text"] or "")
        title = row["title"] or ""
        ok = plain is not None
        if ok:
            changed = False
            if redact:
                plain, n1 = redact_fn(plain)
                title, n2 = redact_fn(title)
                if n1 or n2:
                    counts["redacted"] += 1
                    changed = True
            if changed or was_enc != self.encrypt_text:
                stored, enc, flag, digest = self._encode_text(plain)
                self._conn.execute(
                    "UPDATE frames SET text=?, text_enc=?, text_encrypted=?, text_hash=?, title=? WHERE id=?",
                    (stored, enc, flag, digest, title, row["id"]),
                )
                if flag and not was_enc:
                    counts["text_encrypted"] += 1
                elif was_enc and not flag:
                    counts["text_decrypted"] += 1
        else:
            counts["failed"] += 1

        rel = row["thumb_path"]
        if not rel:
            return
        is_enc = rel.endswith(".enc")
        if is_enc == self.encrypt_thumbnails:
            return
        p = self.dir / rel
        try:
            data = p.read_bytes()
        except OSError:
            return
        if is_enc:
            c = self._cipher(create=False)
            try:
                if c is None:
                    raise ValueError("no key")
                data = c.decrypt_blob(data)
            except Exception:
                if ok:  # don't double-count a frame whose text already failed
                    counts["failed"] += 1
                return
        day = Path(rel).parent.name
        new_rel, size = self._write_thumb(day, int(row["id"]), data, self.encrypt_thumbnails)
        if new_rel == rel:
            return
        with contextlib.suppress(OSError):
            p.unlink()
        self._conn.execute(
            "UPDATE frames SET thumb_path=?, thumb_bytes=? WHERE id=?", (new_rel, size, row["id"]),
        )
        counts["thumbs_decrypted" if is_enc else "thumbs_encrypted"] += 1

    # -- retention ---------------------------------------------------------

    def enforce_retention(self, retention_days: int) -> int:
        """Delete frames older than *retention_days*.  Returns count."""
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM frames WHERE ts < ?", (cutoff,)
            ).fetchall()
        return self.delete([r["id"] for r in rows])

    def enforce_cap(self, cap_bytes: int, *, target_fraction: float = 0.9) -> int:
        """Rolling deletion: drop oldest frames until on-disk usage is under
        ``cap_bytes * target_fraction``.  Returns count deleted.

        Usage is thumbnails + the main database file (the WAL is checkpointed
        first so its transient size doesn't count).  Each deleted frame is
        assumed to free its thumbnail plus roughly three times its text
        length (row + FTS index); the database is vacuumed afterwards so the
        file actually shrinks.  At least one frame is always kept.
        """
        if cap_bytes <= 0:
            return 0
        self._checkpoint()
        usage = self._disk_usage(include_wal=False)
        if usage <= cap_bytes:
            return 0
        target = int(cap_bytes * target_fraction)
        deleted = 0
        with self._lock:
            while usage > target:
                total = self._conn.execute("SELECT COUNT(*) AS n FROM frames").fetchone()["n"]
                if total <= 1:
                    break
                batch = max(1, min(200, total - 1))
                rows = self._conn.execute(
                    "SELECT id, thumb_bytes, length(text) + COALESCE(length(text_enc), 0) AS tlen "
                    "FROM frames ORDER BY ts ASC LIMIT ?", (batch,),
                ).fetchall()
                if not rows:
                    break
                ids = [r["id"] for r in rows]
                freed = sum(int(r["thumb_bytes"] or 0) + 3 * int(r["tlen"] or 0) for r in rows)
                deleted += self.delete(ids)
                usage -= freed
        if deleted:
            self._checkpoint()
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute("VACUUM")
        return deleted

    def _checkpoint(self) -> None:
        with self._lock, contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _disk_usage(self, *, include_wal: bool = True) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(thumb_bytes), 0) AS tb FROM frames"
            ).fetchone()
        total = int(row["tb"] or 0)
        suffixes = ("", "-wal", "-shm") if include_wal else ("",)
        for suffix in suffixes:
            with contextlib.suppress(OSError):
                total += os.path.getsize(str(self.db_path) + suffix)
        return total

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, frame_id: int) -> Optional[Frame]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM frames WHERE id=?", (frame_id,)
            ).fetchone()
        return self._row_to_frame(row) if row else None

    def thumbnail(self, frame: Frame) -> Optional[bytes]:
        """JPEG bytes of *frame*'s thumbnail, decrypting ``.jpg.enc`` files.
        None when there is none or it cannot be decrypted."""
        if not frame.thumb_path:
            return None
        p = self.dir / frame.thumb_path
        try:
            data = p.read_bytes()
        except OSError:
            return None
        if not frame.thumb_path.endswith(".enc"):
            return data
        c = self._cipher(create=False)
        if c is None:
            self._warn_once("nothumbkey", "cannot decrypt thumbnails: memory key unavailable")
            return None
        try:
            return c.decrypt_blob(data)
        except Exception:
            self._warn_once("badthumb", "thumbnail failed to decrypt (wrong or replaced key?)")
            return None

    def search(
        self,
        query: str,
        *,
        start: float = 0.0,
        end: Optional[float] = None,
        app: Optional[str] = None,
        limit: int = 20,
    ) -> list[Frame]:
        """Full-text search.  All query terms must match (AND); when that
        yields nothing, retries as OR so near-misses still surface.

        Encrypted frames are matched through the blind index: exact words
        only (``foo*`` matches just the word ``foo`` there)."""
        terms = _terms(query)
        if not terms:
            return self.timeline(start=start, end=end, app=app, limit=limit)
        hits = self._search_terms(terms, "AND", start, end, app, limit)
        if not hits and len(terms) > 1:
            hits = self._search_terms(terms, "OR", start, end, app, limit)
        return hits

    def timeline(
        self,
        *,
        start: float = 0.0,
        end: Optional[float] = None,
        app: Optional[str] = None,
        limit: int = 50,
        newest_first: bool = True,
    ) -> list[Frame]:
        where, params = self._filters(start, end, app)
        order = "DESC" if newest_first else "ASC"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM frames{where} ORDER BY ts {order} LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._row_to_frame(r) for r in rows]

    def apps(self, *, start: float = 0.0, end: Optional[float] = None) -> list[tuple[str, int]]:
        where, params = self._filters(start, end, None)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT app, COUNT(*) AS n FROM frames{where} GROUP BY app ORDER BY n DESC",
                params,
            ).fetchall()
        return [(r["app"], int(r["n"])) for r in rows]

    def stats(self) -> StoreStats:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, MIN(ts) AS oldest, MAX(ts) AS newest, "
                "COALESCE(SUM(thumb_bytes), 0) AS tb, "
                "COALESCE(SUM(text_encrypted), 0) AS enc, "
                "COALESCE(SUM(thumb_path IS NOT NULL), 0) AS thumbs, "
                "COALESCE(SUM(thumb_path LIKE '%.enc'), 0) AS enc_thumbs "
                "FROM frames"
            ).fetchone()
            apps = self._conn.execute(
                "SELECT app, COUNT(*) AS n FROM frames GROUP BY app ORDER BY n DESC LIMIT 10"
            ).fetchall()
            sources = self._conn.execute(
                "SELECT text_source AS s, COUNT(*) AS n FROM frames GROUP BY text_source"
            ).fetchall()
        db_bytes = 0
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(OSError):
                db_bytes += os.path.getsize(str(self.db_path) + suffix)
        info = crypto.read_keyinfo(self.home) or {}
        return StoreStats(
            frames=int(row["n"] or 0),
            oldest=row["oldest"],
            newest=row["newest"],
            thumb_bytes=int(row["tb"] or 0),
            db_bytes=db_bytes,
            apps=[(a["app"], int(a["n"])) for a in apps],
            text_sources={(r["s"] or "unknown"): int(r["n"]) for r in sources},
            encrypted_text_frames=int(row["enc"] or 0),
            thumbnails=int(row["thumbs"] or 0),
            encrypted_thumbnails=int(row["enc_thumbs"] or 0),
            encrypt_text=self.encrypt_text,
            encrypt_thumbnails=self.encrypt_thumbnails,
            key_store=info.get("store"),
            encryption_available=crypto.crypto_available(),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _search_terms(
        self, terms: list[str], joiner: str, start: float, end: Optional[float],
        app: Optional[str], limit: int,
    ) -> list[Frame]:
        where, params = self._filters(start, end, app, alias="f")
        cipher = self._cipher(create=False) if self.has_encrypted_text() else None
        # Encrypted rows are re-checked after decryption, so over-fetch a little.
        fetch = limit * 3 + 10 if cipher else limit
        if self._fts:
            parts = []
            for t in terms:
                expr = _fts_quote(t)
                toks = tokenize(t.rstrip("*")) if cipher else []
                if toks:
                    blind = " AND ".join(f'"{cipher.token(x)}"' for x in toks)  # type: ignore[union-attr]
                    expr = f"({expr} OR ({blind}))"
                parts.append(expr)
            match = f" {joiner} ".join(parts)
            sql = (
                "SELECT f.*, "
                "snippet(frames_fts, 0, '[', ']', ' … ', 12) AS snippet, "
                "bm25(frames_fts) AS score "
                "FROM frames_fts JOIN frames f ON f.id = frames_fts.rowid "
                f"WHERE frames_fts MATCH ?{where.replace(' WHERE ', ' AND ', 1)} "
                "ORDER BY score, f.ts DESC LIMIT ?"
            )
            with self._lock:
                try:
                    rows = self._conn.execute(sql, (match, *params, fetch)).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            frames = [self._row_to_frame(r, with_snippet=True) for r in rows]
        else:
            # LIKE fallback.
            clauses = []
            like_params: list[Any] = []
            for t in terms:
                clause = "f.text LIKE ? OR f.app LIKE ? OR f.title LIKE ?"
                like_params += [f"%{t}%"] * 3
                toks = tokenize(t.rstrip("*")) if cipher else []
                if toks:
                    clause += " OR (f.text_encrypted = 1" + " AND f.text LIKE ?" * len(toks) + ")"
                    like_params += [f"%{cipher.token(x)}%" for x in toks]  # type: ignore[union-attr]
                clauses.append(f"({clause})")
            cond = f" {joiner} ".join(clauses)
            extra = where.replace(" WHERE ", " AND ", 1)
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT f.* FROM frames f WHERE ({cond}){extra} ORDER BY f.ts DESC LIMIT ?",
                    (*like_params, *params, fetch),
                ).fetchall()
            frames = [self._row_to_frame(r) for r in rows]
            for fr in frames:
                if not fr.encrypted:
                    fr.snippet = _like_snippet(fr.text, terms)

        out: list[Frame] = []
        for fr in frames:
            if fr.encrypted:
                if not _local_match(fr, terms, joiner):
                    continue
                fr.snippet = _highlight_snippet(fr.text, terms)
            out.append(fr)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _filters(
        start: float, end: Optional[float], app: Optional[str], *, alias: str = "",
    ) -> tuple[str, list[Any]]:
        pre = f"{alias}." if alias else ""
        clauses: list[str] = []
        params: list[Any] = []
        if start and start > 0:
            clauses.append(f"{pre}ts >= ?")
            params.append(start)
        if end is not None:
            clauses.append(f"{pre}ts < ?")
            params.append(end)
        if app:
            clauses.append(f"({pre}app LIKE ? OR {pre}title LIKE ?)")
            params += [f"%{app}%", f"%{app}%"]
        if not clauses:
            return "", params
        return " WHERE " + " AND ".join(clauses), params

    def _row_to_frame(self, row: sqlite3.Row, *, with_snippet: bool = False) -> Frame:
        keys = row.keys()
        encrypted = bool(row["text_encrypted"]) if "text_encrypted" in keys else False
        text = row["text"] or ""
        if encrypted:
            text = self._decrypt_text(row["text_enc"]) or ""
        return Frame(
            id=int(row["id"]),
            ts=float(row["ts"]),
            app=row["app"] or "",
            title=row["title"] or "",
            text=text,
            thumb_path=row["thumb_path"],
            thumb_bytes=int(row["thumb_bytes"] or 0),
            width=int(row["width"] or 0),
            height=int(row["height"] or 0),
            snippet=(row["snippet"] or "") if with_snippet and "snippet" in keys else "",
            score=float(row["score"]) if with_snippet and "score" in keys else 0.0,
            text_source=(row["text_source"] or "") if "text_source" in keys else "",
            encrypted=encrypted,
        )

    def _unlink_thumb(self, rel: Optional[str]) -> None:
        if not rel:
            return
        with contextlib.suppress(OSError):
            (self.dir / rel).unlink()

    def _prune_empty_thumb_dirs(self) -> None:
        with contextlib.suppress(OSError):
            for d in self.thumbs_dir.iterdir():
                if d.is_dir():
                    with contextlib.suppress(OSError):
                        d.rmdir()  # only succeeds when empty


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[\w#@.\-/:]+", re.UNICODE)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _terms(query: str) -> list[str]:
    """Split a free-text query into search terms.  Quoted phrases stay
    together.  Punctuation-only tokens are dropped."""
    out: list[str] = []
    for m in re.finditer(r'"([^"]+)"|(\S+)', query or ""):
        tok = (m.group(1) or m.group(2)).strip()
        if not tok:
            continue
        if m.group(1):
            out.append(tok)
            continue
        cleaned = tok.strip("\"'.,;:!?()[]{}")
        if cleaned and any(ch.isalnum() for ch in cleaned):
            out.append(cleaned)
    return out


def _fts_quote(term: str) -> str:
    """Quote a term for FTS5.  Trailing ``*`` keeps prefix semantics."""
    prefix = term.endswith("*")
    core = term.rstrip("*").replace('"', '""')
    q = f'"{core}"'
    return q + "*" if prefix else q


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(normalize(text))


def _contains_seq(words: list[str], seq: list[str], *, prefix_last: bool = False) -> bool:
    n = len(seq)
    if n == 0:
        return False
    for i in range(len(words) - n + 1):
        if words[i:i + n - 1] == seq[:-1]:
            last = words[i + n - 1]
            if last == seq[-1] or (prefix_last and last.startswith(seq[-1])):
                return True
    return False


def _local_match(frame: Frame, terms: list[str], joiner: str) -> bool:
    """Re-check a decrypted frame against the query with FTS-like token
    semantics (phrases must be contiguous).  Prefix terms match only exact
    words in the text, but prefixes in the plaintext app/title."""
    text_words = _words(frame.text)
    meta_words = _words(f"{frame.app}\n{frame.title}")
    results = []
    for t in terms:
        seq = _words(t.rstrip("*"))
        prefix = t.endswith("*")
        results.append(
            _contains_seq(text_words, seq) or _contains_seq(meta_words, seq, prefix_last=prefix)
        )
    return all(results) if joiner == "AND" else any(results)


def _highlight_snippet(text: str, terms: list[str], width: int = 100) -> str:
    """Window of *text* around the first matched term, with ``[hit]``."""
    low = text.lower()
    for t in terms:
        needle = t.lower().rstrip("*")
        i = low.find(needle)
        if i >= 0:
            a = max(0, i - width // 2)
            b = min(len(text), i + len(needle) + width // 2)
            s = (
                ("…" if a > 0 else "") + text[a:i] + "[" + text[i:i + len(needle)] + "]"
                + text[i + len(needle):b] + ("…" if b < len(text) else "")
            )
            return re.sub(r"\s+", " ", s)
    return re.sub(r"\s+", " ", text[:width])


def _like_snippet(text: str, terms: list[str], width: int = 80) -> str:
    low = text.lower()
    for t in terms:
        i = low.find(t.lower().rstrip("*"))
        if i >= 0:
            a = max(0, i - width // 2)
            b = min(len(text), i + len(t) + width // 2)
            return ("…" if a > 0 else "") + text[a:b].replace("\n", " ") + ("…" if b < len(text) else "")
    return text[:width].replace("\n", " ")


def _chunks(seq: list[Any], n: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
