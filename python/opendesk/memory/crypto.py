"""Encryption at rest for screen memory.

One random 32-byte *memory key* protects everything.  Where it lives:

* macOS — the login Keychain, generic password ``service=opendesk``,
  ``account=memory-key`` (hex-encoded), via ``/usr/bin/security``.
* elsewhere — the ``keyring`` package (Secret Service / Credential
  Manager) when installed, same service/account.
* otherwise — a ``0600`` file ``<home>/memory/key``.  Weaker: anyone who
  can read your home directory (backups included) can read the key.

Which store holds the key is recorded (no secret material) in
``<home>/memory/keyinfo.json`` so both SDKs and every later run use the
same one.  ``OPENDESK_MEMORY_KEYSTORE=keychain|keyring|file`` forces a
store.  In ``auto`` mode a *custom* ``--home`` (tests, sandboxes, portable
installs) keeps its key in the file store so it never touches the OS
keychain; the default ``~/.opendesk`` uses the OS store.

Sub-keys are derived with HMAC-SHA256 so the raw key is never used
directly:

* ``enc``   = HMAC(key, "opendesk-memory:enc")    → AES-256-GCM
* ``index`` = HMAC(key, "opendesk-memory:index")  → blind-index tokens

Blob format (thumbnails ``*.jpg.enc``): ``b"ODE1" | nonce(12) | ciphertext+tag``.
Text format: ``"v1:" + base64(nonce(12) | ciphertext+tag)``.
Kept in sync with ``js/src/memory/crypto.ts``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import platform
import re
import secrets
import shutil
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Optional, Protocol

log = logging.getLogger("opendesk.memory")

KEY_SERVICE = "opendesk"
KEY_ACCOUNT = "memory-key"
KEYINFO_FILE = "keyinfo.json"
KEY_FILE = "key"
ENV_KEYSTORE = "OPENDESK_MEMORY_KEYSTORE"

BLOB_MAGIC = b"ODE1"
TEXT_PREFIX = "v1:"
NONCE_LEN = 12
THUMB_AAD = b"opendesk-memory-thumb"
TEXT_AAD = b"opendesk-memory-text"
TOKEN_HEX_LEN = 20  # 80-bit blind-index tokens


class CryptoUnavailable(RuntimeError):
    """Raised when the ``cryptography`` package is not installed."""


def crypto_available() -> bool:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Key stores
# ---------------------------------------------------------------------------


class KeyProvider(Protocol):
    name: str

    def load(self) -> Optional[bytes]: ...

    def save(self, key: bytes) -> None: ...


def _decode_key(text: Optional[str]) -> Optional[bytes]:
    if not text:
        return None
    t = text.strip()
    try:
        raw = bytes.fromhex(t)
    except ValueError:
        return None
    return raw if len(raw) == 32 else None


class KeychainProvider:
    """macOS login Keychain through ``/usr/bin/security``."""

    name = "keychain"

    def __init__(self, service: str = KEY_SERVICE, account: str = KEY_ACCOUNT) -> None:
        self.service = service
        self.account = account

    @staticmethod
    def available() -> bool:
        return platform.system() == "Darwin" and shutil.which("security") is not None

    def load(self) -> Optional[bytes]:
        try:
            r = subprocess.run(
                ["security", "find-generic-password", "-s", self.service, "-a", self.account, "-w"],
                capture_output=True, text=True, timeout=15,
            )
        except Exception as exc:
            log.warning("keychain lookup failed: %s", exc)
            return None
        return _decode_key(r.stdout) if r.returncode == 0 else None

    def save(self, key: bytes) -> None:
        # ``security -i`` reads the command from stdin so the secret never
        # appears in the process list.
        cmd = (
            f'add-generic-password -U -s "{self.service}" -a "{self.account}" '
            f'-l "opendesk screen memory" -w "{key.hex()}"\n'
        )
        r = subprocess.run(["security", "-i"], input=cmd, capture_output=True, text=True, timeout=15)
        if r.returncode != 0 or self.load() != key:
            raise RuntimeError(f"could not store the memory key in the Keychain: {r.stderr.strip()}")

    def delete(self) -> None:
        subprocess.run(
            ["security", "delete-generic-password", "-s", self.service, "-a", self.account],
            capture_output=True, timeout=15,
        )


class KeyringProvider:
    """The ``keyring`` package (Secret Service, Windows Credential Manager …)."""

    name = "keyring"

    def __init__(self, service: str = KEY_SERVICE, account: str = KEY_ACCOUNT) -> None:
        self.service = service
        self.account = account

    @staticmethod
    def available() -> bool:
        try:
            import keyring  # type: ignore[import-not-found]
            from keyring.backends import fail  # type: ignore[import-not-found]
            return not isinstance(keyring.get_keyring(), fail.Keyring)
        except Exception:
            return False

    def load(self) -> Optional[bytes]:
        try:
            import keyring  # type: ignore[import-not-found]
            return _decode_key(keyring.get_password(self.service, self.account))
        except Exception as exc:
            log.warning("keyring lookup failed: %s", exc)
            return None

    def save(self, key: bytes) -> None:
        import keyring  # type: ignore[import-not-found]
        keyring.set_password(self.service, self.account, key.hex())


class FileProvider:
    """``<home>/memory/key`` with ``0600`` permissions."""

    name = "file"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> Optional[bytes]:
        try:
            return _decode_key(self.path.read_text())
        except OSError:
            return None

    def save(self, key: bytes) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(key.hex())
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


class StaticProvider:
    """In-memory key — for tests and embedding."""

    name = "static"

    def __init__(self, key: Optional[bytes] = None) -> None:
        self.key = key

    def load(self) -> Optional[bytes]:
        return self.key

    def save(self, key: bytes) -> None:
        self.key = key


def _mem_dir(home: Optional[Path]) -> Path:
    from opendesk.memory.config import memory_dir
    return memory_dir(home)


def _is_default_home(home: Optional[Path]) -> bool:
    if home is None:
        return True
    from opendesk.memory.config import DEFAULT_HOME
    try:
        return Path(home).expanduser().resolve() == DEFAULT_HOME.resolve()
    except OSError:
        return False


def read_keyinfo(home: Optional[Path] = None) -> Optional[dict]:
    p = _mem_dir(home) / KEYINFO_FILE
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_keyinfo(home: Optional[Path], store: str, key: bytes) -> None:
    from opendesk.memory.config import _atomic_write
    _atomic_write(
        _mem_dir(home) / KEYINFO_FILE,
        json.dumps({"store": store, "key_id": key_id(key), "created": time.time()}, indent=2),
    )


def _provider_named(name: str, home: Optional[Path]) -> Optional[KeyProvider]:
    if name == "keychain":
        return KeychainProvider() if KeychainProvider.available() else None
    if name == "keyring":
        return KeyringProvider() if KeyringProvider.available() else None
    if name == "file":
        return FileProvider(_mem_dir(home) / KEY_FILE)
    return None


def resolve_key_provider(home: Optional[Path] = None) -> KeyProvider:
    """Pick the key store for *home*.

    Honour ``keyinfo.json`` (the store chosen when the key was created),
    then ``$OPENDESK_MEMORY_KEYSTORE``, then the platform default.  Tests
    monkeypatch this function.
    """
    info = read_keyinfo(home) or {}
    for name in (info.get("store"), os.environ.get(ENV_KEYSTORE, "").strip().lower()):
        if name and name != "auto":
            p = _provider_named(name, home)
            if p is not None:
                return p
            if name == info.get("store"):
                # The existing key lives somewhere this process can't reach;
                # never silently mint a second key.
                raise RuntimeError(f"the memory key is in the {name!r} store, which is not available here")
            log.warning("memory key store %r is not available here; falling back", name)
    if _is_default_home(home):
        if KeychainProvider.available():
            return KeychainProvider()
        if KeyringProvider.available():
            return KeyringProvider()
    return FileProvider(_mem_dir(home) / KEY_FILE)


def load_or_create_key(home: Optional[Path] = None, *, create: bool) -> tuple[Optional[bytes], str]:
    """Return ``(key, store_name)``; *create* generates and stores a new key
    when none exists yet."""
    provider = resolve_key_provider(home)
    key = provider.load()
    if key is None and create:
        key = secrets.token_bytes(32)
        provider.save(key)
        if provider.name != "static":
            _write_keyinfo(home, provider.name, key)
        log.info("created screen-memory key in %s", provider.name)
    elif key is not None and provider.name != "static" and read_keyinfo(home) is None:
        _write_keyinfo(home, provider.name, key)
    return key, provider.name


def key_id(key: bytes) -> str:
    """Non-secret fingerprint of a key (first 16 hex of an HMAC)."""
    return hmac.new(key, b"opendesk-memory-key-id", hashlib.sha256).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cipher
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Lower-case and strip diacritics (matches FTS5 ``unicode61``)."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def tokenize(text: str) -> list[str]:
    """Unique word tokens, in first-seen order."""
    seen: dict[str, None] = {}
    for tok in _WORD_RE.findall(normalize(text)):
        if len(tok) <= 64:
            seen.setdefault(tok, None)
    return list(seen)


class MemoryCipher:
    """AES-256-GCM for blobs and text, HMAC-SHA256 for blind-index tokens."""

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("memory key must be 32 bytes")
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except Exception as exc:  # pragma: no cover - exercised when missing
            raise CryptoUnavailable(
                "the 'cryptography' package is required for screen-memory encryption "
                "(pip install cryptography)"
            ) from exc
        self.key_id = key_id(key)
        self._aead = AESGCM(hmac.new(key, b"opendesk-memory:enc", hashlib.sha256).digest())
        self._idx = hmac.new(key, b"opendesk-memory:index", hashlib.sha256).digest()

    # -- blobs ---------------------------------------------------------

    def encrypt_blob(self, data: bytes, aad: bytes = THUMB_AAD) -> bytes:
        nonce = secrets.token_bytes(NONCE_LEN)
        return BLOB_MAGIC + nonce + self._aead.encrypt(nonce, data, aad)

    def decrypt_blob(self, blob: bytes, aad: bytes = THUMB_AAD) -> bytes:
        if not blob.startswith(BLOB_MAGIC):
            raise ValueError("not an opendesk encrypted blob")
        nonce = blob[len(BLOB_MAGIC):len(BLOB_MAGIC) + NONCE_LEN]
        return self._aead.decrypt(nonce, blob[len(BLOB_MAGIC) + NONCE_LEN:], aad)

    # -- text ----------------------------------------------------------

    def encrypt_text(self, text: str) -> str:
        nonce = secrets.token_bytes(NONCE_LEN)
        ct = self._aead.encrypt(nonce, text.encode("utf-8"), TEXT_AAD)
        return TEXT_PREFIX + base64.b64encode(nonce + ct).decode("ascii")

    def decrypt_text(self, token: str) -> str:
        if not token.startswith(TEXT_PREFIX):
            raise ValueError("unknown encrypted-text format")
        raw = base64.b64decode(token[len(TEXT_PREFIX):])
        return self._aead.decrypt(raw[:NONCE_LEN], raw[NONCE_LEN:], TEXT_AAD).decode("utf-8")

    # -- blind index ---------------------------------------------------

    def token(self, word: str) -> str:
        """Keyed token for one (already normalised) word."""
        return hmac.new(self._idx, word.encode("utf-8"), hashlib.sha256).hexdigest()[:TOKEN_HEX_LEN]

    def index_tokens(self, text: str) -> str:
        """Space-separated, sorted, de-duplicated blind tokens for *text*.

        Sorting hides word order; de-duplication hides frequency."""
        return " ".join(sorted({self.token(t) for t in tokenize(text)}))

    def text_digest(self, text: str) -> str:
        return hmac.new(self._idx, b"digest:" + text.encode("utf-8"), hashlib.sha256).hexdigest()[:40]
