"""Secrets: ``{{secret:NAME}}`` placeholders resolved at execution time.

An agent never sees a password.  It types ``{{secret:github}}`` and the guard
substitutes the value from the OS credential store right before the tool
runs, then scrubs the value from the tool output, errors, the chained audit
log and the in-memory sandbox audit / replay params.

Stores (service ``opendesk``, account ``NAME``):

* macOS   — Keychain via ``security find-generic-password -s opendesk -a NAME -w``
* Linux   — Secret Service via ``secret-tool``
* Windows / fallback — the ``keyring`` package if installed
* otherwise secrets are refused.

Placeholders are only honoured in text that is *typed* — ``keyboard`` /
``ui`` / ``browser`` ``type`` (and ``fill``) — never in clipboard writes,
scripts, URLs, etc.  Using a secret needs approval by default; the index
file ``<home>/secrets.json`` (names + settings only, never values) can turn
that off per secret and restrict which apps may receive it::

    {"secrets": {"github": {"created": 1760000000.0,
                            "confirm": true,
                            "allowed_apps": ["Safari", "Google Chrome"]}}}
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from opendesk.safety._home import read_json, resolve_home, write_json

SERVICE = "opendesk"
INDEX_FILE = "secrets.json"
PLACEHOLDER_RE = re.compile(r"\{\{\s*secret:([A-Za-z0-9_.\-]+)\s*\}\}")
NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

# (tool, action) -> param keys where placeholders are allowed.
SECRET_FIELDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("keyboard", "type"): ("text",),
    ("ui", "type"): ("text",),
    ("browser", "type"): ("text",),
    ("browser", "fill"): ("text", "value"),
}


class SecretError(RuntimeError):
    """A secret could not be stored, found or used.  Never contains a value."""


# ---------------------------------------------------------------------------
# OS store backends
# ---------------------------------------------------------------------------


def _run(argv: list[str], *, input: Optional[str] = None, timeout: float = 15.0) -> subprocess.CompletedProcess:
    """Run a store command.  Separate function so tests can replace it."""
    return subprocess.run(
        argv, input=input, capture_output=True, text=True, timeout=timeout, check=False,
    )


def _keyring() -> Any:
    try:
        import keyring  # type: ignore[import-not-found]
        return keyring
    except Exception:
        return None


def store_backend(platform: Optional[str] = None) -> Optional[str]:
    """Name of the usable backend: ``macos`` | ``secret-tool`` | ``keyring`` | ``None``."""
    platform = platform or sys.platform
    if platform == "darwin" and shutil.which("security"):
        return "macos"
    if platform.startswith("linux") and shutil.which("secret-tool"):
        return "secret-tool"
    if _keyring() is not None:
        return "keyring"
    return None


def _no_store() -> SecretError:
    return SecretError(
        "no OS secret store available (macOS Keychain, Linux secret-tool, or the "
        "'keyring' package) — secrets are refused on this machine"
    )


def store_get(name: str) -> str:
    backend = store_backend()
    if backend == "macos":
        p = _run(["security", "find-generic-password", "-s", SERVICE, "-a", name, "-w"])
        if p.returncode != 0:
            raise SecretError(f"secret {name!r} not found in the macOS Keychain")
        return p.stdout[:-1] if p.stdout.endswith("\n") else p.stdout
    if backend == "secret-tool":
        p = _run(["secret-tool", "lookup", "service", SERVICE, "account", name])
        if p.returncode != 0 or not p.stdout:
            raise SecretError(f"secret {name!r} not found in the Secret Service keyring")
        return p.stdout[:-1] if p.stdout.endswith("\n") else p.stdout
    if backend == "keyring":
        val = _keyring().get_password(SERVICE, name)
        if val is None:
            raise SecretError(f"secret {name!r} not found in the keyring")
        return val
    raise _no_store()


def store_set(name: str, value: str) -> None:
    backend = store_backend()
    if backend == "macos":
        # Stored with /usr/bin/security so the item's ACL trusts the same
        # binary that reads it back (no Keychain prompt on every use).  The
        # value is briefly visible on this process's argv — see docs/safety.md.
        p = _run(["security", "add-generic-password", "-U", "-s", SERVICE, "-a", name,
                  "-l", f"opendesk secret {name}", "-w", value])
        if p.returncode != 0:
            raise SecretError(f"could not store {name!r} in the macOS Keychain")
        return
    if backend == "secret-tool":
        p = _run(["secret-tool", "store", f"--label=opendesk secret {name}",
                  "service", SERVICE, "account", name], input=value)
        if p.returncode != 0:
            raise SecretError(f"could not store {name!r} with secret-tool")
        return
    if backend == "keyring":
        _keyring().set_password(SERVICE, name, value)
        return
    raise _no_store()


def store_delete(name: str) -> None:
    backend = store_backend()
    if backend == "macos":
        _run(["security", "delete-generic-password", "-s", SERVICE, "-a", name])
        return
    if backend == "secret-tool":
        _run(["secret-tool", "clear", "service", SERVICE, "account", name])
        return
    if backend == "keyring":
        try:
            _keyring().delete_password(SERVICE, name)
        except Exception:
            pass
        return
    raise _no_store()


# ---------------------------------------------------------------------------
# Index (names + settings; never values)
# ---------------------------------------------------------------------------


@dataclass
class SecretInfo:
    name: str
    created: Optional[float] = None
    confirm: bool = True
    allowed_apps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"created": self.created, "confirm": self.confirm}
        if self.allowed_apps:
            d["allowed_apps"] = self.allowed_apps
        return d


def index_path(home: Any = None) -> Path:
    return resolve_home(home) / INDEX_FILE


def load_index(home: Any = None) -> dict[str, SecretInfo]:
    data = read_json(index_path(home)) or {}
    out: dict[str, SecretInfo] = {}
    for name, info in (data.get("secrets") or {}).items():
        info = info if isinstance(info, dict) else {}
        out[name] = SecretInfo(
            name=name,
            created=info.get("created"),
            confirm=bool(info.get("confirm", True)),
            allowed_apps=[str(a) for a in info.get("allowed_apps") or []],
        )
    return out


def _save_index(index: dict[str, SecretInfo], home: Any = None) -> None:
    write_json(index_path(home), {"secrets": {n: i.to_dict() for n, i in sorted(index.items())}})


def set_secret(
    name: str, value: str, home: Any = None, *,
    allowed_apps: Optional[list[str]] = None, confirm: bool = True,
) -> SecretInfo:
    if not NAME_RE.match(name):
        raise SecretError("secret names may only contain letters, digits, '_', '-', '.' (max 64)")
    if not value:
        raise SecretError("refusing to store an empty secret")
    store_set(name, value)
    index = load_index(home)
    info = SecretInfo(name=name, created=time.time(), confirm=confirm,
                      allowed_apps=list(allowed_apps or []))
    index[name] = info
    _save_index(index, home)
    return info


def delete_secret(name: str, home: Any = None) -> bool:
    index = load_index(home)
    existed = index.pop(name, None) is not None
    try:
        store_delete(name)
    except SecretError:
        if not existed:
            raise
    _save_index(index, home)
    return existed


def list_secrets(home: Any = None) -> list[SecretInfo]:
    return sorted(load_index(home).values(), key=lambda i: i.name)


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------


def find_placeholders(obj: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    """All ``(param_path, secret_name)`` occurrences inside *obj*."""
    out: list[tuple[tuple[str, ...], str]] = []
    if isinstance(obj, str):
        out.extend((path, m.group(1)) for m in PLACEHOLDER_RE.finditer(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(find_placeholders(v, path + (str(k),)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(find_placeholders(v, path + (str(i),)))
    return out


def _allowed_location(tool: str, params: dict[str, Any], path: tuple[str, ...]) -> bool:
    if tool == "fleet" and len(path) >= 2 and path[0] in ("params", "arguments", "args"):
        inner = params.get("tool")
        inner_params = params.get(path[0])
        if isinstance(inner, str) and isinstance(inner_params, dict):
            return _allowed_location(inner, inner_params, path[1:])
        return False
    action = params.get("action")
    fields = SECRET_FIELDS.get((tool, action if isinstance(action, str) else ""))
    return bool(fields) and len(path) == 1 and path[0] in fields


def secret_names(tool: str, params: dict[str, Any]) -> list[str]:
    """Validate placement and return the referenced secret names (ordered, unique).

    Raises :class:`SecretError` when a placeholder appears where secrets are
    not allowed.
    """
    names: list[str] = []
    for path, name in find_placeholders(params):
        if not _allowed_location(tool, params, path):
            where = ".".join(path) or "params"
            raise SecretError(
                f"secret placeholders are only allowed in typed text "
                f"(keyboard/ui/browser type); found one in {tool}.{where}"
            )
        if name not in names:
            names.append(name)
    return names


def resolve(obj: Any, values: dict[str, str]) -> Any:
    """Return a copy of *obj* with placeholders replaced by *values*."""
    if isinstance(obj, str):
        return PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: resolve(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve(v, values) for v in obj]
    if isinstance(obj, tuple):
        return tuple(resolve(v, values) for v in obj)
    return obj


def fetch_values(names: list[str], home: Any = None) -> dict[str, str]:
    index = load_index(home)
    values: dict[str, str] = {}
    for name in names:
        if name not in index:
            raise SecretError(
                f"unknown secret {name!r} — ask the user to run `opendesk secrets set {name}`"
            )
        values[name] = store_get(name)
    return values


class Redactor:
    """Replaces secret values with their placeholders in arbitrary data."""

    def __init__(self, values: dict[str, str]) -> None:
        # Longest first so overlapping values redact fully.
        self._pairs = sorted(
            ((v, "{{secret:%s}}" % n) for n, v in values.items() if v),
            key=lambda p: -len(p[0]),
        )

    def __bool__(self) -> bool:
        return bool(self._pairs)

    def text(self, s: str) -> str:
        for value, placeholder in self._pairs:
            if value in s:
                s = s.replace(value, placeholder)
        return s

    def any(self, obj: Any) -> Any:
        if not self._pairs:
            return obj
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, dict):
            return {k: self.any(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.any(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.any(v) for v in obj)
        return obj
