"""Secret redaction for screen-memory text.

Applied to every frame's text (and window title) *before* it is written to
the index, so credentials that flash past on screen never reach disk.
Each hit is replaced by ``[REDACTED:<kind>]``.

Kinds: ``card`` (Luhn-valid payment card numbers), ``aws_key``,
``github_token``, ``anthropic_key``, ``openai_key``, ``slack_token``,
``jwt``, ``private_key`` (PEM blocks), ``password`` / ``secret``
(``password: hunter2``-style pairs).

Kept byte-for-byte equivalent with ``js/src/memory/redact.ts``.
"""

from __future__ import annotations

import re
from typing import Callable, Union

Replacement = Union[str, Callable[["re.Match[str]"], str]]


def _tag(kind: str) -> str:
    return f"[REDACTED:{kind}]"


_PEM_RE = re.compile(
    r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"
    r"(?:[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|[A-Za-z0-9+/=\s]*)"
)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
_ANTHROPIC_RE = re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")
_OPENAI_RE = re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}")
_GITHUB_RE = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{22,255})")
_SLACK_RE = re.compile(r"\bxox[abposre]-[A-Za-z0-9-]{10,}")
_AWS_RE = re.compile(
    r"\b(?:AKIA|ASIA|ABIA|ACCA|AGPA|AIDA|AIPA|ANPA|ANVA|APKA|AROA|ASCA)[A-Z0-9]{16}\b"
)
_CARD_RE = re.compile(r"(?<![\d])\d(?:[ -]?\d){12,18}(?![\d])")
_PAIR_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(password|passwd|pwd|passphrase|pass code|passcode|pin code"
    r"|secret|client[_ -]?secret|secret[_ -]?key|api[_ -]?key|access[_ -]?key"
    r"|access[_ -]?token|auth[_ -]?token|token|aws_secret_access_key)"
    r"(\s*[:=]\s*)([\"']?)([^\s\"']{1,200})\3"
)
_BEARER_RE = re.compile(r"(?i)\b(bearer)(\s+)([A-Za-z0-9._~+/-]{20,}=*)")
_PASSWORD_LABELS = {"password", "passwd", "pwd", "passphrase", "pass code", "passcode", "pin code"}


def luhn_ok(digits: str) -> bool:
    """Luhn checksum over a string of digits."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card(m: "re.Match[str]") -> str:
    raw = m.group(0)
    digits = re.sub(r"\D", "", raw)
    if (
        13 <= len(digits) <= 19
        and digits[0] in "23456"
        and len(set(digits)) > 1
        and luhn_ok(digits)
    ):
        return _tag("card")
    return raw


def _pair(m: "re.Match[str]") -> str:
    label, sep, quote, value = m.group(1), m.group(2), m.group(3), m.group(4)
    if value.startswith("[REDACTED:"):
        return m.group(0)
    kind = "password" if label.lower() in _PASSWORD_LABELS else "secret"
    return f"{label}{sep}{quote}{_tag(kind)}{quote}"


# Order matters: specific token shapes first, generic label pairs last.
_RULES: list[tuple[str, "re.Pattern[str]", Replacement]] = [
    ("private_key", _PEM_RE, _tag("private_key")),
    ("jwt", _JWT_RE, _tag("jwt")),
    ("anthropic_key", _ANTHROPIC_RE, _tag("anthropic_key")),
    ("openai_key", _OPENAI_RE, _tag("openai_key")),
    ("github_token", _GITHUB_RE, _tag("github_token")),
    ("slack_token", _SLACK_RE, _tag("slack_token")),
    ("aws_key", _AWS_RE, _tag("aws_key")),
    ("secret", _BEARER_RE, lambda m: f"{m.group(1)}{m.group(2)}{_tag('secret')}"),
    ("card", _CARD_RE, _card),
    ("password", _PAIR_RE, _pair),
]


def redact(text: str) -> tuple[str, int]:
    """Return ``(redacted_text, number_of_redactions)``."""
    if not text:
        return text or "", 0
    count = 0
    out = text
    for _kind, pattern, repl in _RULES:
        def _sub(m: "re.Match[str]", _repl: Replacement = repl) -> str:
            nonlocal count
            new = _repl(m) if callable(_repl) else _repl
            if new != m.group(0):
                count += 1
            return new
        out = pattern.sub(_sub, out)
    return out, count


def redact_text(text: str) -> str:
    """Convenience wrapper returning only the redacted text."""
    return redact(text)[0]
