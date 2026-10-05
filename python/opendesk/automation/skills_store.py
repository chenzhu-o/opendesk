"""Persistent, parameterised skill library for computer-use agents.

A *skill* is a reusable procedure — an ordered list of tool calls with
``{{placeholder}}`` templates that can be re-bound on every run.  Unlike a
recorded trajectory (one concrete sequence of coordinates and keystrokes), a
skill captures the *policy*: what to do, in terms of tool calls, with the
arguments left open.

This is the lightweight, dependency-free version of the "persistent skill"
idea from recent computer-use work (interaction traces → reusable policies,
neuro-symbolic reuse, recursive skill abstraction).  Skills accumulate in a
project directory and are retrieved by relevance at task time.

Layout::

    <project_dir>/.opendesk/skills/<name>.json

Skill document::

    {
      "name": "open_invoice_portal",
      "description": "Open the billing portal and download the latest invoice",
      "tags": ["browser", "finance"],
      "version": 1,
      "params": {
        "month": {"type": "string", "required": true},
        "dest":  {"type": "string", "default": "~/Downloads"}
      },
      "steps": [
        {"tool": "browser", "params": {"action": "open", "url": "https://…"}},
        {"tool": "ui", "params": {"action": "click", "title": "Invoices"}}
      ],
      "applicability": {"os": ["darwin", "linux"], "apps": ["Google Chrome"]}
    }

Two ideas from the literature are encoded directly:

* **Parameterisation** — steps reference ``{{params}}`` so one skill serves
  many concrete inputs instead of replaying one frozen trajectory.
* **Relevance is not applicability** — :func:`find_skills` returns a
  relevance score *and* the skill's declared applicability so the caller can
  tell "this looks like my task" from "this will work here".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

# Tokens too common to carry signal in a relevance score.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "open", "run", "do", "my", "me", "it", "is", "then", "this", "that",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Paths & persistence
# ---------------------------------------------------------------------------


def skills_dir(project_dir: Path) -> Path:
    d = project_dir / ".opendesk" / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\- ]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name or "skill"


def skill_path(project_dir: Path, name: str) -> Path:
    return skills_dir(project_dir) / f"{_safe_name(name)}.json"


def save_skill(project_dir: Path, name: str, data: dict) -> Path:
    """Persist a skill document, stamping its canonical name into the file."""
    doc = dict(data)
    doc["name"] = doc.get("name") or name
    path = skill_path(project_dir, name)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return path


def load_skill(project_dir: Path, name: str) -> Optional[dict]:
    """Load a skill by exact (safe) name, falling back to substring match."""
    path = skill_path(project_dir, name)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    needle = _safe_name(name)
    for f in sorted(skills_dir(project_dir).glob("*.json")):
        if needle in f.stem or f.stem in needle:
            try:
                return json.loads(f.read_text())
            except Exception:
                pass
    return None


def list_skills(project_dir: Path) -> list[dict]:
    """Return a short summary (name, description, tags, steps) for each skill."""
    results: list[dict] = []
    for f in sorted(skills_dir(project_dir).glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        results.append(
            {
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "tags": list(data.get("tags", [])),
                "params": list((data.get("params") or {}).keys()),
                "steps": len(data.get("steps", [])),
            }
        )
    return results


def delete_skill(project_dir: Path, name: str) -> bool:
    path = skill_path(project_dir, name)
    if path.exists():
        path.unlink()
        return True
    needle = _safe_name(name)
    for f in sorted(skills_dir(project_dir).glob("*.json")):
        if needle in f.stem:
            f.unlink()
            return True
    return False


# ---------------------------------------------------------------------------
# Relevance retrieval
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {
        w for w in _WORD_RE.findall(text.lower())
        if w not in _STOPWORDS and len(w) > 1
    }


def _skill_text(doc: dict) -> str:
    """Flatten the searchable surface of a skill: name, description, tags, tools."""
    parts: list[str] = [
        str(doc.get("name", "")),
        str(doc.get("description", "")),
        " ".join(str(t) for t in doc.get("tags", []) or []),
        " ".join(str(p) for p in (doc.get("params") or {}).keys()),
        " ".join(
            str(step.get("tool", "")) for step in doc.get("steps", []) or []
        ),
    ]
    return " ".join(parts)


def score_skill(query: str, doc: dict) -> float:
    """Jaccard-style overlap between query tokens and the skill's tokens.

    Deliberately simple and dependency-free: exact token overlap is enough to
    shortlist candidates, and the returned *applicability* (see
    :func:`find_skills`) is what disambiguates them.
    """
    q = _tokens(query)
    if not q:
        return 0.0
    s = _tokens(_skill_text(doc))
    if not s:
        return 0.0
    inter = q & s
    if not inter:
        return 0.0
    # Overlap weighted toward coverage of the query.
    return round(len(inter) / len(q), 4)


def find_skills(
    project_dir: Path,
    query: str,
    *,
    tags: Optional[Iterable[str]] = None,
    limit: int = 5,
) -> list[dict]:
    """Rank saved skills against *query* and return the top ``limit``.

    Each result carries a ``score`` plus the skill's ``params`` schema and
    ``applicability`` so the caller can judge whether a relevant-looking skill
    will actually work in the current environment.
    """
    tag_filter = {t.lower() for t in tags} if tags else None
    matches: list[dict] = []

    for f in sorted(skills_dir(project_dir).glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        if tag_filter is not None:
            have = {str(t).lower() for t in doc.get("tags", []) or []}
            if not (tag_filter & have):
                continue
        score = score_skill(query, doc)
        if score <= 0.0:
            continue
        matches.append(
            {
                "name": doc.get("name", f.stem),
                "description": doc.get("description", ""),
                "tags": list(doc.get("tags", []) or []),
                "params": doc.get("params") or {},
                "applicability": doc.get("applicability") or {},
                "steps": len(doc.get("steps", []) or []),
                "score": score,
            }
        )

    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches[: max(1, limit)]


# ---------------------------------------------------------------------------
# Parameter binding
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([\w.\-]+)\s*\}\}")


class SkillParamError(ValueError):
    """Raised when required skill parameters are missing or malformed."""


def bind_params(doc: dict, arguments: dict[str, Any]) -> dict[str, Any]:
    """Merge caller-supplied *arguments* with declared defaults.

    Raises :class:`SkillParamError` when a required parameter is missing.
    """
    declared: dict[str, dict] = doc.get("params") or {}
    bindings: dict[str, Any] = {}

    for key, spec in declared.items():
        spec = spec or {}
        if key in arguments and arguments[key] is not None:
            bindings[key] = arguments[key]
        elif "default" in spec:
            bindings[key] = spec["default"]
        elif spec.get("required"):
            raise SkillParamError(f"missing required parameter: {key!r}")

    # Allow ad-hoc extras (useful for quick experimentation) but keep declared
    # params authoritative.
    for key, value in arguments.items():
        if key not in bindings and value is not None:
            bindings[key] = value

    return bindings


def render(value: Any, bindings: dict[str, Any]) -> Any:
    """Recursively substitute ``{{placeholder}}`` tokens in a JSON-like value.

    A string that is *exactly* one placeholder adopts the bound value's type
    (so ``"{{count}}"`` can yield an ``int``); mixed strings are stringified.
    """
    if isinstance(value, str):
        whole = _PLACEHOLDER_RE.fullmatch(value.strip())
        if whole:
            key = whole.group(1)
            if key not in bindings:
                raise SkillParamError(f"unbound placeholder: {key!r}")
            return bindings[key]

        def _sub(m: re.Match) -> str:
            key = m.group(1)
            if key not in bindings:
                raise SkillParamError(f"unbound placeholder: {key!r}")
            return str(bindings[key])

        return _PLACEHOLDER_RE.sub(_sub, value)

    if isinstance(value, dict):
        return {k: render(v, bindings) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, bindings) for v in value]
    return value


def render_steps(doc: dict, bindings: dict[str, Any]) -> list[dict]:
    """Return the skill's steps with all placeholders resolved."""
    steps = doc.get("steps") or []
    if not isinstance(steps, list):
        raise SkillParamError("skill 'steps' must be a list")
    return [render(step, bindings) for step in steps]
