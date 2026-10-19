"""Storage for learned task procedures.

All procedures are saved as JSON files under:
    <project_dir>/.opendesk/learned/<task_name>.json

Sidecar files live next to them and are never listed as procedures:
    <task_name>.program.json     — compiled, deterministic program (see program.py)
    <task_name>.trajectory.json  — raw recorded trajectory (no screenshots)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


def procedures_dir(project_dir: Path) -> Path:
    d = project_dir / ".opendesk" / "learned"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(task_name: str) -> str:
    name = task_name.lower().strip()
    name = re.sub(r"[^\w\- ]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name or "task"


def procedure_path(project_dir: Path, task_name: str) -> Path:
    return procedures_dir(project_dir) / f"{_safe_name(task_name)}.json"


def _is_procedure_file(f: Path) -> bool:
    # Procedure stems are sanitised by _safe_name and never contain a dot, so
    # any dotted stem (foo.program, foo.trajectory) is a sidecar file.
    return "." not in f.stem


def program_path(project_dir: Path, task_name: str) -> Path:
    return procedures_dir(project_dir) / f"{_safe_name(task_name)}.program.json"


def trajectory_path(project_dir: Path, task_name: str) -> Path:
    return procedures_dir(project_dir) / f"{_safe_name(task_name)}.trajectory.json"


def find_sidecar(project_dir: Path, task_name: str, kind: str) -> Optional[Path]:
    """Return ``<name>.<kind>.json`` for *task_name* (exact, then fuzzy match)."""
    exact = procedures_dir(project_dir) / f"{_safe_name(task_name)}.{kind}.json"
    if exact.exists():
        return exact
    needle = _safe_name(task_name)
    for f in sorted(procedures_dir(project_dir).glob(f"*.{kind}.json")):
        if needle in f.name[: -len(f".{kind}.json")]:
            return f
    return None


def save_procedure(project_dir: Path, task_name: str, data: dict) -> Path:
    path = procedure_path(project_dir, task_name)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def load_procedure(project_dir: Path, task_name: str) -> Optional[dict]:
    path = procedure_path(project_dir, task_name)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    needle = _safe_name(task_name)
    for f in sorted(procedures_dir(project_dir).glob("*.json")):
        if _is_procedure_file(f) and needle in f.stem:
            try:
                return json.loads(f.read_text())
            except Exception:
                pass
    return None


def list_procedures(project_dir: Path) -> list:
    results = []
    for f in sorted(procedures_dir(project_dir).glob("*.json")):
        if not _is_procedure_file(f):
            continue
        try:
            data = json.loads(f.read_text())
            results.append({
                "name": f.stem,
                "description": data.get("description", ""),
                "compiled": (f.parent / f"{f.stem}.program.json").exists(),
            })
        except Exception:
            results.append({
                "name": f.stem, "description": "",
                "compiled": (f.parent / f"{f.stem}.program.json").exists(),
            })
    return results
