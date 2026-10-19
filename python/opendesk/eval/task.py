"""Eval task format and loading.

A task is a JSON object::

    {
      "id": "textedit-type-save",
      "instruction": "Type 'hello' into the TextEdit document {workdir}/hello.txt and save it.",
      "setup":    ["printf '' > {workdir}/hello.txt", {"open": "{workdir}/hello.txt", "app": "TextEdit"}],
      "check":    "grep -q hello {workdir}/hello.txt",
      "teardown": ["osascript -e 'tell application \\"TextEdit\\" to close every document saving no'"],
      "timeout":  300,
      "tags":     ["textedit", "smoke"],
      "platform": "darwin"
    }

``setup`` / ``teardown`` entries are shell strings or one of these dicts:

* ``{"shell": "<cmd>"}``
* ``{"open_app": "TextEdit"}`` — launch / activate an application
* ``{"open": "<path or url>", "app": "<optional app>"}``
* ``{"write_file": "<path>", "content": "<text>"}`` — no shell quoting needed
* ``{"sleep": 1.5}``

``check`` is a shell command; exit status 0 means the task passed.

Placeholders ``{workdir}`` (a fresh temp dir per trial, removed afterwards),
``{home}`` and ``{task_id}`` are substituted in every string field.

Learn-regression tasks set ``program`` (and optionally ``program_json``) instead
of relying on an agent: the harness runs the compiled program
deterministically and then evaluates ``check``.

A file may hold one task, a list of tasks, or ``{"suite": "...", "tasks": [...]}``;
a directory loads every ``*.json`` inside it.
"""

from __future__ import annotations

import json
import os
import platform as _platform
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

SetupAction = Union[str, dict]


@dataclass
class EvalTask:
    id: str
    instruction: str = ""
    setup: list[SetupAction] = field(default_factory=list)
    check: Optional[str] = None
    teardown: list[SetupAction] = field(default_factory=list)
    timeout: int = 300
    tags: list[str] = field(default_factory=list)
    platform: Optional[str] = None  # "darwin" | "linux" | "windows"
    max_turns: Optional[int] = None
    program: Optional[str] = None  # learn-regression: compiled program name
    program_json: Optional[dict] = None  # written to {workdir}/.opendesk/learned/ before running
    project: Optional[str] = None  # where to load `program` from (default: {workdir})
    suite: str = ""

    @classmethod
    def from_dict(cls, data: dict, *, suite: str = "") -> "EvalTask":
        if "id" not in data:
            raise ValueError(f"task is missing 'id': {data!r}")
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"task {data['id']!r} has unknown fields: {sorted(unknown)}")
        kwargs = dict(data)
        kwargs.setdefault("suite", suite)
        for key in ("setup", "teardown"):
            v = kwargs.get(key)
            if isinstance(v, (str, dict)):
                kwargs[key] = [v]
        task = cls(**kwargs)
        if not task.program and not task.instruction:
            raise ValueError(f"task {task.id!r} needs an 'instruction' (or a 'program')")
        return task

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "")}

    def supported_here(self) -> bool:
        if not self.platform:
            return True
        return self.platform.lower() == current_platform()

    def is_program_task(self) -> bool:
        return bool(self.program)


def current_platform() -> str:
    return {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}.get(
        _platform.system(), _platform.system().lower(),
    )


def substitute(value: Any, *, workdir: str, task_id: str) -> Any:
    """Replace ``{workdir}``, ``{home}``, ``{task_id}`` in strings (recursively)."""
    if isinstance(value, str):
        return (
            value.replace("{workdir}", workdir)
            .replace("{home}", os.path.expanduser("~"))
            .replace("{task_id}", task_id)
        )
    if isinstance(value, list):
        return [substitute(v, workdir=workdir, task_id=task_id) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, workdir=workdir, task_id=task_id) for k, v in value.items()}
    return value


def action_to_shell(action: SetupAction, *, plat: Optional[str] = None) -> Optional[str]:
    """Shell form of a setup/teardown action (``None`` for non-shell actions)."""
    plat = plat or current_platform()
    if isinstance(action, str):
        return action
    if "shell" in action:
        return str(action["shell"])
    if "open_app" in action:
        app = str(action["open_app"])
        if plat == "darwin":
            return f"open -a {shlex.quote(app)}"
        if plat == "windows":
            return f'start "" {shlex.quote(app)}'
        return f"({shlex.quote(app)} >/dev/null 2>&1 &)"
    if "open" in action:
        target = str(action["open"])
        app = action.get("app")
        if plat == "darwin":
            return f"open -a {shlex.quote(app)} {shlex.quote(target)}" if app else f"open {shlex.quote(target)}"
        if plat == "windows":
            return f'start "" {shlex.quote(target)}'
        return f"(xdg-open {shlex.quote(target)} >/dev/null 2>&1 &)"
    return None


def describe_action(action: SetupAction) -> str:
    sh = action_to_shell(action)
    if sh is not None:
        return sh
    if "write_file" in action:
        n = len(str(action.get("content", "")))
        return f"<write {n} chars to {action['write_file']}>"
    if "sleep" in action:
        return f"<sleep {action['sleep']}s>"
    return json.dumps(action)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _tasks_from_obj(obj: Any, *, suite: str) -> list[EvalTask]:
    if isinstance(obj, list):
        return [EvalTask.from_dict(t, suite=suite) for t in obj]
    if isinstance(obj, dict) and "tasks" in obj:
        name = obj.get("suite", suite)
        return [EvalTask.from_dict(t, suite=name) for t in obj["tasks"]]
    if isinstance(obj, dict):
        return [EvalTask.from_dict(obj, suite=suite)]
    raise ValueError("expected a task object, a list of tasks, or {'tasks': [...]}")


def load_tasks(spec: str, *, project_dir: Optional[Path] = None) -> tuple[str, list[EvalTask]]:
    """Resolve *spec* (built-in suite name, JSON file or directory) to tasks.

    Returns ``(suite_name, tasks)``.
    """
    from opendesk.eval.suites import BUILTIN_SUITES, builtin_suite

    if spec in BUILTIN_SUITES:
        return spec, builtin_suite(spec, project_dir=project_dir)

    path = Path(spec).expanduser()
    if path.is_dir():
        tasks: list[EvalTask] = []
        for f in sorted(path.glob("*.json")):
            tasks.extend(_tasks_from_obj(json.loads(f.read_text()), suite=path.name))
        _check_unique(tasks)
        return path.name, tasks
    if path.is_file():
        obj = json.loads(path.read_text())
        name = obj.get("suite", path.stem) if isinstance(obj, dict) else path.stem
        tasks = _tasks_from_obj(obj, suite=name)
        _check_unique(tasks)
        return name, tasks
    raise FileNotFoundError(
        f"No built-in suite or task file named {spec!r}. Built-in suites: {', '.join(BUILTIN_SUITES)}"
    )


def _check_unique(tasks: list[EvalTask]) -> None:
    ids = [t.id for t in tasks]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate task ids: {sorted(dup)}")
