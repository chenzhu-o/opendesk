"""Compiled learn programs — deterministic, LLM-free replay of learned tasks.

A *program* is the compiled form of a learned task.  It lives next to the
natural-language procedure::

    <project_dir>/.opendesk/learned/<name>.json          # NL procedure (LLM replay)
    <project_dir>/.opendesk/learned/<name>.program.json  # compiled program

Each step targets UI elements by **semantic identity** (app, role, title,
accessibility identifier, optional ``ref``) instead of pixel coordinates, and
may carry a **postcondition** that is verified (with retries) after the step
runs.  When a step cannot be resolved or its postcondition never holds, the
run stops and returns a structured *self-heal* result — failed step, reason,
screenshot, remaining steps and the original NL procedure — so an agent can
finish the task by hand and then ``patch`` the program.

Programs come from two sources:

* :func:`compile_trajectory` — a recording made with ``learn(start/stop)``
  (the recorder captures accessibility context at every click and keystroke).
* :func:`compile_audit_log` — the current session's audit log
  (``replay_params`` recorded by every input tool), via
  ``learn(action="save_session")``.

Example program::

    {
      "name": "new-note", "version": 1, "source": "trajectory",
      "steps": [
        {"id": "s1", "tool": "app", "params": {"action": "open", "name": "TextEdit"},
         "target": {"app": "TextEdit"},
         "postcondition": {"kind": "app_frontmost", "value": "TextEdit"},
         "wait_ms": 500},
        {"id": "s2", "tool": "ui", "params": {"action": "click"},
         "target": {"app": "TextEdit", "role": "AXButton", "title": "New Document"},
         "postcondition": {"kind": "window_title_contains", "value": "Untitled"},
         "wait_ms": 300}
      ]
    }
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

PROGRAM_VERSION = 1

PostconditionKind = Literal[
    "app_frontmost", "window_title_contains", "element_exists", "text_visible",
]
StepTool = Literal["ui", "keyboard", "app", "mouse", "clipboard", "wait"]

# ui actions that need a resolved element; everything else acts on the
# focused element / app and is dispatched straight to the tool.
_ELEMENT_UI_ACTIONS = frozenset({"click", "get_value"})
_ELEMENT_MOUSE_ACTIONS = frozenset({
    "click", "double_click", "triple_click", "right_click", "middle_click", "move",
})

PROGRAM_SCHEMA_HINT = """Program step schema (JSON):
  {"id": "s1",
   "tool": "ui" | "keyboard" | "app" | "mouse" | "clipboard" | "wait",
   "params": {...tool params, e.g. {"action": "click"} / {"action": "type", "text": "hi"} /
              {"action": "press_key", "key": "s", "modifiers": ["command"]} /
              {"action": "open", "name": "TextEdit"} / {"ms": 500} for wait},
   "target": {"app": "TextEdit", "role": "button", "title": "Save", "identifier": "...", "ref": "..."},
   "postcondition": {"kind": "app_frontmost" | "window_title_contains" | "element_exists" | "text_visible",
                     "value": "..." , "timeout_ms": 5000},
   "wait_ms": 300}
Target elements by role/title/identifier — never by pixel coordinates."""

# Recorder modifier names -> ui tool modifier names.
_MOD_NAMES = {"cmd": "command", "alt": "option", "ctrl": "ctrl", "shift": "shift"}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Target(BaseModel):
    """Semantic identity of the element (or app) a step acts on."""

    model_config = ConfigDict(extra="ignore")

    app: Optional[str] = None
    role: Optional[str] = None
    title: Optional[str] = None
    identifier: Optional[str] = None
    ref: Optional[str] = None

    def is_element(self) -> bool:
        return bool(self.role or self.title or self.identifier or self.ref)

    def describe(self) -> str:
        parts: list[str] = []
        if self.role:
            parts.append(f"[{self.role}]")
        if self.title:
            parts.append(repr(self.title))
        if self.identifier:
            parts.append(f"#{self.identifier}")
        if self.ref:
            parts.append(f"ref={self.ref}")
        if self.app:
            parts.append(f"in {self.app}")
        return " ".join(parts) or "(no target)"


class Postcondition(BaseModel):
    """What must be true after a step runs.

    ``value`` is a string for every kind; ``element_exists`` also accepts a
    target-like dict (``{"role": ..., "title": ..., "identifier": ...}``).
    """

    model_config = ConfigDict(extra="ignore")

    kind: PostconditionKind
    value: Union[str, dict[str, Any]]
    timeout_ms: int = 5000

    def describe(self) -> str:
        if isinstance(self.value, str):
            return f"{self.kind}={self.value!r}"
        return f"{self.kind}={json.dumps(self.value, ensure_ascii=False)}"


class Step(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    tool: StepTool
    params: dict[str, Any] = Field(default_factory=dict)
    target: Target = Field(default_factory=Target)
    postcondition: Optional[Postcondition] = None
    wait_ms: int = 300
    note: str = ""

    def describe(self) -> str:
        """One human-readable line (used by show, self-heal and NL fallbacks)."""
        p = self.params
        action = p.get("action", "")
        tgt = self.target
        if self.tool == "app":
            text = f"{action.capitalize() or 'Use'} the {p.get('name') or tgt.app} app"
        elif self.tool == "ui" and action == "click":
            text = f"Click {tgt.describe()}"
        elif self.tool == "ui" and action == "click_menu":
            text = f"Choose menu {p.get('menu')} → {p.get('menu_item')} in {tgt.app}"
        elif action == "type" or (self.tool == "ui" and action == "type"):
            where = f" into {tgt.describe()}" if tgt.is_element() else (f" in {tgt.app}" if tgt.app else "")
            text = f"Type {p.get('text', '')!r}{where}"
        elif self.tool == "ui" and action == "press_key":
            mods = "+".join(p.get("modifiers") or [])
            combo = f"{mods}+{p.get('key')}" if mods else str(p.get("key"))
            text = f"Press {combo}" + (f" in {tgt.app}" if tgt.app else "")
        elif self.tool == "keyboard" and action == "hotkey":
            text = f"Press {'+'.join(p.get('keys') or [])}"
        elif self.tool == "keyboard" and action == "press":
            text = f"Press {p.get('key')}"
        elif self.tool == "mouse":
            if tgt.is_element():
                text = f"{action.replace('_', ' ').capitalize()} {tgt.describe()}"
            else:
                text = f"Mouse {action} at ({p.get('x')}, {p.get('y')})"
        elif self.tool == "clipboard":
            text = f"Set clipboard to {p.get('text', '')!r}"
        elif self.tool == "wait":
            text = f"Wait {p.get('ms', self.wait_ms)} ms"
        else:
            text = f"{self.tool} {json.dumps(p, ensure_ascii=False)}"
        if self.postcondition is not None:
            text += f"  ⇒ expect {self.postcondition.describe()}"
        return text


class Program(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    version: int = PROGRAM_VERSION
    description: str = ""
    source: str = "manual"  # "trajectory" | "session" | "manual"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    steps: list[Step] = Field(default_factory=list)

    def index_of(self, step_id: str) -> int:
        for i, s in enumerate(self.steps):
            if s.id == step_id:
                return i
        raise KeyError(f"No step with id {step_id!r}. Ids: {[s.id for s in self.steps]}")

    def next_id(self) -> str:
        n = 0
        for s in self.steps:
            if s.id.startswith("s") and s.id[1:].isdigit():
                n = max(n, int(s.id[1:]))
        return f"s{n + 1}"

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=True)

    def describe(self) -> str:
        lines = [f"Program '{self.name}' — {len(self.steps)} step(s), source={self.source}"]
        if self.description:
            lines.append(f"Goal: {self.description}")
        for s in self.steps:
            lines.append(f"  {s.id:>4}. [{s.tool}] {s.describe()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def save_program(project_dir: Path, program: Program) -> Path:
    from opendesk.automation.storage import program_path

    program.updated_at = time.time()
    path = program_path(project_dir, program.name)
    path.write_text(program.to_json())
    return path


def load_program(project_dir: Path, task_name: str) -> Optional[Program]:
    from opendesk.automation.storage import find_sidecar

    path = find_sidecar(project_dir, task_name, "program")
    if path is None:
        return None
    try:
        return Program.model_validate_json(path.read_text())
    except Exception:
        return None


def parse_program(raw: Union[str, dict, list], *, name: str) -> Program:
    """Validate an agent-authored program: a full program dict or a bare step list."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, list):
        data = {"name": name, "steps": data}
    if not isinstance(data, dict):
        raise ValueError("program must be a JSON object or a list of steps")
    data = dict(data)
    data["name"] = name
    data.setdefault("source", "manual")
    steps = data.get("steps") or []
    for i, s in enumerate(steps, 1):
        if isinstance(s, dict) and "id" not in s:
            s["id"] = f"s{i}"
    prog = Program.model_validate(data)
    ids = [s.id for s in prog.steps]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate step ids: {ids}")
    return prog


# ---------------------------------------------------------------------------
# Compilation — trajectory
# ---------------------------------------------------------------------------

_DOUBLE_CLICK_S = 0.5
_DOUBLE_CLICK_PX = 4


def _clamp_wait(gap_s: float, lo: int = 100, hi: int = 1000) -> int:
    return int(min(max(gap_s * 1000.0, lo), hi))


def compile_trajectory(trajectory: Any, *, name: Optional[str] = None, description: str = "") -> Program:
    """Compile a recorded :class:`~opendesk.automation.trajectory.Trajectory` (or its dict)."""
    from opendesk.automation.trajectory import (
        Trajectory, _is_learn_command, trajectory_from_dict,
    )

    traj: Trajectory = trajectory if isinstance(trajectory, Trajectory) else trajectory_from_dict(trajectory)
    events = [e for e in traj.events if not _is_learn_command(e)]

    # Merge rapid same-spot clicks into double clicks.
    merged: list[tuple[Any, str]] = []  # (event, action override)
    for ev in events:
        if (
            merged
            and ev.action_type == "click"
            and merged[-1][0].action_type == "click"
            and merged[-1][1] == "click"
            and ev.x is not None and merged[-1][0].x is not None
            and abs(ev.x - merged[-1][0].x) <= _DOUBLE_CLICK_PX
            and abs((ev.y or 0) - (merged[-1][0].y or 0)) <= _DOUBLE_CLICK_PX
            and ev.timestamp - merged[-1][0].timestamp <= _DOUBLE_CLICK_S
        ):
            merged[-1] = (merged[-1][0], "double_click")
            continue
        merged.append((ev, ev.action_type))

    prog = Program(name=name or traj.task_name, description=description, source="trajectory")
    seen_apps: set[str] = set()
    current_app: Optional[str] = None

    def add(step_kwargs: dict) -> Step:
        step = Step(id=f"s{len(prog.steps) + 1}", **step_kwargs)
        prog.steps.append(step)
        return step

    for i, (ev, action) in enumerate(merged):
        ax = ev.accessibility_context or {}
        app = ax.get("app") or current_app
        nxt = merged[i + 1][0] if i + 1 < len(merged) else None
        gap = (nxt.timestamp - ev.timestamp) if nxt is not None else 0.3
        nax = (nxt.accessibility_context or {}) if nxt is not None else {}

        if app and app != current_app:
            add({
                "tool": "app",
                "params": {"action": "focus" if app in seen_apps else "open", "name": app},
                "target": Target(app=app),
                "postcondition": Postcondition(kind="app_frontmost", value=app),
                "wait_ms": 500,
            })
            seen_apps.add(app)
            current_app = app

        target = Target(
            app=app, role=ax.get("role") or None, title=ax.get("title") or None,
            identifier=ax.get("identifier") or None,
        )
        post: Optional[Postcondition] = None
        if (
            nax.get("window") and nax.get("window") != ax.get("window")
            and (not nax.get("app") or nax.get("app") == app)
        ):
            post = Postcondition(kind="window_title_contains", value=nax["window"])

        wait_ms = _clamp_wait(gap)

        if action in ("click", "right_click", "double_click"):
            if target.is_element():
                if action == "click":
                    add({"tool": "ui", "params": {"action": "click"}, "target": target,
                         "postcondition": post, "wait_ms": wait_ms})
                else:
                    add({"tool": "mouse", "params": {"action": action}, "target": target,
                         "postcondition": post, "wait_ms": wait_ms})
            else:
                add({
                    "tool": "mouse", "params": {"action": action, "x": ev.x, "y": ev.y},
                    "target": Target(app=app), "postcondition": post, "wait_ms": wait_ms,
                    "note": "coordinate fallback — no accessibility info was recorded for this click",
                })
        elif action == "type" and ev.text:
            if app:
                add({"tool": "ui", "params": {"action": "type", "text": ev.text}, "target": target,
                     "postcondition": post, "wait_ms": wait_ms})
            else:
                add({"tool": "keyboard", "params": {"action": "type", "text": ev.text},
                     "postcondition": post, "wait_ms": wait_ms})
        elif action == "key" and ev.key:
            parts = ev.key.split("+")
            key, mods = parts[-1], [_MOD_NAMES.get(m, m) for m in parts[:-1]]
            if app:
                params: dict[str, Any] = {"action": "press_key", "key": key}
                if mods:
                    params["modifiers"] = mods
                add({"tool": "ui", "params": params, "target": Target(app=app),
                     "postcondition": post, "wait_ms": wait_ms})
            elif mods:
                add({"tool": "keyboard", "params": {"action": "hotkey", "keys": parts},
                     "postcondition": post, "wait_ms": wait_ms})
            else:
                add({"tool": "keyboard", "params": {"action": "press", "key": key},
                     "postcondition": post, "wait_ms": wait_ms})
        elif action == "scroll":
            dy, dx = ev.scroll_dy or 0, ev.scroll_dx or 0
            if dy:
                direction, amount = ("down" if dy < 0 else "up"), abs(int(dy))
            else:
                direction, amount = ("left" if dx < 0 else "right"), abs(int(dx))
            add({
                "tool": "mouse",
                "params": {"action": "scroll", "x": ev.x, "y": ev.y,
                           "direction": direction, "amount": max(amount, 1)},
                "target": Target(app=app), "wait_ms": wait_ms,
                "note": "scroll position is coordinate-based",
            })
    return prog


# ---------------------------------------------------------------------------
# Compilation — session audit log
# ---------------------------------------------------------------------------

_AUDIT_SKIP = frozenset({"screenshot", "cursor_position", "app_list", "clipboard_read", "ocr"})
_KEYBOARD_DEFAULTS = {"interval": 0.02, "hold_duration": 1.0}
_MOUSE_DEFAULTS = {"amount": 3, "duration": 0.25, "settle_ms": 500}


def _drop_empty(d: dict, defaults: Optional[dict] = None) -> dict:
    defaults = defaults or {}
    return {
        k: v for k, v in d.items()
        if v is not None and v != [] and not (k in defaults and defaults[k] == v)
    }


def compile_audit_log(
    log: list[dict], *, name: str, description: str = "", skip_errors: bool = True,
) -> Program:
    """Compile a sandbox audit log (``export_audit_log()``) into a program."""
    prog = Program(name=name, description=description, source="session")
    current_app: Optional[str] = None
    entries = [
        e for e in log
        if e.get("action") not in _AUDIT_SKIP
        and e.get("replay_params") is not None
        and not (skip_errors and e.get("error"))
    ]

    for i, e in enumerate(entries):
        rp = dict(e["replay_params"])
        action_type = e["action"]
        nxt = entries[i + 1] if i + 1 < len(entries) else None
        gap = (nxt["timestamp"] - e["timestamp"]) if nxt is not None else 0.3
        wait_ms = _clamp_wait(gap)
        step: Optional[dict] = None

        if action_type == "ui_action":
            act = rp.get("action")
            if act in ("get_tree",):
                continue
            app = rp.pop("app", None) or current_app
            current_app = app or current_app
            target = Target(app=app, role=rp.pop("role", None), title=rp.pop("title", None))
            params = _drop_empty(rp, {"window_index": 1})
            if act == "get_value":
                # A read — becomes an existence check rather than an action.
                step = {"tool": "wait", "params": {"ms": 0}, "target": target,
                        "postcondition": Postcondition(kind="element_exists", value=_target_value(target))}
            else:
                step = {"tool": "ui", "params": params, "target": target}
        elif action_type.startswith("keyboard_"):
            step = {"tool": "keyboard", "params": _drop_empty(rp, _KEYBOARD_DEFAULTS),
                    "target": Target(app=current_app)}
        elif action_type.startswith("app_"):
            act = rp.get("action")
            name_ = rp.get("name")
            step = {"tool": "app", "params": _drop_empty(rp), "target": Target(app=name_)}
            if act in ("open", "focus") and name_:
                step["postcondition"] = Postcondition(kind="app_frontmost", value=name_)
                current_app = name_
        elif action_type.startswith("mouse_"):
            step = {"tool": "mouse", "params": _drop_empty(rp, _MOUSE_DEFAULTS),
                    "target": Target(app=current_app),
                    "note": "coordinate-based — consider patching to a ui click"}
        elif action_type == "clipboard_write":
            step = {"tool": "clipboard", "params": _drop_empty(rp)}
        if step is None:
            continue
        step.setdefault("wait_ms", wait_ms)
        prog.steps.append(Step(id=f"s{len(prog.steps) + 1}", **step))
    return prog


def _target_value(t: Target) -> dict:
    return {k: v for k, v in t.model_dump().items() if v and k != "app"}


def procedure_from_program(program: Program) -> dict:
    """A minimal NL procedure for programs that have none (LLM fallback / self-heal)."""
    steps = [s.describe() for s in program.steps]
    return {
        "task_name": program.name,
        "description": program.description or f"Compiled task '{program.name}'",
        "steps": steps,
        "procedure": " Then ".join(steps),
    }


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

PatchMode = Literal["replace", "insert_before", "insert_after", "append", "delete"]


def patch_program(
    program: Program, *, mode: PatchMode = "replace", step_id: Optional[str] = None,
    step: Optional[Union[str, dict]] = None,
) -> Program:
    """Return *program* with one step replaced / inserted / deleted (in place)."""
    new_step: Optional[Step] = None
    if mode != "delete":
        if step is None:
            raise ValueError(f"'step' JSON is required for mode={mode!r}")
        raw = json.loads(step) if isinstance(step, str) else dict(step)
        if mode == "replace":
            if not step_id:
                raise ValueError("step_id is required for mode='replace'")
            raw["id"] = step_id
        else:
            raw.setdefault("id", program.next_id())
            if any(s.id == raw["id"] for s in program.steps):
                raw["id"] = program.next_id()
        new_step = Step.model_validate(raw)

    if mode == "append":
        program.steps.append(new_step)  # type: ignore[arg-type]
    else:
        if not step_id:
            raise ValueError(f"step_id is required for mode={mode!r}")
        idx = program.index_of(step_id)
        if mode == "replace":
            program.steps[idx] = new_step  # type: ignore[assignment]
        elif mode == "insert_before":
            program.steps.insert(idx, new_step)  # type: ignore[arg-type]
        elif mode == "insert_after":
            program.steps.insert(idx + 1, new_step)  # type: ignore[arg-type]
        elif mode == "delete":
            program.steps.pop(idx)
    program.updated_at = time.time()
    return program


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class StepOutcome:
    step_id: str
    ok: bool
    detail: str
    elapsed_ms: int


@dataclass
class RunResult:
    program: Program
    status: Literal["ok", "failed"]
    outcomes: list[StepOutcome] = field(default_factory=list)
    failed_step: Optional[Step] = None
    reason: str = ""
    remaining: list[Step] = field(default_factory=list)
    screenshot: Optional[bytes] = None
    screenshot_media_type: str = "image/png"
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def summary(self) -> str:
        n_ok = sum(1 for o in self.outcomes if o.ok)
        if self.ok:
            return f"ok: replayed {n_ok} step(s) of '{self.program.name}' (compiled, {self.elapsed_ms} ms)"
        return (
            f"failed at step {self.failed_step.id if self.failed_step else '?'} "
            f"after {n_ok} OK step(s): {self.reason}"
        )

    def self_heal(self, procedure: Optional[dict] = None) -> dict:
        """Structured hand-off for an agent to finish the task and patch the program."""
        return {
            "status": self.status,
            "program": self.program.name,
            "completed_steps": [o.step_id for o in self.outcomes if o.ok],
            "failed_step": self.failed_step.model_dump(exclude_none=True) if self.failed_step else None,
            "reason": self.reason,
            "remaining_steps": [s.model_dump(exclude_none=True) for s in self.remaining],
            "procedure": procedure,
        }


class StepFailed(RuntimeError):
    pass


def _norm_role(role: Optional[str]) -> str:
    r = (role or "").lower().replace(" ", "").replace("_", "")
    return r[2:] if r.startswith("ax") else r


def _flatten(root: Any) -> list[Any]:
    out = [root]
    for c in getattr(root, "children", []) or []:
        out.extend(_flatten(c))
    return out


def _node_identifier(node: Any) -> Optional[str]:
    md = getattr(node, "metadata", None) or {}
    for key in ("identifier", "AXIdentifier", "automation_id", "auto_id"):
        if md.get(key):
            return str(md[key])
    return getattr(node, "identifier", None)


def _node_ref(node: Any) -> Optional[str]:
    ref = getattr(node, "ref", None)
    if ref is None:
        ref = (getattr(node, "metadata", None) or {}).get("ref")
    return str(ref) if ref is not None else None


def _role_ok(node: Any, role: Optional[str]) -> bool:
    if not role:
        return True
    want, have = _norm_role(role), _norm_role(getattr(node, "role", ""))
    return bool(have) and (want == have or want in have or have in want)


def _title_score(node: Any, title: Optional[str]) -> int:
    """2 = exact (case-insensitive), 1 = substring, 0 = no match. No title → 2."""
    if not title:
        return 2
    t = title.strip().lower()
    for cand in (getattr(node, "name", "") or "", getattr(node, "value", "") or ""):
        c = str(cand).strip().lower()
        if c == t:
            return 2
    for cand in (getattr(node, "name", "") or "", getattr(node, "value", "") or ""):
        if t and t in str(cand).lower():
            return 1
    return 0


def find_element(root: Any, target: Target) -> Optional[Any]:
    """Resolve *target* inside a UI tree by semantic identity.

    Order: ``ref`` (only when the node also agrees with role/title), then the
    accessibility ``identifier``, then role + exact title, then role +
    substring title.  Actionable nodes (bounds or actions) win ties.
    """
    nodes = _flatten(root)

    def actionable(n: Any) -> bool:
        return getattr(n, "bounds", None) is not None or bool(getattr(n, "actions", None))

    if target.ref:
        for n in nodes:
            if _node_ref(n) == target.ref and _role_ok(n, target.role) and _title_score(n, target.title):
                return n
    if target.identifier:
        for n in nodes:
            if _node_identifier(n) == target.identifier:
                return n
    if not (target.role or target.title):
        return None
    best: Optional[Any] = None
    best_score = -1
    for n in nodes:
        if n is root and len(nodes) > 1:
            continue
        if not _role_ok(n, target.role):
            continue
        ts = _title_score(n, target.title)
        if ts == 0:
            continue
        score = ts * 2 + (1 if actionable(n) else 0)
        if score > best_score:
            best, best_score = n, score
    return best


def _tree_has_text(root: Any, text: str) -> bool:
    t = text.lower()
    for n in _flatten(root):
        for cand in (getattr(n, "name", ""), getattr(n, "value", "")):
            if cand and t in str(cand).lower():
                return True
    return False


Sleep = Callable[[float], Awaitable[None]]


class ProgramRunner:
    """Execute a :class:`Program` against ``ctx.computer`` — no LLM involved."""

    def __init__(
        self,
        ctx: Any,
        *,
        resolve_timeout_ms: int = 5000,
        poll_ms: int = 250,
        ocr_fallback: bool = True,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.ctx = ctx
        self.resolve_timeout_ms = resolve_timeout_ms
        self.poll_ms = poll_ms
        self.ocr_fallback = ocr_fallback
        self._sleep = sleep

    @property
    def comp(self) -> Any:
        return self.ctx.computer

    async def run(self, program: Program, *, start_at: Optional[str] = None) -> RunResult:
        t0 = time.monotonic()
        steps = program.steps
        if start_at:
            steps = steps[program.index_of(start_at):]
        result = RunResult(program=program, status="ok")
        for i, step in enumerate(steps):
            s0 = time.monotonic()
            try:
                detail = await self._run_step(step)
                if step.wait_ms > 0:
                    await self._sleep(step.wait_ms / 1000.0)
                if step.postcondition is not None:
                    await self._verify(step, step.postcondition)
                    detail += f"; verified {step.postcondition.kind}"
            except Exception as exc:  # StepFailed or a backend error
                result.status = "failed"
                result.failed_step = step
                result.reason = str(exc) or type(exc).__name__
                result.remaining = list(steps[i + 1:])
                result.outcomes.append(StepOutcome(step.id, False, result.reason, _ms(s0)))
                await self._attach_screenshot(result)
                break
            result.outcomes.append(StepOutcome(step.id, True, detail, _ms(s0)))
        result.elapsed_ms = _ms(t0)
        return result

    # -- steps --------------------------------------------------------

    async def _resolve(self, step: Step) -> Any:
        target = step.target
        deadline = time.monotonic() + self.resolve_timeout_ms / 1000.0
        last_err = ""
        while True:
            try:
                tree = await self.comp.ui_tree(app=target.app)
                el = find_element(tree, target)
                if el is not None:
                    return el
                last_err = f"no element matching {target.describe()}"
            except Exception as exc:
                last_err = f"ui_tree failed: {exc}"
            if time.monotonic() >= deadline:
                raise StepFailed(f"target not found after {self.resolve_timeout_ms} ms: {last_err}")
            await self._sleep(self.poll_ms / 1000.0)

    async def _run_step(self, step: Step) -> str:
        p = dict(step.params)
        action = p.get("action")
        tgt = step.target

        if step.tool == "wait":
            ms = int(p.get("ms", 0))
            if ms:
                await self._sleep(ms / 1000.0)
            return f"waited {ms} ms"

        if step.tool == "ui" and action in _ELEMENT_UI_ACTIONS and tgt.is_element():
            el = await self._resolve(step)
            if action == "get_value":
                return f"value={getattr(el, 'value', None) or getattr(el, 'name', '')!r}"
            from opendesk.tools.ui import _try_perform_action
            via = "a11y"
            if not await _try_perform_action(self.comp, el, "click", tgt.app):
                if getattr(el, "bounds", None) is None:
                    raise StepFailed(f"element {tgt.describe()} has no bounds and no native action")
                await self.comp.click(el.bounds.center)
                via = "bounds-center"
            await self._audit_ui(step, f"clicked {tgt.describe()} via {via}")
            return f"clicked {tgt.describe()} via {via}"

        if step.tool == "mouse" and action in _ELEMENT_MOUSE_ACTIONS and tgt.is_element():
            el = await self._resolve(step)
            if getattr(el, "bounds", None) is None:
                raise StepFailed(f"element {tgt.describe()} has no bounds")
            from opendesk.computer.types import PointerAction, PointerButton, PointerEvent
            pt = el.bounds.center
            if action == "move":
                await self.comp.pointer(PointerEvent(action=PointerAction.MOVE, point=pt))
            else:
                button = {
                    "right_click": PointerButton.RIGHT, "middle_click": PointerButton.MIDDLE,
                }.get(action, PointerButton.LEFT)
                count = {"double_click": 2, "triple_click": 3}.get(action, 1)
                await self.comp.click(pt, button=button, count=count)
            await self._audit_ui(step, f"{action} {tgt.describe()}")
            return f"{action} on {tgt.describe()} at ({int(pt.x)}, {int(pt.y)})"

        # Everything else goes through the real tool (audited, permission-checked).
        if step.tool == "ui":
            p.setdefault("app", tgt.app)
            if tgt.title and "title" not in p:
                p["title"] = tgt.title
            if tgt.role and "role" not in p:
                p["role"] = tgt.role
            if p.get("action") == "type":
                p.pop("title", None)
                p.pop("role", None)
            if not p.get("app"):
                raise StepFailed("ui step has no app (set target.app)")
        tool = _load_tool(step.tool)
        res = await tool.execute(self.ctx, tool.parse_params(p))
        if res.error:
            raise StepFailed(f"{step.tool} tool error: {res.output[:300]}")
        return res.output[:200]

    async def _audit_ui(self, step: Step, msg: str) -> None:
        try:
            from opendesk.computer.sandbox import ActionType, get_sandbox
            t = step.target
            await get_sandbox(self.ctx.session_id).record_action(
                ActionType.UI_ACTION if step.tool == "ui" else ActionType.MOUSE_CLICK,
                {"action": step.params.get("action"), "app": t.app, "title": t.title, "program_step": step.id},
                result=msg[:200],
                replay_params={"action": "click", "app": t.app, "title": t.title, "role": t.role}
                if step.tool == "ui" else None,
            )
        except Exception:
            pass

    # -- postconditions -------------------------------------------------

    async def check(self, step: Step, post: Postcondition, *, final: bool = False) -> tuple[bool, str]:
        kind, value = post.kind, post.value
        if kind in ("app_frontmost", "window_title_contains"):
            win = await self.comp.focused_window()
            if win is None:
                return False, "no focused window"
            want = str(value).lower()
            if kind == "app_frontmost":
                have = (win.app_name or "").lower().removesuffix(".app")
                ok = bool(have) and (want.removesuffix(".app") in have or have in want)
                return ok, f"frontmost app is {win.app_name!r}"
            titles = [win.title or ""]
            if want not in titles[0].lower():
                try:
                    tree = await self.comp.ui_tree(app=step.target.app or win.app_name)
                    titles.append(getattr(tree, "name", "") or "")
                except Exception:
                    pass
            ok = any(want in t.lower() for t in titles)
            return ok, f"window title is {titles[-1]!r}"
        if kind == "element_exists":
            tgt = Target(**value) if isinstance(value, dict) else Target(title=str(value))
            if not tgt.app:
                tgt.app = step.target.app
            tree = await self.comp.ui_tree(app=tgt.app)
            ok = find_element(tree, tgt) is not None
            return ok, f"element {tgt.describe()} {'found' if ok else 'not found'}"
        if kind == "text_visible":
            text = str(value)
            tree = await self.comp.ui_tree(app=step.target.app)
            if _tree_has_text(tree, text):
                return True, "text found in accessibility tree"
            if final and self.ocr_fallback:
                try:
                    from opendesk.computer.ocr import ocr_image
                    pix = await self.comp.capture()
                    ocr = await asyncio.to_thread(ocr_image, pix.data, width=pix.width, height=pix.height)
                    if text.lower() in (ocr or "").lower():
                        return True, "text found via OCR"
                except Exception:
                    pass
            return False, f"text {text!r} not visible"
        return False, f"unknown postcondition kind {kind!r}"

    async def _verify(self, step: Step, post: Postcondition) -> None:
        deadline = time.monotonic() + max(post.timeout_ms, 0) / 1000.0
        detail = ""
        while True:
            final = time.monotonic() >= deadline
            try:
                ok, detail = await self.check(step, post, final=final)
            except Exception as exc:
                ok, detail = False, f"check error: {exc}"
            if ok:
                return
            if final:
                raise StepFailed(
                    f"postcondition {post.describe()} not met within {post.timeout_ms} ms ({detail})"
                )
            await self._sleep(self.poll_ms / 1000.0)

    async def _attach_screenshot(self, result: RunResult) -> None:
        try:
            pix = await self.comp.capture()
            result.screenshot = pix.data
            fmt = getattr(pix.format, "value", str(pix.format))
            result.screenshot_media_type = "image/jpeg" if fmt == "jpeg" else "image/png"
        except Exception:
            result.screenshot = None


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


_TOOL_CLASSES = {
    "mouse": "opendesk.tools.mouse:MouseTool",
    "keyboard": "opendesk.tools.keyboard:KeyboardTool",
    "app": "opendesk.tools.app:AppTool",
    "clipboard": "opendesk.tools.clipboard:ClipboardTool",
    "ui": "opendesk.tools.ui:UITool",
}


def _load_tool(name: str) -> Any:
    import importlib
    module_path, cls_name = _TOOL_CLASSES[name].split(":")
    return getattr(importlib.import_module(module_path), cls_name)()


async def run_program(program: Program, ctx: Any, *, start_at: Optional[str] = None, **kwargs: Any) -> RunResult:
    """Convenience wrapper around :class:`ProgramRunner`."""
    return await ProgramRunner(ctx, **kwargs).run(program, start_at=start_at)


def format_self_heal(result: RunResult, procedure: Optional[dict]) -> str:
    """Agent-facing text for a failed run."""
    fs = result.failed_step
    lines = [
        f"Compiled replay of '{result.program.name}' STOPPED at step {fs.id if fs else '?'}.",
        f"Reason: {result.reason}",
    ]
    done = [o.step_id for o in result.outcomes if o.ok]
    lines.append(f"Completed steps: {', '.join(done) if done else '(none)'}")
    if fs is not None:
        lines.append(f"Failed step: {fs.describe()}")
        lines.append(f"  {fs.model_dump_json(exclude_none=True)}")
    if result.remaining:
        lines.append("Remaining steps:")
        for s in result.remaining:
            lines.append(f"  {s.id}. {s.describe()}")
    if procedure:
        lines.append("")
        lines.append(f"Original procedure — goal: {procedure.get('description', '')}")
        for i, s in enumerate(procedure.get("steps", []), 1):
            lines.append(f"  {i}. {s}")
        if procedure.get("procedure"):
            lines.append(f"  {procedure['procedure']}")
    lines += [
        "",
        "Self-heal: look at the attached screenshot, finish the failed and remaining "
        "steps yourself with the ui/keyboard/app tools, then fix the program with "
        f"learn(action='patch', task_name='{result.program.name}', step_id='{fs.id if fs else ''}', "
        "step='<step JSON>') so the next run succeeds without you. "
        f"Resume a partial run with learn(action='run', task_name='{result.program.name}', start_at='<step id>').",
    ]
    return "\n".join(lines)
