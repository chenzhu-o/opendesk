"""Automation tools — record, replay, and schedule desktop tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from opendesk.tools.base import Tool, ToolContext, ToolResult

# Module-level recorder singleton — only one recording at a time
_active_recorder: Any = None


class LearnTool(Tool):
    name = "learn"
    title = "Record & Replay Workflows"
    annotations = {
        "readOnlyHint": False, "destructiveHint": True,
        "idempotentHint": False, "openWorldHint": True,
    }
    description = (
        "Record and replay computer tasks. "
        "Actions:\n"
        "- start: begin recording mouse/keyboard/screenshots for a named task\n"
        "- stop: stop recording; returns trajectory summary and screenshots for you to summarize into a procedure, then call learn(save) with the JSON\n"
        "- save: save a procedure JSON returned from stop\n"
        "- replay: load a saved procedure and return step-by-step replay instructions\n"
        "- list: list all saved procedures\n"
        "- compile: build a deterministic program (<name>.program.json) from the recorded "
        "trajectory, from this session's actions (source='session'), or from program JSON you "
        "write from the NL procedure (program='[{\"tool\": ...}]')\n"
        "- save_session: compile the actions taken so far in this session into a program "
        "(and a procedure if none exists) — teach by doing\n"
        "- run: execute the compiled program WITHOUT an LLM, verifying postconditions; on failure "
        "returns the failed step, reason, screenshot, remaining steps and the NL procedure\n"
        "- patch: fix a compiled program — replace/insert/delete one step (step_id, step JSON, mode)\n"
        "- show: print the compiled program\n\n"
        "Typical workflow: learn(start) → user performs task → learn(stop) → "
        "summarize the trajectory → learn(save) → later: learn(run) (fast, deterministic) "
        "or learn(replay) (agent-driven). If run fails, finish the task, then learn(patch)."
    )

    class Params(BaseModel):
        action: Literal[
            "start", "stop", "save", "replay", "list",
            "compile", "save_session", "run", "patch", "show",
        ]
        task_name: Optional[str] = None
        procedure: Optional[str] = None  # JSON string for save action
        program: Optional[str] = Field(
            default=None,
            description="compile: program JSON (object with 'steps', or a list of steps) to save as-is.",
        )
        source: Optional[Literal["trajectory", "session"]] = Field(
            default=None,
            description="compile: where to compile from (default: recorded trajectory).",
        )
        description: Optional[str] = Field(default=None, description="compile/save_session: one-line goal.")
        step_id: Optional[str] = Field(
            default=None, description="patch: id of the step to replace / insert around / delete.",
        )
        step: Optional[str] = Field(
            default=None,
            description=(
                "patch: step JSON, e.g. {\"tool\":\"ui\",\"params\":{\"action\":\"click\"},"
                "\"target\":{\"app\":\"TextEdit\",\"role\":\"button\",\"title\":\"Save\"},"
                "\"postcondition\":{\"kind\":\"window_title_contains\",\"value\":\"Saved\"}}"
            ),
        )
        mode: Optional[Literal["replace", "insert_before", "insert_after", "append", "delete"]] = Field(
            default=None, description="patch: how to apply 'step' (default: replace, or append without step_id).",
        )
        start_at: Optional[str] = Field(default=None, description="run: resume from this step id.")
        session_id: Optional[str] = Field(
            default=None,
            description="save_session / compile(source=session): session to read (default: current).",
        )

    async def execute(self, ctx: ToolContext, params: "LearnTool.Params") -> ToolResult:
        await ctx.check_permission(self.name, params.action, f"learn action={params.action}")

        if params.action == "start":
            return await self._start(params.task_name)
        if params.action == "stop":
            return await self._stop()
        if params.action == "save":
            return await self._save(params.task_name, params.procedure)
        if params.action == "replay":
            return await self._replay(params.task_name)
        if params.action == "list":
            return await self._list()
        if params.action == "compile":
            return await self._compile(ctx, params)
        if params.action == "save_session":
            return await self._save_session(ctx, params)
        if params.action == "run":
            return await self._run(ctx, params)
        if params.action == "patch":
            return await self._patch(params)
        if params.action == "show":
            return await self._show(params.task_name)

        return ToolResult(title="learn", output=f"Unknown action: {params.action}")

    async def _start(self, task_name: Optional[str]) -> ToolResult:
        global _active_recorder
        if not task_name:
            return ToolResult(title="learn", output="Error: task_name is required for start")
        if _active_recorder is not None:
            return ToolResult(title="learn", output="A recording is already active. Call learn(stop) first.")
        try:
            from opendesk.automation.recorder import LearnRecorder
        except ImportError:
            return ToolResult(title="learn", output="Error: pynput is required. Run: pip install pynput")

        _active_recorder = LearnRecorder(task_name)
        _active_recorder.start()
        return ToolResult(
            title="learn",
            output=f"Recording started for task '{task_name}'. Perform the task now, then call learn(action=stop) when done."
        )

    async def _stop(self) -> ToolResult:
        global _active_recorder
        if _active_recorder is None:
            return ToolResult(title="learn", output="No active recording. Start one with learn(action=start, task_name=...)")

        from opendesk.automation.trajectory import build_display_summary
        from opendesk.tools.base import Attachment

        trajectory = _active_recorder.stop()
        _active_recorder = None
        summary = build_display_summary(trajectory)
        compiled_note = _persist_and_compile(trajectory)

        attachments = []
        if trajectory.initial_screenshot:
            attachments.append(Attachment(
                content=_b64_to_bytes(trajectory.initial_screenshot),
                media_type="image/jpeg",
                filename="initial_screen.jpg",
            ))

        snap_count = 0
        for i, ev in enumerate(trajectory.events):
            if ev.screenshot_after and snap_count < 6:
                if ev.action_type in ("click", "right_click") or (
                    ev.action_type == "key" and ev.key in ("return", "escape")
                ):
                    attachments.append(Attachment(
                        content=_b64_to_bytes(ev.screenshot_after),
                        media_type="image/jpeg",
                        filename=f"step_{i+1}.jpg",
                    ))
                    snap_count += 1

        output = (
            f"{summary}\n\n{compiled_note}\n\n---\n"
            "Now summarize this recording into a procedure JSON and call:\n"
            "  learn(action=save, task_name=..., procedure='{\"task_name\":\"...\",\"description\":\"...\","
            "\"steps\":[...],\"procedure\":\"...\"}')\n"
            "Guidelines: describe goals, not file paths or app names. "
            "Write steps an agent can replay in any environment."
        )
        return ToolResult(title="learn", output=output, attachments=attachments)

    async def _save(self, task_name: Optional[str], procedure: Optional[str]) -> ToolResult:
        if not task_name:
            return ToolResult(title="learn", output="Error: task_name is required for save")
        if not procedure:
            return ToolResult(title="learn", output="Error: procedure JSON string is required for save")
        try:
            data = json.loads(procedure)
        except json.JSONDecodeError as e:
            return ToolResult(title="learn", output=f"Error: invalid JSON — {e}")

        from opendesk.automation.storage import save_procedure
        path = save_procedure(Path.cwd(), task_name, data)
        return ToolResult(title="learn", output=f"Procedure '{task_name}' saved to {path}")

    async def _replay(self, task_name: Optional[str]) -> ToolResult:
        if not task_name:
            return ToolResult(title="learn", output="Error: task_name is required for replay")

        from opendesk.automation.storage import load_procedure
        proc = load_procedure(Path.cwd(), task_name)
        if proc is None:
            return ToolResult(
                title="learn",
                output=f"No procedure found for '{task_name}'. Run learn(action=list) to see available tasks."
            )

        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(proc.get("steps", [])))
        prompt = (
            f"Replaying task: {proc.get('task_name', task_name)}\n"
            f"Goal: {proc.get('description', '')}\n\n"
            f"Steps:\n{steps_text}\n\n"
            f"Procedure:\n{proc.get('procedure', '')}\n\n"
            "Execute each step using the available tools (ui, screenshot, mouse, keyboard, app). "
            "Use the ui tool first, screenshot(marks=True) if needed, mouse as last resort. "
            "Adapt to the current environment — do not rely on specific file paths or app versions."
        )
        return ToolResult(title="learn", output=prompt)

    async def _list(self) -> ToolResult:
        from opendesk.automation.storage import list_procedures
        procs = list_procedures(Path.cwd())
        if not procs:
            return ToolResult(title="learn", output="No learned procedures yet. Record one with learn(action=start, task_name=...)")

        lines = ["Learned procedures:", ""]
        for p in procs:
            desc = f" — {p['description']}" if p.get("description") else ""
            tag = " [compiled]" if p.get("compiled") else ""
            lines.append(f"  {p['name']}{tag}{desc}")
        if any(p.get("compiled") for p in procs):
            lines += ["", "[compiled] tasks run deterministically with learn(action=run)."]
        return ToolResult(title="learn", output="\n".join(lines))

    # -- compiled programs ------------------------------------------------

    async def _compile(self, ctx: ToolContext, params: "LearnTool.Params") -> ToolResult:
        from opendesk.automation.program import (
            PROGRAM_SCHEMA_HINT, compile_trajectory, parse_program, save_program,
        )
        from opendesk.automation.storage import find_sidecar, load_procedure

        name = params.task_name
        if not name:
            return ToolResult(title="learn", output="Error: task_name is required for compile", error=True)

        if params.program:
            try:
                prog = parse_program(params.program, name=name)
            except Exception as e:
                return ToolResult(
                    title="learn", output=f"Error: invalid program — {e}\n\n{PROGRAM_SCHEMA_HINT}", error=True,
                )
            if params.description:
                prog.description = params.description
            path = save_program(Path.cwd(), prog)
            return ToolResult(title="learn", output=f"Saved program to {path}\n\n{prog.describe()}")

        if params.source == "session":
            return await self._save_session(ctx, params)

        traj_path = find_sidecar(Path.cwd(), name, "trajectory")
        proc = load_procedure(Path.cwd(), name)
        if traj_path is None:
            hint = ""
            if proc is not None:
                hint = (
                    "\nAn NL procedure exists — compile it yourself by writing steps and calling "
                    f"learn(action='compile', task_name='{name}', program='[...]').\n"
                    f"Goal: {proc.get('description', '')}\n"
                    + "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(proc.get("steps", [])))
                )
            return ToolResult(
                title="learn",
                output=(
                    f"No recorded trajectory for '{name}'. Record with learn(start/stop), use "
                    f"learn(action='save_session'), or pass program JSON.{hint}\n\n{PROGRAM_SCHEMA_HINT}"
                ),
                error=True,
            )
        prog = compile_trajectory(
            json.loads(traj_path.read_text()), name=name,
            description=params.description or (proc or {}).get("description", ""),
        )
        path = save_program(Path.cwd(), prog)
        return ToolResult(
            title="learn", output=f"Compiled {len(prog.steps)} step(s) to {path}\n\n{prog.describe()}",
        )

    async def _save_session(self, ctx: ToolContext, params: "LearnTool.Params") -> ToolResult:
        from opendesk.automation.program import (
            compile_audit_log, procedure_from_program, save_program,
        )
        from opendesk.automation.storage import load_procedure, save_procedure
        from opendesk.computer.sandbox import get_sandbox

        name = params.task_name
        if not name:
            return ToolResult(title="learn", output="Error: task_name is required for save_session", error=True)
        session_id = params.session_id or ctx.session_id
        log = get_sandbox(session_id).export_audit_log()
        prog = compile_audit_log(log, name=name, description=params.description or "")
        if not prog.steps:
            return ToolResult(
                title="learn",
                output=f"Nothing to save — session '{session_id}' has no replayable actions yet.",
                error=True,
            )
        path = save_program(Path.cwd(), prog)
        extra = ""
        if load_procedure(Path.cwd(), name) is None:
            proc_path = save_procedure(Path.cwd(), name, procedure_from_program(prog))
            extra = f"\nAlso wrote a basic NL procedure to {proc_path} (refine it with learn(save))."
        return ToolResult(
            title="learn",
            output=(
                f"Saved {len(prog.steps)} step(s) from session '{session_id}' to {path}{extra}\n\n"
                f"{prog.describe()}"
            ),
        )

    async def _run(self, ctx: ToolContext, params: "LearnTool.Params") -> ToolResult:
        from opendesk.automation.program import format_self_heal, load_program, run_program
        from opendesk.automation.storage import load_procedure
        from opendesk.tools.base import Attachment

        name = params.task_name
        if not name:
            return ToolResult(title="learn", output="Error: task_name is required for run", error=True)
        prog = load_program(Path.cwd(), name)
        if prog is None:
            return ToolResult(
                title="learn",
                output=(
                    f"No compiled program for '{name}'. Compile one with learn(action='compile') "
                    "or fall back to learn(action='replay')."
                ),
                error=True,
            )
        try:
            result = await run_program(prog, ctx, start_at=params.start_at)
        except KeyError as e:
            return ToolResult(title="learn", output=f"Error: {e}", error=True)
        steps_meta = [
            {"step_id": o.step_id, "ok": o.ok, "detail": o.detail, "elapsed_ms": o.elapsed_ms}
            for o in result.outcomes
        ]
        if result.ok:
            return ToolResult(
                title=f"learn run: {prog.name}",
                output=result.summary(),
                metadata={"status": "ok", "steps": steps_meta},
            )
        proc = load_procedure(Path.cwd(), name)
        attachments = []
        if result.screenshot:
            ext = "jpg" if result.screenshot_media_type == "image/jpeg" else "png"
            failed_id = result.failed_step.id if result.failed_step else "step"
            attachments.append(Attachment(
                filename=f"failed_{failed_id}.{ext}",
                content=result.screenshot,
                media_type=result.screenshot_media_type,
            ))
        return ToolResult(
            title=f"learn run: {prog.name} — needs self-heal",
            output=format_self_heal(result, proc),
            error=True,
            attachments=attachments,
            metadata={"status": "failed", "self_heal": result.self_heal(proc), "steps": steps_meta},
        )

    async def _patch(self, params: "LearnTool.Params") -> ToolResult:
        from opendesk.automation.program import load_program, patch_program, save_program

        name = params.task_name
        if not name:
            return ToolResult(title="learn", output="Error: task_name is required for patch", error=True)
        prog = load_program(Path.cwd(), name)
        if prog is None:
            return ToolResult(title="learn", output=f"No compiled program for '{name}'.", error=True)
        mode = params.mode or ("replace" if params.step_id else "append")
        try:
            patch_program(prog, mode=mode, step_id=params.step_id, step=params.step)
        except Exception as e:
            return ToolResult(title="learn", output=f"Error: patch failed — {e}", error=True)
        path = save_program(Path.cwd(), prog)
        return ToolResult(title="learn", output=f"Patched ({mode}) {path}\n\n{prog.describe()}")

    async def _show(self, task_name: Optional[str]) -> ToolResult:
        from opendesk.automation.program import load_program
        from opendesk.automation.storage import program_path

        if not task_name:
            return ToolResult(title="learn", output="Error: task_name is required for show", error=True)
        prog = load_program(Path.cwd(), task_name)
        if prog is None:
            return ToolResult(title="learn", output=f"No compiled program for '{task_name}'.", error=True)
        return ToolResult(
            title=f"learn program: {prog.name}",
            output=f"{prog.describe()}\n\nFile: {program_path(Path.cwd(), prog.name)}\n\n{prog.to_json()}",
        )


class ScheduleTool(Tool):
    name = "schedule"
    title = "Schedule Desktop Tasks"
    annotations = {
        "readOnlyHint": False, "destructiveHint": True,
        "idempotentHint": False, "openWorldHint": True,
    }
    description = (
        "Schedule tasks to run automatically on a timer.\n\n"
        "Actions:\n"
        "- add: schedule a task. task can be natural language ('take a screenshot and save it') "
        "or a learned procedure ('replay expense-report'). "
        "timing examples: 'every 30m', 'every 2h', 'every day at 09:00', 'every friday at 17:00'\n"
        "- remove: remove a schedule by name\n"
        "- list: show all scheduled tasks\n"
        "- run: run a scheduled task immediately (for testing)\n\n"
        "After adding schedules, start the background runner with: opendesk scheduler start"
    )

    class Params(BaseModel):
        action: Literal["add", "remove", "list", "run"]
        name: Optional[str] = None
        task: Optional[str] = None
        timing: Optional[str] = None

    async def execute(self, ctx: ToolContext, params: "ScheduleTool.Params") -> ToolResult:
        await ctx.check_permission(self.name, params.action, f"schedule action={params.action}")

        if params.action == "add":
            return self._add(params.name, params.task, params.timing)
        if params.action == "remove":
            return self._remove(params.name)
        if params.action == "list":
            return self._list()
        if params.action == "run":
            return await self._run(params.name)

        return ToolResult(title="schedule", output=f"Unknown action: {params.action}")

    def _add(self, name: Optional[str], task: Optional[str], timing: Optional[str]) -> ToolResult:
        if not name:
            return ToolResult(title="schedule", output="Error: name is required")
        if not task:
            return ToolResult(title="schedule", output="Error: task is required")
        if not timing:
            return ToolResult(title="schedule", output="Error: timing is required")

        from opendesk.automation.schedule_store import ScheduleStore
        try:
            store = ScheduleStore(Path.cwd())
            entry = store.add(name=name, task=task, timing=timing)
        except ValueError as e:
            return ToolResult(title="schedule", output=f"Error: {e}")

        trigger = f"cron: {entry.cron}" if entry.cron else f"every {entry.interval_seconds}s"
        return ToolResult(
            title="schedule",
            output=f"Scheduled '{name}' ({trigger})\nTask: {task}\n\nStart the background runner:\n  opendesk scheduler start",
        )

    def _remove(self, name: Optional[str]) -> ToolResult:
        if not name:
            return ToolResult(title="schedule", output="Error: name is required")

        from opendesk.automation.schedule_store import ScheduleStore
        store = ScheduleStore(Path.cwd())
        if store.remove(name):
            return ToolResult(title="schedule", output=f"Removed schedule '{name}'")
        return ToolResult(title="schedule", output=f"No schedule found with name '{name}'")

    def _list(self) -> ToolResult:
        import time
        from opendesk.automation.schedule_store import ScheduleStore
        store = ScheduleStore(Path.cwd())
        entries = store.all()

        if not entries:
            return ToolResult(title="schedule", output="No schedules yet.\nAdd one with: schedule(action=add, name=..., task=..., timing=...)")

        lines = ["Scheduled tasks:", ""]
        for e in entries:
            last = "never" if not e.last_run else _ago(e.last_run)
            last_result = f" [{e.last_status}]" if e.last_status else ""
            lines.append(f"  {e.name}  ({e.timing})  {'enabled' if e.enabled else 'disabled'}")
            lines.append(f"    task: {e.task}")
            lines.append(f"    last run: {last}{last_result}")
            lines.append("")

        lines.append("Run 'opendesk scheduler start' to activate scheduled tasks.")
        return ToolResult(title="schedule", output="\n".join(lines))

    async def _run(self, name: Optional[str]) -> ToolResult:
        if not name:
            return ToolResult(title="schedule", output="Error: name is required")

        from opendesk.automation.schedule_store import ScheduleStore
        from opendesk.automation.runner import run_task

        store = ScheduleStore(Path.cwd())
        entry = store.get(name)
        if not entry:
            return ToolResult(title="schedule", output=f"No schedule found with name '{name}'")

        status = await run_task(entry.task, Path.cwd())
        store.update_run(name, status)
        return ToolResult(title="schedule", output=f"Ran '{name}': {status}")


# ---------------------------------------------------------------------------

def _persist_and_compile(trajectory: Any) -> str:
    """Save the trajectory sidecar and compile a program from it (best effort)."""
    try:
        from opendesk.automation.program import compile_trajectory, save_program
        from opendesk.automation.storage import trajectory_path
        from opendesk.automation.trajectory import trajectory_to_dict

        data = trajectory_to_dict(trajectory)
        trajectory_path(Path.cwd(), trajectory.task_name).write_text(json.dumps(data, indent=2))
        prog = compile_trajectory(trajectory)
        if not prog.steps:
            return "No compilable steps were recorded."
        path = save_program(Path.cwd(), prog)
        return (
            f"Compiled {len(prog.steps)} deterministic step(s) to {path} — "
            "run with learn(action=run), inspect with learn(action=show)."
        )
    except Exception as e:  # never block the NL flow on compilation
        return f"(Could not compile a program: {e})"


def _b64_to_bytes(b64: str) -> bytes:
    import base64
    return base64.b64decode(b64)


def _ago(ts: float) -> str:
    import time
    diff = int(time.time() - ts)
    if diff < 60:
        return f"{diff}s ago"
    if diff < 3600:
        return f"{diff // 60}m ago"
    if diff < 86400:
        return f"{diff // 3600}h ago"
    return f"{diff // 86400}d ago"
