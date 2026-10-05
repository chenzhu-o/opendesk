"""SkillTool — save, retrieve, and run reusable parameterised procedures.

Where ``learn`` records one concrete trajectory and replays it as prose,
``skill`` stores a *policy*: an ordered list of tool calls with
``{{placeholder}}`` parameters that can be re-bound on every run.  Skills
accumulate in the project directory and are retrieved by relevance, so an
agent that has solved a task once does not re-plan it from scratch next time.

The tool is deliberately local-session state (like ``learn`` / ``schedule``):
skills live next to the project, not on a remote peer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field

from opendesk.tools.base import Tool, ToolContext, ToolResult

#: Refuse to run a skill with more steps than this — a guard against loops.
MAX_STEPS = 50


class SkillTool(Tool):
    """Persist and run reusable, parameterised skills."""

    name = "skill"
    description = (
        "Save, find, and run reusable *skills* — parameterised procedures that "
        "capture how a task is done so it can be repeated on new inputs without "
        "re-planning every step.\n\n"
        "Actions:\n"
        "- save: store a skill. Provide name, steps (JSON list of "
        "{tool, params}), optionally description, params (JSON schema of "
        "{{placeholders}}) and tags.\n"
        "- find: retrieve the skills most relevant to a task, each with its "
        "parameter schema and applicability. Relevance is not applicability — "
        "check `applicability` (os / apps) before running.\n"
        "- run: bind parameters and execute the skill's steps through the same "
        "tools. Pass execute=false to only render the plan.\n"
        "- list: list all saved skills.\n"
        "- delete: remove a skill by name.\n\n"
        "Workflow: solve a task once → skill(save) the winning procedure → "
        "next time skill(find, query=...) then skill(run, name=..., "
        "arguments={...}).\n\n"
        "Steps may invoke any tool except `skill` itself (no recursion). "
        "Reference parameters with {{name}} in any step param."
    )

    class Params(Tool.Params):
        action: Literal["save", "find", "run", "list", "delete"] = Field(
            description="Skill action."
        )
        name: Optional[str] = Field(
            default=None, description="Skill name. Required for save/run/delete."
        )
        description: Optional[str] = Field(
            default=None, description="Human/agent-readable summary of the skill."
        )
        params: Optional[str] = Field(
            default=None,
            description=(
                "JSON object mapping parameter names to their schema, e.g. "
                '\'{"month": {"type": "string", "required": true}, '
                '"dest": {"type": "string", "default": "~/Downloads"}}\'. '
                "Used by action='save'."
            ),
        )
        steps: Optional[str] = Field(
            default=None,
            description=(
                "JSON list of steps to save, e.g. "
                '\'[{"tool": "system", "params": {"action": "mkdir", '
                '"path": "{{dest}}"}}, {"tool": "ui", "params": '
                '{"action": "click", "title": "Invoices"}}]\'. '
                "Used by action='save'."
            ),
        )
        tags: Optional[list[str]] = Field(
            default=None, description="Optional tags for filtering, e.g. ['browser']."
        )
        applicability: Optional[str] = Field(
            default=None,
            description=(
                "JSON object describing where this skill works, e.g. "
                '\'{"os": ["darwin", "linux"], "apps": ["Google Chrome"]}\'. '
                "Shown by action='find' so the caller can judge applicability."
            ),
        )
        query: Optional[str] = Field(
            default=None,
            description="Natural-language task description. Required for action='find'.",
        )
        arguments: Optional[dict[str, Any]] = Field(
            default=None,
            description="Parameter bindings for action='run', e.g. {'month': '2026-09'}.",
        )
        execute: bool = Field(
            default=True,
            description=(
                "For action='run': execute the rendered steps (default) or, "
                "when false, only return the rendered plan."
            ),
        )
        limit: int = Field(
            default=5, description="Maximum number of results for action='find'."
        )

    async def execute(self, ctx: ToolContext, params: "SkillTool.Params") -> ToolResult:
        from opendesk.computer.sandbox import ActionType, get_sandbox

        await ctx.check_permission(
            tool="skill",
            argument=f"{params.action} {params.name or params.query or ''}".strip(),
            description=f"Skill: {params.action}",
        )
        sandbox = get_sandbox(ctx.session_id)
        project_dir = Path.cwd()

        try:
            if params.action == "save":
                return await self._save(sandbox, project_dir, params)
            if params.action == "find":
                return self._find(project_dir, params)
            if params.action == "run":
                return await self._run(ctx, sandbox, project_dir, params)
            if params.action == "list":
                return self._list(project_dir)
            if params.action == "delete":
                return await self._delete(sandbox, project_dir, params)
        except Exception as exc:
            return ToolResult(
                title=f"skill: {params.action} failed",
                output=f"`{params.action}` failed: {exc}",
                error=True,
            )

        return ToolResult(title="skill", output=f"Unknown action: {params.action}")

    # ------------------------------------------------------------------

    async def _save(self, sandbox, project_dir: Path, params: "SkillTool.Params") -> ToolResult:
        from opendesk.automation.skills_store import save_skill
        from opendesk.computer.sandbox import ActionType

        if not params.name:
            return ToolResult(title="skill", output="Error: name is required for save", error=True)
        if not params.steps:
            return ToolResult(title="skill", output="Error: steps is required for save", error=True)

        try:
            steps = json.loads(params.steps)
        except json.JSONDecodeError as e:
            return ToolResult(title="skill", output=f"Error: invalid steps JSON — {e}", error=True)
        if not isinstance(steps, list) or not steps:
            return ToolResult(title="skill", output="Error: steps must be a non-empty JSON list", error=True)

        parsed_params: dict[str, Any] = {}
        if params.params:
            try:
                parsed_params = json.loads(params.params)
            except json.JSONDecodeError as e:
                return ToolResult(title="skill", output=f"Error: invalid params JSON — {e}", error=True)

        applicability: dict[str, Any] = {}
        if params.applicability:
            try:
                applicability = json.loads(params.applicability)
            except json.JSONDecodeError as e:
                return ToolResult(
                    title="skill", output=f"Error: invalid applicability JSON — {e}", error=True
                )

        doc = {
            "name": params.name,
            "description": params.description or "",
            "tags": params.tags or [],
            "version": 1,
            "params": parsed_params,
            "steps": steps,
            "applicability": applicability,
        }
        path = save_skill(project_dir, params.name, doc)

        await sandbox.record_action(
            ActionType.SKILL_SAVE, {"name": params.name, "steps": len(steps)},
            result=str(path), replay_params={"tool": "skill", "params": {"action": "save"}},
        )
        return ToolResult(
            title=f"skill saved: {params.name}",
            output=(
                f"Saved skill '{params.name}' ({len(steps)} steps, "
                f"{len(parsed_params)} params) to {path}."
            ),
        )

    def _find(self, project_dir: Path, params: "SkillTool.Params") -> ToolResult:
        from opendesk.automation.skills_store import find_skills

        if not params.query:
            return ToolResult(title="skill", output="Error: query is required for find", error=True)

        matches = find_skills(project_dir, params.query, tags=params.tags, limit=params.limit)
        if not matches:
            return ToolResult(
                title="skill: find",
                output=(
                    f"No skill matches '{params.query}'. "
                    "Solve the task, then skill(action=save) it for next time."
                ),
            )

        lines = [f"Skills matching '{params.query}' ({len(matches)}):", ""]
        for m in matches:
            lines.append(f"• {m['name']}  (relevance {m['score']})")
            if m["description"]:
                lines.append(f"    {m['description']}")
            if m["tags"]:
                lines.append(f"    tags: {', '.join(m['tags'])}")
            if m["params"]:
                spec = ", ".join(
                    f"{k}{'*' if (v or {}).get('required') else ''}"
                    for k, v in m["params"].items()
                )
                lines.append(f"    params: {spec}   (* = required)")
            if m["applicability"]:
                lines.append(f"    applicability: {json.dumps(m['applicability'])}")
            lines.append(
                f"    run: skill(action=run, name={m['name']!r}, arguments={{...}})"
            )
            lines.append("")
        lines.append(
            "Check `applicability` before running — a relevant skill may still "
            "not apply on this OS or app set."
        )
        return ToolResult(title="skill: find", output="\n".join(lines))

    async def _run(
        self, ctx: ToolContext, sandbox, project_dir: Path, params: "SkillTool.Params"
    ) -> ToolResult:
        from opendesk.automation.skills_store import (
            SkillParamError,
            bind_params,
            load_skill,
            render_steps,
        )
        from opendesk.computer.sandbox import ActionType

        if not params.name:
            return ToolResult(title="skill", output="Error: name is required for run", error=True)

        doc = load_skill(project_dir, params.name)
        if doc is None:
            return ToolResult(
                title="skill: run",
                output=f"No skill named '{params.name}'. Use skill(action=list).",
                error=True,
            )

        if params.arguments and not isinstance(params.arguments, dict):
            return ToolResult(title="skill", output="Error: arguments must be a JSON object", error=True)

        try:
            bindings = bind_params(doc, params.arguments or {})
            steps = render_steps(doc, bindings)
        except SkillParamError as e:
            return ToolResult(title="skill: run", output=f"Error: {e}", error=True)

        if len(steps) > MAX_STEPS:
            return ToolResult(
                title="skill: run",
                output=f"Error: skill has {len(steps)} steps (> {MAX_STEPS} cap).",
                error=True,
            )

        for step in steps:
            if not isinstance(step, dict) or "tool" not in step:
                return ToolResult(
                    title="skill: run",
                    output=f"Error: malformed step: {step!r}",
                    error=True,
                )
            if step["tool"] == "skill":
                return ToolResult(
                    title="skill: run",
                    output="Error: a skill may not invoke the `skill` tool (no recursion).",
                    error=True,
                )

        plan = _format_plan(doc.get("name", params.name), bindings, steps)

        if not params.execute:
            return ToolResult(
                title=f"skill plan: {params.name}",
                output=plan + "\n\n(execute=false — plan only; re-run with execute=true to run it.)",
            )

        lines = [plan, "", "Results:"]
        failure: Optional[str] = None

        for i, step in enumerate(steps, 1):
            tool_name = step["tool"]
            step_params = step.get("params", {}) or {}
            tool = _registry_tool(tool_name)
            if tool is None:
                failure = f"step {i}: unknown tool {tool_name!r}"
                lines.append(f"  {i}. {tool_name} — ERROR: unknown tool")
                break
            try:
                parsed = tool.parse_params(step_params)
                result = await tool.execute(ctx, parsed)
            except Exception as exc:
                failure = f"step {i}: {exc}"
                lines.append(f"  {i}. {tool_name} — ERROR: {exc}")
                break

            status = "ERROR" if result.error else "ok"
            summary = result.output.strip().splitlines()
            head = summary[0][:160] if summary else ""
            lines.append(f"  {i}. {tool_name} — {status}: {head}")
            if result.error:
                failure = f"step {i}: {result.output[:200]}"
                break

        await sandbox.record_action(
            ActionType.SKILL_RUN,
            {"name": doc.get("name", params.name), "steps": len(steps)},
            result=failure or "ok",
            error=failure,
            replay_params={
                "tool": "skill",
                "params": {
                    "action": "run",
                    "name": doc.get("name", params.name),
                    "arguments": params.arguments,
                },
            },
        )

        if failure:
            lines.append("")
            lines.append(f"Stopped: {failure}")
            return ToolResult(
                title=f"skill run: {params.name} (failed)",
                output="\n".join(lines),
                error=True,
            )

        lines.append("")
        lines.append(f"Completed {len(steps)} step(s).")
        return ToolResult(title=f"skill run: {params.name}", output="\n".join(lines))

    def _list(self, project_dir: Path) -> ToolResult:
        from opendesk.automation.skills_store import list_skills

        skills = list_skills(project_dir)
        if not skills:
            return ToolResult(
                title="skill: list",
                output="No skills saved yet. Solve a task, then skill(action=save).",
            )
        lines = [f"Saved skills ({len(skills)}):", ""]
        for s in skills:
            desc = f" — {s['description']}" if s["description"] else ""
            params = f"  params: {', '.join(s['params'])}" if s["params"] else ""
            lines.append(f"  {s['name']}  ({s['steps']} steps){desc}{params}")
        return ToolResult(title="skill: list", output="\n".join(lines))

    async def _delete(self, sandbox, project_dir: Path, params: "SkillTool.Params") -> ToolResult:
        from opendesk.automation.skills_store import delete_skill
        from opendesk.computer.sandbox import ActionType

        if not params.name:
            return ToolResult(title="skill", output="Error: name is required for delete", error=True)
        removed = delete_skill(project_dir, params.name)
        if not removed:
            return ToolResult(
                title="skill: delete", output=f"No skill named '{params.name}'.", error=True
            )
        await sandbox.record_action(
            ActionType.SKILL_DELETE, {"name": params.name},
            replay_params={"tool": "skill", "params": {"action": "delete", "name": params.name}},
        )
        return ToolResult(title="skill: delete", output=f"Deleted skill '{params.name}'.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry_tool(name: str):
    """Look up a tool in a fresh registry (tool objects are stateless)."""
    from opendesk.registry import create_registry

    try:
        return create_registry().get(name)
    except KeyError:
        return None


def _format_plan(name: str, bindings: dict[str, Any], steps: list[dict]) -> str:
    lines = [f"Skill '{name}'" + (f"  bindings: {json.dumps(bindings, ensure_ascii=False)}" if bindings else "")]
    for i, step in enumerate(steps, 1):
        params = json.dumps(step.get("params", {}), ensure_ascii=False)
        lines.append(f"  {i}. {step.get('tool')} {params}")
    return "\n".join(lines)
