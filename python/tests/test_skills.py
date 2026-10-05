"""Tests for the persistent parameterised skill library (``skill``)."""

from __future__ import annotations

import json

import pytest

from opendesk.automation.skills_store import (
    SkillParamError,
    bind_params,
    find_skills,
    list_skills,
    load_skill,
    render,
    render_steps,
    save_skill,
    score_skill,
)
from opendesk.tools.base import ToolContext
from opendesk.tools.skills import SkillTool

from tests._fakes import FakeComputer


def _ctx(computer: FakeComputer) -> ToolContext:
    return ToolContext(session_id="test-skill", computer=computer)


async def _run(tool: SkillTool, computer: FakeComputer, **params):
    return await tool.execute(_ctx(computer), tool.parse_params(params))


SAMPLE = {
    "name": "scaffold_project",
    "description": "Create a project folder and open it in the browser",
    "tags": ["browser", "setup"],
    "params": {
        "dest": {"type": "string", "required": True},
        "name": {"type": "string", "default": "untitled"},
    },
    "steps": [
        {"tool": "system", "params": {"action": "mkdir", "path": "{{dest}}"}},
        {"tool": "system", "params": {"action": "shell", "command": "echo {{name}}"}},
    ],
    "applicability": {"os": ["linux", "darwin"]},
}


# ---------------------------------------------------------------------------
# Store unit tests
# ---------------------------------------------------------------------------


class TestStore:
    def test_save_load_list(self, tmp_path):
        save_skill(tmp_path, "scaffold_project", SAMPLE)
        loaded = load_skill(tmp_path, "scaffold_project")
        assert loaded["name"] == "scaffold_project"
        assert len(list_skills(tmp_path)) == 1

    def test_bind_required_and_default(self):
        bindings = bind_params(SAMPLE, {"dest": "/tmp/x"})
        assert bindings == {"dest": "/tmp/x", "name": "untitled"}

    def test_bind_missing_required_raises(self):
        with pytest.raises(SkillParamError):
            bind_params(SAMPLE, {})

    def test_render_typed_placeholder(self):
        assert render("{{n}}", {"n": 3}) == 3
        assert render("count={{n}}", {"n": 3}) == "count=3"

    def test_render_nested(self):
        out = render({"a": ["{{x}}", {"b": "v={{x}}"}]}, {"x": "hi"})
        assert out == {"a": ["hi", {"b": "v=hi"}]}

    def test_render_unbound_raises(self):
        with pytest.raises(SkillParamError):
            render("{{missing}}", {})

    def test_render_steps(self):
        steps = render_steps(SAMPLE, {"dest": "/tmp/x", "name": "proj"})
        assert steps[0]["params"]["path"] == "/tmp/x"
        assert steps[1]["params"]["command"] == "echo proj"

    def test_score_prefers_matching_skill(self, tmp_path):
        save_skill(tmp_path, "scaffold_project", SAMPLE)
        save_skill(tmp_path, "other", {"name": "other", "description": "send an email"})
        matches = find_skills(tmp_path, "create project folder")
        assert matches and matches[0]["name"] == "scaffold_project"
        assert matches[0]["score"] > 0

    def test_find_returns_applicability(self, tmp_path):
        save_skill(tmp_path, "scaffold_project", SAMPLE)
        matches = find_skills(tmp_path, "create project folder")
        assert matches[0]["applicability"] == {"os": ["linux", "darwin"]}

    def test_tag_filter(self, tmp_path):
        save_skill(tmp_path, "scaffold_project", SAMPLE)
        assert find_skills(tmp_path, "create project", tags=["nope"]) == []
        assert find_skills(tmp_path, "create project", tags=["browser"])


# ---------------------------------------------------------------------------
# Tool integration tests
# ---------------------------------------------------------------------------


class TestSkillTool:
    @pytest.mark.asyncio
    async def test_save_list_find_delete(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tool = SkillTool()
        comp = FakeComputer()

        saved = await _run(
            tool, comp, action="save", name="scaffold_project",
            description=SAMPLE["description"],
            params=json.dumps(SAMPLE["params"]),
            steps=json.dumps(SAMPLE["steps"]),
            tags=["browser", "setup"],
            applicability=json.dumps(SAMPLE["applicability"]),
        )
        assert not saved.error, saved.output

        listing = await _run(tool, comp, action="list")
        assert "scaffold_project" in listing.output

        found = await _run(tool, comp, action="find", query="create project folder")
        assert "scaffold_project" in found.output
        assert "applicability" in found.output

        deleted = await _run(tool, comp, action="delete", name="scaffold_project")
        assert not deleted.error
        assert list_skills(tmp_path) == []

    @pytest.mark.asyncio
    async def test_save_requires_steps(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = await _run(SkillTool(), FakeComputer(), action="save", name="x")
        assert result.error
        assert "steps is required" in result.output

    @pytest.mark.asyncio
    async def test_save_rejects_bad_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = await _run(
            SkillTool(), FakeComputer(), action="save", name="x", steps="{not json"
        )
        assert result.error
        assert "invalid steps JSON" in result.output

    @pytest.mark.asyncio
    async def test_run_executes_steps_through_registry(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_skill(tmp_path, "scaffold_project", SAMPLE)
        comp = FakeComputer()

        result = await _run(
            SkillTool(), comp, action="run", name="scaffold_project",
            arguments={"dest": "/tmp/proj", "name": "proj"},
        )
        assert not result.error, result.output
        assert ("mkdir", {"path": "/tmp/proj"}) in comp.calls
        assert ("shell", {"command": "echo proj"}) in comp.calls
        assert "Completed 2 step(s)." in result.output

    @pytest.mark.asyncio
    async def test_run_execute_false_is_plan_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_skill(tmp_path, "scaffold_project", SAMPLE)
        comp = FakeComputer()

        result = await _run(
            SkillTool(), comp, action="run", name="scaffold_project",
            arguments={"dest": "/tmp/proj"}, execute=False,
        )
        assert not result.error
        assert "plan only" in result.output
        assert comp.calls == []  # nothing executed

    @pytest.mark.asyncio
    async def test_run_missing_required_param(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_skill(tmp_path, "scaffold_project", SAMPLE)
        result = await _run(SkillTool(), FakeComputer(), action="run", name="scaffold_project")
        assert result.error
        assert "missing required parameter" in result.output

    @pytest.mark.asyncio
    async def test_run_unknown_skill(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = await _run(SkillTool(), FakeComputer(), action="run", name="nope")
        assert result.error

    @pytest.mark.asyncio
    async def test_run_rejects_recursive_skill_step(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_skill(tmp_path, "loop", {
            "name": "loop",
            "params": {},
            "steps": [{"tool": "skill", "params": {"action": "list"}}],
        })
        result = await _run(SkillTool(), FakeComputer(), action="run", name="loop")
        assert result.error
        assert "recursion" in result.output

    @pytest.mark.asyncio
    async def test_run_stops_on_failing_step(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_skill(tmp_path, "bad", {
            "name": "bad",
            "params": {},
            "steps": [
                {"tool": "system", "params": {"action": "shell", "command": "ok"}},
                {"tool": "system", "params": {"action": "shell"}},  # missing command
            ],
        })
        comp = FakeComputer()
        result = await _run(SkillTool(), comp, action="run", name="bad")
        assert result.error
        assert "Stopped" in result.output
