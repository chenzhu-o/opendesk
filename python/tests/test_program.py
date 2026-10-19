"""Tests for compiled learn programs: compile, run, self-heal, patch, scheduler."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from opendesk.automation import runner as sched_runner
from opendesk.automation.program import (
    Program,
    ProgramRunner,
    Step,
    Target,
    compile_audit_log,
    compile_trajectory,
    find_element,
    load_program,
    parse_program,
    patch_program,
    run_program,
    save_program,
)
from opendesk.automation.storage import list_procedures, load_procedure, save_procedure
from opendesk.automation.trajectory import (
    Trajectory,
    TrajectoryEvent,
    trajectory_from_dict,
    trajectory_to_dict,
)
from opendesk.computer import Rect, UIElement, Window
from opendesk.computer.sandbox import ActionType, clear_sandbox, get_sandbox
from opendesk.tools.automation import LearnTool
from opendesk.tools.base import ToolContext

from tests._fakes import FakeComputer


# ---------------------------------------------------------------------------
# Scripted computer
# ---------------------------------------------------------------------------


def _btn(name: str, **kw) -> UIElement:
    return UIElement(role="AXButton", name=name, bounds=Rect(x=10, y=10, width=40, height=20),
                     actions=["click"], **kw)


class ScriptedComputer(FakeComputer):
    """FakeComputer whose UI tree / focused window change as actions happen."""

    def __init__(self, trees: Optional[dict] = None, window: Optional[Window] = None) -> None:
        super().__init__()
        self.trees = trees or {}
        self.window = window or Window(id="w", title="Untitled", app_name="TextEdit", focused=True)
        self.on_click = None  # callable(element) -> None

    async def ui_tree(self, *, window_id=None, app=None, max_depth=8) -> UIElement:
        self._record("ui_tree", app=app)
        return self.trees.get(app) or UIElement(role="window", name=self.window.title, children=[])

    async def focused_window(self):
        return self.window

    async def open_app(self, name: str) -> None:
        self._record("open_app", name=name)
        self.window = Window(id=name, title=name, app_name=name, focused=True)

    async def focus_app(self, name: str) -> None:
        self._record("focus_app", name=name)
        self.window = Window(id=name, title=self.window.title, app_name=name, focused=True)

    async def perform_ui_action(self, element, action="click", *, app=None) -> None:
        self._record("perform_ui_action", element=element, action=action, app=app)
        if self.on_click:
            self.on_click(element)


def _ctx(comp, sid="prog-test") -> ToolContext:
    clear_sandbox(sid)
    return ToolContext(session_id=sid, computer=comp)


def _fast(**kw) -> dict:
    return {"resolve_timeout_ms": 60, "poll_ms": 10, "ocr_fallback": False, **kw}


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _synthetic_trajectory() -> Trajectory:
    t0 = 1000.0
    ev = TrajectoryEvent
    return Trajectory(
        task_name="new-note",
        started_at=t0,
        stopped_at=t0 + 20,
        events=[
            ev(t0 + 1, "type", text="/learn start new-note",
               accessibility_context={"app": "Terminal"}),
            ev(t0 + 2, "click", x=300, y=200, button="left",
               accessibility_context={"app": "TextEdit", "window": "Open", "role": "AXButton",
                                      "title": "New Document", "identifier": "new-doc-btn"}),
            ev(t0 + 3, "type", text="hello world",
               accessibility_context={"app": "TextEdit", "window": "Untitled", "role": "AXTextArea"}),
            ev(t0 + 4, "key", key="cmd+s",
               accessibility_context={"app": "TextEdit", "window": "Untitled"}),
            ev(t0 + 5, "click", x=5, y=5, button="left", accessibility_context={"app": "TextEdit"}),
            ev(t0 + 6, "click", x=400, y=400, button="left",
               accessibility_context={"app": "Finder", "role": "AXRow", "title": "notes.txt"}),
            ev(t0 + 6.2, "click", x=401, y=400, button="left",
               accessibility_context={"app": "Finder", "role": "AXRow", "title": "notes.txt"}),
            ev(t0 + 7, "scroll", x=100, y=100, scroll_dy=-3),
        ],
    )


class TestCompileTrajectory:
    def test_compiles_semantic_steps(self):
        prog = compile_trajectory(_synthetic_trajectory())
        tools = [(s.tool, s.params.get("action")) for s in prog.steps]
        # /learn command and its Terminal context are skipped entirely.
        assert all(s.target.app != "Terminal" for s in prog.steps)
        assert tools[0] == ("app", "open")
        assert prog.steps[0].postcondition.kind == "app_frontmost"
        assert prog.steps[0].postcondition.value == "TextEdit"

        click = prog.steps[1]
        assert click.tool == "ui" and click.params == {"action": "click"}
        assert click.target.role == "AXButton"
        assert click.target.title == "New Document"
        assert click.target.identifier == "new-doc-btn"
        assert "x" not in click.params  # semantic, not coordinates
        # State observed at the next event becomes the postcondition.
        assert click.postcondition.kind == "window_title_contains"
        assert click.postcondition.value == "Untitled"

        typ = prog.steps[2]
        assert typ.tool == "ui" and typ.params == {"action": "type", "text": "hello world"}
        key = prog.steps[3]
        assert key.params == {"action": "press_key", "key": "s", "modifiers": ["command"]}

        fallback = prog.steps[4]
        assert fallback.tool == "mouse" and fallback.params["x"] == 5
        assert "coordinate fallback" in fallback.note

    def test_app_switch_and_double_click_and_scroll(self):
        prog = compile_trajectory(_synthetic_trajectory())
        finder_open = [s for s in prog.steps if s.tool == "app" and s.params.get("name") == "Finder"]
        assert finder_open and finder_open[0].params["action"] == "open"
        dbl = [s for s in prog.steps if s.params.get("action") == "double_click"]
        assert len(dbl) == 1
        assert dbl[0].tool == "mouse" and dbl[0].target.title == "notes.txt"
        scroll = prog.steps[-1]
        assert scroll.params["action"] == "scroll" and scroll.params["direction"] == "down"
        ids = [s.id for s in prog.steps]
        assert ids == [f"s{i}" for i in range(1, len(ids) + 1)]

    def test_compiles_from_dict_and_roundtrip(self):
        traj = _synthetic_trajectory()
        traj.events[1].screenshot_after = "AAAA"
        data = trajectory_to_dict(traj)
        assert "screenshot_after" not in json.dumps(data)
        back = trajectory_from_dict(json.loads(json.dumps(data)))
        assert back.events[1].accessibility_context["title"] == "New Document"
        assert len(compile_trajectory(data).steps) == len(compile_trajectory(traj).steps)


class TestCompileAuditLog:
    @pytest.mark.asyncio
    async def test_compiles_session_log(self):
        sb = get_sandbox("audit-compile")
        sb.audit_log.clear()
        await sb.record_action(ActionType.APP_OPEN, {}, replay_params={"action": "open", "name": "TextEdit"})
        await sb.record_action(ActionType.SCREENSHOT, {})  # skipped: read-only
        await sb.record_action(ActionType.UI_ACTION, {}, replay_params={
            "action": "get_tree", "app": "TextEdit", "title": None, "role": None, "text": None,
            "menu": None, "menu_item": None, "key": None, "modifiers": [], "window_index": 1,
        })  # skipped
        await sb.record_action(ActionType.UI_ACTION, {}, replay_params={
            "action": "click", "app": "TextEdit", "title": "Save", "role": "button", "text": None,
            "menu": None, "menu_item": None, "key": None, "modifiers": [], "window_index": 1,
        })
        await sb.record_action(ActionType.KEYBOARD_TYPE, {}, error="boom",
                               replay_params={"action": "type", "text": "bad"})  # skipped: errored
        await sb.record_action(ActionType.KEYBOARD_TYPE, {}, replay_params={
            "action": "type", "text": "hi", "key": None, "keys": None, "interval": 0.02, "hold_duration": 1.0,
        })
        prog = compile_audit_log(sb.export_audit_log(), name="sess")
        assert [s.tool for s in prog.steps] == ["app", "ui", "keyboard"]
        app, click, kb = prog.steps
        assert app.postcondition.kind == "app_frontmost" and app.postcondition.value == "TextEdit"
        assert click.params == {"action": "click"}
        assert (click.target.app, click.target.role, click.target.title) == ("TextEdit", "button", "Save")
        assert kb.params == {"action": "type", "text": "hi"}  # defaults dropped
        assert kb.target.app == "TextEdit"
        assert prog.source == "session"


# ---------------------------------------------------------------------------
# Storage & LearnTool compile/save_session/show/list
# ---------------------------------------------------------------------------


class TestStorageAndTool:
    @pytest.mark.asyncio
    async def test_save_session_writes_program_and_procedure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sid = "save-session-test"
        sb = get_sandbox(sid)
        sb.audit_log.clear()
        await sb.record_action(ActionType.APP_OPEN, {}, replay_params={"action": "open", "name": "Notes"})
        tool = LearnTool()
        res = await tool.execute(ToolContext(session_id=sid, computer=FakeComputer()),
                                 LearnTool.Params(action="save_session", task_name="My Task"))
        assert not res.error, res.output
        assert (tmp_path / ".opendesk/learned/my_task.program.json").exists()
        assert (tmp_path / ".opendesk/learned/my_task.json").exists()

        # Sidecars are not procedures; fuzzy procedure lookup ignores them.
        procs = list_procedures(tmp_path)
        assert [p["name"] for p in procs] == ["my_task"]
        assert procs[0]["compiled"] is True
        assert "steps" in load_procedure(tmp_path, "my")
        assert load_program(tmp_path, "my").name == "My Task"

        listed = await tool.execute(ToolContext(computer=FakeComputer()), LearnTool.Params(action="list"))
        assert "[compiled]" in listed.output
        shown = await tool.execute(ToolContext(computer=FakeComputer()),
                                   LearnTool.Params(action="show", task_name="my task"))
        assert "Open the Notes app" in shown.output

    @pytest.mark.asyncio
    async def test_save_session_empty_log_errors(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        clear_sandbox("empty-sess")
        res = await LearnTool().execute(ToolContext(session_id="empty-sess", computer=FakeComputer()),
                                        LearnTool.Params(action="save_session", task_name="x"))
        assert res.error and "no replayable" in res.output

    @pytest.mark.asyncio
    async def test_compile_from_trajectory_sidecar_and_program_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tool = LearnTool()
        ctx = ToolContext(computer=FakeComputer())
        missing = await tool.execute(ctx, LearnTool.Params(action="compile", task_name="new-note"))
        assert missing.error and "Program step schema" in missing.output

        from opendesk.automation.storage import trajectory_path
        trajectory_path(tmp_path, "new-note").write_text(json.dumps(trajectory_to_dict(_synthetic_trajectory())))
        res = await tool.execute(ctx, LearnTool.Params(action="compile", task_name="new-note"))
        assert not res.error, res.output
        assert load_program(tmp_path, "new-note").steps[1].target.title == "New Document"

        steps = json.dumps([{"tool": "app", "params": {"action": "open", "name": "Calculator"}}])
        res = await tool.execute(ctx, LearnTool.Params(action="compile", task_name="calc", program=steps))
        assert not res.error
        assert load_program(tmp_path, "calc").steps[0].id == "s1"
        bad = await tool.execute(ctx, LearnTool.Params(action="compile", task_name="calc", program="[{\"tool\": \"nope\"}]"))
        assert bad.error

    def test_parse_program_rejects_duplicate_ids(self):
        with pytest.raises(ValueError):
            parse_program([{"id": "a", "tool": "wait"}, {"id": "a", "tool": "wait"}], name="x")


# ---------------------------------------------------------------------------
# Element resolution
# ---------------------------------------------------------------------------


class TestFindElement:
    def test_identifier_beats_ambiguous_title(self):
        tree = UIElement(role="window", name="w", children=[
            _btn("OK"), _btn("OK", metadata={"identifier": "confirm"}),
        ])
        el = find_element(tree, Target(role="button", title="OK", identifier="confirm"))
        assert el.metadata.get("identifier") == "confirm"

    def test_exact_title_beats_substring_and_role_normalised(self):
        tree = UIElement(role="window", name="w", children=[_btn("Save As…"), _btn("Save")])
        assert find_element(tree, Target(role="button", title="save")).name == "Save"
        assert find_element(tree, Target(role="AXButton", title="Save")).name == "Save"
        assert find_element(tree, Target(role="checkbox", title="Save")) is None

    def test_ref_used_when_present_and_consistent(self):
        node_a = SimpleNamespace(role="button", name="Go", value=None, bounds=None, actions=["click"],
                                 children=[], metadata={}, ref="e1")
        node_b = SimpleNamespace(role="button", name="Go", value=None, bounds=None, actions=["click"],
                                 children=[], metadata={}, ref="e2")
        root = SimpleNamespace(role="window", name="w", value=None, bounds=None, actions=[],
                               children=[node_a, node_b], metadata={})
        assert find_element(root, Target(role="button", title="Go", ref="e2")) is node_b
        # A stale ref that now points at a different kind of element is ignored.
        assert find_element(root, Target(role="checkbox", title="Go", ref="e2")) is None
        # ref may also live in metadata.
        tree = UIElement(role="window", name="w", children=[_btn("Go", metadata={"ref": "e9"})])
        assert find_element(tree, Target(ref="e9")).name == "Go"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _textedit_program() -> Program:
    return parse_program({
        "description": "save a note",
        "steps": [
            {"tool": "app", "params": {"action": "open", "name": "TextEdit"},
             "target": {"app": "TextEdit"},
             "postcondition": {"kind": "app_frontmost", "value": "TextEdit", "timeout_ms": 200},
             "wait_ms": 0},
            {"tool": "ui", "params": {"action": "click"},
             "target": {"app": "TextEdit", "role": "button", "title": "Save"},
             "postcondition": {"kind": "window_title_contains", "value": "saved", "timeout_ms": 200},
             "wait_ms": 0},
            {"tool": "ui", "params": {"action": "type", "text": "done"},
             "target": {"app": "TextEdit"},
             "postcondition": {"kind": "element_exists", "value": {"role": "button", "title": "Close"},
                               "timeout_ms": 200},
             "wait_ms": 0},
            {"tool": "keyboard", "params": {"action": "hotkey", "keys": ["cmd", "w"]}, "wait_ms": 0},
        ],
    }, name="te")


def _scripted_textedit() -> ScriptedComputer:
    comp = ScriptedComputer(trees={
        "TextEdit": UIElement(role="window", name="Untitled", children=[_btn("Save"), _btn("Close")]),
    })

    def clicked(el):
        if el.name == "Save":
            comp.window = Window(id="w", title="notes — saved", app_name="TextEdit", focused=True)
    comp.on_click = clicked
    return comp


class TestRun:
    @pytest.mark.asyncio
    async def test_runs_all_steps_and_verifies_postconditions(self):
        comp = _scripted_textedit()
        res = await run_program(_textedit_program(), _ctx(comp), **_fast())
        assert res.ok, res.reason
        assert [o.step_id for o in res.outcomes] == ["s1", "s2", "s3", "s4"]
        methods = [c[0] for c in comp.calls]
        assert "open_app" in methods
        ui_clicks = [c for c in comp.calls if c[0] == "perform_ui_action"]
        assert ui_clicks and ui_clicks[0][1]["element"].name == "Save"
        assert any(c[0] == "text" and c[1]["text"] == "done" for c in comp.calls)
        # The compiled click is audited like a regular ui action.
        log = get_sandbox("prog-test").export_audit_log()
        assert any(e["params"].get("program_step") == "s2" for e in log)

    @pytest.mark.asyncio
    async def test_start_at_skips_earlier_steps(self):
        comp = _scripted_textedit()
        res = await run_program(_textedit_program(), _ctx(comp), start_at="s3", **_fast())
        assert res.ok and [o.step_id for o in res.outcomes] == ["s3", "s4"]
        assert "open_app" not in [c[0] for c in comp.calls]

    @pytest.mark.asyncio
    async def test_postcondition_failure_returns_self_heal(self):
        comp = _scripted_textedit()
        comp.on_click = None  # clicking Save no longer changes the title
        res = await run_program(_textedit_program(), _ctx(comp), **_fast())
        assert not res.ok
        assert res.failed_step.id == "s2"
        assert "window_title_contains" in res.reason
        assert [s.id for s in res.remaining] == ["s3", "s4"]
        assert res.screenshot  # captured for the agent
        heal = res.self_heal({"description": "save a note", "steps": ["open", "save"]})
        assert heal["failed_step"]["id"] == "s2"
        assert heal["completed_steps"] == ["s1"]
        assert [s["id"] for s in heal["remaining_steps"]] == ["s3", "s4"]
        assert heal["procedure"]["description"] == "save a note"

    @pytest.mark.asyncio
    async def test_missing_target_fails_fast_with_reason(self):
        comp = ScriptedComputer(trees={"TextEdit": UIElement(role="window", name="w", children=[])})
        prog = parse_program([{"tool": "ui", "params": {"action": "click"},
                               "target": {"app": "TextEdit", "title": "Nope"}}], name="x")
        res = await ProgramRunner(_ctx(comp), **_fast()).run(prog)
        assert not res.ok and "target not found" in res.reason

    @pytest.mark.asyncio
    async def test_mouse_step_with_semantic_target_clicks_element_center(self):
        comp = ScriptedComputer(trees={"Finder": UIElement(role="window", name="w", children=[
            UIElement(role="AXRow", name="notes.txt", bounds=Rect(x=100, y=200, width=50, height=10)),
        ])})
        prog = parse_program([{"tool": "mouse", "params": {"action": "double_click"},
                               "target": {"app": "Finder", "role": "row", "title": "notes.txt"}}], name="x")
        res = await run_program(prog, _ctx(comp), **_fast())
        assert res.ok, res.reason
        downs = [c[1]["event"] for c in comp.calls if c[0] == "pointer" and c[1]["event"].action.value == "down"]
        assert len(downs) == 2 and downs[0].point.x == 125 and downs[0].point.y == 205

    @pytest.mark.asyncio
    async def test_text_visible_postcondition(self):
        comp = ScriptedComputer(trees={"Notes": UIElement(role="window", name="w", children=[
            UIElement(role="AXTextArea", name="", value="Shopping: apples"),
        ])})
        ok = parse_program([{"tool": "wait", "params": {"ms": 0}, "target": {"app": "Notes"},
                             "postcondition": {"kind": "text_visible", "value": "apples", "timeout_ms": 50}}],
                           name="t")
        assert (await run_program(ok, _ctx(comp), **_fast())).ok
        bad = parse_program([{"tool": "wait", "params": {"ms": 0}, "target": {"app": "Notes"},
                              "postcondition": {"kind": "text_visible", "value": "pears", "timeout_ms": 50}}],
                            name="t")
        assert not (await run_program(bad, _ctx(comp), **_fast())).ok


class TestLearnRunAndPatch:
    @pytest.mark.asyncio
    async def test_run_failure_then_patch_then_success(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_procedure(tmp_path, "te", {"task_name": "te", "description": "save a note",
                                        "steps": ["Open TextEdit", "Save"], "procedure": "Open and save."})
        save_program(tmp_path, _textedit_program())
        comp = _scripted_textedit()
        comp.on_click = None
        tool = LearnTool()
        ctx = _ctx(comp, "learn-run")

        res = await tool.execute(ctx, LearnTool.Params(action="run", task_name="te"))
        assert res.error
        heal = res.metadata["self_heal"]
        assert heal["failed_step"]["id"] == "s2" and heal["reason"]
        assert [s["id"] for s in heal["remaining_steps"]] == ["s3", "s4"]
        assert heal["procedure"]["steps"] == ["Open TextEdit", "Save"]
        assert res.attachments and res.attachments[0].media_type == "image/png"
        assert "learn(action='patch'" in res.output and "Original procedure" in res.output

        # Agent fixes the step: the postcondition was wrong — the title stays "Untitled".
        fixed = {"tool": "ui", "params": {"action": "click"},
                 "target": {"app": "TextEdit", "role": "button", "title": "Save"},
                 "postcondition": {"kind": "window_title_contains", "value": "Untitled", "timeout_ms": 200},
                 "wait_ms": 0}
        res = await tool.execute(ctx, LearnTool.Params(action="patch", task_name="te", step_id="s2",
                                                       step=json.dumps(fixed)))
        assert not res.error, res.output
        assert load_program(tmp_path, "te").steps[1].postcondition.value == "Untitled"

        res = await tool.execute(ctx, LearnTool.Params(action="run", task_name="te"))
        assert not res.error, res.output
        assert res.metadata["status"] == "ok"

    def test_patch_modes(self):
        prog = _textedit_program()
        patch_program(prog, mode="insert_after", step_id="s1", step={"tool": "wait", "params": {"ms": 5}})
        assert [s.id for s in prog.steps] == ["s1", "s5", "s2", "s3", "s4"]
        patch_program(prog, mode="insert_before", step_id="s1", step={"id": "s2", "tool": "wait"})
        assert prog.steps[0].id == "s6"  # colliding id was renumbered
        patch_program(prog, mode="delete", step_id="s5")
        patch_program(prog, mode="append", step={"tool": "wait"})
        assert [s.id for s in prog.steps] == ["s6", "s1", "s2", "s3", "s4", "s7"]
        with pytest.raises(KeyError):
            patch_program(prog, mode="replace", step_id="zz", step={"tool": "wait"})
        with pytest.raises(ValueError):
            patch_program(prog, mode="replace", step_id="s1")

    @pytest.mark.asyncio
    async def test_existing_actions_still_work(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tool = LearnTool()
        ctx = ToolContext(computer=FakeComputer())
        proc = json.dumps({"task_name": "a", "description": "d", "steps": ["x"], "procedure": "p"})
        assert not (await tool.execute(ctx, LearnTool.Params(action="save", task_name="a", procedure=proc))).error
        rep = await tool.execute(ctx, LearnTool.Params(action="replay", task_name="a"))
        assert "Goal: d" in rep.output
        run = await tool.execute(ctx, LearnTool.Params(action="run", task_name="a"))
        assert run.error and "No compiled program" in run.output


# ---------------------------------------------------------------------------
# Scheduler: replay uses the compiled program first
# ---------------------------------------------------------------------------


class TestSchedulerReplay:
    @pytest.mark.asyncio
    async def test_compiled_program_runs_without_llm(self, tmp_path, monkeypatch):
        save_program(tmp_path, _textedit_program())
        comp = _scripted_textedit()
        monkeypatch.setattr(sched_runner, "_replay_context", lambda name: _ctx(comp, "sched"))

        async def no_llm(*a, **k):
            raise AssertionError("LLM must not be called")
        monkeypatch.setattr(sched_runner, "_run_with_claude", no_llm)
        status = await sched_runner.run_task("replay te", tmp_path)
        assert status.startswith("ok: replayed 4 step(s)")

    @pytest.mark.asyncio
    async def test_falls_back_to_llm_with_self_heal_context(self, tmp_path, monkeypatch):
        save_procedure(tmp_path, "te", {"task_name": "te", "description": "save a note",
                                        "steps": ["Open TextEdit"], "procedure": "Open it."})
        prog = _textedit_program()
        prog.steps[1].target.title = "Missing Button"
        save_program(tmp_path, prog)
        comp = _scripted_textedit()
        monkeypatch.setattr(sched_runner, "_replay_context", lambda name: _ctx(comp, "sched2"))
        from opendesk.automation import program as program_mod
        orig = program_mod.ProgramRunner.__init__

        def fast_init(self, ctx, **kw):
            orig(self, ctx, **_fast())
        monkeypatch.setattr(program_mod.ProgramRunner, "__init__", fast_init)
        prompts = []

        async def fake_llm(prompt, project_dir=None):
            prompts.append(prompt)
            return "ok: finished"
        monkeypatch.setattr(sched_runner, "_run_with_claude", fake_llm)
        status = await sched_runner.run_task("replay te", tmp_path)
        assert status == "compiled replay failed, LLM fallback → ok: finished"
        assert "STOPPED at step s2" in prompts[0] and "Do NOT redo" in prompts[0]

    @pytest.mark.asyncio
    async def test_no_program_uses_llm_path(self, tmp_path, monkeypatch):
        save_procedure(tmp_path, "plain", {"task_name": "plain", "description": "d", "steps": [], "procedure": ""})
        called = []

        async def fake_llm(prompt, project_dir=None):
            called.append(prompt)
            return "ok"
        monkeypatch.setattr(sched_runner, "_run_with_claude", fake_llm)
        assert await sched_runner.run_task("replay plain", tmp_path) == "ok"
        assert "Execute this learned task" in called[0]

    @pytest.mark.asyncio
    async def test_missing_everything(self, tmp_path):
        assert "no procedure found" in await sched_runner.run_task("replay ghost", tmp_path)

    @pytest.mark.asyncio
    async def test_no_api_key_uses_claude_cli(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(sched_runner.shutil, "which", lambda name: "/usr/bin/claude")
        seen = {}

        async def fake_headless(prompt, *, cwd=None, **kw):
            seen["prompt"], seen["cwd"] = prompt, cwd
            from opendesk.eval.agents import AgentRun
            return AgentRun(final_text="all done")
        monkeypatch.setattr("opendesk.eval.backends.run_claude_headless", fake_headless)
        status = await sched_runner.run_task("take a screenshot", tmp_path)
        assert status == "ok (claude-code): all done"
        assert seen == {"prompt": "take a screenshot", "cwd": str(tmp_path)}
