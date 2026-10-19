"""Tests for the evaluation harness (no real agents, no desktop actions)."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from opendesk.eval.agents import (
    AgentRun,
    NativeComputerExecutor,
    run_native_computer_loop,
    run_opendesk_loop,
)
from opendesk.eval.backends import (
    AnthropicAPIBackend,
    AnthropicNativeComputerBackend,
    ClaudeCodeBackend,
    FakeBackend,
    ProgramBackend,
    build_claude_code_command,
    get_backend,
    parse_claude_code_output,
)
from opendesk.eval.harness import EvalRunner, dry_run_plan, format_plan, load_results
from opendesk.eval.osworld import OpenDeskOSWorldAgent, stratified_subset
from opendesk.eval.report import render_report
from opendesk.eval.suites import BUILTIN_SUITES, builtin_suite
from opendesk.eval.task import EvalTask, action_to_shell, load_tasks, substitute

from tests._fakes import FakeComputer


def _task(**kw) -> EvalTask:
    base = {"id": "t1", "instruction": "do it in {workdir}", "platform": None}
    base.update(kw)
    return EvalTask.from_dict(base)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


class TestTasks:
    def test_builtin_suites_are_well_formed(self):
        for name in BUILTIN_SUITES:
            tasks = builtin_suite(name)
            ids = [t.id for t in tasks]
            assert len(ids) == len(set(ids)), name
            for t in tasks:
                assert t.check, t.id
                assert t.teardown or name == "macos-smoke" and not t.setup, t.id
        smoke = builtin_suite("macos-smoke")
        assert 6 <= len(smoke) <= 10
        assert all(t.platform == "darwin" for t in smoke)
        for t in builtin_suite("injection"):
            # Passing requires the benign result AND no sign of obeying the planted instruction.
            assert "opendesk-eval-pwned" in t.check and "&&" in t.check
            assert any("rm -f" in str(a) for a in t.teardown)
        lr = builtin_suite("learn-regression")
        assert lr[0].is_program_task() and lr[0].program_json["steps"]

    def test_learn_regression_includes_project_programs(self, tmp_path):
        learned = tmp_path / ".opendesk" / "learned"
        learned.mkdir(parents=True)
        (learned / "expense.program.json").write_text("{}")
        (learned / "expense.json").write_text("{}")
        tasks = builtin_suite("learn-regression", project_dir=tmp_path)
        user = [t for t in tasks if t.id == "program:expense"]
        assert user and user[0].project == str(tmp_path.resolve()) and user[0].program == "expense"

    def test_load_file_list_suite_and_dir(self, tmp_path):
        single = tmp_path / "one.json"
        single.write_text(json.dumps({"id": "a", "instruction": "x", "setup": "echo hi"}))
        name, tasks = load_tasks(str(single))
        assert name == "one" and tasks[0].setup == ["echo hi"]

        suite = tmp_path / "suite.json"
        suite.write_text(json.dumps({"suite": "mine", "tasks": [{"id": "b", "instruction": "y"}]}))
        name, tasks = load_tasks(str(suite))
        assert name == "mine" and tasks[0].suite == "mine"

        d = tmp_path / "dir"
        d.mkdir()
        (d / "1.json").write_text(json.dumps([{"id": "c", "instruction": "z"}]))
        (d / "2.json").write_text(json.dumps({"id": "d", "instruction": "w"}))
        name, tasks = load_tasks(str(d))
        assert name == "dir" and [t.id for t in tasks] == ["c", "d"]

        name, tasks = load_tasks("macos-smoke")
        assert name == "macos-smoke" and tasks

    def test_load_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tasks(str(tmp_path / "nope.json"))
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"id": "a", "instruction": "x", "bogus": 1}))
        with pytest.raises(ValueError, match="unknown fields"):
            load_tasks(str(bad))
        bad.write_text(json.dumps({"id": "a"}))
        with pytest.raises(ValueError, match="instruction"):
            load_tasks(str(bad))
        bad.write_text(json.dumps([{"id": "a", "instruction": "x"}, {"id": "a", "instruction": "y"}]))
        with pytest.raises(ValueError, match="duplicate"):
            load_tasks(str(bad))

    def test_substitute_and_shell_actions(self):
        out = substitute({"a": ["{workdir}/x", "{task_id}"], "b": "{home}"}, workdir="/w", task_id="t")
        assert out == {"a": ["/w/x", "t"], "b": os.path.expanduser("~")}
        assert action_to_shell({"open_app": "Text Edit"}, plat="darwin") == "open -a 'Text Edit'"
        assert action_to_shell({"open": "/w/a b.txt", "app": "TextEdit"}, plat="darwin") == \
            "open -a TextEdit '/w/a b.txt'"
        assert "xdg-open" in action_to_shell({"open": "/w/a"}, plat="linux")
        assert action_to_shell({"write_file": "/x", "content": ""}) is None


# ---------------------------------------------------------------------------
# Harness with a fake backend
# ---------------------------------------------------------------------------


class TestHarness:
    @pytest.mark.asyncio
    async def test_pass_fail_teardown_and_cleanup(self, tmp_path):
        marker = tmp_path / "teardown-ran"
        seen_workdirs = []

        def script(task, instruction, workdir):
            seen_workdirs.append(workdir)
            assert instruction == f"do it in {workdir}"
            if task.id == "good":
                Path(workdir, "out.txt").write_text("ok")
            return AgentRun(final_text="done", turns=3, input_tokens=100, output_tokens=20, cost_usd=0.01)

        common = {
            "instruction": "do it in {workdir}",
            "setup": [{"write_file": "{workdir}/in.txt", "content": "seed"}, "test -f {workdir}/in.txt"],
            "check": "test -f {workdir}/out.txt",
            "teardown": [f"touch {marker}"],
            "tags": ["demo"],
        }
        tasks = [EvalTask.from_dict({"id": "good", **common}), EvalTask.from_dict({"id": "bad", **common})]
        backend = FakeBackend(script, model="m")
        result = await EvalRunner(backend).run("s", tasks, repeat=2)

        by = [(r.task_id, r.trial, r.status) for r in result.results]
        assert by == [("good", 1, "pass"), ("bad", 1, "fail"), ("good", 2, "pass"), ("bad", 2, "fail")]
        good = result.results[0]
        assert good.turns == 3 and good.input_tokens == 100 and good.check_returncode == 0
        assert good.backend == "fake" and good.model == "m" and good.tags == ["demo"]
        assert marker.exists()
        assert all(not Path(w).exists() for w in seen_workdirs)  # temp dirs removed

        out = result.save(tmp_path / "res.json")
        data = load_results(out)
        assert data["summary"] == {"trials": 4, "passed": 2, "success_rate": 0.5}

    @pytest.mark.asyncio
    async def test_setup_error_skip_and_agent_error(self, tmp_path):
        backend = FakeBackend(lambda t, i, w: AgentRun(error="model refused"))
        tasks = [
            _task(id="setup-fails", setup=["exit 3"], check="true"),
            _task(id="other-os", platform="plan9"),
            _task(id="agent-err"),  # no check: success == agent reported no error
        ]
        res = await EvalRunner(backend).run("s", tasks)
        statuses = {r.task_id: r.status for r in res.results}
        assert statuses == {"setup-fails": "setup_error", "other-os": "skipped", "agent-err": "agent_error"}
        assert "exited 3" in res.results[0].error
        assert backend.calls and [c[0] for c in backend.calls] == ["agent-err"]

    @pytest.mark.asyncio
    async def test_program_tasks_use_program_backend(self, tmp_path):
        prog = {"name": "ignored", "steps": [
            {"id": "s1", "tool": "clipboard", "params": {"action": "write", "text": "from program"},
             "wait_ms": 0},
        ]}
        task = _task(id="lr", instruction="", program="demo", program_json=prog,
                     check="test -f {workdir}/.opendesk/learned/demo.program.json")
        fake_agent = FakeBackend()
        comp = FakeComputer()
        runner = EvalRunner(fake_agent, program_backend=ProgramBackend(computer_factory=lambda: comp))
        res = await runner.run("learn-regression", [task])
        assert res.results[0].status == "pass", res.results[0].error
        assert res.results[0].backend == "program" and res.results[0].turns == 0
        assert fake_agent.calls == []
        assert comp._clipboard == "from program"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _results(backend, rows):
    return {"schema": "opendesk-eval/1", "suite": "macos-smoke", "backend": backend, "model": "m",
            "host": "Darwin 25", "results": rows}


def _row(tid, ok, **kw):
    r = {"task_id": tid, "trial": 1, "status": "pass" if ok else "fail", "success": ok,
         "turns": 4, "wall_time_s": 10.0, "input_tokens": 1000, "output_tokens": 100,
         "cost_usd": 0.05, "tags": ["textedit"], "error": None if ok else "check failed",
         "check_returncode": 0 if ok else 1}
    r.update(kw)
    return r


class TestReport:
    def test_markdown_table_two_arms(self):
        a = _results("claude-code", [_row("t1", True), _row("t2", True)])
        b = _results("anthropic-native-computer", [_row("t1", True), _row("t2", False, cost_usd=None)])
        md = render_report([a, b])
        assert "| Arm | Success |" in md
        assert "| claude-code · m | 2/2 (100%)" in md
        assert "| anthropic-native-computer · m | 1/2 (50%)" in md
        assert "Δ success (claude-code · m − anthropic-native-computer · m): +50.0 pts" in md
        assert "| t2 |" in md and "## Per tag" in md and "## Failures" in md
        assert "**t2** #1 [anthropic-native-computer · m] fail: check failed" in md

    def test_skipped_rows_are_not_counted(self):
        md = render_report([_results("x", [_row("t1", True), _row("t2", False, status="skipped")])])
        assert "1/1 (100%)" in md and "skipped" in md


# ---------------------------------------------------------------------------
# Backends — dry-run command construction
# ---------------------------------------------------------------------------


class TestBackendsDryRun:
    def test_claude_code_command(self):
        task = _task(max_turns=12)
        be = get_backend("claude-code", model="claude-opus-5-5", max_turns=40)
        d = be.describe(task, "open calculator", workdir="/w")
        cmd = d["command"]
        assert cmd[:3] == ["claude", "-p", "open calculator"]
        assert cmd[cmd.index("--allowedTools") + 1] == "mcp__opendesk__*"
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert cmd[cmd.index("--max-turns") + 1] == "12"  # task override wins
        assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
        assert cmd[cmd.index("--tools") + 1] == ""  # built-in tools disabled
        assert "--strict-mcp-config" in cmd and "--mcp-config" in cmd
        server = d["mcp_config"]["mcpServers"]["opendesk"]
        assert "opendesk" in server["command"] or server["args"][-1] == "opendesk.integrations.mcp"
        assert d["cwd"] == "/w"

    def test_build_command_without_model(self):
        cmd = build_claude_code_command("x", mcp_config_path="/c.json", model=None, max_turns=None,
                                        system_prompt=None)
        assert "--model" not in cmd and "--max-turns" not in cmd and "--append-system-prompt" not in cmd

    def test_api_backends_describe(self):
        task = _task()
        api = AnthropicAPIBackend(model="m", max_turns=7).describe(task, "go", workdir="/w")
        assert api["request"]["model"] == "m" and api["request"]["max_turns"] == 7
        assert {"ui", "screenshot", "keyboard"} <= set(api["request"]["tools"])
        native = AnthropicNativeComputerBackend(model="m").describe(task, "go", workdir="/w")
        assert native["request"]["tools"] == [{"type": "computer_toolset_20260801"}]
        legacy = AnthropicNativeComputerBackend(model="m", tool_version="computer_20251124")
        assert legacy.describe(task, "go", workdir="/w")["request"]["tools"][0]["name"] == "computer"
        with pytest.raises(ValueError):
            get_backend("nope")

    def test_dry_run_plan_executes_nothing(self, tmp_path):
        sentinel = tmp_path / "should-not-exist"
        task = _task(setup=[f"touch {sentinel}", {"write_file": str(sentinel), "content": "x"}],
                     check="true", teardown=[f"rm -f {sentinel}"])
        for name in ("claude-code", "anthropic-api", "anthropic-native-computer"):
            plan = dry_run_plan([task], get_backend(name, model="m"), repeat=3)
            text = format_plan(plan)
            assert plan[0]["repeat"] == 3 and plan[0]["agent"]["backend"] == name
            assert "do it in <workdir>" in text
        assert not sentinel.exists()
        lr = builtin_suite("learn-regression")
        plan = dry_run_plan(lr, get_backend("claude-code"))
        assert plan[0]["agent"]["backend"] == "program"

    def test_parse_claude_output(self):
        out = json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 6,
                          "result": "Done.", "total_cost_usd": 0.12, "duration_ms": 5000,
                          "usage": {"input_tokens": 10, "cache_read_input_tokens": 90,
                                    "cache_creation_input_tokens": 0, "output_tokens": 40}})
        run = parse_claude_code_output(out)
        assert (run.turns, run.cost_usd, run.input_tokens, run.output_tokens) == (6, 0.12, 100, 40)
        assert run.final_text == "Done." and run.error is None
        maxed = parse_claude_code_output(json.dumps({"subtype": "error_max_turns", "is_error": True,
                                                     "num_turns": 40}))
        assert maxed.error is None and maxed.stop_reason == "error_max_turns"
        assert parse_claude_code_output("garbage").error

    @pytest.mark.asyncio
    async def test_claude_code_run_with_stub_binary(self, tmp_path):
        args_file = tmp_path / "args.json"
        stub = tmp_path / "claude"
        stub.write_text(
            f"#!{sys.executable}\nimport json, sys, os\n"
            f"json.dump({{'argv': sys.argv[1:], 'cwd': os.getcwd()}}, open({str(args_file)!r}, 'w'))\n"
            "cfg = json.load(open(sys.argv[sys.argv.index('--mcp-config') + 1]))\n"
            "assert 'opendesk' in cfg['mcpServers']\n"
            "print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,"
            " 'num_turns': 2, 'result': 'ok', 'total_cost_usd': 0.01,"
            " 'usage': {'input_tokens': 5, 'output_tokens': 7}}))\n"
        )
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        workdir = tmp_path / "wd"
        workdir.mkdir()
        be = ClaudeCodeBackend(claude_bin=str(stub), model="sonnet", max_turns=5)
        run = await be.run(_task(), "hello", workdir=str(workdir), timeout=30)
        assert run.error is None and run.turns == 2 and run.output_tokens == 7
        seen = json.loads(args_file.read_text())
        assert seen["argv"][:2] == ["-p", "hello"] and Path(seen["cwd"]).resolve() == workdir.resolve()
        cfg_path = seen["argv"][seen["argv"].index("--mcp-config") + 1]
        assert not Path(cfg_path).exists()  # temp MCP config cleaned up
        assert list(workdir.iterdir()) == []  # nothing written into the task's workdir

    @pytest.mark.asyncio
    async def test_claude_code_missing_binary(self, tmp_path):
        be = ClaudeCodeBackend(claude_bin=str(tmp_path / "no-such-claude"))
        run = await be.run(_task(), "x", workdir=str(tmp_path), timeout=5)
        assert "not found" in run.error


# ---------------------------------------------------------------------------
# Agent loops with a fake Messages client
# ---------------------------------------------------------------------------


def _resp(content, stop="tool_use", usage=(10, 5)):
    return SimpleNamespace(content=content, stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1],
                                                 cache_read_input_tokens=0, cache_creation_input_tokens=0))


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})  # snapshot: the loop appends
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)
        self.beta = SimpleNamespace(messages=self.messages)


class TestAgentLoops:
    @pytest.mark.asyncio
    async def test_opendesk_loop_dispatches_tools(self):
        comp = FakeComputer()
        client = FakeClient([
            _resp([SimpleNamespace(type="tool_use", id="t1", name="clipboard",
                                   input={"action": "write", "text": "42"})]),
            _resp([SimpleNamespace(type="text", text="Copied 42.")], stop="end_turn"),
        ])
        run = await run_opendesk_loop(client, model="m", instruction="copy 42", computer=comp, max_turns=5)
        assert run.error is None and run.turns == 2 and run.final_text == "Copied 42."
        assert (run.input_tokens, run.output_tokens) == (20, 10)
        assert comp._clipboard == "42"
        second = client.messages.requests[1]["messages"]
        assert second[-1]["content"][0]["tool_use_id"] == "t1"
        assert any(t["name"] == "ui" for t in client.messages.requests[0]["tools"])

    @pytest.mark.asyncio
    async def test_opendesk_loop_respects_max_turns(self):
        block = SimpleNamespace(type="tool_use", id="t", name="clipboard", input={"action": "read"})
        client = FakeClient([_resp([block]) for _ in range(3)])
        run = await run_opendesk_loop(client, model="m", instruction="x", computer=FakeComputer(), max_turns=3)
        assert run.turns == 3 and run.stop_reason == "max_turns"

    @pytest.mark.asyncio
    async def test_native_toolset_loop(self):
        comp = FakeComputer()
        client = FakeClient([
            _resp([
                SimpleNamespace(type="tool_use", id="a", name="screenshot", input={}, toolset_name="computer"),
                SimpleNamespace(type="tool_use", id="b", name="left_click", input={"coordinate": [50, 50]},
                                toolset_name="computer"),
                SimpleNamespace(type="tool_use", id="c", name="key", input={"text": "ctrl+s"},
                                toolset_name="computer"),
            ]),
            _resp([SimpleNamespace(type="text", text="done")], stop="end_turn"),
        ])
        run = await run_native_computer_loop(client, model="m", instruction="x", computer=comp)
        assert run.error is None and run.turns == 2
        req = client.messages.requests[0]
        assert req["tools"] == [{"type": "computer_toolset_20260801"}] and "betas" not in req
        results = client.messages.requests[1]["messages"][-1]["content"]
        assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]
        assert all(r["toolset_name"] == "computer" for r in results)
        assert results[0]["content"][0]["type"] == "image"
        assert results[1]["content"][0]["text"] == "OK"
        downs = [c[1]["event"] for c in comp.calls if c[0] == "pointer" and c[1]["event"].action.value == "down"]
        assert downs and downs[0].point.x == 50  # pixel space == logical for the fake capture
        keys = [c[1]["event"].keysym for c in comp.calls if c[0] == "key"]
        assert keys[:2] == ["ctrl", "s"]

    @pytest.mark.asyncio
    async def test_native_legacy_tool_uses_beta(self):
        client = FakeClient([_resp([SimpleNamespace(type="tool_use", id="a", name="computer",
                                                    input={"action": "screenshot"})]),
                             _resp([], stop="end_turn")])
        run = await run_native_computer_loop(client, model="m", instruction="x", computer=FakeComputer(),
                                             tool_version="computer_20251124")
        req = client.messages.requests[0]
        assert req["betas"] == ["computer-use-2025-11-24"]
        assert req["tools"][0]["display_width_px"] == 1920
        res = client.messages.requests[1]["messages"][-1]["content"][0]
        assert "toolset_name" not in res and res["content"][0]["type"] == "image"
        assert run.turns == 2

    @pytest.mark.asyncio
    async def test_native_executor_errors_become_tool_errors(self):
        client = FakeClient([_resp([SimpleNamespace(type="tool_use", id="a", name="teleport", input={},
                                                    toolset_name="computer")]),
                             _resp([], stop="end_turn")])
        await run_native_computer_loop(client, model="m", instruction="x", computer=FakeComputer())
        res = client.messages.requests[1]["messages"][-1]["content"][0]
        assert res["is_error"] is True and "unsupported" in res["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_executor_scales_coordinates(self):
        from opendesk.computer import Pixmap, PixmapFormat

        class Retina(FakeComputer):
            async def capture(self, *, display_id=None, region=None, downscale=True):
                return Pixmap(data=b"png", format=PixmapFormat.PNG, width=1000, height=500,
                              logical_width=2000, logical_height=1000)
        comp = Retina()
        ex = NativeComputerExecutor(comp)
        await ex.execute("screenshot", {})
        await ex.execute("left_click", {"coordinate": [100, 50]})
        down = next(c[1]["event"] for c in comp.calls if c[0] == "pointer" and c[1]["event"].action.value == "down")
        assert (down.point.x, down.point.y) == (200, 100)


# ---------------------------------------------------------------------------
# OSWorld adapter
# ---------------------------------------------------------------------------


class TestOSWorld:
    def test_stratified_subset(self):
        test_all = {"chrome": [f"c{i}" for i in range(46)], "os": [f"o{i}" for i in range(24)],
                    "vlc": [f"v{i}" for i in range(17)], "tiny": ["t0"]}
        sub = stratified_subset(test_all, n=40, seed=1)
        assert sum(len(v) for v in sub.values()) == 40
        assert all(len(v) >= 1 for v in sub.values())
        assert len(sub["chrome"]) > len(sub["os"]) > len(sub["vlc"])
        assert sub == stratified_subset(test_all, n=40, seed=1)
        assert sub != stratified_subset(test_all, n=40, seed=2)
        assert sum(len(v) for v in stratified_subset(test_all, n=500).values()) == 88
        with pytest.raises(ValueError):
            stratified_subset(test_all, n=2)

    def test_agent_predict_done_and_infeasible(self):
        def make(final):
            return lambda: FakeClient([_resp([SimpleNamespace(type="text", text=final)], stop="end_turn")])
        for arm in ("opendesk", "native"):
            agent = OpenDeskOSWorldAgent(model="m", arm=arm, client_factory=make("Finished."),
                                         computer_factory=FakeComputer)
            agent.reset()
            assert agent.predict("do x", {}) == ("Finished.", ["DONE"])
            assert agent.predict("do x", {})[1] == ["DONE"]  # already finished
        agent = OpenDeskOSWorldAgent(client_factory=make("INFEASIBLE"), computer_factory=FakeComputer)
        assert agent.predict("do y", {})[1] == ["FAIL"]

    def test_agent_unreachable_peer_fails(self):
        def boom():
            raise ConnectionError("no route")
        agent = OpenDeskOSWorldAgent(computer_factory=boom, client_factory=lambda: None)
        text, actions = agent.predict("x", {})
        assert actions == ["FAIL"] and "no route" in text
        with pytest.raises(ValueError):
            OpenDeskOSWorldAgent(arm="other")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def _main(self, monkeypatch, argv):
        from opendesk.cli import main
        monkeypatch.setattr(sys, "argv", ["opendesk", *argv])
        with pytest.raises(SystemExit) as exc:
            main()
        return exc.value.code

    def test_dry_run_and_report(self, monkeypatch, capsys, tmp_path):
        tasks = tmp_path / "t.json"
        tasks.write_text(json.dumps([{"id": "a", "instruction": "hi", "check": "true", "tags": ["x"]},
                                     {"id": "b", "instruction": "yo", "tags": ["y"]}]))
        assert self._main(monkeypatch, ["eval", "run", str(tasks), "--dry-run", "--tag", "x",
                                        "--model", "claude-sonnet-5-5", "--repeat", "3"]) == 0
        out = capsys.readouterr().out
        assert "DRY RUN" in out and "── a  (x3" in out and "── b" not in out
        assert "--model claude-sonnet-5-5" in out

        assert self._main(monkeypatch, ["eval", "run", "macos-smoke", "--dry-run", "--json"]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert len(plan) == len(builtin_suite("macos-smoke"))

        res = tmp_path / "r.json"
        res.write_text(json.dumps(_results("claude-code", [_row("a", True)])))
        assert self._main(monkeypatch, ["eval", "report", str(res)]) == 0
        assert "| claude-code · m | 1/1 (100%)" in capsys.readouterr().out
        assert self._main(monkeypatch, ["eval", "report", str(tasks)]) == 2  # not a results file

    def test_unknown_suite(self, monkeypatch, capsys):
        assert self._main(monkeypatch, ["eval", "run", "no-such-suite", "--dry-run"]) == 2
        assert "Built-in suites" in capsys.readouterr().err
