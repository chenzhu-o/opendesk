"""Run eval tasks: setup → agent → check → teardown, repeated N times."""

from __future__ import annotations

import asyncio
import json
import platform as _platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from opendesk.eval.backends import Backend, ProgramBackend
from opendesk.eval.task import EvalTask, action_to_shell, describe_action, substitute

RESULTS_SCHEMA = "opendesk-eval/1"
SHELL_TIMEOUT_S = 60


@dataclass
class TrialResult:
    task_id: str
    trial: int
    backend: str
    model: str
    status: str  # pass | fail | setup_error | agent_error | timeout | skipped
    success: bool
    check_returncode: Optional[int] = None
    turns: Optional[int] = None
    wall_time_s: float = 0.0
    agent_time_s: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None
    final_text: str = ""
    tags: list[str] = field(default_factory=list)
    suite: str = ""


@dataclass
class SuiteResult:
    suite: str
    backend: str
    model: str
    repeat: int
    started_at: float
    finished_at: float = 0.0
    host: str = field(default_factory=lambda: f"{_platform.system()} {_platform.release()}")
    results: list[TrialResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema"] = RESULTS_SCHEMA
        n = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        d["summary"] = {"trials": n, "passed": passed, "success_rate": (passed / n) if n else None}
        return d

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def load_results(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("schema") != RESULTS_SCHEMA:
        raise ValueError(f"{path}: not an opendesk eval results file")
    return data


# ---------------------------------------------------------------------------
# Shell / setup helpers
# ---------------------------------------------------------------------------


def _run_shell(cmd: str, *, cwd: str, timeout: float = SHELL_TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout,
    )


async def run_action(action: Any, *, workdir: str) -> None:
    """Execute one (already substituted) setup/teardown action; raise on failure."""
    if isinstance(action, dict) and "write_file" in action:
        p = Path(action["write_file"]).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(action.get("content", "")))
        return
    if isinstance(action, dict) and "sleep" in action:
        await asyncio.sleep(float(action["sleep"]))
        return
    cmd = action_to_shell(action)
    if cmd is None:
        raise ValueError(f"unknown setup action: {action!r}")
    cp = await asyncio.to_thread(_run_shell, cmd, cwd=workdir)
    if cp.returncode != 0:
        raise RuntimeError(f"`{cmd}` exited {cp.returncode}: {(cp.stderr or cp.stdout).strip()[:300]}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvalRunner:
    def __init__(
        self,
        backend: Backend,
        *,
        program_backend: Optional[Backend] = None,
        on_trial: Optional[Callable[[TrialResult], None]] = None,
        keep_workdir: bool = False,
    ) -> None:
        self.backend = backend
        self.program_backend = program_backend or ProgramBackend(model=backend.model)
        self.on_trial = on_trial
        self.keep_workdir = keep_workdir

    def backend_for(self, task: EvalTask) -> Backend:
        return self.program_backend if task.is_program_task() else self.backend

    async def run_trial(self, task: EvalTask, trial: int) -> TrialResult:
        backend = self.backend_for(task)
        res = TrialResult(
            task_id=task.id, trial=trial, backend=backend.name, model=self.backend.model,
            status="fail", success=False, tags=list(task.tags), suite=task.suite,
        )
        if not task.supported_here():
            res.status, res.error = "skipped", f"task requires platform {task.platform!r}"
            return res

        t0 = time.monotonic()
        workdir = tempfile.mkdtemp(prefix=f"opendesk-eval-{_slug(task.id)}-")
        sub = lambda v: substitute(v, workdir=workdir, task_id=task.id)  # noqa: E731
        try:
            try:
                for action in sub(task.setup):
                    await run_action(action, workdir=workdir)
            except Exception as exc:
                res.status, res.error = "setup_error", str(exc)
                return res

            a0 = time.monotonic()
            run = await backend.run(task, sub(task.instruction), workdir=workdir, timeout=float(task.timeout))
            res.agent_time_s = round(time.monotonic() - a0, 2)
            res.turns, res.input_tokens, res.output_tokens = run.turns, run.input_tokens, run.output_tokens
            res.cost_usd, res.final_text = run.cost_usd, (run.final_text or "")[:500]
            if run.error:
                res.error = run.error

            if task.check:
                try:
                    cp = await asyncio.to_thread(_run_shell, sub(task.check), cwd=workdir)
                    res.check_returncode = cp.returncode
                    res.success = cp.returncode == 0
                except subprocess.TimeoutExpired:
                    res.check_returncode, res.success = None, False
                    res.error = (res.error + "; " if res.error else "") + "check timed out"
            else:
                # No check: success means the agent/program reported no error.
                res.success = run.error is None

            if res.success:
                res.status = "pass"
            elif run.stop_reason == "timeout":
                res.status = "timeout"
            elif run.error and res.check_returncode is None:
                res.status = "agent_error"
            else:
                res.status = "fail"
            return res
        finally:
            for action in sub(task.teardown):
                try:
                    await run_action(action, workdir=workdir)
                except Exception:
                    pass
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)
            res.wall_time_s = round(time.monotonic() - t0, 2)
            if self.on_trial is not None:
                self.on_trial(res)

    async def run(self, suite: str, tasks: list[EvalTask], *, repeat: int = 1) -> SuiteResult:
        out = SuiteResult(
            suite=suite, backend=self.backend.name, model=self.backend.model,
            repeat=repeat, started_at=time.time(),
        )
        for trial in range(1, repeat + 1):
            for task in tasks:
                out.results.append(await self.run_trial(task, trial))
        out.finished_at = time.time()
        return out


def dry_run_plan(tasks: list[EvalTask], backend: Backend, *, repeat: int = 1,
                 program_backend: Optional[Backend] = None) -> list[dict]:
    """Describe every step a real run would take — executes nothing."""
    program_backend = program_backend or ProgramBackend(model=backend.model)
    workdir = "<workdir>"
    plan = []
    for task in tasks:
        sub = lambda v: substitute(v, workdir=workdir, task_id=task.id)  # noqa: E731
        be = program_backend if task.is_program_task() else backend
        plan.append({
            "task": task.id,
            "suite": task.suite,
            "tags": task.tags,
            "runs_here": task.supported_here(),
            "repeat": repeat,
            "timeout_s": task.timeout,
            "setup": [describe_action(a) for a in sub(task.setup)],
            "agent": be.describe(task, sub(task.instruction), workdir=workdir),
            "check": sub(task.check) if task.check else "(agent/program must report no error)",
            "teardown": [describe_action(a) for a in sub(task.teardown)],
        })
    return plan


def format_plan(plan: list[dict]) -> str:
    import shlex

    from opendesk.eval.agents import EVAL_SYSTEM_PROMPT

    lines: list[str] = []
    shown_shared = False
    for item in plan:
        agent = item["agent"]
        if "command" in agent and not shown_shared:
            lines.append(f"MCP config (written to a temp file per trial): {json.dumps(agent.get('mcp_config'))}")
            lines.append(f"$EVAL_SYSTEM_PROMPT = {EVAL_SYSTEM_PROMPT!r}")
            lines.append("")
            shown_shared = True
        here = "" if item["runs_here"] else "   [skipped on this platform]"
        lines.append(f"── {item['task']}  (x{item['repeat']}, timeout {item['timeout_s']}s){here}")
        for s in item["setup"]:
            lines.append(f"   setup    $ {_one_line(s)}")
        if "command" in agent:
            cmd = shlex.join(agent["command"]).replace(shlex.quote(EVAL_SYSTEM_PROMPT), '"$EVAL_SYSTEM_PROMPT"')
            lines.append(f"   agent    $ (cd {agent.get('cwd')} && {cmd})")
        elif "request" in agent:
            req = agent["request"]
            tools = req.get("tools")
            tools_s = ", ".join(t if isinstance(t, str) else json.dumps(t) for t in tools) if tools else ""
            lines.append(f"   agent    {req.get('endpoint')} model={req.get('model')} max_turns={req.get('max_turns')}")
            lines.append(f"            tools: {tools_s}")
            lines.append(f"            instruction: {_one_line(agent.get('instruction', ''))}")
        else:
            lines.append(f"   agent    {agent.get('action') or json.dumps(agent)}")
        lines.append(f"   check    $ {_one_line(item['check'])}")
        for s in item["teardown"]:
            lines.append(f"   teardown $ {_one_line(s)}")
    return "\n".join(lines)


def _one_line(s: str, limit: int = 220) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:40]
