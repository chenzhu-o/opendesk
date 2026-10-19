"""Evaluation harness — measure agents on real desktop tasks.

Run ``opendesk eval run macos-smoke --dry-run`` to see what a suite does, then
``opendesk eval run <suite> --backend ... --model ... --repeat N`` and
``opendesk eval report results.json [more.json ...]``.
"""

from opendesk.eval.agents import AgentRun
from opendesk.eval.backends import (
    AnthropicAPIBackend,
    AnthropicNativeComputerBackend,
    Backend,
    ClaudeCodeBackend,
    FakeBackend,
    ProgramBackend,
    get_backend,
)
from opendesk.eval.harness import EvalRunner, SuiteResult, TrialResult, dry_run_plan, load_results
from opendesk.eval.report import render_report
from opendesk.eval.task import EvalTask, load_tasks

__all__ = [
    "AgentRun",
    "AnthropicAPIBackend",
    "AnthropicNativeComputerBackend",
    "Backend",
    "ClaudeCodeBackend",
    "EvalRunner",
    "EvalTask",
    "FakeBackend",
    "ProgramBackend",
    "SuiteResult",
    "TrialResult",
    "dry_run_plan",
    "get_backend",
    "load_results",
    "load_tasks",
    "render_report",
]
