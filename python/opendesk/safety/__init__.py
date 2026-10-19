"""opendesk.safety — policy, approvals, kill switch, injection guard, chained audit, secrets.

Everything an agent does through opendesk's integrations (MCP server,
OpenAI / Anthropic / LangChain adapters) passes :func:`run_tool`.  See
``docs/safety.md`` for the user-facing guide.

Quick reference::

    from opendesk.safety import classify, decide, run_tool

    classify("script", {"action": "shell"}).level      # RiskLevel.HIGH
    decide("keyboard", {"action": "type"}).decision     # "allow" (standard preset)
    result = await run_tool(tool, ctx, {"action": "type", "text": "{{secret:github}}"})

Files under the opendesk home (``ctx.metadata["opendesk_home"]`` →
``$OPENDESK_HOME`` → ``~/.opendesk``):

* ``policy.json``  — preset, ordered rules, kill hotkey, approval, taint window
* ``control.json`` — running | stopped | takeover
* ``secrets.json`` — secret names + settings (values live in the OS store)
* ``audit/tools-YYYY-MM-DD.jsonl`` — hash-chained tool-call log

Set ``OPENDESK_SAFETY=off`` to disable the guard entirely.
"""

from opendesk.safety._home import resolve_home, safety_enabled
from opendesk.safety.approval import ApprovalOutcome, ApprovalRequest, request_approval
from opendesk.safety.control import ControlState, check_control, read_control, set_mode
from opendesk.safety.guard import guarded_call, run_tool
from opendesk.safety.injection import scan, untrusted_source, wrap
from opendesk.safety.policy import (
    Decision,
    PolicyConfig,
    PolicyEngine,
    Risk,
    RiskLevel,
    classify,
    decide,
    load_policy,
    save_policy,
)

__all__ = [
    "ApprovalOutcome",
    "ApprovalRequest",
    "ControlState",
    "Decision",
    "PolicyConfig",
    "PolicyEngine",
    "Risk",
    "RiskLevel",
    "check_control",
    "classify",
    "decide",
    "guarded_call",
    "load_policy",
    "read_control",
    "request_approval",
    "resolve_home",
    "run_tool",
    "safety_enabled",
    "save_policy",
    "scan",
    "set_mode",
    "untrusted_source",
    "wrap",
]
