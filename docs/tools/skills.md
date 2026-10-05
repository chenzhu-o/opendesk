# `skill` — Persistent Parameterised Procedures

The `skill` tool saves, retrieves, and runs reusable skills. A *skill* is an
ordered list of tool calls with `{{placeholder}}` parameters that can be
re-bound on every run — a *policy*, not a frozen recording.

This is the lightweight version of the "persistent skill" idea from recent
computer-use work: interaction traces become reusable policies that accumulate
across tasks, instead of being re-planned from scratch every time.

## How it differs from `learn`

| | `learn` | `skill` |
|---|---|---|
| Captures | one concrete trajectory | a parameterised procedure |
| Stored as | prose steps + screenshots | tool calls with `{{params}}` |
| Reuse | re-plan each time | re-bind parameters |
| Retrieval | by name | by relevance + applicability |
| Execution | returns instructions for the model | runs the steps directly |

They compose: record with `learn`, then distil the winning procedure into a
`skill` so it can run unattended later.

## Actions

| Action | Required | What it does |
|---|---|---|
| `save` | `name`, `steps` | Store a skill (optionally `description`, `params`, `tags`, `applicability`) |
| `find` | `query` | Rank skills by relevance; return each skill's params schema and applicability |
| `run` | `name` | Bind `arguments` and execute the steps (or return the plan with `execute=false`) |
| `list` | — | List saved skills |
| `delete` | `name` | Remove a skill |

## Skill format

```json
{
  "name": "open_invoice_portal",
  "description": "Open the billing portal and download the latest invoice",
  "tags": ["browser", "finance"],
  "params": {
    "month": {"type": "string", "required": true},
    "dest":  {"type": "string", "default": "~/Downloads"}
  },
  "steps": [
    {"tool": "system", "params": {"action": "mkdir", "path": "{{dest}}"}},
    {"tool": "ui",     "params": {"action": "click", "title": "Invoices"}}
  ],
  "applicability": {"os": ["darwin", "linux"], "apps": ["Google Chrome"]}
}
```

A string that is *exactly* one placeholder adopts the bound value's type, so
`"{{count}}"` can yield an `int`; mixed strings such as `"step-{{count}}"` are
stringified.

## Examples

```python
from opendesk.registry import create_registry
from opendesk.tools.base import allow_all_context

tools = create_registry()
ctx = allow_all_context()
skill = tools.get("skill")

# Save a parameterised skill
await skill.execute(ctx, skill.parse_params({
    "action": "save",
    "name": "scaffold_project",
    "description": "Create a project folder",
    "params": '{"dest": {"type": "string", "required": true}}',
    "steps":  '[{"tool": "system", "params": {"action": "mkdir", "path": "{{dest}}"}}]',
    "tags": ["setup"],
}))

# Retrieve it later
await skill.execute(ctx, skill.parse_params({
    "action": "find", "query": "create a new project folder",
}))

# Run it with concrete arguments
await skill.execute(ctx, skill.parse_params({
    "action": "run", "name": "scaffold_project",
    "arguments": {"dest": "~/work/new-app"},
}))
```

## Design notes

**Relevance is not applicability.** `find` returns a relevance score *and* the
skill's declared `applicability` (OS, apps). A skill can look like the right
one and still not work in the current environment — the caller is given what
it needs to tell the two apart.

**Execution is guarded.** Steps may invoke any tool except `skill` itself (no
recursion), and a run is capped at 50 steps. A failing step stops the run and
reports where it stopped.

**Local session state.** Like `learn` and `schedule`, skills live in
`<project_dir>/.opendesk/skills/` and are not routed to remote peers.

## Storage

```
<project_dir>/.opendesk/skills/<name>.json
```
