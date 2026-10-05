# `system` — Hybrid CLI & Filesystem Layer

The `system` tool drives the computer through its command line and filesystem
instead of its pixels. It is the fast path for any task that has a
command-line or file equivalent.

```
system (CLI / filesystem) ──(no equivalent)──► ui ──► screenshot(marks=True) ──► mouse
```

## Why it exists

Computer-use research over the past few months converges on *hybrid
interfaces*: mixing GUI actions with native CLI / API calls. Clicking through
a UI is slow, costs a screenshot and a vision-model call per step, and breaks
when the layout changes. A single shell command is faster, cheaper, and
reproducible across app versions.

The `Computer` abstraction always exposed these primitives (`shell`, `exec`,
`read_file`, `write_file`, `list_dir`, `processes`, …) but no agent-facing
tool surfaced them — so agents could only click. `system` closes that gap and
encodes the routing rule in its description, so any model that reads the tool
list follows it without a separate orchestration layer.

## Actions

| Action | Arguments | What it does |
|---|---|---|
| `shell` | `command`, `cwd?`, `timeout?` | Run a command string through the platform shell |
| `exec` | `argv`, `cwd?`, `timeout?` | Spawn an argv vector with no shell interpolation (safer) |
| `read_file` | `path` | Read a file's UTF-8 contents |
| `write_file` | `path`, `content` | Overwrite a file (parent dirs created) |
| `list_dir` | `path` | List directory contents with type and size |
| `stat` | `path` | Size, mtime, mode, type of a path |
| `mkdir` | `path` | Create a directory (parents included) |
| `move` | `path`, `dst` | Move / rename |
| `delete` | `path` | Delete a file or directory |
| `processes` | — | List running processes |
| `environment` | — | OS, hostname, locale, timezone, displays |
| `notifications` | — | Recent desktop notifications |

`max_bytes` (default 20000) truncates returned stdout/stderr/file text so a
chatty command can't flood the agent's context.

## Examples

```python
from opendesk.registry import create_registry
from opendesk.tools.base import allow_all_context

tools = create_registry()
ctx = allow_all_context()

# Run a command
await tools.get("system").execute(ctx, tools.get("system").parse_params({
    "action": "shell",
    "command": "git status --short",
    "cwd": "~/projects/app",
}))

# Write a file, then read it back
sys = tools.get("system")
await sys.execute(ctx, sys.parse_params({
    "action": "write_file", "path": "~/notes/todo.md", "content": "# TODO\n",
}))
await sys.execute(ctx, sys.parse_params({
    "action": "read_file", "path": "~/notes/todo.md",
}))
```

## Routing guidance

Prefer `system` when:

- the task runs a command (`git`, `docker`, `pytest`, `ffmpeg`, package managers),
- the task reads, writes, moves, or lists files,
- the task queries processes, the environment, or notifications,
- an app exposes a CLI or a URL scheme that reaches the target state directly.

Fall back to `ui` (then `screenshot(marks=True)`, then `mouse`) only when the
target has no CLI surface — canvas editors, games, and some desktop apps.

## Audit & replay

Every `system` action is recorded in the session audit log with a complete
`replay_params` set, so the action can be re-executed without the original
screen or UI tree. See the `audit` tool.

## Remote peers

`system` is peer-aware: pass `peer: "<name>"` (MCP) to run the command on a
paired remote machine instead of the local one.
