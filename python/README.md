# OpenDesk — Python SDK and MCP server

**OpenDesk by VitalOps — open-source computer-use MCP server that lets AI agents see and control macOS, Windows and Linux desktops, locally or across paired machines.**

**macOS · Linux · Windows** · Not affiliated with the openDesk office suite.

[![PyPI](https://img.shields.io/pypi/v/opendesk)](https://pypi.org/project/opendesk/)
[![Python](https://img.shields.io/pypi/pyversions/opendesk)](https://pypi.org/project/opendesk/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/vitalops/opendesk/blob/main/LICENSE)

<!-- mcp-name: io.github.vitalops/opendesk -->

---

## Install

Run the MCP server with no install step (needs [uv](https://docs.astral.sh/uv/)):

```bash
claude mcp add opendesk -- uvx --from opendesk opendesk-mcp     # Claude Code
uvx opendesk mcp                                                # any client, stdio
```

Or install the package:

```bash
pip install opendesk
opendesk install          # registers the MCP server with Claude Code
```

`pip install opendesk` includes the MCP server and local computer use (screen capture, mouse, keyboard). For Claude Desktop, Cursor, VS Code and other clients, see [MCP integrations](#mcp-integrations).

**macOS:** run `opendesk check --open` and grant **Accessibility** and **Screen Recording** to the app that runs OpenDesk (Claude Desktop, your terminal for Claude Code, Cursor or VS Code), then restart that app.

---

## Quick start

```python
import asyncio
from opendesk import create_registry, allow_all_context

async def main():
    registry = create_registry()
    ctx = allow_all_context()

    # Screenshot with Set-of-Marks
    shot = registry.get("screenshot")
    result = await shot.execute(ctx, shot.Params(marks=True))
    print(result.output)

    # Click a button by name — no coordinates needed
    ui = registry.get("ui")
    await ui.execute(ctx, ui.Params(action="click", app="TextEdit", title="File"))

asyncio.run(main())
```

---

## Installation options

```bash
pip install opendesk                     # MCP server + local computer use
pip install 'opendesk[learn]'            # + task recording and replay
pip install 'opendesk[learn,schedule]'   # + scheduled tasks
pip install 'opendesk[remote]'           # + remote machine control
pip install 'opendesk[memory]'           # + screen memory (searchable desktop history)
pip install 'opendesk[all]'              # everything
```

The `core` and `mcp` extras from earlier versions still work; they are now part of the base install.

---

## Tools

| Tool | What it does |
|------|-------------|
| `screenshot` | Capture screen with Set-of-Marks on every interactive element |
| `ui` | Click and type by element name — no coordinates needed |
| `mouse` | Pixel-level mouse control for anything `ui` can't reach |
| `keyboard` | Type text, press keys, send hotkeys |
| `app` | Open, close, and focus applications |
| `clipboard` | Read and write the system clipboard |
| `ocr` | Extract text from any region of the screen |
| `learn` | Record a workflow once, replay it anytime |
| `schedule` | Run any task on a timer |
| `audit` | Show the session audit log in any MCP session |
| `memory` | Search the local screen memory (read-only) |
| `memory_manage` | Pause capture, edit the deny list, delete history |

Over MCP every tool has a `title` and `readOnlyHint` / `destructiveHint` annotations, failures return `isError: true`, and the server sends `instructions` describing the recommended workflow.

Full reference: [tools](https://vitalops.ai/opendesk/docs/tools/index.html)

---

## Remote machine control

OpenDesk can control remote machines over an end-to-end encrypted WebSocket connection with mDNS peer discovery.

```bash
# On the machine to be controlled:
pip install 'opendesk[remote]'
opendesk pair            # prints a pairing code
opendesk serve           # after pairing: accept paired controllers

# On the controlling machine:
pip install 'opendesk[remote]'
opendesk pair-with <host> <code>
```

See [remote control](https://vitalops.ai/opendesk/docs/remote/index.html) and the [protocol](https://vitalops.ai/opendesk/docs/protocol/index.html) for full details.

---

## MCP integrations

### Claude Code

```bash
claude mcp add opendesk -- uvx --from opendesk opendesk-mcp
# or, after pip install:
opendesk install        # register globally
opendesk uninstall      # remove
```

### Claude Desktop

Install the one-click `.mcpb` bundle from [GitHub Releases](https://github.com/vitalops/opendesk/releases), or add:

```json
{
  "mcpServers": {
    "opendesk": { "command": "uvx", "args": ["--from", "opendesk", "opendesk-mcp"] }
  }
}
```

### Cursor / Windsurf / Continue

```json
{
  "mcpServers": {
    "opendesk": { "command": "uvx", "args": ["--from", "opendesk", "opendesk-mcp"] }
  }
}
```

### VS Code (`.vscode/mcp.json`)

```json
{
  "servers": {
    "opendesk": { "type": "stdio", "command": "uvx", "args": ["--from", "opendesk", "opendesk-mcp"] }
  }
}
```

### Streamable HTTP

```bash
opendesk mcp --transport http --host 127.0.0.1 --port 8765
```

Clients connect to `http://127.0.0.1:8765/mcp` with `Authorization: Bearer <token>`. The token comes from `--token`, `$OPENDESK_MCP_TOKEN`, or is generated once into `~/.opendesk/mcp-token` (mode 0600) and printed on start. Binding beyond loopback without auth is refused.

---

## Agent integrations

### Anthropic SDK

```python
import anthropic
from opendesk.integrations.claude_code import ClaudeCodeAdapter
from opendesk.registry import create_registry

client = anthropic.Anthropic()
adapter = ClaudeCodeAdapter(create_registry())

result = await adapter.run_loop(
    client=client,
    model="claude-opus-4-6",
    messages=[{"role": "user", "content": "Open TextEdit and type Hello."}],
    system="Use the ui tool first. Mouse is a last resort.",
)
```

### OpenAI / on-device models (Ollama, vLLM, llama.cpp)

```python
from openai import OpenAI
from opendesk.integrations.openai_compat import OpenAIAdapter

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
adapter = OpenAIAdapter()
result = await adapter.run_loop(client, model="qwen2.5:72b", messages=messages)
```

### LangChain

```python
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from opendesk.integrations.langchain_compat import as_langchain_tools
from opendesk.registry import create_registry

tools = as_langchain_tools(create_registry())
agent = create_react_agent(ChatAnthropic(model="claude-opus-4-6"), tools)
```

---

## Build from source

```bash
git clone https://github.com/vitalops/opendesk
cd opendesk/python
pip install -e '.[all]'
python -m pytest -q
```

---

## Docs

- [Getting started](https://vitalops.ai/opendesk/docs/getting-started/python.html)
- [Permissions](https://vitalops.ai/opendesk/docs/getting-started/permissions.html)
- [MCP setup](https://vitalops.ai/opendesk/docs/mcp/index.html)
- [Tools reference](https://vitalops.ai/opendesk/docs/tools/index.html)
- [Remote control](https://vitalops.ai/opendesk/docs/remote/index.html)
- [Protocol](https://vitalops.ai/opendesk/docs/protocol/index.html)
- [Integrations](https://vitalops.ai/opendesk/docs/integrations/index.html)
- [Architecture](https://vitalops.ai/opendesk/docs/architecture/index.html)

## License

MIT
