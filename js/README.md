# @vitalops/opendesk-sdk — JavaScript/TypeScript SDK and MCP server

**OpenDesk by VitalOps — open-source computer-use MCP server that lets AI agents see and control macOS, Windows and Linux desktops, locally or across paired machines.**

Not affiliated with the openDesk office suite. No Python required. All desktop automation runs natively in Node.js — screenshot capture, mouse/keyboard control, accessibility APIs, OCR, clipboard, and audit logging.

**Requirements:** Node.js 18+

---

## MCP installation

OpenDesk works with any MCP-compatible client — Claude Code, Claude Desktop, Cursor, VS Code, Windsurf, Continue, or any custom tool.

### Claude Code (one command)

```bash
claude mcp add opendesk -- npx -y -p @vitalops/opendesk-sdk opendesk-js-mcp
```

Or, with the package installed: `npx opendesk-js install` (remove with `npx opendesk-js uninstall`).

### Claude Desktop / Cursor / Windsurf / Continue

Add to your MCP config file:

```json
{
  "mcpServers": {
    "opendesk": { "command": "npx", "args": ["-y", "-p", "@vitalops/opendesk-sdk", "opendesk-js-mcp"] }
  }
}
```

`npx -y @vitalops/opendesk-sdk` also starts the MCP server (the package's default `opendesk-sdk` bin).

### VS Code (`.vscode/mcp.json`)

```json
{
  "servers": {
    "opendesk": { "type": "stdio", "command": "npx", "args": ["-y", "-p", "@vitalops/opendesk-sdk", "opendesk-js-mcp"] }
  }
}
```

### Streamable HTTP

```bash
npx -p @vitalops/opendesk-sdk opendesk-js mcp --http --host 127.0.0.1 --port 8765
```

Clients connect to `http://127.0.0.1:8765/mcp` with `Authorization: Bearer <token>`. The token comes from `--token`, `$OPENDESK_MCP_TOKEN`, or is generated once into `~/.opendesk/mcp-token` (mode 0600) and printed on start. `--no-auth` is only allowed on loopback.

### macOS permissions

Grant **Accessibility** and **Screen Recording** to the app that runs OpenDesk (Claude Desktop, your terminal for Claude Code, Cursor or VS Code) under System Settings → Privacy & Security, then restart that app. With the Python package, `uvx opendesk check --open` opens the right pane. [Details](https://vitalops.ai/opendesk/docs/getting-started/permissions.html)

Config file locations:
- **Claude Desktop (macOS):** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Cursor:** `.cursor/mcp.json` in your project or `~/.cursor/mcp.json` globally
- **Windsurf:** `~/.codeium/windsurf/mcp_config.json`
- **Continue:** `.continue/config.json`

---

## SDK install

```bash
npm install @vitalops/opendesk-sdk
```

---

## Usage

### Programmatic (agent loop)

```typescript
import { OpenDeskClient } from "@vitalops/opendesk-sdk";

const client = new OpenDeskClient();

// Take a screenshot with Set-of-Marks
const shot = await client.screenshot({ marks: true });
console.log(shot.output);

// Click a button by name — no coordinates needed
await client.ui({ action: "click", app: "Safari", title: "Go" });

// Type text
await client.keyboard({ action: "type", text: "Hello from JS" });

// Open an app
await client.app({ action: "open", name: "Spotify" });

// Read clipboard
const clip = await client.clipboard({ action: "read" });
console.log(clip.output);

// OCR a region
const text = await client.ocr({ region: [0, 0, 800, 400] });
```

### With Vercel AI SDK

```typescript
import { OpenDeskClient } from "@vitalops/opendesk-sdk";
import { generateText } from "ai";
import { anthropic } from "@ai-sdk/anthropic";

const client = new OpenDeskClient();

const shot = await client.screenshot({ marks: true });
const response = await generateText({
  model: anthropic("claude-opus-4-6"),
  messages: [
    {
      role: "user",
      content: [
        { type: "text", text: "What do you see on screen? Click the most prominent button." },
        { type: "image", image: shot.attachments[0].content },
      ],
    },
  ],
});
```

### Expose as MCP server

```typescript
import { createMcpServer } from "@vitalops/opendesk-sdk";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = createMcpServer();
const transport = new StdioServerTransport();
await server.connect(transport);
```

### Custom permission handler

```typescript
import { OpenDeskClient } from "@vitalops/opendesk-sdk";

const client = new OpenDeskClient({
  sessionId: "my-agent-session",
  permissionHandler: async (tool, action, description) => {
    console.log(`Allow ${description}?`);
    return true;
  },
});
```

---

## Remote machine control

Control another machine on your LAN — same tools, forwarded over an encrypted WebSocket with mutual X25519 authentication and mDNS peer discovery.

```bash
# On the machine to be controlled:
npx opendesk-js pair            # prints a 6-digit pairing code

# On the controlling machine:
npx opendesk-js pair-with <host> <code> --name mini
npx opendesk-js serve           # start the long-running daemon (controlled machine)
```

See [remote control (JS)](https://vitalops.ai/opendesk/docs/remote-js/index.html) and the [protocol](https://vitalops.ai/opendesk/docs/protocol/index.html) for full details.

### MCP remote tools

Run `npx opendesk-js install` on the controller as normal. Every computer-use tool (`screenshot`, `mouse`, `keyboard`, etc.) accepts an optional `peer` argument to target a remote machine.

**Admin tools** available to the agent:

| Tool | Purpose |
|---|---|
| `opendesk_peers` | List `local` + every trusted peer |
| `opendesk_discover` | Browse the LAN for opendesk peers |
| `opendesk_use <peer>` | Set default peer for subsequent calls (`"local"` to revert) |
| `opendesk_status` | Show current default peer and open connections |
| `opendesk_describe <peer>` | What a peer is for and what it has access to |
| `opendesk_capabilities [peer]` | Capability manifest of a peer |
| `opendesk_disconnect [peer]` | Close cached connection |

---

## Tools

Full reference: [tools](https://vitalops.ai/opendesk/docs/tools/index.html)

| Tool | Method | Description |
|------|--------|-------------|
| `screenshot` | `client.screenshot(params?)` | Capture screen, optional SoM marks |
| `ui` | `client.ui(params)` | Click/type by element name — no coordinates |
| `mouse` | `client.mouse(params)` | Pixel-level mouse control |
| `keyboard` | `client.keyboard(params)` | Type, press keys, hotkeys |
| `app` | `client.app(params)` | Open, close, focus applications |
| `clipboard` | `client.clipboard(params)` | Read/write system clipboard |
| `ocr` | `client.ocr(params?)` | Extract text from screen |
| `audit` | `client.audit(params?)` | Session audit log |
| `memory` | `client.memory(params)` | Screen memory (read-only) — search a local, OCR'd history of what was on screen |
| `memory_manage` | `client.memoryManage(params)` | Pause capture, edit the deny list, change storage, delete or export history |

Over MCP every tool has a `title` and `readOnlyHint` / `destructiveHint` annotations, failures return `isError: true`, and the server sends `instructions` describing the recommended workflow.

---

## Screen memory

A searchable, fully local history of what was on your screen. Whenever you switch apps or windows (plus an idle fallback), a background daemon reads the window's text from the accessibility tree or with OCR **on your machine**, redacts secrets, and stores the text plus an encrypted thumbnail under `~/.opendesk/memory`. Nothing is uploaded.

```bash
npx opendesk-js memory start              # foreground, Ctrl-C to stop
npx opendesk-js memory install-service    # or: run at login (launchd / systemd --user / Task Scheduler)
npx opendesk-js memory status
npx opendesk-js memory export --format html   # decrypted, browsable copy (also json / csv)
```

Then ask the agent:

```
"What was the error message I saw in the terminal on Tuesday?"
"Find the invoice number I had open last week"
"Show me every time I opened that dashboard this month"
```

Or from code:

```typescript
const hits = await client.memory({ action: "search", query: "error", app: "Terminal", since: "tuesday" });
const moment = await client.memory({ action: "show", id: hits.metadata.ids[0] });
moment.attachments[0]            // { mediaType: "image/jpeg", content: Buffer }

await client.memoryManage({ action: "pause", duration: "1h" });
await client.memoryManage({ action: "deny", denyAdd: "bank" });
await client.memoryManage({ action: "config", storageCapMb: 512, retentionDays: 14 });
await client.memoryManage({ action: "export", format: "csv", since: "last week", path: "./memory-export" });
```

- **Per-app deny list** — matched against app name *and* window title; password managers are excluded by default. `opendesk-js memory deny add "bank"`.
- **Pause hotkey** — `Cmd/Ctrl+Shift+Alt+P`, powered by the optional `uiohook-napi` package (`npm install uiohook-napi`). Without it, pause via `opendesk-js memory pause 2h` or the tool.
- **Storage cap with rolling deletion** — default 2 GB / 30 days; oldest frames go first.
- **OCR** — macOS Vision (compiled once, offline) or Windows WinRT when available, otherwise the bundled tesseract.js (downloads its English model once and caches it locally).
- The store format is shared with the Python SDK's `opendesk memory` config, pause flag, and deny list. Run only one daemon per home.

Full reference → [screen memory](https://vitalops.ai/opendesk/docs/tools/memory.html)

---

## How it works

### Local tools

```
Your JS/TS code
      │
      ▼
@vitalops/opendesk-sdk (Node.js)
      │
      ├── screenshot  (screenshot-desktop)
      ├── memory  (background daemon → JSONL index + thumbnails, local OCR)
      ├── mouse/keyboard  (@nut-tree-fork/nut-js)
      ├── ui  (osascript / PowerShell UI Automation / xdotool)
      ├── ocr  (tesseract.js)
      └── clipboard  (clipboardy)
```

### Remote tools

```
Controller (your machine)                Controlled machine
      │                                        │
      ▼                                        ▼
connect("mini")           ◄──────────►  opendesk-js serve
      │               ws + ChaCha20-Poly1305    │
      ▼                                        ▼
RemoteComputer                          ToolDispatcher
  .capture()                              maps tool.* RPC → local tools
  .pointer() / .key()                     records every call to audit log
```

All platform-specific automation runs directly in Node.js. No external process is required.

---

## Docs

- [Getting started (JS)](https://vitalops.ai/opendesk/docs/getting-started/javascript.html)
- [MCP setup](https://vitalops.ai/opendesk/docs/mcp/index.html)
- [Remote control (JS)](https://vitalops.ai/opendesk/docs/remote-js/index.html)
- [Protocol](https://vitalops.ai/opendesk/docs/protocol/index.html)
- [Tools reference](https://vitalops.ai/opendesk/docs/tools/index.html)
- [Architecture](https://vitalops.ai/opendesk/docs/architecture/index.html)

---

## License

MIT
