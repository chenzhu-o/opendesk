/**
 * Standalone MCP server — exposes all native opendesk tools over the
 * Model Context Protocol.  No Python required.
 *
 * Supports both local tools and remote-peer tools:
 *
 *   - Local tools (screenshot, mouse, etc.) target the machine running this process.
 *   - Remote tools: pass `peer` argument to any tool to target a paired remote machine.
 *   - Session admin tools: opendesk_peers, opendesk_discover, opendesk_use,
 *     opendesk_status, opendesk_describe, opendesk_capabilities, opendesk_disconnect.
 *
 * Every tool is listed with a `title` and MCP `annotations`, failures come back
 * with `isError: true`, and the server sends `instructions` on initialize.
 *
 * Usage (stdio, for Claude Code / Claude Desktop / Cursor):
 *   npx -y @vitalops/opendesk-sdk          # or: opendesk-js-mcp
 *
 * Streamable HTTP (bearer-token auth, loopback by default):
 *   opendesk-js mcp --http --port 8765
 *
 * Or programmatically:
 *   import { createMcpServer, runMcpStdio, runMcpHttp } from "@vitalops/opendesk-sdk";
 */

import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { ListToolsRequestSchema, CallToolRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { createRegistry, ToolRegistry } from "./registry.js";
import {
  allowAllContext, displayTitle, permissionHint, toolAnnotations,
  type ToolAnnotations, type ToolContext,
} from "./tools/base.js";
import { TrustedPeers } from "./protocol/auth/storage.js";
import { DEFAULT_HOME } from "./protocol/auth/identity.js";
import type { RemoteComputer } from "./computer/remote.js";
import { guardedCall, runTool } from "./safety/guard.js";
import { makeMcpElicitationApprover, type Approver } from "./safety/approval.js";
import type { ToolResult } from "./tools/base.js";

export const SERVER_VERSION = "0.3.0";

/** Sent to clients in the MCP `initialize` result. */
export const SERVER_INSTRUCTIONS = `OpenDesk lets you see and control a real desktop (macOS, Windows, Linux): this machine, or paired remote machines.

Recommended workflow, cheapest and most reliable first:
1. If the task needs no visual interaction (files, shell, an app's API or CLI), do it with scripts/APIs instead of driving the GUI.
2. \`ui\` — the accessibility API. Inspect with action='get_tree', then act on elements by name/ref (click, type, click_menu). No coordinates needed.
3. \`screenshot\` with marks=true — numbered boxes over interactive elements when the accessibility tree is not enough.
4. \`mouse\` / \`keyboard\` — raw input as a last resort; pass image_width/image_height from the latest screenshot so coordinates scale correctly.
Verify every state-changing step with a fresh \`screenshot\` before moving on.

Remote machines: \`opendesk_peers\` lists paired peers, \`opendesk_use\` sets the default target, or pass \`peer\` on a single call. \`opendesk_describe\` explains what a peer is for.
Screen history: \`memory\` searches what was on screen earlier (read-only); \`memory_manage\` pauses capture or deletes history.

Safety: some actions may need the user's approval; if a call is denied, do not retry it, ask the user. The user can halt all actions with the kill switch at any time; if a call reports it is engaged, stop.
If a tool reports missing macOS permissions, ask the user to run \`opendesk check --open\` and grant Accessibility and Screen Recording to the app running OpenDesk (Claude Desktop, their terminal for Claude Code, Cursor, VS Code), then restart that app.`;

/** Description of the optional `peer` argument added to remotable tools. */
export const PEER_FIELD_DESCRIPTION =
  "Machine to run on: 'local' or a name from `opendesk_peers`. Omit to use the session default (`opendesk_use`).";

// ---------------------------------------------------------------------------
// Per-MCP-session peer tracking
// ---------------------------------------------------------------------------

const LOCAL = "local";

/** Tools that always run on this machine, never routed to a peer.  Screen
 *  memory is deliberately local-only: the history never leaves the machine
 *  it was recorded on.  The browser tool drives a browser on this machine
 *  (local-only in v1). */
const LOCAL_ONLY_TOOLS = new Set(["memory", "memory_manage", "browser"]);

interface SessionState {
  defaultPeer: string; // "local" or peer name
  remotes: Map<string, RemoteComputer>;
}

function makeSession(): SessionState {
  return { defaultPeer: LOCAL, remotes: new Map() };
}

// One state object shared across the whole server process (single-user server)
const globalState: SessionState = makeSession();

async function getRemote(peerName: string, home?: string): Promise<RemoteComputer> {
  if (globalState.remotes.has(peerName)) return globalState.remotes.get(peerName)!;
  const { connect } = await import("./remote/client.js");
  const remote = await connect(peerName, { home });
  globalState.remotes.set(peerName, remote);
  return remote;
}

// ---------------------------------------------------------------------------
// MCP shapes
// ---------------------------------------------------------------------------

export interface McpToolDefinition {
  name: string;
  title: string;
  description: string;
  inputSchema: Record<string, unknown>;
  annotations: ToolAnnotations & { title?: string };
}

export interface McpContent {
  type: string;
  text?: string;
  data?: string;
  mimeType?: string;
}

export interface McpCallResult {
  content: McpContent[];
  isError: boolean;
  [key: string]: unknown;
}

function errorResult(message: string): McpCallResult {
  return { content: [{ type: "text", text: `ERROR: ${message}` }], isError: true };
}

// ---------------------------------------------------------------------------
// Admin tools (opendesk_*)
// ---------------------------------------------------------------------------

const ADMIN_READ_ONLY: ToolAnnotations = { readOnlyHint: true, openWorldHint: false };
const ADMIN_SESSION_STATE: ToolAnnotations = {
  readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false,
};

const ADMIN_TOOLS: McpToolDefinition[] = [
  {
    name: "opendesk_peers",
    title: "List Peers",
    description: "List trusted peers this controller has paired with. The default peer is marked [default].",
    inputSchema: {
      type: "object",
      properties: { home: { type: "string", description: "Override ~/.opendesk directory" } },
    },
    annotations: ADMIN_READ_ONLY,
  },
  {
    name: "opendesk_discover",
    title: "Discover LAN Peers",
    description:
      "Discover opendesk peers advertising on the local network via mDNS. Only paired peers can be used; " +
      "pair new ones from the CLI (`opendesk-js pair-with <host> <code>`).",
    inputSchema: {
      type: "object",
      properties: {
        timeout: { type: "number", description: "Browse duration in ms (default 2000)" },
      },
    },
    annotations: { readOnlyHint: true, openWorldHint: true },
  },
  {
    name: "opendesk_use",
    title: "Set Default Peer",
    description: "Set the default peer for subsequent computer-use calls. Pass 'local' to control this machine again.",
    inputSchema: {
      type: "object",
      required: ["peer"],
      properties: {
        peer: { type: "string", description: "Peer name or 'local'" },
        home: { type: "string", description: "Override ~/.opendesk directory" },
      },
    },
    annotations: ADMIN_SESSION_STATE,
  },
  {
    name: "opendesk_status",
    title: "Session Status",
    description: "Show the current default peer and the open remote connections for this session.",
    inputSchema: { type: "object", properties: {} },
    annotations: ADMIN_READ_ONLY,
  },
  {
    name: "opendesk_describe",
    title: "Describe Peer",
    description:
      "Return the full description of a paired peer — what that machine is for and what apps / data it has. " +
      "Use this to decide which peer is right for a task.",
    inputSchema: {
      type: "object",
      required: ["peer"],
      properties: {
        peer: { type: "string", description: "Peer name (from `opendesk_peers`)." },
        home: { type: "string", description: "Override ~/.opendesk directory" },
      },
    },
    annotations: ADMIN_READ_ONLY,
  },
  {
    name: "opendesk_capabilities",
    title: "Peer Capabilities",
    description: "Show what a peer's backend can do (input, screen capture, …). Omit `peer` for the current default.",
    inputSchema: {
      type: "object",
      properties: {
        peer: { type: "string", description: "Peer name or 'local'." },
        home: { type: "string", description: "Override ~/.opendesk directory" },
      },
    },
    annotations: { readOnlyHint: true, openWorldHint: true },
  },
  {
    name: "opendesk_disconnect",
    title: "Disconnect Peer",
    description: "Close the cached connection to a remote peer (the peer stays paired).",
    inputSchema: {
      type: "object",
      properties: {
        peer: { type: "string", description: "Peer to disconnect; defaults to the current default peer." },
        home: { type: "string", description: "Override ~/.opendesk directory" },
      },
    },
    annotations: ADMIN_SESSION_STATE,
  },
];

class AdminError extends Error {}

async function callAdminTool(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  switch (name) {
    case "opendesk_peers": {
      const store = new TrustedPeers(args["home"] as string | undefined);
      const peers = store.list();
      if (!peers.length) return "No trusted peers. Pair one from the CLI: `opendesk-js pair-with <host> <code>`.";
      const def = store.getDefault();
      return peers
        .map((p) => {
          const marker = p.name === def ? " [default]" : "";
          const ep = p.lastHost ? `${p.lastHost}:${p.lastPort}` : "(unknown)";
          const desc = p.descriptionOverride || p.description;
          return `${p.name}  fp=${p.publicKey.slice(0, 8)}…  ${ep}${desc ? "  " + desc.slice(0, 60) : ""}${marker}`;
        })
        .join("\n");
    }

    case "opendesk_discover": {
      const { discover } = await import("./remote/discovery.js");
      const timeout = (args["timeout"] as number) ?? 2000;
      const found = await discover(timeout);
      if (!found.length) return "No opendesk peers found on the LAN.";
      return found
        .map((p) => `${p.name}  ${p.host}:${p.port}  fp=${p.fingerprint}  ${p.description}`)
        .join("\n");
    }

    case "opendesk_use": {
      const peer = args["peer"];
      if (typeof peer !== "string" || !peer.trim()) throw new AdminError("`peer` is required (use 'local' to revert)");
      if (peer !== LOCAL) {
        // Open connection eagerly so errors surface now
        await getRemote(peer, args["home"] as string | undefined);
      }
      globalState.defaultPeer = peer;
      return `Default peer set to: ${peer}`;
    }

    case "opendesk_status": {
      const cur = globalState.defaultPeer;
      const connected = [...globalState.remotes.keys()];
      return `Default peer: ${cur}\nConnected: ${connected.length ? connected.join(", ") : "none"}`;
    }

    case "opendesk_describe": {
      const peerName = args["peer"];
      if (peerName === undefined || peerName === "" || peerName === LOCAL) {
        return "local — this machine (the controller). No remote description.";
      }
      const peer = new TrustedPeers(args["home"] as string | undefined).findByName(String(peerName));
      if (!peer) throw new AdminError(`unknown peer: '${peerName}'`);
      const parts = [`Peer: ${peer.name}`, `Fingerprint: ${peer.publicKey.slice(0, 16)}…`];
      if (peer.descriptionOverride) parts.push("Description (override):", peer.descriptionOverride);
      else if (peer.description) parts.push("Description (broadcast):", peer.description);
      else parts.push("(no description — the peer hasn't broadcast one and no override is set)");
      return parts.join("\n");
    }

    case "opendesk_capabilities": {
      const peerName = (args["peer"] as string) ?? globalState.defaultPeer;
      if (peerName === LOCAL) return "Local machine (all capabilities).";
      const remote = await getRemote(peerName, args["home"] as string | undefined);
      const caps = remote.capabilities();
      return JSON.stringify(caps, null, 2);
    }

    case "opendesk_disconnect": {
      const peerName = (args["peer"] as string) ?? globalState.defaultPeer;
      if (peerName === LOCAL) return "Cannot disconnect local machine.";
      const remote = globalState.remotes.get(peerName);
      if (!remote) return `No active connection to '${peerName}'.`;
      await remote.close();
      globalState.remotes.delete(peerName);
      if (globalState.defaultPeer === peerName) globalState.defaultPeer = LOCAL;
      return `Disconnected from '${peerName}'.`;
    }

    default:
      throw new AdminError(`unknown admin tool: ${name}`);
  }
}

// ---------------------------------------------------------------------------
// Library-neutral list / call (exported for tests and custom transports)
// ---------------------------------------------------------------------------

/** Every tool the server exposes, with titles, annotations and the `peer` field. */
export function listMcpTools(registry?: ToolRegistry): McpToolDefinition[] {
  const reg = registry ?? createRegistry();
  const localTools = reg.all().map((t): McpToolDefinition => {
    const title = displayTitle(t);
    const base = {
      name: t.name,
      title,
      description: t.description,
      annotations: { title, ...toolAnnotations(t) },
    };
    if (LOCAL_ONLY_TOOLS.has(t.name)) {
      return { ...base, inputSchema: t.schema as Record<string, unknown> };
    }
    return {
      ...base,
      inputSchema: {
        ...(t.schema as Record<string, unknown>),
        properties: {
          ...((t.schema as Record<string, unknown>)["properties"] as Record<string, unknown> ?? {}),
          peer: { type: "string", description: PEER_FIELD_DESCRIPTION },
        },
      },
    };
  });
  const admin = ADMIN_TOOLS.map((a) => ({ ...a, annotations: { title: a.title, ...a.annotations } }));
  return [...localTools, ...admin];
}

/** Dispatch one `tools/call`; never throws — failures come back with `isError: true`. */
export async function callMcpTool(
  name: string,
  args: Record<string, unknown> = {},
  options: { registry?: ToolRegistry; ctx?: ToolContext; home?: string; approver?: Approver | null } = {},
): Promise<McpCallResult> {
  const reg = options.registry ?? createRegistry();
  const context = options.ctx ?? allowAllContext();
  const { home, approver } = options;

  // Admin tools (also pass through the safety guard: kill switch, policy, audit)
  if (ADMIN_TOOLS.some((a) => a.name === name)) {
    const res = await guardedCall(name, args, context, async (p) => {
      try {
        return { title: name, output: await callAdminTool(name, p), error: false, attachments: [], metadata: {} };
      } catch (e) {
        return { title: name, output: e instanceof Error ? e.message : String(e), error: true, attachments: [], metadata: {} };
      }
    }, { home, peer: LOCAL, approver });
    if (res.error) return errorResult(res.output);
    return { content: [{ type: "text", text: res.output }], isError: false };
  }

  let tool;
  try {
    tool = reg.get(name);
  } catch {
    return errorResult(`unknown tool '${name}'`);
  }

  // Determine peer
  const peerArg = args["peer"] as string | undefined;
  const effectivePeer = LOCAL_ONLY_TOOLS.has(name) ? LOCAL : (peerArg || globalState.defaultPeer);
  const { peer: _p, ...toolArgs } = args;
  void _p; // consumed

  let result: { output?: string; error?: boolean; attachments?: Array<{ mediaType: string; content: Buffer; filename: string }> };
  try {
    if (effectivePeer === LOCAL || effectivePeer === undefined) {
      result = await runTool(tool, context, toolArgs, { home, peer: LOCAL, approver });
    } else {
      // Remote call: forward as a tool method call over the wire (guarded here too)
      const peer = effectivePeer;
      result = await guardedCall(name, toolArgs, context, async (p) => {
        const remote = await getRemote(peer, home);
        const wireResult = await remote.call(`tool.${name}`, p);
        // Reconstruct a ToolResult-shaped object for the MCP response
        return {
          title: name,
          output: String((wireResult?.["output"] as string) ?? JSON.stringify(wireResult)),
          error: Boolean(wireResult?.["error"]),
          attachments: ((wireResult?.["attachments"] as unknown[]) ?? []) as ToolResult["attachments"],
          metadata: {},
        };
      }, { home, peer, approver });
    }
  } catch (e) {
    const msg = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
    const hint = permissionHint(msg);
    return errorResult(hint ? `${msg}\n\n${hint}` : msg);
  }

  const isError = Boolean(result.error);
  const contents: McpContent[] = [];
  const prefix = effectivePeer && effectivePeer !== LOCAL ? `[on ${effectivePeer}] ` : "";
  let output = result.output ?? "";
  if (isError && output) {
    const hint = permissionHint(output);
    if (hint) output = `${output}\n\n${hint}`;
  }
  if (output) contents.push({ type: "text", text: prefix + output });
  for (const att of result.attachments ?? []) {
    if (att.mediaType.startsWith("image/")) {
      contents.push({ type: "image", data: att.content.toString("base64"), mimeType: att.mediaType });
    } else {
      contents.push({ type: "text", text: `[${att.filename}] data:${att.mediaType};base64,${att.content.toString("base64")}` });
    }
  }
  if (!contents.length) contents.push({ type: "text", text: "(no output)" });
  return { content: contents, isError };
}

// ---------------------------------------------------------------------------
// Main factory
// ---------------------------------------------------------------------------

export function createMcpServer(registry?: ToolRegistry, ctx?: ToolContext, home?: string): Server {
  const reg = registry ?? createRegistry();
  const context = ctx ?? allowAllContext();

  const server = new Server(
    {
      name: "opendesk",
      title: "OpenDesk",
      version: SERVER_VERSION,
      websiteUrl: "https://github.com/vitalops/opendesk",
    },
    { capabilities: { tools: {} }, instructions: SERVER_INSTRUCTIONS },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: listMcpTools(reg) }));

  // Safety: ask the MCP client for approval (elicitation) before falling back
  // to a native dialog / console prompt.
  const approver = makeMcpElicitationApprover(server);

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args = {} } = request.params;
    return callMcpTool(name, args as Record<string, unknown>, { registry: reg, ctx: context, home, approver });
  });

  return server;
}

export async function runMcpStdio(registry?: ToolRegistry, ctx?: ToolContext, home?: string): Promise<void> {
  const server = createMcpServer(registry, ctx, home);
  const transport = new StdioServerTransport();
  await server.connect(transport);
  const { startKillHotkey } = await import("./safety/control.js");
  void startKillHotkey(home); // no-op unless policy.json sets kill_hotkey
}

// ---------------------------------------------------------------------------
// Streamable HTTP transport + bearer auth
// ---------------------------------------------------------------------------

export const MCP_HTTP_PATH = "/mcp";
export const MCP_TOKEN_FILENAME = "mcp-token";

export function isLoopback(host: string): boolean {
  const h = host.replace(/^\[|\]$/g, "");
  if (h === "localhost" || h === "localhost.localdomain") return true;
  if (net.isIPv4(h)) return h.startsWith("127.");
  if (net.isIPv6(h)) return h === "::1" || h === "0:0:0:0:0:0:0:1";
  return false;
}

/** Bearer token: explicit → $OPENDESK_MCP_TOKEN → `<home>/mcp-token` (created 0600 on first use). */
export function resolveMcpToken(token?: string, home?: string): { token: string; source: string } {
  if (token) return { token, source: "--token" };
  const env = (process.env["OPENDESK_MCP_TOKEN"] ?? "").trim();
  if (env) return { token: env, source: "$OPENDESK_MCP_TOKEN" };
  const file = path.join(home ?? DEFAULT_HOME, MCP_TOKEN_FILENAME);
  if (fs.existsSync(file)) {
    const stored = fs.readFileSync(file, "utf8").trim();
    if (stored) return { token: stored, source: `read from ${file}` };
  }
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const fresh = crypto.randomBytes(32).toString("base64url");
  fs.writeFileSync(file, fresh + "\n", { mode: 0o600 });
  try { fs.chmodSync(file, 0o600); } catch { /* e.g. Windows */ }
  return { token: fresh, source: `generated and saved to ${file}` };
}

export interface McpHttpOptions {
  host?: string;
  port?: number;
  /** Bearer token; `null` disables auth (loopback only). Default: resolveMcpToken(). */
  token?: string | null;
  home?: string;
  registry?: ToolRegistry;
  ctx?: ToolContext;
}

function bearerOk(req: http.IncomingMessage, token: string): boolean {
  const header = req.headers["authorization"];
  if (typeof header !== "string") return false;
  const [scheme, presented = ""] = header.split(/\s+/, 2);
  if (scheme.toLowerCase() !== "bearer") return false;
  const a = Buffer.from(presented);
  const b = Buffer.from(token);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

function hostAllowed(req: http.IncomingMessage): boolean {
  const host = req.headers["host"];
  if (!host) return false;
  const name = host.startsWith("[") ? host.slice(0, host.indexOf("]") + 1) : host.split(":")[0];
  return isLoopback(name);
}

function originAllowed(req: http.IncomingMessage): boolean {
  const origin = req.headers["origin"];
  if (!origin) return true;
  try {
    return isLoopback(new URL(origin).hostname);
  } catch {
    return false;
  }
}

function sendJson(res: http.ServerResponse, status: number, message: string, headers: Record<string, string> = {}): void {
  const body = JSON.stringify({ jsonrpc: "2.0", id: null, error: { code: -32001, message } });
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(body), ...headers });
  res.end(body);
}

/**
 * Node `http` request listener serving the MCP streamable HTTP endpoint at
 * `/mcp`.  Stateful: one transport + server per MCP session.
 */
export function createMcpHttpHandler(
  opts: { token: string | null; loopbackOnly: boolean; registry?: ToolRegistry; ctx?: ToolContext; home?: string },
): http.RequestListener & { close(): Promise<void> } {
  const transports = new Map<string, StreamableHTTPServerTransport>();

  const listener = async (req: http.IncomingMessage, res: http.ServerResponse): Promise<void> => {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (url.pathname !== MCP_HTTP_PATH && url.pathname !== `${MCP_HTTP_PATH}/`) {
      sendJson(res, 404, "Not found — the MCP endpoint is /mcp");
      return;
    }
    if (opts.loopbackOnly) {
      // DNS-rebinding protection for loopback binds.
      if (!hostAllowed(req)) { sendJson(res, 421, "Invalid Host header"); return; }
      if (!originAllowed(req)) { sendJson(res, 403, "Invalid Origin header"); return; }
    }
    if (opts.token && !bearerOk(req, opts.token)) {
      sendJson(res, 401, "Unauthorized: missing or invalid bearer token", { "www-authenticate": 'Bearer realm="opendesk"' });
      return;
    }

    try {
      const sid = req.headers["mcp-session-id"];
      const existing = typeof sid === "string" ? transports.get(sid) : undefined;
      if (existing) {
        await existing.handleRequest(req, res);
        return;
      }
      if (typeof sid === "string") {
        sendJson(res, 404, "Unknown MCP session");
        return;
      }
      if (req.method !== "POST") {
        sendJson(res, 400, "Bad request: no MCP session — send initialize first");
        return;
      }
      const transport: StreamableHTTPServerTransport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => crypto.randomUUID(),
        onsessioninitialized: (id) => { transports.set(id, transport); },
      });
      transport.onclose = () => {
        if (transport.sessionId) transports.delete(transport.sessionId);
      };
      const server = createMcpServer(opts.registry, opts.ctx, opts.home);
      await server.connect(transport);
      await transport.handleRequest(req, res);
    } catch (e) {
      if (!res.headersSent) sendJson(res, 500, e instanceof Error ? e.message : String(e));
    }
  };

  return Object.assign((req: http.IncomingMessage, res: http.ServerResponse) => { void listener(req, res); }, {
    close: async (): Promise<void> => {
      const all = [...transports.values()];
      transports.clear();
      await Promise.allSettled(all.map((t) => t.close()));
    },
  });
}

/** Serve the MCP server over streamable HTTP. Resolves once listening. */
export async function runMcpHttp(options: McpHttpOptions = {}): Promise<http.Server> {
  const host = options.host ?? "127.0.0.1";
  const port = options.port ?? 8765;
  const loopback = isLoopback(host);

  let token: string | null;
  let source: string;
  if (options.token === null) {
    if (!loopback) {
      throw new Error(
        `refusing to serve on non-loopback address '${host}' without a token; ` +
        "drop --no-auth (a token is generated for you) or bind to 127.0.0.1",
      );
    }
    token = null;
    source = "disabled (--no-auth)";
  } else {
    ({ token, source } = resolveMcpToken(options.token, options.home));
  }

  const handler = createMcpHttpHandler({
    token, loopbackOnly: loopback, registry: options.registry, ctx: options.ctx, home: options.home,
  });
  const server = http.createServer(handler);
  server.on("close", () => { void handler.close(); });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => resolve());
  });

  const urlHost = host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;
  const addr = server.address();
  const boundPort = typeof addr === "object" && addr ? addr.port : port;
  process.stderr.write(`OpenDesk MCP (streamable HTTP) on http://${urlHost}:${boundPort}${MCP_HTTP_PATH}\n`);
  process.stderr.write(token ? `  Auth: Authorization: Bearer ${token}\n  Token ${source}\n` : `  Auth: ${source}\n`);
  if (!loopback) {
    process.stderr.write("  WARNING: listening beyond loopback; anyone with the token controls this desktop.\n");
  }
  return server;
}
