import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import type { AddressInfo } from "node:net";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

import {
  callMcpTool, createMcpHttpHandler, createMcpServer, isLoopback, listMcpTools,
  PEER_FIELD_DESCRIPTION, resolveMcpToken, runMcpHttp, SERVER_INSTRUCTIONS,
} from "../src/mcp.js";
import { createRegistry, ToolRegistry } from "../src/registry.js";
import { displayTitle, Tool, toolAnnotations, type ToolContext, type ToolResult } from "../src/tools/base.js";

class EchoTool extends Tool {
  name = "echo_tool";
  description = "Echo text.";
  schema = { type: "object", properties: { text: { type: "string" } } };
  async execute(_ctx: ToolContext, params: Record<string, unknown>): Promise<ToolResult> {
    return this.ok("Echo", String(params["text"] ?? ""));
  }
}

class FailingTool extends Tool {
  name = "failing";
  title = "Always Fails";
  annotations = { readOnlyHint: true };
  description = "Returns an error result.";
  schema = { type: "object", properties: {} };
  async execute(): Promise<ToolResult> {
    return this.err("UI error", "macOS Accessibility permission is required.");
  }
}

class ThrowingTool extends Tool {
  name = "throwing";
  title = "Throws";
  description = "Throws.";
  schema = { type: "object", properties: {} };
  async execute(): Promise<ToolResult> {
    throw new Error("boom");
  }
}

function fakeRegistry(): ToolRegistry {
  return new ToolRegistry().register(new EchoTool()).register(new FailingTool()).register(new ThrowingTool());
}

let home: string;
beforeEach(() => { home = fs.mkdtempSync(path.join(os.tmpdir(), "opendesk-mcp-")); });
afterEach(() => { fs.rmSync(home, { recursive: true, force: true }); });

// ---------------------------------------------------------------------------

describe("tool metadata", () => {
  it("every tool has a title and annotations", () => {
    const tools = listMcpTools(createRegistry());
    expect(tools.length).toBeGreaterThanOrEqual(17);
    for (const t of tools) {
      expect(t.title, t.name).toBeTruthy();
      expect(t.title).not.toBe(t.name);
      expect(typeof t.annotations.readOnlyHint, t.name).toBe("boolean");
      if (!t.annotations.readOnlyHint) expect(typeof t.annotations.destructiveHint, t.name).toBe("boolean");
    }
  });

  it("uses the expected hints", () => {
    const by = Object.fromEntries(listMcpTools(createRegistry()).map((t) => [t.name, t.annotations]));
    for (const ro of ["screenshot", "ocr", "memory", "opendesk_peers", "opendesk_status", "opendesk_describe"]) {
      expect(by[ro].readOnlyHint, ro).toBe(true);
    }
    for (const act of ["mouse", "keyboard", "app", "ui", "audit", "memory_manage"]) {
      expect(by[act].readOnlyHint, act).toBe(false);
      expect(by[act].destructiveHint, act).toBe(true);
    }
    expect(by["clipboard"].destructiveHint).toBe(false);
    for (const idem of ["opendesk_use", "opendesk_disconnect"]) {
      expect(by[idem]).toMatchObject({ readOnlyHint: false, destructiveHint: false, idempotentHint: true });
    }
  });

  it("lists all seven admin tools", () => {
    const names = new Set(listMcpTools(createRegistry()).map((t) => t.name));
    for (const n of ["opendesk_peers", "opendesk_discover", "opendesk_use", "opendesk_status",
      "opendesk_describe", "opendesk_capabilities", "opendesk_disconnect"]) {
      expect(names.has(n), n).toBe(true);
    }
  });

  it("describes `peer` once in the schema instead of in every description", () => {
    const tools = listMcpTools(createRegistry());
    for (const t of tools) {
      expect(t.description).not.toContain("to target a remote machine instead");
    }
    const props = (name: string) =>
      (tools.find((t) => t.name === name)!.inputSchema["properties"] as Record<string, { description?: string }>);
    expect(props("screenshot")["peer"].description).toBe(PEER_FIELD_DESCRIPTION);
    expect(props("memory")["peer"]).toBeUndefined();
    expect(props("memory_manage")["peer"]).toBeUndefined();
  });

  it("derives defaults for tools without explicit metadata", () => {
    const t = new EchoTool();
    expect(displayTitle(t)).toBe("Echo Tool");
    expect(toolAnnotations(t)).toEqual({ readOnlyHint: false, destructiveHint: true });
    expect(toolAnnotations(new FailingTool())).toEqual({ readOnlyHint: true });
  });

  it("instructions cover the workflow", () => {
    for (const needle of ["`ui`", "marks=true", "`mouse`", "`screenshot`", "opendesk_peers", "opendesk_use",
      "`peer`", "`memory`", "approval", "kill switch", "opendesk check --open"]) {
      expect(SERVER_INSTRUCTIONS, needle).toContain(needle);
    }
  });
});

// ---------------------------------------------------------------------------

describe("isError", () => {
  it("is false on success", async () => {
    const r = await callMcpTool("echo_tool", { text: "hi" }, { registry: fakeRegistry() });
    expect(r.isError).toBe(false);
    expect(r.content[0].text).toBe("hi");
  });

  it("is set for unknown tools, error results, exceptions and admin errors", async () => {
    const reg = fakeRegistry();
    expect((await callMcpTool("nope", {}, { registry: reg })).isError).toBe(true);
    const failed = await callMcpTool("failing", {}, { registry: reg });
    expect(failed.isError).toBe(true);
    expect(failed.content[0].text).toContain("opendesk check --open");
    expect(failed.content[0].text).toContain("Claude Desktop");
    const thrown = await callMcpTool("throwing", {}, { registry: reg });
    expect(thrown.isError).toBe(true);
    expect(thrown.content[0].text).toContain("boom");
    const admin = await callMcpTool("opendesk_describe", { peer: "ghost", home }, { registry: reg });
    expect(admin.isError).toBe(true);
    const missing = await callMcpTool("opendesk_use", {}, { registry: reg });
    expect(missing.isError).toBe(true);
    expect((await callMcpTool("opendesk_status", {}, { registry: reg })).isError).toBe(false);
  });
});

// ---------------------------------------------------------------------------

describe("MCP server wiring (in-memory client)", () => {
  it("exposes instructions, titles, annotations and isError", async () => {
    const server = createMcpServer(fakeRegistry());
    const client = new Client({ name: "vitest", version: "0" });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    try {
      expect(client.getInstructions()).toBe(SERVER_INSTRUCTIONS);
      expect(client.getServerVersion()?.version).toBe("0.3.0");
      const { tools } = await client.listTools();
      const failing = tools.find((t) => t.name === "failing")!;
      expect(failing.title).toBe("Always Fails");
      expect(failing.annotations?.readOnlyHint).toBe(true);
      const ok = await client.callTool({ name: "echo_tool", arguments: { text: "x" } });
      expect(ok.isError).toBe(false);
      const bad = await client.callTool({ name: "failing", arguments: {} });
      expect(bad.isError).toBe(true);
    } finally {
      await client.close();
      await server.close();
    }
  });
});

// ---------------------------------------------------------------------------

const INIT = {
  jsonrpc: "2.0", id: 1, method: "initialize",
  params: { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "vitest", version: "0" } },
};
const HEADERS = { "content-type": "application/json", accept: "application/json, text/event-stream" };

describe("streamable HTTP transport", () => {
  let server: http.Server;
  let base: string;
  let handler: ReturnType<typeof createMcpHttpHandler>;

  beforeEach(async () => {
    handler = createMcpHttpHandler({ token: "s3cret", loopbackOnly: true, registry: fakeRegistry() });
    server = http.createServer(handler);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", () => r()));
    base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });
  afterEach(async () => {
    await handler.close();
    await new Promise<void>((r) => server.close(() => r()));
  });

  it("rejects a missing or wrong bearer token", async () => {
    let r = await fetch(`${base}/mcp`, { method: "POST", headers: HEADERS, body: JSON.stringify(INIT) });
    expect(r.status).toBe(401);
    expect(r.headers.get("www-authenticate")).toMatch(/^Bearer/);
    r = await fetch(`${base}/mcp`, { method: "POST", headers: { ...HEADERS, authorization: "Bearer nope" }, body: JSON.stringify(INIT) });
    expect(r.status).toBe(401);
    r = await fetch(`${base}/mcp`, { method: "POST", headers: { ...HEADERS, authorization: "Basic s3cret" }, body: JSON.stringify(INIT) });
    expect(r.status).toBe(401);
  });

  it("accepts the right token and initializes a session", async () => {
    const r = await fetch(`${base}/mcp`, {
      method: "POST", headers: { ...HEADERS, authorization: "Bearer s3cret" }, body: JSON.stringify(INIT),
    });
    expect(r.status).toBe(200);
    expect(r.headers.get("mcp-session-id")).toBeTruthy();
    const text = await r.text();
    expect(text).toContain("instructions");
  });

  it("rejects a non-loopback Host header (DNS rebinding)", async () => {
    const port = (server.address() as AddressInfo).port;
    const status = await new Promise<number>((resolve, reject) => {
      const req = http.request({
        host: "127.0.0.1", port, path: "/mcp", method: "POST",
        headers: { ...HEADERS, authorization: "Bearer s3cret", host: "evil.example" },
      }, (res) => { res.resume(); resolve(res.statusCode ?? 0); });
      req.on("error", reject);
      req.end(JSON.stringify(INIT));
    });
    expect(status).toBe(421);
  });

  it("404s other paths", async () => {
    const r = await fetch(`${base}/other`, { headers: { authorization: "Bearer s3cret" } });
    expect(r.status).toBe(404);
  });
});

describe("token handling", () => {
  it("generates a 0600 token file once and reuses it", () => {
    const saved = process.env["OPENDESK_MCP_TOKEN"];
    delete process.env["OPENDESK_MCP_TOKEN"];
    try {
      const a = resolveMcpToken(undefined, home);
      const b = resolveMcpToken(undefined, home);
      expect(a.token).toBe(b.token);
      expect(a.token.length).toBeGreaterThanOrEqual(32);
      expect(a.source).toContain("generated");
      expect(b.source).toContain("read from");
      if (process.platform !== "win32") {
        expect(fs.statSync(path.join(home, "mcp-token")).mode & 0o777).toBe(0o600);
      }
      expect(resolveMcpToken("explicit", home)).toEqual({ token: "explicit", source: "--token" });
      process.env["OPENDESK_MCP_TOKEN"] = "from-env";
      expect(resolveMcpToken(undefined, home).token).toBe("from-env");
    } finally {
      if (saved === undefined) delete process.env["OPENDESK_MCP_TOKEN"];
      else process.env["OPENDESK_MCP_TOKEN"] = saved;
    }
  });

  it("refuses non-loopback binds without auth", async () => {
    expect(isLoopback("127.0.0.1")).toBe(true);
    expect(isLoopback("::1")).toBe(true);
    expect(isLoopback("localhost")).toBe(true);
    expect(isLoopback("0.0.0.0")).toBe(false);
    expect(isLoopback("192.168.1.5")).toBe(false);
    await expect(runMcpHttp({ host: "0.0.0.0", port: 0, token: null, home })).rejects.toThrow(/non-loopback/);
  });
});
