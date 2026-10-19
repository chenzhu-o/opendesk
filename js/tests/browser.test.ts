import { describe, it, expect, afterEach } from "vitest";
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import net from "node:net";
import { WebSocketServer } from "ws";
import type { AddressInfo } from "node:net";
import { BrowserTool, closeBrowserSession, getBrowserSession, renderAx, PROFILE_MARKER } from "../src/tools/browser.js";
import { CDPConnection, DevToolsHTTP, findBrowser } from "../src/computer/cdp.js";
import { getSandbox } from "../src/computer/sandbox.js";
import type { ToolContext } from "../src/tools/base.js";

// ---------------------------------------------------------------------------
// Fake DevTools endpoint (HTTP discovery + ws CDP) over a toy page model
// ---------------------------------------------------------------------------

interface Page {
  id: string; url: string; title: string; value: string; out: string;
  focused?: string; mouse: Array<Record<string, unknown>>; keys: Array<Record<string, unknown>>;
  covered: boolean; domClicks: number;
}

const ELEMENTS: Record<number, [string, string, string]> = {
  10: ["heading", "Hello", "h1"],
  11: ["textbox", "Search", "input"],
  12: ["button", "Press me", "button"],
  13: ["StaticText", "idle", "p"],
};
const SELECTORS: Record<string, number> = { "#q": 11, "#b": 12, button: 12, h1: 10 };
const TEXT: Record<string, number> = { "press me": 12, hello: 10 };

class FakeCDP {
  pages = new Map<string, Page>();
  png = Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.alloc(2_500_000)]);
  lastShot: Record<string, unknown> = {};
  httpPort = 0;
  wsPort = 0;
  private counter = 1;
  private httpServer?: http.Server;
  private wss?: WebSocketServer;

  constructor() {
    this.pages.set("T1", this.page("T1", "https://example.test/", "Example"));
  }

  page(id: string, url: string, title: string): Page {
    return { id, url, title, value: "", out: "idle", mouse: [], keys: [], covered: false, domClicks: 0 };
  }

  text(p: Page): string {
    return `Hello\nPress me\n${p.out}`;
  }

  targetJson(p: Page) {
    return { id: p.id, type: "page", title: p.title, url: p.url, webSocketDebuggerUrl: `ws://127.0.0.1:${this.wsPort}/devtools/page/${p.id}` };
  }

  async start(): Promise<void> {
    this.wss = new WebSocketServer({ port: 0, host: "127.0.0.1" });
    await new Promise<void>((r) => this.wss!.once("listening", () => r()));
    this.wsPort = (this.wss.address() as AddressInfo).port;
    this.wss.on("connection", (ws, req) => {
      const page = this.pages.get(String(req.url).split("/").pop()!);
      if (!page) return ws.close();
      ws.on("message", (raw) => {
        const msg = JSON.parse(raw.toString());
        const [result, error, events] = this.handle(page, msg.method, msg.params ?? {});
        ws.send(JSON.stringify(error ? { id: msg.id, error } : { id: msg.id, result }));
        for (const ev of events) ws.send(JSON.stringify(ev));
      });
    });
    this.httpServer = http.createServer((req, res) => {
      const send = (payload: unknown, code = 200) => {
        res.writeHead(code, { "Content-Type": "application/json" });
        res.end(typeof payload === "string" ? payload : JSON.stringify(payload));
      };
      const u = String(req.url);
      if (u === "/json/version") return send({ Browser: "FakeChrome/1.0" });
      if (u === "/json/list" || u === "/json") {
        return send([...[...this.pages.values()].map((p) => this.targetJson(p)), { id: "SW", type: "service_worker" }]);
      }
      if (u.startsWith("/json/new")) {
        if (req.method !== "PUT") return send("Using unsafe HTTP verb GET", 405);
        const id = `T${++this.counter}`;
        const url = u.includes("?") ? decodeURIComponent(u.slice(u.indexOf("?") + 1)) : "about:blank";
        this.pages.set(id, this.page(id, url, "New tab"));
        return send(this.targetJson(this.pages.get(id)!));
      }
      if (u.startsWith("/json/close/")) {
        this.pages.delete(u.split("/").pop()!);
        return send("Target is closing");
      }
      return send({ error: "nf" }, 404);
    });
    await new Promise<void>((r) => this.httpServer!.listen(0, "127.0.0.1", () => r()));
    this.httpPort = (this.httpServer.address() as AddressInfo).port;
  }

  async stop(): Promise<void> {
    for (const c of this.wss?.clients ?? []) c.terminate();
    await new Promise<void>((r) => this.wss?.close(() => r()) ?? r());
    this.httpServer?.closeAllConnections?.();
    await new Promise<void>((r) => this.httpServer?.close(() => r()) ?? r());
  }

  obj(b: number) {
    return { type: "object", subtype: "node", objectId: `obj-${b}` };
  }

  handle(page: Page, method: string, p: Record<string, any>): [unknown, unknown, unknown[]] {
    const val = (value: unknown, type = typeof value) => [{ result: { type, value } }, null, []] as [unknown, unknown, unknown[]];
    const nul = [{ result: { type: "object", subtype: "null", value: null } }, null, []] as [unknown, unknown, unknown[]];
    if (["Page.enable", "Accessibility.enable"].includes(method)) return [{}, null, []];
    if (method === "Runtime.evaluate") {
      const e: string = p.expression;
      if (e === "document.readyState") return val("complete");
      if (e.startsWith("({title: document.title")) return val({ title: page.title, url: page.url, text: this.text(page) }, "object");
      if (e.startsWith("({t: document.title")) return val({ t: page.title, u: page.url }, "object");
      if (e.startsWith("!!document.querySelector(")) return val(JSON.parse(e.slice(25, -1)) in SELECTORS);
      if (e.startsWith("!!(document.body")) return val(this.text(page).includes(JSON.parse(e.split(".includes(")[1].slice(0, -2))));
      if (e.startsWith("document.querySelector(")) {
        const sel = JSON.parse(e.slice(23, -1));
        return sel in SELECTORS ? [{ result: this.obj(SELECTORS[sel]) }, null, []] : nul;
      }
      if (e.includes("(function (needle)")) {
        const needle = JSON.parse(e.slice(e.lastIndexOf("(") + 1, -1)).toLowerCase();
        return needle in TEXT ? [{ result: this.obj(TEXT[needle]) }, null, []] : nul;
      }
      if (e === "6 * 7") return val(42);
      if (e === "undefined") return [{ result: { type: "undefined" } }, null, []];
      if (e.startsWith("throw")) return [{ result: {}, exceptionDetails: { text: "Uncaught", exception: { description: "Error: boom" } } }, null, []];
      return val(`eval:${e}`);
    }
    if (method === "Accessibility.getFullAXTree") {
      return [{ nodes: [
        { nodeId: "1", role: { value: "RootWebArea" }, name: { value: page.title }, childIds: ["2", "3", "4", "5", "6"], backendDOMNodeId: 1,
          properties: [{ name: "focused", value: { value: true } }] },
        { nodeId: "2", role: { value: "heading" }, name: { value: "Hello" }, childIds: ["7"], backendDOMNodeId: 10 },
        { nodeId: "3", role: { value: "textbox" }, name: { value: "Search" }, value: { value: page.value }, backendDOMNodeId: 11 },
        { nodeId: "4", role: { value: "button" }, name: { value: "Press me" }, backendDOMNodeId: 12 },
        { nodeId: "5", role: { value: "StaticText" }, name: { value: page.out }, backendDOMNodeId: 13 },
        { nodeId: "6", ignored: true, role: { value: "generic" }, backendDOMNodeId: 14 },
        { nodeId: "7", role: { value: "StaticText" }, name: { value: "Hello" }, backendDOMNodeId: 15 },
      ] }, null, []];
    }
    if (method === "DOM.resolveNode") {
      return p.backendNodeId in ELEMENTS ? [{ object: this.obj(p.backendNodeId) }, null, []] : [null, { code: -32000, message: "No node" }, []];
    }
    if (method === "Runtime.callFunctionOn") {
      const b = Number(String(p.objectId).split("-")[1]);
      const fn: string = p.functionDeclaration;
      const [, name, tag] = ELEMENTS[b];
      if (fn.includes("elementFromPoint")) {
        return val({ ok: true, x: 100 + b, y: 50, hittable: !(b === 12 && page.covered), tag, label: name }, "object");
      }
      if (fn.includes("el.focus()")) {
        page.focused = tag;
        if (p.arguments?.[0]?.value && b === 11) page.value = "";
        return val(tag);
      }
      if (fn.includes("el.click()")) {
        page.domClicks++;
        if (b === 12) page.out = `clicked ${page.value}`;
        return val(true);
      }
      if (fn.includes('"value" in el')) return val(b === 11 ? page.value : name);
      return [{ result: { type: "undefined" } }, null, []];
    }
    if (method === "Input.dispatchMouseEvent") {
      page.mouse.push(p);
      if (p.type === "mouseReleased" && p.x === 112) page.out = `clicked ${page.value}`;
      return [{}, null, []];
    }
    if (method === "Input.insertText") {
      if (page.focused === "input") page.value += p.text;
      return [{}, null, []];
    }
    if (method === "Input.dispatchKeyEvent") {
      page.keys.push(p);
      return [{}, null, []];
    }
    if (method === "Page.getLayoutMetrics") return [{ cssContentSize: { width: 800, height: 3000 } }, null, []];
    if (method === "Page.captureScreenshot") {
      this.lastShot = p;
      return [{ data: this.png.toString("base64") }, null, []];
    }
    if (method === "Page.navigate") {
      if (String(p.url).startsWith("bad://")) return [{ frameId: "F", errorText: "net::ERR_ABORTED" }, null, []];
      page.url = p.url;
      page.title = "Navigated";
      return [{ frameId: "F", loaderId: "L1" }, null, [{ method: "Page.loadEventFired", params: {} }]];
    }
    return [null, { code: -32601, message: `'${method}' wasn't found` }, []];
  }
}

function freePort(): Promise<number> {
  return new Promise((resolve) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => {
      const port = (s.address() as AddressInfo).port;
      s.close(() => resolve(port));
    });
  });
}

const fakes: FakeCDP[] = [];
const sessionsUsed: string[] = [];

async function attached(sessionId: string) {
  const fake = new FakeCDP();
  await fake.start();
  fakes.push(fake);
  sessionsUsed.push(sessionId);
  const tool = new BrowserTool();
  const c: ToolContext = { sessionId };
  const drive = (params: Record<string, unknown>) => tool.execute(c, params);
  const r = await drive({ action: "attach", port: fake.httpPort });
  expect(r.error, r.output).toBe(false);
  return { fake, drive };
}

afterEach(async () => {
  for (const s of sessionsUsed.splice(0)) await closeBrowserSession(s);
  for (const f of fakes.splice(0)) await f.stop();
});

// ---------------------------------------------------------------------------

describe("CDP client", () => {
  it("talks HTTP discovery and WebSocket CDP", async () => {
    const fake = new FakeCDP();
    await fake.start();
    fakes.push(fake);
    const httpc = new DevToolsHTTP(fake.httpPort);
    expect((await httpc.version())["Browser"]).toBe("FakeChrome/1.0");
    const targets = await httpc.targets();
    expect(targets.map((t) => t.id)).toEqual(["T1"]);
    const conn = await new CDPConnection(targets[0].webSocketDebuggerUrl!).connect();
    expect((await conn.send("Runtime.evaluate", { expression: "6 * 7" }))["result"]).toMatchObject({ value: 42 });
    await expect(conn.send("Bogus.method")).rejects.toThrow(/wasn't found/);
    await conn.close();
    const t = await httpc.newTab("https://a.test/x?y=1#frag");
    expect(t.url).toBe("https://a.test/x?y=1#frag");
  });

  it("explains a missing endpoint", async () => {
    await expect(new DevToolsHTTP(await freePort()).version()).rejects.toThrow(/remote-debugging-port/);
  });

  it("renders the AX tree with refs", () => {
    const { lines, refs, total } = renderAx([
      { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "T" }, childIds: ["2", "3"], backendDOMNodeId: 1 },
      { nodeId: "2", role: { value: "generic" }, childIds: ["4"], backendDOMNodeId: 2 },
      { nodeId: "3", role: { value: "checkbox" }, name: { value: "Agree" }, properties: [{ name: "checked", value: { value: "true" } }], backendDOMNodeId: 3 },
      { nodeId: "4", role: { value: "link" }, name: { value: "Docs" }, backendDOMNodeId: 4 },
    ], 100);
    expect(lines[0]).toBe('[e1] RootWebArea "T"');
    expect(lines).toContain('  [e2] link "Docs"');
    expect(lines).toContain('  [e3] checkbox "Agree" (checked)');
    expect([...refs.entries()]).toEqual([["e1", 1], ["e2", 4], ["e3", 3]]);
    expect(total).toBe(3);
  });
});

describe("BrowserTool", () => {
  it("exposes the metadata contract and action names", () => {
    const t = new BrowserTool();
    expect(t.name).toBe("browser");
    expect(t.title).toBeTruthy();
    expect(t.annotations).toEqual({ readOnlyHint: false, destructiveHint: true, openWorldHint: true });
    expect((t.schema.properties.action as { enum: string[] }).enum).toEqual([
      "launch", "attach", "tabs", "open", "navigate", "read", "click", "type", "wait_for", "screenshot", "eval", "close_tab",
    ]);
  });

  it("lists tabs and reads text + accessibility refs", async () => {
    const { drive } = await attached("jb-read");
    expect((await drive({ action: "tabs" })).output).toContain("*[0] Example — https://example.test/");
    const r = await drive({ action: "read" });
    expect(r.error, r.output).toBe(false);
    expect(r.output).toContain("Title: Example");
    expect(r.output).toContain('[e4] button "Press me"');
    expect(r.output).toContain('[e3] textbox "Search"');
    expect(r.output.match(/"Hello"/g)?.length).toBe(1);
    expect(getBrowserSession("jb-read").refs.get("T1")?.get("e4")).toBe(12);
  });

  it("clicks by ref with mouse events, by selector, by text, and falls back to DOM click", async () => {
    const { fake, drive } = await attached("jb-click");
    await drive({ action: "read" });
    let r = await drive({ action: "click", ref: "e4" });
    expect(r.output).toContain("mouse click");
    expect(fake.pages.get("T1")!.mouse.map((m) => m.type)).toEqual(["mouseMoved", "mousePressed", "mouseReleased"]);
    r = await drive({ action: "click", selector: "#b" });
    expect(r.output).toContain("selector '#b'");
    r = await drive({ action: "click", text: "Press me" });
    expect(r.output).toContain("text 'Press me'");
    fake.pages.get("T1")!.covered = true;
    r = await drive({ action: "click", selector: "button" });
    expect(r.output).toContain("dom click");
    expect(fake.pages.get("T1")!.domClicks).toBe(1);
    expect((await drive({ action: "click", ref: "e99" })).output).toContain("call read again");
    expect((await drive({ action: "click", selector: ".nope" })).output).toContain("No element matches");
    expect((await drive({ action: "click", text: "zzz" })).output).toContain("No visible element");
  });

  it("types into fields, clears, submits, and the page reflects it", async () => {
    const { fake, drive } = await attached("jb-type");
    fake.pages.get("T1")!.value = "old";
    let r = await drive({ action: "type", selector: "#q", text: "abc", clear: true });
    expect(r.output).toContain('"abc"');
    expect(fake.pages.get("T1")!.value).toBe("abc");
    await drive({ action: "read" });
    r = await drive({ action: "type", ref: "e3", text: "de", submit: true });
    expect(r.output).toContain("pressed Enter");
    expect(fake.pages.get("T1")!.keys.map((k) => k.type)).toEqual(["keyDown", "keyUp"]);
    await drive({ action: "click", ref: "e4" });
    expect((await drive({ action: "wait_for", text: "clicked abcde", timeout: 2 })).error).toBe(false);
    expect((await drive({ action: "wait_for", text: "never", timeout: 0.3 })).output).toContain("Timed out");
  });

  it("evaluates expressions and surfaces exceptions", async () => {
    const { drive } = await attached("jb-eval");
    expect((await drive({ action: "eval", expression: "6 * 7" })).output).toBe("42");
    expect((await drive({ action: "eval", expression: "undefined" })).output).toBe("undefined");
    const r = await drive({ action: "eval", expression: "throw new Error('boom')" });
    expect(r.error).toBe(true);
    expect(r.output).toContain("Error: boom");
  });

  it("returns large screenshots as PNG attachments", async () => {
    const { fake, drive } = await attached("jb-shot");
    const r = await drive({ action: "screenshot" });
    expect(r.attachments[0].mediaType).toBe("image/png");
    expect(r.attachments[0].content.equals(fake.png)).toBe(true);
    await drive({ action: "screenshot", fullPage: true });
    expect((fake.lastShot["clip"] as Record<string, number>)["height"]).toBe(3000);
  });

  it("opens, navigates and closes tabs", async () => {
    const { fake, drive } = await attached("jb-nav");
    let r = await drive({ action: "open", url: "https://two.test/" });
    expect(r.error, r.output).toBe(false);
    const id = getBrowserSession("jb-nav").current!;
    expect(id).not.toBe("T1");
    r = await drive({ action: "navigate", url: "https://three.test/" });
    expect(r.output).toContain("Navigated: Navigated — https://three.test/");
    expect((await drive({ action: "navigate", url: "bad://x" })).output).toContain("ERR_ABORTED");
    await drive({ action: "close_tab" });
    expect(fake.pages.has(id)).toBe(false);
    expect(getBrowserSession("jb-nav").current).toBe("T1");
    expect((await drive({ action: "eval", expression: "6 * 7", tab: 9 })).output).toContain("No tab at index 9");
  });

  it("records audit entries", async () => {
    const { drive } = await attached("jb-audit");
    await drive({ action: "eval", expression: "6 * 7" });
    expect(getSandbox("jb-audit").exportAuditLog().at(-1)?.action).toBe("browser");
  });

  it("refuses to launch over a foreign profile or an occupied port", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "odbrowser-"));
    try {
      fs.writeFileSync(path.join(dir, "Local State"), "{}");
      const r = await new BrowserTool().execute({ sessionId: "jb-prof" }, {
        action: "launch", port: await freePort(), profileDir: dir, browser: "/bin/sh",
      });
      expect(r.output).toContain("Refusing");
      expect(fs.existsSync(path.join(dir, PROFILE_MARKER))).toBe(false);

      const { fake } = await attached("jb-port");
      const r2 = await new BrowserTool().execute({ sessionId: "jb-port2" }, {
        action: "launch", port: fake.httpPort, profileDir: path.join(dir, "p"), browser: "/nonexistent/browser",
      });
      expect(r2.error).toBe(true);
      expect(r2.output).toContain("already has a DevTools endpoint");
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

const realBrowser = findBrowser();

describe.runIf(realBrowser !== null)("real headless browser", () => {
  it("launches, reads, clicks, evals and screenshots", async () => {
    const profile = fs.mkdtempSync(path.join(os.tmpdir(), "odbrowser-real-"));
    fs.rmSync(profile, { recursive: true, force: true });
    const tool = new BrowserTool();
    const c: ToolContext = { sessionId: "jb-real" };
    const drive = (params: Record<string, unknown>) => tool.execute(c, params);
    const html =
      "data:text/html,<title>Smoke</title><h1>Hello</h1><input id=q placeholder=Search>" +
      "<button id=b onclick=\"document.getElementById('out').textContent='clicked '+document.getElementById('q').value\">Press me</button>" +
      "<p id=out>idle</p>";
    try {
      let r = await drive({ action: "launch", headless: true, port: await freePort(), profileDir: profile, url: html, timeout: 30 });
      expect(r.error, r.output).toBe(false);
      r = await drive({ action: "read" });
      const line = r.output.split("\n").find((l) => l.includes('] button "Press me"'))!;
      const ref = line.split("[")[1].split("]")[0];
      await drive({ action: "type", selector: "#q", text: "abc" });
      expect((await drive({ action: "click", ref })).error).toBe(false);
      expect((await drive({ action: "wait_for", text: "clicked abc", timeout: 5 })).error).toBe(false);
      expect(JSON.parse((await drive({ action: "eval", expression: "document.getElementById('out').textContent" })).output)).toBe("clicked abc");
      r = await drive({ action: "screenshot" });
      expect(r.attachments[0].content.subarray(0, 4).toString("hex")).toBe("89504e47");
    } finally {
      await closeBrowserSession("jb-real");
      fs.rmSync(profile, { recursive: true, force: true });
      // Chrome's helper processes can touch the profile dir while exiting.
      await new Promise((r) => setTimeout(r, 1000));
      fs.rmSync(profile, { recursive: true, force: true });
    }
    expect(fs.existsSync(profile)).toBe(false);
  }, 60_000);
});
