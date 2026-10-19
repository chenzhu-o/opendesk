import { afterAll, afterEach, describe, expect, it } from "vitest";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as native from "../src/computer/macosNative.js";
import { AXHelper, NativeAXError, SWIFT_SOURCE, formatTree, findNode, setClient, windowText } from "../src/computer/macosNative.js";
import { MouseTool } from "../src/tools/mouse.js";
import type { ToolContext } from "../src/tools/base.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..", "..");

// ---------------------------------------------------------------------------
// Fake helper: a tiny Node script that speaks the JSON-lines protocol.
// ---------------------------------------------------------------------------

const FAKE = String.raw`
const readline = require("node:readline");
const trusted = process.env.FAKE_TRUSTED !== "0";
const rl = readline.createInterface({ input: process.stdin });
const reply = (o) => process.stdout.write(JSON.stringify(o) + "\n");
rl.on("line", (line) => {
  const req = JSON.parse(line);
  switch (req.cmd) {
    case "ping": return reply({ id: req.id, ok: true, result: { version: 1, trusted, pid: process.pid } });
    case "tree": return reply({ id: req.id, ok: true, result: {
      pid: 42, app: req.app || "Fake", window_title: "Doc", truncated: false, nodes: [
        { ref: "w00001", parent: null, role: "AXWindow", title: "Doc", enabled: true, focused: false },
        { ref: "g00001", parent: "w00001", role: "AXGroup", enabled: true, focused: false },
        { ref: "b00001", parent: "g00001", role: "AXButton", title: "Save", enabled: true, focused: false,
          frame: { x: 10, y: 20, w: 30, h: 40 }, actions: ["AXPress"] },
        { ref: "t00001", parent: "w00001", role: "AXTextArea", value: "hello", enabled: true, focused: true },
      ] } });
    case "window_text": return reply({ id: req.id, ok: true, result: { text: "Doc\nhello " + (req.max_chars || 0) } });
    case "echo": return reply({ id: req.id, ok: true, result: req });
    case "fail": return reply({ id: req.id, ok: false, error: "boom" });
    case "hang": return;
    case "crash": process.exit(3);
    default: return reply({ id: req.id, ok: false, error: "unknown command: " + req.cmd });
  }
});
`;

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "opendesk-fake-ax-"));
const fakePath = path.join(tmp, "fake-ax.cjs");
fs.writeFileSync(fakePath, FAKE);
const fakeArgv = [process.execPath, fakePath];

afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));
afterEach(() => {
  setClient(null);
  delete process.env["FAKE_TRUSTED"];
  delete process.env["OPENDESK_BACKGROUND"];
});

describe("Swift source", () => {
  it("is byte-identical to the Python copy", () => {
    const py = fs.readFileSync(path.join(repo, "python", "opendesk", "computer", "macos_native.py"), "utf8");
    const marker = "SWIFT_SOURCE = r'''";
    const start = py.indexOf(marker) + marker.length;
    const end = py.indexOf("'''", start);
    expect(start).toBeGreaterThan(marker.length);
    expect(py.slice(start, end)).toBe(SWIFT_SOURCE);
  });

  it("supports every documented command", () => {
    for (const cmd of ["ping", "frontmost", "tree", "get", "perform", "set_value", "focus",
      "element_at", "window_text", "click_pid", "key_pid"]) {
      expect(SWIFT_SOURCE).toContain(`case "${cmd}"`);
    }
    expect(SWIFT_SOURCE).toContain("AXUIElementCopyMultipleAttributeValues");
    expect(SWIFT_SOURCE).toContain("AXManualAccessibility");
    expect(SWIFT_SOURCE).toContain("postToPid");
  });
});

describe("AXHelper protocol", () => {
  it("round-trips requests by id and drops null args", async () => {
    const h = new AXHelper(fakeArgv);
    const r = await h.request<Record<string, unknown>>("echo", { a: 1, b: null, c: undefined });
    expect(r["cmd"]).toBe("echo");
    expect(r["a"]).toBe(1);
    expect("b" in r).toBe(false);
    const [x, y] = await Promise.all([h.request("echo", { n: 1 }), h.request("echo", { n: 2 })]);
    expect((x as { n: number }).n).toBe(1);
    expect((y as { n: number }).n).toBe(2);
    h.close();
  });

  it("surfaces helper errors", async () => {
    const h = new AXHelper(fakeArgv);
    await expect(h.request("fail")).rejects.toThrow("boom");
    h.close();
  });

  it("times out, kills the helper, and restarts on the next request", async () => {
    const h = new AXHelper(fakeArgv, 300);
    const pid1 = (await h.request<{ pid: number }>("ping")).pid;
    await expect(h.request("hang")).rejects.toBeInstanceOf(NativeAXError);
    const pid2 = (await h.request<{ pid: number }>("ping")).pid;
    expect(pid2).not.toBe(pid1);
    expect(h.restarts).toBe(1);
    h.close();
  });

  it("restarts after a crash", async () => {
    const h = new AXHelper(fakeArgv);
    await expect(h.request("crash")).rejects.toThrow(/exited/);
    const r = await h.request<{ trusted: boolean }>("ping");
    expect(r.trusted).toBe(true);
    h.close();
  });
});

describe("public API with a fake helper", () => {
  it("windowText returns text, and null when not trusted", async () => {
    setClient(new AXHelper(fakeArgv));
    expect(await windowText(undefined, 123)).toBe("Doc\nhello 123");
    process.env["FAKE_TRUSTED"] = "0";
    setClient(new AXHelper(fakeArgv));
    expect(await native.available()).toBe(false);
    expect(await windowText()).toBeNull();
  });

  it("windowText is null when the helper cannot start", async () => {
    setClient(new AXHelper(["/nonexistent/opendesk-ax"]));
    expect(await windowText()).toBeNull();
  });

  it("formats trees with refs and hides layout groups", async () => {
    setClient(new AXHelper(fakeArgv));
    const t = await native.tree({ app: "Fake" });
    const text = formatTree(t);
    expect(text).toContain("Window: Doc  [w00001]");
    expect(text).toContain("[b00001] [AXButton] Save");
    expect(text).toContain('[t00001] [AXTextArea] (unnamed) = "hello"');
    expect(text).not.toContain("AXGroup");
    expect(findNode(t, { title: "save" })?.ref).toBe("b00001");
    expect(findNode(t, { role: "button" })?.ref).toBe("b00001");
  });
});

describe("mouse background mode", () => {
  const ctx = { sessionId: "ax-test" } as ToolContext;

  it("refuses cursor-moving actions when OPENDESK_BACKGROUND=1", async () => {
    process.env["OPENDESK_BACKGROUND"] = "1";
    const r = await new MouseTool().execute(ctx, { action: "drag", x: 1, y: 2, endX: 3, endY: 4 });
    expect(r.error).toBe(true);
    expect(r.output).toContain("ref");
  });

  it("refuses background=true for non-click actions", async () => {
    const r = await new MouseTool().execute(ctx, { action: "move", x: 1, y: 2, background: true });
    expect(r.error).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Real helper (macOS with swiftc only)
// ---------------------------------------------------------------------------

function hasSwiftc(): boolean {
  try {
    execFileSync("which", ["swiftc"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

const real = process.platform === "darwin" && hasSwiftc() && process.env["OPENDESK_NATIVE_AX"] !== "0";

describe.skipIf(!real)("real opendesk-ax helper", () => {
  it("answers ping, frontmost, tree and window_text", async () => {
    if (!(await native.available())) return; // no Accessibility permission here
    const front = await native.frontmost();
    expect(front?.app).toBeTruthy();
    const t = await native.tree({ app: "Finder" });
    expect(t.app).toBe("Finder");
    expect(t.nodes.length).toBeGreaterThan(0);
    const again = await native.tree({ app: "Finder" });
    expect(again.nodes.map((n) => n.ref).slice(0, 20)).toEqual(t.nodes.map((n) => n.ref).slice(0, 20));
    expect(typeof (await windowText())).toBe("string");
  }, 400_000);
});
