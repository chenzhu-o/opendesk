import { describe, it, expect } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { ScriptTool, buildArgv, normalizeOs, truncateStream, MAX_STREAM_CHARS } from "../src/tools/script.js";
import { getSandbox } from "../src/computer/sandbox.js";
import { createRegistry } from "../src/registry.js";
import type { ToolContext } from "../src/tools/base.js";

const posix = process.platform !== "win32";
const mac = process.platform === "darwin";

function ctx(sessionId = "script-test"): ToolContext {
  return { sessionId };
}

describe("script helpers", () => {
  it("normalizes OS names", () => {
    expect(normalizeOs("Darwin")).toBe("darwin");
    expect(normalizeOs("Windows")).toBe("windows");
    expect(normalizeOs("win32")).toBe("windows");
    expect(normalizeOs("MINGW64_NT-10.0")).toBe("windows");
    expect(normalizeOs("linux")).toBe("linux");
    expect(normalizeOs("fake")).toBe("unknown");
  });

  it("builds argv per action and OS", () => {
    expect(buildArgv("shell", "ls -la", "darwin")).toEqual([["/bin/sh", "-c", "ls -la"], null]);
    expect(buildArgv("shell", "dir", "windows")[0][0]).toBe("cmd.exe");
    expect(buildArgv("applescript", 'return "hi"', "darwin")).toEqual([["osascript", "-e", 'return "hi"'], null]);
    expect(buildArgv("jxa", "1 + 1", "darwin")).toEqual([["osascript", "-l", "JavaScript", "-e", "1 + 1"], null]);
    const [ps] = buildArgv("powershell", "Write-Output 'é'", "windows");
    expect(ps[0]).toBe("powershell.exe");
    expect(Buffer.from(ps[ps.length - 1], "base64").toString("utf16le")).toBe("Write-Output 'é'");
    expect(buildArgv("powershell", "x", "linux")[0][0]).toBe("pwsh");
  });

  it("rejects AppleScript/JXA off macOS", () => {
    expect(() => buildArgv("applescript", "beep", "linux")).toThrow(/macOS/);
    expect(() => buildArgv("jxa", "1", "windows")).toThrow(/macOS/);
  });

  it("truncates long output keeping head and tail", () => {
    const [out, cut] = truncateStream("A".repeat(20_000) + "TAIL");
    expect(cut).toBe(true);
    expect(out.endsWith("TAIL")).toBe(true);
    expect(out).toContain("truncated");
    expect(truncateStream("short")).toEqual(["short", false]);
  });
});

describe("ScriptTool", () => {
  it("exposes the metadata contract", () => {
    const t = new ScriptTool();
    expect(t.name).toBe("script");
    expect(t.title).toBeTruthy();
    expect(t.annotations).toEqual({ readOnlyHint: false, destructiveHint: true, openWorldHint: true });
    expect(t.description.toLowerCase()).toContain("prefer");
    expect(createRegistry().get("script")).toBeInstanceOf(ScriptTool);
  });

  it.runIf(posix)("runs a shell command with cwd and captures both streams", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "odscript-"));
    try {
      const r = await new ScriptTool().execute(ctx(), { action: "shell", code: "echo out; echo err 1>&2; pwd", cwd: dir });
      expect(r.error).toBe(false);
      expect(r.output).toContain("Exit code: 0");
      expect(r.output).toContain("out");
      expect(r.output).toContain("err");
      expect(r.output).toContain(path.basename(dir));
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it.runIf(posix)("reports non-zero exit as an error", async () => {
    const r = await new ScriptTool().execute(ctx(), { action: "shell", code: "echo nope 1>&2; exit 3" });
    expect(r.error).toBe(true);
    expect(r.output).toContain("Exit code: 3");
    expect(r.metadata["returncode"]).toBe(3);
  });

  it.runIf(posix)("kills on timeout", async () => {
    const t0 = Date.now();
    const r = await new ScriptTool().execute(ctx(), { action: "shell", code: "sleep 5", timeout: 0.3 });
    expect(r.error).toBe(true);
    expect(r.output).toContain("Timed out");
    expect(Date.now() - t0).toBeLessThan(4000);
  });

  it.runIf(posix)("truncates long output and says so", async () => {
    const r = await new ScriptTool().execute(ctx(), { action: "shell", code: `head -c ${MAX_STREAM_CHARS * 3} /dev/zero | tr '\\0' x` });
    expect(r.output).toContain("truncated");
    expect(r.metadata["stdoutTruncated"]).toBe(true);
  });

  it.runIf(mac)("runs AppleScript and JXA through osascript", async () => {
    const a = await new ScriptTool().execute(ctx(), { action: "applescript", code: "set x to 1 + 2\nreturn x" });
    expect(a.error).toBe(false);
    expect(a.output).toContain("3");
    const j = await new ScriptTool().execute(ctx(), { action: "jxa", code: "const a = 2;\na * 21" });
    expect(j.output).toContain("42");
  });

  it("gives a clear error for AppleScript on non-mac targets", async () => {
    const r = await new ScriptTool({ platform: "linux" }).execute(ctx(), { action: "applescript", code: "beep" });
    expect(r.error).toBe(true);
    expect(r.output).toContain("macOS");
  });

  it("validates inputs", async () => {
    const t = new ScriptTool();
    expect((await t.execute(ctx(), { action: "shell", code: "  " })).error).toBe(true);
    expect((await t.execute(ctx(), { action: "shell", code: "x", timeout: 601 })).output).toContain("timeout");
    const missing = await t.execute(ctx(), { action: "shell", code: "pwd", cwd: "/definitely/not/here" });
    expect(missing.output).toContain("Working directory not found");
  });

  it("reports a missing interpreter", async () => {
    const r = await new ScriptTool({ platform: "linux" }).execute(ctx(), { action: "powershell", code: "Get-Date" });
    // pwsh is normally absent on CI/dev boxes; if present it should simply run.
    if (r.error) expect(r.output).toMatch(/not found|Exit code/);
  });

  it.runIf(posix)("records an audit entry", async () => {
    await new ScriptTool().execute(ctx("script-audit-js"), { action: "shell", code: "echo hi" });
    const log = getSandbox("script-audit-js").exportAuditLog();
    expect(log.at(-1)?.action).toBe("script");
    expect(log.at(-1)?.params["code"]).toBe("echo hi");
  });

  it("asks the permission handler with the code", async () => {
    const seen: string[] = [];
    const c: ToolContext = { sessionId: "perm", permissionHandler: async (tool, arg) => { seen.push(`${tool}|${arg}`); } };
    await new ScriptTool({ platform: "linux" }).execute(c, { action: "applescript", code: "beep" });
    expect(seen[0]).toBe("script|applescript: beep");
  });
});
