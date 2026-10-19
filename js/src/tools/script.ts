/**
 * ScriptTool — run shell / AppleScript / JXA / PowerShell code.
 * Mirrors Python's tools/script.py.
 *
 * Locally the code runs via child_process (argv, no host-side shell
 * interpolation).  On a paired peer the MCP server forwards the call as
 * `tool.script`, so it runs on that machine through its ToolDispatcher.
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import { Tool, ToolContext, ToolResult, checkPermission } from "./base.js";
import { getSandbox } from "../computer/sandbox.js";

export const MAX_TIMEOUT = 600;
export const DEFAULT_TIMEOUT = 60;
export const MAX_STREAM_CHARS = 16_000;
const HEAD_CHARS = 12_000;
/** Hard cap on bytes buffered per stream before we stop collecting. */
const MAX_CAPTURE_BYTES = 8 * 1024 * 1024;

export type ScriptAction = "shell" | "applescript" | "jxa" | "powershell";
export type OsName = "darwin" | "windows" | "linux" | "unknown";

export function normalizeOs(raw: string): OsName {
  const v = (raw ?? "").trim().toLowerCase();
  if (["darwin", "macos", "mac", "osx", "mac os x"].includes(v)) return "darwin";
  if (v.startsWith("win") || ["mingw", "cygwin", "msys"].some((k) => v.includes(k))) return "windows";
  if (v.startsWith("linux") || ["freebsd", "openbsd", "netbsd"].includes(v)) return "linux";
  return "unknown";
}

/** Return [argv, stdin] for running `code` with `action` on `os`. Throws on unsupported combos. */
export function buildArgv(action: string, code: string, os: OsName): [string[], string | null] {
  switch (action) {
    case "shell":
      return os === "windows" ? [["cmd.exe", "/d", "/s", "/c", code], null] : [["/bin/sh", "-c", code], null];
    case "applescript":
    case "jxa": {
      if (os !== "darwin" && os !== "unknown") {
        throw new Error(
          `action='${action}' needs macOS (osascript); the target is ${os}. Use action='shell'` +
            (os === "windows" ? " or 'powershell'" : "") + " instead.",
        );
      }
      const argv = action === "applescript" ? ["osascript"] : ["osascript", "-l", "JavaScript"];
      // One multi-line -e works, and keeps the source visible to approvers / audit logs.
      return [[...argv, "-e", code], null];
    }
    case "powershell": {
      const encoded = Buffer.from(code, "utf16le").toString("base64");
      const exe = os === "windows" ? "powershell.exe" : "pwsh";
      return [[exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], null];
    }
    default:
      throw new Error(`Unknown script action: '${action}'`);
  }
}

/** Keep head + tail of `text` when longer than `limit`. */
export function truncateStream(text: string, limit = MAX_STREAM_CHARS): [string, boolean] {
  if (text.length <= limit) return [text, false];
  const head = Math.min(HEAD_CHARS, Math.floor((limit * 3) / 4));
  const tail = Math.max(0, limit - head);
  const dropped = text.length - head - tail;
  return [
    text.slice(0, head) +
      `\n… [truncated: ${dropped.toLocaleString("en-US")} of ${text.length.toLocaleString("en-US")} chars omitted] …\n` +
      (tail ? text.slice(-tail) : ""),
    true,
  ];
}

export interface RunResult {
  returncode: number;
  stdout: string;
  stderr: string;
  duration: number;
  timedOut: boolean;
}

/** Spawn argv, feed stdin, collect output, kill after `timeoutS`. */
export function runProcess(
  argv: string[],
  opts: { stdin?: string | null; cwd?: string; timeoutS: number },
): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const child = spawn(argv[0], argv.slice(1), {
      cwd: opts.cwd,
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const out: Buffer[] = [];
    const err: Buffer[] = [];
    let outBytes = 0;
    let errBytes = 0;
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, opts.timeoutS * 1000);

    child.stdout.on("data", (d: Buffer) => {
      if (outBytes < MAX_CAPTURE_BYTES) out.push(d);
      outBytes += d.length;
    });
    child.stderr.on("data", (d: Buffer) => {
      if (errBytes < MAX_CAPTURE_BYTES) err.push(d);
      errBytes += d.length;
    });
    child.on("error", (e) => {
      clearTimeout(timer);
      reject(e);
    });
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      resolve({
        returncode: code ?? (signal ? 128 + 9 : 0),
        stdout: Buffer.concat(out).toString("utf8"),
        stderr: Buffer.concat(err).toString("utf8"),
        duration: (Date.now() - start) / 1000,
        timedOut,
      });
    });
    child.stdin.on("error", () => {
      /* process may exit before reading stdin */
    });
    if (opts.stdin != null) child.stdin.write(opts.stdin);
    child.stdin.end();
  });
}

export interface ScriptToolOptions {
  /** Override the detected platform (tests). */
  platform?: string;
}

export class ScriptTool extends Tool {
  name = "script";
  title = "Run script";
  annotations = { readOnlyHint: false, destructiveHint: true, openWorldHint: true };
  description =
    "Run code on the target machine and return exit code, stdout and stderr.\n\n" +
    "ROUTING RULE — prefer this over GUI clicks whenever the task does not need visual " +
    "interaction: reading/writing files, CLIs (git, curl, ffmpeg …), macOS `defaults` / `open` / " +
    "`mdfind`, an app's AppleScript dictionary (Finder, Mail, Safari, Music, System Events), " +
    "Windows PowerShell/WMI/registry. One script call usually replaces many screenshot → click " +
    "steps and is deterministic. Fall back to `ui` (accessibility) and only then `mouse` when the " +
    "task genuinely needs the visual UI.\n\n" +
    "Actions:\n" +
    "  shell       — platform shell (/bin/sh -c on macOS/Linux, cmd.exe on Windows)\n" +
    "  applescript — osascript (macOS only), e.g. 'tell application \"Finder\" to get name of every window'\n" +
    "  jxa         — JavaScript for Automation, osascript -l JavaScript (macOS only)\n" +
    "  powershell  — powershell.exe on Windows, pwsh elsewhere if installed\n\n" +
    `timeout defaults to ${DEFAULT_TIMEOUT}s (max ${MAX_TIMEOUT}). Long output is truncated ` +
    "(head and tail kept). A non-zero exit code is reported as an error.";

  schema = {
    type: "object",
    required: ["action", "code"],
    properties: {
      action: { type: "string", enum: ["shell", "applescript", "jxa", "powershell"], description: "Interpreter." },
      code: { type: "string", description: "The command / script source to run." },
      timeout: { type: "number", description: `Seconds before the process is killed (default ${DEFAULT_TIMEOUT}, max ${MAX_TIMEOUT}).` },
      cwd: { type: "string", description: "Working directory." },
    },
  };

  private platform: string;

  constructor(opts: ScriptToolOptions = {}) {
    super();
    this.platform = opts.platform ?? process.platform;
  }

  async execute(ctx: ToolContext, params: Record<string, unknown>): Promise<ToolResult> {
    const action = String(params["action"] ?? "");
    const code = typeof params["code"] === "string" ? (params["code"] as string) : "";
    const cwd = typeof params["cwd"] === "string" ? (params["cwd"] as string) : undefined;
    const rawTimeout = params["timeout"] == null ? DEFAULT_TIMEOUT : Number(params["timeout"]);
    if (!Number.isFinite(rawTimeout) || rawTimeout <= 0 || rawTimeout > MAX_TIMEOUT) {
      return this.err("Script error", `timeout must be in (0, ${MAX_TIMEOUT}] seconds.`);
    }
    const timeout = rawTimeout;

    let preview = code.trim().replace(/\n/g, " ⏎ ");
    if (preview.length > 200) preview = preview.slice(0, 200) + "…";
    await checkPermission(ctx, "script", `${action}: ${preview}`, `Run ${action} script: ${preview}`);

    const sandbox = getSandbox(ctx.sessionId);
    const auditParams = { action, code: code.slice(0, 1000), cwd, timeout };

    if (!code.trim()) return this.err("Script error", "code must not be empty.");

    const os = normalizeOs(this.platform === "win32" ? "windows" : this.platform);
    let argv: string[];
    let stdin: string | null;
    try {
      [argv, stdin] = buildArgv(action, code, os);
    } catch (e) {
      const msg = (e as Error).message;
      sandbox.recordAction("script", auditParams, undefined, msg);
      return this.err("Script unsupported", msg);
    }
    if (cwd && !fs.existsSync(cwd)) {
      const msg = `Working directory not found: ${cwd}`;
      sandbox.recordAction("script", auditParams, undefined, msg);
      return this.err("Script error", msg);
    }

    let res: RunResult;
    try {
      res = await runProcess(argv, { stdin, cwd, timeoutS: timeout });
    } catch (e) {
      const errno = e as NodeJS.ErrnoException;
      const msg = errno.code === "ENOENT"
        ? `Interpreter not found for action='${action}' (${argv[0]}): ${errno.message}`
        : `${action} failed: ${errno.message ?? e}`;
      sandbox.recordAction("script", auditParams, undefined, msg);
      return this.err("Script error", msg);
    }
    if (res.timedOut) {
      const msg = `Timed out after ${timeout}s (process killed).`;
      sandbox.recordAction("script", auditParams, undefined, msg);
      return this.err("Script timeout", msg);
    }

    const [stdout, outCut] = truncateStream(res.stdout);
    const [stderr, errCut] = truncateStream(res.stderr);
    const rc = res.returncode;
    const lines = [`Exit code: ${rc}  (${res.duration.toFixed(2)}s, ${action} on ${os})`];
    lines.push("--- stdout ---" + (outCut ? " (truncated)" : ""));
    lines.push(stdout ? stdout.replace(/\n+$/, "") : "(empty)");
    if (stderr || rc !== 0) {
      lines.push("--- stderr ---" + (errCut ? " (truncated)" : ""));
      lines.push(stderr ? stderr.replace(/\n+$/, "") : "(empty)");
    }
    sandbox.recordAction(
      "script", auditParams,
      `exit ${rc}, ${Buffer.byteLength(res.stdout)}B stdout, ${Buffer.byteLength(res.stderr)}B stderr`,
      rc === 0 ? undefined : `exit ${rc}`,
    );
    return {
      title: `Script (${action}) exit ${rc}`,
      output: lines.join("\n"),
      error: rc !== 0,
      attachments: [],
      metadata: { returncode: rc, duration: res.duration, stdoutTruncated: outCut, stderrTruncated: errCut, os },
    };
  }
}
