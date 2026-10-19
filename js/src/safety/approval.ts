/**
 * Human approval for `confirm` decisions — mirrors python/opendesk/safety/approval.py.
 *
 * Chain: MCP elicitation (when the client supports it) → native OS dialog →
 * console prompt (TTY only) → deny.  An approver returns true / false, or
 * null for "can't ask here — try the next one".  One shared timeout.
 * OPENDESK_APPROVAL_CHAIN (e.g. "native,console" or "none") overrides the chain.
 */

import { execFile } from "child_process";

export interface ApprovalRequest {
  tool: string;
  action?: string;
  summary: string;
  reason: string;
  risk: string;
  peer?: string | null;
  secrets?: string[];
  timeout: number; // seconds
}

export interface ApprovalOutcome {
  approved: boolean;
  via: string;
  message: string;
}

export type Approver = (req: ApprovalRequest) => Promise<boolean | null>;

export const TITLE = "OpenDesk: approve agent action?";

export function approvalMessage(req: ApprovalRequest): string {
  const target = req.peer && req.peer !== "local" ? ` on ${req.peer}` : "";
  const lines = [
    `An AI agent wants to run \`${req.tool}\`${req.action ? ` (${req.action})` : ""}${target}.`,
    "",
    req.summary,
    "",
    `Why approval is needed: ${req.reason}`,
  ];
  if (req.secrets?.length) lines.push(`Uses stored secret(s): ${req.secrets.join(", ")}`);
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// MCP elicitation
// ---------------------------------------------------------------------------

interface ElicitingServer {
  getClientCapabilities?: () => { elicitation?: unknown } | undefined;
  elicitInput?: (params: unknown) => Promise<{ action: string; content?: Record<string, unknown> | null }>;
}

/** Approver backed by MCP `elicitation/create` (null when unsupported). */
export function makeMcpElicitationApprover(server: unknown): Approver {
  const s = server as ElicitingServer;
  return async (req) => {
    if (typeof s.elicitInput !== "function" || typeof s.getClientCapabilities !== "function") return null;
    if (!s.getClientCapabilities()?.elicitation) return null;
    const res = await s.elicitInput({
      message: approvalMessage(req),
      requestedSchema: {
        type: "object",
        properties: {
          allow: { type: "boolean", title: "Allow this action", description: "Tick to let the agent perform this action once.", default: false },
        },
        required: ["allow"],
      },
    });
    if (res.action !== "accept") return false;
    if (!res.content || !("allow" in res.content)) return true;
    return Boolean(res.content["allow"]);
  };
}

// ---------------------------------------------------------------------------
// Native dialog
// ---------------------------------------------------------------------------

export interface RunResult { code: number; stdout: string; stderr: string }

function appleStr(s: string): string {
  return `"${s.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

export function nativeDialogArgv(req: ApprovalRequest, platform: string = process.platform): string[] | null {
  const secs = Math.max(1, Math.floor(req.timeout));
  const msg = approvalMessage(req);
  if (platform === "darwin") {
    return ["osascript", "-e",
      `display dialog ${appleStr(msg)} with title ${appleStr(TITLE)} buttons {"Deny", "Allow"} ` +
      `default button "Deny" cancel button "Deny" with icon caution giving up after ${secs}`];
  }
  if (platform === "win32") {
    const m = msg.replace(/'/g, "''"), t = TITLE.replace(/'/g, "''");
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command",
      `$r = (New-Object -ComObject WScript.Shell).Popup('${m}', ${secs}, '${t}', 4 + 48 + 256 + 4096); Write-Output $r`];
  }
  if (!process.env["DISPLAY"] && !process.env["WAYLAND_DISPLAY"]) return null;
  return ["zenity", "--question", "--title", TITLE, "--text", msg, "--ok-label", "Allow", "--cancel-label", "Deny", `--timeout=${secs}`];
}

export function parseNativeResult(argv: string[], r: RunResult): boolean | null {
  const exe = argv[0].split(/[\\/]/).pop()!.toLowerCase();
  const out = r.stdout.trim();
  if (exe === "osascript") {
    if (r.code === 0) return out.includes("button returned:Allow") && !out.includes("gave up:true");
    if (r.stderr.includes("-128") || r.stderr.includes("User canceled")) return false;
    return null;
  }
  if (exe.startsWith("powershell") || exe.startsWith("pwsh")) {
    if (r.code !== 0) return null;
    return out.split(/\r?\n/).pop()?.trim() === "6";
  }
  if (exe === "zenity") return [0, 1, 5].includes(r.code) ? r.code === 0 : null;
  return null;
}

function run(argv: string[], timeoutMs: number): Promise<RunResult> {
  return new Promise((resolve) => {
    execFile(argv[0], argv.slice(1), { timeout: timeoutMs }, (err, stdout, stderr) => {
      const code = err ? (typeof (err as { code?: unknown }).code === "number" ? (err as { code: number }).code : -1) : 0;
      resolve({ code, stdout: String(stdout ?? ""), stderr: String(stderr ?? "") || (err && code === -1 ? String(err) : "") });
    });
  });
}

async function nativeDialog(req: ApprovalRequest): Promise<boolean | null> {
  const argv = nativeDialogArgv(req);
  if (!argv) return null;
  const r = await hooks.run(argv, (req.timeout + 5) * 1000);
  if (r.code === -1 && !r.stdout) return null;
  return parseNativeResult(argv, r);
}

// ---------------------------------------------------------------------------
// Console
// ---------------------------------------------------------------------------

async function consolePrompt(req: ApprovalRequest): Promise<boolean | null> {
  if (!process.stdin.isTTY) return null;
  const readline = await import("readline");
  const rl = readline.createInterface({ input: process.stdin, output: process.stderr });
  const answer = await new Promise<string>((resolve) =>
    rl.question(`\n[opendesk] ${approvalMessage(req).replace(/\n/g, "\n[opendesk] ")}\nAllow? [y/N] `, resolve),
  );
  rl.close();
  return ["y", "yes"].includes(answer.trim().toLowerCase());
}

/** Replaceable hooks (tests swap these so no real dialog / prompt is shown). */
export const hooks: {
  native: Approver;
  console: Approver;
  run: (argv: string[], timeoutMs: number) => Promise<RunResult>;
} = { native: nativeDialog, console: consolePrompt, run };

// ---------------------------------------------------------------------------
// Chain
// ---------------------------------------------------------------------------

export function resolveChain(configured?: string[] | null): string[] {
  const env = process.env["OPENDESK_APPROVAL_CHAIN"];
  let names: string[];
  if (env !== undefined) names = env.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
  else if (configured) names = configured.map((s) => String(s).toLowerCase());
  else names = ["mcp", "native", "console"];
  return names.filter((n) => ["mcp", "native", "console"].includes(n));
}

export async function requestApproval(
  req: ApprovalRequest,
  opts: { mcpApprover?: Approver | null; chain?: string[] | null } = {},
): Promise<ApprovalOutcome> {
  const deadline = Date.now() + req.timeout * 1000;
  for (const name of resolveChain(opts.chain)) {
    const approver = name === "mcp" ? opts.mcpApprover : name === "native" ? hooks.native : hooks.console;
    if (!approver) continue;
    const remaining = deadline - Date.now();
    const timedOut: ApprovalOutcome = { approved: false, via: name, message: `approval timed out after ${req.timeout}s` };
    if (remaining <= 0) return timedOut;
    let timer: NodeJS.Timeout | undefined;
    let answer: boolean | null | "timeout";
    try {
      answer = await Promise.race([
        approver(req),
        new Promise<"timeout">((resolve) => { timer = setTimeout(() => resolve("timeout"), remaining); }),
      ]);
    } catch {
      continue;
    } finally {
      if (timer) clearTimeout(timer);
    }
    if (answer === "timeout") return timedOut;
    if (answer === null || answer === undefined) continue;
    return answer
      ? { approved: true, via: name, message: `approved via ${name}` }
      : { approved: false, via: name, message: `the user denied it (${name})` };
  }
  return {
    approved: false,
    via: "none",
    message: "no way to ask the user for approval here (no MCP elicitation support, no desktop dialog, no TTY). " +
      `Ask the user to allow it with a policy rule, e.g. \`opendesk-js policy add-rule --tool ${req.tool}` +
      `${req.action ? ` --action ${req.action}` : ""} --decision allow\`, or perform it themselves.`,
  };
}
