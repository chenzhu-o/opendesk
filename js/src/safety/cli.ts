/**
 * `opendesk-js stop|resume|takeover|handback|secrets|policy|audit verify`.
 * Same files as the Python CLI (`opendesk ...`), so either can manage them.
 */

import path from "path";
import { readControl, setMode } from "./control.js";
import { describeVerify } from "./chain.js";
import { loadPolicy, policyPath, PRESETS, savePolicy, validateRule, approvalTimeout, taintWindow } from "./policy.js";
import { deleteSecret, listSecrets, SecretError, setSecret } from "./secrets.js";
import { auditFiles, verifyFiles } from "./toolAudit.js";

export interface CliIO {
  out: (s: string) => void;
  err: (s: string) => void;
  readSecret: (prompt: string) => Promise<string>;
  exit: (code: number) => never;
}

function parse(argv: string[]): { pos: string[]; flags: Record<string, string | boolean> } {
  const pos: string[] = [];
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const eq = a.indexOf("=");
      if (eq > 2) flags[a.slice(2, eq)] = a.slice(eq + 1);
      else if (argv[i + 1] !== undefined && !argv[i + 1].startsWith("--")) flags[a.slice(2)] = argv[++i];
      else flags[a.slice(2)] = true;
    } else pos.push(a);
  }
  return { pos, flags };
}

async function readHidden(prompt: string): Promise<string> {
  const stdin = process.stdin;
  if (!stdin.isTTY) {
    const chunks: Buffer[] = [];
    for await (const c of stdin) chunks.push(c as Buffer);
    return Buffer.concat(chunks).toString("utf8").split("\n")[0];
  }
  process.stderr.write(prompt);
  return new Promise((resolve) => {
    let buf = "";
    stdin.setRawMode(true);
    stdin.resume();
    stdin.setEncoding("utf8");
    const onData = (ch: string) => {
      for (const c of ch) {
        if (c === "\r" || c === "\n") {
          stdin.setRawMode(false); stdin.pause(); stdin.removeListener("data", onData);
          process.stderr.write("\n");
          resolve(buf);
          return;
        }
        if (c === "\u0003") { stdin.setRawMode(false); process.exit(130); }
        if (c === "\u007f") buf = buf.slice(0, -1);
        else buf += c;
      }
    };
    stdin.on("data", onData);
  });
}

const defaultIO: CliIO = {
  out: (s) => console.log(s),
  err: (s) => process.stderr.write(s + "\n"),
  readSecret: readHidden,
  exit: (code) => process.exit(code),
};

export async function runSafetyCommand(command: string, argv: string[], io: CliIO = defaultIO): Promise<void> {
  const { pos, flags } = parse(argv);
  const home = flags["home"] ? path.resolve(String(flags["home"])) : undefined;
  const reason = typeof flags["reason"] === "string" ? flags["reason"] : "";

  switch (command) {
    case "stop":
      setMode("stopped", home, reason || "opendesk stop");
      io.out("Stopped. Every agent tool call is now refused. Run `opendesk-js resume` to continue.");
      return;
    case "takeover":
      setMode("takeover", home, reason || "user took control");
      io.out("You have control. Agent tool calls are refused until `opendesk-js handback`.");
      return;
    case "resume":
    case "handback": {
      const prev = readControl(home).mode;
      setMode("running", home, reason || (command === "resume" ? "resumed" : "handed back"));
      if (prev !== "running") io.out(`(was ${prev})`);
      io.out(command === "resume" ? "Resumed. Agents may use tools again." : "Handed back. Agents may use tools again.");
      return;
    }
    case "secrets":
      return secretsCmd(pos, flags, home, io);
    case "policy":
      return policyCmd(pos, flags, home, io);
    case "audit-verify": {
      const files = auditFiles(home, typeof flags["date"] === "string" ? flags["date"] : undefined);
      if (!files.length) { io.out("No audit files found."); return; }
      const results = verifyFiles(files);
      for (const r of results) io.out(describeVerify(r));
      const bad = results.filter((r) => !r.ok).length;
      if (bad) {
        io.out(`\n${bad} file(s) failed verification — the log may have been tampered with.`);
        io.exit(1);
      }
      io.out(`\nAll ${results.length} file(s) verified.`);
      return;
    }
    default:
      io.err(`unknown safety command: ${command}`);
      io.exit(2);
  }
}

async function secretsCmd(pos: string[], flags: Record<string, string | boolean>, home: string | undefined, io: CliIO): Promise<void> {
  const sub = pos[0] ?? "list";
  try {
    if (sub === "set") {
      const name = pos[1];
      if (!name) { io.err("usage: secrets set NAME [--allowed-apps A,B] [--no-confirm]"); io.exit(2); }
      const value = await io.readSecret(`Value for secret '${name}': `);
      const apps = typeof flags["allowed-apps"] === "string" ? flags["allowed-apps"].split(",").map((s) => s.trim()).filter(Boolean) : [];
      const info = await setSecret(name, value, home, { allowedApps: apps, confirm: !flags["no-confirm"] });
      io.out(`Stored secret '${info.name}'${apps.length ? ` (allowed apps: ${apps.join(", ")})` : ""}. Use it as {{secret:${info.name}}}.`);
    } else if (sub === "delete") {
      const name = pos[1];
      if (!name) { io.err("usage: secrets delete NAME"); io.exit(2); }
      io.out((await deleteSecret(name, home)) ? `Deleted '${name}'.` : `'${name}' was not in the index.`);
    } else {
      const items = listSecrets(home);
      if (!items.length) { io.out("No secrets stored. Add one with `opendesk-js secrets set NAME`."); return; }
      for (const i of items) {
        const f = [...(i.confirm ? [] : ["no-confirm"]), ...(i.allowed_apps.length ? [`apps=${i.allowed_apps.join(",")}`] : [])];
        io.out(`  ${i.name}${f.length ? `  [${f.join("; ")}]` : ""}`);
      }
    }
  } catch (e) {
    if (e instanceof SecretError) { io.err(`ERROR: ${e.message}`); io.exit(1); }
    throw e;
  }
}

function policyCmd(pos: string[], flags: Record<string, string | boolean>, home: string | undefined, io: CliIO): void {
  const sub = pos[0] ?? "show";
  const cfg = loadPolicy(home);
  if (sub === "preset") {
    const name = pos[1];
    if (!name || !(PRESETS as readonly string[]).includes(name)) { io.err(`preset must be one of ${PRESETS.join(", ")}`); io.exit(2); }
    cfg.preset = name as (typeof PRESETS)[number];
    io.out(`Preset set to '${name}' → ${savePolicy(cfg, home)}`);
    return;
  }
  if (sub === "add-rule") {
    let rule;
    try {
      rule = validateRule({
        tool: flags["tool"] as string, action: flags["action"] as string, app: flags["app"] as string,
        peer: flags["peer"] as string, decision: flags["decision"] as never, reason: flags["reason"] as string,
      });
    } catch (e) { io.err(`ERROR: ${(e as Error).message}`); io.exit(2); }
    if (flags["first"]) cfg.rules.unshift(rule); else cfg.rules.push(rule);
    savePolicy(cfg, home);
    io.out(`Added rule: ${JSON.stringify(rule)}`);
    return;
  }
  if (sub === "remove-rule") {
    const idx = Number(pos[1]);
    if (!Number.isInteger(idx) || idx < 0 || idx >= cfg.rules.length) { io.err(`no rule #${pos[1]}`); io.exit(2); }
    const [removed] = cfg.rules.splice(idx, 1);
    savePolicy(cfg, home);
    io.out(`Removed rule: ${JSON.stringify(removed)}`);
    return;
  }
  if (sub === "kill-hotkey") {
    if (flags["clear"]) delete cfg.kill_hotkey; else if (pos[1]) cfg.kill_hotkey = pos[1];
    savePolicy(cfg, home);
    io.out(`Kill hotkey: ${cfg.kill_hotkey ?? "(none)"} — takes effect on the next MCP server start.`);
    return;
  }
  const state = readControl(home);
  const w = taintWindow(cfg);
  io.out(`Policy file: ${policyPath(home)}`);
  io.out(`Preset:      ${cfg.preset}`);
  io.out(`Control:     ${state.mode}${state.reason ? ` (${state.reason})` : ""}`);
  io.out(`Approval:    timeout=${approvalTimeout(cfg)}s${cfg.approval?.chain ? ` chain=${cfg.approval.chain.join(",")}` : ""}`);
  io.out(`Taint:       ${w.calls} calls / ${w.minutes} min after a suspected injection`);
  io.out(`Kill hotkey: ${cfg.kill_hotkey ?? "(none)"}`);
  io.out("Rules (first match wins):");
  if (!cfg.rules.length) io.out("  (none)");
  cfg.rules.forEach((r, i) => {
    const cond = (["tool", "action", "app", "peer"] as const).filter((k) => r[k]).map((k) => `${k}=${r[k]}`).join(" ");
    io.out(`  [${i}] ${cond} → ${r.decision}${r.reason ? `  — ${r.reason}` : ""}`);
  });
}
