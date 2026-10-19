/**
 * The safety guard — mirrors python/opendesk/safety/guard.py.
 *
 * control check → secret placement → classify/decide (+taint, +secrets) →
 * approval → secret resolution → execute → redaction → untrusted wrapping +
 * injection scan → chained audit.  OPENDESK_SAFETY=off makes it a pass-through.
 */

import { getSandbox } from "../computer/sandbox.js";
import { PermissionDeniedError, Tool, ToolContext, ToolResult } from "../tools/base.js";
import { Approver, requestApproval } from "./approval.js";
import { checkControl } from "./control.js";
import { resolveHome, safetyEnabled } from "./home.js";
import { scan, taint, taintState, untrustedSource, warningText, wrap } from "./injection.js";
import { approvalTimeout, decide, Decision, loadPolicy, taintWindow } from "./policy.js";
import { fetchValues, loadIndex, Redactor, resolvePlaceholders, SecretError, secretNames } from "./secrets.js";
import { summarizeParams, ToolAuditLog } from "./toolAudit.js";

export interface GuardOptions {
  /** opendesk home (else ctx.metadata.opendesk_home → $OPENDESK_HOME → ~/.opendesk). */
  home?: string;
  /** Peer the call targets ("local" or a peer name). */
  peer?: string | null;
  /** First approver in the chain (MCP elicitation). */
  approver?: Approver | null;
  /** Target app for rule matching when params has none. */
  app?: string | null;
}

export type Invoker = (params: Record<string, unknown>) => Promise<ToolResult>;

/** Replaceable hooks (tests swap the frontmost-app lookup). */
export const guardHooks = {
  async frontmostApp(): Promise<string | null> {
    try {
      const { frontmost } = await import("../memory/frontmost.js");
      return (await frontmost()).app || null;
    } catch {
      return null;
    }
  },
};

function refused(title: string, message: string, decision: string, extra: Record<string, unknown> = {}): ToolResult {
  return { title, output: `REFUSED: ${message}`, error: true, attachments: [], metadata: { safety: { decision, ...extra } } };
}

function summaryLine(tool: string, params: Record<string, unknown>): string {
  const s = summarizeParams(params);
  delete s["action"];
  const body = Object.entries(s).map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join(", ");
  return body ? `${tool}: ${body}` : tool;
}

function appMatches(patterns: string[], app?: string | null): boolean {
  if (!app) return false;
  const a = app.toLowerCase();
  return patterns.some((p) => a.includes(p.toLowerCase()) || p.toLowerCase().includes(a));
}

/** Run `tool` through the guard. */
export async function runTool(
  tool: Tool, ctx: ToolContext, params: Record<string, unknown>, opts: GuardOptions = {},
): Promise<ToolResult> {
  return guardedCall(tool.name, params, ctx, (p) => tool.execute(ctx, p), opts);
}

export async function guardedCall(
  toolName: string,
  rawParams: Record<string, unknown> | undefined,
  ctx: ToolContext | undefined,
  invoke: Invoker,
  opts: GuardOptions = {},
): Promise<ToolResult> {
  const params: Record<string, unknown> = { ...(rawParams ?? {}) };
  if (!safetyEnabled()) return invoke(params);

  const meta = ((ctx as unknown as { metadata?: Record<string, unknown> })?.metadata ?? {}) as Record<string, unknown>;
  const home = resolveHome(opts.home, ctx);
  const peer = opts.peer ?? (meta["peer"] as string | undefined) ?? null;
  const approver = opts.approver ?? (meta["approver"] as Approver | undefined) ?? null;
  const session = ctx?.sessionId ?? "default";
  const audit = new ToolAuditLog(home);
  const t0 = Date.now();
  const action = typeof params["action"] === "string" ? (params["action"] as string) : undefined;

  const doAudit = (decision: string, outcome: string, o: { risk?: string; error?: string; extra?: Record<string, unknown> } = {}) => {
    try {
      audit.record({ tool: toolName, params, decision, outcome, session, peer, risk: o.risk, durationMs: Date.now() - t0, error: o.error, extra: o.extra });
    } catch { /* audit must never break a call */ }
  };

  // 1. control
  let msg = checkControl(home);
  if (msg) {
    doAudit("refused", "refused", { error: "control" });
    return refused("Refused: control", msg, "refused", { reason: "control" });
  }

  // 2. secret placement
  let names: string[];
  try {
    names = secretNames(toolName, params);
  } catch (e) {
    doAudit("block", "blocked", { error: String((e as Error).message) });
    return refused("Refused: secret", (e as Error).message, "block");
  }

  // 3. decide
  const policy = loadPolicy(home);
  const taintReason = taintState({ consume: true });
  const index = names.length ? loadIndex(home) : {};
  let app = typeof params["app"] === "string" ? (params["app"] as string) : (opts.app ?? null);
  const needsApp = policy.rules.some((r) => r.app) || names.some((n) => index[n]?.allowed_apps.length);
  if (!app && needsApp && ["keyboard", "mouse", "browser", "fleet"].includes(toolName)) {
    app = await guardHooks.frontmostApp();
  }
  let decision: Decision = decide(toolName, params, { home, peer, app, tainted: !!taintReason, taintReason }, policy);
  const risk = decision.risk.level;

  if (names.length && decision.decision !== "block") {
    for (const n of names) {
      const info = index[n];
      if (!info) {
        const err = `unknown secret '${n}' — ask the user to run \`opendesk-js secrets set ${n}\``;
        doAudit("block", "blocked", { risk, error: err, extra: { secrets: names } });
        return refused("Refused: secret", err, "block");
      }
      if (info.allowed_apps.length && !appMatches(info.allowed_apps, app)) {
        const err = `secret '${n}' may only be typed into ${info.allowed_apps.join(", ")} (target app: ${app ?? "unknown"})`;
        doAudit("block", "blocked", { risk, error: err, extra: { secrets: names } });
        return refused("Refused: secret", err, "block");
      }
    }
    if (decision.decision === "allow" && names.some((n) => index[n].confirm)) {
      decision = { ...decision, decision: "confirm", reason: `uses stored secret(s): ${names.join(", ")}` };
    }
  }

  const extra: Record<string, unknown> = {};
  if (names.length) extra["secrets"] = names;
  if (taintReason) extra["tainted"] = true;

  if (decision.decision === "block") {
    doAudit("block", "blocked", { risk, error: decision.reason, extra });
    return refused("Refused: policy", `blocked by policy — ${decision.reason}`, "block", { risk, reason: decision.reason });
  }

  // 4. approval
  if (decision.decision === "confirm") {
    const outcome = await requestApproval(
      { tool: toolName, action, summary: summaryLine(toolName, params), reason: decision.reason, risk, peer, secrets: names, timeout: approvalTimeout(policy) },
      { mcpApprover: approver, chain: policy.approval?.chain ?? null },
    );
    if (!outcome.approved) {
      doAudit("confirm", "denied", { risk, error: outcome.message, extra: { ...extra, approval: outcome.via } });
      return refused(
        "Refused: not approved",
        `this action needs the user's approval (${decision.reason}) and was not approved: ${outcome.message}`,
        "denied", { risk, reason: decision.reason },
      );
    }
    extra["approval"] = outcome.via;
    msg = checkControl(home);
    if (msg) {
      doAudit("refused", "refused", { risk, error: "control", extra });
      return refused("Refused: control", msg, "refused", { reason: "control" });
    }
  }

  // 5. secrets
  let values: Record<string, string> = {};
  if (names.length) {
    try {
      values = await fetchValues(names, home);
    } catch (e) {
      const m = e instanceof SecretError ? e.message : String(e);
      doAudit(decision.decision, "error", { risk, error: m, extra });
      return { title: "Secret error", output: `ERROR: ${m}`, error: true, attachments: [], metadata: {} };
    }
  }
  const redactor = new Redactor(values);
  const execParams = names.length ? resolvePlaceholders(params, values) : params;

  // 6. execute
  const sandbox = getSandbox(session);
  const start = sandbox.size;
  let result: ToolResult;
  try {
    result = await invoke(execParams);
  } catch (e) {
    if (redactor.active) {
      sandbox.scrubFrom(start, (v) => redactor.any(v));
      const safe = redactor.text(e instanceof Error ? e.message : String(e));
      doAudit(decision.decision, "exception", { risk, error: safe, extra });
      if (e instanceof PermissionDeniedError) throw new PermissionDeniedError(safe);
      throw new Error(safe);
    }
    doAudit(decision.decision, "exception", { risk, error: e instanceof Error ? e.message : String(e), extra });
    throw e;
  }
  if (redactor.active) {
    sandbox.scrubFrom(start, (v) => redactor.any(v));
    result = { ...result, title: redactor.text(result.title ?? ""), output: redactor.text(result.output ?? ""), metadata: redactor.any(result.metadata ?? {}) };
  }

  // 7. untrusted content
  const src = untrustedSource(toolName, params);
  if (src && result.output && !result.error) {
    const findings = scan(result.output);
    let wrapped = wrap(result.output, src);
    if (findings.length) {
      const w = taintWindow(policy);
      taint(`injection-like text in ${src}`, { calls: w.calls, minutes: w.minutes });
      wrapped = warningText(findings, src) + "\n" + wrapped;
      extra["injection"] = [...new Set(findings.map((f) => f.pattern))].sort();
    }
    result = { ...result, output: wrapped };
  }

  // 8. audit
  doAudit(decision.decision, result.error ? "error" : "ok", { risk, error: result.error ? result.output.slice(0, 200) : undefined, extra });
  return { ...result, metadata: { ...(result.metadata ?? {}), safety: { decision: decision.decision, risk, ...extra } } };
}
