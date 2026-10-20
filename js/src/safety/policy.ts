/**
 * Policy engine — mirrors python/opendesk/safety/policy.py.
 *
 * Same `<home>/policy.json` format and semantics:
 *   { "preset": "relaxed" | "standard" | "strict",
 *     "rules": [{ "tool", "action"?, "app"?, "peer"?, "decision": "allow"|"confirm"|"block", "reason"? }],
 *     "kill_hotkey"?: string, "approval"?: { "timeout"?, "chain"? }, "taint"?: { "calls"?, "minutes"? } }
 */

import path from "path";
import { readJson, resolveHome, writeJson } from "./home.js";

export type RiskLevel = "low" | "medium" | "high";
export type DecisionKind = "allow" | "confirm" | "block";
export const PRESETS = ["relaxed", "standard", "strict"] as const;
export const DECISIONS: DecisionKind[] = ["allow", "confirm", "block"];

export interface Risk {
  level: RiskLevel;
  reason: string;
  tool: string;
  action?: string;
  readOnly: boolean;
}

export interface Rule {
  tool: string;
  action?: string;
  app?: string;
  peer?: string;
  decision: DecisionKind;
  reason?: string;
}

export interface Decision {
  decision: DecisionKind;
  reason: string;
  risk: Risk;
  rule?: Rule;
}

export interface PolicyConfig {
  preset: (typeof PRESETS)[number];
  rules: Rule[];
  kill_hotkey?: string;
  approval?: { timeout?: number; chain?: string[] };
  taint?: { calls?: number; minutes?: number };
  [extra: string]: unknown;
}

const L: RiskLevel = "low", M: RiskLevel = "medium", H: RiskLevel = "high";

const TABLE: Record<string, [Record<string, RiskLevel>, RiskLevel]> = {
  screenshot: [{}, L],
  ocr: [{}, L],
  ui: [{ get_tree: L, get_value: L, find: L }, M],
  mouse: [{ cursor_position: L }, M],
  keyboard: [{}, M],
  app: [{ list: L }, M],
  clipboard: [{ read: L }, M],
  audit: [{ show: L, verify: L }, M],
  schedule: [{ list: L }, M],
  learn: [{ list: L, show: L }, M],
  script: [{ list: L, show: L, shell: H, applescript: H, jxa: H, powershell: H, bash: H, python: H, exec: H }, H],
  browser: [{ read: L, tabs: L, screenshot: L, text: L, find: L, eval: H, evaluate: H, js: H }, M],
  memory: [{ search: L, show: L, timeline: L, status: L, recall: L, delete: H, clear: H }, M],
  memory_manage: [{ status: L, list: L, delete: H, clear: H, export: H }, M],
  fs: [{ read: L, list: L, stat: L, delete: H }, M],
  opendesk_peers: [{}, L],
  opendesk_discover: [{}, L],
  opendesk_status: [{}, L],
  opendesk_capabilities: [{}, L],
  opendesk_describe: [{}, L],
  opendesk_use: [{}, M],
  opendesk_disconnect: [{}, M],
};

const READ_VERBS = new Set(["list", "show", "status", "read", "get", "search", "describe", "info"]);
const DESCRIBE: Record<RiskLevel, string> = { low: "read-only", medium: "normal input", high: "high-risk" };

function actionOf(params?: Record<string, unknown>): string | undefined {
  const a = params?.["action"];
  return typeof a === "string" && a ? a : undefined;
}

function mk(level: RiskLevel, reason: string, tool: string, action?: string): Risk {
  return { level, reason, tool, action, readOnly: level === "low" };
}

export function fleetInner(params: Record<string, unknown>): [string | undefined, Record<string, unknown>] {
  const inner = params["tool"];
  const ip = (params["params"] ?? params["arguments"] ?? params["args"] ?? {}) as unknown;
  const innerParams = ip && typeof ip === "object" && !Array.isArray(ip) ? (ip as Record<string, unknown>) : {};
  return [typeof inner === "string" && inner ? inner : undefined, innerParams];
}

/** Built-in risk of calling `tool` with `params`. */
export function classify(tool: string, params: Record<string, unknown> = {}): Risk {
  const action = actionOf(params);
  if (tool === "fleet") {
    const [inner, innerParams] = fleetInner(params);
    if (inner) {
      const r = classify(inner, innerParams);
      return mk(r.level, `fleet → ${r.reason}`, tool, action ?? r.action);
    }
    const lvl = (action && READ_VERBS.has(action)) || (!action && !Object.keys(params).length) ? L : M;
    return mk(lvl, `fleet ${action ?? "call"}`, tool, action);
  }
  const entry = TABLE[tool];
  if (entry) {
    const [table, def] = entry;
    const lvl = action ? (table[action] ?? def) : def;
    const label = action ? `${tool} ${action}` : tool;
    return mk(lvl, `${label} is ${DESCRIBE[lvl]}`, tool, action);
  }
  if (tool.startsWith("opendesk_")) return mk(L, `${tool} is session admin`, tool, action);
  const lvl = action && READ_VERBS.has(action) ? L : M;
  return mk(lvl, `${tool} ${action ?? ""}`.trim() + " (unclassified tool)", tool, action);
}

// ---------------------------------------------------------------------------
// Policy file
// ---------------------------------------------------------------------------

export function policyPath(home?: string): string {
  return path.join(resolveHome(home), "policy.json");
}

export function defaultPolicy(): PolicyConfig {
  return { preset: "standard", rules: [] };
}

export function loadPolicy(home?: string): PolicyConfig {
  const data = readJson(policyPath(home));
  if (!data) return defaultPolicy();
  const preset = String(data["preset"] ?? "standard").toLowerCase();
  const rules = Array.isArray(data["rules"])
    ? (data["rules"] as unknown[]).filter((r): r is Rule => !!r && typeof r === "object")
    : [];
  return {
    ...data,
    preset: (PRESETS as readonly string[]).includes(preset) ? (preset as PolicyConfig["preset"]) : "standard",
    rules,
  } as PolicyConfig;
}

export function savePolicy(cfg: PolicyConfig, home?: string): string {
  const p = policyPath(home);
  writeJson(p, cfg);
  return p;
}

export function approvalTimeout(cfg: PolicyConfig): number {
  const t = Number(cfg.approval?.timeout ?? 60);
  return Number.isFinite(t) && t > 0 ? t : 60;
}

export function taintWindow(cfg: PolicyConfig): { calls: number; minutes: number } {
  const calls = Number(cfg.taint?.calls ?? 10);
  const minutes = Number(cfg.taint?.minutes ?? 10);
  return { calls: Number.isFinite(calls) ? calls : 10, minutes: Number.isFinite(minutes) ? minutes : 10 };
}

export function validateRule(rule: Partial<Rule> & Record<string, unknown>): Rule {
  if (!rule.tool) throw new Error("rule needs a 'tool' (use '*' for any tool)");
  const decision = String(rule.decision ?? "").toLowerCase() as DecisionKind;
  if (!DECISIONS.includes(decision)) throw new Error(`rule decision must be one of ${DECISIONS.join(", ")}`);
  // Key order matches the Python CLI: tool, action, app, peer, decision, reason.
  const out: Record<string, string> = { tool: String(rule.tool) };
  for (const k of ["action", "app", "peer"] as const) if (rule[k]) out[k] = String(rule[k]);
  out["decision"] = decision;
  if (rule.reason) out["reason"] = String(rule.reason);
  return out as unknown as Rule;
}

// ---------------------------------------------------------------------------
// Decisions
// ---------------------------------------------------------------------------

function globToRegex(p: string): RegExp {
  let re = "";
  for (const ch of p) {
    if (ch === "*") re += ".*";
    else if (ch === "?") re += ".";
    else re += ch.replace(/[.+^${}()|[\]\\]/g, "\\$&");
  }
  return new RegExp(`^${re}$`, "s");
}

function glob(pattern: string, value: string | undefined | null, substring = false): boolean {
  if (value == null) return false;
  const p = pattern.toLowerCase(), v = value.toLowerCase();
  if (globToRegex(p).test(v)) return true;
  return substring && !/[*?[]/.test(p) && v.includes(p);
}

export interface DecideContext {
  home?: string;
  app?: string | null;
  peer?: string | null;
  method?: string;
  action?: string;
  tainted?: boolean;
  taintReason?: string | null;
}

export function ruleMatches(rule: Rule, tool: string, params: Record<string, unknown>, ctx: DecideContext): boolean {
  const rt = String(rule.tool || "*");
  if (!(glob(rt, tool) || (ctx.method && glob(rt, ctx.method)))) return false;
  if (rule.action && !glob(rule.action, ctx.action ?? actionOf(params))) return false;
  if (rule.app) {
    const app = ctx.app ?? (typeof params["app"] === "string" ? (params["app"] as string) : undefined);
    if (!glob(rule.app, app ?? undefined, true)) return false;
  }
  if (rule.peer && !glob(rule.peer, ctx.peer || "local")) return false;
  return true;
}

function presetDecision(preset: string, risk: Risk): [DecisionKind, string] {
  if (preset === "relaxed") return ["allow", `relaxed preset: ${risk.reason}`];
  if (preset === "strict") {
    return risk.readOnly ? ["allow", `strict preset: ${risk.reason}`] : ["confirm", `strict preset: ${risk.reason} — needs approval`];
  }
  if (risk.level === "high") return ["confirm", `${risk.reason} — needs approval`];
  return ["allow", risk.reason];
}

function ruleLabel(rule: Rule): string {
  const parts = (["tool", "action", "app", "peer"] as const).filter((k) => rule[k]).map((k) => `${k}=${rule[k]}`);
  return `[${parts.join(" ")} → ${rule.decision}]`;
}

/** Decide allow / confirm / block for one call. */
export function decide(
  tool: string,
  params: Record<string, unknown> = {},
  ctx: DecideContext = {},
  policy?: PolicyConfig,
): Decision {
  const cfg = policy ?? loadPolicy(ctx.home);
  let risk = classify(tool, params);
  if (ctx.action && !risk.action) risk = { ...risk, action: ctx.action };
  let decision: Decision | undefined;
  for (const rule of cfg.rules) {
    if (!DECISIONS.includes(rule.decision)) continue;
    if (ruleMatches(rule, tool, params, ctx)) {
      decision = { decision: rule.decision, reason: rule.reason || `policy rule ${ruleLabel(rule)}`, risk, rule };
      break;
    }
  }
  if (!decision) {
    const [d, reason] = presetDecision(cfg.preset, risk);
    decision = { decision: d, reason, risk };
  }
  if (decision.decision === "allow" && ctx.tainted && !risk.readOnly) {
    const why = ctx.taintReason || "possible prompt injection seen on screen";
    decision = { ...decision, decision: "confirm", reason: `session is tainted (${why}); input actions need approval` };
  }
  return decision;
}
