/**
 * Tamper-evident per-tool-call log: `<home>/audit/tools-YYYY-MM-DD.jsonl`.
 * Mirrors python/opendesk/safety/audit.py (same fields, same chain).
 */

import fs from "fs";
import path from "path";
import { appendChained, verifyFile, VerifyResult } from "./chain.js";
import { resolveHome } from "./home.js";

const SENSITIVE_KEY = /pass(word|wd|phrase)?|secret|token|api[_-]?key|credential|cookie|auth|private[_-]?key|otp/i;
const LONG_KEYS = new Set(["command", "script", "code", "source", "url", "query"]);

function pyRepr(s: string): string {
  // Close enough to Python's repr() for a log preview.
  return `'${s.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, "\\n")}'`;
}

function summValue(key: string, value: unknown, depth = 0): unknown {
  if (SENSITIVE_KEY.test(key)) return "[redacted]";
  if (Buffer.isBuffer(value)) return `<${value.length} bytes>`;
  if (typeof value === "string") {
    const limit = LONG_KEYS.has(key) ? 160 : 40;
    if (["image", "content", "data"].includes(key) && value.length > 200) return `<${value.length} chars>`;
    if (value.length > limit) return `${pyRepr(value.slice(0, limit))}… (${value.length} chars)`;
    return value;
  }
  if (Array.isArray(value)) {
    if (depth >= 2 || value.length > 20) return `<list, ${value.length} items>`;
    return value.map((v) => summValue(key, v, depth + 1));
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (depth >= 2) return `<object, ${entries.length} keys>`;
    return Object.fromEntries(entries.slice(0, 20).map(([k, v]) => [k, summValue(k, v, depth + 1)]));
  }
  return value;
}

export function summarizeParams(params: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(params ?? {}).map(([k, v]) => [k, summValue(k, v)]));
}

function today(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export interface ToolAuditRecord {
  tool: string;
  params: Record<string, unknown>;
  decision: string;
  outcome: string;
  session?: string;
  peer?: string | null;
  risk?: string | null;
  durationMs?: number;
  error?: string | null;
  extra?: Record<string, unknown>;
}

export class ToolAuditLog {
  readonly directory: string;

  constructor(home?: string) {
    this.directory = path.join(resolveHome(home), "audit");
  }

  path(date?: string): string {
    return path.join(this.directory, `tools-${date ?? today()}.jsonl`);
  }

  record(r: ToolAuditRecord): Record<string, unknown> {
    const action = typeof r.params?.["action"] === "string" ? r.params["action"] : null;
    const entry: Record<string, unknown> = {
      ts: Math.round(Date.now() * 1000) / 1_000_000,
      type: "tool",
      session: r.session ?? "",
      tool: r.tool,
      action,
      peer: r.peer || "local",
      params: summarizeParams(r.params),
      risk: r.risk ?? null,
      decision: r.decision,
      outcome: r.outcome,
    };
    if (r.durationMs !== undefined) entry["duration_ms"] = Math.round(r.durationMs);
    if (r.error) entry["error"] = r.error.slice(0, 300);
    if (r.extra) Object.assign(entry, r.extra);
    return appendChained(this.path(), entry);
  }

  iterEntries(date?: string): Array<Record<string, unknown>> {
    try {
      return fs.readFileSync(this.path(date), "utf8").split("\n").filter(Boolean).flatMap((l) => {
        try { return [JSON.parse(l)]; } catch { return []; }
      });
    } catch { return []; }
  }
}

/** All audit JSONL files (tool + server), optionally only for `date`. */
export function auditFiles(home?: string, date?: string): string[] {
  const dir = path.join(resolveHome(home), "audit");
  if (!fs.existsSync(dir)) return [];
  if (date) return [path.join(dir, `tools-${date}.jsonl`), path.join(dir, `${date}.jsonl`)].filter((p) => fs.existsSync(p));
  return fs.readdirSync(dir).filter((f) => f.endsWith(".jsonl")).sort().map((f) => path.join(dir, f));
}

export function verifyFiles(files: string[]): VerifyResult[] {
  return files.map(verifyFile);
}
