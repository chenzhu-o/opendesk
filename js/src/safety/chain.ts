/**
 * Hash chaining for JSONL audit logs — byte-compatible with
 * python/opendesk/safety/chain.py, so either SDK can verify the other's logs.
 *
 * Each chained line carries `prev` (previous line's `hash`, 64 zeros for the
 * first) and `hash` = sha256(canonical JSON of the entry without `hash`).
 * Canonical JSON: sorted keys, no whitespace, no ASCII escaping.  Lines
 * without `hash` are legacy and only accepted before the chain starts.
 */

import crypto from "crypto";
import fs from "fs";
import path from "path";

export const GENESIS = "0".repeat(64);

export function canonicalJson(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "number") return Number.isFinite(value) ? JSON.stringify(value) : "null";
  if (typeof value === "boolean" || typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((v) => (v === undefined ? "null" : canonicalJson(v))).join(",")}]`;
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).filter((k) => obj[k] !== undefined).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(obj[k])}`).join(",")}}`;
  }
  return JSON.stringify(String(value));
}

export function entryHash(entry: Record<string, unknown>): string {
  const { hash: _h, ...body } = entry;
  void _h;
  return crypto.createHash("sha256").update(canonicalJson(body), "utf8").digest("hex");
}

export function seal<T extends Record<string, unknown>>(entry: T, prev: string): T & { prev: string; hash: string } {
  const { prev: _p, hash: _h, ...rest } = entry as Record<string, unknown>;
  void _p; void _h;
  const out = { ...rest, prev } as Record<string, unknown>;
  out["hash"] = entryHash(out);
  return out as T & { prev: string; hash: string };
}

export function lastHash(file: string): string {
  let text: string;
  try { text = fs.readFileSync(file, "utf8"); } catch { return GENESIS; }
  const lines = text.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const raw = lines[i].trim();
    if (!raw) continue;
    try {
      const obj = JSON.parse(raw);
      if (obj && typeof obj.hash === "string") return obj.hash;
    } catch { /* skip */ }
  }
  return GENESIS;
}

/** Append `entry` to `file` with chaining; returns the sealed entry. */
export function appendChained(file: string, entry: Record<string, unknown>): Record<string, unknown> {
  fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const sealed = seal(entry, lastHash(file));
  fs.appendFileSync(file, JSON.stringify(sealed) + "\n", { encoding: "utf8", mode: 0o600 });
  return sealed;
}

export interface VerifyResult {
  path: string;
  ok: boolean;
  chained: number;
  legacy: number;
  error?: string;
  line?: number;
}

export function describeVerify(r: VerifyResult): string {
  const name = path.basename(r.path);
  if (r.ok) return `OK    ${name}: ${r.chained} chained entries${r.legacy ? `, ${r.legacy} legacy unchained` : ""}`;
  return `FAIL  ${name}: line ${r.line}: ${r.error}`;
}

export function verifyFile(file: string): VerifyResult {
  const res: VerifyResult = { path: file, ok: true, chained: 0, legacy: 0 };
  let text: string;
  try { text = fs.readFileSync(file, "utf8"); } catch (e) { return { ...res, ok: false, error: String(e) }; }
  let prev = GENESIS;
  let started = false;
  const lines = text.split("\n");
  const fail = (line: number, error: string): VerifyResult => ({ ...res, ok: false, line, error });
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i].trim();
    if (!raw) continue;
    let obj: Record<string, unknown>;
    try { obj = JSON.parse(raw); } catch { return fail(i + 1, "line is not valid JSON"); }
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) return fail(i + 1, "line is not a JSON object");
    if (!("hash" in obj)) {
      if (started) return fail(i + 1, "unchained entry after the chain started (inserted line?)");
      res.legacy++;
      continue;
    }
    started = true;
    if (obj["prev"] !== prev) return fail(i + 1, "prev does not match the previous entry's hash (deleted or reordered line?)");
    if (entryHash(obj) !== obj["hash"]) return fail(i + 1, "hash mismatch (entry was modified)");
    prev = obj["hash"] as string;
    res.chained++;
  }
  return res;
}
