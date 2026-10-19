/**
 * Shared helpers for the safety layer — mirrors python/opendesk/safety/_home.py.
 */

import fs from "fs";
import os from "os";
import path from "path";

/**
 * Resolve the opendesk home directory.
 * Priority: explicit `home` → `ctx.metadata.opendesk_home` → $OPENDESK_HOME → ~/.opendesk.
 */
export function resolveHome(home?: string | null, ctx?: unknown): string {
  if (home) return path.resolve(expandUser(home));
  const meta = (ctx as { metadata?: Record<string, unknown> } | undefined)?.metadata;
  const metaHome = meta?.["opendesk_home"];
  if (typeof metaHome === "string" && metaHome) return path.resolve(expandUser(metaHome));
  const env = process.env["OPENDESK_HOME"];
  if (env) return path.resolve(expandUser(env));
  return path.join(os.homedir(), ".opendesk");
}

function expandUser(p: string): string {
  return p.startsWith("~") ? path.join(os.homedir(), p.slice(1)) : p;
}

export function readJson(file: string): Record<string, unknown> | null {
  try {
    const data = JSON.parse(fs.readFileSync(file, "utf8"));
    return data && typeof data === "object" && !Array.isArray(data) ? data : null;
  } catch {
    return null;
  }
}

/** Atomically write pretty JSON (mode 0600). */
export function writeJson(file: string, data: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const tmp = path.join(path.dirname(file), `.${path.basename(file)}.${process.pid}.${Date.now()}.tmp`);
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2) + "\n", { encoding: "utf8", mode: 0o600 });
  fs.renameSync(tmp, file);
}

/** False when OPENDESK_SAFETY is off / 0 / false / no / disabled. */
export function safetyEnabled(): boolean {
  const v = (process.env["OPENDESK_SAFETY"] ?? "").trim().toLowerCase();
  return !["off", "0", "false", "no", "disabled"].includes(v);
}
