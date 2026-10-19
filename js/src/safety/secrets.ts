/**
 * Secrets: `{{secret:NAME}}` placeholders resolved at execution time.
 * Mirrors python/opendesk/safety/secrets.py.  Store: macOS Keychain via
 * `security` (service "opendesk", account NAME).  Other platforms get a clear
 * error (use the Python SDK for Linux secret-tool / Windows keyring).
 */

import { execFile } from "child_process";
import path from "path";
import { readJson, resolveHome, writeJson } from "./home.js";

export const SERVICE = "opendesk";
export const PLACEHOLDER_RE = /\{\{\s*secret:([A-Za-z0-9_.-]+)\s*\}\}/g;
const NAME_RE = /^[A-Za-z0-9_.-]{1,64}$/;

/** (tool, action) → param keys where placeholders are allowed. */
export const SECRET_FIELDS: Record<string, string[]> = {
  "keyboard:type": ["text"],
  "ui:type": ["text"],
  "browser:type": ["text"],
  "browser:fill": ["text", "value"],
};

export class SecretError extends Error {}

// ---------------------------------------------------------------------------
// Store (replaceable for tests)
// ---------------------------------------------------------------------------

interface RunResult { code: number; stdout: string }

function run(cmd: string, args: string[]): Promise<RunResult> {
  return new Promise((resolve) => {
    execFile(cmd, args, { timeout: 15_000 }, (err, stdout) => {
      const code = err ? (typeof (err as { code?: unknown }).code === "number" ? (err as { code: number }).code : 1) : 0;
      resolve({ code, stdout: String(stdout ?? "") });
    });
  });
}

function noStore(): SecretError {
  return new SecretError(
    `no supported secret store on ${process.platform} in the JS SDK (macOS Keychain only) — ` +
      "secrets are refused here; use the Python SDK (secret-tool / keyring) on Linux or Windows",
  );
}

export const store = {
  run,
  platform: process.platform as string,
  async get(name: string): Promise<string> {
    if (store.platform !== "darwin") throw noStore();
    const r = await store.run("security", ["find-generic-password", "-s", SERVICE, "-a", name, "-w"]);
    if (r.code !== 0) throw new SecretError(`secret '${name}' not found in the macOS Keychain`);
    return r.stdout.endsWith("\n") ? r.stdout.slice(0, -1) : r.stdout;
  },
  async set(name: string, value: string): Promise<void> {
    if (store.platform !== "darwin") throw noStore();
    const r = await store.run("security", ["add-generic-password", "-U", "-s", SERVICE, "-a", name, "-l", `opendesk secret ${name}`, "-w", value]);
    if (r.code !== 0) throw new SecretError(`could not store '${name}' in the macOS Keychain`);
  },
  async delete(name: string): Promise<void> {
    if (store.platform !== "darwin") throw noStore();
    await store.run("security", ["delete-generic-password", "-s", SERVICE, "-a", name]);
  },
};

// ---------------------------------------------------------------------------
// Index (names + settings, never values)
// ---------------------------------------------------------------------------

export interface SecretInfo {
  name: string;
  created?: number | null;
  confirm: boolean;
  allowed_apps: string[];
}

function indexPath(home?: string): string {
  return path.join(resolveHome(home), "secrets.json");
}

export function loadIndex(home?: string): Record<string, SecretInfo> {
  const data = readJson(indexPath(home));
  const secrets = (data?.["secrets"] ?? {}) as Record<string, Record<string, unknown>>;
  const out: Record<string, SecretInfo> = {};
  for (const [name, info] of Object.entries(secrets)) {
    out[name] = {
      name,
      created: (info?.["created"] as number) ?? null,
      confirm: info?.["confirm"] === undefined ? true : Boolean(info["confirm"]),
      allowed_apps: Array.isArray(info?.["allowed_apps"]) ? (info["allowed_apps"] as unknown[]).map(String) : [],
    };
  }
  return out;
}

function saveIndex(index: Record<string, SecretInfo>, home?: string): void {
  const secrets: Record<string, unknown> = {};
  for (const name of Object.keys(index).sort()) {
    const i = index[name];
    secrets[name] = { created: i.created ?? null, confirm: i.confirm, ...(i.allowed_apps.length ? { allowed_apps: i.allowed_apps } : {}) };
  }
  writeJson(indexPath(home), { secrets });
}

export async function setSecret(
  name: string, value: string, home?: string, opts: { allowedApps?: string[]; confirm?: boolean } = {},
): Promise<SecretInfo> {
  if (!NAME_RE.test(name)) throw new SecretError("secret names may only contain letters, digits, '_', '-', '.' (max 64)");
  if (!value) throw new SecretError("refusing to store an empty secret");
  await store.set(name, value);
  const index = loadIndex(home);
  const info: SecretInfo = { name, created: Date.now() / 1000, confirm: opts.confirm ?? true, allowed_apps: opts.allowedApps ?? [] };
  index[name] = info;
  saveIndex(index, home);
  return info;
}

export async function deleteSecret(name: string, home?: string): Promise<boolean> {
  const index = loadIndex(home);
  const existed = name in index;
  delete index[name];
  try { await store.delete(name); } catch (e) { if (!existed) throw e; }
  saveIndex(index, home);
  return existed;
}

export function listSecrets(home?: string): SecretInfo[] {
  return Object.values(loadIndex(home)).sort((a, b) => a.name.localeCompare(b.name));
}

// ---------------------------------------------------------------------------
// Placeholders
// ---------------------------------------------------------------------------

export function findPlaceholders(obj: unknown, p: string[] = []): Array<[string[], string]> {
  const out: Array<[string[], string]> = [];
  if (typeof obj === "string") {
    for (const m of obj.matchAll(PLACEHOLDER_RE)) out.push([p, m[1]]);
  } else if (Array.isArray(obj)) {
    obj.forEach((v, i) => out.push(...findPlaceholders(v, [...p, String(i)])));
  } else if (obj && typeof obj === "object" && !Buffer.isBuffer(obj)) {
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) out.push(...findPlaceholders(v, [...p, k]));
  }
  return out;
}

function allowedLocation(tool: string, params: Record<string, unknown>, p: string[]): boolean {
  if (tool === "fleet" && p.length >= 2 && ["params", "arguments", "args"].includes(p[0])) {
    const inner = params["tool"];
    const innerParams = params[p[0]];
    if (typeof inner === "string" && innerParams && typeof innerParams === "object") {
      return allowedLocation(inner, innerParams as Record<string, unknown>, p.slice(1));
    }
    return false;
  }
  const action = typeof params["action"] === "string" ? params["action"] : "";
  const fields = SECRET_FIELDS[`${tool}:${action}`];
  return !!fields && p.length === 1 && fields.includes(p[0]);
}

/** Validate placement; return referenced secret names (unique, ordered). */
export function secretNames(tool: string, params: Record<string, unknown>): string[] {
  const names: string[] = [];
  for (const [p, name] of findPlaceholders(params)) {
    if (!allowedLocation(tool, params, p)) {
      throw new SecretError(
        `secret placeholders are only allowed in typed text (keyboard/ui/browser type); found one in ${tool}.${p.join(".") || "params"}`,
      );
    }
    if (!names.includes(name)) names.push(name);
  }
  return names;
}

export function resolvePlaceholders<T>(obj: T, values: Record<string, string>): T {
  if (typeof obj === "string") {
    return obj.replace(PLACEHOLDER_RE, (m, n: string) => (n in values ? values[n] : m)) as unknown as T;
  }
  if (Array.isArray(obj)) return obj.map((v) => resolvePlaceholders(v, values)) as unknown as T;
  if (obj && typeof obj === "object" && !Buffer.isBuffer(obj)) {
    return Object.fromEntries(Object.entries(obj as Record<string, unknown>).map(([k, v]) => [k, resolvePlaceholders(v, values)])) as T;
  }
  return obj;
}

export async function fetchValues(names: string[], home?: string): Promise<Record<string, string>> {
  const index = loadIndex(home);
  const values: Record<string, string> = {};
  for (const n of names) {
    if (!(n in index)) throw new SecretError(`unknown secret '${n}' — ask the user to run \`opendesk-js secrets set ${n}\``);
    values[n] = await store.get(n);
  }
  return values;
}

export class Redactor {
  private pairs: Array<[string, string]>;

  constructor(values: Record<string, string>) {
    this.pairs = Object.entries(values)
      .filter(([, v]) => v)
      .map(([n, v]): [string, string] => [v, `{{secret:${n}}}`])
      .sort((a, b) => b[0].length - a[0].length);
  }

  get active(): boolean {
    return this.pairs.length > 0;
  }

  text(s: string): string {
    for (const [v, ph] of this.pairs) if (s.includes(v)) s = s.split(v).join(ph);
    return s;
  }

  any<T>(obj: T): T {
    if (!this.pairs.length) return obj;
    if (typeof obj === "string") return this.text(obj) as unknown as T;
    if (Array.isArray(obj)) return obj.map((v) => this.any(v)) as unknown as T;
    if (obj && typeof obj === "object" && !Buffer.isBuffer(obj)) {
      return Object.fromEntries(Object.entries(obj as Record<string, unknown>).map(([k, v]) => [k, this.any(v)])) as T;
    }
    return obj;
  }
}
