/**
 * Minimal Chrome DevTools Protocol client — mirrors Python's computer/cdp.py.
 *
 *  - findBrowser / launchBrowser: locate Chrome, Chromium, Edge or Brave and
 *    start it with --remote-debugging-port and a dedicated profile dir.
 *  - DevToolsHTTP: the http://127.0.0.1:<port>/json/* endpoints.
 *  - CDPConnection: one WebSocket to one target, id-matched replies plus a
 *    buffer of recent events.
 */

import { spawn, ChildProcess } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import WebSocket from "ws";

export const DEFAULT_CDP_PORT = 9222;

export class CDPError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CDPError";
  }
}

export interface CDPTarget {
  id: string;
  type: string;
  title?: string;
  url?: string;
  webSocketDebuggerUrl?: string;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Discovery / launch
// ---------------------------------------------------------------------------

const MAC_APPS: Record<string, string[]> = {
  chrome: ["Google Chrome.app/Contents/MacOS/Google Chrome"],
  chromium: ["Chromium.app/Contents/MacOS/Chromium"],
  edge: ["Microsoft Edge.app/Contents/MacOS/Microsoft Edge"],
  brave: ["Brave Browser.app/Contents/MacOS/Brave Browser"],
};
const LINUX_BINS: Record<string, string[]> = {
  chrome: ["google-chrome", "google-chrome-stable"],
  chromium: ["chromium", "chromium-browser"],
  edge: ["microsoft-edge", "microsoft-edge-stable"],
  brave: ["brave-browser", "brave"],
};
const WIN_PATHS: Record<string, string[]> = {
  chrome: ["Google\\Chrome\\Application\\chrome.exe"],
  chromium: ["Chromium\\Application\\chrome.exe"],
  edge: ["Microsoft\\Edge\\Application\\msedge.exe"],
  brave: ["BraveSoftware\\Brave-Browser\\Application\\brave.exe"],
};
export const BROWSER_ORDER = ["chrome", "chromium", "edge", "brave"];

function which(name: string): string | null {
  for (const dir of (process.env.PATH ?? "").split(path.delimiter)) {
    if (!dir) continue;
    const p = path.join(dir, name);
    try {
      fs.accessSync(p, fs.constants.X_OK);
      return p;
    } catch {
      /* keep looking */
    }
  }
  return null;
}

/** Path of a Chromium-family browser, or null. `preferred` is a family name or a full path. */
export function findBrowser(preferred?: string): string | null {
  const want = preferred ?? process.env.OPENDESK_BROWSER ?? undefined;
  if (want && (want.includes("/") || want.includes("\\"))) return fs.existsSync(want) ? want : null;
  const families = want ? [want.toLowerCase()] : BROWSER_ORDER;
  for (const fam of families) {
    if (process.platform === "darwin") {
      for (const root of ["/Applications", path.join(os.homedir(), "Applications")]) {
        for (const rel of MAC_APPS[fam] ?? []) {
          const p = path.join(root, rel);
          if (fs.existsSync(p)) return p;
        }
      }
    } else if (process.platform === "win32") {
      const roots = [process.env["PROGRAMFILES"], process.env["PROGRAMFILES(X86)"], process.env["LOCALAPPDATA"]];
      for (const root of roots) {
        if (!root) continue;
        for (const rel of WIN_PATHS[fam] ?? []) {
          const p = path.join(root, rel);
          if (fs.existsSync(p)) return p;
        }
      }
    } else {
      for (const name of LINUX_BINS[fam] ?? []) {
        const p = which(name);
        if (p) return p;
      }
    }
  }
  return null;
}

export function launchBrowser(
  binary: string,
  opts: { port: number; profileDir: string; headless?: boolean; url?: string; extraArgs?: string[] },
): ChildProcess {
  fs.mkdirSync(opts.profileDir, { recursive: true });
  const args = [
    `--remote-debugging-port=${opts.port}`,
    `--user-data-dir=${opts.profileDir}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
  ];
  if (opts.headless) args.push("--headless=new", "--disable-gpu", "--window-size=1280,900");
  args.push(...(opts.extraArgs ?? []));
  args.push(opts.url ?? "about:blank");
  const child = spawn(binary, args, { stdio: "ignore", detached: true });
  child.on("error", () => {
    /* surfaced by waitReady timing out */
  });
  child.unref();
  return child;
}

// ---------------------------------------------------------------------------
// HTTP endpoints
// ---------------------------------------------------------------------------

export class DevToolsHTTP {
  constructor(public port: number = DEFAULT_CDP_PORT, public host = "127.0.0.1") {}

  get base(): string {
    return `http://${this.host}:${this.port}`;
  }

  async request(p: string, method = "GET", timeoutMs = 5000): Promise<unknown> {
    let res: Response;
    try {
      res = await fetch(this.base + p, { method, signal: AbortSignal.timeout(timeoutMs) });
    } catch (e) {
      throw new CDPError(
        `No DevTools endpoint at ${this.base} (${(e as Error).message}). Launch a browser with ` +
          `browser action='launch', or start one with --remote-debugging-port=${this.port}.`,
      );
    }
    const body = await res.text();
    if (!res.ok) throw new CDPError(`DevTools ${method} ${p} → HTTP ${res.status}`);
    try {
      return JSON.parse(body);
    } catch {
      return body;
    }
  }

  async version(): Promise<Record<string, unknown>> {
    return (await this.request("/json/version")) as Record<string, unknown>;
  }

  async targets(): Promise<CDPTarget[]> {
    const items = await this.request("/json/list");
    if (!Array.isArray(items)) throw new CDPError(`Unexpected /json/list payload: ${JSON.stringify(items)}`);
    return (items as CDPTarget[]).filter((t) => t.type === "page");
  }

  async newTab(url = "about:blank"): Promise<CDPTarget> {
    const p = "/json/new?" + encodeURI(url).replace(/#/g, "%23");
    try {
      return (await this.request(p, "PUT")) as CDPTarget;
    } catch {
      return (await this.request(p, "GET")) as CDPTarget;
    }
  }

  async activate(id: string): Promise<void> {
    await this.request(`/json/activate/${id}`);
  }

  async close(id: string): Promise<void> {
    await this.request(`/json/close/${id}`);
  }

  async waitReady(timeoutS = 20): Promise<Record<string, unknown>> {
    const deadline = Date.now() + timeoutS * 1000;
    let last: unknown;
    while (Date.now() < deadline) {
      try {
        return (await this.request("/json/version", "GET", 2000)) as Record<string, unknown>;
      } catch (e) {
        last = e;
        await sleep(250);
      }
    }
    throw new CDPError(`Browser did not open DevTools on port ${this.port} within ${timeoutS}s: ${last}`);
  }
}

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ---------------------------------------------------------------------------
// WebSocket connection
// ---------------------------------------------------------------------------

interface Pending {
  resolve: (msg: Record<string, unknown>) => void;
  reject: (e: Error) => void;
  timer: NodeJS.Timeout;
}

export interface CDPEvent {
  method: string;
  params?: Record<string, unknown>;
}

export class CDPConnection {
  private ws?: WebSocket;
  private nextId = 0;
  private pending = new Map<number, Pending>();
  events: CDPEvent[] = [];
  closed = false;

  constructor(public wsUrl: string) {}

  connect(timeoutMs = 10_000): Promise<this> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.wsUrl, { perMessageDeflate: false, maxPayload: 0 });
      const timer = setTimeout(() => {
        ws.terminate();
        reject(new CDPError(`Could not open DevTools WebSocket ${this.wsUrl}: timeout`));
      }, timeoutMs);
      ws.once("open", () => {
        clearTimeout(timer);
        this.ws = ws;
        resolve(this);
      });
      ws.once("error", (e) => {
        clearTimeout(timer);
        if (!this.ws) reject(new CDPError(`Could not open DevTools WebSocket ${this.wsUrl}: ${e.message}`));
      });
      ws.on("message", (data) => this.onMessage(data.toString()));
      ws.on("close", () => this.onClose());
    });
  }

  private onMessage(raw: string): void {
    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    const id = msg["id"] as number | undefined;
    if (id !== undefined && this.pending.has(id)) {
      const p = this.pending.get(id)!;
      this.pending.delete(id);
      clearTimeout(p.timer);
      p.resolve(msg);
    } else if (typeof msg["method"] === "string") {
      this.events.push(msg as unknown as CDPEvent);
      if (this.events.length > 200) this.events.shift();
    }
  }

  private onClose(): void {
    this.closed = true;
    for (const p of this.pending.values()) {
      clearTimeout(p.timer);
      p.reject(new CDPError("DevTools connection closed"));
    }
    this.pending.clear();
  }

  send(method: string, params: Record<string, unknown> = {}, timeoutMs = 30_000): Promise<Record<string, unknown>> {
    if (!this.ws || this.closed) return Promise.reject(new CDPError("DevTools connection is closed"));
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new CDPError(`${method} timed out after ${timeoutMs / 1000}s`));
      }, timeoutMs);
      this.pending.set(id, {
        timer,
        reject,
        resolve: (msg) => {
          if (msg["error"]) {
            const err = msg["error"] as { message?: string };
            reject(new CDPError(`${method}: ${err.message ?? JSON.stringify(err)}`));
          } else {
            resolve((msg["result"] as Record<string, unknown>) ?? {});
          }
        },
      });
      this.ws!.send(JSON.stringify({ id, method, params }));
    });
  }

  /** Wait for (and consume) the next buffered event named `method`; null on timeout. */
  async waitEvent(method: string, timeoutMs: number): Promise<CDPEvent | null> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const idx = this.events.findIndex((e) => e.method === method);
      if (idx >= 0) return this.events.splice(idx, 1)[0];
      if (this.closed || Date.now() >= deadline) return null;
      await sleep(50);
    }
  }

  async close(): Promise<void> {
    this.closed = true;
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        /* ignore */
      }
    }
  }
}
