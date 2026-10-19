/**
 * BrowserTool — drive Chrome / Chromium / Edge / Brave over the Chrome
 * DevTools Protocol.  Mirrors Python's tools/browser.py.
 *
 * Local-only in v1: the MCP server never routes `browser` to a paired peer.
 * `launch` uses a dedicated profile under <home>/browser-profile — never the
 * user's everyday browser profile.
 */

import fs from "node:fs";
import path from "node:path";
import type { ChildProcess } from "node:child_process";
import { Tool, ToolContext, ToolResult, checkPermission } from "./base.js";
import { getSandbox } from "../computer/sandbox.js";
import { DEFAULT_HOME } from "../protocol/auth/identity.js";
import {
  CDPConnection,
  CDPError,
  CDPTarget,
  DEFAULT_CDP_PORT,
  DevToolsHTTP,
  findBrowser,
  launchBrowser,
  sleep,
} from "../computer/cdp.js";

export const PROFILE_MARKER = ".opendesk-profile";
const MAX_TEXT_CHARS = 8_000;
const MAX_AX_NODES = 400;
const MAX_EVAL_CHARS = 16_000;
const SKIP_ROLES = new Set(["none", "generic", "InlineTextBox", "LineBreak", "presentation", "ignored"]);
const AX_FLAGS = new Set(["focused", "checked", "selected", "expanded", "disabled", "required", "pressed"]);
const UNLABELLED_CONTAINERS = new Set(["paragraph", "group", "section", "list", "listitem", "div"]);

export const BROWSER_ACTIONS = [
  "launch", "attach", "tabs", "open", "navigate", "read", "click",
  "type", "wait_for", "screenshot", "eval", "close_tab",
] as const;

// ---------------------------------------------------------------------------
// In-page JavaScript (identical to the Python tool)
// ---------------------------------------------------------------------------

const FIND_BY_TEXT_JS = String.raw`
(function (needle) {
  const norm = (s) => (s || "").replace(/\s+/g, " ").trim().toLowerCase();
  const n = norm(needle);
  if (!n) return null;
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none";
  };
  const label = (el) => norm(
    el.innerText || el.value || el.getAttribute("aria-label") ||
    el.getAttribute("title") || el.getAttribute("placeholder") || el.alt || ""
  );
  const clickable = "a,button,input,select,textarea,summary,label,option," +
    "[role=button],[role=link],[role=menuitem],[role=tab],[role=checkbox]," +
    "[role=radio],[role=option],[role=switch],[onclick],[tabindex]";
  const cands = Array.from(document.querySelectorAll(clickable)).filter(visible);
  let hit = cands.find((el) => label(el) === n) || cands.find((el) => label(el).includes(n));
  if (!hit && document.body) {
    const all = Array.from(document.body.querySelectorAll("*"))
      .filter((el) => visible(el) && norm(el.innerText).includes(n));
    hit = all.find((el) => !all.some((o) => o !== el && el.contains(o))) || null;
  }
  return hit || null;
})
`;

const CLICK_PREP_FN = String.raw`
function () {
  const el = this.nodeType === 1 ? this : this.parentElement;
  if (!el) return { ok: false };
  el.scrollIntoView({ block: "center", inline: "center" });
  const r = el.getBoundingClientRect();
  const x = r.left + r.width / 2, y = r.top + r.height / 2;
  const top = (r.width > 0 && r.height > 0) ? document.elementFromPoint(x, y) : null;
  const hittable = !!top && (top === el || el.contains(top) || top.contains(el));
  const label = (el.innerText || el.value || el.getAttribute("aria-label") || "")
    .replace(/\s+/g, " ").trim().slice(0, 80);
  return { ok: true, x, y, hittable, tag: el.tagName.toLowerCase(), label };
}
`;

const JS_CLICK_FN = String.raw`
function () { const el = this.nodeType === 1 ? this : this.parentElement; el.click(); return true; }
`;

const FOCUS_FN = String.raw`
function (clear) {
  const el = this.nodeType === 1 ? this : this.parentElement;
  el.scrollIntoView({ block: "center" });
  el.focus();
  if (clear) {
    if ("value" in el) {
      el.value = "";
      el.dispatchEvent(new Event("input", { bubbles: true }));
    } else if (el.isContentEditable) {
      document.execCommand("selectAll", false, null);
      document.execCommand("delete", false, null);
    }
  }
  return el.tagName.toLowerCase();
}
`;

const VALUE_FN = String.raw`
function () {
  const el = this.nodeType === 1 ? this : this.parentElement;
  return ("value" in el) ? String(el.value) : (el.innerText || "");
}
`;

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

export class BrowserSession {
  port = DEFAULT_CDP_PORT;
  host = "127.0.0.1";
  process?: ChildProcess;
  profileDir?: string;
  current?: string;
  conns = new Map<string, CDPConnection>();
  refs = new Map<string, Map<string, number>>();

  get http(): DevToolsHTTP {
    return new DevToolsHTTP(this.port, this.host);
  }

  async dropConnections(): Promise<void> {
    for (const c of this.conns.values()) await c.close().catch(() => undefined);
    this.conns.clear();
    this.refs.clear();
  }
}

const sessions = new Map<string, BrowserSession>();

export function getBrowserSession(sessionId: string): BrowserSession {
  if (!sessions.has(sessionId)) sessions.set(sessionId, new BrowserSession());
  return sessions.get(sessionId)!;
}

/** Close DevTools connections and (if we launched it) kill the browser. */
export async function closeBrowserSession(sessionId: string, kill = true): Promise<void> {
  const sess = sessions.get(sessionId);
  if (!sess) return;
  sessions.delete(sessionId);
  await sess.dropConnections();
  const proc = sess.process;
  if (kill && proc && proc.exitCode === null && proc.pid) {
    const exited = new Promise<void>((r) => proc.once("exit", () => r()));
    try {
      // Detached → own process group; kill the group so helpers go too.
      if (process.platform !== "win32") process.kill(-proc.pid, "SIGTERM");
      else proc.kill();
    } catch {
      proc.kill();
    }
    await Promise.race([exited, sleep(5000)]);
    if (proc.exitCode === null && proc.signalCode === null) {
      try {
        if (process.platform !== "win32") process.kill(-proc.pid, "SIGKILL");
        else proc.kill("SIGKILL");
      } catch {
        /* already gone */
      }
    }
  }
}

// ---------------------------------------------------------------------------
// AX rendering
// ---------------------------------------------------------------------------

interface AXNode {
  nodeId: string;
  ignored?: boolean;
  role?: { value?: unknown };
  name?: { value?: unknown };
  value?: { value?: unknown };
  childIds?: string[];
  backendDOMNodeId?: number;
  properties?: Array<{ name: string; value?: { value?: unknown } }>;
}

export function renderAx(nodes: AXNode[], maxNodes: number): { lines: string[]; refs: Map<string, number>; total: number } {
  const byId = new Map(nodes.map((n) => [n.nodeId, n]));
  const childIds = new Set(nodes.flatMap((n) => n.childIds ?? []));
  let roots = nodes.filter((n) => !childIds.has(n.nodeId));
  if (!roots.length) roots = nodes.slice(0, 1);
  const lines: string[] = [];
  const refs = new Map<string, number>();
  let total = 0;

  const visit = (node: AXNode, depth: number, parentName: string): void => {
    const role = String(node.role?.value ?? "");
    const name = String(node.name?.value ?? "").trim();
    let shown = !node.ignored && !SKIP_ROLES.has(role);
    if (shown && role === "StaticText" && (!name || name === parentName)) shown = false;
    if (shown && !name && UNLABELLED_CONTAINERS.has(role)) shown = false;
    let nextDepth = depth;
    if (shown) {
      total += 1;
      if (lines.length < maxNodes) {
        let line = "  ".repeat(depth);
        if (node.backendDOMNodeId !== undefined) {
          const ref = `e${refs.size + 1}`;
          refs.set(ref, node.backendDOMNodeId);
          line += `[${ref}] `;
        }
        line += role;
        if (name) line += ` ${JSON.stringify(name.slice(0, 100))}`;
        const value = node.value?.value;
        if (value !== undefined && value !== null && value !== "") line += ` value=${JSON.stringify(String(value).slice(0, 80))}`;
        const flags: string[] = [];
        for (const prop of node.properties ?? []) {
          const v = prop.value?.value;
          if (AX_FLAGS.has(prop.name) && v !== undefined && v !== null && v !== false && v !== "false") {
            flags.push(v === true || v === "true" ? prop.name : `${prop.name}=${v}`);
          }
        }
        if (flags.length) line += ` (${flags.join(", ")})`;
        lines.push(line);
      }
      nextDepth = depth + 1;
    }
    for (const cid of node.childIds ?? []) {
      const child = byId.get(cid);
      if (child) visit(child, nextDepth, shown ? name : parentName);
    }
  };
  for (const r of roots) visit(r, 0, "");
  return { lines, refs, total };
}

function exceptionText(details: Record<string, unknown>): string {
  const exc = (details["exception"] ?? {}) as Record<string, unknown>;
  return String(exc["description"] ?? details["text"] ?? "exception");
}

// ---------------------------------------------------------------------------
// The tool
// ---------------------------------------------------------------------------

interface Params {
  action: string;
  url?: string;
  port?: number;
  browser?: string;
  headless?: boolean;
  profileDir?: string;
  tab?: number | string;
  ref?: string;
  selector?: string;
  text?: string;
  expression?: string;
  clear?: boolean;
  submit?: boolean;
  fullPage?: boolean;
  maxChars?: number;
  timeout?: number;
}

export interface BrowserToolOptions {
  /** Base directory for the default profile (default ~/.opendesk). */
  home?: string;
}

export class BrowserTool extends Tool {
  name = "browser";
  title = "Web Browser (DevTools)";
  annotations = { readOnlyHint: false, destructiveHint: true, openWorldHint: true };
  description =
    "Drive a Chromium-family browser (Chrome, Chromium, Edge, Brave) through the Chrome DevTools " +
    "Protocol — read the DOM and click elements directly instead of screenshots + mouse. Prefer this " +
    "over `mouse` for anything inside a web page; use `script` for non-visual work (files, CLIs, curl).\n\n" +
    "Actions:\n" +
    "  launch   — start a browser with remote debugging and a dedicated profile (<home>/browser-profile, " +
    "never your everyday profile). Options: browser, port, headless, url.\n" +
    "  attach   — connect to a browser already started with --remote-debugging-port.\n" +
    "  tabs     — list open tabs (index, id, title, url; * = current).\n" +
    "  open     — open url in a new tab and make it current.\n" +
    "  navigate — load url in the current (or given) tab and wait for load.\n" +
    "  read     — title, url, visible text and an accessibility snapshot where each node has a short ref like e12.\n" +
    "  click    — click by ref (from read), CSS selector, or visible text.\n" +
    "  type     — type text into ref/selector (or the focused element); clear=true empties it first, submit=true presses Enter.\n" +
    "  wait_for — wait until a selector or text appears (timeout seconds).\n" +
    "  screenshot — PNG of the tab (fullPage=true for the whole page).\n" +
    "  eval     — evaluate a JavaScript expression and return its JSON value.\n" +
    "  close_tab — close the current (or given) tab.\n\n" +
    "Typical flow: launch → navigate → read → click ref=eN → read.\n" +
    "Runs on this machine only (not routed to paired peers).";

  schema = {
    type: "object",
    required: ["action"],
    properties: {
      action: { type: "string", enum: [...BROWSER_ACTIONS], description: "Browser action. See tool description." },
      url: { type: "string", description: "URL for launch/open/navigate." },
      port: { type: "integer", description: `DevTools port for launch/attach (default ${DEFAULT_CDP_PORT}).` },
      browser: { type: "string", description: "launch: chrome, chromium, edge, brave, or a full binary path." },
      headless: { type: "boolean", description: "launch: run without a window." },
      profileDir: { type: "string", description: "launch: profile directory override (default <home>/browser-profile). Must be empty or opendesk-created." },
      tab: { type: ["integer", "string"], description: "Tab index (from tabs) or target id. Default: current tab." },
      ref: { type: "string", description: "Element ref from the last read, e.g. 'e12'." },
      selector: { type: "string", description: "CSS selector." },
      text: { type: "string", description: "click/wait_for: visible text to find. type: the text to type." },
      expression: { type: "string", description: "eval: JavaScript expression." },
      clear: { type: "boolean", description: "type: clear the field first." },
      submit: { type: "boolean", description: "type: press Enter afterwards." },
      fullPage: { type: "boolean", description: "screenshot: capture the full page." },
      maxChars: { type: "integer", description: `read: maximum characters of visible text (default ${MAX_TEXT_CHARS}).` },
      timeout: { type: "number", description: "Seconds to wait (wait_for, page loads, launch). Default 15." },
    },
  };

  private home?: string;

  constructor(opts: BrowserToolOptions = {}) {
    super();
    this.home = opts.home;
  }

  async execute(ctx: ToolContext, raw: Record<string, unknown>): Promise<ToolResult> {
    const p = normalizeParams(raw);
    if (!(BROWSER_ACTIONS as readonly string[]).includes(p.action)) {
      return this.err("Browser error", `Unknown action '${p.action}'. One of: ${BROWSER_ACTIONS.join(", ")}`);
    }
    const targetDesc = p.url ?? p.ref ?? p.selector ?? p.text ?? (p.expression ?? "").slice(0, 120);
    await checkPermission(
      ctx, "browser", `${p.action} ${targetDesc}`.trim(),
      `Browser ${p.action}` + (targetDesc ? `: ${targetDesc}` : ""),
    );

    const audit: Record<string, unknown> = { action: p.action };
    for (const k of ["url", "ref", "selector", "tab", "port"] as const) if (p[k] !== undefined) audit[k] = p[k];
    if (p.text !== undefined) audit["text"] = p.text.slice(0, 200);
    if (p.expression !== undefined) audit["expression"] = p.expression.slice(0, 500);

    const sess = getBrowserSession(ctx.sessionId);
    let result: ToolResult;
    try {
      result = await this.dispatch(sess, p);
    } catch (e) {
      result = this.err("Browser error", `${p.action} failed: ${(e as Error).message ?? e}`);
    }
    getSandbox(ctx.sessionId).recordAction(
      "browser", audit,
      result.error ? undefined : result.output.slice(0, 200),
      result.error ? result.output.slice(0, 200) : undefined,
    );
    return result;
  }

  private dispatch(sess: BrowserSession, p: Params): Promise<ToolResult> {
    switch (p.action) {
      case "launch": return this.doLaunch(sess, p);
      case "attach": return this.doAttach(sess, p);
      case "tabs": return this.doTabs(sess);
      case "open": return this.doOpen(sess, p);
      case "navigate": return this.doNavigate(sess, p);
      case "read": return this.doRead(sess, p);
      case "click": return this.doClick(sess, p);
      case "type": return this.doType(sess, p);
      case "wait_for": return this.doWaitFor(sess, p);
      case "screenshot": return this.doScreenshot(sess, p);
      case "eval": return this.doEval(sess, p);
      default: return this.doCloseTab(sess, p);
    }
  }

  // ---------------------------------------------------------------- helpers

  private async target(sess: BrowserSession, tab?: number | string): Promise<CDPTarget> {
    const targets = await sess.http.targets();
    if (!targets.length) throw new CDPError("The browser has no open tabs. Use action='open' with a url.");
    if (tab === undefined || tab === null || tab === "") {
      const cur = targets.find((t) => t.id === sess.current);
      if (cur) return cur;
      sess.current = targets[0].id;
      return targets[0];
    }
    const key = String(tab).trim();
    if (/^\d+$/.test(key)) {
      const idx = Number(key);
      if (idx >= targets.length) throw new CDPError(`No tab at index ${idx} (${targets.length} open). Use action='tabs'.`);
      return targets[idx];
    }
    const byId = targets.find((t) => t.id === key || t.id.startsWith(key));
    if (byId) return byId;
    const low = key.toLowerCase();
    const byText = targets.find((t) => (t.title ?? "").toLowerCase().includes(low) || (t.url ?? "").toLowerCase().includes(low));
    if (byText) return byText;
    throw new CDPError(`No tab matches '${tab}'. Use action='tabs'.`);
  }

  private async conn(sess: BrowserSession, target: CDPTarget): Promise<CDPConnection> {
    const existing = sess.conns.get(target.id);
    if (existing && !existing.closed) return existing;
    if (!target.webSocketDebuggerUrl) {
      throw new CDPError(`Tab ${target.id} has no webSocketDebuggerUrl (another DevTools client may own it).`);
    }
    const c = await new CDPConnection(target.webSocketDebuggerUrl).connect();
    sess.conns.set(target.id, c);
    await c.send("Page.enable", {}, 5000).catch(() => undefined);
    return c;
  }

  private async current(sess: BrowserSession, tab?: number | string): Promise<[CDPTarget, CDPConnection]> {
    const t = await this.target(sess, tab);
    sess.current = t.id;
    return [t, await this.conn(sess, t)];
  }

  private async evaluate(c: CDPConnection, expression: string, timeoutMs = 30_000): Promise<unknown> {
    const res = await c.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }, timeoutMs);
    if (res["exceptionDetails"]) throw new CDPError(exceptionText(res["exceptionDetails"] as Record<string, unknown>));
    return ((res["result"] ?? {}) as Record<string, unknown>)["value"];
  }

  private async waitLoaded(c: CDPConnection, timeoutS: number): Promise<boolean> {
    const deadline = Date.now() + timeoutS * 1000;
    while (Date.now() < deadline) {
      try {
        if ((await this.evaluate(c, "document.readyState", 5000)) === "complete") return true;
      } catch {
        /* navigating */
      }
      await sleep(200);
    }
    return false;
  }

  private async titleUrl(c: CDPConnection): Promise<[string, string]> {
    try {
      const v = (await this.evaluate(c, "({t: document.title, u: location.href})", 5000)) as Record<string, unknown>;
      if (v && typeof v === "object") return [String(v["t"] || "(untitled)"), String(v["u"] ?? "")];
    } catch {
      /* fall through */
    }
    return ["(unknown)", ""];
  }

  private async resolveElement(
    c: CDPConnection, sess: BrowserSession, targetId: string, p: Params, allowText: boolean,
  ): Promise<[string | null, string]> {
    if (p.ref) {
      const ref = p.ref.trim().replace(/^\[/, "").replace(/\]$/, "");
      const backendNodeId = sess.refs.get(targetId)?.get(ref);
      if (backendNodeId === undefined) {
        throw new CDPError(`Unknown ref '${p.ref}' — refs come from the latest action='read' on this tab; call read again.`);
      }
      const res = await c.send("DOM.resolveNode", { backendNodeId });
      const oid = ((res["object"] ?? {}) as Record<string, unknown>)["objectId"] as string | undefined;
      if (!oid) throw new CDPError(`Element ${ref} is no longer in the page; call read again.`);
      return [oid, `ref ${ref}`];
    }
    if (p.selector) {
      const res = await c.send("Runtime.evaluate", { expression: `document.querySelector(${JSON.stringify(p.selector)})` });
      if (res["exceptionDetails"]) {
        throw new CDPError(`Invalid selector '${p.selector}': ${exceptionText(res["exceptionDetails"] as Record<string, unknown>)}`);
      }
      const obj = (res["result"] ?? {}) as Record<string, unknown>;
      if (obj["subtype"] === "null" || !obj["objectId"]) throw new CDPError(`No element matches selector '${p.selector}'.`);
      return [obj["objectId"] as string, `selector '${p.selector}'`];
    }
    if (allowText && p.text) {
      const res = await c.send("Runtime.evaluate", { expression: `${FIND_BY_TEXT_JS}(${JSON.stringify(p.text)})` });
      const obj = (res["result"] ?? {}) as Record<string, unknown>;
      if (res["exceptionDetails"] || obj["subtype"] === "null" || !obj["objectId"]) {
        throw new CDPError(`No visible element with text '${p.text}'.`);
      }
      return [obj["objectId"] as string, `text '${p.text}'`];
    }
    return [null, ""];
  }

  private defaultProfile(): string {
    return path.join(this.home ?? DEFAULT_HOME, "browser-profile");
  }

  // ---------------------------------------------------------------- actions

  private async doLaunch(sess: BrowserSession, p: Params): Promise<ToolResult> {
    const port = p.port ?? DEFAULT_CDP_PORT;
    const http = new DevToolsHTTP(port);
    let info: Record<string, unknown> | null = null;
    try {
      info = (await http.request("/json/version", "GET", 1000)) as Record<string, unknown>;
    } catch {
      info = null; // port is free — the normal case
    }
    if (info !== null) {
      const owned = sess.process && sess.process.exitCode === null && sess.port === port;
      if (owned) return this.ok("Browser already running", `Browser already launched on port ${port}: ${info["Browser"] ?? "?"}.`);
      throw new CDPError(
        `Port ${port} already has a DevTools endpoint (${info["Browser"] ?? "?"}). ` +
          "Use action='attach' to drive it, or launch with a different port.",
      );
    }
    const binary = findBrowser(p.browser);
    if (!binary) {
      throw new CDPError(
        `Could not find ${p.browser ?? "Chrome, Chromium, Edge or Brave"}. Install one, pass ` +
          "browser='/path/to/binary', or set $OPENDESK_BROWSER.",
      );
    }
    const profile = p.profileDir ?? this.defaultProfile();
    if (fs.existsSync(profile) && fs.readdirSync(profile).length && !fs.existsSync(path.join(profile, PROFILE_MARKER))) {
      throw new CDPError(
        `Refusing to use ${profile} as the browser profile: it is not empty and was not created by opendesk ` +
          "(it may be a real browser profile). Pick an empty directory.",
      );
    }
    fs.mkdirSync(profile, { recursive: true });
    fs.writeFileSync(path.join(profile, PROFILE_MARKER), "opendesk browser profile\n");

    await sess.dropConnections();
    const proc = launchBrowser(binary, { port, profileDir: profile, headless: p.headless, url: p.url ?? "about:blank" });
    sess.port = port;
    sess.process = proc;
    sess.profileDir = profile;
    sess.current = undefined;
    let ver: Record<string, unknown>;
    try {
      ver = await http.waitReady(Math.max(p.timeout ?? 15, 20));
    } catch (e) {
      try {
        proc.kill("SIGKILL");
      } catch {
        /* ignore */
      }
      sess.process = undefined;
      throw e;
    }
    let tabs: CDPTarget[] = [];
    const deadline = Date.now() + 5000;
    while (Date.now() < deadline) {
      tabs = await http.targets();
      if (tabs.length) break;
      await sleep(200);
    }
    if (tabs.length) {
      sess.current = tabs[0].id;
      if (p.url) await this.waitLoaded(await this.conn(sess, tabs[0]), p.timeout ?? 15);
    }
    return this.ok(
      "Browser launched",
      `Launched ${ver["Browser"] ?? path.basename(binary)} (${p.headless ? "headless" : "windowed"}) on DevTools port ${port}.\n` +
        `Profile: ${profile}\nPID: ${proc.pid}\nTabs: ${tabs.length}`,
      [],
      { port, pid: proc.pid, profileDir: profile },
    );
  }

  private async doAttach(sess: BrowserSession, p: Params): Promise<ToolResult> {
    const port = p.port ?? DEFAULT_CDP_PORT;
    const http = new DevToolsHTTP(port);
    const info = await http.version();
    await sess.dropConnections();
    if (sess.port !== port) sess.process = undefined;
    sess.port = port;
    sess.current = undefined;
    const tabs = await http.targets();
    if (tabs.length) sess.current = tabs[0].id;
    return this.ok("Browser attached", `Attached to ${info["Browser"] ?? "browser"} on port ${port}. ${tabs.length} tab(s) open.`, [], { port });
  }

  private async doTabs(sess: BrowserSession): Promise<ToolResult> {
    const tabs = await sess.http.targets();
    if (!tabs.length) return this.ok("Browser tabs", "No open tabs.");
    if (!tabs.some((t) => t.id === sess.current)) sess.current = tabs[0].id;
    const lines = [`${tabs.length} tab(s):`];
    tabs.forEach((t, i) => {
      const mark = t.id === sess.current ? "*" : " ";
      lines.push(` ${mark}[${i}] ${t.title || "(untitled)"} — ${t.url ?? ""}  (id ${t.id})`);
    });
    return this.ok("Browser tabs", lines.join("\n"), [], { tabs: tabs.map((t) => ({ id: t.id, title: t.title, url: t.url })) });
  }

  private async doOpen(sess: BrowserSession, p: Params): Promise<ToolResult> {
    const url = p.url ?? "about:blank";
    const target = await sess.http.newTab(url);
    if (!target || typeof target !== "object" || !target.id) throw new CDPError(`Browser did not return a new tab: ${JSON.stringify(target)}`);
    sess.current = target.id;
    const c = await this.conn(sess, target);
    const loaded = url === "about:blank" ? true : await this.waitLoaded(c, p.timeout ?? 15);
    const [title, cur] = await this.titleUrl(c);
    const note = loaded ? "" : ` (still loading after ${p.timeout ?? 15}s)`;
    return this.ok("Browser open", `Opened new tab${note}: ${title} — ${cur}`, [], { tab: target.id });
  }

  private async doNavigate(sess: BrowserSession, p: Params): Promise<ToolResult> {
    if (!p.url) throw new CDPError("url is required for action='navigate'.");
    const timeout = p.timeout ?? 15;
    const [target, c] = await this.current(sess, p.tab);
    c.events.length = 0;
    const res = await c.send("Page.navigate", { url: p.url }, (timeout + 5) * 1000);
    if (res["errorText"]) throw new CDPError(`Navigation to ${p.url} failed: ${res["errorText"]}`);
    sess.refs.delete(target.id);
    let loaded = true;
    if (res["loaderId"]) {
      const ev = await c.waitEvent("Page.loadEventFired", timeout * 1000);
      loaded = ev !== null || (await this.waitLoaded(c, 1));
    }
    const [title, cur] = await this.titleUrl(c);
    const note = loaded ? "" : ` (load event not seen within ${timeout}s)`;
    return this.ok("Browser navigate", `Navigated${note}: ${title} — ${cur}`);
  }

  private async doRead(sess: BrowserSession, p: Params): Promise<ToolResult> {
    const maxChars = p.maxChars ?? MAX_TEXT_CHARS;
    const [target, c] = await this.current(sess, p.tab);
    const page = ((await this.evaluate(
      c, "({title: document.title, url: location.href, text: document.body ? document.body.innerText : ''})",
    )) ?? {}) as Record<string, unknown>;
    const text = String(page["text"] ?? "");
    const cut = text.length > maxChars;
    await c.send("Accessibility.enable", {}, 5000).catch(() => undefined);
    const ax = await c.send("Accessibility.getFullAXTree", {}, 30_000);
    const { lines, refs, total } = renderAx((ax["nodes"] as AXNode[]) ?? [], MAX_AX_NODES);
    sess.refs.set(target.id, refs);
    const out = [
      `Title: ${page["title"] || "(untitled)"}`,
      `URL: ${page["url"] || target.url || ""}`,
      "",
      `--- visible text (${text.length.toLocaleString("en-US")} chars` +
        (cut ? `, showing first ${maxChars.toLocaleString("en-US")}; raise maxChars for more` : "") + ") ---",
      text.slice(0, maxChars).trim() || "(none)",
      "",
      `--- accessibility snapshot (${refs.size} refs` +
        (total > lines.length ? `; ${total - lines.length} more nodes omitted` : "") +
        "; use ref=eN with click/type) ---",
      ...lines,
    ];
    return this.ok(`Browser read: ${page["title"] ?? ""}`.trim(), out.join("\n"), [], {
      url: page["url"], title: page["title"], textTruncated: cut, refs: refs.size,
    });
  }

  private async doClick(sess: BrowserSession, p: Params): Promise<ToolResult> {
    const [target, c] = await this.current(sess, p.tab);
    const [oid, how] = await this.resolveElement(c, sess, target.id, p, true);
    if (!oid) throw new CDPError("click needs ref, selector, or text.");
    const prep = await c.send("Runtime.callFunctionOn", { objectId: oid, functionDeclaration: CLICK_PREP_FN, returnByValue: true });
    const info = (((prep["result"] ?? {}) as Record<string, unknown>)["value"] ?? {}) as Record<string, unknown>;
    if (!info["ok"]) throw new CDPError(`Element for ${how} cannot be clicked (detached node).`);
    let method = "mouse";
    if (info["hittable"]) {
      const x = Number(info["x"]);
      const y = Number(info["y"]);
      await c.send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
      for (const type of ["mousePressed", "mouseReleased"]) {
        await c.send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1 });
      }
    } else {
      method = "dom";
      await c.send("Runtime.callFunctionOn", { objectId: oid, functionDeclaration: JS_CLICK_FN, returnByValue: true });
    }
    return this.ok(
      "Browser click",
      `Clicked <${info["tag"] ?? "?"}> ${JSON.stringify(info["label"] ?? "")} via ${how} (${method} click). Call read to see the result.`,
      [], { how, method },
    );
  }

  private async doType(sess: BrowserSession, p: Params): Promise<ToolResult> {
    if (p.text === undefined) throw new CDPError("text is required for action='type'.");
    const [target, c] = await this.current(sess, p.tab);
    const [oid, how] = await this.resolveElement(c, sess, target.id, p, false);
    if (oid) {
      await c.send("Runtime.callFunctionOn", {
        objectId: oid, functionDeclaration: FOCUS_FN, arguments: [{ value: Boolean(p.clear) }], returnByValue: true,
      });
    }
    await c.send("Input.insertText", { text: p.text });
    if (p.submit) {
      for (const type of ["keyDown", "keyUp"]) {
        const ev: Record<string, unknown> = { type, key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13 };
        if (type === "keyDown") ev["text"] = "\r";
        await c.send("Input.dispatchKeyEvent", ev);
      }
    }
    let valueNote = "";
    if (oid && !p.submit) {
      try {
        const res = await c.send("Runtime.callFunctionOn", { objectId: oid, functionDeclaration: VALUE_FN, returnByValue: true });
        const v = ((res["result"] ?? {}) as Record<string, unknown>)["value"];
        if (typeof v === "string") valueNote = ` Field now contains ${JSON.stringify(v.slice(0, 120))}.`;
      } catch {
        /* ignore */
      }
    }
    return this.ok(
      "Browser type",
      `Typed ${p.text.length} char(s) into ${how || "the focused element"}${p.submit ? " and pressed Enter" : ""}.${valueNote}`,
    );
  }

  private async doWaitFor(sess: BrowserSession, p: Params): Promise<ToolResult> {
    if (!p.selector && !p.text) throw new CDPError("wait_for needs selector or text.");
    const timeout = p.timeout ?? 15;
    const [, c] = await this.current(sess, p.tab);
    const expr = p.selector
      ? `!!document.querySelector(${JSON.stringify(p.selector)})`
      : `!!(document.body && document.body.innerText.includes(${JSON.stringify(p.text)}))`;
    const what = p.selector ? `selector '${p.selector}'` : `text '${p.text}'`;
    const start = Date.now();
    for (;;) {
      try {
        if (await this.evaluate(c, expr, 5000)) {
          return this.ok("Browser wait_for", `Found ${what} after ${((Date.now() - start) / 1000).toFixed(1)}s.`);
        }
      } catch {
        /* page navigating */
      }
      if (Date.now() - start >= timeout * 1000) {
        return this.err("Browser wait_for timeout", `Timed out after ${timeout}s waiting for ${what}.`);
      }
      await sleep(250);
    }
  }

  private async doScreenshot(sess: BrowserSession, p: Params): Promise<ToolResult> {
    const [, c] = await this.current(sess, p.tab);
    const args: Record<string, unknown> = { format: "png" };
    if (p.fullPage) {
      const m = await c.send("Page.getLayoutMetrics");
      const size = (m["cssContentSize"] ?? m["contentSize"] ?? {}) as Record<string, number>;
      if (size["width"] && size["height"]) {
        args["clip"] = { x: 0, y: 0, width: size["width"], height: size["height"], scale: 1 };
        args["captureBeyondViewport"] = true;
      }
    }
    const res = await c.send("Page.captureScreenshot", args, Math.max(p.timeout ?? 15, 30) * 1000);
    const data = Buffer.from(String(res["data"] ?? ""), "base64");
    if (!data.length) throw new CDPError("Page.captureScreenshot returned no data.");
    const [title, url] = await this.titleUrl(c);
    return this.ok(
      "Browser screenshot",
      `Captured ${p.fullPage ? "full page" : "viewport"} of ${title} — ${url} (${data.length.toLocaleString("en-US")} bytes PNG).`,
      [{ filename: "browser.png", mediaType: "image/png", content: data }],
    );
  }

  private async doEval(sess: BrowserSession, p: Params): Promise<ToolResult> {
    if (!p.expression) throw new CDPError("expression is required for action='eval'.");
    const [, c] = await this.current(sess, p.tab);
    const res = await c.send(
      "Runtime.evaluate",
      { expression: p.expression, returnByValue: true, awaitPromise: true, userGesture: true },
      ((p.timeout ?? 15) + 5) * 1000,
    );
    if (res["exceptionDetails"]) {
      return this.err("Browser eval error", `JavaScript threw: ${exceptionText(res["exceptionDetails"] as Record<string, unknown>)}`);
    }
    const obj = (res["result"] ?? {}) as Record<string, unknown>;
    let text: string;
    if ("value" in obj) text = JSON.stringify(obj["value"], null, 2) ?? String(obj["value"]);
    else if (obj["type"] === "undefined") text = "undefined";
    else text = String(obj["description"] ?? JSON.stringify(obj));
    if (text.length > MAX_EVAL_CHARS) text = text.slice(0, MAX_EVAL_CHARS) + `\n… [truncated: ${text.length} chars total]`;
    return this.ok("Browser eval", text, [], { type: obj["type"] });
  }

  private async doCloseTab(sess: BrowserSession, p: Params): Promise<ToolResult> {
    const target = await this.target(sess, p.tab);
    const c = sess.conns.get(target.id);
    if (c) await c.close();
    sess.conns.delete(target.id);
    sess.refs.delete(target.id);
    await sess.http.close(target.id);
    if (sess.current === target.id) sess.current = undefined;
    try {
      const remaining = (await sess.http.targets()).filter((t) => t.id !== target.id);
      if (remaining.length && !sess.current) sess.current = remaining[0].id;
    } catch {
      /* ignore */
    }
    return this.ok("Browser close_tab", `Closed tab ${target.title || target.id} — ${target.url ?? ""}.`);
  }
}

function normalizeParams(raw: Record<string, unknown>): Params {
  const pick = (a: string, b?: string): unknown => raw[a] ?? (b ? raw[b] : undefined);
  const str = (v: unknown): string | undefined => (v === undefined || v === null ? undefined : String(v));
  const num = (v: unknown): number | undefined => (v === undefined || v === null || v === "" ? undefined : Number(v));
  return {
    action: String(raw["action"] ?? ""),
    url: str(raw["url"]),
    port: num(raw["port"]),
    browser: str(raw["browser"]),
    headless: Boolean(raw["headless"]),
    profileDir: str(pick("profileDir", "profile_dir")),
    tab: raw["tab"] as number | string | undefined,
    ref: str(raw["ref"]),
    selector: str(raw["selector"]),
    text: str(raw["text"]),
    expression: str(raw["expression"]),
    clear: Boolean(raw["clear"]),
    submit: Boolean(raw["submit"]),
    fullPage: Boolean(pick("fullPage", "full_page")),
    maxChars: num(pick("maxChars", "max_chars")),
    timeout: num(raw["timeout"]),
  };
}
