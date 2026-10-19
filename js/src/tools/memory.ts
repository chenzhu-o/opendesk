/**
 * Screen-memory tools — recall what was on screen at any point in the past.
 *
 * Backed by the local screen-memory index in `~/.opendesk/memory` (see
 * src/memory/).  The index is populated by the background daemon
 * (`opendesk-js memory start`).  Two tools share this module:
 *
 *   - `MemoryTool` (`memory`) — read-only: search / show / timeline / status.
 *   - `MemoryManageTool` (`memory_manage`) — pause / resume / deny list /
 *     config / delete.
 *
 * Always local — screen memory never leaves the machine it was recorded on.
 */

import { Tool, ToolContext, ToolResult, checkPermission, type Attachment, type ToolAnnotations } from "./base.js";

const MAX_TEXT_CHARS = 6000;

const TIME_PHRASES =
  "Time phrases for since/until: '2h', '3d', 'yesterday', 'tuesday', " +
  "'last week', 'this month', 'yesterday afternoon', '2026-09-01'.";

export interface MemoryToolOptions {
  /** Override the opendesk home (default ~/.opendesk or $OPENDESK_HOME). */
  home?: string;
}

/** Shared plumbing for `MemoryTool` and `MemoryManageTool`. */
export abstract class MemoryToolBase extends Tool {
  private home?: string;

  constructor(opts: MemoryToolOptions = {}) {
    super();
    this.home = opts.home;
  }

  protected resolveHome(ctx: ToolContext): string | undefined {
    if (this.home) return this.home;
    const meta = (ctx as { metadata?: Record<string, unknown> }).metadata;
    const h = meta?.["opendeskHome"] ?? meta?.["opendesk_home"];
    return typeof h === "string" && h ? h : undefined;
  }

  /** Run `action`; return undefined for an action this tool doesn't own. */
  protected abstract dispatch(action: string, home: string | undefined, p: Params): Promise<ToolResult> | undefined;

  async execute(ctx: ToolContext, params: Record<string, unknown>): Promise<ToolResult> {
    const p = params as Params;
    const action = String(p.action ?? "");
    await checkPermission(ctx, this.name, action, `Screen memory: ${action}${p.query ? ` '${p.query}'` : ""}`);
    const home = this.resolveHome(ctx);

    try {
      const pending = this.dispatch(action, home, p);
      if (!pending) return this.err("Memory error", `Unknown action '${action}' for ${this.name}`);
      return await pending;
    } catch (e) {
      return this.err("Memory error", e instanceof Error ? e.message : String(e));
    }
  }

  // -- search / show / timeline ------------------------------------------------

  protected async search(home: string | undefined, p: Params): Promise<ToolResult> {
    const { MemoryStore } = await import("../memory/store.js");
    const { fmtRange, parseRange } = await import("../memory/timeparse.js");
    if (!p.query || !String(p.query).trim()) {
      throw new Error("query is required for action='search' (use action='timeline' to browse).");
    }
    const [start, end] = parseRange(p.since, p.until);
    const store = new MemoryStore(home);
    const frames = store.search(String(p.query), { start, end, app: p.app ?? null, limit: limitOf(p) });
    const hint = await this.emptyHint(store, home);
    let header = `Screen memory search '${p.query}' — ${fmtRange(start, end)}`;
    if (p.app) header += ` — app~'${p.app}'`;
    if (!frames.length) {
      return this.ok("Memory: no matches", `${header}\nNo matching moments.${hint}`, [], { count: 0 });
    }
    const lines = [header, `${frames.length} match(es), best first:`, ""];
    for (const f of frames) lines.push(frameLine(f, true));
    lines.push("", "Use memory(action='show', id=<id>) for the full text and thumbnail.");
    return this.ok(`Memory: ${frames.length} match(es)`, lines.join("\n"), [], { count: frames.length, ids: frames.map((f) => f.id) });
  }

  protected async timeline(home: string | undefined, p: Params): Promise<ToolResult> {
    const { MemoryStore } = await import("../memory/store.js");
    const { fmtRange, parseRange } = await import("../memory/timeparse.js");
    const [start, end] = parseRange(p.since, p.until);
    const store = new MemoryStore(home);
    const frames = store.timeline({ start, end, app: p.app ?? null, limit: limitOf(p) });
    const apps = store.apps({ start, end });
    const hint = await this.emptyHint(store, home);
    let header = `Screen memory timeline — ${fmtRange(start, end)}`;
    if (p.app) header += ` — app~'${p.app}'`;
    if (!frames.length) {
      return this.ok("Memory: nothing recorded", `${header}\nNo moments recorded in this window.${hint}`, [], { count: 0 });
    }
    const lines = [header, ""];
    if (apps.length && !p.app) {
      lines.push("Apps in window: " + apps.slice(0, 8).map(([a, n]) => `${a || "(unknown)"} ×${n}`).join(", "), "");
    }
    lines.push(`Most recent ${frames.length}:`);
    for (const f of frames) lines.push(frameLine(f, false));
    lines.push("", "Use memory(action='show', id=<id>) to read a moment in full.");
    return this.ok(`Memory timeline (${frames.length})`, lines.join("\n"), [], { count: frames.length, ids: frames.map((f) => f.id) });
  }

  protected async show(home: string | undefined, p: Params): Promise<ToolResult> {
    const { MemoryStore, frameWhen } = await import("../memory/store.js");
    if (p.id === undefined || p.id === null) throw new Error("id is required for action='show'.");
    const store = new MemoryStore(home);
    const frame = store.get(Number(p.id));
    if (!frame) return this.err("Memory: not found", `No moment with id ${p.id}.`);
    const includeImage = p.includeImage !== false;
    const thumb = includeImage ? store.thumbnail(frame) : null;

    let text = frame.text.trim() || "(no text was recognised in this frame)";
    let truncated = "";
    if (text.length > MAX_TEXT_CHARS) {
      text = text.slice(0, MAX_TEXT_CHARS);
      truncated = `\n… [truncated, ${frame.text.length} chars total]`;
    }
    const lines = [`Moment #${frame.id} — ${frameWhen(frame)}`, `App:    ${frame.app || "(unknown)"}`];
    if (frame.title) lines.push(`Window: ${frame.title}`);
    lines.push("", "Screen text:", text + truncated);
    const attachments: Attachment[] = thumb
      ? [{ filename: `memory-${frame.id}.jpg`, mediaType: "image/jpeg", content: thumb }]
      : [];
    return this.ok(`Memory #${frame.id} (${frame.app || "unknown"})`, lines.join("\n"), attachments, {
      id: frame.id, ts: frame.ts, when: frameWhen(frame), app: frame.app, title: frame.title, text: frame.text, thumb_path: frame.thumb_path,
    });
  }

  // -- status / control --------------------------------------------------------

  protected async status(home: string | undefined): Promise<ToolResult> {
    const { MemoryStore } = await import("../memory/store.js");
    const { daemonAlive, describePause, getPause, loadConfig, readDaemonState } = await import("../memory/config.js");
    const { fmtTs } = await import("../memory/timeparse.js");
    const cfg = loadConfig(home);
    const pause = getPause(home);
    const alive = daemonAlive(home);
    const state: Partial<import("../memory/config.js").DaemonState> = readDaemonState(home) ?? {};
    const store = new MemoryStore(home);
    const st = store.stats();
    const daemonLine = alive
      ? `running (pid ${state.pid}, last tick: ${state.status ?? "?"})`
      : "NOT running — start it with `opendesk-js memory start`";
    const span = st.frames && st.oldest !== null && st.newest !== null ? `  (${fmtTs(st.oldest)} → ${fmtTs(st.newest)})` : "";
    const lines = [
      "Screen memory status",
      `  daemon:    ${daemonLine}`,
      `  capture:   ${pause ? "PAUSED — " + describePause(pause) : "active"}`,
      `  store:     ${store.dir}`,
      `  frames:    ${st.frames}${span}`,
      `  size:      ${mb(st.totalBytes)} MB of ${cfg.storage_cap_mb} MB cap (thumbnails ${mb(st.thumbBytes)} MB, index ${mb(st.indexBytes)} MB)`,
      cfg.trigger === "interval"
        ? `  trigger:   interval — every ${cfg.interval_seconds}s`
        : `  trigger:   events — on app/window change (poll ${cfg.poll_seconds}s, ≥${cfg.min_interval_seconds}s apart), idle fallback ${cfg.interval_seconds}s`,
      `  retention: ${cfg.retention_days} days`,
      `  text:      ${Object.entries(st.textSources ?? {}).sort().map(([k, v]) => `${k} ×${v}`).join(", ")}${Object.keys(st.textSources ?? {}).length ? "   " : ""}redaction ${cfg.redact ? "on" : "off"}`,
      `  encrypt:   thumbnails ${cfg.encrypt_thumbnails ? "on" : "off"} (${st.encryptedThumbnails}/${st.thumbnails} stored encrypted); ` +
        `text ${cfg.encrypt_text ? "on" : "off"} (${st.encryptedTextFrames}/${st.frames}); key: ${st.keyStore ?? "not created yet"}`,
      ...(st.encryptionAvailable === false ? ["             WARNING: encryption unavailable — new frames are stored unencrypted"] : []),
      ...(state["redactions"] ? [`  redacted:  ${state["redactions"]} secret(s) this daemon run`] : []),
      `  deny list: ${cfg.deny_apps.length ? cfg.deny_apps.join(", ") : "(empty)"}`,
      `  hotkey:    ${cfg.pause_hotkey || "(disabled)"}`,
    ];
    if (st.apps.length) lines.push("  top apps:  " + st.apps.slice(0, 6).map(([a, n]) => `${a || "(unknown)"} ×${n}`).join(", "));
    return this.ok("Memory status", lines.join("\n"), [], {
      daemonAlive: alive, paused: pause !== null, frames: st.frames, bytes: st.totalBytes, config: cfg.toDict(), stats: st,
    });
  }

  protected async pause(home: string | undefined, p: Params): Promise<ToolResult> {
    const { describePause, parseDuration, setPause } = await import("../memory/config.js");
    const secs = p.duration ? parseDuration(String(p.duration)) : undefined;
    const state = setPause(home, { durationSeconds: secs, reason: "tool" });
    return this.ok("Memory paused", `Screen memory capture ${describePause(state)}.`);
  }

  protected async resume(home: string | undefined): Promise<ToolResult> {
    const { clearPause } = await import("../memory/config.js");
    const cleared = clearPause(home);
    return this.ok("Memory resumed", cleared ? "Screen memory capture resumed." : "Capture was not paused.");
  }

  protected async deny(home: string | undefined, p: Params): Promise<ToolResult> {
    const { loadConfig, saveConfig } = await import("../memory/config.js");
    const cfg = loadConfig(home);
    const msgs: string[] = [];
    if (p.denyAdd) msgs.push(cfg.denyAdd(String(p.denyAdd)) ? `Added '${p.denyAdd}'.` : `'${p.denyAdd}' already listed.`);
    if (p.denyRemove) msgs.push(cfg.denyRemove(String(p.denyRemove)) ? `Removed '${p.denyRemove}'.` : `'${p.denyRemove}' not found.`);
    if (p.denyAdd || p.denyRemove) saveConfig(cfg, home);
    const listing = cfg.deny_apps.map((d) => `  - ${d}`).join("\n") || "  (empty)";
    const out = (msgs.length ? msgs.join("\n") + "\n\n" : "") + "Deny list (apps / window titles never captured):\n" + listing;
    return this.ok("Memory deny list", out, [], { denyApps: cfg.deny_apps });
  }

  protected async configure(home: string | undefined, p: Params): Promise<ToolResult> {
    const { loadConfig, saveConfig } = await import("../memory/config.js");
    const cfg = loadConfig(home);
    const changed: string[] = [];
    if (p.intervalSeconds !== undefined) { cfg.interval_seconds = Number(p.intervalSeconds); changed.push(`interval_seconds=${cfg.interval_seconds}`); }
    if (p.storageCapMb !== undefined) { cfg.storage_cap_mb = Math.floor(Number(p.storageCapMb)); changed.push(`storage_cap_mb=${cfg.storage_cap_mb}`); }
    if (p.retentionDays !== undefined) { cfg.retention_days = Math.floor(Number(p.retentionDays)); changed.push(`retention_days=${cfg.retention_days}`); }
    if (p.trigger !== undefined) {
      if (p.trigger !== "events" && p.trigger !== "interval") throw new Error("trigger must be 'events' or 'interval'");
      cfg.trigger = p.trigger; changed.push(`trigger=${cfg.trigger}`);
    }
    let privacyChanged = false;
    const toggles: Array<[keyof Params, "redact" | "encrypt_text" | "encrypt_thumbnails"]> = [
      ["redact", "redact"], ["encryptText", "encrypt_text"], ["encryptThumbnails", "encrypt_thumbnails"],
    ];
    for (const [param, key] of toggles) {
      const v = p[param];
      if (v !== undefined && Boolean(cfg[key]) !== Boolean(v)) {
        cfg[key] = Boolean(v); changed.push(`${key}=${v ? "on" : "off"}`); privacyChanged = true;
      }
    }
    if (changed.length) saveConfig(cfg, home);
    const lines: string[] = [];
    if (changed.length) {
      lines.push("Updated: " + changed.join(", "), "(the running daemon picks this up within a few ticks)");
      if (privacyChanged) lines.push("Privacy settings apply to new frames. To convert existing frames, the user can run `opendesk-js memory reindex`.");
      lines.push("");
    }
    lines.push("Current config:");
    for (const [k, v] of Object.entries(cfg.toDict())) lines.push(`  ${k}: ${Array.isArray(v) ? JSON.stringify(v) : v}`);
    return this.ok("Memory config", lines.join("\n"), [], cfg.toDict() as unknown as Record<string, unknown>);
  }

  protected async remove(home: string | undefined, p: Params): Promise<ToolResult> {
    const { MemoryStore } = await import("../memory/store.js");
    const { fmtRange, parseRange } = await import("../memory/timeparse.js");
    if (!(p.since || p.until || p.app)) {
      throw new Error("Refusing to delete everything: pass since/until and/or app. Use the CLI `opendesk-js memory clear` to wipe the whole index.");
    }
    const [start, end] = parseRange(p.since, p.until);
    const scope = fmtRange(start, end) + (p.app ? `, app~'${p.app}'` : "");
    const store = new MemoryStore(home);
    const candidates = store.timeline({ start, end, app: p.app ?? null, limit: Infinity });
    if (!p.confirm) {
      return this.ok("Memory delete (dry run)", `${candidates.length} moment(s) would be deleted for ${scope}. Re-run with confirm=true to delete.`, [], { count: candidates.length });
    }
    const n = store.delete(candidates.map((f) => f.id));
    return this.ok("Memory deleted", `Deleted ${n} moment(s) for ${scope}.`, [], { count: n });
  }

  protected async emptyHint(store: { stats(): { frames: number } }, home?: string): Promise<string> {
    const { daemonAlive, getPause } = await import("../memory/config.js");
    if (store.stats().frames === 0) {
      if (!daemonAlive(home)) return "\nThe index is empty and the capture daemon is not running. Start it with `opendesk-js memory start`.";
      return "\nThe index is empty — the daemon has only just started.";
    }
    if (getPause(home) !== null) return "\n(Capture is currently paused.)";
    return "";
  }
}

// ---------------------------------------------------------------------------

interface Params {
  action?: string;
  query?: string;
  since?: string;
  until?: string;
  app?: string;
  limit?: number;
  id?: number;
  includeImage?: boolean;
  duration?: string;
  denyAdd?: string;
  denyRemove?: string;
  intervalSeconds?: number;
  storageCapMb?: number;
  retentionDays?: number;
  trigger?: "events" | "interval";
  redact?: boolean;
  encryptText?: boolean;
  encryptThumbnails?: boolean;
  confirm?: boolean;
}

function limitOf(p: Params): number {
  const n = Number(p.limit);
  if (!Number.isFinite(n) || n < 1) return 15;
  return Math.min(200, Math.floor(n));
}

function frameLine(f: import("../memory/store.js").Frame, snippet: boolean): string {
  const app = f.app || "(unknown)";
  const title = f.title && f.title !== f.app ? ` — ${f.title.slice(0, 70)}` : "";
  const when = new Date(f.ts * 1000);
  const pad = (n: number) => String(n).padStart(2, "0");
  const stamp = `${when.getFullYear()}-${pad(when.getMonth() + 1)}-${pad(when.getDate())} ${pad(when.getHours())}:${pad(when.getMinutes())}:${pad(when.getSeconds())}`;
  let line = `  #${String(f.id).padEnd(6)} ${stamp}  [${app}]${title}`;
  if (snippet) {
    const snip = (f.snippet || f.text.slice(0, 120)).replace(/\n/g, " ").trim();
    if (snip) line += `\n          ${snip.slice(0, 220)}`;
  }
  return line;
}

function mb(n: number): string {
  return (n / (1024 * 1024)).toFixed(1);
}

// ---------------------------------------------------------------------------
// Public tools
// ---------------------------------------------------------------------------

const TIME_WINDOW_PROPS = {
  since: { type: "string", description: "Start of the time window: '2h', 'yesterday', 'tuesday', 'last week', ISO date." },
  until: { type: "string", description: "End of the time window (same formats as since)." },
  app: { type: "string", description: "Only moments whose app name or window title contains this (case-insensitive)." },
};

/** Read-only recall over the local screen-memory history. */
export class MemoryTool extends MemoryToolBase {
  name = "memory";
  title = "Search Screen Memory";
  annotations: ToolAnnotations = { readOnlyHint: true, openWorldHint: false };
  description =
    "Recall what the user saw on screen in the past. A local background daemon " +
    "(`opendesk-js memory start`) captures the screen every ~30s, OCRs it, and " +
    "indexes the text with a thumbnail. Nothing is uploaded. Read-only.\n\n" +
    "Actions:\n" +
    "  search   — full-text search. query='error', since='tuesday', app='Terminal'. " +
    "Returns matching moments (id, time, app, window title, snippet).\n" +
    "  show     — fetch one moment by id: full OCR text + thumbnail image.\n" +
    "  timeline — list captured moments in a period, optionally filtered by app " +
    "(e.g. 'every time the dashboard was open this month').\n" +
    "  status   — is the daemon running/paused, how much is stored, config.\n\n" +
    TIME_PHRASES + "\n" +
    "Typical flow: search → pick an id → show to read the full text / image. " +
    "To pause capture, edit the deny list, change config or delete history, use `memory_manage`.";

  schema = {
    type: "object",
    required: ["action"],
    properties: {
      action: {
        type: "string",
        enum: ["search", "show", "timeline", "status"],
        description: "What to do. See tool description.",
      },
      query: { type: "string", description: "Search terms (action=search). All terms must appear; quote phrases; 'invoice*' for prefix." },
      ...TIME_WINDOW_PROPS,
      limit: { type: "integer", minimum: 1, maximum: 200, default: 15, description: "Max results for search/timeline." },
      id: { type: "integer", description: "Frame id (action=show)." },
      includeImage: { type: "boolean", default: true, description: "For action=show: attach the thumbnail image." },
    },
  };

  protected dispatch(action: string, home: string | undefined, p: Params): Promise<ToolResult> | undefined {
    switch (action) {
      case "search": return this.search(home, p);
      case "show": return this.show(home, p);
      case "timeline": return this.timeline(home, p);
      case "status": return this.status(home);
      default: return undefined;
    }
  }
}

/** State-changing controls for screen memory (pause, deny list, config, delete). */
export class MemoryManageTool extends MemoryToolBase {
  name = "memory_manage";
  title = "Manage Screen Memory";
  annotations: ToolAnnotations = { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: false };
  description =
    "Control the local screen-memory recorder (see the `memory` tool for searching it). " +
    "Changes state on this machine only.\n\n" +
    "Actions:\n" +
    "  pause / resume — stop or restart capture (pause accepts duration='1h').\n" +
    "  deny     — manage the per-app deny list (denyAdd / denyRemove); with neither, lists it.\n" +
    "  config   — change intervalSeconds, storageCapMb, retentionDays, trigger (events|interval), redact, encryptText, encryptThumbnails; with none, shows the config.\n" +
    "  delete   — remove stored moments for a time range and/or app. Dry run unless confirm=true.\n\n" +
    TIME_PHRASES;

  schema = {
    type: "object",
    required: ["action"],
    properties: {
      action: {
        type: "string",
        enum: ["pause", "resume", "deny", "config", "delete"],
        description: "What to do. See tool description.",
      },
      duration: { type: "string", description: "For action=pause: how long ('30m', '2h'). Omit to pause until resumed." },
      denyAdd: { type: "string", description: "For action=deny: app name / title substring to add." },
      denyRemove: { type: "string", description: "For action=deny: entry to remove." },
      intervalSeconds: { type: "number", minimum: 5, description: "For action=config: seconds between captures." },
      storageCapMb: { type: "integer", minimum: 16, description: "For action=config: storage ceiling in MB." },
      retentionDays: { type: "integer", minimum: 1, description: "For action=config: delete frames older than this." },
      trigger: { type: "string", enum: ["events", "interval"], description: "For action=config: capture on app/window changes ('events') or on a fixed timer ('interval')." },
      redact: { type: "boolean", description: "For action=config: scrub secrets from text before storing." },
      encryptText: { type: "boolean", description: "For action=config: store frame text encrypted (exact-word search only)." },
      encryptThumbnails: { type: "boolean", description: "For action=config: store thumbnails encrypted." },
      ...TIME_WINDOW_PROPS,
      confirm: { type: "boolean", default: false, description: "For action=delete: must be true to actually delete." },
    },
  };

  protected dispatch(action: string, home: string | undefined, p: Params): Promise<ToolResult> | undefined {
    switch (action) {
      case "pause": return this.pause(home, p);
      case "resume": return this.resume(home);
      case "deny": return this.deny(home, p);
      case "config": return this.configure(home, p);
      case "delete": return this.remove(home, p);
      default: return undefined;
    }
  }
}
