/**
 * Background capture loop for screen memory — mirrors Python's
 * opendesk/memory/recorder.py.
 *
 * Two triggers (config.trigger):
 *   - "events" (default): every poll_seconds, cheaply probe the frontmost
 *     app + window title; capture when it changes (debounced by
 *     min_interval_seconds), else fall back to one attempt every
 *     interval_seconds that is kept only if the screen changed.
 *   - "interval": one attempt every interval_seconds (v1 behaviour).
 *
 * Each capture attempt (tick):
 *   1. checks the shared pause flag (hotkey / tool / CLI),
 *   2. reads the frontmost app + window title and consults the deny list,
 *   3. grabs the screen,
 *   4. skips the frame when it is visually identical to the previous one
 *      (except right after an app switch),
 *   5. reads the window's accessibility text when available ("ax"), else
 *      runs OCR locally ("ocr"),
 *   6. redacts secrets, builds a JPEG thumbnail, stores both (encrypted as
 *      configured),
 *   7. periodically enforces retention and the storage cap.
 *
 * Nothing leaves the machine.
 */

import {
  MemoryConfig,
  clearDaemonState,
  clearPause,
  getPause,
  loadConfig,
  setPause,
  writeDaemonState,
} from "./config.js";
import { normalize } from "./crypto.js";
import { coerceFrontmost, frontmost as detectFrontmost, type Frontmost } from "./frontmost.js";
import { frameSignature, makeThumbnail, signatureDelta } from "./image.js";
import { loadMacosNative } from "./native.js";
import { availableBackend, createOcrEngine, ensureVisionHelper, type OcrEngine, type OcrPreference } from "./ocr.js";
import { redact } from "./redact.js";
import { MemoryStore } from "./store.js";

export type CaptureFn = () => Promise<Buffer>;
export type OcrFn = (png: Buffer) => Promise<string>;
export type FrontmostFn = () => Promise<Frontmost | unknown> | Frontmost | unknown;
/** Same contract as macosNative.windowText(pid?, maxChars). */
export type WindowTextFn = (pid?: number, maxChars?: number) => Promise<string | null | undefined> | string | null | undefined;

export const AX_MAX_CHARS = 20000;
const AX_TIMEOUT_MS = 5000;
const CONFIG_RELOAD_SECONDS = 10;
const HEARTBEAT_SECONDS = 10;

export interface RecorderOptions {
  home?: string;
  config?: MemoryConfig;
  store?: MemoryStore;
  /** Screen capture; defaults to screenshot-desktop. */
  capture?: CaptureFn;
  /** OCR; defaults to the best local engine (see ocr.ts). */
  ocr?: OcrFn;
  /** Frontmost app/title/pid probe; defaults to the platform probe. */
  frontmost?: FrontmostFn;
  /**
   * Accessibility text of the focused window.  Defaults to the native macOS
   * helper when capturing the real screen (no `capture` override).
   */
  windowText?: WindowTextFn | null;
  /** Monotonic clock in seconds (tests). */
  clock?: () => number;
  /** CLI override that survives config reloads. */
  intervalOverride?: number;
  log?: (msg: string) => void;
}

async function defaultCapture(): Promise<Buffer> {
  const mod = await import("screenshot-desktop");
  return mod.default({ format: "png" });
}

const monotonic = (): number => performance.now() / 1000;

/**
 * Comparison key for window titles: letters only.  Spinners (◐ ◑),
 * counters ("(3) Slack"), progress ("45%") and clocks change a title every
 * second without changing what is on screen; they must not count as a
 * window change.  The idle fallback still captures such windows.
 */
export function titleKey(title: string): string {
  return normalize(title ?? "").replace(/[^\p{L}\s]+/gu, " ").replace(/\s+/g, " ").trim();
}

export class ScreenMemoryRecorder {
  readonly home?: string;
  config: MemoryConfig;
  readonly store: MemoryStore;
  private capture: CaptureFn;
  private ocrFn?: OcrFn;
  private engine: OcrEngine | null = null;
  private engineTried = false;
  private probe: FrontmostFn;
  private windowTextFn: WindowTextFn | null | undefined;
  private local: boolean;
  private clock: () => number;
  private intervalOverride?: number;
  private log: (msg: string) => void;
  private lastSig: number[] | null = null;
  private ticks = 0;
  private _captured = 0;
  private _skipped = 0;
  private _redactions = 0;
  private sources: Record<string, number> = { ax: 0, ocr: 0, none: 0 };
  /** Captured frames per trigger reason. */
  readonly reasons: Record<string, number> = {};
  private _lastStatus = "idle";
  private lastReason = "";
  private lastSource = "";
  private lastKey: [string, string] | null = null;
  private lastAttempt: number | null = null;
  private lastReload: number;
  private lastHeartbeat = 0;
  private ocrWarned = false;
  private stopping = false;
  private wake: (() => void) | null = null;
  private hotkey: { stop(): void } | null = null;

  constructor(opts: RecorderOptions = {}) {
    this.home = opts.home;
    this.config = opts.config ?? loadConfig(opts.home);
    this.intervalOverride = opts.intervalOverride;
    if (this.intervalOverride) this.config.interval_seconds = this.intervalOverride;
    this.log = opts.log ?? ((m) => process.stderr.write(`[memory] ${m}\n`));
    this.store = opts.store ?? new MemoryStore(opts.home, { config: this.config, log: this.log });
    this.capture = opts.capture ?? defaultCapture;
    // Native probes only make sense for the real local screen.
    this.local = !opts.capture;
    this.ocrFn = opts.ocr;
    this.probe = opts.frontmost ?? detectFrontmost;
    this.windowTextFn = opts.windowText === undefined ? undefined : opts.windowText;
    this.clock = opts.clock ?? monotonic;
    this.lastReload = this.clock();
  }

  get captured(): number { return this._captured; }
  get skipped(): number { return this._skipped; }
  get lastStatus(): string { return this._lastStatus; }
  get redactions(): number { return this._redactions; }
  /** Frames captured this run, per text source. */
  get textSources(): Record<string, number> { return { ...this.sources }; }

  /** Pick up deny-list / interval / encryption edits made by other processes. */
  reloadConfig(): void {
    const fresh = loadConfig(this.home);
    if (this.intervalOverride) fresh.interval_seconds = this.intervalOverride;
    this.config = fresh;
    this.store.applyConfig(fresh);
    this.lastReload = this.clock();
  }

  // -- event trigger -------------------------------------------------------

  /**
   * Why a capture should happen now, or null: "start", "app_change",
   * "title_change" (debounced by min_interval_seconds), or "interval".
   */
  triggerReason(front: Frontmost, now: number): string | null {
    if (this.lastAttempt === null || this.lastKey === null) return "start";
    const since = now - this.lastAttempt;
    if (front.app !== this.lastKey[0] || titleKey(front.title) !== this.lastKey[1]) {
      if (since < this.config.min_interval_seconds) return null;
      return front.app !== this.lastKey[0] ? "app_change" : "title_change";
    }
    if (since >= this.config.interval_seconds) return "interval";
    return null;
  }

  /** One step of the events trigger.  Returns the tick status, "paused", or null. */
  async poll(): Promise<string | null> {
    const now = this.clock();
    if (now - this.lastReload >= CONFIG_RELOAD_SECONDS) this.reloadConfig();
    if (getPause(this.home) !== null) {
      this._lastStatus = "paused";
      this.heartbeat();
      return "paused";
    }
    const front = await this.frontmost();
    const reason = this.triggerReason(front, now);
    if (reason === null) {
      this.heartbeat();
      return null;
    }
    return this.tick({ reason, front });
  }

  // -- one tick ------------------------------------------------------------

  async tick(opts: { reason?: string; front?: Frontmost } = {}): Promise<string> {
    const reason = opts.reason ?? "manual";
    this.ticks++;
    if (this.ticks % 10 === 0) this.reloadConfig();

    if (getPause(this.home) !== null) return this.done("paused");

    const front = opts.front ?? await this.frontmost();
    const app = front.app;
    let title = front.title;
    this.lastKey = [app, titleKey(title)];
    this.lastAttempt = this.clock();
    this.lastReason = reason;
    if (this.config.isDenied(app, title)) return this.done(`denied:${app || title}`);

    let png: Buffer;
    try {
      png = await this.capture();
    } catch (e) {
      this.log(`capture failed: ${e}`);
      return this.done(`error:capture:${e}`);
    }

    let sig: number[] | null = null;
    try {
      sig = await frameSignature(png);
    } catch (e) {
      this.log(`signature failed: ${e}`);
    }
    // An app switch is a new context even when few pixels moved (small window,
    // large display): keep it.  Everything else must actually change the screen.
    if (reason !== "app_change" && sig && this.lastSig && signatureDelta(sig, this.lastSig) < this.config.dedupe_threshold) {
      return this.done("duplicate");
    }
    this.lastSig = sig;

    let { text, source } = await this.readText(png, front.pid ?? undefined);
    if (this.config.redact) {
      const a = redact(text);
      const b = redact(title);
      text = a.text;
      title = b.text;
      this._redactions += a.count + b.count;
    }

    let thumb: Buffer | null = null;
    let tw = 0;
    let th = 0;
    try {
      const t = await makeThumbnail(png, this.config.thumbnail_width, this.config.thumbnail_quality);
      thumb = t.jpeg; tw = t.width; th = t.height;
    } catch (e) {
      this.log(`thumbnail failed: ${e}`);
    }

    this.store.add({ ts: Date.now() / 1000, app, title, text, thumb, width: tw, height: th, text_source: source });
    this._captured++;
    this.sources[source] = (this.sources[source] ?? 0) + 1;
    this.reasons[reason] = (this.reasons[reason] ?? 0) + 1;
    this.lastSource = source;
    if (this._captured % 20 === 1) this.housekeep();
    return this.done("captured");
  }

  /** Accessibility text when there is enough of it, else OCR. */
  private async readText(png: Buffer, pid?: number): Promise<{ text: string; source: string }> {
    const ax = await this.axText(pid);
    if (ax && ax.trim().length >= Math.max(1, this.config.ax_min_chars)) {
      let text = ax.trim();
      if (String(this.config.ocr_backend).toLowerCase() === "always") {
        const extra = await this.runOcr(png);
        if (extra && !text.includes(extra)) text = `${text}\n${extra}`;
      }
      return { text, source: "ax" };
    }
    const text = await this.runOcr(png);
    return { text, source: text ? "ocr" : "none" };
  }

  private async runOcr(png: Buffer): Promise<string> {
    try {
      const t = (await this.recognize(png)) ?? "";
      return t === "(no text detected)" ? "" : t;
    } catch (e) {
      if (!this.ocrWarned) {
        this.log(`OCR failed — frames will be stored without text: ${e}`);
        this.ocrWarned = true;
      }
      return "";
    }
  }

  private async resolveWindowText(): Promise<WindowTextFn | null> {
    if (this.windowTextFn === undefined) {
      const native = this.local ? await loadMacosNative() : null;
      const fn = native?.windowText;
      this.windowTextFn = fn ? ((pid, max) => fn(pid, max) as ReturnType<WindowTextFn>) : null;
    }
    return this.windowTextFn ?? null;
  }

  private async axText(pid?: number): Promise<string | null> {
    const fn = await this.resolveWindowText();
    if (!fn) return null;
    let timer: NodeJS.Timeout | undefined;
    try {
      const result = await Promise.race([
        Promise.resolve().then(() => fn(pid, AX_MAX_CHARS)),
        new Promise<null>((resolve) => { timer = setTimeout(() => resolve(null), AX_TIMEOUT_MS); }),
      ]);
      return typeof result === "string" ? result : null;
    } catch {
      return null;
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  /** Apply retention + storage cap. */
  housekeep(): { expired: number; capped: number } {
    const expired = this.store.enforceRetention(this.config.retention_days);
    const capped = this.store.enforceCap(this.config.storage_cap_mb * 1024 * 1024);
    if (expired || capped) this.log(`housekeeping: removed ${expired} expired, ${capped} over-cap frames`);
    return { expired, capped };
  }

  private async recognize(png: Buffer): Promise<string> {
    if (this.ocrFn) return this.ocrFn(png);
    if (!this.engine && !this.engineTried) {
      this.engineTried = true;
      const pref = String(this.config.ocr_backend).toLowerCase() === "always" ? "auto" : this.config.ocr_backend;
      this.engine = await createOcrEngine(pref as OcrPreference, this.home);
      if (!this.engine) throw new Error("no OCR backend available");
    }
    if (!this.engine) throw new Error("no OCR backend available");
    return this.engine.recognize(png);
  }

  private async frontmost(): Promise<Frontmost> {
    try {
      const f = coerceFrontmost(await this.probe());
      return { app: f.app, title: f.title === f.app ? "" : f.title, pid: f.pid ?? null };
    } catch {
      return { app: "", title: "", pid: null };
    }
  }

  private stateFields(): Record<string, unknown> {
    return {
      status: this._lastStatus,
      captured: this._captured,
      skipped: this._skipped,
      interval: this.config.interval_seconds,
      trigger: this.config.trigger,
      poll_seconds: this.config.poll_seconds,
      min_interval_seconds: this.config.min_interval_seconds,
      last_reason: this.lastReason,
      last_text_source: this.lastSource,
      text_sources: { ...this.sources },
      reasons: { ...this.reasons },
      redactions: this._redactions,
    };
  }

  private done(status: string): string {
    this._lastStatus = status;
    if (status !== "captured") this._skipped++;
    writeDaemonState(this.home, this.stateFields());
    this.lastHeartbeat = monotonic();
    return status;
  }

  private heartbeat(): void {
    if (monotonic() - this.lastHeartbeat >= HEARTBEAT_SECONDS) {
      writeDaemonState(this.home, this.stateFields());
      this.lastHeartbeat = monotonic();
    }
  }

  // -- loop ----------------------------------------------------------------

  async run(): Promise<void> {
    this.stopping = false;
    this._lastStatus = "starting";
    writeDaemonState(this.home, this.stateFields());
    await this.startHotkey();
    try {
      while (!this.stopping) {
        const started = Date.now();
        const events = this.config.trigger !== "interval";
        try {
          if (events) await this.poll();
          else await this.tick({ reason: "interval" });
        } catch (e) {
          this.log(`tick failed: ${e}`);
          this.done(`error:${e}`);
        }
        const elapsed = (Date.now() - started) / 1000;
        const delay = events
          ? Math.max(0.2, this.config.poll_seconds - elapsed)
          : Math.max(1, this.config.interval_seconds - elapsed);
        if (this.stopping) break;
        await new Promise<void>((resolve) => {
          const t = setTimeout(() => { this.wake = null; resolve(); }, delay * 1000);
          this.wake = () => { clearTimeout(t); this.wake = null; resolve(); };
        });
      }
    } finally {
      this.stopHotkey();
      if (this.engine) await this.engine.close();
      clearDaemonState(this.home);
      this.store.close();
    }
  }

  stop(): void {
    this.stopping = true;
    this.wake?.();
  }

  // -- pause hotkey ----------------------------------------------------------

  /** Flip the pause flag.  Returns the new paused state. */
  togglePause(): boolean {
    if (getPause(this.home) !== null) {
      clearPause(this.home);
      console.log(`[${now()}] screen memory resumed`);
      return false;
    }
    setPause(this.home, { reason: "hotkey" });
    console.log(`[${now()}] screen memory paused — press the hotkey again to resume`);
    return true;
  }

  private async startHotkey(): Promise<void> {
    const combo = (this.config.pause_hotkey ?? "").trim();
    if (!combo) return;
    try {
      const { registerHotkey } = await import("./hotkey.js");
      this.hotkey = await registerHotkey(combo, () => this.togglePause());
      this.log(`pause hotkey armed: ${combo}`);
    } catch (e) {
      this.log(`pause hotkey unavailable (${(e as Error).message}). Use 'opendesk-js memory pause' instead.`);
      this.hotkey = null;
    }
  }

  private stopHotkey(): void {
    try { this.hotkey?.stop(); } catch { /* ignore */ }
    this.hotkey = null;
  }
}

// ---------------------------------------------------------------------------
// Daemon entry point
// ---------------------------------------------------------------------------

export async function startDaemon(opts: { home?: string; interval?: number } = {}): Promise<void> {
  const config = loadConfig(opts.home);
  const pref = (String(config.ocr_backend).toLowerCase() === "always" ? "auto" : config.ocr_backend) as OcrPreference;
  const backend = availableBackend(pref, opts.home);
  if (backend === "macos-vision") {
    console.log("Preparing the macOS Vision OCR helper (one-time compile, may take a minute)…");
    await ensureVisionHelper(opts.home);
  }
  if (!backend) {
    process.stderr.write(
      "WARNING: no OCR backend available — frames will be stored without text.\n" +
      "  tesseract.js is bundled; on macOS install Xcode command-line tools for the faster Vision engine.\n",
    );
  }

  const recorder = new ScreenMemoryRecorder({ home: opts.home, config, intervalOverride: opts.interval });
  const cfg = recorder.config;
  const native = await loadMacosNative();
  const trigger = cfg.trigger === "interval"
    ? `interval — every ${cfg.interval_seconds}s`
    : `events — app/window changes (poll ${cfg.poll_seconds}s, min ${cfg.min_interval_seconds}s apart), idle fallback every ${cfg.interval_seconds}s`;
  const enc = [cfg.encrypt_thumbnails ? "thumbnails" : "", cfg.encrypt_text ? "text" : ""].filter(Boolean);
  console.log("opendesk screen memory (js)");
  console.log(`  store:      ${recorder.store.dir}`);
  console.log(`  trigger:    ${trigger}`);
  console.log(`  text:       ${native?.windowText ? "accessibility text first, OCR fallback" : "OCR"}; redaction ${cfg.redact ? "on" : "OFF"}`);
  console.log(`  encryption: ${enc.join(", ") || "off"}`);
  console.log(`  cap:        ${cfg.storage_cap_mb} MB, retention ${cfg.retention_days} days`);
  console.log(`  deny list:  ${cfg.deny_apps.join(", ") || "(empty)"}`);
  console.log(`  OCR:        ${backend ?? "unavailable"}`);
  console.log(`  pause key:  ${cfg.pause_hotkey || "(disabled)"}`);
  if (getPause(opts.home) !== null) {
    console.log("  state:      PAUSED (resume with `opendesk-js memory resume` or the hotkey)");
  }
  console.log("Ctrl-C to stop.\n");

  const stop = () => recorder.stop();
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  await recorder.run();
  console.log(`\nStopped. ${recorder.captured} frames captured this run.`);
}

function now(): string {
  return new Date().toTimeString().slice(0, 8);
}
