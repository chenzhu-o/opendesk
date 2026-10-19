import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  MemoryConfig, loadConfig, saveConfig, setPause, clearPause, isPaused, getPause,
  parseDuration, writeDaemonState, daemonAlive,
} from "../src/memory/config.js";
import { parseWhen, parseRange } from "../src/memory/timeparse.js";
import { MemoryStore, queryTerms, makeSnippet } from "../src/memory/store.js";
import { makeThumbnail, frameSignature, signatureDelta } from "../src/memory/image.js";
import { ScreenMemoryRecorder } from "../src/memory/recorder.js";
import { renderLaunchdPlist, renderSystemdUnit, renderSchtasksCommand, LAUNCHD_LABEL } from "../src/memory/service.js";
import { parseCombo } from "../src/memory/hotkey.js";
import { MemoryTool, MemoryManageTool } from "../src/tools/memory.js";
import { createRegistry } from "../src/registry.js";
import type { ToolContext } from "../src/tools/base.js";
import { setKeyProviderResolver, StaticKeyProvider, ENV_KEYSTORE } from "../src/memory/crypto.js";

const TEST_KEY = Buffer.from(Array.from({ length: 32 }, (_, i) => i));

let home: string;
let keyProvider: StaticKeyProvider;
beforeEach(() => {
  home = fs.mkdtempSync(path.join(os.tmpdir(), "odmem-"));
  // Never touch the real Keychain / Secret Service from tests.
  keyProvider = new StaticKeyProvider(TEST_KEY);
  setKeyProviderResolver(() => keyProvider);
  process.env[ENV_KEYSTORE] = "file";
});
afterEach(() => {
  setKeyProviderResolver(null);
  delete process.env[ENV_KEYSTORE];
  fs.rmSync(home, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function png(color = 0xffffffff, size: [number, number] = [64, 48], seed = 0): Promise<Buffer> {
  const mod = (await import("jimp")) as unknown as typeof import("jimp") & { default?: typeof import("jimp") };
  const Jimp = mod.default ?? mod;
  const img = new Jimp(size[0], size[1], color);
  if (seed) {
    for (let y = 5; y < 25; y++) for (let x = seed % 40; x < seed % 40 + 20; x++) img.setPixelColor(0x000000ff, x, y);
  }
  return img.getBufferAsync(Jimp.MIME_PNG);
}

function ts(daysAgo: number, hour = 12): number {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  d.setHours(hour, 0, 0, 0);
  return d.getTime() / 1000;
}

function seeded(h: string): MemoryStore {
  const s = new MemoryStore(h);
  s.add({ ts: ts(3), app: "Terminal", title: "zsh", text: "Error: ECONNREFUSED 127.0.0.1:5432", thumb: Buffer.from("jpg1") });
  s.add({ ts: ts(2), app: "Google Chrome", title: "Grafana — Ops dashboard", text: "Ops dashboard p95 latency 120ms", thumb: Buffer.from("jpg2") });
  s.add({ ts: ts(1), app: "Preview", title: "invoice.pdf", text: "Invoice INV-2026-0042 total $1,234.00", thumb: Buffer.from("jpg3") });
  s.add({ ts: ts(0, 9), app: "Google Chrome", title: "Grafana — Ops dashboard", text: "Ops dashboard error rate 0.2%", thumb: Buffer.from("jpg4") });
  s.add({ ts: ts(0, 10), app: "Slack", title: "#general", text: "lunch at noon?", thumb: null });
  return s;
}

const ctx = (h: string): ToolContext => ({ sessionId: "mem-test", metadata: { opendeskHome: h } } as ToolContext);

// ---------------------------------------------------------------------------
// Config / pause
// ---------------------------------------------------------------------------

describe("MemoryConfig", () => {
  it("round-trips through config.json with python-compatible keys", () => {
    const cfg = loadConfig(home);
    expect(cfg.interval_seconds).toBe(30);
    expect(cfg.deny_apps).toContain("1Password");
    cfg.interval_seconds = 12;
    cfg.denyAdd("Signal");
    saveConfig(cfg, home);
    const raw = JSON.parse(fs.readFileSync(path.join(home, "memory", "config.json"), "utf8"));
    expect(raw.interval_seconds).toBe(12);
    expect(raw.storage_cap_mb).toBe(2048);
    expect(loadConfig(home).deny_apps).toContain("Signal");
  });

  it("ignores unknown keys and survives corrupt files", () => {
    fs.mkdirSync(path.join(home, "memory"), { recursive: true });
    fs.writeFileSync(path.join(home, "memory", "config.json"), '{"interval_seconds": 5, "bogus": 1}');
    expect(loadConfig(home).interval_seconds).toBe(5);
    fs.writeFileSync(path.join(home, "memory", "config.json"), "{not json");
    expect(loadConfig(home).interval_seconds).toBe(30);
  });

  it("deny matching is case-insensitive substring on app or title", () => {
    const cfg = new MemoryConfig({ deny_apps: ["1password", "bank"] });
    expect(cfg.isDenied("1Password 8")).toBe(true);
    expect(cfg.isDenied("Safari", "My Bank — Accounts")).toBe(true);
    expect(cfg.isDenied("Safari", "News")).toBe(false);
    expect(cfg.isDenied("", "")).toBe(false);
  });

  it("denyAdd / denyRemove", () => {
    const cfg = new MemoryConfig({ deny_apps: [] });
    expect(cfg.denyAdd("Signal")).toBe(true);
    expect(cfg.denyAdd("signal")).toBe(false);
    expect(cfg.denyRemove("SIGNAL")).toBe(true);
    expect(cfg.denyRemove("Signal")).toBe(false);
    expect(cfg.denyAdd("  ")).toBe(false);
  });

  it("pause until resumed and pause with expiry", async () => {
    expect(isPaused(home)).toBe(false);
    expect(setPause(home).until).toBeNull();
    expect(isPaused(home)).toBe(true);
    expect(clearPause(home)).toBe(true);
    expect(clearPause(home)).toBe(false);
    setPause(home, { durationSeconds: 0.05 });
    expect(isPaused(home)).toBe(true);
    await new Promise((r) => setTimeout(r, 100));
    expect(getPause(home)).toBeNull();
    expect(fs.existsSync(path.join(home, "memory", "paused.json"))).toBe(false);
  });

  it("parseDuration", () => {
    expect(parseDuration("30m")).toBe(1800);
    expect(parseDuration("2h")).toBe(7200);
    expect(parseDuration("1d")).toBe(86400);
    expect(parseDuration("90")).toBe(90);
    expect(parseDuration("2 hours")).toBe(7200);
    expect(() => parseDuration("soon")).toThrow();
  });

  it("daemonAlive needs a live pid and a fresh heartbeat", () => {
    expect(daemonAlive(home)).toBe(false);
    writeDaemonState(home, { status: "captured" });
    expect(daemonAlive(home)).toBe(true);
    expect(daemonAlive(home, 0)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Time parsing
// ---------------------------------------------------------------------------

describe("parseWhen", () => {
  const NOW = new Date(2026, 8, 4, 15, 30); // Friday 2026-09-04
  const local = (...a: number[]) => new Date(a[0], a[1] - 1, a[2], a[3] ?? 0, a[4] ?? 0).getTime() / 1000;

  it("relative durations are open-ended", () => {
    const [s, e] = parseWhen("2h", NOW);
    expect(e).toBeNull();
    expect(s).toBeCloseTo(NOW.getTime() / 1000 - 7200, 3);
  });
  it("yesterday is a whole day", () => {
    expect(parseWhen("yesterday", NOW)).toEqual([local(2026, 9, 3), local(2026, 9, 4)]);
  });
  it("weekday → most recent; last <weekday>", () => {
    expect(parseWhen("tuesday", NOW)).toEqual([local(2026, 9, 1), local(2026, 9, 2)]);
    expect(parseWhen("friday", NOW)[0]).toBe(local(2026, 9, 4));
    expect(parseWhen("last friday", NOW)[0]).toBe(local(2026, 8, 28));
  });
  it("day parts", () => {
    expect(parseWhen("tuesday afternoon", NOW)).toEqual([local(2026, 9, 1, 12), local(2026, 9, 1, 17)]);
    expect(parseWhen("yesterday morning", NOW)).toEqual([local(2026, 9, 3, 5), local(2026, 9, 3, 12)]);
  });
  it("calendar periods", () => {
    expect(parseWhen("last week", NOW)).toEqual([local(2026, 8, 24), local(2026, 8, 31)]);
    expect(parseWhen("this month", NOW)).toEqual([local(2026, 9, 1), null]);
    expect(parseWhen("last month", NOW)).toEqual([local(2026, 8, 1), local(2026, 9, 1)]);
  });
  it("ISO date and datetime", () => {
    expect(parseWhen("2026-09-01", NOW)).toEqual([local(2026, 9, 1), local(2026, 9, 2)]);
    expect(parseWhen("2026-09-01 14:30", NOW)).toEqual([local(2026, 9, 1, 14, 30), null]);
  });
  it("N days ago", () => {
    expect(parseWhen("3 days ago", NOW)[0]).toBe(local(2026, 9, 1));
  });
  it("garbage throws", () => {
    expect(() => parseWhen("whenever", NOW)).toThrow(/Cannot parse/);
  });
  it("parseRange combines and validates", () => {
    expect(parseRange("tuesday", "thursday", NOW)).toEqual([local(2026, 9, 1), local(2026, 9, 4)]);
    expect(parseRange(undefined, undefined, NOW)).toEqual([0, null]);
    expect(() => parseRange("today", "yesterday", NOW)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

describe("MemoryStore", () => {
  it("add writes a thumbnail and a jsonl record", () => {
    const s = new MemoryStore(home);
    const f = s.add({ ts: Date.now() / 1000, app: "A", title: "t", text: "hello world", thumb: Buffer.from("jpeg") });
    expect(f.id).toBe(1);
    expect(f.thumb_path!.endsWith(".jpg.enc")).toBe(true); // encrypted by default
    const onDisk = fs.readFileSync(path.join(s.dir, f.thumb_path!));
    expect(onDisk.subarray(0, 4).toString()).toBe("ODE1");
    expect(onDisk.includes(Buffer.from("jpeg"))).toBe(false);
    expect(s.get(1)?.text).toBe("hello world");
    expect(s.thumbnail(f)?.toString()).toBe("jpeg");
    expect(s.stats().frames).toBe(1);
    expect(s.add({ ts: Date.now() / 1000 }).id).toBe(2);
  });

  it("search finds terms with snippets, matches title/app, falls back to OR", () => {
    const s = seeded(home);
    let hits = s.search("ECONNREFUSED");
    expect(hits.map((h) => h.app)).toEqual(["Terminal"]);
    expect(hits[0].snippet).toContain("[ECONNREFUSED]");
    expect(s.search("invoice total")).toHaveLength(1);
    expect(s.search("Grafana")).toHaveLength(2);
    expect(s.search("slack")).toHaveLength(1);
    hits = s.search("invoice latency");
    expect(new Set(hits.map((h) => h.app))).toEqual(new Set(["Preview", "Google Chrome"]));
    expect(s.search("INV-2026*")).toHaveLength(1);
    expect(s.search('"$1,234.00"')).toHaveLength(1);
    expect(s.search("(((").length).toBe(s.timeline().length);
  });

  it("time and app filters", () => {
    const s = seeded(home);
    const [start, end] = parseWhen("today");
    expect(s.search("dashboard", { start, end })).toHaveLength(1);
    expect(s.search("dashboard", { app: "chrome" })).toHaveLength(2);
    expect(s.search("dashboard", { app: "Terminal" })).toHaveLength(0);
  });

  it("timeline and apps", () => {
    const s = seeded(home);
    expect(s.timeline({ limit: 2 }).map((f) => f.app)).toEqual(["Slack", "Google Chrome"]);
    expect(s.apps()[0]).toEqual(["Google Chrome", 2]);
    expect(s.timeline({ start: parseWhen("today")[0] })).toHaveLength(2);
    expect(s.timeline({ limit: 10, newestFirst: false })[0].app).toBe("Terminal");
  });

  it("delete removes thumbnails and records", () => {
    const s = seeded(home);
    const f = s.get(1)!;
    const p = path.join(s.dir, f.thumb_path!);
    expect(fs.existsSync(p)).toBe(true);
    expect(s.delete([1])).toBe(1);
    expect(fs.existsSync(p)).toBe(false);
    expect(s.get(1)).toBeNull();
    expect(s.search("ECONNREFUSED")).toHaveLength(0);
    const [start, end] = parseWhen("today");
    expect(s.deleteRange(start, end)).toBe(2);
    expect(s.deleteApp("chrome")).toBe(1);
    expect(s.stats().frames).toBe(1);
  });

  it("retention", () => {
    const s = new MemoryStore(home);
    const now = Date.now() / 1000;
    for (const d of [10, 5, 2.5, 1, 0]) s.add({ ts: now - d * 86400, app: "A", text: `${d}d`, thumb: Buffer.from("x") });
    expect(s.enforceRetention(0)).toBe(0);
    expect(s.enforceRetention(2)).toBe(3);
    expect(new Set(s.timeline().map((f) => f.text))).toEqual(new Set(["1d", "0d"]));
  });

  it("storage cap rolls oldest first and keeps the newest", () => {
    const s = new MemoryStore(home);
    const base = Date.now() / 1000 - 3600;
    for (let i = 0; i < 20; i++) s.add({ ts: base + i, app: "A", text: "x".repeat(10), thumb: Buffer.alloc(100_000, 1) });
    expect(s.stats().totalBytes).toBeGreaterThan(2_000_000);
    const deleted = s.enforceCap(1_000_000);
    expect(deleted).toBeGreaterThan(0);
    expect(deleted).toBeLessThan(20);
    const rem = s.timeline({ limit: 100, newestFirst: false });
    expect(rem[rem.length - 1].ts).toBe(base + 19);
    expect(rem[0].ts).toBeGreaterThan(base);
    expect(s.stats().totalBytes).toBeLessThanOrEqual(1_000_000);
    expect(s.enforceCap(1e12)).toBe(0);
  });

  it("cap never deletes the last frame", () => {
    const s = new MemoryStore(home);
    for (let i = 0; i < 3; i++) s.add({ ts: 1000 + i, app: "A", thumb: Buffer.alloc(100_000, 1) });
    expect(s.enforceCap(1)).toBe(2);
    expect(s.stats().frames).toBe(1);
    expect(s.get(3)).not.toBeNull();
  });

  it("clear and helpers", () => {
    const s = seeded(home);
    expect(s.clear()).toBe(5);
    expect(s.stats().frames).toBe(0);
    expect(fs.readdirSync(s.thumbsDir)).toHaveLength(0);
    expect(queryTerms('error "connection refused" port:5432')).toEqual(["error", "connection refused", "port:5432"]);
    expect(queryTerms("...")).toEqual([]);
    expect(makeSnippet("a b error c d", ["error"])).toContain("[error]");
  });
});

// ---------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------

describe("image helpers", () => {
  it("thumbnail downscales to JPEG", async () => {
    const t = await makeThumbnail(await png(0xffffffff, [1600, 900]), 400);
    expect([t.width, t.height]).toEqual([400, 225]);
    expect(t.jpeg.subarray(0, 3)).toEqual(Buffer.from([0xff, 0xd8, 0xff]));
  });
  it("signature delta separates similar from different frames", async () => {
    const a = await frameSignature(await png(0xffffffff));
    const b = await frameSignature(await png(0xfafafaff));
    const c = await frameSignature(await png(0x000000ff));
    expect(signatureDelta(a, b)).toBeLessThan(0.05);
    expect(signatureDelta(a, c)).toBeGreaterThan(0.9);
    expect(signatureDelta(null, a)).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

function recorder(h: string, frames: Array<{ png: Buffer; app: string; title: string }>, extra: Partial<ConstructorParameters<typeof ScreenMemoryRecorder>[0]> = {}) {
  const queue = [...frames];
  const cur = () => queue[0];
  return new ScreenMemoryRecorder({
    home: h,
    config: extra.config ?? new MemoryConfig({ deny_apps: ["1Password"] }),
    capture: async () => { const f = queue.length > 1 ? queue.shift()! : queue[0]; return f.png; },
    frontmost: async () => ({ app: cur().app, title: cur().title }),
    ocr: extra.ocr ?? (async (b) => `text for ${b.length} bytes`),
    log: () => {},
    ...extra,
  });
}

describe("ScreenMemoryRecorder", () => {
  it("captures, OCRs, thumbnails, and indexes", async () => {
    const rec = recorder(home, [{ png: await png(0xffffffff, [64, 48], 1), app: "Terminal", title: "zsh" }], { ocr: async () => "npm ERR! code ELIFECYCLE" });
    expect(await rec.tick()).toBe("captured");
    const s = new MemoryStore(home);
    const hits = s.search("ELIFECYCLE");
    expect(hits).toHaveLength(1);
    expect(hits[0].app).toBe("Terminal");
    expect(hits[0].title).toBe("zsh");
    expect(s.thumbnail(hits[0])!.subarray(0, 3)).toEqual(Buffer.from([0xff, 0xd8, 0xff]));
    expect(hits[0].width).toBe(64);
  });

  it("skips duplicate frames", async () => {
    const same = await png(0xffffffff, [64, 48], 3);
    const rec = recorder(home, [
      { png: same, app: "A", title: "" }, { png: same, app: "A", title: "" }, { png: await png(0x000000ff), app: "A", title: "" },
    ]);
    expect(await rec.tick()).toBe("captured");
    expect(await rec.tick()).toBe("duplicate");
    expect(await rec.tick()).toBe("captured");
    expect(rec.captured).toBe(2);
    expect(rec.skipped).toBe(1);
  });

  it("denied apps and titles are never captured", async () => {
    let captures = 0;
    const rec = recorder(home, [{ png: await png(), app: "1Password 8", title: "Vault" }], {
      capture: async () => { captures++; return png(); },
    });
    expect(await rec.tick()).toMatch(/^denied:/);
    expect(captures).toBe(0);
    const rec2 = recorder(home, [{ png: await png(), app: "Safari", title: "Acme Bank — Login" }], { config: new MemoryConfig({ deny_apps: ["bank"] }) });
    expect(await rec2.tick()).toMatch(/^denied:/);
  });

  it("pause flag skips capture; hotkey toggles it", async () => {
    const rec = recorder(home, [{ png: await png(), app: "A", title: "" }]);
    setPause(home);
    expect(await rec.tick()).toBe("paused");
    clearPause(home);
    expect(await rec.tick()).toBe("captured");
    expect(rec.togglePause()).toBe(true);
    expect(isPaused(home)).toBe(true);
    expect(rec.togglePause()).toBe(false);
    expect(isPaused(home)).toBe(false);
  });

  it("OCR failure stores the frame without text; capture errors are reported", async () => {
    const rec = recorder(home, [{ png: await png(), app: "A", title: "" }], { ocr: async () => { throw new Error("no engine"); } });
    expect(await rec.tick()).toBe("captured");
    expect(new MemoryStore(home).get(1)!.text).toBe("");
    const rec2 = recorder(home, [{ png: await png(), app: "A", title: "" }], { capture: async () => { throw new Error("no permission"); } });
    expect(await rec2.tick()).toMatch(/^error:capture:/);
  });

  it("reloads config so deny-list edits from another process apply", async () => {
    const cfg = new MemoryConfig({ deny_apps: [] });
    saveConfig(cfg, home);
    const frames = [];
    for (let i = 1; i < 30; i++) frames.push({ png: await png(0xffffffff, [64, 48], i), app: "Slack", title: "" });
    const rec = recorder(home, frames, { config: cfg });
    expect(await rec.tick()).toBe("captured");
    const cfg2 = loadConfig(home);
    cfg2.denyAdd("Slack");
    saveConfig(cfg2, home);
    let last = "";
    for (let i = 0; i < 10; i++) last = await rec.tick();
    expect(last).toMatch(/^denied:/);
  });

  it("interval override survives reload; housekeeping applies the cap", async () => {
    const cfg = new MemoryConfig({ storage_cap_mb: 1, retention_days: 365 });
    saveConfig(cfg, home);
    const rec = recorder(home, [{ png: await png(), app: "A", title: "" }], { intervalOverride: 7, config: cfg });
    expect(rec.config.interval_seconds).toBe(7);
    rec.reloadConfig();
    expect(rec.config.interval_seconds).toBe(7);
    expect(rec.config.storage_cap_mb).toBe(1);
    const now = Date.now() / 1000;
    for (let i = 0; i < 30; i++) rec.store.add({ ts: now - 30 + i, app: "A", thumb: Buffer.alloc(100_000, 1) });
    const r = rec.housekeep();
    expect(r.expired).toBe(0);
    expect(r.capped).toBeGreaterThan(0);
    expect(rec.store.stats().totalBytes).toBeLessThanOrEqual(1024 * 1024);
  });

  it("run loop stops cleanly and clears the heartbeat", async () => {
    const rec = recorder(home, [{ png: await png(0xffffffff, [64, 48], 1), app: "A", title: "" }, { png: await png(0xffffffff, [64, 48], 9), app: "A", title: "" }], {
      config: new MemoryConfig({ interval_seconds: 5, pause_hotkey: "" }),
    });
    const done = rec.run();
    await new Promise((r) => setTimeout(r, 400));
    expect(daemonAlive(home)).toBe(true);
    rec.stop();
    await done;
    expect(daemonAlive(home)).toBe(false);
    expect(rec.captured).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

describe("MemoryTool", () => {
  it("is registered with the expected schema", () => {
    const reg = createRegistry();
    const t = reg.get("memory");
    expect((t.schema as { properties: { action: { enum: string[] } } }).properties.action.enum).toContain("search");
  });

  it("splits read-only actions from management actions", async () => {
    const reg = createRegistry();
    const enumOf = (name: string) =>
      (reg.get(name).schema as { properties: { action: { enum: string[] } } }).properties.action.enum;
    expect(new Set(enumOf("memory"))).toEqual(new Set(["search", "show", "timeline", "status"]));
    expect(new Set(enumOf("memory_manage"))).toEqual(new Set(["pause", "resume", "deny", "config", "delete"]));
    expect(reg.get("memory").annotations?.readOnlyHint).toBe(true);
    expect(reg.get("memory_manage").annotations?.destructiveHint).toBe(true);
    const r = await new MemoryTool().execute(ctx(home), { action: "delete", app: "chrome", confirm: true });
    expect(r.error).toBe(true);
    const r2 = await new MemoryManageTool().execute(ctx(home), { action: "search", query: "x" });
    expect(r2.error).toBe(true);
  });

  it("search returns hits with ids and honours time phrases", async () => {
    seeded(home);
    const tool = new MemoryTool();
    let r = await tool.execute(ctx(home), { action: "search", query: "error", app: "Terminal" });
    expect(r.error).toBe(false);
    expect(r.output).toContain("ECONNREFUSED");
    expect(r.metadata["count"]).toBe(1);
    r = await tool.execute(ctx(home), { action: "search", query: "dashboard", since: "today" });
    expect(r.metadata["count"]).toBe(1);
    r = await tool.execute(ctx(home), { action: "search", query: "dashboard", since: "this week" });
    expect(r.metadata["count"]).toBeGreaterThanOrEqual(1);
  });

  it("validates input and hints when empty", async () => {
    const tool = new MemoryTool();
    let r = await tool.execute(ctx(home), { action: "search" });
    expect(r.error).toBe(true);
    r = await tool.execute(ctx(home), { action: "search", query: "x", since: "whenever" });
    expect(r.error).toBe(true);
    expect(r.output).toMatch(/Cannot parse/);
    r = await tool.execute(ctx(home), { action: "search", query: "anything" });
    expect(r.output).toContain("opendesk-js memory start");
  });

  it("show returns text and thumbnail attachment", async () => {
    seeded(home);
    const tool = new MemoryTool();
    let r = await tool.execute(ctx(home), { action: "show", id: 3 });
    expect(r.output).toContain("INV-2026-0042");
    expect(r.attachments[0].mediaType).toBe("image/jpeg");
    expect(r.attachments[0].content.toString()).toBe("jpg3");
    r = await tool.execute(ctx(home), { action: "show", id: 3, includeImage: false });
    expect(r.attachments).toHaveLength(0);
    r = await tool.execute(ctx(home), { action: "show", id: 999 });
    expect(r.error).toBe(true);
  });

  it("timeline, status, pause/resume, deny, config", async () => {
    seeded(home);
    const tool = new MemoryTool();
    const manage = new MemoryManageTool();
    let r = await tool.execute(ctx(home), { action: "timeline", since: "7d", app: "chrome" });
    expect(r.metadata["count"]).toBe(2);
    expect(r.output).toContain("Grafana");
    r = await tool.execute(ctx(home), { action: "status" });
    expect(r.output).toContain("frames:    5");
    expect(r.output).toContain("NOT running");
    r = await manage.execute(ctx(home), { action: "pause", duration: "1h" });
    expect(isPaused(home)).toBe(true);
    r = await tool.execute(ctx(home), { action: "status" });
    expect(r.output).toContain("PAUSED");
    await manage.execute(ctx(home), { action: "resume" });
    expect(isPaused(home)).toBe(false);
    r = await manage.execute(ctx(home), { action: "deny", denyAdd: "Signal" });
    expect(loadConfig(home).deny_apps).toContain("Signal");
    await manage.execute(ctx(home), { action: "deny", denyRemove: "Signal" });
    expect(loadConfig(home).deny_apps).not.toContain("Signal");
    r = await manage.execute(ctx(home), { action: "config", intervalSeconds: 15, storageCapMb: 512, retentionDays: 7 });
    const cfg = loadConfig(home);
    expect([cfg.interval_seconds, cfg.storage_cap_mb, cfg.retention_days]).toEqual([15, 512, 7]);
    expect(r.output).toContain("Updated");
  });

  it("delete requires scope and confirm", async () => {
    seeded(home);
    const tool = new MemoryManageTool();
    let r = await tool.execute(ctx(home), { action: "delete" });
    expect(r.error).toBe(true);
    r = await tool.execute(ctx(home), { action: "delete", app: "chrome" });
    expect(r.output).toContain("would be deleted");
    expect(new MemoryStore(home).stats().frames).toBe(5);
    r = await tool.execute(ctx(home), { action: "delete", app: "chrome", confirm: true });
    expect(r.metadata["count"]).toBe(2);
    expect(new MemoryStore(home).stats().frames).toBe(3);
  });

  it("consults the permission handler and accepts home via constructor", async () => {
    const { PermissionDeniedError } = await import("../src/tools/base.js");
    const denyCtx: ToolContext = { sessionId: "x", permissionHandler: async () => { throw new PermissionDeniedError("nope"); } };
    await expect(new MemoryTool({ home }).execute(denyCtx, { action: "status" })).rejects.toBeInstanceOf(PermissionDeniedError);
    seeded(home);
    const r = await new MemoryTool({ home }).execute({ sessionId: "y" }, { action: "status" });
    expect(r.output).toContain("frames:    5");
  });
});

// ---------------------------------------------------------------------------
// Service renderers + hotkey combo parsing
// ---------------------------------------------------------------------------

describe("service renderers", () => {
  it("launchd plist is well-formed and carries the args", () => {
    const plist = renderLaunchdPlist("/usr/local/bin/node", "/x/bin/opendesk.js", 20, "/tmp/h");
    expect(plist).toContain(`<string>${LAUNCHD_LABEL}</string>`);
    for (const s of ["memory", "start", "--interval", "20", "--home", "/tmp/h"]) expect(plist).toContain(`<string>${s}</string>`);
    expect(plist).toContain("/tmp/h/memory/daemon.log");
  });
  it("systemd unit", () => {
    const unit = renderSystemdUnit("/usr/bin/node", "/x/bin/opendesk.js");
    expect(unit).toContain("ExecStart=/usr/bin/node /x/bin/opendesk.js memory start\n");
    expect(unit).not.toContain("--home");
  });
  it("schtasks command", () => {
    const cmd = renderSchtasksCommand("C:\\node\\node.exe", "C:\\x\\opendesk.js", 45, "C:\\h");
    expect(cmd.startsWith('"C:\\node\\node.exe" "C:\\x\\opendesk.js" memory start')).toBe(true);
    expect(cmd).toContain("--interval 45 --home C:\\h");
  });
  it("parseCombo understands pynput syntax", () => {
    expect(parseCombo("<cmd>+<shift>+<alt>+p")).toEqual({ ctrl: false, shift: true, alt: true, meta: true, key: "p" });
    expect(parseCombo("<ctrl>+<alt>+m")).toEqual({ ctrl: true, shift: false, alt: true, meta: false, key: "m" });
    expect(() => parseCombo("<ctrl>+<alt>")).toThrow();
  });
});

// ===========================================================================
// Screen memory v2 — events trigger, AX text, redaction, encryption
// ===========================================================================

import { redact } from "../src/memory/redact.js";
import {
  MemoryCipher, tokenize, loadOrCreateKey, readKeyInfo, keyId, defaultResolveKeyProvider, isDefaultHome,
} from "../src/memory/crypto.js";
import { coerceFrontmost } from "../src/memory/frontmost.js";
import { titleKey } from "../src/memory/recorder.js";

function v2Recorder(h: string, extra: Partial<ConstructorParameters<typeof ScreenMemoryRecorder>[0]> & {
  probe?: { cur: [string, string, number?]; calls: number }; clockRef?: { t: number };
} = {}) {
  const probe = extra.probe ?? { cur: ["Terminal", "zsh", 42] as [string, string, number?], calls: 0 };
  const clockRef = extra.clockRef ?? { t: 1000 };
  const { probe: _p, clockRef: _c, ...rest } = extra;
  void _p; void _c;
  let n = 0;
  return new ScreenMemoryRecorder({
    home: h,
    config: new MemoryConfig({ deny_apps: ["1Password"], pause_hotkey: "" }),
    capture: async () => png(n++ % 2 ? 0xffffffff : 0x000000ff),
    frontmost: () => { probe.calls++; return { app: probe.cur[0], title: probe.cur[1], pid: probe.cur[2] }; },
    ocr: async () => "ocr text from the screen",
    clock: () => clockRef.t,
    log: () => {},
    ...rest,
  });
}

describe("MemoryConfig v2", () => {
  it("has event-trigger, redaction and encryption defaults", () => {
    const cfg = new MemoryConfig();
    expect(cfg.trigger).toBe("events");
    expect([cfg.poll_seconds, cfg.min_interval_seconds, cfg.interval_seconds]).toEqual([1, 2, 30]);
    expect([cfg.redact, cfg.encrypt_thumbnails, cfg.encrypt_text]).toEqual([true, true, false]);
    for (const p of ["Private Browsing", "Incognito", "InPrivate"]) expect(cfg.deny_apps).toContain(p);
    expect(cfg.isDenied("Google Chrome", "New Tab - Google Chrome (Incognito)")).toBe(true);
    expect(cfg.isDenied("Safari", "GitHub — Private Browsing")).toBe(true);
    expect(cfg.isDenied("Safari", "GitHub")).toBe(false);
  });

  it("migrates a v1 config file and normalises hand edits", () => {
    fs.mkdirSync(path.join(home, "memory"), { recursive: true });
    const file = path.join(home, "memory", "config.json");
    fs.writeFileSync(file, JSON.stringify({ interval_seconds: 45, deny_apps: ["1Password", "Signal"], ocr_backend: "auto" }));
    const cfg = loadConfig(home);
    expect(cfg.interval_seconds).toBe(45);
    expect(cfg.trigger).toBe("events");
    expect(cfg.deny_apps.slice(0, 2)).toEqual(["1Password", "Signal"]);
    expect(cfg.deny_apps).toContain("InPrivate");
    saveConfig(cfg, home);
    expect(JSON.parse(fs.readFileSync(file, "utf8")).config_version).toBe(2);
    cfg.denyRemove("Incognito");
    saveConfig(cfg, home);
    expect(loadConfig(home).deny_apps).not.toContain("Incognito");
    fs.writeFileSync(file, JSON.stringify({ config_version: 2, trigger: "sometimes", poll_seconds: 0, redact: "off", encrypt_text: "on" }));
    const c2 = loadConfig(home);
    expect([c2.trigger, c2.poll_seconds, c2.redact, c2.encrypt_text]).toEqual(["events", 0.2, false, true]);
  });
});

describe("redaction", () => {
  it("cards need a valid Luhn checksum", () => {
    expect(redact("Visa 4111 1111 1111 1111 exp 12/29")).toEqual({ text: "Visa [REDACTED:card] exp 12/29", count: 1 });
    expect(redact("4111-1111-1111-1111").text).toBe("[REDACTED:card]");
    expect(redact("4111 1111 1111 1112").count).toBe(0);
    expect(redact("call +1 415 555 0100 or order 1234567").count).toBe(0);
  });

  it("tokens and keys", () => {
    const cases: Record<string, string> = {
      AKIAIOSFODNN7EXAMPLE: "aws_key",
      ["ghp_" + "a".repeat(36)]: "github_token",
      ["github_pat_11ABCDEFG0" + "x".repeat(40)]: "github_token",
      ["sk-ant-api03-" + "Z".repeat(40)]: "anthropic_key",
      ["sk-proj-" + "q".repeat(40)]: "openai_key",
      ["xoxb-1234567890-abcdefghij"]: "slack_token",
      ["eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"]: "jwt",
    };
    for (const [secret, kind] of Object.entries(cases)) {
      expect(redact(`token ${secret} end`)).toEqual({ text: `token [REDACTED:${kind}] end`, count: 1 });
    }
  });

  it("PEM blocks and password pairs", () => {
    const pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234\nabcd==\n-----END RSA PRIVATE KEY-----";
    expect(redact(`before\n${pem}\nafter`).text).toBe("before\n[REDACTED:private_key]\nafter");
    expect(redact("password: hunter2").text).toBe("password: [REDACTED:password]");
    expect(redact('DB_PASSWORD="s3cr3t!"').text).toBe('DB_PASSWORD="[REDACTED:password]"');
    expect(redact("api_key=abc123def").text).toBe("api_key=[REDACTED:secret]");
    expect(redact("Bearer abcdefghijklmnopqrstuvwxyz012345").text).toBe("Bearer [REDACTED:secret]");
    expect(redact("Forgot password? Reset it here").count).toBe(0);
  });
});

describe("crypto", () => {
  it("AES-GCM round trips and detects tampering", () => {
    const c = new MemoryCipher(TEST_KEY);
    const blob = c.encryptBlob(Buffer.from("jpeg"));
    expect(blob.subarray(0, 4).toString()).toBe("ODE1");
    expect(c.decryptBlob(blob).toString()).toBe("jpeg");
    const bad = Buffer.from(blob);
    bad[bad.length - 1] ^= 1;
    expect(() => c.decryptBlob(bad)).toThrow();
    expect(() => new MemoryCipher(Buffer.alloc(32)).decryptBlob(blob)).toThrow();
    const t = c.encryptText("héllo wörld");
    expect(t.startsWith("v1:")).toBe(true);
    expect(c.decryptText(t)).toBe("héllo wörld");
    expect(c.encryptText("x")).not.toBe(c.encryptText("x"));
  });

  it("blind tokens are keyed, normalised, sorted and unique", () => {
    const a = new MemoryCipher(TEST_KEY);
    const b = new MemoryCipher(Buffer.alloc(32));
    expect(a.token("error")).toBe(a.token("error"));
    expect(a.token("error")).not.toBe(b.token("error"));
    expect(tokenize("Error: ECONNREFUSED error Café_bar")).toEqual(["error", "econnrefused", "cafe", "bar"]);
    expect(a.indexTokens("b a b a").split(" ")).toEqual([a.token("a"), a.token("b")].sort());
  });

  it("file key store for a custom home (0600, keyinfo without the secret)", () => {
    setKeyProviderResolver(null);
    delete process.env[ENV_KEYSTORE];
    expect(isDefaultHome(home)).toBe(false);
    expect(defaultResolveKeyProvider(home).name).toBe("file");
    expect(loadOrCreateKey(home, false).key).toBeNull();
    const { key, store } = loadOrCreateKey(home, true);
    expect(store).toBe("file");
    expect(key!.length).toBe(32);
    const kf = path.join(home, "memory", "key");
    if (process.platform !== "win32") expect(fs.statSync(kf).mode & 0o777).toBe(0o600);
    const info = readKeyInfo(home)!;
    expect(info.store).toBe("file");
    expect(info.key_id).toBe(keyId(key!));
    expect(fs.readFileSync(path.join(home, "memory", "keyinfo.json"), "utf8")).not.toContain(key!.toString("hex"));
    expect(loadOrCreateKey(home, true).key!.equals(key!)).toBe(true);
  });
});

describe("MemoryStore encryption", () => {
  it("plaintext thumbnails when disabled", () => {
    const s = new MemoryStore(home, { encryptThumbnails: false });
    const f = s.add({ ts: Date.now() / 1000, app: "A", text: "x", thumb: Buffer.from("jpeg") });
    expect(f.thumb_path!.endsWith(".jpg")).toBe(true);
    expect(fs.readFileSync(path.join(s.dir, f.thumb_path!)).toString()).toBe("jpeg");
    const st = s.stats();
    expect([st.thumbnails, st.encryptedThumbnails, st.encryptThumbnails]).toEqual([1, 0, false]);
  });

  it("encrypted text round-trips and is searchable through the blind index", () => {
    const s = new MemoryStore(home, { encryptText: true });
    const f = s.add({ ts: ts(0, 9), app: "Terminal", title: "zsh", text: "Error: connection refused on port 5432", thumb: Buffer.from("jpg"), text_source: "ax" });
    s.add({ ts: ts(0, 10), app: "Slack", title: "#ops", text: "refused the connection politely" });
    expect(f.encrypted).toBe(true);
    expect(f.text_source).toBe("ax");
    const raw = fs.readFileSync(path.join(s.framesDir, fs.readdirSync(s.framesDir).find((n) => n.endsWith(".jsonl"))!), "utf8");
    expect(raw).not.toContain("refused");
    expect(raw).toContain('"text_enc":"v1:');
    expect(s.get(1)!.text).toBe("Error: connection refused on port 5432");
    const hits = s.search("refused");
    expect(hits.map((h) => h.id).sort()).toEqual([1, 2]);
    expect(hits[0].snippet).toContain("[refused]");
    expect(s.search('"connection refused"').map((h) => h.id)).toEqual([1]);
    expect(s.search("5432").map((h) => h.id)).toEqual([1]);
    expect(s.search("refu*")).toEqual([]); // no prefix search in encrypted text
    expect(s.search("zsh").map((h) => h.id)).toEqual([1]); // title stays plaintext
    expect(s.search("refused", { app: "slack" }).map((h) => h.id)).toEqual([2]);
    const st = s.stats();
    expect(st.encryptedTextFrames).toBe(2);
    expect(st.textSources).toEqual({ ax: 1, ocr: 1 });
  });

  it("mixed plaintext + encrypted store", () => {
    seeded(home);
    const s = new MemoryStore(home, { encryptText: true });
    s.add({ ts: ts(0, 11), app: "Terminal", title: "psql", text: "ECONNREFUSED again on 5432", thumb: Buffer.from("j") });
    const hits = s.search("ECONNREFUSED");
    expect(hits.map((h) => h.app)).toEqual(["Terminal", "Terminal"]);
    expect(hits.every((h) => h.snippet!.includes("ECONNREFUSED"))).toBe(true);
    expect(new Set(s.search("invoice again").map((h) => h.app))).toEqual(new Set(["Preview", "Terminal"]));
    expect(s.stats().encryptedTextFrames).toBe(1);
  });

  it("reindex encrypts, redacts, and decrypts back", () => {
    const s0 = new MemoryStore(home, { encryptThumbnails: false });
    s0.add({ ts: ts(1), app: "Notes", text: "password: hunter2 and invoice 42", thumb: Buffer.from("jpg1") });
    s0.add({ ts: ts(0), app: "Notes", text: "lunch plans", thumb: Buffer.from("jpg2") });
    const s1 = new MemoryStore(home, { encryptText: true, encryptThumbnails: true });
    expect(s1.reindex({ redact: true })).toEqual({
      frames: 2, text_encrypted: 2, text_decrypted: 0, thumbs_encrypted: 2, thumbs_decrypted: 0, redacted: 1, failed: 0,
    });
    const f = s1.get(1)!;
    expect(f.encrypted).toBe(true);
    expect(f.text).toBe("password: [REDACTED:password] and invoice 42");
    expect(s1.thumbnail(f)!.toString()).toBe("jpg1");
    expect(s1.search("invoice").map((h) => h.id)).toEqual([1]);
    expect(s1.search("hunter2")).toEqual([]);
    expect(s1.reindex().text_encrypted).toBe(0);
    const s2 = new MemoryStore(home, { encryptText: false, encryptThumbnails: false });
    const c = s2.reindex();
    expect([c.text_decrypted, c.thumbs_decrypted]).toEqual([2, 2]);
    const g = s2.get(2)!;
    expect(g.encrypted).toBe(false);
    expect(fs.readFileSync(path.join(s2.dir, g.thumb_path!)).toString()).toBe("jpg2");
    expect(s2.stats().encryptedTextFrames).toBe(0);
  });

  it("missing or wrong key degrades gracefully", () => {
    new MemoryStore(home, { encryptText: true }).add({ ts: Date.now() / 1000, app: "A", text: "secret plans", thumb: Buffer.from("jpg") });
    setKeyProviderResolver(() => new StaticKeyProvider(null));
    const s = new MemoryStore(home, { log: () => {} });
    const f = s.get(1)!;
    expect(f.encrypted).toBe(true);
    expect(f.text).toBe("");
    expect(s.thumbnail(f)).toBeNull();
    expect(s.search("plans")).toEqual([]);
    expect(s.reindex().failed).toBe(1);
    setKeyProviderResolver(() => new StaticKeyProvider(Buffer.alloc(32)));
    const s2 = new MemoryStore(home, { log: () => {} });
    expect(s2.get(1)!.text).toBe("");
  });

  it("reads v1 records (no text_source) and infers the source", () => {
    const s = new MemoryStore(home);
    fs.writeFileSync(path.join(s.framesDir, "2026-01-01.jsonl"),
      JSON.stringify({ id: 1, ts: new Date(2026, 0, 1, 12).getTime() / 1000, app: "Terminal", title: "zsh", text: "old ECONNREFUSED", thumb_path: null, thumb_bytes: 0, width: 0, height: 0 }) + "\n" +
      JSON.stringify({ id: 2, ts: new Date(2026, 0, 1, 13).getTime() / 1000, app: "Finder", title: "", text: "", thumb_path: null, thumb_bytes: 0, width: 0, height: 0 }) + "\n");
    fs.writeFileSync(path.join(s.framesDir, "seq"), "2");
    expect(s.get(1)!.text_source).toBe("ocr");
    expect(s.get(2)!.text_source).toBe("none");
    expect(s.search("ECONNREFUSED").map((h) => h.id)).toEqual([1]);
    expect(s.add({ ts: Date.now() / 1000, app: "X", text: "new", text_source: "ax" }).id).toBe(3);
    expect(s.stats().textSources).toEqual({ ocr: 1, none: 1, ax: 1 });
  });
});

describe("event trigger", () => {
  it("triggerReason decisions", () => {
    const rec = v2Recorder(home);
    rec.config.min_interval_seconds = 2;
    rec.config.interval_seconds = 30;
    const a = { app: "Terminal", title: "zsh" };
    expect(rec.triggerReason(a, 0)).toBe("start");
    (rec as unknown as { lastKey: [string, string]; lastAttempt: number }).lastKey = ["Terminal", "zsh"];
    (rec as unknown as { lastAttempt: number }).lastAttempt = 0;
    expect(rec.triggerReason(a, 1)).toBeNull();
    expect(rec.triggerReason({ app: "Terminal", title: "vim" }, 1.5)).toBeNull();
    expect(rec.triggerReason({ app: "Terminal", title: "vim" }, 2)).toBe("title_change");
    expect(rec.triggerReason({ app: "Safari", title: "News" }, 2.5)).toBe("app_change");
    expect(rec.triggerReason(a, 29.9)).toBeNull();
    expect(rec.triggerReason(a, 30)).toBe("interval");
  });

  it("poll captures on switches, debounces, dedupes, and falls back to the interval", async () => {
    const clockRef = { t: 0 };
    const probe = { cur: ["Terminal", "zsh", 1] as [string, string, number?], calls: 0 };
    const same = await png(0xffffffff, [64, 48], 5);
    const black = await png(0x000000ff);
    const queue = [same, same, same, black, same, same];
    const rec = v2Recorder(home, {
      clockRef, probe, capture: async () => (queue.length > 1 ? queue.shift()! : queue[0]),
      config: new MemoryConfig({ interval_seconds: 30, min_interval_seconds: 2, pause_hotkey: "", deny_apps: [] }),
    });
    expect(await rec.poll()).toBe("captured");
    clockRef.t = 1; expect(await rec.poll()).toBeNull();
    probe.cur = ["Safari", "News", 7];
    clockRef.t = 1.5; expect(await rec.poll()).toBeNull();
    clockRef.t = 2.5; expect(await rec.poll()).toBe("captured"); // app switch kept even if pixels barely moved
    probe.cur = ["Safari", "Mail", 7];
    clockRef.t = 5; expect(await rec.poll()).toBe("duplicate"); // title change, identical screen
    probe.cur = ["Safari", "◐ Mail (3) — 45%", 7];
    clockRef.t = 7.5; expect(await rec.poll()).toBeNull(); // spinner/counter-only change ignored
    probe.cur = ["TextEdit", "Untitled", 9];
    clockRef.t = 8; expect(await rec.poll()).toBe("captured");
    clockRef.t = 20; expect(await rec.poll()).toBeNull();
    clockRef.t = 38.5; expect(await rec.poll()).toBe("captured");
    expect(rec.reasons).toEqual({ start: 1, app_change: 2, interval: 1 });
    expect(rec.store.timeline({ newestFirst: false }).map((f) => f.app)).toEqual(["Terminal", "Safari", "TextEdit", "TextEdit"]);
  });

  it("titleKey ignores spinners, counters and percentages", () => {
    expect(titleKey("opendesk — ◐ Screen memory — 80×24")).toBe(titleKey("opendesk — ◑ Screen memory — 80×24"));
    expect(titleKey("(3) Slack | #ops")).toBe(titleKey("(12) Slack | #ops"));
    expect(titleKey("Build 45%")).toBe(titleKey("Build 46%"));
    expect(titleKey("Inbox — Mail")).not.toBe(titleKey("Drafts — Mail"));
  });

  it("paused polls never probe; denied windows aren't retried every poll", async () => {
    const probe = { cur: ["1Password 8", "Vault"] as [string, string, number?], calls: 0 };
    const clockRef = { t: 0 };
    let captures = 0;
    const rec = v2Recorder(home, { probe, clockRef, capture: async () => { captures++; return png(); } });
    setPause(home);
    expect(await rec.poll()).toBe("paused");
    expect(probe.calls).toBe(0);
    clearPause(home);
    expect(await rec.poll()).toMatch(/^denied:/);
    clockRef.t = 5;
    expect(await rec.poll()).toBeNull();
    expect(captures).toBe(0);
  });

  it("run loop in events mode captures each switch", async () => {
    const apps = ["Finder", "TextEdit", "Safari"];
    let i = 0;
    let n = 0;
    const rec = new ScreenMemoryRecorder({
      home,
      config: new MemoryConfig({ trigger: "events", poll_seconds: 0.2, min_interval_seconds: 0, pause_hotkey: "", deny_apps: [] }),
      capture: async () => png(n++ % 2 ? 0xffffffff : 0x000000ff),
      frontmost: () => [apps[i++ % 3], "w"],
      ocr: async () => "",
      log: () => {},
    });
    const done = rec.run();
    await new Promise((r) => setTimeout(r, 1100));
    const state = JSON.parse(fs.readFileSync(path.join(home, "memory", "daemon.json"), "utf8"));
    expect(state.trigger).toBe("events");
    rec.stop();
    await done;
    expect(rec.captured).toBeGreaterThanOrEqual(3);
    expect(rec.textSources.none).toBe(rec.captured);
  });

  it("coerces frontmost shapes", () => {
    expect(coerceFrontmost({ app: "A", title: "T", pid: 3 })).toEqual({ app: "A", title: "T", pid: 3 });
    expect(coerceFrontmost(["A", "T"])).toEqual({ app: "A", title: "T", pid: null });
    expect(coerceFrontmost({ appName: "Mail", windowTitle: "Inbox", processId: "12" })).toEqual({ app: "Mail", title: "Inbox", pid: 12 });
    expect(coerceFrontmost(null)).toEqual({ app: "", title: "", pid: null });
  });
});

describe("accessibility text", () => {
  it("uses AX text and skips OCR", async () => {
    const seen: unknown[] = [];
    let ocrCalls = 0;
    const rec = v2Recorder(home, {
      probe: { cur: ["Pages", "Report", 314], calls: 0 },
      windowText: (pid, max) => { seen.push(pid, max); return "Quarterly report — revenue grew 12% quarter over quarter in EMEA"; },
      ocr: async () => { ocrCalls++; return "ocr"; },
    });
    expect(await rec.tick()).toBe("captured");
    const f = rec.store.get(1)!;
    expect(f.text_source).toBe("ax");
    expect(f.text.startsWith("Quarterly report")).toBe(true);
    expect(ocrCalls).toBe(0);
    expect(seen).toEqual([314, 20000]);
    expect(rec.store.search("EMEA").map((h) => h.id)).toEqual([1]);
  });

  it("short / missing / failing AX text falls back to OCR; async + ocr_backend=always", async () => {
    const r1 = v2Recorder(home, { windowText: () => "OK", ocr: async () => "from ocr" });
    await r1.tick();
    expect(r1.store.get(1)!.text_source).toBe("ocr");
    const h2 = path.join(home, "b");
    const r2 = v2Recorder(h2, { windowText: () => null, ocr: async () => "(no text detected)" });
    await r2.tick();
    expect(r2.store.get(1)!.text_source).toBe("none");
    const h3 = path.join(home, "c");
    const r3 = v2Recorder(h3, { windowText: () => { throw new Error("AX denied"); }, ocr: async () => "ocr wins" });
    await r3.tick();
    expect(r3.store.get(1)!.text_source).toBe("ocr");
    const h4 = path.join(home, "d");
    const r4 = v2Recorder(h4, {
      windowText: async () => "A".repeat(30) + " accessibility text body here",
      ocr: async () => "pixels said this",
      config: new MemoryConfig({ ocr_backend: "always", pause_hotkey: "" }),
    });
    await r4.tick();
    const f4 = r4.store.get(1)!;
    expect(f4.text_source).toBe("ax");
    expect(f4.text.endsWith("pixels said this")).toBe(true);
  });

  it("redacts secrets in AX text and titles before storing", async () => {
    const secret = "ghp_" + "b".repeat(36);
    const rec = v2Recorder(home, {
      probe: { cur: ["Terminal", `export GH=${secret}`], calls: 0 },
      windowText: () => `$ export GITHUB_TOKEN=${secret}\n$ git push origin main --force`,
    });
    expect(await rec.tick()).toBe("captured");
    const day = fs.readdirSync(rec.store.framesDir).find((n) => n.endsWith(".jsonl"))!;
    expect(fs.readFileSync(path.join(rec.store.framesDir, day), "utf8")).not.toContain(secret);
    const f = rec.store.get(1)!;
    expect(f.text).toContain("[REDACTED:github_token]");
    expect(f.title).toContain("[REDACTED:github_token]");
    expect(rec.redactions).toBe(2);
  });

  it("injected capture never consults the native module", async () => {
    const rec = v2Recorder(home);
    const resolved = await (rec as unknown as { resolveWindowText(): Promise<unknown> }).resolveWindowText();
    expect(resolved).toBeNull();
  });
});


describe("MemoryManageTool privacy config", () => {
  it("sets trigger and privacy flags, and status reports them", async () => {
    const manage = new MemoryManageTool();
    const r = await manage.execute(ctx(home), { action: "config", trigger: "interval", redact: false, encryptText: true, encryptThumbnails: false });
    const cfg = loadConfig(home);
    expect([cfg.trigger, cfg.redact, cfg.encrypt_text, cfg.encrypt_thumbnails]).toEqual(["interval", false, true, false]);
    expect(r.output).toContain("opendesk-js memory reindex");
    const bad = await manage.execute(ctx(home), { action: "config", trigger: "sometimes" });
    expect(bad.error).toBe(true);
    const st = await new MemoryTool().execute(ctx(home), { action: "status" });
    expect(st.output).toContain("trigger:   interval");
    expect(st.output).toContain("encrypt:");
  });
});
