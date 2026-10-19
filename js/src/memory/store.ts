/**
 * Local, searchable index of screen captures — the JS counterpart of
 * Python's opendesk/memory/store.py.
 *
 * Storage layout under `<home>/memory`:
 *
 *   frames/YYYY-MM-DD.jsonl   one JSON record per kept frame, per day
 *   frames/seq                monotonically increasing frame id
 *   thumbs/YYYY-MM-DD/<id>.jpg      (or <id>.jpg.enc when encrypted)
 *
 * Why JSONL instead of SQLite: it needs no native module and works on every
 * Node.js version this SDK supports.  Queries are almost always bounded by a
 * time window ("tuesday", "last week"), so only the matching day files are
 * scanned; a day is a few thousand records at most.  Nothing leaves the
 * machine.
 *
 * Encryption at rest (crypto.ts):
 *   - thumbnails (`encrypt_thumbnails`, default on) are AES-GCM encrypted and
 *     decrypted transparently by thumbnail();
 *   - text (`encrypt_text`, opt-in) is stored as `text_enc` with a blind
 *     index `text_tokens` (sorted, de-duplicated HMAC tokens of the words).
 *     Search checks the blind index first and only decrypts candidates, so
 *     matching is exact-word: prefix terms (`foo*`) match only the word
 *     `foo` in encrypted text.  App and title stay plaintext.
 * Old records (no `text_source`, plaintext) and new encrypted ones can share
 * a day file; reindex() converts between the two.
 */

import fs from "node:fs";
import path from "node:path";
import { MemoryConfig, loadConfig, memoryDir } from "./config.js";
import { MemoryCipher, loadOrCreateKey, readKeyInfo, tokenize } from "./crypto.js";
import { redact as redactFn } from "./redact.js";
import { fmtDay, fmtTs } from "./timeparse.js";

export type TextSource = "ax" | "ocr" | "none";
export const TEXT_SOURCES: TextSource[] = ["ax", "ocr", "none"];

export interface Frame {
  id: number;
  ts: number;
  app: string;
  title: string;
  text: string;
  thumb_path: string | null;
  thumb_bytes: number;
  width: number;
  height: number;
  /** Where the text came from: "ax" (accessibility tree), "ocr", or "none". */
  text_source: string;
  /** True when the text is stored encrypted at rest. */
  encrypted?: boolean;
  /** Populated by search(). */
  snippet?: string;
  score?: number;
}

/** What is written to the JSONL day files. */
interface StoredRecord {
  id: number;
  ts: number;
  app: string;
  title: string;
  /** Plaintext text; "" when encrypted. */
  text: string;
  thumb_path: string | null;
  thumb_bytes: number;
  width: number;
  height: number;
  text_source?: string;
  /** AES-GCM encrypted text ("v1:…") when encrypt_text was on. */
  text_enc?: string;
  /** Blind index: space-separated HMAC tokens of the words. */
  text_tokens?: string;
}

export interface StoreStats {
  frames: number;
  oldest: number | null;
  newest: number | null;
  thumbBytes: number;
  indexBytes: number;
  totalBytes: number;
  apps: Array<[string, number]>;
  /** Frame counts per text source, e.g. { ax: 12, ocr: 40, none: 3 }. */
  textSources: Record<string, number>;
  /** Frames whose text is encrypted at rest. */
  encryptedTextFrames: number;
  /** Frames with a thumbnail / of those, stored encrypted. */
  thumbnails: number;
  encryptedThumbnails: number;
  /** Current settings for new frames. */
  encryptText: boolean;
  encryptThumbnails: boolean;
  /** Where the memory key lives (keychain | keyring | file), or null if none yet. */
  keyStore: string | null;
  /** Always true in Node.js (node:crypto); present for parity with Python. */
  encryptionAvailable: boolean;
}

export interface ReindexCounts {
  frames: number;
  text_encrypted: number;
  text_decrypted: number;
  thumbs_encrypted: number;
  thumbs_decrypted: number;
  redacted: number;
  failed: number;
}

export interface QueryOpts {
  start?: number;
  end?: number | null;
  app?: string | null;
  limit?: number;
}

export interface StoreOptions {
  config?: MemoryConfig;
  encryptThumbnails?: boolean;
  encryptText?: boolean;
  log?: (msg: string) => void;
}

export function frameWhen(f: Frame): string {
  return fmtTs(f.ts);
}

interface DayCache {
  mtimeMs: number;
  size: number;
  records: StoredRecord[];
}

export class MemoryStore {
  readonly home?: string;
  readonly dir: string;
  readonly framesDir: string;
  readonly thumbsDir: string;
  encryptThumbnails: boolean;
  encryptText: boolean;
  private cache = new Map<string, DayCache>();
  private plain = new WeakMap<StoredRecord, string | null>();
  private tokenSets = new WeakMap<StoredRecord, Set<string>>();
  private cipherObj: MemoryCipher | null = null;
  private warned = new Set<string>();
  private log: (msg: string) => void;

  constructor(home?: string, opts: StoreOptions = {}) {
    this.home = home;
    this.dir = memoryDir(home);
    this.framesDir = path.join(this.dir, "frames");
    this.thumbsDir = path.join(this.dir, "thumbs");
    fs.mkdirSync(this.framesDir, { recursive: true });
    fs.mkdirSync(this.thumbsDir, { recursive: true });
    const cfg = opts.config ?? loadConfig(home);
    this.encryptThumbnails = opts.encryptThumbnails ?? cfg.encrypt_thumbnails;
    this.encryptText = opts.encryptText ?? cfg.encrypt_text;
    this.log = opts.log ?? ((m) => process.stderr.write(`[memory] ${m}\n`));
  }

  close(): void {
    this.cache.clear();
  }

  /** Pick up encryption settings from a (re)loaded config. */
  applyConfig(cfg: MemoryConfig): void {
    this.encryptThumbnails = Boolean(cfg.encrypt_thumbnails);
    this.encryptText = Boolean(cfg.encrypt_text);
  }

  // ------------------------------------------------------------------
  // Encryption
  // ------------------------------------------------------------------

  private warnOnce(key: string, msg: string): void {
    if (this.warned.has(key)) return;
    this.warned.add(key);
    this.log(msg);
  }

  /** The cipher, loading the key lazily.  `create` mints a key (writes only). */
  private cipher(create: boolean): MemoryCipher | null {
    if (this.cipherObj) return this.cipherObj;
    try {
      const { key } = loadOrCreateKey(this.home, create);
      if (!key) return null;
      this.cipherObj = new MemoryCipher(key);
      return this.cipherObj;
    } catch (e) {
      this.warnOnce("nokey", `screen-memory key unavailable: ${(e as Error).message}`);
      return null;
    }
  }

  private decrypt(r: StoredRecord): string | null {
    if (!r.text_enc) return r.text;
    if (this.plain.has(r)) return this.plain.get(r)!;
    let out: string | null = null;
    const c = this.cipher(false);
    if (!c) {
      this.warnOnce("nodecrypt", "cannot decrypt frame text: memory key unavailable");
    } else {
      try {
        out = c.decryptText(r.text_enc);
      } catch {
        this.warnOnce("baddecrypt", "frame text failed to decrypt (wrong or replaced key?)");
      }
    }
    this.plain.set(r, out);
    return out;
  }

  private tokenSet(r: StoredRecord): Set<string> {
    let s = this.tokenSets.get(r);
    if (!s) {
      s = new Set((r.text_tokens ?? "").split(" ").filter(Boolean));
      this.tokenSets.set(r, s);
    }
    return s;
  }

  /** Text fields for a record under the current settings. */
  private encodeText(text: string): Pick<StoredRecord, "text" | "text_enc" | "text_tokens"> {
    if (this.encryptText) {
      const c = this.cipher(true);
      if (c) return { text: "", text_enc: c.encryptText(text), text_tokens: c.indexTokens(text) };
      this.warnOnce("plaintext", "encrypt_text is on but no key is available; storing plaintext");
    }
    return { text };
  }

  private writeThumb(day: string, id: number, jpeg: Buffer, encrypt: boolean): { rel: string; bytes: number } {
    const dayDir = path.join(this.thumbsDir, day);
    fs.mkdirSync(dayDir, { recursive: true });
    let data = jpeg;
    let name = `${id}.jpg`;
    if (encrypt) {
      const c = this.cipher(true);
      if (c) {
        data = c.encryptBlob(jpeg);
        name = `${id}.jpg.enc`;
      } else {
        this.warnOnce("plainthumb", "encrypt_thumbnails is on but no key is available; storing plaintext");
      }
    }
    const p = path.join(dayDir, name);
    fs.writeFileSync(p, data, { mode: 0o600 });
    return { rel: path.relative(this.dir, p).split(path.sep).join("/"), bytes: data.length };
  }

  // ------------------------------------------------------------------
  // Writes
  // ------------------------------------------------------------------

  add(input: {
    ts: number; app?: string; title?: string; text?: string;
    thumb?: Buffer | null; width?: number; height?: number; text_source?: string;
  }): Frame {
    const id = this.nextId();
    const day = fmtDay(input.ts);
    const text = input.text ?? "";
    const source = TEXT_SOURCES.includes(input.text_source as TextSource)
      ? input.text_source! : (text ? "ocr" : "none");
    let thumbPath: string | null = null;
    let thumbBytes = 0;
    if (input.thumb && input.thumb.length) {
      const t = this.writeThumb(day, id, input.thumb, this.encryptThumbnails);
      thumbPath = t.rel;
      thumbBytes = t.bytes;
    }
    const rec: StoredRecord = {
      id,
      ts: input.ts,
      app: input.app ?? "",
      title: input.title ?? "",
      ...this.encodeText(text),
      thumb_path: thumbPath,
      thumb_bytes: thumbBytes,
      width: input.width ?? 0,
      height: input.height ?? 0,
      text_source: source,
    };
    // Stable key order: text first, like v1 records.
    const ordered: StoredRecord = {
      id: rec.id, ts: rec.ts, app: rec.app, title: rec.title, text: rec.text,
      thumb_path: rec.thumb_path, thumb_bytes: rec.thumb_bytes, width: rec.width, height: rec.height,
      text_source: rec.text_source,
      ...(rec.text_enc ? { text_enc: rec.text_enc, text_tokens: rec.text_tokens } : {}),
    };
    fs.appendFileSync(this.dayFile(day), JSON.stringify(ordered) + "\n", { mode: 0o600 });
    this.cache.delete(day);
    return {
      id, ts: input.ts, app: ordered.app, title: ordered.title, text,
      thumb_path: thumbPath, thumb_bytes: thumbBytes, width: ordered.width, height: ordered.height,
      text_source: source, encrypted: Boolean(ordered.text_enc),
    };
  }

  /** Delete frames (and their thumbnails) by id.  Returns count. */
  delete(ids: number[]): number {
    if (!ids.length) return 0;
    const want = new Set(ids);
    let n = 0;
    for (const day of this.days()) {
      const recs = this.loadDay(day);
      if (!recs.some((r) => want.has(r.id))) continue;
      const keep: StoredRecord[] = [];
      for (const r of recs) {
        if (want.has(r.id)) {
          n++;
          if (r.thumb_path) fs.rmSync(path.join(this.dir, r.thumb_path), { force: true });
        } else {
          keep.push(r);
        }
      }
      this.writeDay(day, keep);
    }
    this.pruneEmptyThumbDirs();
    return n;
  }

  deleteRange(start = 0, end: number | null = null): number {
    return this.delete(this.raw({ start, end }).map((r) => r.id));
  }

  deleteApp(appPattern: string): number {
    return this.delete(this.raw({ app: appPattern }).map((r) => r.id));
  }

  clear(): number {
    return this.delete(this.raw({}).map((r) => r.id));
  }

  /** Delete frames older than `retentionDays`.  Returns count. */
  enforceRetention(retentionDays: number): number {
    if (retentionDays <= 0) return 0;
    const cutoff = Date.now() / 1000 - retentionDays * 86400;
    return this.deleteRange(0, cutoff);
  }

  /**
   * Rolling deletion: drop the oldest frames until on-disk usage is under
   * `capBytes * targetFraction`.  At least one frame is always kept.
   */
  enforceCap(capBytes: number, targetFraction = 0.9): number {
    if (capBytes <= 0) return 0;
    let usage = this.stats().totalBytes;
    if (usage <= capBytes) return 0;
    const target = Math.floor(capBytes * targetFraction);
    let deleted = 0;
    while (usage > target) {
      const all = this.raw({}).sort((a, b) => a.ts - b.ts);
      if (all.length <= 1) break;
      const batch = all.slice(0, Math.max(1, Math.min(200, all.length - 1)));
      const freed = batch.reduce((s, r) => s + r.thumb_bytes + JSON.stringify(r).length + 1, 0);
      deleted += this.delete(batch.map((r) => r.id));
      usage -= freed;
    }
    return deleted;
  }

  /**
   * Rewrite every frame under the current settings: encrypt or decrypt text
   * and thumbnails, and (with `redact`) scrub secrets from old text/titles.
   * Frames that cannot be decrypted are left untouched and counted as failed.
   */
  reindex(opts: { redact?: boolean; progress?: (done: number, total: number) => void } = {}): ReindexCounts {
    const counts: ReindexCounts = {
      frames: 0, text_encrypted: 0, text_decrypted: 0, thumbs_encrypted: 0, thumbs_decrypted: 0, redacted: 0, failed: 0,
    };
    const days = this.days();
    const total = this.raw({}).length;
    for (const day of days) {
      const out: StoredRecord[] = [];
      for (const r0 of this.loadDay(day)) {
        counts.frames++;
        const r: StoredRecord = { ...r0 };
        const wasEnc = Boolean(r.text_enc);
        let text = this.decrypt(r0);
        let ok = text !== null;
        if (ok) {
          let changed = false;
          if (opts.redact) {
            const a = redactFn(text!);
            const b = redactFn(r.title);
            if (a.count || b.count) {
              counts.redacted++;
              changed = true;
              text = a.text;
              r.title = b.text;
            }
          }
          if (changed || wasEnc !== this.encryptText) {
            delete r.text_enc;
            delete r.text_tokens;
            Object.assign(r, this.encodeText(text!));
            if (r.text_enc && !wasEnc) counts.text_encrypted++;
            else if (wasEnc && !r.text_enc) counts.text_decrypted++;
          }
        } else {
          counts.failed++;
        }
        if (r.thumb_path && r.thumb_path.endsWith(".enc") !== this.encryptThumbnails) {
          const isEnc = r.thumb_path.endsWith(".enc");
          const abs = path.join(this.dir, r.thumb_path);
          try {
            let data: Buffer = fs.readFileSync(abs);
            if (isEnc) {
              const c = this.cipher(false);
              if (!c) throw new Error("no key");
              data = c.decryptBlob(data);
            }
            const t = this.writeThumb(path.basename(path.dirname(abs)), r.id, data, this.encryptThumbnails);
            if (t.rel !== r.thumb_path) {
              fs.rmSync(abs, { force: true });
              r.thumb_path = t.rel;
              r.thumb_bytes = t.bytes;
              if (isEnc) counts.thumbs_decrypted++;
              else counts.thumbs_encrypted++;
            }
          } catch {
            if (ok) { counts.failed++; ok = false; }
          }
        }
        out.push(r);
      }
      this.writeDay(day, out);
      opts.progress?.(counts.frames, total);
    }
    return counts;
  }

  // ------------------------------------------------------------------
  // Reads
  // ------------------------------------------------------------------

  get(id: number): Frame | null {
    for (const day of this.days().reverse()) {
      const hit = this.loadDay(day).find((r) => r.id === id);
      if (hit) return this.materialize(hit);
    }
    return null;
  }

  /** JPEG bytes of the thumbnail (decrypting `.jpg.enc`), or null. */
  thumbnail(frame: Frame): Buffer | null {
    if (!frame.thumb_path) return null;
    let data: Buffer;
    try {
      data = fs.readFileSync(path.join(this.dir, frame.thumb_path));
    } catch {
      return null;
    }
    if (!frame.thumb_path.endsWith(".enc")) return data;
    const c = this.cipher(false);
    if (!c) {
      this.warnOnce("nothumbkey", "cannot decrypt thumbnails: memory key unavailable");
      return null;
    }
    try {
      return c.decryptBlob(data);
    } catch {
      this.warnOnce("badthumb", "thumbnail failed to decrypt (wrong or replaced key?)");
      return null;
    }
  }

  /**
   * Full-text search.  All terms must match (AND); when that yields
   * nothing, retries as OR so near-misses still surface.
   */
  search(query: string, opts: QueryOpts = {}): Frame[] {
    const terms = queryTerms(query);
    if (!terms.length) return this.timeline(opts);
    let hits = this.searchTerms(terms, "AND", opts);
    if (!hits.length && terms.length > 1) hits = this.searchTerms(terms, "OR", opts);
    return hits;
  }

  timeline(opts: QueryOpts & { newestFirst?: boolean } = {}): Frame[] {
    const { start = 0, end = null, app = null, limit = 50, newestFirst = true } = opts;
    const out: Frame[] = [];
    const days = this.daysInRange(start, end);
    if (newestFirst) days.reverse();
    for (const day of days) {
      const recs = this.loadDay(day).filter((r) => this.matches(r, start, end, app));
      if (newestFirst) recs.sort((a, b) => b.ts - a.ts);
      else recs.sort((a, b) => a.ts - b.ts);
      for (const r of recs) {
        out.push(this.materialize(r));
        if (out.length >= limit) return out;
      }
    }
    return out;
  }

  apps(opts: QueryOpts = {}): Array<[string, number]> {
    const counts = new Map<string, number>();
    for (const r of this.raw(opts)) counts.set(r.app, (counts.get(r.app) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }

  stats(): StoreStats {
    let frames = 0;
    let oldest: number | null = null;
    let newest: number | null = null;
    let thumbBytes = 0;
    let indexBytes = 0;
    let encryptedTextFrames = 0;
    let thumbnails = 0;
    let encryptedThumbnails = 0;
    const counts = new Map<string, number>();
    const textSources: Record<string, number> = {};
    for (const day of this.days()) {
      try { indexBytes += fs.statSync(this.dayFile(day)).size; } catch { /* gone */ }
      for (const r of this.loadDay(day)) {
        frames++;
        thumbBytes += r.thumb_bytes;
        if (oldest === null || r.ts < oldest) oldest = r.ts;
        if (newest === null || r.ts > newest) newest = r.ts;
        counts.set(r.app, (counts.get(r.app) ?? 0) + 1);
        const src = sourceOf(r);
        textSources[src] = (textSources[src] ?? 0) + 1;
        if (r.text_enc) encryptedTextFrames++;
        if (r.thumb_path) {
          thumbnails++;
          if (r.thumb_path.endsWith(".enc")) encryptedThumbnails++;
        }
      }
    }
    const apps = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
    return {
      frames, oldest, newest, thumbBytes, indexBytes, totalBytes: thumbBytes + indexBytes, apps,
      textSources, encryptedTextFrames, thumbnails, encryptedThumbnails,
      encryptText: this.encryptText, encryptThumbnails: this.encryptThumbnails,
      keyStore: readKeyInfo(this.home)?.store ?? null, encryptionAvailable: true,
    };
  }

  // ------------------------------------------------------------------
  // Internals
  // ------------------------------------------------------------------

  private materialize(r: StoredRecord): Frame {
    const text = r.text_enc ? (this.decrypt(r) ?? "") : r.text;
    return {
      id: r.id, ts: r.ts, app: r.app, title: r.title, text,
      thumb_path: r.thumb_path, thumb_bytes: r.thumb_bytes, width: r.width, height: r.height,
      text_source: sourceOf(r), encrypted: Boolean(r.text_enc),
    };
  }

  private raw(opts: QueryOpts): StoredRecord[] {
    const { start = 0, end = null, app = null } = opts;
    const out: StoredRecord[] = [];
    for (const day of this.daysInRange(start, end)) {
      for (const r of this.loadDay(day)) if (this.matches(r, start, end, app)) out.push(r);
    }
    return out;
  }

  private searchTerms(terms: string[], joiner: "AND" | "OR", opts: QueryOpts): Frame[] {
    const { start = 0, end = null, app = null, limit = 20 } = opts;
    const needles = terms.map((t) => t.toLowerCase().replace(/\*$/, ""));
    let blind: string[][] | null = null; // per-term HMAC tokens, computed on first encrypted record
    const scored: Frame[] = [];
    for (const day of this.daysInRange(start, end)) {
      for (const r of this.loadDay(day)) {
        if (!this.matches(r, start, end, app)) continue;
        const meta = `${r.app}\n${r.title}`.toLowerCase();
        let textLow: string | null = r.text_enc ? null : r.text.toLowerCase();
        let matched = 0;
        let score = 0;
        for (let i = 0; i < terms.length; i++) {
          const needle = needles[i];
          const inMeta = meta.includes(needle);
          let inText = 0;
          if (r.text_enc) {
            if (blind === null) {
              const c = this.cipher(false);
              blind = terms.map((t) => (c ? tokenize(t.replace(/\*$/, "")).map((w) => c.token(w)) : []));
            }
            const toks = blind[i];
            const set = this.tokenSet(r);
            if (toks.length && toks.every((tk) => set.has(tk))) {
              if (textLow === null) textLow = (this.decrypt(r) ?? "").toLowerCase();
              inText = countOccurrences(textLow, needle);
            }
          } else {
            inText = countOccurrences(textLow!, needle);
          }
          if (inText || inMeta) {
            matched++;
            score += Math.min(inText, 5) + (inMeta ? 3 : 0);
          }
        }
        if (matched === 0) continue;
        if (joiner === "AND" && matched < terms.length) continue;
        const f = this.materialize(r);
        scored.push({ ...f, score: score + matched * 10, snippet: makeSnippet(f.text, terms) });
      }
    }
    scored.sort((a, b) => (b.score! - a.score!) || (b.ts - a.ts));
    return scored.slice(0, limit);
  }

  private matches(r: StoredRecord, start: number, end: number | null, app: string | null): boolean {
    if (start > 0 && r.ts < start) return false;
    if (end !== null && r.ts >= end) return false;
    if (app) {
      const a = app.toLowerCase();
      if (!r.app.toLowerCase().includes(a) && !r.title.toLowerCase().includes(a)) return false;
    }
    return true;
  }

  private dayFile(day: string): string {
    return path.join(this.framesDir, `${day}.jsonl`);
  }

  /** All day keys on disk, ascending. */
  days(): string[] {
    let names: string[];
    try {
      names = fs.readdirSync(this.framesDir);
    } catch {
      return [];
    }
    return names
      .filter((n) => /^\d{4}-\d{2}-\d{2}\.jsonl$/.test(n))
      .map((n) => n.slice(0, 10))
      .sort();
  }

  private daysInRange(start: number, end: number | null): string[] {
    const lo = start > 0 ? fmtDay(start) : "";
    const hi = end !== null ? fmtDay(end) : "￿";
    return this.days().filter((d) => d >= lo && d <= hi);
  }

  private loadDay(day: string): StoredRecord[] {
    const file = this.dayFile(day);
    let st: fs.Stats;
    try {
      st = fs.statSync(file);
    } catch {
      this.cache.delete(day);
      return [];
    }
    const cached = this.cache.get(day);
    if (cached && cached.mtimeMs === st.mtimeMs && cached.size === st.size) return cached.records;
    const records: StoredRecord[] = [];
    for (const line of fs.readFileSync(file, "utf8").split("\n")) {
      if (!line.trim()) continue;
      try {
        const r = JSON.parse(line) as StoredRecord;
        const rec: StoredRecord = {
          id: Number(r.id), ts: Number(r.ts), app: r.app ?? "", title: r.title ?? "",
          text: r.text ?? "", thumb_path: r.thumb_path ?? null,
          thumb_bytes: Number(r.thumb_bytes ?? 0), width: Number(r.width ?? 0), height: Number(r.height ?? 0),
        };
        if (r.text_source) rec.text_source = String(r.text_source);
        if (r.text_enc) {
          rec.text_enc = String(r.text_enc);
          rec.text_tokens = String(r.text_tokens ?? "");
        }
        records.push(rec);
      } catch {
        /* skip corrupt line */
      }
    }
    this.cache.set(day, { mtimeMs: st.mtimeMs, size: st.size, records });
    return records;
  }

  private writeDay(day: string, records: StoredRecord[]): void {
    const file = this.dayFile(day);
    this.cache.delete(day);
    if (!records.length) {
      fs.rmSync(file, { force: true });
      return;
    }
    const tmp = file + ".tmp";
    fs.writeFileSync(tmp, records.map((r) => JSON.stringify(r)).join("\n") + "\n", { mode: 0o600 });
    fs.renameSync(tmp, file);
  }

  private nextId(): number {
    const seqFile = path.join(this.framesDir, "seq");
    let n = 0;
    try { n = parseInt(fs.readFileSync(seqFile, "utf8"), 10) || 0; } catch { /* first */ }
    n += 1;
    fs.writeFileSync(seqFile, String(n), { mode: 0o600 });
    return n;
  }

  private pruneEmptyThumbDirs(): void {
    let names: string[] = [];
    try { names = fs.readdirSync(this.thumbsDir); } catch { return; }
    for (const n of names) {
      const p = path.join(this.thumbsDir, n);
      try {
        if (fs.statSync(p).isDirectory() && fs.readdirSync(p).length === 0) fs.rmdirSync(p);
      } catch { /* ignore */ }
    }
  }
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

/** v1 records have no text_source: all their text came from OCR. */
function sourceOf(r: StoredRecord): string {
  if (r.text_source) return r.text_source;
  return r.text || r.text_enc ? "ocr" : "none";
}

/** Split a query into terms.  Quoted phrases stay together. */
export function queryTerms(query: string): string[] {
  const out: string[] = [];
  const re = /"([^"]+)"|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(query ?? "")) !== null) {
    if (m[1]) {
      const phrase = m[1].trim();
      if (phrase) out.push(phrase);
      continue;
    }
    const cleaned = m[2].replace(/^["'.,;:!?()[\]{}]+|["'.,;:!?()[\]{}]+$/g, "");
    if (cleaned && /[\p{L}\p{N}]/u.test(cleaned)) out.push(cleaned);
  }
  return out;
}

function countOccurrences(hay: string, needle: string): number {
  if (!needle) return 0;
  let n = 0;
  let i = 0;
  while ((i = hay.indexOf(needle, i)) !== -1) {
    n++;
    i += needle.length;
  }
  return n;
}

/** Highlighted window of text around the first matched term. */
export function makeSnippet(text: string, terms: string[], width = 100): string {
  const low = text.toLowerCase();
  for (const t of terms) {
    const needle = t.toLowerCase().replace(/\*$/, "");
    const i = low.indexOf(needle);
    if (i >= 0) {
      const a = Math.max(0, i - Math.floor(width / 2));
      const b = Math.min(text.length, i + needle.length + Math.floor(width / 2));
      const before = text.slice(a, i);
      const hit = text.slice(i, i + needle.length);
      const after = text.slice(i + needle.length, b);
      return ((a > 0 ? "…" : "") + before + "[" + hit + "]" + after + (b < text.length ? "…" : "")).replace(/\s+/g, " ");
    }
  }
  return text.slice(0, width).replace(/\s+/g, " ");
}
