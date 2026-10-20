/**
 * Export screen memory to plain files: JSON, CSV or a browsable HTML page.
 *
 * The export is decrypted — text and thumbnails come out readable, so the
 * folder can be opened anywhere without the memory key.  It is written with
 * owner-only permissions where the OS supports them; treat it like the
 * history it contains.
 *
 *     out/
 *       frames.json | frames.csv | index.html
 *       thumbs/<id>.jpg            (with thumbnails: true)
 */

import fs from "fs";
import os from "os";
import path from "path";

import { MemoryStore, frameWhen, type Frame } from "./store.js";
import { fmtRange } from "./timeparse.js";

export const EXPORT_FORMATS = ["json", "csv", "html"] as const;
export type ExportFormat = (typeof EXPORT_FORMATS)[number];

const FILES: Record<ExportFormat, string> = { json: "frames.json", csv: "frames.csv", html: "index.html" };
const CSV_FIELDS = ["id", "when", "ts", "app", "title", "text_source", "thumbnail", "text"] as const;

export interface ExportOptions {
  home?: string;
  format?: ExportFormat;
  start?: number;
  end?: number | null;
  app?: string | null;
  query?: string | null;
  thumbnails?: boolean;
  limit?: number;
  store?: MemoryStore;
}

export interface ExportResult {
  /** The folder the export was written to. */
  path: string;
  /** The main file (frames.json, frames.csv or index.html). */
  file: string;
  frames: number;
  thumbnails: number;
  /** Frames whose encrypted text could not be decrypted (key unavailable). */
  unreadableText: number;
  missingThumbnails: number;
  warnings: string[];
}

interface Row {
  id: number;
  when: string;
  ts: number;
  app: string;
  title: string;
  text_source: string;
  thumbnail: string | null;
  text: string;
}

/** `<base>/opendesk-memory-<timestamp>`; base defaults to ~/Downloads (or ~). */
export function defaultExportDir(base?: string): string {
  if (!base) {
    const downloads = path.join(os.homedir(), "Downloads");
    base = isDir(downloads) ? downloads : os.homedir();
  }
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  return path.join(base, `opendesk-memory-${stamp}`);
}

/**
 * Write the frames matching the filters to `out` (oldest first).
 *
 * `out` must not exist or be an empty directory.  `query` narrows the export
 * to search hits; without it every frame in the window is exported.  Throws
 * for a bad format or a non-empty target.
 */
export function exportMemory(out: string | undefined, opts: ExportOptions = {}): ExportResult {
  const format = opts.format ?? "json";
  if (!EXPORT_FORMATS.includes(format)) {
    throw new Error(`format must be one of ${EXPORT_FORMATS.join(", ")}, not '${format}'`);
  }
  const dir = out ? path.resolve(expandHome(out)) : defaultExportDir();
  if (fs.existsSync(dir) && (!isDir(dir) || fs.readdirSync(dir).length > 0)) {
    throw new Error(`${dir} already exists and is not an empty folder`);
  }

  const own = !opts.store;
  const store = opts.store ?? new MemoryStore(opts.home);
  try {
    const q = { start: opts.start ?? 0, end: opts.end ?? null, app: opts.app ?? null, limit: opts.limit ?? Infinity };
    let frames: Frame[];
    if (opts.query && opts.query.trim()) {
      frames = store.search(opts.query, q).sort((a, b) => a.ts - b.ts);
    } else {
      frames = store.timeline({ ...q, newestFirst: false });
    }

    mkdirPrivate(dir);
    const res: ExportResult = {
      path: dir, file: path.join(dir, FILES[format]), frames: frames.length,
      thumbnails: 0, unreadableText: 0, missingThumbnails: 0, warnings: [],
    };
    const thumbs = new Map<number, string>();
    if (opts.thumbnails ?? true) {
      for (const f of frames) {
        if (!f.thumb_path) continue;
        const data = store.thumbnail(f);
        if (!data) {
          res.missingThumbnails++;
          continue;
        }
        if (!thumbs.size) mkdirPrivate(path.join(dir, "thumbs"));
        const rel = `thumbs/${f.id}.jpg`;
        writePrivate(path.join(dir, rel), data);
        thumbs.set(f.id, rel);
      }
      res.thumbnails = thumbs.size;
    }
    res.unreadableText = frames.filter((f) => f.encrypted && !f.text).length;
    if (res.unreadableText) {
      res.warnings.push(
        `${res.unreadableText} frame(s) have encrypted text that could not be decrypted (memory key unavailable); exported without text.`,
      );
    }
    if (res.missingThumbnails) {
      res.warnings.push(`${res.missingThumbnails} thumbnail(s) were missing or could not be decrypted.`);
    }

    const rows: Row[] = frames.map((f) => ({
      id: f.id, when: frameWhen(f), ts: f.ts, app: f.app, title: f.title,
      text_source: f.text_source, thumbnail: thumbs.get(f.id) ?? null, text: f.text,
    }));
    if (format === "json") {
      writePrivate(res.file, JSON.stringify({ exported_at: new Date().toISOString(), frames: rows }, null, 2));
    } else if (format === "csv") {
      writePrivate(res.file, toCsv(rows));
    } else {
      writePrivate(res.file, toHtml(rows, q.start, q.end, opts.app ?? null, opts.query ?? null));
    }
    return res;
  } finally {
    if (own) store.close();
  }
}

// ---------------------------------------------------------------------------

function expandHome(p: string): string {
  return p === "~" || p.startsWith("~/") ? path.join(os.homedir(), p.slice(1)) : p;
}

function isDir(p: string): boolean {
  try {
    return fs.statSync(p).isDirectory();
  } catch {
    return false;
  }
}

function mkdirPrivate(p: string): void {
  fs.mkdirSync(p, { recursive: true, mode: 0o700 });
  try {
    fs.chmodSync(p, 0o700);
  } catch {
    /* best effort (Windows) */
  }
}

function writePrivate(p: string, data: string | Buffer): void {
  fs.writeFileSync(p, data, { mode: 0o600 });
}

function csvCell(v: unknown): string {
  const s = v === null || v === undefined ? "" : String(v);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function toCsv(rows: Row[]): string {
  const lines = [CSV_FIELDS.join(",")];
  for (const r of rows) lines.push(CSV_FIELDS.map((k) => csvCell(r[k])).join(","));
  return lines.join("\r\n") + "\r\n";
}

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#x27;");
}

function toHtml(rows: Row[], start: number, end: number | null, app: string | null, query: string | null): string {
  let scope = fmtRange(start, end);
  if (app) scope += ` · app ~ ${app}`;
  if (query) scope += ` · search “${query}”`;
  const cards = rows.map((r) => {
    const img = r.thumbnail ? `<a href="${esc(r.thumbnail)}"><img loading="lazy" src="${esc(r.thumbnail)}" alt=""></a>` : "";
    const title = r.title && r.title !== r.app ? r.title : "";
    return (
      `<article id="f${r.id}"><header><b>${esc(r.app || "(unknown)")}</b>` +
      `<span>${esc(title)}</span><time>${esc(r.when)}</time>` +
      `<small>#${r.id} · ${esc(r.text_source || "?")}</small></header>` +
      `${img}<details><summary>Text</summary><pre>${esc(r.text || "")}</pre></details></article>`
    );
  });
  const now = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  const exported = `${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())} ${p(now.getHours())}:${p(now.getMinutes())}`;
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenDesk screen memory</title>
<style>
:root{--bg:#fafafa;--fg:#1a1a1a;--card:#fff;--muted:#666;--line:#e3e3e3}
@media (prefers-color-scheme:dark){:root{--bg:#151515;--fg:#eee;--card:#1f1f1f;--muted:#999;--line:#333}}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:960px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}p.meta{color:var(--muted);margin:0 0 20px}
input{width:100%;box-sizing:border-box;padding:8px 10px;margin-bottom:16px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
article{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:12px}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline;margin-bottom:8px}header span{flex:1;color:var(--muted);overflow-wrap:anywhere}
time,small{color:var(--muted)}img{max-width:100%;border-radius:4px;display:block;margin-bottom:8px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.4 ui-monospace,monospace;max-height:320px;overflow:auto}
</style></head><body><main>
<h1>OpenDesk screen memory</h1>
<p class="meta">${rows.length} moment(s) · ${esc(scope)} · exported ${exported}. This page contains your screen history; keep it private.</p>
<input type="search" placeholder="Filter…" oninput="const q=this.value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.hidden=!a.textContent.toLowerCase().includes(q))">
${cards.join("") || "<p>No moments in this range.</p>"}
</main></body></html>
`;
}
