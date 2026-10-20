"""Export screen memory to plain files: JSON, CSV or a browsable HTML page.

The export is decrypted — text and thumbnails come out readable, so the
folder can be opened anywhere without the memory key.  It is written with
owner-only permissions where the OS supports them; treat it like the
history it contains.

    out/
      frames.json | frames.csv | index.html
      thumbs/<id>.jpg            (with thumbnails=True)
"""

from __future__ import annotations

import csv
import datetime as _dt
import html
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from opendesk.memory.store import Frame, MemoryStore

FORMATS = ("json", "csv", "html")
_FILES = {"json": "frames.json", "csv": "frames.csv", "html": "index.html"}
_CSV_FIELDS = ["id", "when", "ts", "app", "title", "text_source", "thumbnail", "text"]


@dataclass
class ExportResult:
    path: Path
    """The folder the export was written to."""
    file: Path
    """The main file (``frames.json``, ``frames.csv`` or ``index.html``)."""
    frames: int = 0
    thumbnails: int = 0
    unreadable_text: int = 0
    """Frames whose encrypted text could not be decrypted (key unavailable)."""
    missing_thumbnails: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file": str(self.file),
            "frames": self.frames,
            "thumbnails": self.thumbnails,
            "unreadable_text": self.unreadable_text,
            "missing_thumbnails": self.missing_thumbnails,
        }


def default_export_dir(base: Optional[Path] = None) -> Path:
    """``<base>/opendesk-memory-<timestamp>``; *base* defaults to ~/Downloads
    (or ~ when there is no Downloads folder)."""
    if base is None:
        downloads = Path.home() / "Downloads"
        base = downloads if downloads.is_dir() else Path.home()
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(base) / f"opendesk-memory-{stamp}"


def export_memory(
    out: Optional[Path] = None,
    *,
    home: Optional[Path] = None,
    fmt: str = "json",
    start: float = 0.0,
    end: Optional[float] = None,
    app: Optional[str] = None,
    query: Optional[str] = None,
    thumbnails: bool = True,
    limit: Optional[int] = None,
    store: Optional[MemoryStore] = None,
) -> ExportResult:
    """Write the frames matching the filters to *out* (oldest first).

    *out* must not exist or be an empty directory.  *query* narrows the
    export to search hits; without it every frame in the window is exported.
    Raises ``ValueError`` for a bad format or a non-empty target.
    """
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}, not {fmt!r}")
    out = Path(out).expanduser() if out is not None else default_export_dir()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"{out} already exists and is not an empty folder")

    own = store is None
    store = store or MemoryStore(home)
    try:
        cap = limit or 10_000_000
        if query and query.strip():
            frames = store.search(query, start=start, end=end, app=app, limit=cap)
            frames.sort(key=lambda f: f.ts)
        else:
            frames = store.timeline(start=start, end=end, app=app, limit=cap, newest_first=False)

        _mkdir_private(out)
        res = ExportResult(path=out, file=out / _FILES[fmt], frames=len(frames))
        thumbs: dict[int, str] = {}
        if thumbnails:
            tdir = out / "thumbs"
            for f in frames:
                if not f.thumb_path:
                    continue
                data = store.thumbnail(f)
                if data is None:
                    res.missing_thumbnails += 1
                    continue
                if not thumbs:
                    _mkdir_private(tdir)
                rel = f"thumbs/{f.id}.jpg"
                _write_private(out / rel, data)
                thumbs[f.id] = rel
            res.thumbnails = len(thumbs)
        res.unreadable_text = sum(1 for f in frames if f.encrypted and not f.text)
        if res.unreadable_text:
            res.warnings.append(
                f"{res.unreadable_text} frame(s) have encrypted text that could not be "
                "decrypted (memory key unavailable); exported without text."
            )
        if res.missing_thumbnails:
            res.warnings.append(
                f"{res.missing_thumbnails} thumbnail(s) were missing or could not be decrypted."
            )

        rows = [_row(f, thumbs.get(f.id)) for f in frames]
        if fmt == "json":
            _write_private(res.file, json.dumps(
                {"exported_at": _dt.datetime.now().isoformat(timespec="seconds"),
                 "frames": rows}, ensure_ascii=False, indent=2,
            ).encode("utf-8"))
        elif fmt == "csv":
            _write_csv(res.file, rows)
        else:
            _write_private(res.file, _html(rows, start, end, app, query).encode("utf-8"))
        return res
    finally:
        if own:
            store.close()


# ---------------------------------------------------------------------------


def _row(f: Frame, thumb: Optional[str]) -> dict[str, Any]:
    return {
        "id": f.id,
        "when": f.when,
        "ts": f.ts,
        "app": f.app,
        "title": f.title,
        "text_source": f.text_source,
        "thumbnail": thumb,
        "text": f.text,
    }


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in _CSV_FIELDS})


def _html(rows: list[dict[str, Any]], start: float, end: Optional[float],
          app: Optional[str], query: Optional[str]) -> str:
    from opendesk.memory.timeparse import fmt_range

    esc = html.escape
    scope = fmt_range(start, end)
    if app:
        scope += f" · app ~ {app}"
    if query:
        scope += f" · search “{query}”"
    cards = []
    for r in rows:
        img = (f'<a href="{esc(r["thumbnail"])}"><img loading="lazy" src="{esc(r["thumbnail"])}" alt=""></a>'
               if r["thumbnail"] else "")
        title = r["title"] if r["title"] and r["title"] != r["app"] else ""
        cards.append(
            f'<article id="f{r["id"]}"><header><b>{esc(r["app"] or "(unknown)")}</b>'
            f'<span>{esc(title)}</span><time>{esc(r["when"])}</time>'
            f'<small>#{r["id"]} · {esc(r["text_source"] or "?")}</small></header>'
            f'{img}<details><summary>Text</summary><pre>{esc(r["text"] or "")}</pre></details></article>'
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenDesk screen memory</title>
<style>
:root{{--bg:#fafafa;--fg:#1a1a1a;--card:#fff;--muted:#666;--line:#e3e3e3}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--card:#1f1f1f;--muted:#999;--line:#333}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif}}
main{{max-width:960px;margin:0 auto}}h1{{font-size:20px;margin:0 0 4px}}p.meta{{color:var(--muted);margin:0 0 20px}}
input{{width:100%;box-sizing:border-box;padding:8px 10px;margin-bottom:16px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}}
article{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:12px}}
header{{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline;margin-bottom:8px}}header span{{flex:1;color:var(--muted);overflow-wrap:anywhere}}
time,small{{color:var(--muted)}}img{{max-width:100%;border-radius:4px;display:block;margin-bottom:8px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.4 ui-monospace,monospace;max-height:320px;overflow:auto}}
</style></head><body><main>
<h1>OpenDesk screen memory</h1>
<p class="meta">{len(rows)} moment(s) · {esc(scope)} · exported {esc(_dt.datetime.now().strftime("%Y-%m-%d %H:%M"))}. This page contains your screen history; keep it private.</p>
<input type="search" placeholder="Filter…" oninput="const q=this.value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.hidden=!a.textContent.toLowerCase().includes(q))">
{"".join(cards) or "<p>No moments in this range.</p>"}
</main></body></html>
"""
