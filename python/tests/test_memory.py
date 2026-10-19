"""Tests for screen memory — store, time parsing, recorder, and the tool.

Everything runs against a ``tmp_path`` home so the user's real
``~/.opendesk/memory`` is never touched.  Screen capture and OCR are faked.
"""

from __future__ import annotations

import datetime as _dt
import io
import time
from pathlib import Path

import pytest

from opendesk.memory.config import (
    MemoryConfig,
    clear_pause,
    get_pause,
    is_paused,
    load_config,
    parse_duration,
    save_config,
    set_pause,
    write_daemon_state,
    daemon_alive,
)
from opendesk.memory.store import MemoryStore, _terms
from opendesk.memory.timeparse import parse_range, parse_when
from opendesk.tools.base import ToolContext
from opendesk.tools.memory import MemoryManageTool, MemoryTool

PIL = pytest.importorskip("PIL")

from opendesk.memory import crypto as _crypto  # noqa: E402

_REAL_RESOLVE_KEY_PROVIDER = _crypto.resolve_key_provider
TEST_KEY = bytes(range(32))


@pytest.fixture(autouse=True)
def _isolated_memory_key(monkeypatch):
    """Never touch the real Keychain / keyring: every test gets an
    in-memory key, and subprocesses are forced onto the file key store."""
    provider = _crypto.StaticProvider(TEST_KEY)
    monkeypatch.setattr(_crypto, "resolve_key_provider", lambda home=None: provider)
    monkeypatch.setenv(_crypto.ENV_KEYSTORE, "file")
    return provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png(color=(255, 255, 255), size=(64, 48), text_seed: int = 0) -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, color)
    if text_seed:
        d = ImageDraw.Draw(img)
        d.rectangle([text_seed % 40, 5, text_seed % 40 + 20, 25], fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _ts(days_ago: float = 0, hour: int = 12) -> float:
    day = _dt.date.today() - _dt.timedelta(days=days_ago)
    return _dt.datetime.combine(day, _dt.time(hour=hour)).timestamp()


def seeded_store(home: Path) -> MemoryStore:
    store = MemoryStore(home)
    store.add(ts=_ts(3), app="Terminal", title="zsh", text="Error: ECONNREFUSED 127.0.0.1:5432", thumb=b"jpg1")
    store.add(ts=_ts(2), app="Google Chrome", title="Grafana — Ops dashboard", text="Ops dashboard p95 latency 120ms", thumb=b"jpg2")
    store.add(ts=_ts(1), app="Preview", title="invoice.pdf", text="Invoice INV-2026-0042 total $1,234.00", thumb=b"jpg3")
    store.add(ts=_ts(0, 9), app="Google Chrome", title="Grafana — Ops dashboard", text="Ops dashboard error rate 0.2%", thumb=b"jpg4")
    store.add(ts=_ts(0, 10), app="Slack", title="#general", text="lunch at noon?", thumb=None)
    return store


# ---------------------------------------------------------------------------
# Config / pause state
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_roundtrip(self, tmp_path: Path):
        cfg = load_config(tmp_path)
        assert cfg.interval_seconds == 30
        assert "1Password" in cfg.deny_apps
        cfg.interval_seconds = 12
        cfg.deny_add("Signal")
        save_config(cfg, tmp_path)
        again = load_config(tmp_path)
        assert again.interval_seconds == 12
        assert "Signal" in again.deny_apps

    def test_unknown_keys_ignored(self, tmp_path: Path):
        (tmp_path / "memory").mkdir(parents=True)
        (tmp_path / "memory" / "config.json").write_text('{"interval_seconds": 5, "bogus": 1}')
        assert load_config(tmp_path).interval_seconds == 5

    def test_corrupt_config_falls_back(self, tmp_path: Path):
        (tmp_path / "memory").mkdir(parents=True)
        (tmp_path / "memory" / "config.json").write_text("{not json")
        assert load_config(tmp_path).interval_seconds == 30

    def test_deny_matching_is_case_insensitive_substring(self):
        cfg = MemoryConfig(deny_apps=["1password", "bank"])
        assert cfg.is_denied("1Password 8", None)
        assert cfg.is_denied("Safari", "My Bank — Accounts")
        assert not cfg.is_denied("Safari", "News")
        assert not cfg.is_denied("", "")

    def test_deny_add_remove(self):
        cfg = MemoryConfig(deny_apps=[])
        assert cfg.deny_add("Signal")
        assert not cfg.deny_add("signal")  # duplicate, case-insensitive
        assert cfg.deny_remove("SIGNAL")
        assert not cfg.deny_remove("Signal")
        assert not cfg.deny_add("   ")

    def test_pause_until_resumed(self, tmp_path: Path):
        assert not is_paused(tmp_path)
        st = set_pause(tmp_path, reason="test")
        assert st.until is None
        assert is_paused(tmp_path)
        assert clear_pause(tmp_path)
        assert not is_paused(tmp_path)
        assert not clear_pause(tmp_path)

    def test_pause_expires(self, tmp_path: Path):
        set_pause(tmp_path, duration_seconds=0.05)
        assert is_paused(tmp_path)
        time.sleep(0.1)
        assert get_pause(tmp_path) is None
        assert not (tmp_path / "memory" / "paused.json").exists()

    def test_parse_duration(self):
        assert parse_duration("30m") == 1800
        assert parse_duration("2h") == 7200
        assert parse_duration("1d") == 86400
        assert parse_duration("90") == 90
        assert parse_duration("2 hours") == 7200
        with pytest.raises(ValueError):
            parse_duration("soon")

    def test_daemon_alive_requires_live_pid_and_fresh_heartbeat(self, tmp_path: Path):
        assert not daemon_alive(tmp_path)
        write_daemon_state(tmp_path, status="captured")
        assert daemon_alive(tmp_path)  # our own pid, fresh heartbeat
        assert not daemon_alive(tmp_path, stale_after=0)


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------


class TestTimeParse:
    NOW = _dt.datetime(2026, 9, 4, 15, 30)  # a Friday

    def test_relative_duration(self):
        start, end = parse_when("2h", now=self.NOW)
        assert end is None
        assert start == pytest.approx((self.NOW - _dt.timedelta(hours=2)).timestamp())

    def test_yesterday_is_whole_day(self):
        start, end = parse_when("yesterday", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 9, 3)
        assert _dt.datetime.fromtimestamp(end) == _dt.datetime(2026, 9, 4)

    def test_weekday_most_recent(self):
        start, end = parse_when("tuesday", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 9, 1)
        assert _dt.datetime.fromtimestamp(end) == _dt.datetime(2026, 9, 2)

    def test_weekday_today_and_last(self):
        start, _ = parse_when("friday", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 9, 4)
        start, _ = parse_when("last friday", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 8, 28)

    def test_day_part(self):
        start, end = parse_when("tuesday afternoon", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 9, 1, 12)
        assert _dt.datetime.fromtimestamp(end) == _dt.datetime(2026, 9, 1, 17)

    def test_last_week_and_this_month(self):
        start, end = parse_when("last week", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 8, 24)
        assert _dt.datetime.fromtimestamp(end) == _dt.datetime(2026, 8, 31)
        start, end = parse_when("this month", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 9, 1)
        assert end is None

    def test_last_month(self):
        start, end = parse_when("last month", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 8, 1)
        assert _dt.datetime.fromtimestamp(end) == _dt.datetime(2026, 9, 1)

    def test_iso_date_and_datetime(self):
        start, end = parse_when("2026-09-01", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 9, 1)
        assert _dt.datetime.fromtimestamp(end) == _dt.datetime(2026, 9, 2)
        start, end = parse_when("2026-09-01 14:30", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 9, 1, 14, 30)
        assert end is None

    def test_n_days_ago(self):
        start, end = parse_when("3 days ago", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 9, 1)
        assert end is not None

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            parse_when("whenever", now=self.NOW)

    def test_parse_range_combines(self):
        start, end = parse_range("tuesday", "thursday", now=self.NOW)
        assert _dt.datetime.fromtimestamp(start) == _dt.datetime(2026, 9, 1)
        assert _dt.datetime.fromtimestamp(end) == _dt.datetime(2026, 9, 4)
        start, end = parse_range(None, None, now=self.NOW)
        assert start == 0.0 and end is None
        with pytest.raises(ValueError):
            parse_range("today", "yesterday", now=self.NOW)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestStore:
    def test_add_writes_thumbnail_and_index(self, tmp_path: Path):
        with MemoryStore(tmp_path) as store:
            f = store.add(ts=time.time(), app="A", title="t", text="hello world", thumb=b"jpeg")
            assert f.id == 1
            on_disk = (store.dir / f.thumb_path).read_bytes()
            assert f.thumb_path.endswith(".jpg.enc")  # encrypted by default
            assert on_disk.startswith(b"ODE1") and b"jpeg" not in on_disk
            assert store.get(1).text == "hello world"
            assert store.thumbnail(f) == b"jpeg"
            assert store.stats().frames == 1

    def test_search_finds_terms_and_snippets(self, tmp_path: Path):
        with seeded_store(tmp_path) as store:
            hits = store.search("ECONNREFUSED")
            assert [h.app for h in hits] == ["Terminal"]
            assert "ECONNREFUSED" in (hits[0].snippet or hits[0].text)

            hits = store.search("invoice total")
            assert len(hits) == 1 and hits[0].app == "Preview"

    def test_search_matches_title_and_app(self, tmp_path: Path):
        with seeded_store(tmp_path) as store:
            assert len(store.search("Grafana")) == 2
            assert len(store.search("slack")) == 1

    def test_search_falls_back_to_or(self, tmp_path: Path):
        with seeded_store(tmp_path) as store:
            hits = store.search("invoice latency")  # never co-occur → OR
            apps = {h.app for h in hits}
            assert apps == {"Preview", "Google Chrome"}

    def test_search_prefix_and_punctuation_safe(self, tmp_path: Path):
        with seeded_store(tmp_path) as store:
            assert len(store.search("INV-2026*")) == 1
            assert store.search('"$1,234.00"')  # phrase with punctuation
            assert store.search("(((") == store.timeline()  # no usable terms → timeline

    def test_time_and_app_filters(self, tmp_path: Path):
        with seeded_store(tmp_path) as store:
            start, end = parse_when("today")
            hits = store.search("dashboard", start=start, end=end)
            assert len(hits) == 1
            hits = store.search("dashboard", app="chrome")
            assert len(hits) == 2
            assert store.search("dashboard", app="Terminal") == []

    def test_timeline_and_apps(self, tmp_path: Path):
        with seeded_store(tmp_path) as store:
            tl = store.timeline(limit=2)
            assert [f.app for f in tl] == ["Slack", "Google Chrome"]
            assert store.apps()[0] == ("Google Chrome", 2)
            start, _ = parse_when("today")
            assert len(store.timeline(start=start)) == 2

    def test_delete_removes_thumbnails(self, tmp_path: Path):
        with seeded_store(tmp_path) as store:
            f = store.get(1)
            path = store.dir / f.thumb_path
            assert path.exists()
            assert store.delete([1]) == 1
            assert not path.exists()
            assert store.get(1) is None
            assert store.search("ECONNREFUSED") == []

    def test_delete_range_and_app(self, tmp_path: Path):
        with seeded_store(tmp_path) as store:
            start, end = parse_when("today")
            assert store.delete_range(start, end) == 2
            assert store.delete_app("chrome") == 1
            assert store.stats().frames == 2

    def test_retention(self, tmp_path: Path):
        now = time.time()
        with MemoryStore(tmp_path) as store:
            for days in (10, 5, 2.5, 1, 0):
                store.add(ts=now - days * 86400, app="A", title="", text=f"{days}d", thumb=b"x")
            assert store.enforce_retention(0) == 0  # disabled
            assert store.enforce_retention(2) == 3  # 10d, 5d, 2.5d
            assert {f.text for f in store.timeline()} == {"1d", "0d"}

    def test_storage_cap_rolls_oldest_first(self, tmp_path: Path):
        with MemoryStore(tmp_path) as store:
            for i in range(20):
                store.add(ts=1000 + i, app="A", title="", text="x" * 10, thumb=b"j" * 100_000)
            before = store.stats()
            assert before.frames == 20
            assert before.total_bytes > 2_000_000
            deleted = store.enforce_cap(1_000_000)
            assert 0 < deleted < 20
            remaining = store.timeline(limit=100, newest_first=False)
            # Oldest go first; newest survive.
            assert remaining[-1].ts == 1019
            assert remaining[0].ts > 1000
            assert store.stats().total_bytes <= 1_000_000
            assert store.enforce_cap(10**12) == 0  # under cap → no-op

    def test_cap_never_deletes_last_frame(self, tmp_path: Path):
        with MemoryStore(tmp_path) as store:
            for i in range(3):
                store.add(ts=1000 + i, app="A", title="", text="", thumb=b"j" * 100_000)
            assert store.enforce_cap(1) == 2
            assert store.stats().frames == 1
            assert store.get(3) is not None

    def test_clear(self, tmp_path: Path):
        with seeded_store(tmp_path) as store:
            assert store.clear() == 5
            assert store.stats().frames == 0
            assert not any(store.thumbs_dir.iterdir())

    def test_terms_tokeniser(self):
        assert _terms('error "connection refused" port:5432') == ["error", "connection refused", "port:5432"]
        assert _terms("...") == []
        assert _terms("foo*") == ["foo*"]


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class _Win:
    def __init__(self, app, title):
        self.app_name = app
        self.title = title


class _Computer:
    """Fake Computer: serves a queue of (png, app, title)."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.captures = 0

    async def focused_window(self):
        _, app, title = self.frames[0]
        return _Win(app, title)

    async def capture(self, **_):
        from opendesk.computer.types import Pixmap, PixmapFormat
        png, _, _ = self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
        self.captures += 1
        return Pixmap(data=png, format=PixmapFormat.PNG, width=64, height=48,
                      logical_width=64, logical_height=48)


def _recorder(tmp_path, frames, **kw):
    from opendesk.memory.recorder import ScreenMemoryRecorder
    cfg = kw.pop("config", MemoryConfig(deny_apps=["1Password"]))
    ocr = kw.pop("ocr", lambda png: f"text for {len(png)} bytes")
    return ScreenMemoryRecorder(tmp_path, computer=_Computer(frames), config=cfg, ocr=ocr, **kw)


class TestRecorder:
    @pytest.mark.asyncio
    async def test_tick_captures_and_indexes(self, tmp_path: Path):
        rec = _recorder(tmp_path, [(_png(text_seed=1), "Terminal", "zsh")], ocr=lambda _: "npm ERR! code ELIFECYCLE")
        assert await rec.tick() == "captured"
        rec.store.close()
        with MemoryStore(tmp_path) as store:
            hits = store.search("ELIFECYCLE")
            assert len(hits) == 1
            assert hits[0].app == "Terminal" and hits[0].title == "zsh"
            assert store.thumbnail(hits[0])[:3] == b"\xff\xd8\xff"  # JPEG magic
            assert hits[0].width == 64

    @pytest.mark.asyncio
    async def test_duplicate_frames_skipped(self, tmp_path: Path):
        png = _png(text_seed=3)
        rec = _recorder(tmp_path, [(png, "A", ""), (png, "A", ""), (_png(color=(0, 0, 0)), "A", "")])
        assert await rec.tick() == "captured"
        assert await rec.tick() == "duplicate"
        assert await rec.tick() == "captured"
        assert rec.captured == 2 and rec.skipped == 1

    @pytest.mark.asyncio
    async def test_denied_app_not_captured(self, tmp_path: Path):
        rec = _recorder(tmp_path, [(_png(), "1Password 8", "Vault")])
        assert (await rec.tick()).startswith("denied:")
        assert rec.computer.captures == 0
        assert rec.store.stats().frames == 0

    @pytest.mark.asyncio
    async def test_denied_by_window_title(self, tmp_path: Path):
        cfg = MemoryConfig(deny_apps=["bank"])
        rec = _recorder(tmp_path, [(_png(), "Safari", "Acme Bank — Login")], config=cfg)
        assert (await rec.tick()).startswith("denied:")

    @pytest.mark.asyncio
    async def test_paused_skips_capture(self, tmp_path: Path):
        rec = _recorder(tmp_path, [(_png(), "A", "")])
        set_pause(tmp_path)
        assert await rec.tick() == "paused"
        assert rec.computer.captures == 0
        clear_pause(tmp_path)
        assert await rec.tick() == "captured"

    @pytest.mark.asyncio
    async def test_toggle_pause_hotkey_handler(self, tmp_path: Path):
        rec = _recorder(tmp_path, [(_png(), "A", "")])
        assert rec.toggle_pause() is True
        assert is_paused(tmp_path)
        assert rec.toggle_pause() is False
        assert not is_paused(tmp_path)

    @pytest.mark.asyncio
    async def test_ocr_failure_stores_frame_without_text(self, tmp_path: Path):
        rec = _recorder(tmp_path, [(_png(), "A", "")], ocr=lambda _: "OCR not available. Install …")
        assert await rec.tick() == "captured"
        assert rec.store.get(1).text == ""

    @pytest.mark.asyncio
    async def test_capture_error_is_reported_not_raised(self, tmp_path: Path):
        rec = _recorder(tmp_path, [(_png(), "A", "")])

        async def boom(**_):
            raise RuntimeError("no screen recording permission")
        rec.computer.capture = boom
        assert (await rec.tick()).startswith("error:capture:")

    @pytest.mark.asyncio
    async def test_config_reload_picks_up_deny_list(self, tmp_path: Path):
        cfg = MemoryConfig(deny_apps=[])
        save_config(cfg, tmp_path)
        rec = _recorder(tmp_path, [(_png(text_seed=i), "Slack", "") for i in range(1, 30)], config=cfg)
        assert await rec.tick() == "captured"
        cfg2 = load_config(tmp_path)
        cfg2.deny_add("Slack")
        save_config(cfg2, tmp_path)
        statuses = [await rec.tick() for _ in range(10)]
        assert statuses[-1].startswith("denied:")

    @pytest.mark.asyncio
    async def test_interval_override_survives_reload(self, tmp_path: Path):
        rec = _recorder(tmp_path, [(_png(), "A", "")], interval_override=7)
        assert rec.config.interval_seconds == 7
        rec.reload_config()
        assert rec.config.interval_seconds == 7

    @pytest.mark.asyncio
    async def test_housekeep_applies_cap(self, tmp_path: Path):
        cfg = MemoryConfig(storage_cap_mb=1, retention_days=365)
        rec = _recorder(tmp_path, [(_png(), "A", "")], config=cfg)
        now = time.time()
        for i in range(30):
            rec.store.add(ts=now - 30 + i, app="A", title="", text="", thumb=b"x" * 100_000)
        result = rec.housekeep()
        assert result["expired"] == 0
        assert result["capped"] > 0
        assert rec.store.stats().total_bytes <= 1024 * 1024

    @pytest.mark.asyncio
    async def test_run_loop_stops_cleanly(self, tmp_path: Path):
        import asyncio
        cfg = MemoryConfig(interval_seconds=5, pause_hotkey="")
        rec = _recorder(tmp_path, [(_png(text_seed=1), "A", ""), (_png(text_seed=9), "A", "")], config=cfg)
        task = asyncio.create_task(rec.run())
        await asyncio.sleep(0.3)
        assert daemon_alive(tmp_path)
        rec.stop()
        await asyncio.wait_for(task, timeout=3)
        assert not daemon_alive(tmp_path)
        assert rec.captured >= 1


class TestImageHelpers:
    def test_thumbnail_downscales(self):
        from opendesk.memory.recorder import make_thumbnail
        jpg, w, h = make_thumbnail(_png(size=(1600, 900)), width=400)
        assert (w, h) == (400, 225)
        assert jpg[:3] == b"\xff\xd8\xff"

    def test_signature_delta(self):
        from opendesk.memory.recorder import frame_signature, signature_delta
        a = frame_signature(_png(color=(255, 255, 255)))
        b = frame_signature(_png(color=(250, 250, 250)))
        c = frame_signature(_png(color=(0, 0, 0)))
        assert signature_delta(a, b) < 0.05
        assert signature_delta(a, c) > 0.9
        assert signature_delta(None, a) == 1.0


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def _ctx(home: Path) -> ToolContext:
    from tests._fakes import FakeComputer
    return ToolContext(session_id="mem-test", computer=FakeComputer(), metadata={"opendesk_home": str(home)})


async def call(tool: MemoryTool | MemoryManageTool, home: Path, **kw):
    return await tool.execute(_ctx(home), tool.parse_params(kw))


class TestMemoryTool:
    @pytest.mark.asyncio
    async def test_registered_and_schema(self):
        from opendesk.registry import create_registry
        reg = create_registry()
        assert "memory" in reg
        schema = reg.get("memory").get_schema()
        assert "search" in schema["properties"]["action"]["enum"]

    def test_read_and_manage_actions_are_split(self):
        from opendesk.registry import create_registry
        reg = create_registry()
        read = set(reg.get("memory").get_schema()["properties"]["action"]["enum"])
        manage = set(reg.get("memory_manage").get_schema()["properties"]["action"]["enum"])
        assert read == {"search", "show", "timeline", "status"}
        assert manage == {"pause", "resume", "deny", "config", "delete"}
        assert reg.get("memory").tool_annotations()["readOnlyHint"] is True
        ann = reg.get("memory_manage").tool_annotations()
        assert ann["readOnlyHint"] is False and ann["destructiveHint"] is True

    def test_read_tool_rejects_control_actions(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MemoryTool().parse_params({"action": "delete", "app": "chrome", "confirm": True})
        with pytest.raises(ValidationError):
            MemoryManageTool().parse_params({"action": "search", "query": "x"})

    @pytest.mark.asyncio
    async def test_search_returns_hits_with_ids(self, tmp_path: Path):
        seeded_store(tmp_path).close()
        tool = MemoryTool()
        r = await call(tool, tmp_path, action="search", query="error", app="Terminal")
        assert not r.error
        assert "ECONNREFUSED" in r.output
        assert r.metadata["count"] == 1
        assert "#1" in r.output

    @pytest.mark.asyncio
    async def test_search_with_time_phrase(self, tmp_path: Path):
        seeded_store(tmp_path).close()
        r = await call(MemoryTool(), tmp_path, action="search", query="dashboard", since="today")
        assert r.metadata["count"] == 1
        r = await call(MemoryTool(), tmp_path, action="search", query="dashboard", since="this week")
        assert r.metadata["count"] >= 1

    @pytest.mark.asyncio
    async def test_search_requires_query_and_validates_time(self, tmp_path: Path):
        r = await call(MemoryTool(), tmp_path, action="search")
        assert r.error
        r = await call(MemoryTool(), tmp_path, action="search", query="x", since="whenever")
        assert r.error and "Cannot parse" in r.output

    @pytest.mark.asyncio
    async def test_empty_index_hints_to_start_daemon(self, tmp_path: Path):
        r = await call(MemoryTool(), tmp_path, action="search", query="anything")
        assert "opendesk memory start" in r.output

    @pytest.mark.asyncio
    async def test_show_returns_text_and_thumbnail(self, tmp_path: Path):
        seeded_store(tmp_path).close()
        r = await call(MemoryTool(), tmp_path, action="show", id=3)
        assert "INV-2026-0042" in r.output
        assert r.attachments and r.attachments[0].media_type == "image/jpeg"
        assert r.attachments[0].content == b"jpg3"
        r = await call(MemoryTool(), tmp_path, action="show", id=3, include_image=False)
        assert not r.attachments
        r = await call(MemoryTool(), tmp_path, action="show", id=999)
        assert r.error

    @pytest.mark.asyncio
    async def test_timeline_groups_apps(self, tmp_path: Path):
        seeded_store(tmp_path).close()
        # "7d", not "this week": the seeded frames span the last 3 days,
        # which crosses a week boundary early in the week.
        r = await call(MemoryTool(), tmp_path, action="timeline", since="7d", app="chrome")
        assert r.metadata["count"] == 2
        assert "Grafana" in r.output

    @pytest.mark.asyncio
    async def test_status(self, tmp_path: Path):
        seeded_store(tmp_path).close()
        r = await call(MemoryTool(), tmp_path, action="status")
        assert "frames:    5" in r.output
        assert "NOT running" in r.output
        assert r.metadata["daemon_alive"] is False

    @pytest.mark.asyncio
    async def test_pause_resume(self, tmp_path: Path):
        r = await call(MemoryManageTool(), tmp_path, action="pause", duration="1h")
        assert "paused" in r.output.lower()
        assert is_paused(tmp_path)
        r = await call(MemoryTool(), tmp_path, action="status")
        assert "PAUSED" in r.output
        r = await call(MemoryManageTool(), tmp_path, action="resume")
        assert not is_paused(tmp_path)

    @pytest.mark.asyncio
    async def test_deny_list_management(self, tmp_path: Path):
        r = await call(MemoryManageTool(), tmp_path, action="deny", deny_add="Signal")
        assert "Signal" in r.metadata["deny_apps"]
        assert "Signal" in load_config(tmp_path).deny_apps
        r = await call(MemoryManageTool(), tmp_path, action="deny", deny_remove="Signal")
        assert "Signal" not in load_config(tmp_path).deny_apps

    @pytest.mark.asyncio
    async def test_config_update(self, tmp_path: Path):
        r = await call(MemoryManageTool(), tmp_path, action="config", interval_seconds=15, storage_cap_mb=512, retention_days=7)
        cfg = load_config(tmp_path)
        assert (cfg.interval_seconds, cfg.storage_cap_mb, cfg.retention_days) == (15, 512, 7)
        assert "Updated" in r.output

    @pytest.mark.asyncio
    async def test_delete_requires_scope_and_confirm(self, tmp_path: Path):
        seeded_store(tmp_path).close()
        r = await call(MemoryManageTool(), tmp_path, action="delete")
        assert r.error
        r = await call(MemoryManageTool(), tmp_path, action="delete", app="chrome")
        assert "would be deleted" in r.output and r.metadata["count"] == 2
        with MemoryStore(tmp_path) as s:
            assert s.stats().frames == 5
        r = await call(MemoryManageTool(), tmp_path, action="delete", app="chrome", confirm=True)
        assert r.metadata["count"] == 2
        with MemoryStore(tmp_path) as s:
            assert s.stats().frames == 3

    @pytest.mark.asyncio
    async def test_permission_handler_is_consulted(self, tmp_path: Path):
        from opendesk.tools.base import PermissionDeniedError

        async def deny(tool, arg, desc):
            raise PermissionDeniedError("nope")

        ctx = ToolContext(permission_handler=deny, metadata={"opendesk_home": str(tmp_path)})
        tool = MemoryTool()
        with pytest.raises(PermissionDeniedError):
            await tool.execute(ctx, tool.parse_params({"action": "status"}))

    @pytest.mark.asyncio
    async def test_home_via_constructor(self, tmp_path: Path):
        seeded_store(tmp_path).close()
        tool = MemoryTool(home=tmp_path)
        r = await tool.execute(ToolContext(), tool.parse_params({"action": "status"}))
        assert "frames:    5" in r.output

    @pytest.mark.asyncio
    async def test_mcp_dispatch_not_peer_aware(self, tmp_path: Path):
        from opendesk.integrations.mcp import MCPDispatcher, PEER_AWARE_TOOLS, TextResult
        from opendesk.integrations.mcp_session import MCPSession
        from opendesk.registry import ToolRegistry
        from tests._fakes import FakeComputer

        assert "memory" not in PEER_AWARE_TOOLS
        reg = ToolRegistry()
        reg.register(MemoryTool(home=tmp_path))
        session = MCPSession(home=tmp_path, local=FakeComputer())
        disp = MCPDispatcher(reg, session)
        entries = await disp.list_tools()
        mem = next(e for e in entries if e.name == "memory")
        assert "peer" not in mem.schema["properties"]
        out = await disp.call_tool("memory", {"action": "status"})
        assert isinstance(out[0], TextResult) and "Screen memory status" in out[0].text


# ---------------------------------------------------------------------------
# Service renderers + CLI wiring
# ---------------------------------------------------------------------------


class TestServiceAndCli:
    def test_launchd_plist(self):
        import xml.etree.ElementTree as ET
        from opendesk.memory.service import LAUNCHD_LABEL, _render_launchd_plist
        plist = _render_launchd_plist("/usr/bin/python3", 20, Path("/tmp/h"))
        ET.fromstring(plist)
        assert f"<string>{LAUNCHD_LABEL}</string>" in plist
        for s in ("memory", "start", "--interval", "20", "--home", "/tmp/h"):
            assert f"<string>{s}</string>" in plist

    def test_systemd_unit(self):
        from opendesk.memory.service import _render_systemd_unit
        unit = _render_systemd_unit("/usr/bin/python3", None, None)
        assert "ExecStart=/usr/bin/python3 -m opendesk.cli memory start\n" in unit
        assert "--home" not in unit

    def test_schtasks_command(self):
        from opendesk.memory.service import _render_schtasks_command
        cmd = _render_schtasks_command("C:\\py\\python.exe", 45, Path("C:\\h"))
        assert cmd.startswith('"C:\\py\\python.exe" -m opendesk.cli memory start')
        assert '--interval 45 --home "C:\\h"' in cmd

    def test_cli_memory_help(self):
        import subprocess, sys
        r = subprocess.run([sys.executable, "-m", "opendesk.cli", "memory", "--help"],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0
        for sub in ("start", "status", "pause", "resume", "search", "deny", "clear"):
            assert sub in r.stdout

    def test_cli_search_and_status(self, tmp_path: Path):
        import subprocess, sys
        seeded_store(tmp_path).close()
        r = subprocess.run(
            [sys.executable, "-m", "opendesk.cli", "memory", "search", "invoice", "--home", str(tmp_path)],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0 and "INV-2026-0042" in r.stdout
        r = subprocess.run(
            [sys.executable, "-m", "opendesk.cli", "memory", "status", "--home", str(tmp_path)],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0 and "frames:    5" in r.stdout


# ===========================================================================
# Screen memory v2 — events trigger, AX text, redaction, encryption
# ===========================================================================

import json as _json
import sqlite3 as _sqlite3


class _Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class _Probe:
    """Fake frontmost probe returning whatever ``.cur`` is set to."""

    def __init__(self, app="Terminal", title="zsh", pid=42):
        self.cur = (app, title, pid)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        app, title, pid = self.cur
        return {"app": app, "title": title, "pid": pid}


def _v2_recorder(tmp_path, *, frames=None, config=None, probe=None, clock=None, ocr=None, window_text=None):
    from opendesk.memory.recorder import ScreenMemoryRecorder
    frames = frames or [(_png(text_seed=i), "X", "") for i in range(1, 40)]
    cfg = config or MemoryConfig(deny_apps=["1Password"], pause_hotkey="")
    return ScreenMemoryRecorder(
        tmp_path, computer=_Computer(frames), config=cfg,
        ocr=ocr or (lambda png: "ocr text from the screen"),
        frontmost=probe or _Probe(), clock=clock or _Clock(), window_text=window_text,
    )


class TestConfigV2:
    def test_new_defaults(self):
        cfg = MemoryConfig()
        assert cfg.trigger == "events"
        assert cfg.poll_seconds == 1.0 and cfg.min_interval_seconds == 2.0
        assert cfg.interval_seconds == 30
        assert cfg.redact is True
        assert cfg.encrypt_thumbnails is True and cfg.encrypt_text is False
        for pat in ("Private Browsing", "Incognito", "InPrivate"):
            assert pat in cfg.deny_apps

    def test_private_windows_denied_by_title(self):
        cfg = MemoryConfig()
        assert cfg.is_denied("Safari", "GitHub — Private Browsing")
        assert cfg.is_denied("Google Chrome", "New Tab - Google Chrome (Incognito)")
        assert cfg.is_denied("Microsoft Edge", "Bing - [InPrivate] - Microsoft Edge")
        assert not cfg.is_denied("Safari", "GitHub")

    def test_v1_config_file_migrates(self, tmp_path: Path):
        (tmp_path / "memory").mkdir(parents=True)
        v1 = {
            "interval_seconds": 45, "storage_cap_mb": 512, "retention_days": 7,
            "deny_apps": ["1Password", "Signal"], "pause_hotkey": "", "thumbnail_width": 480,
            "thumbnail_quality": 60, "dedupe_threshold": 0.02, "ocr_backend": "auto",
        }
        (tmp_path / "memory" / "config.json").write_text(_json.dumps(v1))
        cfg = load_config(tmp_path)
        assert cfg.interval_seconds == 45 and cfg.storage_cap_mb == 512
        assert cfg.trigger == "events" and cfg.redact and cfg.encrypt_thumbnails
        assert cfg.deny_apps[:2] == ["1Password", "Signal"]
        assert "Incognito" in cfg.deny_apps
        save_config(cfg, tmp_path)
        raw = _json.loads((tmp_path / "memory" / "config.json").read_text())
        assert raw["config_version"] == 2 and raw["trigger"] == "events"
        # A v2 file is taken at its word: removed defaults stay removed.
        cfg.deny_remove("Incognito")
        save_config(cfg, tmp_path)
        assert "Incognito" not in load_config(tmp_path).deny_apps

    def test_normalize_hand_edited_values(self, tmp_path: Path):
        (tmp_path / "memory").mkdir(parents=True)
        (tmp_path / "memory" / "config.json").write_text(_json.dumps({
            "config_version": 2, "trigger": "sometimes", "poll_seconds": 0, "redact": "off",
            "encrypt_text": "on", "min_interval_seconds": "x",
        }))
        cfg = load_config(tmp_path)
        assert cfg.trigger == "events"
        assert cfg.poll_seconds == 0.2
        assert cfg.min_interval_seconds == 2.0
        assert cfg.redact is False and cfg.encrypt_text is True


class TestRedaction:
    def _r(self, text):
        from opendesk.memory.redact import redact
        return redact(text)

    def test_cards_need_luhn(self):
        out, n = self._r("Visa 4111 1111 1111 1111 exp 12/29")
        assert out == "Visa [REDACTED:card] exp 12/29" and n == 1
        assert self._r("card 4111-1111-1111-1111")[0] == "card [REDACTED:card]"
        assert self._r("4111 1111 1111 1112")[1] == 0  # fails Luhn
        assert self._r("call +1 415 555 0100 or order 1234567")[1] == 0

    def test_cloud_and_vcs_tokens(self):
        cases = {
            "AKIAIOSFODNN7EXAMPLE": "aws_key",
            "ghp_" + "a" * 36: "github_token",
            "github_pat_11ABCDEFG0" + "x" * 40: "github_token",
            "sk-ant-api03-" + "Z" * 40: "anthropic_key",
            "sk-proj-" + "q" * 40: "openai_key",
            "sk-" + "Q1" * 20: "openai_key",
            "xoxb-1234567890-abcdefghij": "slack_token",
        }
        for secret, kind in cases.items():
            out, n = self._r(f"token {secret} end")
            assert out == f"token [REDACTED:{kind}] end", (secret, out)
            assert n == 1

    def test_jwt_and_pem(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        assert self._r(f"Authorization: {jwt}")[0] == "Authorization: [REDACTED:jwt]"
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234\nabcd==\n-----END RSA PRIVATE KEY-----"
        out, n = self._r(f"before\n{pem}\nafter")
        assert out == "before\n[REDACTED:private_key]\nafter" and n == 1
        # OCR often cuts the END line off — still redacted.
        assert "MIIE" not in self._r("-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\n")[0]

    def test_password_pairs(self):
        assert self._r("password: hunter2")[0] == "password: [REDACTED:password]"
        assert self._r('DB_PASSWORD="s3cr3t!"')[0] == 'DB_PASSWORD="[REDACTED:password]"'
        assert self._r("api_key=abc123def")[0] == "api_key=[REDACTED:secret]"
        assert self._r("Bearer abcdefghijklmnopqrstuvwxyz012345")[0] == "Bearer [REDACTED:secret]"
        assert self._r("Forgot password? Reset it here")[1] == 0
        assert self._r("")[0] == ""


class TestCrypto:
    def test_cipher_round_trip_and_tamper(self):
        from opendesk.memory.crypto import MemoryCipher
        c = MemoryCipher(TEST_KEY)
        blob = c.encrypt_blob(b"\xff\xd8\xffjpeg")
        assert blob.startswith(b"ODE1") and c.decrypt_blob(blob) == b"\xff\xd8\xffjpeg"
        bad = blob[:-1] + bytes([blob[-1] ^ 1])
        with pytest.raises(Exception):
            c.decrypt_blob(bad)
        with pytest.raises(Exception):
            MemoryCipher(bytes(32)).decrypt_blob(blob)
        tok = c.encrypt_text("héllo wörld")
        assert tok.startswith("v1:") and c.decrypt_text(tok) == "héllo wörld"
        assert c.encrypt_text("x") != c.encrypt_text("x")  # random nonce

    def test_blind_tokens(self):
        from opendesk.memory.crypto import MemoryCipher, tokenize
        a, b = MemoryCipher(TEST_KEY), MemoryCipher(bytes(32))
        assert a.token("error") == a.token("error") != b.token("error")
        assert tokenize("Error: ECONNREFUSED error Café_bar") == ["error", "econnrefused", "cafe", "bar"]
        idx = a.index_tokens("b a b a")
        assert idx.split() == sorted({a.token("a"), a.token("b")})

    def test_file_key_store_for_custom_home(self, tmp_path: Path, monkeypatch):
        import os
        from opendesk.memory.crypto import load_or_create_key, read_keyinfo
        monkeypatch.setattr(_crypto, "resolve_key_provider", _REAL_RESOLVE_KEY_PROVIDER)
        monkeypatch.delenv(_crypto.ENV_KEYSTORE, raising=False)
        assert load_or_create_key(tmp_path, create=False) == (None, "file")
        key, store = load_or_create_key(tmp_path, create=True)
        assert store == "file" and len(key) == 32
        kf = tmp_path / "memory" / "key"
        assert oct(os.stat(kf).st_mode & 0o777) == "0o600"
        info = read_keyinfo(tmp_path)
        assert info["store"] == "file" and info["key_id"] == _crypto.key_id(key)
        assert key.hex() not in (tmp_path / "memory" / "keyinfo.json").read_text()
        assert load_or_create_key(tmp_path, create=True)[0] == key

    def test_keyinfo_store_is_honoured(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(_crypto, "resolve_key_provider", _REAL_RESOLVE_KEY_PROVIDER)
        monkeypatch.delenv(_crypto.ENV_KEYSTORE, raising=False)
        (tmp_path / "memory").mkdir(parents=True)
        (tmp_path / "memory" / "keyinfo.json").write_text('{"store": "file"}')
        assert _crypto.resolve_key_provider(tmp_path).name == "file"
        assert not _crypto._is_default_home(tmp_path)

    def test_keychain_provider_keeps_secret_off_argv(self, monkeypatch):
        import subprocess as sp
        calls = []
        stored = {}

        def fake_run(argv, **kw):
            calls.append((argv, kw.get("input")))
            if argv[:2] == ["security", "-i"]:
                stored["hex"] = kw["input"].split('-w "')[1].split('"')[0]
                return sp.CompletedProcess(argv, 0, "", "")
            if argv[1] == "find-generic-password":
                if "hex" in stored:
                    return sp.CompletedProcess(argv, 0, stored["hex"] + "\n", "")
                return sp.CompletedProcess(argv, 44, "", "not found")
            raise AssertionError(argv)

        monkeypatch.setattr(_crypto.subprocess, "run", fake_run)
        p = _crypto.KeychainProvider(service="opendesk-test", account="memory-key")
        assert p.load() is None
        p.save(TEST_KEY)
        assert p.load() == TEST_KEY
        for argv, _ in calls:
            assert TEST_KEY.hex() not in " ".join(argv)
        assert any(inp and TEST_KEY.hex() in inp for _, inp in calls)


class TestEncryptedStore:
    def test_plaintext_thumbnails_when_disabled(self, tmp_path: Path):
        with MemoryStore(tmp_path, encrypt_thumbnails=False) as store:
            f = store.add(ts=time.time(), app="A", title="", text="x", thumb=b"jpeg")
            assert f.thumb_path.endswith(".jpg") and (store.dir / f.thumb_path).read_bytes() == b"jpeg"
            st = store.stats()
            assert (st.thumbnails, st.encrypted_thumbnails, st.encrypt_thumbnails) == (1, 0, False)

    def test_encrypted_text_round_trip_and_blind_search(self, tmp_path: Path):
        with MemoryStore(tmp_path, encrypt_text=True) as store:
            f = store.add(ts=_ts(0, 9), app="Terminal", title="zsh",
                          text="Error: connection refused on port 5432", thumb=b"jpg", text_source="ax")
            store.add(ts=_ts(0, 10), app="Slack", title="#ops", text="refused the connection politely", thumb=None)
            assert f.encrypted and f.text_source == "ax"
            raw = store._conn.execute("SELECT text, text_enc, text_hash FROM frames WHERE id=1").fetchone()
            assert "refused" not in raw["text"] and "refused" not in raw["text_enc"]
            assert raw["text_enc"].startswith("v1:")
            assert store.get(1).text == "Error: connection refused on port 5432"
            hits = store.search("refused")
            assert {h.id for h in hits} == {1, 2}
            assert "[refused]" in hits[0].snippet
            # Phrases must be contiguous even through the blind index.
            assert [h.id for h in store.search('"connection refused"')] == [1]
            assert [h.id for h in store.search("5432")] == [1]
            # Prefix search does not reach encrypted text …
            assert store.search("refu*") == []
            # … but app / title stay plaintext and searchable / filterable.
            assert [h.id for h in store.search("zsh")] == [1]
            assert [h.id for h in store.search("refused", app="slack")] == [2]
            st = store.stats()
            assert st.encrypted_text_frames == 2 and st.encrypt_text
            assert st.text_sources == {"ax": 1, "ocr": 1}

    def test_mixed_plaintext_and_encrypted_store(self, tmp_path: Path):
        seeded_store(tmp_path).close()  # v1-style plaintext text
        with MemoryStore(tmp_path, encrypt_text=True) as store:
            store.add(ts=_ts(0, 11), app="Terminal", title="psql", text="ECONNREFUSED again on 5432", thumb=b"j")
            hits = store.search("ECONNREFUSED")
            assert sorted(h.app for h in hits) == ["Terminal", "Terminal"]
            assert all("ECONNREFUSED" in h.snippet for h in hits)
            assert [h.app for h in store.search("invoice")] == ["Preview"]
            # AND finds nothing across frames → OR fallback spans both kinds.
            apps = {h.app for h in store.search("invoice again")}
            assert apps == {"Preview", "Terminal"}
            assert store.stats().encrypted_text_frames == 1

    def test_reindex_encrypts_then_decrypts(self, tmp_path: Path):
        with MemoryStore(tmp_path, encrypt_thumbnails=False) as store:
            store.add(ts=_ts(1), app="Notes", title="", text="password: hunter2 and invoice 42", thumb=b"jpg1")
            store.add(ts=_ts(0), app="Notes", title="", text="lunch plans", thumb=b"jpg2")
        cfg = MemoryConfig(encrypt_text=True, encrypt_thumbnails=True)
        with MemoryStore(tmp_path, config=cfg) as store:
            counts = store.reindex(redact=True)
            assert counts == {"frames": 2, "text_encrypted": 2, "text_decrypted": 0,
                              "thumbs_encrypted": 2, "thumbs_decrypted": 0, "redacted": 1, "failed": 0}
            f = store.get(1)
            assert f.encrypted and f.text == "password: [REDACTED:password] and invoice 42"
            assert f.thumb_path.endswith(".enc") and store.thumbnail(f) == b"jpg1"
            assert not list(store.thumbs_dir.rglob("*.jpg"))
            assert [h.id for h in store.search("invoice")] == [1]
            assert store.search("hunter2") == []
            # No plaintext left behind in free pages, FTS segments, or the WAL.
            for suffix in ("", "-wal"):
                p = tmp_path / "memory" / f"index.db{suffix}"
                if p.exists():
                    raw = p.read_bytes()
                    assert b"lunch" not in raw and b"hunter2" not in raw
            assert store.reindex()["text_encrypted"] == 0  # idempotent
        with MemoryStore(tmp_path, encrypt_text=False, encrypt_thumbnails=False) as store:
            counts = store.reindex()
            assert counts["text_decrypted"] == 2 and counts["thumbs_decrypted"] == 2
            f = store.get(2)
            assert not f.encrypted and f.text == "lunch plans"
            assert (store.dir / f.thumb_path).read_bytes() == b"jpg2"
            assert [h.id for h in store.search("lunch")] == [2]
            assert store.stats().encrypted_text_frames == 0

    def test_missing_key_degrades_gracefully(self, tmp_path: Path, monkeypatch):
        with MemoryStore(tmp_path, encrypt_text=True) as store:
            store.add(ts=time.time(), app="A", title="t", text="secret plans", thumb=b"jpg")
        monkeypatch.setattr(_crypto, "resolve_key_provider", lambda home=None: _crypto.StaticProvider(None))
        with MemoryStore(tmp_path) as store:
            f = store.get(1)
            assert f.encrypted and f.text == ""
            assert store.thumbnail(f) is None
            assert store.search("plans") == []
            assert store.reindex()["failed"] == 1
            assert store.get(1).encrypted  # left untouched
        monkeypatch.setattr(_crypto, "resolve_key_provider", lambda home=None: _crypto.StaticProvider(bytes(32)))
        with MemoryStore(tmp_path) as store:  # wrong key
            assert store.get(1).text == "" and store.thumbnail(store.get(1)) is None

    def test_v1_database_is_migrated(self, tmp_path: Path):
        mem = tmp_path / "memory"
        mem.mkdir(parents=True)
        db = _sqlite3.connect(str(mem / "index.db"))
        db.executescript("""
            CREATE TABLE frames (id INTEGER PRIMARY KEY, ts REAL NOT NULL, app TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '', text TEXT NOT NULL DEFAULT '', text_hash TEXT NOT NULL DEFAULT '',
                thumb_path TEXT, thumb_bytes INTEGER NOT NULL DEFAULT 0, width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO meta VALUES ('schema_version', '1');
            CREATE VIRTUAL TABLE frames_fts USING fts5(text, app, title, content='frames', content_rowid='id', tokenize='unicode61');
            CREATE TRIGGER frames_ai AFTER INSERT ON frames BEGIN
                INSERT INTO frames_fts(rowid, text, app, title) VALUES (new.id, new.text, new.app, new.title); END;
            CREATE TRIGGER frames_ad AFTER DELETE ON frames BEGIN
                INSERT INTO frames_fts(frames_fts, rowid, text, app, title) VALUES ('delete', old.id, old.text, old.app, old.title); END;
            CREATE TRIGGER frames_au AFTER UPDATE ON frames BEGIN
                INSERT INTO frames_fts(frames_fts, rowid, text, app, title) VALUES ('delete', old.id, old.text, old.app, old.title);
                INSERT INTO frames_fts(rowid, text, app, title) VALUES (new.id, new.text, new.app, new.title); END;
        """)
        (mem / "thumbs" / "2026-01-01").mkdir(parents=True)
        (mem / "thumbs" / "2026-01-01" / "1.jpg").write_bytes(b"oldjpeg")
        db.execute("INSERT INTO frames(ts, app, title, text, thumb_path, thumb_bytes) VALUES (?, 'Terminal', 'zsh', 'old ECONNREFUSED', 'thumbs/2026-01-01/1.jpg', 7)", (time.time() - 60,))
        db.execute("INSERT INTO frames(ts, app, title, text) VALUES (?, 'Finder', '', '')", (time.time() - 30,))
        db.commit()
        db.close()

        with MemoryStore(tmp_path) as store:
            cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(frames)")}
            assert {"text_source", "text_enc", "text_encrypted"} <= cols
            assert store._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "2"
            trig = store._conn.execute("SELECT sql FROM sqlite_master WHERE name='frames_au'").fetchone()[0]
            assert "UPDATE OF text, app, title" in trig
            assert store.get(1).text_source == "ocr" and store.get(2).text_source == "none"
            old = store.search("ECONNREFUSED")
            assert [h.id for h in old] == [1] and store.thumbnail(old[0]) == b"oldjpeg"
            new = store.add(ts=time.time(), app="Terminal", title="zsh", text="new ECONNREFUSED", thumb=b"j", text_source="ax")
            assert new.id == 3 and new.thumb_path.endswith(".enc")
            assert len(store.search("ECONNREFUSED")) == 2
            assert store.stats().text_sources == {"ocr": 1, "none": 1, "ax": 1}
        with MemoryStore(tmp_path) as store:  # re-open: migration is idempotent
            assert store.stats().frames == 3


class TestEventTrigger:
    def test_trigger_reason_decisions(self, tmp_path: Path):
        from opendesk.memory.recorder import FrontmostInfo
        rec = _v2_recorder(tmp_path, config=MemoryConfig(interval_seconds=30, min_interval_seconds=2, pause_hotkey=""))
        a = FrontmostInfo("Terminal", "zsh")
        assert rec.trigger_reason(a, 0) == "start"
        rec._last_key, rec._last_attempt = ("Terminal", "zsh"), 0.0
        assert rec.trigger_reason(a, 1) is None
        assert rec.trigger_reason(FrontmostInfo("Terminal", "vim"), 1.5) is None  # debounced
        assert rec.trigger_reason(FrontmostInfo("Terminal", "vim"), 2.0) == "title_change"
        assert rec.trigger_reason(FrontmostInfo("Safari", "News"), 2.5) == "app_change"
        assert rec.trigger_reason(a, 29.9) is None
        assert rec.trigger_reason(a, 30.0) == "interval"

    @pytest.mark.asyncio
    async def test_poll_captures_on_switch_and_idle_fallback(self, tmp_path: Path):
        clock, probe = _Clock(0.0), _Probe("Terminal", "zsh")
        same = _png(text_seed=5)
        frames = [(same, "", "")] * 3 + [(_png(color=(0, 0, 0)), "", "")] + [(same, "", "")] * 5
        cfg = MemoryConfig(interval_seconds=30, min_interval_seconds=2, pause_hotkey="", deny_apps=[])
        rec = _v2_recorder(tmp_path, frames=frames, config=cfg, probe=probe, clock=clock)
        assert await rec.poll() == "captured"           # start
        clock.t = 1.0
        assert await rec.poll() is None                 # nothing changed
        probe.cur = ("Safari", "News", 7)
        clock.t = 1.5
        assert await rec.poll() is None                 # debounced
        clock.t = 2.5
        assert await rec.poll() == "captured"           # app switch: kept even if pixels barely moved
        clock.t = 5.0
        probe.cur = ("Safari", "Mail", 7)
        assert await rec.poll() == "duplicate"          # title change, identical screen → deduped
        clock.t = 7.5
        probe.cur = ("Safari", "◐ Mail (3) — 45%", 7)
        assert await rec.poll() is None                 # spinner/counter-only title change ignored
        probe.cur = ("TextEdit", "Untitled", 9)
        clock.t = 8.0
        assert await rec.poll() == "captured"           # different pixels
        clock.t = 20.0
        assert await rec.poll() is None
        clock.t = 38.5
        assert await rec.poll() == "captured"           # idle fallback, screen changed
        assert rec._reasons == {"start": 1, "app_change": 2, "interval": 1}
        frames_ = rec.store.timeline(limit=10, newest_first=False)
        assert [f.app for f in frames_] == ["Terminal", "Safari", "TextEdit", "TextEdit"]

    def test_title_key_ignores_spinners_and_counters(self):
        from opendesk.memory.recorder import title_key
        assert title_key("opendesk — ◐ Screen memory — 80×24") == title_key("opendesk — ◑ Screen memory — 80×24")
        assert title_key("(3) Slack | #ops") == title_key("(12) Slack | #ops")
        assert title_key("Build 45%") == title_key("Build 46%")
        assert title_key("Inbox — Mail") != title_key("Drafts — Mail")

    @pytest.mark.asyncio
    async def test_poll_paused_does_not_probe(self, tmp_path: Path):
        probe = _Probe()
        rec = _v2_recorder(tmp_path, probe=probe)
        set_pause(tmp_path)
        assert await rec.poll() == "paused"
        assert probe.calls == 0 and rec.computer.captures == 0

    @pytest.mark.asyncio
    async def test_denied_switch_is_not_retried_every_poll(self, tmp_path: Path):
        clock, probe = _Clock(0.0), _Probe("1Password 8", "Vault")
        rec = _v2_recorder(tmp_path, probe=probe, clock=clock)
        assert (await rec.poll()).startswith("denied:")
        clock.t = 5
        assert await rec.poll() is None
        assert rec.computer.captures == 0

    @pytest.mark.asyncio
    async def test_run_loop_events_mode(self, tmp_path: Path):
        import asyncio
        import itertools
        apps = itertools.cycle(["Finder", "TextEdit", "Safari"])
        cfg = MemoryConfig(trigger="events", poll_seconds=0.2, min_interval_seconds=0, pause_hotkey="", deny_apps=[])
        frames = [(_png(color=(255, 255, 255) if i % 2 else (0, 0, 0)), "", "") for i in range(40)]
        from opendesk.memory.recorder import ScreenMemoryRecorder
        rec = ScreenMemoryRecorder(tmp_path, computer=_Computer(frames), config=cfg, ocr=lambda _: "",
                                   frontmost=lambda: (next(apps), "w"))
        task = asyncio.create_task(rec.run())
        await asyncio.sleep(1.1)
        state = _json.loads((tmp_path / "memory" / "daemon.json").read_text())
        assert state["trigger"] == "events"
        rec.stop()
        await asyncio.wait_for(task, timeout=3)
        assert rec.captured >= 3
        assert rec.text_sources["none"] == rec.captured

    @pytest.mark.asyncio
    async def test_run_loop_interval_mode(self, tmp_path: Path):
        import asyncio
        cfg = MemoryConfig(trigger="interval", interval_seconds=5, pause_hotkey="")
        rec = _recorder(tmp_path, [(_png(text_seed=1), "A", "")], config=cfg)
        task = asyncio.create_task(rec.run())
        await asyncio.sleep(0.3)
        rec.stop()
        await asyncio.wait_for(task, timeout=3)
        assert rec.captured == 1 and rec._reasons == {"interval": 1}

    def test_coerce_frontmost_shapes(self):
        from opendesk.memory.recorder import FrontmostInfo, _coerce_front

        class Obj:
            app_name, title, pid = "Mail", "Inbox", "12"
        assert _coerce_front({"app": "A", "title": "T", "pid": 3}) == FrontmostInfo("A", "T", 3)
        assert _coerce_front(("A", "T")) == FrontmostInfo("A", "T", None)
        assert _coerce_front(Obj()) == FrontmostInfo("Mail", "Inbox", 12)
        assert _coerce_front(None) == FrontmostInfo()


class TestAccessibilityText:
    @pytest.mark.asyncio
    async def test_ax_text_skips_ocr(self, tmp_path: Path):
        seen = {}
        ocr_calls = []

        def window_text(pid=None, max_chars=20000):
            seen.update(pid=pid, max_chars=max_chars)
            return "Quarterly report — revenue grew 12% quarter over quarter in EMEA"

        rec = _v2_recorder(tmp_path, window_text=window_text, probe=_Probe("Pages", "Report", 314),
                           ocr=lambda png: ocr_calls.append(1) or "ocr")
        assert await rec.tick() == "captured"
        f = rec.store.get(1)
        assert f.text_source == "ax" and f.text.startswith("Quarterly report")
        assert ocr_calls == [] and seen == {"pid": 314, "max_chars": 20000}
        assert [h.id for h in rec.store.search("EMEA")] == [1]

    @pytest.mark.asyncio
    async def test_short_or_missing_ax_text_falls_back_to_ocr(self, tmp_path: Path):
        rec = _v2_recorder(tmp_path, window_text=lambda **_: "OK", ocr=lambda _: "from ocr")
        assert await rec.tick() == "captured"
        assert rec.store.get(1).text_source == "ocr" and rec.store.get(1).text == "from ocr"
        rec2 = _v2_recorder(tmp_path / "b", window_text=lambda **_: None, ocr=lambda _: "(no text detected)")
        assert await rec2.tick() == "captured"
        assert rec2.store.get(1).text_source == "none"

    @pytest.mark.asyncio
    async def test_async_window_text_and_ocr_always(self, tmp_path: Path):
        async def window_text(pid=None, max_chars=20000):
            return "A" * 30 + " accessibility text body here"

        cfg = MemoryConfig(ocr_backend="always", pause_hotkey="")
        rec = _v2_recorder(tmp_path, window_text=window_text, config=cfg, ocr=lambda _: "pixels said this")
        assert await rec.tick() == "captured"
        f = rec.store.get(1)
        assert f.text_source == "ax" and f.text.endswith("pixels said this")

    @pytest.mark.asyncio
    async def test_window_text_errors_fall_back(self, tmp_path: Path):
        def boom(**_):
            raise RuntimeError("AX denied")
        rec = _v2_recorder(tmp_path, window_text=boom, ocr=lambda _: "ocr wins")
        assert await rec.tick() == "captured"
        assert rec.store.get(1).text_source == "ocr"

    @pytest.mark.asyncio
    async def test_secrets_redacted_before_storage(self, tmp_path: Path):
        secret = "ghp_" + "b" * 36
        rec = _v2_recorder(
            tmp_path, probe=_Probe("Terminal", f"export GH={secret}"),
            window_text=lambda **_: f"$ export GITHUB_TOKEN={secret}\n$ git push origin main --force",
        )
        assert await rec.tick() == "captured"
        rec.store.close()
        raw = (tmp_path / "memory" / "index.db").read_bytes()
        assert secret.encode() not in raw
        with MemoryStore(tmp_path) as store:
            f = store.get(1)
            assert "[REDACTED:github_token]" in f.text and "[REDACTED:github_token]" in f.title
        assert rec.redactions == 2
        rec2 = _v2_recorder(tmp_path / "b", config=MemoryConfig(redact=False, pause_hotkey=""),
                            window_text=lambda **_: f"token {secret} " + "x" * 40)
        await rec2.tick()
        assert secret in rec2.store.get(1).text

    def test_native_module_only_used_for_the_local_screen(self, tmp_path: Path, monkeypatch):
        import sys
        import types
        from opendesk.memory import recorder as rmod
        fake = types.ModuleType("opendesk.computer.macos_native")
        fake.window_text = lambda pid=None, max_chars=20000: "native"
        fake.frontmost = lambda: {"app": "Native", "title": "", "pid": 1}
        monkeypatch.setitem(sys.modules, "opendesk.computer.macos_native", fake)
        monkeypatch.setattr(rmod.platform, "system", lambda: "Darwin")
        injected = _recorder(tmp_path, [(_png(), "A", "")])
        assert injected._resolve_window_text() is None
        local = rmod.ScreenMemoryRecorder(tmp_path / "l", config=MemoryConfig(pause_hotkey=""))
        assert local._resolve_window_text() is fake.window_text
        monkeypatch.delitem(sys.modules, "opendesk.computer.macos_native")
        monkeypatch.setattr(rmod, "_native", lambda name: None)
        local2 = rmod.ScreenMemoryRecorder(tmp_path / "m", config=MemoryConfig(pause_hotkey=""))
        assert local2._resolve_window_text() is None


class TestCliV2:
    def _run(self, *args, home):
        import subprocess
        import sys
        return subprocess.run(
            [sys.executable, "-m", "opendesk.cli", "memory", *args, "--home", str(home)],
            capture_output=True, text=True, timeout=60,
        )

    def test_config_flags(self, tmp_path: Path):
        r = self._run("config", "--trigger", "interval", "--encrypt-text", "on", "--redact", "off",
                      "--poll", "0.5", "--min-interval", "3", home=tmp_path)
        assert r.returncode == 0, r.stderr
        assert "reindex" in r.stdout
        cfg = load_config(tmp_path)
        assert (cfg.trigger, cfg.encrypt_text, cfg.redact) == ("interval", True, False)
        assert (cfg.poll_seconds, cfg.min_interval_seconds) == (0.5, 3.0)
        r = self._run("config", "--trigger", "sometimes", home=tmp_path)
        assert r.returncode != 0

    def test_reindex_show_and_status(self, tmp_path: Path, monkeypatch):
        # Subprocesses use the file key store, so this test must too.
        monkeypatch.setattr(_crypto, "resolve_key_provider", _REAL_RESOLVE_KEY_PROVIDER)
        from opendesk.memory.recorder import make_thumbnail
        jpeg, _, _ = make_thumbnail(_png(size=(200, 100)), width=100)
        with MemoryStore(tmp_path, encrypt_thumbnails=False) as store:
            store.add(ts=_ts(0, 9), app="Terminal", title="zsh", text="ECONNREFUSED 5432", thumb=jpeg)
        save_config(MemoryConfig(encrypt_text=True), tmp_path)
        r = self._run("reindex", "--yes", home=tmp_path)
        assert r.returncode == 0, r.stderr
        assert "text encrypted 1" in r.stdout and "thumbnails encrypted 1" in r.stdout
        assert not list((tmp_path / "memory" / "thumbs").rglob("*.jpg"))
        out = tmp_path / "out.jpg"
        r = self._run("show", "1", "--save", str(out), home=tmp_path)
        assert r.returncode == 0, r.stderr
        assert "ECONNREFUSED 5432" in r.stdout and "encrypted at rest" in r.stdout
        assert out.read_bytes() == jpeg
        r = self._run("search", "econnrefused", home=tmp_path)
        assert "[ECONNREFUSED]" in r.stdout
        r = self._run("status", home=tmp_path)
        assert "trigger:   events" in r.stdout
        assert "thumbnails on (1/1 stored encrypted)" in r.stdout and "key: file" in r.stdout
