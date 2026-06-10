"""Tests for persistent trading journal and replay helpers."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from daytrading.journal.store import SQLITE_BUSY_TIMEOUT_MS, TradingJournal
from daytrading.models import Bar, Timeframe


def _bar(symbol: str = "TST", close: float = 5.0) -> Bar:
    return Bar(
        symbol=symbol,
        ts=datetime.now(timezone.utc),
        open=close - 0.02,
        high=close + 0.03,
        low=close - 0.03,
        close=close,
        volume=10_000,
        timeframe=Timeframe.MIN_1,
    )


class TestTradingJournal:
    def test_sqlite_connection_is_configured_for_live_dashboard_reads(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))

        busy_timeout = journal._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        journal_mode = journal._conn.execute("PRAGMA journal_mode").fetchone()[0]

        assert busy_timeout == SQLITE_BUSY_TIMEOUT_MS
        assert journal_mode.lower() == "wal"
        journal.close()

    def test_record_and_load_events(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        journal.record("trade_fill", {"symbol": "AIIO", "price": 5.5, "qty": 100})
        journal.record("mistake", {"symbol": "AIIO", "reason": "false breakout"})

        events = journal.load_events()
        assert len(events) == 2
        assert events[0]["type"] == "trade_fill"
        assert events[1]["type"] == "mistake"

    def test_skipped_event_types_are_not_recorded(self, tmp_path) -> None:
        journal = TradingJournal(
            base_dir=str(tmp_path / "journal"),
            daily_prune_enabled=False,
            skipped_event_types=["classification", "market_regime"],
        )
        journal.record("classification", {"symbol": "AIIO", "style": "scalping"})
        journal.record("market_regime", {"cycle": 1, "symbols": {"AIIO": {}}})
        journal.record("trade_fill", {"symbol": "AIIO", "price": 5.5})

        events = journal.load_events()

        assert [e["type"] for e in events] == ["trade_fill"]

    def test_candle_snapshot(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        bars = [_bar(close=5.0 + i * 0.01) for i in range(5)]
        snap = journal.candle_snapshot(bars, limit=3)
        assert len(snap) == 3
        assert snap[-1]["close"] > snap[0]["close"]
        assert snap[-1]["timeframe"] == "1m"

    def test_candle_snapshots_are_not_recorded_by_default(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        bars = journal.candle_snapshot([_bar()], limit=1)
        journal.record("scan_hit", {"symbol": "AIIO", "candle_snapshot": bars})

        events = journal.load_events()
        assert "candle_snapshot" not in events[0]["payload"]

        conn = sqlite3.connect(journal.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM candle_snapshots").fetchone()[0]
        finally:
            conn.close()
        assert count == 0

    def test_candle_snapshots_can_be_enabled_for_debugging(self, tmp_path) -> None:
        journal = TradingJournal(
            base_dir=str(tmp_path / "journal"),
            record_candle_snapshots=True,
        )
        bars = journal.candle_snapshot([_bar()], limit=1)
        journal.record("scan_hit", {"symbol": "AIIO", "candle_snapshot": bars})

        events = journal.load_events()
        assert len(events[0]["payload"]["candle_snapshot"]) == 1

        conn = sqlite3.connect(journal.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM candle_snapshots").fetchone()[0]
        finally:
            conn.close()
        assert count == 1

    def test_save_screenshot_from_base64(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        # 1x1 transparent PNG
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAusB9Y9x2jkAAAAASUVORK5CYII="
        )
        meta = journal.save_screenshot("AIIO", image_b64=png_b64, context={"setup": "vwap_pullback"})
        assert meta["symbol"] == "AIIO"
        assert meta["path"].endswith(".png")

        events = journal.load_events()
        assert any(e["type"] == "screenshot" for e in events)

    def test_replay_frames_filters_event_types(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        journal.record("classification", {"symbol": "AIIO", "style": "scalping"})
        journal.record("trade_fill", {"symbol": "AIIO", "price": 5.6})
        journal.record("internal_debug", {"x": 1})

        frames = journal.replay_frames()
        assert len(frames) == 2
        kinds = {f["type"] for f in frames}
        assert "classification" in kinds
        assert "trade_fill" in kinds
        assert "internal_debug" not in kinds

    def test_replay_frames_include_entry_decisions(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        journal.record(
            "entry_decision",
            {
                "symbol": "AIIO",
                "stage": "final_entry_guard",
                "passed": False,
                "blocked_layer": "entry_guard",
                "reason": "entry score too low",
            },
        )

        frames = journal.replay_frames()

        assert len(frames) == 1
        assert frames[0]["type"] == "entry_decision"
        assert frames[0]["payload"]["stage"] == "final_entry_guard"

    def test_prune_removes_old_journal_rows(self, tmp_path) -> None:
        journal = TradingJournal(
            base_dir=str(tmp_path / "journal"),
            retention_days=0,
            prune_on_start=False,
        )
        old_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        journal.record("trade_fill", {"symbol": "OLD", "price": 1.0}, ts=old_ts)
        journal.record("trade_fill", {"symbol": "NEW", "price": 2.0})

        deleted = journal.prune(retention_days=2)

        assert deleted == 1
        events = journal.load_events()
        assert [e["payload"]["symbol"] for e in events] == ["NEW"]

    def test_prune_handles_large_old_event_sets(self, tmp_path) -> None:
        journal = TradingJournal(
            base_dir=str(tmp_path / "journal"),
            retention_days=0,
            prune_on_start=False,
        )
        old_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for idx in range(1100):
            journal.record("trade_fill", {"symbol": "OLD", "price": idx}, ts=old_ts)
        journal.record("trade_fill", {"symbol": "NEW", "price": 2.0})

        deleted = journal.prune(retention_days=2)

        assert deleted == 1100
        assert [e["payload"]["symbol"] for e in journal.load_events()] == ["NEW"]

    def test_daily_prune_runs_when_startup_prune_is_off(self, tmp_path) -> None:
        journal = TradingJournal(
            base_dir=str(tmp_path / "journal"),
            retention_days=2,
            prune_on_start=False,
            daily_prune_enabled=True,
            async_prune=False,
        )
        old_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        journal.record("trade_fill", {"symbol": "OLD", "price": 1.0}, ts=old_ts)
        journal.record("trade_fill", {"symbol": "NEW", "price": 2.0})

        assert [e["payload"]["symbol"] for e in journal.load_events()] == ["NEW"]

    def test_daily_prune_reads_retention_days_from_env(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("DAYTRADING_JOURNAL_DIR", str(tmp_path / "journal"))
        monkeypatch.setenv("DAYTRADING_JOURNAL_RETENTION_DAYS", "2")
        monkeypatch.setenv("DAYTRADING_JOURNAL_PRUNE_ON_START", "false")
        monkeypatch.setenv("DAYTRADING_JOURNAL_DAILY_PRUNE_ENABLED", "true")
        monkeypatch.setenv("DAYTRADING_JOURNAL_ASYNC_PRUNE", "false")

        journal = TradingJournal()
        old_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        journal.record("trade_fill", {"symbol": "OLD", "price": 1.0}, ts=old_ts)
        journal.record("trade_fill", {"symbol": "NEW", "price": 2.0})

        assert [e["payload"]["symbol"] for e in journal.load_events()] == ["NEW"]
