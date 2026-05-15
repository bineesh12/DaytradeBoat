"""Tests for persistent trading journal and replay helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from daytrading.journal.store import TradingJournal
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
    def test_record_and_load_events(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        journal.record("trade_fill", {"symbol": "AIIO", "price": 5.5, "qty": 100})
        journal.record("mistake", {"symbol": "AIIO", "reason": "false breakout"})

        events = journal.load_events()
        assert len(events) == 2
        assert events[0]["type"] == "trade_fill"
        assert events[1]["type"] == "mistake"

    def test_candle_snapshot(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        bars = [_bar(close=5.0 + i * 0.01) for i in range(5)]
        snap = journal.candle_snapshot(bars, limit=3)
        assert len(snap) == 3
        assert snap[-1]["close"] > snap[0]["close"]
        assert snap[-1]["timeframe"] == "1m"

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

