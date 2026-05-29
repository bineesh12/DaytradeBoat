"""Tests for multi-timeframe bar aggregation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from daytrading.data.bar_aggregator import BarAggregator, _5m_bucket_key
from daytrading.models import Bar, Side, Tick, Timeframe


def _1m_bar(symbol: str, ts: datetime, o: float, h: float, l: float, c: float, v: float = 10_000) -> Bar:
    return Bar(symbol=symbol, ts=ts, open=o, high=h, low=l, close=c, volume=v, timeframe=Timeframe.MIN_1)


def _tick(symbol: str, ts: datetime, price: float, size: float = 100) -> Tick:
    return Tick(symbol=symbol, ts=ts, price=price, size=size, side=Side.BUY)


class Test5mBucketKey:
    def test_buckets_align_to_five_minutes(self) -> None:
        ts = datetime(2026, 5, 19, 14, 32, 0, tzinfo=timezone.utc)
        assert _5m_bucket_key(ts) == "2026-05-19_14:30"
        assert _5m_bucket_key(ts + timedelta(minutes=2)) == "2026-05-19_14:30"
        assert _5m_bucket_key(ts + timedelta(minutes=5)) == "2026-05-19_14:35"


class TestBuild5mBars:
    def test_fewer_than_five_1m_bars_returns_cached_or_empty(self) -> None:
        agg = BarAggregator()
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)
        bars = [_1m_bar("TST", base + timedelta(minutes=i), 5, 5.1, 4.9, 5.05) for i in range(4)]
        assert agg.build_5m_bars("TST", bars) == []

    def test_ten_1m_bars_produce_two_5m_bars(self) -> None:
        agg = BarAggregator()
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)
        bars = []
        for i in range(10):
            p = 5.0 + i * 0.05
            bars.append(_1m_bar("TST", base + timedelta(minutes=i), p, p + 0.03, p - 0.02, p + 0.01))

        result = agg.build_5m_bars("TST", bars)
        assert len(result) == 2
        assert all(b.timeframe == Timeframe.MIN_5 for b in result)
        assert result[0].open == pytest.approx(5.0)
        assert result[0].close == pytest.approx(bars[4].close)
        assert result[0].volume == sum(b.volume for b in bars[:5])
        assert result[1].open == pytest.approx(bars[5].open)

    def test_update_all_5m(self) -> None:
        agg = BarAggregator()
        base = datetime(2026, 5, 19, 9, 30, 0, tzinfo=timezone.utc)
        universe = {
            "A": [_1m_bar("A", base + timedelta(minutes=i), 10, 10.1, 9.9, 10.05) for i in range(10)],
            "B": [_1m_bar("B", base + timedelta(minutes=i), 3, 3.1, 2.9, 3.05) for i in range(10)],
        }
        agg.update_all_5m(universe)
        assert len(agg.get_5m_bars("A")) == 2
        assert len(agg.get_5m_bars("B")) == 2


class Test10sBarsFromTicks:
    def test_first_tick_does_not_complete_bar(self) -> None:
        agg = BarAggregator()
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)
        assert agg.on_tick(_tick("TST", base, 5.0)) is None
        assert agg.get_10s_bars("TST") == []

    def test_window_closes_after_ten_seconds(self) -> None:
        agg = BarAggregator()
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)
        agg.on_tick(_tick("TST", base, 5.00))
        completed = agg.on_tick(_tick("TST", base + timedelta(seconds=10), 5.10))
        assert completed is not None
        assert completed.timeframe == Timeframe.SEC_10
        assert completed.open == pytest.approx(5.00)
        assert completed.close == pytest.approx(5.00)
        assert completed.high >= completed.low
        assert len(agg.get_10s_bars("TST")) == 1

    def test_get_latest_10s(self) -> None:
        agg = BarAggregator()
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)
        for i in range(4):
            agg.on_tick(_tick("TST", base + timedelta(seconds=i * 10), 5.0 + i * 0.01))
            agg.on_tick(_tick("TST", base + timedelta(seconds=i * 10 + 10), 5.0 + i * 0.02))
        latest = agg.get_latest_10s("TST", count=2)
        assert len(latest) <= 2

    def test_clear_symbol(self) -> None:
        agg = BarAggregator()
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)
        agg.on_tick(_tick("TST", base, 5.0))
        agg.on_tick(_tick("TST", base + timedelta(seconds=10), 5.1))
        agg.clear_symbol("TST")
        assert agg.get_10s_bars("TST") == []
        assert agg.get_5m_bars("TST") == []
