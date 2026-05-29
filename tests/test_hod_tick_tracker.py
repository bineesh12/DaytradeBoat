"""Tests for HODTickTracker on live trades."""

from datetime import datetime, timezone

from daytrading.models import Bar, Side, Tick
from daytrading.scanner.hod_momentum.alert_store import HODAlertStore
from daytrading.scanner.hod_momentum.tick_tracker import HODTickTracker


class _FloatChecker:
    def __init__(self, floats: dict) -> None:
        self._floats = floats

    def get_float(self, symbol: str):
        return self._floats.get(symbol)

    def get_float_cached(self, symbol: str):
        return self._floats.get(symbol)


def _bar(i: int, high: float, vol: float = 100_000) -> Bar:
    ts = datetime(2026, 5, 18, 14, i, 0, tzinfo=timezone.utc)
    return Bar(
        symbol="POP",
        ts=ts,
        open=high - 0.1,
        high=high,
        low=high - 0.2,
        close=high - 0.05,
        volume=vol,
    )


def test_new_hod_emits_tick_alert_after_bar_seed() -> None:
    store = HODAlertStore(max_rows=50)
    rows: list = []
    store.set_on_change(rows.extend)

    tracker = HODTickTracker(
        store,
        float_checker=_FloatChecker({"POP": 10_000_000}),
        min_price=2.0,
        max_price=20.0,
        max_float=20_000_000,
        min_day_volume=200_000,
        volume_surge_ratio=1.5,
        tick_cooldown_seconds=0.0,
        known_symbols={"POP"},
    )

    bars = [_bar(i, 5.0 + i * 0.01, vol=80_000) for i in range(5)]
    tracker.update_session_from_bars("POP", bars)

    ts = datetime(2026, 5, 18, 14, 10, 0, tzinfo=timezone.utc)
    tracker.on_trade(Tick(symbol="POP", price=5.2, size=30_000, ts=ts, side=Side.BUY))
    tracker.on_trade(Tick(symbol="POP", price=5.6, size=40_000, ts=ts, side=Side.BUY))

    assert len(rows) >= 1
    row = rows[-1]
    assert row["symbol"] == "POP"
    assert row["alert_name"] == "New HOD Breakout"
    assert row["source"] == "tick"
    assert row["day_volume"] >= 200_000


def test_no_alert_without_bar_seed() -> None:
    store = HODAlertStore(max_rows=50)
    rows: list = []
    store.set_on_change(rows.extend)

    tracker = HODTickTracker(
        store,
        float_checker=_FloatChecker({"POP": 10_000_000}),
        tick_cooldown_seconds=0.0,
        known_symbols={"POP"},
    )

    ts = datetime(2026, 5, 18, 14, 0, 0, tzinfo=timezone.utc)
    tracker.on_trade(Tick(symbol="POP", price=5.0, size=30_000, ts=ts, side=Side.BUY))
    tracker.on_trade(Tick(symbol="POP", price=5.5, size=40_000, ts=ts, side=Side.BUY))

    assert rows == []


def test_high_float_skipped() -> None:
    store = HODAlertStore(max_rows=50)
    rows: list = []
    store.set_on_change(rows.extend)

    tracker = HODTickTracker(
        store,
        float_checker=_FloatChecker({"BIG": 50_000_000}),
        max_float=20_000_000,
        tick_cooldown_seconds=0.0,
        known_symbols={"BIG"},
    )

    bars = [_bar(i, 5.0, vol=100_000) for i in range(5)]
    tracker.update_session_from_bars("BIG", bars)

    ts = datetime(2026, 5, 18, 14, 0, 0, tzinfo=timezone.utc)
    tracker.on_trade(Tick(symbol="BIG", price=5.0, size=30_000, ts=ts, side=Side.BUY))
    tracker.on_trade(Tick(symbol="BIG", price=6.0, size=40_000, ts=ts, side=Side.BUY))

    assert rows == []


def test_low_day_volume_skipped() -> None:
    store = HODAlertStore(max_rows=50)
    rows: list = []
    store.set_on_change(rows.extend)

    tracker = HODTickTracker(
        store,
        float_checker=_FloatChecker({"LOW": 5_000_000}),
        min_day_volume=500_000,
        tick_cooldown_seconds=0.0,
        known_symbols={"LOW"},
    )

    bars = [_bar(i, 5.0, vol=10_000) for i in range(3)]
    tracker.update_session_from_bars("LOW", bars)

    ts = datetime(2026, 5, 18, 14, 0, 0, tzinfo=timezone.utc)
    tracker.on_trade(Tick(symbol="LOW", price=5.5, size=5_000, ts=ts, side=Side.BUY))

    assert rows == []
