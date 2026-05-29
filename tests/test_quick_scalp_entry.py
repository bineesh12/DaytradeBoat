from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from daytrading.models import Bar, Quote, Side, Tick, Timeframe
from daytrading.runner import AlpacaRunner


TS = datetime(2026, 5, 29, 13, 30, tzinfo=timezone.utc)


class _QuickScalpRunner:
    _check_quick_scalp_entry = AlpacaRunner._check_quick_scalp_entry
    _quick_scalp_tick_rr = AlpacaRunner._quick_scalp_tick_rr

    def __init__(self) -> None:
        self._quote_buffer = defaultdict(lambda: deque(maxlen=100))
        self._tick_buffer = defaultdict(lambda: deque(maxlen=200))


def _bar(
    close: float,
    *,
    idx: int,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 75_000,
) -> Bar:
    return Bar(
        symbol="OLOX",
        ts=TS + timedelta(minutes=idx),
        open=close if open_ is None else open_,
        high=close + 0.10 if high is None else high,
        low=close - 0.10 if low is None else low,
        close=close,
        volume=volume,
        timeframe=Timeframe.MIN_1,
    )


def _add_clean_execution_context(runner: _QuickScalpRunner, symbol: str = "OLOX") -> None:
    for i in range(5):
        runner._quote_buffer[symbol].append(
            Quote(
                symbol=symbol,
                ts=TS + timedelta(minutes=12, seconds=i),
                bid=8.07,
                ask=8.11,
                bid_size=1500,
                ask_size=1300,
            )
        )
    for i in range(20):
        runner._tick_buffer[symbol].append(
            Tick(
                symbol=symbol,
                ts=TS + timedelta(minutes=12, seconds=i),
                price=8.00 + min(i, 9) * 0.01,
                size=1000,
                side=Side.BUY if i < 16 else Side.SELL,
            )
        )


def test_quick_scalp_allows_recent_hod_push_when_session_open_is_misleading() -> None:
    runner = _QuickScalpRunner()
    _add_clean_execution_context(runner)
    bars = [
        _bar(9.20, idx=0, open_=9.54, high=9.70, low=9.00, volume=40_000),
        _bar(6.40, idx=1, high=6.60, low=6.25, volume=90_000),
        _bar(6.65, idx=2, high=6.75, low=6.32, volume=95_000),
        _bar(6.55, idx=3, high=6.75, low=6.40, volume=70_000),
        _bar(6.85, idx=4, high=6.95, low=6.55, volume=85_000),
        _bar(6.70, idx=5, high=6.95, low=6.45, volume=75_000),
        _bar(6.92, idx=6, high=7.10, low=6.55, volume=90_000),
        _bar(7.95, idx=7, high=8.15, low=7.45, volume=130_000),
        _bar(8.25, idx=8, high=8.50, low=7.92, volume=140_000),
        _bar(8.09, idx=9, high=8.42, low=7.86, volume=150_000),
    ]

    assert runner._check_quick_scalp_entry("OLOX", bars) is None


def test_quick_scalp_still_rejects_weak_move_even_near_hod() -> None:
    runner = _QuickScalpRunner()
    _add_clean_execution_context(runner)
    bars = [
        _bar(8.30, idx=0, open_=10.00, high=8.55, low=8.10, volume=90_000),
        _bar(8.28, idx=1, high=8.45, low=8.05, volume=90_000),
        _bar(8.35, idx=2, high=8.50, low=8.10, volume=90_000),
        _bar(8.32, idx=3, high=8.48, low=8.05, volume=90_000),
        _bar(8.40, idx=4, high=8.56, low=8.15, volume=90_000),
        _bar(8.38, idx=5, high=8.54, low=8.12, volume=90_000),
    ]

    reject = runner._check_quick_scalp_entry("OLOX", bars)

    assert reject is not None
    assert "quick scalp movement too small day=" in reject
    assert "recent=" in reject


def test_quick_scalp_allows_big_volume_runner_just_under_50k_recent_volume() -> None:
    runner = _QuickScalpRunner()
    _add_clean_execution_context(runner, symbol="IOTR")
    bars = [
        _bar(3.40, idx=0, open_=3.40, high=3.45, low=3.36, volume=900_000),
        _bar(3.70, idx=1, open_=3.40, high=3.75, low=3.38, volume=900_000),
        _bar(3.95, idx=2, open_=3.70, high=4.00, low=3.65, volume=900_000),
        _bar(4.10, idx=3, open_=3.95, high=4.19, low=3.90, volume=900_000),
        _bar(4.30, idx=4, open_=4.10, high=4.35, low=4.05, volume=16_000),
        _bar(4.60, idx=5, open_=4.30, high=4.62, low=4.25, volume=16_000),
        _bar(5.00, idx=6, open_=4.60, high=5.00, low=4.55, volume=16_263),
    ]

    assert runner._check_quick_scalp_entry("IOTR", bars) is None


def test_quick_scalp_tick_rr_uses_tactical_risk_for_hod_scalp() -> None:
    runner = _QuickScalpRunner()
    _add_clean_execution_context(runner)
    bars = [
        _bar(7.95, idx=0, high=8.15, low=7.45, volume=130_000),
        _bar(8.25, idx=1, high=8.50, low=7.92, volume=140_000),
        _bar(8.09, idx=2, high=8.42, low=7.86, volume=150_000),
    ]

    rr = runner._quick_scalp_tick_rr("OLOX", bars, alert_price=8.09)

    assert rr is not None
    price, stop, target, note = rr
    assert stop < price < target
    assert (price - stop) / price <= 0.04
    assert "risk=" in note
