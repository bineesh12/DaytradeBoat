"""Tests for 10-second execution timing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from daytrading.models import Bar, ScanResult, SignalAction, Timeframe, TradeSignal
from daytrading.strategy.execution_timer import ExecutionTimer


def _signal(symbol: str = "TST", price: float = 5.0) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        reason="test_setup",
    )


def _hot_hod_signal(symbol: str = "HOT", price: float = 3.15) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        reason="hot vwap setup",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="vwap_pullback",
            ts=datetime.now(timezone.utc),
            score=30.0,
            criteria={
                "pattern": "vwap_pullback",
                "direction": "up",
                "close": price,
                "rally_pct": 31.8,
                "volume": 966_141,
            },
        ),
    )


def _hot_momentum_signal(symbol: str = "MOMO", price: float = 4.84) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        reason="momentum burst",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="momentum_burst",
            ts=datetime.now(timezone.utc),
            score=30.0,
            criteria={
                "pattern": "momentum_burst",
                "direction": "up",
                "close": price,
                "volume": 300_000,
            },
        ),
    )


def _10s_bar(
    symbol: str,
    ts: datetime,
    o: float,
    c: float,
    h: float | None = None,
    l: float | None = None,
) -> Bar:
    hi = h if h is not None else max(o, c) + 0.02
    lo = l if l is not None else min(o, c) - 0.02
    return Bar(
        symbol=symbol,
        ts=ts,
        open=o,
        high=hi,
        low=lo,
        close=c,
        volume=1000,
        timeframe=Timeframe.SEC_10,
    )


class TestExecutionTimerQueue:
    def test_disabled_returns_false(self) -> None:
        timer = ExecutionTimer(enabled=False)
        assert timer.queue(_signal()) is False
        assert timer.pending_symbols == []

    def test_queue_adds_pending(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        assert timer.queue(_signal("AAA")) is True
        assert timer.pending_symbols == ["AAA"]

    def test_duplicate_symbol_not_double_queued(self) -> None:
        timer = ExecutionTimer(enabled=True)
        timer.queue(_signal("AAA"))
        timer.queue(_signal("AAA"))
        assert timer.pending_symbols == ["AAA"]


class TestExecutionTimerTriggers:
    def test_micro_pullback_bounce_releases(self) -> None:
        timer = ExecutionTimer(max_wait_bars=5, enabled=True)
        timer.queue(_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("TST", base, 5.10, 5.05)) is None  # red
        released = timer.on_10s_bar(_10s_bar("TST", base + timedelta(seconds=10), 5.04, 5.12))
        assert released is not None
        assert released.symbol == "TST"
        assert "TST" not in timer.pending_symbols

    def test_strong_green_bar_releases(self) -> None:
        timer = ExecutionTimer(max_wait_bars=5, enabled=True)
        timer.queue(_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)
        # Body > 30% of range, green
        bar = _10s_bar("TST", base, 5.00, 5.08, h=5.10, l=4.99)
        released = timer.on_10s_bar(bar)
        assert released is not None
        assert released.symbol == "TST"

    def test_max_wait_bars_fallback(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)
        # Doji / flat bars — no bounce trigger
        b = _10s_bar("TST", base, 5.00, 5.00, h=5.01, l=4.99)
        assert timer.on_10s_bar(b) is None
        released = timer.on_10s_bar(_10s_bar("TST", base + timedelta(seconds=10), 5.00, 5.00, h=5.01, l=4.99))
        assert released is not None

    def test_no_pending_returns_none(self) -> None:
        timer = ExecutionTimer(enabled=True)
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)
        assert timer.on_10s_bar(_10s_bar("TST", base, 5, 5.1)) is None


class TestExecutionTimerTimeouts:
    def test_check_timeouts_releases_stale_pending(self) -> None:
        timer = ExecutionTimer(max_wait_bars=1, enabled=True)
        sig = _signal()
        timer.queue(sig)
        # Force old queue time
        pending = timer._pending["TST"]
        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=25)

        released = timer.check_timeouts()
        assert len(released) == 1
        assert released[0].symbol == "TST"

    def test_hot_hod_signal_uses_short_timeout(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _hot_hod_signal()
        timer.queue(sig)
        pending = timer._pending["HOT"]
        assert pending.max_wait_seconds == 12.0

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=13)
        released = timer.check_timeouts()
        assert len(released) == 1
        assert released[0].symbol == "HOT"

    def test_hot_momentum_signal_uses_short_timeout(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _hot_momentum_signal()
        timer.queue(sig)
        pending = timer._pending["MOMO"]
        assert pending.max_wait_seconds == 10.0

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)
        released = timer.check_timeouts()
        assert len(released) == 1
        assert released[0].symbol == "MOMO"

    def test_normal_signal_keeps_standard_timeout(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _signal()
        timer.queue(sig)
        pending = timer._pending["TST"]
        assert pending.max_wait_seconds is None

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=13)
        assert timer.check_timeouts() == []

    def test_cancel_removes_pending(self) -> None:
        timer = ExecutionTimer(enabled=True)
        timer.queue(_signal())
        timer.cancel("TST")
        assert timer.pending_symbols == []
