"""Tests for 10-second execution timing."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from daytrading.models import Bar, ScanResult, Side, SignalAction, Tick, Timeframe, TradeSignal
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


def _veru_vwap_signal(symbol: str = "VERU", price: float = 5.8878) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=147,
        entry_price=price,
        stop_loss=5.56,
        reason="vwap pullback",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="vwap_pullback",
            ts=datetime.now(timezone.utc),
            score=46.0,
            criteria={
                "pattern": "vwap_pullback",
                "direction": "up",
                "close": price,
                "vwap": 5.5529,
                "pullback_low": 5.58,
                "stop_price": 5.56,
                "rally_pct": 31.8,
                "volume": 794_641,
            },
        ),
    )


def _bgms_vwap_signal(symbol: str = "BGMS", price: float = 2.93) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=237,
        entry_price=price,
        stop_loss=2.72,
        reason="vwap pullback",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="vwap_pullback",
            ts=datetime.now(timezone.utc),
            score=16.3,
            criteria={
                "pattern": "vwap_pullback",
                "direction": "up",
                "close": price,
                "vwap": 2.74,
                "pullback_low": 2.74,
                "stop_price": 2.72,
                "rally_pct": 9.7,
                "volume": 402_530,
            },
        ),
    )


def _vwap_reclaim_scout_signal(symbol: str = "AIIO", price: float = 2.67) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=107,
        entry_price=price,
        stop_loss=2.53,
        reason="vwap reclaim scout",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="vwap_pullback",
            ts=datetime.now(timezone.utc),
            score=77.0,
            criteria={
                "pattern": "vwap_pullback",
                "setup_tier": "A+ setup",
                "entry_tier": "vwap_reclaim_scout",
                "direction": "up",
                "close": price,
                "vwap": 2.60,
                "base_low": 2.55,
                "stop_price": 2.53,
                "volume": 3_700_000,
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


def _elite_momentum_signal(symbol: str = "ANY", price: float = 4.94) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=192,
        entry_price=price,
        reason="momentum burst",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="momentum_burst",
            ts=datetime.now(timezone.utc),
            score=97.0,
            criteria={
                "pattern": "momentum_burst",
                "direction": "up",
                "close": price,
                "volume": 4_422_000,
            },
        ),
    )


def _foxx_momentum_signal(symbol: str = "FOXX", price: float = 6.4801) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=87,
        entry_price=price,
        reason="momentum burst",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="momentum_burst",
            ts=datetime.now(timezone.utc),
            score=3.3509,
            criteria={
                "pattern": "momentum_burst",
                "direction": "up",
                "close": price,
                "volume": 1_276_633,
                "burst_pct": 3.3509,
                "velocity": 0.07,
            },
        ),
    )


def _foxx_opening_range_signal(symbol: str = "FOXX", price: float = 5.38) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=333,
        entry_price=price,
        reason="opening range breakout",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="opening_range_breakout",
            ts=datetime.now(timezone.utc),
            score=3.6,
            criteria={
                "pattern": "opening_range_breakout",
                "direction": "up",
                "close": price,
                "volume": 1_105_000,
            },
        ),
    )


def _bgms_opening_range_signal(symbol: str = "BGMS", price: float = 3.26) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=416,
        entry_price=price,
        stop_loss=2.98,
        reason="opening range breakout",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="opening_range_breakout",
            ts=datetime.now(timezone.utc),
            score=31.0,
            criteria={
                "pattern": "opening_range_breakout",
                "direction": "up",
                "close": price,
                "breakout_level": 2.98,
                "orb_high": 3.16,
                "orb_low": 2.58,
                "stop_price": 2.98,
                "volume": 2_988_000,
            },
        ),
    )


def _mnts_momentum_signal(symbol: str = "MNTS", price: float = 15.5899) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=66,
        entry_price=price,
        reason="momentum burst",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="momentum_burst",
            ts=datetime.now(timezone.utc),
            score=1.7617,
            criteria={
                "pattern": "momentum_burst",
                "direction": "up",
                "close": price,
                "volume": 109_489,
                "burst_pct": 1.7617,
                "velocity": 0.09,
            },
        ),
    )


def _thin_low_dollar_momentum_signal(symbol: str = "THIN", price: float = 3.25) -> TradeSignal:
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
            score=1.9,
            criteria={
                "pattern": "momentum_burst",
                "direction": "up",
                "close": price,
                "volume": 109_489,
                "burst_pct": 1.9,
                "velocity": 0.09,
            },
        ),
    )


def _first_pullback_signal(symbol: str = "FPB", price: float = 8.25) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        reason="first pullback reclaim",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="first_pullback_reclaim",
            ts=datetime.now(timezone.utc),
            score=43.0,
            criteria={
                "pattern": "first_pullback_reclaim",
                "direction": "up",
                "close": price,
                "volume": 316_504,
            },
        ),
    )


def _level_breakout_signal(symbol: str = "DAIC", price: float = 4.32) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        stop_loss=3.98,
        reason="level breakout reclaim",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="level_breakout_reclaim",
            ts=datetime.now(timezone.utc),
            score=44.0,
            criteria={
                "pattern": "level_breakout_reclaim",
                "direction": "up",
                "close": price,
                "breakout_level": 4.12,
                "base_high": 4.12,
                "base_low": 4.00,
                "stop_price": 3.98,
                "volume": 165_000,
            },
        ),
    )


def _bull_flag_signal(symbol: str = "ANY", price: float = 6.14) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=181,
        entry_price=price,
        reason="bull flag",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="bull_flag",
            ts=datetime.now(timezone.utc),
            score=2.4,
            criteria={
                "pattern": "bull_flag",
                "direction": "up",
                "close": price,
                "volume": 10_782_000,
            },
        ),
    )


def _generic_scanner_signal(symbol: str = "GEN", price: float = 5.50) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        reason="generic scanner setup",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="custom_scanner",
            ts=datetime.now(timezone.utc),
            score=40.0,
            criteria={
                "pattern": "custom_pattern",
                "direction": "up",
                "close": price,
                "volume": 250_000,
            },
        ),
    )


def _elite_abc_signal(symbol: str = "ELITE", price: float = 6.00) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        reason="abc continuation",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="abc_continuation",
            ts=datetime.now(timezone.utc),
            score=100.0,
            criteria={
                "pattern": "abc_continuation",
                "direction": "up",
                "close": price,
                "volume": 719_000,
                "rally_pct": 18.0,
            },
        ),
    )


def _pullback_base_signal(
    symbol: str = "PB",
    price: float = 7.99,
    score: float = 86.0,
    quantity: float = 100,
    volume: float = 195_756,
    day_move_pct: float | None = None,
    pullback_pct: float | None = None,
    base_range_pct: float | None = None,
    bars: list[Bar] | None = None,
) -> TradeSignal:
    criteria = {
        "pattern": "pullback_base",
        "direction": "up",
        "close": price,
        "volume": volume,
    }
    if day_move_pct is not None:
        criteria["day_move_pct"] = day_move_pct
    if pullback_pct is not None:
        criteria["pullback_pct"] = pullback_pct
    if base_range_pct is not None:
        criteria["base_range_pct"] = base_range_pct
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=quantity,
        entry_price=price,
        reason="pullback base",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="pullback_base",
            ts=datetime.now(timezone.utc),
            score=score,
            criteria=criteria,
            bars=bars or [],
        ),
    )


def _10s_bar(
    symbol: str,
    ts: datetime,
    o: float,
    c: float,
    h: float | None = None,
    l: float | None = None,
    volume: float = 1000,
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
        volume=volume,
        timeframe=Timeframe.SEC_10,
    )


def _pullback_profile_bars(
    symbol: str = "ANY",
    *,
    price: float = 5.36,
    impulse_volume: int = 180_000,
    pullback_volume: int = 65_000,
    reclaim_volume: int = 110_000,
    heavy_red: bool = False,
) -> list[Bar]:
    base = datetime(2026, 6, 3, 16, 20, 0, tzinfo=timezone.utc)
    rows = [
        (4.70, 4.74, 4.76, 4.68, 30_000),
        (4.74, 4.82, 4.84, 4.72, 35_000),
        (4.82, 5.10, 5.16, 4.80, impulse_volume),
        (5.10, 5.24, 5.30, 5.06, 140_000),
        (5.24, 5.42, 5.50, 5.22, 120_000),
        (5.42, 5.30, 5.46, 5.26, pullback_volume),
        (5.30, 5.22, 5.33, 5.18, pullback_volume),
        (5.22, 5.24, 5.29, 5.18, int(pullback_volume * 0.8)),
        (5.24, 5.28, 5.32, 5.22, int(pullback_volume * 0.9)),
        (5.28, price, max(5.40, price + 0.03), 5.25, reclaim_volume),
    ]
    if heavy_red:
        rows[6] = (5.34, 5.18, 5.36, 5.16, int(impulse_volume * 1.05))
    return [
        Bar(
            symbol=symbol,
            ts=base + timedelta(minutes=i),
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
        )
        for i, (o, c, h, l, v) in enumerate(rows)
    ]


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

    def test_max_wait_bars_fallback_for_plain_signal(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)
        # Doji / flat bars — no bounce trigger
        b = _10s_bar("TST", base, 5.00, 5.00, h=5.01, l=4.99)
        assert timer.on_10s_bar(b) is None
        released = timer.on_10s_bar(_10s_bar("TST", base + timedelta(seconds=10), 5.00, 5.00, h=5.01, l=4.99))
        assert released is not None

    def test_structured_signal_cancels_without_favorable_10s_entry(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_hot_hod_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("HOT", base, 3.15, 3.16, h=3.17, l=3.15)) is None
        released = timer.on_10s_bar(_10s_bar("HOT", base + timedelta(seconds=10), 3.16, 3.16, h=3.17, l=3.15))

        assert released is None
        assert "HOT" not in timer.pending_symbols

    def test_hot_momentum_cancels_without_favorable_10s_entry(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_hot_momentum_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("MOMO", base, 4.84, 4.80, h=4.86, l=4.79)) is None
        released = timer.on_10s_bar(_10s_bar("MOMO", base + timedelta(seconds=10), 4.80, 4.80, h=4.82, l=4.78))

        assert released is None
        assert "MOMO" not in timer.pending_symbols

    def test_hot_momentum_allows_strong_green_10s_entry(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_hot_momentum_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(_10s_bar("MOMO", base, 4.84, 4.92, h=4.94, l=4.83))

        assert released is not None
        assert released.symbol == "MOMO"

    def test_early_strength_releases_near_base_above_vwap(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _level_breakout_signal(symbol="BATL", price=1.66)
        sig.scan_result.criteria.update({
            "breakout_level": 1.64,
            "base_high": 1.64,
            "base_low": 1.57,
            "stop_price": 1.55,
            "vwap": 1.58,
            "volume": 300_000,
        })
        timer.queue(sig)
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("BATL", base, 1.65, 1.67, h=1.68, l=1.64, volume=20_000)
        )

        assert released is not None
        assert released.symbol == "BATL"

    def test_first_breakout_10s_bar_gets_volume_grace(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _level_breakout_signal(symbol="BATL", price=1.66)
        sig.scan_result.criteria.update({
            "breakout_level": 1.64,
            "base_high": 1.64,
            "base_low": 1.57,
            "stop_price": 1.55,
            "vwap": 1.58,
            "volume": 300_000,
        })
        timer.queue(sig)
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("BATL", base, 1.66, 1.66, h=1.67, l=1.64, volume=500)
        )

        assert released is None
        assert "BATL" in timer.pending_symbols

    def test_vwap_pullback_cancels_when_10s_bar_breaks_stop(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_veru_vwap_signal())
        base = datetime(2026, 6, 4, 16, 59, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("VERU", base, 5.86, 5.80, h=5.90, l=5.54)
        )

        assert released is None
        assert "VERU" not in timer.pending_symbols

    def test_vwap_pullback_waits_when_10s_bounce_does_not_reclaim_setup(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_veru_vwap_signal())
        base = datetime(2026, 6, 4, 16, 59, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(
            _10s_bar("VERU", base, 5.90, 5.86, h=5.91, l=5.82)
        ) is None
        released = timer.on_10s_bar(
            _10s_bar("VERU", base + timedelta(seconds=10), 5.76, 5.80, h=5.82, l=5.70)
        )

        assert released is None
        assert "VERU" not in timer.pending_symbols

    def test_vwap_pullback_releases_after_clean_10s_reclaim(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_veru_vwap_signal())
        base = datetime(2026, 6, 4, 16, 59, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(
            _10s_bar("VERU", base, 5.90, 5.86, h=5.91, l=5.82)
        ) is None
        released = timer.on_10s_bar(
            _10s_bar("VERU", base + timedelta(seconds=10), 5.86, 5.90, h=5.91, l=5.84)
        )

        assert released is not None
        assert released.symbol == "VERU"

    def test_vwap_pullback_releases_when_10s_holds_setup_flat(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_bgms_vwap_signal())
        base = datetime(2026, 6, 5, 10, 51, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(
            _10s_bar("BGMS", base, 2.93, 2.93, h=2.95, l=2.91, volume=18_000)
        ) is None
        released = timer.on_10s_bar(
            _10s_bar("BGMS", base + timedelta(seconds=10), 2.93, 2.94, h=2.95, l=2.92, volume=20_000)
        )

        assert released is not None
        assert released.symbol == "BGMS"
        assert released.entry_price == pytest.approx(2.93)

    def test_vwap_pullback_releases_sub_two_runner_after_clean_hold(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        signal = _bgms_vwap_signal(symbol="SUBT", price=1.87)
        signal.scan_result.criteria["vwap"] = 1.86
        signal.scan_result.criteria["pullback_low"] = 1.75
        timer.queue(signal)
        base = datetime(2026, 6, 5, 10, 51, 0, tzinfo=timezone.utc)

        pending = timer._pending["SUBT"]
        assert ExecutionTimer._allows_vwap_reclaim_release(
            pending,
            latest_bar=_10s_bar("SUBT", base, 1.87, 1.88, h=1.89, l=1.85, volume=20_000),
        )

    def test_vwap_reclaim_scout_can_release_after_clean_vwap_hold_even_after_dip(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        signal = _vwap_reclaim_scout_signal()
        timer.queue(signal)
        base = datetime(2026, 6, 1, 10, 15, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(
            _10s_bar("AIIO", base, 2.67, 2.64, h=2.68, l=2.58, volume=18_000)
        ) is None
        released = timer.on_10s_bar(
            _10s_bar("AIIO", base + timedelta(seconds=10), 2.64, 2.69, h=2.70, l=2.62, volume=22_000)
        )

        assert released is not None
        assert released.symbol == "AIIO"
        assert released.scan_result.criteria["entry_tier"] == "vwap_reclaim_scout"

    def test_vwap_reclaim_scout_can_release_on_first_clean_10s_hold(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        signal = _vwap_reclaim_scout_signal()
        timer.queue(signal)
        base = datetime(2026, 6, 1, 10, 16, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("AIIO", base, 2.672, 2.75, h=2.78, l=2.665, volume=40_342)
        )

        assert released is not None
        assert released.symbol == "AIIO"

    def test_vwap_reclaim_scout_still_waits_on_dead_10s_tape(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        signal = _vwap_reclaim_scout_signal()
        timer.queue(signal)
        base = datetime(2026, 6, 1, 10, 15, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("AIIO", base, 2.67, 2.68, h=2.69, l=2.65, volume=300)
        )

        assert released is None
        assert "AIIO" in timer.pending_symbols

    def test_sub_two_runner_does_not_release_on_dead_10s_tape(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        signal = _bgms_vwap_signal(symbol="PALI", price=1.80)
        signal.scan_result.criteria.update({
            "pattern": "first_pullback_reclaim",
            "setup_tier": "A+ setup",
            "vwap": 1.78,
            "pullback_low": 1.76,
            "stop_price": 1.70,
            "volume": 905_000,
        })
        signal = replace(signal, stop_loss=1.70)
        timer.queue(signal)
        base = datetime(2026, 6, 11, 15, 17, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("PALI", base, 1.79, 1.80, h=1.81, l=1.79, volume=900),
        )

        assert released is None
        assert "PALI" in timer.pending_symbols

    def test_sub_two_runner_releases_when_10s_tape_is_active(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        signal = _bgms_vwap_signal(symbol="PALI", price=1.80)
        signal.scan_result.criteria.update({
            "pattern": "first_pullback_reclaim",
            "setup_tier": "A+ setup",
            "vwap": 1.78,
            "pullback_low": 1.76,
            "stop_price": 1.70,
            "volume": 905_000,
        })
        signal = replace(signal, stop_loss=1.70)
        timer.queue(signal)
        base = datetime(2026, 6, 11, 15, 17, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("PALI", base, 1.79, 1.80, h=1.81, l=1.79, volume=12_000),
        )

        assert released is not None
        assert released.symbol == "PALI"

    def test_vwap_pullback_does_not_release_after_price_chases_too_far(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_bgms_vwap_signal())
        base = datetime(2026, 6, 5, 10, 51, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(
            _10s_bar("BGMS", base, 2.93, 2.94, h=2.95, l=2.92, volume=18_000)
        ) is None
        released = timer.on_10s_bar(
            _10s_bar("BGMS", base + timedelta(seconds=10), 3.08, 3.10, h=3.12, l=3.07, volume=80_000)
        )

        assert released is None
        assert "BGMS" not in timer.pending_symbols

    def test_elite_signal_allows_clean_10s_fallback(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_elite_abc_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("ELITE", base, 6.00, 6.00, h=6.03, l=5.99)) is None
        released = timer.on_10s_bar(_10s_bar("ELITE", base + timedelta(seconds=10), 6.00, 6.01, h=6.03, l=6.00))

        assert released is not None
        assert released.symbol == "ELITE"

    def test_elite_signal_cancels_after_red_10s_bar(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_elite_abc_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("ELITE", base, 6.03, 6.00, h=6.04, l=5.99)) is None
        released = timer.on_10s_bar(_10s_bar("ELITE", base + timedelta(seconds=10), 6.00, 6.00, h=6.01, l=5.98))

        assert released is None
        assert "ELITE" not in timer.pending_symbols

    def test_elite_momentum_releases_reduced_scout_after_mild_red_wait(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_elite_momentum_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("ANY", base, 4.94, 4.94, h=4.97, l=4.92)) is None
        released = timer.on_10s_bar(
            _10s_bar("ANY", base + timedelta(seconds=10), 4.96, 4.94, h=5.00, l=4.90)
        )

        assert released is not None
        assert released.symbol == "ANY"
        assert released.quantity == 67.0
        assert "continuation_scout" in released.reason

    def test_elite_momentum_cancels_after_ugly_red_wait(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        timer.queue(_elite_momentum_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("ANY", base, 4.94, 4.94, h=4.97, l=4.92)) is None
        released = timer.on_10s_bar(
            _10s_bar("ANY", base + timedelta(seconds=10), 5.05, 4.88, h=5.06, l=4.87)
        )

        assert released is None
        assert "ANY" not in timer.pending_symbols

    def test_strong_pullback_base_releases_reduced_scout_after_mild_red_wait(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        sig = _pullback_base_signal(
            symbol="ANY",
            price=5.36,
            score=80.0,
            quantity=161,
            volume=10_368_276,
            pullback_pct=6.9,
            base_range_pct=5.7,
        )
        timer.queue(sig)
        base = datetime(2026, 6, 3, 16, 32, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("ANY", base, 5.36, 5.36, h=5.40, l=5.32)) is None
        released = timer.on_10s_bar(
            _10s_bar("ANY", base + timedelta(seconds=10), 5.38, 5.36, h=5.42, l=5.30)
        )

        assert released is not None
        assert released.symbol == "ANY"
        assert released.quantity == 56.0
        assert "pullback_scout" in released.reason

    def test_strong_one_minute_pullback_base_releases_full_size_on_clean_flat_10s(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        sig = _pullback_base_signal(
            symbol="ANY",
            price=5.36,
            score=96.0,
            quantity=161,
            volume=110_000,
            day_move_pct=37.0,
            pullback_pct=6.9,
            base_range_pct=5.7,
            bars=_pullback_profile_bars(
                impulse_volume=180_000,
                pullback_volume=55_000,
                reclaim_volume=110_000,
            ),
        )
        timer.queue(sig)
        base = datetime(2026, 6, 3, 16, 32, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("ANY", base, 5.36, 5.36, h=5.40, l=5.32)) is None
        released = timer.on_10s_bar(
            _10s_bar("ANY", base + timedelta(seconds=10), 5.36, 5.37, h=5.40, l=5.33)
        )

        assert released is not None
        assert released.symbol == "ANY"
        assert released.quantity == 161
        assert "pullback_scout" not in released.reason

    def test_scout_only_pullback_never_releases_full_size(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        sig = _pullback_base_signal(
            symbol="ANY",
            price=5.36,
            score=96.0,
            quantity=161,
            volume=110_000,
            day_move_pct=17.0,
            pullback_pct=5.0,
            base_range_pct=3.7,
            bars=_pullback_profile_bars(
                impulse_volume=180_000,
                pullback_volume=55_000,
                reclaim_volume=110_000,
            ),
        )
        assert sig.scan_result is not None
        sig.scan_result.criteria["entry_tier"] = "pullback_scout"
        timer.queue(sig)
        base = datetime(2026, 6, 3, 16, 32, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("ANY", base, 5.36, 5.36, h=5.40, l=5.32)) is None
        released = timer.on_10s_bar(
            _10s_bar("ANY", base + timedelta(seconds=10), 5.36, 5.37, h=5.40, l=5.33)
        )

        assert released is not None
        assert released.quantity == 56.0
        assert "pullback_scout" in released.reason

    def test_one_minute_pullback_volume_profile_can_pass_with_lower_absolute_volume(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        sig = _pullback_base_signal(
            symbol="ANY",
            price=5.36,
            score=96.0,
            quantity=161,
            volume=82_000,
            day_move_pct=37.0,
            pullback_pct=6.9,
            base_range_pct=5.7,
            bars=_pullback_profile_bars(
                impulse_volume=95_000,
                pullback_volume=28_000,
                reclaim_volume=82_000,
            ),
        )
        timer.queue(sig)
        base = datetime(2026, 6, 3, 16, 32, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("ANY", base, 5.36, 5.36, h=5.40, l=5.32)) is None
        released = timer.on_10s_bar(
            _10s_bar("ANY", base + timedelta(seconds=10), 5.36, 5.37, h=5.40, l=5.33)
        )

        assert released is not None
        assert released.quantity == 161
        assert "pullback_scout" not in released.reason

    def test_one_minute_pullback_rejects_heavy_red_pullback_volume(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        sig = _pullback_base_signal(
            symbol="ANY",
            price=5.36,
            score=96.0,
            quantity=161,
            volume=120_000,
            day_move_pct=37.0,
            pullback_pct=6.9,
            base_range_pct=5.7,
            bars=_pullback_profile_bars(
                impulse_volume=160_000,
                pullback_volume=55_000,
                reclaim_volume=120_000,
                heavy_red=True,
            ),
        )
        timer.queue(sig)
        base = datetime(2026, 6, 3, 16, 32, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("ANY", base, 5.36, 5.36, h=5.40, l=5.32)) is None
        released = timer.on_10s_bar(
            _10s_bar("ANY", base + timedelta(seconds=10), 5.36, 5.37, h=5.40, l=5.33)
        )

        assert released is None
        assert "ANY" not in timer.pending_symbols

    def test_one_minute_pullback_base_release_requires_tight_pullback(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        sig = _pullback_base_signal(
            symbol="ANY",
            price=5.36,
            score=96.0,
            quantity=161,
            volume=10_368_276,
            day_move_pct=37.0,
            pullback_pct=10.4,
            base_range_pct=5.7,
            bars=_pullback_profile_bars(),
        )
        timer.queue(sig)
        base = datetime(2026, 6, 3, 16, 32, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("ANY", base, 5.36, 5.36, h=5.40, l=5.32)) is None
        released = timer.on_10s_bar(
            _10s_bar("ANY", base + timedelta(seconds=10), 5.36, 5.37, h=5.40, l=5.33)
        )

        assert released is not None
        assert released.quantity == 56.0
        assert "pullback_scout" in released.reason

    def test_pullback_base_scout_cancels_wide_base(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        sig = _pullback_base_signal(
            symbol="ANY",
            price=5.36,
            score=82.0,
            quantity=161,
            volume=10_368_276,
            pullback_pct=6.9,
            base_range_pct=9.1,
        )
        timer.queue(sig)
        base = datetime(2026, 6, 3, 16, 32, 0, tzinfo=timezone.utc)

        assert timer.on_10s_bar(_10s_bar("ANY", base, 5.36, 5.36, h=5.40, l=5.32)) is None
        released = timer.on_10s_bar(
            _10s_bar("ANY", base + timedelta(seconds=10), 5.38, 5.36, h=5.42, l=5.30)
        )

        assert released is None
        assert "ANY" not in timer.pending_symbols

    def test_structured_signal_allows_green_reclaim_after_intrabar_dip(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        timer.queue(_hot_hod_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(_10s_bar("HOT", base, 3.13, 3.18, h=3.19, l=3.12))

        assert released is not None
        assert released.symbol == "HOT"

    def test_first_pullback_reclaim_allows_strong_green_without_second_dip(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        timer.queue(_first_pullback_signal())
        base = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(_10s_bar("FPB", base, 8.25, 8.34, h=8.36, l=8.24))

        assert released is not None
        assert released.symbol == "FPB"

    def test_level_breakout_releases_with_10s_volume_above_level(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        timer.queue(_level_breakout_signal())
        base = datetime(2026, 6, 5, 8, 44, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("DAIC", base, 4.15, 4.20, h=4.22, l=4.12, volume=25_000)
        )

        assert released is not None
        assert released.symbol == "DAIC"

    def test_level_breakout_cancels_when_10s_reclaim_is_late_chase(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        timer.queue(_level_breakout_signal())
        base = datetime(2026, 6, 5, 8, 44, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("DAIC", base, 4.24, 4.30, h=4.32, l=4.20, volume=25_000)
        )

        assert released is None

    def test_level_breakout_cancels_when_10s_close_loses_level(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        timer.queue(_level_breakout_signal())
        base = datetime(2026, 6, 5, 8, 44, 0, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("DAIC", base, 4.08, 4.10, h=4.16, l=4.05, volume=25_000)
        )

        assert released is None
        assert "DAIC" not in timer.pending_symbols

    def test_opening_range_breakout_cancels_late_failed_push_bounce(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        timer.queue(_bgms_opening_range_signal())
        base = datetime(2026, 6, 5, 8, 10, 30, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("BGMS", base, 3.06, 3.11, h=3.12, l=3.05)
        )

        assert released is None
        assert "BGMS" not in timer.pending_symbols

    def test_opening_range_breakout_releases_when_near_signal_and_above_breakout_level(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        timer.queue(_bgms_opening_range_signal())
        base = datetime(2026, 6, 5, 8, 10, 30, tzinfo=timezone.utc)

        released = timer.on_10s_bar(
            _10s_bar("BGMS", base, 3.18, 3.20, h=3.22, l=3.16, volume=35_000)
        )

        assert released is not None
        assert released.symbol == "BGMS"

    def test_opening_range_breakout_cancels_when_level_reclaim_has_weak_volume(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        timer.queue(_bgms_opening_range_signal())
        base = datetime(2026, 6, 5, 8, 10, 30, tzinfo=timezone.utc)

        assert timer.on_10s_bar(
            _10s_bar("BGMS", base, 3.18, 3.20, h=3.22, l=3.16, volume=2_000)
        ) is None
        released = timer.on_10s_bar(
            _10s_bar("BGMS", base + timedelta(seconds=10), 3.20, 3.21, h=3.22, l=3.18, volume=2_000)
        )

        assert released is None
        assert "BGMS" not in timer.pending_symbols

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

    def test_hot_hod_signal_timeout_cancels_instead_of_chasing(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _hot_hod_signal()
        timer.queue(sig)
        pending = timer._pending["HOT"]
        assert pending.max_wait_seconds == 12.0
        assert pending.require_pullback_reclaim is True

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=13)
        released = timer.check_timeouts()
        assert released == []
        assert "HOT" not in timer.pending_symbols

    def test_hot_momentum_signal_timeout_cancels_without_10s_bars(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _hot_momentum_signal()
        timer.queue(sig)
        pending = timer._pending["MOMO"]
        assert pending.max_wait_seconds == 10.0
        assert pending.require_pullback_reclaim is False
        assert pending.require_micro_signal is True

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)
        released = timer.check_timeouts()
        assert released == []
        assert "MOMO" not in timer.pending_symbols

    def test_bull_flag_timeout_cancels_without_10s_bars(self) -> None:
        timer = ExecutionTimer(max_wait_bars=1, enabled=True)
        sig = _bull_flag_signal()
        timer.queue(sig)
        pending = timer._pending["ANY"]
        assert pending.require_micro_signal is True
        assert pending.require_pullback_reclaim is True

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=25)
        released = timer.check_timeouts()

        assert released == []
        assert "ANY" not in timer.pending_symbols

    def test_elite_momentum_timeout_releases_reduced_scout(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _elite_momentum_signal()
        timer.queue(sig)
        pending = timer._pending["ANY"]
        pending.last_bar = _10s_bar("ANY", datetime.now(timezone.utc), 4.96, 4.94, h=5.00, l=4.90)
        pending.bars_seen = 1
        pending.saw_red = True
        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)

        released = timer.check_timeouts()

        assert len(released) == 1
        assert released[0].symbol == "ANY"
        assert released[0].quantity == 67.0
        assert "continuation_scout" in released[0].reason

    def test_foxx_momentum_timeout_without_10s_bars_cancels(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _foxx_momentum_signal()
        timer.queue(sig)
        pending = timer._pending["FOXX"]
        assert pending.max_wait_seconds == 10.0
        assert pending.bars_seen == 0

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)
        released = timer.check_timeouts()

        assert released == []
        assert "FOXX" not in timer.pending_symbols

    def test_opening_range_timeout_without_10s_bars_cancels(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _foxx_opening_range_signal()
        timer.queue(sig)
        pending = timer._pending["FOXX"]
        assert pending.max_wait_seconds is None
        assert pending.bars_seen == 0

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=31)
        released = timer.check_timeouts()

        assert released == []
        assert "FOXX" not in timer.pending_symbols

    def test_mnts_momentum_timeout_without_10s_bars_cancels(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _mnts_momentum_signal()
        timer.queue(sig)
        pending = timer._pending["MNTS"]
        assert pending.max_wait_seconds == 10.0
        assert pending.bars_seen == 0

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)
        released = timer.check_timeouts()

        assert released == []
        assert "MNTS" not in timer.pending_symbols

    def test_low_price_low_dollar_momentum_still_cancels_without_10s_bars(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _thin_low_dollar_momentum_signal()
        timer.queue(sig)
        pending = timer._pending["THIN"]
        assert pending.max_wait_seconds == 10.0

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)
        released = timer.check_timeouts()

        assert released == []
        assert "THIN" not in timer.pending_symbols

    def test_pullback_base_timeout_releases_reduced_scout_with_recent_10s_bar(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _pullback_base_signal(
            symbol="ANY",
            price=5.36,
            score=80.0,
            quantity=293,
            volume=10_368_276,
            pullback_pct=10.4,
            base_range_pct=3.4,
        )
        timer.queue(sig)
        pending = timer._pending["ANY"]
        pending.last_bar = _10s_bar("ANY", datetime.now(timezone.utc), 5.38, 5.36, h=5.42, l=5.30)
        pending.bars_seen = 1
        pending.saw_red = True
        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)

        released = timer.check_timeouts()

        assert len(released) == 1
        assert released[0].symbol == "ANY"
        assert released[0].quantity == 75.0
        assert "pullback_scout" in released[0].reason

    def test_one_minute_pullback_timeout_releases_full_size_with_clean_recent_10s_bar(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _pullback_base_signal(
            symbol="ANY",
            price=5.36,
            score=96.0,
            quantity=161,
            volume=10_368_276,
            day_move_pct=37.0,
            pullback_pct=6.9,
            base_range_pct=5.7,
            bars=_pullback_profile_bars(
                impulse_volume=180_000,
                pullback_volume=55_000,
                reclaim_volume=110_000,
            ),
        )
        timer.queue(sig)
        pending = timer._pending["ANY"]
        pending.last_bar = _10s_bar("ANY", datetime.now(timezone.utc), 5.36, 5.37, h=5.40, l=5.33)
        pending.bars_seen = 1
        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)

        released = timer.check_timeouts()

        assert len(released) == 1
        assert released[0].symbol == "ANY"
        assert released[0].quantity == 161
        assert "pullback_scout" not in released[0].reason

    def test_first_pullback_reclaim_timeout_cancels_without_10s_bars(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _first_pullback_signal()
        timer.queue(sig)
        pending = timer._pending["FPB"]
        assert pending.max_wait_seconds == 10.0
        assert pending.require_pullback_reclaim is False
        assert pending.require_micro_signal is True

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)
        released = timer.check_timeouts()
        assert released == []
        assert "FPB" not in timer.pending_symbols

    def test_strong_pullback_base_timeout_cancels_without_10s_bars(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _pullback_base_signal(score=86.0)
        timer.queue(sig)
        pending = timer._pending["PB"]
        assert pending.max_wait_seconds == 10.0
        assert pending.require_pullback_reclaim is False
        assert pending.require_micro_signal is True

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)
        released = timer.check_timeouts()
        assert released == []
        assert "PB" not in timer.pending_symbols

    def test_elite_pullback_base_timeout_without_10s_bars_cancels(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _pullback_base_signal(
            symbol="MOBX",
            price=3.0478,
            score=153.576,
            quantity=240,
            volume=466_434,
            day_move_pct=43.94,
            pullback_pct=5.05,
            base_range_pct=6.64,
            bars=_pullback_profile_bars(
                symbol="MOBX",
                price=3.0478,
                impulse_volume=420_000,
                pullback_volume=145_000,
                reclaim_volume=466_434,
            ),
        )
        timer.queue(sig)
        pending = timer._pending["MOBX"]
        assert pending.max_wait_seconds == 10.0
        assert pending.bars_seen == 0

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)
        released = timer.check_timeouts()

        assert released == []
        assert "MOBX" not in timer.pending_symbols

    def test_sti_style_pullback_timeout_without_10s_bars_cancels(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _pullback_base_signal(
            symbol="STI",
            price=18.34,
            score=151.454,
            quantity=54,
            volume=438_228,
            day_move_pct=43.06,
            pullback_pct=4.83,
            base_range_pct=7.79,
            bars=_pullback_profile_bars(
                symbol="STI",
                price=18.34,
                impulse_volume=390_000,
                pullback_volume=120_000,
                reclaim_volume=438_228,
            ),
        )
        timer.queue(sig)
        pending = timer._pending["STI"]
        assert pending.max_wait_seconds == 10.0
        assert pending.bars_seen == 0

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)
        released = timer.check_timeouts()

        assert released == []
        assert "STI" not in timer.pending_symbols

    def test_elite_pullback_base_timeout_without_10s_bars_still_rejects_wide_base(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _pullback_base_signal(
            symbol="WIDE",
            price=3.0478,
            score=153.576,
            quantity=240,
            volume=466_434,
            day_move_pct=43.94,
            pullback_pct=5.05,
            base_range_pct=8.2,
            bars=_pullback_profile_bars(
                symbol="WIDE",
                price=3.0478,
                impulse_volume=420_000,
                pullback_volume=145_000,
                reclaim_volume=466_434,
            ),
        )
        timer.queue(sig)
        pending = timer._pending["WIDE"]
        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)

        released = timer.check_timeouts()

        assert released == []
        assert "WIDE" not in timer.pending_symbols

    def test_elite_signal_timeout_without_10s_bars_cancels(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _elite_abc_signal()
        timer.queue(sig)
        pending = timer._pending["ELITE"]
        assert pending.max_wait_seconds == 10.0
        assert pending.require_micro_signal is True

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)
        released = timer.check_timeouts()
        assert released == []
        assert "ELITE" not in timer.pending_symbols

    def test_weak_pullback_base_still_cancels_without_reclaim(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _pullback_base_signal(score=50.0)
        timer.queue(sig)
        pending = timer._pending["PB"]
        assert pending.max_wait_seconds is None
        assert pending.require_pullback_reclaim is True

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=45)
        released = timer.check_timeouts()
        assert released == []
        assert "PB" not in timer.pending_symbols

    def test_any_scanner_signal_timeout_cancels_without_10s_confirmation(self) -> None:
        timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        sig = _generic_scanner_signal()
        timer.queue(sig)
        pending = timer._pending["GEN"]
        assert pending.max_wait_seconds is None
        assert pending.require_micro_signal is True

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=45)
        released = timer.check_timeouts()

        assert released == []
        assert "GEN" not in timer.pending_symbols

    def test_scanner_signal_timeout_matches_logged_one_bar_wait(self) -> None:
        timer = ExecutionTimer(max_wait_bars=1, enabled=True)
        sig = _generic_scanner_signal()
        timer.queue(sig)
        pending = timer._pending["GEN"]
        assert pending.max_wait_seconds is None
        assert pending.require_micro_signal is True

        pending.queued_at = datetime.now(timezone.utc) - timedelta(seconds=11)
        released = timer.check_timeouts()

        assert released == []
        assert "GEN" not in timer.pending_symbols

    def test_seconds_until_next_timeout_reports_urgent_pending_entry(self) -> None:
        timer = ExecutionTimer(max_wait_bars=1, enabled=True)
        sig = _generic_scanner_signal()
        timer.queue(sig)
        now = datetime.now(timezone.utc)
        pending = timer._pending["GEN"]

        pending.queued_at = now - timedelta(seconds=9)
        assert 0 < timer.seconds_until_next_timeout(now) <= 1

        pending.queued_at = now - timedelta(seconds=11)
        assert timer.seconds_until_next_timeout(now) == 0.0

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


# ---------------------------------------------------------------------------
# Tick-based early entry trigger
# ---------------------------------------------------------------------------

def _tick(symbol: str = "TCK", price: float = 2.0) -> Tick:
    return Tick(
        symbol=symbol,
        ts=datetime.now(timezone.utc),
        price=price,
        size=100,
        side=Side.BUY,
    )


def _tick_signal(symbol: str = "TCK", price: float = 2.0, vwap: float = 1.98) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        reason="tick entry test",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="vwap_pullback",
            ts=datetime.now(timezone.utc),
            score=40.0,
            criteria={
                "pattern": "vwap_pullback",
                "close": price,
                "setup_anchor": price,
                "vwap": vwap,
            },
        ),
    )


def _tick_timer(confirm: int = 2) -> ExecutionTimer:
    return ExecutionTimer(
        max_wait_bars=3,
        enabled=True,
        tick_entry_enabled=True,
        tick_entry_confirm_count=confirm,
    )


def test_tick_entry_releases_after_confirm_count():
    timer = _tick_timer(confirm=2)
    timer.queue(_tick_signal(price=2.00, vwap=1.98))  # anchor=2.00
    assert timer.on_tick(_tick("TCK", 2.01)) is None       # confirm=1
    released = timer.on_tick(_tick("TCK", 2.02))           # confirm=2 -> release
    assert released is not None
    assert released.symbol == "TCK"
    assert "TCK" not in timer.pending_symbols              # popped on release


def test_tick_entry_no_release_when_extended():
    timer = _tick_timer(confirm=2)
    timer.queue(_tick_signal(price=2.00, vwap=1.98))       # ceiling = 2.00 * 1.02 = 2.04
    assert timer.on_tick(_tick("TCK", 2.10)) is None
    assert timer.on_tick(_tick("TCK", 2.10)) is None       # never confirms while extended


def test_tick_entry_no_release_below_vwap():
    timer = _tick_timer(confirm=2)
    timer.queue(_tick_signal(price=2.00, vwap=2.05))       # vwap above price
    assert timer.on_tick(_tick("TCK", 2.00)) is None
    assert timer.on_tick(_tick("TCK", 2.00)) is None


def test_tick_entry_confirmation_resets_on_bad_tick():
    timer = _tick_timer(confirm=2)
    timer.queue(_tick_signal(price=2.00, vwap=1.98))
    assert timer.on_tick(_tick("TCK", 2.01)) is None       # confirm=1
    assert timer.on_tick(_tick("TCK", 2.10)) is None       # extended -> reset to 0
    assert timer.on_tick(_tick("TCK", 2.01)) is None       # confirm=1 again, not 2 -> no release
    assert "TCK" in timer.pending_symbols


def test_tick_entry_disabled_returns_none():
    timer = ExecutionTimer(max_wait_bars=3, enabled=True, tick_entry_enabled=False)
    timer.queue(_tick_signal(price=2.00, vwap=1.98))
    assert timer.on_tick(_tick("TCK", 2.01)) is None
    assert timer.on_tick(_tick("TCK", 2.01)) is None       # still none with flag off
    assert "TCK" in timer.pending_symbols


def test_tick_entry_ignores_unqueued_symbol():
    timer = _tick_timer(confirm=1)
    assert timer.on_tick(_tick("NOPE", 2.0)) is None
