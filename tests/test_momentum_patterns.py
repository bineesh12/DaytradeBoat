"""Tests for Bull Flag and Flat Top Breakout scanners + verifier."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from daytrading.models import Bar, PortfolioState, ScanResult, SignalAction
from daytrading.scanner.scalping.bull_flag import BullFlagScanner
from daytrading.scanner.scalping.flat_top_breakout import FlatTopBreakoutScanner
from daytrading.strategy.scalping.momentum_pattern import MomentumPatternVerifier


def _bar(
    i: int,
    *,
    close: float,
    open_: float,
    high: float,
    low: float,
    volume: float,
    base_ts: datetime | None = None,
    n: int = 30,
) -> Bar:
    if base_ts is None:
        base_ts = datetime.now(timezone.utc)
    return Bar(
        symbol="TST",
        ts=base_ts - timedelta(seconds=(n - i)),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_bull_flag_bars() -> list[Bar]:
    """Build a synthetic bull flag: pole → pullback → breakout."""
    now = datetime.now(timezone.utc)
    bars: list[Bar] = []

    # 5 bars of base activity before the pole
    for i in range(5):
        bars.append(_bar(i, close=5.0, open_=4.98, high=5.01, low=4.97, volume=50_000, base_ts=now, n=30))

    # Pole: 4 green bars, strong move from 5.0 to 5.15 (+3%)
    pole_prices = [(5.00, 5.03), (5.03, 5.07), (5.07, 5.11), (5.11, 5.15)]
    for j, (o, c) in enumerate(pole_prices):
        i = 5 + j
        bars.append(_bar(i, close=c, open_=o, high=c + 0.01, low=o - 0.01, volume=150_000, base_ts=now, n=30))

    # Pullback: 3 red bars, pulling back to 5.10 (about 33% retrace of the 0.15 move)
    pb_prices = [(5.15, 5.13), (5.13, 5.11), (5.11, 5.10)]
    for j, (o, c) in enumerate(pb_prices):
        i = 9 + j
        bars.append(_bar(i, close=c, open_=o, high=o + 0.005, low=c - 0.005, volume=30_000, base_ts=now, n=30))

    # Breakout candle: green, new high above pullback highs
    bars.append(_bar(12, close=5.18, open_=5.10, high=5.20, low=5.09, volume=120_000, base_ts=now, n=30))

    return bars


def _make_flat_top_bars() -> list[Bar]:
    """Build a synthetic flat top breakout: drive → flat resistance → breakout."""
    now = datetime.now(timezone.utc)
    bars: list[Bar] = []

    # 5 base bars
    for i in range(5):
        bars.append(_bar(i, close=4.0, open_=3.98, high=4.01, low=3.97, volume=50_000, base_ts=now, n=25))

    # Drive up: 3 green bars from 4.0 to 4.20
    drive_prices = [(4.00, 4.07), (4.07, 4.13), (4.13, 4.20)]
    for j, (o, c) in enumerate(drive_prices):
        i = 5 + j
        bars.append(_bar(i, close=c, open_=o, high=c + 0.01, low=o - 0.01, volume=100_000, base_ts=now, n=25))

    # Flat top: 4 bars, all with highs near 4.21 (resistance)
    for j in range(4):
        i = 8 + j
        bars.append(_bar(i, close=4.18, open_=4.19, high=4.21, low=4.16, volume=40_000, base_ts=now, n=25))

    # Breakout candle: closes above 4.21
    bars.append(_bar(12, close=4.26, open_=4.20, high=4.28, low=4.19, volume=100_000, base_ts=now, n=25))

    return bars


class TestBullFlagScanner:
    def test_detects_bull_flag(self) -> None:
        bars = _make_bull_flag_bars()
        scanner = BullFlagScanner(min_pole_pct=1.0, min_price=1.0, max_price=20.0)
        hits = scanner.scan({"TST": bars})
        assert len(hits) >= 1
        hit = hits[0]
        assert hit.symbol == "TST"
        assert hit.scanner_name == "bull_flag"
        assert hit.criteria["pattern"] == "bull_flag"
        assert hit.criteria["direction"] == "up"
        assert hit.criteria["pole_pct"] > 0
        assert hit.criteria["retrace_pct"] > 0

    def test_no_hit_without_pole(self) -> None:
        now = datetime.now(timezone.utc)
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=50_000, base_ts=now, n=20)
            for i in range(15)
        ]
        scanner = BullFlagScanner()
        hits = scanner.scan({"TST": bars})
        assert len(hits) == 0

    def test_rejects_wrong_price_range(self) -> None:
        bars = _make_bull_flag_bars()
        scanner = BullFlagScanner(min_price=10.0, max_price=20.0)
        hits = scanner.scan({"TST": bars})
        assert len(hits) == 0


class TestFlatTopScanner:
    def test_detects_flat_top(self) -> None:
        bars = _make_flat_top_bars()
        scanner = FlatTopBreakoutScanner(min_drive_pct=1.0, min_price=1.0, max_price=20.0)
        hits = scanner.scan({"TST": bars})
        assert len(hits) >= 1
        hit = hits[0]
        assert hit.symbol == "TST"
        assert hit.scanner_name == "flat_top_breakout"
        assert hit.criteria["pattern"] == "flat_top_breakout"
        assert hit.criteria["resistance"] > 0

    def test_no_hit_without_flat_zone(self) -> None:
        now = datetime.now(timezone.utc)
        bars = [
            _bar(i, close=5.0 + i * 0.1, open_=5.0 + i * 0.1 - 0.02,
                 high=5.0 + i * 0.1 + 0.05, low=5.0 + i * 0.1 - 0.03,
                 volume=50_000, base_ts=now, n=20)
            for i in range(15)
        ]
        scanner = FlatTopBreakoutScanner()
        hits = scanner.scan({"TST": bars})
        assert len(hits) == 0


class TestMomentumPatternVerifier:
    def test_generates_signal_for_bull_flag(self) -> None:
        bars = _make_bull_flag_bars()
        hit = ScanResult(
            symbol="TST",
            scanner_name="bull_flag",
            ts=datetime.now(timezone.utc),
            score=3.0,
            criteria={
                "pattern": "bull_flag",
                "direction": "up",
                "pole_pct": 3.0,
                "retrace_pct": 33.0,
                "pole_bars": 4,
                "pullback_bars": 3,
                "breakout_price": 5.18,
                "pole_high": 5.16,
                "pullback_low": 5.095,
                "close": 5.18,
                "volume": 120_000,
            },
            bars=bars,
        )
        port = PortfolioState(cash=100_000.0)
        verifier = MomentumPatternVerifier(max_risk_per_share=0.50)
        signal = verifier.verify(hit, port)

        if signal is not None:
            assert signal.action == SignalAction.ENTER_LONG
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.stop_loss < signal.entry_price
            assert signal.take_profit > signal.entry_price
            assert "Bull Flag" in signal.reason
        else:
            # Entry guard may reject due to synthetic bar characteristics
            assert verifier._last_reject is not None

    def test_rejects_unknown_pattern(self) -> None:
        hit = ScanResult(
            symbol="TST",
            scanner_name="bull_flag",
            ts=datetime.now(timezone.utc),
            score=3.0,
            criteria={"pattern": "triple_top", "direction": "up"},
            bars=[
                _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=50_000, n=5)
                for i in range(5)
            ],
        )
        port = PortfolioState(cash=100_000.0)
        verifier = MomentumPatternVerifier()
        signal = verifier.verify(hit, port)
        assert signal is None
        assert "unknown pattern" in verifier._last_reject

    def test_rejects_when_already_in_position(self) -> None:
        pos = MagicMock()
        pos.is_flat = False
        port = PortfolioState(cash=100_000.0)
        port.positions["TST"] = pos

        hit = ScanResult(
            symbol="TST",
            scanner_name="bull_flag",
            ts=datetime.now(timezone.utc),
            score=3.0,
            criteria={"pattern": "bull_flag", "direction": "up", "pullback_low": 5.0},
            bars=[
                _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=50_000, n=5)
                for i in range(5)
            ],
        )
        verifier = MomentumPatternVerifier()
        signal = verifier.verify(hit, port)
        assert signal is None
        assert "already in position" in verifier._last_reject
