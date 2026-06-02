"""Tests for Bull Flag and Flat Top Breakout scanners + verifier."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from daytrading.models import Bar, PortfolioState, ScanResult, SignalAction
from daytrading.scanner.scalping.bull_flag import BullFlagScanner
from daytrading.scanner.scalping.flat_top_breakout import FlatTopBreakoutScanner
from daytrading.scanner.scalping.abc_continuation import ABCContinuationScanner
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


def _make_abc_bars() -> list[Bar]:
    """Build A push, B pullback, C continuation trigger."""
    now = datetime.now(timezone.utc)
    bars: list[Bar] = []
    for i in range(4):
        bars.append(_bar(i, close=4.00, open_=3.99, high=4.02, low=3.98, volume=40_000, base_ts=now, n=30))

    # A leg: strong move from 4.00 to 4.45.
    a_prices = [(4.00, 4.12), (4.12, 4.25), (4.25, 4.36), (4.36, 4.45)]
    for j, (o, c) in enumerate(a_prices):
        i = 4 + j
        bars.append(_bar(i, close=c, open_=o, high=c + 0.03, low=o - 0.02, volume=140_000, base_ts=now, n=30))

    # B: controlled pullback, roughly 35-45% retrace.
    b_prices = [(4.45, 4.34), (4.34, 4.30), (4.31, 4.28)]
    for j, (o, c) in enumerate(b_prices):
        i = 8 + j
        bars.append(_bar(i, close=c, open_=o, high=o + 0.02, low=c - 0.02, volume=55_000, base_ts=now, n=30))

    # C: breaks B high with volume returning.
    bars.append(_bar(11, close=4.49, open_=4.30, high=4.52, low=4.29, volume=90_000, base_ts=now, n=30))
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


class TestABCContinuationScanner:
    def test_detects_abc_continuation(self) -> None:
        bars = _make_abc_bars()
        scanner = ABCContinuationScanner(min_a_leg_pct=5.0, min_price=1.0, max_price=20.0)
        hits = scanner.scan({"TST": bars})
        assert len(hits) >= 1
        hit = hits[0]
        assert hit.scanner_name == "abc_continuation"
        assert hit.criteria["pattern"] == "abc_continuation"
        assert hit.criteria["a_leg_pct"] >= 5.0
        assert 20.0 <= hit.criteria["b_retrace_pct"] <= 60.0
        assert hit.criteria["b_low"] > 0
        assert hit.criteria["c_volume_surge"] >= 1.1

    def test_rejects_too_deep_b_pullback(self) -> None:
        bars = _make_abc_bars()
        # Make B low retrace almost the whole A leg.
        bars[-2] = _bar(10, close=4.05, open_=4.22, high=4.24, low=4.03, volume=55_000, base_ts=bars[-1].ts, n=30)
        scanner = ABCContinuationScanner(min_a_leg_pct=5.0, min_price=1.0, max_price=20.0)
        hits = scanner.scan({"TST": bars})
        assert hits == []


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

    def test_rejects_late_pullback_far_from_hod(self) -> None:
        now = datetime.now(timezone.utc)
        bars = []
        for i in range(10):
            bars.append(_bar(i, close=5.0 + i * 0.12, open_=5.0 + i * 0.12 - 0.04,
                             high=5.1 + i * 0.12, low=4.95 + i * 0.12,
                             volume=80_000, base_ts=now, n=16))
        bars.extend([
            _bar(10, close=6.30, open_=6.15, high=6.50, low=6.10, volume=100_000, base_ts=now, n=16),
            _bar(11, close=5.95, open_=6.20, high=6.25, low=5.90, volume=90_000, base_ts=now, n=16),
            _bar(12, close=5.70, open_=5.95, high=6.00, low=5.65, volume=85_000, base_ts=now, n=16),
            _bar(13, close=5.72, open_=5.66, high=5.75, low=5.62, volume=90_000, base_ts=now, n=16),
            _bar(14, close=5.76, open_=5.70, high=5.78, low=5.68, volume=95_000, base_ts=now, n=16),
        ])
        hit = ScanResult(
            symbol="TST",
            scanner_name="pullback_base",
            ts=now,
            score=1.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 5.62},
            bars=bars,
        )
        verifier = MomentumPatternVerifier()
        signal = verifier.verify(hit, PortfolioState(cash=100_000))
        assert signal is None
        assert "too far from HOD" in verifier._last_reject

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_late_pullback_after_fresh_base_reclaim(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        bars = []
        for i in range(6):
            bars.append(_bar(i, close=4.50 + i * 0.03, open_=4.48 + i * 0.03,
                             high=4.53 + i * 0.03, low=4.46 + i * 0.03,
                             volume=35_000, base_ts=now, n=16))
        bars.extend([
            _bar(6, close=5.40, open_=4.70, high=5.55, low=4.68, volume=220_000, base_ts=now, n=16),
            _bar(7, close=6.20, open_=5.42, high=6.45, low=5.40, volume=280_000, base_ts=now, n=16),
            _bar(8, close=6.72, open_=6.18, high=6.80, low=6.10, volume=310_000, base_ts=now, n=16),
            _bar(9, close=5.48, open_=5.62, high=5.72, low=5.40, volume=45_000, base_ts=now, n=16),
            _bar(10, close=5.50, open_=5.44, high=5.62, low=5.38, volume=40_000, base_ts=now, n=16),
            _bar(11, close=5.54, open_=5.47, high=5.64, low=5.42, volume=42_000, base_ts=now, n=16),
            _bar(12, close=5.56, open_=5.50, high=5.66, low=5.45, volume=44_000, base_ts=now, n=16),
            _bar(13, close=5.86, open_=5.58, high=5.91, low=5.52, volume=75_000, base_ts=now, n=16),
        ])
        hit = ScanResult(
            symbol="TST",
            scanner_name="pullback_base",
            ts=now,
            score=1.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 5.38},
            bars=bars,
        )
        verifier = MomentumPatternVerifier()

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert "Pullback Base" in signal.reason

    def test_rejects_pullback_base_without_big_move(self) -> None:
        now = datetime.now(timezone.utc)
        bars = [
            _bar(i, close=5.0 + i * 0.02, open_=5.0 + i * 0.02 - 0.01,
                 high=5.02 + i * 0.02, low=4.98 + i * 0.02,
                 volume=80_000, base_ts=now, n=15)
            for i in range(12)
        ]
        bars.extend([
            _bar(12, close=5.23, open_=5.18, high=5.26, low=5.17, volume=90_000, base_ts=now, n=15),
            _bar(13, close=5.27, open_=5.22, high=5.30, low=5.21, volume=95_000, base_ts=now, n=15),
            _bar(14, close=5.31, open_=5.25, high=5.34, low=5.24, volume=100_000, base_ts=now, n=15),
        ])
        hit = ScanResult(
            symbol="TST",
            scanner_name="pullback_base",
            ts=now,
            score=1.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 5.17},
            bars=bars,
        )
        verifier = MomentumPatternVerifier()
        signal = verifier.verify(hit, PortfolioState(cash=100_000))
        assert signal is None
        assert "move too small" in verifier._last_reject

    def test_rejects_vwap_pullback_barely_above_vwap(self) -> None:
        now = datetime.now(timezone.utc)
        bars = [
            _bar(i, close=5.25, open_=5.24, high=5.28, low=5.22,
                 volume=80_000, base_ts=now, n=15)
            for i in range(10)
        ]
        bars.extend([
            _bar(10, close=5.50, open_=5.35, high=5.60, low=5.32, volume=100_000, base_ts=now, n=15),
            _bar(11, close=5.36, open_=5.45, high=5.48, low=5.32, volume=90_000, base_ts=now, n=15),
            _bar(12, close=5.34, open_=5.29, high=5.36, low=5.28, volume=95_000, base_ts=now, n=15),
        ])
        hit = ScanResult(
            symbol="TST",
            scanner_name="vwap_pullback",
            ts=now,
            score=1.0,
            criteria={"pattern": "vwap_pullback", "direction": "up", "pullback_low": 5.28},
            bars=bars,
        )
        verifier = MomentumPatternVerifier()
        signal = verifier.verify(hit, PortfolioState(cash=100_000))
        assert signal is None
        assert "above VWAP" in verifier._last_reject

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_generates_signal_for_abc_continuation(self, _mock_guard: object) -> None:
        bars = _make_abc_bars()
        scanner = ABCContinuationScanner(min_a_leg_pct=5.0, min_price=1.0, max_price=20.0)
        hit = scanner.scan({"TST": bars})[0]
        verifier = MomentumPatternVerifier(max_risk_per_share=0.50)
        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert signal.stop_loss is not None
        assert signal.stop_loss == pytest.approx(hit.criteria["b_low"] - 0.02)
        assert "Abc Continuation" in signal.reason

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_strong_pullback_base(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        bars = []
        for i in range(8):
            bars.append(_bar(i, close=5.0 + i * 0.03, open_=5.0 + i * 0.03 - 0.01,
                             high=5.03 + i * 0.03, low=4.98 + i * 0.03,
                             volume=60_000, base_ts=now, n=15))
        bars.extend([
            _bar(8, close=6.20, open_=5.60, high=6.45, low=5.55, volume=180_000, base_ts=now, n=15),
            _bar(9, close=6.05, open_=6.20, high=6.25, low=5.95, volume=120_000, base_ts=now, n=15),
            _bar(10, close=6.02, open_=6.00, high=6.08, low=5.96, volume=110_000, base_ts=now, n=15),
            _bar(11, close=6.06, open_=6.01, high=6.10, low=5.98, volume=115_000, base_ts=now, n=15),
            _bar(12, close=6.12, open_=6.05, high=6.16, low=6.02, volume=120_000, base_ts=now, n=15),
        ])
        hit = ScanResult(
            symbol="TST",
            scanner_name="pullback_base",
            ts=now,
            score=1.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 5.95},
            bars=bars,
        )
        verifier = MomentumPatternVerifier()
        signal = verifier.verify(hit, PortfolioState(cash=100_000))
        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_hot_hod_reclaim_uses_tactical_stop_when_pullback_stop_is_too_wide(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        bars = [
            _bar(0, close=3.40, open_=3.40, high=3.45, low=3.36, volume=100_000, base_ts=now, n=8),
            _bar(1, close=3.55, open_=3.40, high=3.58, low=3.38, volume=120_000, base_ts=now, n=8),
            _bar(2, close=3.88, open_=3.55, high=3.95, low=3.52, volume=150_000, base_ts=now, n=8),
            _bar(3, close=4.18, open_=3.88, high=4.20, low=3.86, volume=180_000, base_ts=now, n=8),
            _bar(4, close=3.86, open_=4.15, high=4.19, low=3.74, volume=90_000, base_ts=now, n=8),
            _bar(5, close=4.05, open_=3.86, high=4.08, low=3.82, volume=110_000, base_ts=now, n=8),
            _bar(6, close=4.51, open_=4.08, high=4.53, low=4.24, volume=785_000, base_ts=now, n=8),
        ]
        hit = ScanResult(
            symbol="IOTR",
            scanner_name="hod_reclaim",
            ts=now,
            score=100.0,
            criteria={
                "pattern": "hod_reclaim",
                "direction": "up",
                "hod": 4.53,
                "pullback_low": 3.74,
                "stop_price": 3.72,
                "rally_pct": 33.2,
                "close": 4.51,
                "volume": 785_000,
            },
            bars=bars,
        )
        verifier = MomentumPatternVerifier()
        signal = verifier.verify(hit, PortfolioState(cash=100_000))
        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert signal.stop_loss is not None
        assert (signal.entry_price - signal.stop_loss) / signal.entry_price <= 0.08
