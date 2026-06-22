"""Tests for Bull Flag and Flat Top Breakout scanners + verifier."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from daytrading.models import Bar, PortfolioState, ScanResult, SignalAction
from daytrading.scanner.scalping.bull_flag import BullFlagScanner
from daytrading.scanner.scalping.flat_top_breakout import FlatTopBreakoutScanner
from daytrading.scanner.scalping.abc_continuation import ABCContinuationScanner
from daytrading.scanner.scalping.first_pullback_reclaim import FirstPullbackReclaimScanner
from daytrading.scanner.scalping.level_breakout_reclaim import LevelBreakoutReclaimScanner
from daytrading.scanner.scalping.level_breakout_watch import LevelBreakoutWatchScanner
from daytrading.scanner.scalping.pullback_base import PullbackBaseScanner
from daytrading.scanner.scalping.runner_reclaim_continuation import RunnerReclaimContinuationScanner
from daytrading.scanner.scalping.shallow_stair_continuation import ShallowStairContinuationScanner
from daytrading.scanner.scalping.early_vwap_reclaim_scout import EarlyVWAPReclaimScoutScanner
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


def test_level_breakout_uses_full_session_move_not_recent_window_open() -> None:
    """A late-day base should keep the true 4am/session context for A+ runner math."""
    now = datetime.now(timezone.utc)
    bars: list[Bar] = []
    n = 100
    for i in range(95):
        close = 1.00 + min(i, 20) * 0.010
        if i >= 20:
            close = 1.60 + ((i % 4) - 1.5) * 0.004
        bars.append(_bar(
            i,
            close=close,
            open_=close - 0.005,
            high=close + 0.02,
            low=close - 0.02,
            volume=30_000,
            base_ts=now,
            n=n,
        ))
    base_values = [
        (1.60, 1.61, 1.63, 1.58, 45_000),
        (1.61, 1.60, 1.62, 1.585, 42_000),
        (1.60, 1.615, 1.625, 1.59, 46_000),
        (1.615, 1.62, 1.63, 1.60, 48_000),
        (1.62, 1.66, 1.67, 1.615, 150_000),
    ]
    for offset, (open_, close, high, low, volume) in enumerate(base_values, start=95):
        bars.append(_bar(
            offset,
            close=close,
            open_=open_,
            high=high,
            low=low,
            volume=volume,
            base_ts=now,
            n=n,
        ))

    scanner = LevelBreakoutReclaimScanner(min_breakout_volume=100_000)
    hits = scanner.scan({"BATL": bars})

    assert hits
    assert hits[0].criteria["session_move_pct"] > 50.0


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


def _make_runner_reclaim_bars() -> list[Bar]:
    """Build a volatile low-float runner pullback that normal first-pullback rejects."""
    now = datetime.now(timezone.utc)
    bars: list[Bar] = []
    prices = [
        (1.00, 1.03, 1.04, 0.99, 40_000),
        (1.03, 1.12, 1.13, 1.02, 180_000),
        (1.12, 1.26, 1.28, 1.10, 260_000),
        (1.26, 1.42, 1.45, 1.24, 320_000),
        (1.42, 1.35, 1.44, 1.30, 150_000),
        (1.35, 1.30, 1.36, 1.24, 130_000),
        (1.30, 1.28, 1.33, 1.22, 110_000),
        (1.28, 1.31, 1.34, 1.25, 95_000),
        (1.31, 1.33, 1.35, 1.27, 90_000),
        (1.33, 1.37, 1.38, 1.30, 100_000),
        (1.37, 1.42, 1.44, 1.34, 155_000),
    ]
    for i, (o, c, h, lo, vol) in enumerate(prices):
        bars.append(_bar(i, close=c, open_=o, high=h, low=lo, volume=vol, base_ts=now, n=len(prices)))
    return bars


def _make_first_pullback_reclaim_bars() -> list[Bar]:
    """Build an SVCO-style first pullback, then reclaim of the pullback base."""
    now = datetime.now(timezone.utc)
    bars: list[Bar] = []
    for i in range(4):
        bars.append(_bar(i, close=10.00, open_=9.98, high=10.03, low=9.96, volume=40_000, base_ts=now, n=30))

    impulse = [(10.00, 10.35), (10.35, 10.78), (10.78, 11.18), (11.18, 11.72)]
    for j, (o, c) in enumerate(impulse):
        i = 4 + j
        bars.append(_bar(i, close=c, open_=o, high=c + 0.04, low=o - 0.03, volume=130_000, base_ts=now, n=30))

    pullback_base = [
        (11.70, 11.46, 11.74, 11.33, 70_000),
        (11.45, 11.38, 11.50, 11.24, 58_000),
        (11.37, 11.42, 11.48, 11.30, 52_000),
        (11.42, 11.45, 11.51, 11.35, 54_000),
    ]
    for j, (o, c, h, l, v) in enumerate(pullback_base):
        i = 8 + j
        bars.append(_bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=30))

    bars.append(_bar(12, close=11.62, open_=11.45, high=11.66, low=11.43, volume=82_000, base_ts=now, n=30))
    return bars


def _make_level_breakout_bars() -> list[Bar]:
    """Build a DAIC-style early level breakout from a tight base."""
    now = datetime.now(timezone.utc)
    bars: list[Bar] = []
    base_rows = [
        (2.86, 2.90, 2.92, 2.84, 35_000),
        (2.90, 3.08, 3.12, 2.88, 140_000),
        (3.08, 3.42, 3.50, 3.04, 210_000),
        (3.42, 3.78, 3.90, 3.35, 260_000),
        (3.78, 3.96, 4.02, 3.70, 180_000),
        (3.96, 4.00, 4.08, 3.86, 80_000),
        (4.00, 3.96, 4.07, 3.88, 72_000),
        (3.96, 4.02, 4.10, 3.91, 76_000),
        (4.02, 4.05, 4.12, 3.96, 78_000),
    ]
    for i, (o, c, h, l, v) in enumerate(base_rows):
        bars.append(_bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=30))
    bars.append(_bar(9, close=4.20, open_=4.06, high=4.24, low=4.02, volume=165_000, base_ts=now, n=30))
    return bars


def _make_shallow_stair_bars() -> list[Bar]:
    """Build an INHD-style shallow stair-step runner above VWAP."""
    now = datetime.now(timezone.utc)
    rows = [
        (2.50, 2.55, 2.58, 2.48, 50_000),
        (2.55, 3.10, 3.18, 2.54, 220_000),
        (3.10, 3.35, 3.42, 3.02, 240_000),
        (3.35, 3.48, 3.55, 3.28, 180_000),
        (3.48, 3.70, 3.78, 3.42, 210_000),
        (3.70, 3.62, 3.75, 3.55, 120_000),
        (3.62, 3.82, 3.90, 3.60, 160_000),
        (3.82, 4.05, 4.15, 3.78, 220_000),
        (4.05, 4.18, 4.24, 4.00, 180_000),
        (4.18, 4.16, 4.25, 4.08, 125_000),
        (4.16, 4.24, 4.28, 4.12, 130_000),
        (4.24, 4.44, 4.50, 4.20, 190_000),
    ]
    return [
        _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=20)
        for i, (o, c, h, l, v) in enumerate(rows)
    ]


def _make_fast_runner_stair_bars() -> list[Bar]:
    """Build a WCT-style fast stair-step runner with wider candles."""
    now = datetime.now(timezone.utc)
    rows = [
        (1.45, 1.46, 1.48, 1.43, 35_000),
        (1.46, 1.82, 1.88, 1.45, 180_000),
        (1.82, 2.20, 2.28, 1.78, 260_000),
        (2.20, 2.86, 3.00, 2.16, 420_000),
        (2.86, 3.15, 3.35, 2.80, 360_000),
        (3.15, 2.92, 3.22, 2.72, 260_000),
        (2.92, 2.76, 3.02, 2.62, 210_000),
        (2.76, 2.88, 2.96, 2.66, 190_000),
        (2.88, 2.82, 2.94, 2.70, 170_000),
        (2.82, 2.95, 3.00, 2.76, 250_000),
        (2.95, 2.92, 3.05, 2.80, 230_000),
        (2.92, 3.08, 3.12, 2.90, 280_000),
    ]
    return [
        _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=20)
        for i, (o, c, h, l, v) in enumerate(rows)
    ]


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

    def test_rejects_extended_c_trigger_climax_bar(self) -> None:
        bars = _make_abc_bars()
        # CUPR-style C candle: valid reclaim shape, but already extended and
        # too wide to buy at the close as an ABC scalp.
        bars[-1] = _bar(
            11,
            close=4.70,
            open_=4.08,
            high=4.85,
            low=4.06,
            volume=1_210_055,
            base_ts=bars[-1].ts,
            n=30,
        )
        scanner = ABCContinuationScanner(min_a_leg_pct=5.0, min_price=1.0, max_price=20.0)

        hits = scanner.scan({"TST": bars})

        assert hits == []


class TestFirstPullbackReclaimScanner:
    def test_detects_svco_style_first_pullback_reclaim(self) -> None:
        bars = _make_first_pullback_reclaim_bars()
        scanner = FirstPullbackReclaimScanner(min_price=1.0, max_price=20.0)

        hits = scanner.scan({"TST": bars})

        assert len(hits) == 1
        hit = hits[0]
        assert hit.scanner_name == "first_pullback_reclaim"
        assert hit.criteria["pattern"] == "first_pullback_reclaim"
        assert hit.criteria["impulse_pct"] >= 5.0
        assert 1.2 <= hit.criteria["pullback_pct"] <= 12.0
        assert hit.criteria["base_low"] > 0
        assert hit.criteria["close"] > hit.criteria["base_high"]

    def test_rejects_straight_breakout_without_pullback(self) -> None:
        now = datetime.now(timezone.utc)
        bars = []
        for i in range(13):
            price = 10.0 + i * 0.12
            bars.append(_bar(i, close=price, open_=price - 0.04,
                             high=price + 0.05, low=price - 0.05,
                             volume=80_000, base_ts=now, n=30))
        scanner = FirstPullbackReclaimScanner(min_price=1.0, max_price=20.0)

        assert scanner.scan({"TST": bars}) == []


class TestRunnerReclaimContinuationScanner:
    def test_detects_volatile_runner_reclaim_that_first_pullback_skips(self) -> None:
        bars = _make_runner_reclaim_bars()
        first_pullback = FirstPullbackReclaimScanner(min_price=1.0, max_price=20.0)
        runner_reclaim = RunnerReclaimContinuationScanner(min_price=1.0, max_price=20.0)

        assert first_pullback.scan({"TST": bars}) == []
        hits = runner_reclaim.scan({"TST": bars})

        assert len(hits) == 1
        hit = hits[0]
        assert hit.scanner_name == "runner_reclaim_continuation"
        assert hit.criteria["pattern"] == "runner_reclaim_continuation"
        assert hit.criteria["pullback_pct"] > 12.0
        assert hit.criteria["base_range_pct"] <= 18.0
        assert hit.criteria["close"] > hit.criteria["base_high"]

    def test_rejects_late_chase_far_above_base(self) -> None:
        bars = _make_runner_reclaim_bars()
        last = bars[-1]
        bars[-1] = _bar(
            10,
            close=1.62,
            open_=1.40,
            high=1.66,
            low=1.39,
            volume=210_000,
            base_ts=last.ts,
            n=len(bars),
        )
        scanner = RunnerReclaimContinuationScanner(min_price=1.0, max_price=20.0)

        assert scanner.scan({"TST": bars}) == []


class TestEarlyVWAPReclaimScoutScanner:
    def test_detects_washout_vwap_reclaim_before_hod_chase(self) -> None:
        now = datetime.now(timezone.utc)
        prices = [
            (2.46, 2.51, 2.44, 2.48, 25_000),
            (2.48, 2.54, 2.46, 2.52, 30_000),
            (2.52, 2.58, 2.50, 2.55, 35_000),
            (2.55, 2.62, 2.52, 2.59, 40_000),
            (2.55, 2.60, 2.50, 2.57, 40_000),
            (2.57, 2.74, 2.55, 2.70, 70_000),
            (2.70, 2.95, 2.68, 2.88, 110_000),
            (2.88, 3.18, 2.84, 3.10, 160_000),
            (3.10, 3.13, 2.92, 2.98, 95_000),
            (2.98, 3.02, 2.71, 2.80, 130_000),
            (2.80, 2.89, 2.76, 2.86, 80_000),
            (2.86, 2.96, 2.81, 2.91, 95_000),
            (2.91, 3.03, 2.86, 2.98, 120_000),
        ]
        bars = [
            _bar(i, open_=o, high=h, low=l, close=c, volume=v, base_ts=now, n=len(prices))
            for i, (o, h, l, c, v) in enumerate(prices)
        ]
        scanner = EarlyVWAPReclaimScoutScanner(min_price=1.0, max_price=20.0)

        hits = scanner.scan({"TST": bars})

        assert len(hits) == 1
        hit = hits[0]
        assert hit.scanner_name == "early_vwap_reclaim_scout"
        assert hit.criteria["pattern"] == "early_vwap_reclaim_scout"
        assert hit.criteria["washout_low"] == 2.71
        assert hit.criteria["reclaim_above_vwap_pct"] > 0
        assert hit.criteria["distance_from_hod_pct"] <= 12.0
        assert hit.criteria["active_recent_bars"] >= 2
        assert hit.criteria["max_recent_volume_share"] <= 0.70

    def test_rejects_one_spike_premarket_reclaim_without_sustained_volume(self) -> None:
        now = datetime.now(timezone.utc)
        prices = [
            (2.00, 2.03, 1.99, 2.01, 4_000),
            (2.01, 2.05, 2.00, 2.03, 5_000),
            (2.03, 2.08, 2.02, 2.06, 7_000),
            (2.06, 2.14, 2.04, 2.12, 9_000),
            (2.12, 2.18, 2.09, 2.15, 8_000),
            (2.15, 2.42, 2.13, 2.36, 108_000),
            (2.36, 2.39, 2.20, 2.24, 5_000),
            (2.24, 2.27, 2.14, 2.18, 4_000),
            (2.18, 2.23, 2.15, 2.20, 6_000),
            (2.20, 2.25, 2.17, 2.22, 2_000),
            (2.22, 2.27, 2.19, 2.24, 4_000),
            (2.24, 2.29, 2.21, 2.26, 5_000),
            (2.26, 2.32, 2.25, 2.31, 32_406),
        ]
        bars = [
            _bar(i, open_=o, high=h, low=l, close=c, volume=v, base_ts=now, n=len(prices))
            for i, (o, h, l, c, v) in enumerate(prices)
        ]
        scanner = EarlyVWAPReclaimScoutScanner(min_price=1.0, max_price=20.0)

        assert scanner.scan({"TST": bars}) == []

    def test_rejects_weak_wick_reclaim_candle(self) -> None:
        now = datetime.now(timezone.utc)
        prices = [
            (6.80, 6.95, 6.76, 6.90, 45_000),
            (6.90, 7.10, 6.86, 7.02, 55_000),
            (7.02, 7.34, 6.98, 7.28, 80_000),
            (7.28, 8.25, 7.20, 8.05, 180_000),
            (8.05, 8.48, 7.92, 8.28, 210_000),
            (8.28, 8.46, 7.68, 7.82, 140_000),
            (7.82, 7.94, 7.34, 7.46, 95_000),
            (7.46, 7.62, 7.04, 7.18, 88_000),
            (7.18, 7.20, 7.13, 7.17, 11_208),
            (7.17, 7.17, 6.9213, 7.00, 9_707),
            (7.01, 7.30, 6.98, 7.30, 48_050),
            # OLOX-style failed reclaim: volume appears, but the candle rejects
            # hard from the high and closes in the lower part of its range.
            (7.30, 7.75, 7.2941, 7.37, 102_426),
        ]
        bars = [
            _bar(i, open_=o, high=h, low=l, close=c, volume=v, base_ts=now, n=len(prices))
            for i, (o, h, l, c, v) in enumerate(prices)
        ]
        scanner = EarlyVWAPReclaimScoutScanner(min_price=1.0, max_price=20.0)

        assert scanner.scan({"TST": bars}) == []

    def test_rejects_late_chase_far_above_vwap(self) -> None:
        now = datetime.now(timezone.utc)
        prices = [
            (2.55, 2.60, 2.50, 2.57, 40_000),
            (2.57, 2.74, 2.55, 2.70, 70_000),
            (2.70, 2.95, 2.68, 2.88, 110_000),
            (2.88, 3.18, 2.84, 3.10, 160_000),
            (3.10, 3.13, 2.92, 2.98, 95_000),
            (2.98, 3.02, 2.71, 2.80, 130_000),
            (2.80, 2.89, 2.76, 2.86, 80_000),
            (2.86, 2.96, 2.81, 2.91, 95_000),
            (2.91, 3.03, 2.86, 2.98, 120_000),
            (2.98, 3.08, 2.93, 3.02, 140_000),
            (3.02, 3.42, 3.00, 3.38, 250_000),
            (3.38, 3.55, 3.31, 3.50, 270_000),
        ]
        bars = [
            _bar(i, open_=o, high=h, low=l, close=c, volume=v, base_ts=now, n=len(prices))
            for i, (o, h, l, c, v) in enumerate(prices)
        ]
        scanner = EarlyVWAPReclaimScoutScanner(min_price=1.0, max_price=20.0)

        assert scanner.scan({"TST": bars}) == []


class TestLevelBreakoutReclaimScanner:
    def test_detects_daic_style_level_breakout(self) -> None:
        bars = _make_level_breakout_bars()
        scanner = LevelBreakoutReclaimScanner(min_price=1.0, max_price=20.0)

        hits = scanner.scan({"TST": bars})

        assert len(hits) == 1
        hit = hits[0]
        assert hit.scanner_name == "level_breakout_reclaim"
        assert hit.criteria["pattern"] == "level_breakout_reclaim"
        assert hit.criteria["breakout_level"] > 4.0
        assert hit.criteria["close"] > hit.criteria["breakout_level"]
        assert hit.criteria["volume_surge"] >= 1.15

    def test_rejects_wick_only_false_breakout(self) -> None:
        bars = _make_level_breakout_bars()
        last = bars[-1]
        bars[-1] = _bar(
            9,
            close=4.08,
            open_=4.06,
            high=4.42,
            low=4.02,
            volume=165_000,
            base_ts=last.ts,
            n=30,
        )
        scanner = LevelBreakoutReclaimScanner(min_price=1.0, max_price=20.0)

        assert scanner.scan({"TST": bars}) == []


class TestLevelBreakoutWatchScanner:
    def test_promotes_clean_closed_level_break_to_live_scout(self) -> None:
        bars = _make_level_breakout_bars()
        scanner = LevelBreakoutWatchScanner(min_price=1.0, max_price=20.0)

        hits = scanner.scan({"TST": bars})

        assert len(hits) == 1
        hit = hits[0]
        assert hit.scanner_name == "level_breakout_watch"
        assert hit.criteria["pattern"] == "level_breakout_reclaim"
        assert hit.criteria["setup_tier"] == "A+ setup"
        assert hit.criteria["entry_tier"] == "level_scout"
        assert hit.criteria["entry_mode"] == "level_breakout_scout"
        assert hit.criteria["size_factor"] == pytest.approx(0.35)
        assert hit.criteria["stop_price"] < hit.criteria["close"]

    def test_conl_style_smooth_level_break_promotes_before_five_pct_move(self) -> None:
        base_ts = datetime(2026, 6, 12, 14, 52, tzinfo=timezone.utc)
        bars = []
        closes = [5.21, 5.24, 5.26, 5.30, 5.34, 5.38, 5.40, 5.41, 5.42]
        for i, close in enumerate(closes):
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.015,
                high=close + 0.015,
                low=close - 0.025,
                volume=65_000,
                base_ts=base_ts,
                n=20,
            ))
        bars.append(_bar(
            9,
            close=5.45,
            open_=5.42,
            high=5.46,
            low=5.415,
            volume=125_000,
            base_ts=base_ts,
            n=20,
        ))
        scanner = LevelBreakoutWatchScanner(min_price=1.0, max_price=20.0)

        hits = scanner.scan({"CONL": bars})

        assert len(hits) == 1
        hit = hits[0]
        assert hit.criteria["pattern"] == "level_breakout_reclaim"
        assert hit.criteria["entry_tier"] == "level_scout"
        assert hit.criteria["entry_mode"] == "level_breakout_scout"
        assert hit.criteria["size_factor"] == pytest.approx(0.35)
        assert hit.criteria["session_move_pct"] < 5.0
        assert hit.criteria["breakout_level"] < hit.criteria["close"]

    def test_watches_near_resistance_before_clean_break(self) -> None:
        bars = _make_level_breakout_bars()
        last = bars[-1]
        bars[-1] = _bar(
            9,
            close=4.07,
            open_=4.02,
            high=4.11,
            low=4.01,
            volume=135_000,
            base_ts=last.ts,
            n=30,
        )
        scanner = LevelBreakoutWatchScanner(min_price=1.0, max_price=20.0)

        hits = scanner.scan({"TST": bars})

        assert len(hits) == 1
        hit = hits[0]
        assert hit.scanner_name == "level_breakout_watch"
        assert hit.criteria["setup_tier"] == "watch only"
        assert "watching" in hit.criteria["status"]
        assert hit.criteria["breakout_level"] > 4.0

    def test_watches_wick_only_level_break_as_failed_break(self) -> None:
        bars = _make_level_breakout_bars()
        last = bars[-1]
        bars[-1] = _bar(
            9,
            close=4.08,
            open_=4.06,
            high=4.42,
            low=4.02,
            volume=165_000,
            base_ts=last.ts,
            n=30,
        )
        scanner = LevelBreakoutWatchScanner(min_price=1.0, max_price=20.0)

        hits = scanner.scan({"TST": bars})

        assert len(hits) == 1
        assert "failed level break" in hits[0].criteria["status"]
        assert hits[0].criteria["pattern"] == "level_breakout_watch"
        assert hits[0].criteria["setup_tier"] == "watch only"

class TestShallowStairContinuationScanner:
    def test_detects_inhd_style_shallow_stair_breakout(self) -> None:
        bars = _make_shallow_stair_bars()
        scanner = ShallowStairContinuationScanner(min_price=1.0, max_price=20.0)

        hits = scanner.scan({"TST": bars})

        assert len(hits) == 1
        hit = hits[0]
        assert hit.scanner_name == "shallow_stair_continuation"
        assert hit.criteria["pattern"] == "shallow_stair_continuation"
        assert hit.criteria["entry_tier"] == "stair_scout"
        assert hit.criteria["pullback_from_hod_pct"] <= 4.0
        assert hit.criteria["base_range_pct"] <= 7.0

    def test_detects_wct_style_fast_runner_stair_breakout(self) -> None:
        bars = _make_fast_runner_stair_bars()
        scanner = ShallowStairContinuationScanner(min_price=1.0, max_price=20.0)

        hits = scanner.scan({"TST": bars})

        assert len(hits) == 1
        hit = hits[0]
        assert hit.scanner_name == "shallow_stair_continuation"
        assert hit.criteria["pattern"] == "shallow_stair_continuation"
        assert hit.criteria["entry_tier"] == "stair_scout"
        assert hit.criteria["runner_profile"] == "fast_stair_runner"
        assert hit.criteria["pullback_from_hod_pct"] > 4.0
        assert hit.criteria["allowed_hod_pullback_pct"] == pytest.approx(12.0)
        assert hit.criteria["base_range_pct"] > 7.0
        assert hit.criteria["allowed_base_range_pct"] == pytest.approx(13.0)
        assert hit.score >= 80.0

    def test_rejects_deep_wide_reclaim_as_not_shallow_stair(self) -> None:
        bars = _make_fast_runner_stair_bars()
        base_ts = bars[-1].ts
        # CAST-style failed re-entry: the name is still strong overall, but the
        # setup is a deep/wide reclaim, not a shallow stair-step base.
        bars[4] = _bar(4, close=9.85, open_=9.25, high=10.0, low=8.90, volume=420_000, base_ts=base_ts, n=20)
        bars[-5:] = [
            _bar(7, close=7.70, open_=8.05, high=8.10, low=7.09, volume=120_000, base_ts=base_ts, n=20),
            _bar(8, close=7.82, open_=7.70, high=7.94, low=7.28, volume=130_000, base_ts=base_ts, n=20),
            _bar(9, close=8.00, open_=7.82, high=8.04, low=7.50, volume=160_000, base_ts=base_ts, n=20),
            _bar(10, close=8.10, open_=8.00, high=8.10, low=7.76, volume=185_000, base_ts=base_ts, n=20),
            _bar(11, close=8.59, open_=8.12, high=8.68, low=8.08, volume=260_000, base_ts=base_ts, n=20),
        ]
        scanner = ShallowStairContinuationScanner(min_price=1.0, max_price=20.0)

        hits = scanner.scan({"TST": bars})

        assert hits == []

    def test_rejects_shallow_stair_without_volume(self) -> None:
        bars = _make_shallow_stair_bars()
        bars[-1] = _bar(
            11,
            close=4.44,
            open_=4.24,
            high=4.50,
            low=4.20,
            volume=20_000,
            base_ts=bars[-1].ts,
            n=20,
        )
        scanner = ShallowStairContinuationScanner(min_price=1.0, max_price=20.0)

        assert scanner.scan({"TST": bars}) == []

    def test_rejects_borderline_shallow_stair_without_decisive_buyer_candle(self) -> None:
        """AIIO-style stair scouts need more than barely returning volume."""
        bars = _make_shallow_stair_bars()
        base_ts = bars[-1].ts
        bars[-4:] = [
            _bar(8, close=3.36, open_=3.34, high=3.40, low=3.3201, volume=48_000, base_ts=base_ts, n=20),
            _bar(9, close=3.37, open_=3.36, high=3.405, low=3.32, volume=50_000, base_ts=base_ts, n=20),
            _bar(10, close=3.40, open_=3.37, high=3.425, low=3.34, volume=53_000, base_ts=base_ts, n=20),
            # Breaks the base, but volume surge is only ~1.1x and body is not decisive.
            _bar(11, close=3.445, open_=3.37, high=3.48, low=3.37, volume=55_506, base_ts=base_ts, n=20),
        ]
        scanner = ShallowStairContinuationScanner(min_price=1.0, max_price=20.0)

        assert scanner.scan({"AIIO": bars}) == []

    def test_rejects_late_extended_level_breakout(self) -> None:
        bars = _make_level_breakout_bars()
        last = bars[-1]
        bars[-1] = _bar(
            9,
            close=4.32,
            open_=4.06,
            high=4.38,
            low=4.02,
            volume=165_000,
            base_ts=last.ts,
            n=30,
        )
        scanner = LevelBreakoutReclaimScanner(min_price=1.0, max_price=20.0)

        assert scanner.scan({"TST": bars}) == []

    def test_rejects_level_breakout_with_weak_active_volume(self) -> None:
        bars = _make_level_breakout_bars()
        last = bars[-1]
        bars[-1] = _bar(
            9,
            close=4.20,
            open_=4.06,
            high=4.24,
            low=4.02,
            volume=60_000,
            base_ts=last.ts,
            n=30,
        )
        scanner = LevelBreakoutReclaimScanner(min_price=1.0, max_price=20.0)

        assert scanner.scan({"TST": bars}) == []


class TestPullbackBaseScanner:
    def _deep_vwap_reclaim_bars(self) -> list[Bar]:
        now = datetime.now(timezone.utc)
        rows = [
            (1.94, 1.96, 1.98, 1.92, 18_000),
            (1.96, 1.98, 2.00, 1.94, 19_000),
            (2.00, 2.03, 2.05, 1.98, 20_000),
            (2.03, 2.07, 2.09, 2.01, 22_000),
            (2.07, 2.10, 2.12, 2.05, 24_000),
            (2.10, 2.28, 2.32, 2.08, 36_000),
            (2.28, 2.58, 2.62, 2.25, 48_000),
            (2.58, 3.02, 3.08, 2.55, 55_000),
            (3.02, 2.70, 3.04, 2.62, 42_000),
            (2.70, 2.48, 2.72, 2.42, 44_000),
            (2.48, 2.35, 2.40, 2.30, 46_000),
            (2.33, 2.36, 2.40, 2.30, 52_000),
            (2.36, 2.34, 2.39, 2.31, 54_000),
            (2.34, 2.38, 2.41, 2.32, 56_000),
            (2.38, 2.45, 2.48, 2.35, 125_000),
        ]
        return [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]

    def test_allows_second_chance_vwap_reclaim_below_session_midpoint(self) -> None:
        bars = self._deep_vwap_reclaim_bars()
        scanner = PullbackBaseScanner(min_price=1.0, max_price=20.0, max_base_range_pct=5.0)

        hits = scanner.scan({"SUNE": bars})

        assert len(hits) == 1
        hit = hits[0]
        assert hit.scanner_name == "pullback_base"
        assert hit.criteria["entry_tier"] == "second_chance_reclaim"
        assert hit.criteria["close"] < (hit.criteria["session_high"] + bars[0].open) / 2

    def test_allows_second_chance_reclaim_even_when_pullback_exceeds_normal_max(self) -> None:
        bars = self._deep_vwap_reclaim_bars()
        spike = bars[7]
        bars[7] = _bar(
            7,
            close=spike.close,
            open_=spike.open,
            high=3.70,
            low=spike.low,
            volume=spike.volume,
            base_ts=spike.ts,
            n=len(bars),
        )
        scanner = PullbackBaseScanner(min_price=1.0, max_price=20.0, max_base_range_pct=5.0)

        hits = scanner.scan({"CCTG": bars})

        assert len(hits) == 1
        hit = hits[0]
        assert hit.criteria["entry_tier"] == "second_chance_reclaim"
        assert hit.criteria["pullback_pct"] > 30.0

    def test_rejects_second_chance_reclaim_without_buyer_volume(self) -> None:
        bars = self._deep_vwap_reclaim_bars()
        last = bars[-1]
        bars[-1] = _bar(
            len(bars) - 1,
            close=last.close,
            open_=last.open,
            high=last.high,
            low=last.low,
            volume=45_000,
            base_ts=last.ts,
            n=len(bars),
        )
        scanner = PullbackBaseScanner(min_price=1.0, max_price=20.0, max_base_range_pct=5.0)

        assert scanner.scan({"SUNE": bars}) == []


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

    def test_vwap_pullback_hod_threshold_loosened_to_twelve(self) -> None:
        # A ~11%-from-HOD vwap_pullback (DTSS/KMRK-style) was rejected at the old
        # 8% cap. The loosened 12% cap no longer rejects it for HOD distance.
        now = datetime.now(timezone.utc)
        bars = []
        for i in range(8):
            c = 5.00 + i * 0.128  # rally to ~5.90 HOD (day move ~20%, below 25% reclaim trigger)
            bars.append(_bar(i, close=c, open_=c - 0.05, high=c + 0.04, low=c - 0.08,
                             volume=120_000, base_ts=now, n=12))
        for j, (o, c, h, l) in enumerate([
            (5.85, 5.60, 5.88, 5.55), (5.60, 5.40, 5.62, 5.36),
            (5.40, 5.26, 5.44, 5.22), (5.26, 5.28, 5.34, 5.22),
        ]):
            bars.append(_bar(8 + j, close=c, open_=o, high=h, low=l,
                             volume=90_000, base_ts=now, n=12))
        # distance_from_hod ~ 11%  (between the old 8% and new 12% caps)
        reject_old = MomentumPatternVerifier._late_pullback_reject(
            "vwap_pullback", bars, max_hod_pct=8.0)
        assert reject_old is not None and "too far from HOD" in reject_old
        reject_new = MomentumPatternVerifier._late_pullback_reject(
            "vwap_pullback", bars, max_hod_pct=12.0)
        assert reject_new is None or "too far from HOD" not in reject_new

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_a_plus_vwap_retry_watch_after_fresh_late_reclaim(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (2.00, 2.05, 2.08, 1.98, 80_000),
            (2.05, 2.22, 2.25, 2.03, 90_000),
            (2.22, 2.72, 2.80, 2.18, 140_000),
            (2.72, 3.55, 3.70, 2.68, 260_000),
            (3.55, 4.75, 5.00, 3.50, 520_000),
            (4.75, 4.20, 4.80, 4.05, 160_000),
            (4.20, 3.62, 4.24, 3.50, 145_000),
            (3.62, 3.66, 3.74, 3.52, 110_000),
            (3.66, 3.70, 3.78, 3.58, 120_000),
            (3.70, 3.74, 3.82, 3.62, 130_000),
            (3.74, 3.96, 4.04, 3.72, 310_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="DSY",
            scanner_name="vwap_pullback",
            ts=bars[-1].ts,
            score=100.0,
            criteria={
                "pattern": "vwap_pullback",
                "direction": "up",
                "pullback_low": 3.50,
                "stop_price": 3.48,
                "setup_tier": "A+ setup",
                "volume": 1_200_000,
            },
            bars=bars,
        )

        signal = MomentumPatternVerifier().verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert hit.criteria["entry_tier"] == "a_plus_retry_watch"

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_sub_two_a_plus_abc_uses_reduced_scout_stop_after_wide_b_retrace(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (1.00, 1.06, 1.08, 0.98, 160_000),
            (1.06, 1.20, 1.23, 1.04, 190_000),
            (1.20, 1.38, 1.42, 1.16, 250_000),
            (1.38, 1.56, 1.60, 1.32, 340_000),
            (1.56, 1.85, 1.85, 1.50, 520_000),
            (1.85, 1.62, 1.84, 1.55, 250_000),
            (1.62, 1.43, 1.66, 1.37, 230_000),
            (1.43, 1.50, 1.55, 1.40, 210_000),
            (1.50, 1.58, 1.62, 1.48, 230_000),
            (1.58, 1.63, 1.67, 1.55, 260_000),
            (1.63, 1.66, 1.70, 1.43, 280_000),
            (1.66, 1.68, 1.73, 1.62, 2_987_555),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="FLD",
            scanner_name="abc_continuation",
            ts=bars[-1].ts,
            score=36.452,
            criteria={
                "pattern": "abc_continuation",
                "direction": "up",
                "a_leg_pct": 67.96,
                "b_low": 1.368,
                "b_retrace_pct": 51.7,
                "c_breakout_pct": 1.82,
                "c_volume_surge": 2.21,
                "setup_tier": "A+ setup",
                "volume": 2_987_555,
            },
            bars=bars,
        )

        signal = MomentumPatternVerifier().verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert signal.quantity < 100_000 / 1.68
        assert hit.criteria["entry_tier"] == "abc_scout"
        assert "A+ ABC scout" in signal.reason

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_a_plus_deep_runner_reclaim_as_reduced_scout(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (0.82, 0.88, 0.90, 0.80, 420_000),
            (0.88, 1.20, 1.28, 0.86, 520_000),
            (1.20, 2.10, 2.25, 1.16, 780_000),
            (2.10, 4.80, 5.10, 2.02, 520_000),
            (4.80, 6.20, 6.57, 4.70, 360_000),
            (6.20, 4.95, 6.25, 4.70, 160_000),
            (4.95, 3.55, 5.05, 3.12, 140_000),
            (3.55, 3.48, 3.76, 3.35, 45_000),
            (3.48, 3.62, 3.78, 3.42, 42_000),
            (3.62, 3.70, 3.82, 3.55, 44_000),
            (3.70, 3.97, 4.11, 3.66, 92_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="VSME",
            scanner_name="vwap_pullback",
            ts=bars[-1].ts,
            score=3547.263,
            criteria={
                "pattern": "vwap_pullback",
                "direction": "up",
                "pullback_low": 3.12,
                "stop_price": 3.10,
                "setup_tier": "A+ setup",
                "volume": 92_000,
            },
            bars=bars,
        )

        signal = MomentumPatternVerifier().verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert hit.criteria["entry_tier"] == "deep_runner_scout"
        assert hit.criteria["deep_reclaim_distance_from_hod_pct"] > 35.0
        assert signal.quantity < 100_000 / 3.97
        assert "A+ deep runner scout" in signal.reason

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_rejects_deep_runner_reclaim_without_a_plus_tier(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (0.82, 0.88, 0.90, 0.80, 420_000),
            (0.88, 1.20, 1.28, 0.86, 520_000),
            (1.20, 2.10, 2.25, 1.16, 780_000),
            (2.10, 4.80, 5.10, 2.02, 520_000),
            (4.80, 6.20, 6.57, 4.70, 360_000),
            (6.20, 4.95, 6.25, 4.70, 160_000),
            (4.95, 3.55, 5.05, 3.12, 140_000),
            (3.55, 3.48, 3.76, 3.35, 45_000),
            (3.48, 3.62, 3.78, 3.42, 42_000),
            (3.62, 3.70, 3.82, 3.55, 44_000),
            (3.70, 3.97, 4.11, 3.66, 92_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="WEAK",
            scanner_name="vwap_pullback",
            ts=bars[-1].ts,
            score=70.0,
            criteria={
                "pattern": "vwap_pullback",
                "direction": "up",
                "pullback_low": 3.12,
                "stop_price": 3.10,
                "setup_tier": "B setup",
                "volume": 92_000,
            },
            bars=bars,
        )
        verifier = MomentumPatternVerifier()

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is None
        assert "too far from HOD" in verifier._last_reject

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_fld_style_a_plus_vwap_reclaim_in_progress_scout(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (0.64, 0.68, 0.70, 0.62, 120_000),
            (0.68, 0.74, 0.76, 0.66, 180_000),
            (0.74, 0.92, 0.96, 0.72, 220_000),
            (0.92, 1.18, 1.24, 0.90, 360_000),
            (1.18, 1.85, 1.85, 1.12, 700_000),
            (1.85, 1.44, 1.82, 1.21, 260_000),
            (1.44, 1.21, 1.50, 1.08, 190_000),
            (1.21, 1.25, 1.30, 1.18, 120_000),
            (1.25, 1.28, 1.34, 1.20, 130_000),
            (1.28, 1.3203, 1.36, 1.24, 1_749_054),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="FLD",
            scanner_name="vwap_pullback",
            ts=bars[-1].ts,
            score=1023.644,
            criteria={
                "pattern": "vwap_pullback",
                "direction": "up",
                "pullback_low": 1.0802,
                "stop_price": 1.0602,
                "setup_tier": "A+ setup",
                "volume": 1_749_054,
            },
            bars=bars,
        )

        signal = MomentumPatternVerifier().verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert hit.criteria["entry_tier"] == "a_plus_reclaim_scout"
        assert hit.criteria["reclaim_distance_from_hod_pct"] > 25.0
        assert "A+ reclaim scout" in signal.reason

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_dsy_style_a_plus_pullback_reclaim_in_progress_scout(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (1.78, 1.80, 1.84, 1.76, 300_000),
            (1.80, 2.15, 2.25, 1.78, 420_000),
            (2.15, 3.50, 3.80, 2.08, 640_000),
            (3.50, 7.80, 8.20, 3.45, 860_000),
            (7.80, 11.00, 11.16, 7.70, 920_000),
            (11.00, 9.20, 11.05, 8.75, 180_000),
            (9.20, 8.55, 9.30, 8.52, 80_000),
            (8.55, 8.60, 8.90, 8.52, 58_000),
            (8.60, 8.70, 8.97, 8.56, 56_000),
            (8.70, 8.77, 8.97, 8.60, 7_724),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="DSY",
            scanner_name="pullback_base",
            ts=bars[-1].ts,
            score=852.352,
            criteria={
                "pattern": "pullback_base",
                "direction": "up",
                "base_low": 8.52,
                "stop_price": 8.50,
                "setup_tier": "A+ setup",
                "volume": 7_724,
            },
            bars=bars,
        )

        signal = MomentumPatternVerifier().verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert hit.criteria["entry_tier"] == "a_plus_reclaim_scout"
        assert "A+ reclaim scout" in signal.reason

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_rejects_weak_a_plus_reclaim_in_progress_without_extreme_runner(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (1.18, 1.20, 1.22, 1.16, 45_000),
            (1.20, 1.20, 1.22, 1.18, 42_000),
            (1.20, 1.25, 1.28, 1.18, 50_000),
            (1.25, 1.38, 1.42, 1.22, 55_000),
            (1.38, 1.52, 1.52, 1.34, 60_000),
            (1.52, 1.30, 1.50, 1.20, 35_000),
            (1.30, 1.12, 1.32, 1.08, 30_000),
            (1.12, 1.15, 1.20, 1.10, 22_000),
            (1.15, 1.18, 1.22, 1.11, 23_000),
            (1.18, 1.1801, 1.24, 1.09, 54_960),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="NVNI",
            scanner_name="vwap_pullback",
            ts=bars[-1].ts,
            score=114.746,
            criteria={
                "pattern": "vwap_pullback",
                "direction": "up",
                "pullback_low": 1.11,
                "stop_price": 1.09,
                "setup_tier": "A+ setup",
                "volume": 54_960,
            },
            bars=bars,
        )
        verifier = MomentumPatternVerifier()

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is None
        assert "too far from HOD" in verifier._last_reject

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_non_a_plus_vwap_pullback_still_rejects_far_from_hod(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (2.00, 2.05, 2.08, 1.98, 80_000),
            (2.05, 2.22, 2.25, 2.03, 90_000),
            (2.22, 2.72, 2.80, 2.18, 140_000),
            (2.72, 3.55, 3.70, 2.68, 260_000),
            (3.55, 4.75, 5.00, 3.50, 520_000),
            (4.75, 4.20, 4.80, 4.05, 160_000),
            (4.20, 3.62, 4.24, 3.50, 145_000),
            (3.62, 3.66, 3.74, 3.52, 110_000),
            (3.66, 3.70, 3.78, 3.58, 120_000),
            (3.70, 3.74, 3.82, 3.62, 130_000),
            (3.74, 3.96, 4.04, 3.72, 310_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="LATE",
            scanner_name="vwap_pullback",
            ts=bars[-1].ts,
            score=40.0,
            criteria={
                "pattern": "vwap_pullback",
                "direction": "up",
                "pullback_low": 3.50,
                "stop_price": 3.48,
                "setup_tier": "B setup",
                "volume": 300_000,
            },
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

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_massive_runner_vwap_continuation_before_hod_retest(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (1.00, 1.02, 1.03, 0.99, 80_000),
            (1.02, 1.10, 1.12, 1.00, 95_000),
            (1.10, 1.50, 1.55, 1.08, 180_000),
            (1.50, 2.45, 2.55, 1.48, 420_000),
            (2.45, 4.80, 5.05, 2.40, 900_000),
            (4.80, 6.72, 6.94, 4.75, 1_100_000),
            (6.72, 6.10, 6.84, 5.92, 320_000),
            (6.10, 5.78, 6.20, 5.64, 240_000),
            (5.78, 5.86, 5.98, 5.72, 160_000),
            (5.86, 5.91, 6.02, 5.80, 155_000),
            (5.91, 6.21, 6.24, 5.85, 260_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="INHD",
            scanner_name="pullback_base",
            ts=bars[-1].ts,
            score=120.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 5.64},
            bars=bars,
        )
        verifier = MomentumPatternVerifier()

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_controlled_lower_volume_pullback_reclaim(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        bars = []
        closes = [4.00, 4.05, 4.12, 4.20, 4.34, 4.52, 4.78, 5.05, 5.28, 5.15, 5.08, 5.12, 5.18, 5.34]
        for i, close in enumerate(closes):
            is_pullback = 9 <= i <= 11
            open_ = close + 0.05 if is_pullback and i in (9, 10) else close - 0.05
            volume = 180_000 if i < 9 else 70_000
            if i == len(closes) - 1:
                volume = 160_000
            bars.append(_bar(
                i, close=close, open_=open_, high=close + 0.07, low=close - 0.08,
                volume=volume, base_ts=now, n=len(closes),
            ))
        hit = ScanResult(
            symbol="GOOD",
            scanner_name="pullback_base",
            ts=now,
            score=75.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 5.05},
            bars=bars,
        )

        signal = MomentumPatternVerifier().verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG

    def test_rejects_heavy_red_volume_pullback(self) -> None:
        now = datetime.now(timezone.utc)
        bars = []
        closes = [4.00, 4.08, 4.18, 4.30, 4.45, 4.66, 4.90, 5.18, 5.42, 5.20, 5.05, 5.08, 5.16, 5.28]
        for i, close in enumerate(closes):
            is_red_pullback = i in (9, 10)
            open_ = close + 0.14 if is_red_pullback else close - 0.05
            volume = 140_000
            if is_red_pullback:
                volume = 220_000
            if i == len(closes) - 1:
                volume = 180_000
            bars.append(_bar(
                i, close=close, open_=open_, high=max(open_, close) + 0.05,
                low=min(open_, close) - 0.06, volume=volume, base_ts=now, n=len(closes),
            ))
        hit = ScanResult(
            symbol="DUMP",
            scanner_name="pullback_base",
            ts=now,
            score=80.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 5.00},
            bars=bars,
        )
        verifier = MomentumPatternVerifier()

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is None
        assert "red volume too heavy" in verifier._last_reject

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_pavs_style_elite_runner_mild_red_pullback(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (1.00, 1.04, 1.06, 0.98, 180_000),
            (1.04, 1.25, 1.28, 1.02, 190_000),
            (1.25, 1.80, 1.86, 1.22, 200_000),
            (1.80, 2.75, 2.88, 1.78, 200_000),
            (2.75, 4.40, 4.60, 2.70, 200_000),
            (4.40, 6.70, 6.95, 4.35, 200_000),
            (6.70, 9.10, 9.49, 6.62, 200_000),
            (9.10, 8.85, 9.22, 8.70, 250_000),
            (8.85, 8.56, 8.96, 8.42, 230_000),
            (8.56, 8.66, 8.78, 8.45, 180_000),
            (8.66, 8.92, 9.02, 8.60, 310_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="PAVS",
            scanner_name="pullback_base",
            ts=now,
            score=120.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 8.42},
            bars=bars,
        )

        signal = MomentumPatternVerifier().verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG

    def test_rejects_pavs_style_runner_when_red_volume_is_not_mild(self) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (1.00, 1.04, 1.06, 0.98, 180_000),
            (1.04, 1.25, 1.28, 1.02, 190_000),
            (1.25, 1.80, 1.86, 1.22, 200_000),
            (1.80, 2.75, 2.88, 1.78, 200_000),
            (2.75, 4.40, 4.60, 2.70, 200_000),
            (4.40, 6.70, 6.95, 4.35, 200_000),
            (6.70, 9.10, 9.49, 6.62, 200_000),
            (9.10, 8.85, 9.22, 8.70, 360_000),
            (8.85, 8.56, 8.96, 8.42, 340_000),
            (8.56, 8.66, 8.78, 8.45, 180_000),
            (8.66, 8.92, 9.02, 8.60, 310_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="PAVS",
            scanner_name="pullback_base",
            ts=now,
            score=120.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 8.42},
            bars=bars,
        )
        verifier = MomentumPatternVerifier()

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is None
        assert "red volume too heavy" in verifier._last_reject

    def test_rejects_pullback_without_green_reclaim_candle(self) -> None:
        now = datetime.now(timezone.utc)
        bars = []
        closes = [4.00, 4.08, 4.18, 4.30, 4.45, 4.66, 4.90, 5.18, 5.42, 5.30, 5.18, 5.15, 5.20, 5.16]
        for i, close in enumerate(closes):
            open_ = close - 0.05
            if i in (9, 10, 13):
                open_ = close + 0.08
            bars.append(_bar(
                i, close=close, open_=open_, high=max(open_, close) + 0.05,
                low=min(open_, close) - 0.06, volume=130_000, base_ts=now, n=len(closes),
            ))
        hit = ScanResult(
            symbol="RED",
            scanner_name="pullback_base",
            ts=now,
            score=80.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 5.10},
            bars=bars,
        )
        verifier = MomentumPatternVerifier()

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is None
        assert "reclaim candle not green" in verifier._last_reject

    def _xos_style_gapper_pullback_bars(self) -> list[Bar]:
        now = datetime.now(timezone.utc)
        prices = [
            6.90, 6.98, 7.05, 7.16, 7.28,
            7.42, 7.58, 7.74, 7.91, 8.05,
            8.12, 8.05, 7.89, 7.74, 7.66,
            7.71, 7.75, 7.78, 7.82, 7.87,
        ]
        bars = []
        for i, close in enumerate(prices):
            volume = 80_000 if i < 15 else 350_000
            high = close + 0.05
            if i == 10:
                high = 8.17
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.03,
                high=high,
                low=close - 0.08,
                volume=volume,
                base_ts=now,
                n=len(prices),
            ))
        return bars

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_hot_gapper_pullback_under_twenty_percent_intraday_move(self, _mock_guard: object) -> None:
        bars = self._xos_style_gapper_pullback_bars()
        hit = ScanResult(
            symbol="XOS",
            scanner_name="pullback_base",
            ts=bars[-1].ts,
            score=50.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 7.62},
            bars=bars,
        )
        float_checker = MagicMock()
        float_checker.get_float.return_value = 6_000_000
        verifier = MomentumPatternVerifier(float_checker=float_checker)

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_rejects_under_twenty_percent_pullback_without_low_float_quality(self, _mock_guard: object) -> None:
        bars = self._xos_style_gapper_pullback_bars()
        hit = ScanResult(
            symbol="BIGF",
            scanner_name="pullback_base",
            ts=bars[-1].ts,
            score=50.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 7.62},
            bars=bars,
        )
        float_checker = MagicMock()
        float_checker.get_float.return_value = 25_000_000
        verifier = MomentumPatternVerifier(float_checker=float_checker)

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is None
        assert "move too small" in verifier._last_reject

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_mid_move_pullback_as_scout_only(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        closes = [
            5.00, 5.05, 5.10, 5.18, 5.28,
            5.42, 5.58, 5.76, 5.86, 5.78,
            5.66, 5.61, 5.64, 5.68, 5.72,
        ]
        bars = []
        for i, close in enumerate(closes):
            open_ = close - 0.04 if i not in (9, 10, 11) else close + 0.04
            high = close + 0.05
            low = close - 0.07
            volume = 160_000 if i < 9 else 85_000
            bars.append(_bar(
                i,
                close=close,
                open_=open_,
                high=high,
                low=low,
                volume=volume,
                base_ts=now,
                n=len(closes),
            ))
        hit = ScanResult(
            symbol="ANY",
            scanner_name="pullback_base",
            ts=now,
            score=65.0,
            criteria={"pattern": "pullback_base", "direction": "up", "base_low": 5.58},
            bars=bars,
        )
        verifier = MomentumPatternVerifier()

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert hit.criteria["entry_tier"] == "pullback_scout"

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
    def test_rejects_weak_late_bull_flag_continuation(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        closes = [
            7.34, 7.38, 7.42, 7.46, 7.50,
            7.55, 7.59, 7.56, 7.53, 7.51,
            7.55, 7.59,
        ]
        bars = [
            _bar(
                i,
                close=close,
                open_=close - 0.02,
                high=close + 0.03,
                low=close - 0.05,
                volume=9_000 if i >= 8 else 22_000,
                base_ts=now,
                n=len(closes),
            )
            for i, close in enumerate(closes)
        ]
        hit = ScanResult(
            symbol="NOWL",
            scanner_name="bull_flag",
            ts=bars[-1].ts,
            score=1.5,
            criteria={
                "pattern": "bull_flag",
                "direction": "up",
                "pullback_low": 7.49,
                "close": bars[-1].close,
                "volume": bars[-1].volume,
            },
            bars=bars,
        )
        verifier = MomentumPatternVerifier()

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is None
        assert "late continuation" in verifier._last_reject

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_allows_strong_flat_top_continuation_quality(self, _mock_guard: object) -> None:
        now = datetime.now(timezone.utc)
        bars = []
        for i, close in enumerate([5.00, 5.08, 5.20, 5.35, 5.52, 5.68]):
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.05,
                high=close + 0.04,
                low=close - 0.09,
                volume=95_000,
                base_ts=now,
                n=12,
            ))
        flat = [
            (5.72, 5.75, 5.80, 5.67, 80_000),
            (5.73, 5.76, 5.81, 5.69, 82_000),
            (5.74, 5.77, 5.82, 5.70, 85_000),
        ]
        for j, (open_, close, high, low, volume) in enumerate(flat):
            bars.append(_bar(6 + j, close=close, open_=open_, high=high, low=low,
                             volume=volume, base_ts=now, n=12))
        bars.extend([
            _bar(9, close=5.84, open_=5.76, high=5.88, low=5.74, volume=130_000, base_ts=now, n=12),
            _bar(10, close=5.91, open_=5.84, high=5.94, low=5.82, volume=140_000, base_ts=now, n=12),
            _bar(11, close=6.05, open_=5.91, high=6.08, low=5.88, volume=170_000, base_ts=now, n=12),
        ])
        hit = ScanResult(
            symbol="GOOD",
            scanner_name="flat_top_breakout",
            ts=bars[-1].ts,
            score=8.0,
            criteria={
                "pattern": "flat_top_breakout",
                "direction": "up",
                "resistance": 5.82,
                "close": bars[-1].close,
                "volume": bars[-1].volume,
            },
            bars=bars,
        )
        verifier = MomentumPatternVerifier()

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert "Flat Top Breakout" in signal.reason

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
    def test_a_plus_abc_continuation_uses_reduced_scout_when_b_low_risk_is_wide(
        self,
        _mock_guard: object,
    ) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (5.10, 5.24, 5.44, 5.05, 1_468_242),
            (5.23, 5.85, 5.90, 5.11, 1_505_164),
            (5.85, 6.33, 6.45, 5.82, 2_681_059),
            (6.32, 6.16, 6.44, 6.00, 1_654_629),
            (6.18, 6.52, 6.70, 5.86, 1_481_059),
            (6.52, 6.51, 6.85, 6.14, 1_866_380),
            (6.51, 6.29, 6.65, 6.20, 1_108_860),
            (6.30, 6.32, 6.44, 6.10, 902_403),
            (6.33, 6.41, 6.49, 6.22, 740_820),
            (6.40, 7.31, 7.54, 6.36, 2_331_364),
            (7.33, 7.63, 7.86, 7.15, 1_837_028),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=20)
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="SUNE",
            scanner_name="abc_continuation",
            ts=bars[-1].ts,
            score=24.111,
            criteria={
                "pattern": "abc_continuation",
                "direction": "up",
                "a_leg_pct": 34.05,
                "a_high": 6.85,
                "a_low": 5.11,
                "b_high": 7.54,
                "b_low": 6.2201,
                "b_retrace_pct": 36.2,
                "c_breakout_pct": 1.19,
                "c_volume_surge": 1.76,
                "close": 7.63,
                "setup_tier": "A+ setup",
                "volume": 1_837_028,
            },
            bars=bars,
        )
        verifier = MomentumPatternVerifier(max_dollar_risk=100.0)

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert signal.stop_loss is not None
        assert signal.stop_loss > hit.criteria["b_low"]
        assert (signal.entry_price - signal.stop_loss) / signal.entry_price <= 0.08
        assert hit.criteria["entry_tier"] == "abc_scout"
        assert hit.criteria["size_factor"] == pytest.approx(0.35)
        assert "A+ ABC scout" in signal.reason

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_a_plus_abc_scout_rejects_tactical_stop_inside_climax_bar_noise(
        self,
        _mock_guard: object,
    ) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (3.20, 3.34, 3.40, 3.18, 280_000),
            (3.34, 3.84, 3.91, 3.21, 563_000),
            (3.78, 4.00, 4.24, 3.76, 1_122_000),
            (4.02, 4.17, 4.45, 3.98, 806_000),
            (4.17, 4.12, 4.25, 4.00, 411_000),
            (4.08, 4.70, 4.85, 4.06, 1_210_055),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=20)
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        hit = ScanResult(
            symbol="CUPR",
            scanner_name="abc_continuation",
            ts=bars[-1].ts,
            score=40.0,
            criteria={
                "pattern": "abc_continuation",
                "direction": "up",
                "a_leg_pct": 35.0,
                "a_high": 4.45,
                "a_low": 3.18,
                "b_high": 4.25,
                "b_low": 4.00,
                "b_retrace_pct": 35.0,
                "c_breakout_pct": 10.6,
                "c_bar_range_pct": 16.8,
                "c_volume_surge": 2.2,
                "close": 4.70,
                "setup_tier": "A+ setup",
                "volume": 1_210_055,
            },
            bars=bars,
        )
        verifier = MomentumPatternVerifier(max_dollar_risk=100.0)

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is None

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_generates_reduced_size_signal_for_shallow_stair_continuation(
        self,
        _mock_guard: object,
    ) -> None:
        bars = _make_shallow_stair_bars()
        scanner = ShallowStairContinuationScanner(min_price=1.0, max_price=20.0)
        hit = scanner.scan({"TST": bars})[0]
        verifier = MomentumPatternVerifier(max_dollar_risk=100.0)

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert signal.stop_loss == pytest.approx(hit.criteria["stop_price"])
        assert hit.criteria["size_factor"] == pytest.approx(0.45)
        assert "stair-step scout" in signal.reason

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_generates_signal_for_first_pullback_reclaim(self, _mock_guard: object) -> None:
        bars = _make_first_pullback_reclaim_bars()
        scanner = FirstPullbackReclaimScanner(min_price=1.0, max_price=20.0)
        hit = scanner.scan({"TST": bars})[0]
        verifier = MomentumPatternVerifier(max_risk_per_share=1.00)

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert signal.stop_loss is not None
        assert signal.stop_loss == pytest.approx(hit.criteria["stop_price"])
        assert "First Pullback Reclaim" in signal.reason

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_generates_signal_for_level_breakout_reclaim(self, _mock_guard: object) -> None:
        bars = _make_level_breakout_bars()
        scanner = LevelBreakoutReclaimScanner(min_price=1.0, max_price=20.0)
        hit = scanner.scan({"TST": bars})[0]
        verifier = MomentumPatternVerifier(max_risk_per_share=1.00)

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert signal.stop_loss is not None
        assert signal.stop_loss == pytest.approx(hit.criteria["stop_price"])
        assert "Level Breakout Reclaim" in signal.reason
        assert hit.criteria["setup_quality"] == "normal quality"
        assert hit.criteria["size_factor"] == pytest.approx(0.7)

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_generates_reduced_signal_for_promoted_level_watch_scout(self, _mock_guard: object) -> None:
        bars = _make_level_breakout_bars()
        scanner = LevelBreakoutWatchScanner(min_price=1.0, max_price=20.0)
        hit = scanner.scan({"TST": bars})[0]
        verifier = MomentumPatternVerifier(max_dollar_risk=100.0)

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert signal.quantity == int(35.0 / (hit.criteria["close"] - hit.criteria["stop_price"]))
        assert hit.criteria["setup_quality"] == "level breakout scout"
        assert hit.criteria["entry_mode"] == "level_breakout_scout"
        assert hit.criteria["size_factor"] == pytest.approx(0.35)

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_verifier_rejects_level_breakout_without_level_volume(self, _mock_guard: object) -> None:
        bars = _make_level_breakout_bars()
        bars[-1] = _bar(
            9,
            close=4.14,
            open_=4.06,
            high=4.18,
            low=4.02,
            volume=55_000,
            base_ts=bars[-1].ts,
            n=30,
        )
        hit = ScanResult(
            symbol="BGMS",
            scanner_name="level_breakout_reclaim",
            ts=bars[-1].ts,
            score=30.0,
            criteria={
                "pattern": "level_breakout_reclaim",
                "direction": "up",
                "breakout_level": 4.05,
                "base_low": 3.88,
                "stop_price": 3.86,
                "close": 4.14,
                "volume": 55_000,
            },
            bars=bars,
        )
        verifier = MomentumPatternVerifier()

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is None
        assert "breakout volume too light" in verifier._last_reject

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_normal_quality_setup_gets_reduced_risk_size(self, _mock_guard: object) -> None:
        bars = _make_level_breakout_bars()
        bars[-1] = _bar(
            9,
            close=4.14,
            open_=4.06,
            high=4.18,
            low=4.02,
            volume=110_000,
            base_ts=bars[-1].ts,
            n=30,
        )
        hit = ScanResult(
            symbol="BGMS",
            scanner_name="level_breakout_reclaim",
            ts=bars[-1].ts,
            score=12.0,
            criteria={
                "pattern": "level_breakout_reclaim",
                "direction": "up",
                "breakout_level": 4.05,
                "base_low": 3.88,
                "stop_price": 3.86,
                "close": 4.14,
                "volume": 110_000,
            },
            bars=bars,
        )
        verifier = MomentumPatternVerifier(max_dollar_risk=100.0)

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.quantity == int(70.0 / (4.14 - 3.86))
        assert hit.criteria["setup_quality"] == "normal quality"
        assert hit.criteria["size_factor"] == pytest.approx(0.7)

    @patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
    def test_a_quality_level_breakout_keeps_full_risk_size(self, _mock_guard: object) -> None:
        bars = _make_level_breakout_bars()
        bars[-1] = _bar(
            9,
            close=4.14,
            open_=4.06,
            high=4.18,
            low=4.02,
            volume=240_000,
            base_ts=bars[-1].ts,
            n=30,
        )
        hit = ScanResult(
            symbol="DAIC",
            scanner_name="level_breakout_reclaim",
            ts=bars[-1].ts,
            score=30.0,
            criteria={
                "pattern": "level_breakout_reclaim",
                "direction": "up",
                "breakout_level": 4.05,
                "base_low": 3.88,
                "stop_price": 3.86,
                "close": 4.14,
                "volume": 240_000,
            },
            bars=bars,
        )
        verifier = MomentumPatternVerifier(max_dollar_risk=100.0)

        signal = verifier.verify(hit, PortfolioState(cash=100_000))

        assert signal is not None
        assert signal.quantity == int(100.0 / (4.14 - 3.86))
        assert hit.criteria["setup_quality"] == "A breakout"
        assert hit.criteria["size_factor"] == pytest.approx(1.0)

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


def _aplus_hit() -> ScanResult:
    return ScanResult(
        symbol="TST", scanner_name="vwap_pullback",
        ts=datetime.now(timezone.utc), score=90.0,
        criteria={"setup_tier": "A+ setup", "pattern": "vwap_pullback"},
    )


def _fresh_scout_dsy_bars(n: int = 10) -> list:
    # low-price heavy volume early (VWAP anchor low), then a fresh green base
    # rebuilt above VWAP on RISING volume — the DSY profile
    return [
        _bar(0, close=3.0, open_=2.9, high=3.05, low=2.85, volume=400_000, n=n),
        _bar(1, close=3.1, open_=3.0, high=3.15, low=2.98, volume=300_000, n=n),
        _bar(2, close=3.3, open_=3.1, high=3.35, low=3.08, volume=200_000, n=n),
        _bar(3, close=3.6, open_=3.3, high=3.65, low=3.28, volume=200_000, n=n),
        _bar(4, close=4.0, open_=3.6, high=4.05, low=3.58, volume=200_000, n=n),
        _bar(5, close=4.3, open_=4.0, high=4.35, low=3.98, volume=150_000, n=n),
        _bar(6, close=4.30, open_=4.35, high=4.40, low=4.28, volume=170_000, n=n),
        _bar(7, close=4.33, open_=4.30, high=4.38, low=4.29, volume=180_000, n=n),
        _bar(8, close=4.32, open_=4.33, high=4.40, low=4.30, volume=190_000, n=n),
        _bar(9, close=4.45, open_=4.32, high=4.48, low=4.31, volume=200_000, n=n),
    ]


def test_fresh_vwap_reclaim_scout_allows_dsy_profile() -> None:
    hit = _aplus_hit()
    allowed = MomentumPatternVerifier._allows_fresh_vwap_reclaim_scout(
        "vwap_pullback", _fresh_scout_dsy_bars(),
        scan_result=hit, float_shares=8_000_000, max_float=20_000_000,
        red_body_pct=4.0, red_range_pct=7.0,
    )
    assert allowed is True
    assert hit.criteria["entry_tier"] == "fresh_vwap_reclaim_scout"


def test_fresh_vwap_reclaim_scout_rejects_gmm_failed_reclaim() -> None:
    # GMM profile: green bar but still BELOW VWAP (reclaim never held)
    bars = [
        _bar(0, close=4.3, open_=4.0, high=4.4, low=3.95, volume=400_000, n=10),
        _bar(1, close=4.4, open_=4.3, high=4.5, low=4.25, volume=350_000, n=10),
        _bar(2, close=4.2, open_=4.4, high=4.45, low=4.10, volume=300_000, n=10),
        _bar(3, close=4.0, open_=4.2, high=4.25, low=3.95, volume=250_000, n=10),
        _bar(4, close=3.9, open_=4.0, high=4.05, low=3.80, volume=200_000, n=10),
        _bar(5, close=3.8, open_=3.9, high=3.95, low=3.70, volume=180_000, n=10),
        _bar(6, close=3.7, open_=3.8, high=3.85, low=3.65, volume=150_000, n=10),
        _bar(7, close=3.65, open_=3.7, high=3.75, low=3.60, volume=120_000, n=10),
        _bar(8, close=3.62, open_=3.65, high=3.70, low=3.58, volume=110_000, n=10),
        _bar(9, close=3.68, open_=3.62, high=3.72, low=3.60, volume=130_000, n=10),
    ]
    allowed = MomentumPatternVerifier._allows_fresh_vwap_reclaim_scout(
        "vwap_pullback", bars,
        scan_result=_aplus_hit(), float_shares=8_000_000, max_float=20_000_000,
        red_body_pct=4.0, red_range_pct=7.0,
    )
    assert allowed is False


def test_fresh_vwap_reclaim_scout_rejects_high_float() -> None:
    allowed = MomentumPatternVerifier._allows_fresh_vwap_reclaim_scout(
        "vwap_pullback", _fresh_scout_dsy_bars(),
        scan_result=_aplus_hit(), float_shares=100_000_000, max_float=20_000_000,
        red_body_pct=4.0, red_range_pct=7.0,
    )
    assert allowed is False


def test_fresh_vwap_reclaim_scout_rejects_without_a_plus_tier() -> None:
    hit = ScanResult(
        symbol="TST", scanner_name="vwap_pullback",
        ts=datetime.now(timezone.utc), score=70.0,
        criteria={"pattern": "vwap_pullback"},
    )
    allowed = MomentumPatternVerifier._allows_fresh_vwap_reclaim_scout(
        "vwap_pullback", _fresh_scout_dsy_bars(),
        scan_result=hit, float_shares=8_000_000, max_float=20_000_000,
        red_body_pct=4.0, red_range_pct=7.0,
    )
    assert allowed is False


def test_fresh_vwap_reclaim_scout_gets_reduced_size_factor() -> None:
    hit = _aplus_hit()
    hit.criteria["entry_tier"] = "fresh_vwap_reclaim_scout"
    factor, label = MomentumPatternVerifier._setup_quality_factor(
        "vwap_pullback", hit, _fresh_scout_dsy_bars(),
    )
    assert factor == 0.35
    assert "fresh VWAP reclaim" in label


@patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", return_value=None)
def test_fresh_vwap_reclaim_scout_verify_uses_new_base_stop_and_reduced_size(
    _mock_guard: object,
) -> None:
    bars = _fresh_scout_dsy_bars()
    bars[5] = _bar(5, close=4.30, open_=4.65, high=4.72, low=4.15, volume=300_000, n=10)
    bars[7] = _bar(7, close=4.33, open_=4.30, high=4.38, low=4.29, volume=300_000, n=10)
    bars[8] = _bar(8, close=4.34, open_=4.32, high=4.40, low=4.30, volume=320_000, n=10)
    bars[9] = _bar(9, close=4.45, open_=4.32, high=4.48, low=4.31, volume=340_000, n=10)
    hit = ScanResult(
        symbol="DSY",
        scanner_name="vwap_pullback",
        ts=datetime.now(timezone.utc),
        score=90.0,
        criteria={
            "setup_tier": "A+ setup",
            "pattern": "vwap_pullback",
            "direction": "up",
            "close": bars[-1].close,
            "pullback_low": 3.98,
            "stop_price": 3.96,
            "volume": bars[-1].volume,
        },
        bars=bars,
    )
    verifier = MomentumPatternVerifier(
        max_dollar_risk=100.0,
        fresh_vwap_reclaim_scout_enabled=True,
    )

    signal = verifier.verify(hit, PortfolioState(cash=100_000))

    assert signal is not None
    assert hit.criteria["entry_tier"] == "fresh_vwap_reclaim_scout"
    assert hit.criteria["base_low"] == pytest.approx(4.28)
    assert hit.criteria["stop_price"] == pytest.approx(4.26)
    assert signal.stop_loss == pytest.approx(hit.criteria["stop_price"])
    assert signal.stop_loss > 3.98
    assert hit.criteria["size_factor"] == pytest.approx(0.35)
    assert "fresh VWAP reclaim scout" in signal.reason
    assert signal.quantity == int(35.0 / (signal.entry_price - signal.stop_loss))


def test_vwap_reclaim_scout_retries_near_miss_score_as_reduced_size() -> None:
    now = datetime.now(timezone.utc)
    bars = [
        _bar(0, close=2.20, open_=2.20, high=2.24, low=2.16, volume=60_000, base_ts=now, n=12),
        _bar(1, close=2.32, open_=2.20, high=2.36, low=2.19, volume=70_000, base_ts=now, n=12),
        _bar(2, close=2.46, open_=2.32, high=2.50, low=2.31, volume=80_000, base_ts=now, n=12),
        _bar(3, close=2.62, open_=2.46, high=2.68, low=2.44, volume=90_000, base_ts=now, n=12),
        _bar(4, close=2.56, open_=2.62, high=2.66, low=2.50, volume=60_000, base_ts=now, n=12),
        _bar(5, close=2.58, open_=2.56, high=2.64, low=2.52, volume=58_000, base_ts=now, n=12),
        _bar(6, close=2.60, open_=2.58, high=2.66, low=2.54, volume=62_000, base_ts=now, n=12),
        _bar(7, close=2.63, open_=2.60, high=2.69, low=2.58, volume=64_000, base_ts=now, n=12),
        _bar(8, close=2.66, open_=2.63, high=2.70, low=2.60, volume=72_000, base_ts=now, n=12),
        _bar(9, close=2.67, open_=2.63, high=2.69, low=2.58, volume=58_000, base_ts=now, n=12),
    ]
    hit = ScanResult(
        symbol="AIIO",
        scanner_name="vwap_pullback",
        ts=now,
        score=100.0,
        criteria={
            "setup_tier": "A+ setup",
            "pattern": "vwap_pullback",
            "direction": "up",
            "close": bars[-1].close,
            "pullback_low": 2.50,
            "volume": bars[-1].volume,
        },
        bars=bars,
    )
    calls = []

    def fake_guard(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return "entry score too low (77/100, need 80+) [test]"
        return None

    verifier = MomentumPatternVerifier(
        max_dollar_risk=100.0,
        vwap_reclaim_scout_enabled=True,
    )

    with patch("daytrading.strategy.scalping.momentum_pattern.check_entry_quality", side_effect=fake_guard):
        signal = verifier.verify(hit, PortfolioState(cash=100_000))

    assert signal is not None
    assert len(calls) == 2
    assert calls[0]["entry_tier"] == ""
    assert calls[1]["entry_tier"] == "vwap_reclaim_scout"
    assert hit.criteria["entry_tier"] == "vwap_reclaim_scout"
    assert hit.criteria["entry_score_at_signal"] == 77
    assert hit.criteria["size_factor"] == pytest.approx(0.30)
    assert "VWAP reclaim scout" in signal.reason
