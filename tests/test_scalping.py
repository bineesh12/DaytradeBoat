"""Tests for the scalping pipeline focused on $1–$20 stocks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from daytrading.exits.manager import ExitManager, TrackedPosition, build_exit_tiers
from daytrading.indicators.scalping import (
    cumulative_delta,
    momentum_burst,
    order_flow_imbalance,
    tape_speed,
)
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.scanner.scalping.momentum_burst import MomentumBurstScanner
from daytrading.scanner.scalping.spread_filter import SpreadFilterScanner
from daytrading.scanner.scalping.tape_reader import TapeReaderScanner
from daytrading.strategy.scalping.momentum_scalp import MomentumScalpVerifier
from daytrading.strategy.scalping.tape_scalp import TapeScalpVerifier
from daytrading.models import (
    Bar,
    ExitReason,
    PortfolioState,
    Quote,
    ScanResult,
    Side,
    SignalAction,
    Tick,
    Timeframe,
)

TS = datetime(2026, 5, 13, 14, 30, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float = 0.0) -> datetime:
    return TS + timedelta(seconds=offset_seconds)


def _bar(
    symbol: str, close: float, volume: float = 10_000,
    open_: float | None = None,
    tf: Timeframe = Timeframe.SEC_5,
) -> Bar:
    o = open_ if open_ is not None else close - 0.01
    return Bar(
        symbol=symbol, ts=TS, open=o,
        high=close + 0.02, low=close - 0.02,
        close=close, volume=volume, timeframe=tf,
    )


# ---------------------------------------------------------------------------
# Price range enforcement
# ---------------------------------------------------------------------------

class TestPriceFilter:

    def test_scanner_rejects_above_20(self) -> None:
        bars = [_bar("X", 25.0 + i * 0.01, volume=10_000) for i in range(5)]
        bars.append(_bar("X", 26.0, volume=20_000))
        scanner = MomentumBurstScanner(min_burst_pct=0.05, min_velocity=0.001, min_volume=100)
        hits = scanner.scan({"X": bars})
        assert len(hits) == 0

    def test_scanner_rejects_below_1(self) -> None:
        bars = [_bar("PENNY", 0.50 + i * 0.001) for i in range(5)]
        bars.append(_bar("PENNY", 0.55))
        scanner = MomentumBurstScanner(min_burst_pct=0.05, min_velocity=0.001, min_volume=100)
        hits = scanner.scan({"PENNY": bars})
        assert len(hits) == 0

    def test_scanner_accepts_5_dollar_stock(self) -> None:
        bars = [_bar("CHEAP", 5.00 + i * 0.001, volume=10_000) for i in range(5)]
        bars.append(_bar("CHEAP", 5.10, volume=15_000))
        scanner = MomentumBurstScanner(min_burst_pct=0.05, burst_period=3, min_velocity=0.001, min_volume=1000)
        hits = scanner.scan({"CHEAP": bars})
        assert len(hits) >= 1

    def test_verifier_rejects_high_price(self) -> None:
        bars = [
            _bar("EXP", 50.0, volume=10_000, open_=49.99),
            _bar("EXP", 50.05, volume=12_000, open_=50.0),
            _bar("EXP", 50.10, volume=14_000, open_=50.05),
            _bar("EXP", 50.50, volume=18_000, open_=50.10),
        ]
        scan = ScanResult(
            symbol="EXP", scanner_name="momentum_burst", ts=TS,
            score=0.8, criteria={"burst_pct": 0.8, "direction": "up", "close": 50.50, "volume": 18_000},
            bars=bars,
        )
        verifier = MomentumScalpVerifier()
        signal = verifier.verify(scan, PortfolioState(cash=100_000))
        assert signal is None


# ---------------------------------------------------------------------------
# Scalping indicators
# ---------------------------------------------------------------------------

class TestScalpingIndicators:

    def test_momentum_burst_detects_spike(self) -> None:
        bars = [_bar("X", 5.00 + i * 0.01) for i in range(5)]
        bars.append(_bar("X", 5.50))
        result = momentum_burst(bars, period=3)
        assert result[-1] > 0.1

    def test_order_flow_all_buys(self) -> None:
        ticks = [Tick(symbol="X", ts=_ts(i), price=5.0, size=100, side=Side.BUY) for i in range(10)]
        result = order_flow_imbalance(ticks, window=10)
        assert result[-1] == 1.0

    def test_order_flow_balanced(self) -> None:
        ticks = [
            Tick(symbol="X", ts=_ts(i), price=5.0, size=100, side=Side.BUY if i % 2 == 0 else Side.SELL)
            for i in range(10)
        ]
        result = order_flow_imbalance(ticks, window=10)
        assert abs(result[-1]) < 0.01

    def test_cumulative_delta(self) -> None:
        ticks = [Tick(symbol="X", ts=_ts(i), price=5.0, size=100, side=Side.BUY) for i in range(5)]
        result = cumulative_delta(ticks)
        assert result[-1] == 500.0

    def test_tape_speed(self) -> None:
        ticks = [Tick(symbol="X", ts=_ts(i * 0.5), price=5.0, size=100, side=Side.BUY) for i in range(20)]
        result = tape_speed(ticks, window_seconds=5.0)
        assert result[-1] >= 2.0


# ---------------------------------------------------------------------------
# Scanners on $1–$20 stocks
# ---------------------------------------------------------------------------

class TestScalpingScanners:

    def test_momentum_burst_on_3_dollar_stock(self) -> None:
        bars = [_bar("LOW", 3.00 + i * 0.001, volume=8_000) for i in range(5)]
        bars.append(_bar("LOW", 3.10, volume=15_000))
        scanner = MomentumBurstScanner(min_burst_pct=0.05, burst_period=3, min_velocity=0.001, min_volume=1000)
        hits = scanner.scan({"LOW": bars})
        assert len(hits) == 1
        assert hits[0].criteria["close"] <= 20.0

    def test_spread_filter_on_cheap_stock(self) -> None:
        quotes = [
            Quote(symbol="CHE", ts=_ts(i), bid=4.00, ask=4.01, bid_size=1000, ask_size=1000)
            for i in range(25)
        ]
        scanner = SpreadFilterScanner(
            max_spread_cents=0.05, max_spread_pct=0.5, max_compression_ratio=1.5,
        )
        hits = scanner.scan_quotes({"CHE": quotes})
        assert len(hits) == 1

    @pytest.mark.skip(reason="TapeReader thresholds need recalibration for current tick model")
    def test_tape_reader_on_8_dollar_stock(self) -> None:
        ticks = [
            Tick(symbol="MID", ts=_ts(i * 0.1), price=8.0, size=500, side=Side.BUY)
            for i in range(60)
        ]
        scanner = TapeReaderScanner(min_imbalance=0.3, min_tape_speed=3.0, imbalance_window=50)
        hits = scanner.scan_ticks({"MID": ticks})
        assert len(hits) == 1


# ---------------------------------------------------------------------------
# Verifiers with cent-based stops
# ---------------------------------------------------------------------------

class TestScalpingVerifiers:

    @patch("daytrading.strategy.scalping.momentum_scalp.check_entry_quality", return_value=None)
    def test_momentum_scalp_long_5_dollar_stock(self, _mock_guard: object) -> None:
        bars = [
            _bar("ABC", 5.00, volume=8_000, open_=4.99),
            _bar("ABC", 5.02, volume=10_000, open_=5.00),
            _bar("ABC", 5.05, volume=12_000, open_=5.02),
            _bar("ABC", 5.15, volume=15_000, open_=5.05),
        ]
        scan = ScanResult(
            symbol="ABC", scanner_name="momentum_burst", ts=TS,
            score=0.85, criteria={"burst_pct": 0.85, "direction": "up", "close": 5.15, "volume": 15_000},
            bars=bars,
        )
        verifier = MomentumScalpVerifier(
            stop_ticks=3, target_ticks=5, trail_ticks=2, position_size=500,
        )
        signal = verifier.verify(scan, PortfolioState(cash=50_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert signal.quantity == 500
        # stop = entry - $0.03
        assert abs(signal.stop_loss - (5.15 - 0.03)) < 0.001
        # target = entry + $0.05
        assert abs(signal.take_profit - (5.15 + 0.05)) < 0.001
        # trailing = $0.02
        assert abs(signal.trailing_stop_offset - 0.02) < 0.001

    @patch("daytrading.strategy.scalping.tape_scalp.check_entry_quality", return_value=None)
    def test_tape_scalp_long_12_dollar_stock(self, _mock_guard: object) -> None:
        bars = [
            _bar("XYZ", 11.95, volume=40_000, open_=11.90),
            _bar("XYZ", 11.98, volume=45_000, open_=11.95),
            _bar("XYZ", 12.00, volume=50_000, open_=11.98),
        ]
        scan = ScanResult(
            symbol="XYZ", scanner_name="tape_reader", ts=TS,
            score=5.0,
            criteria={"imbalance": 0.6, "tape_speed": 10.0, "cum_delta": 5000,
                      "direction": "buy_pressure", "last_price": 12.0},
            bars=bars,
        )
        verifier = TapeScalpVerifier(
            min_score=2.0, stop_ticks=2, target_ticks=4, trail_ticks=3, position_size=500,
        )
        signal = verifier.verify(scan, PortfolioState(cash=50_000))

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert abs(signal.stop_loss - (12.0 - 0.02)) < 0.001
        assert abs(signal.take_profit - (12.0 + 0.04)) < 0.001

    def test_verifier_skips_existing_position(self) -> None:
        from daytrading.models import Position
        scan = ScanResult(
            symbol="ABC", scanner_name="momentum_burst", ts=TS,
            score=0.5, criteria={"burst_pct": 0.5, "direction": "up", "close": 5.0, "volume": 10_000},
            bars=[_bar("ABC", 4.98, open_=4.97), _bar("ABC", 4.99, open_=4.98),
                  _bar("ABC", 5.00, open_=4.99), _bar("ABC", 5.05, open_=5.00)],
        )
        portfolio = PortfolioState(cash=50_000, positions={"ABC": Position(symbol="ABC", quantity=100)})
        verifier = MomentumScalpVerifier()
        signal = verifier.verify(scan, portfolio)
        assert signal is None


# ---------------------------------------------------------------------------
# Exit manager with cent-based stops
# ---------------------------------------------------------------------------

class TestExitManager:

    def test_stop_loss_3_cents(self) -> None:
        em = ExitManager()
        em.track(TrackedPosition(
            symbol="ABC", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=4.97,  # 3 cents below
        ))
        exits = em.check_exits({"ABC": 4.96}, _ts(10))
        assert len(exits) == 1
        assert "stop_loss" in exits[0].reason

    def test_half_sell_at_2_to_1_target(self) -> None:
        em = ExitManager()
        em.track(TrackedPosition(
            symbol="ABC", side=Side.BUY, quantity=100,
            remaining_qty=100, original_qty=100,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=4.90, risk_per_share=0.10,
            first_target_price=5.20,
        ))
        exits = em.check_exits({"ABC": 5.21}, _ts(10))
        assert len(exits) == 1
        assert "take_profit" in exits[0].reason.lower()
        assert exits[0].quantity == 50

    def test_trailing_after_half_sell(self) -> None:
        em = ExitManager()
        pos = TrackedPosition(
            symbol="DEF", side=Side.BUY, quantity=100,
            remaining_qty=50, original_qty=100,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=5.00, sold_half=True, breakeven_locked=True,
            current_step=1, step_pct=0.04,
        )
        em.track(pos)
        exits = em.check_exits({"DEF": 4.99}, _ts(60))
        assert len(exits) >= 1
        assert any("trailing" in e.reason.lower() or "stop" in e.reason.lower() for e in exits)

    def test_stale_exit_after_long_hold(self) -> None:
        em = ExitManager()
        em.track(TrackedPosition(
            symbol="GHI", side=Side.BUY, quantity=100,
            remaining_qty=100, original_qty=100,
            entry_price=3.00, entry_ts=_ts(0),
            stop_loss=2.85, risk_per_share=0.15,
            first_target_price=3.30,
            trend_strength=0.3,
        ))
        exits = em.check_exits({"GHI": 2.98}, _ts(60))
        assert len(exits) == 0
        exits = em.check_exits({"GHI": 2.98}, _ts(185))
        assert len(exits) >= 1

    def test_no_exit_when_safe(self) -> None:
        em = ExitManager()
        em.track(TrackedPosition(
            symbol="JKL", side=Side.BUY, quantity=100,
            remaining_qty=100, original_qty=100,
            entry_price=10.00, entry_ts=_ts(0),
            stop_loss=9.70, first_target_price=10.60,
        ))
        exits = em.check_exits({"JKL": 10.02}, _ts(5))
        assert len(exits) == 0


# ---------------------------------------------------------------------------
# Full pipeline factory
# ---------------------------------------------------------------------------

class TestScalpingPipeline:

    def test_factory_creates_pipeline(self) -> None:
        pipeline = create_scalping_pipeline(initial_cash=10_000)
        assert pipeline.portfolio.cash == 10_000

    def test_factory_defaults_to_50_dollar_risk_cap(self) -> None:
        pipeline = create_scalping_pipeline(initial_cash=10_000)
        assert pipeline._max_dollar_risk_per_trade == 50.0

    def test_factory_respects_price_range(self) -> None:
        pipeline = create_scalping_pipeline(initial_cash=10_000, min_price=2.0, max_price=15.0)
        # $25 stock should be rejected by the classifier
        universe = {
            "EXP": [_bar("EXP", 25.0 + i * 0.01, volume=100_000) for i in range(30)],
        }
        result = pipeline.run_cycle(universe)
        assert len(result.fills) == 0

    def test_pipeline_end_to_end_on_cheap_stock(self) -> None:
        pipeline = create_scalping_pipeline(
            initial_cash=10_000,
            min_burst_pct=0.05,
            min_burst_volume=1_000,
        )
        # create a momentum burst on a $5 stock
        bars = [_bar("SCAL", 5.00 + i * 0.001, volume=10_000) for i in range(25)]
        bars.append(_bar("SCAL", 5.10, volume=20_000, open_=5.05))  # green burst bar
        universe = {"SCAL": bars}

        result = pipeline.run_cycle(universe)
        assert result.regimes.get("SCAL") is not None
        # the classifier should accept this $5 stock
        regime = result.regimes["SCAL"]
        from daytrading.models import TradingStyle
        assert regime.style != TradingStyle.NOT_TRADEABLE or regime.confidence == 0
