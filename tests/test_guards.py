"""Tests for risk guards — false breakouts, liquidity traps, halts, market panic, slippage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from daytrading.models import Bar, Quote, SignalAction, Timeframe, TradeSignal
from daytrading.risk.guards import (
    FalseBreakoutDetector,
    HaltTracker,
    LiquidityTrapDetector,
    MarketPanicDetector,
    SlippageGuard,
    TradeGuard,
)

TS = datetime(2026, 5, 15, 14, 30, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float = 0.0) -> datetime:
    return TS + timedelta(seconds=offset_seconds)


def _bar(
    i: int,
    *,
    close: float,
    open_: float,
    high: float,
    low: float,
    volume: float,
    symbol: str = "TST",
) -> Bar:
    return Bar(
        symbol=symbol,
        ts=_ts(i * 60),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        timeframe=Timeframe.MIN_1,
    )


def _signal(symbol: str = "TST", price: float = 5.0) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        stop_loss=price - 0.10,
        take_profit=price + 0.20,
        reason="test signal",
    )


# ---------------------------------------------------------------------------
# 1. False Breakout Detector
# ---------------------------------------------------------------------------

class TestFalseBreakoutDetector:

    def test_pass_with_strong_volume(self) -> None:
        """Real breakout: volume increases on the breakout bar."""
        bars = [
            _bar(i, close=5.0 + i * 0.01, open_=5.0 + i * 0.01 - 0.005,
                 high=5.0 + i * 0.01 + 0.01, low=5.0 + i * 0.01 - 0.01,
                 volume=10_000)
            for i in range(5)
        ]
        # Breakout bar with 2x volume, close near high
        bars.append(_bar(5, close=5.12, open_=5.05, high=5.13, low=5.04, volume=25_000))
        detector = FalseBreakoutDetector()
        assert detector.check(bars) is None

    def test_reject_declining_volume(self) -> None:
        """False breakout: volume drops on the breakout bar."""
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=20_000)
            for i in range(5)
        ]
        # "Breakout" bar but with declining volume (< 0.8x avg)
        bars.append(_bar(5, close=5.05, open_=5.01, high=5.06, low=5.00, volume=10_000))
        detector = FalseBreakoutDetector()
        result = detector.check(bars)
        assert result is not None
        assert "volume declining" in result

    def test_reject_rejection_wick(self) -> None:
        """False breakout: big upper wick = sellers rejecting the move."""
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=10_000)
            for i in range(5)
        ]
        # Bar with huge upper wick and only modest volume
        bars.append(_bar(5, close=5.02, open_=5.00, high=5.10, low=4.99, volume=11_000))
        detector = FalseBreakoutDetector()
        result = detector.check(bars)
        assert result is not None
        assert "rejection wick" in result

    def test_reject_close_near_low(self) -> None:
        """Buyers couldn't hold: close in lower 30% of range despite green candle."""
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=10_000)
            for i in range(5)
        ]
        # Close barely above open but near the low of the range
        bars.append(_bar(5, close=5.005, open_=5.00, high=5.10, low=5.00, volume=15_000))
        detector = FalseBreakoutDetector()
        result = detector.check(bars)
        assert result is not None
        assert "lower 30%" in result or "rejection wick" in result

    def test_not_enough_bars(self) -> None:
        """Should return None if fewer than 6 bars."""
        bars = [_bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=10_000) for i in range(4)]
        detector = FalseBreakoutDetector()
        assert detector.check(bars) is None


# ---------------------------------------------------------------------------
# 2. Liquidity Trap Detector
# ---------------------------------------------------------------------------

class TestLiquidityTrapDetector:

    def test_pass_tight_spread_good_volume(self) -> None:
        """No trap: tight spread, healthy volume."""
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=10_000)
            for i in range(6)
        ]
        quotes = [Quote(symbol="TST", ts=_ts(0), bid=4.99, ask=5.01, bid_size=1000, ask_size=1000)]
        detector = LiquidityTrapDetector()
        assert detector.check(bars, quotes=quotes) is None

    def test_reject_wide_spread_weak_volume(self) -> None:
        """Trap: wide spread + weak volume = thin book spike."""
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=10_000)
            for i in range(5)
        ]
        # Latest bar with less volume than average
        bars.append(_bar(5, close=5.05, open_=5.00, high=5.06, low=4.99, volume=8_000))
        quotes = [Quote(symbol="TST", ts=_ts(0), bid=4.95, ask=5.10, bid_size=100, ask_size=100)]
        detector = LiquidityTrapDetector(max_spread_pct=0.5)
        result = detector.check(bars, quotes=quotes)
        assert result is not None
        assert "liquidity trap" in result

    def test_reject_spike_and_fade(self) -> None:
        """Trap: bar spiked high but closed near open (huge wick)."""
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=10_000)
            for i in range(5)
        ]
        # Upper wick = 5.20 - 5.02 = 0.18; body = 5.02 - 5.00 = 0.02
        bars.append(_bar(5, close=5.02, open_=5.00, high=5.20, low=4.99, volume=12_000))
        detector = LiquidityTrapDetector()
        result = detector.check(bars)
        assert result is not None
        assert "spike-and-fade" in result

    def test_reject_gap_up_reversal(self) -> None:
        """Trap: opens above prior high but closes red."""
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=10_000)
            for i in range(5)
        ]
        # Opens at 5.05 (above prior high 5.01) but closes red at 5.00
        bars.append(_bar(5, close=5.00, open_=5.05, high=5.06, low=4.99, volume=12_000))
        detector = LiquidityTrapDetector()
        result = detector.check(bars)
        assert result is not None
        assert "gap-up reversal" in result

    def test_no_quotes_ok(self) -> None:
        """No quotes available: only candle checks run."""
        bars = [
            _bar(i, close=5.0 + i * 0.01, open_=5.0 + i * 0.01 - 0.005,
                 high=5.0 + i * 0.01 + 0.01, low=5.0 + i * 0.01 - 0.01,
                 volume=10_000)
            for i in range(6)
        ]
        detector = LiquidityTrapDetector()
        assert detector.check(bars, quotes=None) is None


# ---------------------------------------------------------------------------
# 3. Halt Tracker
# ---------------------------------------------------------------------------

class TestHaltTracker:

    def test_manual_halt_blocks_entry(self) -> None:
        tracker = HaltTracker(resume_cooldown_secs=60)
        tracker.mark_halted("TST", _ts(0))
        result = tracker.check("TST", _ts(10))
        assert result is not None
        assert "HALTED" in result

    def test_resumed_with_cooldown(self) -> None:
        tracker = HaltTracker(resume_cooldown_secs=120)
        tracker.mark_halted("TST", _ts(0))
        tracker.mark_resumed("TST", _ts(30))
        # 30s after resume: still in cooldown
        result = tracker.check("TST", _ts(60))
        assert result is not None
        assert "cooldown" in result

    def test_cooldown_expires(self) -> None:
        tracker = HaltTracker(resume_cooldown_secs=60)
        tracker.mark_halted("TST", _ts(0))
        tracker.mark_resumed("TST", _ts(30))
        # 120s after resume: cooldown expired
        result = tracker.check("TST", _ts(200))
        assert result is None

    def test_price_freeze_detection(self) -> None:
        tracker = HaltTracker()
        tracker.update_price("TST", 5.00, _ts(0))
        # 40 seconds later, price hasn't moved
        tracker.update_price("TST", 5.00, _ts(40))
        result = tracker.check("TST", _ts(45))
        assert result is not None
        assert "HALTED" in result

    def test_post_halt_gap_detection(self) -> None:
        tracker = HaltTracker(resume_cooldown_secs=60)
        tracker.update_price("TST", 5.00, _ts(0))
        # Price freezes
        tracker.update_price("TST", 5.00, _ts(40))
        # Now a big gap = halt resumed
        tracker.update_price("TST", 5.50, _ts(80))
        result = tracker.check("TST", _ts(85))
        assert result is not None
        assert "cooldown" in result

    def test_no_halt_normal_trading(self) -> None:
        tracker = HaltTracker()
        tracker.update_price("TST", 5.00, _ts(0))
        tracker.update_price("TST", 5.01, _ts(5))
        tracker.update_price("TST", 5.02, _ts(10))
        assert tracker.check("TST", _ts(15)) is None

    def test_halted_symbols_property(self) -> None:
        tracker = HaltTracker()
        tracker.mark_halted("AAA", _ts(0))
        tracker.mark_halted("BBB", _ts(0))
        assert set(tracker.halted_symbols) == {"AAA", "BBB"}


# ---------------------------------------------------------------------------
# 4. Market Panic Detector
# ---------------------------------------------------------------------------

class TestMarketPanicDetector:

    def _spy_bar(self, i: int, open_: float, close: float) -> Bar:
        return Bar(
            symbol="SPY",
            ts=_ts(i * 60),
            open=open_,
            high=max(open_, close) + 0.1,
            low=min(open_, close) - 0.1,
            close=close,
            volume=1_000_000,
            timeframe=Timeframe.MIN_1,
        )

    def test_no_panic_normal_market(self) -> None:
        detector = MarketPanicDetector(panic_drop_pct=0.5, lookback_bars=5)
        for i in range(10):
            detector.update_spy_bar(self._spy_bar(i, 450.0 + i * 0.05, 450.0 + i * 0.05 + 0.02))
        assert detector.check() is None
        assert not detector.is_panic

    def test_panic_on_sharp_drop(self) -> None:
        detector = MarketPanicDetector(panic_drop_pct=0.5, lookback_bars=5)
        # 5 bars of SPY dropping 0.6% (450 -> 447.3)
        prices = [450.0, 449.5, 449.0, 448.5, 448.0, 447.3]
        for i in range(len(prices) - 1):
            detector.update_spy_bar(self._spy_bar(i, prices[i], prices[i + 1]))
        result = detector.check()
        assert result is not None
        assert "panic" in result.lower()
        assert detector.is_panic

    def test_recovery_after_green_bars(self) -> None:
        detector = MarketPanicDetector(panic_drop_pct=0.5, lookback_bars=5, recovery_bars=3)
        # Trigger panic
        prices = [450.0, 449.5, 449.0, 448.5, 448.0, 447.3]
        for i in range(len(prices) - 1):
            detector.update_spy_bar(self._spy_bar(i, prices[i], prices[i + 1]))
        assert detector.is_panic

        # 3 green recovery bars
        recovery = [(447.3, 447.8), (447.8, 448.3), (448.3, 448.8)]
        for j, (o, c) in enumerate(recovery):
            detector.update_spy_bar(self._spy_bar(10 + j, o, c))
        assert not detector.is_panic
        assert detector.check() is None


# ---------------------------------------------------------------------------
# 5. Slippage Guard
# ---------------------------------------------------------------------------

class TestSlippageGuard:

    def test_smart_limit_buy(self) -> None:
        guard = SlippageGuard()
        guard.update_quote(Quote(symbol="TST", ts=_ts(0), bid=5.00, ask=5.02, bid_size=1000, ask_size=1000))
        price = guard.get_limit_price("TST", "buy")
        assert price == 5.03  # ask + 1 cent

    def test_smart_limit_sell(self) -> None:
        guard = SlippageGuard()
        guard.update_quote(Quote(symbol="TST", ts=_ts(0), bid=5.00, ask=5.02, bid_size=1000, ask_size=1000))
        price = guard.get_limit_price("TST", "sell")
        assert price == 4.99  # bid - 1 cent

    def test_no_quote_returns_none(self) -> None:
        guard = SlippageGuard()
        assert guard.get_limit_price("UNKNOWN", "buy") is None

    def test_spread_too_wide(self) -> None:
        guard = SlippageGuard(max_spread_pct=1.0)
        guard.update_quote(Quote(symbol="TST", ts=_ts(0), bid=4.90, ask=5.10, bid_size=100, ask_size=100))
        result = guard.check_spread("TST")
        assert result is not None
        assert "spread too wide" in result

    def test_spread_ok(self) -> None:
        guard = SlippageGuard(max_spread_pct=1.0)
        guard.update_quote(Quote(symbol="TST", ts=_ts(0), bid=5.00, ask=5.02, bid_size=1000, ask_size=1000))
        assert guard.check_spread("TST") is None

    def test_slippage_tracking(self) -> None:
        guard = SlippageGuard()
        guard.record_fill("TST", 5.00, 5.03)
        guard.record_fill("TST", 5.00, 5.01)
        avg = guard.avg_slippage("TST")
        assert abs(avg - 0.02) < 0.001

    def test_no_slippage_history(self) -> None:
        guard = SlippageGuard()
        assert guard.avg_slippage("UNKNOWN") == 0.0


# ---------------------------------------------------------------------------
# 6. Unified TradeGuard
# ---------------------------------------------------------------------------

class TestTradeGuard:

    def test_all_clear(self) -> None:
        """No guards triggered = entry allowed."""
        guard = TradeGuard()
        bars = [
            _bar(i, close=5.0 + i * 0.01, open_=5.0 + i * 0.01 - 0.005,
                 high=5.0 + i * 0.01 + 0.01, low=5.0 + i * 0.01 - 0.01,
                 volume=10_000)
            for i in range(7)
        ]
        bars[-1] = _bar(6, close=5.08, open_=5.06, high=5.09, low=5.05, volume=20_000)
        ok, reason = guard.check_entry(_signal(), bars=bars)
        assert ok is True
        assert reason is None

    def test_halt_blocks_entry(self) -> None:
        guard = TradeGuard()
        guard.halt_tracker.mark_halted("TST", _ts(0))
        ok, reason = guard.check_entry(_signal(), bars=None)
        assert ok is False
        assert "HALTED" in reason

    def test_market_panic_blocks_entry(self) -> None:
        guard = TradeGuard()
        guard.market_panic._panic_active = True
        guard.market_panic._panic_start = _ts(0)
        ok, reason = guard.check_entry(_signal())
        assert ok is False
        assert "panic" in reason.lower()

    def test_wide_spread_blocks_entry(self) -> None:
        guard = TradeGuard()
        guard.slippage.update_quote(
            Quote(symbol="TST", ts=_ts(0), bid=4.80, ask=5.20, bid_size=100, ask_size=100),
        )
        ok, reason = guard.check_entry(_signal())
        assert ok is False
        assert "spread too wide" in reason

    def test_false_breakout_blocks_entry(self) -> None:
        guard = TradeGuard()
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=20_000)
            for i in range(5)
        ]
        # Declining volume on "breakout"
        bars.append(_bar(5, close=5.05, open_=5.01, high=5.06, low=5.00, volume=10_000))
        ok, reason = guard.check_entry(_signal(), bars=bars)
        assert ok is False
        assert "false breakout" in reason

    def test_liquidity_trap_blocks_entry(self) -> None:
        guard = TradeGuard()
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=10_000)
            for i in range(5)
        ]
        # Gap-up reversal: opens above prior high but closes red
        # close > low ensures close position is above 30% so false breakout doesn't fire
        bars.append(_bar(5, close=5.00, open_=5.05, high=5.06, low=4.97, volume=12_000))
        ok, reason = guard.check_entry(_signal(), bars=bars)
        assert ok is False
        assert "liquidity trap" in reason

    def test_guard_priority_panic_first(self) -> None:
        """Market panic is checked before symbol-specific guards."""
        guard = TradeGuard()
        guard.market_panic._panic_active = True
        guard.market_panic._panic_start = _ts(0)
        guard.halt_tracker.mark_halted("TST", _ts(0))
        ok, reason = guard.check_entry(_signal())
        assert ok is False
        assert "panic" in reason.lower()

    def test_no_bars_skips_candle_checks(self) -> None:
        """With no bars, only halt/panic/spread guards run."""
        guard = TradeGuard()
        ok, reason = guard.check_entry(_signal(), bars=None)
        assert ok is True
