"""Advanced risk guards — false breakouts, liquidity traps, halts, market panic.

These guards sit between the verifier and the broker, providing additional
protection beyond basic position-sizing risk checks.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from daytrading.models import Bar, Quote, TradeSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. False Breakout Detector
# ---------------------------------------------------------------------------

class FalseBreakoutDetector:
    """Reject entries where price breaks a level but volume doesn't confirm.

    A real breakout has:
      - Volume on the breakout bar >= 1.5x the average of the prior 5 bars
      - Follow-through: the bar closes in the upper 60% of its range
      - No immediate rejection wick (upper wick < 40% of total range)

    A false breakout has:
      - Declining volume on the "breakout" bar
      - Long upper wick (sellers rejecting the move)
      - Close near the low of the bar (buyers couldn't hold)
    """

    def check(self, bars: Sequence[Bar], symbol: str = "") -> Optional[str]:
        if len(bars) < 6:
            return None

        latest = bars[-1]
        prior_5 = list(bars[-6:-1])
        avg_vol = sum(b.volume for b in prior_5) / 5

        if avg_vol <= 0:
            return None

        vol_ratio = latest.volume / avg_vol

        # Volume declining on the breakout candle = weak conviction
        if vol_ratio < 0.8:
            bar_range = latest.high - latest.low
            close_position = (latest.close - latest.low) / bar_range if bar_range > 0 else 0.0
            prior_high = max(b.high for b in bars[:-1])
            day_volume = sum(b.volume for b in bars)
            pv = sum(((b.high + b.low + b.close) / 3.0) * b.volume for b in bars)
            vwap = pv / day_volume if day_volume > 0 else 0.0
            recent_follow_through = len(bars) >= 3 and latest.close > bars[-3].close
            vwap_reclaim = vwap > 0 and latest.close >= vwap * 1.01
            near_high = prior_high > 0 and latest.close >= prior_high * 0.97
            strong_close = latest.close > latest.open and close_position >= 0.55
            if (
                day_volume >= 500_000
                and vwap_reclaim
                and near_high
                and (strong_close or recent_follow_through)
            ):
                return None
            return "false breakout: volume declining ({:.1f}x avg on breakout bar)".format(vol_ratio)

        bar_range = latest.high - latest.low
        if bar_range <= 0:
            return None

        # Upper wick analysis (for longs): big upper wick = sellers rejecting
        if latest.close > latest.open:
            upper_wick = latest.high - latest.close
            wick_ratio = upper_wick / bar_range
            if wick_ratio > 0.50 and vol_ratio < 1.2:
                return "false breakout: rejection wick {:.0f}% of range with weak volume {:.1f}x".format(
                    wick_ratio * 100, vol_ratio)

        # Close position in the bar range
        close_position = (latest.close - latest.low) / bar_range
        if close_position < 0.30 and latest.close > latest.open:
            return "false breakout: closed in lower 30% of range (buyers couldn't hold)"

        return None


# ---------------------------------------------------------------------------
# 2. Liquidity Trap Detector
# ---------------------------------------------------------------------------

class LiquidityTrapDetector:
    """Detect thin-book spikes that are likely to reverse.

    Signs of a liquidity trap:
      - Wide spread (> 0.5% of price) — thin order book
      - Spike bar with volume < average — no real buying
      - Immediate reversal: bar closes below its midpoint after a spike
      - Gap-up with no follow-through
    """

    def __init__(self, max_spread_pct: float = 0.5) -> None:
        self._max_spread_pct = max_spread_pct

    def check(
        self,
        bars: Sequence[Bar],
        quotes: Optional[Sequence[Quote]] = None,
        symbol: str = "",
    ) -> Optional[str]:
        if len(bars) < 5:
            return None

        latest = bars[-1]
        bar_range = latest.high - latest.low
        if bar_range <= 0 or latest.close <= 0:
            return None

        # Check spread from live quotes
        if quotes and len(quotes) >= 1:
            recent_quote = quotes[-1]
            if recent_quote.spread_pct > self._max_spread_pct:
                # Wide spread + spike = likely liquidity trap
                prior_5 = list(bars[-6:-1])
                avg_vol = sum(b.volume for b in prior_5) / 5 if len(prior_5) >= 5 else latest.volume
                if latest.volume < avg_vol * 1.2:
                    return "liquidity trap: spread {:.2f}% with weak volume {:.0f} vs avg {:.0f}".format(
                        recent_quote.spread_pct, latest.volume, avg_vol)

        # Spike-and-fade: bar spiked high but closed near the low
        if latest.close > latest.open:
            upper_wick = latest.high - latest.close
            body = latest.close - latest.open
            if body > 0 and upper_wick > body * 2:
                return "liquidity trap: spike-and-fade (wick {:.2f} > 2x body {:.2f})".format(
                    upper_wick, body)

        # Gap-up trap: current bar opens above prior bar's high but can't hold
        prev = bars[-2]
        if latest.open > prev.high and latest.close < latest.open:
            return "liquidity trap: gap-up reversal (opened {:.2f} above prior high {:.2f}, closed red)".format(
                latest.open, prev.high)

        return None


# ---------------------------------------------------------------------------
# 3. Halt Tracker
# ---------------------------------------------------------------------------

class HaltTracker:
    """Track trading halts and block entries on halted/recently-halted stocks.

    Detects halts via:
      - Alpaca trade status codes (if available)
      - Price freeze: no tick movement for 30+ seconds
      - Gap after silence: price jumps >5% after a quiet period

    After a halt resumes, imposes a cooldown (default 120s) before
    allowing entries — post-halt price action is extremely volatile.
    """

    def __init__(self, resume_cooldown_secs: float = 120.0) -> None:
        self._resume_cooldown = resume_cooldown_secs
        self._halted: Dict[str, datetime] = {}
        self._resumed: Dict[str, datetime] = {}
        self._last_price: Dict[str, Tuple[float, datetime]] = {}
        self._lock = threading.Lock()

    def mark_halted(self, symbol: str, ts: Optional[datetime] = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        with self._lock:
            self._halted[symbol] = ts
            self._resumed.pop(symbol, None)
        logger.warning("HALT MARKED %s at %s", symbol, ts.isoformat())

    def mark_resumed(self, symbol: str, ts: Optional[datetime] = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        with self._lock:
            self._halted.pop(symbol, None)
            self._resumed[symbol] = ts
        logger.warning("HALT RESUMED %s at %s — cooldown %.0fs", symbol, ts.isoformat(), self._resume_cooldown)

    def update_price(self, symbol: str, price: float, ts: Optional[datetime] = None) -> None:
        """Feed tick prices to detect halts via price freeze or post-halt gaps."""
        ts = ts or datetime.now(timezone.utc)
        with self._lock:
            prev = self._last_price.get(symbol)
            self._last_price[symbol] = (price, ts)

            if prev is not None:
                prev_price, prev_ts = prev
                silence = (ts - prev_ts).total_seconds()

                # Price freeze: no meaningful price change for 30+ seconds
                if silence >= 30 and abs(price - prev_price) / max(prev_price, 0.01) < 0.001:
                    if symbol not in self._halted:
                        self._halted[symbol] = prev_ts
                        logger.info("LOW ACTIVITY %s — no trades for %.0fs (blocking entry until volume returns)", symbol, silence)

                # Post-halt gap: price jumped >5% after silence
                elif silence >= 30 and prev_price > 0:
                    gap_pct = abs(price - prev_price) / prev_price
                    if gap_pct > 0.05 and symbol in self._halted:
                        self._halted.pop(symbol, None)
                        self._resumed[symbol] = ts
                        logger.warning("HALT RESUMED (gap) %s — %.1f%% jump after %.0fs silence",
                                     symbol, gap_pct * 100, silence)

    def check(self, symbol: str, now: Optional[datetime] = None) -> Optional[str]:
        """Return rejection reason if symbol is halted or in post-halt cooldown."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            if symbol in self._halted:
                halt_ts = self._halted[symbol]
                duration = (now - halt_ts).total_seconds()
                return "HALTED: trading suspended for {:.0f}s".format(duration)

            if symbol in self._resumed:
                resume_ts = self._resumed[symbol]
                elapsed = (now - resume_ts).total_seconds()
                if elapsed < self._resume_cooldown:
                    remaining = self._resume_cooldown - elapsed
                    return "post-halt cooldown: {:.0f}s remaining (resumed {:.0f}s ago)".format(
                        remaining, elapsed)
                else:
                    self._resumed.pop(symbol, None)

        return None

    @property
    def halted_symbols(self) -> List[str]:
        with self._lock:
            return list(self._halted.keys())


# ---------------------------------------------------------------------------
# 4. Market Panic Detector (SPY/QQQ breadth monitor)
# ---------------------------------------------------------------------------

class MarketPanicDetector:
    """Monitor broad market (SPY) for sudden drops that signal panic.

    When SPY drops >0.5% in 5 minutes, the whole market is selling.
    Low-float momentum stocks get hit hardest during these events.

    Blocks ALL new entries during panic, allows exits to proceed.
    """

    def __init__(
        self,
        panic_drop_pct: float = 0.5,
        lookback_bars: int = 5,
        recovery_bars: int = 3,
    ) -> None:
        self._panic_drop = panic_drop_pct / 100.0
        self._lookback = lookback_bars
        self._recovery_bars = recovery_bars
        self._spy_bars: deque = deque(maxlen=100)
        self._panic_active = False
        self._panic_start: Optional[datetime] = None
        self._green_count = 0
        self._lock = threading.Lock()

    def update_spy_bar(self, bar: Bar) -> None:
        """Feed SPY 1-minute bars to monitor market health."""
        with self._lock:
            self._spy_bars.append(bar)

            if len(self._spy_bars) >= self._lookback:
                lookback_bars = list(self._spy_bars)[-self._lookback:]
                start_price = lookback_bars[0].open
                end_price = lookback_bars[-1].close

                if start_price > 0:
                    change = (end_price - start_price) / start_price

                    if change <= -self._panic_drop:
                        if not self._panic_active:
                            self._panic_active = True
                            self._panic_start = bar.ts or datetime.now(timezone.utc)
                            self._green_count = 0
                            logger.warning(
                                "MARKET PANIC: SPY dropped %.1f%% in last %d bars — blocking entries",
                                change * 100, self._lookback,
                            )
                    elif self._panic_active:
                        if bar.close > bar.open:
                            self._green_count += 1
                        else:
                            self._green_count = 0

                        if self._green_count >= self._recovery_bars:
                            self._panic_active = False
                            logger.info("MARKET RECOVERY: SPY had %d green bars — entries resumed", self._green_count)

    def check(self) -> Optional[str]:
        """Return rejection reason if market is in panic mode."""
        with self._lock:
            if self._panic_active:
                duration = 0
                if self._panic_start:
                    duration = (datetime.now(timezone.utc) - self._panic_start).total_seconds()
                return "market panic: SPY selloff detected {:.0f}s ago — no new entries".format(duration)
        return None

    @property
    def is_panic(self) -> bool:
        with self._lock:
            return self._panic_active


# ---------------------------------------------------------------------------
# 5. Slippage Guard (quote-aware limit pricing)
# ---------------------------------------------------------------------------

class SlippageGuard:
    """Use live quotes to set smarter limit prices and reject bad fills.

    Instead of fixed buffer percentages, uses the actual bid/ask:
      - BUY: limit at ask + 1 cent (not a % above close)
      - SELL: limit at bid - 1 cent (not 3% below)
      - Reject if spread > max threshold (market too thin)
      - Track slippage per symbol to detect deteriorating conditions
    """

    def __init__(self, max_spread_pct: float = 1.0) -> None:
        self._max_spread_pct = max_spread_pct
        self._latest_quotes: Dict[str, Quote] = {}
        self._slippage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self._lock = threading.Lock()

    def update_quote(self, quote: Quote) -> None:
        with self._lock:
            self._latest_quotes[quote.symbol] = quote

    def get_limit_price(self, symbol: str, side: str) -> Optional[float]:
        """Return a smart limit price based on live quotes.

        Returns None if no quote available (caller should use fallback).
        """
        with self._lock:
            q = self._latest_quotes.get(symbol)
            if q is None or q.bid <= 0 or q.ask <= 0:
                return None

            if side == "buy":
                return round(q.ask + 0.01, 2)
            else:
                return round(q.bid - 0.01, 2)

    def check_spread(self, symbol: str) -> Optional[str]:
        """Reject if current spread is too wide. Skip check if no quote available."""
        with self._lock:
            q = self._latest_quotes.get(symbol)
            if q is None:
                return None  # no quote data — allow trade, classifier already assessed spread

            if q.spread_pct > self._max_spread_pct:
                return "spread too wide: {:.2f}% (max {:.2f}%) — bid {:.2f} / ask {:.2f}".format(
                    q.spread_pct, self._max_spread_pct, q.bid, q.ask)

        return None

    def record_fill(self, symbol: str, expected_price: float, fill_price: float) -> None:
        """Track actual slippage for monitoring."""
        slip = fill_price - expected_price
        with self._lock:
            self._slippage_history[symbol].append(slip)

        if abs(slip) > 0.02:
            logger.warning(
                "SLIPPAGE %s: expected %.4f, filled %.4f (slip $%.4f)",
                symbol, expected_price, fill_price, slip,
            )

    def avg_slippage(self, symbol: str) -> float:
        with self._lock:
            history = self._slippage_history.get(symbol)
            if not history:
                return 0.0
            return sum(history) / len(history)


# ---------------------------------------------------------------------------
# Unified Guard — combines all guards into a single pre-trade check
# ---------------------------------------------------------------------------

class TradeGuard:
    """Unified pre-trade risk guard combining all detectors.

    Call ``check_entry(signal, bars, quotes)`` before submitting any order.
    Returns (ok, rejection_reason).
    """

    def __init__(self) -> None:
        self.false_breakout = FalseBreakoutDetector()
        self.liquidity_trap = LiquidityTrapDetector()
        self.halt_tracker = HaltTracker()
        self.market_panic = MarketPanicDetector()
        self.slippage = SlippageGuard()

    def check_entry(
        self,
        signal: TradeSignal,
        bars: Optional[Sequence[Bar]] = None,
        quotes: Optional[Sequence[Quote]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Run all guards. Returns (True, None) if OK, or (False, reason) to reject."""

        # 1. Market panic — blocks everything
        reason = self.market_panic.check()
        if reason:
            return False, reason

        # 2. Halt check
        reason = self.halt_tracker.check(signal.symbol)
        if reason:
            return False, reason

        # 3. Spread check
        reason = self.slippage.check_spread(signal.symbol)
        if reason:
            return False, reason

        if bars and len(bars) >= 6:
            # 4. False breakout — skip for pullback patterns (volume is expected to be low)
            pattern = ""
            if signal.scan_result:
                pattern = signal.scan_result.criteria.get("pattern", "")
                if not pattern and signal.scan_result.scanner_name == "momentum_burst":
                    pattern = "momentum_burst"
            pullback_patterns = (
                "vwap_pullback",
                "pullback_base",
                "hod_reclaim",
                "first_pullback_reclaim",
            )
            if pattern not in pullback_patterns:
                reason = self.false_breakout.check(bars, symbol=signal.symbol)
                if reason:
                    return False, reason

            # 5. Liquidity trap
            reason = self.liquidity_trap.check(bars, quotes=quotes, symbol=signal.symbol)
            if reason:
                return False, reason

        return True, None
