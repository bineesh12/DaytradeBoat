"""Advanced risk guards — false breakouts, liquidity traps, halts, market panic.

These guards sit between the verifier and the broker, providing additional
protection beyond basic position-sizing risk checks.
"""

from __future__ import annotations
from daytrading.data.bar_provenance import canonical_minute

import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from daytrading.models import Timeframe, Bar, ConfirmationWindow, Quote, TradeSignal
from daytrading.data.completed_bars import NY, Rejection, validate_micro_window

MICRO_PULLBACK = "micro_pullback_reclaim"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. False Breakout Detector
# ---------------------------------------------------------------------------

def _completed_session_context(bars, symbol, now):
    """Validated broad context; partial/future bars cannot grant an exception."""
    if now is None or now.tzinfo is None:
        return []
    by_time = {}
    for bar in bars or ():
        if (not canonical_minute(bar, now) or bar.symbol != symbol or bar.timeframe != Timeframe.MIN_1
                or bar.ts is None or bar.ts.tzinfo is None
                or bar.ts.astimezone(NY).date() != now.astimezone(NY).date()
                or (now-bar.ts).total_seconds() < 60):
            continue
        values = (bar.open,bar.high,bar.low,bar.close,bar.volume)
        if (not all(math.isfinite(v) for v in values) or bar.volume < 0
                or not 0 < bar.low <= min(bar.open,bar.close)
                   <= max(bar.open,bar.close) <= bar.high):
            return []
        if bar.ts in by_time and by_time[bar.ts] != bar:
            return []
        by_time[bar.ts] = bar
    return [by_time[t] for t in sorted(by_time)]


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

    def check(self, bars: Sequence[Bar], symbol: str = "", *,
              session_bars: Optional[Sequence[Bar]] = None,
              now: Optional[datetime] = None) -> Optional[str]:
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
            context = (list(bars) if session_bars is None else
                       _completed_session_context(session_bars, symbol, now))
            prior = context[:-1] if session_bars is None else context
            prior_high = max((b.high for b in prior), default=0.0)
            day_volume = sum(b.volume for b in context)
            pv = sum(((b.high + b.low + b.close) / 3.0) * b.volume for b in context)
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

    def __init__(self, max_spread_pct: Optional[float] = None) -> None:
        self._max_spread_pct = max_spread_pct

    def check(
        self,
        bars: Sequence[Bar],
        quotes: Optional[Sequence[Quote]] = None,
        symbol: str = "",
        *,
        wick_bar: Optional[Bar] = None,
        now: Optional[datetime] = None,
        diagnostics: Optional[dict] = None,
    ) -> Optional[str]:
        # The caller validates the original completed micro window. Keep the
        # wick candle separate from minute/session spread and gap context.
        if wick_bar is not None:
            values = (wick_bar.open, wick_bar.high, wick_bar.low,
                      wick_bar.close, wick_bar.volume)
            if (wick_bar.timeframe != Timeframe.SEC_10
                    or wick_bar.ts.tzinfo is None
                    or (symbol and wick_bar.symbol != symbol)
                    or not all(math.isfinite(v) and v > 0 for v in values)
                    or not wick_bar.low <= min(wick_bar.open, wick_bar.close)
                       <= max(wick_bar.open, wick_bar.close) <= wick_bar.high):
                return "liquidity trap: invalid original 10s wick evidence"
        if len(bars) < 5:
            return self._wick_rejection(wick_bar, micro=True) if wick_bar is not None else None

        latest = bars[-1]
        bar_range = latest.high - latest.low
        if bar_range <= 0 or latest.close <= 0:
            return self._wick_rejection(wick_bar, micro=True) if wick_bar is not None else None

        # Check spread from live quotes
        if quotes and len(quotes) >= 1:
            recent_quote = quotes[-1]
            from daytrading.strategy.entry_spread import get_policy,check as spread_check
            from daytrading.strategy.entry_checks import record
            policy=get_policy()
            # Old native checkpoints store the former default literal0.5.
            # It is the shared default, not an independent custom override.
            override=self._max_spread_pct if self._max_spread_pct not in (None,.5) else None
            limit=policy.limit(recent_quote.mid,'trap') if override is None else recent_quote.mid*override/100
            tolerance=1e-6 if override is None and policy.trap_tick_floor>0 else 0.
            wide=recent_quote.spread>limit+tolerance
            record(diagnostics,'liquidity_trap_current_spread',now=now,measured=recent_quote.spread,threshold=limit,units='USD/share',passed=not wide,window={'source':'current quote'},quote=recent_quote,conditional_gate='only blocks together with weak completed-minute volume')
            prior_5 = list(bars[-6:-1])
            avg_vol = sum(b.volume for b in prior_5) / 5 if len(prior_5) >= 5 else latest.volume
            record(diagnostics,'liquidity_trap_minute_volume',now=now,measured=latest.volume,threshold=avg_vol*policy.trap_minute_volume_ratio,units='shares',comparison='>=',passed=latest.volume>=avg_vol*policy.trap_minute_volume_ratio,window={'source':'1m context','start':latest.ts.isoformat(),'end':(latest.ts+timedelta(minutes=1)).isoformat(),'complete':now is not None and latest.ts.tzinfo is not None and latest.ts+timedelta(minutes=1)<=now,'prior_five_mean':avg_vol},quote=recent_quote,conditional_gate='only blocks together with wide current spread')
            record(diagnostics,'liquidity_trap_joint',now=now,measured=wide and latest.volume<avg_vol*policy.trap_minute_volume_ratio,
                threshold=False,units='boolean',comparison='must be false',passed=not (wide and latest.volume<avg_vol*policy.trap_minute_volume_ratio),
                window={'source':'current quote AND completed-minute volume'},quote=recent_quote)
            if wide:
                if latest.volume < avg_vol * policy.trap_minute_volume_ratio:
                    return "liquidity trap: spread {:.2f}% with weak volume {:.0f} vs avg {:.0f}".format(
                        recent_quote.spread_pct, latest.volume, avg_vol)

        # Same 2x rule, applied only to the correct trigger candle.
        wick_reason = self._wick_rejection(
            wick_bar if wick_bar is not None else latest,
            micro=wick_bar is not None,
        )
        if wick_reason:
            return wick_reason

        # Gap-up trap: current bar opens above prior bar's high but can't hold
        prev = bars[-2]
        if latest.open > prev.high and latest.close < latest.open:
            return "liquidity trap: gap-up reversal (opened {:.2f} above prior high {:.2f}, closed red)".format(
                latest.open, prev.high)

        return None

    @staticmethod
    def _wick_rejection(bar: Bar, *, micro: bool = False) -> Optional[str]:
        if bar.close > bar.open:
            upper_wick = bar.high - bar.close
            body = bar.close - bar.open
            if body > 0 and upper_wick > body * 2:
                if micro:
                    return ("liquidity trap: 10s spike-and-fade "
                            "(wick {:.6f} > 2x body {:.6f})").format(upper_wick, body)
                return "liquidity trap: spike-and-fade (wick {:.2f} > 2x body {:.2f})".format(
                    upper_wick, body)
        return None


# ---------------------------------------------------------------------------
# 3. Halt Tracker
# ---------------------------------------------------------------------------

class HaltTracker:
    """Track trading halts and block entries on halted/recently-halted stocks.

    Explicit halts require an explicit resume. A quiet gap with unchanged
    price creates a separate inferred low-activity block. Recovery requires
    three fresh one-second trade samples spanning two seconds within 10s,
    followed by the same post-resume cooldown.

    After a halt resumes, imposes a cooldown (default 120s) before
    allowing entries — post-halt price action is extremely volatile.
    """

    def __init__(self, resume_cooldown_secs: float = 120.0) -> None:
        self._resume_cooldown = resume_cooldown_secs
        self._halted: Dict[str, datetime] = {}
        self._resumed: Dict[str, datetime] = {}
        self._last_price: Dict[str, Tuple[float, datetime]] = {}
        # Silence is evidence of low activity, not an exchange halt. Explicit
        # halts remain blocked until an explicit resume notification arrives.
        self._low_activity: set = set()
        self._activity_ticks: Dict[str, deque] = {}
        self._data_uncertain: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def mark_halted(self, symbol: str, ts: Optional[datetime] = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        with self._lock:
            self._halted[symbol] = ts
            self._low_activity.discard(symbol)
            self._activity_ticks.pop(symbol, None)
            self._resumed.pop(symbol, None)
        logger.warning("HALT MARKED %s at %s", symbol, ts.isoformat())

    def mark_resumed(self, symbol: str, ts: Optional[datetime] = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        with self._lock:
            self._halted.pop(symbol, None)
            self._low_activity.discard(symbol)
            self._activity_ticks.pop(symbol, None)
            self._resumed[symbol] = ts
        logger.warning("HALT RESUMED %s at %s — cooldown %.0fs", symbol, ts.isoformat(), self._resume_cooldown)

    def update_price(self, symbol: str, price: float, ts: Optional[datetime] = None,
                     *, observed_at: Optional[datetime] = None) -> None:
        """Feed ordered trades; live callers reject delayed/future batches."""
        ts = ts or datetime.now(timezone.utc)
        # Discarded data diagnoses delivery/processing health, not exchange
        # silence. It can neither update the accepted price nor prove recovery.
        invalid = None
        if not math.isfinite(price) or price <= 0:
            invalid = 'invalid_price'
        elif ts.tzinfo is None or ts.utcoffset() is None:
            invalid = 'invalid_exchange_clock'
        elif observed_at is not None:
            if observed_at.tzinfo is None or observed_at.utcoffset() is None:
                invalid = 'invalid_observation_clock'
            else:
                age = (observed_at-ts).total_seconds()
                if not 0 <= age <= 10:
                    invalid = 'future_trade' if age < 0 else 'stale_trade'
        with self._lock:
            if invalid:
                stamp = observed_at if observed_at is not None else ts
                if stamp.tzinfo is None:
                    stamp = datetime.now(timezone.utc)
                row = self._data_uncertain.get(symbol)
                if row is None:
                    row = self._data_uncertain[symbol] = dict(
                        since=stamp, reason=invalid, discarded=0,
                        last_observed_at=stamp, last_exchange_ts=ts)
                    logger.warning('DATA HEALTH %s — %s; new entries blocked', symbol, invalid)
                row.update(reason=invalid, discarded=row['discarded']+1,
                           last_observed_at=stamp, last_exchange_ts=ts)
                self._activity_ticks.pop(symbol, None)
                return
            prev = self._last_price.get(symbol)
            if prev is not None and ts <= prev[1]:
                return  # Delayed/duplicate prints cannot demonstrate recovery.
            self._last_price[symbol] = (price, ts)

            if prev is not None:
                prev_price, prev_ts = prev
                silence = (ts - prev_ts).total_seconds()

                # Price freeze: no meaningful price change for 30+ seconds
                if (symbol not in self._data_uncertain and silence >= 30
                        and abs(price - prev_price) / max(prev_price, 0.01) < 0.001):
                    if symbol not in self._halted:
                        self._halted[symbol] = prev_ts
                        self._low_activity.add(symbol)
                        self._resumed.pop(symbol, None)
                        logger.info("LOW ACTIVITY %s — no trades for %.0fs (blocking entry until volume returns)", symbol, silence)
            if symbol in self._low_activity or symbol in self._data_uncertain:
                ticks = self._activity_ticks.setdefault(symbol, deque(maxlen=12))
                # Sample at most once per second: a buffered burst must not
                # count as sustained recovery, and busy tape stays bounded.
                if not ticks or (ts - ticks[-1]).total_seconds() >= 1:
                    ticks.append(ts)

    def adopt_legacy_low_activity(self, symbol: str, halt_ts: datetime) -> None:
        """Classify a legacy flag only after its original log proves its source.

        This never removes a block or reuses pre-restart ticks for recovery.
        Unknown legacy flags keep explicit-halt behavior.
        """
        with self._lock:
            if self._halted.get(symbol) != halt_ts:
                raise ValueError('Legacy activity flag changed during migration')
            self._low_activity.add(symbol)
            self._activity_ticks.pop(symbol, None)

    def check(self, symbol: str, now: Optional[datetime] = None) -> Optional[str]:
        """Return rejection reason if symbol is halted or in post-halt cooldown."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            if symbol in self._halted and symbol not in self._low_activity:
                duration = (now-self._halted[symbol]).total_seconds()
                return "HALTED: trading suspended for {:.0f}s".format(duration)
            if symbol in self._low_activity or symbol in self._data_uncertain:
                uncertain = self._data_uncertain.get(symbol)
                ticks = [t for t in self._activity_ticks.get(symbol, ())
                         if 0 <= (now - t).total_seconds() <= 10]
                latest = self._last_price.get(symbol)
                fresh = latest is not None and 0 <= (now - latest[1]).total_seconds() <= 10
                if fresh and len(ticks) >= 3 and (ticks[-1] - ticks[0]).total_seconds() >= 2:
                    self._halted.pop(symbol, None)
                    self._low_activity.discard(symbol)
                    self._data_uncertain.pop(symbol, None)
                    self._activity_ticks.pop(symbol, None)
                    self._resumed[symbol] = now
                    logger.info('ACTIVITY RECOVERED %s — fresh sustained trades; cooldown %.0fs',
                                symbol, self._resume_cooldown)
                else:
                    if uncertain:
                        return ('DATA HEALTH: {}; {} discarded observations; waiting for '
                                'sustained fresh trades ({}/3 in 10s; need 2s span); '
                                'market inactivity not established').format(
                                    uncertain['reason'], uncertain['discarded'], min(len(ticks), 3))
                    return ('LOW ACTIVITY: waiting for sustained fresh trades '
                            '({}/3 sampled trades in 10s; need 2s span); not a confirmed halt').format(min(len(ticks), 3))
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

    def activity_snapshot(self, now: Optional[datetime] = None) -> dict:
        """Refresh inferred recovery on the runner thread and explain blocks."""
        now=now or datetime.now(timezone.utc)
        with self._lock:
            inferred=list(self._low_activity | self._data_uncertain.keys())
        for symbol in inferred:
            self.check(symbol,now)
        with self._lock:
            result={}
            for symbol in set(self._halted) | set(self._resumed) | set(self._data_uncertain):
                remaining=max(0,self._resume_cooldown-(now-self._resumed[symbol]).total_seconds()) if symbol in self._resumed else 0
                if symbol not in self._halted and symbol not in self._data_uncertain and remaining<=0:
                    continue
                last=self._last_price.get(symbol)
                result[symbol]=dict(
                    source='explicit_or_legacy_halt' if symbol in self._halted and symbol not in self._low_activity else 'data_uncertain' if symbol in self._data_uncertain else 'low_activity' if symbol in self._low_activity else 'resume_cooldown',
                    data_health={k:v.isoformat() if isinstance(v,datetime) else v for k,v in self._data_uncertain[symbol].items()} if symbol in self._data_uncertain else None,
                    blocked_since=(self._halted[symbol] if symbol in self._halted else self._data_uncertain[symbol]['since'] if symbol in self._data_uncertain else self._resumed[symbol]).isoformat(),
                    last_trade_ts=last[1].isoformat() if last else None,
                    last_trade_age_seconds=round((now-last[1]).total_seconds(),1) if last else None,
                    sampled_trades_10s=sum(0<=(now-t).total_seconds()<=10 for t in self._activity_ticks.get(symbol,())),
                    cooldown_remaining_seconds=round(remaining,1))
            return result

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

    def __init__(self, max_spread_pct: Optional[float] = None) -> None:
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

    def get_executable_price(self, symbol: str, side: str) -> Optional[float]:
        """Actual quote side without the legacy discretionary one-cent buffer.

        A risk-reserved hard limit must be compared with this market price;
        adding a second buffer would reject an unchanged ask at its own cap.
        Callers retain quote-age checks and the original absolute price ceiling.
        """
        with self._lock:
            quote = self._latest_quotes.get(symbol)
            if quote is None or quote.bid <= 0 or quote.ask <= 0:
                return None
            return quote.ask if side == "buy" else quote.bid

    def quote_age_error(self, symbol: str, max_age_seconds: float = 10.0) -> Optional[str]:
        with self._lock:
            quote = self._latest_quotes.get(symbol)
            if quote is None:
                return None
            if quote.ts is None or quote.ts.tzinfo is None:
                return "quote has no reliable timestamp"
            age = (datetime.now(timezone.utc) - quote.ts).total_seconds()
            if age > max_age_seconds or age < -5:
                return "quote stale ({:.1f}s)".format(age)
        return None

    def check_spread(self, symbol: str, *, diagnostics=None, now=None) -> Optional[str]:
        """Reject if current spread is too wide. Skip check if no quote available."""
        with self._lock:
            q = self._latest_quotes.get(symbol)
            if q is None:
                return None  # no quote data — allow trade, classifier already assessed spread

            from daytrading.strategy.entry_spread import get_policy
            from daytrading.strategy.entry_checks import record
            # Preserve nondefault legacy overrides; migrate the old default1.0
            # to the named central safety ceiling after native state restore.
            maximum=get_policy().broker_safety_pct*100 if self._max_spread_pct in (None,1.) else self._max_spread_pct
            record(diagnostics,'broker_safety_spread',now=now,measured=q.spread,threshold=q.mid*maximum/100,units='USD/share',passed=q.spread_pct<=maximum,window={'source':'current broker safety quote'},quote=q)
            if q.spread_pct > maximum:
                return "spread too wide: {:.2f}% (max {:.2f}%) — bid {:.2f} / ask {:.2f}".format(
                    q.spread_pct, maximum, q.bid, q.ask)

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
        *,
        breakout_context: Optional[ConfirmationWindow] = None,
        now: Optional[datetime] = None,
        event_context: object = None,
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
        from daytrading.strategy.entry_checks import diagnostics
        reason = self.slippage.check_spread(signal.symbol,diagnostics=diagnostics(signal.scan_result),now=now)
        if reason:
            return False, reason

        # A micro trigger has a separate immutable confirmation window. Never
        # substitute a partial minute or the latest unrelated micro candle.
        hit = signal.scan_result
        native_micro = hit is not None and hit.scanner_name == MICRO_PULLBACK
        event_micro = hit is not None and hit.scanner_name == 'prepared_reclaim_tick'
        micro = native_micro or event_micro
        breakout_bars = bars
        if event_micro:
            if breakout_context is not None or event_context is None or now is None:
                return False, 'event confirmation: separate typed event evidence and observation time required'
            from daytrading.strategy.scalping.event_reclaim import validate_event_guard
            checked = validate_event_guard(signal, event_context, now)
            if isinstance(checked, Rejection):
                return False, 'event confirmation: {}: {}'.format(checked.code, checked.reason)
            current_quote = quotes[-1] if quotes else None
            if (current_quote is None or current_quote.ts.tzinfo is None
                    or not 0 < current_quote.bid < current_quote.ask
                    or min(current_quote.bid_size, current_quote.ask_size) <= 0
                    or not 0 <= (now-current_quote.ts).total_seconds() <= 10
                    or current_quote.bid < event_context.peak):
                return False, 'event confirmation: fresh current quote must still hold original reclaim'
            breakout_bars = checked
        elif native_micro:
            if event_context is not None:
                return False, 'micro confirmation: event evidence cannot replace original completed window'
            if breakout_context is None or now is None:
                return False, "micro confirmation: completed 10s context and observation time required"
            if (
                breakout_context.observed_at.tzinfo is None
                or now.tzinfo is None
                or breakout_context.observed_at > now
            ):
                return False, "micro confirmation: invalid or future context observation"
            invalid = validate_micro_window(
                signal.symbol, breakout_context.bars, now=now,
            )
            if invalid is not None:
                return False, "micro confirmation: {}: {}".format(invalid.code, invalid.reason)
            trigger = breakout_context.bars[-1]
            try:
                expected_start = datetime.fromisoformat(hit.criteria["micro_bar_start"])
                expected_end = datetime.fromisoformat(hit.criteria["micro_confirmed_at"])
            except (KeyError, TypeError, ValueError):
                return False, "micro confirmation: trigger identity missing or invalid"
            if (
                trigger.bar.ts != expected_start
                or trigger.closed_at != expected_end
                or trigger.closed_at > breakout_context.observed_at
            ):
                return False, "micro confirmation: window does not match original completed trigger"
            breakout_bars = [x.bar for x in breakout_context.bars[-6:]]
        elif breakout_context is not None or event_context is not None:
            return False, "confirmation context unsupported for this pattern"

        if micro:
            from daytrading.strategy.entry_checks import diagnostics,record
            trace=diagnostics(hit)
            complete=[b for b in (bars or ()) if b.timeframe==Timeframe.MIN_1 and b.ts.tzinfo is not None and b.ts+timedelta(minutes=1)<=now]
            record(trace,'minute_context_complete',now=now,measured=len(complete),threshold=len(bars or ()),units='bars',comparison='complete/available',passed=len(complete)==len(bars or ()),window={'source':'1m context','partial_bars_excluded':len(bars or ())-len(complete)},quote=quotes[-1] if quotes else None,informational=True)
            bars=complete
        else:trace=None
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
            if micro or pattern not in pullback_patterns:
                reason = self.false_breakout.check(breakout_bars, symbol=signal.symbol,
                    **({"session_bars": bars or [], "now": now} if micro else {}))
                if reason:
                    return False, reason

            # 5. Liquidity trap
            trap_kwargs = dict(quotes=quotes, symbol=signal.symbol,now=now,diagnostics=trace)
            if micro:
                trap_kwargs["wick_bar"] = breakout_bars[-1]
            reason = self.liquidity_trap.check(bars, **trap_kwargs)
            if reason:
                return False, reason

        elif micro:
            # Even when minute history is short, confirmation is mandatory.
            reason = self.false_breakout.check(breakout_bars, symbol=signal.symbol,
                    **({"session_bars": bars or [], "now": now} if micro else {}))
            if reason:
                return False, reason
            # Wick evidence remains mandatory even with short minute history.
            reason = self.liquidity_trap.check(
                bars or [], quotes=quotes, symbol=signal.symbol,
                wick_bar=breakout_bars[-1], now=now, diagnostics=trace,
            )
            if reason:
                return False, reason

        return True, None
