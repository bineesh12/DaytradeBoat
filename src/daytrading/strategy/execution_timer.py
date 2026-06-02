"""10-second execution timer — waits for micro-pullback before entering.

When a 1-minute pattern is confirmed, instead of buying immediately,
this module watches 10-second bars for a better entry moment:

  1. Micro-pullback bounce: a red 10s bar followed by a green 10s bar
  2. Green 10s candle after a small dip (close > open, low < prev close)
  3. Any green 10s candle with volume

If no favorable micro-signal appears within max_wait_bars (default 1 = 10 sec),
the trade executes at market to avoid missing the move entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from daytrading.models import Bar, SignalAction, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class PendingEntry:
    """A signal waiting for 10-sec execution timing."""
    signal: TradeSignal
    queued_at: datetime
    bars_seen: int = 0
    max_wait_bars: int = 3
    max_wait_seconds: Optional[float] = None
    saw_red: bool = False
    best_price: Optional[float] = None


class ExecutionTimer:
    """Holds signals and waits for favorable 10-sec bar before executing.

    Flow:
      1. Pipeline generates a signal (all checks passed)
      2. Signal goes into ExecutionTimer.queue()
      3. Each 10-sec bar, call check() to see if any signal is ready
      4. Returns signals that should execute NOW

    Fallback: if max_wait_bars pass with no good micro-entry,
    the signal is released for immediate execution.
    """

    def __init__(self, max_wait_bars: int = 1, enabled: bool = True) -> None:
        self._max_wait = max_wait_bars
        self._enabled = enabled
        self._pending: Dict[str, PendingEntry] = {}

    @property
    def pending_symbols(self) -> List[str]:
        return list(self._pending.keys())

    def queue(self, signal: TradeSignal) -> bool:
        """Queue a signal for 10-sec execution timing.

        Returns True if queued, False if timer is disabled (execute immediately).
        """
        if not self._enabled:
            return False

        sym = signal.symbol
        if sym in self._pending:
            logger.debug("EXEC_TIMER: %s already pending, skipping duplicate", sym)
            return True

        hot_wait = self._hot_momentum_wait_seconds(signal)
        self._pending[sym] = PendingEntry(
            signal=signal,
            queued_at=datetime.now(timezone.utc),
            max_wait_bars=self._max_wait,
            max_wait_seconds=hot_wait,
        )
        if hot_wait is not None:
            logger.info(
                "EXEC_TIMER: queued %s — hot HOD squeeze, waiting briefly for 10s micro-entry (max %.0fs)",
                sym, hot_wait,
            )
        else:
            logger.info(
                "EXEC_TIMER: queued %s — waiting for 10s micro-entry (max %d bars = %ds)",
                sym, self._max_wait, self._max_wait * 10,
            )
        return True

    def on_10s_bar(self, bar: Bar) -> Optional[TradeSignal]:
        """Feed a 10-sec bar. Returns a signal if it's time to execute."""
        sym = bar.symbol
        pending = self._pending.get(sym)
        if pending is None:
            return None

        pending.bars_seen += 1

        is_green = bar.close > bar.open
        is_red = bar.close < bar.open

        if pending.best_price is None or bar.low < pending.best_price:
            pending.best_price = bar.low

        # Track if we've seen a micro-pullback (red bar)
        if is_red:
            pending.saw_red = True

        # Trigger 1: Micro-pullback bounce — red bar followed by green bar
        if pending.saw_red and is_green:
            logger.info(
                "EXEC_TIMER: %s — micro-pullback bounce detected (bar %d), executing",
                sym, pending.bars_seen,
            )
            return self._release(sym)

        # Trigger 2: Green bar with body > 30% of range (conviction)
        if is_green and bar.high > bar.low:
            body = bar.close - bar.open
            rng = bar.high - bar.low
            if rng > 0 and body / rng > 0.30:
                logger.info(
                    "EXEC_TIMER: %s — strong green 10s bar (bar %d), executing",
                    sym, pending.bars_seen,
                )
                return self._release(sym)

        # Fallback: waited long enough, execute at market
        if pending.bars_seen >= pending.max_wait_bars:
            logger.info(
                "EXEC_TIMER: %s — max wait reached (%d bars), executing at market",
                sym, pending.bars_seen,
            )
            return self._release(sym)

        return None

    def check_timeouts(self) -> List[TradeSignal]:
        """Release any signals that have been waiting too long (safety net).

        Call this periodically (e.g., every second) in case 10-sec bars
        stop arriving for a symbol.
        """
        now = datetime.now(timezone.utc)
        released = []
        for sym in list(self._pending.keys()):
            pending = self._pending[sym]
            age = (now - pending.queued_at).total_seconds()
            max_wait_seconds = pending.max_wait_seconds
            if max_wait_seconds is None:
                max_wait_seconds = (pending.max_wait_bars + 1) * 10
            if age > max_wait_seconds:
                logger.warning(
                    "EXEC_TIMER: %s — timeout after %.0fs with no 10s bars, forcing execution",
                    sym, age,
                )
                released.append(self._release(sym))
        return released

    def cancel(self, symbol: str) -> None:
        """Remove a pending signal (e.g., if conditions changed)."""
        if symbol in self._pending:
            del self._pending[symbol]
            logger.info("EXEC_TIMER: cancelled pending entry for %s", symbol)

    def _release(self, symbol: str) -> TradeSignal:
        pending = self._pending.pop(symbol)
        return pending.signal

    @staticmethod
    def _hot_momentum_wait_seconds(signal: TradeSignal) -> Optional[float]:
        """Shorten the fallback wait for CMND-style HOD squeezes.

        Normal signals still wait for the usual 10s-bar confirmation. Very hot
        HOD/vwap reclaim setups can move without sending usable 10s bars, so
        waiting the full safety timeout often turns a good alert into a chase.
        """
        if signal.action not in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT):
            return None
        hit = signal.scan_result
        if hit is None:
            return None

        pattern = str(hit.criteria.get("pattern", ""))
        scanner = hit.scanner_name
        hot_patterns = {
            "hod_reclaim",
            "vwap_pullback",
            "breakout_scalp",
            "momentum_burst",
            "abc_continuation",
        }
        if pattern not in hot_patterns:
            if scanner not in hot_patterns:
                return None

        price = float(hit.criteria.get("close") or signal.entry_price or 0.0)
        if not (2.0 <= price <= 20.0):
            return None

        rally_pct = float(
            hit.criteria.get("rally_pct")
            or hit.criteria.get("change_session_pct")
            or 0.0
        )
        latest_volume = float(hit.criteria.get("volume") or 0.0)
        if pattern == "breakout_scalp" or scanner == "breakout_scalp":
            return 8.0
        if pattern in ("momentum_burst", "abc_continuation") or scanner in (
            "momentum_burst", "abc_continuation",
        ):
            return 10.0
        if rally_pct >= 25.0 and latest_volume >= 100_000:
            return 12.0
        return None
