"""10-second execution timer — waits for micro-pullback before entering.

When a 1-minute pattern is confirmed, instead of buying immediately,
this module watches 10-second bars for a better entry moment:

  1. Micro-pullback bounce: a red 10s bar followed by a green 10s bar
  2. Green 10s candle after a small dip (close > open, low < prev close)
  3. Any green 10s candle with volume

For structured HOD/pullback/reclaim setups, no favorable micro-signal usually
means cancel instead of chasing. Elite hot-watch setups can fall back only if
the 10s tape stayed clean; the runner still applies live chase/spread guards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
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
    require_pullback_reclaim: bool = False
    require_micro_signal: bool = False
    last_bar: Optional[Bar] = None


class ExecutionTimer:
    """Holds signals and waits for favorable 10-sec bar before executing.

    Flow:
      1. Pipeline generates a signal (all checks passed)
      2. Signal goes into ExecutionTimer.queue()
      3. Each 10-sec bar, call check() to see if any signal is ready
      4. Returns signals that should execute NOW

    Fallback: if max_wait_bars pass with no good micro-entry, normal signals
    are released. Structured pullback/reclaim signals are cancelled.
    """

    def __init__(self, max_wait_bars: int = 1, enabled: bool = True) -> None:
        self._max_wait = max_wait_bars
        self._enabled = enabled
        self._pending: Dict[str, PendingEntry] = {}

    @property
    def pending_symbols(self) -> List[str]:
        return list(self._pending.keys())

    def seconds_until_next_timeout(self, now: Optional[datetime] = None) -> Optional[float]:
        """Return seconds until the next pending signal must be handled."""
        if not self._pending:
            return None
        now = now or datetime.now(timezone.utc)
        waits = []
        for pending in self._pending.values():
            age = (now - pending.queued_at).total_seconds()
            waits.append(self._timeout_seconds(pending) - age)
        return max(0.0, min(waits))

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

        if signal.scan_result is not None:
            signal.scan_result.criteria.setdefault("queued_entry_price", signal.entry_price)

        hot_wait = self._hot_momentum_wait_seconds(signal)
        require_pullback_reclaim = self._requires_pullback_reclaim(signal)
        require_micro_signal = signal.scan_result is not None
        self._pending[sym] = PendingEntry(
            signal=signal,
            queued_at=datetime.now(timezone.utc),
            max_wait_bars=self._max_wait,
            max_wait_seconds=hot_wait,
            require_pullback_reclaim=require_pullback_reclaim,
            require_micro_signal=require_micro_signal,
        )
        if require_pullback_reclaim:
            wait_text = "{:.0f}s".format(hot_wait) if hot_wait is not None else "{}s".format(self._max_wait * 10)
            logger.info(
                "EXEC_TIMER: queued %s — waiting for 10s pullback/reclaim, no chase fallback (max %s)",
                sym, wait_text,
            )
        elif hot_wait is not None:
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
        pending.last_bar = bar

        is_green = bar.close > bar.open
        is_red = bar.close < bar.open

        if pending.best_price is None or bar.low < pending.best_price:
            pending.best_price = bar.low

        # Track if we've seen a micro-pullback (red bar)
        if is_red:
            pending.saw_red = True

        confirmation_failure = self._pullback_confirmation_failure(pending, bar)
        if confirmation_failure is not None:
            logger.info(
                "EXEC_TIMER: %s — 10s confirmation failed: %s; cancelling",
                sym, confirmation_failure,
            )
            self.cancel(sym)
            return None

        if self._allows_early_strength_release(pending, bar):
            logger.info(
                "EXEC_TIMER: %s — early strength 10s hold near setup base (bar %d), executing",
                sym, pending.bars_seen,
            )
            return self._release(sym)

        # Trigger 1: Micro-pullback bounce — red bar followed by green bar
        if pending.saw_red and is_green:
            if not self._passes_pullback_reclaim_confirmation(pending, bar):
                logger.info(
                    "EXEC_TIMER: %s — green 10s bounce did not reclaim setup yet, waiting",
                    sym,
                )
            else:
                logger.info(
                    "EXEC_TIMER: %s — micro-pullback bounce confirmed (bar %d), executing",
                    sym, pending.bars_seen,
                )
                return self._release(sym)

        # Trigger 2: Green bar with body > 30% of range (conviction)
        if is_green and bar.high > bar.low:
            body = bar.close - bar.open
            rng = bar.high - bar.low
            had_intrabar_dip = self._had_intrabar_dip(pending, bar)
            if rng > 0 and body / rng > 0.30 and (
                not pending.require_pullback_reclaim
                or (
                    had_intrabar_dip
                    and self._passes_pullback_reclaim_confirmation(pending, bar)
                )
            ):
                logger.info(
                    "EXEC_TIMER: %s — strong green 10s bar (bar %d), executing",
                    sym, pending.bars_seen,
                )
                return self._release(sym)

        # Fallback: waited long enough. Scanner signals must prove a favorable
        # micro-entry; otherwise a timeout can become a late chase into a dump.
        if pending.bars_seen >= pending.max_wait_bars:
            if pending.require_micro_signal:
                if self._allows_elite_fallback(pending, latest_bar=bar):
                    logger.info(
                        "EXEC_TIMER: %s — elite setup held clean through 10s wait, releasing with chase guards",
                        sym,
                    )
                    return self._release(sym)
                if self._allows_continuation_scout(pending, latest_bar=bar):
                    logger.info(
                        "EXEC_TIMER: %s — elite continuation scout after 10s wait, releasing reduced size",
                        sym,
                    )
                    return self._release(sym, scout_reason="continuation_scout")
                if self._allows_one_minute_pullback_release(pending, latest_bar=bar):
                    logger.info(
                        "EXEC_TIMER: %s — strong 1m pullback held clean through 10s wait, releasing with chase guards",
                        sym,
                    )
                    return self._release(sym)
                if self._allows_vwap_reclaim_release(pending, latest_bar=bar):
                    logger.info(
                        "EXEC_TIMER: %s — VWAP reclaim held clean through 10s wait, releasing with chase guards",
                        sym,
                    )
                    return self._release(sym)
                if self._allows_pullback_scout(pending, latest_bar=bar):
                    logger.info(
                        "EXEC_TIMER: %s — strong pullback scout after 10s wait, releasing reduced size",
                        sym,
                    )
                    return self._release(sym, scout_reason="pullback_scout")
                logger.info(
                    "EXEC_TIMER: %s — max wait reached without favorable 10s entry, cancelling to avoid chase",
                    sym,
                )
                self.cancel(sym)
                return None
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
            max_wait_seconds = self._timeout_seconds(pending)
            if age >= max_wait_seconds:
                if pending.require_micro_signal:
                    if self._allows_elite_fallback(pending):
                        logger.warning(
                            "EXEC_TIMER: %s — timeout after %.0fs with elite setup still clean, releasing with chase guards",
                            sym, age,
                        )
                        released.append(self._release(sym))
                        continue
                    if self._allows_continuation_scout(pending, latest_bar=pending.last_bar):
                        logger.warning(
                            "EXEC_TIMER: %s — timeout after %.0fs, releasing reduced-size continuation scout",
                            sym, age,
                        )
                        released.append(self._release(sym, scout_reason="continuation_scout"))
                        continue
                    if self._allows_one_minute_pullback_release(
                        pending,
                        latest_bar=pending.last_bar,
                    ):
                        logger.warning(
                            "EXEC_TIMER: %s — timeout after %.0fs with strong 1m pullback still clean, releasing with chase guards",
                            sym, age,
                        )
                        released.append(self._release(sym))
                        continue
                    if self._allows_vwap_reclaim_release(
                        pending,
                        latest_bar=pending.last_bar,
                    ):
                        logger.warning(
                            "EXEC_TIMER: %s — timeout after %.0fs with VWAP reclaim still clean, releasing with chase guards",
                            sym, age,
                        )
                        released.append(self._release(sym))
                        continue
                    if self._allows_pullback_scout(
                        pending,
                        latest_bar=pending.last_bar,
                    ):
                        logger.warning(
                            "EXEC_TIMER: %s — timeout after %.0fs, releasing reduced-size pullback scout",
                            sym, age,
                        )
                        released.append(self._release(sym, scout_reason="pullback_scout"))
                        continue
                    logger.warning(
                        "EXEC_TIMER: %s — timeout after %.0fs with no favorable 10s entry, cancelling to avoid chase",
                        sym, age,
                    )
                    self.cancel(sym)
                    continue
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

    @staticmethod
    def _timeout_seconds(pending: PendingEntry) -> float:
        if pending.max_wait_seconds is not None:
            return pending.max_wait_seconds
        return max(1.0, float(pending.max_wait_bars * 10))

    def _release(
        self,
        symbol: str,
        *,
        scout_reason: Optional[str] = None,
    ) -> TradeSignal:
        pending = self._pending.pop(symbol)
        if not scout_reason:
            return pending.signal
        scout_qty = max(1, int(float(pending.signal.quantity or 0) * 0.35))
        scout_qty = min(scout_qty, 75)
        return replace(
            pending.signal,
            quantity=float(scout_qty),
            reason="{} | {}".format(pending.signal.reason, scout_reason),
        )

    @staticmethod
    def _had_intrabar_dip(pending: PendingEntry, bar: Bar) -> bool:
        original = float(pending.signal.entry_price or 0.0)
        if original <= 0:
            return False
        dip_price = min(
            float(bar.low or original),
            float(pending.best_price or original),
        )
        return dip_price <= original * 0.995

    @staticmethod
    def _pullback_confirmation_failure(
        pending: PendingEntry,
        bar: Bar,
    ) -> Optional[str]:
        """Return a hard failure reason for pullback/reclaim micro-confirmation."""
        if not pending.require_pullback_reclaim:
            return None

        hit = pending.signal.scan_result
        criteria = hit.criteria if hit is not None else {}
        low = float(bar.low or 0.0)
        close = float(bar.close or 0.0)

        stop = ExecutionTimer._criteria_float(criteria, "stop_price")
        if stop is None and pending.signal.stop_loss is not None:
            stop = float(pending.signal.stop_loss)
        if stop is not None and stop > 0 and low <= stop:
            return "10s low {:.4f} broke stop {:.4f}".format(low, stop)

        pullback_low = (
            ExecutionTimer._criteria_float(criteria, "pullback_low")
            or ExecutionTimer._criteria_float(criteria, "base_low")
        )
        if pullback_low is not None and pullback_low > 0 and close < pullback_low:
            return "10s close {:.4f} lost pullback low {:.4f}".format(close, pullback_low)

        pattern = str(criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "") if hit is not None else ""
        if pattern == "opening_range_breakout" or scanner == "opening_range_breakout":
            setup_price = ExecutionTimer._criteria_float(criteria, "close")
            if setup_price is None or setup_price <= 0:
                setup_price = float(pending.signal.entry_price or 0.0)
            if setup_price > 0 and close < setup_price * 0.97:
                return "10s close {:.4f} pulled back too far from breakout signal {:.4f}".format(
                    close, setup_price,
                )

        volume_failure = ExecutionTimer._breakout_volume_failure(pending.signal, bar)
        if volume_failure is not None and pending.bars_seen > 1:
            return volume_failure

        breakout_level = ExecutionTimer._breakout_reclaim_level(pending.signal)
        if breakout_level is not None and breakout_level > 0 and close < breakout_level:
            return "10s close {:.4f} below breakout reclaim {:.4f}".format(
                close, breakout_level,
            )

        return None

    @staticmethod
    def _passes_pullback_reclaim_confirmation(
        pending: PendingEntry,
        bar: Bar,
    ) -> bool:
        """Require scanner pullbacks to reclaim the actual setup before release."""
        if not pending.require_pullback_reclaim:
            return True

        close = float(bar.close or 0.0)
        open_ = float(bar.open or 0.0)
        if close <= open_:
            return False

        hit = pending.signal.scan_result
        criteria = hit.criteria if hit is not None else {}

        breakout_level = ExecutionTimer._breakout_reclaim_level(pending.signal)
        setup_price = ExecutionTimer._criteria_float(criteria, "close")
        if setup_price is None or setup_price <= 0:
            setup_price = float(pending.signal.entry_price or 0.0)
        if breakout_level is None and setup_price > 0 and close < setup_price * 0.995:
            return False

        if breakout_level is not None and breakout_level > 0:
            if close < breakout_level:
                return False
            if (
                ExecutionTimer._is_level_breakout_signal(pending.signal)
                and close > breakout_level * 1.025
            ):
                return False

        vwap = ExecutionTimer._criteria_float(criteria, "vwap")
        if vwap is not None and vwap > 0 and close < vwap * 1.003:
            return False

        pullback_low = (
            ExecutionTimer._criteria_float(criteria, "pullback_low")
            or ExecutionTimer._criteria_float(criteria, "base_low")
        )
        if pullback_low is not None and pullback_low > 0 and close < pullback_low:
            return False

        if ExecutionTimer._breakout_volume_failure(pending.signal, bar) is not None:
            return False

        return True

    @staticmethod
    def _allows_early_strength_release(pending: PendingEntry, bar: Bar) -> bool:
        """Release clean grind-up setups before they turn into late chase entries."""
        if not pending.require_micro_signal:
            return False
        if pending.bars_seen > 2 or pending.saw_red:
            return False
        if bar.close <= bar.open or bar.close <= 0:
            return False

        hit = pending.signal.scan_result
        if hit is None:
            return False
        criteria = hit.criteria
        pattern = str(criteria.get("pattern") or hit.scanner_name or "")
        if pattern not in {
            "abc_continuation",
            "first_pullback_reclaim",
            "vwap_pullback",
            "pullback_base",
            "hod_reclaim",
            "level_breakout_reclaim",
            "early_vwap_reclaim_scout",
            "shallow_stair_continuation",
            "momentum_burst",
            "opening_range_breakout",
        }:
            return False

        setup_tier = str(criteria.get("setup_tier") or "").lower()
        entry_tier = str(criteria.get("entry_tier") or "").lower()
        early_context = (
            "a+" in setup_tier
            or entry_tier in {
                "a_plus_reclaim_scout",
                "a_plus_retry_watch",
                "deep_runner_scout",
                "level_scout",
                "stair_scout",
            }
            or pattern in {
                "level_breakout_reclaim",
                "early_vwap_reclaim_scout",
                "shallow_stair_continuation",
            }
        )
        if not early_context:
            return False

        base = (
            ExecutionTimer._criteria_float(criteria, "setup_anchor")
            or ExecutionTimer._criteria_float(criteria, "queued_entry_price")
            or ExecutionTimer._criteria_float(criteria, "breakout_level")
            or ExecutionTimer._criteria_float(criteria, "base_high")
            or ExecutionTimer._criteria_float(criteria, "close")
            or float(pending.signal.entry_price or 0.0)
        )
        if base <= 0:
            return False
        if bar.close > base * 1.02:
            return False
        if bar.low < base * 0.985:
            return False

        vwap_level = ExecutionTimer._criteria_float(criteria, "vwap")
        if vwap_level is None or vwap_level <= 0:
            return False
        if bar.close < vwap_level * 1.003:
            return False

        return True

    @staticmethod
    def _is_level_breakout_signal(signal: TradeSignal) -> bool:
        hit = signal.scan_result
        if hit is None:
            return False
        pattern = str(hit.criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")
        return pattern == "level_breakout_reclaim" or scanner == "level_breakout_reclaim"

    @staticmethod
    def _breakout_reclaim_level(signal: TradeSignal) -> Optional[float]:
        """Return the level that must be reclaimed for breakout-style entries.

        A lower fill can be good for a real pullback. It is not good for an
        opening-range breakout after price has fallen back below the breakout
        line; that is a failed breakout until the level is reclaimed.
        """
        hit = signal.scan_result
        if hit is None:
            return None
        criteria = hit.criteria
        pattern = str(criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")

        if pattern == "opening_range_breakout" or scanner == "opening_range_breakout":
            return (
                ExecutionTimer._criteria_float(criteria, "breakout_level")
                or ExecutionTimer._criteria_float(criteria, "resistance")
            )
        if pattern == "flat_top_breakout" or scanner == "flat_top_breakout":
            return (
                ExecutionTimer._criteria_float(criteria, "resistance")
                or ExecutionTimer._criteria_float(criteria, "breakout_level")
            )
        if pattern == "level_breakout_reclaim" or scanner == "level_breakout_reclaim":
            return (
                ExecutionTimer._criteria_float(criteria, "breakout_level")
                or ExecutionTimer._criteria_float(criteria, "base_high")
            )
        return None

    @staticmethod
    def _breakout_volume_failure(signal: TradeSignal, bar: Bar) -> Optional[str]:
        hit = signal.scan_result
        if hit is None:
            return None
        criteria = hit.criteria
        pattern = str(criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")
        if pattern not in {
            "opening_range_breakout",
            "flat_top_breakout",
            "level_breakout_reclaim",
        } and scanner not in {
            "opening_range_breakout",
            "flat_top_breakout",
            "level_breakout_reclaim",
        }:
            return None

        bar_volume = float(bar.volume or 0.0)
        setup_volume = ExecutionTimer._criteria_float(criteria, "volume") or 0.0
        min_volume = 1_000.0
        if setup_volume >= 100_000:
            min_volume = min(75_000.0, max(5_000.0, setup_volume * 0.01))
        if bar_volume < min_volume:
            return "10s breakout volume {:.0f} below {:.0f}".format(
                bar_volume, min_volume,
            )
        return None

    @staticmethod
    def _criteria_float(criteria: dict, key: str) -> Optional[float]:
        value = criteria.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _allows_elite_fallback(
        pending: PendingEntry,
        latest_bar: Optional[Bar] = None,
    ) -> bool:
        """Allow only top-quality scanner setups to leave the timer after a clean 10s wait.

        This is intentionally narrow: it fixes missed FABC/QMCO-style elite
        signals that saw actual 10s bars without reopening ordinary pullback/HOD
        signals to late chase entries. The runner performs the final live
        spread/chase/red-bar checks before any order is sent.
        """
        signal = pending.signal
        if signal.action is not SignalAction.ENTER_LONG:
            return False
        hit = signal.scan_result
        if hit is None:
            return False
        if pending.saw_red:
            return False
        if latest_bar is None:
            return False
        if latest_bar.close < latest_bar.open:
            return False

        pattern = str(hit.criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")
        elite_patterns = {
            "abc_continuation",
            "momentum_burst",
            "first_pullback_reclaim",
            "hod_reclaim",
        }
        if pattern not in elite_patterns and scanner not in elite_patterns:
            return False

        price = float(hit.criteria.get("close") or signal.entry_price or 0.0)
        if not (1.5 <= price <= 20.0):
            return False

        score = float(hit.score or 0.0)
        volume = float(hit.criteria.get("volume") or 0.0)
        if score < 95.0 or volume < 500_000:
            return False

        return True

    @staticmethod
    def _allows_continuation_scout(
        pending: PendingEntry,
        latest_bar: Optional[Bar] = None,
    ) -> bool:
        """Permit a small scout for exceptional momentum that never pulls back.

        This is intentionally narrower than the clean elite fallback. It is for
        ANY-style HOD squeezes where the setup is strong enough to participate,
        but not clean enough for full size.
        """
        signal = pending.signal
        if signal.action is not SignalAction.ENTER_LONG:
            return False
        hit = signal.scan_result
        if hit is None:
            return False

        pattern = str(hit.criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")
        scout_patterns = {
            "momentum_burst",
            "hod_reclaim",
            "opening_range_breakout",
        }
        if pattern not in scout_patterns and scanner not in scout_patterns:
            return False

        if not ExecutionTimer._is_strong_continuation_scout_signal(signal):
            return False

        if latest_bar is None:
            return pending.bars_seen > 0

        price = float(hit.criteria.get("close") or signal.entry_price or 0.0)
        close = float(latest_bar.close or 0.0)
        open_ = float(latest_bar.open or 0.0)
        high = float(latest_bar.high or max(open_, close))
        low = float(latest_bar.low or min(open_, close))
        if close <= 0:
            return False
        if close < price * 0.99:
            return False
        if close < open_ and high > low:
            red_body = open_ - close
            if red_body / (high - low) > 0.45:
                return False
        return True

    @staticmethod
    def _allows_no_10s_continuation_scout(pending: PendingEntry) -> bool:
        """Allow a reduced scout when strong continuation gets no 10s bars.

        FOXX-style runners can pass the 1-minute scanner/entry guard while the
        10s stream never sends a usable bar before timeout. This keeps the
        no-chase behavior for ordinary names, but lets exceptional liquid
        continuation setups participate with small size.
        """
        if pending.last_bar is not None or pending.bars_seen > 0:
            return False
        if pending.saw_red:
            return False
        return ExecutionTimer._is_strong_continuation_scout_signal(pending.signal)

    @staticmethod
    def _is_strong_continuation_scout_signal(signal: TradeSignal) -> bool:
        if signal.action is not SignalAction.ENTER_LONG:
            return False
        hit = signal.scan_result
        if hit is None:
            return False

        pattern = str(hit.criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")
        scout_patterns = {
            "momentum_burst",
            "hod_reclaim",
            "opening_range_breakout",
        }
        if pattern not in scout_patterns and scanner not in scout_patterns:
            return False

        price = float(hit.criteria.get("close") or signal.entry_price or 0.0)
        if not (1.5 <= price <= 20.0):
            return False

        volume = float(hit.criteria.get("volume") or 0.0)
        dollar_volume = volume * price
        if volume < 500_000 and dollar_volume < 1_000_000:
            return False

        score = float(hit.score or 0.0)
        burst_pct = float(hit.criteria.get("burst_pct") or 0.0)
        velocity = float(hit.criteria.get("velocity") or 0.0)
        if pattern == "momentum_burst" or scanner == "momentum_burst":
            if burst_pct > 0 or velocity > 0:
                return burst_pct >= 1.5 or velocity >= 0.05
            return score >= 95.0
        if pattern == "abc_continuation" or scanner == "abc_continuation":
            return score >= 8.0 or volume >= 700_000
        return score >= 3.0 or volume >= 700_000

    @staticmethod
    def _allows_one_minute_pullback_release(
        pending: PendingEntry,
        latest_bar: Optional[Bar] = None,
    ) -> bool:
        """Use a very strong 1-minute pullback as the primary setup decision.

        A clean 10s candle should still enter full size through the normal
        trigger. This fallback covers the common scalp case where the latest
        1-minute candle has already reclaimed/held the base, while 10s tape is
        flat instead of strongly green. Red or ugly 10s tape still falls back to
        reduced scout/cancel logic.
        """
        signal = pending.signal
        if signal.action is not SignalAction.ENTER_LONG:
            return False
        hit = signal.scan_result
        if hit is None:
            return False
        if ExecutionTimer._is_pullback_scout_only_signal(signal):
            return False

        pattern = str(hit.criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")
        if pattern != "pullback_base" and scanner != "pullback_base":
            return False

        price = float(hit.criteria.get("close") or signal.entry_price or 0.0)
        if not (1.5 <= price <= 20.0):
            return False

        score = float(hit.score or 0.0)
        if score < 90.0:
            return False

        day_move_pct = float(hit.criteria.get("day_move_pct") or 0.0)
        if day_move_pct > 0 and day_move_pct < 25.0:
            return False

        pullback_pct = hit.criteria.get("pullback_pct")
        if pullback_pct is not None:
            pullback = float(pullback_pct)
            if pullback < 3.0 or pullback > 8.0:
                return False

        base_range_pct = hit.criteria.get("base_range_pct")
        if base_range_pct is not None and float(base_range_pct) > 6.0:
            return False

        volume_score = ExecutionTimer._pullback_volume_profile_score(signal)
        if volume_score < 70.0:
            return False

        if latest_bar is None:
            return False

        close = float(latest_bar.close or 0.0)
        open_ = float(latest_bar.open or 0.0)
        if close <= 0:
            return False
        if close < open_:
            return False
        if close < price * 0.995:
            return False
        return True

    @staticmethod
    def _allows_vwap_reclaim_release(
        pending: PendingEntry,
        latest_bar: Optional[Bar] = None,
    ) -> bool:
        """Release a clean VWAP reclaim when 10s tape is quiet but not failing.

        BGMS-style VWAP entries can pass the 1-minute scanner/guard at the
        right price, then miss because the 10s bar is flat instead of a strong
        green trigger. This keeps hard failure checks active and only releases
        if the 10s close is still near the original setup and above VWAP.
        """
        signal = pending.signal
        if signal.action is not SignalAction.ENTER_LONG:
            return False
        hit = signal.scan_result
        if hit is None:
            return False

        pattern = str(hit.criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")
        if pattern != "vwap_pullback" and scanner != "vwap_pullback":
            return False

        if pending.saw_red:
            return False
        if latest_bar is None:
            return False

        price = float(hit.criteria.get("close") or signal.entry_price or 0.0)
        if not (1.5 <= price <= 20.0):
            return False

        close = float(latest_bar.close or 0.0)
        open_ = float(latest_bar.open or 0.0)
        if close <= 0:
            return False
        if close < open_ * 0.998:
            return False
        if close < price * 0.995:
            return False
        if close > price * 1.015:
            return False

        vwap_level = ExecutionTimer._criteria_float(hit.criteria, "vwap")
        if vwap_level is None or vwap_level <= 0:
            return False
        if close < vwap_level * 1.003:
            return False

        pullback_low = ExecutionTimer._criteria_float(hit.criteria, "pullback_low")
        if pullback_low is not None and pullback_low > 0 and close < pullback_low:
            return False

        score = float(hit.score or 0.0)
        volume = float(hit.criteria.get("volume") or 0.0)
        if score < 12.0 and volume < 100_000:
            return False

        return True

    @staticmethod
    def _allows_pullback_scout(
        pending: PendingEntry,
        latest_bar: Optional[Bar] = None,
    ) -> bool:
        """Permit a small scout for strong pullback bases that never print a clean 10s trigger.

        This is for ANY-style missed pullbacks: the 1-minute pullback/base setup
        already passed entry guard/ML, but the 10s tape stayed too flat to
        release. It remains narrower than normal execution and always reduced
        size.
        """
        signal = pending.signal
        if signal.action is not SignalAction.ENTER_LONG:
            return False
        hit = signal.scan_result
        if hit is None:
            return False

        pattern = str(hit.criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")
        if pattern != "pullback_base" and scanner != "pullback_base":
            return False

        price = float(hit.criteria.get("close") or signal.entry_price or 0.0)
        if not (1.5 <= price <= 20.0):
            return False

        score = float(hit.score or 0.0)
        scout_only = ExecutionTimer._is_pullback_scout_only_signal(signal)
        if score < (50.0 if scout_only else 80.0):
            return False

        pullback_pct = hit.criteria.get("pullback_pct")
        if pullback_pct is not None:
            pullback = float(pullback_pct)
            if pullback < 3.0 or pullback > 12.0:
                return False

        base_range_pct = hit.criteria.get("base_range_pct")
        if base_range_pct is not None and float(base_range_pct) > 8.0:
            return False

        volume_score = ExecutionTimer._pullback_volume_profile_score(signal)
        if volume_score < 50.0:
            return False

        if latest_bar is None:
            return pending.bars_seen > 0

        close = float(latest_bar.close or 0.0)
        open_ = float(latest_bar.open or 0.0)
        high = float(latest_bar.high or max(open_, close))
        low = float(latest_bar.low or min(open_, close))
        if close <= 0:
            return False
        if close < price * 0.99:
            return False
        if close < open_ and high > low:
            red_body = open_ - close
            if red_body / (high - low) > 0.45:
                return False
        return True

    @staticmethod
    def _allows_no_10s_pullback_scout(pending: PendingEntry) -> bool:
        """Allow a small scout when a strong 1m pullback gets no 10s bars.

        This handles MOBX-style cases where the 1-minute pullback/base setup is
        clean, but the micro-entry feed never produces a usable confirmation
        before timeout. It never releases full size and stays limited to strong
        pullback-base structures.
        """
        if pending.last_bar is not None or pending.bars_seen > 0:
            return False
        if pending.saw_red:
            return False

        signal = pending.signal
        if signal.action is not SignalAction.ENTER_LONG:
            return False
        hit = signal.scan_result
        if hit is None:
            return False

        pattern = str(hit.criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")
        if pattern != "pullback_base" and scanner != "pullback_base":
            return False

        price = float(hit.criteria.get("close") or signal.entry_price or 0.0)
        if not (1.5 <= price <= 20.0):
            return False

        if float(hit.score or 0.0) < 100.0:
            return False

        day_move_pct = float(hit.criteria.get("day_move_pct") or 0.0)
        if day_move_pct < 25.0:
            return False

        pullback_pct = hit.criteria.get("pullback_pct")
        if pullback_pct is None:
            return False
        pullback = float(pullback_pct)
        if pullback < 3.0 or pullback > 8.0:
            return False

        base_range_pct = hit.criteria.get("base_range_pct")
        if base_range_pct is None:
            return False
        if float(base_range_pct) > 8.0:
            return False

        volume_score = ExecutionTimer._pullback_volume_profile_score(signal)
        return volume_score >= 60.0

    @staticmethod
    def _is_pullback_scout_only_signal(signal: TradeSignal) -> bool:
        hit = signal.scan_result
        if hit is None:
            return False
        return str(hit.criteria.get("entry_tier", "")) == "pullback_scout"

    @staticmethod
    def _pullback_volume_profile_score(signal: TradeSignal) -> float:
        """Score 1-minute pullback volume shape instead of raw latest volume.

        Good pullback entries usually show buying pressure on the impulse,
        quieter red/base candles during the pullback, and volume returning on
        the reclaim. This allows lower absolute-volume names to participate
        when the relative pattern is strong, while rejecting heavy red selling.
        """
        hit = signal.scan_result
        if hit is None:
            return 0.0
        bars = list(hit.bars or [])
        if len(bars) < 8:
            latest_volume = float(hit.criteria.get("volume") or 0.0)
            if latest_volume >= 1_000_000:
                return 65.0
            if latest_volume >= 500_000:
                return 50.0
            return 0.0

        latest = bars[-1]
        lookback = bars[-18:] if len(bars) >= 18 else bars
        base = lookback[-5:-1] if len(lookback) >= 6 else lookback[:-1]
        impulse_candidates = [
            b for b in lookback[:-1]
            if b.close > b.open and float(b.volume or 0.0) > 0
        ]
        if not impulse_candidates:
            return 0.0
        impulse = max(impulse_candidates, key=lambda b: float(b.volume or 0.0))
        impulse_vol = float(impulse.volume or 0.0)
        if impulse_vol <= 0:
            return 0.0

        pullback_red = [b for b in base if b.close < b.open and float(b.volume or 0.0) > 0]
        pullback_sample = pullback_red or [b for b in base if float(b.volume or 0.0) > 0]
        pullback_avg = (
            sum(float(b.volume or 0.0) for b in pullback_sample) / len(pullback_sample)
            if pullback_sample else 0.0
        )
        latest_vol = float(latest.volume or hit.criteria.get("volume") or 0.0)
        score = 0.0

        prior = [b for b in lookback[:-6] if float(b.volume or 0.0) > 0]
        prior_avg = (
            sum(float(b.volume or 0.0) for b in prior) / len(prior)
            if prior else 0.0
        )

        if impulse_vol >= 75_000 or (prior_avg > 0 and impulse_vol >= prior_avg * 2.0):
            score += 25.0
        elif impulse_vol >= 25_000 or (prior_avg > 0 and impulse_vol >= prior_avg * 1.4):
            score += 15.0

        if pullback_avg > 0:
            dry_ratio = pullback_avg / impulse_vol
            if dry_ratio <= 0.55:
                score += 25.0
            elif dry_ratio <= 0.75:
                score += 20.0
            elif dry_ratio <= 0.95:
                score += 10.0

        if latest.close > latest.open:
            if pullback_avg > 0 and latest_vol >= pullback_avg * 1.2:
                score += 25.0
            elif pullback_avg > 0 and latest_vol >= pullback_avg:
                score += 15.0
            elif latest_vol >= 50_000:
                score += 10.0

        if prior_avg > 0 and latest_vol >= prior_avg * 1.5:
            score += 10.0

        heavy_red = False
        for b in pullback_red:
            if float(b.volume or 0.0) >= impulse_vol * 0.90:
                heavy_red = True
                break
            if b.high > b.low:
                red_body = float(b.open - b.close)
                red_range = float(b.high - b.low)
                if red_range > 0 and red_body / red_range >= 0.65 and float(b.volume or 0.0) >= impulse_vol * 0.70:
                    heavy_red = True
                    break
        if heavy_red:
            score -= 35.0

        return max(0.0, min(score, 100.0))

    @staticmethod
    def _requires_pullback_reclaim(signal: TradeSignal) -> bool:
        if signal.action not in (
            SignalAction.ENTER_LONG,
            SignalAction.ENTER_SHORT,
            SignalAction.SCALE_UP_LONG,
            SignalAction.SCALE_UP_SHORT,
            SignalAction.REENTER_LONG,
            SignalAction.REENTER_SHORT,
        ):
            return False
        hit = signal.scan_result
        if hit is None:
            return False

        pattern = str(hit.criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")
        if ExecutionTimer._is_strong_pullback_base_signal(signal):
            return False
        pullback_patterns = {
            "abc_continuation",
            "breakout_scalp",
            "bull_flag",
            "flat_top_breakout",
            "hod_reclaim",
            "level_breakout_reclaim",
            "opening_range_breakout",
            "pullback_base",
            "runner_readd",
            "vwap_pullback",
            "shallow_stair_continuation",
            "early_vwap_reclaim_scout",
        }
        return pattern in pullback_patterns or scanner in pullback_patterns

    @staticmethod
    def _is_strong_pullback_base_signal(signal: TradeSignal) -> bool:
        hit = signal.scan_result
        if hit is None:
            return False
        pattern = str(hit.criteria.get("pattern", ""))
        scanner = str(hit.scanner_name or "")
        if pattern != "pullback_base" and scanner != "pullback_base":
            return False
        return float(hit.score or 0.0) >= 80.0

    @staticmethod
    def _hot_momentum_wait_seconds(signal: TradeSignal) -> Optional[float]:
        """Shorten the fallback wait for CMND-style HOD squeezes.

        Normal signals still wait for the usual 10s-bar confirmation. Very hot
        HOD/vwap reclaim setups can move without sending usable 10s bars, so
        waiting the full safety timeout often turns a good alert into a chase.
        """
        if signal.action not in (
            SignalAction.ENTER_LONG,
            SignalAction.ENTER_SHORT,
            SignalAction.SCALE_UP_LONG,
            SignalAction.SCALE_UP_SHORT,
            SignalAction.REENTER_LONG,
            SignalAction.REENTER_SHORT,
        ):
            return None
        hit = signal.scan_result
        if hit is None:
            return None

        pattern = str(hit.criteria.get("pattern", ""))
        scanner = hit.scanner_name
        if ExecutionTimer._is_strong_pullback_base_signal(signal):
            return 10.0

        hot_patterns = {
            "hod_reclaim",
            "level_breakout_reclaim",
            "vwap_pullback",
            "early_vwap_reclaim_scout",
            "breakout_scalp",
            "momentum_burst",
            "abc_continuation",
            "first_pullback_reclaim",
            "shallow_stair_continuation",
            "runner_readd",
        }
        if pattern not in hot_patterns:
            if scanner not in hot_patterns:
                return None

        price = float(hit.criteria.get("close") or signal.entry_price or 0.0)
        if not (1.5 <= price <= 20.0):
            return None

        rally_pct = float(
            hit.criteria.get("rally_pct")
            or hit.criteria.get("change_session_pct")
            or 0.0
        )
        latest_volume = float(hit.criteria.get("volume") or 0.0)
        if pattern == "breakout_scalp" or scanner == "breakout_scalp":
            return 8.0
        if pattern == "first_pullback_reclaim" or scanner == "first_pullback_reclaim":
            return 10.0
        if pattern in ("momentum_burst", "abc_continuation") or scanner in (
            "momentum_burst", "abc_continuation",
        ):
            return 10.0
        if rally_pct >= 25.0 and latest_volume >= 100_000:
            return 12.0
        return None
