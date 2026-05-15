"""Position scaler — adds shares to winning trades + re-enters after exits.

Two strategies:

SCALE UP (pyramiding):
  You're already in a trade and it's working. Instead of just holding,
  you add more shares at confirmed pullback levels. Rules:

  1. Only scale up into WINNERS — position must be profitable
  2. Each add-on is SMALLER than the previous (pyramid shape)
  3. Stop loss moves up on each add — risk never increases
  4. Maximum 3 scale-ups per trade
  5. Must wait for a pullback + bounce (don't chase)

  Example on a $5 stock:
    Entry:     500 shares @ $5.00   (stop $4.97)
    Scale 1:   250 shares @ $5.08   (stop → $5.03) — half the size
    Scale 2:   125 shares @ $5.18   (stop → $5.13) — half again
    Now holding 875 shares with an average cost of ~$5.06
    and the stock is at $5.20. If it hits $5.50, that's 875 × $0.44 = $386

RE-ENTRY:
  You exited fully (all tiers done), but the stock is STILL RUNNING.
  The re-entry detector watches recently exited symbols for:

  1. Pullback after the exit (proves the exit was right at that moment)
  2. New bounce from the pullback (move is resuming)
  3. Volume still elevated (the move isn't dead)

  If all 3 pass → re-enter with a smaller size and tighter stops.
  Cooldown prevents re-entering the same stock too fast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

from daytrading.exits.manager import ExitManager, TrackedPosition, build_exit_tiers
from daytrading.models import Bar, ExitReason, Side, SignalAction, TradeSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scale-up (pyramiding)
# ---------------------------------------------------------------------------

@dataclass
class ScaleUpConfig:
    """Rules for adding to a winning position."""
    max_scale_ups: int = 3
    size_decay: float = 0.5           # each add-on is this × previous size
    min_profit_cents: float = 5.0     # position must be up ≥ this to consider
    pullback_pct: float = 0.3         # price must pull back ≥ 30% of the move
    bounce_pct: float = 0.5           # then bounce back ≥ 50% of the pullback
    stop_advance_cents: float = 3.0   # move stop up by this much on each scale


class PositionScaler:
    """Monitors winning positions and generates scale-up signals."""

    def __init__(self, config: Optional[ScaleUpConfig] = None) -> None:
        self._config = config or ScaleUpConfig()
        self._scale_counts: Dict[str, int] = {}  # symbol → how many times scaled
        self._original_qty: Dict[str, float] = {}  # symbol → initial entry size
        self._last_scale_price: Dict[str, float] = {}
        self._pullback_seen: Dict[str, bool] = {}
        self._pullback_low: Dict[str, float] = {}  # lowest price during pullback

    def check_scale_ups(
        self,
        exit_manager: ExitManager,
        prices: Dict[str, float],
        bars: Dict[str, Sequence[Bar]],
    ) -> List[TradeSignal]:
        """Check all tracked positions for scale-up opportunities."""
        signals: List[TradeSignal] = []
        cfg = self._config

        for symbol, pos in exit_manager.tracked.items():
            price = prices.get(symbol)
            if price is None:
                continue

            count = self._scale_counts.get(symbol, 0)
            if count >= cfg.max_scale_ups:
                continue

            is_long = pos.side is Side.BUY
            profit_cents = (price - pos.entry_price) * 100.0 if is_long else (pos.entry_price - price) * 100.0

            if profit_cents < cfg.min_profit_cents:
                self._pullback_seen[symbol] = False
                continue

            high_water = pos.highest_price if is_long else pos.lowest_price
            last_scale = self._last_scale_price.get(symbol, pos.entry_price)
            move_from_last = abs(price - last_scale)

            # detect pullback
            if is_long:
                if price < high_water * (1.0 - cfg.pullback_pct / 100.0):
                    self._pullback_seen[symbol] = True
                    pb_low = self._pullback_low.get(symbol, price)
                    self._pullback_low[symbol] = min(pb_low, price)
            else:
                if price > pos.lowest_price * (1.0 + cfg.pullback_pct / 100.0):
                    self._pullback_seen[symbol] = True
                    pb_high = self._pullback_low.get(symbol, price)
                    self._pullback_low[symbol] = max(pb_high, price)

            if not self._pullback_seen.get(symbol, False):
                continue

            # detect bounce from pullback
            bounced = False
            pb_extreme = self._pullback_low.get(symbol)
            if pb_extreme is not None:
                if is_long:
                    pullback_depth = high_water - pb_extreme
                    recovery = price - pb_extreme
                    bounced = pullback_depth > 0 and (recovery / pullback_depth) >= cfg.bounce_pct
                else:
                    pullback_depth = pb_extreme - pos.lowest_price
                    recovery = pb_extreme - price
                    bounced = pullback_depth > 0 and (recovery / pullback_depth) >= cfg.bounce_pct

            if not bounced:
                continue

            # volume check: latest bar should have decent volume
            sym_bars = bars.get(symbol)
            if sym_bars and len(sym_bars) >= 2:
                if sym_bars[-1].volume < sym_bars[-2].volume * 0.6:
                    continue  # volume dying, don't scale up

            # calculate scale-up size (pyramid: each smaller than original entry)
            if symbol not in self._original_qty:
                self._original_qty[symbol] = pos.quantity
            base_size = self._original_qty[symbol]
            scale_size = round(base_size * (cfg.size_decay ** (count + 1)))
            if scale_size < 1:
                continue

            # new stop: advance it
            stop_advance = cfg.stop_advance_cents / 100.0
            if is_long:
                new_stop = (pos.stop_loss or pos.entry_price) + stop_advance
                action = SignalAction.SCALE_UP_LONG
            else:
                new_stop = (pos.stop_loss or pos.entry_price) - stop_advance
                action = SignalAction.SCALE_UP_SHORT

            signals.append(TradeSignal(
                symbol=symbol,
                action=action,
                quantity=scale_size,
                entry_price=price,
                stop_loss=new_stop,
                reason=f"Scale #{count+1}: +{scale_size} shares @ ${price:.2f}, profit={profit_cents:.0f}¢",
            ))

            self._scale_counts[symbol] = count + 1
            self._last_scale_price[symbol] = price
            self._pullback_seen[symbol] = False
            self._pullback_low.pop(symbol, None)

            logger.info(
                "SCALE UP %s %s +%d @ %.4f (scale #%d, profit=%.0f¢)",
                symbol, pos.side.value, scale_size, price, count + 1, profit_cents,
            )

        return signals

    def clear(self, symbol: str) -> None:
        """Reset scale-up tracking when a position is fully closed."""
        self._scale_counts.pop(symbol, None)
        self._original_qty.pop(symbol, None)
        self._last_scale_price.pop(symbol, None)
        self._pullback_seen.pop(symbol, None)
        self._pullback_low.pop(symbol, None)


# ---------------------------------------------------------------------------
# Re-entry after full exit
# ---------------------------------------------------------------------------

@dataclass
class _ExitRecord:
    symbol: str
    side: Side
    exit_price: float
    exit_ts: datetime
    highest_price: float   # how far the trade got before exit
    entry_price: float

@dataclass
class ReentryConfig:
    """Rules for re-entering after a full exit."""
    enabled: bool = True
    cooldown_seconds: float = 30.0     # min time after exit before re-entry
    max_reentries: int = 2             # max re-entries per symbol per session
    reentry_size_pct: float = 0.5      # re-enter with 50% of original size
    min_continuation_cents: float = 3.0  # price must move ≥ 3¢ past exit price
    pullback_max_cents: float = 5.0    # pullback must be ≤ 5¢ (not a reversal)
    stop_cents: float = 3.0
    trail_cents: float = 2.0
    max_hold_seconds: int = 90


class ReentryDetector:
    """Watches recently exited symbols and detects re-entry opportunities."""

    def __init__(self, config: Optional[ReentryConfig] = None) -> None:
        self._config = config or ReentryConfig()
        self._exit_history: Dict[str, List[_ExitRecord]] = {}
        self._reentry_counts: Dict[str, int] = {}
        self._pullback_prices: Dict[str, float] = {}

    def record_full_exit(
        self,
        symbol: str,
        side: Side,
        exit_price: float,
        exit_ts: datetime,
        highest_price: float,
        entry_price: float,
    ) -> None:
        """Record that a position was fully closed — start watching for re-entry."""
        records = self._exit_history.setdefault(symbol, [])
        records.append(_ExitRecord(
            symbol=symbol, side=side, exit_price=exit_price,
            exit_ts=exit_ts, highest_price=highest_price,
            entry_price=entry_price,
        ))
        self._pullback_prices.pop(symbol, None)

    def check_reentries(
        self,
        prices: Dict[str, float],
        bars: Dict[str, Sequence[Bar]],
        now: datetime,
        original_sizes: Dict[str, float],
    ) -> List[TradeSignal]:
        """Check recently exited symbols for re-entry conditions."""
        if not self._config.enabled:
            return []

        signals: List[TradeSignal] = []
        cfg = self._config

        for symbol, records in list(self._exit_history.items()):
            if not records:
                continue

            last_exit = records[-1]
            price = prices.get(symbol)
            if price is None:
                continue

            reentry_count = self._reentry_counts.get(symbol, 0)
            if reentry_count >= cfg.max_reentries:
                continue

            # cooldown check
            elapsed = (now - last_exit.exit_ts).total_seconds()
            if elapsed < cfg.cooldown_seconds:
                continue

            is_long = last_exit.side is Side.BUY
            min_cont = cfg.min_continuation_cents / 100.0
            max_pb = cfg.pullback_max_cents / 100.0

            # has the stock continued past where we exited?
            if is_long:
                continued = price > last_exit.exit_price + min_cont
            else:
                continued = price < last_exit.exit_price - min_cont

            if not continued:
                # track pullback depth to detect bounce later
                pb = self._pullback_prices.get(symbol)
                if is_long:
                    self._pullback_prices[symbol] = min(pb, price) if pb is not None else price
                else:
                    self._pullback_prices[symbol] = max(pb, price) if pb is not None else price
                continue

            # verify pullback happened (not just a straight continuation)
            pb_price = self._pullback_prices.get(symbol)
            if pb_price is not None:
                if is_long:
                    pullback_depth = last_exit.exit_price - pb_price
                else:
                    pullback_depth = pb_price - last_exit.exit_price
                if pullback_depth > max_pb:
                    # too deep — this might be a reversal, not a continuation
                    continue

            # volume check
            sym_bars = bars.get(symbol)
            if sym_bars and len(sym_bars) >= 2:
                if sym_bars[-1].volume < sym_bars[-2].volume * 0.5:
                    continue

            # re-enter with smaller size
            orig_size = original_sizes.get(symbol, 500)
            reentry_size = round(orig_size * cfg.reentry_size_pct)
            if reentry_size < 1:
                continue

            stop_offset = cfg.stop_cents / 100.0
            if is_long:
                action = SignalAction.REENTER_LONG
                stop = price - stop_offset
            else:
                action = SignalAction.REENTER_SHORT
                stop = price + stop_offset

            signals.append(TradeSignal(
                symbol=symbol,
                action=action,
                quantity=reentry_size,
                entry_price=price,
                stop_loss=stop,
                trailing_stop_offset=cfg.trail_cents / 100.0,
                max_hold_seconds=cfg.max_hold_seconds,
                reason=f"Re-entry #{reentry_count+1}: still running past ${last_exit.exit_price:.2f}",
            ))

            self._reentry_counts[symbol] = reentry_count + 1
            self._pullback_prices.pop(symbol, None)

            logger.info(
                "RE-ENTRY %s %s %d @ %.4f (exit was %.4f, re-entry #%d)",
                symbol, action.value, reentry_size, price,
                last_exit.exit_price, reentry_count + 1,
            )

        return signals

    def clear_session(self) -> None:
        """Reset all re-entry tracking for a new session."""
        self._exit_history.clear()
        self._reentry_counts.clear()
        self._pullback_prices.clear()
