"""Smart exit manager — Warrior Trading exit rules + stepping stop.

Exit Indicators (Warrior Trading Momentum Strategy):

  Exit #1 — Half-position sell at first profit target:
    Sell 1/2 when unrealized profit hits 2:1 reward-to-risk.
    Move stop to breakeven on the remaining half.

  Exit #2 — First red candle exit (full position only):
    If still holding the full position (no partial yet), the first candle
    that closes red is an exit signal.  If already sold 1/2, hold through
    red candles as long as the breakeven stop doesn't hit.

  Exit #3 — Extension bar exit (sell into the spike):
    If a single candle produces an unrealized gain of $2-4+ per share,
    sell into the spike before the reversal.

  Also retains the stepping stop, range-bound exit, halt detection,
  and stale exit from the original system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from daytrading.models import ExitReason, Side, SignalAction, TradeSignal

logger = logging.getLogger(__name__)

TICK = 0.01
STEP_PCT = 0.01            # 1% of entry price — lock breakeven quickly to protect capital
STEP_PCT_AFTER_HALF = 0.04  # 4% steps after selling half (let winner run)
MOMENTUM_THRESHOLD = 7  # ticks moved from entry to be considered "high momentum"


@dataclass
class ExitTier:
    """Kept for backward compatibility with synced positions."""
    shares: float
    target_price: Optional[float] = None
    trail_cents: Optional[float] = None
    filled: bool = False


@dataclass
class TrackedPosition:
    """An open position being monitored with the stepping stop system."""

    symbol: str
    side: Side
    quantity: float
    remaining_qty: float = 0.0
    entry_price: float = 0.0
    entry_ts: Optional[datetime] = None
    stop_loss: Optional[float] = None
    max_hold_seconds: Optional[int] = None
    reason: str = ""

    tiers: List[ExitTier] = field(default_factory=list)

    # Stepping stop state
    current_step: int = 0           # how many step levels have been locked
    step_pct: float = STEP_PCT      # percentage of entry price per step
    _step_adjusted: bool = False    # whether step size has been finalized

    # Watermarks
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    breakeven_locked: bool = False

    # Warrior Trading exit state
    sold_half: bool = False         # True after Exit #1 partial sell
    original_qty: float = 0.0      # full position size at entry
    risk_per_share: float = 0.0    # entry_price - stop_loss (set at entry)
    first_target_price: float = 0.0  # entry + 2 * risk_per_share
    last_bar_close: float = 0.0    # previous bar close for red candle detection
    extension_threshold: float = 0.15  # 15% gain triggers extension exit
    entry_time: Optional[datetime] = None  # when the position was opened
    trend_strength: float = 0.5           # 0-1, from classifier (higher = stronger trend)
    consecutive_red: int = 0              # count of consecutive red candles
    prev_bar_open: float = 0.0           # previous bar open for red candle check
    _first_pullback_done: bool = False   # True after first red candle (grace period)

    # Volume tracking for momentum-aware red candle exits
    last_bar_volume: int = 0             # volume of the most recent completed bar
    prev_bar_volume: int = 0             # volume of the bar before that
    avg_bar_volume: float = 0.0          # running average volume of recent bars
    _vol_history: List[int] = field(default_factory=list)
    _max_vol_history: int = 10
    consecutive_green_declining_vol: int = 0  # streak of green bars with declining volume

    # Range-bound detection: exit if price chops in a tight range after step-up
    _range_high: float = 0.0
    _range_low: float = float("inf")
    _range_start_ts: Optional[datetime] = None
    _range_pct: float = 0.01             # max 1% range width before considered "ranging"
    _range_timeout_secs: float = 120  # exit after 120s of ranging

    # Halt detection: exit if no price update for too long
    _last_price_update: Optional[datetime] = None
    _halt_timeout_secs: float = 60   # assume halted if no update for 60s

    _recent_prices: List[float] = field(default_factory=list)
    _max_recent: int = 10

    def __post_init__(self) -> None:
        self.remaining_qty = self.remaining_qty or self.quantity
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
        self.original_qty = self.original_qty or self.quantity
        if self.stop_loss and self.entry_price > 0 and self.risk_per_share == 0:
            self.risk_per_share = abs(self.entry_price - self.stop_loss)
        if self.risk_per_share > 0 and self.first_target_price == 0:
            self.first_target_price = self.entry_price + self.risk_per_share * 1.0

    def record_price(self, price: float, now: Optional[datetime] = None) -> None:
        self._recent_prices.append(price)
        if len(self._recent_prices) > self._max_recent:
            self._recent_prices.pop(0)
        self._last_price_update = now or datetime.now(timezone.utc)

    @property
    def momentum_factor(self) -> float:
        prices = self._recent_prices
        if len(prices) < 3:
            return 1.0
        is_long = self.side is Side.BUY
        recent_half = prices[len(prices) // 2:]
        early_half = prices[: len(prices) // 2]
        if not recent_half or not early_half:
            return 1.0
        recent_move = recent_half[-1] - recent_half[0]
        early_move = early_half[-1] - early_half[0]
        if not is_long:
            recent_move = -recent_move
            early_move = -early_move
        if abs(early_move) < 0.001:
            return 1.5 if recent_move > 0 else 0.7
        ratio = recent_move / abs(early_move)
        return max(0.5, min(2.0, ratio))


def build_exit_tiers(
    quantity: float,
    entry_price: float,
    side: Side,
    *,
    stop_loss: Optional[float] = None,
    **kwargs,
) -> List[ExitTier]:
    """No tiers needed for stepping stop — returns empty list."""
    return []


class ExitManager:
    """Stepping stop exit manager.

    Instead of tiered exits, all shares are held together.
    The stop ratchets up every time price gains another 10 ticks.
    """

    def __init__(self) -> None:
        self._positions: Dict[str, TrackedPosition] = {}

    @property
    def tracked(self) -> Dict[str, TrackedPosition]:
        return dict(self._positions)

    def track(self, pos: TrackedPosition) -> None:
        if pos.entry_time is None:
            pos.entry_time = datetime.now(timezone.utc)
        # Initialize range tracking from entry
        pos._range_high = pos.entry_price
        pos._range_low = pos.entry_price
        pos._range_start_ts = datetime.now(timezone.utc)
        self._positions[pos.symbol] = pos
        target = pos.first_target_price if pos.first_target_price > 0 else pos.entry_price + 0.20
        logger.info(
            "Tracking %s %s %.0f @ %.4f | SL=%.4f → 1:1 target=%.4f (risk=$%.2f, trend=%.2f)",
            pos.side.value, pos.symbol, pos.quantity, pos.entry_price,
            pos.stop_loss or 0, target, pos.risk_per_share, pos.trend_strength,
        )

    def untrack(self, symbol: str) -> None:
        self._positions.pop(symbol, None)

    def update_bar_close(self, symbol: str, close_price: float, open_price: float = 0.0, volume: int = 0) -> None:
        """Called when a new 1-min bar closes. Updates red candle + volume tracking."""
        pos = self._positions.get(symbol)
        if pos is not None:
            if pos.last_bar_close > 0 and close_price < pos.last_bar_close:
                pos.consecutive_red += 1
            else:
                pos.consecutive_red = 0
            pos.prev_bar_open = open_price if open_price > 0 else pos.last_bar_close
            pos.prev_bar_volume = pos.last_bar_volume
            pos.last_bar_volume = volume
            pos.last_bar_close = close_price

            # Track green bars with declining volume (exhaustion detection)
            is_green = close_price > open_price if open_price > 0 else close_price > pos.prev_bar_open
            if is_green and pos.prev_bar_volume > 0 and volume < pos.prev_bar_volume:
                pos.consecutive_green_declining_vol += 1
            else:
                pos.consecutive_green_declining_vol = 0

            if volume > 0:
                pos._vol_history.append(volume)
                if len(pos._vol_history) > pos._max_vol_history:
                    pos._vol_history.pop(0)
                pos.avg_bar_volume = sum(pos._vol_history) / len(pos._vol_history)

    def scale_up(
        self,
        symbol: str,
        add_qty: float,
        add_price: float,
        new_stop: Optional[float] = None,
    ) -> None:
        pos = self._positions.get(symbol)
        if pos is None:
            return
        old_cost = pos.entry_price * pos.remaining_qty
        new_cost = add_price * add_qty
        pos.remaining_qty += add_qty
        pos.quantity += add_qty
        pos.entry_price = (old_cost + new_cost) / pos.remaining_qty
        if new_stop is not None:
            pos.stop_loss = new_stop
        logger.info(
            "SCALED UP %s +%.0f @ %.4f → avg=%.4f, total=%.0f, stop=%.4f",
            symbol, add_qty, add_price, pos.entry_price, pos.remaining_qty,
            pos.stop_loss or 0,
        )

    def register_from_signal(
        self, signal: TradeSignal, ts: datetime, fill_price: float = 0.0,
    ) -> None:
        """Convert a filled TradeSignal into a tracked position.
        
        Uses fill_price (actual execution price) if provided, otherwise
        falls back to signal.entry_price.
        """
        side = Side.BUY if signal.action is SignalAction.ENTER_LONG else Side.SELL
        actual_price = fill_price if fill_price > 0 else signal.entry_price

        # Use the signal's stop loss (set by pattern scanner) — don't override
        stop_loss = signal.stop_loss
        if stop_loss is None or stop_loss <= 0:
            if side is Side.BUY:
                stop_loss = actual_price * 0.98
            else:
                stop_loss = actual_price * 1.02

        # If slippage made the risk too tight, widen stop to maintain
        # a minimum risk of 2% of entry price (same as dynamic risk calc).
        # Example: signal $5.36, stop $5.25 (risk $0.11), but filled at $5.27
        # → effective risk $0.02 which is way too tight. Recalc to 2% = $0.105.
        if side is Side.BUY and actual_price > 0 and stop_loss > 0:
            effective_risk = actual_price - stop_loss
            min_risk = max(0.05, actual_price * 0.03)
            if effective_risk < min_risk:
                old_stop = stop_loss
                stop_loss = round(actual_price - min_risk, 4)
                logger.warning(
                    "STOP ADJUST %s: slippage shrank risk to $%.2f "
                    "(entry %.2f, old stop %.2f) → new stop %.2f (risk $%.2f)",
                    signal.symbol, effective_risk, actual_price, old_stop,
                    stop_loss, min_risk,
                )

        self.track(TrackedPosition(
            symbol=signal.symbol,
            side=side,
            quantity=signal.quantity,
            remaining_qty=signal.quantity,
            entry_price=actual_price,
            entry_ts=ts,
            stop_loss=stop_loss,
            max_hold_seconds=signal.max_hold_seconds,
            reason=signal.reason,
            trend_strength=signal.trend_strength,
        ))

    def check_exits(
        self,
        prices: Dict[str, float],
        now: datetime,
    ) -> List[TradeSignal]:
        """Check all tracked positions for stop loss or step-up."""
        exits: List[TradeSignal] = []

        for symbol, pos in list(self._positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue

            pos.record_price(price, now=now)
            if price > pos.highest_price:
                pos.highest_price = price
            if price < pos.lowest_price:
                pos.lowest_price = price

            signals = self._check_position(pos, price, now)
            exits.extend(signals)

            if pos.remaining_qty <= 0:
                self.untrack(symbol)

        return exits

    def _check_position(
        self,
        pos: TrackedPosition,
        price: float,
        now: datetime,
    ) -> List[TradeSignal]:
        is_long = pos.side is Side.BUY

        # --- dynamic step sizing based on momentum and trend strength ---
        if not pos._step_adjusted and pos.current_step == 0:
            if is_long:
                move_pct = (pos.highest_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
            else:
                move_pct = (pos.entry_price - pos.lowest_price) / pos.entry_price if pos.entry_price > 0 else 0

            # Strong trends get wider steps: let pullbacks play out
            trend_step_bonus = 0.01 if pos.trend_strength >= 0.8 else (0.005 if pos.trend_strength >= 0.6 else 0.0)

            if move_pct >= 0.03:  # 3% move = high momentum
                pos.step_pct = 0.03 + trend_step_bonus
                pos._step_adjusted = True
                logger.info(
                    "MOMENTUM HIGH %s → step size %.1f%% (moved %.1f%%, trend=%.2f)",
                    pos.symbol, pos.step_pct * 100, move_pct * 100, pos.trend_strength,
                )

        # --- hard stop loss: exit all shares ---
        if pos.stop_loss is not None:
            if (is_long and price <= pos.stop_loss) or \
               (not is_long and price >= pos.stop_loss):
                reason = ExitReason.STOP_LOSS if pos.current_step == 0 and not pos.sold_half else ExitReason.TRAILING_STOP
                logger.info(
                    "STOP HIT %s @ %.4f (stop=%.4f, step=%d, entry=%.4f, sold_half=%s)",
                    pos.symbol, price, pos.stop_loss, pos.current_step, pos.entry_price, pos.sold_half,
                )
                sig = self._make_exit(pos, pos.remaining_qty, price, reason)
                pos.remaining_qty = 0
                return [sig]

        # --- Exit #3: Extension bar — sell into spike (15%+ gain) ---
        if is_long and pos.remaining_qty > 0:
            gain_pct = (price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
            if gain_pct >= pos.extension_threshold:
                logger.info(
                    "EXTENSION BAR %s @ %.4f | gain %.1f%% — selling into spike (entry=%.4f)",
                    pos.symbol, price, gain_pct * 100, pos.entry_price,
                )
                sig = self._make_exit(pos, pos.remaining_qty, price, ExitReason.TAKE_PROFIT)
                pos.remaining_qty = 0
                return [sig]

        # --- Exit #1: Sell half at first profit target (2:1 R:R) ---
        if is_long and not pos.sold_half and pos.first_target_price > 0:
            if price >= pos.first_target_price:
                half_qty = max(1, int(pos.remaining_qty / 2))
                logger.info(
                    "HALF SELL %s @ %.4f | hit 1:1 target %.4f | selling %d of %d shares, moving stop to breakeven %.4f",
                    pos.symbol, price, pos.first_target_price, half_qty, int(pos.remaining_qty), pos.entry_price,
                )
                pos.sold_half = True
                pos.remaining_qty -= half_qty
                pos.stop_loss = pos.entry_price  # breakeven stop on remaining
                pos.breakeven_locked = True
                sig = self._make_exit(pos, half_qty, price, ExitReason.TAKE_PROFIT)
                return [sig]

        # --- Exit #2: Red candle exit (full position only) ---
        # MOMENTUM APPROACH: On momentum stocks, one red candle is normal.
        # RED CANDLE ANALYSIS — only for pre-half positions.
        # The hard stop already handles the "I'm wrong" scenario.
        # This is for recognizing distribution (real selling) before the
        # stop is hit, saving some money.  We do NOT exit on every tiny
        # dip — that's what the stop is for.
        #
        # Requirements to trigger:
        #   1. Hold time >= 120s (give the trade 2 minutes to develop)
        #   2. Either heavy selling (high volume + drop) OR sustained fade (3+ red bars)
        if is_long and not pos.sold_half and pos.last_bar_close > 0:
            drop_pct = (pos.last_bar_close - price) / pos.last_bar_close if pos.last_bar_close > 0 else 0
            hold_secs = (now - pos.entry_ts).total_seconds() if pos.entry_ts else 0

            should_exit = False
            exit_detail = ""

            vol_ratio = 0.0
            if pos.avg_bar_volume > 0 and pos.last_bar_volume > 0:
                vol_ratio = pos.last_bar_volume / pos.avg_bar_volume

            is_red = drop_pct > 0
            high_vol_red = is_red and vol_ratio >= 1.5
            low_vol_red = is_red and vol_ratio < 0.8

            if hold_secs >= 120:
                if high_vol_red and drop_pct >= 0.01:
                    should_exit = True
                    exit_detail = f"distribution: high-vol red (vol {vol_ratio:.1f}x avg, drop {drop_pct*100:.1f}%, held {hold_secs:.0f}s)"
                elif pos.consecutive_red >= 3 and not low_vol_red and drop_pct >= 0.005:
                    should_exit = True
                    exit_detail = f"sustained fade: {pos.consecutive_red} red candles, drop {drop_pct*100:.1f}%, held {hold_secs:.0f}s"
            elif low_vol_red:
                logger.debug(
                    "HOLD THROUGH PULLBACK %s @ %.4f | low-vol red (vol %.1fx avg), still developing (%.0fs)",
                    pos.symbol, price, vol_ratio, hold_secs,
                )

            if should_exit:
                logger.info(
                    "RED CANDLE EXIT %s @ %.4f | %s (entry=%.4f, last_close=%.4f, vol=%d, avg_vol=%.0f, ratio=%.1fx)",
                    pos.symbol, price, exit_detail, pos.entry_price, pos.last_bar_close,
                    pos.last_bar_volume, pos.avg_bar_volume, vol_ratio,
                )
                sig = self._make_exit(pos, pos.remaining_qty, price, ExitReason.STOP_LOSS)
                pos.remaining_qty = 0
                return [sig]

        # --- Volume exhaustion exit: green bars with declining volume → momentum fading ---
        if is_long and pos.remaining_qty > 0 and pos.consecutive_green_declining_vol >= 3:
            hold_secs = (now - pos.entry_ts).total_seconds() if pos.entry_ts else 0
            if hold_secs >= 120:
                gain_pct = (price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
                if gain_pct > 0.005:
                    logger.info(
                        "VOLUME EXHAUSTION EXIT %s @ %.4f | %d green bars w/ declining vol, gain %.1f%%, held %.0fs (entry=%.4f)",
                        pos.symbol, price, pos.consecutive_green_declining_vol,
                        gain_pct * 100, hold_secs, pos.entry_price,
                    )
                    sig = self._make_exit(pos, pos.remaining_qty, price, ExitReason.TAKE_PROFIT)
                    pos.remaining_qty = 0
                    return [sig]

        # --- stepping stop: check if price crossed the next step level ---
        # BEFORE selling half: only step up to BREAKEVEN (entry price) to protect
        # the position. Don't aggressively lock profits — let price reach the
        # 2:1 target so the half-sell can trigger first.
        # AFTER selling half: use wider steps (4%) and let the winner run.
        if not pos.sold_half:
            # Pre-half: only move stop to breakeven once price moves 2% above entry
            if is_long and not pos.breakeven_locked:
                breakeven_trigger = pos.entry_price * (1 + STEP_PCT)
                if price >= breakeven_trigger:
                    pos.stop_loss = pos.entry_price
                    pos.breakeven_locked = True
                    logger.info(
                        "BREAKEVEN LOCK %s | price %.4f hit +%.1f%% → stop moved to entry %.4f (waiting for 1:1 target %.4f)",
                        pos.symbol, price, STEP_PCT * 100, pos.entry_price,
                        pos.first_target_price,
                    )

            # Trailing stop for no-target positions (adopted after restart):
            # once breakeven is locked, trail 10 ticks behind the highest price.
            if pos.breakeven_locked and pos.first_target_price == 0:
                TRAIL_TICKS = 10
                trail_offset = TRAIL_TICKS * TICK
                if is_long:
                    trail_stop = round(pos.highest_price - trail_offset, 4)
                    if trail_stop > (pos.stop_loss or 0):
                        pos.stop_loss = trail_stop
                else:
                    trail_stop = round(pos.lowest_price + trail_offset, 4)
                    if trail_stop < (pos.stop_loss or float("inf")):
                        pos.stop_loss = trail_stop
        else:
            # Post-half: step up in 4% increments of entry price to let winner run
            step_amount = pos.entry_price * STEP_PCT_AFTER_HALF
            next_step = pos.current_step + 1
            step_dist = next_step * step_amount
            stepped_up = False

            if is_long:
                target_level = pos.entry_price + step_dist
                if price >= target_level:
                    new_stop = target_level
                    pos.stop_loss = new_stop
                    pos.current_step = next_step
                    stepped_up = True

                    next_next = pos.entry_price + (next_step + 1) * step_amount
                    logger.info(
                        "STEP UP %s → step %d | stop locked at %.4f (+%.1f%%) | next target %.4f (+%.1f%%)",
                        pos.symbol, pos.current_step,
                        new_stop, (new_stop - pos.entry_price) / pos.entry_price * 100,
                        next_next, (next_next - pos.entry_price) / pos.entry_price * 100,
                    )
            else:
                target_level = pos.entry_price - step_dist
                if price <= target_level:
                    new_stop = target_level
                    pos.stop_loss = new_stop
                    pos.current_step = next_step
                    stepped_up = True

                    next_next = pos.entry_price - (next_step + 1) * step_amount
                    logger.info(
                        "STEP UP %s → step %d | stop locked at %.4f (-%.1f%%) | next target %.4f (-%.1f%%)",
                        pos.symbol, pos.current_step,
                        new_stop, (pos.entry_price - new_stop) / pos.entry_price * 100,
                        next_next, (pos.entry_price - next_next) / pos.entry_price * 100,
                    )

            if stepped_up:
                pos._range_high = price
                pos._range_low = price
                pos._range_start_ts = now
                return self._check_position(pos, price, now)

        # --- halt detection: no price update for 60s → exit immediately ---
        if pos._last_price_update is not None:
            silence = (now - pos._last_price_update).total_seconds()
            if silence >= pos._halt_timeout_secs:
                logger.warning(
                    "HALT DETECTED %s — no price update for %.0fs, exiting at last price %.4f (entry=%.4f)",
                    pos.symbol, silence, price, pos.entry_price,
                )
                reason = ExitReason.TAKE_PROFIT if price > pos.entry_price else ExitReason.STOP_LOSS
                sig = self._make_exit(pos, pos.remaining_qty, price, reason)
                pos.remaining_qty = 0
                return [sig]

        # --- stale exit: no movement toward target within timeout → exit ---
        # Applies at ANY stage: before half-sell OR after half-sell.
        # If the stock is just sitting there not reaching the target, get out.
        if pos.trend_strength >= 0.8:
            stale_timeout = 300  # 5 minutes for strong trends
        elif pos.trend_strength >= 0.6:
            stale_timeout = 240  # 4 minutes for moderate trends
        else:
            stale_timeout = 180  # 3 minutes for weak/no trend

        if pos.entry_ts is not None:
            hold_secs = (now - pos.entry_ts).total_seconds()
            # Before half-sell: exit if no progress toward 2:1 target
            if not pos.sold_half and not pos.breakeven_locked and hold_secs >= stale_timeout:
                # Guard: skip stale exit if currently in profit (trade is working, just slow)
                if price > pos.entry_price:
                    pass
                # Guard: extend timeout 50% if price came close to breakeven
                elif pos.highest_price >= pos.entry_price * (1 + STEP_PCT * 0.7) and hold_secs < stale_timeout * 1.5:
                    pass
                else:
                    logger.info(
                        "STALE EXIT %s @ %.4f | held %.0fs, never reached breakeven (timeout=%ds, trend=%.2f, entry=%.4f)",
                        pos.symbol, price, hold_secs, stale_timeout, pos.trend_strength, pos.entry_price,
                    )
                    reason = ExitReason.TAKE_PROFIT if price > pos.entry_price else ExitReason.STOP_LOSS
                    sig = self._make_exit(pos, pos.remaining_qty, price, reason)
                    pos.remaining_qty = 0
                    return [sig]

        # --- range-bound exit: if price chops in tight range ---
        # Works at ANY stage (before or after half-sell).
        # If in a tight range for too long, the momentum is dead — exit.
        range_timeout = pos._range_timeout_secs
        if pos.trend_strength >= 0.8:
            range_timeout = 180
        elif pos.trend_strength >= 0.6:
            range_timeout = 150

        if pos._range_start_ts is not None:
            pos._range_high = max(pos._range_high, price)
            pos._range_low = min(pos._range_low, price)
            range_width = pos._range_high - pos._range_low

            if pos.entry_price > 0 and range_width <= pos.entry_price * pos._range_pct:
                elapsed = (now - pos._range_start_ts).total_seconds()
                if elapsed >= range_timeout:
                    logger.info(
                        "RANGE EXIT %s @ %.4f | stuck %.0fs in %.4f–%.4f (%d tick range), sold_half=%s",
                        pos.symbol, price, elapsed,
                        pos._range_low, pos._range_high,
                        int(range_width / TICK), pos.sold_half,
                    )
                    sig = self._make_exit(pos, pos.remaining_qty, price, ExitReason.TAKE_PROFIT)
                    pos.remaining_qty = 0
                    return [sig]
            else:
                pos._range_high = price
                pos._range_low = price
                pos._range_start_ts = now

        return []

    def _make_exit(
        self,
        pos: TrackedPosition,
        qty: float,
        price: float,
        reason: ExitReason,
    ) -> TradeSignal:
        action = SignalAction.EXIT_LONG if pos.side is Side.BUY else SignalAction.EXIT_SHORT
        return TradeSignal(
            symbol=pos.symbol,
            action=action,
            quantity=qty,
            entry_price=price,
            reason="{}: {}".format(reason.value, pos.reason),
        )
