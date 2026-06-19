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
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from daytrading.models import ExitReason, Side, SignalAction, TradeSignal

logger = logging.getLogger(__name__)

TICK = 0.01
STEP_PCT = 0.01            # 1% of entry price — lock breakeven quickly to protect capital
STEP_PCT_AFTER_HALF = 0.04  # 4% steps after selling half (let winner run)
MOMENTUM_THRESHOLD = 7  # ticks moved from entry to be considered "high momentum"
QUICK_SCALP_PARTIAL_PCT = 0.01  # harvest first quick scalp pop before it fades
RUNNER_MIN_CONFIRM_PCT = 0.018  # prove strength before giving the back half room
RUNNER_TRAIL_PCT = 0.03         # protected runners trail 3% instead of 1%

# Strategy labels that share the hit-run lifecycle: full 1R first-target exit,
# per-symbol give-back/daily-stop P&L tracking, and win/loss cooldowns. The
# post-blowoff micro-base scout is a hit-run re-entry under a different label,
# so it must be treated the same everywhere or it escapes those controls.
HIT_RUN_STRATEGIES = frozenset({
    "momentum_burst_hit_run",
    "post_blowoff_micro_base_scout",
    "warrior_squeeze_playbook",
})


def is_hit_run_strategy(label: Optional[str]) -> bool:
    """True if a strategy/reason label belongs to the hit-run lifecycle."""
    text = (label or "").lower()
    return any(name in text for name in HIT_RUN_STRATEGIES)


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
    entry_strategy: str = ""
    entry_pattern: str = ""
    entry_score: float = 0.0

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

    # Runner handling.  These stay false for ordinary scalps.  A position only
    # becomes confirmed after it has already paid the first partial profit.
    runner_candidate: bool = False
    runner_confirmed: bool = False
    runner_trail_pct: float = RUNNER_TRAIL_PCT

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

    # Recent per-bar range % (high-low)/close, for the volatility-adaptive runner
    # trail. A wider-swinging name earns a wider trail so normal noise does not
    # stop it; a smooth name trails tight.
    _recent_ranges: List[float] = field(default_factory=list)
    _max_ranges: int = 10

    def record_bar_range(self, range_pct: float) -> None:
        if range_pct <= 0:
            return
        self._recent_ranges.append(float(range_pct))
        if len(self._recent_ranges) > self._max_ranges:
            self._recent_ranges.pop(0)

    def recent_range_pct(self) -> float:
        """Median recent per-bar range as a fraction (e.g. 0.027 = 2.7%)."""
        if not self._recent_ranges:
            return 0.0
        return statistics.median(self._recent_ranges)

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

    def __init__(
        self,
        *,
        max_unrealized_loss: float = 50.0,
        runner_trail_pct: float = RUNNER_TRAIL_PCT,
        runner_min_confirm_pct: float = RUNNER_MIN_CONFIRM_PCT,
        runner_trail_adaptive: bool = False,
        runner_trail_atr_mult: float = 2.5,
        runner_trail_cap: float = 0.10,
        runner_give_room_after_partial: bool = False,
        step_trail_exit_enabled: bool = False,
        step_trail_pct: float = 0.025,
    ) -> None:
        self._positions: Dict[str, TrackedPosition] = {}
        self._max_unrealized_loss = max_unrealized_loss
        # Step-trail exit (default off): ride a long while it keeps clearing
        # step_trail_pct-sized steps up; cut the moment it stalls back below the
        # last step. Overrides the partial/breakeven/runner-trail logic when on.
        self._step_trail_exit = bool(step_trail_exit_enabled)
        self._step_trail_pct = max(0.001, float(step_trail_pct))
        # How wide the back half of a confirmed runner trails behind the high, and
        # how far it must run past the partial before it earns that wider trail.
        # Tunable: a wider trail rides continuation runners (CUPR) further but
        # gives back more on top-and-fade names (EDHL). Default stays tight.
        self._runner_trail_pct = max(0.0, float(runner_trail_pct))
        self._runner_min_confirm_pct = max(0.0, float(runner_min_confirm_pct))
        # Adaptive trail: scale the trail to the name's own recent volatility, so
        # a wide-swinging runner (CUPR) breathes while a smooth one (EDHL) trails
        # tight. trail = clamp(atr_mult * median recent bar range, _pct floor, cap).
        # Off by default -> flat _runner_trail_pct (current behavior).
        self._runner_trail_adaptive = bool(runner_trail_adaptive)
        self._runner_trail_atr_mult = max(0.0, float(runner_trail_atr_mult))
        self._runner_trail_cap = max(0.0, float(runner_trail_cap))
        # Give runner candidates room: after the first partial, keep the wider
        # entry stop through the first pullback instead of snapping to breakeven
        # (breakeven shakes a runner out on a normal dip right before it
        # continues). Off by default. Trade-off: bigger give-back on the ones
        # that break down after the partial.
        self._runner_give_room = bool(runner_give_room_after_partial)

    def _apply_post_partial_stop(self, pos: "TrackedPosition") -> None:
        """Stop to set after the first partial: breakeven, or keep the wider
        original stop for a give-room runner candidate."""
        pos.breakeven_locked = True
        if (
            self._runner_give_room
            and pos.runner_candidate
            and pos.stop_loss is not None
            and 0 < pos.stop_loss < pos.entry_price
        ):
            return  # keep the original (wider) stop — give the runner room
        pos.stop_loss = pos.entry_price

    def _runner_trail_for(self, pos: "TrackedPosition") -> float:
        """Trail width for a confirmed runner: flat, or volatility-adaptive."""
        if not self._runner_trail_adaptive:
            return pos.runner_trail_pct
        vol = pos.recent_range_pct()
        if vol <= 0:
            return pos.runner_trail_pct
        adaptive = self._runner_trail_atr_mult * vol
        return min(self._runner_trail_cap, max(self._runner_trail_pct, adaptive))

    @property
    def tracked(self) -> Dict[str, TrackedPosition]:
        return dict(self._positions)

    def track(self, pos: TrackedPosition) -> None:
        entry_time = pos.entry_ts or datetime.now(timezone.utc)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        pos.entry_ts = entry_time
        if pos.entry_time is None:
            pos.entry_time = entry_time
        # Initialize range tracking from entry
        pos._range_high = pos.entry_price
        pos._range_low = pos.entry_price
        pos._range_start_ts = entry_time
        self._positions[pos.symbol] = pos
        target = pos.first_target_price if pos.first_target_price > 0 else pos.entry_price + 0.20
        logger.info(
            "Tracking %s %s %.0f @ %.4f | SL=%.4f → 1:1 target=%.4f (risk=$%.2f, trend=%.2f)",
            pos.side.value, pos.symbol, pos.quantity, pos.entry_price,
            pos.stop_loss or 0, target, pos.risk_per_share, pos.trend_strength,
        )

    def untrack(self, symbol: str) -> None:
        self._positions.pop(symbol, None)

    def update_bar_close(
        self,
        symbol: str,
        close_price: float,
        open_price: float = 0.0,
        volume: int = 0,
        high_price: float = 0.0,
        low_price: float = 0.0,
    ) -> None:
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
            # Volatility sample for the adaptive runner trail: true range when
            # high/low are supplied, else fall back to the candle body.
            if close_price > 0:
                if high_price > 0 and low_price > 0 and high_price >= low_price:
                    pos.record_bar_range((high_price - low_price) / close_price)
                elif open_price > 0:
                    pos.record_bar_range(abs(close_price - open_price) / close_price)

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
        # enough breathing room, but never widen beyond the configured dollar
        # risk for the actual filled quantity.  Timed scalps can slip on entry;
        # without this cap a fill can silently turn a ~$50 planned risk into a
        # much larger loss before the software dollar stop reacts.
        # Example: signal $5.36, stop $5.25 (risk $0.11), but filled at $5.27
        # → effective risk $0.02 which is way too tight. Recalc to 2% = $0.105.
        if side is Side.BUY and actual_price > 0 and stop_loss > 0:
            effective_risk = actual_price - stop_loss
            min_risk = max(0.05, actual_price * 0.03)
            target_risk = effective_risk
            if effective_risk < min_risk:
                target_risk = min_risk
            if signal.quantity > 0 and self._max_unrealized_loss > 0:
                max_risk = self._max_unrealized_loss / signal.quantity
                if target_risk > max_risk:
                    target_risk = max_risk
            if abs(target_risk - effective_risk) >= 0.005:
                old_stop = stop_loss
                stop_loss = round(actual_price - target_risk, 4)
                logger.warning(
                    "STOP ADJUST %s: fill risk $%.2f adjusted "
                    "(entry %.2f, old stop %.2f) → new stop %.2f (risk $%.2f)",
                    signal.symbol, effective_risk, actual_price, old_stop,
                    stop_loss, target_risk,
                )

        scan = signal.scan_result
        entry_strategy = scan.scanner_name if scan is not None else ""
        entry_pattern = ""
        entry_score = 0.0
        if scan is not None:
            entry_pattern = str(scan.criteria.get("pattern") or scan.scanner_name or "")
            try:
                entry_score = float(scan.score)
            except (TypeError, ValueError):
                entry_score = 0.0
        runner_candidate = self._is_runner_candidate_signal(signal, entry_strategy, entry_pattern, entry_score)

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
            entry_strategy=entry_strategy,
            entry_pattern=entry_pattern,
            entry_score=entry_score,
            trend_strength=signal.trend_strength,
            runner_candidate=runner_candidate,
        ))

    def check_exits(
        self,
        prices: Dict[str, float],
        now: datetime,
    ) -> List[TradeSignal]:
        """Check all tracked positions for stop loss or step-up."""
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)
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

        # --- step-trail exit (overrides the rest when enabled) ---
        # Ride while price keeps clearing step_trail_pct steps up; cut the moment
        # it stalls back below the last cleared step. No partial — full position
        # rides, so winners run and faders are cut on the first stall.
        if self._step_trail_exit and is_long and pos.entry_price > 0:
            steps = int(max(0.0, pos.highest_price / pos.entry_price - 1.0) / self._step_trail_pct)
            if steps <= 0:
                trail_stop = pos.entry_price * (1.0 - self._step_trail_pct)
            else:
                trail_stop = pos.entry_price * (1.0 + (steps - 1) * self._step_trail_pct)
            if pos.stop_loss is None or trail_stop > pos.stop_loss:
                pos.stop_loss = round(trail_stop, 4)
            if price <= pos.stop_loss:
                reason = ExitReason.TRAILING_STOP if steps > 0 else ExitReason.STOP_LOSS
                sig = self._make_exit(pos, pos.remaining_qty, price, reason)
                pos.remaining_qty = 0
                return [sig]
            return []

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

        # --- software dollar stop: do not wait for a wide broker stop if
        # the position has already exceeded the intended dollar risk.
        if self._max_unrealized_loss > 0 and pos.remaining_qty > 0:
            if is_long:
                unrealized_loss = max(0.0, pos.entry_price - price) * pos.remaining_qty
            else:
                unrealized_loss = max(0.0, price - pos.entry_price) * pos.remaining_qty
            if unrealized_loss >= self._max_unrealized_loss:
                logger.info(
                    "DOLLAR STOP %s @ %.4f | unrealized loss $%.2f >= $%.2f "
                    "(entry=%.4f, qty=%d, broker_stop=%.4f)",
                    pos.symbol, price, unrealized_loss, self._max_unrealized_loss,
                    pos.entry_price, int(pos.remaining_qty), pos.stop_loss or 0.0,
                )
                sig = self._make_exit(pos, pos.remaining_qty, price, ExitReason.STOP_LOSS)
                pos.remaining_qty = 0
                return [sig]

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

        # --- Quick scalp partial: sell half at +1% before moving to breakeven ---
        if (
            is_long
            and not pos.sold_half
            and pos.remaining_qty > 1
            and self._uses_quick_scalp_partial(pos)
        ):
            quick_partial_price = pos.entry_price * (1 + QUICK_SCALP_PARTIAL_PCT)
            if price >= quick_partial_price:
                partial_qty = self._first_partial_qty(pos)
                logger.info(
                    "QUICK PARTIAL %s @ %.4f | hit +%.1f%% scalp pop %.4f | "
                    "selling %d of %d shares, moving stop to breakeven %.4f",
                    pos.symbol, price, QUICK_SCALP_PARTIAL_PCT * 100,
                    quick_partial_price, partial_qty, int(pos.remaining_qty),
                    pos.entry_price,
                )
                pos.sold_half = True
                pos.remaining_qty -= partial_qty
                self._maybe_confirm_runner(pos, price)
                self._apply_post_partial_stop(pos)
                sig = self._make_exit(pos, partial_qty, price, ExitReason.TAKE_PROFIT)
                return [sig]

        # --- Exit #1: Sell half at first profit target (1:1 R:R) ---
        if is_long and not pos.sold_half and pos.first_target_price > 0:
            if price >= pos.first_target_price:
                if self._uses_full_first_target(pos):
                    logger.info(
                        "FULL TARGET %s @ %.4f | hit 1:1 target %.4f | selling all %.0f shares",
                        pos.symbol, price, pos.first_target_price, pos.remaining_qty,
                    )
                    sig = self._make_exit(pos, pos.remaining_qty, price, ExitReason.TAKE_PROFIT)
                    pos.remaining_qty = 0
                    return [sig]
                partial_qty = self._first_partial_qty(pos)
                logger.info(
                    "HALF SELL %s @ %.4f | hit 1:1 target %.4f | selling %d of %d shares, moving stop to breakeven %.4f",
                    pos.symbol, price, pos.first_target_price, partial_qty, int(pos.remaining_qty), pos.entry_price,
                )
                pos.sold_half = True
                pos.remaining_qty -= partial_qty
                self._maybe_confirm_runner(pos, price)
                self._apply_post_partial_stop(pos)
                sig = self._make_exit(pos, partial_qty, price, ExitReason.TAKE_PROFIT)
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
            if is_long and not pos.breakeven_locked and not self._is_vwap_pullback(pos):
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
        elif pos.runner_confirmed and pos.runner_trail_pct > 0:
            # Post-half on a CONFIRMED runner: trail behind the high so a normal
            # pullback does not shake us out of the move. Width is flat or, when
            # adaptive is on, scaled to the name's own recent volatility. The stop
            # only ever ratchets up, never down.
            trail = self._runner_trail_for(pos)
            if is_long:
                trail_stop = round(pos.highest_price * (1 - trail), 4)
                if trail_stop > (pos.stop_loss or 0):
                    pos.stop_loss = trail_stop
            else:
                trail_stop = round(pos.lowest_price * (1 + trail), 4)
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

    @staticmethod
    def _uses_quick_scalp_partial(pos: TrackedPosition) -> bool:
        if ExitManager._uses_full_first_target(pos):
            return False
        if ExitManager._is_vwap_pullback(pos):
            return False
        text = ExitManager._position_text(pos)
        return any(
            key in text
            for key in (
                "momentum",
                "quick",
                "scalp",
                "pullback",
                "abc",
                "hod",
                "breakout",
            )
        )

    @staticmethod
    def _is_vwap_pullback(pos: TrackedPosition) -> bool:
        text = ExitManager._position_text(pos)
        return "vwap_pullback" in text or "vwap pullback" in text

    @staticmethod
    def _position_text(pos: TrackedPosition) -> str:
        return " ".join(
            str(part or "").lower()
            for part in (
                pos.reason,
                pos.entry_strategy,
                pos.entry_pattern,
            )
        )

    @staticmethod
    def _uses_full_first_target(pos: TrackedPosition) -> bool:
        text = " ".join(
            str(part or "").lower()
            for part in (
                pos.reason,
                pos.entry_strategy,
                pos.entry_pattern,
            )
        )
        if "warrior_squeeze_playbook" in text:
            return False
        return is_hit_run_strategy(text)

    @staticmethod
    def _first_partial_qty(pos: TrackedPosition) -> int:
        """Bank less on runner candidates so more size can ride the move."""
        remaining = int(pos.remaining_qty)
        if remaining <= 1:
            return max(1, remaining)
        if pos.runner_candidate:
            return max(1, int(remaining / 3))
        return max(1, int(remaining / 2))

    @staticmethod
    def _is_runner_candidate_signal(
        signal: TradeSignal,
        entry_strategy: str,
        entry_pattern: str,
        entry_score: float,
    ) -> bool:
        """Return True for setups that deserve protected-runner handling.

        This does not buy or hold blindly.  It only marks the position as
        eligible; it still has to earn the first partial profit before the
        back half gets wider trailing room.
        """
        text = " ".join(
            str(v or "").lower()
            for v in (
                signal.reason,
                entry_strategy,
                entry_pattern,
                signal.scan_result.scanner_name if signal.scan_result else "",
            )
        )
        runner_patterns = (
            "hod",
            "breakout",
            "reclaim",
            "abc",
            "runner",
            "pullback",
            "squeeze",
            "momentum",
        )
        if not any(key in text for key in runner_patterns):
            return False
        if signal.trend_strength >= 0.8:
            return True
        if entry_score >= 80:
            return True
        criteria = signal.scan_result.criteria if signal.scan_result else {}
        try:
            day_change = float(
                criteria.get("day_change_pct")
                or criteria.get("change_pct")
                or criteria.get("day_change")
                or 0.0
            )
        except (TypeError, ValueError):
            day_change = 0.0
        try:
            volume = float(criteria.get("volume") or criteria.get("day_volume") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        return day_change >= 20.0 and volume >= 1_000_000

    def _maybe_confirm_runner(self, pos: TrackedPosition, price: float) -> None:
        if not pos.runner_candidate or pos.runner_confirmed or pos.entry_price <= 0:
            return
        max_run_pct = (max(pos.highest_price, price) - pos.entry_price) / pos.entry_price
        if max_run_pct < self._runner_min_confirm_pct:
            return
        pos.runner_confirmed = True
        pos.runner_trail_pct = self._runner_trail_pct
        logger.info(
            "RUNNER CONFIRMED %s | first partial paid, max run %.1f%% — "
            "back half uses %.1f%% trail",
            pos.symbol, max_run_pct * 100.0, pos.runner_trail_pct * 100.0,
        )
