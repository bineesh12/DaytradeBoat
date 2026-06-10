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
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from daytrading.exits.manager import ExitManager, TrackedPosition, build_exit_tiers
from daytrading.models import Bar, ExitReason, ScanResult, Side, SignalAction, TradeSignal

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
    require_protected_runner: bool = False
    require_clean_pullback_profile: bool = False
    min_pullback_depth_pct: float = 1.0
    max_pullback_depth_pct: float = 12.0
    max_base_range_pct: float = 8.0
    max_add_risk_pct: float = 3.0


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
            if cfg.require_protected_runner and (
                not is_long
                or not pos.sold_half
                or not pos.breakeven_locked
                or (pos.stop_loss is not None and pos.stop_loss < pos.entry_price * 0.995)
            ):
                continue

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

            sym_bars = bars.get(symbol)
            profile = self._clean_pullback_profile(symbol, pos, price, sym_bars)
            if cfg.require_clean_pullback_profile:
                if profile is None:
                    continue
            elif sym_bars and len(sym_bars) >= 2:
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
                profile_stop = profile["stop_price"] if profile is not None else None
                new_stop = profile_stop or (pos.stop_loss or pos.entry_price) + stop_advance
                new_stop = max(new_stop, pos.entry_price)
                if (
                    (cfg.require_protected_runner or cfg.require_clean_pullback_profile)
                    and price > 0
                    and (price - new_stop) / price * 100.0 > cfg.max_add_risk_pct
                ):
                    continue
                action = SignalAction.SCALE_UP_LONG
            else:
                new_stop = (pos.stop_loss or pos.entry_price) - stop_advance
                action = SignalAction.SCALE_UP_SHORT

            criteria = {
                "pattern": "runner_readd",
                "direction": "up" if is_long else "down",
                "close": price,
                "volume": float(sym_bars[-1].volume) if sym_bars else 0.0,
                "stop_price": new_stop,
            }
            if profile is not None:
                criteria.update(profile)

            signals.append(TradeSignal(
                symbol=symbol,
                action=action,
                quantity=scale_size,
                entry_price=price,
                stop_loss=new_stop,
                reason=(
                    f"Protected runner re-add #{count+1}: +{scale_size} shares "
                    f"@ ${price:.2f}, profit={profit_cents:.0f}¢"
                ),
                scan_result=ScanResult(
                    symbol=symbol,
                    scanner_name="runner_readd",
                    ts=datetime.now(timezone.utc),
                    score=float(profile.get("score", 0.0) if profile else 0.0),
                    criteria=criteria,
                    bars=list(sym_bars or []),
                ),
                trend_strength=pos.trend_strength,
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

    def _clean_pullback_profile(
        self,
        symbol: str,
        pos: TrackedPosition,
        price: float,
        bars: Optional[Sequence[Bar]],
    ) -> Optional[Dict[str, float]]:
        """Validate 1-minute runner re-add structure.

        Looks for the pattern the user described: strong impulse, controlled
        lower-volume pullback, then green reclaim with rising volume.
        """
        cfg = self._config
        if pos.side is not Side.BUY:
            return None
        if not bars or len(bars) < 8:
            return None
        recent = list(bars[-12:])
        latest = recent[-1]
        if latest.close <= latest.open:
            return None

        vwap = self._session_vwap(recent)
        if vwap <= 0 or latest.close < vwap * 1.002:
            return None

        high = max(float(b.high or b.close) for b in recent[:-1])
        pullback_low = min(float(b.low or b.close) for b in recent[-5:])
        if high <= 0 or pullback_low <= 0:
            return None
        pullback_depth_pct = (high - pullback_low) / high * 100.0
        if (
            pullback_depth_pct < cfg.min_pullback_depth_pct
            or pullback_depth_pct > cfg.max_pullback_depth_pct
        ):
            return None

        base_high = max(float(b.high or b.close) for b in recent[-5:])
        base_low = min(float(b.low or b.close) for b in recent[-5:])
        base_range_pct = (base_high - base_low) / latest.close * 100.0 if latest.close > 0 else 999.0
        if base_range_pct > cfg.max_base_range_pct:
            return None

        impulse_candidates = [
            b for b in recent[:-3]
            if b.close > b.open and float(b.volume or 0.0) > 0
        ]
        if not impulse_candidates:
            return None
        impulse = max(impulse_candidates, key=lambda b: float(b.volume or 0.0))
        impulse_vol = float(impulse.volume or 0.0)
        pullback_bars = recent[-5:-1]
        red_pullback = [b for b in pullback_bars if b.close < b.open and float(b.volume or 0.0) > 0]
        sample = red_pullback or [b for b in pullback_bars if float(b.volume or 0.0) > 0]
        if not sample:
            return None
        pullback_avg_vol = sum(float(b.volume or 0.0) for b in sample) / len(sample)
        reclaim_vol = float(latest.volume or 0.0)
        if impulse_vol <= 0 or pullback_avg_vol <= 0:
            return None
        if pullback_avg_vol > impulse_vol * 0.85:
            return None
        if reclaim_vol < pullback_avg_vol * 1.05:
            return None
        if latest.high > latest.low:
            upper_wick = float(latest.high - latest.close)
            rng = float(latest.high - latest.low)
            if rng > 0 and upper_wick / rng > 0.45:
                return None

        stop_price = max(pos.entry_price, pullback_low - max(0.01, latest.close * 0.003))
        score = 50.0
        score += min(20.0, max(0.0, (impulse_vol / pullback_avg_vol - 1.0) * 10.0))
        score += min(20.0, max(0.0, (reclaim_vol / pullback_avg_vol - 1.0) * 15.0))
        score += 10.0 if latest.close >= high * 0.985 else 0.0

        return {
            "vwap": round(vwap, 4),
            "pullback_low": round(pullback_low, 4),
            "base_low": round(base_low, 4),
            "base_range_pct": round(base_range_pct, 3),
            "pullback_pct": round(pullback_depth_pct, 3),
            "impulse_volume": impulse_vol,
            "pullback_volume": pullback_avg_vol,
            "reclaim_volume": reclaim_vol,
            "stop_price": round(stop_price, 4),
            "score": round(min(score, 100.0), 3),
        }

    @staticmethod
    def _session_vwap(bars: Sequence[Bar]) -> float:
        total_volume = 0.0
        total_value = 0.0
        for bar in bars:
            volume = float(bar.volume or 0.0)
            if volume <= 0:
                continue
            typical = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
            total_value += typical * volume
            total_volume += volume
        if total_volume <= 0:
            return 0.0
        return total_value / total_volume

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
    cooldown_seconds: float = 300.0    # 5 min cooldown after stop-out
    max_reentries: int = 1             # max 1 re-entry per symbol per session
    reentry_size_pct: float = 0.5      # re-enter with 50% of original size
    min_continuation_cents: float = 3.0  # price must move ≥ 3¢ past exit price
    pullback_max_cents: float = 5.0    # pullback must be ≤ 5¢ (not a reversal)
    stop_cents: float = 3.0
    trail_cents: float = 2.0
    max_hold_seconds: int = 90
    require_clean_continuation_profile: bool = False
    min_pullback_depth_pct: float = 1.0
    max_pullback_depth_pct: float = 12.0
    max_base_range_pct: float = 8.0
    max_reentry_risk_pct: float = 3.0


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
                if pullback_depth > max_pb and not cfg.require_clean_continuation_profile:
                    # too deep — this might be a reversal, not a continuation
                    continue

            # volume check
            sym_bars = bars.get(symbol)
            profile = self._clean_continuation_profile(
                symbol, last_exit, price, sym_bars,
            )
            if cfg.require_clean_continuation_profile:
                if profile is None:
                    continue
            elif sym_bars and len(sym_bars) >= 2:
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
                stop = (
                    profile["stop_price"]
                    if profile is not None
                    else price - stop_offset
                )
                if price > 0 and (price - stop) / price * 100.0 > cfg.max_reentry_risk_pct:
                    continue
            else:
                action = SignalAction.REENTER_SHORT
                stop = price + stop_offset

            criteria = {
                "pattern": "abc_reentry",
                "direction": "up" if is_long else "down",
                "close": price,
                "volume": float(sym_bars[-1].volume) if sym_bars else 0.0,
                "stop_price": stop,
                "setup_quality": "reentry_continuation" if profile else "simple_reentry",
                "size_factor": cfg.reentry_size_pct,
            }
            if profile is not None:
                criteria.update(profile)

            signals.append(TradeSignal(
                symbol=symbol,
                action=action,
                quantity=reentry_size,
                entry_price=price,
                stop_loss=stop,
                trailing_stop_offset=cfg.trail_cents / 100.0,
                max_hold_seconds=cfg.max_hold_seconds,
                reason=(
                    f"ABC re-entry #{reentry_count+1}: reclaim after exit "
                    f"${last_exit.exit_price:.2f}"
                ),
                scan_result=ScanResult(
                    symbol=symbol,
                    scanner_name="abc_reentry",
                    ts=now,
                    score=float(profile.get("score", 0.0) if profile else 0.0),
                    criteria=criteria,
                    bars=list(sym_bars or []),
                ),
            ))

            self._reentry_counts[symbol] = reentry_count + 1
            self._pullback_prices.pop(symbol, None)

            logger.info(
                "RE-ENTRY %s %s %d @ %.4f (exit was %.4f, re-entry #%d)",
                symbol, action.value, reentry_size, price,
                last_exit.exit_price, reentry_count + 1,
            )

        return signals

    def _clean_continuation_profile(
        self,
        symbol: str,
        last_exit: _ExitRecord,
        price: float,
        bars: Optional[Sequence[Bar]],
    ) -> Optional[Dict[str, float]]:
        """Validate ABC/pullback continuation after a full exit."""
        cfg = self._config
        if last_exit.side is not Side.BUY:
            return None
        if not bars or len(bars) < 8:
            return None
        recent = list(bars[-12:])
        latest = recent[-1]
        if latest.close <= latest.open:
            return None

        vwap_value = PositionScaler._session_vwap(recent)
        if vwap_value <= 0 or latest.close < vwap_value * 1.002:
            return None

        a_high = max(float(b.high or b.close) for b in recent[:-1])
        pullback_window = recent[-5:-1]
        if len(pullback_window) < 3 or a_high <= 0:
            return None
        b_low = min(float(b.low or b.close) for b in pullback_window)
        base_high = max(float(b.high or b.close) for b in pullback_window)
        base_low = b_low
        if b_low <= 0:
            return None

        pullback_depth_pct = (a_high - b_low) / a_high * 100.0
        if (
            pullback_depth_pct < cfg.min_pullback_depth_pct
            or pullback_depth_pct > cfg.max_pullback_depth_pct
        ):
            return None

        base_range_pct = (base_high - base_low) / latest.close * 100.0 if latest.close > 0 else 999.0
        if base_range_pct > cfg.max_base_range_pct:
            return None

        if latest.close < base_high * 0.995:
            return None

        impulse_green = [
            b for b in recent[:-4]
            if b.close > b.open and float(b.volume or 0.0) > 0
        ]
        if not impulse_green:
            return None
        impulse_vol = max(float(b.volume or 0.0) for b in impulse_green)
        pullback_sample = [
            b for b in pullback_window
            if float(b.volume or 0.0) > 0
        ]
        if not pullback_sample:
            return None
        pullback_avg_vol = sum(float(b.volume or 0.0) for b in pullback_sample) / len(pullback_sample)
        reclaim_vol = float(latest.volume or 0.0)
        if impulse_vol <= 0 or pullback_avg_vol <= 0:
            return None
        if pullback_avg_vol > impulse_vol * 0.90:
            return None
        if reclaim_vol < pullback_avg_vol * 1.10:
            return None
        if latest.high > latest.low:
            upper_wick_ratio = (latest.high - latest.close) / (latest.high - latest.low)
            if upper_wick_ratio > 0.45:
                return None

        stop_price = max(last_exit.entry_price, b_low - max(0.01, latest.close * 0.003))
        score = 55.0
        score += min(20.0, max(0.0, (impulse_vol / pullback_avg_vol - 1.0) * 8.0))
        score += min(20.0, max(0.0, (reclaim_vol / pullback_avg_vol - 1.0) * 12.0))
        score += 5.0 if latest.close > last_exit.exit_price else 0.0

        return {
            "vwap": round(vwap_value, 4),
            "a_high": round(a_high, 4),
            "b_low": round(b_low, 4),
            "base_low": round(base_low, 4),
            "base_high": round(base_high, 4),
            "base_range_pct": round(base_range_pct, 3),
            "pullback_pct": round(pullback_depth_pct, 3),
            "impulse_volume": impulse_vol,
            "pullback_volume": pullback_avg_vol,
            "reclaim_volume": reclaim_vol,
            "stop_price": round(stop_price, 4),
            "score": round(min(score, 100.0), 3),
        }

    def clear_session(self) -> None:
        """Reset all re-entry tracking for a new session."""
        self._exit_history.clear()
        self._reentry_counts.clear()
        self._pullback_prices.clear()
