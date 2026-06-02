"""Momentum pattern verifier — Bull Flag, Flat Top Breakout, VWAP Pullback,
Opening Range Breakout & HOD Reclaim.

Implements Warrior Trading's momentum day trading entry rules:
- Entry on the first candle making a new high after pullback (bull flag)
  or breaking above flat resistance (flat top breakout)
- First pullback to VWAP after a rally
- First 5-minute opening range breakout
- High-of-day reclaim after pullback
- Stop loss below the pattern low
- Target at 2:1 reward-to-risk ratio
- Trailing stop to lock profits
- Max 20 cent risk per share; if stop is wider, reduce position size
"""

from __future__ import annotations

import logging
from typing import Optional

from daytrading.indicators.core import vwap
from daytrading.models import PortfolioState, ScanResult, SignalAction, TradeSignal
from daytrading.strategy.entry_guard import check_entry_quality, record_rule_rejection

logger = logging.getLogger(__name__)

TICK = 0.01


class MomentumPatternVerifier:
    """Verifies bull flag and flat top breakout scan hits."""

    def __init__(
        self,
        *,
        max_risk_per_share: float = 0.50,
        reward_risk_ratio: float = 1.0,
        trail_ticks: int = 5,
        max_hold_seconds: int = 600,
        max_dollar_risk: float = 100.0,
        min_price: float = 1.0,
        max_price: float = 500.0,
        float_checker: object = None,
    ) -> None:
        self._max_risk = max_risk_per_share
        self._rr_ratio = reward_risk_ratio
        self._trail = trail_ticks * TICK
        self._max_hold = max_hold_seconds
        self._max_dollar_risk = max_dollar_risk
        self._min_price = min_price
        self._max_price = max_price
        self._float_checker = float_checker
        self._last_reject: Optional[str] = None
        self._bar_aggregator = None  # set by factory/runner for 5m context
        self._tick_buffer = None  # set by runner: Dict[str, deque] of recent ticks
        self._quote_buffer = None  # set by runner: Dict[str, deque] of recent quotes
        self._current_symbol: Optional[str] = None

    @property
    def name(self) -> str:
        return "momentum_pattern"

    def _reject(self, reason: str) -> None:
        self._last_reject = reason
        record_rule_rejection(symbol=self._current_symbol, reason=reason)

    def _get_avg_volume(self, symbol: str) -> Optional[float]:
        if self._float_checker is not None and hasattr(self._float_checker, "get_avg_volume"):
            return self._float_checker.get_avg_volume(symbol)
        return None

    def _get_float_shares(self, symbol: str) -> Optional[float]:
        if self._float_checker is not None and hasattr(self._float_checker, "get_float"):
            return self._float_checker.get_float(symbol)
        return None

    @staticmethod
    def _is_hot_hod_reclaim(pattern: str, scan_result: ScanResult, bars: list) -> bool:
        if pattern != "hod_reclaim" or not bars:
            return False
        price = bars[-1].close
        if price <= 0:
            return False
        hod = float(scan_result.criteria.get("hod") or max(b.high for b in bars))
        distance_from_hod = (hod - price) / hod * 100 if hod > 0 else 100.0
        rally_pct = float(scan_result.criteria.get("rally_pct") or 0.0)
        volume = float(scan_result.criteria.get("volume") or bars[-1].volume or 0.0)
        return (
            distance_from_hod <= 2.0
            and rally_pct >= 20.0
            and volume >= 100_000
            and scan_result.score >= 50.0
        )

    @staticmethod
    def _hot_hod_reclaim_stop(price: float, bars: list) -> float:
        """Use a tactical stop for explosive HOD reclaims.

        The technical pullback low can be far below price after a fast reclaim.
        For quick scalping, that creates an untradeable 15-20% risk even when
        momentum is clean. Cap this setup to a smaller scalp-style stop.
        """
        recent = bars[-2:] if len(bars) >= 2 else bars[-1:]
        recent_lows = [b.low for b in recent if b.low > 0]
        if recent_lows:
            recent_stop = min(recent_lows) - 0.02
            recent_risk_pct = (price - recent_stop) / price if price > 0 else 1.0
            if 0.005 < recent_risk_pct <= 0.08:
                return recent_stop
        return price * 0.94

    @staticmethod
    def _is_fresh_late_reclaim(
        pattern: str,
        bars: list,
        *,
        session_high: float,
        distance_from_hod: float,
        day_move_pct: float,
    ) -> bool:
        """Allow a late pullback only after a fresh base reclaim.

        This keeps the normal no-chase HOD-distance rule intact, but gives
        strong runners a second chance after they build a tight base and reclaim
        it with volume. It is meant for DXST/OLOX-style continuation watches,
        not for buying a falling pullback.
        """
        if pattern not in ("vwap_pullback", "pullback_base") or len(bars) < 12:
            return False
        if session_high <= 0 or day_move_pct < 25.0:
            return False

        max_reclaim_distance = 18.0 if pattern == "pullback_base" else 16.0
        if distance_from_hod > max_reclaim_distance:
            return False

        latest = bars[-1]
        if latest.close <= latest.open or latest.close <= 0:
            return False
        candle_range = latest.high - latest.low
        body_ratio = (
            (latest.close - latest.open) / candle_range
            if candle_range > 0 else 0.0
        )
        if body_ratio < 0.40:
            return False

        base = list(bars[-5:-1])
        if len(base) < 4:
            return False
        base_high = max(b.high for b in base)
        base_low = min(b.low for b in base)
        if base_low <= 0:
            return False
        base_range_pct = (base_high - base_low) / base_low * 100.0
        max_base_range_pct = min(8.0, 4.0 + day_move_pct / 20.0)
        if base_range_pct > max_base_range_pct:
            return False
        if latest.close <= base_high * 1.002:
            return False
        if latest.low < base_low:
            return False

        avg_base_vol = sum(b.volume for b in base) / len(base)
        recent_volume = sum(b.volume for b in bars[-3:])
        if recent_volume < 75_000:
            return False
        if avg_base_vol > 0 and latest.volume < avg_base_vol * 1.10:
            return False

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap <= 0:
            return False
        above_vwap_pct = (latest.close - current_vwap) / current_vwap * 100.0
        return above_vwap_pct >= 1.0

    @staticmethod
    def _late_pullback_reject(pattern: str, bars: list) -> Optional[str]:
        """Extra quality gate for slower pullback entries."""
        if len(bars) < 10:
            return "late pullback needs more bars"

        latest = bars[-1]
        price = latest.close
        if price <= 0:
            return "invalid price"

        session_open = bars[0].open
        session_high = max(b.high for b in bars)
        if session_open <= 0 or session_high <= 0:
            return "invalid session range"

        day_move_pct = (session_high - session_open) / session_open * 100
        distance_from_hod = (session_high - price) / session_high * 100

        fresh_late_reclaim = False
        max_hod_distance = 8.0 if pattern == "vwap_pullback" else 10.0
        if distance_from_hod > max_hod_distance:
            fresh_late_reclaim = MomentumPatternVerifier._is_fresh_late_reclaim(
                pattern,
                bars,
                session_high=session_high,
                distance_from_hod=distance_from_hod,
                day_move_pct=day_move_pct,
            )
            if not fresh_late_reclaim:
                return (
                    "late pullback too far from HOD {:.1f}% "
                    "(max {:.1f}%; watching for fresh reclaim)"
                ).format(distance_from_hod, max_hod_distance)
            logger.info(
                "ENTRY GUARD LATE RECLAIM %s: %.1f%% from HOD but fresh base reclaim",
                pattern, distance_from_hod,
            )

        if pattern == "pullback_base" and day_move_pct < 20.0:
            return (
                "pullback base move too small {:.1f}% "
                "(need 20%+ for late pullback)"
            ).format(day_move_pct)

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap > 0:
            above_vwap_pct = (price - current_vwap) / current_vwap * 100
            if above_vwap_pct < 1.0:
                return (
                    "late pullback not strong above VWAP {:.1f}% "
                    "(need 1.0%+)"
                ).format(above_vwap_pct)

        recent = bars[-3:]
        recent_volume = sum(b.volume for b in recent)
        recent_avg = recent_volume / len(recent)
        earlier = bars[:-3]
        earlier_avg = (
            sum(b.volume for b in earlier) / len(earlier)
            if earlier else 0.0
        )
        if recent_volume < 75_000:
            return (
                "late pullback tape too slow {:.0f} recent volume "
                "(need 75K+)"
            ).format(recent_volume)
        if (
            earlier_avg > 0
            and recent_avg < earlier_avg * 0.60
            and not fresh_late_reclaim
        ):
            return (
                "late pullback volume faded {:.1f}x prior "
                "(need 0.6x+)"
            ).format(recent_avg / earlier_avg)

        return None

    def verify(
        self,
        scan_result: ScanResult,
        portfolio: PortfolioState,
    ) -> Optional[TradeSignal]:
        self._current_symbol = scan_result.symbol
        bars = scan_result.bars
        if len(bars) < 3:
            self._reject("insufficient bar data")
            return None

        latest = bars[-1]
        price = latest.close

        if not (self._min_price <= price <= self._max_price):
            self._reject("price ${:.2f} outside range".format(price))
            return None

        pos = portfolio.positions.get(scan_result.symbol)
        if pos and not pos.is_flat:
            self._reject("already in position")
            return None

        pattern = scan_result.criteria.get("pattern", "")
        direction = scan_result.criteria.get("direction", "")
        if direction != "up":
            self._reject("not an upward pattern")
            return None

        # Momentum burst scanner hits use the recent low as stop
        is_momentum_burst = scan_result.scanner_name == "momentum_burst"

        known_patterns = (
            "bull_flag", "flat_top_breakout", "vwap_pullback",
            "opening_range_breakout", "hod_reclaim", "pullback_base",
            "abc_continuation",
        )
        if not is_momentum_burst and pattern not in known_patterns:
            self._reject("unknown pattern: {}".format(pattern))
            return None

        if pattern in ("vwap_pullback", "pullback_base"):
            late_reject = self._late_pullback_reject(pattern, bars)
            if late_reject is not None:
                logger.info("ENTRY GUARD REJECT %s: %s", scan_result.symbol, late_reject)
                self._reject(late_reject)
                return None

        bars_5m = None
        if self._bar_aggregator is not None:
            bars_5m = self._bar_aggregator.get_5m_bars(scan_result.symbol)

        reject = check_entry_quality(
            bars,
            symbol=scan_result.symbol,
            avg_daily_volume=self._get_avg_volume(scan_result.symbol),
            bars_5m=bars_5m,
            float_shares=self._get_float_shares(scan_result.symbol),
            ticks=list(self._tick_buffer.get(scan_result.symbol, [])) if self._tick_buffer else None,
            quotes=list(self._quote_buffer.get(scan_result.symbol, [])) if self._quote_buffer else None,
        )
        if reject is not None:
            logger.info("ENTRY GUARD REJECT %s: %s", scan_result.symbol, reject)
            self._last_reject = reject
            return None

        logger.info("ENTRY GUARD PASS %s pattern=%s price=%.2f",
                     scan_result.symbol, pattern or scan_result.scanner_name, price)

        # --- Stop placement: Warrior Trading rule ---
        # The stop goes at the technical level where the pattern is invalidated.
        # We do NOT clamp it to an arbitrary % range.
        # If the risk per share is too large for our account, we take FEWER
        # shares to keep total dollar risk at $100. If the risk is absurdly
        # wide (>10% of price), we skip the trade entirely — the setup is
        # too loose to scalp.

        if is_momentum_burst:
            pattern = "momentum_burst"
            # For big movers, use last 3 bars (tighter stop near current action)
            # For normal stocks, use last 5 bars
            lookback = 3 if len(bars) >= 3 and (bars[-1].close / bars[0].open - 1) > 0.5 else 5
            lookback = min(lookback, len(bars))
            recent_low = min(b.low for b in bars[-lookback:]) if len(bars) >= lookback else price * 0.97
            pattern_stop = recent_low - 0.02
        elif pattern == "bull_flag":
            pullback_low = float(scan_result.criteria.get("pullback_low", 0))
            if pullback_low <= 0:
                self._reject("no pullback low")
                return None
            pattern_stop = pullback_low - 0.02
        elif pattern == "flat_top_breakout":
            resistance = float(scan_result.criteria.get("resistance", 0))
            if resistance <= 0:
                self._reject("no resistance level")
                return None
            pattern_stop = resistance - 0.05
        elif pattern == "vwap_pullback":
            pullback_low = float(scan_result.criteria.get("pullback_low", 0))
            stop_from_criteria = float(scan_result.criteria.get("stop_price", 0))
            if stop_from_criteria > 0:
                pattern_stop = stop_from_criteria
            elif pullback_low > 0:
                pattern_stop = pullback_low - 0.02
            else:
                recent_low = min(b.low for b in bars[-5:]) if len(bars) >= 5 else price * 0.97
                pattern_stop = recent_low - 0.02
        elif pattern == "opening_range_breakout":
            orb_high = float(scan_result.criteria.get("orb_high", 0))
            orb_low = float(scan_result.criteria.get("orb_low", 0))
            if orb_high > 0:
                pattern_stop = orb_high - 0.02
            elif orb_low > 0:
                pattern_stop = orb_low - 0.02
            else:
                recent_low = min(b.low for b in bars[-5:]) if len(bars) >= 5 else price * 0.97
                pattern_stop = recent_low - 0.02
        elif pattern == "hod_reclaim":
            pullback_low = float(scan_result.criteria.get("pullback_low", 0))
            stop_from_criteria = float(scan_result.criteria.get("stop_price", 0))
            if stop_from_criteria > 0:
                pattern_stop = stop_from_criteria
            elif pullback_low > 0:
                pattern_stop = pullback_low - 0.02
            else:
                recent_low = min(b.low for b in bars[-5:]) if len(bars) >= 5 else price * 0.97
                pattern_stop = recent_low - 0.02
        elif pattern == "pullback_base":
            base_low = float(scan_result.criteria.get("base_low", 0))
            pullback_low = float(scan_result.criteria.get("pullback_low", 0))
            if base_low > 0:
                pattern_stop = base_low - 0.02
            elif pullback_low > 0:
                pattern_stop = pullback_low - 0.02
            else:
                recent_low = min(b.low for b in bars[-5:]) if len(bars) >= 5 else price * 0.97
                pattern_stop = recent_low - 0.02
        elif pattern == "abc_continuation":
            b_low = float(scan_result.criteria.get("b_low", 0))
            if b_low <= 0:
                self._reject("no ABC B-low")
                return None
            pattern_stop = b_low - 0.02
        else:
            self._reject("unhandled pattern")
            return None

        # Use the technical stop as-is. Only sanity-check it.
        stop_price = pattern_stop

        risk_per_share = price - stop_price
        if risk_per_share <= 0:
            self._reject("stop above entry")
            return None

        risk_pct = risk_per_share / price
        # For big intraday movers, allow slightly wider risk (up to 15%)
        # but if still too wide, tighten the stop to last 3 bars low
        max_risk_pct = 0.10
        if len(bars) >= 5:
            day_move = (bars[-1].close - bars[0].open) / bars[0].open if bars[0].open > 0 else 0
            if day_move > 0.30:
                max_risk_pct = 0.15

        if risk_pct > max_risk_pct:
            # Try tighter stop: last 3 bars low
            tight_stop = min(b.low for b in bars[-3:]) - 0.02
            tight_risk = price - tight_stop
            tight_risk_pct = tight_risk / price if price > 0 else 1.0
            if self._is_hot_hod_reclaim(pattern, scan_result, bars):
                hot_stop = self._hot_hod_reclaim_stop(price, bars)
                hot_risk = price - hot_stop
                hot_risk_pct = hot_risk / price if price > 0 else 1.0
                if 0.005 < hot_risk_pct <= 0.08:
                    stop_price = hot_stop
                    risk_per_share = hot_risk
                    risk_pct = hot_risk_pct
                    logger.info(
                        "ENTRY GUARD %s: hot HOD reclaim tactical stop from $%.2f to $%.2f (risk %.0f%% → %.0f%%)",
                        scan_result.symbol, pattern_stop, stop_price,
                        (price - pattern_stop) / price * 100, risk_pct * 100,
                    )
                else:
                    logger.info(
                        "ENTRY GUARD REJECT %s: hot HOD reclaim risk too wide: $%.2f (%.0f%% of $%.2f)",
                        scan_result.symbol, hot_risk, hot_risk_pct * 100, price,
                    )
                    self._reject(
                        "hot HOD reclaim risk too wide: ${:.2f} ({:.0f}% of ${:.2f})".format(
                            hot_risk, hot_risk_pct * 100, price)
                    )
                    return None
            elif 0.005 < tight_risk_pct <= max_risk_pct:
                stop_price = tight_stop
                risk_per_share = tight_risk
                risk_pct = tight_risk_pct
                logger.info("ENTRY GUARD %s: tightened stop from $%.2f to $%.2f (risk %.0f%% → %.0f%%)",
                            scan_result.symbol, pattern_stop, stop_price, 
                            (price - pattern_stop) / price * 100, risk_pct * 100)
            else:
                logger.info("ENTRY GUARD REJECT %s: risk too wide: $%.2f (%.0f%% of $%.2f) — skip loose setup",
                            scan_result.symbol, risk_per_share, risk_pct * 100, price)
                self._reject(
                    "risk too wide: ${:.2f} ({:.0f}% of ${:.2f}) — skip loose setup".format(
                        risk_per_share, risk_pct * 100, price)
                )
                return None
        if risk_pct < 0.005:
            logger.info("ENTRY GUARD REJECT %s: risk too tight: $%.2f (%.1f%% of $%.2f) — will stop on noise",
                        scan_result.symbol, risk_per_share, risk_pct * 100, price)
            self._reject(
                "risk too tight: ${:.2f} ({:.1f}% of ${:.2f}) — will stop on noise".format(
                    risk_per_share, risk_pct * 100, price)
            )
            return None

        target_price = price + (risk_per_share * self._rr_ratio)

        # Position sizing based on max dollar risk
        quantity = int(self._max_dollar_risk / risk_per_share)
        quantity = max(1, min(quantity, 2000))

        reason = "{} {} ${:.2f}, stop=${:.2f} (risk=${:.2f}), target=${:.2f} (1:1)".format(
            pattern.replace("_", " ").title(),
            scan_result.symbol,
            price,
            stop_price,
            risk_per_share,
            target_price,
        )

        return TradeSignal(
            symbol=scan_result.symbol,
            action=SignalAction.ENTER_LONG,
            quantity=quantity,
            entry_price=price,
            stop_loss=stop_price,
            take_profit=target_price,
            trailing_stop_offset=self._trail,
            max_hold_seconds=self._max_hold,
            reason=reason,
            scan_result=scan_result,
        )
