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

from daytrading.models import PortfolioState, ScanResult, SignalAction, TradeSignal
from daytrading.strategy.entry_guard import check_entry_quality

logger = logging.getLogger(__name__)

TICK = 0.01


class MomentumPatternVerifier:
    """Verifies bull flag and flat top breakout scan hits."""

    def __init__(
        self,
        *,
        max_risk_per_share: float = 0.50,
        reward_risk_ratio: float = 2.0,
        trail_ticks: int = 5,
        max_hold_seconds: int = 600,
        max_dollar_risk: float = 100.0,
        min_price: float = 1.0,
        max_price: float = 20.0,
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

    @property
    def name(self) -> str:
        return "momentum_pattern"

    def _get_avg_volume(self, symbol: str) -> Optional[float]:
        if self._float_checker is not None and hasattr(self._float_checker, "get_avg_volume"):
            return self._float_checker.get_avg_volume(symbol)
        return None

    def verify(
        self,
        scan_result: ScanResult,
        portfolio: PortfolioState,
    ) -> Optional[TradeSignal]:
        bars = scan_result.bars
        if len(bars) < 3:
            self._last_reject = "insufficient bar data"
            return None

        latest = bars[-1]
        price = latest.close

        if not (self._min_price <= price <= self._max_price):
            self._last_reject = "price ${:.2f} outside range".format(price)
            return None

        pos = portfolio.positions.get(scan_result.symbol)
        if pos and not pos.is_flat:
            self._last_reject = "already in position"
            return None

        pattern = scan_result.criteria.get("pattern", "")
        direction = scan_result.criteria.get("direction", "")
        if direction != "up":
            self._last_reject = "not an upward pattern"
            return None

        # Momentum burst scanner hits use the recent low as stop
        is_momentum_burst = scan_result.scanner_name == "momentum_burst"

        known_patterns = (
            "bull_flag", "flat_top_breakout", "vwap_pullback",
            "opening_range_breakout", "hod_reclaim",
        )
        if not is_momentum_burst and pattern not in known_patterns:
            self._last_reject = "unknown pattern: {}".format(pattern)
            return None

        reject = check_entry_quality(
            bars,
            symbol=scan_result.symbol,
            avg_daily_volume=self._get_avg_volume(scan_result.symbol),
        )
        if reject is not None:
            logger.info("ENTRY GUARD REJECT %s: %s", scan_result.symbol, reject)
            self._last_reject = reject
            return None

        logger.info("ENTRY GUARD PASS %s pattern=%s price=%.2f",
                     scan_result.symbol, pattern or scan_result.scanner_name, price)

        # --- Fixed stop / target for small-cap momentum scalps ---
        # These are $2-$10 stocks that move in cents, not dollars.
        # Risk 10 cents per share, target 20 cents (2:1).
        # Pattern-specific stop is used if it's tighter than 10c,
        # but capped at 10c max so the 2:1 target stays reachable.
        FIXED_RISK = 0.10  # 10 cents risk per share

        if is_momentum_burst:
            pattern = "momentum_burst"
            pattern_stop = price - FIXED_RISK
        elif pattern == "bull_flag":
            pullback_low = float(scan_result.criteria.get("pullback_low", 0))
            if pullback_low <= 0:
                self._last_reject = "no pullback low"
                return None
            pattern_stop = pullback_low - 0.02
        elif pattern == "flat_top_breakout":
            resistance = float(scan_result.criteria.get("resistance", 0))
            if resistance <= 0:
                self._last_reject = "no resistance level"
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
                pattern_stop = price - FIXED_RISK
        elif pattern == "opening_range_breakout":
            orb_high = float(scan_result.criteria.get("orb_high", 0))
            orb_low = float(scan_result.criteria.get("orb_low", 0))
            if orb_high > 0:
                pattern_stop = orb_high - 0.02
            elif orb_low > 0:
                pattern_stop = orb_low - 0.02
            else:
                pattern_stop = price - FIXED_RISK
        elif pattern == "hod_reclaim":
            pullback_low = float(scan_result.criteria.get("pullback_low", 0))
            stop_from_criteria = float(scan_result.criteria.get("stop_price", 0))
            if stop_from_criteria > 0:
                pattern_stop = stop_from_criteria
            elif pullback_low > 0:
                pattern_stop = pullback_low - 0.02
            else:
                pattern_stop = price - FIXED_RISK
        else:
            self._last_reject = "unhandled pattern"
            return None

        # Cap risk at 10 cents — if pattern stop is wider, tighten it
        stop_price = max(pattern_stop, price - FIXED_RISK)

        risk_per_share = price - stop_price
        if risk_per_share <= 0:
            self._last_reject = "stop above entry"
            return None

        target_price = price + (risk_per_share * self._rr_ratio)

        # Position sizing based on max dollar risk
        quantity = int(self._max_dollar_risk / risk_per_share)
        quantity = max(1, min(quantity, 2000))

        reason = "{} {} ${:.2f}, stop=${:.2f} (risk=${:.2f}), target=${:.2f} (2:1)".format(
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
