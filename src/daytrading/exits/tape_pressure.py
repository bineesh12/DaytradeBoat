"""Tape pressure exit — detect selling pressure for profit protection.

Uses existing indicators from indicators/scalping.py to score real-time
selling pressure on an open position. When score exceeds threshold and the
position is in profit, triggers an exit to protect gains before a pullback.
"""

from __future__ import annotations

import logging
import math
from typing import List

from daytrading.indicators.scalping import (
    cumulative_delta,
    order_flow_imbalance,
    spread_compression_ratio,
    tape_speed,
)
from daytrading.models import Quote, Tick

logger = logging.getLogger(__name__)


class TapePressureExit:
    """Score selling pressure from tick/quote microstructure.

    Activates only when a position is in profit and has been held for a
    minimum duration.  Designed to protect gains before breakeven stop locks.
    """

    def __init__(
        self,
        *,
        threshold: int = 60,
        min_hold_secs: float = 30.0,
        min_ticks: int = 20,
    ) -> None:
        self._threshold = threshold
        self._min_hold = min_hold_secs
        self._min_ticks = min_ticks

    def check(
        self,
        ticks: List[Tick],
        quotes: List[Quote],
        entry_price: float,
        current_price: float,
        hold_secs: float,
    ) -> bool:
        """Return True if selling pressure warrants a profit-protection exit."""
        if hold_secs < self._min_hold:
            return False
        if current_price <= entry_price:
            return False
        if len(ticks) < self._min_ticks:
            return False

        score = self._compute_pressure(ticks, quotes)
        if score >= self._threshold:
            logger.info(
                "TAPE_PRESSURE score=%d (threshold=%d) | ticks=%d quotes=%d",
                score, self._threshold, len(ticks), len(quotes),
            )
            return True
        return False

    def _compute_pressure(self, ticks: List[Tick], quotes: List[Quote]) -> int:
        score = 0

        # 1. Order flow imbalance (0-30 pts) — sellers dominating recent trades
        if len(ticks) >= 30:
            imb_values = order_flow_imbalance(ticks, window=30)
            current_imb = imb_values[-1] if imb_values else 0.0
            if current_imb <= -0.3:
                score += min(30, int(abs(current_imb) * 30))

        # 2. Cumulative delta declining (0-25 pts) — net selling volume
        if len(ticks) >= 20:
            deltas = cumulative_delta(ticks[-20:])
            if len(deltas) >= 10:
                recent_change = deltas[-1] - deltas[-10]
                if recent_change < 0:
                    avg_size = sum(t.size for t in ticks[-20:]) / 20
                    normalized = abs(recent_change) / max(avg_size * 5, 1.0)
                    score += min(25, int(normalized * 25))

        # 3. Spread widening (0-20 pts) — market makers stepping back
        if len(quotes) >= 20:
            ratios = spread_compression_ratio(quotes[-20:], short=5, long=15)
            valid_ratios = [r for r in ratios if not math.isnan(r)]
            if valid_ratios and valid_ratios[-1] > 1.3:
                score += min(20, int((valid_ratios[-1] - 1.0) * 40))

        # 4. Tape speed dying (0-25 pts) — buying interest evaporating
        if len(ticks) >= 20:
            speeds = tape_speed(ticks[-20:], window_seconds=5.0)
            if len(speeds) >= 10 and speeds[-10] > 0:
                speed_ratio = speeds[-1] / speeds[-10]
                if speed_ratio < 0.5:
                    score += min(25, int((1.0 - speed_ratio) * 30))

        return score
