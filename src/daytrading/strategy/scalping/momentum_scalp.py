"""Momentum scalping verifier for $1–$20 stocks.

Entry: scanner detected a sharp micro-move (momentum burst).
Verify: confirm trend continuation — latest bar closes in direction of burst,
volume is increasing, and the move has not already exhausted itself.

Uses tick-based stops/targets (1 tick = $0.01).
5 tick stop / 10 tick target — stepping stop system locks profit every 10 ticks.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from daytrading.models import PortfolioState, ScanResult, SignalAction, TradeSignal
from daytrading.strategy.entry_guard import check_entry_quality

logger = logging.getLogger(__name__)

TICK = 0.01


class MomentumScalpVerifier:

    def __init__(
        self,
        *,
        stop_ticks: int = 5,
        target_ticks: int = 10,
        trail_ticks: int = 3,
        max_hold_seconds: int = 300,
        position_size: float = 500,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._stop = stop_ticks * TICK
        self._target = target_ticks * TICK
        self._trail = trail_ticks * TICK
        self._max_hold = max_hold_seconds
        self._size = position_size
        self._min_price = min_price
        self._max_price = max_price
        self._last_reject: Optional[str] = None

    @property
    def name(self) -> str:
        return "momentum_scalp"

    def verify(
        self,
        scan_result: ScanResult,
        portfolio: PortfolioState,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[TradeSignal]:
        bars = scan_result.bars
        if len(bars) < 3:
            self._last_reject = "insufficient bar data"
            return None

        direction = scan_result.criteria.get("direction", "")
        burst_pct = float(scan_result.criteria.get("burst_pct", 0.0) or 0.0)
        latest = bars[-1]
        prev = bars[-2]

        if not (self._min_price <= latest.close <= self._max_price):
            self._last_reject = "price ${:.2f} outside range".format(latest.close)
            return None

        pos = portfolio.positions.get(scan_result.symbol)
        if pos and not pos.is_flat:
            self._last_reject = "already in position"
            return None

        if direction == "up":
            if latest.close <= latest.open:
                self._last_reject = "bearish candle (close <= open)"
                return None
            if latest.volume < prev.volume * 0.8:
                self._last_reject = "volume declining"
                return None
            if burst_pct < 0.8:
                self._last_reject = "burst {:.2f}% < 0.80%".format(burst_pct)
                return None

            reject = check_entry_quality(bars, symbol=scan_result.symbol, now=now)
            if reject is not None:
                logger.info("ENTRY GUARD REJECT %s: %s", scan_result.symbol, reject)
                self._last_reject = reject
                return None

            price = latest.close

            return TradeSignal(
                symbol=scan_result.symbol,
                action=SignalAction.ENTER_LONG,
                quantity=self._size,
                entry_price=price,
                stop_loss=price - self._stop,
                take_profit=price + self._target,
                trailing_stop_offset=self._trail,
                max_hold_seconds=self._max_hold,
                reason="Momentum scalp long ${:.2f}, burst={:.3f}%, SL={:.0f}t TP={:.0f}t".format(
                    price, burst_pct,
                    self._stop / TICK, self._target / TICK),
                scan_result=scan_result,
            )

        return None
