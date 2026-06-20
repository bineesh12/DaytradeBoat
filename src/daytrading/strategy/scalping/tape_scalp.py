"""Tape reading scalp verifier for $1–$20 stocks.

Entry: TapeReaderScanner detected strong order flow imbalance.
Verify: confirm the imbalance is sustained and tape speed is high —
this signals institutional/algo activity that retail can ride.

Uses tick-based stops/targets (1 tick = $0.01) with 1:2 risk-reward.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from daytrading.models import PortfolioState, ScanResult, SignalAction, TradeSignal
from daytrading.strategy.entry_guard import check_entry_quality

logger = logging.getLogger(__name__)

TICK = 0.01


class TapeScalpVerifier:

    def __init__(
        self,
        *,
        min_score: float = 2.0,
        stop_ticks: int = 5,
        target_ticks: int = 10,
        trail_ticks: int = 3,
        max_hold_seconds: int = 300,
        position_size: float = 500,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._min_score = min_score
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
        return "tape_scalp"

    def verify(
        self,
        scan_result: ScanResult,
        portfolio: PortfolioState,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[TradeSignal]:
        if scan_result.score < self._min_score:
            return None

        direction = scan_result.criteria.get("direction", "")
        imbalance = scan_result.criteria.get("imbalance", 0.0)
        last_price = scan_result.criteria.get("last_price", 0.0)

        if last_price <= 0:
            return None
        if not (self._min_price <= last_price <= self._max_price):
            return None

        pos = portfolio.positions.get(scan_result.symbol)
        if pos and not pos.is_flat:
            return None

        cum_delta = scan_result.criteria.get("cum_delta", 0.0)
        if cum_delta < 100:
            return None

        bars = scan_result.bars
        if not bars or len(bars) < 3:
            logger.info("ENTRY GUARD REJECT %s: no bar data for tape signal", scan_result.symbol)
            self._last_reject = "no bar data for tape signal"
            return None

        reject = check_entry_quality(bars, symbol=scan_result.symbol, now=now)
        if reject is not None:
            logger.info("ENTRY GUARD REJECT %s: %s", scan_result.symbol, reject)
            self._last_reject = reject
            return None

        if direction == "buy_pressure":
            return TradeSignal(
                symbol=scan_result.symbol,
                action=SignalAction.ENTER_LONG,
                quantity=self._size,
                entry_price=last_price,
                stop_loss=last_price - self._stop,
                take_profit=last_price + self._target,
                trailing_stop_offset=self._trail,
                max_hold_seconds=self._max_hold,
                reason="Tape scalp long ${:.2f}, imbalance={:.2f}, SL={:.0f}t TP={:.0f}t".format(
                    last_price, imbalance,
                    self._stop / TICK, self._target / TICK),
                scan_result=scan_result,
            )

        return None
