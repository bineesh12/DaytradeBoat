"""Spread-based scalp verifier for $1-$20 stocks.

Entry: SpreadFilterScanner detected a tight, compressing spread.
Verify: confirm spread is still tight and there's enough volume to scalp
the bid-ask efficiently.

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


class SpreadScalpVerifier:

    def __init__(
        self,
        *,
        max_spread_pct: float = 0.15,
        stop_ticks: int = 5,
        target_ticks: int = 10,
        trail_ticks: int = 3,
        max_hold_seconds: int = 300,
        position_size: float = 500,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._max_spread_pct = max_spread_pct
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
        return "spread_scalp"

    def verify(
        self,
        scan_result: ScanResult,
        portfolio: PortfolioState,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[TradeSignal]:
        spread_pct = scan_result.criteria.get("spread_pct", 1.0)
        compression = scan_result.criteria.get("compression_ratio", 1.0)
        bid = scan_result.criteria.get("bid", 0.0)
        ask = scan_result.criteria.get("ask", 0.0)

        if bid <= 0 or ask <= 0:
            return None

        mid = (bid + ask) / 2.0
        if not (self._min_price <= mid <= self._max_price):
            return None

        if spread_pct > self._max_spread_pct:
            return None

        bars = scan_result.bars
        if not bars or len(bars) < 3:
            logger.info("ENTRY GUARD REJECT %s: no bar data for spread signal", scan_result.symbol)
            self._last_reject = "no bar data for spread signal"
            return None

        reject = check_entry_quality(bars, symbol=scan_result.symbol, now=now)
        if reject is not None:
            logger.info("ENTRY GUARD REJECT %s: %s", scan_result.symbol, reject)
            self._last_reject = reject
            return None

        pos = portfolio.positions.get(scan_result.symbol)
        if pos and not pos.is_flat:
            return None

        entry = ask

        return TradeSignal(
            symbol=scan_result.symbol,
            action=SignalAction.ENTER_LONG,
            quantity=self._size,
            entry_price=entry,
            stop_loss=entry - self._stop,
            take_profit=entry + self._target,
            trailing_stop_offset=self._trail,
            max_hold_seconds=self._max_hold,
            reason="Spread scalp long ${:.2f}, spread={:.4f}%, SL={:.0f}t TP={:.0f}t".format(
                entry, spread_pct,
                self._stop / TICK, self._target / TICK),
            scan_result=scan_result,
        )
