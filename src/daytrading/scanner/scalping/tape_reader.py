"""Tape reader scanner for scalping.

Analyzes time & sales (tick data) for order flow imbalance —
when aggressive buyers heavily outweigh sellers (or vice versa),
price is likely to continue in that direction.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from daytrading.indicators.scalping import cumulative_delta, order_flow_imbalance, tape_speed
from daytrading.models import Bar, ScanResult, Tick


class TapeReaderScanner:
    """Flags symbols with strong order flow imbalance and high tape speed."""

    def __init__(
        self,
        *,
        min_imbalance: float = 0.4,
        min_tape_speed: float = 5.0,
        imbalance_window: int = 50,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._min_imbalance = min_imbalance
        self._min_tape_speed = min_tape_speed
        self._imbalance_window = imbalance_window
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "tape_reader"

    def scan_ticks(self, universe: Dict[str, Sequence[Tick]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, ticks in universe.items():
            if len(ticks) < self._imbalance_window:
                continue
            if not (self._min_price <= ticks[-1].price <= self._max_price):
                continue

            # Price must be actually moving — reject flat/dead ticks
            window_ticks = list(ticks[-self._imbalance_window:])
            prices = [t.price for t in window_ticks]
            price_range = max(prices) - min(prices)
            min_move = max(0.03, window_ticks[-1].price * 0.003)  # at least 3 ticks or 0.3%
            if price_range < min_move:
                continue

            imb = order_flow_imbalance(ticks, window=self._imbalance_window)
            current_imb = imb[-1] if imb else 0.0
            if abs(current_imb) < self._min_imbalance:
                continue

            speed = tape_speed(ticks, window_seconds=5.0)
            current_speed = speed[-1] if speed else 0.0
            if current_speed < self._min_tape_speed:
                continue

            delta = cumulative_delta(ticks)
            current_delta = delta[-1] if delta else 0.0

            latest = ticks[-1]
            results.append(ScanResult(
                symbol=symbol,
                scanner_name=self.name,
                ts=latest.ts,
                score=abs(current_imb) * current_speed,
                criteria={
                    "imbalance": round(current_imb, 3),
                    "tape_speed": round(current_speed, 1),
                    "cum_delta": round(current_delta, 0),
                    "direction": "buy_pressure" if current_imb > 0 else "sell_pressure",
                    "last_price": latest.price,
                },
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        return []
