"""Spread filter scanner for scalping.

Finds symbols with tight, stable spreads — a prerequisite for profitable
scalping. Wide or erratic spreads eat into the tiny per-trade edge.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from daytrading.indicators.scalping import avg_spread, spread_compression_ratio
from daytrading.models import Bar, Quote, ScanResult


class SpreadFilterScanner:
    """Screens for symbols with spread ≤ max and compressing (tightening)."""

    def __init__(
        self,
        *,
        max_spread_cents: float = 2.0,
        max_spread_pct: float = 0.15,  # wider for $1-$20 stocks
        max_compression_ratio: float = 0.85,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._max_spread_cents = max_spread_cents
        self._max_spread_pct = max_spread_pct
        self._max_compression = max_compression_ratio
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "spread_filter"

    def scan_quotes(self, universe: Dict[str, Sequence[Quote]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, quotes in universe.items():
            if len(quotes) < 20:
                continue

            latest = quotes[-1]
            if not (self._min_price <= latest.mid <= self._max_price):
                continue
            if latest.spread > self._max_spread_cents:
                continue
            if latest.spread_pct > self._max_spread_pct:
                continue

            cr = spread_compression_ratio(quotes, short=5, long=20)
            current_cr = cr[-1] if cr else 1.0
            if math.isnan(current_cr):
                current_cr = 1.0
            if current_cr > self._max_compression:
                continue

            avg_s = avg_spread(quotes, window=10)
            current_avg = avg_s[-1] if avg_s else latest.spread

            results.append(ScanResult(
                symbol=symbol,
                scanner_name=self.name,
                ts=latest.ts,
                score=1.0 / max(latest.spread, 0.001),
                criteria={
                    "spread": round(latest.spread, 4),
                    "spread_pct": round(latest.spread_pct, 4),
                    "avg_spread": round(current_avg if not math.isnan(current_avg) else latest.spread, 4),
                    "compression_ratio": round(current_cr, 3),
                    "bid": latest.bid,
                    "ask": latest.ask,
                },
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        return []
