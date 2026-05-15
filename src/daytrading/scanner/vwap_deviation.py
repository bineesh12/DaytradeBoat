"""VWAP deviation scanner.

Flags symbols trading significantly above or below their session VWAP —
used for mean-reversion or trend-continuation setups.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.indicators.core import vwap as calc_vwap
from daytrading.models import Bar, ScanResult


class VWAPDeviationScanner:
    """Flags symbols where price is ≥ `min_dev_pct`% away from VWAP."""

    def __init__(
        self,
        min_dev_pct: float = 2.0,
        min_price: float = 1.0,
        min_volume: float = 100_000,
    ) -> None:
        self._min_dev_pct = min_dev_pct
        self._min_price = min_price
        self._min_volume = min_volume

    @property
    def name(self) -> str:
        return "vwap_deviation"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, bars in universe.items():
            if len(bars) < 2:
                continue
            latest = bars[-1]
            if latest.close < self._min_price:
                continue
            if latest.volume < self._min_volume:
                continue

            vwap_values = calc_vwap(bars)
            current_vwap = vwap_values[-1]
            if current_vwap <= 0:
                continue

            dev_pct = ((latest.close - current_vwap) / current_vwap) * 100.0
            if abs(dev_pct) < self._min_dev_pct:
                continue

            results.append(ScanResult(
                symbol=symbol,
                scanner_name=self.name,
                ts=latest.ts,
                score=abs(dev_pct),
                criteria={
                    "dev_pct": round(dev_pct, 2),
                    "vwap": round(current_vwap, 4),
                    "close": latest.close,
                    "direction": "above" if dev_pct > 0 else "below",
                },
                bars=list(bars[-5:]),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results
