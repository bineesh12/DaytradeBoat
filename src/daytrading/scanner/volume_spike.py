"""Volume spike scanner.

Detects symbols with unusually high relative volume compared to their
recent average — indicates institutional interest or breaking news.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.indicators.core import relative_volume
from daytrading.models import Bar, ScanResult


class VolumeSpikeScanner:
    """Flags symbols whose latest bar volume is ≥ `min_rvol`× their average."""

    def __init__(
        self,
        min_rvol: float = 2.0,
        lookback: int = 20,
        min_price: float = 1.0,
        min_avg_volume: float = 100_000,
    ) -> None:
        self._min_rvol = min_rvol
        self._lookback = lookback
        self._min_price = min_price
        self._min_avg_volume = min_avg_volume

    @property
    def name(self) -> str:
        return "volume_spike"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, bars in universe.items():
            if len(bars) < self._lookback + 1:
                continue
            latest = bars[-1]
            if latest.close < self._min_price:
                continue

            avg_vol = sum(b.volume for b in bars[-(self._lookback + 1):-1]) / self._lookback
            if avg_vol < self._min_avg_volume:
                continue

            rvol = latest.volume / avg_vol if avg_vol > 0 else 0.0
            if rvol < self._min_rvol:
                continue

            results.append(ScanResult(
                symbol=symbol,
                scanner_name=self.name,
                ts=latest.ts,
                score=rvol,
                criteria={
                    "rvol": round(rvol, 2),
                    "volume": latest.volume,
                    "avg_volume": round(avg_vol, 0),
                    "close": latest.close,
                },
                bars=list(bars[-5:]),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results
