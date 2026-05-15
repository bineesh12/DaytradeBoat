"""Pre-market gap scanner.

Detects symbols that gapped up (or down) significantly from the previous
session's close to the current session's open — a classic day-trading catalyst.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.models import Bar, ScanResult


class PremarketGapScanner:
    """Flags symbols where today's open is ≥ `min_gap_pct`% above yesterday's close."""

    def __init__(
        self,
        min_gap_pct: float = 3.0,
        min_price: float = 1.0,
        max_price: float = 10_000.0,
        min_volume: float = 50_000,
    ) -> None:
        self._min_gap_pct = min_gap_pct
        self._min_price = min_price
        self._max_price = max_price
        self._min_volume = min_volume

    @property
    def name(self) -> str:
        return "premarket_gap"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, bars in universe.items():
            if len(bars) < 2:
                continue
            prev_close = bars[-2].close
            today_open = bars[-1].open
            today_bar = bars[-1]

            if prev_close <= 0:
                continue
            gap_pct = ((today_open - prev_close) / prev_close) * 100.0
            if abs(gap_pct) < self._min_gap_pct:
                continue
            if not (self._min_price <= today_bar.close <= self._max_price):
                continue
            if today_bar.volume < self._min_volume:
                continue

            results.append(ScanResult(
                symbol=symbol,
                scanner_name=self.name,
                ts=today_bar.ts,
                score=abs(gap_pct),
                criteria={
                    "gap_pct": round(gap_pct, 2),
                    "prev_close": prev_close,
                    "open": today_open,
                    "volume": today_bar.volume,
                },
                bars=list(bars[-5:]),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results
