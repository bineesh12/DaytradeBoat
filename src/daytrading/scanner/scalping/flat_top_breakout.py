"""Flat Top Breakout pattern scanner.

Detects the Warrior Trading flat top breakout pattern:
1. Strong move up (initial drive)
2. Consolidation at a resistance level — multiple bars with highs at roughly
   the same price (the "flat top")
3. Breakout candle — price breaks above the flat top resistance on volume

Short sellers stack stop orders above the resistance, so the breakout
triggers a cascade of buy orders.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Sequence

from daytrading.models import Bar, ScanResult


class FlatTopBreakoutScanner:
    """Detects flat top breakout setups on 1-min bars."""

    def __init__(
        self,
        *,
        min_drive_pct: float = 1.0,
        min_flat_bars: int = 3,
        max_flat_bars: int = 10,
        flat_tolerance_pct: float = 0.3,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._min_drive_pct = min_drive_pct
        self._min_flat_bars = min_flat_bars
        self._max_flat_bars = max_flat_bars
        self._flat_tolerance_pct = flat_tolerance_pct
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "flat_top_breakout"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, bars in universe.items():
            if len(bars) < 10:
                continue

            latest = bars[-1]
            if not (self._min_price <= latest.close <= self._max_price):
                continue

            hit = self._detect_flat_top(symbol, list(bars[-60:]))
            if hit is not None:
                hit_with_bars = ScanResult(
                    symbol=hit.symbol,
                    scanner_name=hit.scanner_name,
                    ts=hit.ts,
                    score=hit.score,
                    criteria=hit.criteria,
                    bars=list(bars[-120:]),
                )
                results.append(hit_with_bars)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _detect_flat_top(
        self, symbol: str, bars: List[Bar]
    ) -> ScanResult | None:
        if len(bars) < 8:
            return None

        latest = bars[-1]
        # Breakout candle must be green
        if latest.close <= latest.open:
            return None

        # Search for a flat top resistance zone ending near bar[-2]
        for flat_len in range(self._min_flat_bars, self._max_flat_bars + 1):
            flat_end = len(bars) - 2
            flat_start = flat_end - flat_len + 1
            if flat_start < 2:
                continue

            flat_bars = bars[flat_start : flat_end + 1]
            highs = [b.high for b in flat_bars]

            if not highs:
                continue

            resistance = max(highs)
            if resistance <= 0:
                continue

            # Check that highs are "flat" — all within tolerance of resistance
            spread = (resistance - min(highs)) / resistance * 100
            if spread > self._flat_tolerance_pct:
                continue

            # Must have had an initial drive up before the flat zone
            drive_start_idx = max(0, flat_start - 8)
            drive_bars = bars[drive_start_idx:flat_start]
            if len(drive_bars) < 2:
                continue

            drive_low = min(b.low for b in drive_bars)
            drive_high = max(b.high for b in drive_bars)
            if drive_low <= 0:
                continue

            drive_pct = (drive_high - drive_low) / drive_low * 100
            if drive_pct < self._min_drive_pct:
                continue

            # Breakout: latest bar must close above the resistance
            if latest.close <= resistance:
                continue

            # Breakout candle high must clearly exceed resistance
            if latest.high <= resistance * 1.001:
                continue

            # Volume on breakout should be higher than flat zone average
            avg_flat_vol = sum(b.volume for b in flat_bars) / len(flat_bars) if flat_bars else 0
            if avg_flat_vol > 0 and latest.volume < avg_flat_vol * 0.8:
                continue

            breakout_pct = (latest.close - resistance) / resistance * 100
            score = drive_pct + breakout_pct * 2

            return ScanResult(
                symbol=symbol,
                scanner_name=self.name,
                ts=latest.ts,
                score=round(score, 3),
                criteria={
                    "pattern": "flat_top_breakout",
                    "direction": "up",
                    "resistance": round(resistance, 4),
                    "drive_pct": round(drive_pct, 2),
                    "breakout_pct": round(breakout_pct, 2),
                    "flat_bars": flat_len,
                    "close": latest.close,
                    "volume": latest.volume,
                },
                bars=[],
            )

        return None
