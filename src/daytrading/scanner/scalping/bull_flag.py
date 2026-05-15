"""Bull Flag pattern scanner.

Detects the Warrior Trading bull flag pattern:
1. Strong opening drive (pole) — sharp move up on high volume
2. Consolidation pullback (flag) — 2-3 red/small candles on low volume
3. Breakout candle — first green candle making a new high after the pullback

The scanner fires when the breakout candle is detected.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from daytrading.indicators.core import relative_volume
from daytrading.models import Bar, ScanResult


class BullFlagScanner:
    """Detects bull flag breakout setups on 1-min bars."""

    def __init__(
        self,
        *,
        min_pole_pct: float = 1.5,
        min_pole_bars: int = 2,
        max_pole_bars: int = 8,
        min_pullback_bars: int = 2,
        max_pullback_bars: int = 6,
        max_pullback_retrace: float = 0.50,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._min_pole_pct = min_pole_pct
        self._min_pole_bars = min_pole_bars
        self._max_pole_bars = max_pole_bars
        self._min_pullback_bars = min_pullback_bars
        self._max_pullback_bars = max_pullback_bars
        self._max_pullback_retrace = max_pullback_retrace
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "bull_flag"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, bars in universe.items():
            if len(bars) < 10:
                continue

            latest = bars[-1]
            if not (self._min_price <= latest.close <= self._max_price):
                continue

            hit = self._detect_bull_flag(symbol, list(bars[-60:]))
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

    def _detect_bull_flag(
        self, symbol: str, bars: List[Bar]
    ) -> ScanResult | None:
        """Walk backwards to find pole → pullback → breakout."""
        if len(bars) < 8:
            return None

        latest = bars[-1]
        # Breakout candle must be green and making a new high
        if latest.close <= latest.open:
            return None

        # Search for the pullback zone ending at bar[-2]
        for pb_len in range(self._min_pullback_bars, self._max_pullback_bars + 1):
            pb_end = len(bars) - 2
            pb_start = pb_end - pb_len + 1
            if pb_start < 1:
                continue

            pullback = bars[pb_start : pb_end + 1]
            pole_end_idx = pb_start - 1

            # Pullback bars: mostly red/flat, lower volume than the pole
            red_count = sum(1 for b in pullback if b.close <= b.open)
            if red_count < len(pullback) * 0.5:
                continue

            # Search for the pole before the pullback
            for pole_len in range(
                self._min_pole_bars, self._max_pole_bars + 1
            ):
                pole_start_idx = pole_end_idx - pole_len + 1
                if pole_start_idx < 0:
                    continue

                pole = bars[pole_start_idx : pole_end_idx + 1]

                pole_low = min(b.low for b in pole)
                pole_high = max(b.high for b in pole)
                if pole_low <= 0:
                    continue

                pole_move_pct = (pole_high - pole_low) / pole_low * 100
                if pole_move_pct < self._min_pole_pct:
                    continue

                # Pole should be mostly green
                pole_green = sum(1 for b in pole if b.close > b.open)
                if pole_green < len(pole) * 0.6:
                    continue

                # Pullback retracement: how much of the pole did the pullback give back?
                pullback_low = min(b.low for b in pullback)
                retrace = (pole_high - pullback_low) / (pole_high - pole_low) if (pole_high - pole_low) > 0 else 1.0
                if retrace > self._max_pullback_retrace:
                    continue

                # Pullback volume should be lower than pole volume
                avg_pole_vol = sum(b.volume for b in pole) / len(pole)
                avg_pb_vol = sum(b.volume for b in pullback) / len(pullback)
                if avg_pb_vol > avg_pole_vol * 1.2:
                    continue

                # Breakout: latest bar must exceed the pullback high
                pb_high = max(b.high for b in pullback)
                if latest.high <= pb_high:
                    continue

                # Volume confirmation on breakout candle
                if latest.volume < avg_pb_vol * 1.2:
                    continue

                score = pole_move_pct * (1 - retrace)

                return ScanResult(
                    symbol=symbol,
                    scanner_name=self.name,
                    ts=latest.ts,
                    score=round(score, 3),
                    criteria={
                        "pattern": "bull_flag",
                        "direction": "up",
                        "pole_pct": round(pole_move_pct, 2),
                        "retrace_pct": round(retrace * 100, 1),
                        "pole_bars": pole_len,
                        "pullback_bars": pb_len,
                        "breakout_price": latest.close,
                        "pole_high": pole_high,
                        "pullback_low": pullback_low,
                        "close": latest.close,
                        "volume": latest.volume,
                    },
                    bars=[],
                )

        return None
