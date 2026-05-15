"""Momentum burst scanner for scalping.

Detects sharp micro-moves on short-timeframe bars (1s–1m) that
indicate a scalping opportunity is forming.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from daytrading.indicators.scalping import momentum_burst, price_velocity
from daytrading.models import Bar, ScanResult


class MomentumBurstScanner:
    """Flags symbols with a sharp price move over a small number of bars."""

    def __init__(
        self,
        *,
        min_burst_pct: float = 0.8,
        burst_period: int = 3,
        min_velocity: float = 0.05,
        min_volume: float = 10_000,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._min_burst_pct = min_burst_pct
        self._burst_period = burst_period
        self._min_velocity = min_velocity
        self._min_volume = min_volume
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "momentum_burst"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, bars in universe.items():
            if len(bars) < self._burst_period + 1:
                continue

            latest = bars[-1]
            if not (self._min_price <= latest.close <= self._max_price):
                continue
            if latest.volume < self._min_volume:
                continue

            burst = momentum_burst(bars, period=self._burst_period)
            current_burst = burst[-1] if burst else 0.0
            if math.isnan(current_burst):
                continue
            # Only upward bursts — we only buy
            if current_burst < self._min_burst_pct:
                continue

            # Price must move at least 5 ticks ($0.05) in absolute terms
            recent = list(bars[-(self._burst_period + 1):])
            price_move = recent[-1].close - min(b.close for b in recent)
            if price_move < 0.05:
                continue

            vel = price_velocity(bars, period=self._burst_period)
            current_vel = vel[-1] if vel else 0.0
            if math.isnan(current_vel):
                current_vel = 0.0
            if current_vel < self._min_velocity:
                continue

            # Don't trigger on the spike itself — we want the first pullback
            # or consolidation after the spike.
            # If the last bar is a big green candle (body > 2x avg), we're
            # IN the spike — skip and wait for it to settle.
            if len(bars) >= 6:
                last = bars[-1]
                prev_bodies = [abs(b.close - b.open) for b in bars[-6:-1]]
                avg_body = sum(prev_bodies) / len(prev_bodies) if prev_bodies else 0
                last_body = last.close - last.open
                if last_body > 0 and avg_body > 0 and last_body > avg_body * 2.5:
                    continue

                # Also skip if last 3 bars are all green with accelerating closes
                # (we're in the middle of a parabolic run — wait for a red candle)
                last3 = list(bars[-3:])
                if all(b.close > b.open for b in last3):
                    move_pct = (last3[-1].close - last3[0].open) / last3[0].open if last3[0].open > 0 else 0
                    if move_pct > 0.04:
                        continue

            results.append(ScanResult(
                symbol=symbol,
                scanner_name=self.name,
                ts=latest.ts,
                score=current_burst,
                criteria={
                    "burst_pct": round(current_burst, 4),
                    "velocity": round(current_vel, 4),
                    "direction": "up",
                    "close": latest.close,
                    "volume": latest.volume,
                },
                bars=list(bars[-120:]),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results
