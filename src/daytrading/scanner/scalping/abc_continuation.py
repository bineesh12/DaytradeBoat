"""ABC continuation scanner.

Detects a common momentum sequence:
    A: strong push
    B: controlled pullback / consolidation
    C: reclaim above the B high with volume returning
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.models import Bar, ScanResult


class ABCContinuationScanner:
    """Detects A-leg, B-pullback, C-breakout continuation setups."""

    def __init__(
        self,
        *,
        min_a_leg_pct: float = 5.0,
        min_b_retrace: float = 0.20,
        max_b_retrace: float = 0.60,
        min_b_bars: int = 2,
        max_b_bars: int = 8,
        c_volume_surge: float = 1.10,
        max_c_breakout_pct: float = 5.0,
        max_c_bar_range_pct: float = 12.0,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._min_a_leg_pct = min_a_leg_pct
        self._min_b_retrace = min_b_retrace
        self._max_b_retrace = max_b_retrace
        self._min_b_bars = min_b_bars
        self._max_b_bars = max_b_bars
        self._c_volume_surge = c_volume_surge
        self._max_c_breakout_pct = max_c_breakout_pct
        self._max_c_bar_range_pct = max_c_bar_range_pct
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "abc_continuation"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, bars in universe.items():
            if len(bars) < 10:
                continue
            latest = bars[-1]
            if not (self._min_price <= latest.close <= self._max_price):
                continue
            hit = self._detect(symbol, list(bars[-80:]))
            if hit is not None:
                results.append(ScanResult(
                    symbol=hit.symbol,
                    scanner_name=hit.scanner_name,
                    ts=hit.ts,
                    score=hit.score,
                    criteria=hit.criteria,
                    bars=list(bars[-120:]),
                ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _detect(self, symbol: str, bars: List[Bar]) -> ScanResult | None:
        if len(bars) < 10:
            return None

        c_bar = bars[-1]
        if c_bar.close <= c_bar.open:
            return None

        for b_len in range(self._min_b_bars, self._max_b_bars + 1):
            b_end = len(bars) - 2
            b_start = b_end - b_len + 1
            if b_start < 3:
                continue

            b_bars = bars[b_start:b_end + 1]
            b_high = max(b.high for b in b_bars)
            b_low = min(b.low for b in b_bars)
            if b_low <= 0 or b_high <= 0:
                continue

            # C trigger: break/reclaim B high.
            if c_bar.close <= b_high or c_bar.high <= b_high * 1.001:
                continue

            avg_b_vol = sum(b.volume for b in b_bars) / len(b_bars)
            if avg_b_vol > 0 and c_bar.volume < avg_b_vol * self._c_volume_surge:
                continue

            # Pullback should be controlled: mostly lower volume than A and not
            # a sharp dump. B can be red or sideways.
            b_red = sum(1 for b in b_bars if b.close < b.open)
            if b_red < max(1, len(b_bars) // 2):
                continue

            a_end = b_start - 1
            for a_len in range(3, min(12, a_end + 1) + 1):
                a_start = a_end - a_len + 1
                if a_start < 0:
                    continue
                a_bars = bars[a_start:a_end + 1]
                a_low = min(b.low for b in a_bars)
                a_high = max(b.high for b in a_bars)
                if a_low <= 0 or a_high <= a_low:
                    continue

                a_leg_pct = (a_high - a_low) / a_low * 100.0
                if a_leg_pct < self._min_a_leg_pct:
                    continue

                a_green = sum(1 for b in a_bars if b.close > b.open)
                if a_green < len(a_bars) * 0.60:
                    continue

                retrace = (a_high - b_low) / (a_high - a_low)
                if retrace < self._min_b_retrace or retrace > self._max_b_retrace:
                    continue

                avg_a_vol = sum(b.volume for b in a_bars) / len(a_bars)
                if avg_a_vol > 0 and avg_b_vol > avg_a_vol * 1.15:
                    continue

                c_breakout_pct = (c_bar.close - b_high) / b_high * 100.0
                c_bar_range_pct = (
                    (c_bar.high - c_bar.low) / c_bar.close * 100.0
                    if c_bar.close > 0 else 0.0
                )
                if c_breakout_pct > self._max_c_breakout_pct:
                    continue
                if c_bar_range_pct > self._max_c_bar_range_pct:
                    continue
                risk_to_b_low_pct = (c_bar.close - b_low) / c_bar.close * 100.0
                score = a_leg_pct * (1.0 - retrace) + c_breakout_pct * 2.0

                return ScanResult(
                    symbol=symbol,
                    scanner_name=self.name,
                    ts=c_bar.ts,
                    score=round(score, 3),
                    criteria={
                        "pattern": "abc_continuation",
                        "direction": "up",
                        "a_leg_pct": round(a_leg_pct, 2),
                        "a_bars": a_len,
                        "a_high": round(a_high, 4),
                        "a_low": round(a_low, 4),
                        "b_bars": b_len,
                        "b_high": round(b_high, 4),
                        "b_low": round(b_low, 4),
                        "b_retrace_pct": round(retrace * 100.0, 1),
                        "c_breakout_pct": round(c_breakout_pct, 2),
                        "c_bar_range_pct": round(c_bar_range_pct, 2),
                        "c_volume_surge": round(c_bar.volume / avg_b_vol, 2) if avg_b_vol > 0 else 0.0,
                        "risk_to_b_low_pct": round(risk_to_b_low_pct, 2),
                        "close": c_bar.close,
                        "volume": c_bar.volume,
                    },
                    bars=[],
                )

        return None
