"""VWAP pullback scanner.

Detects the first pullback to VWAP after an initial run-up:
1. Stock must have rallied (made a strong move up from open)
2. Price pulls back to touch or slightly pierce VWAP
3. Bounce candle: a green candle off VWAP = entry signal

Classic Warrior Trading entry — buying the first dip to support.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.indicators.core import vwap
from daytrading.models import Bar, ScanResult


class VWAPPullbackScanner:
    """Detects first pullback to VWAP after a strong move up."""

    def __init__(
        self,
        *,
        min_rally_pct: float = 3.0,
        vwap_touch_tolerance: float = 0.005,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._min_rally_pct = min_rally_pct
        self._vwap_touch_tolerance = vwap_touch_tolerance
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "vwap_pullback"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, bars in universe.items():
            if len(bars) < 10:
                continue

            latest = bars[-1]
            if not (self._min_price <= latest.close <= self._max_price):
                continue

            hit = self._detect(symbol, list(bars[-120:]))
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

    def _detect(self, symbol: str, bars: List[Bar]) -> ScanResult | None:
        if len(bars) < 10:
            return None

        latest = bars[-1]
        prev = bars[-2]
        vwap_vals = vwap(bars)
        if not vwap_vals or len(vwap_vals) < len(bars):
            return None

        current_vwap = vwap_vals[-1]
        if current_vwap <= 0:
            return None

        # Latest bar must be green (bounce)
        if latest.close <= latest.open:
            return None

        # Latest bar must close above VWAP (bouncing off it)
        if latest.close < current_vwap:
            return None

        # Previous bar or the low of latest bar must have touched VWAP
        prev_vwap = vwap_vals[-2]
        tolerance = current_vwap * self._vwap_touch_tolerance
        touched = (
            prev.low <= prev_vwap + tolerance
            or latest.low <= current_vwap + tolerance
        )
        if not touched:
            return None

        # Must have rallied first — find the high of day before the pullback
        session_high = max(b.high for b in bars)
        session_open = bars[0].open
        if session_open <= 0:
            return None

        rally_pct = (session_high - session_open) / session_open * 100
        if rally_pct < self._min_rally_pct:
            return None

        # Price must have pulled back from the high (not still at the top)
        pullback_depth = (session_high - prev.low) / session_high * 100
        if pullback_depth < 0.5:
            return None

        # Stop below the VWAP touch low
        stop_price = min(prev.low, latest.low) - 0.02
        score = rally_pct * (1 + pullback_depth / 10)

        return ScanResult(
            symbol=symbol,
            scanner_name=self.name,
            ts=latest.ts,
            score=round(score, 3),
            criteria={
                "pattern": "vwap_pullback",
                "direction": "up",
                "session_high": round(session_high, 4),
                "rally_pct": round(rally_pct, 2),
                "vwap": round(current_vwap, 4),
                "pullback_low": round(min(prev.low, latest.low), 4),
                "stop_price": round(stop_price, 4),
                "close": latest.close,
                "volume": latest.volume,
            },
            bars=[],
        )
