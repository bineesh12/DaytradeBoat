"""High-of-Day (HOD) reclaim scanner.

Detects when a stock reclaims its high of day after pulling back:
1. Stock made a clear high of day earlier in the session
2. Pulled back significantly from that high
3. Now a green candle reclaims (closes at or above) the previous HOD

This triggers short-covering and new momentum buying as the stock
proves it can push through the previous resistance.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.models import Bar, ScanResult


class HODReclaimScanner:
    """Detects high-of-day reclaim after a pullback."""

    def __init__(
        self,
        *,
        min_pullback_pct: float = 1.5,
        min_rally_from_open_pct: float = 3.0,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._min_pullback_pct = min_pullback_pct
        self._min_rally_pct = min_rally_from_open_pct
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "hod_reclaim"

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

        # Latest bar must be green (reclaim candle)
        if latest.close <= latest.open:
            return None

        # Find HOD from all bars except the latest
        prev_bars = bars[:-1]
        hod = max(b.high for b in prev_bars)
        session_open = bars[0].open
        if session_open <= 0 or hod <= 0:
            return None

        # Must have had a significant rally from open to HOD
        rally_pct = (hod - session_open) / session_open * 100
        if rally_pct < self._min_rally_pct:
            return None

        # Find the HOD bar index, then find the lowest low AFTER the HOD
        hod_idx = 0
        for i, b in enumerate(prev_bars):
            if b.high >= hod:
                hod_idx = i

        # Need bars between HOD and current bar for the pullback
        if hod_idx >= len(prev_bars) - 2:
            return None

        pullback_bars = prev_bars[hod_idx + 1:]
        if not pullback_bars:
            return None

        pullback_low = min(b.low for b in pullback_bars)
        pullback_pct = (hod - pullback_low) / hod * 100
        if pullback_pct < self._min_pullback_pct:
            return None

        # Reclaim: latest bar must close at or above the previous HOD
        if latest.close < hod * 0.998:
            return None

        # Latest bar's high must make a new HOD or come very close
        if latest.high < hod * 0.999:
            return None

        stop_price = pullback_low - 0.02
        reclaim_strength = (latest.close - hod) / hod * 100
        score = rally_pct + pullback_pct + reclaim_strength * 5

        return ScanResult(
            symbol=symbol,
            scanner_name=self.name,
            ts=latest.ts,
            score=round(score, 3),
            criteria={
                "pattern": "hod_reclaim",
                "direction": "up",
                "hod": round(hod, 4),
                "pullback_low": round(pullback_low, 4),
                "rally_pct": round(rally_pct, 2),
                "pullback_pct": round(pullback_pct, 2),
                "stop_price": round(stop_price, 4),
                "close": latest.close,
                "volume": latest.volume,
            },
            bars=[],
        )
