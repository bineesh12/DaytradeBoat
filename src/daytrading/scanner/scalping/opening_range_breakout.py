"""Opening Range Breakout (ORB) scanner.

Detects breakout above the first 5-minute high:
1. Track the high and low of the first 5 one-minute bars (opening range)
2. Wait for a candle that breaks above the opening range high
3. Breakout bar must be green and close above the range high

Classic Warrior Trading entry — the first 5-minute range captures the
initial battle between buyers and sellers. A breakout above it signals
momentum continuation.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Sequence

from daytrading.models import Bar, ScanResult


class OpeningRangeBreakoutScanner:
    """Detects breakout above the first 5-minute opening range."""

    def __init__(
        self,
        *,
        orb_bars: int = 5,
        min_range_pct: float = 0.5,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._orb_bars = orb_bars
        self._min_range_pct = min_range_pct
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "opening_range_breakout"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, bars in universe.items():
            if len(bars) < self._orb_bars + 2:
                continue

            latest = bars[-1]
            if not (self._min_price <= latest.close <= self._max_price):
                continue

            hit = self._detect(symbol, list(bars))
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
        if len(bars) < self._orb_bars + 2:
            return None

        # Identify the opening range bars (first N bars of the session)
        # Use the earliest bar's date to find session start
        first_bar = bars[0]
        if first_bar.ts is None:
            return None

        session_date = first_bar.ts.date()
        orb_bars = []
        post_orb_bars = []

        for b in bars:
            if b.ts is None:
                continue
            if b.ts.date() != session_date:
                continue
            if len(orb_bars) < self._orb_bars:
                orb_bars.append(b)
            else:
                post_orb_bars.append(b)

        if len(orb_bars) < self._orb_bars or len(post_orb_bars) < 1:
            return None

        orb_high = max(b.high for b in orb_bars)
        orb_low = min(b.low for b in orb_bars)
        if orb_low <= 0:
            return None

        range_pct = (orb_high - orb_low) / orb_low * 100
        if range_pct < self._min_range_pct:
            return None

        latest = post_orb_bars[-1]

        # Breakout candle must be green
        if latest.close <= latest.open:
            return None

        # Must break above the opening range high
        if latest.close <= orb_high:
            return None
        if latest.high <= orb_high:
            return None

        # Must not have already broken out earlier (we want the FIRST breakout)
        already_broke = False
        for b in post_orb_bars[:-1]:
            if b.close > orb_high and b.close > b.open:
                already_broke = True
                break
        if already_broke:
            return None

        breakout_pct = (latest.close - orb_high) / orb_high * 100
        score = range_pct + breakout_pct * 3

        return ScanResult(
            symbol=symbol,
            scanner_name=self.name,
            ts=latest.ts,
            score=round(score, 3),
            criteria={
                "pattern": "opening_range_breakout",
                "direction": "up",
                "orb_high": round(orb_high, 4),
                "orb_low": round(orb_low, 4),
                "range_pct": round(range_pct, 2),
                "breakout_pct": round(breakout_pct, 2),
                "close": latest.close,
                "volume": latest.volume,
            },
            bars=[],
        )
