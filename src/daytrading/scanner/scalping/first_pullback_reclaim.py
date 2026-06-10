"""First pullback reclaim scanner.

Detects the early continuation setup after a fresh momentum push:
1. Stock makes an initial impulse from the session open.
2. It pulls back in a controlled way and holds above VWAP.
3. Latest candle reclaims the pullback/base high.

This is for SVCO-style trades where the best entry is before a late HOD chase.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.indicators.core import vwap
from daytrading.models import Bar, ScanResult


class FirstPullbackReclaimScanner:
    """Detects first controlled pullback reclaim after an opening impulse."""

    def __init__(
        self,
        *,
        min_impulse_pct: float = 5.0,
        min_pullback_pct: float = 1.2,
        max_pullback_pct: float = 12.0,
        max_base_range_pct: float = 6.0,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._min_impulse_pct = min_impulse_pct
        self._min_pullback_pct = min_pullback_pct
        self._max_pullback_pct = max_pullback_pct
        self._max_base_range_pct = max_base_range_pct
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "first_pullback_reclaim"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, bars in universe.items():
            if len(bars) < 12:
                continue
            latest = bars[-1]
            if not (self._min_price <= latest.close <= self._max_price):
                continue

            hit = self._detect(symbol, list(bars[-120:]))
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
        if len(bars) < 12:
            return None

        latest = bars[-1]
        if latest.close <= latest.open:
            return None

        session_open = bars[0].open
        if session_open <= 0:
            return None

        prev_bars = bars[:-1]
        impulse_high = max(b.high for b in prev_bars)
        impulse_idx = max(i for i, b in enumerate(prev_bars) if b.high >= impulse_high)
        impulse_pct = (impulse_high - session_open) / session_open * 100.0
        if impulse_pct < self._min_impulse_pct:
            return None

        after_impulse = prev_bars[impulse_idx + 1:]
        if len(after_impulse) < 2:
            return None

        pullback_low = min(b.low for b in after_impulse)
        if pullback_low <= 0:
            return None
        pullback_low_idx = min(
            i for i, b in enumerate(after_impulse) if b.low <= pullback_low
        )
        pullback_pct = (impulse_high - pullback_low) / impulse_high * 100.0
        if pullback_pct < self._min_pullback_pct:
            return None
        if pullback_pct > self._max_pullback_pct:
            return None

        # The setup base starts after the pullback low forms. Do not include the
        # first red candle from the high, otherwise the scanner waits for a
        # late HOD-style reclaim instead of the first consolidation reclaim.
        base_source = after_impulse[pullback_low_idx:]
        if len(base_source) < 2:
            return None
        base = base_source[-min(5, len(base_source)):]
        base_high = max(b.high for b in base)
        base_low = min(b.low for b in base)
        if base_low <= 0:
            return None
        base_range_pct = (base_high - base_low) / base_low * 100.0
        if base_range_pct > self._max_base_range_pct:
            return None

        if latest.close <= base_high * 1.001 or latest.high <= base_high * 1.001:
            return None

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap > 0:
            if latest.close < current_vwap:
                return None
            # First-pullback continuation should hold VWAP cleanly.
            if pullback_low < current_vwap * 0.985:
                return None

        avg_base_vol = sum(b.volume for b in base) / len(base)
        if avg_base_vol > 0 and latest.volume < avg_base_vol * 0.8:
            return None

        risk_to_base_low_pct = (latest.close - base_low) / latest.close * 100.0
        if risk_to_base_low_pct > 8.0:
            return None

        reclaim_pct = (latest.close - base_high) / base_high * 100.0
        score = impulse_pct * 0.8 + pullback_pct * 2.0 + reclaim_pct * 5.0

        return ScanResult(
            symbol=symbol,
            scanner_name=self.name,
            ts=latest.ts,
            score=round(score, 3),
            criteria={
                "pattern": "first_pullback_reclaim",
                "direction": "up",
                "impulse_high": round(impulse_high, 4),
                "impulse_pct": round(impulse_pct, 2),
                "pullback_low": round(pullback_low, 4),
                "pullback_pct": round(pullback_pct, 2),
                "base_high": round(base_high, 4),
                "base_low": round(base_low, 4),
                "base_range_pct": round(base_range_pct, 2),
                "vwap": round(current_vwap, 4) if current_vwap > 0 else 0.0,
                "reclaim_pct": round(reclaim_pct, 2),
                "risk_to_base_low_pct": round(risk_to_base_low_pct, 2),
                "stop_price": round(base_low - 0.02, 4),
                "close": latest.close,
                "volume": latest.volume,
            },
            bars=[],
        )
