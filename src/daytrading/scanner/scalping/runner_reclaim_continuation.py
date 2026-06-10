"""Runner-watch reclaim continuation scanner.

Detects early runner pullbacks that are wider than normal first-pullback
setups, but still reclaim a recent base while holding above VWAP.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.indicators.core import vwap
from daytrading.models import Bar, ScanResult


class RunnerReclaimContinuationScanner:
    """Find controlled VWAP/base reclaims on volatile runner-watch names."""

    def __init__(
        self,
        *,
        min_impulse_pct: float = 12.0,
        min_pullback_pct: float = 3.0,
        max_pullback_pct: float = 35.0,
        max_base_range_pct: float = 18.0,
        max_extension_from_base_pct: float = 8.0,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self._min_impulse_pct = min_impulse_pct
        self._min_pullback_pct = min_pullback_pct
        self._max_pullback_pct = max_pullback_pct
        self._max_base_range_pct = max_base_range_pct
        self._max_extension_from_base_pct = max_extension_from_base_pct
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "runner_reclaim_continuation"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, seq in universe.items():
            bars = list(seq)
            if len(bars) < 10:
                continue
            latest = bars[-1]
            if not (self._min_price <= latest.close <= self._max_price):
                continue
            hit = self._detect(symbol, bars[-80:], session_bars=bars)
            if hit is not None:
                results.append(hit)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _detect(
        self,
        symbol: str,
        bars: List[Bar],
        *,
        session_bars: Sequence[Bar] | None = None,
    ) -> ScanResult | None:
        if len(bars) < 10:
            return None

        latest = bars[-1]
        if latest.close <= latest.open:
            return None

        session = list(session_bars or bars)
        session_open = session[0].open
        if session_open <= 0:
            return None

        prior = bars[:-1]
        session_high = max(b.high for b in session)
        impulse_high = max(b.high for b in prior)
        impulse_idx = max(i for i, b in enumerate(prior) if b.high >= impulse_high)
        impulse_pct = (impulse_high - session_open) / session_open * 100.0
        if impulse_pct < self._min_impulse_pct:
            return None

        after_impulse = prior[impulse_idx + 1:]
        if len(after_impulse) < 2:
            return None

        pullback_low = min(b.low for b in after_impulse)
        if pullback_low <= 0:
            return None
        pullback_pct = (impulse_high - pullback_low) / impulse_high * 100.0
        if pullback_pct < self._min_pullback_pct:
            return None
        if pullback_pct > self._max_pullback_pct:
            return None

        pullback_low_idx = min(
            i for i, b in enumerate(after_impulse) if b.low <= pullback_low
        )
        base_source = after_impulse[pullback_low_idx:]
        base = base_source[-min(6, len(base_source)):]
        if len(base) < 2:
            return None
        base_high = max(b.high for b in base)
        base_low = min(b.low for b in base)
        if base_low <= 0:
            return None
        base_range_pct = (base_high - base_low) / base_low * 100.0
        if base_range_pct > self._max_base_range_pct:
            return None

        reclaim_pct = (latest.close - base_high) / base_high * 100.0
        if reclaim_pct < 0.4:
            return None
        if reclaim_pct > self._max_extension_from_base_pct:
            return None

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap > 0:
            if latest.close < current_vwap:
                return None
            # Runner reclaims can pierce VWAP, but should not lose it deeply.
            if pullback_low < current_vwap * 0.94:
                return None

        avg_base_vol = sum(float(b.volume or 0.0) for b in base) / len(base)
        if avg_base_vol > 0 and float(latest.volume or 0.0) < avg_base_vol * 0.75:
            return None

        risk_to_base_low_pct = (latest.close - base_low) / latest.close * 100.0
        if risk_to_base_low_pct > 18.0:
            return None

        recent_high = session_high
        pullback_from_hod_pct = (
            (recent_high - latest.close) / recent_high * 100.0
            if recent_high > 0 else 0.0
        )
        if pullback_from_hod_pct < -0.5:
            return None

        score = (
            impulse_pct * 0.7
            + pullback_pct * 1.5
            + reclaim_pct * 4.0
            - max(0.0, base_range_pct - 10.0)
        )

        return ScanResult(
            symbol=symbol,
            scanner_name=self.name,
            ts=latest.ts,
            score=round(score, 3),
            criteria={
                "pattern": "runner_reclaim_continuation",
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
                "pullback_from_hod_pct": round(pullback_from_hod_pct, 2),
                "stop_price": round(base_low - 0.02, 4),
                "close": latest.close,
                "volume": latest.volume,
            },
            bars=list(bars[-80:]),
        )
