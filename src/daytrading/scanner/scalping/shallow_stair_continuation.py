"""Shallow stair-step continuation scanner.

Some strong runners never give the classic 3%+ pullback. They walk higher in
small bases above VWAP, then break the next level. This scanner catches that
specific continuation shape with a tight tactical stop and reduced sizing.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.indicators.core import vwap
from daytrading.models import Bar, ScanResult


class ShallowStairContinuationScanner:
    """Detect strong runners that continue through shallow stair-step bases."""

    def __init__(
        self,
        *,
        min_price: float = 1.0,
        max_price: float = 20.0,
        min_day_move_pct: float = 20.0,
        max_pullback_from_hod_pct: float = 4.0,
        max_base_range_pct: float = 7.0,
        min_recent_volume: float = 150_000,
        min_volume_surge: float = 0.85,
        elite_min_day_move_pct: float = 50.0,
        elite_max_pullback_from_hod_pct: float = 12.0,
        elite_max_base_range_pct: float = 13.0,
        elite_min_recent_volume: float = 250_000,
        elite_min_volume_surge: float = 0.65,
    ) -> None:
        self._min_price = min_price
        self._max_price = max_price
        self._min_day_move_pct = min_day_move_pct
        self._max_pullback_from_hod_pct = max_pullback_from_hod_pct
        self._max_base_range_pct = max_base_range_pct
        self._min_recent_volume = min_recent_volume
        self._min_volume_surge = min_volume_surge
        self._elite_min_day_move_pct = elite_min_day_move_pct
        self._elite_max_pullback_from_hod_pct = elite_max_pullback_from_hod_pct
        self._elite_max_base_range_pct = elite_max_base_range_pct
        self._elite_min_recent_volume = elite_min_recent_volume
        self._elite_min_volume_surge = elite_min_volume_surge

    @property
    def name(self) -> str:
        return "shallow_stair_continuation"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, seq in universe.items():
            bars = list(seq)
            if len(bars) < 12:
                continue
            latest = bars[-1]
            if not (self._min_price <= latest.close <= self._max_price):
                continue
            hit = self._detect(symbol, bars[-90:], session_bars=bars)
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
        latest = bars[-1]
        price = float(latest.close or 0.0)
        if price <= 0 or latest.close <= latest.open:
            return None

        session = list(session_bars or bars)
        session_open = float(session[0].open or 0.0)
        session_high = max(float(b.high or 0.0) for b in session)
        if session_open <= 0 or session_high <= 0:
            return None

        day_move_pct = (session_high - session_open) / session_open * 100.0
        if day_move_pct < self._min_day_move_pct:
            return None

        recent = bars[-3:]
        recent_volume = sum(float(b.volume or 0.0) for b in recent)
        elite_runner = (
            day_move_pct >= self._elite_min_day_move_pct
            and recent_volume >= self._elite_min_recent_volume
            and 2.0 <= price <= self._max_price
        )

        pullback_from_hod_pct = (session_high - price) / session_high * 100.0
        allowed_hod_pullback_pct = (
            self._elite_max_pullback_from_hod_pct
            if elite_runner else self._max_pullback_from_hod_pct
        )
        if pullback_from_hod_pct > allowed_hod_pullback_pct:
            return None

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap <= 0 or price < current_vwap * 1.01:
            return None

        prior = bars[:-1]
        allowed_base_range_pct = (
            self._elite_max_base_range_pct
            if elite_runner else self._max_base_range_pct
        )
        best_base = None
        for width in (3, 4, 5):
            if len(prior) < width:
                continue
            candidate = list(prior[-width:])
            candidate_high = max(float(b.high or 0.0) for b in candidate)
            candidate_low = min(float(b.low or 0.0) for b in candidate)
            if candidate_high <= 0 or candidate_low <= 0:
                continue
            candidate_range_pct = (candidate_high - candidate_low) / candidate_low * 100.0
            if candidate_range_pct <= allowed_base_range_pct + 1e-6:
                best_base = (candidate, candidate_high, candidate_low, candidate_range_pct)
                break

        if best_base is None:
            return None
        base, base_high, base_low, base_range_pct = best_base

        if price <= base_high * 1.002:
            return None
        if latest.low < base_low * 0.985:
            return None

        min_recent_volume = (
            self._elite_min_recent_volume if elite_runner else self._min_recent_volume
        )
        if recent_volume < min_recent_volume:
            return None

        base_avg_vol = sum(float(b.volume or 0.0) for b in base) / len(base)
        volume_surge = float(latest.volume or 0.0) / base_avg_vol if base_avg_vol > 0 else 0.0
        min_volume_surge = (
            self._elite_min_volume_surge if elite_runner else self._min_volume_surge
        )
        if volume_surge < min_volume_surge:
            return None

        candle_range = float(latest.high or 0.0) - float(latest.low or 0.0)
        body_ratio = (
            (float(latest.close or 0.0) - float(latest.open or 0.0)) / candle_range
            if candle_range > 0 else 0.0
        )
        if body_ratio < 0.25:
            return None

        stop_price = max(base_low - 0.02, price * 0.94)
        risk_pct = (price - stop_price) / price * 100.0
        if risk_pct < 0.5 or risk_pct > 6.0 + 1e-6:
            return None

        breakout_pct = (price - base_high) / base_high * 100.0
        score = (
            day_move_pct * 0.75
            + min(volume_surge, 4.0) * 10.0
            + min(recent_volume / 100_000.0, 6.0) * 4.0
            + breakout_pct * 6.0
            - base_range_pct * 0.5
            - pullback_from_hod_pct * 0.8
        )

        return ScanResult(
            symbol=symbol,
            scanner_name=self.name,
            ts=latest.ts,
            score=round(score, 3),
            criteria={
                "pattern": "shallow_stair_continuation",
                "direction": "up",
                "setup_tier": "A+ setup",
                "entry_tier": "stair_scout",
                "entry_tier_reason": "strong runner shallow stair-step breakout; reduced-size scout",
                "runner_profile": "fast_stair_runner" if elite_runner else "shallow_stair",
                "day_move_pct": round(day_move_pct, 2),
                "pullback_from_hod_pct": round(pullback_from_hod_pct, 2),
                "allowed_hod_pullback_pct": round(allowed_hod_pullback_pct, 2),
                "base_high": round(base_high, 4),
                "base_low": round(base_low, 4),
                "base_range_pct": round(base_range_pct, 2),
                "allowed_base_range_pct": round(allowed_base_range_pct, 2),
                "breakout_pct": round(breakout_pct, 2),
                "volume_surge": round(volume_surge, 2),
                "recent_volume": round(recent_volume, 0),
                "body_ratio": round(body_ratio, 2),
                "vwap": round(current_vwap, 4),
                "stop_price": round(stop_price, 4),
                "close": price,
                "volume": latest.volume,
            },
            bars=list(bars[-90:]),
        )
