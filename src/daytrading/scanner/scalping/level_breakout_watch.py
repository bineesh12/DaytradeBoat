"""Watch-only scanner for resistance/HOD breakout attempts.

This does not create live orders. It makes wide or late level-break attempts
visible in the dashboard so we can review whether the stricter live scanner is
missing a real setup or correctly avoiding a chase.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.indicators.core import vwap
from daytrading.models import Bar, ScanResult


class LevelBreakoutWatchScanner:
    """Surface near-miss level breakouts as watch-only scanner hits."""

    def __init__(
        self,
        *,
        min_price: float = 1.0,
        max_price: float = 20.0,
        min_session_move_pct: float = 8.0,
        min_live_scout_session_move_pct: float = 3.0,
        watch_distance_pct: float = 1.5,
        max_breakout_pct: float = 6.0,
        max_watch_base_range_pct: float = 22.0,
        min_recent_volume: float = 100_000,
        min_volume_surge: float = 0.75,
        live_scout_enabled: bool = True,
    ) -> None:
        self._min_price = min_price
        self._max_price = max_price
        self._min_session_move_pct = min_session_move_pct
        self._min_live_scout_session_move_pct = min_live_scout_session_move_pct
        self._watch_distance_pct = watch_distance_pct
        self._max_breakout_pct = max_breakout_pct
        self._max_watch_base_range_pct = max_watch_base_range_pct
        self._min_recent_volume = min_recent_volume
        self._min_volume_surge = min_volume_surge
        self._live_scout_enabled = live_scout_enabled

    @property
    def name(self) -> str:
        return "level_breakout_watch"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, seq in universe.items():
            bars = list(seq)
            if len(bars) < 8:
                continue
            latest = bars[-1]
            if not (self._min_price <= latest.close <= self._max_price):
                continue
            hit = self._detect(symbol, bars[-90:])
            if hit is not None:
                results.append(hit)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _detect(self, symbol: str, bars: List[Bar]) -> ScanResult | None:
        latest = bars[-1]
        price = float(latest.close or 0.0)
        if price <= 0:
            return None

        session_open = float(bars[0].open or 0.0)
        if session_open <= 0:
            return None
        session_move_pct = (price - session_open) / session_open * 100.0
        if session_move_pct < min(self._min_session_move_pct, self._min_live_scout_session_move_pct):
            return None

        prior = bars[:-1]
        if len(prior) < 7:
            return None

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap > 0 and price < current_vwap * 0.995:
            return None

        recent_volume = sum(float(b.volume or 0.0) for b in bars[-3:])
        if recent_volume < self._min_recent_volume:
            return None

        best_hit = None
        for width in (5, 6, 8, 10, 12):
            if len(prior) < width:
                continue
            base = prior[-width:]
            level = max(float(b.high or 0.0) for b in base)
            base_low = min(float(b.low or 0.0) for b in base)
            if level <= 0 or base_low <= 0:
                continue

            base_range_pct = (level - base_low) / base_low * 100.0
            if base_range_pct > self._max_watch_base_range_pct:
                continue

            avg_base_vol = sum(float(b.volume or 0.0) for b in base) / len(base)
            if avg_base_vol <= 0:
                continue
            volume_surge = float(latest.volume or 0.0) / avg_base_vol
            if volume_surge < self._min_volume_surge:
                continue

            distance_to_level_pct = (level - price) / level * 100.0
            breakout_pct = (price - level) / level * 100.0
            pierced_level = float(latest.high or 0.0) >= level * 1.002
            closed_above = price >= level * 1.002

            if closed_above:
                if breakout_pct > self._max_breakout_pct:
                    continue
                status = "live level scout: closed above level; needs guard/10s hold"
            elif pierced_level:
                if session_move_pct < self._min_session_move_pct:
                    continue
                status = "watching failed level break: wick closed below"
            elif 0 <= distance_to_level_pct <= self._watch_distance_pct:
                if session_move_pct < self._min_session_move_pct:
                    continue
                status = "watching near resistance"
            else:
                continue

            candle_range = float(latest.high or 0.0) - float(latest.low or 0.0)
            upper_wick = float(latest.high or 0.0) - price
            wick_pct = upper_wick / candle_range * 100.0 if candle_range > 0 else 0.0
            score = (
                session_move_pct * 0.45
                + min(volume_surge, 4.0) * 10.0
                + max(0.0, breakout_pct) * 4.0
                - max(0.0, distance_to_level_pct) * 2.0
                - base_range_pct * 0.35
                - wick_pct * 0.12
            )
            live_scout = (
                self._live_scout_enabled
                and closed_above
                and session_move_pct >= self._min_live_scout_session_move_pct
                and breakout_pct >= 0.2
                and breakout_pct <= min(2.5, self._max_breakout_pct)
                and wick_pct <= 35.0
                and latest.low >= base_low * 0.985
                and float(latest.volume or 0.0) >= 100_000
                and recent_volume >= 150_000
                and volume_surge >= 1.0
            )
            pattern = "level_breakout_reclaim" if live_scout else "level_breakout_watch"
            setup_tier = "A+ setup" if live_scout else "watch only"
            candidate = ScanResult(
                symbol=symbol,
                scanner_name=self.name,
                ts=latest.ts,
                score=round(score, 3),
                criteria={
                    "pattern": pattern,
                    "direction": "up",
                    "setup_tier": setup_tier,
                    **(
                        {
                            "entry_tier": "level_scout",
                            "entry_mode": "level_breakout_scout",
                            "setup_quality": "level scout",
                            "size_factor": 0.35,
                            "stop_price": round(base_low - 0.02, 4),
                        }
                        if live_scout else {}
                    ),
                    "entry_tier_reason": status,
                    "status": status,
                    "breakout_level": round(level, 4),
                    "base_high": round(level, 4),
                    "base_low": round(base_low, 4),
                    "base_range_pct": round(base_range_pct, 2),
                    "distance_to_level_pct": round(distance_to_level_pct, 2),
                    "breakout_pct": round(breakout_pct, 2),
                    "session_move_pct": round(session_move_pct, 2),
                    "volume_surge": round(volume_surge, 2),
                    "recent_volume": round(recent_volume, 0),
                    "wick_pct": round(wick_pct, 2),
                    "vwap": round(current_vwap, 4) if current_vwap > 0 else 0.0,
                    "close": price,
                    "volume": latest.volume,
                },
                bars=list(bars[-90:]),
            )
            if best_hit is None or candidate.score > best_hit.score:
                best_hit = candidate

        return best_hit
