"""Pullback-base scanner — detects consolidation after a pullback from a big move.

The classic Warrior Trading continuation entry:
1. Stock had a big move (up > 10% from session open)
2. Pulled back from high of day (at least 3%)
3. Formed a base: tight range for 3+ bars (consolidation)
4. Bounce: latest bar is green with a higher low than the prior bar

This catches stocks like PIII that run from $4→$14, pull back to $11,
consolidate for a few bars, then start bouncing — the re-entry point.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from daytrading.indicators.core import vwap
from daytrading.models import Bar, ScanResult

logger = logging.getLogger(__name__)


class PullbackBaseScanner:
    """Detects a consolidation base after a pullback from a strong move."""

    def __init__(
        self,
        *,
        min_day_move_pct: float = 10.0,
        min_pullback_pct: float = 3.0,
        max_pullback_pct: float = 30.0,
        base_bars: int = 3,
        max_base_range_pct: float = 3.0,
        min_price: float = 2.0,
        max_price: float = 20.0,
    ) -> None:
        self._min_day_move_pct = min_day_move_pct
        self._min_pullback_pct = min_pullback_pct
        self._max_pullback_pct = max_pullback_pct
        self._base_bars = base_bars
        self._max_base_range_pct = max_base_range_pct
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "pullback_base"

    @staticmethod
    def _is_second_chance_vwap_reclaim(bars: List[Bar], session_high: float) -> bool:
        """Allow deep runner pullbacks only after a fresh VWAP/base reclaim."""
        if len(bars) < 10 or session_high <= 0:
            return False

        latest = bars[-1]
        if latest.close <= latest.open or latest.close <= 0:
            return False

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap <= 0 or latest.close < current_vwap * 1.005:
            return False

        base = list(bars[-5:-1])
        if len(base) < 4:
            return False
        base_high = max(b.high for b in base)
        base_low = min(b.low for b in base)
        if base_low <= 0 or latest.close <= base_high * 1.002:
            return False

        base_range_pct = (base_high - base_low) / base_low * 100.0
        if base_range_pct > 9.0:
            return False

        candle_range = latest.high - latest.low
        body_ratio = (latest.close - latest.open) / candle_range if candle_range > 0 else 0.0
        if body_ratio < 0.35:
            return False

        avg_base_vol = sum(float(b.volume or 0.0) for b in base) / len(base)
        if avg_base_vol <= 0 or float(latest.volume or 0.0) < avg_base_vol * 1.05:
            return False

        recent_volume = sum(float(b.volume or 0.0) for b in bars[-3:])
        return recent_volume >= 75_000

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, bars in universe.items():
            if len(bars) < 15:
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
        if len(bars) < 15:
            return None

        latest = bars[-1]
        session_open = bars[0].open
        if session_open <= 0:
            return None

        session_high = max(b.high for b in bars)
        day_move_pct = (session_high - session_open) / session_open * 100
        if day_move_pct < self._min_day_move_pct:
            return None

        pullback_pct = (session_high - latest.close) / session_high * 100
        second_chance_reclaim = self._is_second_chance_vwap_reclaim(bars, session_high)
        if pullback_pct < self._min_pullback_pct:
            logger.info(
                "PULLBACK_BASE %s: pullback too small %.1f%% (min %.1f%%)",
                symbol, pullback_pct, self._min_pullback_pct,
            )
            return None
        if pullback_pct > self._max_pullback_pct and not second_chance_reclaim:
            logger.info(
                "PULLBACK_BASE %s: pullback too large %.1f%% (max %.1f%%)",
                symbol, pullback_pct, self._max_pullback_pct,
            )
            return None

        base = list(bars[-self._base_bars:])
        base_high = max(b.high for b in base)
        base_low = min(b.low for b in base)
        if base_low <= 0:
            return None
        base_range_pct = (base_high - base_low) / base_low * 100

        effective_max_range = self._max_base_range_pct * (1 + day_move_pct / 50)
        effective_max_range = min(effective_max_range, 15.0)
        if base_range_pct > effective_max_range and not second_chance_reclaim:
            logger.info(
                "PULLBACK_BASE %s: base range too wide %.1f%% (max %.1f%% adj for %.0f%% move), "
                "base_high=%.2f base_low=%.2f, close=%.2f, HOD=%.2f, pullback=%.1f%%",
                symbol, base_range_pct, effective_max_range, day_move_pct,
                base_high, base_low, latest.close, session_high, pullback_pct,
            )
            return None

        if latest.close <= latest.open:
            logger.info(
                "PULLBACK_BASE %s: latest bar not green (O=%.2f C=%.2f)",
                symbol, latest.open, latest.close,
            )
            return None

        prev = bars[-2]
        if latest.low < prev.low:
            logger.info(
                "PULLBACK_BASE %s: no higher low (%.2f < %.2f)",
                symbol, latest.low, prev.low,
            )
            return None

        session_mid = (session_high + session_open) / 2
        if latest.close < session_mid and not second_chance_reclaim:
            logger.info(
                "PULLBACK_BASE %s: below session midpoint (%.2f < %.2f)",
                symbol, latest.close, session_mid,
            )
            return None

        logger.info(
            "PULLBACK_BASE %s: *** HIT *** close=%.2f HOD=%.2f pullback=%.1f%% "
            "base_range=%.1f%% day_move=%.0f%%%s",
            symbol, latest.close, session_high, pullback_pct,
            base_range_pct, day_move_pct,
            " second-chance VWAP reclaim" if second_chance_reclaim else "",
        )

        stop_price = base_low - 0.02
        score = day_move_pct * (1 + (self._max_pullback_pct - pullback_pct) / 10)
        entry_tier = "second_chance_reclaim" if second_chance_reclaim else None

        return ScanResult(
            symbol=symbol,
            scanner_name=self.name,
            ts=latest.ts,
            score=round(score, 3),
            criteria={
                "pattern": "pullback_base",
                "direction": "up",
                "session_high": round(session_high, 4),
                "day_move_pct": round(day_move_pct, 2),
                "pullback_pct": round(pullback_pct, 2),
                "base_range_pct": round(base_range_pct, 2),
                "base_low": round(base_low, 4),
                "stop_price": round(stop_price, 4),
                "close": latest.close,
                "volume": latest.volume,
                **(
                    {
                        "entry_tier": entry_tier,
                        "entry_tier_reason": "deep pullback reclaimed VWAP/base with buyers returning",
                    }
                    if entry_tier else {}
                ),
            },
            bars=[],
        )
