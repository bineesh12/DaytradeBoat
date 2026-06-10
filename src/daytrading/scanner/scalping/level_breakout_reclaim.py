"""Level breakout/reclaim scanner.

Detects the first clean break above a tight intraday base/resistance level.
This is earlier than HOD reclaim: it targets the yellow-line breakout before
the stock has already stretched into a late HOD chase.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.indicators.core import vwap
from daytrading.models import Bar, ScanResult


class LevelBreakoutReclaimScanner:
    """Find early base-level breakouts with volume confirmation."""

    def __init__(
        self,
        *,
        min_price: float = 1.0,
        max_price: float = 20.0,
        min_session_move_pct: float = 5.0,
        min_breakout_pct: float = 0.6,
        max_breakout_pct: float = 2.5,
        max_base_range_pct: float = 8.0,
        min_breakout_volume: float = 100_000,
        min_volume_surge: float = 1.15,
    ) -> None:
        self._min_price = min_price
        self._max_price = max_price
        self._min_session_move_pct = min_session_move_pct
        self._min_breakout_pct = min_breakout_pct
        self._max_breakout_pct = max_breakout_pct
        self._max_base_range_pct = max_base_range_pct
        self._min_breakout_volume = min_breakout_volume
        self._min_volume_surge = min_volume_surge

    @property
    def name(self) -> str:
        return "level_breakout_reclaim"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        results: List[ScanResult] = []
        for symbol, seq in universe.items():
            bars = list(seq)
            if len(bars) < 8:
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
        latest = bars[-1]
        if latest.close <= latest.open:
            return None

        session = list(session_bars or bars)
        session_open = session[0].open
        if session_open <= 0:
            return None
        session_move_pct = (latest.close - session_open) / session_open * 100.0
        if session_move_pct < self._min_session_move_pct:
            return None

        prior = bars[:-1]
        if len(prior) < 7:
            return None

        best_hit = None
        # Prefer the tightest recent base that the latest candle is reclaiming.
        for width in (4, 5, 6, 7, 8):
            if len(prior) < width:
                continue
            base = prior[-width:]
            base_high = max(b.high for b in base)
            base_low = min(b.low for b in base)
            if base_low <= 0 or base_high <= 0:
                continue

            base_range_pct = (base_high - base_low) / base_low * 100.0
            if base_range_pct > self._max_base_range_pct:
                continue

            # The latest bar must break and close above the level. A wick-only
            # poke is a false-breakout candidate, not an entry.
            if latest.high <= base_high * (1.0 + self._min_breakout_pct / 100.0):
                continue
            if latest.close <= base_high * (1.0 + self._min_breakout_pct / 100.0):
                continue

            breakout_pct = (latest.close - base_high) / base_high * 100.0
            if breakout_pct > self._max_breakout_pct:
                continue
            if float(latest.volume or 0.0) < self._min_breakout_volume:
                continue

            avg_base_vol = sum(float(b.volume or 0.0) for b in base) / len(base)
            if avg_base_vol <= 0:
                continue
            volume_surge = float(latest.volume or 0.0) / avg_base_vol
            if volume_surge < self._min_volume_surge:
                continue

            vwap_vals = vwap(bars)
            current_vwap = vwap_vals[-1] if vwap_vals else 0.0
            if current_vwap > 0 and latest.close < current_vwap * 1.002:
                continue

            upper_wick = latest.high - latest.close
            candle_range = latest.high - latest.low
            wick_pct = upper_wick / candle_range * 100.0 if candle_range > 0 else 0.0
            if wick_pct > 45.0:
                continue

            risk_to_base_low_pct = (latest.close - base_low) / latest.close * 100.0
            if risk_to_base_low_pct > 12.0:
                continue

            score = (
                session_move_pct * 0.7
                + breakout_pct * 6.0
                + min(volume_surge, 4.0) * 8.0
                - base_range_pct * 1.5
                - wick_pct * 0.2
            )
            candidate = ScanResult(
                symbol=symbol,
                scanner_name=self.name,
                ts=latest.ts,
                score=round(score, 3),
                criteria={
                    "pattern": "level_breakout_reclaim",
                    "direction": "up",
                    "breakout_level": round(base_high, 4),
                    "base_high": round(base_high, 4),
                    "base_low": round(base_low, 4),
                    "base_range_pct": round(base_range_pct, 2),
                    "breakout_pct": round(breakout_pct, 2),
                    "session_move_pct": round(session_move_pct, 2),
                    "volume_surge": round(volume_surge, 2),
                    "wick_pct": round(wick_pct, 2),
                    "vwap": round(current_vwap, 4) if current_vwap > 0 else 0.0,
                    "stop_price": round(base_low - 0.02, 4),
                    "close": latest.close,
                    "volume": latest.volume,
                },
                bars=list(bars[-80:]),
            )
            if best_hit is None or candidate.score > best_hit.score:
                best_hit = candidate

        return best_hit
