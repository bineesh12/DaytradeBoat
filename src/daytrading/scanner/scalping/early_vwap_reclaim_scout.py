"""Early VWAP reclaim scout scanner.

Targets the controlled quick-scalp entry after a runner washes out, reclaims
VWAP/key support, and holds a higher low before the later HOD breakout chase.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.indicators.core import vwap
from daytrading.models import Bar, ScanResult


class EarlyVWAPReclaimScoutScanner:
    """Find early VWAP/level reclaim scouts on hot premarket runners."""

    def __init__(
        self,
        *,
        min_price: float = 1.0,
        max_price: float = 20.0,
        min_session_move_pct: float = 8.0,
        min_washout_below_vwap_pct: float = 1.0,
        min_reclaim_above_vwap_pct: float = 0.3,
        max_extension_from_vwap_pct: float = 5.5,
        max_distance_from_hod_pct: float = 12.0,
        max_reclaim_risk_pct: float = 9.0,
        min_latest_volume: float = 25_000,
        min_recent_volume: float = 90_000,
        min_active_recent_bars: int = 2,
        active_bar_volume: float = 20_000,
        max_single_bar_volume_share: float = 0.70,
    ) -> None:
        self._min_price = min_price
        self._max_price = max_price
        self._min_session_move_pct = min_session_move_pct
        self._min_washout_below_vwap_pct = min_washout_below_vwap_pct
        self._min_reclaim_above_vwap_pct = min_reclaim_above_vwap_pct
        self._max_extension_from_vwap_pct = max_extension_from_vwap_pct
        self._max_distance_from_hod_pct = max_distance_from_hod_pct
        self._max_reclaim_risk_pct = max_reclaim_risk_pct
        self._min_latest_volume = min_latest_volume
        self._min_recent_volume = min_recent_volume
        self._min_active_recent_bars = min_active_recent_bars
        self._active_bar_volume = active_bar_volume
        self._max_single_bar_volume_share = max_single_bar_volume_share

    @property
    def name(self) -> str:
        return "early_vwap_reclaim_scout"

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
        prev = bars[-2]
        if latest.close <= latest.open:
            return None
        if latest.volume < self._min_latest_volume:
            return None
        recent_activity = list(bars[-4:])
        recent_volume = sum(float(b.volume or 0.0) for b in recent_activity)
        active_recent_bars = sum(
            1 for b in recent_activity
            if float(b.volume or 0.0) >= self._active_bar_volume
        )
        max_recent_volume = max((float(b.volume or 0.0) for b in recent_activity), default=0.0)
        max_recent_volume_share = (
            max_recent_volume / recent_volume
            if recent_volume > 0 else 1.0
        )
        if recent_volume < self._min_recent_volume:
            return None
        if active_recent_bars < self._min_active_recent_bars:
            return None
        if max_recent_volume_share > self._max_single_bar_volume_share:
            return None

        session = list(session_bars or bars)
        session_open = float(session[0].open or 0.0)
        if session_open <= 0:
            return None
        session_high = max(float(b.high or 0.0) for b in session)
        if session_high <= 0:
            return None
        recent_window_high = max(float(b.high or 0.0) for b in bars)
        session_high_idx = max(
            i for i, b in enumerate(bars)
            if float(b.high or 0.0) >= recent_window_high
        )
        session_move_pct = (session_high - session_open) / session_open * 100.0
        if session_move_pct < self._min_session_move_pct:
            return None

        vwap_vals = vwap(bars)
        if len(vwap_vals) < len(bars):
            return None
        current_vwap = float(vwap_vals[-1] or 0.0)
        prev_vwap = float(vwap_vals[-2] or 0.0)
        if current_vwap <= 0 or prev_vwap <= 0:
            return None

        washout_start = max(session_high_idx + 1, len(bars) - 10)
        washout_window = bars[washout_start:-1]
        if len(washout_window) < 4:
            return None
        washout_low = min(float(b.low or 0.0) for b in washout_window)
        if washout_low <= 0:
            return None
        washout_idx = min(
            i for i, b in enumerate(washout_window)
            if float(b.low or 0.0) <= washout_low
        )
        washout_below_vwap_pct = (
            (prev_vwap - washout_low) / prev_vwap * 100.0
        )
        if washout_below_vwap_pct < self._min_washout_below_vwap_pct:
            return None

        reclaim_above_vwap_pct = (
            (latest.close - current_vwap) / current_vwap * 100.0
        )
        if reclaim_above_vwap_pct < self._min_reclaim_above_vwap_pct:
            return None
        if reclaim_above_vwap_pct > self._max_extension_from_vwap_pct:
            return None

        # Entry should happen on a reclaim/hold, not a late candle far above it.
        if prev.close > prev_vwap and prev.low > prev_vwap * 1.015:
            return None
        if latest.low < current_vwap * 0.985:
            return None

        recent_lows = [float(b.low or 0.0) for b in bars[-4:-1]]
        if recent_lows and latest.low < min(recent_lows) * 0.995:
            return None

        reclaim_hold_bars = list(washout_window[washout_idx + 1:]) + [latest]
        if len(reclaim_hold_bars) < 2:
            return None
        reclaim_low = min(float(b.low or 0.0) for b in reclaim_hold_bars)
        risk_to_reclaim_low_pct = (
            (latest.close - reclaim_low) / latest.close * 100.0
            if latest.close > 0 else 100.0
        )
        if risk_to_reclaim_low_pct > self._max_reclaim_risk_pct:
            return None

        distance_from_hod_pct = (
            (session_high - latest.close) / session_high * 100.0
            if session_high > 0 else 100.0
        )
        if distance_from_hod_pct > self._max_distance_from_hod_pct:
            return None

        avg_recent_vol = (
            sum(float(b.volume or 0.0) for b in bars[-6:-1]) / 5.0
            if len(bars) >= 6 else 0.0
        )
        volume_surge = (
            float(latest.volume or 0.0) / avg_recent_vol
            if avg_recent_vol > 0 else 1.0
        )

        score = (
            session_move_pct * 0.45
            + washout_below_vwap_pct * 4.0
            + reclaim_above_vwap_pct * 6.0
            + min(volume_surge, 4.0) * 6.0
            - distance_from_hod_pct * 0.4
            - risk_to_reclaim_low_pct * 0.8
        )

        return ScanResult(
            symbol=symbol,
            scanner_name=self.name,
            ts=latest.ts,
            score=round(score, 3),
            criteria={
                "pattern": "early_vwap_reclaim_scout",
                "direction": "up",
                "session_high": round(session_high, 4),
                "session_move_pct": round(session_move_pct, 2),
                "vwap": round(current_vwap, 4),
                "washout_low": round(washout_low, 4),
                "washout_below_vwap_pct": round(washout_below_vwap_pct, 2),
                "reclaim_above_vwap_pct": round(reclaim_above_vwap_pct, 2),
                "distance_from_hod_pct": round(distance_from_hod_pct, 2),
                "risk_to_reclaim_low_pct": round(risk_to_reclaim_low_pct, 2),
                "volume_surge": round(volume_surge, 2),
                "recent_volume": round(recent_volume, 0),
                "active_recent_bars": active_recent_bars,
                "max_recent_volume_share": round(max_recent_volume_share, 2),
                "stop_price": round(max(reclaim_low - 0.02, current_vwap * 0.97), 4),
                "close": latest.close,
                "volume": latest.volume,
                "setup_tier": "A+ setup",
            },
            bars=list(bars[-90:]),
        )
