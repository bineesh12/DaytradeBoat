"""Feature computation for XGBoost entry model.

30+ features covering:
- Price action & momentum
- Volume analysis
- Candle patterns
- Volatility & range
- Time context
- VWAP / support-resistance
- Tape / order flow
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from daytrading.models import Bar

FEATURE_NAMES = [
    # --- Price & Momentum (7) ---
    "gap_pct",
    "change_from_close_pct",
    "distance_from_hod_pct",
    "distance_from_lod_pct",
    "price_vs_vwap_pct",
    "momentum_5bar_pct",
    "momentum_10bar_pct",

    # --- Volume (6) ---
    "float_millions",
    "day_volume_millions",
    "rel_vol",
    "vol_surge",
    "vol_trend",
    "volume_price_confirm",

    # --- Candle Patterns (7) ---
    "green_bar_ratio",
    "avg_body_pct",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "last_candle_body_pct",
    "doji_count",
    "consecutive_green",

    # --- Volatility & Range (4) ---
    "atr_pct",
    "atr_expansion",
    "bar_range_vs_atr",
    "high_low_range_pct",

    # --- Time Context (3) ---
    "minutes_since_open",
    "is_first_30min",
    "is_power_hour",

    # --- Support / Resistance (4) ---
    "distance_from_premarket_high_pct",
    "pullback_depth_pct",
    "higher_lows_count",
    "breakout_attempt_number",
]


def compute_entry_features(
    price: float,
    *,
    float_shares: Optional[float] = None,
    day_volume: float = 0.0,
    rel_vol: float = 0.0,
    session_high: float = 0.0,
    session_open: float = 0.0,
    prior_close: float = 0.0,
    bars: Optional[Sequence[Bar]] = None,
    minutes_since_open: int = 0,
) -> List[float]:
    """Compute 31-feature vector for an entry decision."""
    bars = bars or []
    n = len(bars)

    # --- Price & Momentum ---
    gap_pct = (
        (session_open - prior_close) / prior_close * 100
        if prior_close > 0 else 0.0
    )
    change_from_close = (
        (price - prior_close) / prior_close * 100
        if prior_close > 0 else 0.0
    )
    distance_from_hod = (
        (session_high - price) / session_high * 100
        if session_high > 0 else 0.0
    )
    session_low = min(b.low for b in bars) if bars else price
    distance_from_lod = (
        (price - session_low) / session_low * 100
        if session_low > 0 else 0.0
    )

    # VWAP approximation (volume-weighted average price)
    if bars and n >= 5:
        total_vp = sum(b.close * b.volume for b in bars)
        total_vol = sum(b.volume for b in bars)
        vwap_est = total_vp / total_vol if total_vol > 0 else price
        price_vs_vwap = (price - vwap_est) / vwap_est * 100
    else:
        price_vs_vwap = 0.0

    # Momentum: last 5 and 10 bars
    if n >= 5:
        momentum_5 = (bars[-1].close - bars[-5].close) / bars[-5].close * 100 if bars[-5].close > 0 else 0.0
    else:
        momentum_5 = 0.0
    if n >= 10:
        momentum_10 = (bars[-1].close - bars[-10].close) / bars[-10].close * 100 if bars[-10].close > 0 else 0.0
    else:
        momentum_10 = momentum_5

    # --- Volume ---
    if n >= 10:
        avg_vol_10 = sum(b.volume for b in bars[-10:]) / 10
        vol_surge = bars[-1].volume / avg_vol_10 if avg_vol_10 > 0 else 1.0
    else:
        avg_vol_10 = 1.0
        vol_surge = 1.0

    # Volume trend: are recent bars' volume increasing or decreasing?
    if n >= 6:
        first_half_vol = sum(b.volume for b in bars[-6:-3]) / 3
        second_half_vol = sum(b.volume for b in bars[-3:]) / 3
        vol_trend = second_half_vol / first_half_vol if first_half_vol > 0 else 1.0
    else:
        vol_trend = 1.0

    # Volume-price confirmation: is volume higher on green bars?
    if n >= 5:
        green_vol = sum(b.volume for b in bars[-5:] if b.close >= b.open) or 1
        red_vol = sum(b.volume for b in bars[-5:] if b.close < b.open) or 1
        volume_price_confirm = green_vol / (green_vol + red_vol)
    else:
        volume_price_confirm = 0.5

    # --- Candle Patterns ---
    recent = list(bars[-5:]) if n >= 5 else list(bars)
    if recent:
        green_count = sum(1 for b in recent if b.close >= b.open)
        green_ratio = green_count / len(recent)
        avg_body = sum(abs(b.close - b.open) for b in recent) / len(recent)
        avg_body_pct = avg_body / price * 100 if price > 0 else 0.0

        # Wick analysis
        upper_wicks = []
        lower_wicks = []
        for b in recent:
            bar_range = b.high - b.low
            if bar_range > 0:
                upper_wick = (b.high - max(b.open, b.close)) / bar_range
                lower_wick = (min(b.open, b.close) - b.low) / bar_range
                upper_wicks.append(upper_wick)
                lower_wicks.append(lower_wick)
        upper_wick_ratio = sum(upper_wicks) / len(upper_wicks) if upper_wicks else 0.0
        lower_wick_ratio = sum(lower_wicks) / len(lower_wicks) if lower_wicks else 0.0

        # Last candle body size
        last_bar = recent[-1]
        last_body = abs(last_bar.close - last_bar.open)
        last_range = last_bar.high - last_bar.low
        last_candle_body_pct = last_body / last_range if last_range > 0 else 0.0

        # Doji count (body < 20% of range)
        doji_count = sum(
            1 for b in recent
            if (b.high - b.low) > 0 and abs(b.close - b.open) / (b.high - b.low) < 0.2
        )

        # Consecutive green bars from the end
        consecutive_green = 0
        for b in reversed(recent):
            if b.close >= b.open:
                consecutive_green += 1
            else:
                break
    else:
        green_ratio = 0.5
        avg_body_pct = 0.0
        upper_wick_ratio = 0.0
        lower_wick_ratio = 0.0
        last_candle_body_pct = 0.0
        doji_count = 0
        consecutive_green = 0

    # --- Volatility & Range ---
    if n >= 10:
        atr = sum(b.high - b.low for b in bars[-10:]) / 10
        atr_pct = atr / price * 100 if price > 0 else 0.0
        # ATR expansion: is current ATR higher than earlier ATR?
        if n >= 20:
            atr_early = sum(b.high - b.low for b in bars[-20:-10]) / 10
            atr_expansion = atr / atr_early if atr_early > 0 else 1.0
        else:
            atr_expansion = 1.0
        # Current bar range vs ATR
        if bars:
            current_range = bars[-1].high - bars[-1].low
            bar_range_vs_atr = current_range / atr if atr > 0 else 1.0
        else:
            bar_range_vs_atr = 1.0
    else:
        atr_pct = 0.0
        atr_expansion = 1.0
        bar_range_vs_atr = 1.0

    # Session high-low range
    if session_high > 0 and session_low > 0:
        high_low_range_pct = (session_high - session_low) / session_low * 100
    else:
        high_low_range_pct = 0.0

    # --- Time Context ---
    is_first_30min = 1.0 if minutes_since_open <= 30 else 0.0
    is_power_hour = 1.0 if minutes_since_open >= 330 else 0.0  # last hour

    # --- Support / Resistance ---
    # Distance from premarket high (approximated as first bar's high)
    premarket_high = bars[0].high if bars else price
    distance_from_premarket_high = (
        (price - premarket_high) / premarket_high * 100
        if premarket_high > 0 else 0.0
    )

    # Pullback depth: how far did price pull back from recent high before this bar?
    if n >= 5:
        recent_high = max(b.high for b in bars[-10:]) if n >= 10 else max(b.high for b in bars)
        recent_low = min(b.low for b in bars[-5:])
        pullback_depth = (recent_high - recent_low) / recent_high * 100 if recent_high > 0 else 0.0
    else:
        pullback_depth = 0.0

    # Higher lows: count how many bars have higher lows than the bar before
    higher_lows = 0
    if n >= 5:
        for i in range(max(0, n - 5), n - 1):
            if bars[i + 1].low > bars[i].low:
                higher_lows += 1

    # Breakout attempt number: how many times price touched session high area
    breakout_attempts = 0
    if n >= 5 and session_high > 0:
        threshold = session_high * 0.98
        for b in bars[-20:] if n >= 20 else bars:
            if b.high >= threshold:
                breakout_attempts += 1

    return [
        # Price & Momentum (7)
        gap_pct,
        change_from_close,
        distance_from_hod,
        distance_from_lod,
        price_vs_vwap,
        momentum_5,
        momentum_10,
        # Volume (6)
        (float_shares or 0) / 1_000_000,
        day_volume / 1_000_000,
        rel_vol,
        vol_surge,
        vol_trend,
        volume_price_confirm,
        # Candle Patterns (7)
        green_ratio,
        avg_body_pct,
        upper_wick_ratio,
        lower_wick_ratio,
        last_candle_body_pct,
        float(doji_count),
        float(consecutive_green),
        # Volatility & Range (4)
        atr_pct,
        atr_expansion,
        bar_range_vs_atr,
        high_low_range_pct,
        # Time Context (3)
        float(minutes_since_open),
        is_first_30min,
        is_power_hour,
        # Support / Resistance (4)
        distance_from_premarket_high,
        pullback_depth,
        float(higher_lows),
        float(breakout_attempts),
    ]
