"""Pure-Python technical indicators operating on lists of Bar objects.

No numpy/pandas required — keeps the core dependency-free.
Each function returns a list aligned to the input bars (NaN-padded where
the lookback is insufficient).
"""

from __future__ import annotations

import math
from typing import List, Sequence

from daytrading.models import Bar

NaN = float("nan")


def sma(bars: Sequence[Bar], period: int) -> List[float]:
    """Simple moving average of close prices."""
    out: List[float] = []
    for i in range(len(bars)):
        if i < period - 1:
            out.append(NaN)
        else:
            s = sum(bars[j].close for j in range(i - period + 1, i + 1))
            out.append(s / period)
    return out


def ema(bars: Sequence[Bar], period: int) -> List[float]:
    """Exponential moving average of close prices."""
    k = 2.0 / (period + 1)
    out: List[float] = []
    prev = NaN
    for i, bar in enumerate(bars):
        if i < period - 1:
            out.append(NaN)
        elif i == period - 1:
            prev = sum(b.close for b in bars[: period]) / period
            out.append(prev)
        else:
            prev = bar.close * k + prev * (1 - k)
            out.append(prev)
    return out


def rsi(bars: Sequence[Bar], period: int = 14) -> List[float]:
    """Wilder RSI."""
    out: List[float] = [NaN]
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(bars)):
        delta = bars[i].close - bars[i - 1].close
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
        if i < period:
            out.append(NaN)
        elif i == period:
            avg_g = sum(gains) / period
            avg_l = sum(losses) / period
            if avg_l == 0:
                out.append(100.0)
            else:
                out.append(100.0 - 100.0 / (1 + avg_g / avg_l))
        else:
            prev_rsi_idx = len(out) - 1
            avg_g = (gains[-2] * (period - 1) + gains[-1]) / period if prev_rsi_idx >= period else sum(gains[-period:]) / period
            avg_l = (losses[-2] * (period - 1) + losses[-1]) / period if prev_rsi_idx >= period else sum(losses[-period:]) / period
            if avg_l == 0:
                out.append(100.0)
            else:
                out.append(100.0 - 100.0 / (1 + avg_g / avg_l))
    return out


def vwap(bars: Sequence[Bar]) -> List[float]:
    """Cumulative VWAP from the first bar (reset at session boundary is caller's job)."""
    cum_pv = 0.0
    cum_vol = 0.0
    out: List[float] = []
    for bar in bars:
        typical = (bar.high + bar.low + bar.close) / 3.0
        cum_pv += typical * bar.volume
        cum_vol += bar.volume
        out.append(cum_pv / cum_vol if cum_vol > 0 else NaN)
    return out


def atr(bars: Sequence[Bar], period: int = 14) -> List[float]:
    """Average True Range."""
    out: List[float] = [NaN]
    trs: List[float] = []
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        )
        trs.append(tr)
        if i < period:
            out.append(NaN)
        elif i == period:
            out.append(sum(trs) / period)
        else:
            prev_atr = out[-1]
            out.append((prev_atr * (period - 1) + tr) / period)
    return out


def relative_volume(bars: Sequence[Bar], period: int = 20) -> List[float]:
    """Current bar volume / average volume over `period` prior bars."""
    out: List[float] = []
    for i in range(len(bars)):
        if i < period:
            out.append(NaN)
        else:
            avg_v = sum(bars[j].volume for j in range(i - period, i)) / period
            out.append(bars[i].volume / avg_v if avg_v > 0 else NaN)
    return out
