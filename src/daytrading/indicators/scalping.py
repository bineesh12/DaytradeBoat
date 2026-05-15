"""Scalping-specific indicators operating on tick/quote-level data.

These measure micro-structure signals: tape speed, order flow imbalance,
momentum bursts, and spread behavior.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from daytrading.models import Bar, Quote, Side, Tick

NaN = float("nan")


# ---------------------------------------------------------------------------
# Spread analysis
# ---------------------------------------------------------------------------

def avg_spread(quotes: Sequence[Quote], window: int = 20) -> List[float]:
    """Rolling average bid-ask spread over `window` quotes."""
    out: List[float] = []
    for i in range(len(quotes)):
        if i < window - 1:
            out.append(NaN)
        else:
            s = sum(quotes[j].spread for j in range(i - window + 1, i + 1))
            out.append(s / window)
    return out


def spread_compression_ratio(quotes: Sequence[Quote], short: int = 5, long: int = 20) -> List[float]:
    """Ratio of short-term avg spread to long-term avg spread.

    < 1.0 means spread is tightening (good for scalping entry).
    """
    short_avg = avg_spread(quotes, short)
    long_avg = avg_spread(quotes, long)
    out: List[float] = []
    for s, l in zip(short_avg, long_avg):
        if math.isnan(s) or math.isnan(l) or l == 0:
            out.append(NaN)
        else:
            out.append(s / l)
    return out


# ---------------------------------------------------------------------------
# Tape / trade flow
# ---------------------------------------------------------------------------

def tape_speed(ticks: Sequence[Tick], window_seconds: float = 5.0) -> List[float]:
    """Trades per second over a rolling time window.

    Higher tape speed = more interest from algos/institutions.
    """
    out: List[float] = []
    for i in range(len(ticks)):
        cutoff_ts = ticks[i].ts.timestamp() - window_seconds
        count = 0
        for j in range(i, -1, -1):
            if ticks[j].ts.timestamp() >= cutoff_ts:
                count += 1
            else:
                break
        out.append(count / window_seconds if window_seconds > 0 else 0.0)
    return out


def order_flow_imbalance(ticks: Sequence[Tick], window: int = 50) -> List[float]:
    """Buy volume minus sell volume over last `window` ticks, normalized.

    Returns values in [-1.0, 1.0]:
      +1.0 = 100% aggressive buying
      -1.0 = 100% aggressive selling
       0.0 = balanced
    """
    out: List[float] = []
    for i in range(len(ticks)):
        start = max(0, i - window + 1)
        buy_vol = 0.0
        sell_vol = 0.0
        for j in range(start, i + 1):
            if ticks[j].side is Side.BUY:
                buy_vol += ticks[j].size
            else:
                sell_vol += ticks[j].size
        total = buy_vol + sell_vol
        out.append((buy_vol - sell_vol) / total if total > 0 else 0.0)
    return out


def cumulative_delta(ticks: Sequence[Tick]) -> List[float]:
    """Running sum of (buy_volume - sell_volume).

    Rising delta = buyers in control. Falling delta = sellers in control.
    """
    out: List[float] = []
    running = 0.0
    for tick in ticks:
        if tick.side is Side.BUY:
            running += tick.size
        else:
            running -= tick.size
        out.append(running)
    return out


# ---------------------------------------------------------------------------
# Momentum burst
# ---------------------------------------------------------------------------

def momentum_burst(bars: Sequence[Bar], period: int = 3) -> List[float]:
    """Price change over `period` bars as a percentage.

    Detects sharp micro-moves suitable for scalping entries.
    """
    out: List[float] = []
    for i in range(len(bars)):
        if i < period:
            out.append(NaN)
        else:
            prev_close = bars[i - period].close
            if prev_close > 0:
                out.append(((bars[i].close - prev_close) / prev_close) * 100.0)
            else:
                out.append(NaN)
    return out


def price_velocity(bars: Sequence[Bar], period: int = 5) -> List[float]:
    """Average price change per bar over `period` — speed of the move.

    Measured in price units per bar.
    """
    out: List[float] = []
    for i in range(len(bars)):
        if i < period:
            out.append(NaN)
        else:
            delta = bars[i].close - bars[i - period].close
            out.append(delta / period)
    return out
