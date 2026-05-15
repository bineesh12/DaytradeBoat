from __future__ import annotations

import math
from datetime import datetime, timezone

from daytrading.indicators.core import atr, ema, relative_volume, rsi, sma, vwap
from daytrading.models import Bar

TS = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)


def _bar(close: float, volume: float = 1000, high: float | None = None, low: float | None = None) -> Bar:
    h = high if high is not None else close + 0.5
    lo = low if low is not None else close - 0.5
    return Bar(symbol="TEST", ts=TS, open=close, high=h, low=lo, close=close, volume=volume)


def test_sma_basic() -> None:
    bars = [_bar(float(i)) for i in range(1, 6)]  # 1,2,3,4,5
    result = sma(bars, period=3)
    assert math.isnan(result[0])
    assert math.isnan(result[1])
    assert result[2] == (1 + 2 + 3) / 3
    assert result[3] == (2 + 3 + 4) / 3
    assert result[4] == (3 + 4 + 5) / 3


def test_ema_starts_at_sma() -> None:
    bars = [_bar(10.0)] * 5
    result = ema(bars, period=3)
    assert math.isnan(result[0])
    assert math.isnan(result[1])
    assert result[2] == 10.0  # SMA of 3 identical values


def test_vwap_single_bar() -> None:
    b = _bar(10.0, volume=100, high=11.0, low=9.0)
    result = vwap([b])
    expected = (11.0 + 9.0 + 10.0) / 3.0
    assert abs(result[0] - expected) < 1e-9


def test_rsi_length() -> None:
    bars = [_bar(float(i)) for i in range(20)]
    result = rsi(bars, period=14)
    assert len(result) == len(bars)
    assert math.isnan(result[0])


def test_atr_length() -> None:
    bars = [_bar(float(i), high=float(i) + 1, low=float(i) - 1) for i in range(20)]
    result = atr(bars, period=5)
    assert len(result) == len(bars)
    assert math.isnan(result[0])
    assert not math.isnan(result[5])


def test_relative_volume() -> None:
    bars = [_bar(10.0, volume=100)] * 21
    bars.append(_bar(10.0, volume=300))
    result = relative_volume(bars, period=20)
    assert abs(result[-1] - 3.0) < 1e-9
