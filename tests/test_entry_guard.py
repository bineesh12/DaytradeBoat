"""Tests for ``entry_guard.check_entry_quality``."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from daytrading.models import Bar, Timeframe
from daytrading.strategy import entry_guard as eg


def _bar(
    i: int,
    *,
    close: float,
    open_: float,
    high: float,
    low: float,
    volume: float,
    base_ts: datetime,
    n: int,
) -> Bar:
    ts = base_ts - timedelta(seconds=(n - i))
    return Bar(
        symbol="TST",
        ts=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        timeframe=Timeframe.SEC_5,
    )


def _uptrend_bars_passing_default_guard() -> list[Bar]:
    """25 bars: uptrend with a pullback, enough volume, all bars tight range."""
    now = datetime.now(timezone.utc)
    n = 25
    bars: list[Bar] = []
    for i in range(n):
        frac = i / (n - 1) if n > 1 else 1.0
        c = 3.0 + (5.0 - 3.0) * frac
        o = c - 0.004
        hi = c + 0.005
        lo = c - 0.005
        vol = 50_000.0 if i < n - 1 else 250_000.0
        bars.append(_bar(i, close=c, open_=o, high=hi, low=lo, volume=vol, base_ts=now, n=n))
    # Add a small pullback in the last 3 bars so the overextension filter
    # doesn't trigger (last 3 bars are NOT all green after the pullback).
    bars[-3] = _bar(
        n - 3, close=4.88, open_=4.90, high=4.91, low=4.87,
        volume=50_000.0, base_ts=now, n=n,
    )
    bars[-2] = _bar(
        n - 2, close=4.92, open_=4.88, high=4.93, low=4.87,
        volume=60_000.0, base_ts=now, n=n,
    )
    i = n - 1
    bars[-1] = _bar(
        i,
        close=5.00,
        open_=4.996,
        high=5.002,
        low=4.993,
        volume=250_000.0,
        base_ts=now,
        n=n,
    )
    return bars


class TestCheckEntryQuality:
    def test_insufficient_bars(self) -> None:
        assert eg.check_entry_quality([]) == "insufficient bars"
        assert eg.check_entry_quality([_bar(0, close=5, open_=4.9, high=5.1, low=4.9, volume=1e6, base_ts=datetime.now(timezone.utc), n=1)]) == "insufficient bars"

    def test_price_below_band(self) -> None:
        now = datetime.now(timezone.utc)
        bars = [_bar(i, close=1.5, open_=1.4, high=1.55, low=1.4, volume=100_000, base_ts=now, n=3) for i in range(3)]
        r = eg.check_entry_quality(bars, symbol="TST")
        assert r is not None and "below" in r

    def test_rvol_below_minimum(self) -> None:
        now = datetime.now(timezone.utc)
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.99, volume=80_000, base_ts=now, n=20)
            for i in range(20)
        ]
        r = eg.check_entry_quality(bars, symbol="TST")
        assert r is not None and "rvol" in r.lower()

    def test_full_pass_synthetic(self) -> None:
        bars = _uptrend_bars_passing_default_guard()
        assert eg.check_entry_quality(bars, symbol="TST") is None
