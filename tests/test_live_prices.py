"""Tests for multi-source live price resolution."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from daytrading.execution.live_prices import resolve_live_prices
from daytrading.models import Bar, Quote, Timeframe


def test_quote_mid_used_when_no_broker_position() -> None:
    ts = datetime.now(timezone.utc)
    quotes = {
        "TST": [
            Quote(symbol="TST", ts=ts, bid=4.90, ask=5.10, bid_size=100, ask_size=100),
        ],
    }
    prices = resolve_live_prices(["TST"], quotes=quotes)
    assert prices["TST"] == pytest.approx(5.0)


def test_bar_close_fallback() -> None:
    ts = datetime.now(timezone.utc)
    bars = {
        "TST": [
            Bar(symbol="TST", ts=ts, open=4.0, high=4.5, low=3.9, close=4.25, volume=1000, timeframe=Timeframe.MIN_1),
        ],
    }
    prices = resolve_live_prices(["TST"], bars=bars)
    assert prices["TST"] == 4.25
