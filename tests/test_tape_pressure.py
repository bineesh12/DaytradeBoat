"""Tests for TapePressureExit module."""

from datetime import datetime, timezone

import pytest

from daytrading.exits.tape_pressure import TapePressureExit
from daytrading.models import Quote, Side, Tick


def _make_tick(price: float, size: float = 100, side: Side = Side.BUY, offset_ms: int = 0) -> Tick:
    ts = datetime(2026, 5, 26, 15, 30, 0, tzinfo=timezone.utc)
    return Tick(symbol="TEST", ts=ts, price=price, size=size, side=side)


def _make_quote(bid: float, ask: float, bid_size: float = 500, ask_size: float = 500) -> Quote:
    ts = datetime(2026, 5, 26, 15, 30, 0, tzinfo=timezone.utc)
    return Quote(symbol="TEST", ts=ts, bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)


class TestTapePressureExit:
    def test_no_exit_when_not_in_profit(self) -> None:
        tp = TapePressureExit(threshold=60)
        ticks = [_make_tick(10.0, side=Side.SELL) for _ in range(50)]
        quotes = [_make_quote(9.95, 10.05) for _ in range(25)]
        assert tp.check(ticks, quotes, entry_price=10.10, current_price=10.00, hold_secs=60) is False

    def test_no_exit_when_hold_too_short(self) -> None:
        tp = TapePressureExit(threshold=60, min_hold_secs=30.0)
        ticks = [_make_tick(10.0, side=Side.SELL) for _ in range(50)]
        quotes = [_make_quote(9.95, 10.05) for _ in range(25)]
        assert tp.check(ticks, quotes, entry_price=9.50, current_price=10.00, hold_secs=10) is False

    def test_no_exit_when_not_enough_ticks(self) -> None:
        tp = TapePressureExit(threshold=60, min_ticks=20)
        ticks = [_make_tick(10.0, side=Side.SELL) for _ in range(10)]
        quotes = [_make_quote(9.95, 10.05) for _ in range(25)]
        assert tp.check(ticks, quotes, entry_price=9.50, current_price=10.00, hold_secs=60) is False

    def test_exits_on_heavy_selling(self) -> None:
        """Heavy sell flow + dying tape → score should exceed threshold."""
        tp = TapePressureExit(threshold=50)
        # All sells — imbalance will be -1.0 → 30 pts
        ticks = [_make_tick(10.0 - i * 0.01, side=Side.SELL, size=200) for i in range(50)]
        # Spread widening
        quotes = [_make_quote(9.90, 10.10) for _ in range(10)]
        quotes += [_make_quote(9.80, 10.20) for _ in range(10)]
        result = tp.check(ticks, quotes, entry_price=9.50, current_price=10.00, hold_secs=60)
        assert result is True

    def test_no_exit_balanced_flow(self) -> None:
        """Balanced buy/sell flow should not trigger exit."""
        tp = TapePressureExit(threshold=60)
        ticks = []
        for i in range(50):
            side = Side.BUY if i % 2 == 0 else Side.SELL
            ticks.append(_make_tick(10.0, side=side))
        quotes = [_make_quote(9.99, 10.01) for _ in range(25)]
        result = tp.check(ticks, quotes, entry_price=9.50, current_price=10.00, hold_secs=60)
        assert result is False

    def test_threshold_configurable(self) -> None:
        """Lower threshold makes exit more sensitive."""
        # All sells → high pressure
        ticks = [_make_tick(10.0, side=Side.SELL, size=200) for _ in range(50)]
        quotes = [_make_quote(9.95, 10.05) for _ in range(25)]

        tp_low = TapePressureExit(threshold=20)
        tp_high = TapePressureExit(threshold=95)

        assert tp_low.check(ticks, quotes, entry_price=9.50, current_price=10.00, hold_secs=60) is True
        assert tp_high.check(ticks, quotes, entry_price=9.50, current_price=10.00, hold_secs=60) is False
