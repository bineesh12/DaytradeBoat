from __future__ import annotations

from datetime import datetime, timezone

import pytest

from daytrading.backtest.engine import BacktestEngine
from daytrading.data.feed import InMemoryBarFeed
from daytrading.execution.broker import PaperBroker
from daytrading.strategy.examples import BuyAndHoldOnce
from daytrading.models import Bar


def _bar(symbol: str, close: float) -> Bar:
    ts = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    return Bar(symbol=symbol, ts=ts, open=close, high=close, low=close, close=close, volume=100.0)


def test_buy_and_hold_backtest() -> None:
    sym = "DEMO"
    bars = [_bar(sym, 10.0), _bar(sym, 11.0)]
    feed = InMemoryBarFeed(bars)
    engine = BacktestEngine(
        feed=feed,
        strategy=BuyAndHoldOnce(sym, size=1.0),
        broker=PaperBroker(commission_per_share=0.0),
        initial_cash=1_000.0,
    )
    result = engine.run()
    assert result.bars_processed == 2
    assert len(result.fills) == 1
    assert result.final_portfolio is not None
    assert result.final_portfolio.cash == pytest.approx(1_000.0 - 10.0)
    assert result.final_portfolio.positions[sym].quantity == 1.0
