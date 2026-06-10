"""Tests for startup trade-history sync from Alpaca orders."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from daytrading.dashboard.hub import DashboardHub
from daytrading.runner import AlpacaRunner


class _Value:
    def __init__(self, value: str) -> None:
        self.value = value


def _order(
    *,
    order_id: str,
    symbol: str = "MASK",
    side: str,
    status: str,
    qty: float,
    price: float,
    ts: datetime,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=order_id,
        symbol=symbol,
        side=_Value(side),
        status=_Value(status),
        filled_qty=qty,
        filled_avg_price=price,
        filled_at=ts,
        submitted_at=ts,
    )


def _runner_with_orders(orders: list[SimpleNamespace]) -> AlpacaRunner:
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._hub = DashboardHub()
    runner._last_synced_order_ids = set()
    runner._broker = SimpleNamespace(
        client=SimpleNamespace(
            get_account=lambda: SimpleNamespace(equity="94601.65", last_equity="94706.71"),
            get_orders=lambda filter=None: orders,
        ),
    )
    return runner


def _runner_for_fill_sync(orders: list[SimpleNamespace]) -> AlpacaRunner:
    runner = _runner_with_orders(orders)
    runner._pipeline = SimpleNamespace(
        exit_manager=SimpleNamespace(tracked={}),
        portfolio=SimpleNamespace(positions={}),
    )
    runner._record_trade_exit = lambda *args, **kwargs: None
    runner._push_positions_from_alpaca = lambda: None
    return runner


def test_sync_trade_history_matches_canceled_buy_with_fill_to_later_sell() -> None:
    ts = datetime.now(timezone.utc).replace(hour=13, minute=30, second=0, microsecond=0)
    # Alpaca returns closed orders newest first; runner reverses to process oldest first.
    orders = [
        _order(order_id="sell-final", side="sell", status="filled", qty=437, price=5.00, ts=ts),
        _order(order_id="sell-partial", side="sell", status="filled", qty=66, price=5.15, ts=ts),
        _order(order_id="buy-late-fill", side="buy", status="canceled", qty=503, price=5.38, ts=ts),
    ]
    runner = _runner_with_orders(orders)

    runner._sync_trade_history()

    exits = [t for t in runner._hub.trades if t.trade_type == "exit"]
    assert len(exits) == 2
    assert exits[0].entry_price == pytest.approx(5.38)
    assert exits[0].pnl == pytest.approx((5.15 - 5.38) * 66)
    assert exits[1].entry_price == pytest.approx(5.38)
    assert exits[1].pnl == pytest.approx((5.00 - 5.38) * 437)
    assert sum(t.pnl for t in exits if t.pnl is not None) == pytest.approx(-181.24)


def test_sync_trade_history_does_not_report_zero_pnl_for_unmatched_sell() -> None:
    ts = datetime.now(timezone.utc).replace(hour=13, minute=30, second=0, microsecond=0)
    orders = [
        _order(order_id="sell-only", side="sell", status="filled", qty=100, price=5.00, ts=ts),
    ]
    runner = _runner_with_orders(orders)

    runner._sync_trade_history()

    [exit_trade] = [t for t in runner._hub.trades if t.trade_type == "exit"]
    assert exit_trade.entry_price == 0.0
    assert exit_trade.pnl is None
    assert runner._hub.winning_trades == 0
    assert runner._hub.losing_trades == 0


def test_check_new_fills_ignores_orders_before_current_session(monkeypatch) -> None:
    now_et = datetime(2026, 6, 3, 5, 0, tzinfo=timezone(timedelta(hours=-4)))
    yesterday = datetime(2026, 6, 2, 17, 32, tzinfo=timezone.utc)
    orders = [
        _order(
            order_id="old-buy",
            side="buy",
            status="filled",
            qty=100,
            price=5.00,
            ts=yesterday,
        ),
    ]
    runner = _runner_for_fill_sync(orders)
    monkeypatch.setattr(AlpacaRunner, "_now_et", classmethod(lambda cls: now_et))

    runner._check_new_fills()

    assert list(runner._hub.trades) == []
    assert runner._last_synced_order_ids == {"old-buy"}
