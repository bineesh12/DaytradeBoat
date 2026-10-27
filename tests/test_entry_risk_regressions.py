from daytrading.models import BarSource
"""Regression checks for fill accounting and risk at delayed order release."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from daytrading.execution.entry_executor import EntryExecutor, EntryExecutionContext
from daytrading.exits.manager import ExitManager, TrackedPosition
from daytrading.models import (
    Bar, Fill, OrderStatus, PortfolioState, Position, Side, SignalAction, TradeSignal,
)
from daytrading.runner import AlpacaRunner
from daytrading.strategy.entry_policy import EntryPolicy


NOW = datetime(2026, 10, 14, 14, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("action,side", [
    (SignalAction.ENTER_LONG, Side.BUY),
    (SignalAction.REENTER_LONG, Side.BUY),
    (SignalAction.ENTER_SHORT, Side.SELL),
    (SignalAction.REENTER_SHORT, Side.SELL),
])
def test_exit_registration_preserves_entry_direction(action, side):
    manager = ExitManager()
    signal = TradeSignal(symbol="TEST", action=action, quantity=100, entry_price=10)
    manager.register_from_signal(signal, NOW, fill_price=10, fill_quantity=40)
    tracked = manager.tracked["TEST"]
    assert tracked.side is side
    assert tracked.quantity == tracked.remaining_qty == tracked.original_qty == 40
    assert (tracked.stop_loss < 10) == (side is Side.BUY)


def test_partial_fill_uses_executed_quantity_for_dollar_stop():
    manager = ExitManager(max_unrealized_loss=10)
    signal = TradeSignal(symbol="TEST", action=SignalAction.ENTER_LONG,
                         quantity=100, entry_price=10, stop_loss=9)
    manager.register_from_signal(signal, NOW, fill_price=10, fill_quantity=40)
    tracked = manager.tracked["TEST"]
    assert tracked.remaining_qty == 40
    assert (tracked.entry_price - tracked.stop_loss) * tracked.remaining_qty == pytest.approx(10)


def test_runner_opens_partial_fill_with_actual_size():
    manager = ExitManager()
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._pipeline = SimpleNamespace(exit_manager=manager, _original_sizes={})
    runner._arm_broker_protection = Mock()
    runner._push_positions_from_alpaca = Mock()
    signal = TradeSignal(symbol="TEST", action=SignalAction.REENTER_LONG,
                         quantity=100, entry_price=10, stop_loss=9)
    fill = Fill(symbol="TEST", side=Side.BUY, quantity=40, price=10, ts=NOW)
    runner._on_position_opened(signal, fill, strategy="test", execution_method="10s_timed")
    assert manager.tracked["TEST"].remaining_qty == 40
    assert manager.tracked["TEST"].side is Side.BUY
    assert runner._pipeline._original_sizes["TEST"] == 40
    assert runner._arm_broker_protection.call_args.args[1] == 40


def _timed_context(positions=None):
    manager = ExitManager()
    pipeline = SimpleNamespace(
        portfolio=PortfolioState(cash=10000, positions=positions or {}),
        exit_manager=manager, _max_positions=1, _max_order_shares=100,
        _max_position_shares=100, _exit_cooldowns={}, _cooldown_seconds=60,
        _daily_losers=set(), _symbol_entry_counts={}, _max_entries_per_symbol=3,
    )
    broker = SimpleNamespace(submit=Mock(return_value=(None, OrderStatus.REJECTED)))
    bar = Bar(symbol="TEST", ts=NOW, open=10, high=10.1, low=9.9, close=10, volume=10000, provenance=BarSource.CANONICAL_INTERVAL)
    ctx = EntryExecutionContext(
        new_entries_blocked=lambda *a: False, pipeline=pipeline,
        bar_buffer={"TEST": [bar]}, hub=SimpleNamespace(add_log=Mock(), on_fill=Mock()),
        broker=broker, journal=SimpleNamespace(record=Mock()),
        chase_reject=lambda *a: None, record_entry_reject=Mock(),
        record_missed_a_plus=Mock(), shared_entry_quality_reject=lambda *a, **k: None,
        execution_learning_context=lambda *a: {}, is_hot_hod_timed_signal=lambda *a: False,
        retry_hot_hod_timed_entry=Mock(), on_position_opened=Mock(),
        market_phase=lambda: "REGULAR", seed_recent_order_ids=Mock(),
    )
    return ctx


@pytest.mark.parametrize("action", [SignalAction.ENTER_LONG, SignalAction.REENTER_LONG])
def test_timed_entry_rechecks_position_limit_after_queueing(action):
    ctx = _timed_context({"OTHER": Position(symbol="OTHER", quantity=10, avg_price=10)})
    signal = TradeSignal(symbol="TEST", action=action, quantity=10, entry_price=10)
    EntryExecutor(EntryPolicy()).execute_timed_signal(ctx, signal)
    ctx.broker.submit.assert_not_called()
    assert ctx.record_entry_reject.call_args.kwargs["reason"] == "position limit"


def test_timed_entry_rechecks_order_share_limit():
    ctx = _timed_context()
    signal = TradeSignal(symbol="TEST", action=SignalAction.ENTER_LONG, quantity=101, entry_price=10)
    EntryExecutor(EntryPolicy()).execute_timed_signal(ctx, signal)
    ctx.broker.submit.assert_not_called()
    assert ctx.record_entry_reject.call_args.kwargs["reason"] == "position_risk_limit"


def test_timed_scale_up_rechecks_total_position_shares():
    ctx = _timed_context({"TEST": Position(symbol="TEST", quantity=90, avg_price=10)})
    ctx.pipeline.exit_manager.track(TrackedPosition(
        symbol="TEST", side=Side.BUY, quantity=90, entry_price=10,
        sold_half=True, breakeven_locked=True,
    ))
    signal = TradeSignal(symbol="TEST", action=SignalAction.SCALE_UP_LONG, quantity=20, entry_price=10)
    EntryExecutor(EntryPolicy()).execute_timed_signal(ctx, signal)
    ctx.broker.submit.assert_not_called()
    assert ctx.record_entry_reject.call_args.kwargs["reason"] == "position_risk_limit"


def test_timed_scale_up_within_share_cap_does_not_consume_new_position_slot():
    ctx = _timed_context({"TEST": Position(symbol="TEST", quantity=90, avg_price=10)})
    ctx.pipeline.exit_manager.track(TrackedPosition(
        symbol="TEST", side=Side.BUY, quantity=90, entry_price=10,
        sold_half=True, breakeven_locked=True,
    ))
    signal = TradeSignal(symbol="TEST", action=SignalAction.SCALE_UP_LONG, quantity=10, entry_price=10)
    EntryExecutor(EntryPolicy()).execute_timed_signal(ctx, signal)
    ctx.broker.submit.assert_called_once()


def test_timed_entry_with_capacity_still_submits():
    ctx = _timed_context()
    signal = TradeSignal(symbol="TEST", action=SignalAction.ENTER_LONG, quantity=10, entry_price=10)
    EntryExecutor(EntryPolicy()).execute_timed_signal(ctx, signal)
    ctx.broker.submit.assert_called_once()
