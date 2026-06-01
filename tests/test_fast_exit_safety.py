"""Safety logic for fast-exit and other non-pipeline exit paths."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace

from daytrading.exits.manager import ExitManager, TrackedPosition
from daytrading.models import Fill, OrderStatus, PortfolioState, Side, SignalAction, TradeSignal
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.runner import AlpacaRunner


def test_loss_exit_blacklists_symbol_for_day() -> None:
    pipeline = create_scalping_pipeline(
        initial_cash=10_000, enable_daily_loser_blacklist=True,
    )
    pnl = pipeline.record_realized_exit("LOSER", 10.0, 9.0, 100)
    assert pnl == -100.0
    assert "LOSER" in pipeline._daily_losers
    assert pipeline._daily_pnl == -100.0


def test_loss_exit_does_not_blacklist_when_disabled() -> None:
    pipeline = create_scalping_pipeline(
        initial_cash=10_000, enable_daily_loser_blacklist=False,
    )
    pipeline.record_realized_exit("LOSER", 10.0, 9.0, 100)
    assert "LOSER" not in pipeline._daily_losers


def test_circuit_breaker_trips_on_max_daily_loss() -> None:
    pipeline = create_scalping_pipeline(initial_cash=10_000)
    pipeline._max_daily_loss = -50.0
    pipeline.record_realized_exit("A", 10.0, 9.0, 100)
    assert pipeline._circuit_breaker_tripped is True


def test_winner_does_not_blacklist() -> None:
    pipeline = create_scalping_pipeline(initial_cash=10_000)
    pipeline.record_realized_exit("WIN", 5.0, 5.5, 100)
    assert "WIN" not in pipeline._daily_losers
    assert pipeline._daily_pnl == 50.0


def _runner_for_exit_recording(exit_manager: ExitManager) -> AlpacaRunner:
    runner = object.__new__(AlpacaRunner)
    runner._pipeline = SimpleNamespace(
        exit_manager=exit_manager,
        record_realized_exit=lambda *args, **kwargs: 0.0,
    )
    runner._hub = SimpleNamespace(
        on_exit_fill=lambda *args, **kwargs: None,
        add_log=lambda *args, **kwargs: None,
    )
    runner._journal = SimpleNamespace(record=lambda *args, **kwargs: None)
    runner._exec_timer = SimpleNamespace(cancel=lambda *args, **kwargs: None)
    runner._timed_signal_queue = deque()
    runner._market_phase = lambda: "TEST"
    return runner


def test_full_exit_record_clears_broker_stop_when_position_qty_zero() -> None:
    exit_manager = ExitManager()
    pos = TrackedPosition(
        symbol="OLOX",
        side=Side.BUY,
        quantity=100,
        remaining_qty=0,
        entry_price=10.0,
    )
    exit_manager.track(pos)
    pos.remaining_qty = 0
    runner = _runner_for_exit_recording(exit_manager)

    calls = []
    runner._clear_broker_stop = lambda symbol: calls.append(("clear", symbol))
    runner._refresh_broker_stop = lambda symbol: calls.append(("refresh", symbol))

    runner._record_trade_exit(
        Fill("OLOX", Side.SELL, 100, 11.0, datetime.now(timezone.utc)),
        entry_price=10.0,
        reason="tick_trailing_stop",
    )

    assert calls == [("clear", "OLOX")]


def test_partial_exit_record_refreshes_broker_stop_for_remaining_qty() -> None:
    exit_manager = ExitManager()
    pos = TrackedPosition(
        symbol="OLOX",
        side=Side.BUY,
        quantity=100,
        remaining_qty=50,
        entry_price=10.0,
        stop_loss=10.0,
    )
    exit_manager.track(pos)
    runner = _runner_for_exit_recording(exit_manager)

    calls = []
    runner._clear_broker_stop = lambda symbol: calls.append(("clear", symbol))
    runner._refresh_broker_stop = lambda symbol: calls.append(("refresh", symbol))

    runner._record_trade_exit(
        Fill("OLOX", Side.SELL, 50, 11.0, datetime.now(timezone.utc)),
        entry_price=10.0,
        reason="take_profit",
    )

    assert calls == [("refresh", "OLOX")]


def test_failed_fast_exit_attempts_are_logged_for_shadow_ml(monkeypatch) -> None:
    class ExitManagerStub:
        def __init__(self) -> None:
            self._positions = {
                "MASK": SimpleNamespace(
                    entry_price=5.59,
                    sold_half=False,
                    remaining_qty=294,
                    risk_per_share=0.10,
                    stop_loss=5.49,
                )
            }

        @property
        def tracked(self):
            return dict(self._positions)

        def check_exits(self, prices, now):
            return [
                TradeSignal(
                    symbol="MASK",
                    action=SignalAction.EXIT_LONG,
                    quantity=294,
                    entry_price=prices["MASK"],
                    reason="take_profit",
                )
            ]

    logged = []

    def fake_log_execution_quality(**kwargs):
        logged.append(kwargs)

    monkeypatch.setattr(
        "daytrading.ml.shadow_collector.log_execution_quality",
        fake_log_execution_quality,
    )

    runner = object.__new__(AlpacaRunner)
    runner._pipeline = SimpleNamespace(
        exit_manager=ExitManagerStub(),
        portfolio=PortfolioState(cash=10_000),
        set_cooldown=lambda *args, **kwargs: None,
    )
    runner._broker = SimpleNamespace(
        submit=lambda *args, **kwargs: (None, OrderStatus.CANCELLED)
    )
    runner._live_prices = lambda symbols: {"MASK": 6.94}

    runner._check_exits_only()

    assert [row["source"] for row in logged] == [
        "fast_exit_limit",
        "fast_exit_guarded_marketable",
    ]
    assert all(row["status"] is OrderStatus.CANCELLED for row in logged)


def test_paused_runner_blocks_breakout_scalp_entries() -> None:
    runner = object.__new__(AlpacaRunner)
    runner._pending_breakout_scalps = deque([("HOT", 5.0, 1.0)])
    runner._hub = SimpleNamespace(trading_paused=True)
    runner._pipeline = SimpleNamespace(_circuit_breaker_tripped=False, _daily_pnl=0.0)

    runner._process_breakout_scalps()

    assert not runner._pending_breakout_scalps


def test_paused_runner_blocks_timed_entries() -> None:
    runner = object.__new__(AlpacaRunner)
    runner._hub = SimpleNamespace(trading_paused=True)
    runner._pipeline = SimpleNamespace(_circuit_breaker_tripped=False, _daily_pnl=0.0)
    calls = []
    runner._broker = SimpleNamespace(submit=lambda *args, **kwargs: calls.append(args))

    runner._execute_timed_signal(
        TradeSignal(
            symbol="HOT",
            action=SignalAction.ENTER_LONG,
            quantity=100,
            entry_price=5.0,
        )
    )

    assert calls == []
