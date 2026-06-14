"""Safety logic for fast-exit and other non-pipeline exit paths."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
import time
from types import SimpleNamespace

from daytrading.exits.manager import ExitManager, TrackedPosition
from daytrading.models import Bar, Fill, Order, OrderStatus, PortfolioState, Position, Side, SignalAction, TradeSignal
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


def test_small_loss_does_not_blacklist_until_second_loss() -> None:
    pipeline = create_scalping_pipeline(
        initial_cash=10_000, enable_daily_loser_blacklist=True,
    )

    first_pnl = pipeline.record_realized_exit("SDOT", 7.67, 7.65, 57)
    assert round(first_pnl, 2) == -1.14
    assert "SDOT" not in pipeline._daily_losers
    assert pipeline._daily_loss_counts["SDOT"] == 1

    second_pnl = pipeline.record_realized_exit("SDOT", 7.70, 7.68, 57)
    assert round(second_pnl, 2) == -1.14
    assert "SDOT" in pipeline._daily_losers
    assert pipeline._daily_loss_counts["SDOT"] == 2


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


def test_flat_portfolio_exit_clears_stale_tracked_stop() -> None:
    exit_manager = ExitManager()
    pos = TrackedPosition(
        symbol="SDOT",
        side=Side.BUY,
        quantity=57,
        remaining_qty=57,
        entry_price=7.6693,
        stop_loss=7.5693,
    )
    exit_manager.track(pos)
    runner = _runner_for_exit_recording(exit_manager)
    runner._pipeline.portfolio = PortfolioState(cash=10_000)
    runner._pipeline.portfolio.positions["SDOT"] = Position(
        symbol="SDOT", quantity=0, avg_price=7.6693,
    )
    runner._reconciler = SimpleNamespace(clear_pending=lambda symbol: None)

    calls = []
    runner._clear_broker_stop = lambda symbol: calls.append(("clear", symbol))
    runner._refresh_broker_stop = lambda symbol: calls.append(("refresh", symbol))

    runner._record_trade_exit(
        Fill("SDOT", Side.SELL, 57, 7.6474, datetime.now(timezone.utc)),
        entry_price=7.6693,
        reason="fast_exit",
    )

    assert calls == [("clear", "SDOT")]
    assert "SDOT" not in exit_manager.tracked


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


def test_confirmed_runner_core_uses_runner_management() -> None:
    pos = TrackedPosition(
        symbol="CHAI",
        side=Side.BUY,
        quantity=499,
        remaining_qty=333,
        entry_price=3.697,
        stop_loss=3.70,
        sold_half=True,
        breakeven_locked=True,
    )
    pos.runner_confirmed = True

    assert AlpacaRunner._uses_runner_core_management(pos) is True


def test_confirmed_runner_skips_10s_profit_protection_exit() -> None:
    exit_manager = ExitManager()
    pos = TrackedPosition(
        symbol="CHAI",
        side=Side.BUY,
        quantity=499,
        remaining_qty=333,
        entry_price=3.697,
        entry_ts=datetime.now(timezone.utc) - timedelta(seconds=120),
        stop_loss=3.70,
        sold_half=True,
        breakeven_locked=True,
    )
    pos.runner_confirmed = True
    pos.highest_price = 3.90
    exit_manager.track(pos)

    runner = object.__new__(AlpacaRunner)
    runner._pipeline = SimpleNamespace(
        exit_manager=exit_manager,
        portfolio=PortfolioState(cash=10_000),
    )
    runner._bar_aggregator = SimpleNamespace(
        get_latest_10s=lambda symbol, count=2: [
            Bar(symbol=symbol, open=3.86, high=3.88, low=3.82, close=3.85, volume=10_000, ts=datetime.now(timezone.utc)),
            Bar(symbol=symbol, open=3.85, high=3.86, low=3.78, close=3.79, volume=12_000, ts=datetime.now(timezone.utc)),
        ]
    )
    calls = []
    runner._broker = SimpleNamespace(submit=lambda *args, **kwargs: calls.append(args))
    runner._record_trade_exit = lambda *args, **kwargs: 0.0
    runner._hub = SimpleNamespace(add_log=lambda *args, **kwargs: None)
    runner._seed_recent_order_ids = lambda: None
    runner._push_positions_from_alpaca = lambda: None

    red_bar = Bar(
        symbol="CHAI",
        open=3.85,
        high=3.86,
        low=3.78,
        close=3.79,
        volume=12_000,
        ts=datetime.now(timezone.utc),
    )
    runner._check_10s_candle_exit("CHAI", red_bar)

    assert calls == []
    assert exit_manager.tracked["CHAI"].remaining_qty == 333


def test_filled_position_stop_is_capped_by_dollar_risk_after_slippage() -> None:
    exit_manager = ExitManager(max_unrealized_loss=50.0)
    signal = TradeSignal(
        symbol="MNTS",
        action=SignalAction.ENTER_LONG,
        quantity=151,
        entry_price=16.92,
        stop_loss=16.59,
        reason="First Pullback Reclaim MNTS",
    )

    exit_manager.register_from_signal(
        signal,
        datetime.now(timezone.utc),
        fill_price=16.99,
    )

    pos = exit_manager.tracked["MNTS"]
    assert round(pos.entry_price - pos.stop_loss, 4) <= round(50.0 / 151, 4)
    assert pos.stop_loss > 16.59


def test_fast_exit_uses_short_broker_wait_then_restores() -> None:
    runner = object.__new__(AlpacaRunner)
    runner._pipeline = SimpleNamespace(portfolio=PortfolioState(cash=10_000))
    waits = []

    class BrokerStub:
        _max_wait = 5.0

        def submit(self, order, bar, portfolio):
            waits.append(self._max_wait)
            return None, OrderStatus.CANCELLED

    runner._broker = BrokerStub()
    order = SimpleNamespace(symbol="MASK")
    bar = SimpleNamespace(symbol="MASK")

    fill, status = runner._submit_fast_exit_order(order, bar)

    assert fill is None
    assert status is OrderStatus.CANCELLED
    assert waits == [1.0]
    assert runner._broker._max_wait == 5.0


def test_fast_exit_clamps_to_fresh_broker_position_qty() -> None:
    runner = object.__new__(AlpacaRunner)
    runner._pipeline = SimpleNamespace(portfolio=PortfolioState(cash=10_000))
    submitted = []

    class BrokerStub:
        _max_wait = 5.0

        def __init__(self) -> None:
            self.invalidated = False

        def _invalidate_position_cache(self):
            self.invalidated = True

        def get_positions(self):
            assert self.invalidated
            return {"VERU": {"qty": 71}}

        def submit(self, order, bar, portfolio):
            submitted.append(order)
            return Fill(order.symbol, order.side, order.quantity, 5.54, datetime.now(timezone.utc)), OrderStatus.FILLED

    broker = BrokerStub()
    runner._broker = broker

    order = Order(symbol="VERU", side=Side.SELL, quantity=147, limit_price=None)
    bar = Bar(symbol="VERU", open=5.55, high=5.55, low=5.55, close=5.55, volume=0, ts=datetime.now(timezone.utc))

    fill, status = runner._submit_fast_exit_order(order, bar)

    assert status is OrderStatus.FILLED
    assert fill is not None
    assert submitted[0].quantity == 71
    assert runner._broker._max_wait == 5.0


def test_fast_exit_skips_when_broker_is_already_flat() -> None:
    runner = object.__new__(AlpacaRunner)
    runner._pipeline = SimpleNamespace(portfolio=PortfolioState(cash=10_000))

    class BrokerStub:
        _max_wait = 5.0

        def _invalidate_position_cache(self):
            pass

        def get_positions(self):
            return {}

        def submit(self, order, bar, portfolio):
            raise AssertionError("flat broker position should not submit an exit")

    runner._broker = BrokerStub()
    order = Order(symbol="VERU", side=Side.SELL, quantity=147, limit_price=None)
    bar = Bar(symbol="VERU", open=5.55, high=5.55, low=5.55, close=5.55, volume=0, ts=datetime.now(timezone.utc))

    fill, status = runner._submit_fast_exit_order(order, bar)

    assert fill is None
    assert status is OrderStatus.CANCELLED


def test_paused_runner_blocks_breakout_scalp_entries() -> None:
    runner = object.__new__(AlpacaRunner)
    runner._pending_breakout_scalps = deque([("HOT", 5.0, 1.0)])
    runner._hub = SimpleNamespace(trading_paused=True)
    runner._pipeline = SimpleNamespace(_circuit_breaker_tripped=False, _daily_pnl=0.0)

    runner._process_breakout_scalps()

    assert not runner._pending_breakout_scalps


def test_breakout_scalp_reject_does_not_reference_missing_status() -> None:
    runner = object.__new__(AlpacaRunner)
    runner._pending_breakout_scalps = deque([("HOT", 5.0, time.time())])
    runner._breakout_scalp_active = False
    runner._breakout_scalp_cooldown = {}
    runner._pipeline = SimpleNamespace(
        portfolio=PortfolioState(cash=10_000),
        _symbol_entry_counts={},
        _max_entries_per_symbol=2,
        _symbol_cooldowns={},
        _symbol_last_entry_time={},
        _exit_cooldowns={},
    )
    runner._bar_buffer = {
        "HOT": deque([
            Bar(symbol="HOT", open=5.0, high=5.1, low=4.9, close=5.0, volume=10_000, ts=datetime.now(timezone.utc)),
            Bar(symbol="HOT", open=5.0, high=5.2, low=4.95, close=5.1, volume=12_000, ts=datetime.now(timezone.utc)),
        ])
    }
    runner._new_entries_blocked = lambda *args, **kwargs: False
    runner._quick_scalp_hod_alert_reject = lambda symbol: "watch only"
    runner._broker = SimpleNamespace(submit=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not submit")))

    runner._process_breakout_scalps()

    assert not runner._pending_breakout_scalps


def test_breakout_scalp_honors_spread_size_factor_before_submit() -> None:
    runner = object.__new__(AlpacaRunner)
    runner._pending_breakout_scalps = deque([("HOT", 5.0, time.time())])
    runner._breakout_scalp_active = False
    runner._breakout_scalp_cooldown = {}
    runner._quick_scalp_spread_size_factors = {"HOT": 0.35}
    runner._pipeline = SimpleNamespace(
        portfolio=PortfolioState(cash=10_000),
        _symbol_entry_counts={},
        _max_entries_per_symbol=2,
        _exit_cooldowns={},
        _cooldown_seconds=0,
    )
    now = datetime.now(timezone.utc)
    runner._bar_buffer = {
        "HOT": deque([
            Bar(symbol="HOT", open=4.80, high=5.00, low=4.75, close=4.90, volume=300_000, ts=now),
            Bar(symbol="HOT", open=4.90, high=5.20, low=4.85, close=5.10, volume=350_000, ts=now),
            Bar(symbol="HOT", open=5.10, high=5.35, low=5.05, close=5.30, volume=400_000, ts=now),
        ])
    }
    submitted = []

    def submit(order, bar, portfolio):
        submitted.append(order)
        return Fill("HOT", Side.BUY, order.quantity, order.limit_price or bar.close, now), OrderStatus.FILLED

    runner._new_entries_blocked = lambda *args, **kwargs: False
    runner._quick_scalp_hod_alert_reject = lambda symbol: None
    runner._quick_scalp_recent_normal_reject = lambda symbol, **kwargs: None
    runner._check_quick_scalp_entry = lambda symbol, bars: None
    runner._quick_scalp_shared_quality_reject = lambda symbol, bars: None
    runner._quick_scalp_10s_reject = lambda symbol: None
    runner._quick_scalp_tick_rr = lambda symbol, bars, alert_price: (5.30, 5.20, 5.43, "test")
    runner._broker = SimpleNamespace(submit=submit)
    runner._on_position_opened = lambda *args, **kwargs: None
    runner._hub = SimpleNamespace(on_fill=lambda *args, **kwargs: None, add_log=lambda *args, **kwargs: None)
    runner._journal = SimpleNamespace(record=lambda *args, **kwargs: None)
    runner._market_phase = lambda: "OPEN"
    runner._seed_recent_order_ids = lambda: None

    runner._process_breakout_scalps()

    assert submitted
    assert submitted[0].quantity == 175
    assert not runner._quick_scalp_spread_size_factors


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


def test_timed_entry_rechecks_shared_entry_quality_before_submit(monkeypatch) -> None:
    runner = object.__new__(AlpacaRunner)
    bars = deque([
        Bar(symbol="HOT", open=5.0, high=5.05, low=4.95, close=5.0, volume=10_000, ts=datetime.now(timezone.utc)),
        Bar(symbol="HOT", open=5.0, high=5.06, low=4.96, close=5.02, volume=10_000, ts=datetime.now(timezone.utc)),
        Bar(symbol="HOT", open=5.02, high=5.08, low=5.0, close=5.04, volume=10_000, ts=datetime.now(timezone.utc)),
    ])
    runner._hub = SimpleNamespace(
        trading_paused=False,
        logs=[],
        add_log=lambda level, message: runner._hub.logs.append((level, message)),
    )
    runner._pipeline = SimpleNamespace(
        _circuit_breaker_tripped=False,
        _daily_pnl=0.0,
        _exit_cooldowns={},
        _cooldown_seconds=60,
        _daily_losers=set(),
        _symbol_entry_counts={},
        _max_entries_per_symbol=3,
        portfolio=PortfolioState(cash=10_000),
    )
    runner._bar_buffer = {"HOT": bars}
    runner._quote_buffer = {}
    runner._tick_buffer = {}
    runner._bar_aggregator = None
    runner._float_checker = None
    runner._live_prices = lambda symbols: {"HOT": 5.04}
    runner._latest_price = lambda symbol: 5.04
    runner._broker = SimpleNamespace(
        submit=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("timed entry should not submit after shared quality reject")
        )
    )

    monkeypatch.setattr(
        "daytrading.runner.check_entry_quality",
        lambda *args, **kwargs: "test shared guard reject",
    )

    runner._execute_timed_signal(
        TradeSignal(
            symbol="HOT",
            action=SignalAction.ENTER_LONG,
            quantity=100,
            entry_price=5.04,
            scan_result=None,
        )
    )

    assert runner._hub.logs == [("WARNING", "ENTRY SKIP HOT: test shared guard reject")]


def _runner_for_short_cleanup() -> AlpacaRunner:
    runner = object.__new__(AlpacaRunner)
    exit_manager = ExitManager()
    runner._pipeline = SimpleNamespace(
        portfolio=PortfolioState(cash=10_000),
        exit_manager=exit_manager,
    )
    runner._pipeline.portfolio.positions["VERU"] = Position(
        symbol="VERU", quantity=-1, avg_price=4.28,
    )
    runner._hub = SimpleNamespace(logs=[], add_log=lambda level, message: runner._hub.logs.append((level, message)))
    runner._journal = SimpleNamespace(events=[], record=lambda event_type, payload, ts=None: runner._journal.events.append((event_type, payload)))
    runner._exec_timer = SimpleNamespace(cancel=lambda symbol: None)
    runner._timed_signal_queue = deque()
    runner._last_synced_order_ids = set()
    runner._accidental_short_cleanup_enabled = True
    runner._accidental_short_max_qty = 0.0
    runner._accidental_short_cooldown_sec = 30.0
    runner._accidental_short_cleanup_at = {}
    runner._clear_broker_stop = lambda symbol: None
    runner._seed_recent_order_ids = lambda: runner._last_synced_order_ids.add("seeded")
    return runner


def test_accidental_short_cleanup_covers_tiny_short() -> None:
    runner = _runner_for_short_cleanup()
    submitted = []

    def fake_submit(order, bar):
        submitted.append((order, bar))
        return Fill(order.symbol, order.side, order.quantity, 6.02, datetime.now(timezone.utc)), OrderStatus.FILLED

    runner._submit_fast_exit_order = fake_submit

    runner._cleanup_accidental_shorts(
        {"VERU": {"qty": -1, "avg_entry": 4.28, "current_price": 6.02}}
    )

    order, bar = submitted[0]
    assert order.symbol == "VERU"
    assert order.side is Side.BUY
    assert order.quantity == 1
    assert bar.close == 6.02
    assert "VERU" not in runner._pipeline.portfolio.positions
    assert runner._journal.events[0][0] == "short_cleanup"
    assert "seeded" in runner._last_synced_order_ids


def test_accidental_short_cleanup_covers_large_short_by_default() -> None:
    runner = _runner_for_short_cleanup()
    calls = []

    def fake_submit(order, bar):
        calls.append(order)
        return Fill(order.symbol, order.side, order.quantity, 6.02, datetime.now(timezone.utc)), OrderStatus.FILLED

    runner._submit_fast_exit_order = fake_submit

    runner._cleanup_accidental_shorts(
        {"VERU": {"qty": -25, "avg_entry": 4.28, "current_price": 6.02}}
    )

    assert len(calls) == 1
    assert calls[0].side is Side.BUY
    assert calls[0].quantity == 25


def test_accidental_short_cleanup_respects_explicit_max_qty_cap() -> None:
    runner = _runner_for_short_cleanup()
    runner._accidental_short_max_qty = 5.0
    calls = []
    runner._submit_fast_exit_order = lambda order, bar: calls.append(order)

    runner._cleanup_accidental_shorts(
        {"VERU": {"qty": -25, "avg_entry": 4.28, "current_price": 6.02}}
    )

    assert calls == []
    assert any("auto-cover skipped" in message for _, message in runner._hub.logs)


def test_accidental_short_cleanup_uses_cooldown() -> None:
    runner = _runner_for_short_cleanup()
    calls = []

    def fake_submit(order, bar):
        calls.append(order)
        return Fill(order.symbol, order.side, order.quantity, 6.02, datetime.now(timezone.utc)), OrderStatus.FILLED

    runner._submit_fast_exit_order = fake_submit

    broker_pos = {"VERU": {"qty": -1, "avg_entry": 4.28, "current_price": 6.02}}
    runner._cleanup_accidental_shorts(broker_pos)
    runner._cleanup_accidental_shorts(broker_pos)

    assert len(calls) == 1
