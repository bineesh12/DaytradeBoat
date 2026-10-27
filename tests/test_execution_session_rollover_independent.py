"""Independent owner-boundary rollover probes. All state and SDKs are local fakes.

No date reset, financial rewrite, broker request, order, or network operation is
performed by these fixtures. Actual staged store/broker/runner classes are used.
"""
from concurrent.futures import Future
from datetime import datetime, timezone
import sqlite3
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.execution import durability
from daytrading.exits.manager import TrackedPosition
from daytrading.models import PortfolioState, Position, Side
from daytrading.pipeline.engine import TradingPipeline
from daytrading.runner import AlpacaRunner


OLD = "2026-10-23"
DAY = "2026-10-26"
NOW = datetime(2026, 10, 26, 8, 5, tzinfo=timezone.utc)


def broker():
    b = AlpacaBroker.__new__(AlpacaBroker)
    b._paper = True
    b._account_scope = "isolated-paper-account"
    b._safety_state()
    b._client = Mock()
    # Merely checking readiness or rotating a store must never perform I/O.
    for name in ("get_all_positions", "get_orders", "get_order_by_id", "get_account",
                 "submit_order", "cancel_order_by_id", "replace_order_by_id"):
        getattr(b._client, name).side_effect = AssertionError("Unexpected SDK call")
    b.bind_position_refresh_owner(("isolated", 0))
    b._position_future = None
    b._pos_cache = {}
    b._pos_cache_ts = 100.0
    b._position_sync_stale = False
    return b


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setattr(durability, "session_day", lambda: OLD)
    monkeypatch.setattr("daytrading.execution.alpaca_broker.time.monotonic", lambda: 100.1)
    monkeypatch.setattr("daytrading.ml.artifacts.data_root", lambda: tmp_path)
    b = broker()
    p = TradingPipeline([], {}, b, PortfolioState(cash=12345.67))
    r = AlpacaRunner.__new__(AlpacaRunner)
    r._broker, r._pipeline = b, p
    r._journal = NS(record=Mock())
    r._last_session_reset_day = OLD
    r._partial_entries = {}
    r._pipeline_protection_pending = {}
    r._last_synced_order_ids = set()
    r._fill_sync_complete = True
    r._execution_recovery_required = True
    r._session_has_fills = False
    r._fill_baseline_verified = True
    old = durability.ExecutionStore(tmp_path / "execution-state" / (OLD + ".sqlite3"), r)
    r._execution_store = old
    r._execution_session_day = OLD
    r._journal._execution_store = old
    b._durability_checkpoint = old.before_broker
    old.save()
    # A normal session reset has already occurred in memory. This fixture never
    # invokes reset or changes a persisted date to make the rollover succeed.
    r._last_session_reset_day = DAY
    p._daily_pnl = 12.75
    p._daily_loss_counts = {"HISTORICAL": 1}
    monkeypatch.setattr(durability, "session_day", lambda: DAY)
    yield r, p, b, old, tmp_path
    for store in (old, getattr(r, "_execution_store", None)):
        if store is not None:
            try:
                store.db.close()
            except sqlite3.Error:
                pass


@pytest.mark.parametrize("age,expected", [
    (0.0, True), (0.499999, True), (-0.000001, False),
    (0.5, False), (5.0, False), (float("nan"), False), (float("inf"), False),
])
def test_pure_position_snapshot_obeys_original_half_second_ttl(monkeypatch, age, expected):
    b = broker()
    monkeypatch.setattr("daytrading.execution.alpaca_broker.time.monotonic", lambda: 100.0 + age)
    before = (b._pos_cache, b._pos_cache_ts, b._position_sync_stale, b._position_future)
    assert b.position_snapshot_current() is expected
    assert (b._pos_cache, b._pos_cache_ts, b._position_sync_stale, b._position_future) == before
    assert b._client.mock_calls == []


@pytest.mark.parametrize("change", ["missing_cache", "stale", "pending", "completed_unadopted", "unbound_owner"])
def test_snapshot_rejects_uncertain_or_unadopted_state_without_sdk(monkeypatch, change):
    b = broker()
    monkeypatch.setattr("daytrading.execution.alpaca_broker.time.monotonic", lambda: 100.1)
    if change == "missing_cache":
        b._pos_cache = None
    elif change == "stale":
        b._position_sync_stale = True
    elif change in ("pending", "completed_unadopted"):
        b._position_future = Future()
        if change == "completed_unadopted":
            b._position_future.set_result([])
    else:
        del b._position_owner
    assert b.position_snapshot_current() is False
    assert b._client.mock_calls == []


def test_snapshot_read_is_owner_only_and_has_no_adoption_side_effect(monkeypatch):
    b = broker()
    monkeypatch.setattr("daytrading.execution.alpaca_broker.time.monotonic", lambda: 100.1)
    out = []
    t = threading.Thread(target=lambda: out.append(b.position_snapshot_current()))
    t.start()
    t.join(timeout=1)
    assert not t.is_alive() and out == [False]
    assert b.position_snapshot_current() is True
    assert b._client.mock_calls == []


def test_normal_owner_boundary_preserves_economics_and_keeps_recovery_latched(ready):
    r, p, b, old, root = ready
    before = durability.capture(r)
    assert r._maybe_rollover_execution_store() is True
    new = r._execution_store
    assert new is not old and new.day == DAY
    assert r._journal._execution_store is new and r._execution_session_day == DAY
    assert r._execution_recovery_required is True
    assert durability.capture(r) == before
    assert p.portfolio.cash == 12345.67 and p._daily_pnl == 12.75
    assert p._daily_loss_counts == {"HISTORICAL": 1}
    assert (root / "execution-state" / (OLD + ".sqlite3")).is_file()
    assert (root / "execution-state" / (DAY + ".sqlite3")).is_file()
    assert r._maybe_rollover_execution_store() is False
    assert r._execution_store is new and b._client.mock_calls == []


def block(r, p, b, old, name):
    if name == "reset_not_complete":
        r._last_session_reset_day = OLD
    elif name == "fill_sync_not_complete":
        r._fill_sync_complete = False
    elif name == "long_position":
        p.portfolio.positions["TEST"] = Position("TEST", 10, 3.)
    elif name == "short_position":
        p.portfolio.positions["TEST"] = Position("TEST", -10, 3.)
    elif name == "tracked_exit":
        p.exit_manager.track(TrackedPosition(symbol="TEST", side=Side.BUY,
            quantity=10, remaining_qty=10, entry_price=3.))
    elif name == "partial_entry":
        r._partial_entries = {"TEST": object()}
    elif name == "protection_pending":
        r._pipeline_protection_pending = {"TEST": "pending"}
    elif name == "uncertain_order":
        b._unsettled_orders = {"TEST": ("order-id", "request-id")}
    elif name == "safety_block":
        b._safety_blocked = True
    elif name == "unconfirmed_cancel":
        b._cancel_unconfirmed = True
    elif name == "retained_cancel":
        b._unconfirmed_cancels = {"order-id"}
    elif name == "broker_position":
        b._pos_cache = {"TEST": {"qty": 10., "avg_entry": 3.}}
    elif name == "stale_snapshot":
        b._position_sync_stale = True
    elif name == "expired_snapshot":
        b._pos_cache_ts = 99.0
    elif name == "future_snapshot":
        b._pos_cache_ts = 101.0
    elif name == "pending_snapshot":
        b._position_future = Future()
    elif name == "store_failed":
        old.failed = True
    elif name == "store_invalid":
        old.invalid_restore = True
    elif name == "nested_operation":
        old.depth = 1
    else:
        raise AssertionError(name)


@pytest.mark.parametrize("reason", [
    "reset_not_complete", "fill_sync_not_complete", "long_position", "short_position",
    "tracked_exit", "partial_entry", "protection_pending", "uncertain_order", "safety_block",
    "unconfirmed_cancel", "retained_cancel", "broker_position", "stale_snapshot", "expired_snapshot",
    "future_snapshot", "pending_snapshot", "store_failed", "store_invalid", "nested_operation",
])
def test_each_uncertain_boundary_defers_rollover_without_creating_new_store(ready, reason):
    r, p, b, old, root = ready
    block(r, p, b, old, reason)
    cash = p.portfolio.cash
    assert r._maybe_rollover_execution_store() is False
    assert r._execution_store is old and r._journal._execution_store is old
    assert r._execution_session_day == OLD
    assert p.portfolio.cash == cash and r._execution_recovery_required is True
    assert not (root / "execution-state" / (DAY + ".sqlite3")).exists()
    assert b._client.mock_calls == []


def test_real_outer_operation_cannot_switch_its_owned_store(ready):
    r, _, b, old, root = ready
    with old.operation():
        assert old.depth == 1
        assert r._maybe_rollover_execution_store() is False
        assert r._execution_store is old
        assert not (root / "execution-state" / (DAY + ".sqlite3")).exists()
    assert old.depth == 0 and b._client.mock_calls == []


def test_initialize_rejects_nesting_before_opening_target(ready):
    r, _, _, old, root = ready
    old.depth = 1
    try:
        with pytest.raises((RuntimeError, ValueError)):
            durability.initialize(r, root, preserve_current=True)
    finally:
        old.depth = 0
    assert r._execution_store is old
    assert not (root / "execution-state" / (DAY + ".sqlite3")).exists()


def test_non_owner_rollover_defers_without_financial_or_sdk_changes(ready):
    r, p, b, old, root = ready
    out = []
    t = threading.Thread(target=lambda: out.append(r._maybe_rollover_execution_store()))
    t.start()
    t.join(timeout=1)
    assert not t.is_alive() and out == [False]
    assert r._execution_store is old and p.portfolio.cash == 12345.67
    assert not (root / "execution-state" / (DAY + ".sqlite3")).exists()
    assert b._client.mock_calls == []


def test_iteration_services_adoption_then_rollover_before_next_sync(monkeypatch):
    r = AlpacaRunner.__new__(AlpacaRunner)
    calls = []
    r._service_position_refresh = lambda: calls.append("adopt")
    r._maybe_rollover_execution_store = lambda: calls.append("rollover") or False
    r._maybe_clear_execution_recovery = lambda: calls.append("recovery")
    r._service_broker_sync = lambda: calls.append("sync")
    r._next_position_sync_at = 0.
    r._now_et = lambda: NOW
    r._market_phase = lambda: "PRE-MARKET"
    r._maybe_daily_session_reset = Mock()
    r._is_in_trading_window = lambda when: False
    r._is_in_pattern_scan_window = lambda when: True
    r._enforce_session_window = Mock()
    monkeypatch.setattr("daytrading.runner.time.monotonic", lambda: 100.1)
    assert r._session_iteration_context() == (NOW, False, True)
    assert calls.index("adopt") < calls.index("rollover") < calls.index("sync")


@pytest.mark.parametrize("scope", ["", "different-paper-account"])
def test_rollover_cannot_change_or_lose_verified_account_identity(ready, scope):
    r, p, b, old, root = ready
    b._account_scope = scope
    assert r._maybe_rollover_execution_store() is False
    assert r._execution_store is old and r._journal._execution_store is old
    assert p.portfolio.cash == 12345.67 and r._execution_recovery_required is True
    assert not (root / "execution-state" / (DAY + ".sqlite3")).exists()
    assert b._client.mock_calls == []


def test_recovery_clear_database_read_failure_retains_block_without_crashing_owner(ready):
    r, _, b, _, _ = ready
    assert r._maybe_rollover_execution_store() is True
    store = r._execution_store
    connection = store.db

    class UnavailableRead:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("isolated unavailable checkpoint read")

    try:
        store.db = UnavailableRead()
        assert r._maybe_clear_execution_recovery() is False
        assert r._execution_recovery_required is True
        assert r._execution_store is store and b._client.mock_calls == []
    finally:
        store.db = connection


def test_queued_old_operation_rechecks_active_store_before_mutating_owner(ready):
    r, p, b, old, _ = ready
    waiting = threading.Event()
    mutated = []
    rejected = []
    checkpoint_before = old.db.execute("SELECT payload,sha256 FROM checkpoint WHERE id=1").fetchone()

    def queued_operation():
        # Capture the old operation before its lock can be acquired, exactly
        # as a durable wrapper can do while the owner is preparing rollover.
        operation = old.operation()
        waiting.set()
        try:
            with operation:
                mutated.append(True)
                p.portfolio.cash -= 1.
        except (RuntimeError, ValueError) as exc:
            rejected.append(type(exc).__name__)

    t = threading.Thread(target=queued_operation)
    try:
        with old.lock:
            t.start()
            assert waiting.wait(timeout=1)
            assert old.depth == 0 and not mutated
            assert r._maybe_rollover_execution_store() is True
            assert r._execution_store is not old
        t.join(timeout=1)
        assert not t.is_alive()
        assert mutated == [] and len(rejected) == 1
        assert p.portfolio.cash == 12345.67
        assert r._execution_recovery_required is True
        assert old.db.execute("SELECT payload,sha256 FROM checkpoint WHERE id=1").fetchone() == checkpoint_before
        assert r._execution_store._validated_checkpoint()[0]["state"] == durability.capture(r)
        assert b._client.mock_calls == []
    finally:
        if t.is_alive():
            t.join(timeout=1)
