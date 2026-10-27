"""Current-state rollover preservation; SQLite and broker fixtures are isolated."""
from collections import deque
from concurrent.futures import Future
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from daytrading.execution import durability
from daytrading.models import PortfolioState
from daytrading.pipeline.engine import TradingPipeline
from daytrading.runner import AlpacaRunner
from daytrading.runtime_state import capture_state
from test_execution_session_rollover_independent import ready, broker, DAY, OLD, NOW


def database(store):
    return {table: store.db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall()
            for table in ('checkpoint', 'canonical_fills', 'evidence', 'report_outbox')}


def seed_fills(r, old):
    r._pipeline.portfolio._fill_projections = {'cash': {
        ('HIST', 'buy', 'paper/buy-id'): (100., 250.),
        ('HIST', 'sell', 'paper/sell-id'): (100., 260.),
    }}
    old.save()


def target_store(r, root):
    target = durability.ExecutionStore(root/'execution-state'/(DAY+'.sqlite3'), r)
    target.save()
    return target


def assert_old_intact(r, b, old, snapshot, graph):
    assert r._execution_store is old and r._journal._execution_store is old
    assert r._execution_session_day == OLD
    assert b._durability_checkpoint.__self__ is old
    assert database(old) == snapshot and durability.capture(r) == graph
    assert r._execution_recovery_required and b._client.mock_calls == []


def test_new_checkpoint_preserves_full_graph_ledger_and_schema2_handover(ready):
    r, p, b, old, root = ready
    seed_fills(r, old)
    r._hub = NS(trading_paused=True)
    r._exec_timer = NS(pending_symbols=[])
    r._timed_signal_queue = deque()
    r._recorded_exit_fill_keys = set()
    old.save()
    graph, before = durability.capture(r), database(old)
    guards = capture_state(r, NOW)
    cash_object = p.portfolio
    assert r._maybe_rollover_execution_store()
    new = r._execution_store
    assert durability.capture(r) == graph and p.portfolio is cash_object
    assert capture_state(r, NOW)['guards'] == guards['guards']
    assert database(old) == before  # No late commit into the old-day store.
    assert database(new)['canonical_fills'] == before['canonical_fills']
    assert new._validated_checkpoint()[0]['state'] == graph
    assert r._execution_recovery_required  # Clearing is a separate guarded step.
    assert r._maybe_clear_execution_recovery()
    assert not r._execution_recovery_required
    assert not r._maybe_rollover_execution_store() and r._execution_store is new
    assert b._client.mock_calls == []


def test_identical_existing_checkpoint_binds_without_restoring_owner_objects(ready, monkeypatch):
    r, p, b, old, root = ready
    seed_fills(r, old)
    target = target_store(r, root)
    target_before = database(target)
    target.db.close()
    graph, before = durability.capture(r), database(old)
    owners = (p.portfolio, p.portfolio.positions, p.exit_manager, p._daily_loss_counts)
    monkeypatch.setattr(durability.ExecutionStore, '_restore', lambda self: pytest.fail('No restore during rollover'))
    assert r._maybe_rollover_execution_store()
    assert owners == (p.portfolio, p.portfolio.positions, p.exit_manager, p._daily_loss_counts)
    assert owners[0] is p.portfolio and owners[1] is p.portfolio.positions
    assert owners[2] is p.exit_manager and owners[3] is p._daily_loss_counts
    assert database(r._execution_store) == target_before
    assert durability.capture(r) == graph and database(old) == before
    assert b._client.mock_calls == []


@pytest.mark.parametrize('change', ['cash', 'risk', 'fill_projection'])
def test_conflicting_target_never_overwrites_current_owner(ready, change):
    r, p, b, old, root = ready
    seed_fills(r, old)
    cash, pnl, projections = p.portfolio.cash, p._daily_pnl, deepcopy(p.portfolio._fill_projections)
    if change == 'cash': p.portfolio.cash += 1.
    elif change == 'risk': p._daily_pnl -= 100.
    else: p.portfolio._fill_projections['cash'][('HIST', 'buy', 'paper/other-id')] = (10., 20.)
    target = target_store(r, root)
    target_before = database(target)
    p.portfolio.cash, p._daily_pnl, p.portfolio._fill_projections = cash, pnl, projections
    graph, before = durability.capture(r), database(old)
    assert not r._maybe_rollover_execution_store()
    assert 'conflicts' in r._execution_rollover_error
    assert_old_intact(r, b, old, before, graph)
    assert database(target) == target_before
    target.db.close()


@pytest.mark.parametrize('field', ['account', 'day', 'build', 'parameters', 'schema', 'digest'])
def test_target_headers_or_digest_fail_before_binding(ready, field):
    r, _, b, old, root = ready
    target = target_store(r, root)
    raw, sha = target.db.execute('SELECT payload,sha256 FROM checkpoint').fetchone()
    payload = json.loads(raw)
    if field != 'digest':
        payload[field] = -1 if field == 'schema' else 'different'
        raw = json.dumps(payload)
        sha = hashlib.sha256(raw.encode()).hexdigest()
    else: sha = 'invalid'
    with target.db: target.db.execute('UPDATE checkpoint SET payload=?,sha256=?', (raw, sha))
    graph, before, target_before = durability.capture(r), database(old), database(target)
    assert not r._maybe_rollover_execution_store()
    assert_old_intact(r, b, old, before, graph)
    assert database(target) == target_before
    target.db.close()


@pytest.mark.parametrize('where,kind', [('old','outbox'), ('old','ledger'), ('old','digest'),
                                     ('old','missing_checkpoint'), ('target','outbox'), ('target','ledger'),
                                     ('target','orphan_ledger')])
def test_uncertain_delivery_or_ledger_never_migrates(ready, where, kind):
    r, _, b, old, root = ready
    seed_fills(r, old)
    other = target_store(r, root) if where == 'target' else old
    with other.db:
        if kind == 'outbox': other.db.execute("INSERT INTO report_outbox VALUES ('pending','{}',0)")
        elif kind == 'ledger': other.db.execute('UPDATE canonical_fills SET quantity=quantity+1')
        elif kind == 'digest': other.db.execute("UPDATE checkpoint SET sha256='invalid'")
        else: other.db.execute('DELETE FROM checkpoint')
    graph, before, other_before = durability.capture(r), database(old), database(other)
    assert not r._maybe_rollover_execution_store()
    assert_old_intact(r, b, old, before, graph)
    assert database(other) == other_before
    if where == 'old': assert not (root/'execution-state'/(DAY+'.sqlite3')).exists()
    else: other.db.close()


@pytest.mark.parametrize('stage', ['before_commit', 'after_commit'])
def test_failed_new_save_keeps_previous_bindings_and_state(ready, monkeypatch, stage):
    r, _, b, old, root = ready
    graph, before = durability.capture(r), database(old)
    save = durability.ExecutionStore.save
    def interrupted(store, *args, **kwargs):
        if store.day == DAY:
            if stage == 'after_commit': save(store, *args, **kwargs)
            raise OSError('isolated disk failure')
        return save(store, *args, **kwargs)
    monkeypatch.setattr(durability.ExecutionStore, 'save', interrupted)
    assert not r._maybe_rollover_execution_store()
    assert_old_intact(r, b, old, before, graph)
    assert (root/'execution-state'/(DAY+'.sqlite3')).exists()  # Retain provisional evidence.


def test_failed_callback_binding_restores_every_previous_binding(ready):
    r, _, b, old, _ = ready
    graph, before = durability.capture(r), database(old)
    base = type(b)
    class OneFailure(base):
        def __setattr__(self, name, value):
            if (name == '_durability_checkpoint' and getattr(self, '_refuse_new_callback', False)
                    and getattr(getattr(value, '__self__', None), 'day', None) == DAY):
                object.__setattr__(self, '_refuse_new_callback', False)
                raise RuntimeError('isolated late binding failure')
            super().__setattr__(name, value)
    b.__class__ = OneFailure
    b._refuse_new_callback = True
    assert not r._maybe_rollover_execution_store()
    assert_old_intact(r, b, old, before, graph)


def test_recovery_clear_rechecks_current_freshness_and_uncertainty(ready):
    r, _, b, _, _ = ready
    assert r._maybe_rollover_execution_store()
    b._position_sync_stale = True
    assert not r._maybe_clear_execution_recovery() and r._execution_recovery_required
    b._position_sync_stale = False
    b._unsettled_orders = {'TEST': (None, 'uncertain')}
    assert not r._maybe_clear_execution_recovery() and r._execution_recovery_required
    b._unsettled_orders.clear()
    assert r._maybe_clear_execution_recovery() and not r._execution_recovery_required


def test_crash_after_new_baseline_before_binding_can_retry_without_rewriting_state(ready, monkeypatch):
    r, _, b, old, _ = ready
    save = durability.ExecutionStore.save
    def interrupted(store, *args, **kwargs):
        save(store, *args, **kwargs)
        if store.day == DAY: raise RuntimeError('crash after atomic baseline')
    monkeypatch.setattr(durability.ExecutionStore, 'save', interrupted)
    graph, before = durability.capture(r), database(old)
    assert not r._maybe_rollover_execution_store()
    assert_old_intact(r, b, old, before, graph)
    monkeypatch.setattr(durability.ExecutionStore, 'save', save)
    assert r._maybe_rollover_execution_store()
    assert durability.capture(r) == graph and database(old) == before


def test_new_session_restart_restores_identical_economics_with_recovery_latched(ready):
    r, _, _, _, root = ready
    seed_fills(r, r._execution_store)
    graph = durability.capture(r)
    assert r._maybe_rollover_execution_store()
    r2 = AlpacaRunner.__new__(AlpacaRunner)
    b2 = broker()
    r2._broker = b2
    r2._pipeline = TradingPipeline([], {}, b2, PortfolioState(cash=1.))
    r2._journal = NS()
    store = durability.ExecutionStore(root/'execution-state'/(DAY+'.sqlite3'), r2)
    try:
        assert store.restore()
        assert durability.capture(r2) == graph
        assert r2._execution_recovery_required
        assert b2._client.mock_calls == []
    finally:
        store.db.close()


@pytest.mark.parametrize('change', ['expired', 'generation', 'context'])
def test_snapshot_change_during_persistence_refuses_binding(ready, monkeypatch, change):
    r, _, b, old, _ = ready
    graph, before = durability.capture(r), database(old)
    save = durability.ExecutionStore.save
    def changed(store, *args, **kwargs):
        save(store, *args, **kwargs)
        if store.day == DAY:
            if change == 'expired': b._pos_cache_ts = 90.
            elif change == 'generation': b._position_generation = getattr(b, '_position_generation', 0) + 1
            else: b._position_context = ('changed',)
    monkeypatch.setattr(durability.ExecutionStore, 'save', changed)
    assert not r._maybe_rollover_execution_store()
    assert_old_intact(r, b, old, before, graph)


def test_captured_old_broker_callback_refuses_before_sdk_checkpoint_or_write(ready):
    r, _, b, old, _ = ready
    graph, before = durability.capture(r), database(old)
    callback = old.before_broker
    assert r._maybe_rollover_execution_store()
    with pytest.raises(ValueError, match='inactive execution store'):
        callback({'kind': 'submit', 'request_id': 'never-submitted'})
    assert r._execution_recovery_required
    assert durability.capture(r) == graph and database(old) == before
    assert b._client.mock_calls == []


@pytest.mark.parametrize('age,rolls', [(0.001, True), (0.5, False), (0.501, False)])
def test_native_owner_schedule_adopts_before_next_three_second_refresh(ready, monkeypatch, age, rolls):
    """Model the observed 130ms response using native owner/store components.

    This is a controlled lifecycle regression, not a complete historical market
    replay or a claim that missing scanner/account inputs were reconstructed.
    """
    r, p, b, old, _ = ready
    graph = durability.capture(r)
    clock = [100.]
    monkeypatch.setattr('daytrading.runner.time.monotonic', lambda: clock[0])
    future = Future()
    b._position_pool = NS(submit=Mock(return_value=future))
    b._pos_cache_ts = 0.
    b.recover_order_safety = Mock()
    r._market_wakeup = NS(set=Mock())
    r._flush_micro_control_evidence = Mock()
    r._check_new_fills = lambda: None  # Completed recorded-empty order read.
    r._reconciler = NS(reconcile=Mock())
    r._cleanup_accidental_shorts = Mock()
    r._hub = NS(on_position_update=Mock())
    r._next_position_sync_at = 100.
    r._now_et = lambda: NOW
    r._market_phase = lambda: 'PRE-MARKET'
    r._maybe_daily_session_reset = Mock()  # Normal reset was already completed.
    r._is_in_trading_window = lambda _: False
    r._is_in_pattern_scan_window = lambda _: True
    r._enforce_session_window = Mock()
    r._session_iteration_context()
    assert r._execution_store is old and b._position_sync_stale
    assert r._next_position_sync_at == 103.
    clock[0] = 100.130
    future.set_result([])  # Explicit modeled availability, no SDK call.
    clock[0] += age
    r._session_iteration_context()
    assert (r._execution_store.day == DAY) is rolls
    assert r._execution_recovery_required is not rolls
    assert durability.capture(r) == graph
    b._position_pool.submit.assert_called_once()
    assert b._client.mock_calls == [] and p.portfolio.cash == 12345.67
