"""Atomic PAPER execution checkpoints, separate from historical trade journals.

A broker request is persisted before sending it. Economic projections and their
watermarks commit together after the outer operation. Recovery loads the last
atomic domain snapshot and resolves retained request IDs before admitting buys.
No recorded session or historical journal is rewritten by this store.
"""
from contextlib import contextmanager
from datetime import datetime as WallDatetime, timezone, timedelta
from functools import wraps
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from zoneinfo import ZoneInfo

from daytrading.replay.codec import pack, unpack

SCHEMA = 1
PIPELINE = ('_fill_reconciliation_required', '_daily_pnl', '_daily_loss_counts', '_daily_losers', '_symbol_entry_counts',
            '_circuit_breaker_tripped', '_exit_cooldowns', '_original_sizes',
            '_realized_trade_legs', '_realized_exit_fills', '_fill_projections')
MANAGER = ('_residual_confirmation_pending', '_fill_reconciliation_required', '_positions', '_retired_positions', '_committed_exit_keys', '_fill_projections')
RUNNER = ('_legacy_fill_checkpoint','_fill_reconciliation_required', '_recorded_exit_fill_keys', '_fill_projections', '_last_synced_order_ids',
          '_legacy_synced_order_ids', '_legacy_sync_watermarks', '_partial_entries', '_pipeline_protection_pending',
          '_last_session_reset_day')
BROKER = ('_order_lineage', '_unconfirmed_cancels', '_protection_states', '_suspended_stops', '_unsettled_orders', '_protective_stop_ids',
          '_failed_stop_updates', '_stop_request_ids', '_stop_submit_outcomes',
          '_conflict_fill_seen', '_entry_order_originals', '_entry_monitor_failed',
          '_entry_progress_failed', '_safety_blocked')


def session_day():
    now=WallDatetime.now(ZoneInfo('America/New_York'))
    return (now.date() if now.hour>=4 else (now-timedelta(days=1)).date()).isoformat()


def fields(obj, names):
    return {name: getattr(obj, name) for name in names if hasattr(obj, name)}


def native_broker(runner):
    broker = runner._broker
    return vars(broker).get('_target', broker)


def capture(runner):
    p = runner._pipeline
    return pack(dict(portfolio=vars(p.portfolio), pipeline=fields(p, PIPELINE),
                     manager=fields(p.exit_manager, MANAGER), runner=fields(runner, RUNNER),
                     broker=fields(native_broker(runner), BROKER)))


def parameter_digest(runner):
    from daytrading.runtime_state import PARAMETERS
    return hashlib.sha256(json.dumps(fields(runner._pipeline, PARAMETERS),
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


class ExecutionStore:
    def __init__(self, path, runner):
        self.runner = runner
        self.lock = threading.RLock()
        self.depth = 0
        self.base = None
        self.failed = False
        self.invalid_restore = False
        self.recovery_broker = {}
        self.parameters = parameter_digest(runner)
        self.account = getattr(native_broker(runner), '_account_scope', '')
        source=Path(__file__).resolve().parents[1]
        self.build=hashlib.sha256(b''.join(str(path.relative_to(source)).encode()+hashlib.sha256(path.read_bytes()).digest()
            for path in sorted(source.rglob('*.py')))).hexdigest()
        self.day = session_day()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS checkpoint (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL, sha256 TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS evidence (sequence INTEGER PRIMARY KEY, observed_at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS report_outbox (execution_event_id TEXT PRIMARY KEY, payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0)')
        self.db.execute('CREATE TABLE IF NOT EXISTS canonical_fills (symbol TEXT, side TEXT, scoped_order_id TEXT, quantity REAL NOT NULL, notional REAL NOT NULL, PRIMARY KEY(symbol,side,scoped_order_id))')
        self.db.commit()

    def queue_report(self, key, payload, stamp, event_type='trade_exit'):
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO report_outbox(execution_event_id,payload) VALUES (?,?)',
                            (key,json.dumps(pack((event_type,payload,stamp)),separators=(',',':'))))

    def retry_reports(self):
        # Journal writes carry a canonical key and are idempotent across the
        # crash window between its commit and this outbox acknowledgement.
        rows=self.db.execute('SELECT execution_event_id,payload FROM report_outbox WHERE delivered=0 ORDER BY rowid').fetchall()
        for key, raw in rows:
            decoded=unpack(json.loads(raw))
            event_type,payload,stamp=decoded if len(decoded)==3 else ('trade_exit',*decoded)
            try:
                getattr(self.runner._journal,'_execution_delivery',self.runner._journal.record)(event_type,payload,ts=stamp)
            except Exception:
                return False
            with self.db:
                self.db.execute('UPDATE report_outbox SET delivered=1 WHERE execution_event_id=?',(key,))
        return True

    def save(self, payload=None, *, kind='commit', evidence=None):
        try:
            if self.invalid_restore:
                raise ValueError('Refusing to overwrite a rejected execution checkpoint')
            state = payload or capture(self.runner)
            value = json.dumps(dict(schema=SCHEMA, day=self.day, parameters=self.parameters, build=self.build, account=self.account,
                                    state=state), separators=(',', ':'), allow_nan=False)
            digest = hashlib.sha256(value.encode()).hexdigest()
            with self.db:
                initial = self.db.execute('SELECT 1 FROM checkpoint WHERE id=1').fetchone() is None
                domain = unpack(state)
                watermarks = domain['portfolio'].get('_fill_projections', {}).get('cash', {})
                existing_keys={(sym,side,oid) for sym,side,oid in self.db.execute('SELECT symbol,side,scoped_order_id FROM canonical_fills')}
                if existing_keys-set(watermarks):
                    raise ValueError('Refusing to drop durable canonical fill watermarks')
                for (symbol, side, order_id), (qty, cost) in watermarks.items():
                    previous = self.db.execute('SELECT quantity,notional FROM canonical_fills WHERE symbol=? AND side=? AND scoped_order_id=?', (symbol,side,order_id)).fetchone() or (0.,0.)
                    delta_qty,delta_cost=qty-previous[0],cost-previous[1]
                    if delta_qty < 0 or (delta_qty == 0 and abs(delta_cost)>1e-7 and side != 'buy'):
                        raise ValueError('Canonical checkpoint watermark regression/correction requires reconciliation')
                    if delta_qty > 0 or abs(delta_cost)>1e-7:
                        fact=dict(symbol=symbol,side=side,scoped_order_id=order_id,
                                  cumulative_quantity=qty,cumulative_notional=cost,
                                  quantity=delta_qty,notional=delta_cost)
                        event_kind='baseline_watermark' if initial else 'accepted_fill_delta' if delta_qty>0 else 'entry_valuation_correction'
                        self.db.execute('INSERT INTO evidence(observed_at,kind,payload) VALUES (?,?,?)',
                            (WallDatetime.now(timezone.utc).isoformat(),event_kind,json.dumps(fact,separators=(',',':'))))
                        self.db.execute('INSERT OR REPLACE INTO canonical_fills VALUES (?,?,?,?,?)',(symbol,side,order_id,qty,cost))
                self.db.execute('INSERT OR REPLACE INTO checkpoint VALUES (1,?,?)', (value, digest))
                if evidence is not None:
                    self.db.execute('INSERT INTO evidence(observed_at,kind,payload) VALUES (?,?,?)',
                        (WallDatetime.now(timezone.utc).isoformat(), kind,
                         json.dumps(pack(evidence), separators=(',', ':'), allow_nan=False)))
        except Exception:
            self.failed = True
            self.runner._execution_recovery_required = True
            raise

    def restore(self):
        try:
            return self._restore()
        except Exception:
            self.invalid_restore=True
            raise

    def _validated_checkpoint(self):
        """Validate without changing owner objects or recovery/protection state."""
        row = self.db.execute('SELECT payload,sha256 FROM checkpoint WHERE id=1').fetchone()
        if row is None:
            return None
        value, digest = row
        if hashlib.sha256(value.encode()).hexdigest() != digest:
            raise ValueError('Execution checkpoint integrity failure')
        payload = json.loads(value)
        if payload['schema'] != SCHEMA or payload['day'] != self.day or payload['parameters'] != self.parameters or payload.get('build') != self.build or payload.get('account') != self.account:
            raise ValueError('Execution checkpoint schema/day/source/account/risk parameters changed')
        # Decode and validate the entire graph before changing native state.
        state = unpack(payload['state'])
        if set(state) != {'portfolio', 'pipeline', 'manager', 'runner', 'broker'}:
            raise ValueError('Invalid execution checkpoint domain')
        for name, allowed in (('pipeline', PIPELINE), ('manager', MANAGER), ('runner', RUNNER), ('broker', BROKER)):
            if set(state[name])-set(allowed):
                raise ValueError('Unexpected execution checkpoint field')
        expected=state['portfolio'].get('_fill_projections',{}).get('cash',{})
        actual={(symbol,side,oid):(qty,cost) for symbol,side,oid,qty,cost in self.db.execute('SELECT symbol,side,scoped_order_id,quantity,notional FROM canonical_fills')}
        if expected != actual:
            raise ValueError('Canonical ledger does not match its atomic projection checkpoint')
        return payload, state

    def _restore(self):
        validated = self._validated_checkpoint()
        if validated is None:
            return False
        payload, state = validated
        p = self.runner._pipeline
        vars(p.portfolio).clear(); vars(p.portfolio).update(state['portfolio'])
        for obj, key in ((p, 'pipeline'), (p.exit_manager, 'manager'), (self.runner, 'runner'), (native_broker(self.runner), 'broker')):
            for name, value in state[key].items():
                setattr(obj, name, value)
        # Requests and existing exposure must be freshly reconciled. Loaded
        # quantities are recovery context, never permission to place an order.
        for row in getattr(self.runner, '_partial_entries', {}).values():
            row.active_submit = False
        self.runner._execution_recovery_required = True
        self.runner._reliability_restored = True
        for row in getattr(native_broker(self.runner), '_protection_states', {}).values():
            row['requires_revalidation'] = True
        return True

    def before_broker(self, evidence):
        with self.lock:
            if getattr(self.runner, '_execution_store', self) is not self:
                self.runner._execution_recovery_required = True
                raise ValueError('Broker request captured an inactive execution store')
            # Persist pre-operation economic state plus newly latched intents
            # and request identities. Never checkpoint a half-applied projection.
            state = unpack(self.base) if self.base is not None else unpack(capture(self.runner))
            current_broker = unpack(pack(fields(native_broker(self.runner), BROKER)))
            for name in ('_unsettled_orders','_entry_order_originals'):
                retained = self.recovery_broker.setdefault(name,{})
                retained.update(current_broker.get(name,{}))
                current_broker[name] = dict(retained)
            state['broker'] = current_broker
            # The initial pending-entry descriptor exists before its first
            # request; restore it with zero accounted quantity, not a later
            # callback's half-applied cost/quantity.
            pending = state['runner'].setdefault('_partial_entries',{})
            for symbol,row in getattr(self.runner,'_partial_entries',{}).items():
                if symbol not in pending and row.quantity == 0:
                    pending[symbol] = unpack(pack(row))
            if self.base is not None:
                base = unpack(self.base)
                base['runner']['_partial_entries'] = pending
                self.base = pack(base)
            current = self.runner._pipeline.exit_manager
            for symbol, pos in state['manager'].get('_positions', {}).items():
                live = current.tracked.get(symbol)
                if live is not None:
                    pos.exit_intent = unpack(pack(live.exit_intent))
                    pos.timeout_due_at = live.timeout_due_at
            self.save(pack(state), kind='broker_request', evidence=evidence)

    @contextmanager
    def operation(self):
        with self.lock:
            if getattr(self.runner, '_execution_store', self) is not self:
                self.runner._execution_recovery_required = True
                raise ValueError('Execution operation captured an inactive store')
            outer = self.depth == 0
            if outer:
                self.base = capture(self.runner)
                self.recovery_broker = {}
            self.depth += 1
            try:
                yield
            except Exception:
                self.runner._execution_recovery_required = True
                self.failed = True
                engine=getattr(self.runner,'_session_engine',None)
                if engine is not None:
                    engine.recorder.fail('execution operation interrupted before atomic checkpoint')
                raise
            else:
                if outer:
                    self.save()
            finally:
                self.depth -= 1
                if outer:
                    self.base = None
                    self.recovery_broker = {}


def durable_operation(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        store = getattr(self, '_execution_store', None)
        if store is None:
            return method(self, *args, **kwargs)
        with store.operation():
            return method(self, *args, **kwargs)
    return call


def initialize(runner, root, *, preserve_current=False):
    if getattr(runner._broker, '_paper', False) is not True:
        return
    day = session_day()
    previous=vars(runner).get('_execution_store')
    # A caller's old operation would otherwise commit after the active pointer
    # changed. Refuse before opening/creating any target store.
    if previous is not None and previous.depth:
        runner._execution_recovery_required = True
        raise ValueError('Execution store switch requires a completed durable operation')
    if preserve_current:
        if previous is None or previous.failed or previous.invalid_restore:
            runner._execution_recovery_required = True
            raise ValueError('Rollover requires a healthy previous execution store')
        with previous.lock:
            return _rollover_preserving_current(runner, root, previous, day)
    try:
        from .alpaca_broker import AlpacaBroker
        if isinstance(native_broker(runner), AlpacaBroker) and not getattr(native_broker(runner), '_account_scope', ''):
            raise ValueError('Verified PAPER account identity is required before execution recovery')
        store = ExecutionStore(Path(root)/'execution-state'/f'{day}.sqlite3', runner)
        restored = store.restore()
        runner._execution_store = store
        runner._journal._execution_store = store
        runner._execution_session_day = store.day
        if not restored:
            if (getattr(runner, '_session_has_fills', False)
                    and not getattr(runner, '_fill_baseline_verified', False)):
                runner._legacy_fill_checkpoint = True
            store.save()
        runner._broker._durability_checkpoint = store.before_broker
        return restored
    except Exception:
        runner._execution_recovery_required = True
        if previous is not None:
            runner._execution_store=previous
            runner._journal._execution_store=previous
            runner._execution_session_day=previous.day
        else:
            vars(runner).pop('_execution_store',None)
            vars(runner._journal).pop('_execution_store',None)
        raise


def _rollover_preserving_current(runner, root, previous, day):
    """Normal owner rollover only; no restore, date rewrite or financial merge."""
    broker = native_broker(runner)
    runner._execution_recovery_required = True
    if (previous.depth or previous.failed or previous.invalid_restore
            or getattr(runner, '_execution_store', None) is not previous
            or getattr(broker, '_position_owner', None) != threading.get_ident()):
        raise ValueError('Rollover requires the idle decision owner')
    if (not getattr(broker, '_account_scope', '')
            or broker._account_scope != previous.account):
        raise ValueError('Verified unchanged PAPER account identity is required for rollover')
    if parameter_digest(runner) != previous.parameters:
        raise ValueError('Rollover risk parameters changed')
    current = getattr(broker, 'position_snapshot_current', None)
    positions = getattr(broker, '_pos_cache', None)
    p = runner._pipeline
    if (getattr(runner, '_last_session_reset_day', None) != day
            or not getattr(runner, '_fill_sync_complete', False)
            or runner._position_snapshot_uncertain()
            or getattr(runner, '_pipeline_protection_pending', {})
            or getattr(runner, '_legacy_fill_checkpoint', False)
            or any(getattr(owner, '_fill_reconciliation_required', False)
                   for owner in (runner, p, p.portfolio, p.exit_manager))
            or p.exit_manager.tracked or any(not pos.is_flat for pos in p.portfolio.positions.values())
            or not isinstance(positions, dict) or positions
            or not callable(current) or not current()):
        raise ValueError('Current owner state is not freshly reconciled and flat')
    snapshot = (getattr(broker, '_position_generation', 0),
                getattr(broker, '_position_context', None), broker._pos_cache_ts)
    if previous.day == day or session_day() != day:
        raise ValueError('Rollover session boundary changed')
    if previous.db.execute('SELECT count(*) FROM report_outbox WHERE delivered=0').fetchone()[0]:
        raise ValueError('Previous execution reports remain undelivered')
    if previous._validated_checkpoint() is None:
        raise ValueError('Previous execution checkpoint is missing')
    before = capture(runner)
    expected = unpack(before)['portfolio'].get('_fill_projections', {}).get('cash', {})
    ledger = {(s, side, oid): (q, n) for s, side, oid, q, n in previous.db.execute(
        'SELECT symbol,side,scoped_order_id,quantity,notional FROM canonical_fills')}
    if expected != ledger:
        raise ValueError('Previous canonical ledger differs from current owner state')
    old_callback = getattr(broker, '_durability_checkpoint', None)
    old_journal_store = getattr(runner._journal, '_execution_store', None)
    old_day = getattr(runner, '_execution_session_day', previous.day)
    store = None
    try:
        store = ExecutionStore(Path(root)/'execution-state'/f'{day}.sqlite3', runner)
        if store.build != previous.build:
            raise ValueError('Rollover source build changed')
        if store.day != day or session_day() != day:
            raise ValueError('Rollover session boundary changed')
        validated = store._validated_checkpoint()
        if store.db.execute('SELECT count(*) FROM report_outbox WHERE delivered=0').fetchone()[0]:
            raise ValueError('Current-session execution reports remain undelivered')
        if validated is not None and validated[0]['state'] != before:
            raise ValueError('Current-session checkpoint conflicts with current owner state')
        if validated is None:
            if any(store.db.execute('SELECT count(*) FROM '+table).fetchone()[0]
                   for table in ('canonical_fills', 'evidence', 'report_outbox')):
                raise ValueError('Current-session evidence has no atomic checkpoint')
            store.save(before)
        # No financial owner objects are restored or replaced. Refuse if the
        # accepted snapshot aged out, context changed, or owner state changed
        # while the bounded SQLite checkpoint was being prepared.
        if (capture(runner) != before or not current()
                or getattr(broker, '_pos_cache', None) != {}
                or snapshot != (getattr(broker, '_position_generation', 0),
                                getattr(broker, '_position_context', None), broker._pos_cache_ts)
                or previous.db.execute('SELECT count(*) FROM report_outbox WHERE delivered=0').fetchone()[0]
                or session_day() != day):
            raise ValueError('Owner state or position freshness changed during rollover')
        runner._execution_store = store
        runner._journal._execution_store = store
        runner._execution_session_day = store.day
        broker._durability_checkpoint = store.before_broker
        return validated is not None
    except Exception:
        runner._execution_store = previous
        runner._journal._execution_store = old_journal_store
        runner._execution_session_day = old_day
        broker._durability_checkpoint = old_callback
        if store is not None:
            store.db.close()
        raise
