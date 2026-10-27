"""Explicit, full-precision PAPER guard checkpoints at a main-loop boundary.

Only whitelisted risk state is serialized. Accounts, clients and credentials
are never traversed. Export is flat-only and paused-only; it places no orders.
"""
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
import threading
import time
from zoneinfo import ZoneInfo

from daytrading.analytics.missed_a_plus import MissedAPlusRecord
from daytrading.models import Bar, Timeframe
from daytrading.pipeline.engine import _RejectCooldown

RISK_FIELDS = (
    '_symbol_entry_counts', '_daily_loss_counts', '_daily_losers', '_daily_pnl',
    '_circuit_breaker_tripped', '_exit_cooldowns', '_original_sizes', '_scan_rejections',
)
PARAMETERS = (
    '_micro_pullback_only', '_micro_minimum_room_r',
    '_max_positions', '_max_position_shares', '_max_order_shares', '_max_dollar_risk_per_trade',
    '_max_entry_risk_pct', '_max_entries_per_symbol', '_max_daily_loss', '_cooldown_seconds',
    '_enable_daily_loser_blacklist', '_daily_loser_blacklist_min_loss', '_daily_loser_blacklist_max_losses',
    '_missed_a_plus_chase_window_sec', '_missed_a_plus_chase_pct_sub5', '_missed_a_plus_chase_pct_5plus',
    '_missed_a_plus_fresh_base_reset', '_missed_a_plus_fresh_base_pct',
    '_entry_chase_pct_low', '_entry_chase_pct_high', '_entry_chase_price_tier',
)


def encode(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError('Checkpoint contains a naive timestamp')
        return {'__datetime__': value.isoformat()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set):
        return {'__set__': sorted((encode(x) for x in value), key=lambda x: json.dumps(x,sort_keys=True))}
    if isinstance(value, tuple):
        return {'__tuple__': [encode(x) for x in value]}
    if isinstance(value, (list, deque)):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {str(k):encode(v) for k,v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError('Unsupported guard checkpoint value')


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if isinstance(value, dict):
        if set(value)=={'__datetime__'}:
            stamp=datetime.fromisoformat(value['__datetime__'])
            if stamp.tzinfo is None: raise ValueError('Naive checkpoint timestamp')
            return stamp
        if set(value)=={'__set__'}: return set(decode(x) for x in value['__set__'])
        if set(value)=={'__tuple__'}: return tuple(decode(x) for x in value['__tuple__'])
        return {k:decode(v) for k,v in value.items()}
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def assert_quiescent(runner):
    if getattr(runner._broker, '_paper', False) is not True:
        raise ValueError('Guard checkpoints require a PAPER broker')
    if runner._hub.trading_paused is not True:
        raise ValueError('Pause new entries before checkpointing')
    if any(not p.is_flat for p in runner._pipeline.portfolio.positions.values()) or runner._pipeline.exit_manager.tracked:
        raise ValueError('Checkpoint requires a flat strategy portfolio')
    if runner._exec_timer.pending_symbols or runner._timed_signal_queue:
        raise ValueError('Wait for pending timed entries to clear before checkpointing')
    book=getattr(runner._pipeline,'pending_entries',None)
    if book is not None and (book.rows or book.inflight):
        raise ValueError('Wait for retained entry setups to clear before checkpointing')
    if getattr(runner, '_partial_entries', {}):
        raise ValueError('First-partial entry protection remains pending')
    if (getattr(runner._broker,'_safety_blocked',False) is True or
            getattr(runner._broker,'_unsettled_orders',{}) or getattr(runner,'_pipeline_protection_pending',{})):
        raise ValueError('Broker reconciliation or protection remains pending')


def capture_state(runner, now=None):
    """Called only from the runner main loop, never from its web thread."""
    assert_quiescent(runner)
    if getattr(runner, '_legacy_fill_checkpoint', False):
        raise ValueError('Legacy fill identities must be verified before exporting schema 2')
    now=now or datetime.now(timezone.utc)
    p=runner._pipeline
    tracker=p._missed_a_plus
    # This is the complete store, including scanner near-misses, in original
    # insertion order. Do not call report(), which sorts, rounds and truncates.
    records={key:asdict(record) for key,record in tracker._records.items()}
    order=list(tracker._order)
    if len(order)!=len(set(order)) or set(order)!=set(records):
        raise ValueError('Guard tracker order is inconsistent')
    halt=p.trade_guard.halt_tracker
    with halt._lock:
        halt_state=deepcopy(dict(halted=halt._halted,resumed=halt._resumed,last_price=halt._last_price,
                                 resume_cooldown=halt._resume_cooldown,
                                 low_activity=halt._low_activity,
                                 data_uncertain=halt._data_uncertain,
                                 activity_ticks={k:list(v) for k,v in halt._activity_ticks.items()}))
    panic=p.trade_guard.market_panic
    with panic._lock:
        panic_state=dict(bars=[asdict(bar) for bar in panic._spy_bars],active=panic._panic_active,
                         start=panic._panic_start,green_count=panic._green_count)
    slippage=p.trade_guard.slippage
    with slippage._lock:
        slippage_state={key:list(values) for key,values in slippage._slippage_history.items()}
    mono=time.monotonic()
    anchors={key:dict(price=value[0],last_seen_wall=now.timestamp()-(mono-value[1]))
             for key,value in getattr(runner,'_timed_entry_anchor',{}).items()}
    guards=encode(dict(
        risk={name:deepcopy(getattr(p,name)) for name in RISK_FIELDS},
        fill_projections={key: [(name, list(values.items())) for name, values in deepcopy(getattr(owner, '_fill_projections', {})).items()]
                          for key, owner in (('runner', runner), ('pipeline', p),
                                             ('portfolio', p.portfolio), ('manager', p.exit_manager))},
        exit_accounting=dict(legs=deepcopy(p._realized_trade_legs),
                             fills=list(deepcopy(p._realized_exit_fills).items())),
        parameters={name:deepcopy(getattr(p,name)) for name in PARAMETERS},
        reject_cooldowns={key:asdict(value) for key,value in p._reject_cooldowns.items()},
        tracker=dict(records=records,order=order,max_records=tracker._max_records,
                     dedupe_seconds=tracker._dedupe_seconds,label_after_seconds=tracker._label_after_seconds),
        halt=halt_state,panic=panic_state,slippage_history=slippage_state,
        runner=dict(timed_anchors=anchors,timed_anchor_ttl=getattr(runner,'_timed_entry_anchor_ttl_sec',300),
                    recorded_exit_fill_keys=getattr(runner,'_recorded_exit_fill_keys',set()),
                    synced_order_ids=getattr(runner,'_last_synced_order_ids',set()),
                    last_session_reset_day=getattr(runner,'_last_session_reset_day',None),
                    hod_eligibility=dict(active=getattr(runner,'_hod_active',{}),
                        tradeable_at=getattr(runner,'_hod_tradeable_alert_at',{}),
                        watermarks=getattr(getattr(runner,'_hod_bar_scanner',None),'_qualification_watermarks',{}),
                        notification_keys=getattr(getattr(runner,'_hod_bar_scanner',None),'_bar_cooldown_keys',set()),
                        gapper_notified=getattr(getattr(runner,'_hod_bar_scanner',None),'_gapper_fired',set()))),
    ))
    return dict(schema_version=2,source_pid=os.getpid(),captured_at=now.isoformat(),
                day_new_york=now.astimezone(ZoneInfo('America/New_York')).date().isoformat(),
                complete=True,record_count=len(records),guards=guards,guards_sha256=digest(guards))


def restore_state(runner, checkpoint, now=None):
    """Validate the entire checkpoint before changing any guard state."""
    now=now or datetime.now(timezone.utc)
    if getattr(runner._broker,'_paper',False) is not True:
        raise ValueError('Guard restoration requires a PAPER broker')
    if checkpoint.get('schema_version') not in (1, 2) or checkpoint.get('complete') is not True:
        raise ValueError('Incomplete or unsupported guard checkpoint')
    if checkpoint['day_new_york']!=now.astimezone(ZoneInfo('America/New_York')).date().isoformat():
        raise ValueError('Guard checkpoint day mismatch')
    age=(now-datetime.fromisoformat(checkpoint['captured_at'])).total_seconds()
    if not 0<=age<=600: raise ValueError('Guard checkpoint expired')
    raw=checkpoint['guards']
    if digest(raw)!=checkpoint['guards_sha256']: raise ValueError('Guard checkpoint digest mismatch')
    state=decode(deepcopy(raw))
    risk,parameters=state['risk'],state['parameters']
    if set(risk)!=set(RISK_FIELDS) or set(parameters)!=set(PARAMETERS):
        raise ValueError('Guard checkpoint risk schema mismatch')
    p=runner._pipeline
    # Optional for schema-1 flat checkpoints exported by the previous build.
    accounting=state.get('exit_accounting',dict(legs={},fills=[]))
    if not isinstance(accounting,dict) or set(accounting)!={'legs','fills'}:
        raise ValueError('Invalid exit accounting checkpoint')
    legs=accounting['legs'];fill_rows=accounting['fills']
    if not isinstance(legs,dict) or not isinstance(fill_rows,list):
        raise ValueError('Invalid exit accounting checkpoint')
    def finite_number(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
    for symbol,values in legs.items():
        if (not isinstance(symbol,str) or not isinstance(values,tuple) or len(values)!=3
                or not all(finite_number(x) for x in values) or values[1]<=0 or values[2]<=0):
            raise ValueError('Invalid exit accounting trade totals')
    fills={}
    for row in fill_rows:
        if not isinstance(row,tuple) or len(row)!=2:
            raise ValueError('Invalid exit accounting fill')
        key,value=row
        broker_key = isinstance(key, tuple) and len(key) == 6 and key[0] == 'broker'
        if broker_key:
            if (not all(isinstance(x,str) and x for x in key[1:4]) or key[2] not in ('buy','sell')
                    or not all(finite_number(x) and x>0 for x in key[4:6])):
                raise ValueError('Invalid broker exit accounting fill')
        else:
            if (not isinstance(key,tuple) or len(key)!=5 or not isinstance(key[0],str)
                    or key[1] not in ('buy','sell') or not all(finite_number(x) and x>0 for x in key[2:4])
                    or not isinstance(key[4],str)):
                raise ValueError('Invalid exit accounting fill')
            stamp=datetime.fromisoformat(key[4])
            if stamp.tzinfo is None: raise ValueError('Invalid exit accounting timestamp')
        if not finite_number(value) or key in fills:
            raise ValueError('Invalid exit accounting fill amount')
        fills[key]=value
    if any(getattr(p,name)!=parameters[name] for name in PARAMETERS):
        raise ValueError('Guard parameters changed across restart')
    tracker=state['tracker']
    records={key:MissedAPlusRecord(**row) for key,row in tracker['records'].items()}
    if (len(records)!=checkpoint['record_count'] or set(tracker['order'])!=set(records) or
            len(tracker['order'])!=len(records) or any(key!=row.id for key,row in records.items())):
        raise ValueError('Guard checkpoint tracker incomplete')
    rejects={key:_RejectCooldown(**row) for key,row in state['reject_cooldowns'].items()}
    spy_bars=[Bar(**dict(row,timeframe=Timeframe(row['timeframe']))) for row in state['panic']['bars']]
    for key in ('halted','resumed','last_price','resume_cooldown'):state['halt'][key]
    low_activity=set(state['halt'].get('low_activity', ()))
    if not low_activity.issubset(state['halt']['halted']):
        raise ValueError('Activity checkpoint contains unblocked symbols')
    activity_ticks={k:deque(v,maxlen=12) for k,v in state['halt'].get('activity_ticks',{}).items()}
    data_uncertain=state['halt'].get('data_uncertain', {})
    if not isinstance(data_uncertain, dict):
        raise ValueError('Invalid data-health checkpoint')
    for symbol,row in data_uncertain.items():
        if (not isinstance(symbol,str) or not isinstance(row,dict)
                or set(row)!={'since','reason','discarded','last_observed_at','last_exchange_ts'}
                or row['reason'] not in {'invalid_price','invalid_exchange_clock','invalid_observation_clock','future_trade','stale_trade'}
                or not isinstance(row['discarded'],int) or row['discarded']<1
                or any(not isinstance(row[k],datetime) or row[k].tzinfo is None for k in ('since','last_observed_at'))
                or not isinstance(row['last_exchange_ts'],datetime)):
            raise ValueError('Invalid data-health checkpoint')
    if not set(activity_ticks).issubset(low_activity | data_uncertain.keys()):
        raise ValueError('Activity checkpoint source mismatch')
    for values in activity_ticks.values():
        if any(not isinstance(t,datetime) or t.tzinfo is None for t in values) or list(values)!=sorted(set(values)):
            raise ValueError('Invalid recovery activity timestamps')
    for key in ('active','start','green_count'):state['panic'][key]
    slippage_history=defaultdict(lambda:deque(maxlen=20),
        {key:deque(values,maxlen=20) for key,values in state['slippage_history'].items()})
    runner_state=state['runner']
    hod=runner_state.get('hod_eligibility',dict(active={},tradeable_at={},watermarks={},notification_keys=set(),gapper_notified=set()))
    if (not isinstance(hod,dict) or set(hod)!={'active','tradeable_at','watermarks','notification_keys','gapper_notified'}
            or any(not isinstance(hod[k],dict) for k in ('active','tradeable_at','watermarks'))
            or any(not isinstance(hod[k],set) or any(not isinstance(v,str) for v in hod[k]) for k in ('notification_keys','gapper_notified'))
            or any(not isinstance(k,str) or not isinstance(v,datetime) or v.tzinfo is None or v>now
                   for rows in (hod['active'],hod['tradeable_at'],hod['watermarks']) for k,v in rows.items())):
        raise ValueError('Invalid HOD eligibility checkpoint')
    reset_day=runner_state.get('last_session_reset_day')
    if reset_day is not None:
        try:
            parsed=datetime.fromisoformat(reset_day)
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid session reset day') from exc
        if parsed.date().isoformat()!=reset_day or reset_day>checkpoint['day_new_york']:
            raise ValueError('Invalid session reset day')
    for key in ('timed_anchor_ttl','recorded_exit_fill_keys','synced_order_ids'):runner_state[key]
    anchors={key:(row['price'],time.monotonic()-(now.timestamp()-row['last_seen_wall']))
             for key,row in runner_state['timed_anchors'].items()}
    # Round-trip every record before mutation: no UI rounding or omitted fields.
    records_encoded=encode({key:asdict(row) for key,row in records.items()})
    if digest(records_encoded)!=digest(raw['tracker']['records']):
        raise ValueError('Guard record precision changed')
    projections = {}
    raw_projections = state.get('fill_projections', {})
    if checkpoint['schema_version'] == 2 and set(raw_projections) != {'runner','pipeline','portfolio','manager'}:
        raise ValueError('Missing fill projection checkpoint')
    for owner, names in raw_projections.items():
        projections[owner] = {}
        for name, rows in names:
            values = dict(rows)
            if len(values) != len(rows) or any(not isinstance(k, tuple) or len(k)!=3 or
                    k[1] not in ('buy','sell') or not all(isinstance(x,str) and x for x in k) or
                    not isinstance(v,tuple) or len(v)!=2 or not all(finite_number(x) and x>0 for x in v)
                    for k,v in rows):
                raise ValueError('Invalid fill projection checkpoint')
            projections[owner][name] = values
    runner._hub.trading_paused=True
    runner._hod_active=dict(hod['active'])
    runner._hod_tradeable_alert_at=dict(hod['tradeable_at'])
    hod_scanner=getattr(runner,'_hod_bar_scanner',None)
    if hod_scanner is not None:
        hod_scanner._qualification_watermarks=dict(hod['watermarks'])
        hod_scanner._bar_cooldown_keys=set(hod['notification_keys'])
        hod_scanner._gapper_fired=set(hod['gapper_notified'])
    for name,value in risk.items(): setattr(p,name,value)
    p._realized_trade_legs=legs
    p._realized_exit_fills=fills
    for name, owner in (('runner',runner),('pipeline',p),('portfolio',p.portfolio),('manager',p.exit_manager)):
        owner._fill_projections = projections.get(name, {})
    runner._legacy_fill_checkpoint = checkpoint['schema_version'] == 1
    runner._fill_baseline_verified = checkpoint['schema_version'] == 2
    p._reject_cooldowns=rejects
    p._missed_a_plus._records=records
    p._missed_a_plus._order=tracker['order']
    p._missed_a_plus._max_records=tracker['max_records']
    p._missed_a_plus._dedupe_seconds=tracker['dedupe_seconds']
    p._missed_a_plus._label_after_seconds=tracker['label_after_seconds']
    halt=p.trade_guard.halt_tracker
    with halt._lock:
        halt._halted=state['halt']['halted'];halt._resumed=state['halt']['resumed']
        halt._last_price=state['halt']['last_price'];halt._resume_cooldown=state['halt']['resume_cooldown']
        # Legacy checkpoints do not prove the origin of a halt. They remain
        # blocked unless the local migration supplies matching log evidence.
        halt._low_activity=low_activity
        halt._data_uncertain=data_uncertain
        # Pre-crash samples cannot certify post-restart transport recovery.
        halt._activity_ticks={k:v for k,v in activity_ticks.items() if k not in data_uncertain}
    panic=p.trade_guard.market_panic
    with panic._lock:
        panic._spy_bars=deque(spy_bars,maxlen=100);panic._panic_active=state['panic']['active']
        panic._panic_start=state['panic']['start'];panic._green_count=state['panic']['green_count']
    with p.trade_guard.slippage._lock:
        p.trade_guard.slippage._slippage_history=slippage_history
    runner._timed_entry_anchor=anchors
    runner._timed_entry_anchor_ttl_sec=runner_state['timed_anchor_ttl']
    runner._recorded_exit_fill_keys=runner_state['recorded_exit_fill_keys']
    runner._last_synced_order_ids=runner_state['synced_order_ids']
    runner._last_session_reset_day=reset_day
    return dict(complete=True,records_restored=len(records),records_sha256=digest(records_encoded),
                risk_sha256=digest(encode(risk)),source_guards_sha256=checkpoint['guards_sha256'],
                last_session_reset_day=reset_day,daily_pnl=p._daily_pnl,entry_counts=dict(p._symbol_entry_counts),
                halted_symbols=len(halt._halted),timed_anchors=len(anchors),trading_paused=True)


def migrate_legacy_fill_checkpoint(runner, *, reviewed_duplicates=None):
    """Read-only broker proof for a paused, flat schema-1 guard migration.

    Preserve every risk value. Seed cumulative watermarks only when legacy
    recorded sales individually match the fresh broker ledger. An explicit
    reviewed duplicate map may retain a known extra legacy identity without
    importing its quantity again. No historical P&L or risk value is corrected.
    This function never submits, cancels or replaces an order or edits a journal.
    """
    assert_quiescent(runner)
    if not getattr(runner, '_legacy_fill_checkpoint', False):
        raise ValueError('No legacy fill checkpoint awaits migration')
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    from daytrading.execution.fill_identity import observation
    from daytrading.models import Side
    client = runner._broker.client

    def prove_flat():
        if client.get_all_positions() or client.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.OPEN, limit=500, nested=True)):
            raise ValueError('Legacy migration requires a fresh flat broker with no open orders')

    prove_flat()
    orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL,
        after=runner._session_start_utc(), limit=500, nested=True))
    if len(orders) >= 500 or len({str(o.id) for o in orders}) != len(orders):
        raise ValueError('Legacy broker ledger is truncated or contradictory')
    facts = []
    broker_sales = {}
    balances = {}
    for order in orders:
        from daytrading.execution.stop_recovery import TERMINAL, value
        if value(order.status) not in TERMINAL:
            raise ValueError('Legacy broker ledger contains a nonterminal order')
        qty = float(order.filled_qty or 0)
        if qty == 0:
            continue
        side = str(getattr(order.side, 'value', order.side))
        if side not in ('buy', 'sell'):
            raise ValueError('Unknown broker side during legacy migration')
        fill = observation(order, order.symbol, Side(side), order.filled_at or order.submitted_at,
                           account_id=getattr(runner._broker, '_account_scope', ''))
        if not all(math.isfinite(x) and x > 0 for x in (fill.quantity, fill.cumulative_notional)):
            raise ValueError('Invalid legacy broker fill')
        facts.append(fill)
        balances[fill.symbol] = balances.get(fill.symbol, 0.) + (qty if side == 'buy' else -qty)
        if side == 'sell':
            broker_sales[fill.order_id] = fill
    if any(abs(qty) > 1e-7 for qty in balances.values()):
        raise ValueError('Legacy broker ledger does not close exactly')
    duplicates = dict(reviewed_duplicates or {})
    matched = set()
    duplicate_matches = {}
    start = runner._session_start_utc()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    for key in runner._pipeline._realized_exit_fills:
        if len(key) != 5 or key[1] != 'sell':
            raise ValueError('Legacy migration requires unambiguous sale identities')
        stamp = datetime.fromisoformat(key[4])
        if stamp.tzinfo is None:
            raise ValueError('Naive legacy fill timestamp')
        if stamp < start:
            if key in duplicates:
                raise ValueError('Reviewed duplicate is outside the current session')
            continue  # Retained historical accounting is not today's broker ledger.
        candidates = [oid for oid, fill in broker_sales.items()
                      if fill.symbol == key[0] and fill.quantity == key[2]
                      and round(fill.price, 4) == key[3]
                      and abs((fill.ts-stamp).total_seconds()) <= .000001]
        if len(candidates) != 1:
            raise ValueError('Legacy accounting does not match broker sales; preserve and review the discrepancy')
        oid = candidates[0]
        if key in duplicates:
            if duplicates[key] != oid:
                raise ValueError('Reviewed duplicate does not match its broker identity')
            duplicate_matches[key] = oid
        elif oid in matched:
            raise ValueError('Legacy accounting does not match broker sales: unreviewed duplicate')
        else:
            matched.add(oid)
    if matched != set(broker_sales) or set(duplicate_matches) != set(duplicates) or any(
            oid not in matched for oid in duplicate_matches.values()):
        raise ValueError('Legacy accounting does not match broker sales; incomplete duplicate proof')
    prove_flat()
    owners = ((runner, 'exit_report'), (runner._pipeline, 'realized'),
              (runner._pipeline.portfolio, 'inventory'), (runner._pipeline.portfolio, 'cash'),
              (runner._pipeline.exit_manager, 'exit_state'))
    for owner, name in owners:
        values = owner.__dict__.setdefault('_fill_projections', {}).setdefault(name, {})
        for fill in facts:
            values[(fill.symbol, fill.side.value, (fill.account_id+'/'+fill.order_id) if fill.account_id else fill.order_id)] = (fill.cumulative_quantity, fill.cumulative_notional)
    runner._legacy_synced_order_ids = {str(o.id) for o in orders}
    runner._legacy_sync_watermarks = {f.order_id: (f.cumulative_quantity, f.cumulative_notional) for f in facts}
    runner._legacy_fill_checkpoint = False
    runner._fill_baseline_verified = True
    return dict(migrated=True, orders=len(orders), fill_watermarks=len(facts),
                retained_duplicate_identities=len(duplicate_matches), risk_values_unchanged=True, trading_paused=True)


class _CheckpointEvent:
    """One-use adapter; capture on the runner thread at its next wait boundary."""
    def __init__(self, runner, freeze):
        self.runner,self.original,self.freeze=runner,runner._new_data,freeze
        self.ready,self.release=threading.Event(),threading.Event()
        self.state,self.error=None,None
        self.cancelled=False

    def __getattr__(self,name): return getattr(self.original,name)

    def wait(self,timeout=None):
        try:
            if not self.cancelled:
                self.state=capture_state(self.runner)
        except Exception as exc:
            self.error=exc
        finally:
            if self.runner._new_data is self: self.runner._new_data=self.original
            self.ready.set()
        if self.freeze and self.state is not None and not self.cancelled:
            self.release.wait(timeout=30)
            self.release.set()
        return self.original.wait(timeout=timeout)


def request_checkpoint(runner, *, freeze=False):
    assert_quiescent(runner)
    current=getattr(runner,'_paper_handover_lease',None)
    if current is not None and not current.release.is_set():
        raise ValueError('A paper checkpoint lease is already active')
    proxy=_CheckpointEvent(runner,freeze)
    runner._paper_handover_lease=proxy
    runner._new_data=proxy
    proxy.original.set()
    if not proxy.ready.wait(timeout=10):
        proxy.cancelled=True;proxy.release.set()
        if runner._new_data is proxy: runner._new_data=proxy.original
        raise ValueError('Main-thread checkpoint timed out')
    if proxy.error is not None:
        proxy.release.set()
        raise ValueError(str(proxy.error))
    if not freeze: proxy.release.set()
    return proxy.state


def release_checkpoint(runner):
    proxy=getattr(runner,'_paper_handover_lease',None)
    if proxy is not None: proxy.release.set()
    return dict(released=True)
