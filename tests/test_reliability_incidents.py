"""October 21 incident fixtures; all broker/market I/O is synthetic."""
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace as NS
from unittest.mock import Mock
import threading
import time

import pytest

from daytrading.execution.broker import apply_fill, apply_cash_observation
from daytrading.execution.fill_identity import observation, FillConflict
from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.execution.stop_recovery import reconcile
from daytrading.execution.durability import ExecutionStore
from daytrading.execution.position_reconciler import PositionReconciler
from daytrading.exits.manager import TrackedPosition
from daytrading.models import Side, Position
from test_trade_loss_accounting import pipeline, reporting_runner

T = datetime(2026,10,21,13,30,tzinfo=timezone.utc)


def seen(oid, qty, avg=2.42, *, symbol='YMAT', offset=0):
    return observation(NS(id=oid,filled_qty=qty,filled_avg_price=avg), symbol, Side.SELL,
                       T+timedelta(microseconds=offset))


def setup(symbol='YMAT', qty=497, entry=2.48, stop=2.42):
    p=pipeline();p.portfolio.positions[symbol]=Position(symbol,qty,entry)
    p.exit_manager.track(TrackedPosition(symbol=symbol,side=Side.BUY,quantity=qty,entry_price=entry,entry_ts=T-timedelta(seconds=300),
        remaining_qty=qty,stop_loss=stop,trend_strength=.9))
    r=reporting_runner(p);r._last_synced_order_ids=set();r._now_et=lambda:T
    return r,p


def book(r,p,fill):
    apply_fill(p.portfolio,fill)
    return r._record_trade_exit(fill,2.48,'stop_loss')


def test_ymat_26_share_duplicate_timestamp_never_changes_economics_twice():
    r,p=setup();p.exit_manager.check_exits({'YMAT':2.40},T)
    f=seen('00000000-0000-4000-8000-000000000001',26)
    book(r,p,f);cash=p.portfolio.cash
    book(r,p,replace(f,ts=f.ts+timedelta(microseconds=1)))
    assert p.portfolio.cash==cash
    assert p.portfolio.positions['YMAT'].quantity==471
    assert p.exit_manager.tracked['YMAT'].remaining_qty==471
    assert p._daily_pnl==pytest.approx(-1.56)
    assert r._journal.record.call_count==1


def test_ymat_remainder_never_becomes_an_ordinary_orphan():
    r,p=setup();first=p.exit_manager.check_exits({'YMAT':2.4},T)[0]
    entered=p.exit_manager.tracked['YMAT'].entry_ts
    for i,(qty,remaining) in enumerate(((26,471),(440,31),(5,26),(26,0))):
        book(r,p,seen(str(i),qty,offset=i))
        if remaining:
            decision=p.exit_manager.check_exits({'YMAT':2.60},T+timedelta(seconds=90+i))[0]
            assert decision.quantity==remaining and decision.reason==first.reason
            assert p.exit_manager.tracked['YMAT'].entry_ts==entered
            assert p.exit_manager.tracked['YMAT'].exit_intent.liquidate
            r._clear_broker_stop.assert_not_called()
    assert 'YMAT' not in p.exit_manager.tracked
    r._clear_broker_stop.assert_called_once_with('YMAT')
    assert p._daily_loss_counts['YMAT']==1


def test_cumulative_quantity_and_notional_use_delta_price():
    r,p=setup();book(r,p,seen('same-order',26,2.42))
    book(r,p,seen('same-order',36,2.43,offset=1))
    book(r,p,seen('same-order',26,2.42,offset=2))  # older poll
    assert p.portfolio.positions['YMAT'].quantity==461
    assert p._daily_pnl==pytest.approx(36*2.43-36*2.48)
    assert r._hub.on_exit_fill.call_args.args[0].price==pytest.approx(2.456)
    assert r._hub.on_exit_fill.call_count==2


def test_equal_price_quantity_timestamp_distinct_order_ids_are_distinct():
    r,p=setup()
    book(r,p,seen('one',26));book(r,p,seen('two',26))
    assert p.portfolio.positions['YMAT'].quantity==445
    assert p._daily_pnl==pytest.approx(-3.12)
    assert r._journal.record.call_count==2


def test_contradictory_same_quantity_cost_blocks_new_entries():
    r,p=setup();book(r,p,seen('one',26))
    with pytest.raises(FillConflict):book(r,p,seen('one',26,2.43))
    assert p.portfolio._fill_reconciliation_required
    assert r._new_entries_blocked('YMAT','TEST')
    assert p.portfolio.positions['YMAT'].quantity==471


def test_inve_ordinary_five_minute_intent_survives_340_15_6():
    r,p=setup('INVE',361,2.729944,2.50)
    pos=p.exit_manager.tracked['INVE'];entered=pos.entry_ts
    decision=p.exit_manager.check_exits({'INVE':2.64},T)[0]
    assert '300s' in decision.reason
    intent=pos.exit_intent
    for i,(qty,px,remaining) in enumerate(((340,2.64,21),(15,2.7,6),(6,2.668333,0))):
        f=seen(str(i),qty,px,symbol='INVE',offset=i)
        apply_fill(p.portfolio,f);r._record_trade_exit(f,2.729944,decision.reason)
        if remaining:
            again=p.exit_manager.check_exits({'INVE':2.80},T+timedelta(seconds=i+1))[0]
            assert again.quantity==remaining and again.reason==decision.reason
            assert p.exit_manager.tracked['INVE'] is pos and pos.entry_ts==entered
            assert pos.exit_intent is intent
    assert p._daily_pnl==pytest.approx(-31.399786,abs=.0001)


def test_sphl_48_of_49_does_not_pay_unfilled_target_share():
    r,p=setup('SPHL',99,2.,1.9);m=p.exit_manager
    sig=m.check_exits({'SPHL':2.10},T)[0]
    assert sig.quantity==49
    f=seen('target',48,2.1,symbol='SPHL');apply_fill(p.portfolio,f)
    r._record_trade_exit(f,2.,sig.reason)
    pos=m.tracked['SPHL'];assert pos.remaining_qty==51 and not pos.sold_half and pos.stop_loss==1.9
    assert m.check_exits({'SPHL':2.11},T+timedelta(seconds=1))[0].quantity==1
    f=seen('target',49,2.1,symbol='SPHL');apply_fill(p.portfolio,f)
    r._record_trade_exit(f,2.,sig.reason)
    assert pos.remaining_qty==50 and pos.sold_half and pos.stop_loss>=2.


def broker_stop(qty=6, price=2.42, oid='stop'):
    return NS(id=oid,symbol='TEST',side='sell',type='stop',status='new',qty=qty,
              filled_qty=0,stop_price=price,filled_avg_price=None)


def safe_broker(quantity, orders, current=2.60):
    b=AlpacaBroker.__new__(AlpacaBroker);b._client=Mock();b._safety_state()
    b._client.get_all_positions.return_value=[NS(symbol='TEST',qty=quantity,current_price=current)] if quantity else []
    b._client.get_orders.return_value=orders
    for order in orders:b._client.get_order_by_id.return_value=order
    b.place_protective_stop=Mock(return_value='placed');b.replace_protective_stop=Mock(return_value='replaced')
    return b


def test_existing_six_share_stop_is_reused_without_reserving_twice():
    current=broker_stop();b=safe_broker(6,[current]);b._failed_stop_updates['TEST']=('stop',6,2.42)
    assert reconcile(b,'TEST','stop',6,2.42)
    assert b.get_protective_stop_id('TEST')=='stop'
    b.place_protective_stop.assert_not_called();b.replace_protective_stop.assert_not_called()


def test_amod_stale_47_share_stop_is_resized_to_fresh_125():
    current=broker_stop(47);b=safe_broker(125,[current])
    assert reconcile(b,'TEST','stop',47,2.42)
    b.replace_protective_stop.assert_called_once_with('TEST','stop',125,2.42)
    b.place_protective_stop.assert_not_called()


def test_ymat_crossed_stop_hands_off_liquidation_without_lowering_stop():
    b=safe_broker(497,[],current=2.40);b._protection_breached_handler=Mock()
    assert not reconcile(b,'TEST',None,497,2.42)
    b._protection_breached_handler.assert_called_once_with('TEST',497,2.42,2.40)
    b.place_protective_stop.assert_not_called()


def test_fresh_order_read_failure_preserves_recovery_block():
    b=safe_broker(6,[]);b._client.get_orders.side_effect=TimeoutError()
    assert b.reconcile_protective_stop('TEST',6,2.42) is None
    assert b._safety_blocked and 'TEST' in b._failed_stop_updates
    b.place_protective_stop.assert_not_called()


def durable_setup(tmp_path):
    r,p=setup();r._broker=NS(_paper=True,_unsettled_orders={},_suspended_stops={})
    store=ExecutionStore(tmp_path/'state.sqlite',r);r._execution_store=store
    return r,p,store


@pytest.mark.parametrize('crash_stage', ['before_effects','after_inventory','after_report'])
def test_restart_recovers_atomic_projections_and_pending_intent(tmp_path,crash_stage):
    r,p,store=durable_setup(tmp_path);m=p.exit_manager
    m.check_exits({'YMAT':2.4},T);reason=m.tracked['YMAT'].exit_intent.reason
    store.save()
    f=seen('restart-order',26)
    with pytest.raises(RuntimeError):
        with store.operation():
            r._broker._unsettled_orders['YMAT']=(None,'stable-request')
            store.before_broker({'kind':'submit','request_id':'stable-request'})
            if crash_stage!='before_effects':apply_fill(p.portfolio,f)
            if crash_stage=='after_report':r._record_trade_exit(f,2.48,'stop')
            raise RuntimeError('simulated crash')
    # A new process restores the durable pre-request domain, then projects the
    # broker observation once; an interrupted in-memory projection is not replayed twice.
    r2,p2,recovered=durable_setup(tmp_path);assert recovered.restore()
    assert r2._broker._unsettled_orders['YMAT']==(None,'stable-request')
    assert p2.exit_manager.tracked['YMAT'].exit_intent.reason==reason
    assert p2.portfolio.positions['YMAT'].quantity==497
    book(r2,p2,f);recovered.save()
    r3,p3,again=durable_setup(tmp_path);assert again.restore()
    assert p3.portfolio.positions['YMAT'].quantity==471
    cash=p3.portfolio.cash;book(r3,p3,replace(f,ts=T+timedelta(microseconds=8)))
    assert p3.portfolio.cash==cash and p3._daily_pnl==pytest.approx(-1.56)


def test_corrupt_durable_checkpoint_has_no_partial_restore(tmp_path):
    r,p,store=durable_setup(tmp_path);store.save()
    store.db.execute("UPDATE checkpoint SET sha256='wrong'");store.db.commit()
    p._daily_pnl=123
    with pytest.raises(ValueError,match='integrity'):store.restore()
    assert p._daily_pnl==123


def test_known_lifecycle_is_restored_with_original_clock_and_policy():
    r,p=setup();m=p.exit_manager;m.check_exits({'YMAT':2.40},T);pos=m.tracked['YMAT']
    book(r,p,seen('partial',471))
    m._retired_positions={'YMAT':pos};m.untrack('YMAT')
    assert 'YMAT' not in m.tracked
    rec=PositionReconciler();out=rec.reconcile({'YMAT':{'qty':26,'avg_entry':2.48}},p.portfolio,m,now=T+timedelta(seconds=500))
    assert out.confirmed==['YMAT'] and not out.adopted and m.tracked['YMAT'] is pos
    assert m.check_exits({'YMAT':2.60},T+timedelta(seconds=501))[0].quantity==26


def test_cash_projection_agrees_when_sync_follows_local_fill():
    r,p=setup();f=seen('cash',26);book(r,p,f);cash=p.portfolio.cash
    apply_cash_observation(p.portfolio,replace(f,ts=T+timedelta(microseconds=1)))
    assert p.portfolio.cash==cash
    apply_cash_observation(p.portfolio,seen('cash',36,2.43))
    assert p.portfolio.cash-cash==pytest.approx(24.56)


def test_recording_failure_blocks_entries_but_consumes_market_input(tmp_path,monkeypatch):
    from test_recorded_session_validation import make_runner
    from daytrading.runner import QuoteEvent
    from daytrading.models import Quote
    r,e=make_runner(tmp_path,monkeypatch);e.recorder.fail('disk fault')
    assert r._new_entries_blocked('NCPL','TEST')
    q=Quote('NCPL',T,2.,2.01,100,100);r._enqueue_market_event(QuoteEvent('NCPL',q));r._drain_events()
    assert r._quote_buffer['NCPL'][-1]==q
    e.recorder.close();assert e.recorder.manifest['state']=='incomplete'


def test_new_capture_boundary_preserves_old_gap_and_queued_input(tmp_path,monkeypatch):
    from test_recorded_session_validation import make_runner
    from daytrading.runner import QuoteEvent
    from daytrading.models import Quote
    from daytrading.replay.store import read_session
    r,e=make_runner(tmp_path,monkeypatch);old=e.recorder;old.fail('overflow')
    q=Quote('NCPL',T,2.,2.01,100,100);r._enqueue_market_event(QuoteEvent('NCPL',q))
    new_id=e.restart_boundary();assert new_id!=old.session_id
    assert old.manifest['state']=='incomplete'
    assert e.recorder.manifest['metadata']['capture_boundary']['gap'] is True
    r._drain_events();e.recorder.close()
    manifest,events=read_session(e.recorder.root,new_id)
    assert manifest['state']=='sealed',manifest['failures']
    assert any(x['kind']=='capture_boundary' for x in events)
    from daytrading.replay.audit import audit_session
    assert audit_session(manifest,events)['status']=='matched'
    assert r._quote_buffer['NCPL'][-1]==q


def test_sdk_worker_keeps_tape_on_decision_owner_and_replays_callbacks(tmp_path,monkeypatch):
    from test_recorded_session_validation import make_runner
    from daytrading.replay.engine import ExternalProxy
    from daytrading.runner import QuoteEvent
    from daytrading.models import Quote
    r,e=make_runner(tmp_path,monkeypatch);owner=threading.get_ident();threads=[]
    def slow():
        threads.append(threading.get_ident());time.sleep(.07);return []
    # Invoke a recorded SDK-shaped boundary and assert FIFO data can be pumped
    # while the private response is still in flight. No strategy mutation in worker.
    proxy=ExternalProxy(NS(slow=slow),'broker_client',e)
    q=Quote('NCPL',T,2.,2.01,100,100);r._enqueue_market_event(QuoteEvent('NCPL',q))
    e.invoke('synthetic_sdk_wait',lambda:proxy.slow(),(),{})
    assert threads and threads[0]!=owner
    assert r._quote_buffer['NCPL'][-1]==q and not r._broker_io_busy
    e.recorder.close();assert e.recorder.manifest['state']=='sealed',e.recorder.manifest['failures']
    e.sdk_worker.shutdown()


def test_outbox_retries_after_journal_commit_without_duplicate_event(tmp_path):
    from daytrading.journal.store import TradingJournal
    # Real isolated SQLite journal, fault injected after its durable primary commit.
    journal=TradingJournal(base_dir=str(tmp_path/'journal'),prune_on_start=False)
    r,p,store=durable_setup(tmp_path);r._journal=journal
    payload=dict(execution_event_id='canonical-exit',symbol='YMAT',side='sell',quantity=26,
                 entry_price=2.48,exit_price=2.42,pnl=-1.56,trade_type='exit')
    store.queue_report('canonical-exit',payload,T)
    real=journal.record
    def commit_then_lose_ack(*args,**kwargs):
        real(*args,**kwargs);raise RuntimeError('response lost after journal commit')
    journal.record=commit_then_lose_ack
    assert store.retry_reports() is False
    journal.record=real;assert store.retry_reports() is True
    rows=journal._conn.execute("SELECT count(*) FROM events WHERE type='trade_exit'").fetchone()[0]
    assert rows==1
    assert journal._conn.execute('SELECT count(*) FROM trades').fetchone()[0]==1


def test_vivo_coordinated_time_exit_stays_latched_across_seven_orders():
    # Preserve the successful same-intent liquidation behavior independently
    # of the entry filter. The incident quantities add to the observed 99 shares.
    r,p=setup('VIVO',99,3.04,2.90);m=p.exit_manager
    pos=m.tracked['VIVO'];pos.entry_ts=T-timedelta(seconds=61)
    # Ordinary intent mechanics are shared with coordinated exits; the native
    # coordinated state itself is exercised by the existing micro regression suite.
    from daytrading.exits.manager import ExitIntent
    pos.exit_intent=ExitIntent('coordinated_micro:time_exit',T,99,True)
    for i,qty in enumerate((45,44,4,2,2,1,1)):
        f=seen('vivo-'+str(i),qty,3.02,symbol='VIVO',offset=i)
        apply_fill(p.portfolio,f);r._record_trade_exit(f,3.04,'coordinated_micro:time_exit')
        if i<6:
            assert m.check_exits({'VIVO':3.05},T+timedelta(seconds=i+1))[0].reason=='coordinated_micro:time_exit'
    assert not p.portfolio.positions and p._daily_pnl==pytest.approx(-1.98)


@pytest.mark.parametrize("durable", [False, True])
def test_real_native_exit_sdk_wait_keeps_exact_audit_matching(tmp_path,monkeypatch,durable):
    from collections import deque
    from daytrading.runner import AlpacaRunner,QuoteEvent
    from daytrading.models import Quote
    from daytrading.replay.engine import shared_pipeline,attach_capture
    from daytrading.replay.store import read_session
    from daytrading.replay.audit import audit_session
    from test_recorded_session_validation import HistFake,StreamFake
    at=datetime.now(timezone.utc)
    completed=NS(id='exit-sdk',symbol='TEST',side='sell',type='limit',status='filled',
                 qty=100,filled_qty=100,filled_avg_price=9.1,filled_at=at,submitted_at=at)
    def delayed_submit(**kwargs):time.sleep(.08);return completed
    client=NS(get_orders=lambda **kw:[],submit_order=delayed_submit,
              get_order_by_id=lambda oid:completed,get_all_positions=lambda:[])
    b=AlpacaBroker.__new__(AlpacaBroker);b._client=client;b._paper=True;b._slippage_guard=None
    b._max_wait=.1;b._poll_interval=.005;b._cancel_grace_seconds=.01;b._safety_state()
    b._pos_cache={'TEST':{'qty':100.,'avg_entry':10.,'current_price':9.1}}
    b._pos_cache_ts=time.monotonic();b._position_sync_stale=False
    from concurrent.futures import Future
    b._position_future=Future()
    b._position_future.set_result([NS(symbol='TEST',qty=100,avg_entry_price=10,current_price=9.1,
        market_value=910,unrealized_pl=-90,unrealized_plpc=-.09)])
    b._position_pool=Mock()

    p=shared_pipeline(broker=b,initial_cash=25000,min_avg_volume=10000,high_liquidity_volume=50000,
                      entry_patterns=('vwap_pullback','hod_reclaim','micro_pullback_reclaim'))
    r=AlpacaRunner(b,HistFake(),StreamFake(),p,['TEST','NCPL','SPY'],dashboard_port=0)
    p.portfolio.positions['TEST']=Position('TEST',100,10)
    p.exit_manager.track(TrackedPosition(symbol='TEST',side=Side.BUY,quantity=100,remaining_qty=100,
                       entry_price=10,stop_loss=9.5,entry_ts=at-timedelta(seconds=10)))
    r._quote_buffer['TEST']=deque([Quote('TEST',at,9.1,9.11,1000,1000)])
    monkeypatch.setenv('DAYTRADING_DATA_DIR',str(tmp_path))
    monkeypatch.setenv('DAYTRADING_JOURNAL_DIR',str(tmp_path/'journal'))
    monkeypatch.setenv('DAYTRADING_SESSION_RECORDING','true')
    e=attach_capture(r)
    if durable:
        store=ExecutionStore(tmp_path/'native-state.sqlite',r);r._execution_store=store;store.save()
        b._durability_checkpoint=store.before_broker
    r._enqueue_market_event(QuoteEvent('NCPL',Quote('NCPL',at,2.,2.01,100,100)))
    r._check_exits_only();e.recorder.close()
    manifest,events=read_session(e.recorder.root,e.recorder.session_id)
    assert manifest['state']=='sealed',manifest['failures']
    assert r._quote_buffer['NCPL']
    result=audit_session(manifest,events)
    assert result['status']=='matched',result
    if hasattr(e,'sdk_worker'):e.sdk_worker.shutdown()


def test_unfilled_micro_target_cannot_mask_original_60_second_exit():
    from daytrading.exits.coordinated_micro import MicroRiskPlan,MicroExitState
    r,p=setup('MICRO',100,2.,1.9);pos=p.exit_manager.tracked['MICRO'];pos.entry_ts=T
    plan=MicroRiskPlan('MICRO',2.,1.9,100,10)
    pos.micro_exit_state=MicroExitState.open(plan,2.,100,T)
    first=p.exit_manager.check_exits({'MICRO':2.11},T+timedelta(seconds=20))[0]
    assert first.quantity==50
    f=seen('partial',49,2.1,symbol='MICRO');apply_fill(p.portfolio,f)
    r._record_trade_exit(f,2.,first.reason)
    assert not pos.sold_half and pos.remaining_qty==51
    assert p.exit_manager.check_exits({'MICRO':2.01},T+timedelta(seconds=30))==[]
    timed=p.exit_manager.check_exits({'MICRO':2.01},T+timedelta(seconds=60))[0]
    assert timed.quantity==51 and 'time_exit' in timed.reason
    assert pos.exit_intent.liquidate


def test_pending_new_stop_is_acknowledged_but_not_working_protection():
    b=AlpacaBroker.__new__(AlpacaBroker);b._client=Mock();b._safety_state()
    pending=broker_stop(6);pending.status='pending_new';b._client.submit_order.return_value=pending
    assert b.place_protective_stop('TEST',6,2.42) is None
    assert b._safety_blocked and b.protection_status()['TEST']['state']=='acknowledged'
    assert 'TEST' in b._failed_stop_updates


def test_uncertain_cancel_retains_identity_until_broker_acknowledges():
    b=AlpacaBroker.__new__(AlpacaBroker);b._client=Mock();b._safety_state()
    order=broker_stop();b._client.get_order_by_id.return_value=order
    b._client.cancel_order_by_id.side_effect=TimeoutError('lost acknowledgement')
    assert not b.cancel_order_by_id('stop')
    assert 'stop' in b._unconfirmed_cancels and b._safety_blocked
    order.status='canceled';b._client.cancel_order_by_id.side_effect=None
    assert b.cancel_order_by_id('stop') and not b._unconfirmed_cancels


def test_replaced_stop_response_loss_adopts_matching_successor():
    old=broker_stop();old.status='replaced';old.replaced_by='successor'
    successor=broker_stop(125,oid='successor');b=safe_broker(125,[successor])
    b._client.get_order_by_id.side_effect=lambda oid:old if oid=='stop' else successor
    b._failed_stop_updates['TEST']=('stop',47,2.42)
    assert reconcile(b,'TEST','stop',47,2.42)
    assert b.get_protective_stop_id('TEST')=='successor'
    b.place_protective_stop.assert_not_called();b.replace_protective_stop.assert_not_called()


def test_report_retry_after_accounting_error_never_loses_fill():
    r,p=setup();f=seen('retry',26);apply_fill(p.portfolio,f)
    real=p.record_realized_exit;p.record_realized_exit=Mock(side_effect=RuntimeError('projection interrupted'))
    with pytest.raises(RuntimeError):r._record_trade_exit(f,2.48,'stop')
    p.record_realized_exit=real;r._record_trade_exit(f,2.48,'stop')
    assert p._daily_pnl==pytest.approx(-1.56)
    assert p.exit_manager.tracked['YMAT'].remaining_qty==471
    assert r._journal.record.call_count==1


def test_same_order_id_is_scoped_to_its_broker_account():
    from dataclasses import replace
    r,p=setup();f=seen('same-id',26)
    book(r,p,replace(f,account_id='account-one'))
    book(r,p,replace(f,account_id='account-two'))
    assert p.portfolio.positions['YMAT'].quantity==445
    assert p._daily_pnl==pytest.approx(-3.12)


def test_execution_checkpoint_refuses_another_account_before_mutation(tmp_path):
    r,p,store=durable_setup(tmp_path);store.save()
    r2,p2=setup();r2._broker=NS(_paper=True,_account_scope='another-account')
    other=ExecutionStore(tmp_path/'state.sqlite',r2)
    p2._daily_pnl=123
    with pytest.raises(ValueError,match='account'):other.restore()
    assert p2._daily_pnl==123


def test_legacy_checkpoint_migration_preserves_risk_and_requires_matching_sales():
    from test_trade_loss_accounting import checkpoint_runner
    from daytrading.runtime_state import migrate_legacy_fill_checkpoint
    r=checkpoint_runner();r._hub.trading_paused=True;r._legacy_fill_checkpoint=True
    r._session_start_utc=lambda:T-timedelta(hours=4)
    r._broker.client=NS(get_all_positions=lambda:[],get_orders=lambda **kwargs:[])
    before=r._pipeline._daily_pnl
    result=migrate_legacy_fill_checkpoint(r)
    assert result['migrated'] and not r._legacy_fill_checkpoint
    assert r._pipeline._daily_pnl==before and r._hub.trading_paused
    r._legacy_fill_checkpoint=True
    r._pipeline._realized_exit_fills={('YMAT','sell',26,2.42,T.isoformat()):-1.56}
    with pytest.raises(ValueError,match='does not match'):migrate_legacy_fill_checkpoint(r)
    assert r._legacy_fill_checkpoint and r._pipeline._daily_pnl==before


def test_broker_stop_fill_cannot_be_counted_as_pending_profit_target():
    from daytrading.exits.coordinated_micro import MicroRiskPlan,MicroExitState
    r,p=setup('MICRO',100,2.,1.9);pos=p.exit_manager.tracked['MICRO'];pos.entry_ts=T
    pos.micro_exit_state=MicroExitState.open(MicroRiskPlan('MICRO',2.,1.9,100,10),2.,100,T)
    sig=p.exit_manager.check_exits({'MICRO':2.11},T+timedelta(seconds=5))[0]
    f=seen('profit',48,2.1,symbol='MICRO');apply_fill(p.portfolio,f);r._record_trade_exit(f,2.,sig.reason)
    stopped=replace(seen('actual-stop',26,1.9,symbol='MICRO'),execution_reason='broker_stop')
    apply_fill(p.portfolio,stopped);r._record_trade_exit(stopped,2.,'broker_fill_sync')
    assert pos.remaining_qty==26 and pos.micro_exit_state.target_filled==48 and not pos.sold_half
    assert pos.exit_intent.liquidate
    assert p.exit_manager.check_exits({'MICRO':2.2},T+timedelta(seconds=7))[0].quantity==26


def test_canonical_delta_and_checkpoint_commit_together_without_duplicate_sequence(tmp_path):
    r,p,store=durable_setup(tmp_path);store.save();f=seen('canonical',26)
    book(r,p,f);store.save();book(r,p,replace(f,ts=T+timedelta(microseconds=2)));store.save()
    assert store.db.execute("SELECT count(*) FROM evidence WHERE kind='accepted_fill_delta'").fetchone()[0]==1
    assert store.db.execute('SELECT quantity FROM canonical_fills').fetchone()[0]==26


def test_rejected_checkpoint_cannot_be_overwritten_by_shutdown_save(tmp_path):
    r,p,store=durable_setup(tmp_path);store.save()
    old=store.db.execute('SELECT payload,sha256 FROM checkpoint').fetchone()
    r2,p2=setup();r2._broker=NS(_paper=True,_account_scope='other')
    rejected=ExecutionStore(tmp_path/'state.sqlite',r2)
    with pytest.raises(ValueError):rejected.restore()
    with pytest.raises(ValueError,match='Refusing'):rejected.save()
    assert rejected.db.execute('SELECT payload,sha256 FROM checkpoint').fetchone()==old


def test_session_checkpoint_day_uses_the_existing_four_am_new_york_boundary(monkeypatch):
    import daytrading.execution.durability as module
    from zoneinfo import ZoneInfo
    current=[datetime(2026,10,22,0,30,tzinfo=ZoneInfo('America/New_York'))]
    class Clock:
        @staticmethod
        def now(tz):return current[0].astimezone(tz)
    monkeypatch.setattr(module,'WallDatetime',Clock)
    assert module.session_day()=='2026-10-21'
    current[0]=current[0].replace(hour=4)
    assert module.session_day()=='2026-10-22'


def test_entry_journal_delivery_uses_the_same_canonical_outbox(tmp_path):
    from daytrading.execution.fill_identity import report_execution
    from daytrading.journal.store import TradingJournal
    r,p,store=durable_setup(tmp_path)
    journal=TradingJournal(base_dir=str(tmp_path/'entries'),prune_on_start=False)
    r._journal=journal;journal._execution_store=store
    f=replace(seen('entry',26,2.48),side=Side.BUY)
    report_execution(journal,'trade_fill',dict(symbol='YMAT',side='buy',quantity=26,price=2.48,trade_type='entry'),f)
    report_execution(journal,'trade_fill',dict(symbol='YMAT',side='buy',quantity=26,price=2.48,trade_type='entry'),replace(f,ts=T+timedelta(microseconds=1)))
    assert journal._conn.execute("SELECT count(*) FROM events WHERE type='trade_fill'").fetchone()[0]==1
    assert store.db.execute('SELECT count(*) FROM report_outbox').fetchone()[0]==1


def test_fully_accounted_sale_does_not_sell_again_on_lagging_position_read():
    r,p=setup();m=p.exit_manager;m.check_exits({'YMAT':2.4},T)
    book(r,p,seen('fully-sold',497))
    rec=PositionReconciler()
    result=rec.reconcile({'YMAT':{'qty':26,'avg_entry':2.48}},p.portfolio,m,now=T+timedelta(seconds=1))
    assert result.still_pending==['YMAT'] and not m.check_exits({'YMAT':2.4},T+timedelta(seconds=2))
    assert r._new_entries_blocked('YMAT','TEST')
    for i in range(3):rec.reconcile({},p.portfolio,m,now=T+timedelta(seconds=i+3))
    assert not m._residual_confirmation_pending and not m._retired_positions


def test_matching_stop_without_ownership_proof_is_not_silently_adopted():
    b=safe_broker(6,[broker_stop(6,oid='unattributed')])
    assert not reconcile(b,'TEST',None,6,2.42)
    assert b.get_protective_stop_id('TEST') is None
    b.place_protective_stop.assert_not_called();b.replace_protective_stop.assert_not_called()


def test_first_install_with_intraday_fills_requires_existing_guard_migration(tmp_path):
    from daytrading.execution.durability import initialize
    r,p=setup();r._broker=NS(_paper=True);r._session_has_fills=True
    initialize(r,tmp_path)
    assert r._legacy_fill_checkpoint
    assert r._new_entries_blocked('YMAT','TEST')


def test_missing_native_account_identity_refuses_initialization(tmp_path):
    from daytrading.execution.durability import initialize
    r,p=setup();b=AlpacaBroker.__new__(AlpacaBroker);b._paper=True;r._broker=b
    with pytest.raises(ValueError,match='account identity'):initialize(r,tmp_path)
    assert not hasattr(r,'_execution_store') and r._execution_recovery_required


def legacy_migration_runner():
    from test_trade_loss_accounting import checkpoint_runner
    r=checkpoint_runner();r._hub.trading_paused=True;r._legacy_fill_checkpoint=True
    r._session_start_utc=lambda:T-timedelta(hours=4)
    rows=[NS(id='buy',symbol='YMAT',side=Side.BUY,status='filled',filled_qty=26,
             filled_avg_price=2.48,filled_at=T-timedelta(seconds=60)),
          NS(id='sell',symbol='YMAT',side=Side.SELL,status='filled',filled_qty=26,
             filled_avg_price=2.424281,filled_at=T)]
    r._broker.client=NS(get_all_positions=lambda:[],get_orders=lambda **kw:
        rows if str(getattr(kw['filter'].status,'value',kw['filter'].status))=='all' else [])
    key=('YMAT','sell',26.,2.4243,T.isoformat())
    r._pipeline._realized_exit_fills={key:-1.56}
    return r,rows,key


def test_legacy_migration_uses_rounded_individual_facts_and_retains_prior_days():
    from daytrading.runtime_state import migrate_legacy_fill_checkpoint,encode,digest,RISK_FIELDS
    r,rows,key=legacy_migration_runner()
    r._pipeline._realized_exit_fills[('WORX','sell',100.,3.,(T-timedelta(days=1)).isoformat())]=1.
    before=digest(encode({k:getattr(r._pipeline,k) for k in RISK_FIELDS}))
    accounting=dict(r._pipeline._realized_exit_fills)
    assert migrate_legacy_fill_checkpoint(r)['fill_watermarks']==2
    assert r._pipeline._realized_exit_fills==accounting
    assert digest(encode({k:getattr(r._pipeline,k) for k in RISK_FIELDS}))==before
    assert r._fill_baseline_verified


def test_reviewed_duplicate_migration_keeps_risk_and_rejects_unreviewed_rows():
    from daytrading.runtime_state import migrate_legacy_fill_checkpoint
    r,rows,key=legacy_migration_runner()
    extra=(*key[:4],(T-timedelta(microseconds=1)).isoformat())
    r._pipeline._realized_exit_fills[extra]=-1.56
    before=r._pipeline._daily_pnl
    with pytest.raises(ValueError,match='unreviewed duplicate'):
        migrate_legacy_fill_checkpoint(r)
    assert r._legacy_fill_checkpoint and not getattr(r,'_fill_projections',{})
    with pytest.raises(ValueError,match='broker identity'):
        migrate_legacy_fill_checkpoint(r,reviewed_duplicates={extra:'wrong-order'})
    result=migrate_legacy_fill_checkpoint(r,reviewed_duplicates={extra:'sell'})
    assert result['retained_duplicate_identities']==1
    assert r._pipeline._daily_pnl==before and len(r._pipeline._realized_exit_fills)==2
    assert len(r._pipeline._fill_projections['realized'])==2


def test_verified_intraday_migration_survives_first_store_initialization(tmp_path):
    from daytrading.runtime_state import migrate_legacy_fill_checkpoint
    from daytrading.execution.durability import initialize
    r,rows,key=legacy_migration_runner()
    migrate_legacy_fill_checkpoint(r)
    r._journal=NS()
    r._session_has_fills=True
    initialize(r,tmp_path)
    assert not r._legacy_fill_checkpoint
    assert r._execution_store.db.execute('SELECT count(*) FROM canonical_fills').fetchone()[0]==2


def test_unverified_legacy_checkpoint_cannot_be_exported_as_schema_two():
    from test_trade_loss_accounting import checkpoint_runner
    from daytrading.runtime_state import capture_state
    r=checkpoint_runner();r._hub.trading_paused=True;r._legacy_fill_checkpoint=True
    with pytest.raises(ValueError,match='verified'):
        capture_state(r)
