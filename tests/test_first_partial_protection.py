from daytrading.models import BarSource
"""Causal first-partial, cancellation and late-fill protection regressions."""
from collections import deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.execution.entry_executor import EntryExecutor
from daytrading.execution import partial_entry as partial
from daytrading.exits.manager import ExitManager
from daytrading.models import Bar, Fill, Order, OrderStatus, PortfolioState, Quote, Side, SignalAction, TradeSignal
T=datetime(2026,10,16,12,47,4,tzinfo=timezone.utc)
class Clock:
    now=0.
    def monotonic(self):return self.now
    def sleep(self,x):self.now+=x

def row(q,avg=2.,status='partially_filled',at=0):
    return NS(id='buy',symbol='TEST',qty=100,filled_qty=q,filled_avg_price=avg,filled_at=T+timedelta(seconds=at),status=status)

def setup(observations,tape=None):
    c=Clock();events=[];source=deque(observations);last=[observations[-1]]
    def get(oid):
        result=source.popleft() if source else last[0]
        if isinstance(result,Exception):raise result
        c.now=max(c.now,(result.filled_at-T).total_seconds())
        last[0]=result;events.append(('observe',result.status,result.filled_qty,c.now));return result
    client=NS(submit_order=lambda **kw:NS(id='buy'),get_order_by_id=get,cancel_order_by_id=lambda oid:events.append(('cancel',oid,c.now)))
    b=AlpacaBroker.__new__(AlpacaBroker);b._client=client;b._slippage_guard=None;b._now=lambda:T;b._max_wait=.45;b._poll_interval=.15;b._cancel_grace_seconds=.8;b._safety_state()
    b._cancel_open_orders_for=lambda symbol:None;b._restore_protection=lambda *a,**kw:True
    sig=TradeSignal('TEST',SignalAction.ENTER_LONG,100,entry_price=2.,stop_loss=1.90,trend_strength=.82)
    order=Order('TEST',Side.BUY,100,limit_price=2.);bar=Bar('TEST',T,2,2,2,2,10000, provenance=BarSource.CANONICAL_INTERVAL)
    manager=ExitManager(max_unrealized_loss=50.);portfolio=PortfolioState(cash=10000)
    r=NS(_broker=b,_pipeline=NS(portfolio=portfolio,exit_manager=manager,_symbol_entry_counts={},_original_sizes={},set_cooldown=lambda s:None),
         _quote_buffer={'TEST':deque()},_tick_buffer={},_micro_market_now=lambda:T+timedelta(seconds=c.now),
         _journal=NS(record=lambda kind,data,**kw:events.append((kind,data))),
         _hub=NS(on_fill=lambda *a,**kw:events.append(('entry_report',a[0].quantity)),add_log=lambda *a:None),_entry_executor=EntryExecutor.__new__(EntryExecutor))
    tape=tape or (lambda t:2.)
    def drain():
        price=tape(c.now)
        if price is not None:r._quote_buffer['TEST'].append(Quote('TEST',T+timedelta(seconds=c.now),price,price+.01,1000,1000))
        events.append(('pump',c.now))
    r._drain_partial_entry_events=drain
    ctx=NS(pipeline=r._pipeline,on_position_opened=lambda signal,fill,**kw:events.append(('arm_stop',fill.quantity,manager._positions['TEST'].stop_loss)),hub=r._hub,journal=r._journal,market_phase=lambda:'PRE-MARKET',seed_recent_order_ids=lambda:None)
    r._entry_execution_context=lambda:ctx;r._push_positions_from_alpaca=lambda:None;r._seed_recent_order_ids=lambda:None
    def sell(order,bar,label):
        assert 'TEST' not in b._unsettled_orders
        events.append(('sell',order.quantity,bar.close));return Fill('TEST',Side.SELL,order.quantity,bar.close,T+timedelta(seconds=c.now)),OrderStatus.FILLED
    r._submit_protective_exit_order=sell;r._record_trade_exit=lambda *a,**kw:events.append(('exit_report',a[0].quantity))
    b.set_entry_lifecycle_callbacks(lambda *a:partial.observe(r,*a),lambda *a:partial.waiting(r,*a),lambda *a:partial.recovered(r,*a))
    def submit():
        with patch('daytrading.execution.alpaca_broker.time',c):
            result=partial.submit(r,sig,order,bar);partial.finish(r,sig)
        return result
    return NS(r=r,b=b,clock=c,events=events,signal=sig,order=order,bar=bar,submit=submit,source=source)

def test_first_partial_tracked_before_next_poll_and_counted_once():
    h=setup([row(20),row(100,status='filled',at=.15)]);fill,status=h.submit()
    first=next(i for i,e in enumerate(h.events) if e[0]=='entry_fill_progress');second=next(i for i,e in enumerate(h.events) if e[:3]==('observe','filled',100))
    assert first<second and h.r._pipeline.portfolio.positions['TEST'].quantity==100
    assert h.r._pipeline._symbol_entry_counts=={'TEST':1}
    assert [e for e in h.events if e[0]=='entry_report']==[('entry_report',100.)]
    assert not h.r._partial_entries and status is OrderStatus.FILLED
    assert fill.price==2 and h.r._pipeline.exit_manager._positions['TEST'].trend_strength==.82

def test_stop_cancels_remainder_includes_late_fills_and_exits_exact_inventory():
    h=setup([row(20),row(40,1.995,'pending_cancel',.15),row(50,1.99,'canceled',.3),row(60,1.985,'canceled',.45),row(60,1.985,'canceled',.6)],tape=lambda t:1.89 if t>=.15 else 2.)
    fill,status=h.submit()
    assert fill.quantity==60 and status is OrderStatus.FILLED and not h.r._pipeline.portfolio.positions
    assert h.r._pipeline._symbol_entry_counts['TEST']==1
    assert [e for e in h.events if e[0]=='sell']==[('sell',60.,1.89)]
    assert next(i for i,e in enumerate(h.events) if e[0]=='cancel')<next(i for i,e in enumerate(h.events) if e[0]=='sell')
    assert h.r._pipeline.portfolio.cash==pytest.approx(10000-60*1.985+60*1.89)
    assert len([e for e in h.events if e[0]=='entry_report'])==1

def test_duplicate_updates_do_not_duplicate_cash_inventory_or_entries():
    h=setup([row(20),row(20),row(20),row(20,status='canceled'),row(20,status='canceled')]);h.submit()
    assert h.r._pipeline.portfolio.cash==9960 and h.r._pipeline.portfolio.positions['TEST'].quantity==20
    assert len([e for e in h.events if e[0]=='entry_fill_progress'])==1

def test_changed_average_uses_incremental_notional():
    h=setup([row(20,2.),row(100,1.98,'filled',.15)]);h.submit()
    progress=[e[1] for e in h.events if e[0]=='entry_fill_progress']
    assert progress[1]['delta_price']==pytest.approx((198.-40.)/80)
    assert h.r._pipeline.portfolio.cash==9802 and h.r._pipeline.portfolio.positions['TEST'].avg_price==pytest.approx(1.98)

def test_lost_cancellation_response_still_accounts_late_fill():
    h=setup([row(20),row(40,1.99,'canceled',.15),row(40,1.99,'canceled',.3)],tape=lambda t:1.89)
    def lost(oid):h.events.append(('cancel_lost',oid));raise TimeoutError()
    h.b._client.cancel_order_by_id=lost;h.submit()
    assert [e for e in h.events if e[0]=='sell']==[('sell',40.,1.89)]
    assert h.r._pipeline.portfolio.cash==pytest.approx(9996.)

def test_unconfirmed_cancel_keeps_monitor_no_exit_until_recovered():
    h=setup([row(20),TimeoutError()],tape=lambda t:1.89);fill,status=h.submit()
    assert fill is None and status is OrderStatus.NEW and h.b._safety_blocked
    assert h.r._partial_entries['TEST'].quantity==20 and not [e for e in h.events if e[0]=='sell']
    h.source.extend([row(50,1.99,'canceled',.3),row(60,1.985,'canceled',.45),row(60,1.985,'canceled',.6)])
    with patch('daytrading.execution.alpaca_broker.time',h.clock):assert h.b.recover_order_safety()
    assert [e for e in h.events if e[0]=='sell']==[('sell',60.,1.89)]
    assert not h.r._partial_entries and not h.r._pipeline.portfolio.positions
    assert len([e for e in h.events if e[0]=='entry_report'])==1

def test_ambiguous_submit_later_fill_is_protected_and_reported_once():
    h=setup([row(20),row(100,status='filled')],tape=lambda t:1.89)
    def lost(**kw):raise TimeoutError()
    h.b._client.submit_order=lost;h.b._client.get_order_by_client_id=lambda cid:row(20)
    fill,status=h.submit();assert fill is None and status is OrderStatus.NEW and h.r._partial_entries
    h.b._client.get_order_by_client_id=lambda cid:row(100,status='filled')
    with patch('daytrading.execution.alpaca_broker.time',h.clock):h.b.recover_order_safety()
    assert [e for e in h.events if e[0]=='sell']==[('sell',100.,1.89)] and h.r._pipeline._symbol_entry_counts['TEST']==1

def test_no_fresh_tape_cancels_remainder_without_fabricating_stop_hit():
    h=setup([row(20),row(20,status='canceled'),row(20,status='canceled')],tape=lambda t:None);h.submit()
    assert [e for e in h.events if e[0]=='cancel'] and [e for e in h.events if e[0]=='arm_stop']
    assert not [e for e in h.events if e[0]=='sell']

def test_future_and_pre_submission_prices_do_not_trigger_stop():
    h=setup([row(20),row(100,status='filled')]);h.r._quote_buffer['TEST'].extend([Quote('TEST',T-timedelta(seconds=1),1.5,1.51,100,100),Quote('TEST',T+timedelta(seconds=30),1.5,1.51,100,100)]);h.submit()
    assert not [e for e in h.events if e[0]=='sell']

def test_profitable_tape_does_not_simulate_unexecuted_profit_partial():
    h=setup([row(20),row(40,at=.15),row(100,status='filled',at=.3)],tape=lambda t:2.05 if t<.3 else 1.99);h.submit()
    assert not [e for e in h.events if e[0]=='sell'] and not h.r._pipeline.exit_manager._positions['TEST'].sold_half

def test_stop_never_widens_and_dollar_limit_unchanged():
    h=setup([row(20,2.),row(100,2.05,'filled',.15)]);h.submit()
    stops=[e[1]['stop'] for e in h.events if e[0]=='entry_fill_progress'];assert stops==sorted(stops)
    pos=h.r._pipeline.exit_manager._positions['TEST'];assert (pos.entry_price-pos.stop_loss)*pos.remaining_qty<=50

def test_checkpoint_rejects_pending_first_partial_even_when_flat():
    from daytrading.runtime_state import assert_quiescent
    r=NS(_broker=NS(_paper=True),_hub=NS(trading_paused=True),_pipeline=NS(portfolio=PortfolioState(cash=10000),exit_manager=ExitManager()),_exec_timer=NS(pending_symbols=[]),_timed_signal_queue=[],_partial_entries={'TEST':object()})
    with pytest.raises(ValueError,match='First-partial'):assert_quiescent(r)

def test_terminal_cancel_observed_before_late_quantity_update_is_not_finalized_early():
    h=setup([row(20),row(40,1.99,'canceled',.15),row(60,1.985,'canceled',.3),row(60,1.985,'canceled',.45)],tape=lambda t:2.)
    fill,status=h.submit()
    assert fill.quantity==60 and h.r._pipeline.portfolio.positions['TEST'].quantity==60
    assert h.r._pipeline.portfolio.cash==pytest.approx(10000-60*1.985)

def test_same_quantity_average_correction_does_not_add_inventory():
    h=setup([row(20,2.),row(20,1.99),row(100,1.98,'filled',.3)]);h.submit()
    assert h.r._pipeline.portfolio.positions['TEST'].quantity==100
    assert h.r._pipeline.portfolio.cash==pytest.approx(9802)
    assert h.r._pipeline._symbol_entry_counts['TEST']==1

def test_partial_protective_exit_retains_remaining_inventory():
    h=setup([row(20),row(60,status='canceled'),row(60,status='canceled')],tape=lambda t:1.89)
    h.r._submit_protective_exit_order=lambda order,bar,label:(Fill('TEST',Side.SELL,20,1.89,T),OrderStatus.FILLED)
    h.submit()
    assert h.r._pipeline.portfolio.positions['TEST'].quantity==40
    assert h.r._pipeline.portfolio.cash==pytest.approx(10000-120+20*1.89)
    assert [e for e in h.events if e[0]=='exit_report']==[('exit_report',20)]

def test_monitor_failure_cancels_and_keeps_recovery_block_not_new_buy():
    h=setup([row(20),row(20,status='canceled'),row(20,status='canceled')])
    def broken():raise RuntimeError('market queue failure')
    h.r._drain_partial_entry_events=broken
    fill,status=h.submit()
    assert fill is None and status is OrderStatus.NEW and h.b._safety_blocked
    assert h.b._unsettled_orders and h.r._partial_entries
    assert not [e for e in h.events if e[0]=='sell']

def test_pending_market_batch_and_protection_audit_exactly_and_missing_receipt_fails(tmp_path,monkeypatch):
    from test_recorded_session_validation import make_runner
    from daytrading.replay.store import read_session
    from daytrading.replay.audit import audit_session
    runner,engine=make_runner(tmp_path,monkeypatch)
    runner._market_clock=NS(now=lambda:datetime.now(timezone.utc),status=lambda:dict(synchronized=True,error=None))
    now=datetime.now(timezone.utc);sig=TradeSignal('NCPL',SignalAction.ENTER_LONG,100,entry_price=2,stop_loss=1.9,trend_strength=.82)
    order=Order('NCPL',Side.BUY,100,limit_price=2);bar=Bar('NCPL',now,2,2,2,2,1000, provenance=BarSource.CANONICAL_INTERVAL)
    runner._partial_entries={'NCPL':partial.PendingEntry(sig,order,bar,monitor_after=now)}
    runner._on_entry_progress('buy',order,Fill('NCPL',Side.BUY,20,2,now))
    runner._on_quote(Quote('NCPL',datetime.now(timezone.utc),1.89,1.90,1000,1000))
    assert runner._on_entry_wait('NCPL') is True
    engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
    assert m['state']=='sealed',m['failures']
    next(e for e in events if e['kind']=='bootstrap')['data']['strict_receipts']=True
    result=audit_session(m,events);assert result['status']=='matched',result
    assert runner._pipeline._symbol_entry_counts['NCPL']==1
    without=[e for e in events if e['kind']!='receipt']
    result=audit_session(m,without);assert result['status']=='failed',result
    assert 'receipt' in result['first_divergence']['reason'].lower()

def test_predictive_inputs_reject_observed_partial_fill_callback():
    from daytrading.replay.predict import PredictiveInputs
    with pytest.raises(ValueError,match='signal'):
        PredictiveInputs({},({'data':{'method':'_on_entry_progress'}},),(),(),())

def test_timed_executor_managed_entry_reports_once_without_double_booking():
    from daytrading.execution.entry_executor import EntryExecutionContext
    h=setup([row(20),row(100,status='filled',at=.15)])
    from daytrading.strategy.entry_policy import EntryPolicy
    h.r._entry_executor.set_policy(EntryPolicy())
    p=h.r._pipeline;p._exit_cooldowns={};p._daily_losers=set();p._max_entries_per_symbol=4;p._max_positions=3;p._max_order_shares=500;p._max_position_shares=1000
    context=EntryExecutionContext(new_entries_blocked=lambda *a:False,pipeline=p,bar_buffer={'TEST':[h.bar]},hub=h.r._hub,broker=h.b,journal=h.r._journal,
        chase_reject=lambda *a:None,record_entry_reject=lambda *a,**kw:None,record_missed_a_plus=lambda *a,**kw:None,
        shared_entry_quality_reject=lambda *a,**kw:None,execution_learning_context=lambda *a:{},is_hot_hod_timed_signal=lambda *a:False,
        retry_hot_hod_timed_entry=lambda *a:None,on_position_opened=lambda *a,**kw:None,market_phase=lambda:'PRE-MARKET',seed_recent_order_ids=lambda:None,
        submit_entry=lambda s,o,b:partial.submit(h.r,s,o,b),finish_entry=lambda s:partial.finish(h.r,s))
    with patch('daytrading.execution.alpaca_broker.time',h.clock):h.r._entry_executor.execute_timed_signal(context,h.signal)
    assert h.r._pipeline.portfolio.cash==9800 and p._symbol_entry_counts=={'TEST':1}
    assert [e for e in h.events if e[0]=='entry_report']==[('entry_report',100.)]

def test_failed_final_protection_keeps_state_and_retries_without_double_inventory():
    h=setup([row(20),row(100,status='filled')]);ctx=h.r._entry_execution_context()
    opened=ctx.on_position_opened
    def fail(*a,**kw):raise RuntimeError('temporary protection failure')
    ctx.on_position_opened=fail
    with pytest.raises(RuntimeError,match='protection'):h.submit()
    assert h.r._partial_entries['TEST'].quantity==100 and not h.r._partial_entries['TEST'].reported
    ctx.on_position_opened=opened
    partial.finish(h.r,h.signal)
    assert not h.r._partial_entries and h.r._pipeline.portfolio.cash==9800
    assert h.r._pipeline._symbol_entry_counts['TEST']==1
    assert len([e for e in h.events if e[0]=='entry_report'])==1

@pytest.mark.parametrize('active',[True,False])
def test_delayed_position_response_cannot_overwrite_partial_inventory(active):
    from daytrading.runner import AlpacaRunner
    r=object.__new__(AlpacaRunner);r._entry_reconcile_epoch=0;r._partial_entries={};calls=[]
    def delayed():
        r._entry_reconcile_epoch=1
        r._partial_entries={'TEST':object()} if active else {}
        return {'TEST':dict(qty=20,avg_entry=2.)}
    r._broker=NS(get_positions=delayed);r._cleanup_accidental_shorts=lambda rows:calls.append('cleanup')
    r._reconciler=NS(reconcile=lambda *a:calls.append('reconcile'))
    r._push_positions_from_alpaca()
    assert not calls
