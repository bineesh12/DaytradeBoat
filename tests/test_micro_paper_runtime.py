from daytrading.models import BarSource
"""PAPER activation gates: causal context, shared gates, bounded orders/protection."""
from collections import deque
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from daytrading.data.bar_aggregator import BarAggregator
from daytrading.execution.entry_executor import EntryExecutionContext, EntryExecutor
from daytrading.execution.pricing import marketable_limit
from daytrading.exits.manager import ExitManager
from daytrading.models import Bar, CompletedBar, ConfirmationWindow, Fill, OrderStatus, PortfolioState, Quote, ScanResult, Side, SignalAction, Tick, Timeframe, TradeSignal
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.runner import AlpacaRunner
from daytrading.strategy.entry_policy import EntryPolicy
from daytrading.strategy.scalping.micro_pullback import NAME, MicroRiskLimits, Rejection, earlier_impulse, evaluate_micro_pullback
from daytrading.strategy.scalping.micro_pullback_runtime import MicroPullbackScanner, MicroPullbackVerifier, bounded_micro_order

AT=datetime(2026,10,16,12,11,28,tzinfo=timezone.utc)

def minutes(symbol='TEST'):
    return [Bar(symbol,AT.replace(minute=1,second=0),2,2.1,2,2.1,10000, provenance=BarSource.CANONICAL_INTERVAL),
            Bar(symbol,AT.replace(minute=6,second=0),2.3,2.6,2.3,2.6,10000, provenance=BarSource.CANONICAL_INTERVAL)]

def window(symbol='TEST'):
    start=AT.replace(minute=10,second=20)
    values=[(2.37,2.39,2.32,2.36)]*3+[(2.35,2.38,2.32,2.34),
        (2.37,2.38,2.33,2.35),(2.35,2.42,2.34,2.41)]
    return tuple(CompletedBar(Bar(symbol,start+timedelta(seconds=i*10),*v,10000,
        Timeframe.SEC_10, provenance=BarSource.CANONICAL_INTERVAL),start+timedelta(seconds=(i+1)*10)) for i,v in enumerate(values))

def hit():
    w=window();return ScanResult('TEST',NAME,AT,90,
        criteria=dict(pattern=NAME,setup_anchor=2.41,micro_setup_id='TEST|micro|one',
            micro_bar_start=w[-1].bar.ts.isoformat(),micro_confirmed_at=w[-1].closed_at.isoformat()),
        bars=minutes(),confirmation_window=ConfirmationWindow(w,AT))

def signal():return TradeSignal('TEST',SignalAction.ENTER_LONG,500,2.41,2.33,2.57,
    max_hold_seconds=60,scan_result=hit())

def quote(**changes):return replace(Quote('TEST',AT,2.40,2.41,1000,1000),**changes)

def test_earlier_context_retains_impulse_with_identical_micro_confirmation():
    old=evaluate_micro_pullback('TEST',window(),now=AT)
    new=evaluate_micro_pullback('TEST',window(),now=AT,minute_bars=minutes())
    assert old.code=='impulse' and new.anchor==2.41 and new.technical_stop==2.33

@pytest.mark.parametrize('symbol',['TEST','OTHER','NCPL','BENF','TCRT'])
def test_generic_rule_and_causal_peak_expiry(symbol):
    bs=minutes(symbol)
    ctx=earlier_impulse(symbol,bs,now=AT,before=window(symbol)[-3].bar.ts)
    assert ctx['impulse_pct']==pytest.approx(30)
    assert isinstance(earlier_impulse(symbol,bs,now=AT+timedelta(minutes=1),before=AT),Rejection)

def test_future_partial_and_wrong_symbol_minutes_cannot_manufacture_context():
    extra=[Bar('TEST',AT.replace(second=0),1,9,1,9,100000, provenance=BarSource.CANONICAL_INTERVAL),
           Bar('WRONG',AT.replace(minute=6,second=0),1,9,1,9,100000, provenance=BarSource.CANONICAL_INTERVAL)]
    assert earlier_impulse('TEST',minutes()+extra,now=AT,before=window()[-3].bar.ts)==earlier_impulse('TEST',minutes(),now=AT,before=window()[-3].bar.ts)
    assert isinstance(earlier_impulse('TEST',extra,now=AT,before=AT),Rejection)
    assert isinstance(earlier_impulse('TEST',[minutes()[1]],now=AT,before=AT),Rejection)

def test_actual_completion_time_is_retained_and_cleared():
    a=BarAggregator();start=AT-timedelta(seconds=13)
    assert a.on_tick(Tick('TEST',start,2,100,Side.BUY)) is None
    a.on_tick(Tick('TEST',AT,2.1,100,Side.BUY))
    assert a.get_completed_10s('TEST')[-1].closed_at==AT
    assert a.get_completed_10s('TEST')[-1].bar.ts==start
    a.clear_symbol('TEST');assert not a.get_completed_10s('TEST')

def scanner():
    a=BarAggregator();a._completed_10s['TEST']=list(window())
    s=MicroPullbackScanner();s._bar_aggregator=a;s._now=lambda:AT
    return s

def test_scanner_deduplicates_original_trigger_and_preserves_both_timeframes():
    s=scanner();h=s.scan({'TEST':minutes()})[0]
    assert h.bars==minutes() and h.confirmation_window.bars==window()
    assert h.criteria['micro_close_location']>.65
    assert s.scan({'TEST':minutes()})==[]

def test_late_or_sparse_micro_data_fails_closed():
    s=scanner();s._now=lambda:AT+timedelta(seconds=16)
    assert s.scan({'TEST':minutes()})==[] and s.rejections['stale']==1
    s=scanner();w=list(window());w[-1]=replace(w[-1],closed_at=AT)
    s._bar_aggregator._completed_10s['TEST']=w
    assert s.scan({'TEST':minutes()})==[] and s.rejections['sparse_bar']==1

def test_default_is_off_and_explicit_activation_cannot_use_live_broker():
    assert NAME not in create_scalping_pipeline()._verifiers
    for b in (None,SimpleNamespace(_paper=False)):
        with pytest.raises(ValueError,match='PAPER'):
            create_scalping_pipeline(entry_patterns=(NAME,),broker=b)
    p=create_scalping_pipeline(entry_patterns=(NAME,),broker=SimpleNamespace(_paper=True))
    assert isinstance(p._verifiers[NAME],MicroPullbackVerifier)
    assert any(isinstance(s,MicroPullbackScanner) for s in p._scanners)

def test_shared_verifier_guard_cannot_be_bypassed(monkeypatch):
    import daytrading.strategy.scalping.micro_pullback_runtime as runtime
    guard=Mock(return_value='spread blocked');monkeypatch.setattr(runtime,'check_entry_quality',guard)
    v=MicroPullbackVerifier(MicroRiskLimits());v._quote_buffer={'TEST':[quote()]}
    assert v.verify(hit(),PortfolioState(cash=25000),now=AT) is None
    assert v._last_reject=='spread blocked';assert guard.call_args.kwargs['entry_pattern']==NAME

def test_hard_execution_price_and_worst_risk_survive_later_repricing():
    o=bounded_micro_order(signal(),quote(),minutes()[-1],now=AT,limits=MicroRiskLimits())
    assert o.quantity<=500 and o.quantity*max(o.hard_limit_price-2.33,.05,o.hard_limit_price*.03)<=40+1e-7
    assert marketable_limit(o,minutes()[-1],quote_price=2.40)==o.hard_limit_price
    with pytest.raises(ValueError,match='hard execution limit'):
        marketable_limit(o,minutes()[-1],quote_price=o.hard_limit_price+.01)

@pytest.mark.parametrize('q,why',[(quote(ts=AT-timedelta(seconds=11)),'quote_age'),
    (quote(ask=2.49),'chase'),(quote(ask_size=0),'quote'),(quote(bid=2.41),'quote')])
def test_stale_chased_and_empty_book_orders_are_rejected(q,why):
    r=bounded_micro_order(signal(),q,minutes()[-1],now=AT,limits=MicroRiskLimits())
    assert isinstance(r,Rejection) and r.code==why

def test_confirmation_dispatch_once_without_another_timer():
    runner=AlpacaRunner.__new__(AlpacaRunner);runner._exec_timer=SimpleNamespace(pending_symbols=set(),queue=Mock())
    runner._execute_timed_signal=Mock();runner._record_entry_reject=Mock()
    runner._entry_policy=EntryPolicy();runner._record_entry_decision=Mock()
    runner._dispatch_confirmed_entry(signal());runner._dispatch_confirmed_entry(signal())
    runner._execute_timed_signal.assert_called_once();runner._exec_timer.queue.assert_not_called()
    runner._record_entry_reject.assert_called_once()

def test_protection_failure_remains_pending_even_with_old_stop_id():
    runner=AlpacaRunner.__new__(AlpacaRunner);runner._reconciler=Mock();runner._hub=Mock()
    runner._broker=SimpleNamespace(get_protective_stop_id=lambda s:'old',
        replace_protective_stop=Mock(return_value=None),place_protective_stop=Mock())
    assert not runner._arm_broker_protection('TEST',123,2.33)
    assert runner._pipeline_protection_pending=={'TEST':(123,2.33)}
    runner._broker.replace_protective_stop.return_value='new'
    assert runner._arm_broker_protection('TEST',123,2.33)
    assert runner._pipeline_protection_pending=={}
    runner._broker.replace_protective_stop.assert_called_with('TEST','old',123,2.33)

def test_stop_exception_stays_pending_without_losing_fill_accounting():
    runner=AlpacaRunner.__new__(AlpacaRunner);runner._reconciler=Mock();runner._hub=Mock()
    runner._broker=SimpleNamespace(get_protective_stop_id=lambda s:None,
        place_protective_stop=Mock(side_effect=RuntimeError('connection lost')))
    assert runner._arm_broker_protection('TEST',123,2.33) is False
    assert runner._pipeline_protection_pending=={'TEST':(123,2.33)}
    runner._hub.add_log.assert_called_once()

def test_old_trigger_cannot_be_resubmitted_after_a_different_confirmation():
    runner=AlpacaRunner.__new__(AlpacaRunner);runner._exec_timer=SimpleNamespace(pending_symbols=set(),queue=Mock())
    runner._execute_timed_signal=Mock();runner._record_entry_reject=Mock()
    runner._entry_policy=EntryPolicy();runner._record_entry_decision=Mock()
    a=signal();b=replace(a,scan_result=replace(a.scan_result,criteria={**a.scan_result.criteria,'micro_setup_id':'second'}))
    for s in (a,b,a):runner._dispatch_confirmed_entry(s)
    assert runner._execute_timed_signal.call_count==2

def test_partial_fill_registers_and_protects_actual_quantity(monkeypatch):
    # Full execution boundary, mock broker fill; real filled-size exit registration.
    import daytrading.execution.entry_executor as module
    monkeypatch.setattr(module,'datetime',SimpleNamespace(now=lambda tz:AT))
    p=SimpleNamespace(portfolio=PortfolioState(cash=25000),exit_manager=ExitManager(max_unrealized_loss=50),
        _exit_cooldowns={},_cooldown_seconds=300,_daily_losers=set(),_symbol_entry_counts={},
        _max_entries_per_symbol=3,_max_positions=3,_max_position_shares=1000,_max_order_shares=500,
        _max_dollar_risk_per_trade=50,_max_entry_risk_pct=.06,_original_sizes={},
        trade_guard=SimpleNamespace(check_entry=Mock(return_value=(True,None))))
    p._verifiers={NAME:SimpleNamespace(_tick_buffer={'TEST':[Tick('TEST',AT,2.41,100,Side.BUY)]})}
    broker=SimpleNamespace(_paper=True,submit=Mock(return_value=(Fill('TEST',Side.BUY,123,2.41,AT),OrderStatus.FILLED)),
        place_protective_stop=Mock(return_value='stop'),get_protective_stop_id=lambda s:None)
    runner=AlpacaRunner.__new__(AlpacaRunner);runner._pipeline=p;runner._broker=broker
    runner._reconciler=Mock();runner._push_positions_from_alpaca=Mock();runner._hub=Mock()
    ctx=EntryExecutionContext(new_entries_blocked=lambda *a:False,pipeline=p,bar_buffer={'TEST':minutes()+[Bar('TEST',AT.replace(minute=3,second=0),2.15,2.2,2.1,2.15,10000, provenance=BarSource.CANONICAL_INTERVAL)]},
        hub=Mock(),broker=broker,journal=Mock(),chase_reject=lambda *a:None,
        record_entry_reject=Mock(),record_missed_a_plus=Mock(),shared_entry_quality_reject=lambda *a,**k:None,
        execution_learning_context=lambda *a:{},is_hot_hod_timed_signal=lambda *a:False,
        retry_hot_hod_timed_entry=Mock(),on_position_opened=runner._on_position_opened,
        market_phase=lambda:'PRE-MARKET',seed_recent_order_ids=Mock(),quote_buffer={'TEST':[quote()]})
    EntryExecutor(EntryPolicy(min_price=1,max_price=20)).execute_timed_signal(ctx,signal())
    broker.submit.assert_called_once();assert p.exit_manager.tracked['TEST'].remaining_qty==123
    broker.place_protective_stop.assert_called_once_with('TEST',123,pytest.approx(2.33))
    assert p._symbol_entry_counts=={'TEST':1} and p.portfolio.positions['TEST'].quantity==123

def test_micro_only_cycle_does_not_run_other_scanners_or_retained_setups():
    p=create_scalping_pipeline(entry_patterns=(NAME,),broker=SimpleNamespace(_paper=True))
    other=[s for s in p._scanners if s.name!=NAME]
    for s in other:s.scan=Mock(side_effect=AssertionError('other scanner ran'))
    p.pending_entries.prepare=Mock(side_effect=AssertionError('other retained setup touched'))
    p.run_cycle({'TEST':minutes()},now=AT,quotes={'TEST':[quote()]},only_scanners=frozenset({NAME}),allow_entries=False)
    for s in other:s.scan.assert_not_called()
