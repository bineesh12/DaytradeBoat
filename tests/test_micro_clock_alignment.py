from daytrading.models import BarSource
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from test_micro_paper_runtime import AT, hit, signal, minutes, quote, scanner, window
from daytrading.data.completed_bars import validate_micro_window
from daytrading.pipeline.engine import _RejectCooldown
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.risk.guards import TradeGuard
from daytrading.strategy.scalping.micro_pullback import NAME, MicroRiskLimits, Rejection, assess_micro_entry
from daytrading.strategy.scalping.micro_pullback_runtime import candidate_from_hit
from daytrading.execution.entry_executor import EntryExecutionContext, EntryExecutor
from daytrading.models import Bar, PortfolioState, OrderStatus, Tick, Side
from daytrading.strategy.entry_policy import EntryPolicy


def pipeline():return create_scalping_pipeline(entry_patterns=(NAME,),broker=SimpleNamespace(_paper=True))

def test_recorded_tcrt_negative_quote_age_corrected_by_independent_clock_only():
    c=candidate_from_hit(hit());q=quote(ts=AT+timedelta(seconds=.752))
    before=assess_micro_entry(c,q,now=AT)
    assert before.code=='quote_age'
    after=assess_micro_entry(c,q,now=AT+timedelta(seconds=.844))
    assert not isinstance(after,Rejection) and after.planned_risk<=50
    # A real future quote remains invalid; a stale quote never becomes fresh.
    assert assess_micro_entry(c,replace(q,ts=AT+timedelta(seconds=2)),now=AT+timedelta(seconds=.844)).code=='quote_age'
    assert assess_micro_entry(c,replace(q,ts=AT-timedelta(seconds=10)),now=AT+timedelta(seconds=.844)).code=='quote_age'

def test_future_completed_candle_rechecked_without_consuming_trigger():
    s=scanner();end=window()[-1].closed_at
    early=end-timedelta(seconds=.5)
    assert s.scan({'TEST':minutes()},now=early)==[]
    assert s.needs_recheck and 'TEST' not in s.seen
    hits=s.scan({'TEST':minutes()},now=end+timedelta(seconds=.1))
    assert len(hits)==1 and not s.needs_recheck
    assert s.scan({'TEST':minutes()},now=end+timedelta(seconds=.2))==[]

def test_future_retry_bounded_without_bypassing_future_validation():
    s=scanner();end=window()[-1].closed_at
    assert not s.scan({'TEST':minutes()},now=end-timedelta(seconds=30))
    assert not s.scan({'TEST':minutes()},now=end-timedelta(seconds=14))
    assert not s.needs_recheck
    assert validate_micro_window('TEST',window(),now=end-timedelta(seconds=14)).code=='future'

def test_one_cycle_uses_identical_observation_for_scanner_and_post_guard():
    s=scanner();observed=AT+timedelta(seconds=.844)
    h=s.scan({'TEST':minutes()},now=observed)[0]
    assert h.confirmation_window.observed_at==observed
    sig=replace(signal(),scan_result=h)
    guard=TradeGuard();guard.liquidity_trap.check=Mock(return_value=None)
    ok,why=guard.check_entry(sig,bars=minutes(),breakout_context=h.confirmation_window,now=observed)
    assert ok,why
    ok,why=guard.check_entry(sig,bars=minutes(),breakout_context=h.confirmation_window,now=AT)
    assert not ok and 'future context observation' in why

def test_clock_unavailable_blocks_scanner_without_consuming_window():
    s=scanner();s._clock_ready=lambda:False
    assert s.scan({'TEST':minutes()},now=AT)==[] and not s.seen
    s._clock_ready=lambda:True
    assert len(s.scan({'TEST':minutes()},now=AT))==1

@pytest.mark.parametrize('reason',[
    'quote_age: quote age -0.752s outside freshness bounds',
    'confirmation_age: confirmation age -0.1s',
    'clock_alignment: fresh broker clock calibration required',
    'micro confirmation: future: micro candle is not completed as of observation',
    'tape shows selling pressure (imbalance=-0.8)',
])
def test_temporary_data_cache_no_longer_blocks_new_micro_checks_but_history_remains(reason):
    p=pipeline();h=hit();key=p._cooldown_key(h)
    saved=_RejectCooldown(reason,AT,2.41,10000,60)
    p._reject_cooldowns[key]=saved
    assert p._reject_cooldown_reason(h,AT+timedelta(seconds=1)) is None
    assert p._reject_cooldowns[key] is saved
    p._record_reject_cooldown(h,reason,AT+timedelta(seconds=1))
    assert p._reject_cooldowns[key] is saved

@pytest.mark.parametrize('reason',['entry score too low (70/100, need 80+)',
    'stop too wide: 8.7% risk > 6.0% cap','micro_profile: trigger volume 0.5x < 0.8x'])
def test_structural_score_volume_and_risk_cache_still_applies(reason):
    p=pipeline();h=hit();p._record_reject_cooldown(h,reason,AT)
    saved=p._reject_cooldowns[p._cooldown_key(h)]
    assert p._reject_cooldown_reason(h,AT+timedelta(seconds=1))
    p._record_reject_cooldown(h,'quote_age: temporary',AT+timedelta(seconds=1))
    assert p._reject_cooldowns[p._cooldown_key(h)] is saved

def test_other_patterns_and_actual_exit_cooldowns_are_unchanged():
    p=pipeline();h=replace(hit(),scanner_name='hod_reclaim',criteria={'pattern':'hod_reclaim'})
    p._record_reject_cooldown(h,'quote_age: temporary',AT)
    assert p._reject_cooldown_reason(h,AT+timedelta(seconds=1))
    p.set_cooldown('TEST',AT)
    assert p._exit_cooldowns['TEST']==AT and p._cooldown_seconds==300


def execution_context(market_time, *, window_open=True, quote_offset=.752):
    p=SimpleNamespace(portfolio=PortfolioState(cash=25000),_exit_cooldowns={},
        _cooldown_seconds=300,_daily_losers=set(),_symbol_entry_counts={},
        _max_entries_per_symbol=3,_max_positions=3,_max_position_shares=1000,
        _max_order_shares=500,_max_dollar_risk_per_trade=50,_max_entry_risk_pct=.06,
        trade_guard=SimpleNamespace(check_entry=Mock(return_value=(True,None))))
    # The final reclaim check uses a consumed trade, independently of the quote.
    p._verifiers={NAME:SimpleNamespace(_tick_buffer={'TEST':[
        Tick('TEST',AT+timedelta(seconds=quote_offset),2.41,100,Side.BUY)]})}
    broker=SimpleNamespace(_paper=True,submit=Mock(return_value=(None,OrderStatus.REJECTED)))
    ctx=EntryExecutionContext(new_entries_blocked=lambda *a:False,pipeline=p,
        bar_buffer={'TEST':minutes()+[Bar('TEST',AT.replace(minute=3,second=0),2.15,2.2,2.1,2.15,10000, provenance=BarSource.CANONICAL_INTERVAL)]},hub=Mock(),broker=broker,journal=Mock(),
        chase_reject=lambda *a:None,record_entry_reject=Mock(),record_missed_a_plus=Mock(),
        shared_entry_quality_reject=lambda *a,**k:None,execution_learning_context=lambda *a:{},
        is_hot_hod_timed_signal=lambda *a:False,retry_hot_hod_timed_entry=Mock(),
        on_position_opened=Mock(),market_phase=lambda:'PRE-MARKET',seed_recent_order_ids=Mock(),
        quote_buffer={'TEST':[quote(ts=AT+timedelta(seconds=quote_offset))]},
        market_now=lambda:market_time,market_entry_window=lambda t:window_open)
    return ctx


def test_final_executor_passes_aligned_time_to_guard_and_preserves_risk(monkeypatch):
    import daytrading.execution.entry_executor as module
    monkeypatch.setattr(module,'datetime',SimpleNamespace(now=lambda tz:AT))
    aligned=AT+timedelta(seconds=.844);ctx=execution_context(aligned)
    EntryExecutor(EntryPolicy(min_price=1,max_price=20)).execute_timed_signal(ctx,signal())
    ctx.broker.submit.assert_called_once()
    assert ctx.pipeline.trade_guard.check_entry.call_args.kwargs['now']==aligned
    order=ctx.broker.submit.call_args.args[0]
    assert order.quantity<=500 and order.quantity*max(order.hard_limit_price-2.33,.05,order.hard_limit_price*.03)<=50


@pytest.mark.parametrize('market_time,window_open,quote_offset',[
    (None,True,.752),(AT+timedelta(seconds=.844),False,.752),
    (AT+timedelta(seconds=.844),True,2),(AT+timedelta(seconds=20),True,.752)])
def test_final_executor_never_submits_without_clock_window_or_fresh_data(monkeypatch,market_time,window_open,quote_offset):
    import daytrading.execution.entry_executor as module
    monkeypatch.setattr(module,'datetime',SimpleNamespace(now=lambda tz:AT))
    ctx=execution_context(market_time,window_open=window_open,quote_offset=quote_offset)
    EntryExecutor(EntryPolicy(min_price=1,max_price=20)).execute_timed_signal(ctx,signal())
    ctx.broker.submit.assert_not_called();ctx.record_entry_reject.assert_called_once()


def test_pipeline_separates_local_risk_time_from_single_market_observation():
    p=pipeline();p._router=None
    s=SimpleNamespace(name=NAME,accepts_cycle_time=True,scan=Mock(return_value=[]))
    p._scanners=[s];aligned=AT+timedelta(seconds=.844)
    p.run_cycle({'TEST':minutes()},now=AT,market_now=aligned,allow_entries=False)
    assert p._cycle_now==AT and p._market_now==aligned
    assert s.scan.call_args.kwargs['now']==aligned


@pytest.mark.parametrize('fault',['missing','stale','future','wrong_symbol','lost_reclaim'])
def test_fresh_quote_cannot_replace_a_valid_submission_trade(monkeypatch,fault):
    import daytrading.execution.entry_executor as module
    monkeypatch.setattr(module,'datetime',SimpleNamespace(now=lambda tz:AT))
    aligned=AT+timedelta(seconds=.844)
    ctx=execution_context(aligned)
    tick=ctx.pipeline._verifiers[NAME]._tick_buffer['TEST'][0]
    if fault=='stale':tick=replace(tick,ts=aligned-timedelta(seconds=1,microseconds=1))
    if fault=='future':tick=replace(tick,ts=aligned+timedelta(microseconds=1))
    if fault=='wrong_symbol':tick=replace(tick,symbol='OTHER')
    if fault=='lost_reclaim':tick=replace(tick,price=2.3799)
    ctx.pipeline._verifiers[NAME]._tick_buffer['TEST']=[] if fault=='missing' else [tick]
    EntryExecutor(EntryPolicy(min_price=1,max_price=20)).execute_timed_signal(ctx,signal())
    ctx.broker.submit.assert_not_called()
    ctx.record_entry_reject.assert_called_once()
    assert ctx.record_entry_reject.call_args.kwargs['stage']=='submission_recheck'
    assert ctx.record_entry_reject.call_args.kwargs['reason'].startswith('submission_reclaim:')
    assert ctx.pipeline._symbol_entry_counts=={} and not ctx.pipeline.portfolio.positions
