"""No network: original deadlines, owner adoption, gate wakeups and final guards."""
from concurrent.futures import Future
from datetime import timedelta
from threading import Event, Thread
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from test_micro_paper_runtime import AT, hit, signal, minutes, quote
from test_staged_entry_boundary import event_signal, inputs
from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.models import Tick, Side, PortfolioState
from daytrading.pipeline.engine import TradingPipeline
from daytrading.runner import AlpacaRunner
from daytrading.strategy.scalping.micro_flow import CoordinatedMicroRetries
from daytrading.strategy.scalping.micro_vwap import deadline, identity
from daytrading.strategy.submission import final_check


def broker_case():
    b=AlpacaBroker.__new__(AlpacaBroker);f=Future();wake=Event()
    b._position_pool=NS(submit=Mock(return_value=f));b._position_future=None
    b._client=NS(get_all_positions=lambda:[]);b._pos_cache={};b._pos_cache_ts=0
    b.bind_position_refresh_owner(('version',0),wake)
    return b,f,wake


def book_case():
    return CoordinatedMicroRetries(NS(counts={},observe=lambda *a,**k:None))


@pytest.mark.parametrize('complete_age',[.1,.499999])
def test_owner_adopts_before_original_expiry_without_another_sdk_request(complete_age):
    b,f,wake=broker_case()
    with patch('daytrading.execution.alpaca_broker.time.monotonic',return_value=100):
        assert b.get_positions()=={} and b._position_sync_stale
    def complete():
        with patch('daytrading.execution.alpaca_broker.time.monotonic',return_value=100):f.set_result([])
        assert b._position_sync_stale  # SDK completion cannot mutate decision state.
        assert not b.adopt_completed_positions()  # Non-owner forbidden.
    t=Thread(target=complete);t.start();t.join();assert wake.is_set()
    with patch('daytrading.execution.alpaca_broker.time.monotonic',return_value=100+complete_age):
        assert b.adopt_completed_positions() and not b._position_sync_stale
        assert not b.adopt_completed_positions()  # Duplicate completion/wakeup harmless.
    rows=b.pop_position_refresh_events()
    assert [r['state'] for r in rows]==['requested','adopted']
    assert rows[-1]['requested_at'] and rows[-1]['completed_at'] and rows[-1]['at']
    b._position_pool.submit.assert_called_once()


@pytest.mark.parametrize('conflict',['context','generation','late','failure','missing_version'])
def test_uncertain_or_conflicting_snapshot_stays_blocked(conflict):
    b,f,_=broker_case()
    with patch('daytrading.execution.alpaca_broker.time.monotonic',return_value=100):b.get_positions()
    if conflict=='context':b.bind_position_refresh_owner(('version',1))
    if conflict=='generation':b._invalidate_position_cache()
    if conflict=='missing_version':del f._daytrade_position_meta
    with patch('daytrading.execution.alpaca_broker.time.monotonic',return_value=100):
        if conflict=='failure':f.set_exception(RuntimeError('unavailable'))
        else:f.set_result([])
    with patch('daytrading.execution.alpaca_broker.time.monotonic',return_value=101 if conflict=='late' else 100.1):
        assert not b.adopt_completed_positions() and b._position_sync_stale
    assert b.pop_position_refresh_events()[-1]['state'] in ('failed','discarded')


@pytest.mark.parametrize('event',[False,True])
@pytest.mark.parametrize('release',['before','at','after'])
def test_confirmed_setup_wakes_only_within_original_deadline(event,release):
    h=event_signal().scan_result if event else hit();b=book_case();end=deadline(h)
    assert b.hold_confirmed(h,'broker_position_sync_stale',AT)
    assert b.take({'TEST':minutes()},AT)==[]
    when=end+timedelta(microseconds={'before':-1,'at':0,'after':1}[release])
    b.wake_confirmed(None,when);rows=b.take({'TEST':minutes()},when)
    if release=='before':
        assert len(rows)==1 and identity(rows[0])==identity(h) and deadline(rows[0])==end
        assert rows[0].confirmation_window==h.confirmation_window
        # Repeated gate-clear callbacks do not schedule duplicate evaluations.
        b.wake_confirmed(None,when);assert b.take({'TEST':minutes()},when)==[]
    else:
        assert rows==[] and not b.rows
        assert not b.hold_confirmed(h,'broker_position_sync_stale',when)


@pytest.mark.parametrize('price',[2.30,2.48])
def test_confirmed_setup_cannot_revive_after_support_or_chase_invalidation(price):
    h=hit();h.criteria.update(base_low=2.32,setup_anchor=2.41)
    b=book_case();assert b.hold_confirmed(h,'broker_position_sync_stale',AT)
    t=Tick('TEST',AT+timedelta(seconds=.1),price,100,Side.BUY);b.mark(t,now=t.ts)
    b.wake_confirmed(None,t.ts);assert not b.rows
    assert not b.hold_confirmed(h,'broker_position_sync_stale',t.ts)
    assert b.last['TEST']['reason']=='original_support_or_chase_broken'


def test_duplicate_confirmation_submission_and_terminal_callbacks_cannot_reschedule():
    h=hit();b=book_case();assert b.hold_confirmed(h,'broker_position_sync_stale',AT)
    assert b.hold_confirmed(h,'broker_position_sync_stale',AT)
    assert len([r for r in b.transitions if r['state']=='confirmed_waiting'])==1
    b.finish(h,'submission_started',AT)
    b.finish(h,'submission_started',AT)
    assert not b.hold_confirmed(h,'broker_position_sync_stale',AT)
    b.wake_confirmed(None,AT);assert not b.take({'TEST':minutes()},AT)


def test_gate_clear_requires_reverification_instead_of_stored_signal():
    h=hit();book=book_case();scanner=NS(name=h.scanner_name,vwap_retries=book)
    p=TradingPipeline.__new__(TradingPipeline);p._scanners=[scanner];p._portfolio=PortfolioState(cash=1000)
    p._market_now=AT;p._entry_global_blocker='broker_position_sync_stale'
    verifier=NS(verify=Mock(return_value=signal()),_last_reject=None)
    assert p._verify_hit(verifier,h,now=AT,session_bars=minutes()) is not None
    assert h.symbol in book.blocked and h.symbol in book.rows
    p._entry_global_blocker=None;book.wake_confirmed(None,AT)
    again=book.take({'TEST':minutes()},AT)[0]
    verifier.verify.return_value=None;verifier._last_reject='reclaim: original no longer valid'
    assert p._verify_hit(verifier,again,now=AT,session_bars=minutes()) is None
    assert not book.rows and verifier.verify.call_count==2


@pytest.mark.parametrize('change',['old_trade','old_quote','lost_reclaim','chase','room','stop','dollar_risk'])
@pytest.mark.parametrize('event',[False,True])
def test_woken_candidate_still_fails_final_submission_guards(change,event):
    from dataclasses import replace
    args=inputs(event_signal() if event else signal())
    assert final_check(**args)[0] is None
    if change=='old_trade':args['ticks']=[replace(args['ticks'][-1],ts=AT-timedelta(seconds=1.01))]
    elif change=='old_quote':args['quote']=replace(args['quote'],ts=AT-timedelta(seconds=10.01))
    elif change=='lost_reclaim':args['ticks']=[replace(args['ticks'][-1],price=2.30)]
    elif change=='chase':args['quote']=replace(args['quote'],bid=2.49,ask=2.50)
    elif change=='room':args['policy'].minimum_room_r=100  # test-only adversarial room input.
    elif change=='stop':args['signal']=replace(args['signal'],stop_loss=2.0)
    elif change=='dollar_risk':args['order']=replace(args['order'],quantity=50000)
    reason,evidence=final_check(**args)
    assert reason is not None and not evidence['passed']


def test_exact_position_block_is_not_reported_as_market_closure():
    r=AlpacaRunner.__new__(AlpacaRunner);r._broker=NS(_position_sync_stale=True)
    r._entry_window_checker=lambda:True;r._hub=NS(trading_paused=False);r._pipeline=NS(_circuit_breaker_tripped=False)
    assert r._entry_block_reason(None,'TEST')=='broker_position_sync_stale'
    r._broker._position_sync_stale=False;assert r._entry_block_reason(None,'TEST') is None


def test_unadopted_snapshot_never_means_confirmed_flat_for_urgent_exit():
    from daytrading.models import Order
    r=AlpacaRunner.__new__(AlpacaRunner)
    r._broker=NS(get_positions=lambda:{},_position_sync_stale=True)
    o=Order('TEST',Side.SELL,100,2.0)
    assert r._broker_has_compatible_exit_position('TEST',Side.SELL)
    assert r._clamp_fast_exit_order_to_broker_position(o)==o


def test_owner_reconciliation_conflict_rejects_completed_snapshot():
    b,f,wake=broker_case();r=AlpacaRunner.__new__(AlpacaRunner)
    r._broker=b;r._market_wakeup=wake;r._partial_entries={};r._entry_reconcile_epoch=0
    r._pipeline=NS(portfolio=PortfolioState(cash=1000),exit_manager=NS(tracked={}),_scanners=[])
    r._reconciler=NS(reconcile=Mock());r._cleanup_accidental_shorts=Mock();r._hub=NS(on_position_update=Mock())
    r._journal=NS(record=Mock())
    with patch('daytrading.execution.alpaca_broker.time.monotonic',return_value=100):
        b.bind_position_refresh_owner(r._position_refresh_context(),wake);b.get_positions();f.set_result([])
        r._pipeline.portfolio.cash=900  # A local fill crossed the *request*, not only this read.
        r._service_position_refresh()
    assert b._position_sync_stale
    r._reconciler.reconcile.assert_not_called()
    assert r._journal.record.call_args_list[-1].args[1]['reason']=='position_version_changed'


@pytest.mark.parametrize('economic_work',[False,True])
def test_reporting_checkpoint_runs_native_cycle_only_after_readiness_and_safe_drain(economic_work):
    from daytrading.pipeline.engine import PipelineResult
    r=AlpacaRunner.__new__(AlpacaRunner);b=book_case();h=hit()
    b.hold_confirmed(h,'broker_position_sync_stale',AT)
    r._pipeline=NS(_scanners=[NS(vwap_retries=b)])
    r._service_position_refresh=Mock();r._entry_block_reason=Mock(return_value=None)
    r._micro_market_now=lambda:AT;r._drain_events=Mock();r._run_one_cycle=Mock()
    r._flush_micro_control_evidence=Mock()
    result=PipelineResult()
    if economic_work:result.fills.append(object())
    r._service_confirmed_micro_readiness(result,5)
    if economic_work:
        r._service_position_refresh.assert_not_called();r._run_one_cycle.assert_not_called()
    else:
        r._drain_events.assert_called_once()
        r._run_one_cycle.assert_called_once_with(5,allow_entries=True,
            only_scanners=frozenset({'micro_pullback_reclaim','prepared_reclaim_tick'}))
        assert h.symbol in b.dirty and not b.blocked
        assert not r._micro_readiness_active


def test_queued_adverse_trade_is_processed_before_gate_clear_retry():
    from daytrading.pipeline.engine import PipelineResult
    r=AlpacaRunner.__new__(AlpacaRunner);b=book_case();h=hit();h.criteria['base_low']=2.32
    b.hold_confirmed(h,'broker_position_sync_stale',AT)
    r._pipeline=NS(_scanners=[NS(vwap_retries=b)]);r._service_position_refresh=Mock()
    r._entry_block_reason=lambda *a:None;r._micro_market_now=lambda:AT
    r._drain_events=lambda:b.mark(Tick('TEST',AT,2.30,100,Side.BUY),now=AT)
    r._run_one_cycle=Mock();r._flush_micro_control_evidence=Mock()
    r._service_confirmed_micro_readiness(PipelineResult(),1)
    r._run_one_cycle.assert_not_called();assert not b.rows


def test_finished_future_waits_for_callback_timestamp_before_owner_adoption():
    b,f,_=broker_case()
    with patch('daytrading.execution.alpaca_broker.time.monotonic',return_value=100):b.get_positions();f.set_result([])
    meta=f._daytrade_position_meta;saved=meta.pop('completed_monotonic')
    assert f.done() and not b.adopt_completed_positions()
    assert b._position_future is f and b._position_sync_stale
    meta['completed_monotonic']=saved
    with patch('daytrading.execution.alpaca_broker.time.monotonic',return_value=100.1):
        assert b.adopt_completed_positions()


def test_terminal_verifier_failure_prevents_duplicate_callback_revival():
    b=book_case();h=hit();b.hold_confirmed(h,'broker_position_sync_stale',AT)
    b.decide(h,'native_invalidation: original_support_or_chase_broken',AT)
    assert not b.hold_confirmed(h,'broker_position_sync_stale',AT)
    assert not b.rows


@pytest.mark.parametrize('complete_before_expiry',[True,False])
def test_refresh_completion_and_setup_expiry_share_original_clock(complete_before_expiry):
    b,f,wake=broker_case();book=book_case();h=hit();book.hold_confirmed(h,'broker_position_sync_stale',AT)
    end=deadline(h);delta=(end-AT).total_seconds()+(-.001 if complete_before_expiry else .001)
    with patch('daytrading.execution.alpaca_broker.time.monotonic',return_value=100):b.get_positions()
    with patch('daytrading.execution.alpaca_broker.time.monotonic',return_value=100+delta):
        f.set_result([]);assert wake.is_set() and b._position_sync_stale
        assert b.adopt_completed_positions()
    when=AT+timedelta(seconds=delta);book.wake_confirmed(None,when)
    assert bool(book.take({'TEST':minutes()},when))==complete_before_expiry
    assert deadline(h)==end
