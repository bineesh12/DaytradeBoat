from daytrading.models import BarSource
"""Real scanner identity and broker-boundary regressions; SDK is always fake."""
import json
import sqlite3
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from test_micro_paper_runtime import AT, minutes, window, signal as native_signal, quote
from daytrading.models import Bar, Order, OrderStatus, PortfolioState, Side, SignalAction, Tick, TradeSignal
from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.execution.origin import signal_origin
from daytrading.journal.store import TradingJournal
from daytrading.runner import AlpacaRunner
from daytrading.strategy.entry_admission import route_rejection
from daytrading.strategy.entry_policy import EntryPolicy
from daytrading.strategy.scalping.event_reclaim import EventTriggerEvidence, EventReclaimScanner, trailing_windows, candidate_from_event
from daytrading.strategy.scalping.micro_pullback import MicroRiskLimits, Rejection
from daytrading.strategy.submission import final_check, persist


def complete_minutes():
    return minutes()+[Bar('TEST',AT.replace(minute=3,second=0),2.15,2.2,2.1,2.15,10000, provenance=BarSource.CANONICAL_INTERVAL)]


def event_signal(age=.5):
    observed=AT-timedelta(seconds=age)
    times=[(-60,2.2),(-50,2.30),(-40,2.32),(-30,2.34),(-20,2.36),(-10,2.38),
           (-4,2.40),(-3,2.37),(-2.5,2.37),(-2,2.38)]
    ticks=tuple(Tick('TEST',observed+timedelta(seconds=t),p,100,Side.BUY) for t,p in times)+(
        Tick('TEST',observed,2.41,100,Side.BUY),)
    q=quote(ts=observed)
    e=EventTriggerEvidence('TEST','TEST|event|one',observed,2.2,ticks[0].ts,2.4,ticks[6].ts,
        2.37,ticks[7].ts,observed,observed+timedelta(seconds=15),2.41,4,ticks,
        window(),tuple(complete_minutes()),q,trailing_windows('TEST',ticks,observed))
    assert not isinstance(candidate_from_event(e,AT),Rejection)
    scanner=EventReclaimScanner();scanner._clock_ready=lambda:True;scanner.pending['TEST']=e
    hit=scanner.scan({'TEST':complete_minutes()},now=AT)[0]
    return TradeSignal('TEST',SignalAction.ENTER_LONG,500,2.41,2.36,2.51,
        reason='PAPER prepared trade/quote reclaim',scan_result=hit)


def inputs(sig=None):
    sig=sig or native_signal()
    return dict(signal=sig,order=Order('TEST',Side.BUY,100,2.42,hard_limit_price=2.42),
        policy=policy(),bars=complete_minutes(),quote=quote(bid=2.41,ask=2.42),
        ticks=[Tick('TEST',AT,2.42,100,Side.BUY)],now=AT,limits=MicroRiskLimits(),boundary='test')


def policy():
    p=EntryPolicy();p.micro_pullback_only=True;return p


def gate(**changes):
    args=inputs();args.update(changes);return final_check(**args)


@pytest.mark.parametrize('event',[False,True])
def test_actual_scanner_routes_preserve_vwap_context_and_stop(event):
    sig=event_signal() if event else native_signal()
    args=inputs(sig);reason,evidence=final_check(**args)
    assert reason is None and evidence['passed']
    assert route_rejection(sig,True) is None
    assert evidence['strategy']==sig.scan_result.scanner_name
    assert evidence['setup_id'] and evidence['trigger']['level']
    assert len(evidence['source_candles']['impulse'])==2
    assert len(evidence['source_candles']['native_event_context' if event else 'confirmation'])==6
    assert evidence['vwap_context']['vwap_threshold']==pytest.approx(evidence['vwap_context']['vwap_reference']*.995)
    assert sig.stop_loss==(2.36 if event else 2.33)
    assert args['policy'].minimum_room_r==1
    # Above reclaim can still lose the existing session VWAP context floor.
    args['bars']=[replace(b,volume=10000000) if b.high==2.6 else replace(b,volume=1) for b in args['bars']]
    why,row=final_check(**args)
    assert why.startswith('below VWAP:') and not row['passed']


def test_ordinary_inve_route_blocked_even_with_micro_pattern_label():
    sig=native_signal();sig=replace(sig,scan_result=replace(sig.scan_result,scanner_name='vwap_pullback'))
    reason,row=gate(signal=sig)
    assert reason.startswith('micro_only_route:') and not row['passed']


@pytest.mark.parametrize('event',[False,True])
@pytest.mark.parametrize('age',[1.000001,-.01])
def test_every_route_requires_fresh_nonfuture_current_trade(event,age):
    sig=event_signal() if event else native_signal()
    why,row=gate(signal=sig,ticks=[Tick('TEST',AT-timedelta(seconds=age),2.42,100,Side.BUY)])
    assert why and not row['passed'] and row['trade_age_seconds']==pytest.approx(age)


def test_olox_recorded_age_lapse_with_real_event_contract():
    sig=event_signal(age=.989223);e=sig.scan_result.criteria['_event_evidence']
    args=inputs(sig);args['ticks']=[e.ticks[-1]]
    assert final_check(**args)[0] is None
    args['now']=AT+timedelta(seconds=.073560)
    why,row=final_check(**args)
    assert why.startswith('event_execution:') and row['trade_age_seconds']==pytest.approx(1.062783)


def test_full_native_detector_cannot_be_bypassed_at_final_boundary():
    sig=native_signal();w=sig.scan_result.confirmation_window
    rows=list(w.bars);rows[-1]=replace(rows[-1],bar=replace(rows[-1].bar,open=2.415))
    sig=replace(sig,scan_result=replace(sig.scan_result,confirmation_window=replace(w,bars=tuple(rows))))
    why,row=gate(signal=sig)
    assert why.startswith('reclaim:') and not row['passed']


@pytest.mark.parametrize('change',[
    {'limits':None}, {'now':AT.replace(tzinfo=None)},
    {'order':Order('TEST',Side.BUY,501,2.42,hard_limit_price=2.42)},
    {'order':Order('TEST',Side.BUY,100,2.5,hard_limit_price=2.5)},
    {'quote':quote(ts=AT-timedelta(seconds=11))},
    {'quote':quote(ts=AT+timedelta(seconds=.01))},
    {'ticks':[Tick('OTHER',AT,2.42,100,Side.BUY)]},
    {'ticks':[Tick('TEST',AT,2.37,100,Side.BUY)]},
])
def test_invalid_final_inputs_fail_closed(change):
    why,row=gate(**change)
    assert why and not row['passed']
    json.dumps(row,allow_nan=False)


def broker_fixture(sig=None):
    sig=sig or native_signal();data=inputs(sig);clock=[AT];rows=[]
    r=AlpacaRunner.__new__(AlpacaRunner)
    v=NS(limits=data['limits'],_tick_buffer={'TEST':data['ticks']})
    r._pipeline=NS(_entry_policy=data['policy'],_verifiers={sig.scan_result.scanner_name:v})
    r._quote_buffer={'TEST':[data['quote']]};r._bar_buffer={'TEST':data['bars']}
    r._micro_market_now=lambda:clock[0];r._record_entry_reject=Mock()
    r._journal=NS(record=lambda kind,row:rows.append(deepcopy(row)))
    b=AlpacaBroker.__new__(AlpacaBroker);b._now=lambda:clock[0];b._safety_state()
    b._strategy_admission=r._broker_strategy_admission;b._restore_protection=Mock()
    b._client=NS(submit_order=Mock(return_value=NS(id='fake',status='new')))
    b._slippage_guard=NS(get_limit_price=lambda *a:2.42,get_executable_price=lambda *a:2.42)
    b._cancel_open_orders_for=Mock();b._observe_entry_progress=Mock()
    b._wait_for_fill=lambda *a:(None,OrderStatus.NEW)
    return sig,data,r,b,clock,rows


def submit(sig,data,b):
    with signal_origin(sig):
        return b.submit(data['order'],Bar('TEST',AT,2.42,2.42,2.42,2.42,100, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=2000))


@pytest.mark.parametrize('event',[False,True])
def test_real_broker_boundary_allows_valid_original_signal_and_correlates_request(event):
    sig,data,r,b,clock,rows=broker_fixture(event_signal() if event else None)
    assert submit(sig,data,b)[1] is OrderStatus.NEW
    b._client.submit_order.assert_called_once()
    request=b._client.submit_order.call_args.kwargs['order_data']
    assert rows[-1]['client_order_id']==request.client_order_id
    assert rows[-1]['passed'] and len(rows)==3


@pytest.mark.parametrize('stage',['cancel','checkpoint','record'])
@pytest.mark.parametrize('event',[False,True])
def test_delay_after_plan_blocks_sdk_and_restores_protection(stage,event):
    sig,data,r,b,clock,rows=broker_fixture(event_signal() if event else None)
    def delay(*args):clock[0]=AT+timedelta(seconds=1.1)
    if stage=='cancel':b._cancel_open_orders_for=delay
    elif stage=='checkpoint':b._checkpoint_request=delay
    else:
        def record(kind,row):
            rows.append(deepcopy(row));delay()
        r._journal.record=record
    assert submit(sig,data,b)[1] is OrderStatus.REJECTED
    b._client.submit_order.assert_not_called()
    assert not b._unsettled_orders and not b._entry_order_originals
    if stage!='record':b._restore_protection.assert_called()
    assert not rows[-1]['passed']


def test_short_cover_is_not_rechecked_as_entry_after_cancel():
    sig,data,r,b,clock,rows=broker_fixture()
    sig=replace(sig,action=SignalAction.EXIT_SHORT)
    book=PortfolioState(cash=2000);from daytrading.models import Position
    book.positions['TEST']=Position('TEST',-100,2.42)
    with signal_origin(sig):
        assert b.submit(data['order'],Bar('TEST',AT,2.42,2.42,2.42,2.42,100, provenance=BarSource.CANONICAL_INTERVAL),book)[1] is OrderStatus.NEW
    b._client.submit_order.assert_called_once();assert rows==[]


def test_final_snapshot_survives_full_history_and_snapshot_and_event_filters(tmp_path):
    sig=native_signal();sig.scan_result.criteria['quality_inputs']={'entry_checks':[{'old':True}]*256}
    why,row=gate(signal=sig)
    assert why is None and row['checks'] and all('old' not in c for c in row['checks'])
    assert len(sig.scan_result.criteria['quality_inputs']['entry_checks'])==256
    journal=TradingJournal(base_dir=str(tmp_path),record_candle_snapshots=False,
        skipped_event_types=['entry_submission_checks'],prune_on_start=False,daily_prune_enabled=False)
    assert persist(row,journal.record,lambda:AT) is None
    conn=sqlite3.connect(journal._db_path)
    stored=conn.execute("select payload_json from events where type='entry_submission_checks'").fetchone()
    assert stored is not None
    saved=json.loads(stored[0]);assert len(saved['source_candles']['confirmation'])==6
    assert saved['checks']==row['checks']
    # Another evaluation carries only its own bounded checks.
    why,second=gate(signal=sig)
    assert why is None and len(second['checks'])==len(row['checks'])
    assert len(json.dumps(second))<20000


def test_evidence_writer_failure_and_expiry_block_submission():
    why,row=gate();assert why is None
    assert 'recording failed' in persist(row,Mock(side_effect=OSError('disk full')),lambda:AT)
    rows=[]
    assert 'aged during' in persist(row,lambda kind,r:rows.append(r),lambda:AT+timedelta(seconds=1.01))
    assert rows[-1]['passed'] is False


def test_final_evidence_identity_is_repeatable_and_missing_event_is_explicit():
    args=inputs()
    first=final_check(**args)[1];second=final_check(**args)[1]
    assert first['evidence_id']==second['evidence_id']
    sig=event_signal();sig.scan_result.criteria.pop('_event_evidence')
    why,row=gate(signal=sig)
    assert why.startswith('event_type:') and not row['passed']


def test_session_capture_keeps_complete_final_snapshot(tmp_path,monkeypatch):
    from test_recorded_session_validation import make_runner
    from daytrading.replay.engine import ACTIVE,LiveOperation
    from daytrading.replay.store import read_session
    runner,engine=make_runner(tmp_path,monkeypatch)
    why,row=gate();assert why is None
    token=ACTIVE.set(LiveOperation(engine,'final-evidence-fixture'))
    try:runner._journal.record('entry_submission_checks',row)
    finally:ACTIVE.reset(token)
    engine.recorder.close()
    manifest,events=read_session(engine.recorder.root,engine.recorder.session_id)
    decision=next(e for e in events if e['kind']=='decision' and e['data']['type']=='entry_submission_checks')
    from daytrading.replay.codec import unpack
    saved=unpack(decision['data']['payload'])
    assert saved['evidence_id']==row['evidence_id']
    assert len(saved['source_candles']['confirmation'])==6 and saved['checks']==row['checks']


def test_recording_must_preserve_stricter_existing_quote_age_limit():
    why,row=gate(quote=quote(ts=AT-timedelta(seconds=4.9),bid=2.41,ask=2.42),
        limits=replace(MicroRiskLimits(),max_quote_age_seconds=5))
    assert why is None
    assert 'quote aged' in persist(row,lambda *a:None,lambda:AT+timedelta(seconds=.2))


@pytest.mark.parametrize('clock',[None,AT.replace(tzinfo=None)])
def test_event_disconnect_clock_loss_fails_closed(clock):
    why,row=gate(signal=event_signal(),now=clock)
    assert why.startswith('clock_alignment:') and not row['passed']


def test_final_boundary_preserves_authoritative_approved_plan():
    from daytrading.exits.coordinated_micro import approve
    sig=native_signal();approved=approve(sig,2.42,100,9.)
    why,row=gate(signal=sig)
    assert why is None and row['structural_stop']==approved.stop
    assert row['approved_risk_plan']['budget']==9.
    why,row=gate(signal=sig,order=Order('TEST',Side.BUY,101,2.42,hard_limit_price=2.42))
    assert why.startswith('risk:') and not row['passed']
    sig.scan_result.criteria['micro_risk_plan']['budget']=10.
    why,row=gate(signal=sig)
    assert why and not row['passed']  # Typed and serialized original must agree.


def test_overhead_source_is_explicit_even_when_it_is_original_trigger_wick():
    why,row=gate(quote=quote(bid=2.40,ask=2.41),ticks=[Tick('TEST',AT,2.41,100,Side.BUY)])
    assert why.startswith('room_risk:')
    assert row['trigger']['level']==2.38  # Reclaim base remains separate.
    assert row['room_reference']['nearby_high']==2.42
    assert row['source_candles']['overhead_matches'][-1]['kind']=='original_completed_micro'
    assert row['source_candles']['overhead_matches'][-1]['bar']['ts']==window()[-1].bar.ts.isoformat()
