from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import threading
import time
from types import SimpleNamespace

import pytest

from daytrading.analytics.missed_a_plus import MissedAPlusRecord
from daytrading.dashboard.hub import DashboardHub
from daytrading.dashboard.server import create_app
from daytrading.execution.broker import PaperBroker
from daytrading.models import PortfolioState
from daytrading.pipeline.engine import TradingPipeline, _RejectCooldown
from daytrading.runtime_state import capture_state, restore_state, digest, request_checkpoint, release_checkpoint

NOW=datetime(2026,10,15,17,0,0,123456,tzinfo=timezone.utc)


def runner():
    p=TradingPipeline([],{},PaperBroker(),PortfolioState(cash=25000))
    p._daily_pnl=-25.00000000000001
    p._symbol_entry_counts={'NCPL':1};p._daily_loss_counts={'NCPL':1}
    p._exit_cooldowns={'NCPL':NOW-timedelta(minutes=3)}
    p._daily_losers={'BAD'}
    p._reject_cooldowns={'TEST':_RejectCooldown('spread too wide',NOW,4.123456789012345,125.5,60)}
    for i in range(65):
        record=MissedAPlusRecord(str(i),'TEST','hod_reclaim','hod_reclaim',NOW,NOW,
            'scanner_near_miss' if i%5==0 else 'verifier','spread too wide',4.123456789012345+i/100,
            a_plus_score=90,correct=False,criteria={'stop_price':3.987654321012345})
        p._missed_a_plus._records[record.id]=record
        p._missed_a_plus._order.append(record.id)
    p.trade_guard.halt_tracker.mark_halted('HALT',NOW)
    p.trade_guard.halt_tracker.mark_resumed('COOL',NOW)
    p.trade_guard.market_panic._panic_active=True
    p.trade_guard.market_panic._panic_start=NOW
    p.trade_guard.slippage._slippage_history['TEST'].append(.012345678901234)
    return SimpleNamespace(_pipeline=p,_broker=SimpleNamespace(_paper=True,secret='DO_NOT_EXPORT'),
        _hub=SimpleNamespace(trading_paused=True),_exec_timer=SimpleNamespace(pending_symbols=[]),
        _timed_signal_queue=deque(),_new_data=threading.Event(),_pipeline_protection_pending={},
        _timed_entry_anchor={'TEST':(4.123456789012345,time.monotonic()-10)},
        _timed_entry_anchor_ttl_sec=300,_recorded_exit_fill_keys={('NCPL','sell',500,2.0,NOW.isoformat())},
        _last_synced_order_ids={'order-one'},_last_session_reset_day='2026-10-15')


def test_full_checkpoint_preserves_more_than_ui_limit_and_all_float_precision():
    source=runner()
    state=capture_state(source,NOW)
    assert state['record_count']==65
    assert len(source._pipeline.missed_a_plus_report())==30
    assert 'DO_NOT_EXPORT' not in json.dumps(state)
    target=runner();target._pipeline._missed_a_plus._records.clear();target._pipeline._missed_a_plus._order.clear()
    evidence=restore_state(target,state,NOW+timedelta(seconds=2))
    restored=capture_state(target,NOW+timedelta(seconds=2))
    for key in ('risk','tracker','reject_cooldowns','halt','panic','slippage_history','parameters'):
        assert restored['guards'][key]==state['guards'][key]
    assert evidence['records_restored']==65
    assert target._pipeline._missed_a_plus._records['0'].price_at_reject==4.123456789012345
    assert target._pipeline._missed_a_plus.chase_reject(symbol='TEST',price=5,now=NOW)
    assert target._pipeline.trade_guard.halt_tracker.check('HALT',NOW).startswith('HALTED')
    assert target._pipeline.trade_guard.halt_tracker.check('COOL',NOW).startswith('post-halt cooldown')
    assert target._pipeline.trade_guard.market_panic.is_panic
    assert target._timed_entry_anchor['TEST'][0]==source._timed_entry_anchor['TEST'][0]
    expected_age=(NOW+timedelta(seconds=2)).timestamp()-state['guards']['runner']['timed_anchors']['TEST']['last_seen_wall']
    assert time.monotonic()-target._timed_entry_anchor['TEST'][1]==pytest.approx(expected_age,abs=.1)
    assert target._recorded_exit_fill_keys==source._recorded_exit_fill_keys
    assert target._hub.trading_paused


@pytest.mark.parametrize('change', ['incomplete','bad_digest','wrong_day','expired','missing_record','changed_parameters'])
def test_invalid_checkpoint_cannot_partially_reset_guards(change):
    state=capture_state(runner(),NOW)
    target=runner();target._pipeline._daily_pnl=-77
    if change=='incomplete':state['complete']=False
    if change=='bad_digest':state['guards_sha256']='wrong'
    if change=='wrong_day':state['day_new_york']='2026-10-14'
    if change=='expired':state['captured_at']=(NOW-timedelta(hours=1)).isoformat()
    if change=='missing_record':
        del state['guards']['tracker']['records']['0']
        state['guards_sha256']=digest(state['guards'])
    if change=='changed_parameters':target._pipeline._max_order_shares+=1
    with pytest.raises(ValueError):restore_state(target,state,NOW)
    assert target._pipeline._daily_pnl==-77
    assert len(target._pipeline._missed_a_plus._records)==65


@pytest.mark.parametrize('change', ['live','unpaused','timed','retained','broker_pending'])
def test_export_requires_paper_paused_flat_and_quiescent(change):
    source=runner()
    if change=='live':source._broker._paper=False
    if change=='unpaused':source._hub.trading_paused=False
    if change=='timed':source._exec_timer.pending_symbols=['TEST']
    if change=='retained':source._pipeline.pending_entries.rows['TEST']=object()
    if change=='broker_pending':source._broker._safety_blocked=True
    with pytest.raises(ValueError):capture_state(source,NOW)


def test_snapshot_runs_at_loop_boundary_and_freeze_releases_cleanly():
    source=runner();original=source._new_data;stop=threading.Event()
    def loop():
        while not stop.is_set():source._new_data.wait(.01)
    thread=threading.Thread(target=loop,daemon=True);thread.start()
    try:
        state=request_checkpoint(source,freeze=True)
        assert state['record_count']==65 and source._new_data is original
        assert not source._paper_handover_lease.release.is_set()
        with pytest.raises(ValueError):request_checkpoint(source,freeze=True)
        release_checkpoint(source)
        assert source._paper_handover_lease.release.is_set()
    finally:
        release_checkpoint(source);stop.set();original.set();thread.join(timeout=1)
    assert not thread.is_alive()


def test_native_route_uses_complete_state_and_inherits_dashboard_access_controls(monkeypatch):
    monkeypatch.delenv('DAYTRADING_DASHBOARD_TOKEN',raising=False)
    hub=DashboardHub();hub.trading_paused=True
    full=capture_state(runner(),NOW)
    hub.paper_handover_callback=lambda freeze=False:full
    hub.paper_handover_release=lambda:{'released':True}
    client=create_app(hub).test_client()
    response=client.post('/api/paper-handover',json={})
    assert response.status_code==200 and response.json['checkpoint']['record_count']==65
    assert client.post('/api/paper-handover',json={},headers={'Origin':'https://example.com'}).status_code==403
    assert client.post('/api/paper-handover',json={'action':'restore'}).status_code==400
    hub.trading_paused=False
    assert client.post('/api/paper-handover',json={}).status_code==409

def test_inferred_activity_source_and_partial_recovery_survive_checkpoint():
    source=runner();h=source._pipeline.trade_guard.halt_tracker
    h.update_price('BENF',1.1,NOW-timedelta(seconds=40))
    h.update_price('BENF',1.1,NOW)
    h.update_price('BENF',1.1,NOW+timedelta(seconds=1))
    state=capture_state(source,NOW+timedelta(seconds=1))
    target=runner();restore_state(target,state,NOW+timedelta(seconds=2))
    restored=target._pipeline.trade_guard.halt_tracker
    assert restored._low_activity=={'BENF'}
    assert 'LOW ACTIVITY' in restored.check('BENF',NOW+timedelta(seconds=2))
    restored.update_price('BENF',1.1,NOW+timedelta(seconds=2))
    assert 'cooldown' in restored.check('BENF',NOW+timedelta(seconds=2))
    assert restored.check('HALT',NOW+timedelta(seconds=2)).startswith('HALTED')


def test_legacy_halt_source_is_not_assumed_from_checkpoint():
    state=capture_state(runner(),NOW)
    del state['guards']['halt']['low_activity'];del state['guards']['halt']['activity_ticks']
    state['guards_sha256']=digest(state['guards'])
    target=runner();restore_state(target,state,NOW+timedelta(seconds=2))
    assert target._pipeline.trade_guard.halt_tracker._low_activity==set()
    assert target._pipeline.trade_guard.halt_tracker.check('HALT',NOW).startswith('HALTED')


@pytest.mark.parametrize('reset_day', ['2026-10-15', None])
def test_overnight_restart_does_not_skip_the_next_premarket_reset(reset_day):
    now=datetime(2026,10,16,6,30,tzinfo=timezone.utc)  # 02:30 New York
    source=runner();source._last_session_reset_day=reset_day
    state=capture_state(source,now)
    assert state['day_new_york']=='2026-10-16'
    target=runner()
    restore_state(target,state,now+timedelta(seconds=2))
    assert target._last_session_reset_day==reset_day
    assert target._pipeline._daily_pnl==source._pipeline._daily_pnl
    assert target._pipeline._symbol_entry_counts=={'NCPL':1}


@pytest.mark.parametrize('reset_day', ['2026-10-17', 'nonsense', 42])
def test_bad_reset_marker_rejected_before_mutating_risk(reset_day):
    state=capture_state(runner(),NOW)
    state['guards']['runner']['last_session_reset_day']=reset_day
    state['guards_sha256']=digest(state['guards'])
    target=runner();target._pipeline._daily_pnl=-77
    with pytest.raises(ValueError,match='session reset day'):
        restore_state(target,state,NOW)
    assert target._pipeline._daily_pnl==-77
