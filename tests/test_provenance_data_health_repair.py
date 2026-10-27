"""Source-boundary regressions; all clients are fakes, no broker I/O."""
from collections import deque
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import pytest
from daytrading.models import Bar, BarSource, Timeframe, Tick, Side, TradeSignal, SignalAction, ScanResult
from daytrading.data.bar_provenance import canonical_minute
from daytrading.data.session_vwap import session_vwap_context
from daytrading.data.completed_bars import Rejection
from daytrading.data.bar_aggregator import BarAggregator
from daytrading.replay.codec import pack, unpack
from daytrading.risk.guards import HaltTracker
from daytrading.strategy.entry_admission import room_check
from daytrading.strategy.scalping.micro_pullback import earlier_impulse
from daytrading.scanner.hod_momentum.session_context import session_context_from_bars
from daytrading.runner import AlpacaRunner

AT=datetime(2026,10,22,12,0,tzinfo=timezone.utc)
def minute(i=0, **kw):
    return replace(Bar('X', AT-timedelta(minutes=10-i),2,2.2,1.9,2.1,10000,
                       provenance=BarSource.CANONICAL_INTERVAL),**kw)

@pytest.mark.parametrize('source', list(BarSource))
def test_provenance_required_even_for_grid_aligned_aged_values(source):
    b=minute(provenance=source)
    assert canonical_minute(b,AT) == (source is BarSource.CANONICAL_INTERVAL)
    assert canonical_minute(unpack(pack(b)),AT) == (source is BarSource.CANONICAL_INTERVAL)

@pytest.mark.parametrize('offset', [timedelta(seconds=1),timedelta(microseconds=1)])
def test_source_declaration_alone_cannot_turn_rolling_time_into_minute(offset):
    assert not canonical_minute(replace(minute(),ts=minute().ts+offset),AT)

@pytest.mark.parametrize('change', [dict(ts=AT),dict(ts=AT+timedelta(minutes=1)),dict(ts=AT.replace(tzinfo=None)),dict(timeframe=Timeframe.SEC_10),dict(timeframe=Timeframe.MIN_5)])
def test_partial_future_unaware_other_timeframes_do_not_enter_minutes(change):
    assert not canonical_minute(minute(**change),AT)

def test_legacy_protocol_one_unlabelled_bar_stays_unknown():
    encoded=pack(minute())
    for node in encoded['nodes']:
        if node.get('type','').endswith(':Bar'):node['fields'].pop('provenance')
    old=unpack(encoded)
    assert old.provenance is BarSource.UNKNOWN
    assert not canonical_minute(old,AT)

@pytest.mark.parametrize('source',[BarSource.UNKNOWN,BarSource.ROLLING_SNAPSHOT,BarSource.PRICE_MARK])
def test_synthetic_highs_volume_and_impulse_cannot_contaminate_context(source):
    bars=[minute(i) for i in range(5)]
    injected=minute(6,high=9,close=9,volume=10**9,provenance=source)
    assert session_vwap_context(bars+[injected],symbol='X',now=AT)==session_vwap_context(bars,symbol='X',now=AT)
    ctx=session_context_from_bars(bars+[injected],now=AT)
    assert ctx.session_high==2.2 and ctx.day_volume==50000
    assert isinstance(earlier_impulse('X',[injected,replace(injected,ts=AT-timedelta(minutes=3),low=1)],now=AT,before=AT),Rejection)
    assert isinstance(session_vwap_context([replace(b,provenance=source) for b in bars],symbol='X',now=AT),Rejection)
    sig=TradeSignal('X',SignalAction.ENTER_LONG,10,2.1,2.,scan_result=ScanResult('X','micro_pullback_reclaim',AT,90))
    why,evidence=room_check(sig,2.1,[injected],AT)
    assert why.endswith('unavailable') and evidence['known_high_count']==0

def test_rolling_injection_does_not_mutate_history_or_include_future_ticks():
    owner=AlpacaRunner.__new__(AlpacaRunner)
    owner._watchlist_set={'X'};owner._watchlist_pinned=set()
    owner._bar_buffer={'X':deque([minute()],maxlen=300)}
    owner._tick_buffer={'X':deque([Tick('X',AT-timedelta(seconds=1),2.11,10,Side.BUY),Tick('X',AT+timedelta(seconds=1),99,999,Side.BUY)])}
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return AT
    universe={'X':list(owner._bar_buffer['X'])}
    with patch('daytrading.runner.datetime',Clock):owner._inject_recent_trade_bars(universe)
    assert len(owner._bar_buffer['X'])==1 and len(universe['X'])==1
    snap=owner._rolling_trade_snapshots['X']
    assert snap.provenance is BarSource.ROLLING_SNAPSHOT
    assert snap.high==snap.low==2.11 and snap.volume==10
    assert not canonical_minute(snap,AT+timedelta(hours=2))

def test_five_minute_aggregation_cannot_reuse_disappeared_canonical_context():
    agg=BarAggregator();bars=[minute(i) for i in range(5)]
    assert agg.build_5m_bars('X',bars)
    assert agg.build_5m_bars('X',[replace(b,provenance=BarSource.ROLLING_SNAPSHOT) for b in bars])==[]
    assert agg.get_5m_bars('X')==[]

@pytest.mark.parametrize('age,reason',[(10.001,'stale_trade'),(30,'stale_trade'),(-.001,'future_trade')])
def test_bad_delivery_blocks_immediately_without_fabricating_inactivity(age,reason):
    h=HaltTracker();h.update_price('X',2,AT,observed_at=AT)
    h.update_price('X',2,AT+timedelta(seconds=1),observed_at=AT+timedelta(seconds=1+age))
    assert reason in h.check('X',AT+timedelta(seconds=50))
    assert 'X' not in h._low_activity and h._last_price['X'][1]==AT
    assert h.activity_snapshot(AT+timedelta(seconds=50))['X']['source']=='data_uncertain'

def test_continuing_stale_tape_is_not_inferred_market_silence():
    h=HaltTracker();h.update_price('X',2,AT,observed_at=AT)
    for s in range(1,32):h.update_price('X',2,AT+timedelta(seconds=s),observed_at=AT+timedelta(seconds=s+11))
    h.update_price('X',2,AT+timedelta(seconds=32),observed_at=AT+timedelta(seconds=32))
    assert 'X' not in h._low_activity and 'DATA HEALTH' in h.check('X',AT+timedelta(seconds=32))
    assert h._data_uncertain['X']['discarded']==31

def test_recovery_requires_fresh_samples_then_full_120_second_cooldown():
    h=HaltTracker();h.update_price('X',2,AT,observed_at=AT+timedelta(seconds=11))
    for s in [20,21]:
        t=AT+timedelta(seconds=s);h.update_price('X',2,t,observed_at=t)
        assert 'DATA HEALTH' in h.check('X',t)
    t=AT+timedelta(seconds=22);h.update_price('X',2,t,observed_at=t)
    assert '120s' in h.check('X',t)
    assert h.check('X',t+timedelta(seconds=119.999)) is not None
    assert h.check('X',t+timedelta(seconds=120)) is None

def test_duplicate_out_of_order_prints_do_not_prove_recovery():
    h=HaltTracker();h.update_price('X',2,AT,observed_at=AT+timedelta(seconds=11))
    t=AT+timedelta(seconds=20)
    for _ in range(10):h.update_price('X',2,t,observed_at=t)
    h.update_price('X',2,t-timedelta(seconds=1),observed_at=t)
    assert 'DATA HEALTH' in h.check('X',t) and len(h._activity_ticks['X'])==1

def test_explicit_halt_is_not_cleared_by_data_recovery():
    h=HaltTracker();h.mark_halted('X',AT)
    h.update_price('X',2,AT,observed_at=AT+timedelta(seconds=11))
    for s in [20,21,22]:
        t=AT+timedelta(seconds=s);h.update_price('X',2,t,observed_at=t)
    assert h.check('X',t).startswith('HALTED')
    h.mark_resumed('X',t)
    assert h.check('X',t).startswith('DATA HEALTH')  # explicit resume does not certify data recovery

def test_genuine_quiet_feed_retains_low_activity_guard():
    h=HaltTracker();h.update_price('X',2,AT,observed_at=AT)
    h.update_price('X',2,AT+timedelta(seconds=31),observed_at=AT+timedelta(seconds=31))
    assert h.check('X',AT+timedelta(seconds=31)).startswith('LOW ACTIVITY')

def test_uncertainty_survives_restart_without_reusing_precrash_samples():
    from test_runtime_state import runner,NOW
    from daytrading.runtime_state import capture_state,restore_state
    a=runner();h=a._pipeline.trade_guard.halt_tracker
    h.update_price('X',2,NOW-timedelta(seconds=11),observed_at=NOW)
    h.update_price('X',2,NOW,observed_at=NOW)
    state=capture_state(a,NOW)
    b=runner();restore_state(b,state,NOW+timedelta(seconds=1));restored=b._pipeline.trade_guard.halt_tracker
    assert 'DATA HEALTH' in restored.check('X',NOW+timedelta(seconds=1))
    assert 'X' not in restored._activity_ticks
    assert restored._data_uncertain['X']['discarded']==1

def test_late_cancel_poll_stays_blocked_without_duplicate_submission(monkeypatch):
    from test_remaining_reliability import broker,filled_order,bar
    from daytrading.execution import alpaca_broker as module
    from daytrading.models import Order,OrderStatus,PortfolioState
    class Clock:
        value=0.
        delay=.005
        def monotonic(self):return self.value
        def sleep(self,seconds):self.value+=max(seconds,self.delay)
    clock=Clock();monkeypatch.setattr(module,'time',clock)
    client=Mock();client.get_orders.return_value=[];client.submit_order.return_value=NS(id='entry')
    client.get_order_by_id.return_value=filled_order('partially_filled',qty=5)
    b=broker(client);b._max_wait=0
    _,status=b.submit(Order('TEST',Side.BUY,20,10),bar(),PortfolioState(cash=10000))
    assert status is OrderStatus.NEW and b._safety_blocked
    client.get_order_by_id.return_value=filled_order('canceled',qty=5)
    assert not b.recover_order_safety() and b._safety_blocked
    clock.delay=0
    assert b.recover_order_safety() and not b._safety_blocked
    assert client.submit_order.call_count==1

@pytest.mark.parametrize('source',[BarSource.UNKNOWN,BarSource.ROLLING_SNAPSHOT,BarSource.PRICE_MARK])
@pytest.mark.parametrize('event',[False,True])
def test_final_submission_fails_closed_for_noncanonical_minute_history(source,event):
    from test_staged_entry_boundary import inputs,event_signal
    from daytrading.strategy.submission import final_check
    args=inputs(event_signal() if event else None)
    args['bars']=[replace(b,provenance=source) for b in args['bars']]
    reason,row=final_check(**args)
    assert reason and not row['passed']
    assert 'minute' in reason.lower() or 'impulse' in reason.lower()

@pytest.mark.parametrize('price',[float('nan'),float('inf'),0,-1])
def test_invalid_prices_block_without_populating_accepted_history(price):
    h=HaltTracker();h.update_price('X',price,AT,observed_at=AT)
    assert 'invalid_price' in h.check('X',AT) and 'X' not in h._last_price

def test_data_health_diagnostics_are_json_safe_without_losing_clock_evidence():
    import json
    h=HaltTracker();h.update_price('X',2,AT,observed_at=AT+timedelta(seconds=11))
    row=h.activity_snapshot(AT+timedelta(seconds=11))['X']
    assert row['data_health']['last_exchange_ts']==AT.isoformat()
    assert row['data_health']['last_observed_at']==(AT+timedelta(seconds=11)).isoformat()
    assert json.loads(json.dumps(row))==row
