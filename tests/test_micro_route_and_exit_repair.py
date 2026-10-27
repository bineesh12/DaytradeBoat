from daytrading.models import BarSource
"""Route isolation, INVE timer boundary and live liquidation failure cases.

All broker/data clients are local fakes. No historical fills are simulated here.
"""
from collections import deque
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.execution.durability import ExecutionStore
from daytrading.execution.live_prices import resolve_exit_prices
from daytrading.execution.origin import signal_origin
from daytrading.execution.protection_status import acknowledge, snapshot
from daytrading.exits.manager import ExitManager, TrackedPosition
from daytrading.models import Bar, Order, OrderStatus, PortfolioState, Position, Quote, ScanResult, Side, SignalAction, Tick, TradeSignal
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.runner import AlpacaRunner
from daytrading.strategy.entry_admission import room_check
from daytrading.strategy.entry_policy import EntryPolicy
from daytrading.strategy.execution_timer import ExecutionTimer

T = datetime(2026, 10, 21, 11, 47, 20, tzinfo=timezone.utc)


def signal(route='micro_pullback_reclaim', price=2.70, stop=2.60):
    return TradeSignal('TEST', SignalAction.ENTER_LONG, 100, price, stop, 3.,
        reason='original reclaim', scan_result=ScanResult('TEST', route, T, 95,
            criteria=dict(pattern=route, close=price, stop_price=stop)))


def bars(high=2.90):
    return [Bar('TEST', T.replace(second=0,microsecond=0)-timedelta(minutes=2), 2.6, high, 2.6, 2.7, 100000, provenance=BarSource.CANONICAL_INTERVAL)]


def policy():
    p = EntryPolicy(guard=lambda *a, **k: None)
    p.micro_pullback_only = True
    return p


@pytest.mark.parametrize('route', ['vwap_pullback', 'hod_reclaim', 'first_pullback_reclaim', 'momentum_burst'])
def test_micro_only_rejects_ordinary_route_even_if_quality_passes(route):
    d = policy().evaluate(signal(route), bars=bars(), stage='final', now=T)
    assert not d.passed and d.blocked_layer == 'entry_admission'
    assert 'micro_only_route' in d.reason


def test_pattern_cannot_disguise_ordinary_scanner_and_exits_stay_allowed():
    sig = signal('vwap_pullback'); sig.scan_result.criteria['pattern'] = 'micro_pullback_reclaim'
    assert not policy().evaluate(sig, bars=bars(), stage='final', now=T).passed
    assert policy().evaluate(replace(sig, action=SignalAction.EXIT_LONG), bars=[], stage='exit', now=T).passed


def test_both_real_micro_routes_survive_route_gate_with_room():
    for route in ('micro_pullback_reclaim', 'prepared_reclaim_tick'):
        assert policy().evaluate(signal(route), bars=bars(), stage='final', now=T).passed


def test_unified_factory_filters_mixed_allowlist_but_keeps_shadow_scanners():
    broker = NS(_paper=True)
    p = create_scalping_pipeline(broker=broker,
        entry_patterns=('micro_pullback_reclaim', 'vwap_pullback', 'hod_reclaim'),
        prepared_reclaim_enabled=True, coordinated_micro_flow_enabled=True, unified_micro_entry_enabled=True)
    assert set(p._verifiers) == {'micro_pullback_reclaim', 'prepared_reclaim_tick'}
    assert p._entry_policy.micro_pullback_only
    assert 'vwap_pullback' in {s.name for s in p._scanners}
    assert 'micro_only_route' in broker._strategy_admission(None, Order('TEST', Side.BUY, 1), 2.7)


def test_worst_fill_cap_can_fail_even_when_signal_price_has_one_r_room():
    sig = signal()
    assert room_check(sig, 2.70, bars(2.80), T)[0] is None
    why, evidence = room_check(sig, 2.72, bars(2.80), T)
    assert why.startswith('room_risk:') and evidence['room_r'] == pytest.approx(2/3)
    # A limit above resistance does not prove the market reclaimed resistance.
    assert room_check(sig, 2.81, bars(2.80), T)[1]['room_r'] < 0


def test_room_uses_original_and_approved_stop_no_future_or_wrong_symbol_highs():
    sig = signal(); sig.scan_result.criteria['micro_risk_plan'] = dict(stop=2.5)
    why, evidence = room_check(sig, 2.7, bars(2.8), T)
    assert why and evidence['structural_stop'] == 2.5
    future = replace(bars()[0], ts=T, high=9.)
    wrong = replace(bars()[0], symbol='OTHER', high=8.)
    assert room_check(sig, 2.7, [future, wrong], T)[0].endswith('unavailable')
    assert room_check(sig, 2.9, bars(2.8), T, market_price=2.9)[0] is None


def test_inve_green_below_original_reference_cannot_confirm_or_timeout(monkeypatch):
    timer = ExecutionTimer(max_wait_bars=3)
    sig = signal('vwap_pullback', 2.7285, 2.59)
    timer.queue(sig)
    pending = timer._pending['TEST']
    green = Bar('TEST', T, 2.7103, 2.73, 2.71, 2.721, 21245, provenance=BarSource.CANONICAL_INTERVAL)
    assert not timer._passes_pullback_reclaim_confirmation(pending, green)
    pending.last_bar = green
    pending.queued_at = datetime.now(timezone.utc)-timedelta(seconds=100)
    monkeypatch.setattr(timer, '_allows_elite_fallback', lambda *a: True)
    assert timer.check_timeouts() == [] and timer.pending_symbols == []
    assert timer._passes_pullback_reclaim_confirmation(pending, replace(green, close=2.7285))


def test_tick_shortcut_cannot_release_below_original_reference():
    timer = ExecutionTimer(tick_entry_enabled=True)
    timer.queue(signal('vwap_pullback', 2.7285, 2.59))
    for i in range(3):
        assert timer.on_tick(Tick('TEST', T+timedelta(milliseconds=i), 2.721, 100, Side.BUY)) is None
    assert timer.pending_symbols == ['TEST']


def manager():
    m = ExitManager()
    m.track(TrackedPosition('TEST', Side.BUY, 361, remaining_qty=361,
        entry_price=2.729944, entry_ts=T-timedelta(seconds=301),
        stop_loss=2.59, trend_strength=.95, reason='vwap_pullback', entry_strategy='vwap_pullback'))
    return m


def test_timeout_latches_with_no_price_and_keeps_original_clock_when_price_arrives():
    m = manager(); pos = m.tracked['TEST']
    assert m.check_exits({}, T) == []
    intent = pos.exit_intent
    assert intent and intent.validate_timeout_price
    assert pos.timeout_due_at == T-timedelta(seconds=1)
    first = m.check_exits({'TEST': 2.64}, T+timedelta(seconds=10))[0]
    assert first.quantity == 361 and 'stale_timeout 300s' in first.reason
    assert pos.exit_intent.requested_at == intent.requested_at
    assert not pos.exit_intent.validate_timeout_price


def test_due_ordinary_timeout_preserves_existing_profit_exceptions():
    m = manager(); m.latch_timeouts(T)
    signals = m.check_exits({'TEST': 2.85}, T)
    assert all('stale_timeout' not in x.reason for x in signals)
    assert not (m.tracked.get('TEST') and m.tracked['TEST'].exit_intent
                and m.tracked['TEST'].exit_intent.validate_timeout_price)


def test_cached_position_and_bar_are_never_executable_quote_substitutes():
    stale = Quote('TEST', T-timedelta(seconds=11), 2.6, 2.61, 100, 100)
    detail = {}
    assert resolve_exit_prices(['TEST'], sides={'TEST': Side.BUY}, now=T,
        quotes={'TEST':[stale]}, bars={'TEST':bars()},
        broker_positions={'TEST':dict(current_price=2.7)}, diagnostics=detail) == {}
    assert detail['TEST']['source'] == 'awaiting_fresh_quote'


def runner():
    r = AlpacaRunner.__new__(AlpacaRunner)
    r._pipeline = NS(exit_manager=manager(), portfolio=PortfolioState(cash=2000))
    r._quote_buffer, r._bar_buffer = {}, {}
    r._broker = NS(_slippage_guard=None)
    r._hist = NS(get_latest_quotes=Mock(return_value={'TEST':Quote('TEST', T, 2.64, 2.65, 100, 100)}))
    r._micro_market_now = lambda:T
    return r


def test_disconnected_stream_uses_timestamped_rest_quote_and_checks_age_after_io():
    r = runner()
    assert r._exit_prices(['TEST']) == {'TEST':2.64}
    assert r._exit_price_evidence['TEST']['quote_fresh']
    r = runner()
    def delayed(symbols):
        r._micro_market_now = lambda:T+timedelta(seconds=11)
        return {'TEST':Quote('TEST', T, 2.64, 2.65, 100, 100)}
    r._hist.get_latest_quotes = delayed
    assert r._exit_prices(['TEST']) == {}


def test_timeout_is_checkpointed_before_failed_quote_io():
    r = runner(); stages = []
    r._execution_store = NS(operation=nullcontext, before_broker=lambda evidence:stages.append('checkpoint'))
    def fail(symbols):
        assert r._pipeline.exit_manager.tracked['TEST'].exit_intent
        assert stages == ['checkpoint']
        raise TimeoutError('disconnected')
    r._hist.get_latest_quotes = fail
    r._check_exits_only()
    assert r._pipeline.exit_manager.tracked['TEST'].exit_intent.validate_timeout_price


def test_broker_route_gate_blocks_missing_context_but_allows_short_cover():
    b = AlpacaBroker.__new__(AlpacaBroker); b._slippage_guard=None; b._now=lambda:T
    b._strategy_admission=lambda sig, order, price: 'micro_only_route' if sig is None else None
    b._submit_protected=Mock(return_value=(None, OrderStatus.NEW))
    bar = Bar('TEST', T, 2.7, 2.7, 2.7, 2.7, 0, provenance=BarSource.CANONICAL_INTERVAL)
    book = PortfolioState(cash=2000)
    assert b.submit(Order('TEST', Side.BUY, 10, 2.7), bar, book)[1] is OrderStatus.REJECTED
    b._submit_protected.assert_not_called()
    book.positions['TEST'] = Position('TEST', -10, 2.7)
    assert b.submit(Order('TEST', Side.BUY, 10, 2.7), bar, book)[1] is OrderStatus.NEW
    b._submit_protected.assert_called_once()


@pytest.mark.parametrize('status,phase,eligible', [
    ('accepted','pre',False), ('new','pre',False), ('accepted','rth',False), ('new','rth',True)])
def test_broker_stop_acceptance_is_separate_from_session_eligibility(status, phase, eligible, monkeypatch):
    at = T if phase == 'pre' else T.replace(hour=15)
    class FixedClock(datetime):
        @classmethod
        def now(cls,tz=None):return at
    monkeypatch.setattr('daytrading.execution.protection_status.WallDatetime',FixedClock)
    b = NS()
    order = NS(id='stop', symbol='TEST', side='sell', type='stop', qty=100, filled_qty=0,
               status=status, stop_price=2.6, time_in_force='day', extended_hours=False)
    assert acknowledge(b, 'TEST', order, 100, 2.6)
    b._exit_quote_evidence = {'TEST':dict(quote_ts=at.isoformat(), quote_fresh=True)}
    row = snapshot(b, at)['TEST']
    assert row['broker_matching_quantity'] == 100
    assert row['session_eligible'] is eligible
    assert row['covered_quantity'] == (100 if eligible else 0)
    assert not row['fill_guaranteed']
    stale = snapshot(b, at+timedelta(seconds=11))['TEST']
    assert not stale['quote_fresh'] and stale['covered_quantity'] == 0
    b._protection_states['TEST']['requires_revalidation'] = True
    assert snapshot(b, at)['TEST']['covered_quantity'] == 0


def test_stop_coverage_respects_early_close_and_unknown_calendar_year():
    from daytrading.execution.protection_status import current_session
    assert current_session(datetime(2026,11,27,18,1,tzinfo=timezone.utc)) == 'AFTER-HOURS'
    assert current_session(T.replace(year=2027)) == 'UNKNOWN'


def test_prior_day_stop_observation_cannot_become_current_cover_with_new_quote():
    b=NS();at=T.replace(hour=15)
    order=NS(id='old',symbol='TEST',side='sell',type='stop',qty=100,filled_qty=0,
        status='new',stop_price=2.6,time_in_force='day',extended_hours=False)
    acknowledge(b,'TEST',order,100,2.6)
    b._protection_states['TEST']['working_observed_at']=(at-timedelta(days=1)).isoformat()
    b._exit_quote_evidence={'TEST':dict(quote_ts=at.isoformat(),quote_fresh=True)}
    row=snapshot(b,at)['TEST']
    assert row['quote_fresh'] and row['broker_matching_quantity']==100
    assert not row['session_eligible'] and row['covered_quantity']==0
    assert row['protection_status']=='awaiting_broker_revalidation'


def test_entry_exit_and_stop_requests_keep_strategy_and_reason_lineage():
    b = AlpacaBroker.__new__(AlpacaBroker); b._slippage_guard=None; b._now=lambda:T
    b._submit_protected=Mock(return_value=(None, OrderStatus.NEW)); b._order_audit=Mock()
    sig = signal(); bar = Bar('TEST', T, 2.7, 2.7, 2.7, 2.7, 0, provenance=BarSource.CANONICAL_INTERVAL)
    with signal_origin(sig):
        b.submit(Order('TEST', Side.BUY, 100, 2.7), bar, PortfolioState(cash=2000))
    for kind in ('submit','protective_stop','submit'):
        b._checkpoint_request(kind, 'TEST', 'id')
    assert b._order_audit.call_count == 3
    for call in b._order_audit.call_args_list:
        assert call.args[0]['strategy'] == 'micro_pullback_reclaim'
        assert call.args[0]['entry_reason'] == 'original reclaim'


def test_liquidation_quote_can_age_during_cancel_without_sending_stale_order():
    b = AlpacaBroker.__new__(AlpacaBroker); b._safety_state()
    b._client=NS(submit_order=Mock()); b._now=lambda:T
    b._cancel_open_orders_for=lambda symbol:setattr(b, '_now', lambda:T+timedelta(seconds=11))
    b._restore_protection=Mock(return_value=True)
    order=Order('TEST',Side.SELL,100,2.64,liquidation_quote_ts=T)
    request=NS(limit_price=2.63, client_order_id='fixed')
    assert b._submit_protected(order,request)[1] is OrderStatus.REJECTED
    b._client.submit_order.assert_not_called()
    b._restore_protection.assert_called_once_with('TEST')


def test_due_timeout_checkpoint_survives_restart_without_a_quote(tmp_path):
    from test_reliability_incidents import durable_setup, T as ENTRY_CLOCK
    r,p,store=durable_setup(tmp_path)
    r._micro_market_now=lambda:ENTRY_CLOCK
    p.exit_manager.tracked['YMAT'].entry_ts=ENTRY_CLOCK-timedelta(seconds=301)
    store.save()
    with store.operation():
        assert r._latch_exit_timeouts()==['YMAT']
    r2,p2,recovered=durable_setup(tmp_path)
    assert recovered.restore()
    pos=p2.exit_manager.tracked['YMAT']
    assert pos.exit_intent.validate_timeout_price and pos.exit_intent.requested_at==ENTRY_CLOCK
    assert pos.timeout_due_at==ENTRY_CLOCK-timedelta(seconds=1)


def test_rest_quote_adapter_preserves_exchange_timestamp_and_feed():
    from daytrading.data.alpaca_feed import AlpacaHistoricalFeed
    h=AlpacaHistoricalFeed.__new__(AlpacaHistoricalFeed);h._feed='sip'
    h._client=NS(get_stock_latest_quote=Mock(return_value={
        'TEST':NS(timestamp=T,bid_price=2.64,ask_price=2.65,bid_size=100,ask_size=200)}))
    assert h.get_latest_quotes(['TEST'])['TEST']==Quote('TEST',T,2.64,2.65,100,200)
    request=h._client.get_stock_latest_quote.call_args.args[0]
    assert request.symbol_or_symbols==['TEST'] and request.feed.value=='sip'


def test_reclaim_lost_during_cancellation_is_rejected_before_http():
    from test_micro_paper_runtime import AT, signal as original_signal, minutes, quote as original_quote
    from daytrading.strategy.scalping.micro_pullback import MicroRiskLimits
    sig=original_signal();r=AlpacaRunner.__new__(AlpacaRunner)
    verifier=NS(limits=MicroRiskLimits(),_tick_buffer={'TEST':[Tick('TEST',AT,2.43,100,Side.BUY)]})
    r._pipeline=NS(_entry_policy=policy(),_verifiers={'micro_pullback_reclaim':verifier})
    r._quote_buffer={'TEST':[original_quote(bid=2.42,ask=2.43)]}
    r._bar_buffer={'TEST':minutes()+[Bar('TEST',AT.replace(minute=3,second=0),2.15,2.2,2.1,2.15,10000, provenance=BarSource.CANONICAL_INTERVAL)]};r._micro_market_now=lambda:AT;r._record_entry_reject=Mock()
    b=AlpacaBroker.__new__(AlpacaBroker);b._now=lambda:AT;b._safety_state()
    b._strategy_admission=r._broker_strategy_admission;b._restore_protection=Mock()
    b._client=NS(submit_order=Mock())
    b._slippage_guard=NS(get_limit_price=lambda *a:2.43,get_executable_price=lambda *a:2.43)
    b._cancel_open_orders_for=lambda symbol:verifier._tick_buffer.update(TEST=[Tick('TEST',AT,2.37,100,Side.SELL)])
    order=Order('TEST',Side.BUY,100,2.44,hard_limit_price=2.44)
    with signal_origin(sig):
        fill,status=b.submit(order,Bar('TEST',AT,2.43,2.43,2.43,2.43,100, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=2000))
    assert fill is None and status is OrderStatus.REJECTED
    b._client.submit_order.assert_not_called()
    assert 'lost original reclaim' in r._record_entry_reject.call_args.kwargs['reason']


def test_micro_max_hold_latches_without_price_even_after_a_partial_target_request():
    from daytrading.exits.coordinated_micro import MicroRiskPlan, MicroExitState
    from daytrading.exits.manager import ExitIntent
    m=manager();pos=m.tracked['TEST']
    plan=MicroRiskPlan('TEST',2.74,2.60,100,20)
    pos.quantity=pos.remaining_qty=100;pos.entry_price=2.73
    pos.micro_exit_state=MicroExitState.open(plan,2.73,100,T-timedelta(seconds=60))
    pos.exit_intent=ExitIntent('take_profit',T-timedelta(seconds=10),50,False)
    assert m.check_exits({},T)==[]
    assert pos.exit_intent.liquidate and 'coordinated_micro:time_exit' in pos.exit_intent.reason
    assert pos.exit_intent.quantity==100
