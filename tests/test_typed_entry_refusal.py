from daytrading.models import BarSource
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from test_micro_paper_runtime import AT, minutes
from test_project_a_release_requirements import case
from daytrading.execution.entry_outcome import EntryRecovery, RecoverableEntryRefusal
from daytrading.execution.origin import entry_attempt, entry_refused, entry_sdk_started
from daytrading.models import Bar, PortfolioState
from daytrading.pipeline.engine import TradingPipeline
from daytrading.strategy.scalping.micro_failure import classify, Disposition
from daytrading.strategy.scalping.micro_vwap import deadline
from daytrading.replay.codec import pack, unpack


def refusal(message='diagnostic wording may change'):
    return RecoverableEntryRefusal('spread', message, EntryRecovery.ORDER_PLAN_SPREAD)


def test_typed_refusal_roundtrip_shared_classification_and_attempt_authority():
    r, ctx, sig, scanner = case()
    typed = unpack(pack(refusal()))
    assert isinstance(typed, RecoverableEntryRefusal)
    assert classify(typed) is Disposition.TEMPORARY
    assert classify(typed.recovery) is Disposition.TEMPORARY
    assert TradingPipeline._transient_micro_rejection(sig.scan_result, typed)
    # A log string with the same code or diagnostic wording grants no authority.
    assert classify('order_plan_spread') is Disposition.TERMINAL
    assert classify('spread: diagnostic wording may change') is Disposition.TERMINAL
    with pytest.raises(TypeError):
        RecoverableEntryRefusal('spread', 'message', 'order_plan_spread')
    with entry_attempt(sig) as row:
        entry_refused(typed)
        entry_refused('generic order rejected')
        assert row['recovery'] is EntryRecovery.ORDER_PLAN_SPREAD
        assert row['reason'] == 'spread: diagnostic wording may change'
    book = scanner.vwap_retries
    book.decide(sig.scan_result, typed, AT)
    assert 'TEST' in book.rows


def test_message_change_does_not_change_recovery(monkeypatch):
    import daytrading.strategy.scalping.micro_pullback_runtime as module
    r, ctx, sig, scanner = case()
    monkeypatch.setattr(module, 'bounded_micro_order', Mock(return_value=refusal('translated or amended text')))
    r._dispatch_confirmed_entry(sig)
    assert 'TEST' in scanner.vwap_retries.rows
    assert scanner.vwap_retries.last['TEST']['reason'] is EntryRecovery.ORDER_PLAN_SPREAD
    ctx.broker.submit.assert_not_called()


def recover():
    r, ctx, sig, scanner = case()
    r._dispatch_confirmed_entry(sig)
    book = scanner.vwap_retries
    original = book.rows['TEST']
    end = deadline(original)
    when = AT + timedelta(seconds=.25)
    ctx.market_now = lambda: when
    r._micro_market_now = lambda: when
    ctx.quote_buffer['TEST'][-1] = replace(ctx.quote_buffer['TEST'][-1], ts=when, bid=2.41, ask=2.42)
    ticks = ctx.pipeline._verifiers['micro_pullback_reclaim']._tick_buffer['TEST']
    ticks[-1] = replace(ticks[-1], ts=when, price=2.42)
    book.mark(ctx.quote_buffer['TEST'][-1], now=when)
    again = book.take({'TEST': minutes()}, when)
    assert len(again) == 1
    assert again[0].criteria['micro_setup_id'] == original.criteria['micro_setup_id']
    assert again[0].criteria['setup_anchor'] == original.criteria['setup_anchor']
    assert again[0].confirmation_window == original.confirmation_window
    assert deadline(again[0]) == end
    assert sig.stop_loss == 2.33
    return r, ctx, replace(sig, scan_result=again[0]), scanner


@pytest.mark.parametrize('fault', ['none', 'stale_quote', 'stale_trade', 'lost_reclaim', 'chase', 'room', 'position'])
def test_recovery_rechecks_current_submission_conditions(fault):
    r, ctx, sig, scanner = recover()
    when = ctx.market_now()
    ticks = ctx.pipeline._verifiers['micro_pullback_reclaim']._tick_buffer['TEST']
    if fault == 'stale_quote':
        ctx.quote_buffer['TEST'][-1] = replace(ctx.quote_buffer['TEST'][-1], ts=when-timedelta(seconds=10.01))
    elif fault == 'stale_trade':
        ticks[-1] = replace(ticks[-1], ts=when-timedelta(seconds=1.01))
    elif fault == 'lost_reclaim':
        ticks[-1] = replace(ticks[-1], price=2.35)
    elif fault == 'chase':
        ctx.quote_buffer['TEST'][-1] = replace(ctx.quote_buffer['TEST'][-1], bid=2.49, ask=2.50)
    elif fault == 'room':
        ctx.bar_buffer['TEST'].append(Bar('TEST', AT.replace(minute=9,second=0), 2.44,2.45,2.40,2.44,10000, provenance=BarSource.CANONICAL_INTERVAL))
    elif fault == 'position':
        from daytrading.models import Position
        ctx.pipeline.portfolio.positions['TEST'] = Position('TEST', 10, 2.42)
    r._dispatch_confirmed_entry(sig)
    if fault == 'none':
        ctx.broker.submit.assert_called_once()
        # A no-response uninstrumented broker remains consumed, never retried.
        r._dispatch_confirmed_entry(sig)
        ctx.broker.submit.assert_called_once()
    else:
        ctx.broker.submit.assert_not_called()


def test_recovered_hit_returns_through_current_verifier_not_a_stored_signal():
    r, ctx, sig, scanner = recover()
    p = TradingPipeline.__new__(TradingPipeline)
    p._scanners = [scanner]
    p._portfolio = PortfolioState(cash=25000)
    p._market_now = ctx.market_now()
    def verify(hit, portfolio, *, now, session_bars):
        return None
    spy = Mock(side_effect=verify)
    import inspect
    spy.__signature__ = inspect.signature(verify)
    verifier = NS(verify=spy, _last_reject='micro_profile: trigger volume 0.5x < 0.8x')
    current_minutes = minutes()+[Bar('TEST', AT.replace(minute=9,second=0),2.4,2.45,2.39,2.4,100, provenance=BarSource.CANONICAL_INTERVAL)]
    assert p._verify_hit(verifier, sig.scan_result, now=ctx.market_now(), session_bars=current_minutes) is None
    verifier.verify.assert_called_once_with(sig.scan_result, p._portfolio, now=ctx.market_now(), session_bars=current_minutes)
    assert not scanner.vwap_retries.rows
    ctx.broker.submit.assert_not_called()


@pytest.mark.parametrize('state', ['unknown_response', 'http_reject', 'partial_fill', 'cancel'])
def test_late_typed_callback_cannot_revive_sdk_started_attempt(state):
    r, ctx, sig, scanner = case()
    def execute(signal):
        entry_sdk_started('TEST', 'original-client-id')
        if state == 'unknown_response':
            r._broker._unsettled_orders = {'TEST': (None, 'original-client-id')}
        elif state == 'partial_fill':
            r._partial_entries = {'TEST': NS(quantity=10, active_submit=False)}
        entry_refused(refusal())
    r._execute_timed_signal = Mock(side_effect=execute)
    r._dispatch_confirmed_entry(sig)
    r._broker._unsettled_orders = {}
    r._partial_entries = {}
    r._dispatch_confirmed_entry(sig)
    r._execute_timed_signal.assert_called_once()
    assert not scanner.vwap_retries.rows
    assert sig.scan_result.criteria['micro_setup_id'] in scanner.vwap_retries.terminal


@pytest.mark.parametrize('kind', ['expiry', 'support', 'chase'])
def test_typed_recovery_preserves_expiry_and_terminal_breaches(kind):
    r, ctx, sig, scanner = recover()
    book = scanner.vwap_retries
    # Reinsert only the same original hit after another temporary quote failure.
    book.decide(sig.scan_result, refusal(), ctx.market_now())
    when = deadline(sig.scan_result) if kind == 'expiry' else ctx.market_now()
    if kind != 'expiry':
        from daytrading.models import Tick, Side
        tick = Tick('TEST', when, 2.30 if kind=='support' else 2.50, 100, Side.BUY)
        scanner.observe_native(tick, when)
        book.mark(tick, now=when)
        # Verifier observes the shared terminal path before any execution.
        book.decide(sig.scan_result, refusal(), when)
    assert not book.take({'TEST': minutes()}, when)
    assert not book.rows
    assert not book.hold_confirmed(sig.scan_result, 'broker_position_sync_stale', when)
