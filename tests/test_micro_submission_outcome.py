"""Explicit pre-SDK refusals can retry; submitted or uncertain IDs cannot."""
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from test_micro_paper_runtime import AT, signal, quote
from test_staged_entry_boundary import inputs, broker_fixture, submit
from daytrading.models import PortfolioState, OrderStatus
from daytrading.strategy.scalping.micro_pullback_runtime import MicroPullbackScanner
from daytrading.strategy.scalping.micro_flow import MicroFlowCollector
from daytrading.strategy.submission import final_check
from daytrading.runner import AlpacaRunner
from daytrading.execution.origin import entry_attempt, entry_refused
from daytrading.execution import partial_entry

@pytest.mark.parametrize('coordinated', [False,True])
def test_explicit_pre_submit_spread_reject_remains_retryable_until_original_expiry(coordinated):
    sig,data,r,b,clock,rows=broker_fixture()
    scanner=MicroPullbackScanner(flow=MicroFlowCollector() if coordinated else None);scanner.track_native(sig.scan_result,AT)
    r._pipeline._scanners=[scanner];r._pipeline._symbol_entry_counts={}
    r._broker=b;r._hub=NS(trading_paused=False)
    r._entry_policy=data['policy'];r._exec_timer=NS(pending_symbols=set())
    r._record_entry_decision=Mock();r._flush_micro_control_evidence=Mock()
    statuses=[]
    def widen(*args,**kwargs):r._quote_buffer['TEST']=[quote(bid=2.38,ask=2.42)]
    b._cancel_open_orders_for=widen
    r._execute_timed_signal=lambda signal:statuses.append(submit(signal,data,b)[1])
    r._dispatch_confirmed_entry(sig)
    # Explicit strategy refusal before any SDK order is not an uncertain response.
    b._client.submit_order.assert_not_called()
    assert any(not row['passed'] for row in rows)
    clock[0]=AT+timedelta(seconds=.25)
    r._quote_buffer['TEST']=[quote(ts=clock[0],bid=2.41,ask=2.42)]
    b._cancel_open_orders_for=Mock()
    recovered={**data,'quote':r._quote_buffer['TEST'][-1],'now':clock[0]}
    assert final_check(**recovered)[0] is None
    assert sig.scan_result.criteria['micro_setup_id'] not in scanner.vwap_retries.terminal, \
        'Temporary pre-submission spread refusal consumed the original setup'
    r._dispatch_confirmed_entry(sig)
    b._client.submit_order.assert_called_once()

@pytest.mark.parametrize('at', ['cancel','checkpoint'])
def test_partial_entry_epoch_bookkeeping_does_not_consume_explicit_no_sdk_refusal(at):
    sig,data,r,b,clock,rows=broker_fixture()
    scanner=MicroPullbackScanner();scanner.track_native(sig.scan_result,AT)
    r._pipeline._scanners=[scanner];r._pipeline._symbol_entry_counts={}
    r._pipeline._portfolio=PortfolioState(cash=25000)
    r._pipeline.portfolio=r._pipeline._portfolio
    r._broker=b;r._hub=NS(trading_paused=False)
    r._entry_policy=data['policy'];r._exec_timer=NS(pending_symbols=set())
    r._record_entry_decision=Mock();r._flush_micro_control_evidence=Mock()
    def widen(*args,**kwargs):r._quote_buffer['TEST']=[quote(bid=2.38,ask=2.42)]
    if at=='cancel':b._cancel_open_orders_for=widen
    else:b._checkpoint_request=widen
    def execute(signal):
        from daytrading.execution.origin import signal_origin
        with signal_origin(signal):
            try:partial_entry.submit(r,signal,data['order'],data['bars'][-1])
            finally:partial_entry.finish(r,signal)
        # Generic reporting must not erase the specific refusal.
        entry_refused('timed entry order rejected')
    r._execute_timed_signal=execute
    r._dispatch_confirmed_entry(sig)
    assert r._entry_reconcile_epoch==2 and not r._partial_entries
    assert r._pipeline.portfolio.cash==25000 and not r._pipeline.portfolio.positions
    b._client.submit_order.assert_not_called()
    assert scanner.vwap_retries.rows['TEST'].criteria['micro_setup_id']==sig.scan_result.criteria['micro_setup_id']
    assert not scanner.vwap_retries.terminal
    assert sig.scan_result.criteria['micro_setup_id'] not in r._micro_consumed_setups['TEST']

@pytest.mark.parametrize('sdk_state', ['http_reject','unknown_response','partial','canceled'])
def test_once_sdk_started_later_temporary_messages_cannot_restore_retry(sdk_state):
    sig,data,r,b,clock,rows=broker_fixture()
    scanner=MicroPullbackScanner();scanner.track_native(sig.scan_result,AT)
    r._pipeline._scanners=[scanner];r._pipeline._symbol_entry_counts={}
    r._broker=b;r._hub=NS(trading_paused=False)
    r._entry_policy=data['policy'];r._exec_timer=NS(pending_symbols=set())
    r._record_entry_decision=Mock();r._flush_micro_control_evidence=Mock()
    if sdk_state=='http_reject':
        error=RuntimeError('explicit fake HTTP rejection');error.status_code=422
        b._client.submit_order.side_effect=error
    elif sdk_state=='unknown_response':
        b._client.submit_order.side_effect=TimeoutError('fake lost response')
    elif sdk_state=='partial':
        b._wait_for_fill=lambda *a:(NS(quantity=10),OrderStatus.NEW)
    else:
        b._wait_for_fill=lambda *a:(None,OrderStatus.CANCELLED)
    outcomes=[]
    def execute(signal):
        outcomes.append(submit(signal,data,b))
        entry_refused('broker_submission: current quote invalid, stale or wider than configured spread limit')
    r._execute_timed_signal=execute
    r._dispatch_confirmed_entry(sig)
    setup=sig.scan_result.criteria['micro_setup_id']
    assert setup in scanner.vwap_retries.terminal
    assert setup in r._micro_consumed_setups['TEST']
    # Even cleared reconciliation and a duplicate wake cannot resubmit this ID.
    b._unsettled_orders.clear();b._safety_blocked=False
    r._dispatch_confirmed_entry(sig)
    b._client.submit_order.assert_called_once()
    assert len(outcomes)==1
    expected={'http_reject':OrderStatus.REJECTED,'unknown_response':OrderStatus.NEW,
              'partial':OrderStatus.NEW,'canceled':OrderStatus.CANCELLED}
    assert outcomes[0][1] is expected[sdk_state]
    if sdk_state=='partial':assert outcomes[0][0].quantity==10

def test_attempt_context_is_reset_and_specific_refusal_is_not_overwritten():
    from daytrading.execution.origin import ENTRY_ATTEMPT
    with entry_attempt(signal()) as row:
        entry_refused('micro_spread: original quote too wide')
        entry_refused('timed entry order rejected')
    assert row['reason']=='micro_spread: original quote too wide'
    assert ENTRY_ATTEMPT.get() is None
    with entry_attempt(signal()) as next_row:assert next_row['state']=='unobserved'
