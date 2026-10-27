from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from observation_mode import install,LatencyProbe,REASON
from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.models import Side,OrderStatus

def case(native=False):
    if native:
        b=AlpacaBroker.__new__(AlpacaBroker);b._paper=True;b._client=NS(submit_order=Mock())
        b._strategy_admission=lambda *x:None
    else:
        b=NS(_paper=True,submit=Mock(return_value=('fill',OrderStatus.FILLED)),
             submit_urgent_exit=Mock(return_value=('exit',OrderStatus.FILLED)),
             place_protective_stop=Mock(return_value='stop'),recover_order_safety=Mock(return_value=True),
             _strategy_admission=lambda *x:None)
    p=NS(set_scan_only=Mock(),positions={})
    r=NS(_broker=b,_pipeline=p,_entry_window_checker=lambda:True,
         _entry_block_reason=lambda *x:None,_hub=NS(trading_paused=True,runtime_metrics={},baseline={}),
         _enqueue_market_event=Mock(return_value=True),_process_market_event=Mock(return_value=False))
    def publish():r._hub.runtime_metrics={};r._hub.baseline={'entry_window_open':True}
    r._publish_runtime_metrics=publish
    return r,b

@pytest.mark.parametrize('side',[Side.BUY,Side.SELL])
@pytest.mark.parametrize('method',['submit','submit_urgent_exit'])
def test_all_opening_sides_refused_before_native_sdk(side,method):
    r,b=case(native=True);probe=install(r)
    result=getattr(b,method)(NS(symbol='TEST',side=side,quantity=10),NS(symbol='TEST'),NS(positions={}))
    assert result==(None,OrderStatus.REJECTED);b._client.submit_order.assert_not_called()
    assert probe.snapshot()['rejected_opening_calls']==1

@pytest.mark.parametrize('side,qty',[ (Side.SELL,10),(Side.BUY,-10)])
@pytest.mark.parametrize('method',['submit','submit_urgent_exit'])
def test_reductions_retain_native_controls_and_context(side,qty,method):
    r,b=case();seen=[]
    def native(*args,**kwargs):
        seen.append((r._entry_window_checker(),r._entry_block_reason('TEST','EXIT')))
        return 'native',OrderStatus.NEW
    setattr(b,method,native);install(r)
    assert getattr(b,method)(NS(symbol='TEST',side=side,quantity=5),None,NS(positions={'TEST':NS(quantity=qty)}))==('native',OrderStatus.NEW)
    assert seen==[(True,None)]
    assert not r._entry_window_checker() and r._entry_block_reason('TEST','ENTRY')==REASON

@pytest.mark.parametrize('quantity',[11,0,-1,float('nan'),float('inf')])
def test_oversell_or_invalid_reduction_cannot_open_exposure(quantity):
    r,b=case();original=b.submit_urgent_exit;install(r)
    assert b.submit_urgent_exit(NS(symbol='TEST',side=Side.SELL,quantity=quantity),None,NS(positions={'TEST':NS(quantity=10)}))==(None,OrderStatus.REJECTED)
    original.assert_not_called()

def test_dashboard_resume_or_pipeline_toggle_does_not_clear_latch():
    r,b=case();install(r);r._hub.trading_paused=False;r._pipeline._scan_only=False
    assert r._entry_block_reason('TEST','TIMED')==REASON
    assert not r._entry_window_checker()
    assert b._strategy_admission(None,None,None)==REASON
    with pytest.raises(ValueError):install(r)

def test_native_stop_and_reconciliation_methods_untouched():
    r,b=case();stop=b.place_protective_stop;recovery=b.recover_order_safety
    install(r)
    assert b.place_protective_stop is stop and b.recover_order_safety is recovery
    assert b.place_protective_stop('TEST',5,2.)=='stop';assert b.recover_order_safety()

def test_context_restored_after_unknown_closing_response():
    r,b=case()
    def unknown(*a,**k):raise TimeoutError('unknown')
    b.submit=unknown;install(r)
    with pytest.raises(TimeoutError):b.submit(NS(symbol='TEST',side=Side.SELL,quantity=5),None,NS(positions={'TEST':NS(quantity=10)}))
    assert not r._entry_window_checker() and r._entry_block_reason('TEST','ENTRY')==REASON

def test_live_money_refused_before_changes():
    r,b=case();b._paper=False
    with pytest.raises(ValueError):install(r)
    assert r._entry_window_checker()

def test_rollback_runner_without_new_global_reason_method():
    r,b=case();del r._entry_block_reason
    install(r)
    assert not r._entry_window_checker()
    assert b.submit(NS(symbol='TEST',side=Side.BUY,quantity=1),None,NS(positions={}))==(None,OrderStatus.REJECTED)

def test_probe_bounds_never_drop_market_events():
    p=LatencyProbe(capacity=1);Q=type('QuoteEvent',(),{});a,b=Q(),Q();p.receipt(a);p.receipt(b)
    p.consume(a);p.consume(b);s=p.snapshot();assert s['probe_overflow']==1 and s['unmatched_consumptions']==1
    assert s['receipt_to_processing']['QuoteEvent']['count']==1

def test_instrumentation_preserves_callback_results_and_order():
    r,b=case();events=[]
    r._enqueue_market_event=lambda e:events.append(('receipt',e)) or True
    r._process_market_event=lambda e:events.append(('process',e)) or 123
    p=install(r);Q=type('QuoteEvent',(),{});e=Q()
    assert r._enqueue_market_event(e) is True;assert r._process_market_event(e)==123
    assert events==[('receipt',e),('process',e)]
    r._publish_runtime_metrics();assert r._hub.runtime_metrics['paper_observation']['automatic_entries'] is False
    assert r._hub.baseline['entry_window_open'] is False and r._hub.trading_paused is False

def test_typed_attempt_is_explicitly_not_submitted():
    from daytrading.execution.origin import entry_attempt
    r,b=case();install(r)
    with entry_attempt(NS(symbol="TEST",scan_result=None)) as attempt:
        b.submit(NS(symbol='TEST',side=Side.BUY,quantity=1),None,NS(positions={}))
    assert attempt['state']=='not_submitted'
