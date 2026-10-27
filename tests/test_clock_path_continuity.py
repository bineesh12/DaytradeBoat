"""Original setup continuity must fail closed across clock outages."""
from dataclasses import replace
from datetime import timedelta
from collections import deque
import pytest
from test_micro_paper_runtime import AT, signal
from test_staged_entry_boundary import inputs
from daytrading.models import Tick, Side
from daytrading.strategy.scalping.micro_pullback_runtime import MicroPullbackScanner
from daytrading.strategy.scalping.micro_flow import MicroFlowCollector
from daytrading.strategy.submission import final_check

@pytest.mark.parametrize('retained', [False,True])
@pytest.mark.parametrize('breach_price', [2.32,2.50])
def test_clock_outage_cannot_forget_consumed_support_breach_after_rollover(retained,breach_price):
    sig=signal();scanner=MicroPullbackScanner(flow=MicroFlowCollector());scanner.track_native(sig.scan_result,AT)
    if retained:scanner.vwap_retries.hold_confirmed(sig.scan_result,'broker_position_sync_stale',AT)
    breach=Tick('TEST',AT+timedelta(seconds=.1),breach_price,100,Side.BUY)
    recovered=replace(breach,ts=AT+timedelta(seconds=.2),price=2.42)
    # Production returns None from _micro_market_now when calibration is stale.
    scanner.observe_native(breach,None)
    scanner.vwap_retries.mark(breach,now=None)
    # Actual runner uses deque(maxlen=200). This is high tape rate, not expiry.
    ticks=deque([breach],maxlen=200)
    for i in range(200):
        recovered=replace(recovered,ts=AT+timedelta(seconds=.2+i*.014))
        ticks.append(recovered);scanner.observe_native(recovered,recovered.ts)
        scanner.vwap_retries.mark(recovered,now=recovered.ts)
    args=inputs(sig);now=AT+timedelta(seconds=3.1)
    args.update(now=now,ticks=list(ticks),quote=replace(args['quote'],ts=now))
    reason,evidence=final_check(**args)
    assert reason is not None, 'Original setup became admissible after an untracked support breach during clock outage'
    assert not evidence['passed']

def test_clock_gap_is_uncertainty_not_a_fabricated_breach_and_new_setup_is_independent():
    sig=signal();s=MicroPullbackScanner();s.track_native(sig.scan_result,AT)
    s.observe_native(Tick('TEST',AT,2.42,100,Side.BUY),None)
    state=sig.scan_result.criteria['_native_setup_state']
    assert state['reason']=='native_path_uncertain' and state['breached_at'] is None
    other=replace(sig.scan_result,criteria={**sig.scan_result.criteria,'micro_setup_id':'new'})
    s.track_native(other,AT)
    assert final_check(**inputs(replace(sig,scan_result=other)))[0] is None
