from dataclasses import replace
from datetime import timedelta

import pytest
from test_micro_paper_runtime import AT, signal
from test_staged_entry_boundary import inputs
from daytrading.models import Tick, Side
from daytrading.strategy.scalping.micro_pullback_runtime import MicroPullbackScanner
from daytrading.strategy.submission import final_check


@pytest.mark.parametrize('price', [2.32, 2.50])
@pytest.mark.parametrize('drop_buffer', [False, True])
def test_observed_breach_survives_recovery_retry_removal_and_buffer_rollover(price, drop_buffer):
    sig = signal()
    hit = sig.scan_result
    scanner = MicroPullbackScanner()
    scanner.track_native(hit, AT)
    # Retry row disappears after verifier success; ledger must remain alive.
    clone = scanner.vwap_retries.clone(hit)
    scanner.vwap_retries.decide(hit, 'below VWAP (temporary)', AT)
    scanner.vwap_retries.decide(hit, None, AT)
    assert not scanner.vwap_retries.rows
    breach = Tick('TEST', AT + timedelta(seconds=.1), price, 100, Side.BUY)
    recovery = replace(breach, ts=AT + timedelta(seconds=.2), price=2.42)
    for tick in (breach, breach, recovery):
        scanner.observe_native(tick, tick.ts)
    assert clone.criteria['_native_setup_state'] is hit.criteria['_native_setup_state']
    args = inputs(replace(sig, scan_result=clone))
    args.update(now=AT + timedelta(seconds=.3), ticks=[recovery] if drop_buffer else [breach, recovery])
    why, evidence = final_check(**args)
    assert why == 'native_invalidation: original_support_or_chase_broken'
    assert not evidence['passed']


def test_future_breach_cannot_change_an_earlier_decision_or_poison_new_setup():
    sig = signal()
    scanner = MicroPullbackScanner()
    scanner.track_native(sig.scan_result, AT)
    future = Tick('TEST', AT + timedelta(seconds=1), 2.50, 100, Side.BUY)
    scanner.observe_native(future, AT)
    assert final_check(**inputs(sig))[0] is None
    scanner.observe_native(future, future.ts)
    args = inputs(sig)
    args.update(now=future.ts, ticks=[replace(future, price=2.42)])
    assert final_check(**args)[0].startswith('native_invalidation:')
    new_hit = replace(sig.scan_result, criteria={**sig.scan_result.criteria, 'micro_setup_id': 'new'})
    scanner.track_native(new_hit, AT)
    assert final_check(**inputs(replace(sig, scan_result=new_hit)))[0] is None


def test_native_state_uses_original_candles_and_cannot_be_reanchored():
    sig = signal()
    scanner = MicroPullbackScanner()
    scanner.track_native(sig.scan_result, AT)
    sig.scan_result.criteria['_native_setup_state']['ceiling'] = 100
    why, row = final_check(**inputs(sig))
    assert why == 'native_invalidation: original state identity/anchors mismatch'
    assert not row['passed']


def test_recorded_terminal_state_remains_terminal_after_codec_restore():
    from daytrading.replay.codec import pack, unpack
    sig = signal()
    scanner = MicroPullbackScanner()
    scanner.track_native(sig.scan_result, AT)
    tick = Tick('TEST', AT, 2.50, 100, Side.BUY)
    scanner.observe_native(tick, AT)
    restored = unpack(pack(sig))
    why, row = final_check(**inputs(restored))
    assert why == 'native_invalidation: original_support_or_chase_broken'
    assert not row['passed']
