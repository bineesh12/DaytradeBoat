"""Owner-thread native invalidation memory, independent of retry scheduling.

Only consumed causal trades count. A new quote, recovered price, cleared gate
or rolling tick buffer cannot undo an observed original support/chase breach.
Plain state dictionaries are preserved by the existing recording codec.
"""
from datetime import timedelta
from math import isfinite


def original_state(hit):
    window = hit.confirmation_window
    if window is None or len(window.bars) != 6:
        return None
    trigger = window.bars[-1]
    return dict(setup_id=hit.criteria.get('micro_setup_id'), symbol=hit.symbol,
                confirmed_at=trigger.closed_at,
                deadline=trigger.closed_at + timedelta(seconds=15),
                support=trigger.bar.low, ceiling=trigger.bar.close * 1.025,
                reason=None, breached_at=None)


def observe(state, tick, now):
    if state['reason'] or tick.symbol != state['symbol']:
        return
    if tick.ts is None or tick.ts.tzinfo is None:
        state.update(reason='native_path_uncertain', breached_at=None)
        return
    if not state['confirmed_at'] <= tick.ts < state['deadline']:
        return
    if now is None or now.tzinfo is None:
        # The consumed path cannot be verified across a calibration outage.
        # Do not label an unaligned observation a proven support/chase breach.
        # Recovery, buffer rollover and gate-clear callbacks cannot restore it.
        state.update(reason='native_path_uncertain', breached_at=None,
                     uncertainty='broker_clock_unavailable_during_consumed_trade')
        return
    if not state['confirmed_at'] <= tick.ts <= now or tick.ts >= state['deadline']:
        return
    if not isfinite(tick.price) or tick.price <= 0:
        state.update(reason='native_path_uncertain', breached_at=tick.ts)
    elif tick.price < state['support'] or tick.price > state['ceiling']:
        state.update(reason='original_support_or_chase_broken', breached_at=tick.ts)


def check(hit, ticks, now):
    if hit.scanner_name != 'micro_pullback_reclaim':
        return None
    expected = original_state(hit)
    if expected is None:
        return 'native_invalidation: original confirmation missing'
    state = hit.criteria.setdefault('_native_setup_state', expected)
    if any(state.get(k) != expected[k] for k in
           ('setup_id', 'symbol', 'confirmed_at', 'deadline', 'support', 'ceiling')):
        return 'native_invalidation: original state identity/anchors mismatch'
    for tick in ticks or ():
        observe(state, tick, now)
    if state['reason']:
        return 'native_invalidation: ' + state['reason']
    return None
