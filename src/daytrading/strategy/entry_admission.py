"""Causal route and overhead-room checks; no market or broker I/O."""
from daytrading.data.bar_provenance import canonical_minute
from datetime import timedelta
from math import isfinite

MICRO_ROUTES = frozenset(('micro_pullback_reclaim', 'prepared_reclaim_tick'))


def route_rejection(signal, micro_only):
    if not micro_only:
        return None
    from daytrading.models import SignalAction
    hit = getattr(signal, 'scan_result', None)
    if (hit is None or hit.scanner_name not in MICRO_ROUTES
            or hit.criteria.get('pattern', hit.scanner_name) not in (
                ('micro_pullback_reclaim', 'prepared_reclaim_tick')
                if hit.scanner_name == 'prepared_reclaim_tick' else ('micro_pullback_reclaim',))
            or hit.criteria.get('entry_trigger', hit.scanner_name) != hit.scanner_name
            or signal.action is not SignalAction.ENTER_LONG):
        return 'micro_only_route: original micro reclaim signal required; ordinary/unknown entries disabled'
    return None


def room_check(signal, cap, bars, now, minimum_r=1., market_price=None):
    """Use worst permitted fill and original stop, never a tighter invented stop.

    Only same-session, completed minute bars and already-completed original
    confirmation bars are eligible. A high stops being overhead only after
    observed price crosses it; a permissive
    limit cap cannot manufacture that crossing. Missing evidence fails closed.
    """
    from daytrading.market_calendar import ET
    hit = signal.scan_result
    criteria = hit.criteria if hit is not None else {}
    stops = [signal.stop_loss, criteria.get('micro_original_technical_stop'), criteria.get('stop_price')]
    approved = criteria.get('micro_risk_plan')
    if isinstance(approved, dict):
        stops.append(approved.get('stop'))
    stops = [float(x) for x in stops if isinstance(x, (int, float)) and isfinite(x) and x > 0]
    stop = min(stops) if stops else None
    result = dict(minimum_r=minimum_r, cap=cap, structural_stop=stop, nearby_high=None,
                  room_r=None, source='completed same-session highs', known_high_count=0)
    if (now is None or now.tzinfo is None or stop is None or not isfinite(cap) or cap <= stop):
        return 'room_risk: aware clock, original structural stop and valid price cap required', result
    highs = []
    for bar in bars:
        if (canonical_minute(bar, now) and bar.symbol == signal.symbol and bar.ts.tzinfo is not None
                and bar.ts.astimezone(ET).date() == now.astimezone(ET).date()
                and bar.ts + timedelta(minutes=1) <= now
                and isfinite(bar.high) and bar.high > 0):
            highs.append(bar.high)
    result['canonical_minute_count'] = len(highs)
    if not highs:
        return 'room_risk: canonical completed-minute context unavailable', result
    window = hit.confirmation_window if hit is not None else None
    if window is not None:
        highs.extend(row.bar.high for row in window.bars
            if row.closed_at.tzinfo is not None and row.closed_at <= now
            and row.bar.ts.astimezone(ET).date() == now.astimezone(ET).date()
            and isfinite(row.bar.high) and row.bar.high > 0)
    result['known_high_count'] = len(highs)
    if not highs:
        return 'room_risk: completed overhead context unavailable', result
    reference = signal.entry_price if market_price is None else market_price
    if not isfinite(reference) or reference <= 0:
        return 'room_risk: valid current execution reference required', result
    result['market_reference'] = reference
    overhead = [high for high in highs if high > reference + 1e-9]
    if not overhead:
        result['source'] = 'no overhead among known completed highs'
        return None, result
    high = min(overhead)
    ratio = (high-cap)/(cap-stop)
    result.update(nearby_high=high, room_r=ratio, risk_per_share=cap-stop,
                  room_per_share=high-cap)
    if ratio + 1e-9 < minimum_r:
        return 'room_risk: {:.3f}R below required {:.3f}R to nearby high {:.4f}'.format(ratio, minimum_r, high), result
    return None, result
