"""Observed protection status; acknowledgement is distinct from working cover."""
from datetime import datetime as WallDatetime, timezone


def update(broker, symbol, state, **details):
    rows = broker.__dict__.setdefault('_protection_states', {})
    row = rows.setdefault(symbol, {})
    stamp = WallDatetime.now(timezone.utc).isoformat()
    row.update(state=state, observed_at=stamp, **details)
    if state == 'cancel_requested':
        row['cancel_requested_at'] = stamp
    elif state == 'absent':
        row['cancel_acknowledged_at'] = stamp
        row.setdefault('coverage_gap_started_at', stamp)
    elif state == 'submitting':
        row['submit_requested_at'] = stamp
    elif state == 'acknowledged':
        row['submit_acknowledged_at'] = stamp
    elif state == 'working':
        row['working_observed_at'] = stamp
        start = row.pop('coverage_gap_started_at', None)
        if start:
            row['last_observed_gap_seconds'] = (WallDatetime.fromisoformat(stamp)-WallDatetime.fromisoformat(start)).total_seconds()
    elif state == 'flat':
        start=row.pop('coverage_gap_started_at',None)
        if start:row['uncovered_exposure_to_flat_seconds']=(WallDatetime.fromisoformat(stamp)-WallDatetime.fromisoformat(start)).total_seconds()
        row['covered_quantity']=0
        row['requires_revalidation']=False
    elif state == 'fill_observed':
        row.setdefault('first_fill_observed_at', stamp)
    return row


def acknowledge(broker, symbol, order, quantity, stop):
    """Only a matching observed accepted/new order establishes coverage."""
    from .stop_recovery import unchanged
    def value(name, default=None):
        item = getattr(order, name, default)
        return getattr(item, 'value', item)
    update(broker, symbol, 'acknowledged', order_id=str(order.id), desired_quantity=quantity, desired_stop=stop,
           broker_status=value('status'), order_type=value('type'), time_in_force=value('time_in_force'),
           extended_hours=value('extended_hours', False), requires_revalidation=False,
           covered_quantity=0)
    try:
        matches = unchanged(order, symbol, quantity, stop)
    except (AttributeError, TypeError, ValueError):
        matches = False
    if matches:
        update(broker, symbol, 'working', covered_quantity=quantity, stop_price=round(stop, 2))
    return matches


def current_session(now):
    """Conservative calendar bounds; unsupported years never imply coverage."""
    from datetime import date
    from daytrading.market_calendar import ET, is_us_market_holiday
    local = now.astimezone(ET)
    if local.year not in (2025, 2026):
        return 'UNKNOWN'
    if is_us_market_holiday(local.date()):
        return 'CLOSED'
    early = {date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
             date(2026, 11, 27), date(2026, 12, 24)}
    minute = local.hour * 60 + local.minute
    close = 13 * 60 if local.date() in early else 16 * 60
    if 570 <= minute < close:
        return 'OPEN'
    if 240 <= minute < 570:
        return 'PRE-MARKET'
    return 'AFTER-HOURS' if close <= minute < 1200 else 'CLOSED'


def snapshot(broker, now=None):
    """Observed broker ownership is separate from current-session eligibility.

    This is a diagnostic projection, never permission to place or remove orders.
    A stop is only eligible to trigger in RTH; no order here guarantees a fill.
    """
    from copy import deepcopy
    from daytrading.market_calendar import ET
    now = now or WallDatetime.now(timezone.utc)
    phase = current_session(now)
    rows = deepcopy(getattr(broker, '_protection_states', {}))
    quotes = getattr(broker, '_exit_quote_evidence', {})
    for symbol, row in rows.items():
        evidence = quotes.get(symbol, {})
        try:
            age = (now-WallDatetime.fromisoformat(evidence['quote_ts'])).total_seconds()
            fresh = bool(evidence.get('quote_fresh')) and 0 <= age <= 10
        except (KeyError, ValueError, TypeError):
            age, fresh = None, False
        matched = row.get('covered_quantity', 0) if row.get('state') == 'working' else 0
        try:
            observed = WallDatetime.fromisoformat(row['working_observed_at'])
            current_day = observed.astimezone(ET).date() == now.astimezone(ET).date()
        except (KeyError, ValueError, TypeError):
            current_day = False
        eligible = (phase == 'OPEN' and row.get('broker_status') == 'new'
                    and row.get('order_type') == 'stop' and row.get('time_in_force') in ('day', 'gtc')
                    and current_day and not row.get('requires_revalidation', False) and matched > 0)
        row.update(broker_matching_quantity=matched, session=phase,
            broker_observation_current_day=current_day,
            session_eligible=eligible, session_eligible_quantity=matched if eligible else 0,
            quote_ts=evidence.get('quote_ts'), quote_age_seconds=age, quote_fresh=fresh,
            covered_quantity=matched if eligible and fresh else 0,
            protection_status=('conditionally_eligible' if eligible and fresh else
                'broker_eligible_local_quote_stale' if eligible else
                'awaiting_broker_revalidation' if row.get('requires_revalidation') or not current_day else
                'not_eligible_in_current_session'), fill_guaranteed=False)
    return rows
