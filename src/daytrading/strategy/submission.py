"""Existing micro contracts at the final boundary; no new quality/room policy."""
from daytrading.data.bar_provenance import canonical_minute
from dataclasses import asdict
from datetime import datetime, timedelta
from math import isfinite
import hashlib
import json
import logging

from daytrading.strategy.entry_checks import capture, diagnostics, record
from daytrading.strategy.entry_admission import MICRO_ROUTES, route_rejection
from daytrading.strategy.scalping.micro_pullback import Rejection, evaluate_micro_pullback, earlier_impulse
from daytrading.strategy.scalping.micro_pullback_runtime import execution_recheck, submission_reclaim_recheck
from daytrading.strategy.scalping.micro_vwap import fresh_micro_vwap

LOG = logging.getLogger(__name__)


def safe(value):
    if hasattr(value, '__dataclass_fields__'):
        return safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, 'value'):
        return value.value
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def age(now, item):
    try:return (now-item.ts).total_seconds()
    except (AttributeError,TypeError):return None


def final_check(signal, order, *, policy, bars, quote, ticks, now, limits, boundary):
    """Return a complete bounded decision; no mutable full-history copies."""
    hit = getattr(signal, 'scan_result', None)
    c = hit.criteria if hit else {}
    from daytrading.strategy.scalping.event_reclaim import EventTriggerEvidence
    event = c.get('_event_evidence')
    if not isinstance(event,EventTriggerEvidence):event = None
    window = hit.confirmation_window if hit else None
    tick = ticks[-1] if ticks else None
    cap = order.hard_limit_price or order.limit_price
    impulse = c.get('earlier_impulse') or {}
    if event is not None and hasattr(event, 'minute_context'):
        impulse = earlier_impulse(signal.symbol, event.minute_context, now=event.armed_at, before=event.peak_at)
        if isinstance(impulse, Rejection):impulse = {}
    source_minutes = [b for b in bars if canonical_minute(b, now) and b.ts.isoformat() in
                      (impulse.get('trough_start'), impulse.get('peak_start'))]
    evidence = dict(schema='final-entry-v1', evidence_id=None,
        boundary=boundary, symbol=order.symbol, strategy=hit.scanner_name if hit else None,
        setup_id=c.get('event_setup_id') or c.get('micro_setup_id'),
        entry_reason=getattr(signal, 'reason', None), evaluated_at=safe(now),
        quantity=order.quantity, hard_cap=cap, stop=getattr(signal, 'stop_loss', None),
        trigger=dict(level=c.get('base_high', c.get('event_peak')),
            anchor=c.get('setup_anchor'), start=c.get('micro_bar_start'),
            confirmed_at=c.get('micro_confirmed_at', c.get('event_reclaimed_at'))),
        source_candles=dict(confirmation=safe(window.bars[-6:]) if window else [],
            event_trailing=safe(event.windows) if event else [], impulse=safe(source_minutes)),
        event_structure={k:safe(getattr(event,k)) for k in
            ('setup_id','trough','trough_at','peak','peak_at','support','armed_at','reclaimed_at','deadline','anchor')} if event else None,
        trade=safe(tick), quote=safe(quote),
        freshness_limits=dict(trade_seconds=1.,quote_seconds=min(10.,limits.max_quote_age_seconds) if limits else 10.), phases={k:'not_evaluated' for k in
            ('route','pattern','quote_risk','reclaim','vwap_context','admission','hard_cap','reserved_risk')}, checks=[])
    from daytrading.execution.origin import EXECUTION_REQUEST_ID
    evidence['client_order_id'] = EXECUTION_REQUEST_ID.get()
    if event:
        evidence['source_candles']['native_event_context'] = safe(event.native_context[-6:])
        path=[t for t in ticks if now is not None and now.tzinfo is not None
            and t.ts.tzinfo is not None and event.reclaimed_at.tzinfo is not None
            and event.reclaimed_at<=t.ts<=now]
        evidence['reclaim_path_extrema'] = safe(dict(count=len(path),
            lowest=min(path,key=lambda t:t.price),highest=max(path,key=lambda t:t.price))) if path else None
    def reject():
        reason = route_rejection(signal, policy.micro_pullback_only)
        evidence['phases']['route'] = reason or 'passed'
        if reason:return reason
        if hit is None or hit.scanner_name not in MICRO_ROUTES:return None
        if now is None or now.tzinfo is None:
            return 'clock_alignment: fresh broker clock calibration required'
        if limits is None:return 'risk: original verifier limits unavailable'
        if hit.scanner_name=='prepared_reclaim_tick' and event is None:
            return 'event_type: original typed event-trigger evidence required'
        if event is None:
            if window is None:return 'confirmation: six original completed micro bars required'
            candidate = evaluate_micro_pullback(signal.symbol, window.bars, now=now, minute_bars=bars)
            if isinstance(candidate, Rejection):return candidate.code+': '+candidate.reason
            evidence['trigger'].update(level=candidate.base_high, anchor=candidate.anchor,
                start=safe(candidate.bar_start), confirmed_at=safe(candidate.confirmed_at))
            if not impulse:
                source = earlier_impulse(signal.symbol, bars, now=now, before=window.bars[-3].bar.ts)
                if not isinstance(source, Rejection):
                    evidence['source_candles']['impulse'] = safe([b for b in bars if b.ts.isoformat() in
                        (source['trough_start'],source['peak_start'])])
            if signal.stop_loss != candidate.technical_stop:
                return 'stop: original micro technical stop changed'
        evidence['phases']['pattern'] = 'passed' if event is None else 'checked_with_reclaim'
        reason = execution_recheck(signal, quote, now, limits, boundary)
        evidence['phases']['quote_risk'] = reason or 'passed'
        if reason:return reason
        if hit.scanner_name == 'prepared_reclaim_tick':
            from daytrading.strategy.scalping.event_reclaim import submission_reclaim_recheck as event_recheck
            invalid = event_recheck(signal, quote, ticks, now)
            reason = invalid.code+': '+invalid.reason if invalid else None
        else:
            reason = submission_reclaim_recheck(signal, quote, tick, now, ticks=ticks)
        evidence['phases']['reclaim'] = reason or 'passed'
        if reason:return reason
        # Same completed-session VWAP floor and same fresh trade/bid basis as
        # check_entry_quality. Ordinary VWAP entries remain a separate route.
        context = fresh_micro_vwap(bars, ticks, [quote] if quote else [], symbol=signal.symbol, now=now)
        if isinstance(context, Rejection):return context.code+': '+context.reason
        from daytrading.data.session_vwap import session_vwap_context
        eligible = session_vwap_context(bars, symbol=signal.symbol, now=now)['bars']
        evidence['vwap_context'] = context
        evidence['vwap_context']['source_minutes_sha256'] = hashlib.sha256(
            json.dumps(safe(eligible), sort_keys=True).encode()).hexdigest()
        ok = context['vwap_comparison_price'] >= context['vwap_threshold']
        record(diagnostics(hit), 'session_vwap_floor', now=now,
            measured=context['vwap_comparison_price'], threshold=context['vwap_threshold'],
            units='USD/share', comparison='>=', passed=ok,
            window={'source':context['vwap_reference_source']}, quote=quote)
        evidence['phases']['vwap_context'] = 'passed' if ok else 'failed'
        if not ok:return 'below VWAP: fresh trade/bid lost existing session VWAP floor'
        reason = policy.admission_reject(signal, cap=cap, bars=bars, now=now, market_price=quote.ask)
        evidence['room_reference'] = safe(c.get('room_risk'))
        if c.get('room_risk'):
            overhead=c['room_risk']['nearby_high']
            from daytrading.market_calendar import ET
            minutes=[b for b in bars if canonical_minute(b, now) and b.high==overhead and b.symbol==signal.symbol
                and b.ts.tzinfo is not None and b.ts.astimezone(ET).date()==now.astimezone(ET).date()
                and b.ts+timedelta(minutes=1)<=now]
            micros=[b for b in window.bars if b.bar.high==overhead and b.closed_at<=now] if window else []
            evidence['overhead_source_count']=len(minutes)+len(micros)
            evidence['source_candles']['overhead_matches'] = safe(
                [dict(kind='completed_session_minute',bar=b) for b in minutes[-2:]]+
                [dict(kind='original_completed_micro',bar=b.bar,closed_at=b.closed_at) for b in micros[-6:]])
        evidence['phases']['admission'] = reason or 'passed'
        if reason:return reason
        cap_ok = order.hard_limit_price is not None and quote.ask <= cap+1e-12
        record(diagnostics(hit), 'submission_hard_cap', now=now, measured=quote.ask,
            threshold=cap, units='USD/share', passed=cap_ok,
            window={'source':'current ask vs immutable order cap'}, quote=quote)
        evidence['phases']['hard_cap'] = 'passed' if cap_ok else 'failed'
        if not cap_ok:return 'fresh ask exceeds immutable order cap'
        effective = max(cap-signal.stop_loss, limits.min_exit_risk_dollars, cap*limits.min_exit_risk_pct)
        original = max(signal.entry_price-signal.stop_loss, limits.min_exit_risk_dollars,
                       signal.entry_price*limits.min_exit_risk_pct)
        budget = min(limits.max_dollar_risk, signal.quantity*original)
        from daytrading.exits.coordinated_micro import from_signal
        approved = from_signal(signal)
        if approved is not None:
            evidence['approved_risk_plan'] = safe(approved)
            budget = min(budget,approved.budget)
            effective = max(effective,cap-approved.stop)
            if cap != approved.cap or order.quantity > approved.quantity:
                return 'risk: order differs from original approved cap or quantity'
        evidence['structural_stop'] = approved.stop if approved else signal.stop_loss
        ok = (0 < order.quantity <= min(signal.quantity, limits.max_order_shares)
              and order.quantity*effective <= budget+1e-8
              and effective/cap <= limits.max_stop_pct+1e-12
              and cap <= float(c.get('setup_anchor') or signal.entry_price)*(1+limits.max_chase_pct)+1e-12)
        record(diagnostics(hit), 'submission_reserved_risk', now=now,
            measured=order.quantity*effective, threshold=budget, units='USD', passed=ok,
            window={'source':'unchanged original stop, quantity and immutable cap'}, quote=quote)
        evidence['phases']['reserved_risk'] = 'passed' if ok else 'failed'
        if not ok:return 'risk: immutable cap/quantity exceeds original risk or extension limits'
        return None
    with capture() as rows:
        try:
            reason = reject()
        except (AttributeError, TypeError, ValueError, KeyError, ArithmeticError):
            LOG.exception('Invalid final entry inputs')
            reason = 'entry_evidence: invalid or incomplete final-check inputs'
    evidence.update(checks=rows, passed=reason is None, rejection_reason=reason,
        trade_age_seconds=age(now, tick),
        quote_age_seconds=age(now, quote),
        original_deadline=safe(event.deadline if event else window.bars[-1].closed_at+timedelta(seconds=min(15.,limits.max_confirmation_age_seconds) if limits else 15.) if window and window.bars else None))
    # Recomputed decisions retain stable identities during offline audit.
    identity={k:evidence[k] for k in ('boundary','symbol','strategy','setup_id','evaluated_at',
        'quantity','hard_cap','stop','trigger','trade','quote','passed','rejection_reason')}
    evidence['evidence_id']=hashlib.sha256(json.dumps(safe(identity),sort_keys=True).encode()).hexdigest()[:32]
    if hit is not None:c['_final_entry_check_id'] = evidence['evidence_id']
    return reason, safe(evidence)


def persist(evidence, writer, clock=None):
    """Keep the final decision independent of summary caps; fail closed on I/O."""
    try:
        if writer is not None:writer('entry_submission_checks', evidence)
        else:LOG.info('ENTRY_FINAL_CHECK %s', json.dumps(evidence, allow_nan=False))
    except Exception:
        LOG.exception('Cannot persist final entry evidence')
        return 'entry_evidence: final-check recording failed'
    if not evidence['passed'] or clock is None:return evidence['rejection_reason']
    try:now = clock()
    except Exception:now = None
    reason = None
    if now is None or now.tzinfo is None:reason = 'clock_alignment: final evidence clock unavailable'
    else:
        for key,maximum in (('trade',evidence['freshness_limits']['trade_seconds']),
                            ('quote',evidence['freshness_limits']['quote_seconds'])):
            stamp = evidence.get(key,{}).get('ts') if evidence.get(key) else None
            if stamp is None or not 0 <= (now-datetime.fromisoformat(stamp)).total_seconds() <= maximum:
                reason = 'submission_reclaim: '+key+' aged during final evidence recording'
                break
        deadline = evidence.get('original_deadline')
        if deadline and now >= datetime.fromisoformat(deadline):reason = 'confirmation: original deadline expired during final evidence recording'
    if reason:
        failed = dict(evidence, passed=False, rejection_reason=reason, completed_at=safe(now))
        try:
            if writer is not None:writer('entry_submission_checks', failed)
            else:LOG.info('ENTRY_FINAL_CHECK %s',json.dumps(failed,allow_nan=False))
        except Exception:LOG.exception('Cannot persist final entry expiry')
    return reason
