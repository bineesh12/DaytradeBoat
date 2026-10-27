"""Causal, JSON-safe entry checks shared by verifier, risk and execution."""
from datetime import datetime, timezone
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging

LOG = logging.getLogger(__name__)
_CAPTURES = ContextVar('entry_check_captures', default=())


@contextmanager
def capture():
    """Fresh boundary evidence, independent of the capped setup history."""
    rows = []
    token = _CAPTURES.set((*_CAPTURES.get(), rows))
    try:
        yield rows
    finally:
        _CAPTURES.reset(token)


def diagnostics(hit):
    if hit is None:return None
    target=hit.criteria.setdefault('quality_inputs', {})
    target['entry_check_context']=dict(symbol=hit.symbol,scanner=hit.scanner_name,
        setup_id=hit.criteria.get('event_setup_id') or hit.criteria.get('micro_setup_id'))
    return target


def record(target, name, *, now, measured, threshold, units, passed, window,
           quote=None, comparison='<=', **details):
    now = now or datetime.now(timezone.utc)
    age = None
    if quote is not None and quote.ts.tzinfo is not None and now.tzinfo is not None:
        age = (now-quote.ts).total_seconds()
    row = dict(check=name, evaluated_at=now.isoformat(), measured=measured,
               threshold=threshold, units=units, comparison=comparison, passed=passed,
               data_window=window, quote_age_seconds=age,
               quote_timestamp=quote.ts.isoformat() if quote is not None else None,
               quote_age_status='available' if age is not None else 'not_applicable_or_unavailable',
               **details)
    if target is not None:
        row.update({k:v for k,v in target.get('entry_check_context',{}).items() if k not in row})
        rows = target.setdefault('entry_checks', [])
        if len(rows) < 256:
            rows.append(row)
        else:
            target['entry_checks_overflow'] = target.get('entry_checks_overflow', 0)+1
    for rows in _CAPTURES.get():
        rows.append(dict(row))
    # Every check is logged even when the bounded decision summary overflows.
    # Scoped final snapshots persist these rows once, without duplicate log lines.
    if not _CAPTURES.get():
        LOG.info('ENTRY_CHECK %s', json.dumps(row, allow_nan=False, default=str))
    return passed
