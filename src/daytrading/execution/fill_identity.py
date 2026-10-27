"""Broker cumulative observations and independent, fill-aware projections.

Order identity and cumulative quantity/notional survive every ingress path.
Timestamps are evidence, never broker execution identity. Legacy simulation
fills retain their original identity. Watermarks advance only after success.
"""
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from datetime import timezone
from functools import wraps
import math
import threading

_LOCK = threading.RLock()


class FillConflict(ValueError):
    """Contradictory broker evidence requires reconciliation, not a guess."""


def scoped_order_id(fill):
    scope = getattr(fill, 'account_id', '')
    return scope+'/'+fill.order_id if scope else fill.order_id


def projection_key(fill):
    return (fill.symbol, fill.side.value, scoped_order_id(fill))


def identity(fill):
    if getattr(fill, 'order_id', ''):
        return ('broker', fill.symbol, fill.side.value, scoped_order_id(fill),
                float(fill.cumulative_quantity), float(fill.cumulative_notional))
    stamp = fill.ts.astimezone(timezone.utc) if fill.ts.tzinfo is not None else fill.ts
    return (fill.symbol, fill.side.value, round(float(fill.quantity), 4),
            round(float(fill.price), 4), stamp.isoformat())


def observation(order, symbol, side, ts, account_id='', execution_reason=''):
    from daytrading.models import Fill
    qty = float(order.filled_qty or 0)
    price = float(order.filled_avg_price or 0)
    oid = str(order.id)
    if not oid or oid == 'None':
        raise FillConflict('Broker fill has no order identity')
    return Fill(symbol, side, qty, price, ts, order_id=oid, account_id=account_id, execution_reason=execution_reason,
                cumulative_quantity=qty,
                cumulative_notional=float(Decimal(str(qty))*Decimal(str(price))))


def delta(fill, state):
    if not getattr(fill, 'order_id', ''):
        return fill, None, None
    key = projection_key(fill)
    qty, cost = fill.cumulative_quantity, fill.cumulative_notional
    if qty is None or cost is None or not all(math.isfinite(x) and x > 0 for x in (qty, cost)):
        raise FillConflict('Invalid cumulative broker fill')
    old_qty, old_cost = state.get(key, (0., 0.))
    if qty < old_qty:
        if cost > old_cost + 1e-7:
            raise FillConflict('Regressed quantity with increased notional')
        return None, key, None  # Older, out-of-order observation.
    if qty == old_qty:
        if abs(cost-old_cost) > 1e-7:
            raise FillConflict('Cumulative fill price correction requires reconciliation')
        return None, key, None
    increment = float(Decimal(str(cost))-Decimal(str(old_cost)))
    if increment <= 0:
        raise FillConflict('Increasing quantity without increasing notional')
    return replace(fill, quantity=qty-old_qty, price=increment/(qty-old_qty)), key, (qty, cost)


@contextmanager
def project(owner, name, fill):
    """Serialize normalization and mutation; commit a watermark after success."""
    with _LOCK:
        states = getattr(owner, '_fill_projections', None)
        if states is None:
            states = {}; owner._fill_projections = states
        state = states.setdefault(name, {})
        try:
            incremental, key, watermark = delta(fill, state)
        except FillConflict:
            owner._fill_reconciliation_required = True
            raise
        yield incremental
        if watermark is not None:
            state[key] = watermark


def projected(name):
    """Decorator for methods whose first argument after self is a Fill."""
    def decorate(method):
        @wraps(method)
        def run(self, fill, *args, **kwargs):
            with project(self, name, fill) as incremental:
                if incremental is None:
                    return 0.0
                return method(self, incremental, *args, **kwargs)
        return run
    return decorate


def journal_identity(fill):
    if not getattr(fill, 'order_id', ''):
        return {}
    import hashlib, json
    return dict(order_id=fill.order_id,account_id=fill.account_id,
                cumulative_quantity=fill.cumulative_quantity,cumulative_notional=fill.cumulative_notional,
                execution_event_id=hashlib.sha256(json.dumps(identity(fill),separators=(',',':')).encode()).hexdigest())


def report_execution(journal, kind, payload, fill):
    """Canonical entry reporting, including replay-safe durable delivery."""
    from daytrading.reporting import report_safely
    payload = dict(payload, **journal_identity(fill))
    store = vars(journal).get('_execution_store')
    if store is not None and fill.order_id:
        engine = getattr(store.runner, '_session_engine', None)
        if engine is not None:
            engine.capture_decision(kind, payload, fill.ts)
        store.queue_report(payload['execution_event_id'], payload, fill.ts, event_type=kind)
        store.retry_reports()
    else:
        report_safely(journal.record, kind, payload, ts=fill.ts)
