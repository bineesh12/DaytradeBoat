"""Reconcile failed stop intents against fresh broker facts before retrying.

The saved quantity is an intent, not evidence of remaining inventory. Unknown
requests, live buys, duplicate sells and failed reads keep the safety block.
"""
from dataclasses import dataclass
import math

TERMINAL = frozenset(('filled', 'canceled', 'cancelled', 'expired', 'rejected', 'replaced', 'done_for_day'))


def value(x):
    return getattr(x, 'value', x)


def number(x):
    n = float(x)
    if not math.isfinite(n):
        raise ValueError('Non-finite broker quantity or price')
    return n


def leaves(order):
    qty, filled = number(order.qty), number(getattr(order, 'filled_qty', 0) or 0)
    if not 0 <= filled <= qty:
        raise ValueError('Invalid broker fill quantity')
    return qty - filled


def is_stop(order, symbol):
    return (order.symbol == symbol and value(order.side) == 'sell'
            and value(order.type) == 'stop')


def unchanged(order, symbol, qty, stop):
    """Compare the actual broker-rounded request, not the unrounded stop."""
    return (is_stop(order, symbol) and value(order.status) in ('new', 'accepted')
            and number(getattr(order, 'filled_qty', 0) or 0) == 0
            and leaves(order) == int(qty) == qty
            and number(order.stop_price) == round(stop, 2))


@dataclass(frozen=True)
class Snapshot:
    quantity: float
    open_orders: tuple
    current_price: float = 0.0


def snapshot(broker, symbol):
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    positions = broker._client.get_all_positions()
    matches = [p for p in positions if p.symbol == symbol]
    if len(matches) > 1:
        raise ValueError('Duplicate broker positions')
    qty = number(matches[0].qty) if matches else 0.
    if qty < 0 or int(qty) != qty:
        raise ValueError('Unexpected short or fractional inventory')
    orders = broker._client.get_orders(filter=GetOrdersRequest(
        status=QueryOrderStatus.OPEN, symbols=[symbol], limit=500, nested=True))
    if len(orders) >= 500:
        raise ValueError('Open-order query may be truncated')
    if any(o.symbol != symbol for o in orders):
        raise ValueError('Unexpected symbol in broker order response')
    if any(value(o.status) in TERMINAL for o in orders):
        raise ValueError('Contradictory open-order response')
    raw_price = getattr(matches[0], 'current_price', None) if matches else None
    price = number(raw_price) if isinstance(raw_price, (str, int, float)) and raw_price else 0.0
    return Snapshot(qty, tuple(orders), price)


def terminal_fill_pending(broker, order):
    qty = number(getattr(order, 'filled_qty', 0) or 0)
    seen = broker._conflict_fill_seen.get(str(order.id), 0.)
    if qty > seen:
        # Use the existing fill-accounting workflow, never discard a late fill.
        broker._unsettled_orders[order.symbol] = (str(order.id), None)
        return True
    return False


def _known_requests(broker, symbol, old_id):
    orders = []
    if old_id:
        orders.append(broker._client.get_order_by_id(old_id))
    cid = broker._stop_request_ids.get(symbol)
    if cid:
        try:
            orders.append(broker._client.get_order_by_client_id(cid))
        except Exception as exc:
            if getattr(exc, 'status_code', None) != 404:
                raise
            # A timeout followed by a 404 is not a rejection acknowledgement.
            outcome = getattr(broker, '_stop_submit_outcomes', {}).get(symbol)
            if outcome != 'rejected':
                raise ValueError('Stop acceptance remains unknown') from exc
    unique = {str(o.id): o for o in orders}
    for original in list(unique.values()):
        current = original
        chain = {str(current.id)}
        while value(current.status) == 'replaced':
            successor = getattr(current, 'replaced_by', None)
            if not successor or str(successor) in chain or len(chain) >= 8:
                raise ValueError('Unresolved protective stop replacement chain')
            current = broker._client.get_order_by_id(str(successor))
            chain.add(str(current.id));unique[str(current.id)] = current
    for o in unique.values():
        if not is_stop(o, symbol):
            raise ValueError('Saved stop identity does not match broker order')
    return tuple(unique.values())


def order_signature(orders):
    return sorted((str(o.id), str(value(o.side)), str(value(o.type)), str(value(o.status)),
                   number(o.qty), number(getattr(o,'filled_qty',0) or 0),
                   str(getattr(o,'stop_price',None))) for o in orders)


def reconcile(broker, symbol, old_id, qty, stop, *, read_only=False, expected_exposure=False):
    """Clear only proven obsolete intents, otherwise protect fresh inventory."""
    first = snapshot(broker, symbol)
    known = _known_requests(broker, symbol, old_id)
    if any(terminal_fill_pending(broker, o) for o in known
           if value(o.status) in TERMINAL):
        return False
    if any(value(o.status) not in TERMINAL for o in known
           if str(o.id) not in {str(x.id) for x in first.open_orders}):
        return False
    # Refresh again after order lookup: a late fill can change remaining size.
    fresh = snapshot(broker, symbol)
    if (fresh.quantity != first.quantity or order_signature(fresh.open_orders) != order_signature(first.open_orders)):
        return False
    if any(value(o.status) in TERMINAL and str(o.id) in {str(x.id) for x in fresh.open_orders}
           for o in known):
        return False
    if fresh.quantity == 0:
        if expected_exposure and qty > 0:
            return False
        if fresh.open_orders:
            return False
        from .protection_status import update
        update(broker, symbol, 'flat')
        broker._failed_stop_updates.pop(symbol, None)
        broker._protective_stop_ids.pop(symbol, None)
        broker._stop_request_ids.pop(symbol, None)
        getattr(broker, '_stop_submit_outcomes', {}).pop(symbol, None)
        broker._invalidate_position_cache()
        return True
    if read_only:
        return False
    stops = [o for o in fresh.open_orders if is_stop(o, symbol)]
    if len(stops) != len(fresh.open_orders) or len(stops) > 1:
        return False  # Pending buys/sells or duplicate protection need reconciliation.
    owned = {str(o.id) for o in known}
    cached = broker._protective_stop_ids.get(symbol)
    if cached:
        owned.add(str(cached))
    owned.update(str(o.id) for o in broker._suspended_stops.get(symbol, []))
    if any(str(o.id) not in owned for o in stops):
        return False  # Matching economics alone do not prove order ownership.
    if stops and unchanged(stops[0], symbol, fresh.quantity, max(number(stops[0].stop_price), number(stop))):
        broker._protective_stop_ids[symbol] = str(stops[0].id)
        from .protection_status import acknowledge
        acknowledge(broker, symbol, stops[0], fresh.quantity, max(number(stops[0].stop_price), number(stop)))
        broker._failed_stop_updates.pop(symbol, None)
        broker._stop_request_ids.pop(symbol, None)
        getattr(broker, '_stop_submit_outcomes', {}).pop(symbol, None)
        return True
    if fresh.current_price > 0 and fresh.current_price <= round(number(stop), 2):
        # A stop above the market cannot be restored as a valid sell stop.
        # Hand off a persistent urgent exit; never lower the stop to fit price.
        from .protection_status import update
        update(broker, symbol, 'crossed_stop_liquidation', desired_quantity=fresh.quantity,
               desired_stop=number(stop), observed_market_price=fresh.current_price)
        handler = getattr(broker, '_protection_breached_handler', None)
        if handler is not None:
            handler(symbol, fresh.quantity, number(stop), fresh.current_price)
        return False
    if stops:
        current = stops[0]
        if number(getattr(current, 'filled_qty', 0) or 0) > 0:
            terminal_fill_pending(broker, current)
            return False
        if value(current.status) not in ('new', 'accepted'):
            return False
        # Never widen a stop while repairing quantity or a lost response.
        price = max(number(current.stop_price), number(stop))
        oid = broker.replace_protective_stop(symbol, str(current.id), fresh.quantity, price)
    else:
        oid = broker.place_protective_stop(symbol, fresh.quantity, number(stop))
    return bool(oid) and symbol not in broker._failed_stop_updates


def reconcile_legacy_flat(broker, symbol, *, expected_orders):
    """One-use migration for a legacy intent that did not record HTTP outcome.

    Requires the caller's freshly broker/journal-reconciled terminal order IDs,
    plus an independent fresh full ledger, two position/open-order reads and
    a definitive lookup of the retained request. No order operation is allowed.
    """
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    if getattr(broker, '_paper', False) is not True:
        raise ValueError('Legacy maintenance requires PAPER')
    if (broker._unsettled_orders or broker._suspended_stops or
            broker._entry_monitor_failed or broker._entry_progress_failed):
        raise ValueError('Other broker recovery is pending')
    intent = broker._failed_stop_updates.get(symbol)
    if intent is None:
        return True
    first = snapshot(broker, symbol)
    if first.quantity or first.open_orders:
        raise ValueError('Legacy recovery requires flat broker and no open orders')
    ledger = broker._client.get_orders(filter=GetOrdersRequest(
        status=QueryOrderStatus.ALL, symbols=[symbol], limit=500, nested=True))
    if (not ledger or len(ledger) >= 500 or
            {str(o.id) for o in ledger} != set(expected_orders) or
            any(o.symbol != symbol or value(o.status) not in TERMINAL for o in ledger)):
        raise ValueError('Legacy terminal ledger changed')
    buys = sum(number(o.filled_qty or 0) for o in ledger if value(o.side) == 'buy')
    sells = sum(number(o.filled_qty or 0) for o in ledger if value(o.side) == 'sell')
    if buys <= 0 or buys != sells:
        raise ValueError('Legacy broker ledger does not close exactly')
    old_id, _, _ = intent
    if old_id and old_id not in expected_orders:
        raise ValueError('Legacy stop missing from reconciled ledger')
    cid = broker._stop_request_ids.get(symbol)
    if cid:
        try:
            request = broker._client.get_order_by_client_id(cid)
        except Exception as exc:
            if getattr(exc, 'status_code', None) != 404:
                raise
        else:
            if str(request.id) not in expected_orders or value(request.status) not in TERMINAL:
                raise ValueError('Legacy request is unresolved')
    if snapshot(broker, symbol) != first:
        raise ValueError('Legacy flat state changed during reconciliation')
    # All accepted orders, including filled stops, were reconciled externally
    # against journal fills before granting this bounded maintenance operation.
    broker._failed_stop_updates.pop(symbol, None)
    broker._protective_stop_ids.pop(symbol, None)
    broker._stop_request_ids.pop(symbol, None)
    broker._invalidate_position_cache()
    return True
