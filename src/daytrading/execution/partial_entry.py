from daytrading.models import BarSource
"""First-partial entry protection using native fills, exits and cancellation.

No new entry rule or execution-price model. The native order poll pumps the
recorded market queue while a buy is pending; only confirmed incremental cost
is booked. A protective decision is latched until the remaining buy is settled.
"""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import logging
import math
import queue
import time

from daytrading.execution.broker import apply_fill
from daytrading.execution.fill_identity import projection_key
from daytrading.models import Bar, Fill, Order, OrderStatus, Side
from daytrading.reporting import report_safely

log = logging.getLogger(__name__)


@dataclass
class PendingEntry:
    signal: object
    order: Order
    bar: Bar
    quantity: float = 0.
    notional: float = 0.
    first_fill_ts: object = None
    last_fill_ts: object = None
    monitor_after: object = None
    order_id: str = ''
    cancel_for_protection: bool = False
    exit_reason: str = ''
    exit_price: float = 0.
    reported: bool = False
    active_submit: bool = True


def entries(runner):
    if not hasattr(runner, '_partial_entries'):
        runner._partial_entries = {}
    return runner._partial_entries


def submit(runner, signal, order, bar):
    pending = entries(runner)
    if order.symbol in pending:
        raise RuntimeError('Previous entry still unsettled: '+order.symbol)
    if order.symbol in runner._pipeline.portfolio.positions:
        raise RuntimeError('Initial partial-entry path requires a flat symbol')
    row = PendingEntry(signal, order, bar, monitor_after=runner._micro_market_now())
    runner._entry_reconcile_epoch = getattr(runner, "_entry_reconcile_epoch", 0)+1
    pending[order.symbol] = row
    try:
        fill, status = runner._broker.submit(order, bar, runner._pipeline.portfolio)
        # Immediate/simulated brokers may not have native progress callbacks.
        if fill is not None:
            observe(runner, row.order_id, order, fill)
        return fill, status
    finally:
        row.active_submit = False


def observe(runner, order_id, order, fill):
    row = entries(runner).get(order.symbol)
    if row is None or fill is None:
        return
    if row.order_id and row.order_id != str(order_id):
        # A deliberately retried unfilled intent can acquire its first ID.
        if row.quantity > 0:
            raise RuntimeError('Two entry orders for one pending position')
    row.order_id = str(order_id)
    row.last_fill_ts = fill.ts
    qty = float(fill.quantity)
    cost = Decimal(str(fill.quantity)) * Decimal(str(fill.price))
    old_cost = Decimal(str(row.notional))
    delta = qty-row.quantity
    if not math.isfinite(qty) or not cost.is_finite() or delta < 0 or qty > row.order.quantity or float(cost) <= 0:
        raise RuntimeError('Invalid cumulative entry fill')
    if delta == 0:
        # Corrected cumulative averages must adjust cost, never add shares.
        correction = float(cost-old_cost)
        if correction:
            position = runner._pipeline.portfolio.positions.get(order.symbol)
            if position is None:
                raise RuntimeError('Missing pending entry inventory')
            runner._pipeline.portfolio.cash -= correction
            position.avg_price = float(cost)/qty
            tracked = runner._pipeline.exit_manager._positions[order.symbol]
            tracked.entry_price = position.avg_price
            preview = deepcopy(runner._pipeline.exit_manager)
            preview.register_from_signal(row.signal, fill.ts, fill_price=fill.price,
                                         fill_quantity=row.order.quantity)
            tracked.stop_loss = max(tracked.stop_loss or 0., preview._positions[order.symbol].stop_loss or 0.)
            cap = float(getattr(runner._pipeline.exit_manager, '_max_unrealized_loss', 0) or 0)
            if cap > 0:
                tracked.stop_loss = max(tracked.stop_loss, fill.price-cap/qty)
            tracked.risk_per_share = max(0., fill.price-tracked.stop_loss)
            tracked.first_target_price = (float(row.signal.take_profit) if
                runner._pipeline.exit_manager._uses_signal_planned_first_target(row.signal) and
                row.signal.take_profit and row.signal.take_profit > fill.price else fill.price+tracked.risk_per_share)
            state=getattr(tracked,'micro_exit_state',None)
            if state is not None:
                state.synchronize_entry(fill.price,qty,observed=True)
                state.stop=max(state.stop,tracked.stop_loss)
                tracked.first_target_price=state.target
            row.notional = float(cost)
            if fill.order_id:
                states = runner._pipeline.portfolio.__dict__.setdefault('_fill_projections', {})
                states.setdefault('inventory', {})[projection_key(fill)] = (qty, float(cost))
                states.setdefault('cash', {})[projection_key(fill)] = (qty, float(cost))
        return
    incremental = Fill(fill.symbol, fill.side, delta, float((cost-old_cost)/Decimal(str(delta))),
                       fill.ts, commission=fill.commission, order_id=fill.order_id, account_id=fill.account_id,
                       cumulative_quantity=fill.cumulative_quantity, cumulative_notional=fill.cumulative_notional)
    if incremental.price <= 0:
        raise RuntimeError("Invalid incremental entry cost")
    manager = runner._pipeline.exit_manager
    previous = manager._positions.get(order.symbol)
    old_stop = previous.stop_loss if previous else None
    # Reuse the native stop calculation and its original full-order risk cap.
    # Then expose only actual inventory, not the requested buy quantity.
    preview = deepcopy(manager)
    preview.register_from_signal(row.signal, fill.ts, fill_price=fill.price,
                                 fill_quantity=row.order.quantity)
    candidate = preview._positions[order.symbol]
    apply_fill(runner._pipeline.portfolio, incremental)
    if previous is None:
        manager._positions[order.symbol] = candidate
        runner._pipeline._symbol_entry_counts[order.symbol] = runner._pipeline._symbol_entry_counts.get(order.symbol, 0)+1
        row.first_fill_ts = fill.ts
    tracked = manager._positions[order.symbol]
    tracked.entry_order_id = row.order_id
    tracked.stop_loss = max(tracked.stop_loss or 0., candidate.stop_loss or 0.)
    tracked.quantity = qty
    tracked.original_qty = qty
    tracked.remaining_qty = qty
    tracked.entry_price = fill.price
    # Never loosen the registered stop as late partials change the average.
    cap = float(getattr(manager, '_max_unrealized_loss', 0) or 0)
    if cap > 0:
        tracked.stop_loss = max(tracked.stop_loss or 0., fill.price-cap/qty)
    if old_stop is not None:
        tracked.stop_loss = max(old_stop, tracked.stop_loss or 0.)
    tracked.risk_per_share = max(0., fill.price-(tracked.stop_loss or fill.price))
    tracked.first_target_price = (float(row.signal.take_profit) if
        manager._uses_signal_planned_first_target(row.signal) and row.signal.take_profit and
        row.signal.take_profit > fill.price else fill.price+tracked.risk_per_share)
    state=getattr(tracked,'micro_exit_state',None)
    if state is not None:
        if previous is None:
            from daytrading.exits.coordinated_micro import MicroExitState
            state=MicroExitState.from_observed_fill(state.plan,fill.price,qty,fill.ts)
            tracked.micro_exit_state=state
        else:
            state.synchronize_entry(fill.price,qty,observed=True)
        state.stop=max(state.stop,tracked.stop_loss)
        tracked.first_target_price=state.target
    runner._pipeline._original_sizes[order.symbol] = qty
    row.quantity, row.notional = qty, float(cost)
    report_safely(runner._journal.record, 'entry_fill_progress', {
        'symbol': order.symbol, 'order_id': row.order_id, 'cumulative_quantity': qty,
        'cumulative_average': fill.price, 'delta_quantity': delta,
        'delta_price': incremental.price, 'stop': tracked.stop_loss,
        'protection': 'local pending-entry monitor; buy cancellation precedes exit',
    }, ts=fill.ts)
    log.info('FIRST/PARTIAL PROTECTION %s %.0f @ %.4f stop %.4f (buy pending)',
             order.symbol, qty, fill.price, tracked.stop_loss or 0.)


def drain_pending_events(runner):
    # A callback executes inside broker.submit's native operation. Record the
    # exact dequeued batch so the audit consumes the same receipt order and lag.
    from daytrading.replay.engine import ACTIVE, LiveOperation
    def collect():
        batch = []
        for _ in range(5000):
            try:
                event = runner._event_queue.get_nowait()
            except queue.Empty:
                break
            enqueued = runner._event_enqueued_at.pop(id(event), None)
            lag = max(0., time.monotonic()-enqueued) if enqueued is not None else 0.
            batch.append((event, lag))
        return batch
    context = ACTIVE.get()
    if isinstance(context, LiveOperation):
        batch = context.external('market_ingress', 'pending_entry_batch', (), {}, collect)
    elif context is not None:
        batch = context.external('market_ingress', 'pending_entry_batch', (), {})
    else:
        batch = collect()
    runner._event_entry_lag_sec = 0.
    for event, lag in batch:
        runner._event_entry_lag_sec = max(runner._event_entry_lag_sec, lag)
        runner._process_market_event(event)


def waiting(runner, symbol):
    row = entries(runner).get(symbol)
    if row is None:
        return False
    # This runs on the entry/main thread, never the stream callback thread.
    runner._drain_partial_entry_events()
    if row.quantity <= 0:
        return False
    now = runner._micro_market_now()
    if now is None:
        row.cancel_for_protection = True
        log.error("PARTIAL PROTECTION %s: clock unavailable; cancel remaining buy", symbol)
        return True
    prices = []
    for quote in runner._quote_buffer.get(symbol, ()):
        age = (now-quote.ts).total_seconds()
        if 0 <= age <= 10 and 0 < quote.bid < quote.ask:
            prices.append((quote.ts, quote.bid))
    for tick in runner._tick_buffer.get(symbol, ()):
        age = (now-tick.ts).total_seconds()
        if 0 <= age <= 10 and tick.price > 0:
            prices.append((tick.ts, tick.price))
    if not prices:
        row.cancel_for_protection = True
        log.error('PARTIAL PROTECTION %s: no fresh tape; cancel remaining buy, retain original stop', symbol)
        return True
    # Check every adverse observation, including a dip followed by a rebound.
    # The same pure dollar/technical-stop predicates are used by normal exits.
    for stamp, price in sorted(prices):
        if row.monitor_after and stamp < row.monitor_after:
            continue
        reason = runner._pipeline.exit_manager.pending_entry_protection_reason(symbol, price)
        if reason is not None:
            row.cancel_for_protection = True
            row.exit_reason, row.exit_price = reason.value, price
    return row.cancel_for_protection


def finish(runner, signal):
    row = entries(runner).get(signal.symbol)
    if row is None or row.active_submit:
        return
    if signal.symbol in getattr(runner._broker, '_unsettled_orders', {}):
        return  # Still monitored by native recovery; never sell against a live buy.
    if row.quantity and not row.reported:
        fill = Fill(signal.symbol, Side.BUY, row.quantity, row.notional/row.quantity,
                    row.last_fill_ts or datetime.now(timezone.utc), order_id=row.order_id,
                    account_id=getattr(runner._broker, '_account_scope', ''),
                    cumulative_quantity=row.quantity, cumulative_notional=row.notional)
        # Incremental inventory/counts are already idempotently committed. The
        # reporting path catches observational failures; a protection failure
        # retains this row for a main-thread finalization retry.
        runner._entry_executor.record_timed_fill(runner._entry_execution_context(), signal, fill, already_applied=True)
        row.reported = True
    entries(runner).pop(signal.symbol, None)
    runner._entry_reconcile_epoch = getattr(runner, "_entry_reconcile_epoch", 0)+1
    if row.quantity:
        runner._push_positions_from_alpaca()
    if row.exit_reason and row.quantity:
        pos = runner._pipeline.portfolio.positions.get(signal.symbol)
        if pos is None:
            return
        now = datetime.now(timezone.utc)
        bar = Bar(signal.symbol, now, row.exit_price, row.exit_price, row.exit_price, row.exit_price, 0, provenance=BarSource.PRICE_MARK)
        order = Order(signal.symbol, Side.SELL, pos.quantity, limit_price=row.exit_price)
        entry_price = pos.avg_price
        from daytrading.exits.manager import ExitIntent
        tracked = runner._pipeline.exit_manager.tracked.get(signal.symbol)
        if tracked is not None and tracked.exit_intent is None:
            tracked.exit_intent = ExitIntent(row.exit_reason, now, pos.quantity, True)
        fill, status = runner._submit_protective_exit_order(order, bar, 'FIRST PARTIAL STOP')
        if fill:
            apply_fill(runner._pipeline.portfolio, fill)
            runner._record_trade_exit(fill, entry_price, row.exit_reason,
                strategy=signal.scan_result.scanner_name if signal.scan_result else '')
            runner._seed_recent_order_ids()
            runner._push_positions_from_alpaca()
            runner._pipeline.set_cooldown(signal.symbol)
        else:
            log.critical('FIRST PARTIAL EXIT unresolved %s status=%s; retaining position and stop', signal.symbol, status)


def recovered(runner, symbol):
    row = entries(runner).get(symbol)
    if row is not None:
        finish(runner, row.signal)
