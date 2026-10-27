"""Alpaca broker — submit real orders to Alpaca paper (or live) trading.

Uses alpaca-py SDK. Implements the same Broker protocol as PaperBroker
so it drops into the pipeline with zero changes.

Usage:
    from daytrading.execution.alpaca_broker import AlpacaBroker

    broker = AlpacaBroker(api_key="...", secret_key="...", paper=True)
    fill, status = broker.submit(order, bar, portfolio)
"""

from __future__ import annotations

from daytrading.execution.protection_status import update as protection_update, acknowledge as acknowledge_protection

import concurrent.futures
import logging
import time
import uuid
import math
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Tuple

from daytrading.execution.pricing import ExecutionPricing, marketable_limit
from daytrading.execution.transport import configure_client_timeouts

from daytrading.models import Bar, Fill, Order, OrderStatus, PortfolioState, Side

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
    _HAS_ALPACA = True
except ImportError:
    _HAS_ALPACA = False


class AlpacaBroker:
    """Submits orders to Alpaca Trading API (paper or live).

    Matches the ``Broker`` protocol:
        submit(order, bar, portfolio) → (Fill | None, OrderStatus)
    """

    requires_fresh_exit_quotes = True

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        paper: bool = True,
        max_wait_seconds: float = 5.0,
        poll_interval: float = 0.15,
        limit_buffer_pct: float = 0.005,
        slippage_guard: object = None,
    ) -> None:
        if not _HAS_ALPACA:
            raise ImportError(
                "alpaca-py is required. Install with: pip install 'daytrading[alpaca]'"
            )
        self._client = TradingClient(api_key, secret_key, paper=paper)
        configure_client_timeouts(self._client)
        self._paper = paper
        self._max_wait = max_wait_seconds
        self._poll_interval = poll_interval
        self._cancel_grace_seconds = 2.0
        self._marketable_limit_slippage_pct = 0.0075
        self._buy_limit_slippage_pct = 0.0075
        self._low_liquidity_buy_slippage_pct = 0.005
        self._momentum_buy_slippage_pct = 0.01
        self._sell_limit_slippage_pct = 0.0075
        self._limit_buffer_pct = limit_buffer_pct
        self._slippage_guard = slippage_guard
        logger.info("AlpacaBroker initialized (paper=%s)", paper)

    @property
    def client(self) -> TradingClient:
        return self._client

    def _pricing_policy(self) -> ExecutionPricing:
        return ExecutionPricing(
            market_slippage_pct=getattr(self, "_marketable_limit_slippage_pct", 0.0075),
            buy_slippage_pct=getattr(self, "_buy_limit_slippage_pct", 0.0075),
            thin_buy_slippage_pct=getattr(self, "_low_liquidity_buy_slippage_pct", 0.005),
            momentum_buy_slippage_pct=getattr(self, "_momentum_buy_slippage_pct", 0.01),
            sell_slippage_pct=getattr(self, "_sell_limit_slippage_pct", 0.0075),
        )

    def set_entry_deadline_clock(self, clock):
        self._entry_deadline_clock=clock

    def _original_deadline_now(self):
        clock=getattr(self,'_entry_deadline_clock',None)
        return clock() if clock is not None else getattr(self,'_now',lambda:datetime.now(timezone.utc))()

    def limit_price_for(self, order: Order, bar: Bar) -> float:
        """Shared request pricing for native paper and independent simulation."""
        if order.side is Side.BUY and getattr(order,'expires_at',None) is not None:
            now=self._original_deadline_now()
            expiry=order.expires_at
            if expiry.tzinfo is None or now is None or now.tzinfo is None or now>=expiry:
                raise ValueError('original micro setup deadline expired or unaligned')
        smart_price = None
        if self._slippage_guard and hasattr(self._slippage_guard, "get_limit_price"):
            if hasattr(self._slippage_guard, "quote_age_error"):
                stale = self._slippage_guard.quote_age_error(order.symbol)
                if stale:
                    raise ValueError(stale)
            raw_price = getattr(self._slippage_guard, "get_executable_price", None)
            if order.hard_limit_price is not None and callable(raw_price):
                smart_price = raw_price(order.symbol, order.side.value)
            else:
                # Preserve original pricing for ordinary requests and legacy
                # guard adapters that do not expose an actual quote side.
                smart_price = self._slippage_guard.get_limit_price(order.symbol, order.side.value)
        if order.side is Side.BUY and smart_price is None:
            if bar.ts is None or bar.ts.tzinfo is None:
                raise ValueError("fallback bar has no reliable timestamp")
            now = getattr(self, "_now", lambda: datetime.now(timezone.utc))()
            age = (now - bar.ts).total_seconds()
            if age > 75 or age < -5:
                raise ValueError("fallback execution bar is stale")
        return marketable_limit(order, bar, quote_price=smart_price, policy=self._pricing_policy())

    def submit(self, order: Order, bar: Bar, portfolio: PortfolioState) -> Tuple[Optional[Fill], OrderStatus]:
        if order.symbol != bar.symbol or not math.isfinite(order.quantity) or int(order.quantity) <= 0:
            return None, OrderStatus.REJECTED
        if order.side is Side.BUY and getattr(self, "_safety_blocked", False):
            logger.error("BUY blocked pending broker order/protection recovery: %s", order.symbol)
            return None, OrderStatus.NEW
        try:
            price = self.limit_price_for(order, bar)
            from daytrading.execution.origin import EXECUTION_SIGNAL, lineage
            signal = EXECUTION_SIGNAL.get()
            guard = getattr(self, '_strategy_admission', None)
            pos = portfolio.positions.get(order.symbol)
            qty = pos.quantity if pos is not None else 0
            reduces = ((order.side is Side.BUY and qty < 0 and order.quantity <= -qty)
                       or (order.side is Side.SELL and qty > 0 and order.quantity <= qty))
            if not reduces and guard is not None:
                reason = guard(signal, order, price)
                if reason:
                    from daytrading.execution.origin import entry_refused
                    entry_refused(reason)
                    logger.warning('ENTRY ROUTE REJECT %s: %s', order.symbol, reason)
                    return None, OrderStatus.REJECTED
            if not reduces and signal is not None:
                self.__dict__.setdefault('_order_lineage', {})[order.symbol] = lineage(signal)
            request = LimitOrderRequest(
                symbol=order.symbol, qty=int(order.quantity),
                side=OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL,
                time_in_force=TimeInForce.DAY, limit_price=price, extended_hours=True,
                client_order_id=uuid.uuid4().hex,
            )
        except Exception as exc:
            logger.error("ORDER price/request rejected %s: %s", order.symbol, exc)
            return None, OrderStatus.REJECTED
        return self._submit_protected(order, request)

    def submit_urgent_exit(self, order: Order, bar: Bar, portfolio: PortfolioState, *, slippage_pct: float = 0.05) -> Tuple[Optional[Fill], OrderStatus]:
        if not math.isfinite(order.quantity) or int(order.quantity) <= 0:
            return None, OrderStatus.REJECTED
        try:
            price = marketable_limit(order, bar, urgent_pct=slippage_pct)
            request = LimitOrderRequest(
                symbol=order.symbol, qty=int(order.quantity),
                side=OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL,
                time_in_force=TimeInForce.DAY, limit_price=price, extended_hours=True,
                client_order_id=uuid.uuid4().hex,
            )
        except Exception as exc:
            logger.error("URGENT EXIT invalid %s: %s", order.symbol, exc)
            return None, OrderStatus.REJECTED
        return self._submit_protected(order, request)

    def _checkpoint_request(self, kind, symbol, request_id):
        row = dict(kind=kind, symbol=symbol, request_id=request_id,
            **getattr(self, '_order_lineage', {}).get(symbol, dict(strategy='unknown_recovery',
                entry_reason='original entry reason unavailable', opportunity_id=None)))
        if kind == 'submit':
            from daytrading.execution.origin import EXECUTION_SIGNAL, lineage
            if EXECUTION_SIGNAL.get() is not None:
                row.update(lineage(EXECUTION_SIGNAL.get()))
        logger.info('ORDER lineage %s', row)
        audit = getattr(self, '_order_audit', None)
        if audit is not None:
            audit(dict(row))
        callback = getattr(self, '_durability_checkpoint', None)
        if callback is not None:
            callback(row)

    def _liquidation_quote_current(self, order):
        stamp = order.liquidation_quote_ts
        if stamp is None:
            return True  # Legacy adapters do not declare live liquidation evidence.
        now = getattr(self, '_now', lambda: datetime.now(timezone.utc))()
        return stamp.tzinfo is not None and 0 <= (now-stamp).total_seconds() <= 10

    def _buy_slippage_window_pct(self, bar: Bar) -> float:
        return self._pricing_policy().buy_window(bar)

    def _safety_state(self) -> None:
        if not hasattr(self, "_suspended_stops"):
            self._suspended_stops = {}
            self._unsettled_orders = {}
            self._protective_stop_ids = {}
            self._failed_stop_updates = {}
            self._stop_request_ids = {}
            self._conflict_fill_seen = {}
            self._entry_order_originals = {}
            self._entry_monitor_failed = set()
            self._entry_progress_failed = set()
            self._safety_blocked = False
        if not hasattr(self, "_stop_submit_outcomes"):
            self._stop_submit_outcomes = {}
        if not hasattr(self, '_unconfirmed_cancels'):
            self._unconfirmed_cancels = set()
        if not hasattr(self, '_conflict_fill_inflight'):
            self._conflict_fill_inflight = set()

    def _submit_protected(self, order: Order, request: object) -> Tuple[Optional[Fill], OrderStatus]:
        started = time.monotonic()
        try:
            return self._submit_protected_impl(order, request)
        finally:
            self._last_order_latency_sec = time.monotonic() - started

    def _submit_protected_impl(self, order: Order, request: object) -> Tuple[Optional[Fill], OrderStatus]:
        self._safety_state()
        self._conflict_fill_during_cancel = False
        if self._unconfirmed_cancels:
            self.recover_order_safety()
            if self._unconfirmed_cancels:
                return None, OrderStatus.NEW
        if order.symbol in self._unsettled_orders:
            self.recover_order_safety()
            if order.symbol in self._unsettled_orders:
                return None, OrderStatus.NEW
        try:
            self._cancel_open_orders_for(order.symbol)
        except Exception as exc:
            self._safety_blocked = True
            logger.error("Cannot confirm cancellation for %s: %s", order.symbol, exc)
            return None, OrderStatus.NEW
        blocker = getattr(self, "_entry_blocker", None)
        if order.side is Side.BUY and (
            self._conflict_fill_during_cancel or (blocker is not None and blocker(order.symbol))
        ):
            stops = self._suspended_stops.get(order.symbol, [])
            remaining = max((float(s.qty) - float(getattr(s, "filled_qty", 0) or 0) for s in stops), default=0.0)
            self._restore_protection(order.symbol, expected_quantity=remaining)
            logger.warning("Queued buy cancelled after stop execution or final entry block: %s", order.symbol)
            return None, OrderStatus.REJECTED
        if order.side is Side.BUY:
            from daytrading.execution.origin import EXECUTION_SIGNAL
            from daytrading.models import SignalAction
            guard = getattr(self, '_strategy_admission', None)
            if (EXECUTION_SIGNAL.get() is not None and guard is not None
                    and EXECUTION_SIGNAL.get().action is SignalAction.ENTER_LONG):
                reason = guard(EXECUTION_SIGNAL.get(), order, float(request.limit_price))
                if reason:
                    from daytrading.execution.origin import entry_refused
                    entry_refused(reason)
                    self._restore_protection(order.symbol)
                    logger.warning('Entry invalidated during cancellation for %s: %s', order.symbol, reason)
                    return None, OrderStatus.REJECTED
            self._entry_order_originals[order.symbol] = order
        if not self._liquidation_quote_current(order):
            self._restore_protection(order.symbol)
            logger.warning('Exit intent retained: quote aged during cancellation for %s', order.symbol)
            return None, OrderStatus.REJECTED
        self._unsettled_orders[order.symbol] = (None, request.client_order_id)
        self._checkpoint_request('submit', order.symbol, request.client_order_id)
        # Cancellation and the durable checkpoint can both consume the original
        # freshness budget. Revalidate after their I/O, immediately before SDK.
        if order.side is Side.BUY:
            from daytrading.execution.origin import EXECUTION_SIGNAL
            from daytrading.models import SignalAction
            guard = getattr(self, '_strategy_admission', None)
            if (EXECUTION_SIGNAL.get() is not None and guard is not None
                    and EXECUTION_SIGNAL.get().action is SignalAction.ENTER_LONG):
                from daytrading.execution.origin import EXECUTION_REQUEST_ID
                token = EXECUTION_REQUEST_ID.set(request.client_order_id)
                try:
                    reason = guard(EXECUTION_SIGNAL.get(), order, float(request.limit_price))
                finally:
                    EXECUTION_REQUEST_ID.reset(token)
                if reason:
                    from daytrading.execution.origin import entry_refused
                    entry_refused(reason)
                    self._unsettled_orders.pop(order.symbol, None)
                    self._entry_order_originals.pop(order.symbol, None)
                    self._restore_protection(order.symbol)
                    logger.warning('Entry invalidated during checkpoint for %s: %s', order.symbol, reason)
                    return None, OrderStatus.REJECTED
        try:
            if order.side is Side.BUY:
                from daytrading.execution.origin import entry_sdk_started
                entry_sdk_started(order.symbol,request.client_order_id)
            placed = self._client.submit_order(order_data=request)
        except Exception as exc:
            # A lost HTTP response does not establish that the order was rejected.
            status_code = getattr(exc, "status_code", None)
            if isinstance(status_code, int) and 400 <= status_code < 500 and status_code not in (408, 429):
                self._unsettled_orders.pop(order.symbol, None)
                stops = self._suspended_stops.get(order.symbol, [])
                remaining = max((float(s.qty) - float(getattr(s, "filled_qty", 0) or 0) for s in stops), default=0.0)
                self._restore_protection(order.symbol, expected_quantity=remaining)
                self._entry_order_originals.pop(order.symbol, None)
                return None, OrderStatus.REJECTED
            self._unsettled_orders[order.symbol] = (None, request.client_order_id)
            self._safety_blocked = True
            logger.error("Order acceptance unknown for %s; entries blocked: %s", order.symbol, exc)
            return None, OrderStatus.NEW
        self._unsettled_orders[order.symbol] = (str(placed.id), request.client_order_id)
        if order.side is Side.SELL:
            protection_update(self, order.symbol, 'liquidating', liquidation_order_id=str(placed.id),
                              liquidation_status=str(getattr(placed.status, 'value', placed.status)) if hasattr(placed, 'status') else 'unknown')
        self._observe_entry_progress(placed.id, order, placed)
        fill, status = self._wait_for_fill(placed.id, order)
        if order.symbol in self._entry_monitor_failed or order.symbol in self._entry_progress_failed:
            self._safety_blocked = True
            return None, OrderStatus.NEW
        if status is not OrderStatus.NEW:
            self._unsettled_orders.pop(order.symbol, None)
            self._entry_order_originals.pop(order.symbol, None)
            stops = self._suspended_stops.get(order.symbol, [])
            previous = max((float(s.qty) - float(getattr(s, "filled_qty", 0) or 0) for s in stops), default=0.0)
            executed = fill.quantity if fill else 0.0
            remaining = previous + executed if order.side is Side.BUY else max(0.0, previous - executed)
            self._restore_protection(order.symbol, expected_quantity=remaining)
        else:
            self._safety_blocked = True
        return fill, status

    def _restore_protection(self, symbol: str, *, expected_quantity: Optional[float] = None) -> bool:
        self._safety_state()
        stops = self._suspended_stops.get(symbol, [])
        if not stops:
            return True
        try:
            from daytrading.execution.stop_recovery import reconcile
            stop_price = max(float(s.stop_price) for s in stops)
            # expected_quantity is only a request-time hint. Every restoration
            # uses the same fresh positions/open-orders reconciliation.
            old_id = str(stops[-1].id)
            self._failed_stop_updates.setdefault(symbol, (old_id, expected_quantity or 0., stop_price))
            if not reconcile(self, symbol, old_id, expected_quantity or 0., stop_price, expected_exposure=True):
                self._safety_blocked = True
                return False
            self._suspended_stops.pop(symbol, None)
            return True
        except Exception as exc:
            self._safety_blocked = True
            logger.error("PROTECTION recovery pending for %s: %s", symbol, exc)
            return False

    def recover_order_safety(self) -> bool:
        """Confirm unknown orders and restore suspended stops before admitting buys."""
        self._safety_state()
        try:
            for symbol, row in getattr(self, '_protection_states', {}).items():
                if row.get('requires_revalidation') and row.get('desired_quantity', 0) > 0:
                    self._failed_stop_updates.setdefault(symbol,
                        (row.get('order_id'), row['desired_quantity'], row['desired_stop']))
            for order_id in list(self._unconfirmed_cancels):
                self.cancel_order_by_id(order_id)
            for symbol, (order_id, client_id) in list(self._unsettled_orders.items()):
                order = (self._client.get_order_by_id(order_id) if order_id else
                         self._client.get_order_by_client_id(client_id))
                original = getattr(self, "_entry_order_originals", {}).get(symbol)
                if original is not None:
                    self._observe_entry_progress(order.id, original, order)
                    self._protect_entry_wait(original)
                status = getattr(order.status, "value", order.status)
                if status not in ("filled", "canceled", "cancelled", "expired", "rejected"):
                    self._client.cancel_order_by_id(order.id)
                    continue
                if original is not None and status != "filled":
                    settled = self._wait_for_cancel_fill(order.id, original)
                    if self._cancel_unconfirmed:
                        continue
                if symbol in self._entry_progress_failed or symbol in self._entry_monitor_failed:
                    continue
                kind = getattr(order, "type", "")
                if str(getattr(getattr(order, 'side', None), 'value', getattr(order, 'side', None))) == 'sell':
                    self._emit_conflicting_stop_fill(order)
                self._unsettled_orders.pop(symbol, None)
                self._entry_order_originals.pop(symbol, None)
                self._invalidate_position_cache()
                if original is not None:
                    handler = getattr(self, "_entry_recovered_handler", None)
                    if handler is not None:
                        handler(symbol)
            for symbol in list(self._suspended_stops):
                if symbol not in self._unsettled_orders:
                    self._restore_protection(symbol)
            for symbol, (old_id, qty, stop) in list(self._failed_stop_updates.items()):
                if symbol in self._suspended_stops:
                    continue
                from daytrading.execution.stop_recovery import reconcile
                reconcile(self, symbol, old_id, qty, stop)
        except Exception as exc:
            logger.warning("Broker recovery still pending: %s", exc)
        self._safety_blocked = bool(self._unconfirmed_cancels or self._unsettled_orders or self._suspended_stops or self._failed_stop_updates or self._entry_monitor_failed or self._entry_progress_failed)
        return not self._safety_blocked

    def reconcile_protective_stop(self, symbol, qty, stop, *, known_id=None):
        self._safety_state()
        old_id = self._protective_stop_ids.get(symbol) or known_id
        self._failed_stop_updates[symbol] = (old_id, qty, stop)
        from daytrading.execution.stop_recovery import reconcile
        try:
            if reconcile(self, symbol, old_id, qty, stop, expected_exposure=True):
                return self._protective_stop_ids.get(symbol)
        except Exception:
            logger.exception('Protection reconciliation pending for %s', symbol)
        self._safety_blocked = True
        return None

    def protection_status(self):
        from daytrading.execution.protection_status import snapshot
        return snapshot(self, getattr(self, '_now', lambda: datetime.now(timezone.utc))())

    def get_protective_stop_id(self, symbol: str) -> Optional[str]:
        return getattr(self, "_protective_stop_ids", {}).get(symbol)

    def set_execution_callbacks(self, conflict_fill, entry_blocked) -> None:
        self._conflict_fill_handler = conflict_fill
        self._entry_blocker = entry_blocked

    def set_entry_lifecycle_callbacks(self, progress, waiting, recovered) -> None:
        self._entry_progress_handler = progress
        self._entry_wait_handler = waiting
        self._entry_recovered_handler = recovered

    def _observe_entry_progress(self, order_id, original, observed):
        if original.side is not Side.BUY:
            return
        handler = getattr(self, "_entry_progress_handler", None)
        if handler is not None:
            try:
                handler(str(order_id), original, self._fill_from_alpaca_order(observed, original))
                self._entry_progress_failed.discard(original.symbol)
            except Exception:
                self._entry_progress_failed.add(original.symbol)
                self._safety_blocked = True
                logger.exception("CRITICAL: partial-entry accounting failed for %s; cancel and reconcile", original.symbol)

    def _protect_entry_wait(self, original):
        handler = getattr(self, "_entry_wait_handler", None)
        if original.side is not Side.BUY:
            return False
        try:
            result = bool(handler(original.symbol)) if handler else False
            if handler is not None:
                self._entry_monitor_failed.discard(original.symbol)
            return result or original.symbol in self._entry_monitor_failed or original.symbol in self._entry_progress_failed
        except Exception:
            self._entry_monitor_failed.add(original.symbol)
            self._safety_blocked = True
            logger.exception("CRITICAL: partial-entry monitor failed for %s; cancel remaining buy", original.symbol)
            return True

    def _emit_conflicting_stop_fill(self, order) -> None:
        quantity = float(getattr(order, "filled_qty", 0) or 0)
        if quantity > 0:
            self._conflict_fill_during_cancel = True
        handler = getattr(self, "_conflict_fill_handler", None)
        if handler is None:
            return
        order_id = str(order.id)
        delta = quantity - self._conflict_fill_seen.get(order_id, 0.0)
        price = float(getattr(order, "filled_avg_price", 0) or 0)
        if delta <= 0 or price <= 0:
            return
        from daytrading.execution.fill_identity import observation
        # Consumers each project the same cumulative fact independently.
        fill = observation(order, order.symbol, Side.SELL,
                           getattr(order, "filled_at", None) or datetime.now(timezone.utc), account_id=getattr(self, '_account_scope', ''),
                           execution_reason='broker_stop' if 'stop' in str(getattr(getattr(order, 'type', ''), 'value', getattr(order, 'type', ''))) else '')
        remaining = max(0.0, float(order.qty) - quantity)
        protection_update(self, order.symbol, 'fill_observed', filled_quantity=quantity, order_id=order_id)
        if order_id in self._conflict_fill_inflight:
            return
        self._conflict_fill_inflight.add(order_id)
        try:
            handler(fill, order_id, remaining)
            self._conflict_fill_seen[order_id] = quantity
        finally:
            self._conflict_fill_inflight.discard(order_id)

    def _wait_for_fill(
        self,
        order_id: str,
        original: Order,
    ) -> Tuple[Optional[Fill], OrderStatus]:
        """Poll Alpaca until the order fills, is rejected, or times out."""
        self._safety_state()
        deadline = time.monotonic() + self._max_wait
        if original.side is Side.BUY and getattr(original,'expires_at',None) is not None:
            now=self._original_deadline_now()
            expiry=original.expires_at
            remaining=(expiry-now).total_seconds() if expiry.tzinfo is not None and now is not None and now.tzinfo is not None else 0.
            deadline=min(deadline,time.monotonic()+max(0.,remaining))

        while time.monotonic() < deadline:
            try:
                alpaca_order = self._client.get_order_by_id(order_id)
            except Exception as exc:
                logger.error("Error checking order %s: %s", order_id, exc)
                self._safety_blocked = True
                return None, OrderStatus.NEW

            self._observe_entry_progress(order_id, original, alpaca_order)
            protect_cancel = self._protect_entry_wait(original)
            status_val = alpaca_order.status.value if hasattr(alpaca_order.status, 'value') else str(alpaca_order.status)

            if status_val == "filled":
                fill = self._fill_from_alpaca_order(alpaca_order, original)
                if fill:
                    self._invalidate_position_cache()
                    logger.info(
                        "ORDER FILLED %s %s %.0f @ %.4f",
                        original.side.value, original.symbol, fill.quantity, fill.price,
                    )
                    self._record_slippage(original, fill)
                    return fill, OrderStatus.FILLED

            if status_val in ("canceled", "cancelled", "expired", "rejected"):
                if original.side is Side.BUY and status_val != "rejected":
                    fill = self._wait_for_cancel_fill(order_id, original)
                    if self._cancel_unconfirmed:
                        self._safety_blocked = True
                        return None, OrderStatus.NEW
                else:
                    fill = self._fill_from_alpaca_order(alpaca_order, original)
                if fill:
                    self._invalidate_position_cache()
                    logger.info(
                        "ORDER %s WITH FILL %s %s %.0f @ %.4f",
                        status_val.upper(), original.side.value, original.symbol,
                        fill.quantity, fill.price,
                    )
                    self._record_slippage(original, fill)
                    return fill, OrderStatus.FILLED
                logger.warning(
                    "ORDER %s %s %s", status_val.upper(), original.symbol, order_id,
                )
                return None, OrderStatus.REJECTED

            if protect_cancel:
                logger.warning("ENTRY PROTECTION %s: cancel remaining buy before exit", original.symbol)
                break
            if original.side is Side.BUY and getattr(original,'expires_at',None) is not None:
                time.sleep(min(self._poll_interval,max(0.,deadline-time.monotonic())))
            else:
                time.sleep(self._poll_interval)

        logger.warning(
            "ORDER TIMEOUT %s %s after %.1fs — cancelling",
            original.symbol, order_id, self._max_wait,
        )
        try:
            self._client.cancel_order_by_id(order_id)
        except Exception:
            pass

        fill = self._wait_for_cancel_fill(order_id, original)
        if fill:
            return fill, OrderStatus.FILLED

        if getattr(self, "_cancel_unconfirmed", False):
            self._safety_blocked = True
            return None, OrderStatus.NEW
        return None, OrderStatus.CANCELLED

    def _fill_from_alpaca_order(self, alpaca_order: object, original: Order) -> Optional[Fill]:
        """Build a fill when Alpaca reports any executed quantity."""
        try:
            fill_qty = float(getattr(alpaca_order, "filled_qty", 0) or 0)
            fill_price = float(getattr(alpaca_order, "filled_avg_price", 0) or 0)
        except (TypeError, ValueError):
            return None
        if fill_qty <= 0 or fill_price <= 0:
            return None
        fill_ts = getattr(alpaca_order, "filled_at", None) or datetime.now(timezone.utc)
        from daytrading.execution.fill_identity import observation
        return observation(alpaca_order, original.symbol, original.side, fill_ts, account_id=getattr(self, '_account_scope', ''))

    def _record_slippage(self, original: Order, fill: Fill) -> None:
        if self._slippage_guard and hasattr(self._slippage_guard, "record_fill"):
            expected = original.limit_price or fill.price
            self._slippage_guard.record_fill(original.symbol, expected, fill.price)

    def _wait_for_cancel_fill(self, order_id: str, original: Order) -> Optional[Fill]:
        """After cancel, give Alpaca time to publish late partial fills.

        Fast-moving names can fill while the cancel is in flight. Returning a
        fill here prevents the runner from treating the entry as missed and then
        discovering an orphan position later through reconciliation.
        """
        deadline = time.monotonic() + float(getattr(self, "_cancel_grace_seconds", 2.0))
        last_fill: Optional[Fill] = None
        self._cancel_unconfirmed = True
        terminal_signature = None
        while time.monotonic() < deadline:
            try:
                final_order = self._client.get_order_by_id(order_id)
            except Exception:
                break
            self._observe_entry_progress(order_id, original, final_order)
            self._protect_entry_wait(original)
            fill = self._fill_from_alpaca_order(final_order, original)
            if fill:
                last_fill = fill
            status = getattr(final_order, "status", "")
            status_val = status.value if hasattr(status, "value") else str(status)
            if status_val in ("filled", "canceled", "cancelled", "expired", "rejected"):
                signature = (status_val, float(getattr(final_order, "filled_qty", 0) or 0),
                             float(getattr(final_order, "filled_avg_price", 0) or 0))
                if status_val == "filled" or signature == terminal_signature:
                    self._cancel_unconfirmed = False
                    break
                terminal_signature = signature
            else:
                terminal_signature = None
            time.sleep(self._poll_interval)

        if self._cancel_unconfirmed:
            return None
        if last_fill:
            self._invalidate_position_cache()
            logger.info(
                "PARTIAL FILL %s %s %.0f of %.0f @ %.4f (cancel confirmed/late fill)",
                original.side.value, original.symbol,
                last_fill.quantity, original.quantity, last_fill.price,
            )
            self._record_slippage(original, last_fill)
        return last_fill

    def _open_order_count_for(self, symbol: str) -> int:
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            return len(self._client.get_orders(filter=req) or [])
        except Exception:
            return 0

    def _invalidate_position_cache(self) -> None:
        self._pos_cache = None
        self._position_generation = getattr(self, '_position_generation', 0) + 1
        self._position_sync_stale = True

    def bind_position_refresh_owner(self, context, wakeup=None) -> None:
        """Only the decision thread may adopt SDK results into entry state."""
        owner = getattr(self, '_position_owner', None)
        if owner is not None and owner != threading.get_ident():
            raise RuntimeError('position refresh owner thread mismatch')
        self._position_owner = threading.get_ident()
        previous = getattr(self, '_position_context', context)
        if previous != context:
            self._invalidate_position_cache()
        self._position_context = context
        self._position_wakeup = wakeup

    def acknowledge_position_context(self, context):
        if (getattr(self, '_position_owner', None) == threading.get_ident()
                and not self.position_refresh_pending() and not self._position_sync_stale):
            # Reconciliation itself changed local state to this accepted snapshot.
            self._position_context = context

    def _position_refresh_note(self, state, **fields):
        events = self.__dict__.setdefault('_position_refresh_events', deque(maxlen=128))
        row = dict(state=state, at=datetime.now(timezone.utc).isoformat(), **fields)
        events.append(row)
        logger.info('position_refresh %s', row)

    def pop_position_refresh_events(self):
        events = self.__dict__.setdefault('_position_refresh_events', deque(maxlen=128))
        rows = list(events)
        events.clear()
        return rows

    def position_refresh_pending(self):
        return getattr(self, '_position_future', None) is not None

    def position_snapshot_current(self):
        """Read-only owner predicate; never request or adopt broker I/O."""
        if (getattr(self, '_position_owner', None) != threading.get_ident()
                or self.position_refresh_pending()
                or getattr(self, '_position_sync_stale', True) is not False
                or getattr(self, '_pos_cache', None) is None):
            return False
        age = time.monotonic() - getattr(self, '_pos_cache_ts', float('-inf'))
        return 0 <= age < 0.5  # The existing position-cache/adoption TTL.

    def adopt_completed_positions(self) -> bool:
        """Nonblocking owner-thread adoption; never starts a network request."""
        owner = getattr(self, '_position_owner', threading.get_ident())
        if owner != threading.get_ident():
            return False
        future = getattr(self, '_position_future', None)
        if future is None or not future.done():
            return False
        meta = getattr(future, '_daytrade_position_meta', {})
        if meta and 'completed_monotonic' not in meta:
            return False  # Future.done() can race its completion callback.
        self._position_future = None  # Exactly one adoption, including duplicate wakeups.
        if not meta:  # Legacy/handover future: request version cannot be proven.
            self._position_sync_stale = True
            self._position_refresh_note('discarded', reason='missing_request_version')
            return False
        fields = {k: meta.get(k) for k in ('request_id', 'requested_at', 'completed_at', 'generation')}
        self._position_sync_stale = True
        try:
            positions = future.result()  # done() above: no wait on the SDK.
            age = time.monotonic() - meta['completed_monotonic']
            if (meta['generation'] != getattr(self, '_position_generation', 0)
                    or meta['context'] != getattr(self, '_position_context', None)):
                self._position_refresh_note('discarded', reason='position_version_changed', **fields)
                return False
            # Cache TTL starts at completion, never at delayed adoption.
            if not 0 <= age < 0.5:
                self._position_refresh_note('discarded', reason='snapshot_age', age_seconds=age, **fields)
                return False
            result = {p.symbol: {
                'qty': float(p.qty), 'avg_entry': float(p.avg_entry_price),
                'current_price': float(p.current_price or p.avg_entry_price),
                'market_value': float(p.market_value), 'unrealized_pl': float(p.unrealized_pl),
                'side': p.side,
            } for p in positions}
            if any(not math.isfinite(v) for row in result.values() for k, v in row.items() if k != 'side'):
                raise ValueError('nonfinite position snapshot')
        except Exception as exc:
            self._position_refresh_note('failed', reason=type(exc).__name__, **fields)
            return False
        self._pos_cache = result
        self._pos_cache_ts = meta['completed_monotonic']
        self._position_sync_stale = False
        self._position_refresh_note('adopted', quantity=len(result), age_seconds=age, **fields)
        return True

    def _cancel_open_orders_for(self, symbol: str, *, preserve_stops: bool = False) -> None:
        """Confirm cancellation and remember protection before submitting a conflict."""
        self._safety_state()
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        orders = self._client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
        for order in orders:
            kind = getattr(order, "type", None) or getattr(order, "order_type", "")
            is_stop = "stop" in str(getattr(kind, "value", kind)).lower()
            if is_stop and preserve_stops:
                continue
            if is_stop:
                # Preserve the snapshot even if cancellation acknowledgement is lost.
                stops = self._suspended_stops.setdefault(symbol, [])
                if not any(str(s.id) == str(order.id) for s in stops):
                    stops.append(order)
            self._unsettled_orders[symbol] = (str(order.id), None)
            if is_stop:
                protection_update(self, symbol, 'cancel_requested', order_id=str(order.id))
            self._checkpoint_request('cancel', symbol, str(order.id))
            self._client.cancel_order_by_id(str(order.id))
            deadline = time.monotonic() + float(getattr(self, "_cancel_grace_seconds", 2.0))
            while True:
                current = self._client.get_order_by_id(str(order.id))
                status = getattr(current.status, "value", current.status)
                if status in ("canceled", "cancelled", "expired", "rejected"):
                    if str(getattr(current.side, 'value', current.side)) == 'sell':
                        self._emit_conflicting_stop_fill(current)
                    if is_stop:
                        protection_update(self, symbol, 'absent', order_id=str(order.id), covered_quantity=0)
                        if self._protective_stop_ids.get(symbol) == str(order.id):
                            self._protective_stop_ids.pop(symbol, None)
                        stops = self._suspended_stops[symbol]
                        self._suspended_stops[symbol] = [current if str(s.id) == str(order.id) else s for s in stops]
                    self._unsettled_orders.pop(symbol, None)
                    break
                if status == "filled":
                    if str(getattr(current.side, 'value', current.side)) == 'sell':
                        self._emit_conflicting_stop_fill(current)
                    self._invalidate_position_cache()
                    raise RuntimeError("conflicting order filled during cancellation; reconcile first")
                if time.monotonic() >= deadline:
                    self._unsettled_orders[symbol] = (str(order.id), None)
                    raise RuntimeError("conflicting order cancellation unconfirmed")
                time.sleep(self._poll_interval)
        self._invalidate_position_cache()

    def place_protective_stop(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
    ) -> Optional[str]:
        """Place a broker-held stop-loss sell (survives bot restarts / sync gaps)."""
        if qty <= 0 or stop_price <= 0:
            return None
        self._safety_state()
        try:
            from alpaca.trading.requests import StopOrderRequest
            prior_request = self._stop_request_ids.get(symbol)
            client_id = self._stop_request_ids.setdefault(symbol, uuid.uuid4().hex)
            # Reuse one idempotency ID across a lost-response retry.
            if prior_request and symbol in self._failed_stop_updates:
                try:
                    existing = self._client.get_order_by_client_id(client_id)
                except Exception as exc:
                    if (getattr(exc, "status_code", None) != 404 or
                            self._stop_submit_outcomes.get(symbol) != 'rejected'):
                        raise
                else:
                    status = getattr(existing.status, "value", existing.status)
                    if status not in ("filled", "canceled", "cancelled", "expired", "rejected"):
                        from daytrading.execution.stop_recovery import unchanged
                        if not unchanged(existing, symbol, qty, stop_price):
                            self._safety_blocked = True
                            return None
                        oid = str(existing.id)
                        self._protective_stop_ids[symbol] = oid
                        self._failed_stop_updates.pop(symbol, None)
                        self._stop_request_ids.pop(symbol, None)
                        acknowledge_protection(self, symbol, existing, qty, stop_price)
                        return oid
                    if status == "filled":
                        positions = self._client.get_all_positions()
                        qty = next((float(p.qty) for p in positions if p.symbol == symbol), 0.0)
                        if qty <= 0:
                            self._failed_stop_updates.pop(symbol, None)
                            self._protective_stop_ids.pop(symbol, None)
                            self._stop_request_ids.pop(symbol, None)
                            self._invalidate_position_cache()
                            return str(existing.id)
                    self._stop_request_ids[symbol] = client_id = uuid.uuid4().hex
            request = StopOrderRequest(
                symbol=symbol,
                qty=int(qty),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=round(stop_price, 2),
                client_order_id=client_id,
            )
            self._failed_stop_updates[symbol] = (None, qty, stop_price)
            self._stop_submit_outcomes[symbol] = 'unknown'
            protection_update(self, symbol, 'submitting', desired_quantity=qty, desired_stop=stop_price, client_order_id=client_id)
            self._checkpoint_request('protective_stop', symbol, client_id)
            alpaca_order = self._client.submit_order(order_data=request)
            self._invalidate_position_cache()
            oid = str(alpaca_order.id)
            self._safety_state()
            self._protective_stop_ids[symbol] = oid
            if not acknowledge_protection(self, symbol, alpaca_order, qty, stop_price):
                self._failed_stop_updates[symbol] = (oid, qty, stop_price)
                self._safety_blocked = True
                return None
            self._failed_stop_updates.pop(symbol, None)
            self._stop_request_ids.pop(symbol, None)
            self._stop_submit_outcomes.pop(symbol, None)
            logger.info(
                "BROKER STOP %s sell %.0f @ stop $%.2f → id=%s",
                symbol, qty, stop_price, oid,
            )
            return oid
        except Exception as exc:
            self._failed_stop_updates[symbol] = (None, qty, stop_price)
            code = getattr(exc, "status_code", None)
            self._stop_submit_outcomes[symbol] = (
                "rejected" if isinstance(code, int) and 400 <= code < 500
                and code not in (408, 429) else "unknown")
            self._safety_blocked = True
            logger.error("BROKER STOP failed %s: %s", symbol, exc)
            return None

    def cancel_order_by_id(self, order_id: str) -> bool:
        """Retain cancellation identity until terminal acknowledgement/fills."""
        self._safety_state()
        self._unconfirmed_cancels.add(str(order_id))
        try:
            order = self._client.get_order_by_id(order_id)
            symbol = order.symbol
            self._unsettled_orders[symbol] = (str(order_id), None)
            self._checkpoint_request('cancel', symbol, str(order_id))
            status = str(getattr(order.status, 'value', order.status))
            terminal = ('filled','canceled','cancelled','expired','rejected','replaced','done_for_day')
            if status not in terminal:
                protection_update(self, symbol, 'cancel_requested', order_id=str(order_id))
                self._client.cancel_order_by_id(order_id)
            deadline = time.monotonic()+float(getattr(self, '_cancel_grace_seconds', 2.0))
            while status not in terminal:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Cancellation remains unconfirmed')
                order = self._client.get_order_by_id(order_id)
                status = str(getattr(order.status, 'value', order.status))
                if status not in terminal:
                    time.sleep(self._poll_interval)
            if str(getattr(order.side, 'value', order.side)) == 'sell':
                self._emit_conflicting_stop_fill(order)
            self._unconfirmed_cancels.discard(str(order_id))
            if self._unsettled_orders.get(symbol, (None,None))[0] == str(order_id):
                self._unsettled_orders.pop(symbol, None)
            if self._protective_stop_ids.get(symbol) == str(order_id):
                self._protective_stop_ids.pop(symbol, None)
            protection_update(self, symbol, 'absent', order_id=str(order_id), covered_quantity=0)
            self._invalidate_position_cache()
            return True
        except Exception as exc:
            self._safety_blocked = True
            logger.warning('Cancellation recovery pending for %s: %s', order_id, exc)
            return False

    def replace_protective_stop(
        self,
        symbol: str,
        old_order_id: Optional[str],
        qty: float,
        stop_price: float,
    ) -> Optional[str]:
        self._safety_state()
        if old_order_id:
            try:
                current = self._client.get_order_by_id(old_order_id)
                status = getattr(current.status, "value", current.status)
                if status in ("filled", "canceled", "cancelled", "expired", "rejected"):
                    self._protective_stop_ids.pop(symbol, None)
                    if status == "filled":
                        self._unsettled_orders[symbol] = (old_order_id, None)
                        self._safety_blocked = True
                        return old_order_id
                    return self.place_protective_stop(symbol, qty, stop_price)
                if status not in ("new", "accepted"):
                    self._safety_blocked = True
                    self._failed_stop_updates[symbol] = (old_order_id, qty, stop_price)
                    return old_order_id
                from daytrading.execution.stop_recovery import unchanged
                if unchanged(current, symbol, qty, stop_price):
                    self._protective_stop_ids[symbol] = str(current.id)
                    self._failed_stop_updates.pop(symbol, None)
                    acknowledge_protection(self, symbol, current, qty, stop_price)
                    return str(current.id)
                from alpaca.trading.requests import ReplaceOrderRequest
                self._failed_stop_updates[symbol] = (old_order_id, qty, stop_price)
                self._checkpoint_request('replace_stop', symbol, old_order_id)
                replacement = self._client.replace_order_by_id(
                    old_order_id, order_data=ReplaceOrderRequest(qty=int(qty), stop_price=round(stop_price, 2)),
                )
                self._safety_state()
                self._protective_stop_ids[symbol] = str(replacement.id)
                if not acknowledge_protection(self, symbol, replacement, qty, stop_price):
                    self._safety_blocked = True
                    return None
                self._failed_stop_updates.pop(symbol, None)
                return str(replacement.id)
            except Exception as exc:
                # Retain existing protection if an update fails; do not cancel it.
                self._safety_state()
                self._safety_blocked = True
                self._failed_stop_updates[symbol] = (old_order_id, qty, stop_price)
                logger.error("Stop replacement failed %s; retaining existing stop: %s", symbol, exc)
                return old_order_id
        return self.place_protective_stop(symbol, qty, stop_price)

    def get_account(self) -> dict:
        """Return Alpaca account info (cash, buying power, etc.)."""
        acct = self._client.get_account()
        self._account_scope = str(acct.id)
        return {
            "account_id": self._account_scope,
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "equity": float(acct.equity),
            "pattern_day_trader": acct.pattern_day_trader,
        }

    def get_positions(self) -> dict:
        lock = self.__dict__.setdefault("_position_request_lock", threading.Lock())
        if not lock.acquire(blocking=False):
            if time.monotonic() - getattr(self, "_pos_cache_ts", 0.0) >= 0.5:
                self._position_sync_stale = True
            return getattr(self, "_pos_cache", None) or {}
        try:
            return self._fetch_positions()
        finally:
            lock.release()

    def _fetch_positions(self) -> dict:
        """One in-flight snapshot; worker completion only signals the owner."""
        if getattr(self, '_position_owner', threading.get_ident()) != threading.get_ident():
            return getattr(self, '_pos_cache', None) or {}
        self.adopt_completed_positions()
        now = time.monotonic()
        if (now - getattr(self, '_pos_cache_ts', 0.0) < 0.5
                and getattr(self, '_pos_cache', None) is not None
                and getattr(self, '_position_sync_stale', False) is not True):
            return self._pos_cache
        if getattr(self, '_position_future', None) is None:
            if not hasattr(self, '_position_pool'):
                self._position_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            request_id = getattr(self, '_position_request_id', 0) + 1
            self._position_request_id = request_id
            meta = dict(request_id=request_id, requested_at=datetime.now(timezone.utc).isoformat(),
                requested_monotonic=now, generation=getattr(self, '_position_generation', 0),
                context=getattr(self, '_position_context', None))
            self._position_sync_stale = True
            future = self._position_pool.submit(self._client.get_all_positions)
            future._daytrade_position_meta = meta
            self._position_future = future
            wakeup = getattr(self, '_position_wakeup', None)
            def completed(done):
                # No portfolio/cache/entry mutation and no journal IO on SDK thread.
                meta['completed_at'] = datetime.now(timezone.utc).isoformat()
                meta['completed_monotonic'] = time.monotonic()
                if wakeup is not None:
                    wakeup.set()
            future.add_done_callback(completed)
            self._position_refresh_note('requested', request_id=request_id,
                requested_at=meta['requested_at'], generation=meta['generation'])
        self._position_sync_stale = True
        self.adopt_completed_positions()
        return getattr(self, '_pos_cache', None) or {}

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        cancelled = self._client.cancel_orders()
        count = len(cancelled) if cancelled else 0
        logger.info("Cancelled %d open orders", count)
        return count

    def close_all_positions(self) -> None:
        """Liquidate everything (end-of-day safety)."""
        self._checkpoint_request('close_all_positions', '*', uuid.uuid4().hex)
        self._client.close_all_positions(cancel_orders=True)
        logger.info("Closed all positions")

    def emergency_close_all_positions(
        self,
        *,
        attempts: int = 3,
        settle_seconds: float = 1.0,
    ) -> dict:
        """Cancel all orders and keep closing positions until Alpaca is flat.

        Dashboard emergency-close must be stronger than the normal EOD close:
        open sell/stop orders can hold shares for a few seconds and make a
        fresh close order reject. This method cancels, waits for held orders to
        clear, retries, and returns the actual remaining broker state.
        """
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        cancelled_total = 0
        submitted = []
        errors = []

        def open_orders():
            try:
                req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
                return list(self._client.get_orders(filter=req) or [])
            except Exception as exc:
                errors.append("get open orders: {}".format(exc))
                return []

        def fresh_positions():
            try:
                return list(self._client.get_all_positions() or [])
            except Exception as exc:
                errors.append("get positions: {}".format(exc))
                return []

        def cancel_open_orders() -> int:
            count = 0
            for order in open_orders():
                try:
                    self._checkpoint_request('cancel', order.symbol, str(order.id))
                    self._client.cancel_order_by_id(str(order.id))
                    count += 1
                except Exception as exc:
                    errors.append("cancel {}: {}".format(getattr(order, "id", ""), exc))
            if count:
                self._invalidate_position_cache()
            return count

        for attempt in range(1, max(1, attempts) + 1):
            cancelled_total += cancel_open_orders()
            deadline = time.monotonic() + max(0.0, settle_seconds)
            while time.monotonic() < deadline:
                if not open_orders():
                    break
                time.sleep(0.1)

            positions = fresh_positions()
            if not positions:
                self._invalidate_position_cache()
                return {
                    "ok": True,
                    "flat": True,
                    "attempts": attempt,
                    "cancelled_orders": cancelled_total,
                    "submitted_orders": submitted,
                    "remaining_positions": {},
                    "errors": errors,
                }

            try:
                self._checkpoint_request('emergency_close_all_positions', '*', uuid.uuid4().hex)
                self._client.close_all_positions(cancel_orders=True)
                submitted.append({"type": "close_all_positions", "attempt": attempt})
            except Exception as exc:
                errors.append("close_all_positions: {}".format(exc))

            time.sleep(max(0.2, settle_seconds))

            # If the bulk close did not flatten everything, submit explicit
            # market closes for remaining quantities.
            for pos in fresh_positions():
                try:
                    qty = abs(float(pos.qty))
                    if qty <= 0:
                        continue
                    side_val = getattr(pos, "side", "")
                    side_str = side_val.value if hasattr(side_val, "value") else str(side_val)
                    order_side = OrderSide.SELL if side_str.lower() == "long" else OrderSide.BUY
                    req = MarketOrderRequest(
                        symbol=pos.symbol,
                        qty=qty,
                        side=order_side,
                        time_in_force=TimeInForce.DAY,
                        client_order_id=uuid.uuid4().hex,
                    )
                    self._checkpoint_request('emergency_close', pos.symbol, req.client_order_id)
                    order = self._client.submit_order(order_data=req)
                    submitted.append({
                        "type": "market_close",
                        "symbol": pos.symbol,
                        "qty": qty,
                        "side": order_side.value,
                        "id": str(order.id),
                    })
                except Exception as exc:
                    errors.append("market close {}: {}".format(getattr(pos, "symbol", ""), exc))

            time.sleep(max(0.2, settle_seconds))

        remaining = {
            p.symbol: {
                "qty": float(p.qty),
                "side": str(getattr(p, "side", "")),
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price) if p.current_price else None,
            }
            for p in fresh_positions()
        }
        self._invalidate_position_cache()
        flat = not remaining
        logger.warning(
            "Emergency close completed flat=%s remaining=%s cancelled=%d errors=%s",
            flat, list(remaining.keys()), cancelled_total, errors,
        )
        return {
            "ok": flat,
            "flat": flat,
            "attempts": max(1, attempts),
            "cancelled_orders": cancelled_total,
            "submitted_orders": submitted,
            "remaining_positions": remaining,
            "errors": errors,
        }
