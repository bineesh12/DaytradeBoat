"""Alpaca broker — submit real orders to Alpaca paper (or live) trading.

Uses alpaca-py SDK. Implements the same Broker protocol as PaperBroker
so it drops into the pipeline with zero changes.

Usage:
    from daytrading.execution.alpaca_broker import AlpacaBroker

    broker = AlpacaBroker(api_key="...", secret_key="...", paper=True)
    fill, status = broker.submit(order, bar, portfolio)
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

from daytrading.models import Bar, Fill, Order, OrderStatus, PortfolioState, Side

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderStatus as AlpacaOrderStatus, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest
    _HAS_ALPACA = True
except ImportError:
    _HAS_ALPACA = False


class AlpacaBroker:
    """Submits orders to Alpaca Trading API (paper or live).

    Matches the ``Broker`` protocol:
        submit(order, bar, portfolio) → (Fill | None, OrderStatus)
    """

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
        self._paper = paper
        self._max_wait = max_wait_seconds
        self._poll_interval = poll_interval
        self._cancel_grace_seconds = 2.0
        self._marketable_limit_slippage_pct = 0.0075
        self._sell_limit_slippage_pct = 0.0075
        self._limit_buffer_pct = limit_buffer_pct
        self._slippage_guard = slippage_guard
        logger.info("AlpacaBroker initialized (paper=%s)", paper)

    @property
    def client(self) -> TradingClient:
        return self._client

    def submit(
        self,
        order: Order,
        bar: Bar,
        portfolio: PortfolioState,
    ) -> Tuple[Optional[Fill], OrderStatus]:
        """Submit an order to Alpaca and wait for fill."""
        if order.quantity <= 0:
            return None, OrderStatus.REJECTED

        # Cancel any open orders for this symbol to avoid wash trade rejections
        self._cancel_open_orders_for(order.symbol)

        alpaca_side = OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL

        try:
            # Use live quote-based pricing when available for tighter fills
            smart_price = None
            if self._slippage_guard and hasattr(self._slippage_guard, "get_limit_price"):
                side_str = "buy" if order.side is Side.BUY else "sell"
                smart_price = self._slippage_guard.get_limit_price(order.symbol, side_str)

            if order.limit_price is None:
                base = smart_price if smart_price is not None else bar.close
                if base <= 0:
                    logger.error("ORDER REJECTED %s: no price for guarded marketable limit", order.symbol)
                    return None, OrderStatus.REJECTED
                guard = float(getattr(self, "_marketable_limit_slippage_pct", 0.01))
                if order.side is Side.SELL:
                    adj_price = round(base * (1.0 - guard), 2)
                else:
                    adj_price = round(base * (1.0 + guard), 2)
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=int(order.quantity),
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=adj_price,
                    extended_hours=True,
                )
                alpaca_order = self._client.submit_order(order_data=request)
                logger.info(
                    "ORDER SUBMITTED %s %s %d guarded_limit=%.2f (%.1f%% window) → id=%s",
                    order.side.value, order.symbol, int(order.quantity),
                    adj_price, guard * 100.0, alpaca_order.id,
                )
                return self._wait_for_fill(alpaca_order.id, order)

            if order.side is Side.SELL:
                if smart_price is not None:
                    adj_price = smart_price
                    logger.info("SMART LIMIT SELL %s: bid-based %.2f", order.symbol, adj_price)
                else:
                    base = order.limit_price
                    sell_guard = float(getattr(self, "_sell_limit_slippage_pct", 0.0075))
                    adj_price = round(base * (1.0 - sell_guard), 2)
            else:
                # Dynamic slippage: momentum stocks need more room to fill
                # Base: 1.5% for stocks under $5, 1% for $5-20, 0.8% for $20+
                base = order.limit_price
                if base < 5.0:
                    max_slippage_pct = 0.015
                elif base < 20.0:
                    max_slippage_pct = 0.01
                else:
                    max_slippage_pct = 0.008

                # If the bar shows strong momentum (big green candle), allow extra room
                if bar.close > bar.open and bar.high > bar.low:
                    bar_range_pct = (bar.high - bar.low) / bar.low if bar.low > 0 else 0
                    if bar_range_pct > 0.02:
                        max_slippage_pct = min(max_slippage_pct + 0.005, 0.025)

                max_limit = round(base * (1.0 + max_slippage_pct), 2)

                if smart_price is not None:
                    if smart_price > max_limit:
                        logger.info(
                            "SLIPPAGE REJECT BUY %s: ask %.2f too far from signal %.2f (max %.2f, slip=%.1f%%)",
                            order.symbol, smart_price, base, max_limit, max_slippage_pct * 100,
                        )
                        return None, OrderStatus.REJECTED
                    adj_price = smart_price
                    logger.info("SMART LIMIT BUY %s: ask-based %.2f (signal %.2f)", order.symbol, adj_price, base)
                else:
                    adj_price = max_limit

            request = LimitOrderRequest(
                symbol=order.symbol,
                qty=int(order.quantity),
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                limit_price=adj_price,
                extended_hours=True,
            )

            alpaca_order = self._client.submit_order(order_data=request)
            price_info = "limit={:.2f}".format(request.limit_price)
            logger.info(
                "ORDER SUBMITTED %s %s %d %s → id=%s",
                order.side.value, order.symbol, int(order.quantity),
                price_info, alpaca_order.id,
            )

        except Exception as exc:
            logger.error("ORDER REJECTED by Alpaca: %s", exc)
            return None, OrderStatus.REJECTED

        return self._wait_for_fill(alpaca_order.id, order)

    def _wait_for_fill(
        self,
        order_id: str,
        original: Order,
    ) -> Tuple[Optional[Fill], OrderStatus]:
        """Poll Alpaca until the order fills, is rejected, or times out."""
        deadline = time.monotonic() + self._max_wait

        while time.monotonic() < deadline:
            try:
                alpaca_order = self._client.get_order_by_id(order_id)
            except Exception as exc:
                logger.error("Error checking order %s: %s", order_id, exc)
                return None, OrderStatus.REJECTED

            status_val = alpaca_order.status.value if hasattr(alpaca_order.status, 'value') else str(alpaca_order.status)

            if status_val == "filled":
                fill = self._fill_from_alpaca_order(alpaca_order, original)
                if fill:
                    logger.info(
                        "ORDER FILLED %s %s %.0f @ %.4f",
                        original.side.value, original.symbol, fill.quantity, fill.price,
                    )
                    self._record_slippage(original, fill)
                    return fill, OrderStatus.FILLED

            if status_val in ("canceled", "cancelled", "expired", "rejected"):
                fill = self._fill_from_alpaca_order(alpaca_order, original)
                if fill:
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
        return Fill(
            symbol=original.symbol,
            side=original.side,
            quantity=fill_qty,
            price=fill_price,
            ts=fill_ts,
            commission=0.0,
        )

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
        while time.monotonic() < deadline:
            try:
                final_order = self._client.get_order_by_id(order_id)
            except Exception:
                break
            fill = self._fill_from_alpaca_order(final_order, original)
            if fill:
                last_fill = fill
            status = getattr(final_order, "status", "")
            status_val = status.value if hasattr(status, "value") else str(status)
            if status_val in ("filled", "canceled", "cancelled", "expired", "rejected") and last_fill:
                break
            if status_val in ("canceled", "cancelled", "expired", "rejected") and not last_fill:
                break
            time.sleep(self._poll_interval)

        if last_fill:
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

    def _cancel_open_orders_for(self, symbol: str, *, preserve_stops: bool = False) -> None:
        """Cancel open orders for a symbol to prevent wash trade conflicts."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            open_orders = self._client.get_orders(filter=req)
            cancelled = 0
            for o in open_orders:
                order_type = getattr(o, "type", None) or getattr(o, "order_type", None)
                type_str = (
                    order_type.value if hasattr(order_type, "value")
                    else str(order_type or "")
                ).lower()
                if preserve_stops and "stop" in type_str:
                    continue
                try:
                    self._client.cancel_order_by_id(str(o.id))
                    cancelled += 1
                    logger.info("Cancelled conflicting order %s for %s", o.id, symbol)
                except Exception:
                    pass
            if cancelled:
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    if self._open_order_count_for(symbol) == 0:
                        break
                    time.sleep(0.1)
                self._invalidate_position_cache()
        except Exception as exc:
            logger.warning("Could not check open orders for %s: %s", symbol, exc)

    def place_protective_stop(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
    ) -> Optional[str]:
        """Place a broker-held stop-loss sell (survives bot restarts / sync gaps)."""
        if qty <= 0 or stop_price <= 0:
            return None
        try:
            from alpaca.trading.requests import StopOrderRequest
            request = StopOrderRequest(
                symbol=symbol,
                qty=int(qty),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=round(stop_price, 2),
            )
            alpaca_order = self._client.submit_order(order_data=request)
            self._invalidate_position_cache()
            oid = str(alpaca_order.id)
            logger.info(
                "BROKER STOP %s sell %.0f @ stop $%.2f → id=%s",
                symbol, qty, stop_price, oid,
            )
            return oid
        except Exception as exc:
            logger.error("BROKER STOP failed %s: %s", symbol, exc)
            return None

    def cancel_order_by_id(self, order_id: str) -> None:
        try:
            self._client.cancel_order_by_id(order_id)
            self._invalidate_position_cache()
        except Exception as exc:
            logger.debug("Cancel order %s: %s", order_id, exc)

    def replace_protective_stop(
        self,
        symbol: str,
        old_order_id: Optional[str],
        qty: float,
        stop_price: float,
    ) -> Optional[str]:
        if old_order_id:
            self.cancel_order_by_id(old_order_id)
        return self.place_protective_stop(symbol, qty, stop_price)

    def get_account(self) -> dict:
        """Return Alpaca account info (cash, buying power, etc.)."""
        acct = self._client.get_account()
        return {
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "equity": float(acct.equity),
            "pattern_day_trader": acct.pattern_day_trader,
        }

    def get_positions(self) -> dict:
        """Return all open Alpaca positions (cached for 0.5s to reduce API calls)."""
        now = time.monotonic()
        if (hasattr(self, '_pos_cache_ts')
                and now - self._pos_cache_ts < 0.5
                and self._pos_cache is not None):
            return self._pos_cache
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._client.get_all_positions)
                positions = future.result(timeout=10)
        except (TimeoutError, concurrent.futures.TimeoutError):
            logger.warning("get_positions timed out (10s) — returning stale cache")
            return self._pos_cache if hasattr(self, '_pos_cache') and self._pos_cache else {}
        except Exception as exc:
            logger.warning("get_positions failed: %s — returning stale cache", exc)
            return self._pos_cache if hasattr(self, '_pos_cache') and self._pos_cache else {}
        result = {
            p.symbol: {
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price) if p.current_price else float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "side": p.side,
            }
            for p in positions
        }
        self._pos_cache = result
        self._pos_cache_ts = now
        return result

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        cancelled = self._client.cancel_orders()
        count = len(cancelled) if cancelled else 0
        logger.info("Cancelled %d open orders", count)
        return count

    def close_all_positions(self) -> None:
        """Liquidate everything (end-of-day safety)."""
        self._client.close_all_positions(cancel_orders=True)
        logger.info("Closed all positions")
