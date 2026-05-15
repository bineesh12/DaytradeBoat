"""Alpaca broker — submit real orders to Alpaca paper (or live) trading.

Uses alpaca-py SDK. Implements the same Broker protocol as PaperBroker
so it drops into the pipeline with zero changes.

Usage:
    from daytrading.execution.alpaca_broker import AlpacaBroker

    broker = AlpacaBroker(api_key="...", secret_key="...", paper=True)
    fill, status = broker.submit(order, bar, portfolio)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

from daytrading.models import Bar, Fill, Order, OrderStatus, PortfolioState, Side

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderStatus as AlpacaOrderStatus, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
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
        max_wait_seconds: float = 15.0,
        poll_interval: float = 0.3,
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

            if order.side is Side.SELL:
                if smart_price is not None:
                    adj_price = smart_price
                    logger.info("SMART LIMIT SELL %s: bid-based %.2f", order.symbol, adj_price)
                else:
                    base = order.limit_price if order.limit_price else bar.close
                    adj_price = round(base * (1.0 - 0.03), 2)
            else:
                if smart_price is not None:
                    adj_price = smart_price
                    logger.info("SMART LIMIT BUY %s: ask-based %.2f", order.symbol, adj_price)
                else:
                    base = order.limit_price if order.limit_price else bar.close
                    adj_price = round(base * (1.0 + self._limit_buffer_pct), 2)

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
                fill_price = float(alpaca_order.filled_avg_price or 0)
                fill_qty = float(alpaca_order.filled_qty or 0)
                fill_ts = alpaca_order.filled_at or datetime.now(timezone.utc)

                fill = Fill(
                    symbol=original.symbol,
                    side=original.side,
                    quantity=fill_qty,
                    price=fill_price,
                    ts=fill_ts,
                    commission=0.0,
                )
                logger.info(
                    "ORDER FILLED %s %s %.0f @ %.4f",
                    original.side.value, original.symbol, fill_qty, fill_price,
                )
                # Track slippage for monitoring
                if self._slippage_guard and hasattr(self._slippage_guard, "record_fill"):
                    expected = original.limit_price or fill_price
                    self._slippage_guard.record_fill(original.symbol, expected, fill_price)
                return fill, OrderStatus.FILLED

            if status_val in ("canceled", "cancelled", "expired", "rejected"):
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

        # Check for partial fill before declaring cancelled
        try:
            final_order = self._client.get_order_by_id(order_id)
            partial_qty = float(final_order.filled_qty or 0)
            partial_price = float(final_order.filled_avg_price or 0)
            if partial_qty > 0 and partial_price > 0:
                logger.info(
                    "PARTIAL FILL %s %s %.0f of %.0f @ %.4f (rest cancelled)",
                    original.side.value, original.symbol,
                    partial_qty, original.quantity, partial_price,
                )
                fill = Fill(
                    symbol=original.symbol,
                    side=original.side,
                    quantity=partial_qty,
                    price=partial_price,
                    ts=datetime.now(timezone.utc),
                )
                return fill, OrderStatus.FILLED
        except Exception:
            pass

        return None, OrderStatus.CANCELLED

    def _cancel_open_orders_for(self, symbol: str) -> None:
        """Cancel any open orders for a symbol to prevent wash trade conflicts."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            open_orders = self._client.get_orders(filter=req)
            for o in open_orders:
                try:
                    self._client.cancel_order_by_id(str(o.id))
                    logger.info("Cancelled conflicting order %s for %s", o.id, symbol)
                except Exception:
                    pass
            if open_orders:
                import time
                time.sleep(0.2)
        except Exception as exc:
            logger.warning("Could not check open orders for %s: %s", symbol, exc)

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
        """Return all open Alpaca positions."""
        positions = self._client.get_all_positions()
        return {
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
