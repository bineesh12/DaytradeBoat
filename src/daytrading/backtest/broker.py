from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
from daytrading.execution.pricing import ExecutionPricing, marketable_limit

from daytrading.models import Bar, Fill, Order, OrderStatus, PortfolioState, Quote, Side


@dataclass(frozen=True)
class FillModel:
    """Conservative bar-based fill model for strategy replay."""

    commission_per_share: float = 0.0
    min_spread_cents: float = 0.01
    spread_pct_of_range: float = 0.20
    max_spread_pct: float = 1.50
    max_volume_participation: float = 0.10


class BacktestBroker:
    """Broker-compatible simulator for the real trading pipeline.

    The pipeline submits limit-like orders. In live trading buys cross the ask
    and sells hit the bid, so this model estimates a bar-local spread and fills
    entries/exits on the costly side of that spread. It keeps the API identical
    to PaperBroker/AlpacaBroker so the backtest can reuse the real pipeline.
    """

    def __init__(self, model: Optional[FillModel] = None) -> None:
        self.model = model or FillModel()
        if not 0 < self.model.max_volume_participation <= 1:
            raise ValueError("volume participation must be between 0 and 1")
        self.pricing = ExecutionPricing()
        self._quote_depth_used: dict[tuple, int] = {}

    def submit_at_quote(self, order: Order, bar: Bar, portfolio: PortfolioState,
                        *, quote: Quote, now) -> Tuple[Optional[Fill], OrderStatus]:
        """Causal NBBO estimate with live request pricing, separate from bar fills.

        The caller supplies the quote available at simulated order arrival.
        A request limit is not assumed to be an execution price. Displayed
        depth can be consumed once; broker price improvement is not invented.
        """
        from daytrading.backtest.execution import quote_request_limit
        try:
            if not math.isfinite(order.quantity) or order.quantity <= 0:
                raise ValueError('Invalid quote order quantity')
            limit = quote_request_limit(order, bar, quote, now=now)
            price = quote.ask if order.side is Side.BUY else quote.bid
            depth = quote.ask_size if order.side is Side.BUY else quote.bid_size
            if not math.isfinite(depth) or depth < 0:
                raise ValueError('Invalid quote depth')
        except (ValueError, TypeError):
            return None, OrderStatus.REJECTED
        if ((order.side is Side.BUY and price > limit + 1e-9)
                or (order.side is Side.SELL and price < limit - 1e-9)):
            return None, OrderStatus.CANCELLED
        key = (quote.symbol, quote.ts, order.side)
        qty = min(int(order.quantity), max(0, int(depth) - self._quote_depth_used.get(key, 0)))
        if order.side is Side.BUY:
            qty = min(qty, int(portfolio.cash / (price + self.model.commission_per_share)))
        if qty <= 0:
            return None, OrderStatus.CANCELLED
        self._quote_depth_used[key] = self._quote_depth_used.get(key, 0) + qty
        return Fill(order.symbol, order.side, qty, price, now,
                    commission=qty*self.model.commission_per_share), OrderStatus.FILLED

    def estimated_spread(self, bar: Bar) -> float:
        price = max(float(bar.close), 0.0)
        if price <= 0:
            return 0.0
        range_spread = max(float(bar.high) - float(bar.low), 0.0) * self.model.spread_pct_of_range
        max_spread = price * (self.model.max_spread_pct / 100.0)
        spread = max(self.model.min_spread_cents, range_spread)
        return min(spread, max_spread)

    def submit(
        self,
        order: Order,
        bar: Bar,
        portfolio: PortfolioState,
    ) -> Tuple[Optional[Fill], OrderStatus]:
        if order.symbol != bar.symbol or not math.isfinite(order.quantity) or int(order.quantity) <= 0:
            return None, OrderStatus.REJECTED
        if not all(math.isfinite(float(value)) for value in (bar.close, bar.volume, bar.high, bar.low)):
            return None, OrderStatus.REJECTED

        spread = self.estimated_spread(bar)
        try:
            limit = marketable_limit(order, bar, policy=self.pricing)
        except ValueError:
            return None, OrderStatus.REJECTED
        # Signals are emitted at the closed-bar boundary. Do not fill at an
        # earlier low/high already observed while computing that signal.
        quantity = min(int(order.quantity), int(max(0, bar.volume) * self.model.max_volume_participation))
        if quantity <= 0:
            return None, OrderStatus.CANCELLED
        if order.side is Side.BUY:
            price = float(bar.close) + spread / 2.0
            if price > limit:
                return None, OrderStatus.CANCELLED
        else:
            price = max(float(bar.close) - spread / 2.0, 0.01)
            if price < limit:
                return None, OrderStatus.CANCELLED
        commission = quantity * self.model.commission_per_share
        if order.side is Side.BUY and portfolio.cash < quantity * price + commission:
            return None, OrderStatus.REJECTED
        return (
            Fill(
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                price=round(price, 4),
                ts=bar.ts,
                commission=commission,
            ),
            OrderStatus.FILLED,
        )
