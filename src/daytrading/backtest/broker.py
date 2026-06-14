from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from daytrading.models import Bar, Fill, Order, OrderStatus, PortfolioState, Side


@dataclass(frozen=True)
class FillModel:
    """Conservative bar-based fill model for strategy replay."""

    commission_per_share: float = 0.0
    min_spread_cents: float = 0.01
    spread_pct_of_range: float = 0.20
    max_spread_pct: float = 1.50


class BacktestBroker:
    """Broker-compatible simulator for the real trading pipeline.

    The pipeline submits limit-like orders. In live trading buys cross the ask
    and sells hit the bid, so this model estimates a bar-local spread and fills
    entries/exits on the costly side of that spread. It keeps the API identical
    to PaperBroker/AlpacaBroker so the backtest can reuse the real pipeline.
    """

    def __init__(self, model: Optional[FillModel] = None) -> None:
        self.model = model or FillModel()

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
        if order.symbol != bar.symbol or order.quantity <= 0:
            return None, OrderStatus.REJECTED

        spread = self.estimated_spread(bar)
        mid = float(order.limit_price if order.limit_price is not None else bar.close)
        if mid <= 0:
            return None, OrderStatus.REJECTED

        if order.side is Side.BUY:
            if (
                order.limit_price is not None
                and float(bar.low) <= mid <= float(bar.high)
                and mid < float(bar.close)
            ):
                price = mid + spread / 2.0
            else:
                price = max(mid, float(bar.close)) + spread / 2.0
            if portfolio.cash < order.quantity * price:
                return None, OrderStatus.REJECTED
        else:
            price = min(mid, float(bar.close)) - spread / 2.0
            price = max(price, 0.01)

        commission = abs(order.quantity) * self.model.commission_per_share
        return (
            Fill(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=round(price, 4),
                ts=bar.ts,
                commission=commission,
            ),
            OrderStatus.FILLED,
        )
