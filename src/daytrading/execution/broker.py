from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from daytrading.models import Bar, Fill, Order, OrderStatus, PortfolioState, Position, Side


@runtime_checkable
class Broker(Protocol):
    """Turns orders into fills at market (or your venue model)."""

    def submit(
        self,
        order: Order,
        bar: Bar,
        portfolio: PortfolioState,
    ) -> Tuple[Optional[Fill], OrderStatus]:
        ...


class PaperBroker:
    """Immediate fill at bar close; optional commission per share."""

    def __init__(self, commission_per_share: float = 0.0) -> None:
        self._commission_per_share = commission_per_share

    def submit(
        self,
        order: Order,
        bar: Bar,
        portfolio: PortfolioState,
    ) -> Tuple[Optional[Fill], OrderStatus]:
        if order.symbol != bar.symbol:
            return None, OrderStatus.REJECTED
        if order.quantity <= 0:
            return None, OrderStatus.REJECTED

        price = order.limit_price if order.limit_price is not None else bar.close
        if order.side is Side.BUY and portfolio.cash < order.quantity * price:
            return None, OrderStatus.REJECTED

        comm = abs(order.quantity) * self._commission_per_share
        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            ts=bar.ts,
            commission=comm,
        )
        return fill, OrderStatus.FILLED


def apply_fill(portfolio: PortfolioState, fill: Fill) -> None:
    """Update cash and position from a fill (mutates portfolio)."""
    sign = 1.0 if fill.side is Side.BUY else -1.0
    cost = sign * fill.quantity * fill.price + fill.commission
    portfolio.cash -= cost

    pos = portfolio.positions.get(fill.symbol) or Position(symbol=fill.symbol)
    old_q = pos.quantity
    new_q = old_q + sign * fill.quantity

    if old_q == 0:
        pos.avg_price = fill.price
    elif (old_q > 0 and sign > 0) or (old_q < 0 and sign < 0):
        pos.avg_price = (pos.avg_price * abs(old_q) + fill.price * fill.quantity) / abs(new_q)
    else:
        # reducing or flipping; keep avg for remaining direction simplified
        if new_q == 0:
            pos.avg_price = 0.0
        elif abs(new_q) < abs(old_q):
            pass  # avg unchanged when scaling down
        else:
            pos.avg_price = fill.price

    pos.quantity = new_q
    if new_q == 0:
        portfolio.positions.pop(fill.symbol, None)
    else:
        portfolio.positions[fill.symbol] = pos
