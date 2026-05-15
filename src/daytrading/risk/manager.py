from __future__ import annotations

from typing import Optional

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from daytrading.models import Bar, Order, PortfolioState, Side


@runtime_checkable
class RiskManager(Protocol):
    def check(self, order: Order, bar: Bar, portfolio: PortfolioState) -> bool:
        ...


def allow_order(
    order: Order,
    bar: Bar,
    portfolio: PortfolioState,
    *,
    max_position_shares: Optional[float] = None,
    max_order_shares: Optional[float] = None,
) -> bool:
    """Lightweight pre-trade checks strategies or a custom RiskManager can call."""

    if max_order_shares is not None and order.quantity > max_order_shares:
        return False

    pos = portfolio.positions.get(order.symbol)
    cur = pos.quantity if pos else 0.0

    if order.side is Side.BUY and max_position_shares is not None:
        if cur + order.quantity > max_position_shares:
            return False
    if order.side is Side.SELL and max_position_shares is not None:
        if cur - order.quantity < -max_position_shares:
            return False

    _ = bar  # reserved for notional / volatility caps later
    return True
