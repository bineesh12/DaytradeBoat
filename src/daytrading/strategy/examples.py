from __future__ import annotations

from daytrading.risk.manager import allow_order
from daytrading.models import Bar, Order, PortfolioState, Side


class BuyAndHoldOnce:
    """Buy one unit on first bar if flat (demo only)."""

    def __init__(self, symbol: str, size: float = 1.0) -> None:
        self._symbol = symbol
        self._size = size
        self._done = False

    def on_bar(self, bar: Bar, portfolio: PortfolioState) -> list[Order]:
        if self._done or bar.symbol != self._symbol:
            return []
        pos = portfolio.positions.get(self._symbol)
        if pos and pos.quantity != 0:
            self._done = True
            return []
        order = Order(symbol=self._symbol, side=Side.BUY, quantity=self._size)
        if not allow_order(order, bar, portfolio, max_order_shares=self._size * 2):
            return []
        self._done = True
        return [order]
