from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from daytrading.data.feed import DataFeed
from daytrading.execution.broker import Broker, apply_fill
from daytrading.strategy.base import Strategy
from daytrading.models import Bar, Fill, OrderStatus, PortfolioState


@dataclass
class BacktestResult:
    fills: List[Fill] = field(default_factory=list)
    final_portfolio: Optional[PortfolioState] = None
    bars_processed: int = 0


class BacktestEngine:
    """Event loop: each bar → strategy orders → broker fills → portfolio update."""

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        broker: Broker,
        initial_cash: float,
    ) -> None:
        self._feed = feed
        self._strategy = strategy
        self._broker = broker
        self._initial_cash = initial_cash

    def run(self) -> BacktestResult:
        portfolio = PortfolioState(cash=self._initial_cash, positions={})
        fills: List[Fill] = []
        n = 0
        for bar in self._feed.iter_bars():
            n += 1
            orders = self._strategy.on_bar(bar, portfolio)
            for order in orders:
                fill, status = self._broker.submit(order, bar, portfolio)
                if status is OrderStatus.FILLED and fill is not None:
                    apply_fill(portfolio, fill)
                    fills.append(fill)
        return BacktestResult(
            fills=fills,
            final_portfolio=portfolio,
            bars_processed=n,
        )
