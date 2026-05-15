from __future__ import annotations

from typing import List

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from daytrading.models import Bar, Order, PortfolioState


@runtime_checkable
class Strategy(Protocol):
    """Pure decision logic: map bar + portfolio to zero or more orders."""

    def on_bar(self, bar: Bar, portfolio: PortfolioState) -> list[Order]:
        ...
