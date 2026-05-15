from __future__ import annotations

from typing import Iterator, Sequence

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from daytrading.models import Bar


@runtime_checkable
class DataFeed(Protocol):
    """Yields time-ordered bars (single- or multi-symbol interleaved by your rules)."""

    def iter_bars(self) -> Iterator[Bar]:
        ...


class InMemoryBarFeed:
    """Simple feed from a pre-sorted list of bars."""

    def __init__(self, bars: Sequence[Bar]) -> None:
        self._bars = list(bars)

    def iter_bars(self) -> Iterator[Bar]:
        yield from self._bars
