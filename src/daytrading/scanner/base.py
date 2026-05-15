from __future__ import annotations

from typing import Dict, List, Sequence

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from daytrading.models import Bar, ScanResult


@runtime_checkable
class Scanner(Protocol):
    """Screens a universe of symbols and returns those meeting criteria."""

    @property
    def name(self) -> str:
        ...

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        """universe maps symbol → recent bars for that symbol.

        Returns a list of ScanResults, one per symbol that passed.
        """
        ...
