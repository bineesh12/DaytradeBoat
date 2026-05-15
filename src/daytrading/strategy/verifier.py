"""Strategy verifier protocol.

A verifier takes a ScanResult (a scanner hit) and decides whether the setup
is actually tradeable — producing a TradeSignal or returning None to skip.
"""

from __future__ import annotations

from typing import List, Optional

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from daytrading.models import PortfolioState, ScanResult, TradeSignal


@runtime_checkable
class StrategyVerifier(Protocol):
    """Validate scanner output against strategy-specific entry rules."""

    @property
    def name(self) -> str:
        ...

    def verify(
        self,
        scan_result: ScanResult,
        portfolio: PortfolioState,
    ) -> Optional[TradeSignal]:
        """Return a TradeSignal if the scan hit meets entry criteria, else None."""
        ...
