"""Risk gate for the experimental Warrior squeeze playbook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WarriorRiskDecision:
    allowed: bool
    reason: str = ""


class WarriorRiskAllocator:
    """Pure Warrior risk gate.

    Phase 2 starts with the existing behavior: one active Warrior/burst scalp at
    a time.  The object is intentionally parameterized now so Phase 4 can raise
    ``max_concurrent_warrior_trades`` without rewriting the orchestration loop.
    """

    def __init__(self, *, max_concurrent_warrior_trades: int = 1) -> None:
        self.max_concurrent_warrior_trades = max(1, int(max_concurrent_warrior_trades or 1))

    def allow(
        self,
        symbol: str,
        state: Any,
        open_positions: int = 0,
    ) -> WarriorRiskDecision:
        sym = str(symbol or "").upper()
        day_blocked = getattr(state, "day_blocked", {}) or {}
        if sym and sym in day_blocked:
            return WarriorRiskDecision(False, str(day_blocked.get(sym) or "Warrior symbol blocked"))
        open_count = max(0, int(open_positions or 0))
        if open_count >= self.max_concurrent_warrior_trades:
            return WarriorRiskDecision(
                False,
                "max concurrent Warrior trades reached "
                f"({open_count}/{self.max_concurrent_warrior_trades})",
            )
        return WarriorRiskDecision(True)
