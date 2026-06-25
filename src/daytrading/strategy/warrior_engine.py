"""Shared Warrior playbook coordinator."""

from __future__ import annotations

from dataclasses import dataclass

from daytrading.strategy.warrior_risk import WarriorRiskAllocator, WarriorRiskDecision
from daytrading.strategy.warrior_watch import WarriorWatchBook


@dataclass
class WarriorEngine:
    watch: WarriorWatchBook
    risk: WarriorRiskAllocator

    @classmethod
    def with_defaults(cls, *, max_concurrent_warrior_trades: int = 1) -> "WarriorEngine":
        watch = WarriorWatchBook()
        return cls(
            watch=watch,
            risk=WarriorRiskAllocator(
                max_concurrent_warrior_trades=max_concurrent_warrior_trades
            ),
        )

    def allow_entry(self, symbol: str, *, open_positions: int = 0) -> WarriorRiskDecision:
        return self.risk.allow(symbol, self.watch, open_positions=open_positions)

    def reset_session(self) -> None:
        self.watch.reset_session()
