"""Shared state owner for the experimental Warrior squeeze playbook.

The live runner and replay driver still orchestrate entries separately today,
but their Warrior state should have one shape and one reset lifecycle.  This
object is intentionally a thin dictionary owner for Phase 1 of the refactor:
call sites can keep their existing behavior while state stops living as loose
attributes scattered across runner/driver classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass
class WarriorWatchBook:
    """Per-symbol state used by Warrior and momentum-burst hit-run flows."""

    armed: Dict[str, Any] = field(default_factory=dict)
    window_high: Dict[str, float] = field(default_factory=dict)
    session_anchor_high: Dict[str, float] = field(default_factory=dict)
    pending: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hit_run_counts: Dict[str, int] = field(default_factory=dict)
    hit_run_block_until: Dict[str, Any] = field(default_factory=dict)
    symbol_pnl: Dict[str, float] = field(default_factory=dict)
    symbol_peak_pnl: Dict[str, float] = field(default_factory=dict)
    day_blocked: Dict[str, str] = field(default_factory=dict)
    rejection_high: Dict[str, float] = field(default_factory=dict)
    rejection_reason: Dict[str, str] = field(default_factory=dict)
    target_wins: Dict[str, int] = field(default_factory=dict)
    last_target_at: Dict[str, Any] = field(default_factory=dict)
    failed_burst: Dict[str, str] = field(default_factory=dict)
    failed_burst_high: Dict[str, float] = field(default_factory=dict)
    post_target_reclaim_allowed: Dict[str, int] = field(default_factory=dict)
    last_entry_trigger: Dict[str, str] = field(default_factory=dict)
    normal_fallback_rejects: Dict[str, int] = field(default_factory=dict)
    normal_fallback_last_reason: Dict[str, str] = field(default_factory=dict)

    def reset_session(self) -> None:
        """Clear all state that must not leak between trading sessions."""
        self.armed.clear()
        self.window_high.clear()
        self.session_anchor_high.clear()
        self.pending.clear()
        self.hit_run_counts.clear()
        self.hit_run_block_until.clear()
        self.symbol_pnl.clear()
        self.symbol_peak_pnl.clear()
        self.day_blocked.clear()
        self.rejection_high.clear()
        self.rejection_reason.clear()
        self.target_wins.clear()
        self.last_target_at.clear()
        self.failed_burst.clear()
        self.failed_burst_high.clear()
        self.post_target_reclaim_allowed.clear()
        self.last_entry_trigger.clear()
        self.normal_fallback_rejects.clear()
        self.normal_fallback_last_reason.clear()

    def clear_watch_symbol(self, symbol: str) -> None:
        """Remove a symbol from active Warrior watch state without erasing P&L.

        Eviction should not make a rediscovered symbol less constrained than a
        continuously watched symbol.  Clear the paired arming/rejection/cooldown
        state together so a symbol cannot keep a stale rejection-high path while
        losing its failed-burst cooldown.
        """
        sym = symbol.upper()
        self.armed.pop(sym, None)
        self.window_high.pop(sym, None)
        self.session_anchor_high.pop(sym, None)
        self.pending.pop(sym, None)
        self.hit_run_counts.pop(sym, None)
        self.hit_run_block_until.pop(sym, None)
        self.rejection_high.pop(sym, None)
        self.rejection_reason.pop(sym, None)
        self.target_wins.pop(sym, None)
        self.last_target_at.pop(sym, None)
        self.failed_burst.pop(sym, None)
        self.failed_burst_high.pop(sym, None)
        self.post_target_reclaim_allowed.pop(sym, None)
        self.last_entry_trigger.pop(sym, None)
        self.normal_fallback_rejects.pop(sym, None)
        self.normal_fallback_last_reason.pop(sym, None)

    def watch_score(self, symbol: str, candidate_high: Optional[float] = None) -> float:
        """Rank watched symbols so low-quality inactive names can be replaced."""
        sym = symbol.upper()
        high = float(
            candidate_high
            if candidate_high is not None
            else self.window_high.get(sym, 0.0)
            or 0.0
        )
        anchor = float(
            self.session_anchor_high.get(sym)
            or self.rejection_high.get(sym)
            or high
            or 0.0
        )
        pct_score = ((high / anchor) - 1.0) * 100.0 if anchor > 0 and high > 0 else 0.0
        score = pct_score + min(high, 50.0) * 0.1
        score += float(self.target_wins.get(sym, 0) or 0) * 100.0
        if sym in self.pending:
            score += 20.0
        if sym in self.day_blocked:
            score -= 1_000.0
        return score

    def ensure_capacity(
        self,
        symbol: str,
        *,
        capacity: int,
        candidate_high: Optional[float] = None,
        active_symbols: Iterable[str] = (),
    ) -> bool:
        """Make room for a new watched symbol by evicting the weakest inactive one.

        ``capacity <= 0`` intentionally means unlimited watch capacity.  Active
        positions and symbols that already banked a target are hard-protected;
        pending-only symbols can be replaced by a stronger fresh candidate.
        """
        sym = symbol.upper()
        if capacity <= 0 or sym in self.armed or len(self.armed) < capacity:
            return True
        protected = {s.upper() for s in active_symbols}
        protected.update(
            s.upper()
            for s, wins in self.target_wins.items()
            if int(wins or 0) > 0
        )
        candidates = [s for s in self.armed.keys() if s.upper() not in protected]
        if not candidates:
            return False
        weakest = min(candidates, key=lambda s: self.watch_score(s))
        candidate_score = self.watch_score(sym, candidate_high)
        weakest_score = self.watch_score(weakest)
        # A fresh Warrior-qualified symbol deserves a shot even if the scores are
        # close, but do not replace a much stronger watched symbol.
        if candidate_score + 25.0 < weakest_score:
            return False
        self.clear_watch_symbol(weakest)
        return True
