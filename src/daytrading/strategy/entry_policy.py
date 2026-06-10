"""Single owner for final entry-policy decisions.

Scanners and pattern verifiers still own pattern discovery. This module owns the
shared final question: can this setup become an order right now?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Sequence

from daytrading.models import Bar, Quote, SignalAction, Tick, TradeSignal
from daytrading.strategy.entry_guard import check_entry_quality


@dataclass(frozen=True)
class EntryDecision:
    symbol: str
    stage: str
    passed: bool
    blocked_layer: str = ""
    reason: str = ""
    action: str = ""
    pattern: str = ""
    scanner: str = ""
    setup_tier: str = ""
    entry_tier: str = ""
    price: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def reject_reason(self) -> Optional[str]:
        return None if self.passed else self.reason

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "symbol": self.symbol,
            "stage": self.stage,
            "passed": self.passed,
            "blocked_layer": self.blocked_layer,
            "reason": self.reason,
            "action": self.action,
            "pattern": self.pattern,
            "scanner": self.scanner,
            "setup_tier": self.setup_tier,
            "entry_tier": self.entry_tier,
            "price": self.price,
            "metadata": dict(self.metadata),
        }
        return payload


class EntryPolicy:
    """Centralized final entry policy used by live order paths."""

    ENTRY_ACTIONS = {
        SignalAction.ENTER_LONG,
        SignalAction.REENTER_LONG,
        SignalAction.SCALE_UP_LONG,
    }

    def __init__(
        self,
        guard: Optional[Callable[..., Optional[str]]] = None,
    ) -> None:
        self._guard = guard or check_entry_quality

    @staticmethod
    def _signal_context(signal: Optional[TradeSignal]) -> Dict[str, Any]:
        if signal is None:
            return {
                "symbol": "",
                "action": "",
                "pattern": "",
                "scanner": "",
                "setup_tier": "",
                "entry_tier": "",
                "price": 0.0,
            }
        hit = signal.scan_result
        criteria = hit.criteria if hit is not None else {}
        scanner = str(hit.scanner_name or "") if hit is not None else ""
        pattern = str(criteria.get("pattern") or scanner or "")
        return {
            "symbol": signal.symbol,
            "action": signal.action.value,
            "pattern": pattern,
            "scanner": scanner,
            "setup_tier": str(criteria.get("setup_tier") or ""),
            "entry_tier": str(criteria.get("entry_tier") or ""),
            "price": float(signal.entry_price or 0.0),
        }

    def decision(
        self,
        *,
        symbol: str,
        stage: str,
        passed: bool,
        reason: str = "",
        blocked_layer: str = "",
        signal: Optional[TradeSignal] = None,
        price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EntryDecision:
        ctx = self._signal_context(signal)
        symbol = symbol or str(ctx["symbol"])
        meta = dict(metadata or {})
        if reason and "entry_score" not in meta:
            match = re.search(r"(\d+)/100", str(reason))
            if match:
                meta["entry_score"] = int(match.group(1))
        return EntryDecision(
            symbol=symbol,
            stage=stage,
            passed=passed,
            blocked_layer="" if passed else (blocked_layer or stage),
            reason="" if passed else str(reason or "rejected"),
            action=str(ctx["action"]),
            pattern=str(ctx["pattern"]),
            scanner=str(ctx["scanner"]),
            setup_tier=str(ctx["setup_tier"]),
            entry_tier=str(ctx["entry_tier"]),
            price=float(price if price is not None else ctx["price"] or 0.0),
            metadata=meta,
        )

    def evaluate(
        self,
        signal: TradeSignal,
        *,
        bars: Sequence[Bar],
        stage: str,
        quotes: Optional[Sequence[Quote]] = None,
        ticks: Optional[Sequence[Tick]] = None,
        bars_5m: Optional[Sequence[Bar]] = None,
        avg_daily_volume: Optional[float] = None,
        float_shares: Optional[float] = None,
        min_day_change_pct: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EntryDecision:
        """Evaluate the shared rule/ML entry gate and return a decision."""
        if signal.action not in self.ENTRY_ACTIONS:
            return self.decision(
                symbol=signal.symbol,
                stage=stage,
                passed=True,
                signal=signal,
                metadata={"skipped": "non-entry action", **dict(metadata or {})},
            )
        ctx = self._signal_context(signal)
        pattern = str(ctx["pattern"])
        reason = self._guard(
            bars,
            symbol=signal.symbol,
            min_day_change_pct=min_day_change_pct,
            avg_daily_volume=avg_daily_volume,
            bars_5m=bars_5m,
            float_shares=float_shares,
            ticks=ticks,
            quotes=quotes,
            entry_pattern=pattern,
            setup_tier=str(ctx["setup_tier"]),
            entry_tier=str(ctx["entry_tier"]),
        )
        return self.decision(
            symbol=signal.symbol,
            stage=stage,
            passed=reason is None,
            reason=reason or "",
            blocked_layer="entry_guard",
            signal=signal,
            metadata=metadata,
        )
