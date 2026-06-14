from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from daytrading.dashboard.hub import _daily_scorecard
from daytrading.models import Fill, Side


@dataclass
class BacktestLedger:
    """Converts fills into dashboard-compatible trade rows."""

    trades: List[Dict[str, Any]] = field(default_factory=list)
    _open_entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record_entry(self, fill: Fill, *, strategy: str = "") -> None:
        row = {
            "symbol": fill.symbol,
            "side": fill.side.value,
            "quantity": fill.quantity,
            "entry_price": fill.price,
            "entry_time": fill.ts.isoformat(),
            "exit_price": None,
            "exit_time": None,
            "pnl": None,
            "exit_reason": None,
            "trade_type": "entry",
            "strategy": strategy,
        }
        self.trades.append(row)
        existing = self._open_entries.get(fill.symbol)
        if existing:
            # Scale-up / reentry: blend into a volume-weighted average cost and
            # accumulate entry-side commission so the eventual exit PnL is
            # computed against the true basis, not just the last add.
            old_qty = float(existing.get("quantity") or 0.0)
            new_qty = old_qty + fill.quantity
            if new_qty > 0:
                existing["entry_price"] = (
                    old_qty * float(existing.get("entry_price") or fill.price)
                    + fill.quantity * fill.price
                ) / new_qty
            existing["quantity"] = new_qty
            existing["commission"] = float(existing.get("commission") or 0.0) + fill.commission
        else:
            self._open_entries[fill.symbol] = {
                "quantity": fill.quantity,
                "entry_price": fill.price,
                "strategy": strategy,
                "commission": fill.commission,
            }

    def record_exit(self, fill: Fill, *, reason: str = "") -> None:
        entry = self._open_entries.get(fill.symbol) or {}
        entry_price = float(entry.get("entry_price") or fill.price)
        strategy = str(entry.get("strategy") or "")
        open_qty = float(entry.get("quantity") or fill.quantity)
        # Charge the entry-side commission for the shares being closed (a round
        # trip pays commission on both legs); leave the rest with the position.
        entry_commission_total = float(entry.get("commission") or 0.0)
        if open_qty > 0:
            entry_commission_share = entry_commission_total * (fill.quantity / open_qty)
        else:
            entry_commission_share = entry_commission_total
        pnl = (fill.price - entry_price) * fill.quantity
        if fill.side is Side.BUY:
            pnl = -pnl
        self.trades.append({
            "symbol": fill.symbol,
            "side": fill.side.value,
            "quantity": fill.quantity,
            "entry_price": entry_price,
            "entry_time": fill.ts.isoformat(),
            "exit_price": fill.price,
            "exit_time": fill.ts.isoformat(),
            "pnl": round(pnl - fill.commission - entry_commission_share, 2),
            "exit_reason": reason,
            "trade_type": "exit",
            "strategy": strategy,
        })
        remaining = open_qty - fill.quantity
        if remaining > 0:
            entry["quantity"] = remaining
            entry["commission"] = entry_commission_total - entry_commission_share
            self._open_entries[fill.symbol] = entry
        else:
            self._open_entries.pop(fill.symbol, None)


def build_backtest_scorecard(
    *,
    trades: List[Dict[str, Any]],
    total_scan_hits: int,
    total_signals: int,
    total_rejected: int,
    cycle_count: int,
    missed_a_plus: List[dict],
    total_deferred: int = 0,
    rejected_by_layer: Optional[Dict[str, int]] = None,
    rejected_reasons_by_layer: Optional[Dict[str, List[dict]]] = None,
) -> dict:
    scorecard = _daily_scorecard(
        trades=trades,
        total_trades=sum(1 for t in trades if t.get("trade_type") == "entry"),
        total_scan_hits=total_scan_hits,
        total_signals=total_signals,
        total_rejected=total_rejected,
        cycle_count=cycle_count,
        missed_a_plus=missed_a_plus,
    )
    funnel = dict(scorecard.get("funnel") or {})
    funnel["deferred"] = int(total_deferred or 0)
    funnel["rejected_by_layer"] = dict(rejected_by_layer or {})
    funnel["top_reject_reasons_by_layer"] = dict(rejected_reasons_by_layer or {})
    scorecard["funnel"] = funnel
    return scorecard
