"""AI Trade Analyzer — learns from completed trades and adapts behavior.

Runs periodically (every N cycles) to:
  1. Analyze recent trade outcomes by symbol, time-of-day, exit type, scanner
  2. Detect losing patterns (e.g. "NOK always stale-exits", "afternoon trades lose")
  3. Generate actionable suggestions shown on the dashboard
  4. Auto-tune parameters: cooldown, step size, momentum threshold, position size

This is a rules-based AI engine — no external API needed, runs locally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str
    entry_time: str
    exit_time: str
    scanner: str = ""
    hold_seconds: float = 0.0


@dataclass
class Insight:
    severity: str       # "critical", "warning", "info", "positive"
    category: str       # "symbol", "timing", "exit", "scanner", "risk"
    message: str
    action_taken: str   # what was auto-adjusted, or "" if suggestion only
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")


@dataclass
class AnalysisResult:
    insights: List[Insight] = field(default_factory=list)
    blocked_symbols: List[str] = field(default_factory=list)
    adjusted_params: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0  # overall session quality 0-100


class TradeAnalyzer:
    """Analyzes trade history and generates insights + auto-adjustments."""

    def __init__(self, min_trades: int = 5, max_block_trades: int = 3) -> None:
        self._min_trades = min_trades
        self._max_block_trades = max_block_trades
        self._blocked_symbols: Dict[str, str] = {}  # symbol -> reason
        self._insights: List[Insight] = []
        self._last_analysis: Optional[datetime] = None
        self._session_adjustments: Dict[str, float] = {}
        self._session_start: datetime = datetime.now(timezone.utc)

    @property
    def blocked_symbols(self) -> Dict[str, str]:
        return dict(self._blocked_symbols)

    @property
    def insights(self) -> List[Insight]:
        return list(self._insights)

    def is_blocked(self, symbol: str) -> bool:
        return symbol in self._blocked_symbols

    def analyze(self, trades: List[TradeRecord]) -> AnalysisResult:
        """Run full analysis on trade history. Returns insights and actions."""
        exits = [t for t in trades if t.exit_price > 0 and t.pnl is not None]
        result = AnalysisResult()

        if len(exits) < self._min_trades:
            return result

        self._insights.clear()

        self._analyze_symbols(exits, result)
        self._analyze_exit_types(exits, result)
        self._analyze_win_streaks(exits, result)
        self._analyze_risk_reward(exits, result)
        self._analyze_position_sizing(exits, result)
        self._calculate_session_score(exits, result)

        self._insights = list(result.insights)
        self._last_analysis = datetime.now(timezone.utc)

        logger.info(
            "Trade analysis: %d trades, %d insights, %d blocked symbols, score=%.0f/100",
            len(exits), len(result.insights), len(result.blocked_symbols), result.score,
        )
        for ins in result.insights:
            logger.info("AI INSIGHT [%s] %s: %s%s",
                        ins.severity.upper(), ins.category,
                        ins.message,
                        f" → {ins.action_taken}" if ins.action_taken else "")

        for k, v in result.adjusted_params.items():
            try:
                self._session_adjustments[k] = float(v)
            except (TypeError, ValueError):
                pass

        return result

    def reset_blocks(self) -> None:
        """Clear all blocked symbols — call on each new session/restart."""
        self._blocked_symbols.clear()
        self._session_start = datetime.now(timezone.utc)

    def _analyze_symbols(self, trades: List[TradeRecord], result: AnalysisResult) -> None:
        """Find symbols that consistently lose money TODAY — blocks reset on restart."""
        by_sym: Dict[str, List[TradeRecord]] = {}
        for t in trades:
            by_sym.setdefault(t.symbol, []).append(t)

        for sym, sym_trades in by_sym.items():
            if len(sym_trades) < 2:
                continue

            wins = sum(1 for t in sym_trades if t.pnl >= 0)
            total_pnl = sum(t.pnl for t in sym_trades)
            win_rate = wins / len(sym_trades) if sym_trades else 0

            # Count consecutive recent losses for this symbol
            consecutive_losses = 0
            for t in reversed(sym_trades):
                if t.pnl < 0:
                    consecutive_losses += 1
                else:
                    break

            # Block only if 3+ consecutive losses TODAY on this symbol
            if consecutive_losses >= self._max_block_trades and total_pnl < -15:
                self._blocked_symbols[sym] = "{}x consecutive losses today, P&L ${:.2f}".format(
                    consecutive_losses, total_pnl)
                result.blocked_symbols.append(sym)
                result.insights.append(Insight(
                    severity="critical",
                    category="symbol",
                    message="{}: {} consecutive losses, P&L ${:.2f} — BLOCKED FOR TODAY".format(
                        sym, consecutive_losses, total_pnl),
                    action_taken="Blocked {} for rest of session".format(sym),
                ))
            elif win_rate < 0.4 and total_pnl < -10 and len(sym_trades) >= 3:
                result.insights.append(Insight(
                    severity="warning",
                    category="symbol",
                    message="{}: {}/{} wins ({:.0f}%), total P&L ${:.2f} — underperforming today".format(
                        sym, wins, len(sym_trades), win_rate * 100, total_pnl),
                    action_taken="",
                ))
            elif win_rate >= 0.6 and total_pnl > 10:
                result.insights.append(Insight(
                    severity="positive",
                    category="symbol",
                    message="{}: {}/{} wins ({:.0f}%), total P&L +${:.2f} — strong performer".format(
                        sym, wins, len(sym_trades), win_rate * 100, total_pnl),
                    action_taken="",
                ))

    def _analyze_exit_types(self, trades: List[TradeRecord], result: AnalysisResult) -> None:
        """Analyze which exit types are profitable."""
        by_exit: Dict[str, List[TradeRecord]] = {}
        for t in trades:
            reason = t.exit_reason or "unknown"
            by_exit.setdefault(reason, []).append(t)

        for reason, exit_trades in by_exit.items():
            if len(exit_trades) < 2:
                continue

            total_pnl = sum(t.pnl for t in exit_trades)
            avg_pnl = total_pnl / len(exit_trades)
            wins = sum(1 for t in exit_trades if t.pnl >= 0)
            win_rate = wins / len(exit_trades)

            if "stop_loss" in reason.lower() and len(exit_trades) > 5:
                if win_rate == 0:
                    result.insights.append(Insight(
                        severity="critical",
                        category="exit",
                        message="Stop losses: {:.0f}% hit rate, avg loss ${:.2f} — stops may be too tight".format(
                            (1 - win_rate) * 100, avg_pnl),
                        action_taken="Consider widening stop from 5 to 7 ticks",
                    ))

            if "stale" in reason.lower():
                stale_pct = len(exit_trades) / len(trades) * 100
                if stale_pct > 50:
                    result.insights.append(Insight(
                        severity="warning",
                        category="exit",
                        message="Stale exits are {:.0f}% of all trades (avg P&L ${:.2f}) — entries lack momentum".format(
                            stale_pct, avg_pnl),
                        action_taken="Tighten momentum quality threshold",
                    ))
                    result.adjusted_params["min_momentum_quality"] = 50

            if "trailing" in reason.lower() and avg_pnl > 0:
                result.insights.append(Insight(
                    severity="positive",
                    category="exit",
                    message="Trailing stops working well: {:.0f}% win rate, avg +${:.2f}".format(
                        win_rate * 100, avg_pnl),
                    action_taken="",
                ))

    def _analyze_win_streaks(self, trades: List[TradeRecord], result: AnalysisResult) -> None:
        """Detect losing streaks and suggest pausing."""
        recent = trades[-10:]
        losses = sum(1 for t in recent if t.pnl < 0)
        recent_pnl = sum(t.pnl for t in recent)

        if losses >= 7:
            result.insights.append(Insight(
                severity="critical",
                category="risk",
                message="Losing streak: {}/10 recent trades are losses (${:.2f})".format(
                    losses, recent_pnl),
                action_taken="Reduce position size to 250 shares",
            ))
            result.adjusted_params["position_size"] = 250

        consecutive_losses = 0
        for t in reversed(trades):
            if t.pnl < 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 4:
            result.insights.append(Insight(
                severity="warning",
                category="risk",
                message="{} consecutive losses — market conditions may be unfavorable".format(
                    consecutive_losses),
                action_taken="Increased cooldown to 300s",
            ))
            result.adjusted_params["cooldown_seconds"] = 300

    def _analyze_risk_reward(self, trades: List[TradeRecord], result: AnalysisResult) -> None:
        """Check if actual R:R matches target R:R."""
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]

        if not wins or not losses:
            return

        avg_win = sum(t.pnl for t in wins) / len(wins)
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses))

        if avg_loss > 0:
            actual_rr = avg_win / avg_loss
            if actual_rr < 1.0:
                result.insights.append(Insight(
                    severity="critical",
                    category="risk",
                    message="Actual R:R is 1:{:.1f} (avg win ${:.2f} vs avg loss ${:.2f}) — need better entries".format(
                        actual_rr, avg_win, avg_loss),
                    action_taken="",
                ))
            elif actual_rr >= 1.5:
                result.insights.append(Insight(
                    severity="positive",
                    category="risk",
                    message="Good R:R of 1:{:.1f} (avg win ${:.2f} vs avg loss ${:.2f})".format(
                        actual_rr, avg_win, avg_loss),
                    action_taken="",
                ))

    def _analyze_position_sizing(self, trades: List[TradeRecord], result: AnalysisResult) -> None:
        """Check if position sizing is appropriate for account."""
        total_pnl = sum(t.pnl for t in trades)
        max_drawdown = 0.0
        running = 0.0
        peak = 0.0
        for t in trades:
            running += t.pnl
            peak = max(peak, running)
            dd = peak - running
            max_drawdown = max(max_drawdown, dd)

        if max_drawdown > 500:
            result.insights.append(Insight(
                severity="warning",
                category="risk",
                message="Max drawdown ${:.2f} — consider reducing position size".format(max_drawdown),
                action_taken="",
            ))

    def _calculate_session_score(self, trades: List[TradeRecord], result: AnalysisResult) -> None:
        """Score the session quality 0-100."""
        score = 50.0  # start neutral

        wins = sum(1 for t in trades if t.pnl >= 0)
        win_rate = wins / len(trades) if trades else 0
        score += (win_rate - 0.5) * 40  # ±20 points for win rate

        total_pnl = sum(t.pnl for t in trades)
        if total_pnl > 0:
            score += min(20, total_pnl / 10)
        else:
            score += max(-20, total_pnl / 10)

        criticals = sum(1 for i in result.insights if i.severity == "critical")
        score -= criticals * 5

        positives = sum(1 for i in result.insights if i.severity == "positive")
        score += positives * 3

        result.score = max(0, min(100, score))

    def snapshot(self) -> dict:
        """Return serializable state for the dashboard."""
        return {
            "insights": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                    "action_taken": i.action_taken,
                    "timestamp": i.timestamp,
                }
                for i in self._insights
            ],
            "blocked_symbols": dict(self._blocked_symbols),
            "session_adjustments": dict(self._session_adjustments),
            "last_analysis": self._last_analysis.strftime("%H:%M:%S") if self._last_analysis else None,
        }
