"""Nightly trade analyst — collects stats and generates a report after market close.

Runs automatically when the runner detects market close (4 PM ET)
on regular trading days (skips weekends and US market holidays).
Produces a structured report at data/reports/YYYY-MM-DD.json and a
human-readable summary at data/reports/YYYY-MM-DD.md that Cursor
can read next morning to propose code fixes.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from daytrading.market_calendar import is_us_market_holiday

logger = logging.getLogger(__name__)


def _is_market_holiday(d: date) -> bool:
    """Check if the given date is a weekend or US stock market holiday."""
    return is_us_market_holiday(d)


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


class NightlyAnalyst:
    """Collects trade data from the journal DB and produces a nightly report."""

    def __init__(self, db_path: str, report_dir: Optional[str] = None) -> None:
        self._db_path = db_path
        self._report_dir = report_dir or os.path.join(
            os.path.dirname(db_path), "..", "reports"
        )
        os.makedirs(self._report_dir, exist_ok=True)

    def run(self, day: Optional[str] = None) -> Dict[str, Any]:
        """Run the full nightly analysis for the given day (default: today)."""
        day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        day_date = date.fromisoformat(day)
        if _is_market_holiday(day_date):
            logger.info("NIGHTLY ANALYST: %s is a weekend/holiday — skipping", day)
            return {"day": day, "status": "holiday"}

        logger.info("NIGHTLY ANALYST: starting analysis for %s", day)

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row

        trades = self._load_trades(conn, day)
        scan_hits = self._load_scan_hits(conn, day)
        mistakes = self._load_mistakes(conn, day)
        cycles = self._load_cycles(conn, day)
        conn.close()

        if not trades:
            logger.info("NIGHTLY ANALYST: no trades for %s — skipping", day)
            return {"day": day, "status": "no_trades"}

        report = {
            "day": day,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": self._build_summary(trades),
            "trade_details": trades,
            "pattern_analysis": self._analyze_patterns(trades),
            "exit_analysis": self._analyze_exits(trades),
            "time_analysis": self._analyze_timing(trades),
            "risk_analysis": self._analyze_risk(trades),
            "scanner_analysis": self._analyze_scanners(scan_hits, trades),
            "rejection_analysis": self._analyze_rejections(mistakes),
            "problems": self._identify_problems(trades, mistakes),
            "cycle_stats": {
                "total_cycles": len(cycles),
                "total_scan_hits": sum(c.get("scan_hits", 0) for c in cycles),
                "total_signals": sum(c.get("signals", 0) for c in cycles),
            },
        }

        json_path = os.path.join(self._report_dir, f"{day}.json")
        md_path = os.path.join(self._report_dir, f"{day}.md")

        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        md_content = self._render_markdown(report)
        with open(md_path, "w") as f:
            f.write(md_content)

        logger.info("NIGHTLY ANALYST: report saved to %s", md_path)
        return report

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_trades(self, conn: sqlite3.Connection, day: str) -> List[Dict]:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM trades WHERE ts LIKE ? ORDER BY ts ASC",
            (f"{day}%",),
        )
        rows = cur.fetchall()
        trades = []
        for r in rows:
            trades.append({
                "symbol": r["symbol"],
                "side": r["side"],
                "trade_type": r["trade_type"],
                "strategy": r["strategy"],
                "quantity": r["quantity"],
                "entry_price": r["entry_price"],
                "exit_price": r["exit_price"],
                "pnl": r["pnl"],
                "reason": r["reason"],
                "ts": r["ts"],
            })
        return trades

    def _load_scan_hits(self, conn: sqlite3.Connection, day: str) -> List[Dict]:
        cur = conn.cursor()
        cur.execute(
            "SELECT payload_json FROM events WHERE day = ? AND type = 'scan_hit' ORDER BY ts ASC",
            (day,),
        )
        return [json.loads(r["payload_json"]) for r in cur.fetchall()]

    def _load_mistakes(self, conn: sqlite3.Connection, day: str) -> List[Dict]:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM mistakes WHERE ts LIKE ? ORDER BY ts ASC",
            (f"{day}%",),
        )
        return [dict(r) for r in cur.fetchall()]

    def _load_cycles(self, conn: sqlite3.Connection, day: str) -> List[Dict]:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM market_context WHERE ts LIKE ? ORDER BY ts ASC",
            (f"{day}%",),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _build_summary(self, trades: List[Dict]) -> Dict:
        entries = [t for t in trades if t["trade_type"] == "entry"]
        exits = [t for t in trades if t["trade_type"] == "exit" and t["pnl"] is not None]

        pnls = [t["pnl"] for t in exits]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls) if pnls else 0
        symbols_traded = list(set(t["symbol"] for t in entries))

        return {
            "total_entries": len(entries),
            "total_exits": len(exits),
            "total_pnl": round(total_pnl, 2),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": round(_safe_div(len(wins), len(exits)) * 100, 1),
            "avg_winner": round(_safe_div(sum(wins), len(wins)), 2) if wins else 0,
            "avg_loser": round(_safe_div(sum(losses), len(losses)), 2) if losses else 0,
            "largest_winner": round(max(wins), 2) if wins else 0,
            "largest_loser": round(min(losses), 2) if losses else 0,
            "profit_factor": round(abs(_safe_div(sum(wins), sum(losses))), 2) if losses else float("inf"),
            "symbols_traded": symbols_traded,
        }

    def _analyze_patterns(self, trades: List[Dict]) -> List[Dict]:
        """Win rate and P&L by pattern/strategy."""
        exits = [t for t in trades if t["trade_type"] == "exit" and t["pnl"] is not None]
        by_pattern: Dict[str, List[float]] = defaultdict(list)
        for t in exits:
            pat = t.get("strategy") or "unknown"
            pat = pat.strip().lower() or "unknown"
            by_pattern[pat].append(t["pnl"])

        results = []
        for pat, pnls in sorted(by_pattern.items()):
            wins = [p for p in pnls if p > 0]
            results.append({
                "pattern": pat,
                "trades": len(pnls),
                "wins": len(wins),
                "losses": len(pnls) - len(wins),
                "win_rate": round(_safe_div(len(wins), len(pnls)) * 100, 1),
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl": round(_safe_div(sum(pnls), len(pnls)), 2),
            })
        return results

    def _analyze_exits(self, trades: List[Dict]) -> List[Dict]:
        """Breakdown by exit reason."""
        exits = [t for t in trades if t["trade_type"] == "exit" and t["pnl"] is not None]
        by_reason: Dict[str, List[float]] = defaultdict(list)
        for t in exits:
            reason = t.get("reason") or "unknown"
            if "stop" in reason.lower():
                key = "stop_loss"
            elif "red candle" in reason.lower():
                key = "red_candle_exit"
            elif "stale" in reason.lower():
                key = "stale_exit"
            elif "range" in reason.lower():
                key = "range_exit"
            elif "target" in reason.lower() or "profit" in reason.lower():
                key = "take_profit"
            elif "extension" in reason.lower():
                key = "extension_exit"
            elif "half" in reason.lower():
                key = "half_sell"
            else:
                key = reason[:30] if reason else "unknown"
            by_reason[key].append(t["pnl"])

        results = []
        for reason, pnls in sorted(by_reason.items(), key=lambda x: sum(x[1])):
            results.append({
                "exit_type": reason,
                "count": len(pnls),
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl": round(_safe_div(sum(pnls), len(pnls)), 2),
            })
        return results

    def _analyze_timing(self, trades: List[Dict]) -> Dict:
        """Performance by time of day (ET hours)."""
        entries = [t for t in trades if t["trade_type"] == "entry"]
        exits_by_sym: Dict[str, Dict] = {}
        for t in trades:
            if t["trade_type"] == "exit" and t["pnl"] is not None:
                exits_by_sym[t["symbol"]] = t

        by_hour: Dict[int, List[float]] = defaultdict(list)
        for e in entries:
            try:
                ts = datetime.fromisoformat(e["ts"].replace("Z", "+00:00"))
                hour_et = (ts.hour - 4) % 24  # UTC to ET approximate
                sym_exit = exits_by_sym.get(e["symbol"])
                if sym_exit and sym_exit["pnl"] is not None:
                    by_hour[hour_et].append(sym_exit["pnl"])
            except Exception:
                continue

        hourly = []
        for hour in sorted(by_hour.keys()):
            pnls = by_hour[hour]
            wins = [p for p in pnls if p > 0]
            hourly.append({
                "hour_et": hour,
                "trades": len(pnls),
                "wins": len(wins),
                "win_rate": round(_safe_div(len(wins), len(pnls)) * 100, 1),
                "total_pnl": round(sum(pnls), 2),
            })
        return {"hourly": hourly}

    def _analyze_risk(self, trades: List[Dict]) -> Dict:
        """Risk metrics: avg risk per trade, R multiples."""
        exits = [t for t in trades if t["trade_type"] == "exit" and t["pnl"] is not None]
        if not exits:
            return {}

        entry_prices = {}
        for t in trades:
            if t["trade_type"] == "entry" and t["entry_price"]:
                entry_prices[t["symbol"]] = t["entry_price"]

        risk_data = []
        for t in exits:
            ep = entry_prices.get(t["symbol"], t.get("entry_price", 0))
            xp = t.get("exit_price", 0)
            qty = t.get("quantity", 0)
            if ep and xp and qty:
                pct_move = _safe_div(xp - ep, ep) * 100
                risk_data.append({
                    "symbol": t["symbol"],
                    "entry": ep,
                    "exit": xp,
                    "pct_move": round(pct_move, 2),
                    "pnl": t["pnl"],
                    "hold_approx": "see logs",
                })

        return {
            "trades": risk_data,
            "avg_pct_move": round(
                _safe_div(sum(r["pct_move"] for r in risk_data), len(risk_data)), 2
            ) if risk_data else 0,
        }

    def _analyze_scanners(self, scan_hits: List[Dict], trades: List[Dict]) -> Dict:
        """How many scan hits turned into trades, and which scanners are productive."""
        by_scanner: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "verified": 0})
        for h in scan_hits:
            name = h.get("scanner", "unknown")
            by_scanner[name]["hits"] += 1
            if h.get("verified"):
                by_scanner[name]["verified"] += 1

        return {
            "scanners": {
                name: {
                    "hits": d["hits"],
                    "verified": d["verified"],
                    "conversion_rate": round(_safe_div(d["verified"], d["hits"]) * 100, 1),
                }
                for name, d in sorted(by_scanner.items())
            },
            "total_scan_hits": len(scan_hits),
        }

    def _analyze_rejections(self, mistakes: List[Dict]) -> Dict:
        """Most common rejection reasons."""
        by_reason: Dict[str, int] = defaultdict(int)
        by_kind: Dict[str, int] = defaultdict(int)
        for m in mistakes:
            reason = m.get("reason", "unknown") or "unknown"
            kind = m.get("kind", "unknown") or "unknown"
            short_reason = reason[:60]
            by_reason[short_reason] += 1
            by_kind[kind] += 1

        top_reasons = sorted(by_reason.items(), key=lambda x: -x[1])[:15]
        return {
            "total_rejections": len(mistakes),
            "by_kind": dict(by_kind),
            "top_reasons": [{"reason": r, "count": c} for r, c in top_reasons],
        }

    def _identify_problems(self, trades: List[Dict], mistakes: List[Dict]) -> List[Dict]:
        """Auto-detect systemic problems from today's data."""
        problems = []
        exits = [t for t in trades if t["trade_type"] == "exit" and t["pnl"] is not None]
        entries = [t for t in trades if t["trade_type"] == "entry"]

        if not exits:
            return problems

        pnls = [t["pnl"] for t in exits]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        win_rate = _safe_div(len(wins), len(exits)) * 100

        # --- Build per-trade detail for the report ---
        entry_map: Dict[str, Dict] = {}
        for t in entries:
            entry_map[t["symbol"]] = t
        trade_details = []
        for t in exits:
            ent = entry_map.get(t["symbol"])
            hold_sec = None
            if ent:
                try:
                    e_ts = datetime.fromisoformat(ent["ts"].replace("Z", "+00:00"))
                    x_ts = datetime.fromisoformat(t["ts"].replace("Z", "+00:00"))
                    hold_sec = (x_ts - e_ts).total_seconds()
                except Exception:
                    pass
            trade_details.append({
                "symbol": t["symbol"],
                "entry_price": ent["entry_price"] if ent else None,
                "exit_price": t.get("exit_price"),
                "pnl": t["pnl"],
                "reason": t.get("reason") or "unknown",
                "strategy": t.get("strategy") or (ent.get("strategy") if ent else "unknown"),
                "hold_seconds": hold_sec,
            })

        # ---- CRITICAL: Zero win rate ----
        if len(wins) == 0 and len(exits) >= 3:
            problems.append({
                "severity": "CRITICAL",
                "problem": f"Every single trade lost money today",
                "what_happened": (
                    f"You took {len(exits)} trades and none of them were winners. "
                    f"Total loss: ${abs(total_pnl):.2f}."
                ),
                "why_it_matters": (
                    "A 0% win rate means either the bot is entering at the wrong time "
                    "(buying at the top instead of on pullbacks), the stop losses are too "
                    "tight (getting knocked out by normal price noise before the trade "
                    "has a chance to work), or the exit logic is selling too quickly."
                ),
                "what_to_check": [
                    "Look at each trade below — did the stock actually go up AFTER you got stopped out?",
                    "If yes → stops are too tight. The stop needs to be below a real support level.",
                    "If no → the entries are bad. The bot is buying stocks that are already extended.",
                    "Check if most exits say 'stop_loss' — that points to tight stops.",
                    "Check if most exits say 'red_candle' — that points to aggressive exit logic.",
                ],
                "file_to_fix": "momentum_pattern.py (stop placement) or entry_guard.py (entry filters)",
            })

        # ---- HIGH: Low win rate ----
        elif 0 < win_rate < 30 and len(exits) >= 5:
            problems.append({
                "severity": "HIGH",
                "problem": f"Very low win rate: only {len(wins)} winners out of {len(exits)} trades ({win_rate:.0f}%)",
                "what_happened": (
                    f"You won {len(wins)} and lost {len(losses)} trades. "
                    f"Total P&L: ${total_pnl:+.2f}."
                ),
                "why_it_matters": (
                    "Warrior Trading targets at least 50-60% win rate on momentum plays. "
                    "Below 30% means something is systematically wrong."
                ),
                "what_to_check": [
                    "Are stops being placed at real technical levels (below candle lows)?",
                    "Is the entry guard letting in low-quality setups?",
                    "Are we buying too far from VWAP (chasing)?",
                ],
                "file_to_fix": "entry_guard.py or momentum_pattern.py",
            })

        # ---- HIGH: Re-entering losers ----
        sym_entries = defaultdict(int)
        sym_pnl = defaultdict(float)
        for t in entries:
            sym_entries[t["symbol"]] += 1
        for t in exits:
            sym_pnl[t["symbol"]] += t["pnl"] or 0

        repeat_losers = [
            (sym, cnt, sym_pnl[sym])
            for sym, cnt in sym_entries.items()
            if cnt >= 2 and sym_pnl.get(sym, 0) < 0
        ]
        if repeat_losers:
            details_lines = []
            for s, c, p in repeat_losers:
                details_lines.append(f"  - {s}: bought {c} times, lost ${abs(p):.2f} total")
            problems.append({
                "severity": "HIGH",
                "problem": "Bot kept buying the same losing stock multiple times",
                "what_happened": (
                    "These stocks were bought, lost money, and then the bot bought them AGAIN:\n"
                    + "\n".join(details_lines)
                ),
                "why_it_matters": (
                    "Warrior Trading rule: if a stock hits your stop, move on. "
                    "Don't revenge-trade the same ticker. Every re-entry on a loser "
                    "adds to the loss."
                ),
                "what_to_check": [
                    "The daily loser blacklist should prevent re-entries on stocks that already lost money today.",
                    "Check if the blacklist is active and working.",
                ],
                "file_to_fix": "runner.py (daily loser blacklist logic)",
            })

        # ---- HIGH: Mostly stopped out ----
        stop_exits = [t for t in exits if "stop" in (t.get("reason") or "").lower()]
        if len(stop_exits) >= len(exits) * 0.8 and len(exits) >= 3:
            problems.append({
                "severity": "HIGH",
                "problem": f"Almost every trade was stopped out ({len(stop_exits)} out of {len(exits)})",
                "what_happened": (
                    f"{len(stop_exits)} of {len(exits)} trades hit their stop loss. "
                    "That means stops are probably placed too close to the entry."
                ),
                "why_it_matters": (
                    "When 80%+ of trades are stop-outs, it usually means the stop is within "
                    "the normal noise range of the stock. A $5 stock can easily wiggle $0.10-0.15 "
                    "in a minute — if your stop is $0.05 below entry, you'll get stopped on noise."
                ),
                "what_to_check": [
                    "Look at the stop prices — are they below a real support level (candle low, VWAP)?",
                    "Or are they just a small fixed amount below entry?",
                    "The stop should be under the pattern low, not an arbitrary number.",
                ],
                "file_to_fix": "momentum_pattern.py (stop calculation section)",
            })

        # ---- HIGH: Very short hold times ----
        quick_exits = 0
        for td in trade_details:
            if td["hold_seconds"] is not None and td["hold_seconds"] < 60:
                quick_exits += 1
        if quick_exits >= len(exits) * 0.5 and len(exits) >= 3:
            problems.append({
                "severity": "HIGH",
                "problem": f"Most trades were held less than 1 minute ({quick_exits} out of {len(exits)})",
                "what_happened": (
                    f"{quick_exits} trades were exited in under 60 seconds. "
                    "Momentum trades need at least 2-5 minutes to play out."
                ),
                "why_it_matters": (
                    "If a trade is exited within seconds, the exit logic is panicking. "
                    "Warrior Trading says give the trade time to work unless it clearly breaks the pattern."
                ),
                "what_to_check": [
                    "Is the 'red candle exit' firing too early?",
                    "The minimum hold time should be at least 120 seconds before any red candle exit.",
                    "Check if stops are so tight that they trigger on the first tiny dip.",
                ],
                "file_to_fix": "exits/manager.py (red candle exit and minimum hold time)",
            })

        # ---- MEDIUM: Afternoon losses ----
        afternoon_pnl = 0.0
        afternoon_count = 0
        for t in exits:
            try:
                ts = datetime.fromisoformat(t["ts"].replace("Z", "+00:00"))
                hour_et = (ts.hour - 4) % 24
                if hour_et >= 12:
                    afternoon_pnl += t["pnl"] or 0
                    afternoon_count += 1
            except Exception:
                pass
        if afternoon_count >= 3 and afternoon_pnl < -50:
            problems.append({
                "severity": "MEDIUM",
                "problem": f"Afternoon trading is losing money (${afternoon_pnl:+.2f} from {afternoon_count} trades)",
                "what_happened": (
                    f"After 12 PM ET, the bot took {afternoon_count} trades and "
                    f"lost ${abs(afternoon_pnl):.2f}. Morning momentum is usually "
                    f"much stronger than afternoon."
                ),
                "why_it_matters": (
                    "Warrior Trading focuses on the 9:30-11:30 AM ET window. "
                    "Afternoon volume drops and setups become unreliable."
                ),
                "what_to_check": [
                    "Consider stopping new entries after 11:30 AM or 12:00 PM ET.",
                    "If you keep afternoon trading, use tighter filters (higher RVOL requirement).",
                ],
                "file_to_fix": "runner.py (trading window cutoff time)",
            })

        # ---- MEDIUM: Specific pattern failing ----
        pat_pnls: Dict[str, List[float]] = defaultdict(list)
        for t in exits:
            pat = t.get("strategy") or "unknown"
            pat_pnls[pat].append(t["pnl"] or 0)
        for pat, plist in pat_pnls.items():
            if len(plist) >= 2 and all(p <= 0 for p in plist):
                problems.append({
                    "severity": "MEDIUM",
                    "problem": f"The '{pat}' pattern lost on every single trade today",
                    "what_happened": (
                        f"'{pat}' was used for {len(plist)} trades and all of them lost money "
                        f"(total: ${sum(plist):+.2f})."
                    ),
                    "why_it_matters": (
                        "If a specific pattern is consistently losing, either the scanner "
                        "is detecting false signals, or the market conditions don't suit this pattern today."
                    ),
                    "what_to_check": [
                        f"Review the scanner thresholds for '{pat}' — are they too loose?",
                        f"Consider temporarily disabling '{pat}' until the thresholds are tuned.",
                        "Check if today's market was choppy/low-volume (bad for momentum patterns).",
                    ],
                    "file_to_fix": f"scanner/scalping/ (the {pat} scanner file)",
                })

        # ---- INFO: Largest single loss ----
        if losses:
            worst = min(losses)
            worst_trade = next(
                (td for td in trade_details if td["pnl"] == worst), None
            )
            if worst_trade and abs(worst) > 50:
                problems.append({
                    "severity": "INFO",
                    "problem": f"Largest single loss was ${abs(worst):.2f} on {worst_trade['symbol']}",
                    "what_happened": (
                        f"{worst_trade['symbol']}: bought at ${worst_trade['entry_price']}, "
                        f"exited at ${worst_trade['exit_price']} (reason: {worst_trade['reason']}). "
                        f"Lost ${abs(worst):.2f}."
                    ),
                    "why_it_matters": (
                        "Per Warrior Trading, max loss per trade should be around $100. "
                        "If a single trade loses significantly more, position sizing or stop logic needs review."
                    ),
                    "what_to_check": [
                        "Was the position size too large for the risk?",
                        "Did slippage cause the stop to fill far from the intended price?",
                    ],
                    "file_to_fix": "momentum_pattern.py (position sizing) or exits/manager.py (slippage handling)",
                })

        # Store trade details in each problem for the markdown renderer
        for p in problems:
            p["_trade_details"] = trade_details

        return problems

    # ------------------------------------------------------------------
    # Report rendering
    # ------------------------------------------------------------------

    def _render_markdown(self, report: Dict) -> str:
        """Render a clear, human-readable report anyone can understand."""
        s = report["summary"]
        day = report["day"]
        lines = []

        # ============================================================
        # HEADER — Quick glance
        # ============================================================
        lines.append(f"# Trading Report — {day}")
        lines.append("")

        # Overall verdict
        if s["total_pnl"] > 0:
            verdict = f"PROFITABLE DAY (+${s['total_pnl']:.2f})"
        elif s["total_pnl"] == 0:
            verdict = "BREAK EVEN"
        else:
            verdict = f"LOSING DAY (-${abs(s['total_pnl']):.2f})"
        lines.append(f"> **{verdict}** — {s['win_count']} wins, {s['loss_count']} losses "
                      f"({s['win_rate']}% win rate)")
        lines.append("")

        # ============================================================
        # TODAY AT A GLANCE
        # ============================================================
        lines.append("## Today at a Glance")
        lines.append("")
        lines.append(f"| What | Value |")
        lines.append(f"|------|-------|")
        lines.append(f"| Total trades | {s['total_entries']} entries, {s['total_exits']} exits |")
        lines.append(f"| Net P&L | **${s['total_pnl']:+.2f}** |")
        lines.append(f"| Win rate | {s['win_rate']}% ({s['win_count']}W / {s['loss_count']}L) |")
        lines.append(f"| Average winning trade | ${s['avg_winner']:+.2f} |")
        lines.append(f"| Average losing trade | ${s['avg_loser']:+.2f} |")
        lines.append(f"| Best trade | ${s['largest_winner']:+.2f} |")
        lines.append(f"| Worst trade | ${s['largest_loser']:+.2f} |")
        lines.append(f"| Profit factor | {s['profit_factor']} |")
        lines.append(f"| Stocks traded | {', '.join(s['symbols_traded'])} |")
        lines.append("")

        # ============================================================
        # TRADE-BY-TRADE LOG
        # ============================================================
        problems = report.get("problems", [])
        trade_details = problems[0].get("_trade_details", []) if problems else []
        if trade_details:
            lines.append("## Every Trade Today")
            lines.append("")
            lines.append("| # | Stock | Entry | Exit | P&L | Hold Time | Exit Reason | Verdict |")
            lines.append("|---|-------|-------|------|-----|-----------|-------------|---------|")
            for i, td in enumerate(trade_details, 1):
                hold = ""
                if td["hold_seconds"] is not None:
                    m, sec = divmod(int(td["hold_seconds"]), 60)
                    hold = f"{m}m {sec}s"
                pnl = td["pnl"] or 0
                icon = "WIN" if pnl > 0 else "LOSS"
                entry_str = f"${td['entry_price']:.2f}" if td["entry_price"] else "?"
                exit_str = f"${td['exit_price']:.2f}" if td["exit_price"] else "?"
                reason_short = (td["reason"] or "")[:25]
                lines.append(
                    f"| {i} | {td['symbol']} | {entry_str} | {exit_str} | "
                    f"${pnl:+.2f} | {hold} | {reason_short} | {icon} |"
                )
            lines.append("")

        # ============================================================
        # PROBLEMS — The most important section
        # ============================================================
        if problems:
            lines.append("---")
            lines.append("")
            lines.append("## Issues Found Today")
            lines.append("")
            lines.append("These are the problems I detected by analyzing your trades. "
                          "Read each one carefully.")
            lines.append("")

            for i, p in enumerate(problems, 1):
                sev = p["severity"]
                if sev == "CRITICAL":
                    sev_label = "CRITICAL"
                elif sev == "HIGH":
                    sev_label = "IMPORTANT"
                elif sev == "MEDIUM":
                    sev_label = "WORTH CHECKING"
                else:
                    sev_label = "FYI"

                lines.append(f"### Issue {i}: {p['problem']} [{sev_label}]")
                lines.append("")

                if "what_happened" in p:
                    lines.append(f"**What happened:** {p['what_happened']}")
                    lines.append("")

                if "why_it_matters" in p:
                    lines.append(f"**Why this matters:** {p['why_it_matters']}")
                    lines.append("")

                checks = p.get("what_to_check", [])
                if checks:
                    lines.append("**What to check:**")
                    for c in checks:
                        lines.append(f"- {c}")
                    lines.append("")

                if "file_to_fix" in p:
                    lines.append(f"**File to look at:** `{p['file_to_fix']}`")
                    lines.append("")

        else:
            lines.append("---")
            lines.append("")
            lines.append("## No Major Issues Found")
            lines.append("")
            lines.append("Everything looks okay today. Keep it up!")
            lines.append("")

        # ============================================================
        # PATTERN PERFORMANCE — Which patterns are working
        # ============================================================
        patterns = report.get("pattern_analysis", [])
        if patterns:
            lines.append("---")
            lines.append("")
            lines.append("## Which Patterns Worked?")
            lines.append("")
            lines.append("This shows how each pattern (Bull Flag, ORB, etc.) performed today.")
            lines.append("")
            lines.append("| Pattern | Trades | Wins | Losses | Win Rate | Total P&L |")
            lines.append("|---------|--------|------|--------|----------|-----------|")
            for p in patterns:
                lines.append(
                    f"| {p['pattern']} | {p['trades']} | {p['wins']} | {p['losses']} | "
                    f"{p['win_rate']}% | ${p['total_pnl']:+.2f} |"
                )
            lines.append("")

        # ============================================================
        # EXIT ANALYSIS — How trades ended
        # ============================================================
        exits = report.get("exit_analysis", [])
        if exits:
            lines.append("## How Did Trades End?")
            lines.append("")
            lines.append("This shows what caused each trade to close "
                          "(stop loss, take profit, red candle, etc.).")
            lines.append("")
            lines.append("| Exit Reason | Count | Total P&L | Avg P&L |")
            lines.append("|-------------|-------|-----------|---------|")
            for e in exits:
                lines.append(
                    f"| {e['exit_type']} | {e['count']} | "
                    f"${e['total_pnl']:+.2f} | ${e['avg_pnl']:+.2f} |"
                )
            lines.append("")

        # ============================================================
        # TIME ANALYSIS
        # ============================================================
        timing = report.get("time_analysis", {})
        hourly = timing.get("hourly", [])
        if hourly:
            lines.append("## What Time of Day Was Best?")
            lines.append("")
            lines.append("| Hour (ET) | Trades | Wins | Win Rate | P&L |")
            lines.append("|-----------|--------|------|----------|-----|")
            for h in hourly:
                hour_label = f"{h['hour_et']}:00-{h['hour_et']+1}:00"
                lines.append(
                    f"| {hour_label} | {h['trades']} | {h['wins']} | "
                    f"{h['win_rate']}% | ${h['total_pnl']:+.2f} |"
                )
            lines.append("")

        # ============================================================
        # REJECTIONS — What the bot decided NOT to trade
        # ============================================================
        rej = report.get("rejection_analysis", {})
        top_reasons = rej.get("top_reasons", [])
        total_rej = rej.get("total_rejections", 0)
        if top_reasons:
            lines.append("## What Did the Bot Reject?")
            lines.append("")
            lines.append(f"The bot rejected {total_rej} potential trades today. "
                          "Top reasons:")
            lines.append("")
            for r in top_reasons[:10]:
                lines.append(f"- **{r['count']}x** — {r['reason']}")
            lines.append("")

        # ============================================================
        # SCANNER STATS
        # ============================================================
        scanners = report.get("scanner_analysis", {}).get("scanners", {})
        if scanners:
            lines.append("## Scanner Stats")
            lines.append("")
            lines.append("How many stocks each scanner found vs. how many passed verification.")
            lines.append("")
            lines.append("| Scanner | Found | Verified | Pass Rate |")
            lines.append("|---------|-------|----------|-----------|")
            for name, d in scanners.items():
                lines.append(
                    f"| {name} | {d['hits']} | {d['verified']} | {d['conversion_rate']}% |"
                )
            lines.append("")

        # ============================================================
        # FOOTER — Instructions
        # ============================================================
        lines.append("---")
        lines.append("")
        lines.append("## What To Do Next")
        lines.append("")
        if problems:
            lines.append("1. Open Cursor (if not already open)")
            lines.append(f'2. Say: **"Read the report at data/reports/{day}.md and fix the issues"**')
            lines.append("3. Cursor will read this file, understand the problems, and propose code changes")
            lines.append("4. Review the changes before applying them")
        else:
            lines.append("No action needed. The system performed within acceptable parameters.")
        lines.append("")

        return "\n".join(lines)
