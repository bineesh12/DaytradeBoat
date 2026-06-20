"""CLI: baseline-vs-experiment sweep over the bot's real past universe.

Run on the box (Alpaca creds + journal there):

    python -m daytrading.backtest.sweep_cli \
        --journal /var/lib/daytrading-data/journal/journal.db \
        --dates 2026-06-10,2026-06-11,2026-06-12

Builds an unbiased symbol basket from the journal, runs ``run_backtest_sweep``,
and prints each experiment's P&L / expectancy delta vs baseline → keep/cut.
"""

from __future__ import annotations

import argparse
import json

from daytrading.backtest.batch import journal_universe
from daytrading.backtest.service import run_backtest_sweep


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline-vs-experiment backtest sweep.")
    parser.add_argument("--journal", required=True, help="journal.db for the unbiased universe")
    parser.add_argument("--dates", required=True, help="Comma-separated YYYY-MM-DD list")
    parser.add_argument("--max-per-day", type=int, default=60)
    parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    symbols, used_dates = journal_universe(args.journal, dates, max_per_day=args.max_per_day)
    if not symbols:
        print("No symbols found in journal for those dates.")
        return 1

    sweep = run_backtest_sweep(symbols, used_dates)

    if args.json:
        print(json.dumps(sweep, indent=2, default=str))
        return 0

    experiments = sweep.get("experiments", {})
    deltas = sweep.get("deltas_vs_baseline", {})
    print("Sweep: {} symbols x {} days\n".format(len(symbols), len(used_dates)))
    print("{:<26} {:>7} {:>8} {:>11} {:>10} {:>12}".format(
        "config", "trades", "win%", "expectancy", "P&L", "ΔP&L vs base"))
    print("-" * 78)
    for name, agg in experiments.items():
        sc = agg.get("scorecard", {})
        d = deltas.get(name, {})
        dp = d.get("total_pnl", 0.0)
        verdict = "baseline" if name == "baseline" else ("keep" if dp > 0 else "cut")
        print("{:<26} {:>7} {:>7.1f}% {:>11.2f} {:>10.2f} {:>10.2f}  {}".format(
            name, sc.get("closed_trades", 0), sc.get("win_rate", 0.0),
            sc.get("expectancy_per_trade", 0.0), sc.get("total_pnl", 0.0), dp, verdict))
    note = sweep.get("universe_note")
    if note:
        print("\n" + str(note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
