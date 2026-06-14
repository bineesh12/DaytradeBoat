from __future__ import annotations

import argparse
import json

from daytrading.backtest.data_loader import load_many_csv, parse_timestamp
from daytrading.backtest.driver import PipelineBacktestDriver


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay historical bars through the scalping pipeline.")
    parser.add_argument("--csv", action="append", required=True, help="CSV file with symbol,ts,open,high,low,close,volume")
    parser.add_argument("--initial-cash", type=float, default=25_000.0)
    parser.add_argument("--start", default=None, help="Optional ISO timestamp")
    parser.add_argument("--end", default=None, help="Optional ISO timestamp")
    parser.add_argument("--max-bars", type=int, default=120)
    args = parser.parse_args()

    bars = load_many_csv(args.csv)
    driver = PipelineBacktestDriver(
        bars,
        initial_cash=args.initial_cash,
        max_bars_per_symbol=args.max_bars,
    )
    result = driver.run(
        start=parse_timestamp(args.start) if args.start else None,
        end=parse_timestamp(args.end) if args.end else None,
    )
    print(json.dumps({
        "cycles": result.cycles,
        "fills": len(result.fills),
        "scorecard": result.scorecard,
        "final_cash": round(result.final_portfolio.cash, 2) if result.final_portfolio else None,
        "open_positions": {
            sym: {"quantity": pos.quantity, "avg_price": pos.avg_price}
            for sym, pos in (result.final_portfolio.positions if result.final_portfolio else {}).items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
