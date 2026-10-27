"""Explicit paper-only entry point: config checks, read-only preflight, or run."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import dotenv_values


def load_paper_config(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(
            f"Create {path} from .env.paper.example and add paper API keys locally"
        )
    values = {
        key: value for key, value in dotenv_values(path).items() if value is not None
    }
    if values.get("DAYTRADING_ALPACA_PAPER", "").lower() not in ("true", "1", "yes"):
        raise ValueError(
            "Paper launcher requires DAYTRADING_ALPACA_PAPER=true in its own config"
        )
    for key in ("DAYTRADING_ALPACA_API_KEY", "DAYTRADING_ALPACA_SECRET_KEY"):
        value = values.get(key, "")
        if not value or value.lower().startswith(("your_", "replace", "put_")):
            raise ValueError(
                f"Set {key} in the local paper config; do not paste keys into chat"
            )
    feed = values.get("DAYTRADING_ALPACA_FEED", "iex").lower()
    if feed not in ("iex", "sip"):
        raise ValueError("Paper baseline supports iex or sip")
    if feed == "iex" and int(values.get("DAYTRADING_MAX_WATCHLIST_SYMBOLS", "30")) > 30:
        raise ValueError(
            "IEX Basic supports at most 30 streamed symbols, including SPY"
        )
    # Dedicated config cannot inherit an experiment from another project/shell.
    for key in list(os.environ):
        if key.startswith("DAYTRADING_"):
            os.environ.pop(key)
    os.environ.update(values)
    os.environ["DAYTRADING_ALPACA_PAPER"] = "true"
    os.environ["DAYTRADING_CONFIG_LOADED"] = "true"
    os.environ["DAYTRADING_DATA_DIR"] = str(path.resolve().parent / "data")
    return values


def preflight(values: dict, *, allow_existing: bool = False) -> dict:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    key, secret = (
        values["DAYTRADING_ALPACA_API_KEY"],
        values["DAYTRADING_ALPACA_SECRET_KEY"],
    )
    trading = TradingClient(key, secret, paper=True)
    from daytrading.execution.transport import configure_client_timeouts

    configure_client_timeouts(trading)
    account = trading.get_account()
    if account.trading_blocked or account.account_blocked:
        raise ValueError("Paper account reports trading/account blocked")
    positions = trading.get_all_positions()
    orders = trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
    if (positions or orders) and not allow_existing:
        raise ValueError(
            "Paper account already has positions/orders; review them before using --adopt-existing"
        )
    feed = values.get("DAYTRADING_ALPACA_FEED", "iex").lower()
    historical = StockHistoricalDataClient(key, secret)
    configure_client_timeouts(historical)
    quotes = historical.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=["SPY"], feed=DataFeed(feed))
    )
    quote = quotes.get("SPY")
    if quote is None or quote.bid_price <= 0 or quote.ask_price <= 0:
        raise ValueError("Selected feed did not return a usable SPY quote")
    return {
        "paper": True,
        "feed": feed,
        "existing_positions": len(positions),
        "open_orders": len(orders),
        "quote_timestamp": quote.timestamp.isoformat(),
        "market_open": trading.get_clock().is_open,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".env.paper"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-config", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--adopt-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        config_path = args.config.resolve()
        values = load_paper_config(config_path)
        os.chdir(config_path.parent)
        if args.check_config:
            print(
                f"Paper config valid; feed={values.get('DAYTRADING_ALPACA_FEED', 'iex')}; no API requests or orders sent"
            )
            return 0
        result = preflight(values, allow_existing=args.adopt_existing)
        print(f"Paper preflight passed: {result}; no orders submitted")
        if args.run:
            from daytrading.runner import AlpacaRunner

            AlpacaRunner.from_env().run()
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- sanitize provider errors at the command boundary
        # API exceptions may include URLs/request details; never print credentials.
        logging.getLogger(__name__).error(
            "Paper preflight error type: %s", type(exc).__name__
        )
        print(
            f"Paper preflight failed: {type(exc).__name__}. Check paper credentials, feed entitlement and connectivity.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
