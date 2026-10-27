"""Prior trading day close/high for HOD vs yesterday comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from daytrading.market_calendar import ET
from typing import Dict, List, Optional, Sequence

import logging

logger = logging.getLogger(__name__)

try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    _HAS_ALPACA = True
except ImportError:
    _HAS_ALPACA = False


@dataclass(frozen=True)
class PriorDayStats:
    prior_close: float
    prior_high: float
    prior_volume: float = 0.0


def gap_and_change_from_close(
    session_open: float,
    price: float,
    prior_close: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    if prior_close is None or prior_close <= 0:
        return None, None
    gap_pct = (session_open - prior_close) / prior_close * 100
    change_from_close_pct = (price - prior_close) / prior_close * 100
    return gap_pct, change_from_close_pct


def fetch_prior_day_stats(
    client: object,
    symbols: Sequence[str],
    *,
    feed: object,
) -> Dict[str, PriorDayStats]:
    """Load the last completed daily bar on today's split-adjusted basis.

    Snapshots expose raw previous bars; comparing those with post-split prices
    can turn a corporate action into a fake gap or HOD breakout.
    """
    if not _HAS_ALPACA or not symbols:
        return {}
    result = {}
    end = datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=10)
    sym_list = list(symbols)
    for i in range(0, len(sym_list), 100):
        batch = sym_list[i:i+100]
        try:
            req = StockBarsRequest(symbol_or_symbols=batch, feed=feed,
                                   timeframe=TimeFrame.Day, start=start, end=end,
                                   adjustment="split")
            raw = client.get_stock_bars(req)
            rows = getattr(raw, "data", raw)
            for symbol in batch:
                completed = [b for b in rows.get(symbol, []) if b.timestamp.astimezone(ET).date() < end.date()]
                if not completed:
                    continue
                previous = max(completed, key=lambda b: b.timestamp)
                close = float(previous.close)
                high = float(previous.high)
                if close > 0:
                    result[symbol] = PriorDayStats(close, high if high > 0 else close, float(previous.volume))
        except Exception as exc:
            logger.warning("Split-adjusted prior-day bar batch unavailable: %s", type(exc).__name__)
            # Missing verified reference data is preferable to a fabricated gap.
            continue
    return result
