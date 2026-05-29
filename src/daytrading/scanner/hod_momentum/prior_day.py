"""Prior trading day close/high for HOD vs yesterday comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import logging

logger = logging.getLogger(__name__)

try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest
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
    """Load previous_daily_bar close/high via Alpaca snapshots."""
    if not _HAS_ALPACA or not symbols:
        return {}

    result: Dict[str, PriorDayStats] = {}
    batch_size = 100
    sym_list = list(symbols)

    for i in range(0, len(sym_list), batch_size):
        batch = sym_list[i : i + batch_size]
        try:
            req = StockSnapshotRequest(symbol_or_symbols=batch, feed=feed)
            snaps = client.get_stock_snapshot(req)
        except Exception as exc:
            logger.warning("Prior day snapshot batch failed: %s", exc)
            continue

        if not isinstance(snaps, dict):
            snaps = {batch[0]: snaps} if len(batch) == 1 else {}

        for sym in batch:
            snap = snaps.get(sym)
            if snap is None:
                continue
            prev = getattr(snap, "previous_daily_bar", None)
            if prev is None:
                continue
            try:
                close = float(getattr(prev, "close", 0))
                high = float(getattr(prev, "high", close))
                volume = float(getattr(prev, "volume", 0))
            except (TypeError, ValueError):
                continue
            if close <= 0:
                continue
            if high <= 0:
                high = close
            result[sym] = PriorDayStats(
                prior_close=close, prior_high=high, prior_volume=volume,
            )

    return result
