"""Resolve best available live price for exit/stop checks."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from daytrading.models import Bar, Quote


def resolve_live_prices(
    symbols: Sequence[str],
    *,
    broker_positions: Optional[Mapping[str, dict]] = None,
    quotes: Optional[Mapping[str, Sequence[Quote]]] = None,
    bars: Optional[Mapping[str, Sequence[Bar]]] = None,
) -> Dict[str, float]:
    """Return a price for each symbol using broker → quote mid → last bar close."""
    prices: Dict[str, float] = {}
    broker_positions = broker_positions or {}
    quotes = quotes or {}
    bars = bars or {}

    for sym in symbols:
        data = broker_positions.get(sym)
        if data:
            px = float(data.get("current_price") or data.get("avg_entry") or 0)
            if px > 0:
                prices[sym] = px
                continue

        qlist: List[Quote] = list(quotes.get(sym) or [])
        if qlist:
            q = qlist[-1]
            if q.bid > 0 and q.ask > 0:
                prices[sym] = (q.bid + q.ask) / 2.0
                continue
            if q.ask > 0:
                prices[sym] = q.ask
                continue
            if q.bid > 0:
                prices[sym] = q.bid
                continue

        blist: List[Bar] = list(bars.get(sym) or [])
        if blist and blist[-1].close > 0:
            prices[sym] = blist[-1].close

    return prices
