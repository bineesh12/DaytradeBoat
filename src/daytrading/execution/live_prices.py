"""Resolve best available live price for exit/stop checks."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Dict, List, Mapping, Optional, Sequence

from daytrading.models import Bar, Quote, Side


def liquidation_quote(quote: Quote, symbol: str, side: Side, now: datetime,
                      max_age_seconds: float = 10.0) -> Optional[float]:
    """Fresh executable side, with no I/O or mutation; never a fill promise.

    A long liquidates at bid and a short at ask. Invalid, crossed/locked,
    empty, future or stale quotes leave the existing protection fallback in
    charge. Require only displayed liquidity, not enough depth for the whole
    position: partial fills still go through the broker's safety workflow.
    """
    if quote.symbol != symbol or side not in (Side.BUY, Side.SELL):
        return None
    if quote.ts.tzinfo is None or now.tzinfo is None:
        return None
    age = (now - quote.ts).total_seconds()
    values = (quote.bid, quote.ask, quote.bid_size, quote.ask_size)
    if (not 0 <= age <= max_age_seconds or
            not all(math.isfinite(value) and value > 0 for value in values) or
            quote.bid >= quote.ask):
        return None
    return quote.bid if side is Side.BUY else quote.ask


def resolve_exit_prices(symbols: Sequence[str], *, sides: Mapping[str, Side],
                        now: datetime, quotes=None, bars=None,
                        broker_positions=None, diagnostics=None) -> Dict[str, float]:
    """Return fresh liquidation sides only; cached marks are not order prices.

    Broker positions and bars remain accepted parameters for old adapters but
    cannot establish freshness of an executable bid/ask.
    """
    prices = {}
    for symbol in symbols:
        qlist = (quotes or {}).get(symbol) or ()
        aware = [quote for quote in qlist if quote.ts.tzinfo is not None]
        quote = max(aware, key=lambda q: q.ts) if aware else None
        price = (liquidation_quote(quote, symbol, sides.get(symbol), now)
                 if quote is not None else None)
        if price is not None:
            prices[symbol] = price
        if diagnostics is not None:
            diagnostics[symbol] = dict(
                price=prices.get(symbol),
                source=("fresh_bid" if sides.get(symbol) is Side.BUY else "fresh_ask")
                       if price is not None else "awaiting_fresh_quote",
                quote_ts=quote.ts.isoformat() if quote is not None else None,
                quote_age_seconds=((now-quote.ts).total_seconds() if quote is not None
                                   and now.tzinfo is not None else None),
                quote_fresh=price is not None,
                checked_at=now.isoformat(),
            )
    return prices


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
