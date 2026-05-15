"""Alpaca data feed — historical bars + real-time streaming.

Two classes:

  AlpacaHistoricalFeed — fetch historical bars via REST API.
    Used at startup to seed the pipeline with recent bar history.

  AlpacaStreamFeed — WebSocket streaming for live bars, quotes, trades.
    Feeds real-time data into the pipeline's run_cycle loop.

Usage:
    from daytrading.data.alpaca_feed import AlpacaHistoricalFeed, AlpacaStreamFeed

    # Historical
    hist = AlpacaHistoricalFeed(api_key, secret_key)
    bars = hist.get_bars(["AAPL", "TSLA"], timeframe="1Min", limit=100)

    # Streaming
    stream = AlpacaStreamFeed(api_key, secret_key, feed="iex")
    stream.subscribe(["AAPL", "TSLA"], on_bar=my_callback)
    stream.start()  # blocking — run in a thread
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from daytrading.models import Bar, Quote, Tick, Side, Timeframe

logger = logging.getLogger(__name__)

try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.live.stock import StockDataStream
    from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    _HAS_ALPACA = True
except ImportError:
    _HAS_ALPACA = False


_TF_MAP = {
    "1Min": (1, "Minute"),
    "5Min": (5, "Minute"),
    "15Min": (15, "Minute"),
    "1Hour": (1, "Hour"),
    "1Day": (1, "Day"),
}

_TF_ENUM_MAP = {
    "1Min": Timeframe.MIN_1,
    "5Min": Timeframe.MIN_5,
    "15Min": Timeframe.MIN_15,
    "1Hour": Timeframe.HOUR_1,
    "1Day": Timeframe.DAILY,
}


def _parse_feed(feed: str) -> "DataFeed":
    """Convert a feed string like 'iex' or 'sip' to the DataFeed enum."""
    return DataFeed(feed.lower())


def _make_timeframe(tf_str: str) -> TimeFrame:
    amount, unit_str = _TF_MAP.get(tf_str, (1, "Minute"))
    unit = getattr(TimeFrameUnit, unit_str)
    return TimeFrame(amount=amount, unit=unit)


# ---------------------------------------------------------------------------
# Historical
# ---------------------------------------------------------------------------

class AlpacaHistoricalFeed:
    """Fetch historical bar data from Alpaca REST API."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        feed: str = "iex",
    ) -> None:
        if not _HAS_ALPACA:
            raise ImportError(
                "alpaca-py is required. Install with: pip install 'daytrading[alpaca]'"
            )
        self._client = StockHistoricalDataClient(api_key, secret_key)
        self._feed = _parse_feed(feed)

    def get_bars(
        self,
        symbols: Sequence[str],
        *,
        timeframe: str = "1Min",
        limit: int = 100,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, List[Bar]]:
        """Fetch recent bars for multiple symbols.

        Returns dict mapping symbol -> list of Bar objects, oldest first.
        Fetches each symbol individually because Alpaca's limit is global.
        """
        if start is None:
            start = datetime.now(timezone.utc) - timedelta(days=5)

        tf = _make_timeframe(timeframe)
        tf_enum = _TF_ENUM_MAP.get(timeframe, Timeframe.MIN_1)
        result: Dict[str, List[Bar]] = {}

        for symbol in symbols:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=tf,
                    start=start,
                    end=end,
                    limit=limit,
                    feed=self._feed,
                )
                raw = self._client.get_stock_bars(request)
                raw_bars = getattr(raw, 'data', {}).get(symbol, [])
                if raw_bars is None:
                    raw_bars = []

                bars: List[Bar] = []
                for b in raw_bars:
                    bars.append(Bar(
                        symbol=symbol,
                        ts=b.timestamp if hasattr(b, 'timestamp') else getattr(b, 't', datetime.now(timezone.utc)),
                        open=float(getattr(b, 'open', 0)),
                        high=float(getattr(b, 'high', 0)),
                        low=float(getattr(b, 'low', 0)),
                        close=float(getattr(b, 'close', 0)),
                        volume=float(getattr(b, 'volume', 0)),
                        timeframe=tf_enum,
                    ))
                if bars:
                    result[symbol] = bars
            except Exception as exc:
                logger.warning("Failed to fetch bars for %s: %s", symbol, exc)

        logger.info(
            "Fetched bars: %s",
            ", ".join(f"{s}={len(bars)}" for s, bars in result.items()),
        )
        return result

    def get_latest_bars(self, symbols: Sequence[str]) -> Dict[str, Bar]:
        """Get the most recent bar for each symbol."""
        request = StockLatestBarRequest(
            symbol_or_symbols=list(symbols),
            feed=self._feed,
        )
        raw = self._client.get_stock_latest_bar(request)
        result: Dict[str, Bar] = {}

        for symbol in symbols:
            b = raw.get(symbol) if isinstance(raw, dict) else getattr(raw, symbol, None)
            if b is None:
                continue
            result[symbol] = Bar(
                symbol=symbol,
                ts=b.timestamp if hasattr(b, 'timestamp') else datetime.now(timezone.utc),
                open=float(getattr(b, 'open', 0)),
                high=float(getattr(b, 'high', 0)),
                low=float(getattr(b, 'low', 0)),
                close=float(getattr(b, 'close', 0)),
                volume=float(getattr(b, 'volume', 0)),
            )

        return result


# ---------------------------------------------------------------------------
# Real-time streaming
# ---------------------------------------------------------------------------

BarCallback = Callable[[Bar], None]
QuoteCallback = Callable[[Quote], None]
TradeCallback = Callable[[Tick], None]


class AlpacaStreamFeed:
    """WebSocket streaming for live bars, quotes, and trades.

    Runs in a background thread. Incoming data is converted to our
    Bar/Quote/Tick types and forwarded to registered callbacks.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        feed: str = "iex",
    ) -> None:
        if not _HAS_ALPACA:
            raise ImportError(
                "alpaca-py is required. Install with: pip install 'daytrading[alpaca]'"
            )
        self._stream = StockDataStream(api_key, secret_key, feed=_parse_feed(feed))
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._bar_callbacks: List[BarCallback] = []
        self._quote_callbacks: List[QuoteCallback] = []
        self._trade_callbacks: List[TradeCallback] = []

    def on_bar(self, callback: BarCallback) -> None:
        self._bar_callbacks.append(callback)

    def on_quote(self, callback: QuoteCallback) -> None:
        self._quote_callbacks.append(callback)

    def on_trade(self, callback: TradeCallback) -> None:
        self._trade_callbacks.append(callback)

    def subscribe(
        self,
        symbols: Sequence[str],
        *,
        bars: bool = True,
        quotes: bool = True,
        trades: bool = False,
    ) -> None:
        """Subscribe to data streams for the given symbols."""
        if bars:
            self._stream.subscribe_bars(self._handle_bar, *symbols)
            logger.info("Subscribed to bars: %s", symbols)

        if quotes:
            self._stream.subscribe_quotes(self._handle_quote, *symbols)
            logger.info("Subscribed to quotes: %s", symbols)

        if trades:
            self._stream.subscribe_trades(self._handle_trade, *symbols)
            logger.info("Subscribed to trades: %s", symbols)

    def subscribe_all_trades(self) -> None:
        """Subscribe to trades for ALL symbols using wildcard '*'.

        Used for real-time market scanning — catches every trade
        across the entire market on a single WebSocket connection.
        """
        self._stream.subscribe_trades(self._handle_trade, "*")
        logger.info("Subscribed to ALL trades (wildcard *) for real-time scanning")

    async def _handle_bar(self, data: Any) -> None:
        bar = Bar(
            symbol=data.symbol,
            ts=data.timestamp,
            open=float(data.open),
            high=float(data.high),
            low=float(data.low),
            close=float(data.close),
            volume=float(data.volume),
            timeframe=Timeframe.MIN_1,
        )
        for cb in self._bar_callbacks:
            try:
                cb(bar)
            except Exception as exc:
                logger.error("Bar callback error: %s", exc)

    async def _handle_quote(self, data: Any) -> None:
        quote = Quote(
            symbol=data.symbol,
            ts=data.timestamp,
            bid=float(data.bid_price),
            ask=float(data.ask_price),
            bid_size=float(data.bid_size),
            ask_size=float(data.ask_size),
        )
        for cb in self._quote_callbacks:
            try:
                cb(quote)
            except Exception as exc:
                logger.error("Quote callback error: %s", exc)

    async def _handle_trade(self, data: Any) -> None:
        tick = Tick(
            symbol=data.symbol,
            ts=data.timestamp,
            price=float(data.price),
            size=float(data.size),
            side=Side.BUY,  # Alpaca doesn't provide aggressor side directly
        )
        for cb in self._trade_callbacks:
            try:
                cb(tick)
            except Exception as exc:
                logger.error("Trade callback error: %s", exc)

    def start(self, background: bool = True, max_retries: int = 5) -> None:
        """Start the WebSocket stream.

        If background=True (default), runs in a daemon thread.
        Retries with exponential backoff on connection failures.
        """
        import time as _time

        def _run_with_retry() -> None:
            for attempt in range(1, max_retries + 1):
                try:
                    self._stream.run()
                    break
                except Exception as exc:
                    if not self._running:
                        break
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "Stream connection failed (attempt %d/%d): %s — retrying in %ds",
                        attempt, max_retries, exc, wait,
                    )
                    _time.sleep(wait)
            else:
                logger.error(
                    "Stream gave up after %d attempts. "
                    "Will retry when new data is needed.", max_retries,
                )

        if background:
            self._thread = threading.Thread(
                target=_run_with_retry, daemon=True, name="alpaca-stream",
            )
            self._running = True
            self._thread.start()
            logger.info("Alpaca stream started in background thread")
        else:
            self._running = True
            _run_with_retry()

    def stop(self) -> None:
        """Stop the WebSocket stream."""
        self._running = False
        try:
            self._stream.stop()
        except Exception:
            pass
        logger.info("Alpaca stream stopped")

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())
