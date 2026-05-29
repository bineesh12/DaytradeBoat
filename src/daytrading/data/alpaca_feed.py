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
import concurrent.futures
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

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


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_bar_window(
    start: Optional[datetime],
    end: Optional[datetime],
) -> tuple[datetime, datetime]:
    """Ensure Alpaca bar requests have start < end (UTC-aware)."""
    now_utc = datetime.now(timezone.utc)
    end_utc = _to_utc(end) if end is not None else now_utc
    start_utc = _to_utc(start) if start is not None else (now_utc - timedelta(days=5))
    if start_utc >= end_utc:
        start_utc = end_utc - timedelta(hours=6)
    return start_utc, end_utc


_BAR_FETCH_MAX_RETRIES = 3
_BAR_FETCH_BACKOFF_SEC = 2.0


def _is_transient_network_error(exc: BaseException) -> bool:
    """True for DNS/connection failures that may succeed on retry."""
    if isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)):
        return True
    msg = str(exc).lower()
    if any(
        needle in msg
        for needle in (
            "failed to resolve",
            "name resolution",
            "nodename nor servname",
            "name or service not known",
            "connection refused",
            "connection reset",
            "timed out",
            "temporary failure in name resolution",
        )
    ):
        return True
    try:
        from urllib3.exceptions import NameResolutionError, NewConnectionError

        if isinstance(exc, (NameResolutionError, NewConnectionError)):
            return True
    except ImportError:
        pass
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return _is_transient_network_error(cause)
    return False


def _chunk_symbols(symbols: Sequence[str], batch_size: int) -> List[List[str]]:
    """Split symbol list into batches of at most batch_size."""
    if batch_size < 1:
        batch_size = 1
    syms = [s for s in symbols if s]
    return [syms[i : i + batch_size] for i in range(0, len(syms), batch_size)]


if _HAS_ALPACA:

    class _ResilientDataStream(StockDataStream):
        """StockDataStream subclass that adds backoff on connection-limit errors.

        The base SDK catches ValueError internally and retries with zero delay,
        hammering Alpaca's server.  We override ``_run_forever`` to insert a
        30-60 s sleep when the server says "connection limit exceeded".
        """

        async def _run_forever(self) -> None:
            import websockets  # noqa: F811 — re-import inside async scope

            while not any(
                v
                for k, v in self._handlers.items()
                if k not in ("cancelErrors", "corrections")
            ):
                if not self._stop_stream_queue.empty():
                    self._stop_stream_queue.get(timeout=1)
                    return
                await asyncio.sleep(0)

            _log = logging.getLogger("alpaca.data.live.websocket")
            _log.info("started %s stream", self._name)
            self._should_run = True
            self._running = False
            _conn_limit_hits = 0

            while True:
                try:
                    if not self._should_run:
                        _log.info("%s stream stopped", self._name)
                        return
                    if not self._running:
                        _log.info("starting %s websocket connection", self._name)
                        await self._start_ws()
                        await self._send_subscribe_msg()
                        self._running = True
                        _conn_limit_hits = 0
                    await self._consume()
                except websockets.WebSocketException as wse:
                    await self.close()
                    self._running = False
                    _log.warning("data websocket error, restarting connection: %s", wse)
                except ValueError as ve:
                    msg = str(ve).lower()
                    if "insufficient subscription" in msg:
                        await self.close()
                        self._running = False
                        _log.error("fatal: %s", ve)
                        return
                    if "connection limit" in msg:
                        _conn_limit_hits += 1
                        await self.close()
                        self._running = False
                        wait = 30 if _conn_limit_hits <= 2 else 60
                        _log.warning(
                            "connection limit exceeded (hit #%d) — "
                            "waiting %ds for Alpaca to release old slot …",
                            _conn_limit_hits, wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    _log.exception("error during websocket communication: %s", ve)
                except Exception as e:
                    _log.exception("error during websocket communication: %s", e)
                finally:
                    await asyncio.sleep(0)


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
        bar_fetch_batch_size: int = 10,
        bar_fetch_batch_delay_sec: float = 0.5,
    ) -> None:
        if not _HAS_ALPACA:
            raise ImportError(
                "alpaca-py is required. Install with: pip install 'daytrading[alpaca]'"
            )
        self._client = StockHistoricalDataClient(api_key, secret_key)
        self._feed = _parse_feed(feed)
        self._bar_fetch_batch_size = max(1, int(bar_fetch_batch_size))
        self._bar_fetch_batch_delay_sec = max(0.0, float(bar_fetch_batch_delay_sec))
        self.last_fetch_failures: int = 0

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
        Uses batched REST requests to reduce connection churn on weak networks.
        """
        start, end = _normalize_bar_window(start, end)

        tf = _make_timeframe(timeframe)
        tf_enum = _TF_ENUM_MAP.get(timeframe, Timeframe.MIN_1)
        result: Dict[str, List[Bar]] = {}
        sym_list = [s for s in symbols if s]
        batches = _chunk_symbols(sym_list, self._bar_fetch_batch_size)
        failures = 0
        total_batches = len(batches)

        for batch_idx, batch in enumerate(batches, start=1):
            if batch_idx > 1 and self._bar_fetch_batch_delay_sec > 0:
                time.sleep(self._bar_fetch_batch_delay_sec)

            batch_result, batch_failures = self._fetch_bar_batch(
                batch,
                tf=tf,
                tf_enum=tf_enum,
                start=start,
                end=end,
                limit=limit,
            )
            result.update(batch_result)
            failures += batch_failures

            if total_batches > 1:
                logger.info(
                    "Fetched bars (batch %d/%d): %s",
                    batch_idx,
                    total_batches,
                    ", ".join(f"{s}={len(b)}" for s, b in batch_result.items())
                    or "(none)",
                )

        self.last_fetch_failures = failures

        if total_batches <= 1:
            logger.info(
                "Fetched bars: %s",
                ", ".join(f"{s}={len(bars)}" for s, bars in result.items()),
            )
        elif failures:
            logger.info(
                "Fetched bars total: %d symbols (%d failures)",
                len(result),
                failures,
            )
        else:
            logger.info("Fetched bars total: %d symbols", len(result))
        return result

    def _fetch_bar_batch(
        self,
        symbols: List[str],
        *,
        tf: "TimeFrame",
        tf_enum: Timeframe,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> tuple:
        """Fetch one multi-symbol batch; returns (symbol->bars, failure_count)."""
        if not symbols:
            return {}, 0

        label = ",".join(symbols[:3]) + ("..." if len(symbols) > 3 else "")
        last_exc: Optional[Exception] = None
        raw_data: Dict[str, List] = {}

        for attempt in range(1, _BAR_FETCH_MAX_RETRIES + 1):
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=list(symbols),
                    timeframe=tf,
                    start=start,
                    end=end,
                    limit=limit,
                    feed=self._feed,
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self._client.get_stock_bars, request)
                    raw = future.result(timeout=15)
                raw_data = dict(getattr(raw, "data", {}) or {})
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if (
                    attempt < _BAR_FETCH_MAX_RETRIES
                    and _is_transient_network_error(exc)
                ):
                    wait = _BAR_FETCH_BACKOFF_SEC * attempt
                    logger.warning(
                        "Network/DNS error (REST data.alpaca.markets, not WebSocket) "
                        "fetching batch [%s] — attempt %d/%d, retry in %.0fs: %s",
                        label,
                        attempt,
                        _BAR_FETCH_MAX_RETRIES,
                        wait,
                        exc,
                    )
                    time.sleep(wait)
                    continue
                break

        if last_exc is not None:
            if _is_transient_network_error(last_exc):
                logger.warning(
                    "Network/DNS error (REST data.alpaca.markets, not WebSocket) "
                    "— batch [%s] failed after %d attempts: %s",
                    label,
                    _BAR_FETCH_MAX_RETRIES,
                    last_exc,
                )
            else:
                logger.warning(
                    "Failed to fetch bars for batch [%s]: %s", label, last_exc,
                )
            return {}, len(symbols)

        batch_result: Dict[str, List[Bar]] = {}
        for symbol in symbols:
            raw_bars = raw_data.get(symbol) or []
            bars: List[Bar] = []
            for b in raw_bars:
                bars.append(Bar(
                    symbol=symbol,
                    ts=b.timestamp if hasattr(b, "timestamp") else getattr(
                        b, "t", datetime.now(timezone.utc),
                    ),
                    open=float(getattr(b, "open", 0)),
                    high=float(getattr(b, "high", 0)),
                    low=float(getattr(b, "low", 0)),
                    close=float(getattr(b, "close", 0)),
                    volume=float(getattr(b, "volume", 0)),
                    timeframe=tf_enum,
                ))
            if bars:
                batch_result[symbol] = bars

        missing = len(symbols) - len(batch_result)
        return batch_result, missing

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
        self._api_key = api_key
        self._secret_key = secret_key
        self._feed_str = feed
        self._stream = _ResilientDataStream(api_key, secret_key, feed=_parse_feed(feed))
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._bar_callbacks: List[BarCallback] = []
        self._quote_callbacks: List[QuoteCallback] = []
        self._trade_callbacks: List[TradeCallback] = []

        self._subscribed_bar_symbols: List[str] = []
        self._subscribed_quote_symbols: List[str] = []
        self._subscribed_trade_symbols: List[str] = []
        self._subscribe_all_trades: bool = False
        self._sub_lock = threading.Lock()
        self._pending_bar_symbols: Set[str] = set()
        self._pending_quote_symbols: Set[str] = set()
        self._pending_trade_symbols: Set[str] = set()
        self._trade_filter_symbols: Set[str] = set()

        # Lee-Ready tick classification state
        self._latest_quotes: Dict[str, Quote] = {}
        self._last_tick_side: Dict[str, Side] = {}

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
        """Queue bar/quote subscriptions; apply on the stream asyncio loop."""
        syms = [s.upper() for s in symbols if s]
        if not syms:
            return
        with self._sub_lock:
            if bars:
                self._pending_bar_symbols.update(syms)
            if quotes:
                self._pending_quote_symbols.update(syms)
            if trades:
                self._pending_trade_symbols.update(syms)
        if trades:
            self.flush_pending_subscriptions()

    def flush_pending_subscriptions(self) -> None:
        """Send queued subscriptions on the stream thread (thread-safe)."""
        if not self._running:
            return
        loop = getattr(self._stream, "_loop", None)
        if loop is None or not loop.is_running():
            return

        with self._sub_lock:
            new_bars = [
                s for s in self._pending_bar_symbols
                if s not in self._subscribed_bar_symbols
            ]
            new_quotes = [
                s for s in self._pending_quote_symbols
                if s not in self._subscribed_quote_symbols
            ]
            new_trades = [
                s for s in self._pending_trade_symbols
                if s not in self._subscribed_trade_symbols
            ]
            for s in new_bars:
                self._pending_bar_symbols.discard(s)
            for s in new_quotes:
                self._pending_quote_symbols.discard(s)
            for s in new_trades:
                self._pending_trade_symbols.discard(s)

        if not new_bars and not new_quotes and not new_trades:
            return

        try:
            if new_bars:
                self._stream.subscribe_bars(self._handle_bar, *new_bars)
                self._subscribed_bar_symbols.extend(new_bars)
            if new_quotes:
                self._stream.subscribe_quotes(self._handle_quote, *new_quotes)
                self._subscribed_quote_symbols.extend(new_quotes)
            if new_trades:
                self._stream.subscribe_trades(self._handle_trade, *new_trades)
                self._subscribed_trade_symbols.extend(new_trades)
            asyncio.run_coroutine_threadsafe(
                self._stream._send_subscribe_msg(), loop,
            )
            logger.info(
                "Stream subscribed — bars +%d quotes +%d trades +%d (e.g. %s)",
                len(new_bars),
                len(new_quotes),
                len(new_trades),
                (new_bars or new_quotes or new_trades)[:3],
            )
        except Exception as exc:
            logger.warning(
                "Stream subscribe flush failed (%d bars, %d quotes, %d trades): %s",
                len(new_bars),
                len(new_quotes),
                len(new_trades),
                exc,
            )
            with self._sub_lock:
                self._pending_bar_symbols.update(new_bars)
                self._pending_quote_symbols.update(new_quotes)
                self._pending_trade_symbols.update(new_trades)

    def subscribe_all_trades(self) -> None:
        """Subscribe to trades for ALL symbols using wildcard '*'.

        Used for real-time market scanning — catches every trade
        across the entire market on a single WebSocket connection.
        """
        self._subscribe_all_trades = True
        logger.info(
            "Trade subscription mode: per-symbol (pool-only, not wildcard *)"
        )

    def set_trade_filter(self, symbols: Set[str]) -> None:
        """Set the symbols that should be processed by the trade handler.

        When using wildcard subscription, trades outside this set are dropped.
        When using per-symbol mode, this subscribes to trades for exactly
        these symbols (much lower network/CPU load than wildcard).
        """
        old = self._trade_filter_symbols
        self._trade_filter_symbols = symbols
        if self._subscribe_all_trades and self._running:
            new_syms = symbols - old
            if new_syms:
                with self._sub_lock:
                    self._pending_trade_symbols.update(new_syms)
                self.flush_pending_subscriptions()

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
        self._latest_quotes[data.symbol] = quote
        for cb in self._quote_callbacks:
            try:
                cb(quote)
            except Exception as exc:
                logger.error("Quote callback error: %s", exc)

    async def _handle_trade(self, data: Any) -> None:
        filt = self._trade_filter_symbols
        if data.symbol not in filt:
            return
        price = float(data.price)
        # Lee-Ready tick classification: compare trade price to latest NBBO
        last_q = self._latest_quotes.get(data.symbol)
        if last_q and last_q.ask > 0 and last_q.bid > 0:
            if price >= last_q.ask:
                side = Side.BUY
            elif price <= last_q.bid:
                side = Side.SELL
            else:
                side = self._last_tick_side.get(data.symbol, Side.BUY)
        else:
            side = self._last_tick_side.get(data.symbol, Side.BUY)
        self._last_tick_side[data.symbol] = side
        tick = Tick(
            symbol=data.symbol,
            ts=data.timestamp,
            price=price,
            size=float(data.size),
            side=side,
        )
        for cb in self._trade_callbacks:
            try:
                cb(tick)
            except Exception as exc:
                logger.error("Trade callback error: %s", exc)

    def _rebuild_stream(self) -> None:
        """Create a fresh StockDataStream and re-register all subscriptions.

        The Alpaca SDK's internal reconnect loop keeps hammering the server
        with the old (broken) connection.  Building a brand-new instance
        resets all internal state and lets us control the retry cadence.
        """
        try:
            self._stream.stop()
        except Exception:
            pass

        self._stream = _ResilientDataStream(
            self._api_key, self._secret_key,
            feed=_parse_feed(self._feed_str),
        )

        if self._subscribed_bar_symbols:
            self._stream.subscribe_bars(self._handle_bar, *self._subscribed_bar_symbols)
        if self._subscribed_quote_symbols:
            self._stream.subscribe_quotes(self._handle_quote, *self._subscribed_quote_symbols)
        if self._subscribed_trade_symbols:
            self._stream.subscribe_trades(self._handle_trade, *self._subscribed_trade_symbols)

    def start(self, background: bool = True, max_retries: int = 12) -> None:
        """Start the WebSocket stream.

        If background=True (default), runs in a daemon thread.
        On "connection limit exceeded" errors, we wait 30-60s for Alpaca
        to release the old slot, then create a completely fresh
        StockDataStream instance to avoid the SDK's own rapid-fire
        reconnect loop.
        """
        import atexit
        import time as _time

        atexit.register(self.stop)

        def _run_with_retry() -> None:
            for attempt in range(1, max_retries + 1):
                try:
                    self._stream.run()
                    break
                except Exception as exc:
                    if not self._running:
                        break
                    err_msg = str(exc).lower()
                    if "connection limit" in err_msg or "429" in err_msg:
                        wait = 30 if attempt <= 2 else 60
                        logger.warning(
                            "Connection limit exceeded (attempt %d/%d) — "
                            "Alpaca needs time to release the old slot. "
                            "Rebuilding stream in %ds …",
                            attempt, max_retries, wait,
                        )
                        _time.sleep(wait)
                        self._rebuild_stream()
                    else:
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
        """Stop the WebSocket stream and close the underlying connection."""
        if not self._running:
            return
        self._running = False

        try:
            self._stream.stop()
        except Exception:
            pass

        try:
            inner = self._stream
            if hasattr(inner, '_loop') and inner._loop is not None and inner._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    inner.close(), inner._loop,
                ).result(timeout=5)
            elif hasattr(inner, '_ws') and inner._ws is not None:
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(inner._ws.close())
                    loop.close()
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("WS force-close (expected): %s", exc)

        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

        logger.info("Alpaca stream stopped")

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())
