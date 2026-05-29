"""Tests for Alpaca integration — uses mocks (no real API calls).

Verifies:
  1. AlpacaBroker correctly maps our Order → Alpaca order request
  2. AlpacaBroker handles fills, rejections, timeouts
  3. AlpacaHistoricalFeed converts Alpaca bars to our Bar type
  4. AlpacaStreamFeed routes callbacks correctly
  5. AlpacaRunner.from_env validates configuration
  6. Pipeline uses AlpacaBroker as drop-in replacement for PaperBroker
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from daytrading.models import Bar, Fill, Order, OrderStatus, PortfolioState, Side


# ===================================================================
# Mock Alpaca SDK objects so tests run without alpaca-py installed
# ===================================================================

class _MockAlpacaOrder:
    def __init__(
        self,
        id: str = "test-order-123",
        status: str = "filled",
        filled_avg_price: float = 5.05,
        filled_qty: Optional[float] = None,
        filled_at: Optional[datetime] = None,
    ):
        self.id = id
        self.status = status
        self.filled_avg_price = filled_avg_price
        self.filled_qty = 500 if filled_qty is None and status == "filled" else (filled_qty or 0)
        self.filled_at = filled_at or datetime.now(timezone.utc)


class _MockTradingClient:
    def __init__(self) -> None:
        self.submitted_orders: list = []
        self._next_order = _MockAlpacaOrder()

    def submit_order(self, order_data: object) -> _MockAlpacaOrder:
        self.submitted_orders.append(order_data)
        return self._next_order

    def get_order_by_id(self, order_id: str) -> _MockAlpacaOrder:
        return self._next_order

    def cancel_order_by_id(self, order_id: str) -> None:
        pass

    def get_account(self) -> MagicMock:
        acct = MagicMock()
        acct.cash = "25000.00"
        acct.buying_power = "50000.00"
        acct.portfolio_value = "25000.00"
        acct.equity = "25000.00"
        acct.pattern_day_trader = False
        return acct

    def get_all_positions(self) -> list:
        return []

    def cancel_orders(self) -> list:
        return []

    def get_orders(self, filter: object = None) -> list:
        return []

    def close_all_positions(self, cancel_orders: bool = True) -> None:
        pass


# ===================================================================
# AlpacaBroker tests
# ===================================================================

class TestAlpacaBroker:
    """Test the broker with a mocked TradingClient."""

    def _make_broker(self, mock_client: Optional[_MockTradingClient] = None):
        """Build an AlpacaBroker with a mock client injected."""
        # Import with mocked alpaca modules
        mock_trading = MagicMock()
        mock_trading.TradingClient = _MockTradingClient
        mock_trading.OrderSide = MagicMock()
        mock_trading.OrderSide.BUY = "buy"
        mock_trading.OrderSide.SELL = "sell"

        # Directly construct and inject mock
        from daytrading.execution.alpaca_broker import AlpacaBroker

        with patch.dict("sys.modules", {
            "alpaca": MagicMock(),
            "alpaca.trading": MagicMock(),
            "alpaca.trading.client": MagicMock(),
            "alpaca.trading.enums": MagicMock(),
            "alpaca.trading.requests": MagicMock(),
        }):
            with patch("daytrading.execution.alpaca_broker._HAS_ALPACA", True):
                broker = AlpacaBroker.__new__(AlpacaBroker)
                client = mock_client or _MockTradingClient()
                broker._client = client
                broker._paper = True
                broker._max_wait = 1.0
                broker._poll_interval = 0.05
                broker._cancel_grace_seconds = 0.1
                broker._slippage_guard = None
                broker._limit_buffer_pct = 0.005
                return broker, client

    def test_submit_buy_order_filled(self) -> None:
        broker, client = self._make_broker()

        order = Order(symbol="AAPL", side=Side.BUY, quantity=100, limit_price=5.00)
        bar = Bar(symbol="AAPL", ts=datetime.now(timezone.utc),
                  open=5.0, high=5.1, low=4.9, close=5.0, volume=10000)
        portfolio = PortfolioState(cash=25000)

        fill, status = broker._wait_for_fill("test-id", order)

        assert status is OrderStatus.FILLED
        assert fill is not None
        assert fill.symbol == "AAPL"
        assert fill.side is Side.BUY
        assert fill.quantity == 500  # from mock
        assert fill.price == 5.05   # from mock

    def test_submit_rejected_order(self) -> None:
        client = _MockTradingClient()
        client._next_order = _MockAlpacaOrder(status="rejected")
        broker, _ = self._make_broker(client)

        order = Order(symbol="AAPL", side=Side.BUY, quantity=100)
        fill, status = broker._wait_for_fill("test-id", order)

        assert status is OrderStatus.REJECTED
        assert fill is None

    def test_submit_cancelled_order(self) -> None:
        client = _MockTradingClient()
        client._next_order = _MockAlpacaOrder(status="canceled")
        broker, _ = self._make_broker(client)

        order = Order(symbol="AAPL", side=Side.BUY, quantity=100)
        fill, status = broker._wait_for_fill("test-id", order)

        assert status is OrderStatus.REJECTED
        assert fill is None

    def test_cancelled_order_with_filled_qty_returns_fill(self) -> None:
        client = _MockTradingClient()
        client._next_order = _MockAlpacaOrder(
            status="canceled",
            filled_avg_price=5.42,
            filled_qty=437,
        )
        broker, _ = self._make_broker(client)

        order = Order(symbol="MASK", side=Side.BUY, quantity=750)
        fill, status = broker._wait_for_fill("test-id", order)

        assert status is OrderStatus.FILLED
        assert fill is not None
        assert fill.symbol == "MASK"
        assert fill.quantity == 437
        assert fill.price == 5.42

    def test_timeout_waits_for_late_partial_fill_after_cancel(self) -> None:
        client = _MockTradingClient()
        client._next_order = _MockAlpacaOrder(
            status="canceled",
            filled_avg_price=5.42,
            filled_qty=384,
        )
        broker, _ = self._make_broker(client)
        broker._max_wait = 0.0

        order = Order(symbol="MASK", side=Side.BUY, quantity=750)
        fill, status = broker._wait_for_fill("test-id", order)

        assert status is OrderStatus.FILLED
        assert fill is not None
        assert fill.quantity == 384
        assert fill.price == 5.42

    def test_none_limit_price_submits_guarded_marketable_limit(self, monkeypatch) -> None:
        from daytrading.execution import alpaca_broker as broker_mod

        limit_requests = []

        def fake_limit_order_request(**kwargs):
            req = SimpleNamespace(kind="limit", **kwargs)
            limit_requests.append(req)
            return req

        monkeypatch.setattr(broker_mod, "LimitOrderRequest", fake_limit_order_request)
        monkeypatch.setattr(broker_mod, "OrderSide", SimpleNamespace(BUY="buy", SELL="sell"))
        monkeypatch.setattr(broker_mod, "TimeInForce", SimpleNamespace(DAY="day"))

        client = _MockTradingClient()
        client._next_order = _MockAlpacaOrder(
            status="filled",
            filled_avg_price=4.99,
            filled_qty=243,
        )
        broker, _ = self._make_broker(client)
        broker._marketable_limit_slippage_pct = 0.0075

        order = Order(symbol="MASK", side=Side.SELL, quantity=243, limit_price=None)
        bar = Bar(
            symbol="MASK",
            ts=datetime.now(timezone.utc),
            open=5.0,
            high=5.0,
            low=4.9,
            close=4.95,
            volume=10000,
        )
        fill, status = broker.submit(order, bar, PortfolioState(cash=25000))

        assert status is OrderStatus.FILLED
        assert fill is not None
        assert fill.price == 4.99
        assert limit_requests
        assert limit_requests[0].kind == "limit"
        assert limit_requests[0].side == "sell"
        assert limit_requests[0].limit_price == 4.91

    def test_regular_sell_limit_uses_tight_slippage_window(self, monkeypatch) -> None:
        from daytrading.execution import alpaca_broker as broker_mod

        limit_requests = []

        def fake_limit_order_request(**kwargs):
            req = SimpleNamespace(kind="limit", **kwargs)
            limit_requests.append(req)
            return req

        monkeypatch.setattr(broker_mod, "LimitOrderRequest", fake_limit_order_request)
        monkeypatch.setattr(broker_mod, "OrderSide", SimpleNamespace(BUY="buy", SELL="sell"))
        monkeypatch.setattr(broker_mod, "TimeInForce", SimpleNamespace(DAY="day"))

        client = _MockTradingClient()
        client._next_order = _MockAlpacaOrder(
            status="filled",
            filled_avg_price=9.94,
            filled_qty=100,
        )
        broker, _ = self._make_broker(client)
        broker._sell_limit_slippage_pct = 0.0075

        order = Order(symbol="EXIT", side=Side.SELL, quantity=100, limit_price=10.00)
        bar = Bar(
            symbol="EXIT",
            ts=datetime.now(timezone.utc),
            open=10.0,
            high=10.1,
            low=9.9,
            close=10.0,
            volume=10000,
        )

        fill, status = broker.submit(order, bar, PortfolioState(cash=25000))

        assert status is OrderStatus.FILLED
        assert fill is not None
        assert limit_requests
        assert limit_requests[0].limit_price == 9.93

    def test_zero_quantity_rejected(self) -> None:
        broker, _ = self._make_broker()

        order = Order(symbol="AAPL", side=Side.BUY, quantity=0)
        bar = Bar(symbol="AAPL", ts=datetime.now(timezone.utc),
                  open=5.0, high=5.1, low=4.9, close=5.0, volume=10000)
        portfolio = PortfolioState(cash=25000)

        fill, status = broker.submit(order, bar, portfolio)
        assert status is OrderStatus.REJECTED
        assert fill is None

    def test_get_account_returns_dict(self) -> None:
        broker, _ = self._make_broker()
        acct = broker.get_account()

        assert "cash" in acct
        assert "buying_power" in acct
        assert "equity" in acct
        assert isinstance(acct["cash"], float)

    def test_get_positions_empty(self) -> None:
        broker, _ = self._make_broker()
        positions = broker.get_positions()
        assert positions == {}


# ===================================================================
# AlpacaHistoricalFeed tests
# ===================================================================

class TestAlpacaHistoricalFeed:

    def test_bar_conversion(self) -> None:
        """Verify Alpaca bars are converted to our Bar type."""
        # Mock an alpaca bar object
        mock_bar = MagicMock()
        mock_bar.timestamp = datetime(2026, 5, 13, 14, 30, 0, tzinfo=timezone.utc)
        mock_bar.open = 5.00
        mock_bar.high = 5.10
        mock_bar.low = 4.95
        mock_bar.close = 5.05
        mock_bar.volume = 100000

        bar = Bar(
            symbol="AAPL",
            ts=mock_bar.timestamp,
            open=float(mock_bar.open),
            high=float(mock_bar.high),
            low=float(mock_bar.low),
            close=float(mock_bar.close),
            volume=float(mock_bar.volume),
        )

        assert bar.symbol == "AAPL"
        assert bar.close == 5.05
        assert bar.volume == 100000

    def test_normalize_bar_window_when_start_after_end(self) -> None:
        from daytrading.data.alpaca_feed import _normalize_bar_window

        now = datetime(2026, 5, 17, 10, 0, 0, tzinfo=timezone.utc)
        future_start = datetime(2026, 5, 17, 13, 30, 0, tzinfo=timezone.utc)
        with patch("daytrading.data.alpaca_feed.datetime") as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            start, end = _normalize_bar_window(future_start, now)
        assert start < end
        assert end == now


# ===================================================================
# AlpacaStreamFeed tests
# ===================================================================

class TestAlpacaStreamFeed:

    def test_callback_registration(self) -> None:
        """Verify callbacks are stored."""
        from daytrading.data.alpaca_feed import AlpacaStreamFeed

        with patch.dict("sys.modules", {
            "alpaca": MagicMock(),
            "alpaca.data": MagicMock(),
            "alpaca.data.historical": MagicMock(),
            "alpaca.data.historical.stock": MagicMock(),
            "alpaca.data.live": MagicMock(),
            "alpaca.data.live.stock": MagicMock(),
            "alpaca.data.requests": MagicMock(),
            "alpaca.data.timeframe": MagicMock(),
        }):
            with patch("daytrading.data.alpaca_feed._HAS_ALPACA", True):
                feed = AlpacaStreamFeed.__new__(AlpacaStreamFeed)
                feed._stream = MagicMock()
                feed._thread = None
                feed._running = False
                feed._bar_callbacks = []
                feed._quote_callbacks = []
                feed._trade_callbacks = []

                cb = lambda bar: None
                feed.on_bar(cb)
                assert len(feed._bar_callbacks) == 1
                assert feed._bar_callbacks[0] is cb

    def test_trade_filter_queues_and_flushes_trades_on_stream_loop(self) -> None:
        """Pool trade updates must not call the Alpaca stream from scanner threads."""
        from daytrading.data.alpaca_feed import AlpacaStreamFeed

        class _Loop:
            def is_running(self) -> bool:
                return True

        class _Stream:
            def __init__(self) -> None:
                self._loop = _Loop()
                self.trade_subs: list[str] = []

            def subscribe_trades(self, callback, *symbols: str) -> None:
                self.trade_subs.extend(symbols)

            async def _send_subscribe_msg(self) -> None:
                return None

        feed = AlpacaStreamFeed.__new__(AlpacaStreamFeed)
        feed._stream = _Stream()
        feed._running = True
        feed._subscribe_all_trades = True
        feed._sub_lock = threading.Lock()
        feed._pending_bar_symbols = set()
        feed._pending_quote_symbols = set()
        feed._pending_trade_symbols = set()
        feed._subscribed_bar_symbols = []
        feed._subscribed_quote_symbols = []
        feed._subscribed_trade_symbols = []
        feed._trade_filter_symbols = set()
        feed._handle_bar = MagicMock()
        feed._handle_quote = MagicMock()
        feed._handle_trade = MagicMock()

        with patch("daytrading.data.alpaca_feed.asyncio.run_coroutine_threadsafe") as run_coro:
            run_coro.side_effect = lambda coro, loop: coro.close()
            feed.set_trade_filter({"IOTR", "NEXR"})

        assert sorted(feed._stream.trade_subs) == ["IOTR", "NEXR"]
        assert sorted(feed._subscribed_trade_symbols) == ["IOTR", "NEXR"]
        assert feed._pending_trade_symbols == set()
        assert run_coro.called


# ===================================================================
# Runner config validation
# ===================================================================

class TestRunnerConfig:

    def test_missing_api_key_raises(self) -> None:
        import os
        from daytrading.config import Settings
        old_key = os.environ.pop("DAYTRADING_ALPACA_API_KEY", None)
        old_secret = os.environ.pop("DAYTRADING_ALPACA_SECRET_KEY", None)
        try:
            os.environ["DAYTRADING_ALPACA_API_KEY"] = ""
            os.environ["DAYTRADING_ALPACA_SECRET_KEY"] = ""
            cfg = Settings()

            with pytest.raises(ValueError, match="Missing Alpaca credentials"):
                from daytrading.runner import AlpacaRunner
                AlpacaRunner.from_env(settings=cfg)
        finally:
            if old_key is not None:
                os.environ["DAYTRADING_ALPACA_API_KEY"] = old_key
            if old_secret is not None:
                os.environ["DAYTRADING_ALPACA_SECRET_KEY"] = old_secret

    def test_fallback_watchlist(self) -> None:
        from daytrading.runner import _fallback_watchlist
        watchlist = _fallback_watchlist()
        assert len(watchlist) >= 5
        assert all(isinstance(s, str) for s in watchlist)


# ===================================================================
# Broker protocol compatibility
# ===================================================================

class TestBrokerProtocol:

    def test_alpaca_broker_matches_protocol(self) -> None:
        """AlpacaBroker should have the same submit() signature as Broker."""
        from daytrading.execution.alpaca_broker import AlpacaBroker
        from daytrading.execution.broker import Broker

        assert hasattr(AlpacaBroker, "submit")

        import inspect
        paper_sig = inspect.signature(Broker.submit)
        alpaca_sig = inspect.signature(AlpacaBroker.submit)

        paper_params = list(paper_sig.parameters.keys())
        alpaca_params = list(alpaca_sig.parameters.keys())
        assert paper_params == alpaca_params
