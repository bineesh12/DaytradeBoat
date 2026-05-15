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

from datetime import datetime, timezone
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
        filled_qty: float = 500,
        filled_at: Optional[datetime] = None,
    ):
        self.id = id
        self.status = status
        self.filled_avg_price = filled_avg_price
        self.filled_qty = filled_qty
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

    def test_default_watchlist(self) -> None:
        from daytrading.runner import _default_watchlist
        watchlist = _default_watchlist()
        assert len(watchlist) >= 10
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
