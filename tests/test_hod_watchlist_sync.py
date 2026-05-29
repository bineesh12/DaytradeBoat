"""Tests for HOD-alert-driven watchlist sync and trade universe."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from threading import Lock
from unittest.mock import MagicMock

from daytrading.models import Bar
from daytrading.runner import AlpacaRunner


class _RunnerStub:
    """Minimal runner state for unbound method tests."""

    _protected_watchlist_symbols = AlpacaRunner._protected_watchlist_symbols
    _hod_watchlist_symbols = AlpacaRunner._hod_watchlist_symbols
    _trade_symbol_set = AlpacaRunner._trade_symbol_set
    _trade_universe = AlpacaRunner._trade_universe
    _sync_watchlist_to_hod_alerts = AlpacaRunner._sync_watchlist_to_hod_alerts
    _publish_trading_watchlist = AlpacaRunner._publish_trading_watchlist
    _watchlist_pinned = {"SPY"}
    _add_symbols_to_watchlist = AlpacaRunner._add_symbols_to_watchlist
    _remove_symbols_from_watchlist = AlpacaRunner._remove_symbols_from_watchlist

    def __init__(self, watchlist: list[str]) -> None:
        self._watchlist = list(watchlist)
        self._watchlist_set = set(watchlist)
        self._watchlist_pinned = {"SPY"}
        self._max_watchlist = 50
        self._hod_watchlist_ttl_minutes = 5.0
        self._hod_last_alert_at: dict = {}
        self._news_checker = None
        self._news_pinned = set()
        self._skip_counts = defaultdict(int)
        self._bar_buffer = defaultdict(list)
        self._quote_buffer = defaultdict(list)
        self._tick_buffer = defaultdict(list)
        self._lock = Lock()
        self._hod_tick_tracker = None
        self._hist = MagicMock()
        self._stream = MagicMock()
        self._hub = MagicMock()
        self._load_session_bars_for_symbols = MagicMock()
        self._load_prior_day_stats = MagicMock()
        self._seed_all_hod_sessions = MagicMock()
        self._enqueue_hod_seed_symbols = MagicMock()
        self._ensure_streaming_symbols = MagicMock()
        portfolio = MagicMock()
        portfolio.positions = {}
        exit_mgr = MagicMock()
        exit_mgr.tracked = {}
        self._pipeline = MagicMock()
        self._pipeline.portfolio = portfolio
        self._pipeline.exit_manager = exit_mgr


def _make_runner(watchlist: list[str]) -> _RunnerStub:
    return _RunnerStub(watchlist)


class _FloatMap:
    def __init__(self, floats: dict) -> None:
        self._floats = floats

    def get_float(self, symbol: str):
        return self._floats.get(symbol)

    def get_float_cached(self, symbol: str):
        return self._floats.get(symbol)

    def warm_from_store(self, symbols):
        return 0, len(symbols)

    def needs_yahoo_refresh(self, symbol: str) -> bool:
        return True


class TestFloatFilteredHodPool:
    def test_excludes_high_float(self) -> None:
        candidates = [
            {"symbol": "BIG", "price": 8.0, "change_pct": 10.0},
            {"symbol": "LOW", "price": 6.0, "change_pct": 8.0},
        ]
        pool = AlpacaRunner.build_float_filtered_hod_pool(
            candidates,
            _FloatMap({"BIG": 50_000_000, "LOW": 5_000_000}),
            max_float=20_000_000,
            pool_max=50,
        )
        assert pool == ["LOW"]

    def test_excludes_price_outside_band(self) -> None:
        candidates = [
            {"symbol": "CHEAP", "price": 1.5, "change_pct": 10.0},
            {"symbol": "OK", "price": 5.0, "change_pct": 8.0},
        ]
        pool = AlpacaRunner.build_float_filtered_hod_pool(
            candidates,
            _FloatMap({"CHEAP": 5_000_000, "OK": 5_000_000}),
            min_price=2.0,
            max_price=20.0,
            max_float=20_000_000,
        )
        assert pool == ["OK"]

    def test_respects_pool_max(self) -> None:
        candidates = [
            {"symbol": "A{}".format(i), "price": 5.0, "change_pct": i}
            for i in range(10)
        ]
        floats = {c["symbol"]: 1_000_000 for c in candidates}
        pool = AlpacaRunner.build_float_filtered_hod_pool(
            candidates,
            _FloatMap(floats),
            max_float=20_000_000,
            pool_max=3,
        )
        assert len(pool) == 3
        assert pool == ["A0", "A1", "A2"]


class TestHodWatchlistSync:
    def test_adds_symbol_on_hod_alert(self) -> None:
        runner = _make_runner(["SPY"])
        runner._add_symbols_to_watchlist(["CISS"])
        assert "CISS" in runner._watchlist_set
        assert runner._watchlist == ["SPY", "CISS"]

    def test_removes_symbol_when_not_on_board(self) -> None:
        runner = _make_runner(["SPY", "VUZI"])
        runner._sync_watchlist_to_hod_alerts()
        assert "VUZI" not in runner._watchlist_set
        assert "SPY" in runner._watchlist_set

    def test_sync_from_recent_alert(self) -> None:
        runner = _make_runner(["SPY"])
        runner._hod_last_alert_at["AIIO"] = datetime.now(timezone.utc)
        runner._sync_watchlist_to_hod_alerts()
        assert "AIIO" in runner._watchlist_set

    def test_ttl_expires_watchlist_symbol(self) -> None:
        runner = _make_runner(["SPY", "OLD"])
        runner._hod_last_alert_at["OLD"] = datetime.now(timezone.utc) - timedelta(
            minutes=10,
        )
        runner._sync_watchlist_to_hod_alerts()
        assert "OLD" not in runner._watchlist_set
        assert "SPY" in runner._watchlist_set

    def test_news_pinned_symbol_stays_active(self) -> None:
        runner = _make_runner(["SPY", "NEWS"])
        runner._news_pinned.add("NEWS")
        runner._sync_watchlist_to_hod_alerts()
        assert "NEWS" in runner._watchlist_set
        assert "SPY" in runner._watchlist_set


class TestTradeUniverse:
    def test_excludes_pool_only_symbols(self) -> None:
        runner = _make_runner(["SPY", "CISS"])
        runner._hod_bar_pool = ["POOLONLY"]
        ts = datetime(2026, 5, 18, 14, 0, 0, tzinfo=timezone.utc)
        bar = Bar(
            symbol="POOLONLY",
            ts=ts,
            open=5.0,
            high=5.1,
            low=4.9,
            close=5.0,
            volume=100_000,
        )
        bar_universe = {
            "SPY": [bar],
            "CISS": [bar],
            "POOLONLY": [bar],
        }
        trade = runner._trade_universe(bar_universe)
        assert "POOLONLY" not in trade
        assert "CISS" in trade
        assert "SPY" in trade
