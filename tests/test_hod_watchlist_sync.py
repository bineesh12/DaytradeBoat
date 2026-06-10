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
    _on_hod_alerts_changed = AlpacaRunner._on_hod_alerts_changed
    _is_tradeable_hod_watchlist_alert = AlpacaRunner._is_tradeable_hod_watchlist_alert
    _parse_hod_alert_time = staticmethod(AlpacaRunner._parse_hod_alert_time)
    _publish_trading_watchlist = AlpacaRunner._publish_trading_watchlist
    _prune_hot_watch = AlpacaRunner._prune_hot_watch
    _hot_watch_snapshot = AlpacaRunner._hot_watch_snapshot
    _hot_watch_live_metrics = AlpacaRunner._hot_watch_live_metrics
    _publish_hot_watch = AlpacaRunner._publish_hot_watch
    _promote_hot_watch = AlpacaRunner._promote_hot_watch
    _hot_watch_reject_reason = AlpacaRunner._hot_watch_reject_reason
    _hot_watch_mode = AlpacaRunner._hot_watch_mode
    is_hot_watch_active = AlpacaRunner.is_hot_watch_active
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
        self._hod_tradeable_alert_at: dict = {}
        self._hod_watchlist_min_day_volume = 500_000
        self._hod_watchlist_min_rel_vol = 1.0
        self._hod_watchlist_min_bar_rvol = 1.2
        self._hod_sub2_enabled = True
        self._hod_sub2_min_price = 1.0
        self._hod_min_price = 2.0
        self._hod_max_price = 20.0
        self._hod_max_float = 20_000_000
        self._hot_watch: dict = {}
        self._hot_watch_ttl_minutes = 8.0
        self._hot_watch_strong_ttl_minutes = 15.0
        self._hot_watch_runner_ttl_minutes = 25.0
        self._hot_watch_max_symbols = 40
        self._hot_watch_enabled = True
        self._hot_watch_min_change_pct = 5.0
        self._hot_watch_min_day_volume = 200_000
        self._hot_watch_min_score = 0.30
        self._hot_watch_sub5_min_day_volume = 500_000
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
        self._journal = MagicMock()
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
        self._phase = "OPEN"

    def _market_phase(self) -> str:
        return self._phase


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


class _CachedOnlyFloat:
    def get_float_cached(self, symbol: str):
        return None

    def get_float(self, symbol: str):
        raise AssertionError("fast scan must not perform network float lookups")


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


class TestHotWatchStreaming:
    def test_ensure_streaming_symbols_requests_trade_ticks_for_10s_bars(self) -> None:
        runner = _make_runner(["SPY"])
        runner._hod_tick_tracker = MagicMock()

        AlpacaRunner._ensure_streaming_symbols(runner, ["fofo", "STAK"])

        runner._stream.subscribe.assert_called_once_with(["FOFO", "STAK"], bars=True, quotes=True)
        runner._hod_tick_tracker.add_known_symbols.assert_called_once_with(["FOFO", "STAK"])
        runner._stream.add_trade_filter_symbols.assert_called_once_with(["FOFO", "STAK"])
        runner._stream.flush_pending_subscriptions.assert_not_called()

    def test_ensure_streaming_symbols_flushes_without_tick_tracker(self) -> None:
        runner = _make_runner(["SPY"])
        runner._hod_tick_tracker = None

        AlpacaRunner._ensure_streaming_symbols(runner, ["FOFO"])

        runner._stream.subscribe.assert_called_once_with(["FOFO"], bars=True, quotes=True)
        runner._stream.flush_pending_subscriptions.assert_called_once_with()
        runner._stream.add_trade_filter_symbols.assert_not_called()


class TestFastScanHandling:
    def test_fast_scan_uses_cached_float_only(self) -> None:
        runner = AlpacaRunner.__new__(AlpacaRunner)
        runner._float_checker = _CachedOnlyFloat()
        runner._hod_bar_pool = []
        runner._hod_max_float = 20_000_000
        runner._hod_pool_max = 1000
        runner._bar_buffer = {}
        runner._prior_day_stats = {}
        runner._max_bars_per_symbol = 100
        runner._hod_hydrate_batch_max = 25
        runner._hub = MagicMock()
        runner._journal = MagicMock()
        runner._fetch_session_bars = MagicMock(return_value={})
        runner._fetch_prior_day_stats = MagicMock(return_value={})
        runner._seed_hod_session = MagicMock()
        runner._sync_tick_tracker_pool = MagicMock()
        runner._hot_watch_reject_reason = MagicMock(return_value=None)
        runner._promote_hot_watch = MagicMock()

        loaded = AlpacaRunner._handle_fast_scan_movers(runner, [{
            "symbol": "FAST",
            "price": 4.5,
            "abs_change_pct": 25.0,
            "change_pct": 25.0,
            "volume": 750_000,
            "score": 0.4,
        }])

        assert loaded == 0
        assert "FAST" in runner._hod_bar_pool
        runner._promote_hot_watch.assert_called_once()
        runner._fetch_session_bars.assert_called_once_with(["FAST"])


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
        runner._hod_tradeable_alert_at["AIIO"] = datetime.now(timezone.utc)
        runner._sync_watchlist_to_hod_alerts()
        assert "AIIO" in runner._watchlist_set

    def test_ttl_expires_watchlist_symbol(self) -> None:
        runner = _make_runner(["SPY", "OLD"])
        runner._hod_tradeable_alert_at["OLD"] = datetime.now(timezone.utc) - timedelta(
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

    def test_hod_alert_with_weak_liquidity_stays_watch_only(self) -> None:
        runner = _make_runner(["SPY"])
        now = datetime.now(timezone.utc).isoformat()

        runner._on_hod_alerts_changed([{
            "symbol": "THIN",
            "time": now,
            "price": 4.25,
            "alert_name": "Gapper Continuation",
            "day_volume": 220_000,
            "rel_vol": 0.4,
            "bar_rvol": 0.7,
        }])

        assert "THIN" in runner._hod_last_alert_at
        assert "THIN" not in runner._hod_tradeable_alert_at
        assert "THIN" not in runner._watchlist_set

    def test_hod_alert_with_liquidity_enters_trading_watchlist(self) -> None:
        runner = _make_runner(["SPY"])
        now = datetime.now(timezone.utc).isoformat()

        runner._on_hod_alerts_changed([{
            "symbol": "LIQD",
            "time": now,
            "price": 4.25,
            "alert_name": "Gapper Continuation",
            "day_volume": 650_000,
            "rel_vol": 1.4,
            "bar_rvol": 0.8,
        }])

        assert "LIQD" in runner._hod_tradeable_alert_at
        assert "LIQD" in runner._watchlist_set

    def test_sub_two_hod_alert_above_dollar_fifty_can_enter_trading_watchlist(self) -> None:
        runner = _make_runner(["SPY"])
        now = datetime.now(timezone.utc).isoformat()

        runner._on_hod_alerts_changed([{
            "symbol": "SUBT",
            "time": now,
            "price": 1.87,
            "alert_name": "Gapper Continuation",
            "day_volume": 2_500_000,
            "rel_vol": 1.4,
            "bar_rvol": 1.1,
        }])

        assert "SUBT" in runner._hod_tradeable_alert_at
        assert "SUBT" in runner._watchlist_set

    def test_hod_alert_below_dollar_fifty_stays_watch_only(self) -> None:
        runner = _make_runner(["SPY"])
        now = datetime.now(timezone.utc).isoformat()

        runner._on_hod_alerts_changed([{
            "symbol": "LOWP",
            "time": now,
            "price": 1.49,
            "alert_name": "Gapper Continuation",
            "day_volume": 5_000_000,
            "rel_vol": 2.0,
            "bar_rvol": 1.5,
        }])

        assert "LOWP" in runner._hod_last_alert_at
        assert "LOWP" not in runner._hod_tradeable_alert_at
        assert "LOWP" not in runner._watchlist_set


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

    def test_includes_active_hot_watch_symbol(self) -> None:
        runner = _make_runner(["SPY"])
        runner._hot_watch["CODX"] = {
            "added_at": datetime.now(timezone.utc),
            "reason": "fast scan mover",
        }
        ts = datetime(2026, 6, 2, 13, 35, 0, tzinfo=timezone.utc)
        bar = Bar(
            symbol="CODX",
            ts=ts,
            open=7.0,
            high=7.5,
            low=6.9,
            close=7.4,
            volume=200_000,
        )
        trade = runner._trade_universe({"SPY": [bar], "CODX": [bar]})
        assert "CODX" in trade

    def test_published_trading_watchlist_includes_hot_watch_symbols(self) -> None:
        runner = _make_runner(["SPY"])
        runner._hot_watch["CODX"] = {
            "added_at": datetime.now(timezone.utc),
            "reason": "fast scan mover",
        }

        runner._publish_trading_watchlist()

        symbols = runner._hub.on_trading_watchlist.call_args.args[0]
        assert symbols == ["CODX", "SPY"]

    def test_promoting_hot_watch_republishes_trading_watchlist(self) -> None:
        runner = _make_runner(["SPY"])

        runner._promote_hot_watch(
            {
                "symbol": "XOS",
                "price": 7.34,
                "change_pct": 229.0,
                "abs_change_pct": 229.0,
                "volume": 98_000_000,
                "score": 0.55,
            },
            flt=6_000_000,
            reason="fast scan mover",
        )

        symbols = runner._hub.on_trading_watchlist.call_args.args[0]
        assert symbols == ["SPY", "XOS"] or symbols == ["XOS", "SPY"]

    def test_expired_hot_watch_symbol_drops_from_trade_universe(self) -> None:
        runner = _make_runner(["SPY"])
        runner._hot_watch_ttl_minutes = 5.0
        runner._hot_watch["OLD"] = {
            "added_at": datetime.now(timezone.utc) - timedelta(minutes=10),
            "reason": "fast scan mover",
        }
        ts = datetime(2026, 6, 2, 13, 35, 0, tzinfo=timezone.utc)
        bar = Bar(
            symbol="OLD",
            ts=ts,
            open=7.0,
            high=7.5,
            low=6.9,
            close=7.4,
            volume=200_000,
        )
        trade = runner._trade_universe({"SPY": [bar], "OLD": [bar]})
        assert "OLD" not in trade
        runner._journal.record.assert_called()


class TestHotWatchTTL:
    def test_normal_hot_watch_expires_after_base_ttl(self) -> None:
        runner = _make_runner(["SPY"])
        runner._hot_watch["NORM"] = {
            "added_at": datetime.now(timezone.utc) - timedelta(minutes=9),
            "ttl_minutes": 8.0,
            "mode": "watch",
        }

        assert not runner.is_hot_watch_active("NORM")
        assert "NORM" not in runner._hot_watch

    def test_strong_hot_watch_stays_longer_than_normal(self) -> None:
        runner = _make_runner(["SPY"])
        runner._hot_watch["STRONG"] = {
            "added_at": datetime.now(timezone.utc) - timedelta(minutes=9),
            "ttl_minutes": 15.0,
            "mode": "strong_watch",
        }

        assert runner.is_hot_watch_active("STRONG")

    def test_runner_hot_watch_stays_longest_but_expires(self) -> None:
        runner = _make_runner(["SPY"])
        runner._hot_watch["RUN"] = {
            "added_at": datetime.now(timezone.utc) - timedelta(minutes=20),
            "ttl_minutes": 25.0,
            "mode": "runner_watch",
        }
        assert runner.is_hot_watch_active("RUN")

        runner._hot_watch["RUN"]["added_at"] = (
            datetime.now(timezone.utc) - timedelta(minutes=26)
        )
        assert not runner.is_hot_watch_active("RUN")

    def test_hot_watch_mode_from_mover_strength(self) -> None:
        runner = _make_runner(["SPY"])

        assert runner._hot_watch_mode({
            "symbol": "BASE", "price": 6.0, "abs_change_pct": 6.0,
            "volume": 250_000, "score": 0.31,
        }) == ("watch", 8.0)

        assert runner._hot_watch_mode({
            "symbol": "STRONG", "price": 4.0, "abs_change_pct": 12.0,
            "volume": 1_100_000, "score": 0.35,
        }) == ("strong_watch", 15.0)

        assert runner._hot_watch_mode({
            "symbol": "RUN", "price": 7.0, "abs_change_pct": 30.0,
            "volume": 2_000_000, "score": 0.45,
        }) == ("runner_watch", 25.0)

    def test_hot_watch_premarket_uses_recent_volume_threshold(self) -> None:
        runner = _make_runner(["SPY"])
        runner._phase = "PRE-MARKET"

        reason = runner._hot_watch_reject_reason({
            "symbol": "MOBX",
            "price": 2.83,
            "abs_change_pct": 26.9,
            "volume": 51_430,
            "score": 0.59,
        }, flt=5_000_000)

        assert reason is None

    def test_hot_watch_open_market_keeps_day_volume_threshold(self) -> None:
        runner = _make_runner(["SPY"])
        runner._phase = "OPEN"

        reason = runner._hot_watch_reject_reason({
            "symbol": "MOBX",
            "price": 2.83,
            "abs_change_pct": 26.9,
            "volume": 51_430,
            "score": 0.59,
        }, flt=5_000_000)

        assert reason == "volume 51430 < 500000"

    def test_expired_symbol_can_be_added_again(self) -> None:
        runner = _make_runner(["SPY"])
        runner._hot_watch["BACK"] = {
            "added_at": datetime.now(timezone.utc) - timedelta(minutes=9),
            "ttl_minutes": 8.0,
            "mode": "watch",
        }
        assert not runner.is_hot_watch_active("BACK")

        runner._promote_hot_watch(
            {
                "symbol": "BACK", "price": 6.0, "abs_change_pct": 7.0,
                "change_pct": 7.0, "volume": 300_000, "score": 0.32,
            },
            flt=5_000_000,
            reason="fast scan mover",
        )

        assert runner.is_hot_watch_active("BACK")
        assert runner._hot_watch["BACK"]["mode"] == "watch"

    def test_strong_refresh_extends_active_window(self) -> None:
        runner = _make_runner(["SPY"])
        old_added_at = datetime.now(timezone.utc) - timedelta(minutes=14)
        runner._hot_watch["STRONG"] = {
            "added_at": old_added_at,
            "ttl_minutes": 15.0,
            "mode": "strong_watch",
        }

        runner._promote_hot_watch(
            {
                "symbol": "STRONG", "price": 6.0, "abs_change_pct": 14.0,
                "change_pct": 14.0, "volume": 900_000, "score": 0.40,
            },
            flt=5_000_000,
            reason="fast scan mover",
        )

        assert runner._hot_watch["STRONG"]["added_at"] > old_added_at
        assert runner._hot_watch["STRONG"]["ttl_minutes"] == 15.0

    def test_hot_watch_snapshot_separates_session_change_from_current_pullback(self) -> None:
        runner = _make_runner(["SPY"])
        runner._hot_watch["HUBC"] = {
            "added_at": datetime.now(timezone.utc),
            "ttl_minutes": 25.0,
            "mode": "runner_watch",
            "reason": "fast scan mover",
            "mover": {
                "symbol": "HUBC",
                "price": 1.76,
                "change_pct": 821.47,
                "abs_change_pct": 821.47,
                "volume": 4_928_425,
                "score": 0.64,
            },
        }
        ts = datetime(2026, 6, 8, 18, 0, tzinfo=timezone.utc)
        runner._bar_buffer["HUBC"] = [
            Bar("HUBC", ts, 1.40, 1.55, 1.35, 1.50, 20_000),
            Bar("HUBC", ts + timedelta(minutes=1), 1.50, 1.70, 1.49, 1.66, 30_000),
            Bar("HUBC", ts + timedelta(minutes=2), 1.66, 1.77, 1.62, 1.73, 4_125),
        ]

        row = runner._hot_watch_snapshot()[0]

        assert row["change_pct"] == 821.47
        assert row["short_change_pct"] == 15.33
        assert row["session_high"] == 1.77
        assert row["pullback_from_high_pct"] == -2.26
