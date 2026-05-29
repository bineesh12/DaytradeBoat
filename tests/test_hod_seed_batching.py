"""Tests for batched HOD seed worker and enqueue filtering."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from unittest.mock import MagicMock

from daytrading.config import Settings
from daytrading.runner import AlpacaRunner


class _SeedStub:
    _eligible_for_hod_seed = AlpacaRunner._eligible_for_hod_seed
    _pull_hod_seed_batch = AlpacaRunner._pull_hod_seed_batch
    _hod_seed_rate_limit = AlpacaRunner._hod_seed_rate_limit

    def __init__(self) -> None:
        self._hod_seed_queue: deque = deque()
        self._hod_seed_pending: set = set()
        self._hod_seed_lock = Lock()
        self._hod_seed_batch_size = 10
        self._hod_seed_max_per_minute = 30
        self._hod_seed_minute_start = time.time()
        self._hod_seed_processed_this_minute = 0
        self._watchlist_pinned = {"SPY"}
        self._watchlist_set: set = set()
        self._hod_bar_pool: list = []


class TestHodSeedEligibility:
    def test_pool_symbol_eligible(self) -> None:
        r = _SeedStub()
        r._hod_bar_pool = ["BKD", "FINV"]
        assert r._eligible_for_hod_seed("BKD") is True

    def test_watchlist_symbol_eligible(self) -> None:
        r = _SeedStub()
        r._watchlist_set = {"AIRS"}
        assert r._eligible_for_hod_seed("AIRS") is True

    def test_random_tape_symbol_not_eligible(self) -> None:
        r = _SeedStub()
        r._hod_bar_pool = ["BKD"]
        assert r._eligible_for_hod_seed("RANDOM") is False

    def test_pinned_not_eligible(self) -> None:
        r = _SeedStub()
        r._hod_bar_pool = ["SPY"]
        assert r._eligible_for_hod_seed("SPY") is False


class TestHodSeedBatchDrain:
    def test_pull_respects_max_count(self) -> None:
        r = _SeedStub()
        for sym in ("A", "B", "C", "D", "E"):
            r._hod_seed_queue.append(sym)
            r._hod_seed_pending.add(sym)
        batch = r._pull_hod_seed_batch(3)
        assert batch == ["A", "B", "C"]
        assert r._hod_seed_pending == {"D", "E"}
        assert list(r._hod_seed_queue) == ["D", "E"]

    def test_pull_clears_pending(self) -> None:
        r = _SeedStub()
        r._hod_seed_queue.append("X")
        r._hod_seed_pending.add("X")
        r._pull_hod_seed_batch(10)
        assert "X" not in r._hod_seed_pending


class TestHodSeedRateLimit:
    def test_quota_decreases_with_processed(self) -> None:
        r = _SeedStub()
        r._hod_seed_max_per_minute = 30
        r._hod_seed_processed_this_minute = 25
        assert r._hod_seed_rate_limit() == 5

    def test_quota_zero_when_exhausted(self) -> None:
        r = _SeedStub()
        r._hod_seed_processed_this_minute = 30
        assert r._hod_seed_rate_limit() == 0

    def test_quota_resets_after_minute(self) -> None:
        r = _SeedStub()
        r._hod_seed_processed_this_minute = 30
        r._hod_seed_minute_start = time.time() - 61.0
        assert r._hod_seed_rate_limit() == 30
        assert r._hod_seed_processed_this_minute == 0


class TestHodSeedConfig:
    def test_config_defaults(self) -> None:
        cfg = Settings()
        assert cfg.hod_seed_batch_size == 10
        assert cfg.hod_seed_max_per_minute == 100


class TestHodSeedWorkerBatching:
    def test_process_batch_calls_batched_load(self) -> None:
        runner = MagicMock()
        runner._bar_hydrate_paused.return_value = False
        runner._watchlist_pinned = set()
        runner._bar_buffer = {}
        runner._hod_bar_pool = ["BKD"]
        runner._hod_pool_max = 50
        runner._watchlist_set = set()
        runner._hod_tick_tracker = None
        runner._fetch_session_bars.return_value = {"BKD": [], "FINV": []}
        runner._fetch_prior_day_stats.return_value = {}

        import queue as _q
        runner._event_queue = _q.Queue(maxsize=100)
        from threading import Event as _Ev
        runner._new_data = _Ev()

        AlpacaRunner._process_hod_seed_batch(runner, ["BKD", "FINV"])

        runner._fetch_session_bars.assert_called_once()
        symbols = runner._fetch_session_bars.call_args[0][0]
        assert set(symbols) == {"BKD", "FINV"}
        runner._fetch_prior_day_stats.assert_called_once_with(symbols)
