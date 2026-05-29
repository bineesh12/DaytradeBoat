"""Tests for HOD bar pool merge, prune, and price band."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from threading import Lock
from unittest.mock import MagicMock

from daytrading.models import Bar
from daytrading.runner import AlpacaRunner


class _RunnerStub:
    _latest_price = AlpacaRunner._latest_price
    _symbol_in_hod_price_band = AlpacaRunner._symbol_in_hod_price_band
    _prune_hod_bar_pool_by_price = AlpacaRunner._prune_hod_bar_pool_by_price
    _merge_hod_bar_pool = AlpacaRunner._merge_hod_bar_pool

    def __init__(self) -> None:
        self._hod_bar_pool: list = []
        self._hod_min_price = 2.0
        self._hod_max_price = 20.0
        self._hod_pool_max = 50
        self._bar_buffer = defaultdict(list)
        self._quote_buffer = defaultdict(list)
        self._lock = Lock()


def _bar(sym: str, close: float) -> Bar:
    return Bar(
        symbol=sym,
        ts=datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc),
        open=close,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=100_000,
    )


class TestHodPoolPriceBand:
    def test_prune_removes_outside_band(self) -> None:
        runner = _RunnerStub()
        runner._hod_bar_pool = ["LOW", "HIGH"]
        runner._bar_buffer["LOW"] = [_bar("LOW", 5.0)]
        runner._bar_buffer["HIGH"] = [_bar("HIGH", 25.0)]
        removed = runner._prune_hod_bar_pool_by_price()
        assert "HIGH" in removed
        assert runner._hod_bar_pool == ["LOW"]

    def test_merge_caps_and_adds(self) -> None:
        runner = _RunnerStub()
        runner._hod_bar_pool = ["OLD"]
        runner._bar_buffer["OLD"] = [_bar("OLD", 6.0)]
        added, removed = runner._merge_hod_bar_pool(["NEW1", "NEW2"])
        assert "NEW1" in runner._hod_bar_pool
        assert "NEW1" in added
        assert len(runner._hod_bar_pool) <= 50

    def test_symbol_in_band(self) -> None:
        runner = _RunnerStub()
        runner._bar_buffer["X"] = [_bar("X", 8.0)]
        assert runner._symbol_in_hod_price_band("X") is True
        runner._bar_buffer["Y"] = [_bar("Y", 30.0)]
        assert runner._symbol_in_hod_price_band("Y") is False


class TestBuildFloatPool:
    def test_stops_at_pool_max(self) -> None:
        class _Checker:
            def warm_from_store(self, symbols):
                return 0, len(symbols)

            def get_float_cached(self, symbol):
                return None

            def get_float(self, symbol):
                return 1_000_000

        candidates = [
            {"symbol": "A{}".format(i), "price": 5.0} for i in range(20)
        ]
        pool = AlpacaRunner.build_float_filtered_hod_pool(
            candidates,
            _Checker(),
            pool_max=5,
        )
        assert len(pool) == 5

    def test_prioritizes_strong_movers_beyond_top_rank(self) -> None:
        class _Checker:
            def warm_from_store(self, symbols):
                return 0, len(symbols)

            def get_float_cached(self, symbol):
                return {
                    "WNW": 155_000,
                    "TOP0": 1_000_000,
                    "TOP1": 1_000_000,
                    "TOP2": 1_000_000,
                }.get(symbol)

            def get_float(self, symbol):
                return 1_000_000

        candidates = [
            {"symbol": "TOP0", "price": 5.0, "change_pct": 5.0, "volume": 1_000_000},
            {"symbol": "TOP1", "price": 5.0, "change_pct": 4.0, "volume": 1_000_000},
            {"symbol": "TOP2", "price": 5.0, "change_pct": 3.0, "volume": 1_000_000},
            {"symbol": "WNW", "price": 5.9, "change_pct": 55.0, "volume": 150_000},
        ]

        pool = AlpacaRunner.build_float_filtered_hod_pool(
            candidates,
            _Checker(),
            pool_max=2,
        )

        assert "WNW" in pool
