"""Tests for SQLite float cache."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

from daytrading.data.float_checker import FloatChecker
from daytrading.data.float_store import FloatRecord, FloatStore
from daytrading.runner import AlpacaRunner


def test_upsert_and_get_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        store = FloatStore(db_path=db)
        store.upsert("aapl", 15_000_000_000, 16_000_000_000, avg_volume=50_000_000)
        rec = store.get("AAPL")
        assert rec is not None
        assert rec.float_shares == 15_000_000_000
        assert rec.outstanding_shares == 16_000_000_000
        assert rec.avg_volume == 50_000_000
        store.close()


def test_fresh_vs_stale() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        store = FloatStore(db_path=db)
        old = datetime.now(timezone.utc) - timedelta(days=10)
        store.upsert("STALE", 5_000_000, fetched_at=old)
        store.upsert("FRESH", 5_000_000)
        assert store.is_fresh("FRESH", ttl_days=7)
        assert not store.is_fresh("STALE", ttl_days=7)
        store.close()


def test_bulk_get() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        store = FloatStore(db_path=db)
        store.upsert("AAA", 1_000_000)
        store.upsert("BBB", 2_000_000)
        rows = store.bulk_get(["AAA", "BBB", "MISSING"])
        assert set(rows.keys()) == {"AAA", "BBB"}
        assert rows["AAA"].float_shares == 1_000_000
        store.close()


def test_float_checker_uses_db_without_yahoo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        store = FloatStore(db_path=db)
        store.upsert("LOW", 5_000_000)
        checker = FloatChecker(store=store, cache_ttl_days=7)
        assert checker.get_float("LOW") == 5_000_000
        assert checker.cache_size == 1
        store.close()


def test_warm_from_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        store = FloatStore(db_path=db)
        store.upsert("X", 3_000_000)
        checker = FloatChecker(store=store, cache_ttl_days=7)
        from_db, need = checker.warm_from_store(["X", "Y"])
        assert from_db == 1
        assert need == 1
        assert checker.get_float("X") == 3_000_000
        store.close()


def test_build_pool_uses_float_checker_cache() -> None:
    class _Checker:
        def __init__(self) -> None:
            self.cached_calls: list = []
            self.network_calls: list = []

        def warm_from_store(self, symbols):
            return len(symbols), 0

        def needs_yahoo_refresh(self, symbol):
            return False

        def get_float_cached(self, symbol):
            self.cached_calls.append(symbol)
            floats = {"LOW": 5_000_000, "BIG": 50_000_000}
            return floats.get(symbol)

        def get_float(self, symbol):
            self.network_calls.append(symbol)
            floats = {"LOW": 5_000_000, "BIG": 50_000_000}
            return floats.get(symbol)

    checker = _Checker()
    pool = AlpacaRunner.build_float_filtered_hod_pool(
        [
            {"symbol": "LOW", "price": 5.0},
            {"symbol": "BIG", "price": 8.0},
        ],
        checker,
        max_float=20_000_000,
        pool_max=50,
    )
    assert pool == ["LOW"]
    assert "BIG" not in pool
    assert checker.cached_calls == ["LOW", "BIG"]
    assert checker.network_calls == []


def test_build_pool_falls_back_to_network_for_uncached_float() -> None:
    class _Checker:
        def __init__(self) -> None:
            self.cached_calls: list = []
            self.network_calls: list = []

        def warm_from_store(self, symbols):
            return 0, len(symbols)

        def get_float_cached(self, symbol):
            self.cached_calls.append(symbol)
            return None

        def get_float(self, symbol):
            self.network_calls.append(symbol)
            floats = {"LOW": 5_000_000, "BIG": 50_000_000}
            return floats.get(symbol)

    checker = _Checker()
    pool = AlpacaRunner.build_float_filtered_hod_pool(
        [
            {"symbol": "LOW", "price": 5.0},
            {"symbol": "BIG", "price": 8.0},
        ],
        checker,
        max_float=20_000_000,
        pool_max=50,
    )
    assert pool == ["LOW"]
    assert checker.cached_calls == ["LOW", "BIG"]
    assert checker.network_calls == ["LOW", "BIG"]
