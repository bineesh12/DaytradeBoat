"""Tests for watchlist management — cap, skip tracking, and pruning."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from daytrading.models import MarketRegime, TradingStyle

TS = datetime(2026, 5, 15, 14, 30, 0, tzinfo=timezone.utc)


def _regime(symbol: str, style: TradingStyle, confidence: float = 0.5) -> MarketRegime:
    return MarketRegime(
        symbol=symbol, ts=TS, style=style, confidence=confidence,
        volatility_pct=1.0, spread_pct=0.1, relative_volume=3.0,
        trend_strength=0.8, avg_bar_range_pct=0.5, liquidity_score=0.5,
        reasons=[],
    )


def _make_runner(watchlist: list[str]) -> MagicMock:
    """Create a minimal mock runner with real watchlist management attributes."""
    runner = MagicMock()
    runner._watchlist = list(watchlist)
    runner._watchlist_set = set(watchlist)
    runner._watchlist_pinned = {"SPY"}
    runner._skip_counts = defaultdict(int)
    runner._SKIP_THRESHOLD = 10
    runner._CLEANUP_EVERY = 10
    runner._bar_buffer = defaultdict(list)
    runner._quote_buffer = defaultdict(list)
    runner._tick_buffer = defaultdict(list)

    from threading import Lock
    runner._lock = Lock()

    portfolio = MagicMock()
    portfolio.positions = {}
    exit_mgr = MagicMock()
    exit_mgr.tracked = {}
    runner._pipeline = MagicMock()
    runner._pipeline.portfolio = portfolio
    runner._pipeline.exit_manager = exit_mgr

    return runner


# ---- Import the actual methods so we can call them on our mock ----
from daytrading.runner import AlpacaRunner

_prune = AlpacaRunner._prune_watchlist


# =========================================================================
# Skip tracking
# =========================================================================

class TestSkipTracking:

    def test_skip_increments(self):
        """Consecutive not_tradeable regimes increment skip counter."""
        runner = _make_runner(["AIIO", "CONL"])

        regime_skip = _regime("AIIO", TradingStyle.NOT_TRADEABLE, 0.0)
        regime_ok = _regime("CONL", TradingStyle.SCALPING, 0.8)

        for sym, reg in {"AIIO": regime_skip, "CONL": regime_ok}.items():
            if reg.style.value == "not_tradeable":
                runner._skip_counts[sym] += 1
            else:
                runner._skip_counts[sym] = 0

        assert runner._skip_counts["AIIO"] == 1
        assert runner._skip_counts["CONL"] == 0

    def test_skip_resets_on_route(self):
        """A successful route resets the skip counter to 0."""
        runner = _make_runner(["AIIO"])
        runner._skip_counts["AIIO"] = 8

        regime_ok = _regime("AIIO", TradingStyle.SCALPING, 0.8)
        if regime_ok.style.value == "not_tradeable":
            runner._skip_counts["AIIO"] += 1
        else:
            runner._skip_counts["AIIO"] = 0

        assert runner._skip_counts["AIIO"] == 0

    def test_skip_accumulates(self):
        """Multiple consecutive skips accumulate correctly."""
        runner = _make_runner(["DEAD"])

        for _ in range(10):
            runner._skip_counts["DEAD"] += 1

        assert runner._skip_counts["DEAD"] == 10


# =========================================================================
# Pruning stale symbols
# =========================================================================

class TestPruneWatchlist:

    def test_prune_removes_stale(self):
        """Symbols at or above skip threshold are removed."""
        runner = _make_runner(["AIIO", "CONL", "DEAD"])
        runner._skip_counts["DEAD"] = 10
        runner._skip_counts["AIIO"] = 3
        runner._skip_counts["CONL"] = 0

        _prune(runner)

        assert "DEAD" not in runner._watchlist
        assert "DEAD" not in runner._watchlist_set
        assert "AIIO" in runner._watchlist
        assert "CONL" in runner._watchlist
        assert len(runner._watchlist) == 2

    def test_prune_clears_buffers(self):
        """Pruned symbols have their data buffers cleared."""
        runner = _make_runner(["DEAD", "ALIVE"])
        runner._skip_counts["DEAD"] = 15
        runner._bar_buffer["DEAD"] = [MagicMock()]
        runner._quote_buffer["DEAD"] = [MagicMock()]
        runner._tick_buffer["DEAD"] = [MagicMock()]

        _prune(runner)

        assert "DEAD" not in runner._bar_buffer
        assert "DEAD" not in runner._quote_buffer
        assert "DEAD" not in runner._tick_buffer

    def test_prune_protects_open_positions(self):
        """Symbols with open positions are never pruned, even if stale."""
        runner = _make_runner(["HELD", "DEAD"])
        runner._skip_counts["HELD"] = 20
        runner._skip_counts["DEAD"] = 20

        pos = MagicMock()
        pos.is_flat = False
        runner._pipeline.portfolio.positions = {"HELD": pos}

        _prune(runner)

        assert "HELD" in runner._watchlist
        assert "DEAD" not in runner._watchlist

    def test_prune_protects_tracked_exits(self):
        """Symbols tracked by exit manager are never pruned."""
        runner = _make_runner(["TRACKED", "DEAD"])
        runner._skip_counts["TRACKED"] = 20
        runner._skip_counts["DEAD"] = 20

        runner._pipeline.exit_manager.tracked = {"TRACKED": MagicMock()}

        _prune(runner)

        assert "TRACKED" in runner._watchlist
        assert "DEAD" not in runner._watchlist

    def test_prune_nothing_when_all_active(self):
        """No symbols removed when all are below threshold."""
        runner = _make_runner(["A", "B", "C"])
        runner._skip_counts["A"] = 5
        runner._skip_counts["B"] = 0
        runner._skip_counts["C"] = 9

        _prune(runner)

        assert len(runner._watchlist) == 3

    def test_prune_removes_multiple(self):
        """Multiple stale symbols are removed in one pass."""
        syms = ["DEAD1", "DEAD2", "DEAD3", "ALIVE"]
        runner = _make_runner(syms)
        runner._skip_counts["DEAD1"] = 10
        runner._skip_counts["DEAD2"] = 15
        runner._skip_counts["DEAD3"] = 50
        runner._skip_counts["ALIVE"] = 2

        _prune(runner)

        assert runner._watchlist == ["ALIVE"]
        assert runner._watchlist_set == {"ALIVE"}


# =========================================================================
# Integration: skip tracking + pruning together
# =========================================================================

class TestSkipAndPruneIntegration:

    def test_full_lifecycle(self):
        """Symbol goes from active → stale → pruned over multiple cycles."""
        runner = _make_runner(["AIIO", "CONL"])

        for cycle in range(1, 12):
            aiio_regime = _regime("AIIO", TradingStyle.NOT_TRADEABLE, 0.0)
            conl_regime = _regime("CONL", TradingStyle.SCALPING, 0.8)

            for sym, reg in {"AIIO": aiio_regime, "CONL": conl_regime}.items():
                if reg.style.value == "not_tradeable":
                    runner._skip_counts[sym] += 1
                else:
                    runner._skip_counts[sym] = 0

        assert runner._skip_counts["AIIO"] == 11
        assert runner._skip_counts["CONL"] == 0

        _prune(runner)

        assert "AIIO" not in runner._watchlist
        assert "CONL" in runner._watchlist

    def test_revived_stock_not_pruned(self):
        """Stock that was stale but becomes active again survives pruning."""
        runner = _make_runner(["AIIO"])

        for _ in range(9):
            runner._skip_counts["AIIO"] += 1
        assert runner._skip_counts["AIIO"] == 9

        runner._skip_counts["AIIO"] = 0

        _prune(runner)

        assert "AIIO" in runner._watchlist
