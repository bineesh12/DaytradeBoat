from __future__ import annotations

from datetime import datetime
from threading import Lock
from types import SimpleNamespace

from daytrading.runner import AlpacaRunner


class _Hub:
    def __init__(self) -> None:
        self.reset_count = 0
        self.logs: list[tuple[str, str]] = []

    def reset_daily_overview(self) -> None:
        self.reset_count += 1

    def add_log(self, level: str, message: str) -> None:
        self.logs.append((level, message))


def _runner() -> AlpacaRunner:
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._bar_buffer = {"OLD": []}
    runner._quote_buffer = {"OLD": []}
    runner._tick_buffer = {"OLD": []}
    runner._watchlist_pinned = {"SPY"}
    runner._watchlist = ["SPY", "OLD"]
    runner._watchlist_set = {"SPY", "OLD"}
    runner._hod_bar_scanner = None
    runner._hod_tick_tracker = None
    runner._hod_seed_retries = {"OLD": 1}
    runner._hod_seed_blacklist = {"OLD"}
    runner._hod_seed_queue = []
    runner._hod_seed_pending = {"OLD"}
    runner._hod_seed_lock = Lock()
    runner._pipeline = SimpleNamespace(_daily_pnl=-10.0, _daily_losers={"OLD"})
    runner._eod_flattened = True
    runner._network_failure_times = [1.0]
    runner._hydrate_paused_until = 1.0
    runner._trade_analyzer = None
    runner._hub = _Hub()
    runner._last_synced_order_ids = {"old-order"}
    runner._recorded_exit_fill_keys = {"old-fill"}
    runner._last_session_reset_day = None
    return runner


def test_daily_session_reset_runs_once_at_premarket(monkeypatch) -> None:
    runner = _runner()
    monkeypatch.setattr(AlpacaRunner, "_is_trading_day", classmethod(lambda cls, when=None: True))
    premarket = datetime(2026, 6, 2, 4, 0)

    assert runner._maybe_daily_session_reset(premarket, "PRE-MARKET") is True
    assert runner._maybe_daily_session_reset(premarket, "PRE-MARKET") is False

    assert runner._last_session_reset_day == "2026-06-02"
    assert runner._hub.reset_count == 1
    assert runner._pipeline._daily_pnl == 0.0
    assert runner._pipeline._daily_losers == set()
    assert runner._last_synced_order_ids == set()
    assert runner._recorded_exit_fill_keys == set()
    assert runner._watchlist == ["SPY"]


def test_daily_session_reset_waits_until_premarket(monkeypatch) -> None:
    runner = _runner()
    monkeypatch.setattr(AlpacaRunner, "_is_trading_day", classmethod(lambda cls, when=None: True))
    before_premarket = datetime(2026, 6, 2, 3, 59)

    assert runner._maybe_daily_session_reset(before_premarket, "CLOSED") is False
    assert runner._hub.reset_count == 0
    assert runner._last_session_reset_day is None
