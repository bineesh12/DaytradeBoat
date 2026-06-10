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
    runner._pipeline = SimpleNamespace(
        _daily_pnl=-10.0,
        _daily_losers={"OLD"},
        _daily_loss_counts={"OLD": 1},
    )
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
    assert runner._pipeline._daily_loss_counts == {}
    assert runner._last_synced_order_ids == {"old-order"}
    assert runner._recorded_exit_fill_keys == set()
    assert runner._watchlist == ["SPY"]


def test_daily_session_reset_restores_today_trade_history_when_broker_exists(monkeypatch) -> None:
    runner = _runner()
    runner._broker = object()
    runner._sync_calls = 0
    monkeypatch.setattr(AlpacaRunner, "_is_trading_day", classmethod(lambda cls, when=None: True))

    def sync_trade_history() -> None:
        runner._sync_calls += 1

    runner._sync_trade_history = sync_trade_history

    assert runner._maybe_daily_session_reset(datetime(2026, 6, 2, 4, 0), "PRE-MARKET") is True

    assert runner._hub.reset_count == 1
    assert runner._sync_calls == 1


def test_daily_session_reset_waits_until_premarket(monkeypatch) -> None:
    runner = _runner()
    monkeypatch.setattr(AlpacaRunner, "_is_trading_day", classmethod(lambda cls, when=None: True))
    before_premarket = datetime(2026, 6, 2, 3, 59)

    assert runner._maybe_daily_session_reset(before_premarket, "CLOSED") is False
    assert runner._hub.reset_count == 0
    assert runner._last_session_reset_day is None


def test_premarket_phase_transition_does_not_block_on_history_reload(monkeypatch) -> None:
    runner = _runner()
    runner._after_hours_enabled = False
    runner._refresh_requests = 0
    runner._history_loads = 0
    monkeypatch.setattr(AlpacaRunner, "_is_trading_day", classmethod(lambda cls, when=None: True))

    def request_refresh() -> None:
        runner._refresh_requests += 1

    def load_history() -> None:
        runner._history_loads += 1

    runner._request_pool_refresh_now = request_refresh
    runner._load_history = load_history

    runner._handle_market_phase_transition(
        "CLOSED",
        "PRE-MARKET",
        datetime(2026, 6, 3, 4, 0),
    )

    assert runner._hub.reset_count == 1
    assert runner._refresh_requests == 1
    assert runner._history_loads == 0


def _runner_for_after_market_training(tmp_path) -> AlpacaRunner:
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._after_hours_enabled = False
    runner._nightly_analysis_day = None
    runner._journal = SimpleNamespace(
        base_dir=str(tmp_path / "journal"),
        db_path=str(tmp_path / "journal" / "journal.db"),
    )
    runner._pipeline = SimpleNamespace(
        exit_manager=SimpleNamespace(tracked={}),
    )
    runner._hub = _Hub()
    runner._nightly_calls = 0

    def run_nightly() -> None:
        runner._nightly_calls += 1

    runner._run_nightly_analysis = run_nightly
    return runner


def test_after_market_training_runs_once_after_regular_close(monkeypatch, tmp_path) -> None:
    runner = _runner_for_after_market_training(tmp_path)
    monkeypatch.setattr(AlpacaRunner, "_is_trading_day", classmethod(lambda cls, when=None: True))

    before_cutoff = datetime(2026, 6, 2, 16, 4)
    at_cutoff = datetime(2026, 6, 2, 16, 5)

    assert runner._maybe_run_after_market_training(before_cutoff, "AFTER-HOURS") is False
    assert runner._maybe_run_after_market_training(at_cutoff, "AFTER-HOURS") is True
    assert runner._maybe_run_after_market_training(at_cutoff, "AFTER-HOURS") is False
    assert runner._nightly_calls == 1


def test_after_market_training_waits_for_open_positions(monkeypatch, tmp_path) -> None:
    runner = _runner_for_after_market_training(tmp_path)
    runner._pipeline.exit_manager.tracked = {"OPEN": object()}
    monkeypatch.setattr(AlpacaRunner, "_is_trading_day", classmethod(lambda cls, when=None: True))

    after_cutoff = datetime(2026, 6, 2, 16, 6)

    assert runner._maybe_run_after_market_training(after_cutoff, "AFTER-HOURS") is False
    assert runner._nightly_calls == 0


def test_after_market_training_waits_until_after_hours_close(monkeypatch, tmp_path) -> None:
    runner = _runner_for_after_market_training(tmp_path)
    runner._after_hours_enabled = True
    monkeypatch.setattr(AlpacaRunner, "_is_trading_day", classmethod(lambda cls, when=None: True))

    early = datetime(2026, 6, 2, 19, 59)
    cutoff = datetime(2026, 6, 2, 20, 5)

    assert runner._maybe_run_after_market_training(early, "AFTER-HOURS") is False
    assert runner._maybe_run_after_market_training(cutoff, "AFTER-HOURS") is True
    assert runner._nightly_calls == 1
