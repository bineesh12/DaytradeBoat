"""New movers outside the last filtered candidate pool must be discoverable."""
from queue import Queue
from types import SimpleNamespace

import pytest

from daytrading.data.market_data_service import MarketDataService
from daytrading.data.watchlist_scanner import WatchlistScanner
from daytrading.runner import AlpacaRunner, FastScanEvent


def _runner(scanner, phase="PRE-MARKET", capacity=80, event_queue=None):
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._scanner = scanner
    runner._market_phase = lambda: phase
    runner._after_hours_enabled = False
    runner._pool_candidates_ready = True
    runner._hod_bar_pool = []
    runner._bar_buffer = {}
    runner._market_data = MarketDataService()
    runner._hot_watch_reject_reason = lambda *args: None
    runner._fast_scan_known = set()
    runner._fast_scan_process_max = capacity
    runner._event_queue = event_queue if event_queue is not None else Queue()
    runner._hub = SimpleNamespace(add_log=lambda *args, **kwargs: None)
    return runner


def test_sleeping_symbol_can_enter_fast_discovery_without_pool_refresh():
    scanner = WatchlistScanner.__new__(WatchlistScanner)
    scanner._candidate_symbols = ["KNOWN"]
    scanner._load_all_symbols = lambda: ["KNOWN", "SPRC"]
    requested = []
    def snapshots(symbols):
        requested.append(list(symbols))
        return [dict(symbol=s, price=4.35, change_pct=20, abs_change_pct=20, volume=222_484)
                for s in symbols]
    scanner._get_snapshots = snapshots
    runner = _runner(scanner)
    runner._hot_watch_reject_reason = lambda *args: "not a hot-watch candidate"
    runner._run_fast_scan()
    event = runner._event_queue.get_nowait()
    assert isinstance(event, FastScanEvent)
    assert requested == [["KNOWN", "SPRC"]]
    assert "SPRC" in [row["symbol"] for row in event.new_movers]
    assert scanner._candidate_symbols == ["KNOWN"]  # Read-only pool ownership.


@pytest.mark.parametrize("phase", ["PRE-MARKET", "OPEN"])
def test_fast_discovery_requests_broad_readonly_snapshots(phase):
    calls = []
    scanner = SimpleNamespace(scan_candidates=lambda **kwargs: calls.append(kwargs) or [])
    _runner(scanner, phase)._run_fast_scan()
    assert calls == [dict(full_universe=True, readonly=True)]


def test_closed_session_does_not_run_broad_requests():
    scanner = SimpleNamespace(scan_candidates=lambda **kwargs: pytest.fail("closed discovery"))
    _runner(scanner, "CLOSED")._run_fast_scan()


def test_batch_overflow_is_not_marked_seen_before_handoff():
    movers = [dict(symbol=s, price=5, abs_change_pct=3, volume=30_000) for s in ("FIRST", "NEXT")]
    scanner = SimpleNamespace(scan_candidates=lambda **kwargs: movers)
    runner = _runner(scanner, "OPEN", capacity=1)
    runner._hot_watch_reject_reason = lambda *args: "not a hot-watch candidate"
    runner._run_fast_scan()
    assert runner._fast_scan_known == {"FIRST"}
    assert runner._event_queue.get_nowait().new_movers[0]["symbol"] == "FIRST"
    runner._run_fast_scan()
    assert runner._event_queue.get_nowait().new_movers[0]["symbol"] == "NEXT"


def test_full_event_queue_leaves_candidate_retryable():
    movers = [dict(symbol="RETRY", price=5, abs_change_pct=3, volume=30_000)]
    scanner = SimpleNamespace(scan_candidates=lambda **kwargs: movers)
    events = Queue(maxsize=1)
    events.put("busy")
    runner = _runner(scanner, "OPEN", event_queue=events)
    runner._run_fast_scan()
    assert runner._fast_scan_known == set()
    events.get_nowait()
    runner._run_fast_scan()
    assert events.get_nowait().new_movers[0]["symbol"] == "RETRY"


def test_broad_scan_keeps_price_volume_filters():
    scanner = WatchlistScanner.__new__(WatchlistScanner)
    scanner._is_premarket = True
    scanner._min_price, scanner._max_price = 1, 20
    scanner._min_volume = scanner._premarket_min_volume = 10_000
    scanner._min_change = 2
    def snapshot(price, volume):
        return SimpleNamespace(daily_bar=SimpleNamespace(close=4, volume=5_000_000),
                               previous_daily_bar=SimpleNamespace(close=4),
                               latest_trade=SimpleNamespace(price=price),
                               minute_bar=SimpleNamespace(volume=volume))
    assert scanner._extract("SPRC", snapshot(4.35, 27_022)) is not None
    assert scanner._extract("SLEEP", snapshot(4.35, 307)) is None
    assert scanner._extract("SUB1", snapshot(.90, 500_000)) is None
    assert scanner._extract("OVER20", snapshot(21, 500_000)) is None
