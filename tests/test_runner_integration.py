"""Integration-style tests for runner wiring (mocked broker, no live Alpaca)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import queue
from time import perf_counter
from threading import Lock
from types import SimpleNamespace
import pytest

from daytrading.models import Bar, OrderStatus, PortfolioState, ScanResult, SignalAction, Timeframe, TradeSignal
from daytrading.pipeline.engine import PipelineResult
from daytrading.runner import AlpacaRunner, BarsLoadedEvent, FastScanEvent
from daytrading.strategy.execution_timer import ExecutionTimer


def _signal(symbol: str = "TST") -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=5.0,
        reason="bull_flag",
    )


def _momentum_signal(
    reason: str = "momentum burst",
    scanner_name: str = "momentum_burst",
    pattern: str = "momentum_burst",
) -> TradeSignal:
    return TradeSignal(
        symbol="ANY",
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=4.94,
        reason=reason,
        scan_result=ScanResult(
            symbol="ANY",
            scanner_name=scanner_name,
            ts=datetime.now(timezone.utc),
            score=97.0,
            criteria={
                "pattern": pattern,
                "close": 4.94,
                "volume": 4_422_000,
            },
        ),
    )


class TestExecutionTimerQueue:
    def test_deferred_signal_queued_not_blocking(self) -> None:
        timer = ExecutionTimer(max_wait_bars=2, enabled=True)
        assert timer.queue(_signal()) is True
        assert "TST" in timer.pending_symbols


class TestPipelineDeferredSignals:
    def test_pipeline_result_has_deferred_list(self) -> None:
        r = PipelineResult()
        r.deferred_signals.append(_signal())
        assert len(r.deferred_signals) == 1


class TestJournalStrategyOnFill:
    def test_trade_fill_payload_includes_strategy(self, tmp_path) -> None:
        from daytrading.journal.store import TradingJournal

        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        ts = datetime(2026, 5, 15, 14, 30, 0, tzinfo=timezone.utc)
        journal.record("trade_fill", {
            "symbol": "WIN",
            "side": "buy",
            "quantity": 100,
            "price": 5.0,
            "trade_type": "entry",
            "strategy": "bull_flag",
        }, ts=ts)

        import sqlite3
        conn = sqlite3.connect(journal.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT strategy FROM trades WHERE symbol='WIN'"
        ).fetchone()
        conn.close()
        assert row["strategy"] == "bull_flag"


class TestTimedSignalQueue:
    def test_strong_green_bar_queues_for_main_loop(self) -> None:
        """Pattern used in runner: on_10s_bar → append to deque, not broker.submit."""
        timer = ExecutionTimer(max_wait_bars=5, enabled=True)
        queue: deque = deque()
        timer.queue(_signal("BBB"))
        bar = Bar(
            symbol="BBB",
            ts=datetime.now(timezone.utc),
            open=5.0, high=5.12, low=4.98, close=5.10,
            volume=1000, timeframe=Timeframe.SEC_10,
        )
        sig = timer.on_10s_bar(bar)
        if sig:
            queue.append(sig)
        assert len(queue) == 1
        assert queue[0].symbol == "BBB"

    def test_timed_entry_cannot_submit_when_final_guard_rejects(self) -> None:
        """10s release is not enough; timed entry still needs final guard/ML pass."""
        runner = AlpacaRunner.__new__(AlpacaRunner)
        runner._new_entries_blocked = lambda *a, **k: False
        runner._pipeline = SimpleNamespace(
            _exit_cooldowns={},
            _cooldown_seconds=60,
            _daily_losers=set(),
            _symbol_entry_counts={},
            _max_entries_per_symbol=3,
            portfolio=PortfolioState(cash=50_000, positions={}),
        )
        runner._bar_buffer = {
            "TST": deque([
                Bar("TST", datetime.now(timezone.utc), 5.0, 5.05, 4.95, 5.0, 200_000),
                Bar("TST", datetime.now(timezone.utc), 5.0, 5.05, 4.95, 5.02, 200_000),
                Bar("TST", datetime.now(timezone.utc), 5.02, 5.05, 4.99, 5.01, 200_000),
            ])
        }
        runner._timed_entry_chase_reject = lambda *a, **k: None
        runner._shared_entry_quality_reject = lambda *a, **k: "entry score too low (65/100, need 80+)"
        runner._hub = SimpleNamespace(add_log=lambda *a, **k: None)

        class _Broker:
            submits = 0

            def submit(self, *args, **kwargs):
                self.submits += 1
                return None, OrderStatus.REJECTED

        runner._broker = _Broker()

        runner._execute_timed_signal(_signal("TST"))

        assert runner._broker.submits == 0

    def test_hot_hod_retry_cannot_submit_when_final_guard_rejects(self) -> None:
        """The controlled hot retry also goes through final guard/ML."""
        runner = AlpacaRunner.__new__(AlpacaRunner)
        runner._live_prices = lambda symbols: {symbols[0]: 5.05}
        runner._latest_price = lambda symbol: 5.05
        runner._bar_buffer = {
            "TST": deque([
                Bar("TST", datetime.now(timezone.utc), 5.0, 5.1, 4.95, 5.05, 250_000),
            ])
        }
        runner._shared_entry_quality_reject = lambda *a, **k: "ML model low confidence (22%, need 30%)"

        class _Broker:
            submits = 0

            def submit(self, *args, **kwargs):
                self.submits += 1
                return None, OrderStatus.REJECTED

        runner._broker = _Broker()
        sig = _signal("TST")

        fill, status = runner._retry_hot_hod_timed_entry(
            sig,
            OrderStatus.REJECTED,
            Bar("TST", datetime.now(timezone.utc), 5.0, 5.1, 4.95, 5.02, 250_000),
        )

        assert fill is None
        assert status is OrderStatus.REJECTED
        assert runner._broker.submits == 0


class TestRunnerTickTrail:
    def test_ordinary_scalp_uses_one_percent_tick_trail(self) -> None:
        pos = SimpleNamespace(highest_price=5.00, runner_confirmed=False)

        assert AlpacaRunner._tick_trail_stop_for(pos) == pytest.approx(4.95)

    def test_confirmed_runner_uses_wider_tick_trail(self) -> None:
        pos = SimpleNamespace(
            highest_price=5.00,
            runner_confirmed=True,
            runner_trail_pct=0.03,
        )

        assert AlpacaRunner._tick_trail_stop_for(pos) == pytest.approx(4.85)


class TestFastScanDeferral:
    def _runner(self) -> AlpacaRunner:
        runner = AlpacaRunner.__new__(AlpacaRunner)
        runner._event_queue = queue.Queue()
        runner._exec_timer = ExecutionTimer(max_wait_bars=1, enabled=True)
        runner._timed_signal_queue = deque()
        runner._deferred_fast_scan_movers = []
        runner._fast_scan_process_max = 80
        runner._candidate_hydrate_pending = set()
        runner._candidate_hydrate_seq = 0
        runner._candidate_hydrate_lock = Lock()
        runner._candidate_hydrate_queue = queue.PriorityQueue()
        runner._candidate_hydrate_batch_max = 10
        runner._hod_active = {}
        runner._watchlist_set = set()
        return runner

    def test_fast_scan_hydration_defers_while_timed_entry_pending(self) -> None:
        runner = self._runner()
        runner._exec_timer.queue(_signal("FOXX"))
        queued = []
        runner._enqueue_candidate_hydration = lambda movers, source="": queued.extend(movers) or len(movers)
        runner._event_queue.put_nowait(FastScanEvent([{"symbol": "XOS"}]))

        got_bars = runner._drain_events()

        assert got_bars is False
        assert queued == []
        assert runner._deferred_fast_scan_movers == [{"symbol": "XOS"}]

    def test_deferred_fast_scan_queues_after_timed_entry_clears(self) -> None:
        runner = self._runner()
        runner._deferred_fast_scan_movers = [{"symbol": "XOS"}]
        queued = []
        runner._enqueue_candidate_hydration = lambda movers, source="": queued.extend(movers) or len(movers)
        runner._event_queue.put_nowait(FastScanEvent([
            {"symbol": "XOS"},
            {"symbol": "FOXX"},
        ]))

        got_bars = runner._drain_events()

        assert got_bars is False
        assert [m["symbol"] for m in queued] == ["XOS", "FOXX"]
        assert runner._deferred_fast_scan_movers == []

    def test_loaded_candidate_bars_signal_new_data(self) -> None:
        runner = self._runner()
        runner._max_bars_per_symbol = 100
        runner._bar_buffer = {}
        runner._prior_day_stats = {}
        runner._now_et = lambda: datetime(2026, 6, 4, tzinfo=timezone.utc)
        runner._bar_is_today = lambda bar, today_et: True
        runner._seed_hod_session = lambda symbol: None
        bar = Bar(
            symbol="XOS",
            ts=datetime(2026, 6, 4, 10, 30, tzinfo=timezone.utc),
            open=5.0,
            high=5.2,
            low=4.9,
            close=5.1,
            volume=1000,
        )
        runner._event_queue.put_nowait(BarsLoadedEvent({"XOS": [bar]}, {}))

        got_bars = runner._drain_events()

        assert got_bars is True
        assert list(runner._bar_buffer["XOS"])[0] == bar

    def test_candidate_hydration_prioritizes_hod_alert_symbols(self) -> None:
        runner = self._runner()
        runner._watchlist_set = {"FOXX"}
        runner._enqueue_candidate_hydration([
            {"symbol": "SLOW", "abs_change_pct": 6.0, "volume": 250_000},
            {"symbol": "FOXX", "abs_change_pct": 8.0, "volume": 250_000},
            {"symbol": "XOS", "abs_change_pct": 30.0, "volume": 1_000_000},
        ])

        batch = runner._pull_candidate_hydration_batch()

        assert [m["symbol"] for m in batch] == ["FOXX", "XOS", "SLOW"]
        assert runner._candidate_hydrate_pending == set()

    def test_fast_scan_drain_does_not_run_heavy_hydration_path(self) -> None:
        runner = self._runner()
        queued = []
        runner._enqueue_candidate_hydration = (
            lambda movers, source="": queued.extend(movers) or len(movers)
        )
        runner._handle_fast_scan_movers = lambda movers, push_event=False: (_ for _ in ()).throw(
            AssertionError("main loop must not hydrate fast-scan movers")
        )
        for idx in range(100):
            runner._event_queue.put_nowait(FastScanEvent([{
                "symbol": f"SYM{idx}",
                "abs_change_pct": 20.0,
                "volume": 1_000_000,
            }]))

        started = perf_counter()
        got_bars = runner._drain_events()
        elapsed = perf_counter() - started

        assert got_bars is False
        assert len(queued) == 100
        assert elapsed < 0.1

    def test_fast_scan_processes_hot_watch_candidate_already_in_hod_pool(self) -> None:
        runner = self._runner()
        runner._scanner = SimpleNamespace(
            _is_premarket=False,
            scan_candidates=lambda readonly=True: [{
                "symbol": "FOXX",
                "price": 4.66,
                "abs_change_pct": 62.94,
                "change_pct": 62.94,
                "volume": 76_774,
                "score": 0.88,
            }],
        )
        runner._market_phase = lambda: "PRE-MARKET"
        runner._after_hours_enabled = False
        runner._pool_candidates_ready = True
        runner._hod_bar_pool = {"FOXX"}
        runner._bar_buffer = {
            "FOXX": deque([
                Bar(
                    symbol="FOXX",
                    ts=datetime.now(timezone.utc),
                    open=4.5,
                    high=4.7,
                    low=4.4,
                    close=4.66,
                    volume=50_000,
                )
            ])
        }
        runner._hot_watch = {}
        runner._hot_watch_enabled = True
        runner._watchlist_pinned = {"SPY"}
        runner._hod_sub2_min_price = 2.0
        runner._hod_max_price = 20.0
        runner._hot_watch_min_change_pct = 5.0
        runner._hot_watch_min_day_volume = 200_000
        runner._hot_watch_sub5_min_day_volume = 500_000
        runner._hot_watch_min_score = 0.30
        runner._hod_max_float = 20_000_000
        runner._fast_scan_known = {"FOXX"}
        runner._hub = SimpleNamespace(add_log=lambda *args, **kwargs: None)

        runner._run_fast_scan()

        evt = runner._event_queue.get_nowait()
        assert isinstance(evt, FastScanEvent)
        assert [m["symbol"] for m in evt.new_movers] == ["FOXX"]


class TestPriorityRefresh:
    def test_stale_reject_symbol_refreshes_before_regular_stale_names(self, monkeypatch) -> None:
        runner = AlpacaRunner.__new__(AlpacaRunner)
        runner._watchlist_pinned = {"SPY"}
        runner._watchlist_set = {"SPY", "FAST", "SLOW"}
        runner._hot_watch = {"FAST": {}, "SLOW": {}}
        runner._priority_bar_refresh = set()
        runner._last_watchlist_bar_refresh = 0.0
        runner._watchlist_bar_refresh_sec = 30.0
        runner._max_bars_per_symbol = 100
        runner._bar_buffer = {
            "FAST": deque([
                Bar(
                    symbol="FAST",
                    ts=datetime(2026, 6, 3, 11, 0, tzinfo=timezone.utc),
                    open=5.0,
                    high=5.0,
                    low=5.0,
                    close=5.0,
                    volume=100,
                )
            ]),
            "SLOW": deque([
                Bar(
                    symbol="SLOW",
                    ts=datetime(2026, 6, 3, 11, 0, tzinfo=timezone.utc),
                    open=5.0,
                    high=5.0,
                    low=5.0,
                    close=5.0,
                    volume=100,
                )
            ]),
        }
        requested_batches = []

        class _Hist:
            def get_bars(self, batch, limit=30):
                requested_batches.append(list(batch))
                return {
                    sym: [
                        Bar(
                            symbol=sym,
                            ts=datetime(2026, 6, 3, 11, 10, tzinfo=timezone.utc),
                            open=6.0,
                            high=6.0,
                            low=6.0,
                            close=6.0,
                            volume=100,
                        )
                    ]
                    for sym in batch
                }

        runner._hist = _Hist()
        monkeypatch.setattr(
            AlpacaRunner,
            "_now_et",
            classmethod(lambda cls: datetime(2026, 6, 3, 7, 10, tzinfo=timezone.utc)),
        )
        monkeypatch.setattr(
            "daytrading.runner.datetime",
            SimpleNamespace(
                now=lambda tz=None: datetime(2026, 6, 3, 11, 10, tzinfo=timezone.utc),
            ),
        )

        runner._request_priority_bar_refresh("FAST")
        runner._refresh_watchlist_bars()

        assert requested_batches
        assert requested_batches[0][0] == "FAST"


class TestTimedEntryChaseGuard:
    def _runner_with_red_10s(self) -> AlpacaRunner:
        runner = AlpacaRunner.__new__(AlpacaRunner)
        runner._live_prices = lambda symbols: {"ANY": 4.95}
        runner._latest_price = lambda symbol: 4.95
        runner._quote_buffer = {"ANY": deque()}
        red_bar = Bar(
            symbol="ANY",
            ts=datetime.now(timezone.utc),
            open=4.96,
            high=5.00,
            low=4.90,
            close=4.94,
            volume=10_000,
            timeframe=Timeframe.SEC_10,
        )
        runner._bar_aggregator = SimpleNamespace(
            get_latest_10s=lambda symbol, count=1: [red_bar],
        )
        return runner

    def test_normal_hot_signal_rejects_latest_red_10s_candle(self) -> None:
        runner = self._runner_with_red_10s()

        reject = runner._timed_entry_chase_reject(
            _momentum_signal(),
            Bar(
                symbol="ANY",
                ts=datetime.now(timezone.utc),
                open=4.94,
                high=4.96,
                low=4.90,
                close=4.94,
                volume=1000,
            ),
        )

        assert reject == "latest 10s candle turned red during entry wait"

    def test_continuation_scout_skips_red_10s_reject_but_keeps_other_guards(self) -> None:
        runner = self._runner_with_red_10s()

        reject = runner._timed_entry_chase_reject(
            _momentum_signal(reason="momentum burst | continuation_scout"),
            Bar(
                symbol="ANY",
                ts=datetime.now(timezone.utc),
                open=4.94,
                high=4.96,
                low=4.90,
                close=4.94,
                volume=1000,
            ),
        )

        assert reject is None

    def test_pullback_scout_skips_red_10s_reject_but_keeps_other_guards(self) -> None:
        runner = self._runner_with_red_10s()
        signal = _momentum_signal(
            reason="pullback base | pullback_scout",
            scanner_name="pullback_base",
            pattern="pullback_base",
        )

        reject = runner._timed_entry_chase_reject(
            signal,
            Bar(
                symbol="ANY",
                ts=datetime.now(timezone.utc),
                open=4.94,
                high=4.96,
                low=4.90,
                close=4.94,
                volume=1000,
            ),
        )

        assert reject is None


def test_watch_only_decisions_excluded_from_funnel():
    """Shadow-scanner 'collecting data' rows must not enter the entry_decision funnel."""
    is_wo = AlpacaRunner._is_watch_only_decision
    # Watch-only monitoring rows → filtered
    assert is_wo({"reason": "watch only: momentum_burst collecting data, not live A+ setup"})
    assert is_wo({"reason": "", "blocked_layer": "scanner", "setup_tier": "watch only"})
    assert is_wo({"reason": "level breakout has not reclaimed", "blocked_layer": "verifier",
                  "setup_tier": "watch only"})
    # Real entry decisions → kept
    assert not is_wo({"reason": "entry score too low (75/100, need 80+)", "blocked_layer": "entry_guard"})
    assert not is_wo({"reason": "spread too wide (1.00c = 0.62% of $1.60)", "blocked_layer": "verifier"})
    assert not is_wo({"passed": True, "reason": "", "blocked_layer": "", "setup_tier": "A+ setup"})
