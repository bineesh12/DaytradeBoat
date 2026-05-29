"""Integration-style tests for runner wiring (mocked broker, no live Alpaca)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import pytest

from daytrading.models import Bar, SignalAction, Timeframe, TradeSignal
from daytrading.pipeline.engine import PipelineResult
from daytrading.strategy.execution_timer import ExecutionTimer


def _signal(symbol: str = "TST") -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=5.0,
        reason="bull_flag",
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
