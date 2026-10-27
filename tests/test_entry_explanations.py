from daytrading.models import BarSource
"""Boundary-risk regressions and observable execution-timer cancellations."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from daytrading.dashboard.hub import DashboardHub
from daytrading.execution.broker import PaperBroker
from daytrading.models import Bar, PortfolioState, ScanResult, SignalAction, TradeSignal
from daytrading.pipeline.engine import TradingPipeline
from daytrading.reporting import reporting_failures
from daytrading.runner import AlpacaRunner
from daytrading.strategy.entry_policy import EntryPolicy
from daytrading.strategy.execution_timer import ExecutionTimer


@pytest.mark.parametrize("action", [SignalAction.ENTER_LONG, SignalAction.REENTER_LONG, SignalAction.SCALE_UP_LONG])
@pytest.mark.parametrize("price", [2.0, 2.0189, 5.82, 6.465, 10.0, 20.0])
@pytest.mark.parametrize("risk", [0.0599, 0.06, 0.060001])
def test_risk_cap_accepts_equality_but_keeps_real_excess_blocked(action, price, risk):
    pipeline = TradingPipeline(scanners=[], verifiers={}, broker=PaperBroker(),
                               portfolio=PortfolioState(cash=25_000))
    pipeline._max_entry_risk_pct = 0.06
    checked = []
    pipeline._entry_policy = EntryPolicy(guard=lambda *a, **k: checked.append(True) or None)
    signal = TradeSignal(symbol="GENK", action=action, quantity=10,
                         entry_price=price, stop_loss=price * (1 - risk))
    reject = pipeline._final_entry_quality_reject(signal, universe={})
    if risk <= 0.06:
        assert reject is None
        assert checked == [True]
    else:
        assert reject.startswith("stop too wide")
        assert checked == []


def test_equal_risk_still_reaches_other_quality_vetoes():
    pipeline = TradingPipeline(scanners=[], verifiers={}, broker=PaperBroker(),
                               portfolio=PortfolioState(cash=25_000))
    pipeline._max_entry_risk_pct = 0.06
    pipeline._entry_policy = EntryPolicy(guard=lambda *a, **k: "spread too wide")
    signal = TradeSignal(symbol="GENK", action=SignalAction.ENTER_LONG, quantity=10,
                         entry_price=2.0189, stop_loss=2.0189 * 0.94)
    assert pipeline._final_entry_quality_reject(signal, universe={}) == "spread too wide"


def _setup():
    now = datetime.now(timezone.utc)
    return TradeSignal(
        symbol="TEST", action=SignalAction.ENTER_LONG, quantity=10,
        entry_price=3.0, stop_loss=2.88,
        scan_result=ScanResult(symbol="TEST", scanner_name="vwap_pullback", ts=now,
                               score=30, criteria={"pattern": "vwap_pullback", "close": 3.0,
                                                   "stop_price": 2.88, "volume": 100_000}),
    )


def _red_bar(low=2.95):
    return Bar(symbol="TEST", ts=datetime.now(timezone.utc),
               open=3.0, high=3.01, low=low, close=2.96, volume=1000, provenance=BarSource.CANONICAL_INTERVAL)


def test_timer_stop_failure_reaches_dashboard_and_journal_once():
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._entry_policy = EntryPolicy()
    runner._hub = DashboardHub()
    rows = []
    runner._journal = SimpleNamespace(record=lambda kind, payload, **k: rows.append((kind, payload)))
    timer = ExecutionTimer(max_wait_bars=3, on_cancel=runner._record_execution_timer_cancel)
    signal = _setup()
    timer.queue(signal)
    assert timer.on_10s_bar(_red_bar(low=2.85)) is None
    assert not timer.pending_symbols
    timer.cancel("TEST")
    assert len(rows) == 1
    kind, payload = rows[0]
    assert kind == "entry_decision"
    assert payload["stage"] == payload["source"] == "execution_timer"
    assert payload["pattern"] == "vwap_pullback"
    assert not payload["passed"]
    assert "broke stop" in payload["reason"]
    assert payload["metadata"]["status"] == "cancelled"
    assert runner._hub.funnel_snapshot()["decisions"][0]["reason"] == payload["reason"]


def test_timer_bar_limit_reports_missing_confirmation():
    rows = []
    timer = ExecutionTimer(max_wait_bars=1, on_cancel=lambda signal, reason: rows.append(reason))
    timer.queue(_setup())
    assert timer.on_10s_bar(_red_bar()) is None
    assert rows == ["no favorable confirmation after 1 10s candle"]


def test_timer_timeout_reports_missing_confirmation():
    rows = []
    timer = ExecutionTimer(on_cancel=lambda signal, reason: rows.append(reason))
    timer.queue(_setup())
    timer._pending["TEST"].queued_at -= timedelta(seconds=60)
    assert timer.check_timeouts() == []
    assert len(rows) == 1 and "timeout" in rows[0]
    assert not timer.pending_symbols


def test_reporting_failure_cannot_restore_a_cancelled_entry():
    def broken(*args):
        raise RuntimeError("dashboard unavailable")
    timer = ExecutionTimer(on_cancel=broken)
    timer.queue(_setup())
    before = reporting_failures()
    timer.cancel("TEST", reason="runtime cancellation")
    assert not timer.pending_symbols
    assert reporting_failures() == before + 1


def test_timer_release_is_not_reported_as_a_cancellation():
    rows = []
    timer = ExecutionTimer(on_cancel=lambda *args: rows.append(args))
    signal = TradeSignal(symbol="TEST", action=SignalAction.ENTER_LONG, quantity=10, entry_price=3)
    timer.queue(signal)
    green = Bar(symbol="TEST", ts=datetime.now(timezone.utc),
                open=3.0, high=3.1, low=2.99, close=3.09, volume=1000, provenance=BarSource.CANONICAL_INTERVAL)
    assert timer.on_10s_bar(green) is signal
    assert not timer.pending_symbols and not rows
