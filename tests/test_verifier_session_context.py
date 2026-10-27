from daytrading.models import BarSource
"""Session guards must not mistake a detector's local window for the day."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from daytrading.classifier.router import RouteResult, StyleConfig
from daytrading.models import Bar, PortfolioState, ScanResult, TradingStyle
from daytrading.pipeline.engine import TradingPipeline
from daytrading.strategy.scalping.momentum_pattern import MomentumPatternVerifier


def recorded(symbol):
    data = json.loads((Path(__file__).parent / "fixtures" /
                       "recovered_sip_session_context.json").read_text())[symbol]
    bars = [Bar(symbol=symbol, ts=datetime.fromisoformat(b["t"].replace("Z", "+00:00")),
                open=b["o"], high=b["h"], low=b["l"], close=b["c"], volume=b["v"], provenance=BarSource.CANONICAL_INTERVAL)
            for b in data["bars"]]
    hit = ScanResult(symbol=symbol, scanner_name="vwap_pullback", ts=bars[-1].ts,
                     score=50, criteria=deepcopy(data["hit"]), bars=bars[-120:])
    verifier = MomentumPatternVerifier(float_checker=Mock(
        get_float=Mock(return_value=data["float_shares"]), get_avg_volume=Mock(return_value=None)))
    return bars, hit, verifier, datetime.fromisoformat(data["decision_at"])


class OneHit:
    name = "vwap_pullback"

    def __init__(self, hit):
        self.hit = hit

    def scan(self, universe):
        return [self.hit]


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("symbol,expected", [
    ("WETO", "late pullback too far from HOD 20.2%"),
    ("SECZ", "late pullback tape too slow 4552 recent volume"),
])
def test_recorded_session_reaches_true_guard_without_rewriting_setup(symbol, expected, routed):
    bars, hit, verifier, now = recorded(symbol)
    scanner = OneHit(hit)
    broker = Mock()
    router = None
    if routed:
        router = Mock()
        router.route.return_value = RouteResult(groups={TradingStyle.SCALPING: {symbol: bars}})
        router.get_config.return_value = StyleConfig(scanners=[scanner],
                                                     verifiers={scanner.name: verifier})
    pipeline = TradingPipeline(scanners=[scanner], verifiers={scanner.name: verifier},
                               broker=broker, portfolio=PortfolioState(cash=50_000), router=router)
    pattern_bars, stop = list(hit.bars), hit.criteria["stop_price"]
    result = pipeline.run_cycle({symbol: bars}, now=now)
    assert pipeline.scan_rejections[symbol].startswith(expected)
    context = hit.criteria["quality_inputs"]
    assert context["session_context"] == "pipeline_universe"
    assert context["session_open"] == bars[0].open
    assert context["session_bar_count"] == len(bars)
    assert context["session_change_pct"] > 5
    assert hit.bars == pattern_bars
    assert hit.criteria["stop_price"] == stop
    assert result.fills == []
    assert result.deferred_signals == []
    broker.submit.assert_not_called()


def test_session_context_excludes_future_other_symbols_and_previous_new_york_day():
    bars, hit, verifier, now = recorded("WETO")
    previous_ny_day = replace(bars[0], ts=datetime(2026, 10, 15, 2, tzinfo=timezone.utc),
                              open=0.01, high=100)
    other_symbol = replace(bars[0], symbol="OTHER", open=0.01, high=100)
    future = replace(bars[-1], ts=bars[-1].ts + timedelta(minutes=1), high=100)
    context = [previous_ny_day, other_symbol, *bars, future]
    assert verifier.verify(hit, PortfolioState(cash=50_000), now=now, session_bars=context) is None
    assert verifier._last_reject.startswith("late pullback too far from HOD 20.2%")
    assert hit.criteria["quality_inputs"]["session_open"] == .79
    assert hit.criteria["quality_inputs"]["session_bar_count"] == len(bars)


@pytest.mark.parametrize("context_kind", ["empty", "missing_trigger", "revised_trigger"])
def test_inconsistent_context_fails_closed(context_kind):
    bars, hit, verifier, now = recorded("WETO")
    context = [] if context_kind == "empty" else list(bars[:-1])
    if context_kind == "revised_trigger":
        context.append(replace(bars[-1], close=1.18))
    assert verifier.verify(hit, PortfolioState(cash=50_000), now=now, session_bars=context) is None
    assert verifier._last_reject == "session context does not match setup candle"


def test_routed_log_counts_each_detector_instead_of_cumulative_hits(caplog):
    bars, hit, _, now = recorded("WETO")
    scanner = OneHit(hit)
    empty = Mock(name="empty_scanner")
    empty.name = "hod_reclaim"
    empty.scan.return_value = []
    # No tick/quote inputs: only this detector's own bar results count.
    router = Mock()
    router.route.return_value = RouteResult(groups={TradingStyle.SCALPING: {"WETO": bars}})
    router.get_config.return_value = StyleConfig(scanners=[scanner, empty], verifiers={})
    pipeline = TradingPipeline(scanners=[], verifiers={}, broker=Mock(),
                               portfolio=PortfolioState(cash=50_000), router=router)
    with caplog.at_level(logging.DEBUG, logger="daytrading.pipeline.engine"):
        pipeline.run_cycle({"WETO": bars}, now=now)
    assert "Scanner vwap_pullback (scalping) found 1 hits" in caplog.text
    assert "Scanner hod_reclaim (scalping) found 0 hits" in caplog.text
    assert "Scanner hod_reclaim (scalping) found 1 hits" not in caplog.text


def test_verifier_legacy_call_remains_supported_without_session_keyword():
    bars, hit, _, now = recorded("WETO")
    class Legacy:
        def verify(self, scan, portfolio):
            assert scan is hit
            return None
    pipeline = TradingPipeline(scanners=[], verifiers={}, broker=Mock(),
                               portfolio=PortfolioState(cash=50_000))
    assert pipeline._verify_hit(Legacy(), hit, now=now, session_bars=bars) is None
