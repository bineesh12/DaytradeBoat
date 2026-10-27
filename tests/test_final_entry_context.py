from daytrading.models import BarSource
"""Final gates must evaluate the same available trend context as verifiers."""
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from daytrading.data.bar_aggregator import BarAggregator
from daytrading.execution.broker import PaperBroker
from daytrading.models import Bar, PortfolioState, Quote, ScanResult, SignalAction, TradeSignal
from daytrading.pipeline.engine import PipelineResult, TradingPipeline
from daytrading.strategy import entry_guard


def _time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.fixture
def sprc(monkeypatch):
    source = json.loads((Path(__file__).parent / "fixtures/sprc_entry_context.json").read_text())
    bars = [Bar("SPRC", _time(b["t"]), b["o"], b["h"], b["l"], b["c"], b["v"], provenance=BarSource.CANONICAL_INTERVAL)
            for b in source["bars"]]
    quotes = [Quote("SPRC", _time(q["t"]), q["bp"], q["ap"], q["bs"], q["as"])
              for q in source["quotes"]]
    # Replay only the deterministic rule gate; do not write research/ML rows.
    monkeypatch.setattr(entry_guard, "ENTRY_ML_ENABLED", False)
    monkeypatch.setattr(entry_guard, "refresh_entry_model", lambda: None)
    monkeypatch.setattr(entry_guard, "record_rule_rejection", lambda **kwargs: None)
    monkeypatch.setattr(entry_guard, "_log_candidate", lambda *args, **kwargs: None)
    pipeline = TradingPipeline(scanners=[], verifiers={}, broker=PaperBroker(),
                               portfolio=PortfolioState(cash=25_000))
    pipeline._max_entry_risk_pct = 0.06
    pipeline._entry_policy.entry_patterns = {"vwap_pullback"}
    signal = TradeSignal("SPRC", SignalAction.ENTER_LONG, 200, source["entry"],
                         stop_loss=source["stop"],
                         scan_result=ScanResult("SPRC", "vwap_pullback", bars[-1].ts, 90,
                                                {"pattern": "vwap_pullback", "setup_tier": "A+ setup"}, bars))
    return pipeline, signal, bars, quotes, _time(source["now"])


@pytest.mark.parametrize("action", [SignalAction.ENTER_LONG, SignalAction.REENTER_LONG, SignalAction.SCALE_UP_LONG])
def test_available_5minute_context_clears_spurious_sprc_79_score(sprc, action):
    pipeline, signal, bars, quotes, now = sprc
    signal = replace(signal, action=action)
    # Prove this fixture reproduces the real rejection without context.
    old = pipeline._entry_policy.evaluate(signal, bars=bars, quotes=quotes,
                                          stage="final_entry_guard", min_day_change_pct=0, now=now)
    assert "79/100" in old.reason and "no5m=5" in old.reason
    pipeline._bar_aggregator = BarAggregator()
    pipeline._bar_aggregator.build_5m_bars("SPRC", bars)
    result = PipelineResult()
    rejection = pipeline._final_entry_quality_reject(signal, universe={"SPRC": bars},
                                                     quotes={"SPRC": quotes}, now=now, result=result)
    assert rejection is None
    assert result.entry_decisions[-1]["passed"]


def test_missing_5minute_data_still_rejects_without_threshold_bypass(sprc):
    pipeline, signal, bars, quotes, now = sprc
    pipeline._bar_aggregator = BarAggregator()  # No context for this symbol yet.
    reason = pipeline._final_entry_quality_reject(signal, universe={"SPRC": bars},
                                                 quotes={"SPRC": quotes}, now=now)
    assert "79/100" in reason and "no5m=5" in reason


def test_trend_context_does_not_bypass_stop_distance_cap(sprc):
    pipeline, signal, bars, quotes, now = sprc
    pipeline._bar_aggregator = BarAggregator()
    pipeline._bar_aggregator.build_5m_bars("SPRC", bars)
    signal = replace(signal, entry_price=5.08)
    reason = pipeline._final_entry_quality_reject(signal, universe={"SPRC": bars},
                                                 quotes={"SPRC": quotes}, now=now)
    assert reason == "stop too wide: 7.9% risk > 6.0% cap"
