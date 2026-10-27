from daytrading.models import BarSource
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from daytrading.data.bar_aggregator import BarAggregator
from daytrading.execution.broker import PaperBroker
from daytrading.models import Bar, PortfolioState, Quote, ScanResult, SignalAction, TradeSignal
from daytrading.opportunities import hit_identity
from daytrading.pipeline.engine import TradingPipeline
from daytrading.strategy.pending_entry import PendingEntryBook
from daytrading.strategy.scalping.momentum_pattern import MomentumPatternVerifier
from daytrading.strategy import entry_guard


def setup():
    now = datetime(2026, 10, 15, 12, 0, 30, tzinfo=timezone.utc)
    bar = Bar("TEST", now-timedelta(seconds=30), 4.9, 5.01, 4.8, 5, 100_000, provenance=BarSource.CANONICAL_INTERVAL)
    hit = ScanResult("TEST", "vwap_pullback", bar.ts, 90,
                     {"pattern": "vwap_pullback", "pullback_low": 4.8, "stop_price": 4.78,
                      "_pending_structure_valid": True}, [bar]*3)
    book = PendingEntryBook()
    book.reject(hit, "spread too wide", now)
    return book, hit, now


def prepare(book, hit, now, **kwargs):
    quote = Quote(hit.symbol, now, 4.99, 5.01, 1000, 1000)
    options = dict(entries_enabled=True, enabled_patterns={"vwap_pullback"})
    options.update(kwargs)
    return book.prepare({hit.symbol: hit.bars}, {hit.symbol: [quote]}, {}, PortfolioState(cash=10_000), now, **options)


def test_only_verified_vwap_structure_with_transient_gate_can_wait():
    book, hit, now = setup()
    book.clear()
    for candidate, reason in [(replace(hit, scanner_name="bull_flag"), "spread too wide"),
                              (replace(hit, criteria={}), "spread too wide"),
                              (hit, "pullback has dump candle"), (hit, "stop too wide")]:
        book.reject(candidate, reason, now)
        assert not book.rows


def test_fresh_data_wakes_but_unchanged_data_does_not_repeat_checks():
    book, hit, now = setup()
    assert not book.due(now)
    book.mark("TEST", 5, now+timedelta(seconds=1))
    assert book.due(now+timedelta(seconds=1))
    retried = prepare(book, hit, now+timedelta(seconds=1))["TEST"]
    assert hit_identity(retried) == hit_identity(hit)
    assert retried.criteria["_pending_recheck"]
    assert not book.due(now+timedelta(seconds=2))
    book.mark("TEST", 5, now+timedelta(seconds=1.1))
    assert not book.due(now+timedelta(seconds=1.1))
    assert book.due(now+timedelta(seconds=2))


@pytest.mark.parametrize("reason", ["spread too wide", "entry score too low", "tape shows selling pressure"])
def test_repeated_failures_never_extend_expiry_or_change_anchor(reason):
    book, hit, now = setup()
    original = book.rows["TEST"]
    book.reject(hit, reason, now+timedelta(seconds=40))
    assert book.rows["TEST"] is original
    assert original.expires == now+timedelta(seconds=60)
    assert prepare(book, hit, now+timedelta(seconds=60)) == {}
    book.reject(hit, reason, now+timedelta(seconds=61))
    assert not book.rows


@pytest.mark.parametrize("price,reason", [(4.79, "support broken"), (5.13, "extended")])
def test_bad_tick_cancels_even_if_price_recovers_before_next_check(price, reason):
    book, hit, now = setup()
    book.mark("TEST", price, now+timedelta(seconds=1), dirty=False)
    book.mark("TEST", 5, now+timedelta(seconds=2))
    assert prepare(book, hit, now+timedelta(seconds=2)) == {}
    assert reason in book.drain_events()[-1][2]


@pytest.mark.parametrize("enabled,patterns", [(False, {"vwap_pullback"}), (True, {"hod_reclaim"})])
def test_pause_closed_window_or_disabled_pattern_cancels(enabled, patterns):
    book, hit, now = setup()
    assert prepare(book, hit, now, entries_enabled=enabled, enabled_patterns=patterns) == {}
    assert not book.rows


def test_stale_quotes_cannot_release_pending_setup():
    book, hit, now = setup()
    book.mark("TEST", 5, now+timedelta(seconds=7))
    old_quote = Quote("TEST", now, 4.99, 5.01, 1000, 1000)
    result = book.prepare({"TEST": hit.bars}, {"TEST": [old_quote]}, {}, PortfolioState(cash=10_000),
                          now+timedelta(seconds=7), entries_enabled=True, enabled_patterns=None)
    assert not result and "TEST" in book.rows
    assert "fresh valid quote" in book.drain_events()[-1][2]


def test_lastmile_spread_rejection_returns_to_same_deadline():
    book, hit, now = setup()
    book.finish("TEST", "confirming", "all gates passed")
    assert not book.rows and "TEST" in book.inflight
    book.reject(hit, "spread 0.8% rejected: quote depth too weak for spread exception", now+timedelta(seconds=20))
    assert not book.inflight and book.rows["TEST"].expires == now+timedelta(seconds=60)
    book.finish("TEST", "submitted", "existing order path")
    book.reject(hit, "spread too wide", now+timedelta(seconds=30))
    assert not book.rows


def test_new_candle_that_broke_support_cannot_retry_on_recovery():
    book, hit, now = setup()
    later_bar = replace(hit.bars[-1], ts=now+timedelta(seconds=30), low=4.79, close=5)
    quote = Quote("TEST", now+timedelta(seconds=31),4.99,5.01,1000,1000)
    book.mark("TEST",5,quote.ts)
    assert not book.prepare({"TEST":hit.bars+[later_bar]},{"TEST":[quote]}, {}, PortfolioState(cash=10_000),
                            quote.ts,entries_enabled=True,enabled_patterns=None)
    assert "support broken" in book.drain_events()[-1][2]


def test_confirming_setup_keeps_original_expiry_and_expires_on_lastmile_failure():
    book, hit, now=setup()
    book.finish("TEST","confirming","gates passed")
    book.reject(hit,"spread too wide",now+timedelta(seconds=61))
    assert not book.rows and not book.inflight
    assert "expired" in book.drain_events()[-1][2]


def test_full_pipeline_retries_sprc_without_new_detector_hit(monkeypatch):
    source = json.loads((Path(__file__).parent/"fixtures/sprc_entry_context.json").read_text())
    def ts(value): return datetime.fromisoformat(value.replace("Z", "+00:00"))
    now = ts(source["now"])
    bars = [Bar("SPRC", ts(b["t"]), b["o"], b["h"], b["l"], b["c"], b["v"], provenance=BarSource.CANONICAL_INTERVAL) for b in source["bars"]]
    hit = ScanResult("SPRC", "vwap_pullback", bars[-1].ts, 90,
                     {"pattern": "vwap_pullback", "direction": "up", "pullback_low": 4.7001,
                      "stop_price": 4.6801, "vwap": 4.7461}, bars)
    hits = [hit]
    scanner = SimpleNamespace(name="vwap_pullback", scan=lambda universe: list(hits))
    verifier = MomentumPatternVerifier(reward_risk_ratio=2, max_dollar_risk=50, max_price=20)
    aggregator = BarAggregator()
    aggregator.build_5m_bars("SPRC", bars)
    verifier._bar_aggregator = aggregator
    wide = Quote("SPRC", now, 4.92, 4.96, 100, 100)
    quotes = {"SPRC": [wide]*3}
    verifier._quote_buffer = quotes
    pipeline = TradingPipeline([scanner], {"vwap_pullback": verifier}, PaperBroker(), PortfolioState(cash=25_000))
    pipeline._bar_aggregator = aggregator
    pipeline._max_entry_risk_pct = .06
    pipeline._entry_policy.entry_patterns = {"vwap_pullback"}
    pipeline._execution_timer = object()  # Existing deferred path; never submit directly.
    monkeypatch.setattr(entry_guard, "ENTRY_ML_ENABLED", False)
    monkeypatch.setattr(entry_guard, "refresh_entry_model", lambda: None)
    monkeypatch.setattr(entry_guard, "record_rule_rejection", lambda **kwargs: None)
    monkeypatch.setattr(entry_guard, "_log_candidate", lambda *args, **kwargs: None)
    first = pipeline.run_cycle({"SPRC": bars}, now=now, quotes=quotes)
    assert "SPRC" in pipeline.pending_entries.rows
    assert not first.deferred_signals and not first.fills
    hits.clear()  # Detector no longer emits a pattern.
    later = now+timedelta(seconds=2)
    quotes["SPRC"] = [Quote("SPRC", later, 4.94, 4.95, 1000, 1000)]*3
    pipeline.pending_entries.mark("SPRC", 4.94, later)
    second = pipeline.run_cycle({"SPRC": bars}, now=later, quotes=quotes)
    assert len(second.deferred_signals) == 1 and not second.fills
    assert hit_identity(second.deferred_signals[0].scan_result) == hit_identity(hit)
    third = pipeline.run_cycle({"SPRC": bars}, now=later+timedelta(seconds=1), quotes=quotes)
    assert not third.deferred_signals and not third.fills
    assert "SPRC" in pipeline.pending_entries.inflight
