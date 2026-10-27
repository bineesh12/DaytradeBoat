from daytrading.models import BarSource
"""HOD memory is not an entry permission: every retry retains shared gates."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from daytrading.data.bar_aggregator import BarAggregator
from daytrading.execution.broker import PaperBroker
from daytrading.models import Bar, PortfolioState, Quote, ScanResult, Side, SignalAction, Tick, TradeSignal
from daytrading.opportunities import hit_identity
from daytrading.pipeline.engine import TradingPipeline
from daytrading.strategy import entry_guard
from daytrading.strategy.execution_timer import ExecutionTimer
from daytrading.strategy.pending_entry import PendingEntryBook
from daytrading.strategy.scalping.momentum_pattern import MomentumPatternVerifier

REASON = "no favorable confirmation after 3 10s candles"


def setup():
    now = datetime(2026, 10, 15, 12, 52, 21, tzinfo=timezone.utc)
    bar = Bar("WETO", now.replace(minute=51, second=0), .9875, 1.09, .986, 1.082, 654972, provenance=BarSource.CANONICAL_INTERVAL)
    hit = ScanResult("WETO", "hod_reclaim", bar.ts, 65,
                     dict(pattern="hod_reclaim", direction="up", hod=1.05, pullback_low=.8835,
                          stop_price=.8635, close=1.082, volume=654972, setup_tier="A+ setup"), [bar]*10)
    signal = TradeSignal("WETO", SignalAction.ENTER_LONG, 500, 1.082, 1.01708, 1.21184, scan_result=hit)
    book = PendingEntryBook()
    book.timer_cancel(signal, REASON, now)
    return book, signal, now


def prepare(book, signal, now, bid=1.08, ask=1.09, **kwargs):
    options = dict(entries_enabled=True, enabled_patterns={"hod_reclaim"})
    options.update(kwargs)
    return book.prepare({signal.symbol: signal.scan_result.bars},
                        {signal.symbol: [Quote(signal.symbol, now, bid, ask, 1000, 1000)]}, {},
                        PortfolioState(cash=25000), now, **options)


def test_hod_requires_pullback_then_recovery_not_just_another_high():
    book, signal, now = setup()
    book.mark("WETO", 1.09, now+timedelta(seconds=1), quote=True)
    assert not prepare(book, signal, now+timedelta(seconds=1), 1.09, 1.10)
    book.mark("WETO", 1.05, now+timedelta(seconds=6), quote=True)
    assert not prepare(book, signal, now+timedelta(seconds=6), 1.05, 1.06)
    book.mark("WETO", 1.08, now+timedelta(seconds=33), quote=True)
    retried = prepare(book, signal, now+timedelta(seconds=33))["WETO"]
    assert hit_identity(retried) == hit_identity(signal.scan_result)
    assert retried.criteria["stop_price"] == signal.stop_loss
    assert retried.criteria["setup_anchor"] == signal.entry_price
    assert retried.criteria["_pending_hod_retry"]


@pytest.mark.parametrize("reason", ["10s low broke stop", "setup support broken", "HOD reclaim 10s confirmation red/flat"])
def test_hard_failures_do_not_create_retry(reason):
    book, signal, now = setup()
    book.clear()
    book.timer_cancel(signal, reason, now)
    assert not book.rows


def test_one_retry_only_and_original_deadline_never_extends():
    book, signal, now = setup()
    retained = book.rows["WETO"].hit
    book.timer_cancel(signal, REASON, now+timedelta(seconds=20))
    assert book.rows["WETO"].expires == now+timedelta(seconds=60)
    book.finish("WETO", "confirming", "all gates passed")
    book.timer_cancel(replace(signal, scan_result=retained), REASON, now+timedelta(seconds=40))
    assert not book.rows and not book.inflight
    book.timer_cancel(signal, REASON, now+timedelta(seconds=41))
    assert not book.rows


def test_retry_cannot_raise_a_previously_pinned_anti_chase_anchor():
    book, signal, now = setup()
    book.clear()
    signal.scan_result.criteria["setup_anchor"] = 1.07
    book.timer_cancel(signal, REASON, now)
    assert book.rows["WETO"].entry == 1.07
    assert book.rows["WETO"].hit.criteria["setup_anchor"] == 1.07


@pytest.mark.parametrize("price,expected", [(1.01, "support broken"), (1.12, "extended")])
def test_quote_break_or_extension_is_terminal_even_after_recovery(price, expected):
    book, signal, now = setup()
    book.mark("WETO", price, now+timedelta(seconds=1), quote=True)
    book.mark("WETO", 1.08, now+timedelta(seconds=2), quote=True)
    assert not prepare(book, signal, now+timedelta(seconds=2))
    assert expected in book.drain_events()[-1][2]


def test_raw_off_quote_print_does_not_redefine_retry_structure():
    book, signal, now = setup()
    book.mark("WETO", .98, now+timedelta(seconds=1), dirty=False)
    assert not book.rows["WETO"].invalid
    assert not book.rows["WETO"].saw_pullback
    # This affects setup memory only; raw ticks still go through unchanged
    # tape pressure, micro-candle and final execution checks in the runner.
    book.mark("WETO", 1.05, now+timedelta(seconds=2), quote=True)
    assert book.rows["WETO"].saw_pullback


@pytest.mark.parametrize("enabled,patterns", [(False, {"hod_reclaim"}), (True, {"vwap_pullback"})])
def test_pause_and_disabled_hod_remove_retry(enabled, patterns):
    book, signal, now = setup()
    assert not prepare(book, signal, now, entries_enabled=enabled, enabled_patterns=patterns)
    assert not book.rows


def test_retry_expires_and_cannot_rearm_old_opportunity():
    book, signal, now = setup()
    book.mark("WETO", 1.05, now+timedelta(seconds=1), quote=True)
    assert not prepare(book, signal, now+timedelta(seconds=60))
    book.timer_cancel(signal, REASON, now+timedelta(seconds=61))
    assert not book.rows


@pytest.mark.parametrize("ask,qty,expected", [(1.08,500,None), (1.09,500,"stop risk"),
                                             (1.08,1000,"dollar risk"), (1.12,500,"extended")])
def test_delayed_entry_checks_original_risk_limits_at_executable_ask(ask, qty, expected):
    book, signal, now = setup()
    retained = book.rows["WETO"].hit
    book.finish("WETO","confirming","all gates passed")
    signal = replace(signal,quantity=qty,scan_result=retained)
    quote = Quote("WETO",now,ask-.001,ask,1000,1000)
    reason = book.execution_reject(signal,quote,now,max_risk_pct=.06,max_dollar_risk=50)
    if expected:
        assert expected in reason
    else:
        assert reason is None


def test_lastmile_cannot_widen_protective_stop_or_execute_an_expired_retry():
    book, signal, now = setup()
    retained = book.rows["WETO"].hit
    book.finish("WETO","confirming","all gates passed")
    signal = replace(signal,scan_result=retained,stop_loss=1.0)
    quote = Quote("WETO",now,1.079,1.08,1000,1000)
    assert "stop widened" in book.execution_reject(signal,quote,now,max_risk_pct=.06,max_dollar_risk=50)
    assert "expired" in book.execution_reject(signal,replace(quote,ts=now+timedelta(seconds=60)),
        now+timedelta(seconds=60),max_risk_pct=.06,max_dollar_risk=50)


def test_retry_ignores_pre_queue_candle_and_does_not_extend_timer_timeout():
    book, signal, now = setup()
    observations = []
    timer = ExecutionTimer(max_wait_bars=3, on_observation=lambda signal, row: observations.append(row))
    signal = replace(signal, scan_result=book.rows["WETO"].hit)
    timer.queue(signal)
    queued = timer._pending["WETO"].queued_at
    old = Bar("WETO", queued-timedelta(seconds=9), 1.07, 1.09, 1.06, 1.08, 100000, provenance=BarSource.CANONICAL_INTERVAL)
    assert timer.on_10s_bar(old) is None
    assert timer._pending["WETO"].bars_seen == 0
    assert observations[-1]["ignored_pre_signal_bar"]
    new = replace(old, ts=queued+timedelta(seconds=1))
    assert timer.on_10s_bar(new) is None  # 1.08 is below the original 1.082 reference.
    new = replace(new, ts=queued+timedelta(seconds=11), close=1.083)
    assert timer.on_10s_bar(new) is signal
    assert not observations[-1]["ignored_pre_signal_bar"]


def test_partial_candle_does_not_extend_original_wall_clock_timeout():
    book, signal, now = setup()
    cancellations = []
    timer = ExecutionTimer(max_wait_bars=3, on_cancel=lambda signal, reason:cancellations.append(reason))
    signal = replace(signal, scan_result=book.rows["WETO"].hit)
    timer.queue(signal)
    queued = timer._pending["WETO"].queued_at
    assert timer.on_10s_bar(Bar("WETO",queued-timedelta(seconds=9),1.07,1.09,1.06,1.08,100000, provenance=BarSource.CANONICAL_INTERVAL)) is None
    class Clock:
        @classmethod
        def now(cls, *args): return queued+timedelta(seconds=30)
    with patch("daytrading.strategy.execution_timer.datetime",Clock):
        assert not timer.check_timeouts()
    assert not timer.pending_symbols and "30s timeout" in cancellations[-1]


def quiet_guard(monkeypatch):
    monkeypatch.setattr(entry_guard, "ENTRY_ML_ENABLED", False)
    monkeypatch.setattr(entry_guard, "refresh_entry_model", lambda: None)
    monkeypatch.setattr(entry_guard, "record_rule_rejection", lambda **kwargs: None)
    monkeypatch.setattr(entry_guard, "_log_candidate", lambda *args, **kwargs: None)
    monkeypatch.setattr("daytrading.strategy.scalping.momentum_pattern.record_rule_rejection", lambda **kwargs: None)


def test_weto_recovered_but_real_shared_guard_still_blocks_at_79(monkeypatch):
    quiet_guard(monkeypatch)
    source = json.loads((Path(__file__).parent/"fixtures/weto_hod_retry_context.json").read_text())
    ts = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
    now = ts(source["now"])
    bars = [Bar("WETO", ts(b['t']), b['o'], b['h'], b['l'], b['c'], b['v'], provenance=BarSource.CANONICAL_INTERVAL) for b in source['bars']]
    quotes = [Quote("WETO", ts(q['t']), q['bp'], q['ap'], q['bs'], q['as']) for q in source['quotes']]
    ticks = [Tick("WETO", ts(t['t']), t['p'], t['s'], Side(t['side'])) for t in source['ticks']]
    book, signal, cancelled = setup()
    book.mark("WETO", 1.05, cancelled+timedelta(seconds=6), quote=True)
    book.mark("WETO", quotes[-1].bid, now, quote=True)
    verifier = MomentumPatternVerifier(max_dollar_risk=50, max_price=20, reward_risk_ratio=2)
    aggregator = BarAggregator()
    aggregator.build_5m_bars("WETO", bars)
    verifier._bar_aggregator = aggregator
    verifier._quote_buffer, verifier._tick_buffer = {"WETO": quotes}, {"WETO": ticks}
    scanner = SimpleNamespace(name="hod_reclaim", scan=lambda universe: [])
    pipeline = TradingPipeline([scanner], {"hod_reclaim": verifier}, PaperBroker(), PortfolioState(cash=25000))
    pipeline.pending_entries = book
    pipeline._entry_policy.entry_patterns = {"hod_reclaim"}
    pipeline._execution_timer = object()
    pipeline._bar_aggregator = aggregator
    pipeline._max_entry_risk_pct = .06
    result = pipeline.run_cycle({"WETO": bars}, now=now, quotes={"WETO": quotes}, ticks={"WETO": ticks})
    assert not result.deferred_signals and not result.fills
    assert any("79/100" in row.get("reason", "") for row in result.entry_decisions)
    assert "WETO" in book.rows  # Still observed until its fixed deadline.


@pytest.mark.parametrize("block", [None, "circuit_breaker", "entry_count", "daily_loser", "position_limit", "stop_risk"])
def test_valid_retained_hod_routes_through_verifier_risk_and_timer_without_new_scanner_hit(monkeypatch, block):
    quiet_guard(monkeypatch)
    source = json.loads((Path(__file__).parent/"fixtures/sprc_entry_context.json").read_text())
    ts = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
    now = ts(source["now"])
    bars = [Bar("SPRC", ts(b['t']), b['o'], b['h'], b['l'], b['c'], b['v'], provenance=BarSource.CANONICAL_INTERVAL) for b in source['bars']]
    # Controlled HOD setup on historical bars, not a historical SPRC fill claim.
    hit = ScanResult("SPRC", "hod_reclaim", bars[-1].ts, 90,
                     dict(pattern="hod_reclaim", direction="up", hod=4.90, pullback_low=4.7001,
                          stop_price=4.6801, close=4.93, volume=bars[-1].volume, setup_tier="A+ setup"), bars)
    signal = TradeSignal("SPRC", SignalAction.ENTER_LONG, 100, 4.93, 4.6801, scan_result=hit)
    if block == "stop_risk":
        signal = replace(signal, stop_loss=4.50)
    verifier = MomentumPatternVerifier(max_dollar_risk=50, max_price=20, reward_risk_ratio=2)
    aggregator = BarAggregator()
    aggregator.build_5m_bars("SPRC", bars)
    verifier._bar_aggregator = aggregator
    quotes = {"SPRC": [Quote("SPRC", now+timedelta(seconds=2),4.94,4.95,1000,1000)]*3}
    verifier._quote_buffer = quotes
    pipeline = TradingPipeline([SimpleNamespace(name="hod_reclaim",scan=lambda universe: [])],
                               {"hod_reclaim":verifier}, PaperBroker(), PortfolioState(cash=25000), max_order_shares=500)
    pipeline._bar_aggregator = aggregator
    pipeline._max_entry_risk_pct = .06
    pipeline._entry_policy.entry_patterns = {"hod_reclaim"}
    pipeline._execution_timer = object()
    if block == "circuit_breaker": pipeline._circuit_breaker_tripped = True
    if block == "entry_count": pipeline._symbol_entry_counts["SPRC"] = 3
    if block == "daily_loser":
        pipeline._enable_daily_loser_blacklist = True
        pipeline._daily_losers.add("SPRC")
    if block == "position_limit": pipeline._max_positions = 0
    pipeline.pending_entries.timer_cancel(signal, REASON, now)
    pipeline.pending_entries.mark("SPRC",4.90,now+timedelta(seconds=1),quote=True)
    pipeline.pending_entries.mark("SPRC",4.94,now+timedelta(seconds=2),quote=True)
    result = pipeline.run_cycle({"SPRC":bars},now=now+timedelta(seconds=2),quotes=quotes)
    if block:
        assert not result.deferred_signals and not result.fills
        return
    assert len(result.deferred_signals) == 1 and not result.fills, [(r['stage'],r['reason']) for r in result.entry_decisions]
    assert result.deferred_signals[0].quantity <= 500
    assert result.deferred_signals[0].stop_loss >= signal.stop_loss
    assert hit_identity(result.deferred_signals[0].scan_result) == hit_identity(hit)
    pipeline._scanners[0].scan = lambda universe: [hit]  # Repeated detector hit cannot duplicate the retry.
    again = pipeline.run_cycle({"SPRC":bars},now=now+timedelta(seconds=3),quotes=quotes)
    assert not again.deferred_signals and not again.fills
