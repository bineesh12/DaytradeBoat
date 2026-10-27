"""Restored offline hypotheses cannot change the registered PAPER strategy."""

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

from daytradeboat_studies import micro_pullback as study
from daytradeboat_studies import micro_pullback_adapter as adapter
from test_micro_paper_runtime import AT, minutes, quote
from test_micro_support_study import base, make_hit

from daytrading.models import PortfolioState, SignalAction, TradeSignal
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.strategy.scalping import micro_pullback as production
from daytrading.strategy.scalping import micro_pullback_runtime as runtime


def test_research_support_cannot_replace_the_registered_production_rule():
    p = create_scalping_pipeline(
        entry_patterns=(production.NAME,), broker=SimpleNamespace(_paper=True)
    )
    scanner = next(s for s in p._scanners if s.name == production.NAME)
    assert type(scanner) is runtime.MicroPullbackScanner
    assert type(scanner.rules) is production.MicroPullbackRules
    assert type(p._verifiers[production.NAME]) is runtime.MicroPullbackVerifier
    assert (
        production.evaluate_micro_pullback(
            "TEST", base(), now=AT, minute_bars=minutes()
        ).code
        == "lows"
    )
    hypothesis = study.evaluate_micro_pullback(
        "TEST",
        base(),
        now=AT,
        minute_bars=minutes(),
        rules=study.MicroPullbackRules(base_mode="support_reclaim"),
    )
    assert not isinstance(hypothesis, study.Rejection)


def test_research_support_stop_cannot_enter_through_production_order_planning():
    hit = make_hit()
    sig = TradeSignal(
        "TEST", SignalAction.ENTER_LONG, 100, 2.41, 2.32, 2.59, scan_result=hit
    )
    result = runtime.bounded_micro_order(
        sig, quote(), minutes()[-1], now=AT, limits=production.MicroRiskLimits()
    )
    assert result.code == "stop"


def test_research_adapter_rejects_future_observation_before_quality_or_order(
    monkeypatch,
):
    hit = make_hit()
    hit = replace(
        hit,
        confirmation_window=replace(
            hit.confirmation_window, observed_at=AT + timedelta(microseconds=1)
        ),
    )
    guard = Mock(return_value=None)
    monkeypatch.setattr(adapter, "check_entry_quality", guard)
    verifier = adapter.MicroPullbackVerifier(study.MicroRiskLimits())
    verifier._quote_buffer = {"TEST": [quote()]}
    assert verifier.verify(hit, PortfolioState(cash=25000), now=AT) is None
    guard.assert_not_called()
    sig = TradeSignal(
        "TEST", SignalAction.ENTER_LONG, 100, 2.41, 2.32, 2.59, scan_result=hit
    )
    assert (
        adapter.bounded_micro_order(
            sig, quote(), minutes()[-1], now=AT, limits=study.MicroRiskLimits()
        ).code
        == "future"
    )


def test_research_adapter_requires_original_typed_identity(monkeypatch):
    hit = make_hit()
    hit.criteria.pop("micro_setup_id")
    guard = Mock(return_value=None)
    monkeypatch.setattr(adapter, "check_entry_quality", guard)
    verifier = adapter.MicroPullbackVerifier(study.MicroRiskLimits())
    verifier._quote_buffer = {"TEST": [quote()]}
    assert verifier.verify(hit, PortfolioState(cash=25000), now=AT) is None
    guard.assert_not_called()
