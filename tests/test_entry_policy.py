from __future__ import annotations

from datetime import datetime, timezone

from daytrading.models import Bar, ScanResult, SignalAction, TradeSignal
from daytrading.strategy.entry_policy import EntryPolicy


def _bars(symbol: str = "AAPL", price: float = 5.0) -> list[Bar]:
    ts = datetime.now(timezone.utc)
    return [
        Bar(symbol=symbol, ts=ts, open=price, high=price + 0.1, low=price - 0.1, close=price, volume=200_000),
        Bar(symbol=symbol, ts=ts, open=price, high=price + 0.1, low=price - 0.1, close=price, volume=220_000),
        Bar(symbol=symbol, ts=ts, open=price, high=price + 0.1, low=price - 0.1, close=price, volume=250_000),
    ]


def _signal(symbol: str = "AAPL", price: float = 5.0) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=10,
        entry_price=price,
        reason="VWAP pullback",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="vwap_pullback",
            ts=datetime.now(timezone.utc),
            score=90.0,
            criteria={
                "pattern": "vwap_pullback",
                "setup_tier": "A+ setup",
                "entry_tier": "a_plus_reclaim_scout",
            },
        ),
    )


def test_entry_policy_returns_structured_reject(monkeypatch) -> None:
    def fake_guard(*args, **kwargs):
        assert kwargs["entry_pattern"] == "vwap_pullback"
        assert kwargs["setup_tier"] == "A+ setup"
        assert kwargs["entry_tier"] == "a_plus_reclaim_scout"
        return "entry score too low (75/100, need 80+)"

    monkeypatch.setattr("daytrading.strategy.entry_policy.check_entry_quality", fake_guard)

    decision = EntryPolicy().evaluate(
        _signal(),
        bars=_bars(),
        stage="final_entry_guard",
    )

    assert decision.passed is False
    assert decision.blocked_layer == "entry_guard"
    assert decision.reason == "entry score too low (75/100, need 80+)"
    assert decision.pattern == "vwap_pullback"
    assert decision.setup_tier == "A+ setup"
    assert decision.entry_tier == "a_plus_reclaim_scout"
    assert decision.metadata["entry_score"] == 75
    assert decision.to_payload()["stage"] == "final_entry_guard"


def test_entry_policy_returns_structured_pass(monkeypatch) -> None:
    monkeypatch.setattr(
        "daytrading.strategy.entry_policy.check_entry_quality",
        lambda *args, **kwargs: None,
    )

    decision = EntryPolicy().evaluate(
        _signal(),
        bars=_bars(),
        stage="timed_entry_final_guard",
    )

    assert decision.passed is True
    assert decision.reject_reason is None
    assert decision.blocked_layer == ""
    assert decision.reason == ""
