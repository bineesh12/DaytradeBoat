from datetime import datetime, timezone

from daytrading.execution.entry_executor import EntryExecutor
from daytrading.models import Bar, ScanResult, SignalAction, Timeframe, TradeSignal
from daytrading.strategy.entry_policy import EntryPolicy


def _bar(symbol="BATL", close=1.66):
    return Bar(
        symbol=symbol,
        ts=datetime.now(timezone.utc),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=10_000,
        timeframe=Timeframe.MIN_1,
    )


def _signal(symbol="BATL"):
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=10,
        entry_price=1.66,
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="pullback_base",
            ts=datetime.now(timezone.utc),
            score=0.9,
            criteria={"pattern": "pullback_base", "setup_tier": "A+"},
        ),
    )


def test_entry_executor_records_policy_decision():
    recorded = []
    policy = EntryPolicy(guard=lambda *args, **kwargs: None)
    executor = EntryExecutor(policy, lambda decision, source: recorded.append((decision, source)))

    decision = executor.evaluate_quality(
        _signal(),
        bars=[_bar()],
        stage="unit_test",
        source="test",
    )

    assert decision.passed
    assert recorded == [(decision, "test")]


def test_entry_executor_reject_preserves_structured_context():
    recorded = []
    executor = EntryExecutor(
        EntryPolicy(guard=lambda *args, **kwargs: None),
        lambda decision, source: recorded.append((decision, source)),
    )

    decision = executor.reject(
        symbol="BATL",
        stage="spread_gate",
        reason="spread too wide",
        source="test",
        signal=_signal(),
    )

    assert not decision.passed
    assert decision.pattern == "pullback_base"
    assert decision.blocked_layer == "spread_gate"
    assert recorded[0][1] == "test"
