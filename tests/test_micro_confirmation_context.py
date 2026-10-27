from daytrading.models import BarSource
"""Keep execution confirmation separate from minute/session quality context."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from daytrading.models import (
    Bar,
    CompletedBar,
    ConfirmationWindow,
    ScanResult,
    SignalAction,
    Timeframe,
    TradeSignal,
)
from daytrading.risk.guards import FalseBreakoutDetector, TradeGuard

CASES = json.loads(
    (Path(__file__).parent / "fixtures/micro_guard_timeframes.json").read_text()
)["cases"]


def context(case=CASES[0]):
    bs = tuple(
        CompletedBar(
            Bar(
                case["symbol"],
                datetime.fromisoformat(x["start"]),
                x["open"],
                x["high"],
                x["low"],
                x["close"],
                x["volume"],
                Timeframe.SEC_10,
             provenance=BarSource.CANONICAL_INTERVAL),
            datetime.fromisoformat(x["closed_at"]),
        )
        for x in case["micro_bars"]
    )
    now = datetime.fromisoformat(case["now"])
    w = ConfirmationWindow(bs, now)
    h = ScanResult(
        case["symbol"],
        "micro_pullback_reclaim",
        now,
        90,
        case["criteria"].copy(),
        confirmation_window=w,
    )
    s = TradeSignal(
        case["symbol"],
        SignalAction.ENTER_LONG,
        100,
        bs[-1].bar.close,
        stop_loss=bs[-1].bar.low - 0.01,
        scan_result=h,
    )
    # Deliberately weak partial minute, yet otherwise valid OHLCV.
    minutes = [
        Bar(s.symbol, now - timedelta(minutes=6 - i), 2, 2.1, 1.9, 2.02, 10000, provenance=BarSource.CANONICAL_INTERVAL)
        for i in range(6)
    ]
    minutes[-1] = replace(minutes[-1], volume=100)
    return s, w, now, minutes


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["symbol"] + c["now"])
def test_real_completed_micro_volumes_preserve_the_expected_rejections(case):
    s, w, now, minutes = context(case)
    g = TradeGuard()
    # Isolate confirmation comparison while spying that liquidity sees minutes.
    g.liquidity_trap.check = Mock(return_value=None)
    assert "volume declining" in FalseBreakoutDetector().check(minutes)
    ok, reason = g.check_entry(s, bars=minutes, breakout_context=w, now=now)
    assert ok == case["expected_micro_pass"]
    if not ok:
        assert "volume declining" in reason
    else:
        g.liquidity_trap.check.assert_called_once_with(
            minutes, quotes=None, symbol=s.symbol, wick_bar=w.bars[-1].bar,
            now=now, diagnostics=s.scan_result.criteria['quality_inputs'],
        )


@pytest.mark.parametrize(
    "change,reason",
    [
        ("missing", "context"),
        ("stale", "stale"),
        ("future", "future"),
        ("short", "six completed"),
        ("symbol", "mixed symbol"),
        ("minute", "non-10-second"),
        ("sparse", "sparse_bar"),
        ("identity", "does not match"),
        ("completion", "does not match"),
        ("observation", "observation"),
        ("observed_before_close", "does not match"),
        ("naive", "observation"),
    ],
)
def test_micro_confirmation_fails_closed(change, reason):
    s, w, now, minutes = context()
    if change == "missing":
        w = None
    elif change == "stale":
        now += timedelta(seconds=16)
    elif change == "future":
        now = w.bars[-1].closed_at - timedelta(microseconds=1)
    elif change == "short":
        w = replace(w, bars=w.bars[-5:])
    elif change == "symbol":
        w = replace(
            w,
            bars=(replace(w.bars[0], bar=replace(w.bars[0].bar, symbol="BAD")),)
            + w.bars[1:],
        )
    elif change == "minute":
        w = replace(
            w,
            bars=(
                replace(
                    w.bars[0], bar=replace(w.bars[0].bar, timeframe=Timeframe.MIN_1)
                ),
            )
            + w.bars[1:],
        )
    elif change == "sparse":
        w = replace(
            w,
            bars=(
                replace(
                    w.bars[0], closed_at=w.bars[0].closed_at + timedelta(seconds=20)
                ),
            )
            + w.bars[1:],
        )
    elif change == "identity":
        s = replace(
            s,
            scan_result=replace(
                s.scan_result,
                criteria={**s.scan_result.criteria, "micro_bar_start": now.isoformat()},
            ),
        )
    elif change == "completion":
        s = replace(
            s,
            scan_result=replace(
                s.scan_result,
                criteria={
                    **s.scan_result.criteria,
                    "micro_confirmed_at": now.isoformat(),
                },
            ),
        )
    elif change == "observation":
        w = replace(w, observed_at=now + timedelta(seconds=1))
    elif change == "observed_before_close":
        w = replace(w, observed_at=w.bars[-1].closed_at - timedelta(seconds=1))
    elif change == "naive":
        now = now.replace(tzinfo=None)
    ok, why = TradeGuard().check_entry(s, bars=minutes, breakout_context=w, now=now)
    assert not ok and reason in why


def test_confirmation_is_immutable_and_survives_retained_setup_copy():
    s, w, now, minutes = context()
    incoming = list(w.bars)
    frozen = ConfirmationWindow(incoming, now)
    incoming.clear()
    assert len(frozen.bars) == 6
    # PendingEntryBook uses deepcopy plus replace to refresh minute context.
    refreshed = replace(deepcopy(s.scan_result), bars=minutes)
    assert refreshed.confirmation_window == w
    assert (
        refreshed.confirmation_window.bars[-1].bar.ts
        == s.scan_result.confirmation_window.bars[-1].bar.ts
    )


@pytest.mark.parametrize(
    "reason", ["spread limit", "liquidity trap", "HALTED", "market panic"]
)
def test_micro_context_never_bypasses_other_guards(reason):
    s, w, now, minutes = context()
    g = TradeGuard()
    gate = {
        "spread limit": g.slippage,
        "liquidity trap": g.liquidity_trap,
        "HALTED": g.halt_tracker,
        "market panic": g.market_panic,
    }[reason]
    if reason == "spread limit":
        gate.check_spread = Mock(return_value=reason)
    else:
        gate.check = Mock(return_value=reason)
    ok, why = g.check_entry(s, bars=minutes, breakout_context=w, now=now)
    assert not ok and why == reason


def test_existing_pattern_uses_original_minute_guard():
    s, _w, _now, minutes = context()
    s = replace(
        s,
        scan_result=replace(
            s.scan_result, scanner_name="bull_flag", criteria={"pattern": "bull_flag"}
        ),
    )
    ok, why = TradeGuard().check_entry(s, bars=minutes)
    assert not ok and "volume declining" in why


def test_micro_criteria_cannot_spoof_the_existing_pullback_exemption():
    s, w, now, minutes = context(CASES[1])
    s.scan_result.criteria["pattern"] = "vwap_pullback"
    ok, why = TradeGuard().check_entry(s, bars=minutes, breakout_context=w, now=now)
    assert not ok and "volume declining" in why
