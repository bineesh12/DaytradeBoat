from daytrading.models import BarSource
"""Safety and generality checks for the unregistered micro research rules."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from daytrading.models import Bar, Quote, ScanResult, Timeframe
from daytrading.opportunities import hit_identity
from daytrading.strategy.scalping.micro_pullback import (
    NAME,
    CompletedMicroBar,
    MicroPullbackRules,
    MicroRiskLimits,
    Rejection,
    assess_micro_entry,
    evaluate_micro_pullback,
    partial_runner_stop,
)

AT = datetime(2026, 10, 15, 12, 4, tzinfo=UTC)


def window(symbol="OTHER"):
    prices = [
        (1.65, 1.70, 1.60, 1.68),
        (1.68, 1.85, 1.67, 1.82),
        (1.82, 1.83, 1.70, 1.72),
        (1.72, 1.78, 1.72, 1.76),
        (1.76, 1.79, 1.75, 1.77),
        (1.7796, 1.83, 1.77, 1.80),
    ]
    return [
        CompletedMicroBar(
            Bar(
                symbol, AT - timedelta(seconds=60 - i * 10), *p, 10000, Timeframe.SEC_10
            , provenance=BarSource.CANONICAL_INTERVAL),
            AT - timedelta(seconds=50 - i * 10),
        )
        for i, p in enumerate(prices)
    ]


def detect(symbol="OTHER", **kw):
    return evaluate_micro_pullback(
        symbol, window(symbol), now=AT, rules=MicroPullbackRules(close_floor=0.5), **kw
    )


def quote(**kw):
    return replace(Quote("OTHER", AT, 1.79, 1.80, 1000, 500), **kw)


@pytest.mark.parametrize("symbol", ["OTHER", "VBIO", "NCPL", "BENF", "GENK"])
def test_same_rules_for_every_symbol(symbol):
    r = detect(symbol)
    assert not isinstance(r, Rejection)
    assert r.anchor == 1.80 and r.technical_stop == 1.76
    assert r.confirmed_at == AT and r.setup_id.startswith(symbol + "|")


def test_primary_65_does_not_accept_50_percent_sensitivity():
    r = evaluate_micro_pullback("OTHER", window(), now=AT)
    assert r.code == "close_location" and "50.0% < 65.0%" in r.reason


@pytest.mark.parametrize(
    "change,code",
    [
        ("future", "future"),
        ("stale", "stale"),
        ("other_session", "session"),
        ("mixed_symbol", "identity"),
        ("wrong_timeframe", "identity"),
        ("naive", "timestamp"),
        ("sparse", "sparse_bar"),
        ("gap", "gap"),
        ("bad_ohlc", "ohlcv"),
        ("nan", "ohlcv"),
    ],
)
def test_bad_timing_and_data_cannot_make_a_trigger(change, code):
    bs = window()
    now = AT
    if change == "future":
        now -= timedelta(microseconds=1)
    elif change == "stale":
        now += timedelta(seconds=16)
    elif change == "other_session":
        now += timedelta(days=1)
    elif change == "mixed_symbol":
        bs[0] = replace(bs[0], bar=replace(bs[0].bar, symbol="BAD"))
    elif change == "wrong_timeframe":
        bs[0] = replace(bs[0], bar=replace(bs[0].bar, timeframe=Timeframe.MIN_1))
    elif change == "naive":
        now = AT.replace(tzinfo=None)
    elif change == "sparse":
        bs[0] = replace(bs[0], closed_at=bs[0].closed_at + timedelta(seconds=6))
    elif change == "gap":
        bs[0] = replace(bs[0], closed_at=bs[0].closed_at + timedelta(seconds=1))
    elif change == "bad_ohlc":
        bs[0] = replace(bs[0], bar=replace(bs[0].bar, low=2.0))
    elif change == "nan":
        bs[0] = replace(bs[0], bar=replace(bs[0].bar, volume=float("nan")))
    r = evaluate_micro_pullback(
        "OTHER", bs, now=now, rules=MicroPullbackRules(close_floor=0.5)
    )
    assert r.code == code


def test_rejection_does_not_prevent_a_later_fresh_valid_window():
    assert evaluate_micro_pullback("OTHER", window()[:3], now=AT).code == "history"
    assert not isinstance(detect(), Rejection)


@pytest.mark.parametrize(
    "quote_change,code",
    [
        ({"ts": AT + timedelta(seconds=1)}, "quote_age"),
        ({"ts": AT - timedelta(seconds=11)}, "quote_age"),
        ({"symbol": "BAD"}, "quote"),
        ({"ask_size": 0}, "quote"),
        ({"bid": 1.90}, "quote"),
        ({"ask": float("inf")}, "quote"),
        ({"ask": 1.85}, "chase"),
        ({"bid": 1.70, "ask": 1.71}, "stop"),
    ],
)
def test_entry_quote_validation(quote_change, code):
    assert assess_micro_entry(detect(), quote(**quote_change), now=AT).code == code


def test_risk_sizes_for_effective_room_and_shares_not_lots():
    p = assess_micro_entry(
        detect(),
        quote(ask_size=2000),
        now=AT,
        limits=MicroRiskLimits(max_order_shares=5000),
    )
    assert p.quantity == 925  # 50 / max(technical .04, .05, 3% of 1.80)
    assert p.planned_risk <= 50
    assert p.estimated_effective_stop == pytest.approx(1.746)
    small = assess_micro_entry(detect(), quote(ask_size=7), now=AT)
    assert small.quantity == 7


def test_structural_stop_not_clamped_to_force_trade():
    r = assess_micro_entry(replace(detect(), technical_stop=1.50), quote(), now=AT)
    assert r.code == "stop" and "never clamped" in r.reason


def test_expired_confirmation_cannot_skip_timer():
    r = assess_micro_entry(
        detect(), quote(ts=AT + timedelta(seconds=16)), now=AT + timedelta(seconds=16)
    )
    assert r.code == "confirmation_age"


def test_fill_activated_trail_never_lowers_stop():
    kw = {"entry": 1.79, "initial_stop": 1.7363, "highest_observed_bid": 2.12}
    assert (
        partial_runner_stop(previous_stop=1.7363, partial_filled_shares=0, **kw)
        == 1.7363
    )
    assert partial_runner_stop(
        previous_stop=1.7363, partial_filled_shares=250, **kw
    ) == pytest.approx(2.0663)
    assert (
        partial_runner_stop(previous_stop=2.10, partial_filled_shares=250, **kw) == 2.10
    )


def test_micro_triggers_have_distinct_ids_inside_same_minute_context():
    bar = Bar("OTHER", AT, 1.7, 1.9, 1.6, 1.8, 10000, provenance=BarSource.CANONICAL_INTERVAL)
    h = ScanResult(
        "OTHER",
        NAME,
        AT,
        10,
        {"micro_bar_start": (AT - timedelta(seconds=20)).isoformat()},
        [bar],
    )
    later = replace(
        h, criteria={"micro_bar_start": (AT - timedelta(seconds=10)).isoformat()}
    )
    assert hit_identity(h) != hit_identity(later)
    assert hit_identity(h) == hit_identity(replace(h, ts=AT + timedelta(seconds=1)))
    # Existing one-minute patterns retain their original identity convention.
    assert hit_identity(replace(h, scanner_name="hod_reclaim")) == hit_identity(
        replace(later, scanner_name="hod_reclaim")
    )


def test_invalid_micro_identity_falls_back_without_breaking_other_records():
    h = ScanResult("OTHER", NAME, AT, 10, {"micro_bar_start": "bad"})
    assert hit_identity(h)[1] == AT.isoformat()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"close_floor": 1.1},
        {"close_floor": float("nan")},
        {"max_completion_seconds": 9},
    ],
)
def test_invalid_detector_config(kwargs):
    with pytest.raises(ValueError):
        MicroPullbackRules(**kwargs)
