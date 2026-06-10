from __future__ import annotations

from datetime import datetime, timedelta, timezone

from daytrading.analytics.missed_a_plus import MissedAPlusTracker
from daytrading.dashboard.hub import DashboardHub
from daytrading.models import Bar, ScanResult


def _bar(
    i: int,
    *,
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 100_000,
    base_ts: datetime | None = None,
    n: int = 20,
) -> Bar:
    base_ts = base_ts or datetime.now(timezone.utc)
    open_px = open_ if open_ is not None else close - 0.02
    return Bar(
        symbol="HOT",
        ts=base_ts + timedelta(seconds=i - n),
        open=open_px,
        high=high if high is not None else close + 0.04,
        low=low if low is not None else min(open_px, close) - 0.04,
        close=close,
        volume=volume,
    )


def _a_plus_bars(base_ts: datetime | None = None) -> list[Bar]:
    base_ts = base_ts or datetime.now(timezone.utc)
    bars: list[Bar] = []
    for i in range(16):
        close = 4.0 + i * 0.08
        bars.append(_bar(
            i,
            close=close,
            open_=close - 0.03,
            high=close + 0.05,
            low=close - 0.06,
            volume=90_000 if i < 11 else 180_000,
            base_ts=base_ts,
            n=20,
        ))
    bars.extend([
        _bar(16, close=5.28, open_=5.22, high=5.31, low=5.18, volume=220_000, base_ts=base_ts, n=20),
        _bar(17, close=5.34, open_=5.28, high=5.37, low=5.26, volume=240_000, base_ts=base_ts, n=20),
        _bar(18, close=5.42, open_=5.34, high=5.45, low=5.32, volume=260_000, base_ts=base_ts, n=20),
    ])
    return bars


def _hit(bars: list[Bar], *, pattern: str = "level_breakout_reclaim") -> ScanResult:
    return ScanResult(
        symbol="HOT",
        scanner_name=pattern,
        ts=bars[-1].ts,
        score=42.0,
        criteria={
            "pattern": pattern,
            "setup_tier": "A+ setup",
            "breakout_level": 5.25,
            "close": bars[-1].close,
            "volume": bars[-1].volume,
        },
        bars=bars,
    )


def test_a_plus_reject_gets_recorded_and_labeled_as_missed_opportunity() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=60)
    bars = _a_plus_bars(now)

    rec = tracker.record_blocked(
        layer="entry_guard",
        reason="entry score too low (78/100, need 80+)",
        universe={"HOT": bars},
        hit=_hit(bars),
        now=now,
    )

    assert rec is not None
    later = list(bars)
    later.append(_bar(19, close=5.72, open_=5.45, high=5.74, low=5.44, volume=300_000, base_ts=now, n=20))
    tracker.update_prices({"HOT": later}, now=now + timedelta(seconds=90))

    report = tracker.report()
    assert report[0]["symbol"] == "HOT"
    assert report[0]["outcome"] == "missed_opportunity"
    assert report[0]["correct"] is False
    assert report[0]["move_after_pct"] >= 5.0
    assert "guard" in report[0]["suggested_fix"].lower()


def test_small_pop_then_hard_dump_is_correct_reject_not_missed() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=60)
    bars = _a_plus_bars(now)

    rec = tracker.record_blocked(
        layer="entry_guard",
        reason="pullback has dump candle 1.1% body/19.8% range",
        universe={"HOT": bars},
        hit=_hit(bars, pattern="vwap_pullback"),
        now=now,
    )

    assert rec is not None
    later = list(bars)
    later.append(_bar(
        19,
        close=5.18,
        open_=5.42,
        high=5.72,
        low=4.05,
        volume=420_000,
        base_ts=now,
        n=20,
    ))
    tracker.update_prices({"HOT": later}, now=now + timedelta(seconds=90))

    report = tracker.report()
    assert report[0]["outcome"] == "correct_reject"
    assert report[0]["correct"] is True
    assert report[0]["move_after_pct"] >= 3.0
    assert report[0]["dump_after_pct"] <= -6.0
    assert "hard dump" in report[0]["suggested_fix"].lower()


def test_large_continuation_still_counts_as_missed_despite_pullback() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=60)
    bars = _a_plus_bars(now)
    tracker.record_blocked(
        layer="entry_guard",
        reason="spread too wide",
        universe={"HOT": bars},
        hit=_hit(bars, pattern="abc_continuation"),
        now=now,
    )

    later = list(bars)
    later.append(_bar(
        19,
        close=6.15,
        open_=5.42,
        high=6.30,
        low=5.05,
        volume=500_000,
        base_ts=now,
        n=20,
    ))
    tracker.update_prices({"HOT": later}, now=now + timedelta(seconds=90))

    report = tracker.report()
    assert report[0]["outcome"] == "missed_opportunity"
    assert report[0]["correct"] is False
    assert report[0]["move_after_pct"] >= 8.0


def test_report_ranks_biggest_missed_runners_first() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=1)
    bars = _a_plus_bars(now)
    tracker.record_blocked(
        layer="ml",
        reason="ML model low confidence (22%, need 30%)",
        universe={"HOT": bars},
        hit=_hit(bars),
        now=now,
    )
    other = [Bar(symbol="RUN", ts=b.ts, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume) for b in bars]
    run_hit = ScanResult(
        symbol="RUN",
        scanner_name="vwap_pullback",
        ts=other[-1].ts,
        score=35.0,
        criteria={"pattern": "vwap_pullback", "setup_tier": "A+ setup", "close": other[-1].close},
        bars=other,
    )
    tracker.record_blocked(
        layer="timed_entry",
        reason="10s confirmation red/flat",
        universe={"RUN": other},
        hit=run_hit,
        now=now,
    )
    tracker.update_prices({
        "HOT": bars + [_bar(19, close=5.60, high=5.62, base_ts=now, n=20)],
        "RUN": other + [Bar(symbol="RUN", ts=now, open=5.4, high=6.2, low=5.35, close=6.1, volume=350_000)],
    }, now=now + timedelta(seconds=5))

    report = tracker.report()
    assert report[0]["symbol"] == "RUN"
    assert report[0]["move_after_pct"] > report[1]["move_after_pct"]


def test_weak_non_a_plus_reject_does_not_pollute_report() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=1)
    weak = [
        _bar(i, close=2.0 + i * 0.005, high=2.02 + i * 0.005, low=1.99, volume=5_000, base_ts=now, n=12)
        for i in range(12)
    ]
    hit = ScanResult(
        symbol="HOT",
        scanner_name="momentum_burst",
        ts=weak[-1].ts,
        score=2.0,
        criteria={"pattern": "momentum_burst", "close": weak[-1].close},
        bars=weak,
    )

    rec = tracker.record_blocked(
        layer="verifier",
        reason="watch only",
        universe={"HOT": weak},
        hit=hit,
        now=now,
    )

    assert rec is None
    assert tracker.report() == []


def test_dashboard_snapshot_exposes_missed_a_plus_rows() -> None:
    hub = DashboardHub()
    hub.on_missed_a_plus([{
        "symbol": "HOT",
        "pattern": "vwap_pullback",
        "blocked_layer": "entry_guard",
        "reason": "entry score too low",
        "move_after_pct": 5.2,
    }])

    snap = hub.snapshot()

    assert snap["missed_a_plus"][0]["symbol"] == "HOT"
