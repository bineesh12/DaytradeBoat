from __future__ import annotations

from datetime import datetime, timedelta, timezone

from daytrading.analytics.missed_a_plus import MissedAPlusTracker
from daytrading.dashboard.hub import DashboardHub
from daytrading.dashboard.server import create_app, _missed_a_plus_risk_summary
from daytrading.models import Bar, Quote, ScanResult, SignalAction, TradeSignal


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
    assert "spread rule" in report[0]["suggested_fix"].lower()


def test_spread_reject_summary_counts_false_and_correct_blocks() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=1)
    hot = _a_plus_bars(now)
    weak = _a_plus_bars(now)
    weak_hit = _hit(weak, pattern="first_pullback_reclaim")
    weak_hit = ScanResult(
        symbol="JUNK",
        scanner_name=weak_hit.scanner_name,
        ts=weak_hit.ts,
        score=weak_hit.score,
        criteria=dict(weak_hit.criteria, close=weak[-1].close),
        bars=weak,
    )

    tracker.record_blocked(
        layer="scanner",
        reason="spread too wide (1.80c = 0.85% of $2.13)",
        universe={"HOT": hot},
        quotes={"HOT": [Quote("HOT", now, 2.12, 2.138, 1200, 1100)] * 3},
        hit=_hit(hot, pattern="runner_reclaim_continuation"),
        now=now,
    )
    tracker.record_blocked(
        layer="scanner",
        reason="spread too wide (6.80c = 2.31% of $2.95)",
        universe={"JUNK": weak},
        hit=weak_hit,
        now=now,
    )

    hot_later = list(hot)
    hot_later.append(_bar(19, close=5.90, high=6.00, low=5.40, volume=350_000, base_ts=now, n=20))
    junk_later = list(weak)
    junk_later.append(_bar(19, close=5.10, high=5.46, low=4.60, volume=80_000, base_ts=now, n=20))
    tracker.update_prices({"HOT": hot_later, "JUNK": junk_later}, now=now + timedelta(seconds=5))

    summary = tracker.spread_summary()

    assert summary["spread_blocked_runners"] == 2
    assert summary["spread_false_blocks"] == 1
    assert summary["spread_correct_rejects"] == 1
    assert summary["symbols"] == {"HOT": 1, "JUNK": 1}
    row = next(r for r in tracker.report(limit=10) if r["symbol"] == "HOT")
    assert row["is_spread_reject"] is True
    assert row["spread_pct"] > 0


def test_wide_risk_reject_records_risk_and_tactical_stop_survival() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=1)
    bars = _a_plus_bars(now)
    bars[-3] = _bar(16, close=5.28, high=5.28, low=5.18, volume=220_000, base_ts=now, n=20)
    bars[-2] = _bar(17, close=5.34, high=5.34, low=5.25, volume=240_000, base_ts=now, n=20)
    bars[-1] = _bar(18, close=5.42, high=5.42, low=5.33, volume=260_000, base_ts=now, n=20)
    hit = _hit(bars, pattern="level_breakout_reclaim")
    hit.criteria["stop_price"] = 4.27

    rec = tracker.record_blocked(
        layer="verifier",
        reason="risk too wide: $1.15 (21% of $5.42) — skip loose setup",
        universe={"HOT": bars},
        hit=hit,
        now=now,
    )

    assert rec is not None
    assert rec.risk_per_share == 1.15
    assert rec.risk_pct == 21
    assert rec.tactical_stop_price == 5.16

    later = list(bars)
    later.append(_bar(19, close=5.95, high=6.05, low=5.24, volume=420_000, base_ts=now, n=20))
    tracker.update_prices({"HOT": later}, now=now + timedelta(seconds=5))

    row = tracker.report(limit=10)[0]
    assert row["is_risk_reject"] is True
    assert row["outcome"] == "missed_opportunity"
    assert row["risk_per_share"] == 1.15
    assert row["risk_pct"] == 21
    assert row["tactical_stop_price"] == 5.16
    assert row["tactical_stop_survived"] is True
    assert row["smooth_for_tactical_stop"] is True
    assert row["tactical_stop_clean_survival"] is True
    assert row["median_bar_range_pct"] <= 2.0
    assert "tactical-stop survival" in row["suggested_fix"].lower()

    summary = tracker.risk_summary()
    assert summary["risk_blocked_runners"] == 1
    assert summary["risk_false_blocks"] == 1
    assert summary["tactical_stop_survived"] == 1
    assert summary["tactical_stop_failed"] == 0
    assert summary["clean_tactical_stop_survived"] == 1
    assert summary["choppy_tactical_stop_survived"] == 0


def test_wide_risk_reject_that_breaks_tactical_stop_stays_correct() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=1)
    bars = _a_plus_bars(now)
    bars[-3] = _bar(16, close=5.28, high=5.28, low=5.18, volume=220_000, base_ts=now, n=20)
    bars[-2] = _bar(17, close=5.34, high=5.34, low=5.25, volume=240_000, base_ts=now, n=20)
    bars[-1] = _bar(18, close=5.42, high=5.42, low=5.33, volume=260_000, base_ts=now, n=20)
    hit = _hit(bars, pattern="pullback_base")
    hit.criteria["stop_price"] = 4.20

    tracker.record_blocked(
        layer="verifier",
        reason="risk too wide: $1.22 (23% of $5.42) — skip loose setup",
        universe={"HOT": bars},
        hit=hit,
        now=now,
    )

    later = list(bars)
    later.append(_bar(19, close=5.10, high=5.48, low=4.95, volume=430_000, base_ts=now, n=20))
    tracker.update_prices({"HOT": later}, now=now + timedelta(seconds=5))

    row = tracker.report(limit=10)[0]
    assert row["is_risk_reject"] is True
    assert row["outcome"] == "correct_reject"
    assert row["tactical_stop_survived"] is False
    assert row["tactical_stop_clean_survival"] is False
    assert row["move_after_pct"] < 8

    summary = tracker.risk_summary()
    assert summary["risk_correct_rejects"] == 1
    assert summary["tactical_stop_failed"] == 1
    assert summary["clean_tactical_stop_failed"] == 1


def test_choppy_wide_risk_survival_is_not_counted_as_clean() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=1)
    bars = _a_plus_bars(now)
    bars[-3] = _bar(16, close=5.28, high=5.62, low=5.05, volume=220_000, base_ts=now, n=20)
    bars[-2] = _bar(17, close=5.34, high=5.70, low=5.10, volume=240_000, base_ts=now, n=20)
    bars[-1] = _bar(18, close=5.42, high=5.82, low=5.16, volume=260_000, base_ts=now, n=20)
    hit = _hit(bars, pattern="level_breakout_reclaim")
    hit.criteria["stop_price"] = 4.27

    tracker.record_blocked(
        layer="verifier",
        reason="risk too wide: $1.15 (21% of $5.42) — skip loose setup",
        universe={"HOT": bars},
        hit=hit,
        now=now,
    )

    later = list(bars)
    later.append(_bar(19, close=5.95, high=6.05, low=5.24, volume=420_000, base_ts=now, n=20))
    tracker.update_prices({"HOT": later}, now=now + timedelta(seconds=5))

    row = tracker.report(limit=10)[0]
    assert row["tactical_stop_survived"] is True
    assert row["smooth_for_tactical_stop"] is False
    assert row["tactical_stop_clean_survival"] is None
    assert row["median_bar_range_pct"] > 2.0

    summary = tracker.risk_summary()
    assert summary["tactical_stop_survived"] == 1
    assert summary["clean_tactical_stop_survived"] == 0
    assert summary["choppy_tactical_stop_survived"] == 1


def test_recent_missed_a_plus_rejects_late_chase_entry() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=60)
    bars = _a_plus_bars(now)
    tracker.record_blocked(
        layer="scanner",
        reason="not on HOD momentum alert board",
        universe={"HOT": bars},
        hit=_hit(bars, pattern="hod_reclaim"),
        now=now,
    )

    reason = tracker.chase_reject(
        symbol="HOT",
        price=6.20,
        now=now + timedelta(minutes=10),
    )

    assert reason is not None
    assert "late chase" in reason
    assert "earlier blocked A+" in reason


def test_fresh_base_reset_skips_stale_chase_anchor() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=60)
    bars = _a_plus_bars(now)
    tracker.record_blocked(
        layer="scanner",
        reason="not on HOD momentum alert board",
        universe={"HOT": bars},
        hit=_hit(bars, pattern="hod_reclaim"),
        now=now,
    )
    # Stale level (~$5.42) blocks a $6.20 late chase by default.
    assert tracker.chase_reject(
        symbol="HOT", price=6.20, now=now + timedelta(minutes=10),
    ) is not None
    # But when the current setup's own base ($6.00) has migrated well above the
    # stale level, the fresh-base reset skips that anchor and allows the entry
    # (the primary own-base chase guard still vets it upstream).
    assert tracker.chase_reject(
        symbol="HOT", price=6.20, now=now + timedelta(minutes=10),
        fresh_base_anchor=6.00, fresh_base_reset_pct=0.08,
    ) is None


def test_old_missed_a_plus_does_not_block_fresh_setup() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=60)
    bars = _a_plus_bars(now)
    tracker.record_blocked(
        layer="scanner",
        reason="not ready",
        universe={"HOT": bars},
        hit=_hit(bars, pattern="hod_reclaim"),
        now=now,
    )

    reason = tracker.chase_reject(
        symbol="HOT",
        price=6.20,
        now=now + timedelta(minutes=45),
        max_age_seconds=1800,
    )

    assert reason is None


def test_hod_distance_anchor_does_not_block_fresh_reclaim_retry() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=60)
    bars = _a_plus_bars(now)
    stale_hit = _hit(bars, pattern="vwap_pullback")
    stale_hit.criteria["close"] = 4.00
    tracker.record_blocked(
        layer="verifier",
        reason="late pullback too far from HOD 17.1% (max 12.0%; watching for fresh reclaim)",
        universe={"HOT": bars},
        hit=stale_hit,
        now=now,
        fallback_price=4.00,
    )
    assert tracker.chase_reject(
        symbol="HOT",
        price=5.20,
        now=now + timedelta(minutes=2),
    ) is None

    reclaim_hit = _hit(bars, pattern="vwap_pullback")
    reclaim_hit.criteria["entry_tier"] = "a_plus_retry_watch"
    reclaim_hit.criteria["entry_tier_reason"] = "A+ runner reclaimed a fresh base after scanner reject"
    signal = TradeSignal(
        symbol="HOT",
        action=SignalAction.ENTER_LONG,
        quantity=10,
        entry_price=5.20,
        scan_result=reclaim_hit,
    )

    reason = tracker.chase_reject(
        symbol="HOT",
        price=5.20,
        now=now + timedelta(minutes=2),
        signal=signal,
    )

    assert reason is None


def test_watch_state_and_loose_risk_rejects_do_not_become_chase_anchors() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=60)
    bars = _a_plus_bars(now)
    for i, reason in enumerate((
        "pullback has dump candle 7.4% body/9.7% range (wait for new base)",
        "risk too wide: $1.15 (22% of $5.13) — skip loose setup",
    )):
        hit = _hit(bars, pattern="vwap_pullback")
        hit.criteria["close"] = 4.00 + i * 0.20
        tracker.record_blocked(
            layer="scanner",
            reason=reason,
            universe={"HOT": bars},
            hit=hit,
            now=now + timedelta(minutes=i),
            fallback_price=4.00 + i * 0.20,
        )

    reason = tracker.chase_reject(
        symbol="HOT",
        price=5.50,
        now=now + timedelta(minutes=5),
    )

    assert reason is None


def test_momentum_burst_reject_does_not_block_later_structured_reclaim() -> None:
    now = datetime.now(timezone.utc)
    tracker = MissedAPlusTracker(label_after_seconds=60)
    bars = _a_plus_bars(now)
    burst_hit = _hit(bars, pattern="momentum_burst")
    burst_hit.criteria["close"] = 4.78
    tracker.record_blocked(
        layer="verifier",
        reason="entry score too low (73/100, need 80+)",
        universe={"HOT": bars},
        hit=burst_hit,
        now=now,
        fallback_price=4.78,
    )

    reclaim_hit = _hit(bars, pattern="hod_reclaim")
    signal = TradeSignal(
        symbol="HOT",
        action=SignalAction.ENTER_LONG,
        quantity=10,
        entry_price=5.13,
        scan_result=reclaim_hit,
    )

    reason = tracker.chase_reject(
        symbol="HOT",
        price=5.13,
        now=now + timedelta(minutes=12),
        signal=signal,
    )

    assert reason is None


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


def test_missed_a_plus_api_exposes_scanner_near_miss_summary() -> None:
    hub = DashboardHub()
    hub.on_scanner_near_miss({
        "scanner_near_misses": 2,
        "scanner_gaps": 1,
        "washouts": 1,
        "gap_symbols": ["ASBP"],
    })
    app = create_app(hub)

    resp = app.test_client().get("/api/missed-a-plus")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["scanner_near_miss"]["scanner_gaps"] == 1
    assert data["scanner_near_miss"]["gap_symbols"] == ["ASBP"]


def test_dashboard_risk_summary_counts_wide_risk_rows() -> None:
    summary = _missed_a_plus_risk_summary([
        {
            "symbol": "HOT",
            "reason": "risk too wide: $1.15 (21% of $5.42)",
            "outcome": "missed_opportunity",
            "tactical_stop_survived": True,
            "tactical_stop_clean_survival": True,
            "smooth_for_tactical_stop": True,
        },
        {
            "symbol": "JUNK",
            "is_risk_reject": True,
            "reason": "risk too wide",
            "outcome": "correct_reject",
            "tactical_stop_survived": False,
            "tactical_stop_clean_survival": False,
            "smooth_for_tactical_stop": True,
        },
        {
            "symbol": "CHOP",
            "is_risk_reject": True,
            "reason": "risk too wide",
            "outcome": "missed_opportunity",
            "tactical_stop_survived": True,
            "tactical_stop_clean_survival": None,
            "smooth_for_tactical_stop": False,
        },
        {"symbol": "SPRD", "reason": "spread too wide", "outcome": "missed_opportunity"},
    ])

    assert summary["risk_blocked_runners"] == 3
    assert summary["risk_false_blocks"] == 2
    assert summary["risk_correct_rejects"] == 1
    assert summary["tactical_stop_survived"] == 2
    assert summary["tactical_stop_failed"] == 1
    assert summary["clean_tactical_stop_survived"] == 1
    assert summary["clean_tactical_stop_failed"] == 1
    assert summary["choppy_tactical_stop_survived"] == 1
    assert summary["symbols"] == {"HOT": 1, "JUNK": 1, "CHOP": 1}


def test_scanner_near_miss_flags_smooth_runner_as_gap() -> None:
    t0 = datetime(2026, 6, 12, 14, 0, tzinfo=timezone.utc)
    tracker = MissedAPlusTracker()
    # smooth, tight ~0.7% bars, decent volume, low float
    bars = [
        _bar(i, close=3.00, open_=2.995, high=3.01, low=2.99, volume=150_000, base_ts=t0, n=20)
        for i in range(20)
    ]
    rec = tracker.record_scanner_near_miss(
        symbol="SMOO", reason="no clean A+ pattern (washout/reclaim)",
        universe={"SMOO": bars}, float_shares=8_000_000, now=t0,
    )
    assert rec is not None
    # 5 min later it ran +10% smoothly; low never broke the tactical stop
    runner = bars + [
        _bar(0, close=3.30, open_=3.28, high=3.31, low=3.27, volume=160_000,
             base_ts=t0 + timedelta(seconds=300), n=1)
    ]
    tracker.update_prices({"SMOO": runner}, now=t0 + timedelta(seconds=300))

    s = tracker.scanner_near_miss_summary()
    assert s["scanner_gaps"] == 1
    assert "SMOO" in s["gap_symbols"]
    assert s["washouts"] == 0


def test_scanner_near_miss_flags_gappy_runner_as_washout() -> None:
    t0 = datetime(2026, 6, 12, 14, 0, tzinfo=timezone.utc)
    tracker = MissedAPlusTracker()
    # gappy ~6% bars (the ASBP profile)
    bars = [
        _bar(i, close=5.00, open_=4.95, high=5.15, low=4.85, volume=200_000, base_ts=t0, n=20)
        for i in range(20)
    ]
    tracker.record_scanner_near_miss(
        symbol="GAPY", reason="volatile HOD washout",
        universe={"GAPY": bars}, float_shares=8_000_000, now=t0,
    )
    runner = bars + [
        _bar(0, close=6.00, open_=5.50, high=6.20, low=5.40, volume=300_000,
             base_ts=t0 + timedelta(seconds=300), n=1)
    ]
    tracker.update_prices({"GAPY": runner}, now=t0 + timedelta(seconds=300))

    s = tracker.scanner_near_miss_summary()
    assert s["moved"] == 1
    assert s["washouts"] == 1          # ran, but gappy -> correctly ignored
    assert s["scanner_gaps"] == 0      # NOT a real miss


def test_scanner_near_miss_skips_high_float() -> None:
    t0 = datetime(2026, 6, 12, 14, 0, tzinfo=timezone.utc)
    tracker = MissedAPlusTracker()
    bars = [
        _bar(i, close=3.00, open_=2.995, high=3.01, low=2.99, volume=150_000, base_ts=t0, n=20)
        for i in range(20)
    ]
    rec = tracker.record_scanner_near_miss(
        symbol="BIG", reason="x", universe={"BIG": bars},
        float_shares=100_000_000, now=t0,
    )
    assert rec is None
