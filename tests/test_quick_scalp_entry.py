from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from daytrading.models import Bar, Quote, Side, Tick, Timeframe
from daytrading.runner import AlpacaRunner


TS = datetime(2026, 5, 29, 13, 30, tzinfo=timezone.utc)


class _QuickScalpRunner:
    _check_quick_scalp_entry = AlpacaRunner._check_quick_scalp_entry
    _quick_scalp_recent_normal_reject = AlpacaRunner._quick_scalp_recent_normal_reject
    _quick_scalp_has_tradeable_hod_alert = AlpacaRunner._quick_scalp_has_tradeable_hod_alert
    _quick_scalp_can_ignore_recent_shape_reject = staticmethod(
        AlpacaRunner._quick_scalp_can_ignore_recent_shape_reject
    )
    _quick_scalp_hod_alert_reject = AlpacaRunner._quick_scalp_hod_alert_reject
    _momentum_breakout_tape_is_smooth = AlpacaRunner._momentum_breakout_tape_is_smooth
    _momentum_breakout_consume = AlpacaRunner._momentum_breakout_consume
    _quick_scalp_shared_quality_reject = AlpacaRunner._quick_scalp_shared_quality_reject
    _shared_entry_quality_reject = AlpacaRunner._shared_entry_quality_reject
    _quick_scalp_10s_reject = AlpacaRunner._quick_scalp_10s_reject
    _quick_scalp_tick_rr = AlpacaRunner._quick_scalp_tick_rr
    _quick_scalp_allows_extreme_hod_runner_alert = staticmethod(
        AlpacaRunner._quick_scalp_allows_extreme_hod_runner_alert
    )

    def __init__(self) -> None:
        self._quote_buffer = defaultdict(lambda: deque(maxlen=100))
        self._tick_buffer = defaultdict(lambda: deque(maxlen=200))
        self._bar_aggregator = None
        self._float_checker = None
        self._hod_alert_store = None
        self._pipeline = SimpleNamespace(scan_rejections={})


def _bar(
    close: float,
    *,
    idx: int,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 75_000,
) -> Bar:
    return Bar(
        symbol="OLOX",
        ts=TS + timedelta(minutes=idx),
        open=close if open_ is None else open_,
        high=close + 0.10 if high is None else high,
        low=close - 0.10 if low is None else low,
        close=close,
        volume=volume,
        timeframe=Timeframe.MIN_1,
    )


def _add_clean_execution_context(runner: _QuickScalpRunner, symbol: str = "OLOX") -> None:
    for i in range(5):
        runner._quote_buffer[symbol].append(
            Quote(
                symbol=symbol,
                ts=TS + timedelta(minutes=12, seconds=i),
                bid=8.07,
                ask=8.11,
                bid_size=1500,
                ask_size=1300,
            )
        )
    for i in range(20):
        runner._tick_buffer[symbol].append(
            Tick(
                symbol=symbol,
                ts=TS + timedelta(minutes=12, seconds=i),
                price=8.00 + min(i, 9) * 0.01,
                size=1000,
                side=Side.BUY if i < 16 else Side.SELL,
            )
        )


def test_quick_scalp_allows_recent_hod_push_when_session_open_is_misleading() -> None:
    runner = _QuickScalpRunner()
    _add_clean_execution_context(runner)
    bars = [
        _bar(9.20, idx=0, open_=9.54, high=9.70, low=9.00, volume=40_000),
        _bar(6.40, idx=1, high=6.60, low=6.25, volume=90_000),
        _bar(6.65, idx=2, high=6.75, low=6.32, volume=95_000),
        _bar(6.55, idx=3, high=6.75, low=6.40, volume=70_000),
        _bar(6.85, idx=4, high=6.95, low=6.55, volume=85_000),
        _bar(6.70, idx=5, high=6.95, low=6.45, volume=75_000),
        _bar(6.92, idx=6, high=7.10, low=6.55, volume=90_000),
        _bar(7.95, idx=7, high=8.15, low=7.45, volume=130_000),
        _bar(8.25, idx=8, high=8.50, low=7.92, volume=140_000),
        _bar(8.09, idx=9, high=8.42, low=7.86, volume=150_000),
    ]

    assert runner._check_quick_scalp_entry("OLOX", bars) is None


def test_quick_scalp_still_rejects_weak_move_even_near_hod() -> None:
    runner = _QuickScalpRunner()
    _add_clean_execution_context(runner)
    bars = [
        _bar(8.30, idx=0, open_=10.00, high=8.55, low=8.10, volume=90_000),
        _bar(8.28, idx=1, high=8.45, low=8.05, volume=90_000),
        _bar(8.35, idx=2, high=8.50, low=8.10, volume=90_000),
        _bar(8.32, idx=3, high=8.48, low=8.05, volume=90_000),
        _bar(8.40, idx=4, high=8.56, low=8.15, volume=90_000),
        _bar(8.38, idx=5, high=8.54, low=8.12, volume=90_000),
    ]

    reject = runner._check_quick_scalp_entry("OLOX", bars)

    assert reject is not None
    assert "quick scalp movement too small day=" in reject
    assert "recent=" in reject


def test_quick_scalp_allows_sub_two_runner_to_reach_quality_checks() -> None:
    runner = _QuickScalpRunner()
    _add_clean_execution_context(runner, symbol="NEXR")
    bars = [
        _bar(1.35, idx=0, open_=1.20, high=1.38, low=1.18, volume=1_200_000),
        _bar(1.55, idx=1, high=1.58, low=1.32, volume=1_500_000),
        _bar(1.72, idx=2, high=1.76, low=1.50, volume=1_600_000),
        _bar(1.87, idx=3, high=1.88, low=1.70, volume=1_800_000),
    ]

    reject = runner._check_quick_scalp_entry("NEXR", bars)

    assert reject != "quick scalp price $1.87 outside range $1.50-$20.00"


def test_quick_scalp_still_rejects_below_dollar_fifty_runner() -> None:
    runner = _QuickScalpRunner()
    _add_clean_execution_context(runner, symbol="NEXR")
    bars = [
        _bar(1.25, idx=0, open_=1.20, high=1.30, low=1.18, volume=1_200_000),
        _bar(1.32, idx=1, high=1.34, low=1.22, volume=1_500_000),
        _bar(1.40, idx=2, high=1.43, low=1.30, volume=1_600_000),
        _bar(1.49, idx=3, high=1.50, low=1.38, volume=1_800_000),
    ]

    reject = runner._check_quick_scalp_entry("NEXR", bars)

    assert reject == "quick scalp price $1.49 outside range $1.50-$20.00"


def test_extreme_hod_runner_alert_allows_sub_two_candidate() -> None:
    row = {
        "price": 1.87,
        "change_session_pct": 92.0,
        "change_from_close_pct": 135.0,
        "day_volume": 8_000_000,
        "rel_vol": 1.2,
        "bar_rvol": 1.0,
        "float_shares": 4_000_000,
    }

    assert _QuickScalpRunner._quick_scalp_allows_extreme_hod_runner_alert(row) is True


def test_extreme_hod_runner_alert_still_rejects_below_dollar_fifty() -> None:
    row = {
        "price": 1.49,
        "change_session_pct": 120.0,
        "change_from_close_pct": 150.0,
        "day_volume": 8_000_000,
        "rel_vol": 1.2,
        "bar_rvol": 1.0,
        "float_shares": 4_000_000,
    }

    assert _QuickScalpRunner._quick_scalp_allows_extreme_hod_runner_alert(row) is False


def test_quick_scalp_respects_recent_hard_entry_guard_reject() -> None:
    runner = _QuickScalpRunner()
    runner._pipeline = SimpleNamespace(
        scan_rejections={
            "STAK": "spread too wide (3.00c = 0.69% of $4.35)",
        }
    )

    reject = runner._quick_scalp_recent_normal_reject("STAK")

    assert reject == "recent normal entry reject: spread too wide (3.00c = 0.69% of $4.35)"


def test_quick_scalp_ignores_stale_late_hod_reject_for_fresh_hod_alert() -> None:
    runner = _QuickScalpRunner()
    runner._pipeline = SimpleNamespace(
        scan_rejections={
            "CUPR": (
                "cached reject: late pullback too far from HOD 17.1% "
                "(max 12.0%; watching for fresh reclaim) (21s left)"
            ),
        }
    )
    runner._hod_alert_store = SimpleNamespace(
        snapshot=lambda: [{
            "symbol": "CUPR",
            "reject_reason": None,
            "price": 5.09,
            "day_volume": 22_000_000,
            "rel_vol": 1.3,
            "bar_rvol": 1.3,
        }]
    )

    reject = runner._quick_scalp_recent_normal_reject(
        "CUPR",
        allow_fresh_hod_breakout=True,
    )

    assert reject is None


def test_quick_scalp_still_respects_recent_spread_reject_on_hod_alert() -> None:
    runner = _QuickScalpRunner()
    runner._pipeline = SimpleNamespace(
        scan_rejections={
            "CUPR": "spread too wide (6.40c = 1.34% of $4.78)",
        }
    )
    runner._hod_alert_store = SimpleNamespace(
        snapshot=lambda: [{
            "symbol": "CUPR",
            "reject_reason": None,
            "price": 5.09,
            "day_volume": 22_000_000,
            "rel_vol": 1.3,
            "bar_rvol": 1.3,
        }]
    )

    reject = runner._quick_scalp_recent_normal_reject(
        "CUPR",
        allow_fresh_hod_breakout=True,
    )

    assert reject == "recent normal entry reject: spread too wide (6.40c = 1.34% of $4.78)"


def test_quick_scalp_rejects_watch_only_hod_alert() -> None:
    runner = _QuickScalpRunner()
    runner._hod_alert_store = SimpleNamespace(
        snapshot=lambda: [{
            "symbol": "SUNE",
            "reject_reason": "watch only: momentum_burst collecting data, not live A+ setup",
            "rel_vol": 0.71,
            "bar_rvol": 0.71,
        }]
    )

    reject = runner._quick_scalp_hod_alert_reject("SUNE")

    assert reject == (
        "HOD alert not tradeable: "
        "watch only: momentum_burst collecting data, not live A+ setup"
    )


def test_quick_scalp_promotes_extreme_momentum_burst_to_real_entry_gates() -> None:
    runner = _QuickScalpRunner()
    runner._hod_alert_store = SimpleNamespace(
        snapshot=lambda: [{
            "symbol": "AHMA",
            "reject_reason": "watch only: momentum_burst collecting data, not live A+ setup",
            "price": 2.90,
            "change_session_pct": 163.6,
            "change_from_close_pct": 168.5,
            "day_volume": 62_000_000,
            "float_shares": 2_100_000,
            "rel_vol": 4.2,
            "bar_rvol": 0.33,
        }]
    )

    assert runner._quick_scalp_hod_alert_reject("AHMA") is None


def test_quick_scalp_does_not_promote_weak_watch_only_momentum_burst() -> None:
    runner = _QuickScalpRunner()
    runner._hod_alert_store = SimpleNamespace(
        snapshot=lambda: [{
            "symbol": "SUNE",
            "reject_reason": "watch only: momentum_burst collecting data, not live A+ setup",
            "price": 4.00,
            "change_session_pct": 18.0,
            "change_from_close_pct": 20.0,
            "day_volume": 800_000,
            "float_shares": 5_000_000,
            "rel_vol": 1.2,
            "bar_rvol": 1.1,
        }]
    )

    reject = runner._quick_scalp_hod_alert_reject("SUNE")

    assert reject == (
        "HOD alert not tradeable: "
        "watch only: momentum_burst collecting data, not live A+ setup"
    )


def test_quick_scalp_rejects_weak_active_hod_rvol() -> None:
    runner = _QuickScalpRunner()
    runner._hod_alert_store = SimpleNamespace(
        snapshot=lambda: [{
            "symbol": "SUNE",
            "reject_reason": None,
            "rel_vol": 0.71,
            "bar_rvol": 0.71,
        }]
    )

    reject = runner._quick_scalp_hod_alert_reject("SUNE")

    assert reject == "HOD alert active RVOL too weak 0.71x (need 1.0x+)"


def test_quick_scalp_requires_10s_confirmation_feed() -> None:
    runner = _QuickScalpRunner()

    reject = runner._quick_scalp_10s_reject("OLOX")

    assert reject == "no 10s confirmation feed"


def test_quick_scalp_rejects_red_10s_confirmation() -> None:
    runner = _QuickScalpRunner()
    red_bar = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc),
        open=8.10,
        high=8.12,
        low=8.02,
        close=8.04,
        volume=22_000,
        timeframe=Timeframe.SEC_10,
    )
    runner._bar_aggregator = SimpleNamespace(
        get_latest_10s=lambda symbol, count=2: [red_bar],
    )

    reject = runner._quick_scalp_10s_reject("OLOX")

    assert reject == "10s confirmation red/flat"


def test_quick_scalp_allows_green_expanding_10s_confirmation() -> None:
    runner = _QuickScalpRunner()
    previous = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc) - timedelta(seconds=10),
        open=8.00,
        high=8.08,
        low=7.98,
        close=8.04,
        volume=18_000,
        timeframe=Timeframe.SEC_10,
    )
    latest = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc),
        open=8.04,
        high=8.16,
        low=8.03,
        close=8.14,
        volume=24_000,
        timeframe=Timeframe.SEC_10,
    )
    runner._bar_aggregator = SimpleNamespace(
        get_latest_10s=lambda symbol, count=2: [previous, latest],
    )

    assert runner._quick_scalp_10s_reject("OLOX") is None


def test_quick_scalp_allows_big_volume_runner_just_under_50k_recent_volume() -> None:
    runner = _QuickScalpRunner()
    _add_clean_execution_context(runner, symbol="IOTR")
    bars = [
        _bar(3.40, idx=0, open_=3.40, high=3.45, low=3.36, volume=900_000),
        _bar(3.70, idx=1, open_=3.40, high=3.75, low=3.38, volume=900_000),
        _bar(3.95, idx=2, open_=3.70, high=4.00, low=3.65, volume=900_000),
        _bar(4.10, idx=3, open_=3.95, high=4.19, low=3.90, volume=900_000),
        _bar(4.30, idx=4, open_=4.10, high=4.35, low=4.05, volume=16_000),
        _bar(4.60, idx=5, open_=4.30, high=4.62, low=4.25, volume=16_000),
        _bar(5.00, idx=6, open_=4.60, high=5.00, low=4.55, volume=16_263),
    ]

    assert runner._check_quick_scalp_entry("IOTR", bars) is None


def test_quick_scalp_tick_rr_uses_tactical_risk_for_hod_scalp() -> None:
    runner = _QuickScalpRunner()
    _add_clean_execution_context(runner)
    bars = [
        _bar(7.95, idx=0, high=8.15, low=7.45, volume=130_000),
        _bar(8.25, idx=1, high=8.50, low=7.92, volume=140_000),
        _bar(8.09, idx=2, high=8.42, low=7.86, volume=150_000),
    ]

    rr = runner._quick_scalp_tick_rr("OLOX", bars, alert_price=8.09)

    assert rr is not None
    price, stop, target, note = rr
    assert stop < price < target
    assert (price - stop) / price <= 0.04
    assert "risk=" in note


def test_momentum_breakout_off_rejects_low_rvol() -> None:
    runner = _QuickScalpRunner()
    runner._hod_alert_store = SimpleNamespace(
        snapshot=lambda: [{"symbol": "VSME", "rel_vol": 0.7, "bar_rvol": 0.5, "day_volume": 19_000_000}]
    )
    # mode OFF (default): low rvol still rejected
    r = runner._quick_scalp_hod_alert_reject("VSME")
    assert r is not None and "RVOL too weak" in r


def _smooth_bars(sym: str, close: float = 2.20) -> list:
    # ~1.4% per-bar range — tight tape where a stop holds
    return [
        Bar(symbol=sym, ts=TS, open=close - 0.005, high=close + 0.015,
            low=close - 0.015, close=close, volume=300_000)
        for _ in range(6)
    ]


def _gappy_bars(sym: str, close: float = 2.20) -> list:
    # ~13% per-bar range — violent tape where a stop slips (VSME-style)
    return [
        Bar(symbol=sym, ts=TS, open=close - 0.10, high=close + 0.15,
            low=close - 0.15, close=close, volume=600_000)
        for _ in range(6)
    ]


def _momentum_runner(bars: list, sym: str) -> _QuickScalpRunner:
    runner = _QuickScalpRunner()
    runner._momentum_breakout_enabled = True
    runner._momentum_breakout_min_rvol = 0.4
    runner._momentum_breakout_min_day_volume = 5_000_000
    runner._momentum_breakout_max_bar_range_pct = 3.0
    runner._momentum_breakout_score_floor = 72.0
    runner._momentum_breakout_armed = {}
    runner._bar_buffer = {sym: deque(bars)}
    return runner


def test_momentum_breakout_allows_high_vol_breakout_on_smooth_tape() -> None:
    runner = _momentum_runner(_smooth_bars("RUNR"), "RUNR")
    runner._hod_alert_store = SimpleNamespace(
        snapshot=lambda: [{"symbol": "RUNR", "rel_vol": 0.7, "bar_rvol": 0.5, "day_volume": 19_000_000}]
    )
    # high abs volume + faded rvol + SMOOTH tape -> allowed
    assert runner._quick_scalp_hod_alert_reject("RUNR") is None
    # ...and the entry is armed for tagging, consumable exactly once
    assert runner._momentum_breakout_consume("RUNR") is True
    assert runner._momentum_breakout_consume("RUNR") is False


def test_momentum_breakout_consume_false_when_not_armed() -> None:
    runner = _momentum_runner(_smooth_bars("RUNR"), "RUNR")
    assert runner._momentum_breakout_consume("RUNR") is False


def test_momentum_breakout_rejects_high_vol_breakout_on_gappy_tape() -> None:
    runner = _momentum_runner(_gappy_bars("VSME"), "VSME")
    runner._hod_alert_store = SimpleNamespace(
        snapshot=lambda: [{"symbol": "VSME", "rel_vol": 0.7, "bar_rvol": 0.5, "day_volume": 19_000_000}]
    )
    # same high abs volume, but GAPPY tape (stops slip) -> still rejected (VSME)
    r = runner._quick_scalp_hod_alert_reject("VSME")
    assert r is not None and "RVOL too weak" in r


def test_momentum_breakout_on_still_rejects_dead_tape() -> None:
    runner = _momentum_runner(_smooth_bars("DEAD"), "DEAD")
    runner._hod_alert_store = SimpleNamespace(
        snapshot=lambda: [{"symbol": "DEAD", "rel_vol": 0.7, "bar_rvol": 0.5, "day_volume": 800_000}]
    )
    # low absolute volume (dead tape) -> still rejected even if smooth
    r = runner._quick_scalp_hod_alert_reject("DEAD")
    assert r is not None and "RVOL too weak" in r


def test_momentum_breakout_score_floor_allows_sub80_on_smooth_tape() -> None:
    runner = _momentum_runner(_smooth_bars("RUNR"), "RUNR")
    runner._shared_entry_quality_reject = (
        lambda *a, **k: "entry score too low (75/100, need 80+) [day+78%=20]"
    )
    # score 75 >= floor 72 + smooth tape -> override allows it
    assert runner._quick_scalp_shared_quality_reject("RUNR", []) is None


def test_momentum_breakout_score_floor_keeps_reject_on_gappy_tape() -> None:
    runner = _momentum_runner(_gappy_bars("VSME"), "VSME")
    runner._shared_entry_quality_reject = (
        lambda *a, **k: "entry score too low (75/100, need 80+) [day+78%=20]"
    )
    # score 75 but GAPPY tape -> override does NOT apply, reject stands
    r = runner._quick_scalp_shared_quality_reject("VSME", [])
    assert r is not None and "entry score too low" in r


def test_momentum_breakout_score_floor_keeps_reject_below_floor() -> None:
    runner = _momentum_runner(_smooth_bars("RUNR"), "RUNR")
    runner._shared_entry_quality_reject = (
        lambda *a, **k: "entry score too low (68/100, need 80+) [day+78%=20]"
    )
    # score 68 < floor 72 -> reject stands even on smooth tape
    r = runner._quick_scalp_shared_quality_reject("RUNR", [])
    assert r is not None and "entry score too low" in r


def test_momentum_breakout_score_floor_off_keeps_reject() -> None:
    runner = _momentum_runner(_smooth_bars("RUNR"), "RUNR")
    runner._momentum_breakout_enabled = False
    runner._shared_entry_quality_reject = (
        lambda *a, **k: "entry score too low (75/100, need 80+) [day+78%=20]"
    )
    # mode OFF -> no override
    r = runner._quick_scalp_shared_quality_reject("RUNR", [])
    assert r is not None and "entry score too low" in r
