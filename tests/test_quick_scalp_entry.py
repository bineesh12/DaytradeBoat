from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import time
from types import SimpleNamespace

from daytrading.models import Bar, Fill, OrderStatus, PortfolioState, Position, Quote, ScanResult, Side, Tick, Timeframe
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
    _breakout_scalp_10s_reject = AlpacaRunner._breakout_scalp_10s_reject
    _quick_scalp_tick_rr = AlpacaRunner._quick_scalp_tick_rr
    _maybe_arm_momentum_burst_scalp = AlpacaRunner._maybe_arm_momentum_burst_scalp
    _warrior_squeeze_should_arm = AlpacaRunner._warrior_squeeze_should_arm
    _process_momentum_burst_scalps = AlpacaRunner._process_momentum_burst_scalps
    _latest_momentum_burst_10s_bar = AlpacaRunner._latest_momentum_burst_10s_bar
    _momentum_burst_recent_10s = AlpacaRunner._momentum_burst_recent_10s
    _momentum_burst_violent_liquid_ok = AlpacaRunner._momentum_burst_violent_liquid_ok
    _momentum_burst_stop_trading_reason = AlpacaRunner._momentum_burst_stop_trading_reason
    _momentum_burst_continuation_base_ok = AlpacaRunner._momentum_burst_continuation_base_ok
    _momentum_burst_level_context = staticmethod(AlpacaRunner._momentum_burst_level_context)
    _maybe_arm_warrior_squeeze_from_10s = AlpacaRunner._maybe_arm_warrior_squeeze_from_10s
    _warrior_squeeze_pullaway_context = AlpacaRunner._warrior_squeeze_pullaway_context
    _warrior_squeeze_equal_high_pullaway_context = (
        AlpacaRunner._warrior_squeeze_equal_high_pullaway_context
    )
    _warrior_squeeze_first_starter_has_proof_hold = (
        AlpacaRunner._warrior_squeeze_first_starter_has_proof_hold
    )
    _warrior_squeeze_curl_reclaim_context = AlpacaRunner._warrior_squeeze_curl_reclaim_context
    _momentum_burst_rebase_pending_after_reject = AlpacaRunner._momentum_burst_rebase_pending_after_reject
    _momentum_burst_hit_run_time_allowed = AlpacaRunner._momentum_burst_hit_run_time_allowed
    _record_momentum_burst_hit_run_pnl = AlpacaRunner._record_momentum_burst_hit_run_pnl
    _execute_momentum_burst_scalp = AlpacaRunner._execute_momentum_burst_scalp
    _capital_aware_quantity = AlpacaRunner._capital_aware_quantity
    _current_equity = AlpacaRunner._current_equity
    _new_entries_blocked = AlpacaRunner._new_entries_blocked
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
        self._risk_pct_of_equity = 0.015
        self._max_dollar_risk_per_trade = 50.0
        self._max_position_pct_of_equity = 1.0
        self._min_risk_dollars = 5.0
        self._fallback_equity = 25_000.0
        self._account_equity = 25_000.0
        self._account_equity_at = time.monotonic()
        self._momentum_burst_cycle_enabled = False
        self._momentum_burst_window_sec = 300.0
        self._momentum_burst_scalp_cooldown_sec = 300.0
        self._momentum_burst_armed = {}
        self._momentum_burst_window_high = {}
        self._momentum_burst_session_anchor_high = {}
        self._momentum_burst_pending = {}
        self._momentum_burst_hit_run_enabled = False
        self._momentum_burst_hit_run_max_entries = 1
        self._momentum_burst_hit_run_win_cooldown_sec = 15.0
        self._momentum_burst_hit_run_loss_cooldown_sec = 90.0
        self._momentum_burst_hit_run_max_hold_sec = 45.0
        self._momentum_burst_hit_run_reward_risk = 1.0
        self._momentum_burst_hit_run_end_et = "11:30"
        self._momentum_burst_hit_run_counts = {}
        self._momentum_burst_hit_run_block_until = {}
        self._momentum_burst_hit_run_stop_after_giveback = True
        self._momentum_burst_hit_run_max_giveback = 50.0
        self._momentum_burst_hit_run_daily_loss_stop = 50.0
        self._momentum_burst_hit_run_symbol_pnl = {}
        self._momentum_burst_hit_run_symbol_peak_pnl = {}
        self._momentum_burst_hit_run_day_blocked = {}
        self._warrior_squeeze_enabled = False
        self._warrior_squeeze_min_reclaim_price = 2.0
        self._warrior_squeeze_starter_size_factor = 0.35
        self._warrior_squeeze_rejection_high = {}
        self._warrior_squeeze_rejection_reason = {}
        self._warrior_squeeze_target_wins = {}
        self._breakout_scalp_active = False
        self._breakout_scalp_cooldown = {}
        self._quick_scalp_spread_size_factors = {}


def test_warrior_squeeze_live_a_plus_reclaim_stop_stays_inside_final_guard() -> None:
    runner = _QuickScalpRunner()
    runner._warrior_squeeze_enabled = True
    runner._warrior_squeeze_min_reclaim_price = 3.50
    runner._warrior_squeeze_rejection_high["MBUR"] = 4.00
    bars = [
        Bar("MBUR", TS, 7.90, 8.00, 7.86, 7.96, 120_000, Timeframe.SEC_10),
        Bar("MBUR", TS + timedelta(seconds=10), 8.05, 8.20, 8.00, 8.15, 180_000, Timeframe.SEC_10),
        Bar("MBUR", TS + timedelta(seconds=20), 8.55, 9.10, 8.35, 9.00, 220_000, Timeframe.SEC_10),
    ]
    runner._bar_aggregator = SimpleNamespace()
    runner._momentum_burst_recent_10s = lambda symbol, count=6: bars[-count:]
    pending = {
        "ts": bars[-2].ts,
        "breakout_close": 8.15,
        "breakout_high": 8.25,
        "breakout_volume": 180_000,
        "entry_trigger": "warrior_a_plus_reclaim",
    }

    context = runner._warrior_squeeze_pullaway_context("MBUR", bars[-1], pending)

    assert context is not None
    entry = context["entry_price_override"]
    stop = context["stop_price_override"]
    assert (entry - stop) / entry <= 0.06


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


def test_capital_aware_quantity_uses_equity_risk_without_explicit_cap() -> None:
    runner = _QuickScalpRunner()
    runner._account_equity = 95_000.0
    runner._account_equity_at = time.monotonic()
    runner._risk_pct_of_equity = 0.015
    runner._max_dollar_risk_per_trade = 50.0
    runner._max_position_pct_of_equity = 1.0
    runner._min_risk_dollars = 5.0

    qty = runner._capital_aware_quantity(1.91, 1.85)

    assert qty == 23750


def test_capital_aware_quantity_can_cap_burst_to_exit_dollar_risk_budget() -> None:
    runner = _QuickScalpRunner()
    runner._account_equity = 95_000.0
    runner._account_equity_at = time.monotonic()
    runner._risk_pct_of_equity = 0.015
    runner._max_position_pct_of_equity = 1.0
    runner._min_risk_dollars = 5.0

    qty = runner._capital_aware_quantity(1.91, 1.85, max_dollar_risk=50.0)

    assert qty == 833
    assert qty * (1.91 - 1.85) <= 50.0


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


def test_breakout_scalp_requires_clean_reclaim_to_ignore_stale_late_reject() -> None:
    runner = _QuickScalpRunner()
    runner._pipeline = SimpleNamespace(
        scan_rejections={
            "CRVO": (
                "cached reject: late pullback too far from HOD 10.7% "
                "(max 10.0%; watching for fresh reclaim) (26s left)"
            ),
        }
    )
    runner._hod_alert_store = SimpleNamespace(
        snapshot=lambda: [{
            "symbol": "CRVO",
            "reject_reason": None,
            "price": 5.16,
            "day_volume": 93_000_000,
            "rel_vol": 1.3,
            "bar_rvol": 1.3,
        }]
    )
    runner._momentum_burst_continuation_base_ok = (
        lambda symbol: (False, "volume faded", {})
    )

    reject = runner._quick_scalp_recent_normal_reject(
        "CRVO",
        allow_fresh_hod_breakout=True,
        require_clean_reclaim=True,
    )

    assert reject == (
        "fresh HOD breakout needs clean 10s reclaim after recent reject: volume faded"
    )


def test_breakout_scalp_can_clear_stale_reject_with_clean_reclaim() -> None:
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
    runner._momentum_burst_continuation_base_ok = (
        lambda symbol: (True, "fresh continuation base", {})
    )

    reject = runner._quick_scalp_recent_normal_reject(
        "CUPR",
        allow_fresh_hod_breakout=True,
        require_clean_reclaim=True,
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


def test_breakout_scalp_rejects_weak_close_10s_confirmation() -> None:
    runner = _QuickScalpRunner()
    previous = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc) - timedelta(seconds=10),
        open=8.00,
        high=8.08,
        low=7.98,
        close=8.04,
        volume=24_000,
        timeframe=Timeframe.SEC_10,
    )
    latest = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc),
        open=8.04,
        high=8.20,
        low=8.00,
        close=8.08,
        volume=25_000,
        timeframe=Timeframe.SEC_10,
    )
    runner._bar_aggregator = SimpleNamespace(
        get_latest_10s=lambda symbol, count=2: [previous, latest],
    )

    reject = runner._breakout_scalp_10s_reject("OLOX")

    assert reject == "10s confirmation weak close (40% location)"


def test_breakout_scalp_rejects_faded_10s_confirmation_volume() -> None:
    runner = _QuickScalpRunner()
    previous = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc) - timedelta(seconds=10),
        open=8.00,
        high=8.08,
        low=7.98,
        close=8.04,
        volume=60_000,
        timeframe=Timeframe.SEC_10,
    )
    latest = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc),
        open=8.04,
        high=8.16,
        low=8.03,
        close=8.14,
        volume=20_000,
        timeframe=Timeframe.SEC_10,
    )
    runner._bar_aggregator = SimpleNamespace(
        get_latest_10s=lambda symbol, count=2: [previous, latest],
    )

    reject = runner._breakout_scalp_10s_reject("OLOX")

    assert reject == "10s confirmation volume faded 20000 < 50% prior 60000"


def test_breakout_scalp_rejects_violent_confirmation_without_high_close() -> None:
    runner = _QuickScalpRunner()
    previous = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc) - timedelta(seconds=10),
        open=9.40,
        high=9.70,
        low=9.30,
        close=9.55,
        volume=130_000,
        timeframe=Timeframe.SEC_10,
    )
    latest = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc),
        open=9.50,
        high=11.34,
        low=9.33,
        close=10.69,
        volume=291_000,
        timeframe=Timeframe.SEC_10,
    )
    runner._bar_aggregator = SimpleNamespace(
        get_latest_10s=lambda symbol, count=4: [previous, latest],
    )

    reject = runner._breakout_scalp_10s_reject("OLOX")

    assert reject == "10s breakout candle too volatile without strong close (68% location, 18.8% range)"


def test_breakout_scalp_rejects_recent_dump_before_breakout() -> None:
    runner = _QuickScalpRunner()
    older = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc) - timedelta(seconds=30),
        open=2.32,
        high=2.38,
        low=2.28,
        close=2.35,
        volume=90_000,
        timeframe=Timeframe.SEC_10,
    )
    dump = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc) - timedelta(seconds=20),
        open=2.29,
        high=2.32,
        low=2.21,
        close=2.22,
        volume=175_000,
        timeframe=Timeframe.SEC_10,
    )
    previous = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc) - timedelta(seconds=10),
        open=2.22,
        high=2.48,
        low=2.20,
        close=2.47,
        volume=130_000,
        timeframe=Timeframe.SEC_10,
    )
    latest = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc),
        open=2.47,
        high=2.62,
        low=2.50,
        close=2.60,
        volume=180_000,
        timeframe=Timeframe.SEC_10,
    )
    runner._bar_aggregator = SimpleNamespace(
        get_latest_10s=lambda symbol, count=4: [older, dump, previous, latest],
    )

    reject = runner._breakout_scalp_10s_reject("OLOX")

    assert reject == "recent 10s dump candle before breakout (3.1% body, 9% close location)"


def test_breakout_scalp_allows_wide_candle_with_high_close_and_no_prior_dump() -> None:
    runner = _QuickScalpRunner()
    older = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc) - timedelta(seconds=30),
        open=4.20,
        high=4.42,
        low=4.15,
        close=4.37,
        volume=180_000,
        timeframe=Timeframe.SEC_10,
    )
    previous = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc) - timedelta(seconds=10),
        open=4.37,
        high=4.80,
        low=4.30,
        close=4.73,
        volume=450_000,
        timeframe=Timeframe.SEC_10,
    )
    latest = Bar(
        symbol="OLOX",
        ts=datetime.now(timezone.utc),
        open=4.73,
        high=5.17,
        low=4.61,
        close=5.17,
        volume=725_000,
        timeframe=Timeframe.SEC_10,
    )
    runner._bar_aggregator = SimpleNamespace(
        get_latest_10s=lambda symbol, count=4: [older, previous, latest],
    )

    assert runner._breakout_scalp_10s_reject("OLOX") is None


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


def _momentum_burst_hit(symbol: str = "MBUR", high: float = 2.40) -> ScanResult:
    bars = [
        Bar(
            symbol=symbol,
            ts=datetime.now(timezone.utc) - timedelta(minutes=idx),
            open=high - 0.05,
            high=high,
            low=high - 0.10,
            close=high - 0.02,
            volume=250_000,
            timeframe=Timeframe.MIN_1,
        )
        for idx in range(3, 0, -1)
    ]
    return ScanResult(
        symbol=symbol,
        scanner_name="momentum_burst",
        ts=bars[-1].ts,
        score=4.2,
        criteria={"pattern": "momentum_burst", "setup_tier": "watch only", "close": high - 0.02},
        bars=bars,
    )


def test_momentum_burst_cycle_arms_only_when_enabled() -> None:
    runner = _QuickScalpRunner()
    hit = _momentum_burst_hit()

    runner._maybe_arm_momentum_burst_scalp(hit)
    assert runner._momentum_burst_armed == {}

    runner._momentum_burst_cycle_enabled = True
    runner._maybe_arm_momentum_burst_scalp(hit)

    assert "MBUR" in runner._momentum_burst_armed
    assert runner._momentum_burst_window_high["MBUR"] == 2.40


def test_momentum_burst_hit_run_arms_when_hit_run_enabled() -> None:
    runner = _QuickScalpRunner()
    hit = _momentum_burst_hit()

    runner._momentum_burst_hit_run_enabled = True
    runner._maybe_arm_momentum_burst_scalp(hit)

    assert "MBUR" in runner._momentum_burst_armed
    assert runner._momentum_burst_window_high["MBUR"] == 2.40


def test_warrior_squeeze_off_preserves_normal_hit_run_arming() -> None:
    runner = _QuickScalpRunner()
    runner._momentum_burst_hit_run_enabled = True
    runner._warrior_squeeze_enabled = False
    cheap_hit = _momentum_burst_hit(high=1.70)

    runner._maybe_arm_momentum_burst_scalp(cheap_hit)

    assert "MBUR" in runner._momentum_burst_armed
    assert runner._momentum_burst_window_high["MBUR"] == 1.70
    assert runner._warrior_squeeze_rejection_high == {}


def test_warrior_squeeze_ignores_first_cheap_spike_then_arms_reclaim() -> None:
    runner = _QuickScalpRunner()
    runner._warrior_squeeze_enabled = True
    runner._warrior_squeeze_min_reclaim_price = 2.0
    cheap_hit = _momentum_burst_hit(high=1.70)

    runner._maybe_arm_momentum_burst_scalp(cheap_hit)

    assert runner._momentum_burst_armed == {}
    assert runner._warrior_squeeze_rejection_high["MBUR"] == 1.70

    reclaim_hit = _momentum_burst_hit(high=2.08)
    runner._maybe_arm_momentum_burst_scalp(reclaim_hit)

    assert "MBUR" in runner._momentum_burst_armed
    assert runner._momentum_burst_window_high["MBUR"] == 2.08


def test_warrior_squeeze_rejects_first_high_volume_shooting_star() -> None:
    runner = _QuickScalpRunner()
    runner._warrior_squeeze_enabled = True
    runner._warrior_squeeze_min_reclaim_price = 2.0
    base_hit = _momentum_burst_hit(high=2.60)
    bars = [
        Bar(
            symbol=bar.symbol,
            ts=bar.ts,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=60_000,
            timeframe=bar.timeframe,
        )
        for bar in base_hit.bars[:-1]
    ] + [
        Bar(
            symbol="MBUR",
            ts=base_hit.bars[-1].ts,
            open=2.35,
            high=2.60,
            low=2.30,
            close=2.34,
            volume=220_000,
            timeframe=Timeframe.MIN_1,
        )
    ]
    hit = ScanResult(
        symbol="MBUR",
        scanner_name="momentum_burst",
        ts=bars[-1].ts,
        score=base_hit.score,
        criteria=base_hit.criteria,
        bars=bars,
    )

    runner._maybe_arm_momentum_burst_scalp(hit)

    assert runner._momentum_burst_armed == {}
    assert runner._warrior_squeeze_rejection_high["MBUR"] == 2.60
    assert runner._warrior_squeeze_rejection_reason["MBUR"] == "high-volume shooting-star rejection"


def test_momentum_burst_cycle_expires_window() -> None:
    runner = _QuickScalpRunner()
    runner._momentum_burst_cycle_enabled = True
    runner._momentum_burst_window_sec = 1.0
    runner._momentum_burst_armed = {"MBUR": time.monotonic() - 2.0}
    runner._momentum_burst_window_high = {"MBUR": 2.40}
    runner._hub = SimpleNamespace(trading_paused=False)
    runner._pipeline = SimpleNamespace(_circuit_breaker_tripped=False, _daily_pnl=0.0)

    runner._process_momentum_burst_scalps()

    assert runner._momentum_burst_armed == {}
    assert runner._momentum_burst_window_high == {}


class _Bars10:
    def __init__(self, bars):
        self._bars = list(bars)

    def append(self, bar):
        self._bars.append(bar)

    def get_latest_10s(self, symbol, count=2):
        return self._bars[-count:]


class _FillBroker:
    def submit(self, order, bar, portfolio):
        return (
            Fill(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.limit_price,
                ts=datetime.now(timezone.utc),
            ),
            OrderStatus.FILLED,
        )


def _momentum_burst_trade_runner() -> _QuickScalpRunner:
    runner = _QuickScalpRunner()
    sym = "MBUR"
    now = datetime.now(timezone.utc)
    runner._momentum_burst_cycle_enabled = True
    runner._momentum_burst_hit_run_end_et = ""
    runner._momentum_burst_window_sec = 300.0
    runner._momentum_burst_scalp_cooldown_sec = 120.0
    runner._momentum_burst_armed = {sym: time.monotonic()}
    runner._momentum_burst_window_high = {sym: 2.40}
    runner._test_now = now
    runner._bar_buffer = {
        sym: deque([
            Bar(sym, now - timedelta(minutes=4 - idx), 1.80 + idx * 0.15, 1.90 + idx * 0.15,
                1.75 + idx * 0.15, 1.85 + idx * 0.15, 250_000, Timeframe.MIN_1)
            for idx in range(5)
        ])
    }
    runner._bar_aggregator = _Bars10([
        Bar(sym, now - timedelta(seconds=10), 2.38, 2.40, 2.36, 2.39, 75_000, Timeframe.SEC_10),
        Bar(sym, now, 2.39, 2.55, 2.38, 2.52, 125_000, Timeframe.SEC_10),
    ])
    for i in range(12):
        runner._tick_buffer[sym].append(
            Tick(
                symbol=sym,
                ts=now - timedelta(seconds=11 - i),
                price=2.48 + i * 0.005,
                size=1000,
                side=Side.BUY,
            )
        )
    runner._pipeline = SimpleNamespace(
        scan_rejections={},
        portfolio=PortfolioState(cash=10_000),
        _exit_cooldowns={},
        _cooldown_seconds=120,
        _symbol_entry_counts={},
        _max_entries_per_symbol=2,
        _circuit_breaker_tripped=False,
        _daily_pnl=0.0,
    )
    runner._broker = _FillBroker()
    runner._hub = SimpleNamespace(
        trading_paused=False,
        on_fill=lambda *args, **kwargs: None,
        add_log=lambda *args, **kwargs: None,
    )
    runner._journal = SimpleNamespace(record=lambda *args, **kwargs: None)
    runner._shared_entry_quality_reject = lambda *args, **kwargs: None
    runner._record_entry_reject = lambda *args, **kwargs: None
    runner._on_position_opened = lambda *args, **kwargs: setattr(runner, "_opened", args)
    runner._seed_recent_order_ids = lambda: None
    runner._market_phase = lambda: "TEST"
    return runner


def test_momentum_burst_cycle_arms_then_fires_on_confirmation_bar() -> None:
    runner = _momentum_burst_trade_runner()
    now = runner._test_now

    # First pass sees the fresh-high spike bar — it must NOT buy it, only arm a
    # pending breakout awaiting the next 10s bar.
    runner._process_momentum_burst_scalps()
    assert runner._breakout_scalp_active is False
    assert "MBUR" in runner._momentum_burst_pending
    assert not getattr(runner, "_opened", None)

    # Next 10s bar confirms the breakout with continuation and a strong close.
    runner._bar_aggregator.append(
        Bar("MBUR", now + timedelta(seconds=10), 2.53, 2.62, 2.52, 2.59, 130_000, Timeframe.SEC_10)
    )
    runner._process_momentum_burst_scalps()

    assert runner._breakout_scalp_active is True
    assert runner._momentum_burst_armed == {}
    assert runner._pipeline._symbol_entry_counts["MBUR"] == 1
    signal = runner._opened[0]
    assert signal.scan_result.scanner_name == "momentum_burst_scalp"
    assert signal.scan_result.criteria["entry_mode"] == "momentum_burst_scalp"
    assert runner._breakout_scalp_cooldown["MBUR"] > time.monotonic()


def test_momentum_burst_hit_run_keeps_window_after_confirmed_entry() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_cycle_enabled = False
    runner._momentum_burst_hit_run_enabled = True
    now = runner._test_now

    runner._process_momentum_burst_scalps()
    runner._bar_aggregator.append(
        Bar("MBUR", now + timedelta(seconds=10), 2.53, 2.62, 2.52, 2.59, 130_000, Timeframe.SEC_10)
    )
    runner._process_momentum_burst_scalps()

    assert runner._breakout_scalp_active is True
    assert "MBUR" in runner._momentum_burst_armed
    assert runner._momentum_burst_hit_run_counts["MBUR"] == 1
    signal = runner._opened[0]
    assert signal.max_hold_seconds == 45.0
    assert signal.scan_result.scanner_name == "momentum_burst_hit_run"
    assert signal.scan_result.criteria["entry_mode"] == "momentum_burst_hit_run"
    assert signal.take_profit == round(
        signal.entry_price + (signal.entry_price - signal.stop_loss),
        2,
    )


def test_warrior_squeeze_entry_is_tagged_and_reduced_starter_size() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_cycle_enabled = False
    runner._warrior_squeeze_enabled = True
    runner._warrior_squeeze_starter_size_factor = 0.35
    runner._warrior_squeeze_min_reclaim_price = 3.50
    runner._warrior_squeeze_rejection_high["MBUR"] = 2.25
    runner._warrior_squeeze_rejection_reason["MBUR"] = "high-volume shooting-star rejection"
    now = runner._test_now

    runner._momentum_burst_window_high["MBUR"] = 3.50
    runner._momentum_burst_pending = {
        "MBUR": {
            "ts": now - timedelta(seconds=20),
            "breakout_close": 3.16,
            "breakout_high": 3.50,
            "breakout_volume": 760_000,
            "entry_trigger": "warrior_a_plus_reclaim",
        }
    }
    runner._bar_aggregator = _Bars10([
        Bar("MBUR", now - timedelta(seconds=10), 2.70, 3.50, 2.70, 3.16, 760_000, Timeframe.SEC_10),
        Bar("MBUR", now, 3.17, 4.08, 3.10, 3.9674, 845_000, Timeframe.SEC_10),
    ])
    runner._process_momentum_burst_scalps()

    assert runner._breakout_scalp_active is True
    assert runner._momentum_burst_hit_run_counts["MBUR"] == 1
    signal = runner._opened[0]
    assert signal.scan_result.scanner_name == "warrior_squeeze_playbook"
    assert signal.scan_result.criteria["entry_mode"] == "warrior_squeeze_playbook"
    assert signal.scan_result.criteria["variant"] == "warrior_clwt_fast_pullaway"
    assert signal.scan_result.criteria["size_factor"] == 0.35
    assert signal.quantity < 750
    assert signal.max_hold_seconds == 180.0


def test_warrior_squeeze_size_factor_does_not_stack_with_violent_hit_run() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_cycle_enabled = False
    runner._momentum_burst_hit_run_enabled = True
    runner._warrior_squeeze_enabled = True
    runner._warrior_squeeze_starter_size_factor = 0.35
    runner._warrior_squeeze_min_reclaim_price = 3.50
    runner._warrior_squeeze_rejection_high["MBUR"] = 2.25
    runner._warrior_squeeze_rejection_reason["MBUR"] = "high-volume shooting-star rejection"
    runner._momentum_burst_violent_liquid_ok = lambda symbol: (
        True,
        {"median_10s_range_pct": 4.0},
    )
    now = runner._test_now

    runner._momentum_burst_window_high["MBUR"] = 3.50
    runner._momentum_burst_pending = {
        "MBUR": {
            "ts": now - timedelta(seconds=20),
            "breakout_close": 3.16,
            "breakout_high": 3.50,
            "breakout_volume": 760_000,
            "entry_trigger": "warrior_a_plus_reclaim",
        }
    }
    runner._bar_aggregator = _Bars10([
        Bar("MBUR", now - timedelta(seconds=10), 2.70, 3.50, 2.70, 3.16, 760_000, Timeframe.SEC_10),
        Bar("MBUR", now, 3.17, 4.08, 3.10, 3.9674, 845_000, Timeframe.SEC_10),
    ])
    runner._process_momentum_burst_scalps()

    signal = runner._opened[0]
    assert signal.scan_result.scanner_name == "warrior_squeeze_playbook"
    assert signal.scan_result.criteria["size_factor"] == 0.35
    # If violent-liquid and warrior both applied 0.35, this would be about 12%.
    assert signal.quantity > 70


def test_warrior_squeeze_pullaway_starter_buys_capped_reclaim_level() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_cycle_enabled = False
    runner._warrior_squeeze_enabled = True
    runner._warrior_squeeze_min_reclaim_price = 3.50
    runner._warrior_squeeze_starter_size_factor = 0.35
    runner._warrior_squeeze_rejection_high["MBUR"] = 2.25
    now = runner._test_now
    runner._momentum_burst_window_high = {"MBUR": 3.50}
    runner._momentum_burst_pending = {
        "MBUR": {
            "ts": now - timedelta(seconds=10),
            "breakout_close": 3.16,
            "breakout_high": 3.50,
            "breakout_volume": 760_000,
        }
    }
    runner._bar_aggregator = _Bars10([
        Bar("MBUR", now - timedelta(seconds=30), 3.46, 3.56, 3.45, 3.53, 260_000, Timeframe.SEC_10),
        Bar("MBUR", now - timedelta(seconds=20), 3.54, 3.62, 3.49, 3.58, 320_000, Timeframe.SEC_10),
        Bar("MBUR", now - timedelta(seconds=10), 3.58, 3.70, 3.50, 3.64, 760_000, Timeframe.SEC_10),
        Bar("MBUR", now, 3.65, 4.08, 3.56, 3.97, 845_000, Timeframe.SEC_10),
    ])

    runner._process_momentum_burst_scalps()

    signal = runner._opened[0]
    assert signal.scan_result.scanner_name == "warrior_squeeze_playbook"
    assert signal.scan_result.criteria["variant"] == "warrior_proof_pullback_hold"
    assert signal.scan_result.criteria["entry_trigger"] == "warrior_level_pullaway"
    assert signal.scan_result.criteria["pullaway_level"] == 3.5
    assert signal.scan_result.criteria["max_pay"] == 3.71
    assert signal.entry_price == 3.71
    assert signal.entry_price < 3.97
    assert signal.max_hold_seconds == 180.0
    assert signal.quantity < 100


def test_warrior_squeeze_clwt_fast_pullaway_does_not_need_slow_proof_hold() -> None:
    runner = _momentum_burst_trade_runner()
    runner._warrior_squeeze_enabled = True
    runner._warrior_squeeze_min_reclaim_price = 3.50
    runner._warrior_squeeze_rejection_high["MBUR"] = 2.25
    runner._warrior_squeeze_rejection_reason["MBUR"] = "high-volume shooting-star rejection"
    now = runner._test_now
    bars = [
        Bar("MBUR", now - timedelta(seconds=20), 2.70, 3.50, 2.70, 3.16, 760_000, Timeframe.SEC_10),
        Bar("MBUR", now - timedelta(seconds=10), 3.17, 4.08, 3.10, 3.9674, 845_000, Timeframe.SEC_10),
    ]
    runner._bar_aggregator = _Bars10(bars)
    pending = {
        "ts": now - timedelta(seconds=20),
        "breakout_close": 3.16,
        "breakout_high": 3.50,
        "breakout_volume": 760_000,
        "entry_trigger": "warrior_a_plus_reclaim",
    }

    context = runner._warrior_squeeze_pullaway_context("MBUR", bars[-1], pending)

    assert context is not None
    assert context["entry_trigger"] == "warrior_level_pullaway"
    assert context["variant_override"] == "warrior_clwt_fast_pullaway"
    assert context["pullaway_level"] == 3.5
    assert context["max_pay"] == 4.025
    assert context["entry_price_override"] == 3.9674
    assert context["target_price_override"] > context["entry_price_override"]


def test_warrior_squeeze_rejects_high_price_generic_pullaway_starter() -> None:
    runner = _momentum_burst_trade_runner()
    runner._warrior_squeeze_enabled = True
    runner._warrior_squeeze_min_reclaim_price = 3.50
    runner._warrior_squeeze_rejection_high["PIII"] = 6.20
    runner._warrior_squeeze_rejection_reason["PIII"] = "first explosive 10s spike"
    now = runner._test_now
    bars = [
        Bar("PIII", now - timedelta(seconds=30), 6.80, 7.20, 6.70, 7.05, 260_000, Timeframe.SEC_10),
        Bar("PIII", now - timedelta(seconds=20), 7.05, 7.35, 6.95, 7.28, 310_000, Timeframe.SEC_10),
        Bar("PIII", now - timedelta(seconds=10), 7.30, 7.75, 7.25, 7.62, 380_000, Timeframe.SEC_10),
    ]
    runner._bar_aggregator = _Bars10(bars)
    pending = {
        "ts": now - timedelta(seconds=20),
        "breakout_close": 7.05,
        "breakout_high": 7.20,
        "breakout_volume": 260_000,
        "entry_trigger": "warrior_a_plus_reclaim",
    }

    context = runner._warrior_squeeze_pullaway_context("PIII", bars[-1], pending)

    assert context is None


def test_warrior_squeeze_equal_high_pullaway_allows_clwt_style_level_hold() -> None:
    runner = _momentum_burst_trade_runner()
    runner._warrior_squeeze_enabled = True
    runner._warrior_squeeze_min_reclaim_price = 3.50
    runner._warrior_squeeze_rejection_high["MBUR"] = 2.25
    runner._warrior_squeeze_rejection_reason["MBUR"] = "high-volume shooting-star rejection"
    now = runner._test_now
    bars = [
        Bar("MBUR", now - timedelta(seconds=40), 3.42, 3.50, 3.36, 3.47, 180_000, Timeframe.SEC_10),
        Bar("MBUR", now - timedelta(seconds=30), 3.48, 3.54, 3.42, 3.52, 230_000, Timeframe.SEC_10),
        Bar("MBUR", now - timedelta(seconds=20), 3.51, 3.58, 3.45, 3.55, 260_000, Timeframe.SEC_10),
        Bar("MBUR", now - timedelta(seconds=10), 3.54, 3.59, 3.49, 3.57, 280_000, Timeframe.SEC_10),
        Bar("MBUR", now, 3.56, 3.60, 3.50, 3.59, 310_000, Timeframe.SEC_10),
    ]
    runner._bar_aggregator = _Bars10(bars)
    pending = {
        "ts": now,
        "breakout_close": 3.57,
        "breakout_high": 3.60,
        "breakout_volume": 280_000,
    }

    context = runner._warrior_squeeze_equal_high_pullaway_context(
        "MBUR",
        bars[-1],
        pending,
        window_high=3.60,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_equal_high_pullaway"
    assert context["variant_override"] == "warrior_equal_high_pullaway"
    assert context["pullaway_level"] == 3.5
    assert context["entry_price_override"] == 3.59
    assert context["target_price_override"] > context["entry_price_override"]


def test_warrior_squeeze_curl_reclaim_rejects_high_price_first_starter() -> None:
    runner = _momentum_burst_trade_runner()
    runner._warrior_squeeze_enabled = True
    runner._warrior_squeeze_min_reclaim_price = 3.50
    now = runner._test_now
    bars = [
        Bar("PIII", now - timedelta(seconds=20), 7.10, 7.35, 7.05, 7.28, 260_000, Timeframe.SEC_10),
        Bar("PIII", now - timedelta(seconds=10), 7.32, 7.72, 7.30, 7.65, 360_000, Timeframe.SEC_10),
    ]
    runner._bar_aggregator = _Bars10(bars)
    pending = {
        "ts": now - timedelta(seconds=20),
        "breakout_close": 7.28,
        "breakout_high": 7.35,
        "breakout_volume": 260_000,
    }

    context = runner._warrior_squeeze_curl_reclaim_context(
        "PIII",
        bars[-1],
        pending,
        window_high=7.35,
    )

    assert context is None


def test_warrior_squeeze_equal_high_pullaway_rejects_topping_tail() -> None:
    runner = _momentum_burst_trade_runner()
    runner._warrior_squeeze_enabled = True
    runner._warrior_squeeze_min_reclaim_price = 3.50
    runner._warrior_squeeze_rejection_high["MBUR"] = 2.25
    runner._warrior_squeeze_rejection_reason["MBUR"] = "high-volume shooting-star rejection"
    now = runner._test_now
    bars = [
        Bar("MBUR", now - timedelta(seconds=30), 3.48, 3.54, 3.42, 3.52, 230_000, Timeframe.SEC_10),
        Bar("MBUR", now - timedelta(seconds=20), 3.51, 3.58, 3.45, 3.55, 260_000, Timeframe.SEC_10),
        Bar("MBUR", now - timedelta(seconds=10), 3.54, 3.59, 3.49, 3.57, 280_000, Timeframe.SEC_10),
        Bar("MBUR", now, 3.58, 3.82, 3.50, 3.60, 410_000, Timeframe.SEC_10),
    ]
    runner._bar_aggregator = _Bars10(bars)

    context = runner._warrior_squeeze_equal_high_pullaway_context(
        "MBUR",
        bars[-1],
        {"breakout_close": 3.57, "breakout_high": 3.60, "breakout_volume": 280_000},
        window_high=3.82,
    )

    assert context is None


def test_momentum_burst_hit_run_max_entries_blocks_extra_entries() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_cycle_enabled = False
    runner._momentum_burst_hit_run_enabled = True
    runner._momentum_burst_hit_run_counts = {"MBUR": 3}

    runner._process_momentum_burst_scalps()

    assert "MBUR" not in runner._momentum_burst_pending
    assert not getattr(runner, "_opened", None)


def test_momentum_burst_hit_run_giveback_blocks_symbol_for_day() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_hit_run_max_giveback = 25.0

    assert runner._record_momentum_burst_hit_run_pnl("MBUR", 60.0) is None
    reason = runner._record_momentum_burst_hit_run_pnl("MBUR", -30.0)

    assert "gave back" in reason
    assert "MBUR" in runner._momentum_burst_hit_run_day_blocked


def test_momentum_burst_hit_run_daily_loss_blocks_symbol_for_day() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_hit_run_daily_loss_stop = 20.0

    reason = runner._record_momentum_burst_hit_run_pnl("MBUR", -22.0)

    assert "daily hit-run loss" in reason
    assert "MBUR" in runner._momentum_burst_hit_run_day_blocked


def test_momentum_burst_hit_run_daily_loss_still_blocks_when_giveback_disabled() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_hit_run_stop_after_giveback = False
    runner._momentum_burst_hit_run_daily_loss_stop = 20.0

    reason = runner._record_momentum_burst_hit_run_pnl("MBUR", -22.0)

    assert "daily hit-run loss" in reason
    assert "MBUR" in runner._momentum_burst_hit_run_day_blocked


def test_momentum_burst_hit_run_giveback_can_be_disabled_independently() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_hit_run_stop_after_giveback = False
    runner._momentum_burst_hit_run_max_giveback = 25.0

    assert runner._record_momentum_burst_hit_run_pnl("MBUR", 60.0) is None
    assert runner._record_momentum_burst_hit_run_pnl("MBUR", -30.0) is None
    assert "MBUR" not in runner._momentum_burst_hit_run_day_blocked


def test_momentum_burst_hit_run_reentry_uses_normal_chase_cap() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_cycle_enabled = False
    runner._momentum_burst_hit_run_enabled = True
    runner._momentum_burst_hit_run_counts = {"MBUR": 1}
    runner._momentum_burst_hit_run_max_entries = 3
    runner._momentum_burst_continuation_base_ok = lambda symbol: (True, "fresh continuation base", {})
    runner._momentum_burst_violent_liquid_ok = lambda symbol: (True, {})
    now = runner._test_now
    runner._momentum_burst_pending = {
        "MBUR": {
            "ts": now - timedelta(seconds=10),
            "breakout_close": 2.40,
            "breakout_high": 2.45,
            "breakout_volume": 100_000,
        }
    }

    # This would pass the first-entry violent cap (8%), but as a re-entry it is
    # a 5% chase from the breakout close and must reject under the normal 3% cap.
    runner._process_momentum_burst_scalps()

    assert runner._breakout_scalp_active is False
    assert not getattr(runner, "_opened", None)
    assert "MBUR" not in runner._momentum_burst_pending


def test_momentum_burst_cycle_skips_unconfirmed_reversal() -> None:
    # The CUPR failure mode: spike bar makes a new high, next bar reverses red.
    # The confirmation rule must skip it (no entry).
    runner = _momentum_burst_trade_runner()
    now = runner._test_now

    runner._process_momentum_burst_scalps()
    assert "MBUR" in runner._momentum_burst_pending

    # Next bar reverses (red, closes below the breakout close) — not confirmed.
    runner._bar_aggregator.append(
        Bar("MBUR", now + timedelta(seconds=10), 2.51, 2.53, 2.30, 2.34, 140_000, Timeframe.SEC_10)
    )
    runner._process_momentum_burst_scalps()

    assert runner._breakout_scalp_active is False
    assert not getattr(runner, "_opened", None)
    assert "MBUR" not in runner._momentum_burst_pending


def test_momentum_burst_hit_run_requires_continuation_high() -> None:
    # CAST 04:13-style failure: spike bar makes the high, the next bar is green
    # and holds the breakout close, but it does not trade above the spike high.
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_cycle_enabled = False
    runner._momentum_burst_hit_run_enabled = True
    now = runner._test_now

    runner._process_momentum_burst_scalps()
    assert "MBUR" in runner._momentum_burst_pending

    runner._bar_aggregator.append(
        Bar("MBUR", now + timedelta(seconds=10), 2.53, 2.55, 2.52, 2.54, 130_000, Timeframe.SEC_10)
    )
    runner._process_momentum_burst_scalps()

    assert runner._breakout_scalp_active is False
    assert not getattr(runner, "_opened", None)
    assert "MBUR" not in runner._momentum_burst_pending


def test_momentum_burst_level_context_marks_half_dollar_break() -> None:
    assert _QuickScalpRunner._momentum_burst_level_context(2.43, 2.51) == {
        "psych_level": 2.5,
        "entry_trigger": "psych_level_break",
    }
    assert _QuickScalpRunner._momentum_burst_level_context(10.40, 11.05) == {
        "psych_level": 11.0,
        "entry_trigger": "psych_level_break",
    }
    assert _QuickScalpRunner._momentum_burst_level_context(2.51, 2.58) == {}


def test_momentum_burst_hit_run_rejects_failed_half_dollar_hold() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_cycle_enabled = False
    runner._momentum_burst_hit_run_enabled = True
    now = runner._test_now
    runner._bar_aggregator = _Bars10([
        Bar("MBUR", now - timedelta(seconds=10), 2.38, 2.40, 2.36, 2.39, 75_000, Timeframe.SEC_10),
        # Spike crosses $2.50 but closes just below it. This only arms pending.
        Bar("MBUR", now, 2.39, 2.501, 2.38, 2.49, 125_000, Timeframe.SEC_10),
    ])

    runner._process_momentum_burst_scalps()
    assert runner._momentum_burst_pending["MBUR"]["psych_level"] == 2.5

    # Next 10s bar is technically green and trades through the spike high, but
    # it cannot hold the $2.50 level. Do not buy this failed level break.
    runner._bar_aggregator.append(
        Bar("MBUR", now + timedelta(seconds=10), 2.48, 2.507, 2.47, 2.496, 130_000, Timeframe.SEC_10)
    )
    runner._process_momentum_burst_scalps()

    assert runner._breakout_scalp_active is False
    assert not getattr(runner, "_opened", None)
    assert "MBUR" not in runner._momentum_burst_pending


def test_momentum_burst_hit_run_accepts_half_dollar_hold() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_cycle_enabled = False
    runner._momentum_burst_hit_run_enabled = True
    now = runner._test_now
    runner._bar_aggregator = _Bars10([
        Bar("MBUR", now - timedelta(seconds=10), 2.38, 2.40, 2.36, 2.39, 75_000, Timeframe.SEC_10),
        Bar("MBUR", now, 2.39, 2.501, 2.38, 2.49, 125_000, Timeframe.SEC_10),
    ])

    runner._process_momentum_burst_scalps()
    runner._bar_aggregator.append(
        Bar("MBUR", now + timedelta(seconds=10), 2.49, 2.56, 2.48, 2.55, 130_000, Timeframe.SEC_10)
    )
    runner._process_momentum_burst_scalps()

    assert runner._breakout_scalp_active is True
    signal = runner._opened[0]
    assert signal.scan_result.scanner_name == "momentum_burst_hit_run"
    assert signal.scan_result.criteria["psych_level"] == 2.5
    assert signal.scan_result.criteria["entry_trigger"] == "psych_level_break"


def test_momentum_burst_hit_run_rebases_after_clean_micro_base_pause() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_cycle_enabled = False
    runner._momentum_burst_hit_run_enabled = True
    runner._momentum_burst_continuation_base_ok = (
        lambda symbol: (True, "fresh micro-base reclaim", {"base_high": 2.55, "base_low": 2.50})
    )
    now = runner._test_now

    runner._process_momentum_burst_scalps()
    assert "MBUR" in runner._momentum_burst_pending

    # The first confirm bar holds green but does not clear the spike high. This
    # used to delete the setup; now it should rebase to the micro-base instead.
    runner._bar_aggregator.append(
        Bar("MBUR", now + timedelta(seconds=10), 2.52, 2.55, 2.50, 2.54, 130_000, Timeframe.SEC_10)
    )
    runner._process_momentum_burst_scalps()

    assert runner._breakout_scalp_active is False
    assert not getattr(runner, "_opened", None)
    assert runner._momentum_burst_pending["MBUR"]["micro_base_reclaim"] is True
    assert runner._momentum_burst_pending["MBUR"]["breakout_close"] == 2.54
    assert runner._momentum_burst_pending["MBUR"]["original_ts"] == now
    assert runner._momentum_burst_pending["MBUR"]["original_breakout_close"] == 2.52
    assert runner._momentum_burst_pending["MBUR"]["rebase_count"] == 1

    # The next 10s bar finally clears that micro-base high with a strong close.
    runner._bar_aggregator.append(
        Bar("MBUR", now + timedelta(seconds=20), 2.54, 2.60, 2.53, 2.59, 140_000, Timeframe.SEC_10)
    )
    runner._process_momentum_burst_scalps()

    assert runner._breakout_scalp_active is True
    assert runner._momentum_burst_hit_run_counts["MBUR"] == 1
    assert runner._opened[0].scan_result.scanner_name == "momentum_burst_hit_run"


def test_momentum_burst_hit_run_rebase_keeps_original_timeout_anchor() -> None:
    runner = _momentum_burst_trade_runner()
    runner._momentum_burst_cycle_enabled = False
    runner._momentum_burst_hit_run_enabled = True
    runner._momentum_burst_continuation_base_ok = (
        lambda symbol: (True, "fresh micro-base reclaim", {"base_high": 2.55, "base_low": 2.50})
    )
    now = runner._test_now
    runner._momentum_burst_pending = {
        "MBUR": {
            "ts": now + timedelta(seconds=20),
            "original_ts": now - timedelta(seconds=20),
            "breakout_close": 2.54,
            "breakout_high": 2.55,
            "original_breakout_close": 2.52,
            "original_breakout_high": 2.55,
            "breakout_volume": 130_000,
            "rebase_count": 1,
        }
    }
    runner._momentum_burst_window_high["MBUR"] = 2.60
    runner._bar_aggregator.append(
        Bar("MBUR", now + timedelta(seconds=20), 2.54, 2.55, 2.50, 2.54, 130_000, Timeframe.SEC_10)
    )

    runner._process_momentum_burst_scalps()

    assert "MBUR" not in runner._momentum_burst_pending
    assert not getattr(runner, "_opened", None)


def test_momentum_burst_hit_run_time_window_allows_early_et() -> None:
    runner = _QuickScalpRunner()

    assert runner._momentum_burst_hit_run_time_allowed(
        datetime(2026, 6, 8, 14, 30, tzinfo=timezone.utc)
    ) is True


def test_momentum_burst_hit_run_time_window_blocks_afternoon_et() -> None:
    runner = _QuickScalpRunner()

    assert runner._momentum_burst_hit_run_time_allowed(
        datetime(2026, 6, 8, 18, 0, tzinfo=timezone.utc)
    ) is False


def test_momentum_burst_hit_run_time_window_blank_allows_all_day() -> None:
    runner = _QuickScalpRunner()
    runner._momentum_burst_hit_run_end_et = ""

    assert runner._momentum_burst_hit_run_time_allowed(
        datetime(2026, 6, 8, 18, 0, tzinfo=timezone.utc)
    ) is True
