"""Tests for Warrior Trading stepping-stop exit manager.

Replaces legacy 3-tier exit tests. Current model:
  - Half sell at 2:1 R:R
  - Breakeven stop after half (or after 2% move pre-half)
  - Stepping trailing stop after half
  - Red candle / extension / stale exits
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from daytrading.exits.manager import (
    ExitManager,
    TrackedPosition,
    build_exit_tiers,
)
from daytrading.models import ExitReason, Side, SignalAction

TS = datetime(2026, 5, 13, 14, 30, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float = 0.0) -> datetime:
    return TS + timedelta(seconds=offset_seconds)


def _long_pos(
    symbol: str = "ABC",
    entry: float = 5.00,
    stop: float = 4.90,
    qty: float = 100,
) -> TrackedPosition:
    risk = entry - stop
    return TrackedPosition(
        symbol=symbol,
        side=Side.BUY,
        quantity=qty,
        remaining_qty=qty,
        original_qty=qty,
        entry_price=entry,
        entry_ts=_ts(0),
        stop_loss=stop,
        risk_per_share=risk,
        first_target_price=entry + risk * 2,
    )


class TestBuildExitTiersCompat:
    def test_returns_empty_list(self) -> None:
        assert build_exit_tiers(500, 5.00, Side.BUY) == []


class TestStopLoss:
    def test_stop_loss_exits_full_position(self) -> None:
        em = ExitManager()
        em.track(_long_pos(stop=4.90))
        exits = em.check_exits({"ABC": 4.89}, _ts(5))
        assert len(exits) == 1
        assert exits[0].quantity == 100
        assert ExitReason.STOP_LOSS.value in exits[0].reason

    def test_no_exit_above_stop(self) -> None:
        em = ExitManager()
        em.track(_long_pos())
        exits = em.check_exits({"ABC": 5.02}, _ts(5))
        assert len(exits) == 0

    def test_dollar_stop_exits_before_wide_broker_stop(self) -> None:
        em = ExitManager(max_unrealized_loss=50.0)
        em.track(_long_pos(entry=5.59, stop=5.28, qty=294))
        exits = em.check_exits({"ABC": 5.40}, _ts(30))
        assert len(exits) == 1
        assert exits[0].quantity == 294
        assert ExitReason.STOP_LOSS.value in exits[0].reason

    def test_dollar_stop_does_not_exit_inside_loss_cap(self) -> None:
        em = ExitManager(max_unrealized_loss=50.0)
        em.track(_long_pos(entry=5.59, stop=5.28, qty=160))
        exits = em.check_exits({"ABC": 5.40}, _ts(30))
        assert len(exits) == 0


class TestHalfSellAt2to1:
    def test_half_sell_at_first_target(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.90)  # risk 0.10, target 5.20
        em.track(pos)
        exits = em.check_exits({"ABC": 5.21}, _ts(10))
        assert len(exits) == 1
        assert exits[0].quantity == 50
        assert "take_profit" in exits[0].reason.lower()
        tracked = em.tracked.get("ABC")
        assert tracked is not None
        assert tracked.sold_half is True
        assert tracked.stop_loss == pytest.approx(5.00)
        assert tracked.remaining_qty == 50

    def test_momentum_scalp_sells_half_at_one_percent(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.80)
        pos.reason = "Momentum Burst CING"
        em.track(pos)

        exits = em.check_exits({"ABC": 5.05}, _ts(10))

        assert len(exits) == 1
        assert exits[0].quantity == 50
        assert "take_profit" in exits[0].reason.lower()
        tracked = em.tracked.get("ABC")
        assert tracked is not None
        assert tracked.sold_half is True
        assert tracked.stop_loss == pytest.approx(5.00)
        assert tracked.remaining_qty == 50

    def test_vwap_pullback_does_not_use_quick_scalp_partial(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.80)
        pos.reason = "Vwap Pullback CRVO"
        pos.entry_strategy = "vwap_pullback"
        pos.entry_pattern = "vwap_pullback"
        em.track(pos)

        exits = em.check_exits({"ABC": 5.05}, _ts(10))

        assert exits == []
        tracked = em.tracked.get("ABC")
        assert tracked is not None

    def test_pullback_base_exits_full_size_on_quick_scalp_pop(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.80, qty=90)
        pos.reason = "Pullback Base UBXG"
        pos.entry_strategy = "pullback_base"
        pos.entry_pattern = "pullback_base"
        em.track(pos)

        exits = em.check_exits({"ABC": 5.05}, _ts(10))

        assert len(exits) == 1
        assert exits[0].quantity == 90
        assert "take_profit" in exits[0].reason.lower()
        assert "ABC" not in em.tracked

    def test_momentum_burst_hit_run_sells_full_at_first_target(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.90, qty=90)
        pos.reason = "Momentum Burst Hit-Run"
        pos.entry_strategy = "momentum_burst_hit_run"
        em.track(pos)

        exits = em.check_exits({"ABC": 5.21}, _ts(10))

        assert len(exits) == 1
        assert exits[0].quantity == 90
        assert "take_profit" in exits[0].reason.lower()
        assert "ABC" not in em.tracked

    def test_hit_run_emergency_dump_exits_full_position(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=10.00, stop=9.20, qty=90)
        pos.reason = "Warrior Squeeze STI"
        pos.entry_strategy = "warrior_squeeze_playbook"
        em.track(pos)

        em.update_bar_close(
            "ABC",
            close_price=9.52,
            open_price=10.05,
            high_price=10.12,
            low_price=9.50,
            volume=180_000,
        )
        exits = em.check_exits({"ABC": 9.52}, _ts(20))

        assert len(exits) == 1
        assert exits[0].quantity == 90
        assert "stop_loss" in exits[0].reason.lower()
        assert "ABC" not in em.tracked

    def test_non_hit_run_does_not_emergency_exit_above_stop(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=10.00, stop=9.20, qty=90)
        pos.reason = "Vwap Pullback ABC"
        pos.entry_strategy = "vwap_pullback"
        em.track(pos)

        em.update_bar_close(
            "ABC",
            close_price=9.52,
            open_price=10.05,
            high_price=10.12,
            low_price=9.50,
            volume=180_000,
        )
        exits = em.check_exits({"ABC": 9.52}, _ts(20))

        assert exits == []
        assert "ABC" in em.tracked

    def test_warrior_squeeze_banks_partial_at_first_target(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.90, qty=90)
        pos.reason = "Warrior Squeeze JRSH"
        pos.entry_strategy = "warrior_squeeze_playbook"
        em.track(pos)

        exits = em.check_exits({"ABC": 5.21}, _ts(10))

        assert len(exits) == 1
        assert exits[0].quantity == 45
        assert "take_profit" in exits[0].reason.lower()
        tracked = em.tracked.get("ABC")
        assert tracked is not None
        assert tracked.sold_half is True
        assert tracked.remaining_qty == 45
        assert tracked.stop_loss == pytest.approx(5.00)

    def test_warrior_halt_resume_trigger_sells_full_without_reason_literal(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=10.00, stop=9.50, qty=90)
        pos.reason = "Warrior Squeeze CUPR"
        pos.entry_strategy = "warrior_squeeze_playbook"
        pos.entry_pattern = "warrior_squeeze_playbook"
        pos.entry_trigger = "warrior_halt_resume_continuation"
        em.track(pos)

        exits = em.check_exits({"ABC": 11.05}, _ts(10))

        assert len(exits) == 1
        assert exits[0].quantity == 90
        assert "take_profit" in exits[0].reason.lower()
        assert "ABC" not in em.tracked

    def test_warrior_stair_step_runner_uses_partial_not_full_target_exit(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=10.00, stop=9.50, qty=90)
        pos.reason = "Warrior Squeeze STI"
        pos.entry_strategy = "warrior_squeeze_playbook"
        pos.entry_pattern = "warrior_squeeze_playbook"
        pos.entry_trigger = "warrior_stair_step_runner"
        em.track(pos)

        exits = em.check_exits({"ABC": 11.05}, _ts(10))

        assert len(exits) == 1
        assert exits[0].quantity == 45
        assert "take_profit" in exits[0].reason.lower()
        tracked = em.tracked.get("ABC")
        assert tracked is not None
        assert tracked.remaining_qty == 45
        assert tracked.stop_loss == pytest.approx(10.00)

    def test_warrior_failed_follow_through_exits_before_full_stop(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=9.74, stop=9.2043, qty=205)
        pos.reason = "Warrior Squeeze PLSM"
        pos.entry_strategy = "warrior_squeeze_playbook"
        pos.entry_pattern = "warrior_squeeze_playbook"
        pos.entry_trigger = "warrior_parabolic_micro_pullback_reclaim"
        pos.first_target_price = 10.4096
        em.track(pos)

        # First bar after entry pushes 0.6R toward target, then closes red near
        # its low on heavy volume. Warrior should cut the failed follow-through
        # at the bar close instead of waiting for the full tactical stop.
        tracked = em.tracked["ABC"]
        tracked.highest_price = 10.08
        em.update_bar_close(
            "ABC",
            close_price=9.59,
            open_price=9.78,
            high_price=10.08,
            low_price=9.48,
            volume=229_569,
        )

        exits = em.check_exits({"ABC": 9.59}, _ts(10))

        assert len(exits) == 1
        assert exits[0].quantity == 205
        assert exits[0].entry_price == pytest.approx(9.59)
        assert "stop_loss" in exits[0].reason.lower()
        assert "ABC" not in em.tracked

    def test_normal_position_does_not_use_warrior_failed_follow_through_exit(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=9.74, stop=9.2043, qty=205)
        pos.reason = "ABC Continuation PLSM"
        pos.entry_strategy = "standard"
        em.track(pos)
        tracked = em.tracked["ABC"]
        tracked.highest_price = 10.08
        em.update_bar_close(
            "ABC",
            close_price=9.59,
            open_price=9.78,
            high_price=10.08,
            low_price=9.48,
            volume=229_569,
        )

        exits = em.check_exits({"ABC": 9.59}, _ts(10))

        assert exits == []
        assert "ABC" in em.tracked

    def test_runner_candidate_confirms_after_first_partial(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.80)
        pos.reason = "Quick Momentum Scalp CHAI breakout"
        pos.runner_candidate = True
        pos.trend_strength = 0.9
        em.track(pos)

        exits = em.check_exits({"ABC": 5.10}, _ts(10))

        assert len(exits) == 1
        tracked = em.tracked.get("ABC")
        assert tracked is not None
        assert tracked.sold_half is True
        assert tracked.runner_confirmed is True
        assert tracked.runner_trail_pct == pytest.approx(0.03)

    def test_runner_candidate_banks_one_third_first_partial(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.80, qty=90)
        pos.reason = "Quick Momentum Scalp CHAI breakout"
        pos.runner_candidate = True
        pos.trend_strength = 0.9
        em.track(pos)

        exits = em.check_exits({"ABC": 5.10}, _ts(10))

        assert len(exits) == 1
        assert exits[0].quantity == 30
        tracked = em.tracked.get("ABC")
        assert tracked is not None
        assert tracked.remaining_qty == 60

    def test_runner_candidate_defaults_to_breakeven_after_first_partial(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.80, qty=90)
        pos.reason = "Quick Momentum Scalp AIIO breakout"
        pos.runner_candidate = True
        pos.trend_strength = 0.9
        em.track(pos)

        exits = em.check_exits({"ABC": 5.10}, _ts(10))

        assert len(exits) == 1
        tracked = em.tracked.get("ABC")
        assert tracked is not None
        assert tracked.sold_half is True
        assert tracked.breakeven_locked is True
        assert tracked.stop_loss == pytest.approx(5.00)

    def test_runner_give_room_keeps_structural_stop_after_first_partial(self) -> None:
        em = ExitManager(runner_give_room_after_partial=True)
        pos = _long_pos(entry=5.00, stop=4.80, qty=90)
        pos.reason = "Quick Momentum Scalp AIIO breakout"
        pos.runner_candidate = True
        pos.trend_strength = 0.9
        em.track(pos)

        exits = em.check_exits({"ABC": 5.10}, _ts(10))

        assert len(exits) == 1
        tracked = em.tracked.get("ABC")
        assert tracked is not None
        assert tracked.sold_half is True
        assert tracked.runner_confirmed is True
        assert tracked.breakeven_locked is True
        assert tracked.stop_loss == pytest.approx(4.80)

    def test_runner_give_room_does_not_apply_to_non_runner_partial(self) -> None:
        em = ExitManager(runner_give_room_after_partial=True)
        pos = _long_pos(entry=5.00, stop=4.80, qty=90)
        pos.reason = "Momentum Burst AIIO"
        pos.runner_candidate = False
        em.track(pos)

        exits = em.check_exits({"ABC": 5.10}, _ts(10))

        assert len(exits) == 1
        tracked = em.tracked.get("ABC")
        assert tracked is not None
        assert tracked.sold_half is True
        assert tracked.breakeven_locked is True
        assert tracked.stop_loss == pytest.approx(5.00)

    def test_ordinary_scalp_partial_does_not_confirm_runner(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.80)
        pos.reason = "Momentum Burst CING"
        em.track(pos)

        exits = em.check_exits({"ABC": 5.10}, _ts(10))

        assert len(exits) == 1
        tracked = em.tracked.get("ABC")
        assert tracked is not None
        assert tracked.sold_half is True
        assert tracked.runner_confirmed is False


class TestBreakevenAfterHalf:
    def test_remaining_stopped_at_breakeven(self) -> None:
        em = ExitManager()
        pos = _long_pos()
        pos.sold_half = True
        pos.remaining_qty = 50
        pos.stop_loss = 5.00
        pos.breakeven_locked = True
        em.track(pos)
        exits = em.check_exits({"ABC": 4.99}, _ts(60))
        assert len(exits) == 1
        assert exits[0].quantity == 50


class TestExtensionExit:
    def test_extension_exit_on_large_gain(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.50)
        pos.extension_threshold = 0.15
        em.track(pos)
        exits = em.check_exits({"ABC": 5.80}, _ts(30))  # 16% gain
        assert len(exits) == 1
        assert exits[0].quantity == 100


class TestRedCandleExit:
    def test_no_red_exit_before_120_seconds(self) -> None:
        em = ExitManager()
        pos = _long_pos()
        pos.last_bar_close = 5.05
        pos.prev_bar_open = 5.04
        em.track(pos)
        em.update_bar_close("ABC", close_price=5.00, open_price=5.05, volume=50_000)
        exits = em.check_exits({"ABC": 4.95}, _ts(30))
        assert len(exits) == 0

    def test_red_exit_after_hold_with_volume_and_drop(self) -> None:
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.90)
        pos.entry_ts = _ts(0)
        pos.last_bar_close = 5.15
        pos.last_bar_volume = 40_000
        pos.avg_bar_volume = 20_000
        pos.consecutive_red = 3
        em.track(pos)
        # Price below last_bar_close by >0.5% after 120s hold
        exits = em.check_exits({"ABC": 5.05}, _ts(130))
        assert len(exits) == 1
        assert exits[0].quantity == 100


class TestStaleExit:
    def test_stale_exit_when_no_progress(self) -> None:
        em = ExitManager()
        em.track(TrackedPosition(
            symbol="GHI", side=Side.BUY, quantity=100,
            remaining_qty=100, original_qty=100,
            entry_price=3.00, entry_ts=_ts(0),
            stop_loss=2.85, risk_per_share=0.15,
            first_target_price=3.30,
            trend_strength=0.3,
        ))
        exits = em.check_exits({"GHI": 2.98}, _ts(60))
        assert len(exits) == 0
        exits = em.check_exits({"GHI": 2.98}, _ts(185))
        assert len(exits) >= 1


class TestVolumeExhaustionExit:
    """Exit on volume exhaustion: 3+ green bars with declining volume while in profit."""

    def _setup_exhausted_pos(self) -> tuple:
        """Create a position with 3 green declining-volume bars."""
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.90)
        pos.entry_ts = _ts(0)
        em.track(pos)
        # First bar establishes baseline volume (no comparison possible yet)
        em.update_bar_close("ABC", close_price=5.03, open_price=5.00, volume=100_000)
        # Next 3 bars: green with declining volume → streak of 3
        em.update_bar_close("ABC", close_price=5.07, open_price=5.03, volume=80_000)
        em.update_bar_close("ABC", close_price=5.11, open_price=5.07, volume=60_000)
        em.update_bar_close("ABC", close_price=5.15, open_price=5.11, volume=40_000)
        return em, pos

    def test_exit_on_exhaustion_in_profit(self) -> None:
        """3 declining-vol green bars + in profit + held 120s -> exit."""
        em, pos = self._setup_exhausted_pos()
        exits = em.check_exits({"ABC": 5.12}, _ts(130))
        assert len(exits) == 1
        assert exits[0].quantity == 100
        assert "take_profit" in exits[0].reason.lower()

    def test_no_exit_before_120s(self) -> None:
        """Exhaustion detected but held less than 120s -> no exit."""
        em, pos = self._setup_exhausted_pos()
        exits = em.check_exits({"ABC": 5.12}, _ts(60))
        assert len(exits) == 0

    def test_no_exit_when_at_loss(self) -> None:
        """Exhaustion detected but position is at a loss -> no exhaustion exit."""
        em, pos = self._setup_exhausted_pos()
        # First call expands the range so range-exit doesn't interfere
        em.check_exits({"ABC": 5.15}, _ts(5))
        em.check_exits({"ABC": 4.92}, _ts(10))
        # Now call with price below entry but above stop, at 130s
        exits = em.check_exits({"ABC": 4.98}, _ts(130))
        # Any exit that fires should NOT be from volume exhaustion
        for e in exits:
            assert "volume exhaustion" not in e.reason.lower()

    def test_no_exit_with_only_2_declining_bars(self) -> None:
        """Only 2 declining-vol green bars -> no exit (need 3)."""
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.90)
        pos.entry_ts = _ts(0)
        em.track(pos)
        em.update_bar_close("ABC", close_price=5.05, open_price=5.00, volume=80_000)
        em.update_bar_close("ABC", close_price=5.10, open_price=5.05, volume=60_000)
        # Only 2 declining bars
        exits = em.check_exits({"ABC": 5.12}, _ts(130))
        assert len(exits) == 0

    def test_red_bar_resets_streak(self) -> None:
        """A red bar in the middle resets the exhaustion streak."""
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.90)
        pos.entry_ts = _ts(0)
        em.track(pos)
        em.update_bar_close("ABC", close_price=5.05, open_price=5.00, volume=80_000)
        em.update_bar_close("ABC", close_price=5.10, open_price=5.05, volume=60_000)
        # Red bar resets the streak
        em.update_bar_close("ABC", close_price=5.08, open_price=5.10, volume=40_000)
        em.update_bar_close("ABC", close_price=5.12, open_price=5.08, volume=30_000)
        # Only 1 qualifying bar after reset
        exits = em.check_exits({"ABC": 5.12}, _ts(130))
        assert len(exits) == 0

    def test_volume_increase_resets_streak(self) -> None:
        """A green bar with increasing volume resets the streak."""
        em = ExitManager()
        pos = _long_pos(entry=5.00, stop=4.90)
        pos.entry_ts = _ts(0)
        em.track(pos)
        em.update_bar_close("ABC", close_price=5.05, open_price=5.00, volume=80_000)
        em.update_bar_close("ABC", close_price=5.10, open_price=5.05, volume=60_000)
        # Volume increases — momentum returning
        em.update_bar_close("ABC", close_price=5.15, open_price=5.10, volume=90_000)
        em.update_bar_close("ABC", close_price=5.20, open_price=5.15, volume=70_000)
        # Only 1 qualifying bar since reset
        exits = em.check_exits({"ABC": 5.18}, _ts(130))
        assert len(exits) == 0


class TestConfigurableRunnerTrail:
    def _confirmed_runner(self, em: ExitManager, peak: float) -> TrackedPosition:
        pos = _long_pos(entry=5.00, stop=4.90, qty=100)
        pos.sold_half = True
        pos.breakeven_locked = True
        pos.stop_loss = 5.00  # breakeven after the partial
        pos.runner_candidate = True
        em._positions[pos.symbol] = pos
        em._maybe_confirm_runner(pos, peak)  # +10% run confirms the runner
        return pos

    def test_confirm_uses_configured_trail_pct(self) -> None:
        em = ExitManager(runner_trail_pct=0.08, runner_min_confirm_pct=0.02)
        pos = self._confirmed_runner(em, 5.50)
        assert pos.runner_confirmed is True
        assert pos.runner_trail_pct == pytest.approx(0.08)

    def test_wider_trail_holds_where_tight_trail_stops(self) -> None:
        # Same 4.5% pullback from a $5.50 high: a 3% trail stops, an 8% trail holds.
        def run(trail: float) -> int:
            em = ExitManager(runner_trail_pct=trail)
            self._confirmed_runner(em, 5.50)
            em.check_exits({"ABC": 5.50}, _ts(10))   # set high + trail stop
            return len(em.check_exits({"ABC": 5.25}, _ts(15)))

        assert run(0.03) == 1   # 5.25 <= 5.335 trail stop → exits
        assert run(0.08) == 0   # 5.25 > 5.06 trail stop → rides


class TestAdaptiveRunnerTrail:
    def test_flat_when_adaptive_off(self) -> None:
        em = ExitManager(runner_trail_pct=0.03, runner_trail_adaptive=False)
        pos = _long_pos()
        pos.runner_trail_pct = 0.03
        for r in (0.05, 0.06, 0.05):  # high vol ignored when adaptive off
            pos.record_bar_range(r)
        assert em._runner_trail_for(pos) == pytest.approx(0.03)

    def test_adaptive_scales_with_volatility_and_clamps(self) -> None:
        em = ExitManager(
            runner_trail_pct=0.03, runner_trail_adaptive=True,
            runner_trail_atr_mult=2.5, runner_trail_cap=0.10,
        )
        smooth = _long_pos(symbol="SMOOTH")
        for _ in range(5):
            smooth.record_bar_range(0.014)  # 1.4% bars -> 3.5%
        assert em._runner_trail_for(smooth) == pytest.approx(0.035)

        wide = _long_pos(symbol="WIDE")
        for _ in range(5):
            wide.record_bar_range(0.027)    # 2.7% bars -> 6.75%
        assert em._runner_trail_for(wide) == pytest.approx(0.0675)

        floored = _long_pos(symbol="FLOOR")
        floored.runner_trail_pct = 0.03
        for _ in range(5):
            floored.record_bar_range(0.005)  # 0.5% -> 1.25%, below floor
        assert em._runner_trail_for(floored) == pytest.approx(0.03)

        crazy = _long_pos(symbol="CRAZY")
        for _ in range(5):
            crazy.record_bar_range(0.10)     # 10% -> 25%, above cap
        assert em._runner_trail_for(crazy) == pytest.approx(0.10)


class TestGiveRunnerRoom:
    def _runner(self, em: ExitManager) -> TrackedPosition:
        pos = _long_pos(entry=5.00, stop=4.90, qty=100)  # risk 0.10, 1:1 target 5.20
        pos.runner_candidate = True
        em._positions[pos.symbol] = pos
        return pos

    def test_breakeven_by_default_after_partial(self) -> None:
        em = ExitManager(runner_give_room_after_partial=False)
        pos = self._runner(em)
        em.check_exits({"ABC": 5.20}, _ts(30))  # hits 1:1 target -> half sell
        assert pos.sold_half is True
        assert pos.stop_loss == pytest.approx(5.00)  # snapped to breakeven

    def test_give_room_keeps_wider_stop_after_partial(self) -> None:
        em = ExitManager(runner_give_room_after_partial=True)
        pos = self._runner(em)
        em.check_exits({"ABC": 5.20}, _ts(30))
        assert pos.sold_half is True
        assert pos.breakeven_locked is True
        assert pos.stop_loss == pytest.approx(4.90)  # kept the wider original stop

    def test_give_room_ignored_for_non_runner(self) -> None:
        em = ExitManager(runner_give_room_after_partial=True)
        pos = _long_pos(entry=5.00, stop=4.90, qty=100)  # runner_candidate stays False
        em._positions[pos.symbol] = pos
        em.check_exits({"ABC": 5.20}, _ts(30))
        assert pos.stop_loss == pytest.approx(5.00)  # non-runners still go breakeven
