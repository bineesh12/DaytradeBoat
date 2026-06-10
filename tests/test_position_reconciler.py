"""Tests for broker ↔ bot position reconciliation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from daytrading.execution.position_reconciler import PositionReconciler
from daytrading.exits.manager import ExitManager, TrackedPosition
from daytrading.models import PortfolioState, Position, Side


def _tracked(symbol: str = "AIIO", qty: float = 100) -> TrackedPosition:
    return TrackedPosition(
        symbol=symbol,
        side=Side.BUY,
        quantity=qty,
        remaining_qty=qty,
        entry_price=5.0,
        entry_ts=datetime.now(timezone.utc),
        stop_loss=4.8,
    )


class TestPositionReconciler:
    def test_pending_entry_not_untracked_on_first_broker_miss(self) -> None:
        rec = PositionReconciler(broker_miss_limit=3)
        exit_mgr = ExitManager()
        portfolio = PortfolioState(cash=10_000)
        portfolio.positions["AIIO"] = Position(symbol="AIIO", quantity=100, avg_price=5.0)
        exit_mgr.track(_tracked())

        now = datetime.now(timezone.utc)
        rec.mark_entry_pending("AIIO", now)
        result = rec.reconcile({}, portfolio, exit_mgr, now=now)

        assert "AIIO" in exit_mgr.tracked
        assert result.still_pending == ["AIIO"]
        assert result.closed == []

    def test_untrack_after_repeated_broker_misses(self) -> None:
        rec = PositionReconciler(broker_miss_limit=2, pending_grace=timedelta(seconds=0))
        exit_mgr = ExitManager()
        portfolio = PortfolioState(cash=10_000)
        exit_mgr.track(_tracked())

        now = datetime.now(timezone.utc)
        rec.reconcile({}, portfolio, exit_mgr, now=now)
        result = rec.reconcile({}, portfolio, exit_mgr, now=now)

        assert "AIIO" not in exit_mgr.tracked
        assert result.closed == ["AIIO"]

    def test_adopt_orphan_broker_position(self) -> None:
        rec = PositionReconciler()
        exit_mgr = ExitManager()
        portfolio = PortfolioState(cash=10_000)

        result = rec.reconcile(
            {"XYZ": {"qty": 50, "avg_entry": 3.0, "current_price": 3.1}},
            portfolio,
            exit_mgr,
        )

        assert result.adopted == ["XYZ"]
        assert "XYZ" in exit_mgr.tracked
        assert portfolio.positions["XYZ"].quantity == 50

    def test_adopt_orphan_10_tick_stop(self) -> None:
        """Adopted position should have a 10-tick ($0.10) stop below entry."""
        rec = PositionReconciler()
        exit_mgr = ExitManager()
        portfolio = PortfolioState(cash=10_000)

        rec.reconcile(
            {"TENX": {"qty": 263, "avg_entry": 12.44}},
            portfolio,
            exit_mgr,
        )

        pos = exit_mgr.tracked["TENX"]
        assert pos.stop_loss == pytest.approx(12.34, abs=0.001)
        assert pos.risk_per_share == pytest.approx(0.10, abs=0.001)

    def test_adopt_orphan_rebuilds_first_target(self) -> None:
        """Adopted positions should still take partial profit at the fallback 1:1 target."""
        rec = PositionReconciler()
        exit_mgr = ExitManager()
        portfolio = PortfolioState(cash=10_000)

        rec.reconcile(
            {"TENX": {"qty": 263, "avg_entry": 12.44}},
            portfolio,
            exit_mgr,
        )

        pos = exit_mgr.tracked["TENX"]
        assert pos.first_target_price == pytest.approx(12.54, abs=0.001)
        assert pos.sold_half is False

    def test_adopt_orphan_can_half_sell_at_rebuilt_target(self) -> None:
        rec = PositionReconciler()
        exit_mgr = ExitManager()
        portfolio = PortfolioState(cash=10_000)

        rec.reconcile(
            {"TENX": {"qty": 100, "avg_entry": 12.44}},
            portfolio,
            exit_mgr,
        )

        signals = exit_mgr.check_exits({"TENX": 12.55}, datetime.now(timezone.utc))

        assert len(signals) == 1
        assert signals[0].symbol == "TENX"
        assert signals[0].quantity == 50
        assert exit_mgr.tracked["TENX"].sold_half is True

    def test_adopt_orphan_trend_strength(self) -> None:
        """Adopted position should get moderate trend_strength for decent stale timeout."""
        rec = PositionReconciler()
        exit_mgr = ExitManager()
        portfolio = PortfolioState(cash=10_000)

        rec.reconcile(
            {"TENX": {"qty": 100, "avg_entry": 12.44}},
            portfolio,
            exit_mgr,
        )

        pos = exit_mgr.tracked["TENX"]
        assert pos.trend_strength == 0.7

    def test_unexpected_broker_short_is_flagged_and_tracked_for_cover(self) -> None:
        rec = PositionReconciler()
        exit_mgr = ExitManager()
        portfolio = PortfolioState(cash=10_000)

        result = rec.reconcile(
            {"VERU": {"qty": -1, "avg_entry": 4.28, "current_price": 6.02}},
            portfolio,
            exit_mgr,
        )

        assert result.accidental_shorts == ["VERU"]
        assert result.adopted == []
        assert "VERU" in exit_mgr.tracked
        tracked = exit_mgr.tracked["VERU"]
        assert tracked.side is Side.SELL
        assert tracked.remaining_qty == 1
        assert tracked.stop_loss == pytest.approx(4.38)
        assert portfolio.positions["VERU"].quantity == -1


class TestAdoptedTrailingStop:
    """Test the trailing stop behavior for adopted (no-target) positions."""

    def _make_adopted_pos(self, entry: float = 12.44) -> TrackedPosition:
        """Create a position that looks like an adopted orphan."""
        TICK = 0.01
        pos = TrackedPosition(
            symbol="TENX",
            side=Side.BUY,
            quantity=263,
            remaining_qty=263,
            entry_price=entry,
            entry_ts=datetime.now(timezone.utc),
            stop_loss=round(entry - 10 * TICK, 4),
            risk_per_share=10 * TICK,
            trend_strength=0.7,
            reason="adopted from broker",
        )
        pos.first_target_price = 0  # override __post_init__ auto-compute
        return pos

    def test_trailing_stop_activates_after_breakeven(self) -> None:
        """After breakeven lock, stop should trail 10 ticks behind highest price."""
        exit_mgr = ExitManager()
        pos = self._make_adopted_pos(entry=10.00)
        exit_mgr.track(pos)

        now = datetime.now(timezone.utc)

        # Price moves up 1% → breakeven triggers, trailing also activates
        signals = exit_mgr.check_exits({"TENX": 10.10}, now)
        assert signals == []
        assert pos.breakeven_locked is True
        assert pos.stop_loss == pytest.approx(10.00, abs=0.001)  # trail: 10.10 - 0.10

        # Price moves to 10.20 → trail = 10.20 - 0.10 = 10.10
        signals = exit_mgr.check_exits({"TENX": 10.20}, now)
        assert signals == []
        assert pos.stop_loss == pytest.approx(10.10, abs=0.001)

        # Price moves to 10.40 → trail = 10.40 - 0.10 = 10.30
        signals = exit_mgr.check_exits({"TENX": 10.40}, now)
        assert signals == []
        assert pos.stop_loss == pytest.approx(10.30, abs=0.001)

    def test_trailing_stop_does_not_move_down(self) -> None:
        """Stop should never decrease — only ratchet up."""
        exit_mgr = ExitManager()
        pos = self._make_adopted_pos(entry=10.00)
        exit_mgr.track(pos)

        now = datetime.now(timezone.utc)

        # Lock breakeven and trail up
        exit_mgr.check_exits({"TENX": 10.10}, now)  # breakeven at +1%
        exit_mgr.check_exits({"TENX": 10.30}, now)  # trail = 10.20
        assert pos.stop_loss == pytest.approx(10.20, abs=0.001)

        # Price pulls back — stop stays at 10.20
        exit_mgr.check_exits({"TENX": 10.15}, now)
        assert pos.stop_loss == pytest.approx(10.20, abs=0.001)

    def test_trailing_stop_triggers_exit(self) -> None:
        """When price drops to trailing stop level, position exits."""
        exit_mgr = ExitManager()
        pos = self._make_adopted_pos(entry=10.00)
        exit_mgr.track(pos)

        now = datetime.now(timezone.utc)

        # Move up and trail
        exit_mgr.check_exits({"TENX": 10.10}, now)  # breakeven at +1%
        exit_mgr.check_exits({"TENX": 10.30}, now)  # trail = 10.20

        # Price drops to trailing stop
        signals = exit_mgr.check_exits({"TENX": 10.20}, now)
        assert len(signals) == 1
        assert signals[0].symbol == "TENX"

    def test_no_trailing_before_breakeven(self) -> None:
        """Before breakeven is locked, stop stays at initial 10-tick level."""
        exit_mgr = ExitManager()
        pos = self._make_adopted_pos(entry=10.00)
        exit_mgr.track(pos)

        now = datetime.now(timezone.utc)

        # Price moves slightly — not enough for breakeven (needs 1%)
        exit_mgr.check_exits({"TENX": 10.05}, now)
        assert pos.breakeven_locked is False
        assert pos.stop_loss == pytest.approx(9.90, abs=0.001)  # initial 10-tick stop

    def test_normal_position_not_affected(self) -> None:
        """Positions with a target should NOT get trailing stop behavior."""
        exit_mgr = ExitManager()
        pos = TrackedPosition(
            symbol="ABC",
            side=Side.BUY,
            quantity=100,
            remaining_qty=100,
            entry_price=10.00,
            entry_ts=datetime.now(timezone.utc),
            stop_loss=9.70,
            risk_per_share=0.30,
            first_target_price=10.30,
            trend_strength=0.7,
        )
        exit_mgr.track(pos)

        now = datetime.now(timezone.utc)

        # Price moves up, locks breakeven
        exit_mgr.check_exits({"ABC": 10.10}, now)
        assert pos.breakeven_locked is True
        # Stop should be at breakeven (entry), NOT trailing
        assert pos.stop_loss == pytest.approx(10.00, abs=0.001)
