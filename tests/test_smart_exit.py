"""Tests for the smart tiered exit system.

Verifies that the platform doesn't dump the whole position at the first
target — instead it scales out in tiers, locks breakeven, and uses
adaptive trailing to ride big moves.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from daytrading.exits.manager import (
    ExitManager,
    ExitTier,
    TrackedPosition,
    build_exit_tiers,
)
from daytrading.models import ExitReason, Side, SignalAction

TS = datetime(2026, 5, 13, 14, 30, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float = 0.0) -> datetime:
    return TS + timedelta(seconds=offset_seconds)


# ---------------------------------------------------------------------------
# Tier construction
# ---------------------------------------------------------------------------

class TestBuildTiers:

    def test_default_3_tiers(self) -> None:
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY)
        assert len(tiers) == 3
        total_shares = sum(t.shares for t in tiers)
        assert total_shares == 500

    def test_tier1_has_fixed_target(self) -> None:
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY)
        assert tiers[0].target_price is not None
        assert abs(tiers[0].target_price - 5.05) < 0.001  # 5 cents above

    def test_tier2_has_trail_only(self) -> None:
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY)
        assert tiers[1].target_price is None
        assert tiers[1].trail_cents is not None

    def test_tier3_has_wider_trail(self) -> None:
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY)
        assert tiers[2].trail_cents > tiers[1].trail_cents

    def test_short_side_targets_below(self) -> None:
        tiers = build_exit_tiers(quantity=500, entry_price=10.00, side=Side.SELL)
        assert tiers[0].target_price < 10.00

    def test_custom_tier_sizes(self) -> None:
        tiers = build_exit_tiers(
            quantity=1000, entry_price=5.00, side=Side.BUY,
            tier1_pct=0.50, tier2_pct=0.25, tier3_pct=0.25,
        )
        assert tiers[0].shares == 500
        assert tiers[1].shares == 250
        assert tiers[2].shares == 250


# ---------------------------------------------------------------------------
# Tier 1: lock in profit
# ---------------------------------------------------------------------------

class TestTier1FixedTarget:

    def test_tier1_exits_partial_at_target(self) -> None:
        """When price hits $5.05, only Tier 1 (40% = 200 shares) exits."""
        em = ExitManager()
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY)
        em.track(TrackedPosition(
            symbol="ABC", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=4.97, tiers=tiers,
        ))

        exits = em.check_exits({"ABC": 5.05}, _ts(10))
        assert len(exits) == 1
        assert exits[0].quantity == 200  # 40% of 500
        assert "take_profit" in exits[0].reason

        # position still tracked with remaining shares
        assert "ABC" in em.tracked
        assert em.tracked["ABC"].remaining_qty == 300

    def test_breakeven_lock_after_tier1(self) -> None:
        """After Tier 1 fills, stop loss moves to entry price."""
        em = ExitManager()
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY)
        em.track(TrackedPosition(
            symbol="ABC", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=4.97, tiers=tiers,
        ))

        em.check_exits({"ABC": 5.05}, _ts(10))

        pos = em.tracked["ABC"]
        assert pos.breakeven_locked is True
        assert pos.stop_loss == 5.00  # moved to entry = can't lose money


# ---------------------------------------------------------------------------
# Tier 2: trailing stop — let it run
# ---------------------------------------------------------------------------

class TestTier2TrailingStop:

    def test_tier2_trails_up_and_exits_on_pullback(self) -> None:
        """After Tier 1, price goes to $5.20, then pulls back → Tier 2 exits."""
        em = ExitManager()
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY)
        em.track(TrackedPosition(
            symbol="XYZ", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=4.97, tiers=tiers,
        ))

        # Tier 1 fills at $5.05
        em.check_exits({"XYZ": 5.05}, _ts(5))
        assert em.tracked["XYZ"].remaining_qty == 300

        # price runs to $5.20 — no exit, trail follows
        em.check_exits({"XYZ": 5.10}, _ts(10))
        em.check_exits({"XYZ": 5.15}, _ts(15))
        em.check_exits({"XYZ": 5.20}, _ts(20))
        assert em.tracked["XYZ"].remaining_qty == 300  # still holding

        # price pulls back to $5.17 — Tier 2 trail is 3¢ from high ($5.20)
        # trail level = $5.20 - $0.03 = $5.17
        exits = em.check_exits({"XYZ": 5.17}, _ts(25))
        tier2_exits = [e for e in exits if "trailing_stop" in e.reason]
        assert len(tier2_exits) >= 1

    def test_tier2_does_not_exit_if_still_running(self) -> None:
        """Trail follows price up — no exit while price keeps going."""
        em = ExitManager()
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY)
        em.track(TrackedPosition(
            symbol="RUN", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=4.97, tiers=tiers,
        ))

        # Tier 1 fills
        em.check_exits({"RUN": 5.05}, _ts(5))

        # steady uptrend — no pullback bigger than trail
        for i in range(10):
            price = 5.10 + i * 0.02
            exits = em.check_exits({"RUN": price}, _ts(10 + i * 2))
            # should not trigger trailing exit while trending
            trailing_exits = [e for e in exits if "trailing_stop" in e.reason]
            assert len(trailing_exits) == 0 or price < em.tracked.get("RUN", TrackedPosition(symbol="", side=Side.BUY, quantity=0, entry_price=0)).highest_price


# ---------------------------------------------------------------------------
# Tier 3: ride the wave (wide trail)
# ---------------------------------------------------------------------------

class TestTier3RideTheWave:

    def test_tier3_catches_big_move(self) -> None:
        """Stock goes from $5.00 to $5.50 — Tier 3 rides most of it."""
        em = ExitManager()
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY,
                                 tier3_trail_cents=5.0)
        em.track(TrackedPosition(
            symbol="BIG", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=4.97, tiers=tiers,
        ))

        # Tier 1 fills at $5.05
        em.check_exits({"BIG": 5.05}, _ts(5))

        # price runs to $5.50
        for i in range(20):
            price = 5.10 + i * 0.02
            em.check_exits({"BIG": price}, _ts(10 + i * 2))

        # Tier 3 should still be holding at $5.48+ because trail is 5¢
        pos = em.tracked.get("BIG")
        if pos:
            assert pos.remaining_qty > 0  # Tier 3 still open

        # now price pulls back to $5.43 (5¢ below $5.48 high)
        exits = em.check_exits({"BIG": 5.43}, _ts(60))
        # at least Tier 3 or remaining position should exit
        total_exited = sum(e.quantity for e in exits)
        assert total_exited > 0


# ---------------------------------------------------------------------------
# Hard stop loss and time exit
# ---------------------------------------------------------------------------

class TestHardExits:

    def test_stop_loss_dumps_everything(self) -> None:
        """If price drops below stop, ALL remaining shares exit."""
        em = ExitManager()
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY)
        em.track(TrackedPosition(
            symbol="BAD", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=4.97, tiers=tiers,
        ))

        exits = em.check_exits({"BAD": 4.96}, _ts(5))
        assert len(exits) == 1
        assert exits[0].quantity == 500  # all shares
        assert "stop_loss" in exits[0].reason
        assert "BAD" not in em.tracked

    def test_breakeven_stop_after_tier1_saves_you(self) -> None:
        """After Tier 1, stop moves to breakeven. If stock drops back, no loss."""
        em = ExitManager()
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY)
        em.track(TrackedPosition(
            symbol="SAFE", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=4.97, tiers=tiers,
        ))

        # Tier 1 fills, stop moves to $5.00
        em.check_exits({"SAFE": 5.05}, _ts(5))
        assert em.tracked["SAFE"].stop_loss == 5.00

        # price drops to exactly breakeven
        exits = em.check_exits({"SAFE": 5.00}, _ts(15))
        assert len(exits) == 1
        assert exits[0].quantity == 300  # remaining shares
        assert "stop_loss" in exits[0].reason
        # net result: +5¢ on 200 shares = $10 profit, $0 loss on 300 = STILL GREEN

    def test_time_exit_dumps_everything(self) -> None:
        em = ExitManager()
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY)
        em.track(TrackedPosition(
            symbol="SLOW", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=4.97, max_hold_seconds=120, tiers=tiers,
        ))

        exits = em.check_exits({"SLOW": 5.02}, _ts(121))
        assert len(exits) == 1
        assert exits[0].quantity == 500
        assert "time_exit" in exits[0].reason


# ---------------------------------------------------------------------------
# Momentum detection
# ---------------------------------------------------------------------------

class TestMomentumDetection:

    def test_momentum_factor_accelerating(self) -> None:
        """Price accelerating → momentum factor > 1."""
        pos = TrackedPosition(
            symbol="X", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
        )
        # slow start, then accelerating
        for p in [5.01, 5.02, 5.03, 5.05, 5.08, 5.12]:
            pos.record_price(p)

        assert pos.momentum_factor > 1.0

    def test_momentum_factor_decelerating(self) -> None:
        """Price decelerating → momentum factor < 1."""
        pos = TrackedPosition(
            symbol="X", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
        )
        # fast start, then stalling
        for p in [5.10, 5.18, 5.24, 5.25, 5.25, 5.26]:
            pos.record_price(p)

        assert pos.momentum_factor < 1.0


# ---------------------------------------------------------------------------
# Short side
# ---------------------------------------------------------------------------

class TestShortSide:

    def test_short_tier1_exits_below_entry(self) -> None:
        em = ExitManager()
        tiers = build_exit_tiers(quantity=500, entry_price=8.00, side=Side.SELL,
                                 tier1_target_cents=5.0)
        # Tier 1 target = 8.00 - 0.05 = 7.95
        assert abs(tiers[0].target_price - 7.95) < 0.001

        em.track(TrackedPosition(
            symbol="SHORT", side=Side.SELL, quantity=500,
            entry_price=8.00, entry_ts=_ts(0),
            stop_loss=8.03, tiers=tiers,
        ))

        exits = em.check_exits({"SHORT": 7.95}, _ts(10))
        assert len(exits) == 1
        assert exits[0].quantity == 200

    def test_short_stop_loss_above(self) -> None:
        em = ExitManager()
        tiers = build_exit_tiers(quantity=500, entry_price=8.00, side=Side.SELL)
        em.track(TrackedPosition(
            symbol="SHORT", side=Side.SELL, quantity=500,
            entry_price=8.00, entry_ts=_ts(0),
            stop_loss=8.03, tiers=tiers,
        ))

        exits = em.check_exits({"SHORT": 8.04}, _ts(5))
        assert len(exits) == 1
        assert exits[0].quantity == 500
        assert "stop_loss" in exits[0].reason


# ---------------------------------------------------------------------------
# Full scenario: the trade that runs
# ---------------------------------------------------------------------------

class TestFullScenario:

    def test_5_dollar_stock_runs_to_5_50(self) -> None:
        """The scenario you asked about: stock keeps going after first target.

        Entry: $5.00, 500 shares
        Tier 1: 200 shares exit at $5.05 → +$10
        Tier 2: 150 shares ride to ~$5.30 (trail at 3¢ from high) → +$40.50
        Tier 3: 150 shares ride to ~$5.45 (trail at 5¢ from high) → +$63
        Total: ~$113.50 instead of $25 (if we had exited everything at $5.05)
        """
        em = ExitManager()
        tiers = build_exit_tiers(quantity=500, entry_price=5.00, side=Side.BUY,
                                 tier1_target_cents=5.0,
                                 tier2_trail_cents=3.0,
                                 tier3_trail_cents=5.0)
        em.track(TrackedPosition(
            symbol="RUNNER", side=Side.BUY, quantity=500,
            entry_price=5.00, entry_ts=_ts(0),
            stop_loss=4.97, tiers=tiers,
        ))

        all_exits = []

        # Tier 1: hits $5.05
        exits = em.check_exits({"RUNNER": 5.05}, _ts(10))
        all_exits.extend(exits)
        assert len(exits) == 1
        assert exits[0].quantity == 200

        # price keeps running
        for price in [5.10, 5.15, 5.20, 5.25, 5.30, 5.35, 5.40, 5.45, 5.50]:
            exits = em.check_exits({"RUNNER": price}, _ts(20))
            all_exits.extend(exits)

        # price pulls back → remaining tiers exit
        for price in [5.48, 5.45, 5.42, 5.38]:
            exits = em.check_exits({"RUNNER": price}, _ts(30))
            all_exits.extend(exits)

        total_shares_exited = sum(e.quantity for e in all_exits)
        assert total_shares_exited == 500  # all shares accounted for

        # calculate P&L
        total_pnl = 0.0
        for ex in all_exits:
            total_pnl += ex.quantity * (ex.entry_price - 5.00)

        # should be WAY more than the $25 from exiting all at $5.05
        assert total_pnl > 25.0, f"PnL was only ${total_pnl:.2f}, should beat $25"
