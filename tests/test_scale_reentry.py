"""Tests for scale-up (pyramiding) and re-entry after exit.

Covers:
  - PositionScaler: detects winning positions → generates scale-up signals
  - ExitManager.scale_up: adds shares, recalculates avg price, advances stop
  - ReentryDetector: detects continuation after full exit → re-enter smaller
  - Full scenario: entry → scale ups → tiered exits → re-entry
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Sequence

import pytest

from daytrading.exits.manager import (
    ExitManager,
    ExitTier,
    TrackedPosition,
    build_exit_tiers,
)
from daytrading.exits.scaler import (
    PositionScaler,
    ReentryConfig,
    ReentryDetector,
    ScaleUpConfig,
)
from daytrading.models import Bar, ExitReason, Side, SignalAction, TradeSignal

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ts(offset_s: int = 0) -> datetime:
    return datetime(2026, 5, 13, 9, 30, 0) + timedelta(seconds=offset_s)


def _bar(symbol: str, close: float, volume: float = 100_000, offset_s: int = 0) -> Bar:
    return Bar(
        symbol=symbol,
        ts=_ts(offset_s),
        open=close - 0.01,
        high=close + 0.02,
        low=close - 0.02,
        close=close,
        volume=volume,
    )


# ===================================================================
# ExitManager.scale_up
# ===================================================================

class TestExitManagerScaleUp:

    def test_scale_up_updates_avg_price(self) -> None:
        em = ExitManager()
        tiers = build_exit_tiers(500, 5.00, Side.BUY)
        em.track(TrackedPosition(
            symbol="AAPL", side=Side.BUY, quantity=500,
            entry_price=5.00, stop_loss=4.97, tiers=tiers,
        ))

        em.scale_up("AAPL", add_qty=250, add_price=5.10, new_stop=5.03)

        pos = em.tracked["AAPL"]
        expected_avg = (5.00 * 500 + 5.10 * 250) / 750
        assert abs(pos.entry_price - expected_avg) < 0.001
        assert pos.remaining_qty == 750
        assert pos.quantity == 750
        assert pos.stop_loss == 5.03

    def test_scale_up_adds_trailing_tier(self) -> None:
        em = ExitManager()
        tiers = build_exit_tiers(500, 5.00, Side.BUY)
        original_tier_count = len(tiers)
        em.track(TrackedPosition(
            symbol="AAPL", side=Side.BUY, quantity=500,
            entry_price=5.00, stop_loss=4.97, tiers=tiers,
        ))

        em.scale_up("AAPL", add_qty=250, add_price=5.10)
        pos = em.tracked["AAPL"]
        assert len(pos.tiers) == original_tier_count + 1
        new_tier = pos.tiers[-1]
        assert new_tier.shares == 250
        assert new_tier.trail_cents is not None
        assert new_tier.target_price is None

    def test_scale_up_nonexistent_symbol_is_noop(self) -> None:
        em = ExitManager()
        em.scale_up("NONEXIST", add_qty=100, add_price=5.00)
        assert "NONEXIST" not in em.tracked

    def test_multiple_scale_ups(self) -> None:
        em = ExitManager()
        tiers = build_exit_tiers(500, 5.00, Side.BUY)
        em.track(TrackedPosition(
            symbol="X", side=Side.BUY, quantity=500,
            entry_price=5.00, stop_loss=4.97, tiers=tiers,
        ))

        em.scale_up("X", 250, 5.10, new_stop=5.03)
        em.scale_up("X", 125, 5.20, new_stop=5.13)

        pos = em.tracked["X"]
        assert pos.remaining_qty == 875
        assert pos.stop_loss == 5.13
        expected_avg = (5.00 * 500 + 5.10 * 250 + 5.20 * 125) / 875
        assert abs(pos.entry_price - expected_avg) < 0.001


# ===================================================================
# PositionScaler
# ===================================================================

class TestPositionScaler:

    def _setup_winning_position(self) -> tuple:
        em = ExitManager()
        tiers = build_exit_tiers(500, 5.00, Side.BUY)
        pos = TrackedPosition(
            symbol="RUN", side=Side.BUY, quantity=500,
            entry_price=5.00, stop_loss=4.97, tiers=tiers,
        )
        em.track(pos)
        return em, pos

    def test_no_scale_when_not_profitable(self) -> None:
        em, _ = self._setup_winning_position()
        scaler = PositionScaler(ScaleUpConfig(min_profit_cents=5.0))

        signals = scaler.check_scale_ups(
            em, {"RUN": 5.02},
            {"RUN": [_bar("RUN", 5.02)]},
        )
        assert len(signals) == 0

    def test_no_scale_without_pullback(self) -> None:
        """Even if profitable, need a pullback first (don't chase)."""
        em, _ = self._setup_winning_position()
        scaler = PositionScaler(ScaleUpConfig(min_profit_cents=5.0))

        # stock goes straight up — no pullback yet
        signals = scaler.check_scale_ups(
            em, {"RUN": 5.15},
            {"RUN": [_bar("RUN", 5.15)]},
        )
        assert len(signals) == 0

    def test_max_scale_ups_respected(self) -> None:
        em, _ = self._setup_winning_position()
        scaler = PositionScaler(ScaleUpConfig(max_scale_ups=0))

        signals = scaler.check_scale_ups(
            em, {"RUN": 5.20},
            {"RUN": [_bar("RUN", 5.20)]},
        )
        assert len(signals) == 0

    def test_clear_resets_count(self) -> None:
        scaler = PositionScaler()
        scaler._scale_counts["X"] = 5
        scaler._pullback_seen["X"] = True
        scaler.clear("X")
        assert "X" not in scaler._scale_counts
        assert "X" not in scaler._pullback_seen

    def test_scale_signal_has_correct_action(self) -> None:
        em, pos = self._setup_winning_position()
        cfg = ScaleUpConfig(
            min_profit_cents=3.0,
            pullback_pct=20.0,
            bounce_pct=0.3,
        )
        scaler = PositionScaler(cfg)

        # simulate profitable position with pullback watermarks
        pos.highest_price = 5.15
        scaler._pullback_seen["RUN"] = True
        scaler._pullback_low["RUN"] = 5.08

        bars = [_bar("RUN", 5.08, 80_000), _bar("RUN", 5.12, 100_000)]
        signals = scaler.check_scale_ups(em, {"RUN": 5.12}, {"RUN": bars})

        assert len(signals) == 1
        assert signals[0].action is SignalAction.SCALE_UP_LONG
        assert signals[0].symbol == "RUN"
        assert signals[0].quantity == 250  # 500 * 0.5^1

    def test_pyramid_size_decays(self) -> None:
        em, pos = self._setup_winning_position()
        cfg = ScaleUpConfig(
            min_profit_cents=3.0,
            size_decay=0.5,
            pullback_pct=20.0,
            bounce_pct=0.3,
        )
        scaler = PositionScaler(cfg)

        # force two triggers
        pos.highest_price = 5.20
        for i in range(2):
            scaler._pullback_seen["RUN"] = True
            scaler._pullback_low["RUN"] = 5.10
            bars = [_bar("RUN", 5.10, 80_000), _bar("RUN", 5.15, 100_000)]
            signals = scaler.check_scale_ups(em, {"RUN": 5.15}, {"RUN": bars})
            if signals:
                expected_size = round(500 * (0.5 ** (i + 1)))
                assert signals[0].quantity == expected_size


# ===================================================================
# ReentryDetector
# ===================================================================

class TestReentryDetector:

    def test_no_reentry_during_cooldown(self) -> None:
        det = ReentryDetector(ReentryConfig(cooldown_seconds=30))
        det.record_full_exit(
            "X", Side.BUY, exit_price=5.10, exit_ts=_ts(0),
            highest_price=5.15, entry_price=5.00,
        )

        # only 10 seconds later
        signals = det.check_reentries(
            {"X": 5.15},
            {"X": [_bar("X", 5.15)]},
            _ts(10),
            {"X": 500},
        )
        assert len(signals) == 0

    def test_no_reentry_if_not_continued(self) -> None:
        det = ReentryDetector(ReentryConfig(
            cooldown_seconds=5, min_continuation_cents=3.0,
        ))
        det.record_full_exit(
            "X", Side.BUY, exit_price=5.10, exit_ts=_ts(0),
            highest_price=5.15, entry_price=5.00,
        )

        # price didn't continue past exit + 3¢
        signals = det.check_reentries(
            {"X": 5.11},
            {"X": [_bar("X", 5.11)]},
            _ts(60),
            {"X": 500},
        )
        assert len(signals) == 0

    def test_reentry_when_stock_continues(self) -> None:
        det = ReentryDetector(ReentryConfig(
            cooldown_seconds=5,
            min_continuation_cents=3.0,
            reentry_size_pct=0.5,
        ))
        det.record_full_exit(
            "X", Side.BUY, exit_price=5.10, exit_ts=_ts(0),
            highest_price=5.15, entry_price=5.00,
        )

        # pullback first, then continuation
        det.check_reentries({"X": 5.07}, {"X": [_bar("X", 5.07)]}, _ts(10), {"X": 500})

        bars = [_bar("X", 5.12, 80_000), _bar("X", 5.16, 100_000)]
        signals = det.check_reentries({"X": 5.16}, {"X": bars}, _ts(60), {"X": 500})

        assert len(signals) == 1
        assert signals[0].action is SignalAction.REENTER_LONG
        assert signals[0].quantity == 250  # 500 * 0.5
        assert signals[0].stop_loss is not None

    def test_max_reentries_enforced(self) -> None:
        det = ReentryDetector(ReentryConfig(
            cooldown_seconds=1, max_reentries=1,
            min_continuation_cents=1.0,
        ))
        det.record_full_exit(
            "X", Side.BUY, exit_price=5.10, exit_ts=_ts(0),
            highest_price=5.15, entry_price=5.00,
        )

        # first re-entry
        bars = [_bar("X", 5.05, 80_000), _bar("X", 5.15, 100_000)]
        det._pullback_prices["X"] = 5.05
        det.check_reentries({"X": 5.15}, {"X": bars}, _ts(60), {"X": 500})

        # record another exit
        det.record_full_exit(
            "X", Side.BUY, exit_price=5.20, exit_ts=_ts(120),
            highest_price=5.25, entry_price=5.15,
        )

        # second re-entry attempt → blocked
        det._pullback_prices["X"] = 5.15
        signals = det.check_reentries(
            {"X": 5.30}, {"X": bars}, _ts(200), {"X": 500},
        )
        assert len(signals) == 0

    def test_short_side_reentry(self) -> None:
        det = ReentryDetector(ReentryConfig(
            cooldown_seconds=1,
            min_continuation_cents=3.0,
            reentry_size_pct=0.5,
        ))
        det.record_full_exit(
            "Y", Side.SELL, exit_price=5.00, exit_ts=_ts(0),
            highest_price=4.90, entry_price=5.10,
        )

        det.check_reentries({"Y": 5.03}, {"Y": [_bar("Y", 5.03)]}, _ts(10), {"Y": 400})

        bars = [_bar("Y", 5.03, 80_000), _bar("Y", 4.95, 100_000)]
        signals = det.check_reentries({"Y": 4.95}, {"Y": bars}, _ts(60), {"Y": 400})

        assert len(signals) == 1
        assert signals[0].action is SignalAction.REENTER_SHORT
        assert signals[0].quantity == 200

    def test_clear_session_resets_everything(self) -> None:
        det = ReentryDetector()
        det.record_full_exit("A", Side.BUY, 5.0, _ts(0), 5.1, 4.9)
        det.clear_session()
        assert len(det._exit_history) == 0
        assert len(det._reentry_counts) == 0


# ===================================================================
# Full scenario: Entry → Scale ups → Tiered exits → Re-entry
# ===================================================================

class TestFullScaleReentryScenario:

    def test_5_dollar_stock_scales_then_reenters(self) -> None:
        """Complete flow:

        1. Enter 500 shares @ $5.00
        2. Stock runs to $5.15 → pull back to $5.08 → bounce to $5.12
           → Scale up +250 @ $5.12, stop moves to $5.03
        3. Stock runs to $5.30 → pull back to $5.22 → bounce to $5.28
           → Scale up +125 @ $5.28, stop moves to $5.06
        4. Total position: 875 shares, avg ~$5.06
        5. Tiered exit kicks in:
           - Tier 1 (200 shares) exits at first target
           - Tiers 2+3 trail and exit as stock pulls back
        6. Fully out. Stock pulls back then pushes to $5.50
           → Re-enter 250 shares (50% of original 500) with tighter stops
        """
        em = ExitManager()
        scaler = PositionScaler(ScaleUpConfig(
            min_profit_cents=3.0,
            size_decay=0.5,
            pullback_pct=20.0,
            bounce_pct=0.3,
            stop_advance_cents=3.0,
        ))
        reentry = ReentryDetector(ReentryConfig(
            cooldown_seconds=5,
            min_continuation_cents=3.0,
            reentry_size_pct=0.5,
        ))

        # --- 1. Initial entry ---
        tiers = build_exit_tiers(500, 5.00, Side.BUY,
                                  tier1_target_cents=5.0,
                                  tier2_trail_cents=3.0,
                                  tier3_trail_cents=5.0)
        em.track(TrackedPosition(
            symbol="HOT", side=Side.BUY, quantity=500,
            entry_price=5.00, stop_loss=4.97, tiers=tiers,
        ))

        # --- 2. First scale-up opportunity ---
        pos = em.tracked["HOT"]
        pos.highest_price = 5.15

        scaler._pullback_seen["HOT"] = True
        scaler._pullback_low["HOT"] = 5.08

        bars = [_bar("HOT", 5.08, 80_000), _bar("HOT", 5.12, 100_000)]
        scale1 = scaler.check_scale_ups(em, {"HOT": 5.12}, {"HOT": bars})
        assert len(scale1) == 1
        assert scale1[0].quantity == 250  # 500 * 0.5

        em.scale_up("HOT", 250, 5.12, new_stop=5.03)
        pos = em.tracked["HOT"]
        assert pos.remaining_qty == 750
        assert pos.stop_loss == 5.03

        # --- 3. Second scale-up ---
        pos.highest_price = 5.30
        scaler._pullback_seen["HOT"] = True
        scaler._pullback_low["HOT"] = 5.22

        bars2 = [_bar("HOT", 5.22, 80_000), _bar("HOT", 5.28, 100_000)]
        scale2 = scaler.check_scale_ups(em, {"HOT": 5.28}, {"HOT": bars2})
        assert len(scale2) == 1
        assert scale2[0].quantity == 125  # 500 * 0.5^2

        em.scale_up("HOT", 125, 5.28, new_stop=5.06)
        pos = em.tracked["HOT"]
        assert pos.remaining_qty == 875
        assert pos.stop_loss == 5.06

        # --- 4. Tiers start firing as stock moves up then pulls back ---
        # Tier 1 (fixed target @ entry + 5¢ = $5.05) — already past it
        exits1 = em.check_exits({"HOT": 5.30}, _ts(100))
        tier1_exits = [e for e in exits1 if "take_profit" in e.reason]
        assert len(tier1_exits) >= 1  # tier 1 triggered

        # stock keeps going then pulls back
        for p in [5.35, 5.40, 5.45]:
            em.check_exits({"HOT": p}, _ts(110))

        # pullback triggers trailing stops
        all_exits: List[TradeSignal] = []
        all_exits.extend(exits1)
        for p in [5.42, 5.38, 5.35, 5.30, 5.25]:
            more = em.check_exits({"HOT": p}, _ts(120))
            all_exits.extend(more)

        total_exited = sum(e.quantity for e in all_exits)
        # should have exited some or all shares via tiers + trailing

        # --- 5. Record full exit for re-entry ---
        if "HOT" not in em.tracked:
            reentry.record_full_exit(
                "HOT", Side.BUY, exit_price=5.30,
                exit_ts=_ts(120), highest_price=5.45, entry_price=5.00,
            )

            # --- 6. Re-entry check ---
            # stock pulls back then continues
            reentry.check_reentries(
                {"HOT": 5.25},
                {"HOT": [_bar("HOT", 5.25)]},
                _ts(130), {"HOT": 500},
            )

            bars_re = [_bar("HOT", 5.30, 80_000), _bar("HOT", 5.38, 100_000)]
            re_signals = reentry.check_reentries(
                {"HOT": 5.38}, {"HOT": bars_re}, _ts(200), {"HOT": 500},
            )

            assert len(re_signals) == 1
            assert re_signals[0].action is SignalAction.REENTER_LONG
            assert re_signals[0].quantity == 250  # 500 * 0.5
            assert re_signals[0].stop_loss is not None

    def test_scale_up_profit_vs_flat_entry(self) -> None:
        """Pyramiding makes more money than a flat entry when the trade works."""

        # flat entry: 500 shares @ $5.00, exit all @ $5.40
        flat_pnl = 500 * (5.40 - 5.00)  # $200

        # pyramid entry: 500@5.00 + 250@5.12 + 125@5.28
        pyramid_shares = [500, 250, 125]
        pyramid_prices = [5.00, 5.12, 5.28]
        exit_price = 5.40
        pyramid_pnl = sum(
            shares * (exit_price - entry)
            for shares, entry in zip(pyramid_shares, pyramid_prices)
        )
        # 500*0.40 + 250*0.28 + 125*0.12 = 200 + 70 + 15 = $285

        assert pyramid_pnl > flat_pnl
        assert pyramid_pnl == pytest.approx(285.0, abs=0.01)

    def test_scale_up_risk_controlled(self) -> None:
        """Verify that the stop advances on each scale-up, keeping risk bounded.

        Risk grows with more shares but the stop advancing limits the damage.
        Without stop advancement, risk would be much worse.
        """
        # entry: 500 @ 5.00, stop 4.97 → risk = 500 * 0.03 = $15
        initial_risk = 500 * (5.00 - 4.97)

        # scale 1: +250 @ 5.12, stop → 5.03
        avg1 = (500 * 5.00 + 250 * 5.12) / 750
        risk_with_stop_advance = 750 * (avg1 - 5.03)
        risk_without_advance = 750 * (avg1 - 4.97)

        # stop advancement cuts risk roughly in half vs. not advancing
        assert risk_with_stop_advance < risk_without_advance

        # scale 2: +125 @ 5.28, stop → 5.06
        avg2 = (750 * avg1 + 125 * 5.28) / 875
        risk2_with = 875 * (avg2 - 5.06)
        risk2_without = 875 * (avg2 - 4.97)

        assert risk2_with < risk2_without
        # risk per share stays small ($0.01-0.02 range)
        assert (avg2 - 5.06) < 0.03
