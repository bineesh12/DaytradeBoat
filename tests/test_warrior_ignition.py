from daytrading.strategy.warrior_ignition import (
    IgnitionSignal,
    ignition_suppression_reason,
)
from daytrading.runner import AlpacaRunner


def _signal(price: float, day_move: float, near_hod: float = 0.98) -> IgnitionSignal:
    return IgnitionSignal(
        detected=True,
        conviction=0.45,
        entry_ref=price,
        stop=price * 0.94,
        base_high=price * 0.98,
        features={"day_move": day_move, "near_hod": near_hod},
    )


def test_ignition_suppresses_post_peak_chop_after_failed_entry() -> None:
    reason = ignition_suppression_reason(
        _signal(3.16, 0.082),
        failed_entries=1,
        peak_price=3.41,
        peak_day_move=0.227,
    )

    assert "post-peak chop" in reason


def test_ignition_first_fire_and_fresh_high_are_not_suppressed() -> None:
    first_fire = ignition_suppression_reason(
        _signal(3.41, 0.227),
        failed_entries=0,
        peak_price=3.41,
        peak_day_move=0.227,
    )
    fresh_recovery = ignition_suppression_reason(
        _signal(3.52, 0.24),
        failed_entries=1,
        peak_price=3.41,
        peak_day_move=0.227,
    )

    assert first_fire == ""
    assert fresh_recovery == ""


def test_ignition_blocks_after_two_failed_entries() -> None:
    reason = ignition_suppression_reason(
        _signal(3.60, 0.25),
        failed_entries=2,
        peak_price=3.41,
        peak_day_move=0.227,
    )

    assert "2 failed ignitions" in reason


def test_ignition_failure_count_uses_completed_net_trade_pnl() -> None:
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._warrior_ignition_failed_entries = {}
    runner._warrior_ignition_trade_pnl = {}

    runner._record_warrior_ignition_exit("SKYQ", 10.0, "partial_target", completed=False)
    runner._record_warrior_ignition_exit("SKYQ", -25.0, "trailing_stop", completed=True)

    assert runner._warrior_ignition_failed_entries["SKYQ"] == 1

    runner._record_warrior_ignition_exit("SKYQ", 10.0, "partial_target", completed=False)
    runner._record_warrior_ignition_exit("SKYQ", 5.0, "final_exit", completed=True)

    assert "SKYQ" not in runner._warrior_ignition_failed_entries
