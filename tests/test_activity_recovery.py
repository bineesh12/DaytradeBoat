from datetime import datetime, timedelta, timezone
import pytest
from daytrading.risk.guards import HaltTracker

NOW = datetime(2026, 10, 15, 18, tzinfo=timezone.utc)


def at(s):
    return NOW + timedelta(seconds=s)


def quiet():
    h = HaltTracker()
    h.update_price("BENF", 1.1, at(0))
    h.update_price("BENF", 1.1, at(40))
    return h


def test_fresh_sustained_trades_clear_inferred_flag_with_full_cooldown():
    h = quiet()
    assert h.check("BENF", at(40)).startswith("LOW ACTIVITY")
    h.update_price("BENF", 1.1, at(41))
    h.update_price("BENF", 1.1, at(42))
    assert h.check("BENF", at(42)).startswith("post-halt cooldown: 120s")
    assert "BENF" not in h.halted_symbols
    assert h.check("BENF", at(161)) is not None
    assert h.check("BENF", at(162)) is None


def test_explicit_halt_is_not_cleared_by_trades_or_large_gap():
    h = quiet()
    h.mark_halted("BENF", at(40))
    for t, p in ((41, 1.11), (42, 1.12), (43, 1.12), (80, 1.5)):
        h.update_price("BENF", p, at(t))
    assert h.check("BENF", at(80)).startswith("HALTED")
    h.mark_resumed("BENF", at(81))
    assert "cooldown" in h.check("BENF", at(82))


@pytest.mark.parametrize(
    "offsets", [(40, 40, 40), (40, 40.1, 40.2), (41, 40, 39), (50, 65, 80)]
)
def test_duplicates_bursts_out_of_order_and_sparse_trades_do_not_recover(offsets):
    h = quiet()
    for t in offsets:
        h.update_price("BENF", 1.1, at(t))
    assert "LOW ACTIVITY" in h.check("BENF", at(max(offsets)))


def test_stale_and_future_batches_do_not_prove_current_recovery():
    h = quiet()
    for t in (41, 42, 43):
        h.update_price("BENF", 1.1, at(t))
    assert "LOW ACTIVITY" in h.check("BENF", at(100))
    assert "LOW ACTIVITY" in h.check("BENF", at(20))
    for t in (101, 102, 103):
        h.update_price("BENF", 1.1, at(t))
    assert "cooldown" in h.check("BENF", at(103))


def test_invalid_prices_and_timestamp_cannot_advance_last_price():
    h = quiet()
    before = dict(h._last_price)
    for p in (0, -1, float("nan"), float("inf")):
        h.update_price("BENF", p, at(41))
    h.update_price("BENF", 1.1, datetime(2026, 10, 15, 18))
    assert h._last_price == before


def test_unknown_legacy_flags_stay_blocked_until_proven_and_get_new_evidence():
    h = HaltTracker()
    h._halted["BENF"] = at(0)
    h._last_price["BENF"] = (1.1, at(40))
    for t in (41, 42, 43):
        h.update_price("BENF", 1.1, at(t))
    assert "HALTED" in h.check("BENF", at(43))
    with pytest.raises(ValueError):
        h.adopt_legacy_low_activity("BENF", at(1))
    h.adopt_legacy_low_activity("BENF", at(0))
    assert "LOW ACTIVITY" in h.check("BENF", at(43))
    for t in (44, 45, 46):
        h.update_price("BENF", 1.1, at(t))
    assert "cooldown" in h.check("BENF", at(46))


def test_live_arrival_check_discards_delayed_and_future_ticks():
    h = quiet()
    before = dict(h._last_price)
    h.update_price("BENF", 1.1, at(41), observed_at=at(100))
    h.update_price("BENF", 1.1, at(110), observed_at=at(100))
    assert h._last_price == before
    for t in (101, 102, 103):
        h.update_price("BENF", 1.1, at(t), observed_at=at(t + 0.1))
    assert h.activity_snapshot(at(104))["BENF"]["source"] == "resume_cooldown"
