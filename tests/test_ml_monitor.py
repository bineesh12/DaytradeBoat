"""Tests for ML monitor: shadow mode and auto-disable."""

import time
from unittest.mock import patch

from daytrading.ml.monitor import MLMonitor


def test_shadow_mode_correct_rejection():
    """If ML rejects and price goes down, shadow marks it correct."""
    m = MLMonitor()
    m.record_ml_rejection("TEST", 10.0, 0.30, 70)

    # Price drops
    m.update_price("TEST", 9.50)

    # Fast-forward: override the reject_time to simulate 5 min passing
    m._shadow_entries[0].reject_time = time.time() - 301
    m.check_shadow_outcomes()

    assert m.stats.shadow_correct == 1
    assert m.stats.shadow_wrong == 0


def test_shadow_mode_wrong_rejection():
    """If ML rejects but price goes up, shadow marks it wrong."""
    m = MLMonitor()
    m.record_ml_rejection("TEST", 10.0, 0.30, 70)

    # Price goes up
    m.update_price("TEST", 11.00)

    m._shadow_entries[0].reject_time = time.time() - 301
    m.check_shadow_outcomes()

    assert m.stats.shadow_correct == 0
    assert m.stats.shadow_wrong == 1


def test_shadow_mode_marks_intraperiod_scalp_run_as_wrong_rejection():
    """If ML rejects, then price gives a scalp and fades, ML was still wrong."""
    m = MLMonitor()
    m.record_ml_rejection("VSME", 4.00, 0.08, 95)

    m.update_price("VSME", 4.20)
    m.update_price("VSME", 3.95)

    m._shadow_entries[0].reject_time = time.time() - 301
    m.check_shadow_outcomes()

    assert m.stats.shadow_correct == 0
    assert m.stats.shadow_wrong == 1


def test_elite_false_rejects_disable_model_quickly():
    """Two high-score rejects that run 8%+ disable ML via the monitor."""
    m = MLMonitor()

    m.record_ml_rejection("VSME", 3.30, 0.08, 95)
    m.update_price("VSME", 3.57)
    m._shadow_entries[0].reject_time = time.time() - 301
    m.check_shadow_outcomes()

    assert m.is_model_enabled is True
    assert m.stats.elite_false_rejects == 1

    m.record_ml_rejection("WCT", 5.00, 0.09, 105)
    m.update_price("WCT", 5.45)
    m._shadow_entries[0].reject_time = time.time() - 301
    m.check_shadow_outcomes()

    assert m.is_model_enabled is False
    assert m.stats.model_disabled is True
    assert "elite false ML rejects" in m.stats.disable_reason


def test_ordinary_small_wrong_reject_does_not_fast_disable_model():
    """Fast disable is only for high-score rejects with a large usable move."""
    m = MLMonitor()

    for i in range(5):
        symbol = f"SYM{i}"
        m.record_ml_rejection(symbol, 10.0, 0.20, 70)
        m.update_price(symbol, 10.20)
        m._shadow_entries[0].reject_time = time.time() - 301
        m.check_shadow_outcomes()

    assert m.stats.shadow_wrong == 5
    assert m.stats.elite_false_rejects == 0
    assert m.is_model_enabled is True


def test_auto_disable_high_rejection_rate():
    """Model is disabled if rejection rate exceeds the current threshold."""
    m = MLMonitor()

    # 24 rejections, 1 pass = 96% rejection rate with 25 scored samples.
    for i in range(24):
        m.record_ml_rejection(f"SYM{i}", 10.0, 0.30, 70)
    m.record_entry_passed()

    m._check_auto_disable()

    assert m.stats.model_disabled is True
    assert m.is_model_enabled is False
    assert "rejection rate" in m.stats.disable_reason


def test_auto_disable_low_shadow_accuracy():
    """Model is disabled if shadow accuracy drops below current threshold."""
    m = MLMonitor()

    # Need at least MIN_SAMPLES_FOR_DISABLE scored.
    for i in range(25):
        m.record_entry_passed()

    # Shadow results: 4 correct, 21 wrong = 16% accuracy.
    m._stats.shadow_correct = 4
    m._stats.shadow_wrong = 21

    m._check_auto_disable()

    assert m.stats.model_disabled is True
    assert m.is_model_enabled is False
    assert "shadow accuracy" in m.stats.disable_reason


def test_no_disable_with_few_shadow_outcomes():
    """Shadow accuracy should not disable ML before enough shadow labels exist."""
    m = MLMonitor()

    # Enough scored entries overall, but only 15 shadow outcomes. This was too
    # noisy and caused premature live disables like 4 correct / 11 wrong.
    for i in range(25):
        m.record_entry_passed()
    m._stats.shadow_correct = 4
    m._stats.shadow_wrong = 11

    m._check_auto_disable()

    assert m.stats.model_disabled is False
    assert m.is_model_enabled is True


def test_no_disable_with_few_samples():
    """Model should not be disabled with insufficient data."""
    m = MLMonitor()

    # Only 24 samples (below threshold of 25), despite high rejection rate.
    for i in range(23):
        m.record_ml_rejection(f"SYM{i}", 10.0, 0.30, 70)
    m.record_entry_passed()

    m._check_auto_disable()

    assert m.stats.model_disabled is False
    assert m.is_model_enabled is True


def test_daily_reset_re_enables():
    """Daily reset re-enables a disabled model."""
    m = MLMonitor()
    m._model_enabled = False
    m._stats.model_disabled = True
    m._stats.entries_passed = 5

    m.reset_daily()

    assert m.is_model_enabled is True
    assert m.stats.entries_passed == 0


def test_rule_rejections_are_grouped_by_symbol_and_reason():
    """Repeated same-symbol rule rejects should not inflate dashboard stats."""
    m = MLMonitor()

    m.record_rule_rejection("ANY", "stale data (300s old, max=300s)")
    m.record_rule_rejection("ANY", "stale data (450s old, max=300s)")
    m.record_rule_rejection("DXST", "late pullback too far from HOD 18.1% (max 10.0%)")
    m.record_rule_rejection("DXST", "late pullback too far from HOD 21.4% (max 10.0%)")
    m.record_rule_rejection("DXST", "below VWAP (4.90 < 5.10)")

    assert m.stats.entries_rejected_by_rules == 3

    m.reset_daily()
    m.record_rule_rejection("ANY", "stale data (700s old, max=300s)")

    assert m.stats.entries_rejected_by_rules == 1


def test_duplicate_ml_rejections_do_not_inflate_disable_stats():
    """Repeated same-setup ML rejects should count once for dashboard stats."""
    m = MLMonitor()

    for _ in range(10):
        m.record_ml_rejection("GNTA", 2.31, 0.17, 70)

    assert m.stats.entries_rejected_by_ml == 1
    assert len(m._shadow_entries) == 1


def test_soft_pass_shadow_does_not_count_as_ml_block():
    """Strong rule-score soft passes learn in shadow without counting as blocks."""
    m = MLMonitor()

    m.record_ml_rejection("ASTI", 8.75, 0.299, 100, counted=False)

    assert m.stats.entries_rejected_by_ml == 0
    assert len(m._shadow_entries) == 1


def test_stats_to_dict():
    """Stats dict has all expected fields."""
    m = MLMonitor()
    m.record_entry_passed()
    m.record_entry_passed()
    m.record_ml_rejection("A", 10.0, 0.35, 72)

    d = m.stats.to_dict()
    assert d["entries_passed"] == 2
    assert d["entries_rejected_by_ml"] == 1
    assert d["rejection_rate_pct"] == 33.3
    assert d["model_disabled"] is False
