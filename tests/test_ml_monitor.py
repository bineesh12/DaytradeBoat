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

    # Shadow results: 1 correct, 5 wrong = 16.7% accuracy.
    m._stats.shadow_correct = 1
    m._stats.shadow_wrong = 5

    m._check_auto_disable()

    assert m.stats.model_disabled is True
    assert m.is_model_enabled is False
    assert "shadow accuracy" in m.stats.disable_reason


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
