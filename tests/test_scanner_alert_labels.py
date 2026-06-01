"""Tests for HOD Momentum scanner alert labels."""

from datetime import datetime, timezone

from daytrading.models import Bar
from daytrading.scanner.scanner_alert_labels import (
    alert_row_class,
    classify_hod_momentum_alerts,
)


def _bar(i: int, close: float, vol: float = 50_000) -> Bar:
    ts = datetime(2026, 5, 18, 14, i, 0, tzinfo=timezone.utc)
    return Bar(
        symbol="TST",
        ts=ts,
        open=close - 0.05,
        high=close + 0.05,
        low=close - 0.1,
        close=close,
        volume=vol,
    )


def test_low_float_high_rel_vol_alert() -> None:
    alerts = classify_hod_momentum_alerts(
        price=5.0,
        float_shares=143_000,
        rel_vol=829.0,
        bar_rvol=10.0,
        change_session_pct=60.0,
        change_5m_pct=6.0,
        change_10m_pct=12.0,
    )
    assert "Low Float - High Rel Vol" in alerts
    assert "Squeeze - Up 5% in 5min" in alerts
    assert "Squeeze - Up 10% in 10min" in alerts


def test_float_over_20m_excluded() -> None:
    alerts = classify_hod_momentum_alerts(
        price=29.60,
        float_shares=87_570_000,
        rel_vol=12.5,
        bar_rvol=2.0,
        change_session_pct=8.0,
        change_5m_pct=2.0,
        change_10m_pct=5.0,
    )
    assert alerts == []


def test_hod_breakout_and_reclaim() -> None:
    alerts = classify_hod_momentum_alerts(
        price=8.0,
        float_shares=5_000_000,
        rel_vol=2.0,
        bar_rvol=1.0,
        change_session_pct=10.0,
        change_5m_pct=1.0,
        change_10m_pct=2.0,
        include_hod_breakout=True,
        include_hod_reclaim=True,
    )
    assert "New HOD Breakout" in alerts
    assert "HOD Reclaim" in alerts


def test_intraday_low_reclaim_is_entry_gate_alert() -> None:
    alerts = classify_hod_momentum_alerts(
        price=2.65,
        float_shares=5_000_000,
        rel_vol=1.0,
        bar_rvol=4.5,
        change_session_pct=-2.0,
        change_5m_pct=4.2,
        change_10m_pct=8.5,
        include_intraday_low_reclaim=True,
    )

    assert alerts == ["Intraday Low Reclaim"]


def test_alert_row_classes() -> None:
    assert alert_row_class("New HOD Breakout") == "hod-breakout"
    assert alert_row_class("HOD Reclaim") == "hod-reclaim"
    assert alert_row_class("Intraday Low Reclaim") == "hod-reclaim"
    assert alert_row_class("Low Float - High Rel Vol") == "hod-low-float"
