"""Document and verify each HOD Momentum alert label condition."""

from datetime import datetime, timezone

from daytrading.models import Bar
from daytrading.scanner.scanner_alert_labels import (
    bar_volume_surge,
    change_pct_10m,
    change_pct_5m,
    classify_hod_momentum_alerts,
)


def _1m(i: int, o: float, h: float, c: float, vol: float = 50_000) -> Bar:
    ts = datetime(2026, 5, 18, 14, i, 0, tzinfo=timezone.utc)
    return Bar(
        symbol="T",
        ts=ts,
        open=o,
        high=h,
        low=o - 0.1,
        close=c,
        volume=vol,
    )


def test_new_hod_breakout_flag_only_when_requested() -> None:
    alerts = classify_hod_momentum_alerts(
        price=5.0,
        float_shares=1_000_000,
        rel_vol=1.0,
        bar_rvol=1.0,
        change_session_pct=10.0,
        change_5m_pct=1.0,
        change_10m_pct=1.0,
        include_hod_breakout=True,
    )
    assert alerts == ["New HOD Breakout"]


def test_today_hod_breakout_flag() -> None:
    alerts = classify_hod_momentum_alerts(
        price=5.0,
        float_shares=1_000_000,
        rel_vol=1.0,
        bar_rvol=1.0,
        change_session_pct=10.0,
        change_5m_pct=1.0,
        change_10m_pct=1.0,
        include_today_hod_breakout=True,
    )
    assert alerts == ["Today HOD Breakout"]


def test_hod_reclaim_flag_only_when_requested() -> None:
    alerts = classify_hod_momentum_alerts(
        price=5.0,
        float_shares=1_000_000,
        rel_vol=1.0,
        bar_rvol=1.0,
        change_session_pct=10.0,
        change_5m_pct=1.0,
        change_10m_pct=1.0,
        include_hod_reclaim=True,
    )
    assert alerts == ["HOD Reclaim"]


def test_low_float_high_rel_vol() -> None:
    alerts = classify_hod_momentum_alerts(
        price=5.0,
        float_shares=5_000_000,
        rel_vol=3.0,
        bar_rvol=1.0,
        change_session_pct=2.0,
        change_5m_pct=None,
        change_10m_pct=None,
    )
    assert "Low Float - High Rel Vol" in alerts
    alerts_low = classify_hod_momentum_alerts(
        price=5.0,
        float_shares=5_000_000,
        rel_vol=2.99,
        bar_rvol=1.0,
        change_session_pct=2.0,
        change_5m_pct=None,
        change_10m_pct=None,
    )
    assert "Low Float - High Rel Vol" not in alerts_low


def test_squeeze_5m_from_5m_bar() -> None:
    b5 = [_1m(0, 4.0, 4.5, 4.5)]
    ch5 = change_pct_5m(b5)
    assert ch5 is not None and ch5 >= 5.0
    alerts = classify_hod_momentum_alerts(
        price=5.0,
        float_shares=1_000_000,
        rel_vol=0.0,
        bar_rvol=1.0,
        change_session_pct=2.0,
        change_5m_pct=ch5,
        change_10m_pct=None,
    )
    assert "Squeeze - Up 5% in 5min" in alerts


def test_squeeze_10m_from_last_ten_1m_bars() -> None:
    bars = [_1m(i, 4.0, 4.1 + i * 0.05, 4.0 + i * 0.1) for i in range(10)]
    ch10 = change_pct_10m(bars)
    assert ch10 is not None and ch10 >= 10.0
    alerts = classify_hod_momentum_alerts(
        price=bars[-1].close,
        float_shares=1_000_000,
        rel_vol=0.0,
        bar_rvol=1.0,
        change_session_pct=2.0,
        change_5m_pct=None,
        change_10m_pct=ch10,
    )
    assert "Squeeze - Up 10% in 10min" in alerts


def test_squeeze_5m_from_bar_volume_surge() -> None:
    quiet = [_1m(i, 5.0, 5.1, 5.05, vol=10_000) for i in range(10)]
    hot = [_1m(10 + i, 5.1, 5.2, 5.15, vol=100_000) for i in range(5)]
    today = quiet + hot
    assert bar_volume_surge(today) >= 5.0
    alerts = classify_hod_momentum_alerts(
        price=5.15,
        float_shares=1_000_000,
        rel_vol=0.0,
        bar_rvol=bar_volume_surge(today),
        change_session_pct=3.0,
        change_5m_pct=None,
        change_10m_pct=None,
    )
    assert "Squeeze - Up 5% in 5min" in alerts


def test_no_alerts_without_float() -> None:
    assert classify_hod_momentum_alerts(
        price=5.0,
        float_shares=None,
        rel_vol=10.0,
        bar_rvol=10.0,
        change_session_pct=20.0,
        change_5m_pct=None,
        change_10m_pct=None,
        include_hod_breakout=True,
    ) == []
