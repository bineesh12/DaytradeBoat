"""Tests for prior-day HOD vs today-only HOD classification."""

from datetime import datetime, timezone

from daytrading.models import Bar
from daytrading.scanner.hod_momentum.alert_store import HODAlertStore
from daytrading.scanner.hod_momentum.bar_scanner import HODMomentumScanner
from daytrading.scanner.hod_momentum.hod_alert_logic import classify_hod_breakout_alerts
from daytrading.scanner.hod_momentum.prior_day import PriorDayStats


def test_classify_breaks_yesterday_vs_today_only() -> None:
    new_hod, today_hod = classify_hod_breakout_alerts(
        6.55, 6.40, 6.50, 6.45, prior_day_high=8.0, require_break_prior_day=True,
    )
    assert new_hod is False
    assert today_hod is True

    new_hod2, today_hod2 = classify_hod_breakout_alerts(
        8.10, 7.90, 8.05, 7.95, prior_day_high=8.0, require_break_prior_day=True,
    )
    assert new_hod2 is True
    assert today_hod2 is False


def _bar(i: int, o: float, h: float, c: float, vol: float = 80_000) -> Bar:
    ts = datetime(2026, 5, 18, 14, i, 0, tzinfo=timezone.utc)
    return Bar(
        symbol="NXTS",
        ts=ts,
        open=o,
        high=h,
        low=o - 0.1,
        close=c,
        volume=vol,
    )


class _FloatChecker:
    def get_float(self, symbol: str):
        return 9_000_000 if symbol == "NXTS" else None

    def get_float_cached(self, symbol: str):
        return self.get_float(symbol)


def test_nxts_gets_today_hod_not_new_hod() -> None:
    store = HODAlertStore(max_rows=50)
    rows: list = []
    store.set_on_change(rows.extend)

    scanner = HODMomentumScanner(
        store,
        float_checker=_FloatChecker(),
        min_session_change_pct=5.0,
        min_day_volume=50_000,
        require_break_prior_day_high=True,
    )

    bars = [
        _bar(0, 5.0, 5.1, 5.05),
        _bar(1, 5.05, 5.2, 5.15),
        _bar(2, 5.15, 5.25, 5.2),
        _bar(3, 5.2, 5.35, 5.3),
        _bar(4, 5.3, 5.45, 5.4),
        _bar(5, 5.4, 5.55, 5.5),
        _bar(6, 5.5, 5.65, 5.6),
        _bar(7, 5.6, 5.75, 5.7),
        _bar(8, 5.7, 5.88, 5.88),
        _bar(9, 5.88, 5.90, 5.89),
    ]
    prior = PriorDayStats(prior_close=7.0, prior_high=8.0)
    scanner.scan(
        {"NXTS": bars},
        rel_vols={"NXTS": 10.0},
        prior_day_stats={"NXTS": prior},
    )

    names = [r["alert_name"] for r in rows]
    # Warrior-style: price ($5.89) is below prior day high ($8.00),
    # so ALL alerts are blocked — no entries below yesterday's resistance.
    assert "New HOD Breakout" not in names
    assert len(names) == 0
