"""Tests for HOD Momentum bar scanner and alert store."""

from datetime import datetime, timezone

from daytrading.models import Bar
from daytrading.scanner.hod_momentum.alert_store import HODAlertStore
from daytrading.scanner.hod_momentum.bar_scanner import HODMomentumScanner
from daytrading.scanner.hod_momentum.prior_day import PriorDayStats


def _bar(i: int, close: float, vol: float = 100_000) -> Bar:
    ts = datetime(2026, 5, 18, 14, i, 0, tzinfo=timezone.utc)
    return Bar(
        symbol="CISS",
        ts=ts,
        open=close - 0.1,
        high=close + 0.2,
        low=close - 0.2,
        close=close,
        volume=vol,
    )


class _FloatChecker:
    def get_float(self, symbol: str):
        if symbol == "MTEK":
            return 5_000_000
        return 9_280_000 if symbol in ("CISS", "AIIO") else None

    def get_float_cached(self, symbol: str):
        return self.get_float(symbol)


def test_bar_scanner_adds_low_float_alert() -> None:
    store = HODAlertStore(max_rows=50)
    rows: list = []
    store.set_on_change(rows.extend)

    scanner = HODMomentumScanner(
        store,
        float_checker=_FloatChecker(),
        min_price=2.0,
        max_price=20.0,
        min_session_change_pct=5.0,
        min_day_volume=50_000,
    )

    bars = [_bar(i, 4.0 + i * 0.3) for i in range(20)]
    scanner.scan(
        {"CISS": bars},
        rel_vols={"CISS": 5.0},
        verified_symbols={"CISS"},
    )

    assert len(rows) >= 1
    assert any(r["symbol"] == "CISS" for r in rows)
    assert any(r["source"] == "bar" for r in rows)


def _aiio_bar(
    i: int,
    o: float,
    h: float,
    l: float,
    c: float,
    vol: float = 80_000,
) -> Bar:
    ts = datetime(2026, 5, 18, 14, i, 0, tzinfo=timezone.utc)
    return Bar(
        symbol="AIIO",
        ts=ts,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=vol,
    )


def test_no_hod_reclaim_below_session_high() -> None:
    """Pullback reclaim under true session HOD must not label HOD Reclaim."""
    store = HODAlertStore(max_rows=50)
    rows: list = []
    store.set_on_change(rows.extend)

    scanner = HODMomentumScanner(
        store,
        float_checker=_FloatChecker(),
        min_price=2.0,
        max_price=20.0,
        min_session_change_pct=5.0,
        min_day_volume=50_000,
    )

    bars = [
        _aiio_bar(0, 4.0, 4.2, 3.9, 4.1),
        _aiio_bar(1, 4.1, 6.5, 4.0, 6.4),  # session HOD 6.50
        _aiio_bar(2, 6.3, 6.4, 5.0, 5.2),
        _aiio_bar(3, 5.2, 5.4, 5.0, 5.1),
        _aiio_bar(4, 5.1, 5.3, 5.0, 5.15),
        _aiio_bar(5, 5.15, 5.4, 5.05, 5.2),
        _aiio_bar(6, 5.2, 5.5, 5.1, 5.3),
        _aiio_bar(7, 5.3, 5.6, 5.2, 5.4),
        _aiio_bar(8, 5.4, 5.7, 5.3, 5.5),
        _aiio_bar(9, 5.5, 5.88, 5.4, 5.88),  # green but below 6.50 HOD
    ]
    scanner.scan({"AIIO": bars}, rel_vols={"AIIO": 10.0})

    names = [r["alert_name"] for r in rows]
    assert "HOD Reclaim" not in names
    assert "New HOD Breakout" not in names


def test_intraday_low_reclaim_alert_can_promote_negative_day_mover() -> None:
    """A strong pop from session low should be watchable even below prior-day high."""
    store = HODAlertStore(max_rows=50)
    rows: list = []
    store.set_on_change(rows.extend)

    scanner = HODMomentumScanner(
        store,
        float_checker=_FloatChecker(),
        min_price=2.0,
        max_price=20.0,
        min_session_change_pct=5.0,
        min_day_volume=50_000,
    )

    bars = [
        _aiio_bar(0, 2.50, 2.52, 2.48, 2.50, 80_000),
        _aiio_bar(1, 2.50, 2.51, 2.35, 2.38, 90_000),
        _aiio_bar(2, 2.38, 2.42, 2.30, 2.34, 100_000),
        _aiio_bar(3, 2.34, 2.37, 2.28, 2.31, 100_000),
        _aiio_bar(4, 2.31, 2.36, 2.25, 2.28, 100_000),
        _aiio_bar(5, 2.28, 2.39, 2.26, 2.36, 100_000),
        _aiio_bar(6, 2.36, 2.46, 2.35, 2.44, 110_000),
        _aiio_bar(7, 2.44, 2.54, 2.43, 2.52, 120_000),
        _aiio_bar(8, 2.52, 2.61, 2.50, 2.59, 140_000),
        _aiio_bar(9, 2.59, 2.68, 2.58, 2.65, 180_000),
    ]

    scanner.scan(
        {"AIIO": bars},
        bars_5m={"AIIO": [_aiio_bar(10, 2.52, 2.68, 2.50, 2.65, 250_000)]},
        rel_vols={"AIIO": 1.0},
        prior_day_stats={
            "AIIO": PriorDayStats(prior_close=2.71, prior_high=2.90),
        },
    )

    assert any(r["alert_name"] == "Intraday Low Reclaim" for r in rows)


def test_sub2_extreme_momentum_can_alert_under_standard_price_band() -> None:
    store = HODAlertStore(max_rows=50)
    rows: list = []
    store.set_on_change(rows.extend)

    scanner = HODMomentumScanner(
        store,
        float_checker=_FloatChecker(),
        min_price=2.0,
        max_price=20.0,
        min_session_change_pct=5.0,
        min_day_volume=50_000,
        sub2_enabled=True,
        sub2_min_price=1.0,
        sub2_max_price=2.0,
        sub2_min_session_change_pct=10.0,
        sub2_min_day_volume=1_000_000,
        sub2_max_float=10_000_000,
    )

    bars = [
        Bar("MTEK", datetime(2026, 5, 18, 14, i, 0, tzinfo=timezone.utc),
            1.18 + i * 0.07, 1.24 + i * 0.07, 1.16 + i * 0.07,
            1.22 + i * 0.07, 120_000 + i * 20_000)
        for i in range(10)
    ]

    scanner.scan(
        {"MTEK": bars},
        rel_vols={"MTEK": 5.0},
        prior_day_stats={
            "MTEK": PriorDayStats(prior_close=1.20, prior_high=1.30),
        },
        is_premarket=True,
    )

    assert rows
    assert all(r["symbol"] == "MTEK" for r in rows)
    assert any(r["alert_name"] in {
        "New HOD Breakout",
        "Low Float - High Rel Vol",
        "Squeeze - Up 10% in 10min",
    } for r in rows)
