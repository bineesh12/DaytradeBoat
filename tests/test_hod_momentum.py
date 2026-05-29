"""Tests for HOD Momentum bar scanner and alert store."""

from datetime import datetime, timezone

from daytrading.models import Bar
from daytrading.scanner.hod_momentum.alert_store import HODAlertStore
from daytrading.scanner.hod_momentum.bar_scanner import HODMomentumScanner


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
