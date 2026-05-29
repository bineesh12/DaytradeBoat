"""Tests for Former Momo Stock HOD alerts ($20+)."""

from datetime import datetime, timedelta, timezone

from daytrading.models import Bar
from daytrading.scanner.hod_momentum.alert_store import HODAlertStore
from daytrading.scanner.hod_momentum.former_momo_scanner import FormerMomoScanner


def _bar(i: int, close: float, vol: float = 200_000) -> Bar:
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
        minutes=20 - i,
    )
    return Bar(
        symbol="OSCR",
        ts=ts,
        open=close - 0.5,
        high=close + 0.2,
        low=close - 0.5,
        close=close,
        volume=vol,
    )


class _FloatChecker:
    def __init__(self, *, cached: bool = True) -> None:
        self._cached = cached

    def get_float(self, symbol: str):
        return 150_000_000 if symbol == "OSCR" else None

    def get_float_cached(self, symbol: str):
        if not self._cached:
            return None
        return self.get_float(symbol)


def test_former_momo_without_prior_close_uses_session_change() -> None:
    store = HODAlertStore(max_rows=50)
    rows: list = []
    store.set_on_change(rows.extend)

    scanner = FormerMomoScanner(
        store,
        float_checker=_FloatChecker(),
        min_change_from_close_pct=3.0,
        min_day_volume=500_000,
    )
    # Open ~20, close 24.5 => ~22.5% session change, no prior stats
    bars = [_bar(i, 20.0 + i * 0.25, vol=60_000) for i in range(20)]
    scanner.scan({"OSCR": bars}, prior_day_stats={})

    assert any(r["symbol"] == "OSCR" and r["alert_name"] == "Former Momo Stock" for r in rows)


def test_former_momo_requires_cached_float() -> None:
    store = HODAlertStore(max_rows=50)
    rows: list = []
    store.set_on_change(rows.extend)

    scanner = FormerMomoScanner(
        store,
        float_checker=_FloatChecker(cached=False),
        min_change_from_close_pct=3.0,
        min_day_volume=500_000,
    )
    bars = [_bar(i, 20.0 + i * 0.25, vol=60_000) for i in range(20)]
    scanner.scan({"OSCR": bars}, prior_day_stats={})

    assert rows == []
