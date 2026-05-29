"""Tests for HODAlertStore sorting."""

from datetime import datetime, timezone, timedelta

from daytrading.scanner.hod_momentum.alert_store import HODAlertStore
from daytrading.scanner.hod_momentum.models import HODAlertRow


def test_alerts_sorted_newest_first() -> None:
    store = HODAlertStore(max_rows=50, ttl_minutes=10)
    now = datetime.now(timezone.utc)
    t1 = (now - timedelta(minutes=3)).isoformat()
    t2 = (now - timedelta(minutes=1)).isoformat()
    t3 = (now - timedelta(minutes=2)).isoformat()

    store.add(HODAlertRow(
        symbol="AAA",
        time=t1,
        price=5.0,
        alert_name="New HOD Breakout",
        source="tick",
    ))
    store.add(HODAlertRow(
        symbol="BBB",
        time=t2,
        price=3.0,
        alert_name="New HOD Breakout",
        source="tick",
    ))
    store.add(HODAlertRow(
        symbol="CCC",
        time=t3,
        price=4.0,
        alert_name="HOD Reclaim",
        source="bar",
    ))

    snap = store.snapshot()
    assert snap[0]["symbol"] == "BBB"
    assert snap[1]["symbol"] == "CCC"
    assert snap[2]["symbol"] == "AAA"
