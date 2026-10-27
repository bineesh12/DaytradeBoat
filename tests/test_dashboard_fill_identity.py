from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from daytrading.dashboard.hub import DashboardHub
from daytrading.models import Fill, Side


def fill():
    return Fill('NCPL', Side.BUY, 500, 2.5847,
                datetime(2026, 10, 16, 12, 47, 7, 377728, timezone.utc))


def test_poll_and_submit_report_one_execution_and_enrich_strategy():
    hub = DashboardHub()
    hub.on_fill(fill())
    hub.on_fill(fill(), strategy='vwap_pullback')
    assert hub.total_trades == 1
    assert len(hub.trades) == 1
    assert hub.trades[0].strategy == 'vwap_pullback'


def test_real_partial_executions_remain_distinct():
    hub = DashboardHub()
    hub.on_fill(fill())
    hub.on_fill(replace(fill(), quantity=200))
    hub.on_fill(replace(fill(), ts=fill().ts+timedelta(microseconds=1)))
    assert hub.total_trades == len(hub.trades) == 3


def test_same_instant_other_timezone_is_same_execution():
    hub = DashboardHub()
    hub.on_fill(fill())
    hub.on_fill(replace(fill(), ts=fill().ts.astimezone(timezone(timedelta(hours=-4)))))
    assert hub.total_trades == 1


def test_concurrent_callbacks_count_once():
    hub = DashboardHub()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: hub.on_fill(fill()), range(20)))
    assert hub.total_trades == len(hub.trades) == 1


def test_daily_reset_clears_reporting_identity():
    hub = DashboardHub()
    hub.on_fill(fill())
    hub.reset_daily_overview()
    hub.on_fill(fill())
    assert hub.total_trades == len(hub.trades) == 1
