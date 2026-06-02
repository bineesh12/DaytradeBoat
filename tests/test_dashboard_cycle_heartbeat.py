from __future__ import annotations

from daytrading.dashboard.hub import DashboardHub


def test_cycle_heartbeat_updates_snapshot_cycle_count() -> None:
    hub = DashboardHub()

    hub.on_cycle_heartbeat(7, "no bars yet")

    snap = hub.snapshot()
    assert snap["stats"]["cycle_count"] == 7


def test_daily_overview_reset_clears_session_stats() -> None:
    hub = DashboardHub()
    hub.total_trades = 3
    hub.winning_trades = 2
    hub.losing_trades = 1
    hub.total_pnl = 42.0
    hub.total_scan_hits = 9
    hub.total_signals = 4
    hub.total_rejected = 2
    hub.cycle_count = 88
    hub.ai_analysis = {"blocked_symbols": {"ABTS": "old day"}}
    hub.pnl_history.append({"ts": "old", "pnl": 42.0})

    hub.reset_daily_overview()
    snap = hub.snapshot()

    assert snap["stats"]["total_trades"] == 0
    assert snap["stats"]["winning_trades"] == 0
    assert snap["stats"]["losing_trades"] == 0
    assert snap["stats"]["total_pnl"] == 0.0
    assert snap["stats"]["total_scan_hits"] == 0
    assert snap["stats"]["total_signals"] == 0
    assert snap["stats"]["total_rejected"] == 0
    assert snap["stats"]["cycle_count"] == 0
    assert snap["pnl_history"] == []
    assert snap["ai_analysis"] == {}
