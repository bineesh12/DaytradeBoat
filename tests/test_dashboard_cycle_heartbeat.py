from __future__ import annotations

from daytrading.dashboard.hub import DashboardHub


def test_cycle_heartbeat_updates_snapshot_cycle_count() -> None:
    hub = DashboardHub()

    hub.on_cycle_heartbeat(7, "no bars yet")

    snap = hub.snapshot()
    assert snap["stats"]["cycle_count"] == 7
