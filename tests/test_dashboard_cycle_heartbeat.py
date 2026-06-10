from __future__ import annotations

from werkzeug.exceptions import BadRequest

from daytrading.dashboard.hub import DashboardHub
from daytrading.dashboard.server import create_app


def test_cycle_heartbeat_updates_snapshot_cycle_count() -> None:
    hub = DashboardHub()

    hub.on_cycle_heartbeat(7, "no bars yet")

    snap = hub.snapshot()
    assert snap["stats"]["cycle_count"] == 7


def test_cycle_heartbeat_broadcasts_live_log_event() -> None:
    hub = DashboardHub()
    q = hub.subscribe()

    hub.on_cycle_heartbeat(8, "entry window closed")

    events = list(q)
    assert any(e["type"] == "cycle" and e["data"]["cycle"] == 8 for e in events)
    log_events = [e for e in events if e["type"] == "log"]
    assert len(log_events) == 1
    assert log_events[0]["data"]["message"] == "Cycle 8 heartbeat: entry window closed"


def test_cycle_heartbeat_ignores_stale_lower_cycle_number() -> None:
    hub = DashboardHub()
    q = hub.subscribe()

    hub.on_cycle_heartbeat(12, "active")
    hub.on_cycle_heartbeat(9, "late stale event")

    snap = hub.snapshot()
    assert snap["stats"]["cycle_count"] == 12
    cycle_events = [e for e in q if e["type"] == "cycle"]
    assert cycle_events[-1]["data"]["cycle"] == 12


def test_dashboard_cycle_display_resets_after_bot_restart_and_polls_status() -> None:
    hub = DashboardHub()
    app = create_app(hub)

    resp = app.test_client().get("/")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "function setCycleCountMonotonic" in html
    assert "setCycleCountMonotonic(msg.data.cycle)" in html
    assert "sameBotRun ? Math.max(savedCycle, incomingCycle) : incomingCycle" in html
    assert "setInterval(syncTradingControlState, 10000)" in html


def test_dashboard_api_fetches_retry_transient_failures() -> None:
    hub = DashboardHub()
    app = create_app(hub)

    resp = app.test_client().get("/")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "function fetchJsonWithRetry" in html
    assert "fetchJsonWithRetry('/api/snapshot')" in html
    assert "fetchJsonWithRetry('/api/ml-stats')" in html


def test_dashboard_api_responses_are_not_cached() -> None:
    hub = DashboardHub()
    app = create_app(hub)

    resp = app.test_client().get("/api/snapshot")

    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "no-cache, no-store, must-revalidate"
    assert resp.headers["Pragma"] == "no-cache"
    assert resp.headers["Expires"] == "0"


def test_dashboard_force_refresh_cache_busts_ml_report() -> None:
    hub = DashboardHub()
    app = create_app(hub)

    resp = app.test_client().get("/")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "let url = '/api/analytics-report' + (force ? ('?_=' + now) : '')" in html


def test_dashboard_bad_request_handler_returns_diagnostic_json() -> None:
    hub = DashboardHub()
    app = create_app(hub)

    def raise_bad_request():
        raise BadRequest("test bad payload")

    app.add_url_rule("/api/test-bad-request", view_func=raise_bad_request)

    resp = app.test_client().get("/api/test-bad-request")

    assert resp.status_code == 400
    assert resp.get_json() == {
        "ok": False,
        "error": "bad request",
        "path": "/api/test-bad-request",
    }


def test_hot_watch_updates_snapshot() -> None:
    hub = DashboardHub()

    hub.on_hot_watch([
        {
            "symbol": "DXST",
            "mode": "runner_watch",
            "price": 3.19,
            "score": 35.0,
            "remaining_seconds": 900,
        }
    ])

    snap = hub.snapshot()
    assert snap["hot_watch"] == [
        {
            "symbol": "DXST",
            "mode": "runner_watch",
            "price": 3.19,
            "score": 35.0,
            "remaining_seconds": 900,
        }
    ]


def test_hot_watch_broadcast_refreshes_visible_trading_watchlist() -> None:
    hub = DashboardHub()
    q = hub.subscribe()
    hub.on_trading_watchlist(["SPY"], pinned=["SPY"])
    q.clear()

    hub.on_hot_watch([
        {
            "symbol": "DXST",
            "mode": "runner_watch",
            "price": 3.19,
            "score": 35.0,
            "remaining_seconds": 900,
        }
    ])

    events = list(q)
    visible = [
        e for e in events
        if e["type"] == "trading_watchlist"
    ][-1]
    assert visible["data"]["symbols"] == ["SPY", "DXST"]
    assert visible["data"]["pinned"] == ["SPY"]


def test_dashboard_defines_compact_volume_helper_for_trading_watchlist() -> None:
    hub = DashboardHub()
    app = create_app(hub)

    resp = app.test_client().get("/")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "function fmtCompact" in html
    assert "fmtCompact(h.volume||0)" in html


def test_dashboard_hot_watch_shows_session_and_current_move_separately() -> None:
    hub = DashboardHub()
    app = create_app(hub)

    resp = app.test_client().get("/")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "<th>Session</th><th>Now</th>" in html
    assert "function hotWatchNowText" in html
    assert "from high" in html
    assert "short-term" in html


def test_candidate_hydration_stats_snapshot_and_broadcast() -> None:
    hub = DashboardHub()
    q = hub.subscribe()

    hub.on_candidate_hydration(
        queued=5,
        hydrated=2,
        skipped_fresh=1,
        pending=3,
        paused_for_entry=True,
        last_batch_size=4,
        last_loaded=2,
        last_source="fast scan",
    )

    snap = hub.snapshot()
    stats = snap["candidate_hydration"]
    assert stats["queued"] == 5
    assert stats["hydrated"] == 2
    assert stats["skipped_fresh"] == 1
    assert stats["pending"] == 3
    assert stats["paused_for_entry"] is True
    assert stats["last_batch_size"] == 4
    assert stats["last_loaded"] == 2
    assert stats["last_source"] == "fast scan"
    assert any(e["type"] == "candidate_hydration" for e in q)


def test_dashboard_renders_candidate_worker_status_hook() -> None:
    hub = DashboardHub()
    app = create_app(hub)

    resp = app.test_client().get("/")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "candidate-worker-status" in html
    assert "function renderCandidateWorker" in html


def test_entry_checks_render_setup_tier_column() -> None:
    hub = DashboardHub()
    app = create_app(hub)

    resp = app.test_client().get("/")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "<th>Tier</th>" in html
    assert "function renderSetupTier" in html
    assert "A_PLUS_SCANNERS" in html


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
