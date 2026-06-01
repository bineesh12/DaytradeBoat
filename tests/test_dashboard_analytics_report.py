from __future__ import annotations

import json
from types import SimpleNamespace

from daytrading.dashboard.hub import DashboardHub
from daytrading.dashboard.server import create_app


def test_dashboard_returns_latest_analytics_report(tmp_path):
    data_dir = tmp_path / "data"
    journal_dir = data_dir / "journal"
    report_dir = data_dir / "reports"
    journal_dir.mkdir(parents=True)
    report_dir.mkdir()
    (report_dir / "2026-05-14.json").write_text(json.dumps({
        "day": "2026-05-14",
        "ml_learning": {"total_rows": 1},
    }))
    (report_dir / "2026-05-15.json").write_text(json.dumps({
        "day": "2026-05-15",
        "ml_learning": {"total_rows": 7},
        "ml_progress": {"all_time_rows": 9},
    }))

    hub = DashboardHub()
    hub.journal = SimpleNamespace(base_dir=str(journal_dir))
    app = create_app(hub)

    resp = app.test_client().get("/api/analytics-report")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ok"] is True
    assert payload["day"] == "2026-05-15"
    assert payload["report"]["ml_learning"]["total_rows"] == 7
    assert payload["report"]["ml_progress"]["all_time_rows"] == 9


def test_dashboard_analytics_report_handles_missing_report_dir(tmp_path):
    journal_dir = tmp_path / "data" / "journal"
    journal_dir.mkdir(parents=True)
    hub = DashboardHub()
    hub.journal = SimpleNamespace(base_dir=str(journal_dir))
    app = create_app(hub)

    resp = app.test_client().get("/api/analytics-report")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ok"] is True
    assert payload["report"] is None
    assert "No nightly analytics report" in payload["message"]


def test_force_close_uses_emergency_close_and_pauses_trading():
    class ExitManagerStub:
        def __init__(self):
            self.tracked = {"MASK": object()}
            self.untracked = []

        def untrack(self, symbol):
            self.untracked.append(symbol)
            self.tracked.pop(symbol, None)

    broker = SimpleNamespace(
        emergency_close_all_positions=lambda **kwargs: {
            "ok": True,
            "flat": True,
            "attempts": kwargs["attempts"],
            "cancelled_orders": 2,
            "submitted_orders": [],
            "remaining_positions": {},
            "errors": [],
        }
    )
    exit_manager = ExitManagerStub()
    hub = DashboardHub()
    hub._broker = broker
    hub._exit_manager = exit_manager
    app = create_app(hub)

    resp = app.test_client().post("/api/force-close")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["flat"] is True
    assert payload["attempts"] == 4
    assert hub.trading_paused is True
    assert exit_manager.untracked == ["MASK"]


def test_force_close_reports_remaining_positions():
    broker = SimpleNamespace(
        emergency_close_all_positions=lambda **kwargs: {
            "ok": False,
            "flat": False,
            "remaining_positions": {"MASK": {"qty": 10}},
            "errors": ["held for orders"],
        }
    )
    hub = DashboardHub()
    hub._broker = broker
    app = create_app(hub)

    resp = app.test_client().post("/api/force-close")

    assert resp.status_code == 500
    payload = resp.get_json()
    assert payload["flat"] is False
    assert payload["remaining_positions"]["MASK"]["qty"] == 10
    assert hub.trading_paused is True
