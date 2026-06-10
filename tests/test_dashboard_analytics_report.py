from __future__ import annotations

import json
from types import SimpleNamespace

from daytrading.analyst import collector
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


def test_dashboard_defaults_to_latest_saved_report_with_journal_db(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    journal_dir = data_dir / "journal"
    report_dir = data_dir / "reports"
    journal_dir.mkdir(parents=True)
    report_dir.mkdir()
    db_path = journal_dir / "journal.db"
    db_path.write_text("")
    (report_dir / "2026-06-05.json").write_text(json.dumps({
        "day": "2026-06-05",
        "ml_learning": {"entry_model": {"labeled": 311}},
        "ml_progress": {"models": []},
    }))

    class AnalystStub:
        def __init__(self, db_path, report_dir):
            pass

        def _analyze_ml_learning(self, day):
            return {"entry_model": {"labeled": 311}}

        def _analyze_ml_progress(self, day):
            return {"models": []}

    monkeypatch.setattr(collector, "NightlyAnalyst", AnalystStub)

    hub = DashboardHub()
    hub.journal = SimpleNamespace(base_dir=str(journal_dir), db_path=str(db_path))
    app = create_app(hub)

    resp = app.test_client().get("/api/analytics-report")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ok"] is True
    assert payload["day"] == "2026-06-05"
    assert payload["report"]["ml_learning"]["entry_model"]["labeled"] == 311


def test_dashboard_latest_analytics_report_does_not_refresh_ml_on_default_load(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    journal_dir = data_dir / "journal"
    report_dir = data_dir / "reports"
    journal_dir.mkdir(parents=True)
    report_dir.mkdir()
    db_path = journal_dir / "journal.db"
    db_path.write_text("")
    (report_dir / "2026-06-05.json").write_text(json.dumps({
        "day": "2026-06-05",
        "ml_learning": {"entry_model": {"labeled": 12}},
        "ml_progress": {"models": []},
    }))

    class AnalystStub:
        def __init__(self, db_path, report_dir):
            raise AssertionError("default dashboard load must not refresh analytics")

    monkeypatch.setattr(collector, "NightlyAnalyst", AnalystStub)

    hub = DashboardHub()
    hub.journal = SimpleNamespace(base_dir=str(journal_dir), db_path=str(db_path))
    app = create_app(hub)

    resp = app.test_client().get("/api/analytics-report")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ok"] is True
    assert payload["day"] == "2026-06-05"
    assert payload["report"]["ml_learning"]["entry_model"]["labeled"] == 12


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


def test_dashboard_generates_current_report_when_missing(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    journal_dir = data_dir / "journal"
    journal_dir.mkdir(parents=True)
    db_path = journal_dir / "journal.db"
    db_path.write_text("")

    class AnalystStub:
        def __init__(self, db_path, report_dir):
            self.report_dir = report_dir

        def run(self, day):
            path = __import__("pathlib").Path(self.report_dir) / f"{day}.json"
            report = {
                "day": day,
                "generated_at": "now",
                "ml_learning": {"entry_model": {"labeled": 99}},
                "ml_progress": {"models": []},
            }
            path.write_text(json.dumps(report))
            return report

        def _analyze_ml_learning(self, day):
            return {"entry_model": {"labeled": 99}}

        def _analyze_ml_progress(self, day):
            return {"models": []}

    monkeypatch.setattr(collector, "NightlyAnalyst", AnalystStub)

    hub = DashboardHub()
    hub.journal = SimpleNamespace(base_dir=str(journal_dir), db_path=str(db_path))
    app = create_app(hub)

    resp = app.test_client().get("/api/analytics-report?day=2026-06-03")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ok"] is True
    assert payload["day"] == "2026-06-03"
    assert payload["report"]["ml_learning"]["entry_model"]["labeled"] == 99


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
