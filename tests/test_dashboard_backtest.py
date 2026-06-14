from __future__ import annotations

from daytrading.dashboard.hub import DashboardHub
from daytrading.dashboard.server import create_app


def test_dashboard_renders_backtest_page() -> None:
    app = create_app(DashboardHub())

    html = app.test_client().get("/").get_data(as_text=True)

    assert 'data-page="backtest"' in html
    assert 'id="page-backtest"' in html
    assert 'id="bt-symbol"' in html
    assert 'id="bt-flag-level" checked' not in html
    assert "function runBacktest" in html
    assert "function renderBacktest" in html
    assert "function renderBacktestChart" in html
    assert "function renderBacktestFunnel" in html
    assert "function renderBacktestLayerBreakdown" in html
    assert "function shortEtTime" in html
    assert "function zoomBacktestChart" in html
    assert "function panBacktestChart" in html
    assert "function beginBacktestChartDrag" in html
    assert "bt-svg-chart" in html
    assert "bt-chart-svg" in html
    assert "window.LightweightCharts" not in html
    assert "function hydrateTradingViewBacktestChart" not in html
    assert "onwheel=\"wheelBacktestChart(event)\"" in html
    assert "Chart times are Eastern Time" in html
    assert ".replace(' ET', '')" in html
    assert "A+ Funnel Detail" in html
    assert "Backtest Gate Breakdown" in html


def test_backtest_endpoint_returns_service_result(monkeypatch) -> None:
    app = create_app(DashboardHub())
    seen = {}

    def fake_run(symbol, session_date, *, flags=None, settings=None):
        seen["settings"] = settings
        return {
            "ok": True,
            "symbol": symbol,
            "date": session_date,
            "bars": 10,
            "cycles": 9,
            "round_trips": [],
            "scorecard": {"trades_taken": 0},
            "funnel": {},
            "flags": flags,
        }

    monkeypatch.setattr("daytrading.backtest.service.run_backtest", fake_run)

    resp = app.test_client().post("/api/backtest", json={
        "symbol": "cupr",
        "date": "2026-06-10",
        "flags": {"level_breakout_scout": True},
    })

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["symbol"] == "CUPR"
    assert data["date"] == "2026-06-10"
    assert data["flags"]["level_breakout_scout"] is True
    assert seen["settings"] is not None


def test_backtest_endpoint_validates_required_fields() -> None:
    app = create_app(DashboardHub())

    resp = app.test_client().post("/api/backtest", json={"date": "2026-06-10"})

    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False


def test_backtest_sweep_endpoint_returns_service_result(monkeypatch) -> None:
    app = create_app(DashboardHub())
    seen = {}

    def fake_sweep(symbols, dates, *, experiments=None, settings=None):
        seen["settings"] = settings
        return {
            "ok": True,
            "symbols": symbols,
            "dates": dates,
            "experiments": {"baseline": {"scorecard": {"total_pnl": 0}}},
            "deltas_vs_baseline": {},
        }

    monkeypatch.setattr("daytrading.backtest.service.run_backtest_sweep", fake_sweep)

    resp = app.test_client().post("/api/backtest/sweep", json={
        "symbols": "cupr, conl",
        "dates": "2026-06-10",
    })

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["symbols"] == ["cupr", "conl"]
    assert data["dates"] == ["2026-06-10"]
    assert seen["settings"] is not None


def test_backtest_sweep_endpoint_validates_required_fields() -> None:
    app = create_app(DashboardHub())

    resp = app.test_client().post("/api/backtest/sweep", json={"dates": ["2026-06-10"]})

    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False
