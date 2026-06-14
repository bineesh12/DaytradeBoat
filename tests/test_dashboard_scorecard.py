from __future__ import annotations

from datetime import datetime

from daytrading.dashboard.hub import DashboardHub, TradeRecord
from daytrading.dashboard.server import create_app
from daytrading.journal.store import TradingJournal


def test_daily_scorecard_reports_expectancy_and_funnel() -> None:
    hub = DashboardHub()
    hub.total_trades = 3
    hub.total_scan_hits = 20
    hub.total_signals = 5
    hub.total_rejected = 7
    hub.cycle_count = 42
    hub.trades.append(
        TradeRecord(
            symbol="WIN",
            side="sell",
            quantity=100,
            entry_price=2.00,
            entry_time="",
            exit_price=2.20,
            exit_time="",
            pnl=20.0,
            trade_type="exit",
        )
    )
    hub.trades.append(
        TradeRecord(
            symbol="LOSS",
            side="sell",
            quantity=100,
            entry_price=2.00,
            entry_time="",
            exit_price=1.90,
            exit_time="",
            pnl=-10.0,
            trade_type="exit",
        )
    )
    hub.on_missed_a_plus([
        {
            "symbol": "MISS",
            "pattern": "pullback_base",
            "outcome": "missed_opportunity",
            "move_after_pct": 12.4,
            "reason": "late_from_HOD",
        },
        {
            "symbol": "GOOD",
            "pattern": "vwap_pullback",
            "outcome": "correct_reject",
            "move_after_pct": 0.4,
            "reason": "selling pressure",
        },
    ])

    scorecard = hub.snapshot()["daily_scorecard"]

    assert scorecard["trades_taken"] == 3
    assert scorecard["closed_trades"] == 2
    assert scorecard["win_rate"] == 50.0
    assert scorecard["total_pnl"] == 10.0
    assert scorecard["avg_win"] == 20.0
    assert scorecard["avg_loss"] == 10.0
    assert scorecard["profit_factor"] == 2.0
    assert scorecard["expectancy_per_trade"] == 5.0
    assert scorecard["cycles"] == 42
    assert scorecard["funnel"]["hit_to_signal_pct"] == 25.0
    assert scorecard["funnel"]["signal_to_entry_pct"] == 60.0
    assert scorecard["funnel"]["reject_rate_pct"] == 58.3
    assert scorecard["missed_a_plus"]["missed_opportunities"] == 1
    assert scorecard["missed_a_plus"]["correct_rejects"] == 1
    assert scorecard["missed_a_plus"]["best_symbol"] == "MISS"


def test_daily_scorecard_counts_partial_exits_as_one_round_trip() -> None:
    hub = DashboardHub()
    hub.total_trades = 1
    # one entry, then TWO partial exits (half + trail) for the same position
    hub.trades.append(
        TradeRecord(
            symbol="PART", side="buy", quantity=200, entry_price=2.00,
            entry_time="t0", exit_price=None, exit_time=None, pnl=None,
            trade_type="entry",
        )
    )
    hub.trades.append(
        TradeRecord(
            symbol="PART", side="sell", quantity=100, entry_price=2.00,
            entry_time="t0", exit_price=2.05, exit_time="t1", pnl=5.0,
            trade_type="exit",
        )
    )
    hub.trades.append(
        TradeRecord(
            symbol="PART", side="sell", quantity=100, entry_price=2.00,
            entry_time="t0", exit_price=2.03, exit_time="t2", pnl=3.0,
            trade_type="exit",
        )
    )

    sc = hub.snapshot()["daily_scorecard"]

    # two exit rows, but ONE round-trip — not two closed trades
    assert sc["closed_trades"] == 1
    assert sc["wins"] == 1
    assert sc["total_pnl"] == 8.0
    # closed-rate can no longer exceed 100% (was 200% counting raw exits)
    assert sc["funnel"]["closed_rate_pct"] <= 100.0


def test_scorecard_isolates_momentum_breakout_mode() -> None:
    from daytrading.dashboard.hub import _daily_scorecard

    # the exact trap: the experimental mode loses while normal entries win
    trades = [
        {"symbol": "MB", "trade_type": "entry", "strategy": "breakout_scalp_momentum",
         "entry_time": "t0", "pnl": None},
        {"symbol": "MB", "trade_type": "exit", "strategy": "breakout_scalp_momentum",
         "exit_time": "t1", "pnl": -8.0},
        {"symbol": "STD", "trade_type": "entry", "strategy": "breakout_scalp",
         "entry_time": "t0", "pnl": None},
        {"symbol": "STD", "trade_type": "exit", "strategy": "breakout_scalp",
         "exit_time": "t1", "pnl": 12.0},
    ]

    sc = _daily_scorecard(
        trades=trades, total_trades=2, total_scan_hits=0, total_signals=0,
        total_rejected=0, cycle_count=0, missed_a_plus=[],
    )

    bm = sc["by_entry_mode"]
    bs = sc["by_strategy"]
    # blended P&L is +$4 (looks fine) but the mode is isolated as a -$8 loser
    assert sc["total_pnl"] == 4.0
    assert bm["momentum_breakout"]["closed_trades"] == 1
    assert bm["momentum_breakout"]["total_pnl"] == -8.0
    assert bm["momentum_breakout"]["wins"] == 0
    assert bm["standard"]["closed_trades"] == 1
    assert bm["standard"]["total_pnl"] == 12.0
    assert bs["breakout_scalp_momentum"]["total_pnl"] == -8.0
    assert bs["breakout_scalp"]["total_pnl"] == 12.0


def test_dashboard_fill_strategy_feeds_entry_mode_scorecard() -> None:
    from daytrading.execution.broker import Fill, Side

    hub = DashboardHub()
    fill = Fill(
        symbol="MB",
        side=Side.BUY,
        quantity=100,
        price=2.00,
        ts=datetime.fromisoformat("2026-06-10T14:30:00+00:00"),
    )
    hub.on_fill(fill, "entry", strategy="breakout_scalp_momentum")
    hub.trades.append(
        TradeRecord(
            symbol="MB", side="sell", quantity=100, entry_price=2.00,
            entry_time="t0", exit_price=1.95, exit_time="t1", pnl=-5.0,
            trade_type="exit",
        )
    )

    sc = hub.snapshot()["daily_scorecard"]

    assert sc["by_entry_mode"]["momentum_breakout"]["closed_trades"] == 1
    assert sc["by_entry_mode"]["momentum_breakout"]["total_pnl"] == -5.0
    assert sc["by_strategy"]["breakout_scalp_momentum"]["total_pnl"] == -5.0


def test_scorecard_counts_reentry_as_new_round_trip_with_strategy() -> None:
    from daytrading.dashboard.hub import _daily_scorecard

    sc = _daily_scorecard(
        trades=[
            {"symbol": "RE", "trade_type": "reentry", "strategy": "hod_reclaim", "entry_time": "t0"},
            {"symbol": "RE", "trade_type": "exit", "exit_time": "t1", "pnl": 7.0},
        ],
        total_trades=1,
        total_scan_hits=0,
        total_signals=0,
        total_rejected=0,
        cycle_count=0,
        missed_a_plus=[],
    )

    assert sc["closed_trades"] == 1
    assert sc["by_strategy"]["hod_reclaim"]["total_pnl"] == 7.0


def test_scorecard_isolates_fresh_vwap_reclaim_scout_mode() -> None:
    from daytrading.dashboard.hub import _daily_scorecard

    trades = [
        {"symbol": "FR", "trade_type": "entry", "strategy": "fresh_vwap_reclaim_scout",
         "entry_time": "t0", "pnl": None},
        {"symbol": "FR", "trade_type": "exit", "strategy": "fresh_vwap_reclaim_scout",
         "exit_time": "t1", "pnl": 16.0},
    ]
    sc = _daily_scorecard(
        trades=trades, total_trades=1, total_scan_hits=0, total_signals=0,
        total_rejected=0, cycle_count=0, missed_a_plus=[],
    )
    bm = sc["by_entry_mode"]
    assert bm["fresh_vwap_reclaim_scout"]["closed_trades"] == 1
    assert bm["fresh_vwap_reclaim_scout"]["total_pnl"] == 16.0


def test_scorecard_isolates_elite_wide_spread_mode() -> None:
    from daytrading.dashboard.hub import _daily_scorecard

    trades = [
        {"symbol": "EW", "trade_type": "entry", "strategy": "elite_wide_spread",
         "entry_time": "t0", "pnl": None},
        {"symbol": "EW", "trade_type": "exit", "strategy": "elite_wide_spread",
         "exit_time": "t1", "pnl": 9.0},
    ]
    sc = _daily_scorecard(
        trades=trades, total_trades=1, total_scan_hits=0, total_signals=0,
        total_rejected=0, cycle_count=0, missed_a_plus=[],
    )
    bm = sc["by_entry_mode"]
    assert bm["elite_wide_spread"]["closed_trades"] == 1
    assert bm["elite_wide_spread"]["total_pnl"] == 9.0
    assert "standard" not in bm


def test_scorecard_isolates_level_breakout_scout_mode() -> None:
    from daytrading.dashboard.hub import _daily_scorecard

    trades = [
        {"symbol": "LB", "trade_type": "entry", "strategy": "level_breakout_scout",
         "entry_time": "t0", "pnl": None},
        {"symbol": "LB", "trade_type": "exit", "strategy": "level_breakout_scout",
         "exit_time": "t1", "pnl": 5.0},
    ]
    sc = _daily_scorecard(
        trades=trades, total_trades=1, total_scan_hits=0, total_signals=0,
        total_rejected=0, cycle_count=0, missed_a_plus=[],
    )
    bm = sc["by_entry_mode"]
    assert bm["level_breakout_scout"]["closed_trades"] == 1
    assert bm["level_breakout_scout"]["total_pnl"] == 5.0
    assert "standard" not in bm


def test_rolling_scorecard_is_cached_between_snapshots(monkeypatch) -> None:
    from daytrading.dashboard import hub as hub_module

    calls = {"n": 0}

    def fake_rolling(journal, window_days=20):
        calls["n"] += 1
        return {"available": True, "verdict": "collecting"}

    monkeypatch.setattr(hub_module, "_rolling_journal_scorecard", fake_rolling)
    hub = DashboardHub()
    hub.snapshot()
    hub.snapshot()
    # second snapshot reuses the cache — SQLite is not re-scanned every render
    assert calls["n"] == 1


def test_dashboard_renders_daily_scorecard_panel() -> None:
    hub = DashboardHub()
    app = create_app(hub)

    resp = app.test_client().get("/")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Daily Scorecard" in html
    assert "Rolling Go-Live Gauge" in html
    assert "Strategy P&amp;L" in html or "Strategy P&L" in html
    assert "function renderDailyScorecard" in html
    assert "daily_scorecard" in html
    assert "rolling_scorecard" in html


def test_rolling_scorecard_reads_trade_expectancy_from_journal(tmp_path) -> None:
    journal = TradingJournal(base_dir=str(tmp_path / "journal"))
    try:
        day = "2026-06-10"
        ts_entry = datetime.fromisoformat(f"{day}T14:30:00+00:00")
        ts_exit = datetime.fromisoformat(f"{day}T14:35:00+00:00")
        journal.record("cycle", {
            "cycle": 1,
            "scan_hits": 10,
            "signals": 4,
            "rejected": 6,
        }, ts=ts_entry)
        journal.record("trade_fill", {
            "symbol": "WIN",
            "side": "buy",
            "quantity": 100,
            "price": 2.0,
            "trade_type": "entry",
            "strategy": "vwap_pullback",
        }, ts=ts_entry)
        journal.record("trade_exit", {
            "symbol": "WIN",
            "side": "sell",
            "quantity": 100,
            "entry_price": 2.0,
            "exit_price": 2.20,
            "pnl": 20.0,
            "trade_type": "exit",
            "reason": "take_profit",
        }, ts=ts_exit)
        journal.record("trade_fill", {
            "symbol": "LOSS",
            "side": "buy",
            "quantity": 100,
            "price": 3.0,
            "trade_type": "entry",
            "strategy": "pullback_base",
        }, ts=ts_entry)
        journal.record("trade_exit", {
            "symbol": "LOSS",
            "side": "sell",
            "quantity": 100,
            "entry_price": 3.0,
            "exit_price": 2.90,
            "pnl": -10.0,
            "trade_type": "exit",
            "reason": "stop_loss",
        }, ts=ts_exit)

        hub = DashboardHub()
        hub.journal = journal

        rolling = hub.snapshot()["rolling_scorecard"]
    finally:
        journal.close()

    assert rolling["available"] is True
    assert rolling["window_days"] == 20
    assert rolling["min_closed_trades"] == 25
    assert rolling["min_sessions"] == 10
    assert rolling["trades_taken"] == 2
    assert rolling["closed_trades"] == 2
    assert rolling["total_pnl"] == 10.0
    assert rolling["expectancy_per_trade"] == 5.0
    assert rolling["funnel"]["hit_to_signal_pct"] == 40.0
    assert rolling["funnel"]["signal_to_entry_pct"] == 50.0
    assert rolling["funnel"]["reject_rate_pct"] == 60.0
    assert rolling["sessions"] == 1
    assert rolling["verdict"] == "collecting"
    assert "25 closed trades" in rolling["verdict_reason"]


def test_rolling_scorecard_is_unavailable_without_journal() -> None:
    hub = DashboardHub()

    rolling = hub.snapshot()["rolling_scorecard"]

    assert rolling["available"] is False
    assert rolling["reason"] == "journal not configured"
