"""Tests for nightly trade analyst."""

from __future__ import annotations

import os
from datetime import date, datetime, timezone

import pytest

from daytrading.analyst.collector import NightlyAnalyst
from daytrading.market_calendar import is_us_market_holiday
from daytrading.journal.store import TradingJournal


class TestMarketHoliday:
    def test_weekend_is_holiday(self) -> None:
        assert is_us_market_holiday(date(2026, 5, 16)) is True

    def test_weekday_non_holiday(self) -> None:
        assert is_us_market_holiday(date(2026, 5, 15)) is False

    def test_us_market_holiday(self) -> None:
        assert is_us_market_holiday(date(2026, 1, 1)) is True


class TestNightlyAnalystRun:
    def _seed_day_trades(self, journal: TradingJournal, day: str) -> None:
        ts_entry = f"{day}T14:30:00+00:00"
        ts_exit = f"{day}T14:35:00+00:00"
        journal.record("trade_fill", {
            "symbol": "WIN",
            "side": "buy",
            "quantity": 100,
            "price": 5.0,
            "trade_type": "entry",
            "strategy": "bull_flag",
        }, ts=datetime.fromisoformat(ts_entry))
        journal.record("trade_exit", {
            "symbol": "WIN",
            "side": "sell",
            "quantity": 100,
            "entry_price": 5.0,
            "exit_price": 5.20,
            "pnl": 20.0,
            "trade_type": "exit",
            "reason": "take_profit",
            "strategy": "bull_flag",
        }, ts=datetime.fromisoformat(ts_exit))
        journal.record("trade_fill", {
            "symbol": "LOSS",
            "side": "buy",
            "quantity": 100,
            "price": 3.0,
            "trade_type": "entry",
            "strategy": "momentum_burst",
        }, ts=datetime.fromisoformat(ts_entry))
        journal.record("trade_exit", {
            "symbol": "LOSS",
            "side": "sell",
            "quantity": 100,
            "entry_price": 3.0,
            "exit_price": 2.90,
            "pnl": -10.0,
            "trade_type": "exit",
            "reason": "stop_loss",
            "strategy": "momentum_burst",
        }, ts=datetime.fromisoformat(ts_exit))

    def test_holiday_skips_report(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        analyst = NightlyAnalyst(db_path=journal.db_path, report_dir=str(tmp_path / "reports"))
        report = analyst.run("2026-05-16")
        assert report["status"] == "holiday"

    def test_no_trades_skips_report(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        analyst = NightlyAnalyst(db_path=journal.db_path, report_dir=str(tmp_path / "reports"))
        report = analyst.run("2026-05-15")
        assert report["status"] == "no_trades"

    def test_full_report_shape_and_files(self, tmp_path) -> None:
        journal = TradingJournal(base_dir=str(tmp_path / "journal"))
        report_dir = str(tmp_path / "reports")
        self._seed_day_trades(journal, "2026-05-15")

        analyst = NightlyAnalyst(db_path=journal.db_path, report_dir=report_dir)
        report = analyst.run("2026-05-15")

        assert report.get("status") not in ("holiday", "no_trades")
        assert report["summary"]["win_count"] == 1
        assert report["summary"]["loss_count"] == 1
        assert report["summary"]["total_pnl"] == pytest.approx(10.0)

        patterns = {p["pattern"]: p for p in report["pattern_analysis"]}
        assert patterns["bull_flag"]["total_pnl"] == pytest.approx(20.0)
        assert patterns["momentum_burst"]["total_pnl"] == pytest.approx(-10.0)

        assert os.path.isfile(os.path.join(report_dir, "2026-05-15.json"))
        assert os.path.isfile(os.path.join(report_dir, "2026-05-15.md"))
        with open(os.path.join(report_dir, "2026-05-15.md")) as f:
            assert "Trading Report" in f.read()
