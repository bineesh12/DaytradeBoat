"""Tests for nightly trade analyst."""

from __future__ import annotations

import os
import json
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

    def _write_jsonl(self, path, rows) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _seed_ml_learning_rows(self, ml_dir, day: str) -> None:
        self._write_jsonl(ml_dir / "missed_opportunities.jsonl", [
            {
                "ts": f"{day}T14:31:00+00:00",
                "symbol": "OLOX",
                "reason": "guard rejected: extended",
                "label": 1,
                "future_return_pct": 3.2,
            },
            {
                "ts": f"{day}T14:34:00+00:00",
                "symbol": "STG",
                "reason": "cooldown",
                "label": 0,
                "future_return_pct": -0.4,
            },
        ])
        self._write_jsonl(ml_dir / "pullback_candidates.jsonl", [
            {
                "ts": f"{day}T14:35:00+00:00",
                "symbol": "PULL",
                "scanner": "vwap_pullback",
                "label": 1,
                "future_return_pct": 1.8,
            },
        ])
        self._write_jsonl(ml_dir / "exit_snapshots.jsonl", [
            {
                "ts": f"{day}T14:36:00+00:00",
                "symbol": "HOLD",
                "label": 1,
                "future_return_pct": 0.7,
            },
            {
                "ts": f"{day}T14:37:00+00:00",
                "symbol": "SELL",
                "label": 0,
                "future_return_pct": -0.8,
            },
        ])
        self._write_jsonl(ml_dir / "execution_quality.jsonl", [
            {
                "ts": f"{day}T14:38:00+00:00",
                "symbol": "FILL",
                "status": "filled",
                "label": 1,
                "slippage_pct": 0.2,
            },
            {
                "ts": f"{day}T14:39:00+00:00",
                "symbol": "BADFILL",
                "status": "filled",
                "label": 0,
                "slippage_pct": 1.1,
            },
            {
                "ts": "2026-05-14T14:39:00+00:00",
                "symbol": "OLD",
                "status": "filled",
                "label": 1,
                "slippage_pct": 0.1,
            },
        ])
        self._write_jsonl(ml_dir / "entry_candidates.jsonl", [
            {
                "ts": f"{day}T14:30:00+00:00",
                "symbol": "WIN",
                "passed": True,
                "outcome_pnl": 2.5,
            },
            {
                "ts": f"{day}T14:31:00+00:00",
                "symbol": "LOSS",
                "passed": True,
                "outcome_pnl": -1.0,
            },
            {
                "ts": f"{day}T14:32:00+00:00",
                "symbol": "PENDING",
                "passed": False,
                "outcome_pnl": None,
            },
        ])
        self._write_jsonl(ml_dir / "shadow_results.jsonl", [
            {
                "ts": f"{day}T14:33:00+00:00",
                "symbol": "RIGHT",
                "ml_correct": True,
                "change_pct": -1.2,
            },
            {
                "ts": f"{day}T14:34:00+00:00",
                "symbol": "WRONG",
                "ml_correct": False,
                "change_pct": 4.4,
            },
        ])

    def _seed_previous_ml_report(self, report_dir) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        previous = {
            "day": "2026-05-14",
            "ml_learning": {
                "total_rows": 4,
                "missed_opportunities": {"positive_rate": 25.0},
                "pullback_candidates": {"positive_rate": 0.0},
                "exit_helper": {"positive_rate": 50.0},
                "execution_quality": {"good_rate": 100.0},
            },
        }
        (report_dir / "2026-05-14.json").write_text(json.dumps(previous))

    def _seed_shadow_model_meta(self, model_dir) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "dataset": "missed_opportunity",
            "samples": 80,
            "positive_rate": 0.42,
            "test_accuracy": 0.61,
            "feature_names": ["momentum_5bar_pct"],
        }
        (model_dir / "missed_opportunity_model.meta.json").write_text(json.dumps(meta))

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
        report_dir_path = tmp_path / "reports"
        report_dir = str(report_dir_path)
        ml_dir = tmp_path / "ml"
        model_dir = tmp_path / "models"
        self._seed_day_trades(journal, "2026-05-15")
        self._seed_ml_learning_rows(ml_dir, "2026-05-15")
        self._seed_previous_ml_report(report_dir_path)
        self._seed_shadow_model_meta(model_dir)

        analyst = NightlyAnalyst(
            db_path=journal.db_path,
            report_dir=report_dir,
            ml_dir=str(ml_dir),
            model_dir=str(model_dir),
        )
        report = analyst.run("2026-05-15")

        assert report.get("status") not in ("holiday", "no_trades")
        assert report["summary"]["win_count"] == 1
        assert report["summary"]["loss_count"] == 1
        assert report["summary"]["total_pnl"] == pytest.approx(10.0)

        patterns = {p["pattern"]: p for p in report["pattern_analysis"]}
        assert patterns["bull_flag"]["total_pnl"] == pytest.approx(20.0)
        assert patterns["momentum_burst"]["total_pnl"] == pytest.approx(-10.0)

        ml = report["ml_learning"]
        assert ml["total_rows"] == 12
        assert ml["entry_model"]["total"] == 3
        assert ml["entry_model"]["labeled"] == 2
        assert ml["entry_model"]["profitable"] == 1
        assert ml["entry_shadow"]["total"] == 2
        assert ml["entry_shadow"]["correct"] == 1
        assert ml["entry_shadow"]["wrong"] == 1
        assert ml["missed_opportunities"]["total"] == 2
        assert ml["missed_opportunities"]["went_up"] == 1
        assert ml["missed_opportunities"]["positive_rate"] == pytest.approx(50.0)
        assert ml["missed_opportunities"]["best"]["symbol"] == "OLOX"
        assert ml["pullback_candidates"]["worked"] == 1
        assert ml["exit_helper"]["hold_helped"] == 1
        assert ml["execution_quality"]["good_fills"] == 1
        assert ml["execution_quality"]["bad_fills"] == 1
        assert ml["execution_quality"]["worst"]["symbol"] == "BADFILL"

        progress = report["ml_progress"]
        assert progress["previous_day"] == "2026-05-14"
        assert progress["rows_today"] == 12
        assert progress["rows_previous"] == 4
        assert progress["rows_change"] == 8
        assert progress["all_time_rows"] == 13
        assert progress["all_time_labeled"] == 12
        missed_progress = {
            row["dataset"]: row for row in progress["datasets"]
        }["missed_opportunities"]
        assert missed_progress["rate_change"] == pytest.approx(25.0)
        models = {m["model"]: m for m in progress["models"]}
        assert models["missed_opportunity"]["status"] == "trained"
        assert models["missed_opportunity"]["samples"] == 80
        assert models["pullback_entry"]["status"] == "collecting_data"

        assert os.path.isfile(os.path.join(report_dir, "2026-05-15.json"))
        assert os.path.isfile(os.path.join(report_dir, "2026-05-15.md"))
        with open(os.path.join(report_dir, "2026-05-15.md")) as f:
            markdown = f.read()
            assert "Trading Report" in markdown
            assert "ML Learning Report" in markdown
            assert "ML Progress" in markdown
            assert "Best missed setup: OLOX" in markdown
            assert "missed_opportunity" in markdown
