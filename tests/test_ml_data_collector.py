from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from daytrading.ml import data_collector as dc
from daytrading.models import Bar


def _rows():
    return [
        json.loads(line)
        for line in dc._CANDIDATES_FILE.read_text().splitlines()
        if line.strip()
    ]


def _bar(symbol: str, close: float, ts: datetime) -> Bar:
    return Bar(
        symbol=symbol,
        ts=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100_000,
    )


def test_deferred_entry_outcomes_label_same_day_candidates():
    now = datetime.now(timezone.utc)
    dc.log_entry_candidate(
        symbol="FOXX",
        price=4.00,
        score=80,
        passed=False,
        reject_reason="rule reject",
        bars=[_bar("FOXX", 4.00, now - timedelta(minutes=5))],
    )
    rows = _rows()
    rows[0]["ts"] = (now - timedelta(minutes=5)).isoformat()
    dc._CANDIDATES_FILE.write_text(json.dumps(rows[0]) + "\n")

    changed = dc.update_deferred_entry_outcomes(
        {"FOXX": [_bar("FOXX", 4.12, now)]},
        wait_seconds=180,
        min_move_pct=1.5,
    )

    assert changed == 1
    [row] = _rows()
    assert row["outcome_source"] == "shadow_future_price"
    assert row["outcome_pnl"] == pytest.approx(3.0)
    assert row["shadow_label"] == 1


def test_deferred_entry_outcomes_skip_passed_entries_waiting_for_real_trade():
    now = datetime.now(timezone.utc)
    dc.log_entry_candidate(
        symbol="FOXX",
        price=4.00,
        score=80,
        passed=True,
        bars=[_bar("FOXX", 4.00, now - timedelta(minutes=5))],
    )
    rows = _rows()
    rows[0]["ts"] = (now - timedelta(minutes=5)).isoformat()
    dc._CANDIDATES_FILE.write_text(json.dumps(rows[0]) + "\n")

    changed = dc.update_deferred_entry_outcomes(
        {"FOXX": [_bar("FOXX", 4.12, now)]},
        wait_seconds=180,
        min_move_pct=1.5,
    )

    assert changed == 0
    [row] = _rows()
    assert row["outcome_pnl"] is None


def test_real_trade_outcome_overwrites_shadow_label():
    now = datetime.now(timezone.utc)
    dc.log_entry_candidate(
        symbol="FOXX",
        price=4.00,
        score=80,
        passed=True,
        bars=[_bar("FOXX", 4.00, now - timedelta(minutes=5))],
    )
    rows = _rows()
    rows[0]["ts"] = (now - timedelta(minutes=5)).isoformat()
    rows[0]["outcome_pnl"] = 3.0
    rows[0]["outcome_pct"] = 3.0
    rows[0]["outcome_duration_s"] = 180
    rows[0]["outcome_source"] = "shadow_future_price"
    rows[0]["shadow_label"] = 1
    rows[0]["labeled_at"] = now.isoformat()
    dc._CANDIDATES_FILE.write_text(json.dumps(rows[0]) + "\n")

    dc.log_trade_outcome(
        symbol="FOXX",
        entry_price=4.00,
        exit_price=3.90,
        entry_time=now - timedelta(minutes=4),
        exit_time=now,
    )

    [row] = _rows()
    assert row["outcome_source"] == "real_trade"
    assert row["outcome_pnl"] == pytest.approx(-2.5)
    assert "shadow_label" not in row
    assert "labeled_at" not in row


def test_deferred_entry_outcomes_do_not_label_different_day_rows():
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1, minutes=5)
    dc.log_entry_candidate(
        symbol="FOXX",
        price=4.00,
        score=80,
        passed=True,
        bars=[_bar("FOXX", 4.00, yesterday)],
    )
    rows = _rows()
    rows[0]["ts"] = yesterday.isoformat()
    dc._CANDIDATES_FILE.write_text(json.dumps(rows[0]) + "\n")

    changed = dc.update_deferred_entry_outcomes(
        {"FOXX": [_bar("FOXX", 4.50, now)]},
        wait_seconds=180,
    )

    assert changed == 0
    [row] = _rows()
    assert row["outcome_pnl"] is None


def test_load_candidates_for_filters_by_symbol_and_day():
    # Two CAST records on 6/15 (one pass, one reject), plus noise that must be excluded.
    dc._CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    dc._CANDIDATES_FILE.write_text("\n".join([
        json.dumps({"ts": "2026-06-15T14:17:32+00:00", "symbol": "CAST", "price": 1.54,
                    "score": 95, "passed": True, "reject_reason": None,
                    "breakdown": "day+170%=20, surge5.7x=15, rvol2.0x=-5", "rel_vol": 2.0}),
        json.dumps({"ts": "2026-06-15T18:50:00+00:00", "symbol": "CAST", "price": 3.63,
                    "score": 72, "passed": False, "reject_reason": "entry score too low (72/100, need 80+)",
                    "breakdown": "rvol0.4x=-25", "rel_vol": 0.4}),
        json.dumps({"ts": "2026-06-14T14:00:00+00:00", "symbol": "CAST", "price": 2.0,
                    "score": 50, "passed": False}),          # wrong day
        json.dumps({"ts": "2026-06-15T15:00:00+00:00", "symbol": "AHMA", "price": 2.3,
                    "score": 80, "passed": True}),            # wrong symbol
    ]) + "\n")

    rows = dc.load_candidates_for("cast", "2026-06-15")  # case-insensitive symbol
    assert [r["score"] for r in rows] == [95, 72]          # sorted by ts, only CAST 6/15
    assert rows[0]["passed"] is True and rows[1]["passed"] is False
    assert rows[0]["rel_vol"] == 2.0
    assert "rvol0.4x=-25" in rows[1]["breakdown"]


def test_load_candidates_for_missing_inputs_and_file():
    assert dc.load_candidates_for("", "2026-06-15") == []
    assert dc.load_candidates_for("CAST", "") == []
    # No file written yet under the isolated dir -> empty, no error.
    assert dc.load_candidates_for("NONE", "2026-06-15") == []
