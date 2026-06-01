from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from daytrading.ml import shadow_collector as sc
from daytrading.models import Bar, Fill, Order, OrderStatus, Side


def _bar(symbol: str = "OLOX", close: float = 10.0, minutes_ago: int = 0) -> Bar:
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return Bar(
        symbol=symbol,
        ts=ts,
        open=close * 0.99,
        high=close * 1.01,
        low=close * 0.98,
        close=close,
        volume=100_000,
    )


@pytest.fixture()
def shadow_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sc, "MISSED_FILE", tmp_path / "missed_opportunities.jsonl")
    monkeypatch.setattr(sc, "PULLBACK_FILE", tmp_path / "pullback_candidates.jsonl")
    monkeypatch.setattr(sc, "EXIT_FILE", tmp_path / "exit_snapshots.jsonl")
    monkeypatch.setattr(sc, "EXECUTION_FILE", tmp_path / "execution_quality.jsonl")
    return tmp_path


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _age_first_row(path):
    rows = _rows(path)
    rows[0]["ts"] = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_missed_opportunity_is_labeled_after_wait(shadow_tmp):
    old = _bar(close=10.0, minutes_ago=5)
    sc.log_missed_opportunity(
        symbol="OLOX",
        price=10.0,
        reason="ML low confidence",
        scanner="hod_reclaim",
        bars=[old],
    )
    _age_first_row(sc.MISSED_FILE)

    sc.update_deferred_outcomes({"OLOX": [_bar(close=10.35)]})

    [row] = _rows(sc.MISSED_FILE)
    assert row["label"] == 1
    assert row["future_return_pct"] == pytest.approx(3.5)


def test_pullback_candidate_gets_negative_label(shadow_tmp):
    old = _bar(close=10.0, minutes_ago=5)
    sc.log_pullback_candidate(
        symbol="OLOX",
        price=10.0,
        scanner="pullback_base",
        criteria={"pattern": "pullback_base"},
        bars=[old],
    )
    _age_first_row(sc.PULLBACK_FILE)

    sc.update_deferred_outcomes({"OLOX": [_bar(close=9.85)]})

    [row] = _rows(sc.PULLBACK_FILE)
    assert row["label"] == 0
    assert row["failed_first"] is True


def test_exit_snapshots_label_after_trade_exit(shadow_tmp):
    sc.log_exit_snapshot(
        symbol="MASK",
        price=5.00,
        entry_price=4.90,
        remaining_qty=100,
        sold_half=False,
        breakeven_locked=True,
        bars=[_bar("MASK", close=5.00)],
    )

    changed = sc.label_exit_snapshots("MASK", 5.08)

    assert changed == 1
    [row] = _rows(sc.EXIT_FILE)
    assert row["label"] == 1
    assert row["future_return_pct"] == pytest.approx(1.6)


def test_execution_quality_labels_bad_slippage(shadow_tmp):
    bar = _bar("IOTR", close=5.00)
    order = Order("IOTR", Side.BUY, 100, limit_price=5.00)
    fill = Fill("IOTR", Side.BUY, 100, price=5.06, ts=bar.ts)

    sc.log_execution_quality(
        order=order,
        bar=bar,
        status=OrderStatus.FILLED,
        fill=fill,
        source="test",
    )

    [row] = _rows(sc.EXECUTION_FILE)
    assert row["label"] == 0
    assert row["slippage_pct"] == pytest.approx(1.2)
