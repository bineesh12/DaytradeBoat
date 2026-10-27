from daytrading.models import BarSource
import json, sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock
from daytrading.opportunities import OpportunityTracker, hit_identity
from daytrading.dashboard.hub import DashboardHub
from daytrading.dashboard.server import create_app
from daytrading.models import Bar, ScanResult, SignalAction, TradeSignal, Fill, Side
from daytrading.strategy.entry_policy import EntryPolicy
from daytrading.strategy.execution_timer import ExecutionTimer

NOW = datetime(2026, 10, 15, 18, tzinfo=timezone.utc)


def decision(stage, passed, reason="", **extra):
    return dict(
        opportunity_id="setup1",
        bar_ts=NOW.isoformat(),
        symbol="BENF",
        scanner="vwap_pullback",
        stage=stage,
        passed=passed,
        reason=reason,
        ts=NOW.isoformat(),
        **extra,
    )


def test_repeated_detection_cannot_erase_guard_rejection():
    t = OpportunityTracker()
    t.observe(decision("pattern", True), ("pattern",))
    t.observe(decision("trade_guard", False, "LOW ACTIVITY: waiting for fresh trades"))
    for _ in range(4):
        t.observe(
            decision("pattern", True, "pattern detected; evaluating"), ("pattern",)
        )
    r = t.snapshot()["decisions"][0]
    assert (
        r["stage"] == "trade_guard"
        and r["status"] == "rejected"
        and "LOW ACTIVITY" in r["reason"]
    )
    assert t.snapshot()["counts"]["pattern"] == 1 and len(r["history"]) == 2


def test_retry_preserves_previous_rejection_and_timer_result_and_snapshot_is_detached():
    t = OpportunityTracker()
    t.observe(decision("verifier", False, "entry score 79/100 needs 80"))
    t.observe(decision("verifier", True))
    t.observe(decision("deferred_to_timer", True), ("qualified",))
    t.observe(decision("execution_confirmation", True, "timer released"))
    t.observe(decision("timed_entry_chase", False, "price ran 4.3%, max 3.5%"))
    r = t.snapshot()["decisions"][0]
    assert (
        len(r["history"]) == 5 and r["last_rejection"]["stage"] == "timed_entry_chase"
    )
    r["history"].clear()
    assert len(t.snapshot()["decisions"][0]["history"]) == 5


def test_watch_only_separate_from_executable_reject_and_query_route(monkeypatch):
    monkeypatch.delenv("DAYTRADING_DASHBOARD_TOKEN", raising=False)
    hub = DashboardHub()
    hub.on_entry_decision(decision("scanner", False, "watch only: collecting data"))
    hub.on_entry_decision(
        dict(
            decision("trade_guard", False, "spread too wide"),
            opportunity_id="setup2",
            symbol="NCPL",
        )
    )
    client = create_app(hub).test_client()
    rows = client.get("/api/entry-setups?query=BENF&status=watch_only").json
    assert (
        rows["matching_setups"] == 1 and rows["decisions"][0]["status"] == "watch_only"
    )
    assert (
        client.get("/api/entry-setups?query=NCPL&status=rejected").json[
            "matching_setups"
        ]
        == 1
    )
    assert not any(rows["counts"].values())


def test_today_journal_restores_exact_ids_and_reasons_without_prior_day_or_invented_fills(
    tmp_path,
):
    p = tmp_path / "journal.db"
    c = sqlite3.connect(p)
    c.execute(
        "create table events(id integer primary key,ts text,type text,payload_json text)"
    )
    rows = [
        (
            NOW - timedelta(days=1),
            "entry_decision",
            dict(decision("verifier", False, "yesterday"), opportunity_id="old"),
        ),
        (
            NOW,
            "scan_hit",
            dict(
                symbol="BENF",
                scanner="vwap_pullback",
                opportunity_id="setup1",
                bar_ts=NOW.isoformat(),
            ),
        ),
        (NOW, "entry_decision", decision("trade_guard", False, "LOW ACTIVITY")),
        (NOW, "scan_hit", dict(symbol="LEGACY", scanner="pullback_base")),
        (NOW, "trade_fill", dict(symbol="BENF", quantity=100, price=1.1)),
    ]
    for ts, kind, payload in rows:
        c.execute(
            "insert into events(ts,type,payload_json) values(?,?,?)",
            (ts.isoformat(), kind, json.dumps(payload)),
        )
    c.commit()
    c.close()
    before = p.read_bytes()
    h = DashboardHub()
    h.journal = SimpleNamespace(db_path=str(p))
    result = h.restore_entry_history(NOW + timedelta(seconds=1))
    r = h.funnel_snapshot()
    assert r["counts"]["pattern"] == 1 and r["counts"]["filled"] == 0
    assert (
        r["decisions"][0]["reason"] == "LOW ACTIVITY"
        and result["unlinked_legacy_events"] == 1
    )
    assert p.read_bytes() == before
    h.restore_entry_history(NOW + timedelta(seconds=2))
    assert h.funnel_snapshot()["counts"]["pattern"] == 1


def test_actual_fill_and_full_exit_are_journalled_and_restore_but_partial_exit_not_closed(
    tmp_path,
):
    p = tmp_path / "journal.db"
    c = sqlite3.connect(p)
    c.execute(
        "create table events(id integer primary key,ts text,type text,payload_json text)"
    )

    def record(kind, payload, **kw):
        c.execute(
            "insert into events(ts,type,payload_json) values(?,?,?)",
            (NOW.isoformat(), kind, json.dumps(payload)),
        )
        c.commit()

    h = DashboardHub()
    h.journal = SimpleNamespace(db_path=str(p), record=record)
    hit = ScanResult(
        "BENF",
        "vwap_pullback",
        NOW,
        90,
        bars=[Bar("BENF", NOW, 1.1, 1.2, 1.0, 1.15, 10000, provenance=BarSource.CANONICAL_INTERVAL)],
    )
    sig = TradeSignal("BENF", SignalAction.ENTER_LONG, 100, 1.15, 1.1, scan_result=hit)
    h.on_opportunity_fill(sig, Fill("BENF", Side.BUY, 100, 1.15, NOW))
    h.on_opportunity_exit("BENF", False)
    assert h.funnel_snapshot()["counts"]["closed"] == 0
    h.on_opportunity_exit("BENF", True)
    assert h.funnel_snapshot()["counts"]["closed"] == 1
    restored = DashboardHub()
    restored.journal = SimpleNamespace(db_path=str(p))
    restored.restore_entry_history(NOW + timedelta(days=0, seconds=1))
    assert (
        restored.funnel_snapshot()["counts"]["filled"] == 1
        and restored.funnel_snapshot()["counts"]["closed"] == 1
    )
    assert restored.funnel_snapshot()["decisions"][0]["status"] == "closed"


def test_timer_release_callback_is_observable_and_failure_does_not_block_execution():
    now = datetime.now(timezone.utc)
    sig = TradeSignal("BENF", SignalAction.ENTER_LONG, 10, 1.1)
    callback = Mock(side_effect=RuntimeError("UI unavailable"))
    timer = ExecutionTimer(on_release=callback)
    timer.queue(sig)
    assert timer.on_10s_bar(Bar("BENF", now, 1.1, 1.2, 1.1, 1.2, 10000, provenance=BarSource.CANONICAL_INTERVAL)) is sig
    callback.assert_called_once_with(sig)
    assert not timer.pending_symbols
