from __future__ import annotations

import json
import sqlite3

from daytrading.backtest.batch import jobs_from_journal, journal_universe


def _make_journal(tmp_path):
    db = tmp_path / "journal.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE events (type TEXT, day TEXT, payload_json TEXT)")
    conn.executemany("INSERT INTO events VALUES (?,?,?)", [
        ("entry_decision", "2026-06-12", json.dumps({"symbol": "CUPR"})),
        ("entry_decision", "2026-06-12", json.dumps({"symbol": "cupr"})),   # dup, case-insensitive
        ("entry_decision", "2026-06-12", json.dumps({"symbol": "DSY"})),
        ("entry_decision", "2026-06-11", json.dumps({"symbol": "RKLZ"})),
        ("other", "2026-06-12", json.dumps({"symbol": "NOPE"})),            # wrong type
    ])
    conn.commit()
    conn.close()
    return str(db)


def test_jobs_from_journal_builds_unbiased_universe(tmp_path) -> None:
    jobs = jobs_from_journal(_make_journal(tmp_path), ["2026-06-12", "2026-06-11"])

    assert ("CUPR", "2026-06-12") in jobs
    assert ("DSY", "2026-06-12") in jobs
    assert ("RKLZ", "2026-06-11") in jobs
    assert ("NOPE", "2026-06-12") not in jobs               # wrong event type excluded
    assert len([j for j in jobs if j[0] == "CUPR"]) == 1     # deduped


def test_journal_universe_returns_distinct_symbols_and_dates(tmp_path) -> None:
    symbols, dates = journal_universe(_make_journal(tmp_path), ["2026-06-12", "2026-06-11"])

    assert symbols == ["CUPR", "DSY", "RKLZ"]
    assert dates == ["2026-06-11", "2026-06-12"]
