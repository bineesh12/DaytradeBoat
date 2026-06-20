"""Unbiased backtest universe from the bot's own journal.

The sweep itself lives in ``service.run_backtest_sweep`` (baseline vs each
experiment). This module only supplies an UNBIASED symbol/date basket for it:
the distinct names the bot actually evaluated on each day — not a hand-picked
set of winners — so a sweep reflects the real population, duds included.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Job = Tuple[str, str]  # (symbol, date)


def jobs_from_journal(
    journal_path: str,
    dates: Sequence[str],
    *,
    event_type: str = "entry_decision",
    max_per_day: int = 60,
) -> List[Job]:
    """Return (symbol, date) jobs for every symbol the bot evaluated on each day."""
    import json
    import sqlite3

    jobs: List[Job] = []
    conn = sqlite3.connect(journal_path)
    try:
        cur = conn.cursor()
        for day in dates:
            symbols: set = set()
            for (payload,) in cur.execute(
                "SELECT payload_json FROM events WHERE type = ? AND day = ?",
                (event_type, day),
            ).fetchall():
                try:
                    sym = json.loads(payload).get("symbol")
                except Exception:
                    sym = None
                if sym:
                    symbols.add(str(sym).upper())
            for sym in sorted(symbols)[:max_per_day]:
                jobs.append((sym, day))
    finally:
        conn.close()
    return jobs


def journal_universe(
    journal_path: str,
    dates: Sequence[str],
    *,
    event_type: str = "entry_decision",
    max_per_day: int = 60,
) -> Tuple[List[str], List[str]]:
    """Distinct symbols + dates for ``service.run_backtest_sweep(symbols, dates)``."""
    jobs = jobs_from_journal(
        journal_path, dates, event_type=event_type, max_per_day=max_per_day,
    )
    symbols = sorted({s for s, _ in jobs})
    used_dates = sorted({d for _, d in jobs})
    return symbols, used_dates
