"""Persistent SQLite cache for symbol float / share data."""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

SQLITE_BUSY_TIMEOUT_MS = 30000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def default_db_path() -> str:
    base = os.environ.get("DAYTRADING_JOURNAL_DIR", "data/journal")
    return os.path.abspath(
        os.environ.get("DAYTRADING_JOURNAL_DB", os.path.join(base, "journal.db")),
    )


@dataclass(frozen=True)
class FloatRecord:
    symbol: str
    float_shares: Optional[float]
    outstanding_shares: Optional[float]
    avg_volume: Optional[float]
    source: Optional[str]
    fetched_at: datetime

    def is_fresh(self, ttl_days: int) -> bool:
        age = _utc_now() - self.fetched_at
        return age <= timedelta(days=ttl_days)


class FloatStore:
    """Thread-safe SQLite store for float data keyed by symbol."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = os.path.abspath(db_path or default_db_path())
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
        )
        self._configure_connection(self._conn)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    @property
    def db_path(self) -> str:
        return self._db_path

    @staticmethod
    def _configure_connection(conn: sqlite3.Connection) -> None:
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS symbol_floats (
                    symbol TEXT PRIMARY KEY,
                    float_shares REAL,
                    outstanding_shares REAL,
                    avg_volume REAL,
                    source TEXT,
                    fetched_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_symbol_floats_fetched
                    ON symbol_floats(fetched_at);
                """,
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _parse_ts(value: str) -> datetime:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    def _row_to_record(self, row: sqlite3.Row) -> FloatRecord:
        return FloatRecord(
            symbol=str(row["symbol"]),
            float_shares=row["float_shares"],
            outstanding_shares=row["outstanding_shares"],
            avg_volume=row["avg_volume"],
            source=row["source"],
            fetched_at=self._parse_ts(row["fetched_at"]),
        )

    def get(self, symbol: str) -> Optional[FloatRecord]:
        sym = symbol.upper().strip()
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM symbol_floats WHERE symbol = ?",
                (sym,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def is_fresh(self, symbol: str, ttl_days: int = 7) -> bool:
        rec = self.get(symbol)
        return rec is not None and rec.is_fresh(ttl_days)

    def upsert(
        self,
        symbol: str,
        float_shares: Optional[float],
        outstanding_shares: Optional[float] = None,
        avg_volume: Optional[float] = None,
        *,
        source: str = "yfinance",
        fetched_at: Optional[datetime] = None,
    ) -> None:
        sym = symbol.upper().strip()
        ts = (fetched_at or _utc_now()).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO symbol_floats (
                    symbol, float_shares, outstanding_shares, avg_volume,
                    source, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    float_shares = excluded.float_shares,
                    outstanding_shares = excluded.outstanding_shares,
                    avg_volume = excluded.avg_volume,
                    source = excluded.source,
                    fetched_at = excluded.fetched_at
                """,
                (sym, float_shares, outstanding_shares, avg_volume, source, ts),
            )
            self._conn.commit()

    def bulk_get(self, symbols: Sequence[str]) -> Dict[str, FloatRecord]:
        syms = sorted({s.upper().strip() for s in symbols if s})
        if not syms:
            return {}
        placeholders = ",".join("?" for _ in syms)
        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM symbol_floats WHERE symbol IN ({placeholders})",
                syms,
            )
            rows = cur.fetchall()
        return {row["symbol"]: self._row_to_record(row) for row in rows}

    def count_stale(self, symbols: Sequence[str], ttl_days: int = 7) -> int:
        records = self.bulk_get(symbols)
        syms = {s.upper().strip() for s in symbols if s}
        stale = 0
        for sym in syms:
            rec = records.get(sym)
            if rec is None or not rec.is_fresh(ttl_days):
                stale += 1
        return stale
