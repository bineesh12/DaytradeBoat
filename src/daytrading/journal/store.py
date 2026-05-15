"""Persistent trade journal for analysis, mistakes, and replay.

Primary storage is SQLite so data is queryable for later research.
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import shutil
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from daytrading.models import Bar


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _day_key(ts: Optional[datetime] = None) -> str:
    return (ts or _utc_now()).strftime("%Y-%m-%d")


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return getattr(value, "value")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class TradingJournal:
    """SQLite-backed journal for strategy learning and replay."""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        screenshot_dir: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> None:
        root = base_dir or os.environ.get("DAYTRADING_JOURNAL_DIR", "data/journal")
        self._base_dir = os.path.abspath(root)
        self._event_dir = os.path.join(self._base_dir, "events")  # legacy mirror (optional)
        self._shot_dir = os.path.abspath(
            screenshot_dir or os.environ.get("DAYTRADING_SCREENSHOT_DIR", os.path.join(self._base_dir, "screenshots"))
        )
        self._db_path = os.path.abspath(db_path or os.environ.get("DAYTRADING_JOURNAL_DB", os.path.join(self._base_dir, "journal.db")))
        os.makedirs(self._event_dir, exist_ok=True)
        os.makedirs(self._shot_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    @property
    def base_dir(self) -> str:
        return self._base_dir

    @property
    def db_path(self) -> str:
        return self._db_path

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            # Migrate older DBs where specialized tables still had payload_json.
            self._migrate_specialized_tables(cur)
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    day TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_day ON events(day);
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    trade_type TEXT,
                    strategy TEXT,
                    quantity REAL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    reason TEXT,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, ts);
                CREATE INDEX IF NOT EXISTS idx_trades_type ON trades(trade_type);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);

                CREATE TABLE IF NOT EXISTS market_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    cycle INTEGER,
                    phase TEXT,
                    symbols_scanned INTEGER,
                    scan_hits INTEGER,
                    signals INTEGER,
                    fills INTEGER,
                    exits INTEGER,
                    rejected INTEGER,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );

                CREATE TABLE IF NOT EXISTS classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    style TEXT,
                    confidence REAL,
                    trend_strength REAL,
                    relative_volume REAL,
                    spread_pct REAL,
                    volatility_pct REAL,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );
                CREATE INDEX IF NOT EXISTS idx_classifications_symbol_ts ON classifications(symbol, ts);

                CREATE TABLE IF NOT EXISTS mistakes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    symbol TEXT,
                    kind TEXT,
                    reason TEXT,
                    scanner TEXT,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );
                CREATE INDEX IF NOT EXISTS idx_mistakes_symbol_ts ON mistakes(symbol, ts);
                CREATE INDEX IF NOT EXISTS idx_mistakes_kind ON mistakes(kind);

                CREATE TABLE IF NOT EXISTS market_regime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    cycle INTEGER,
                    phase TEXT,
                    spy_change_pct REAL,
                    regime_label TEXT,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );

                CREATE TABLE IF NOT EXISTS screenshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    path TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );
                CREATE INDEX IF NOT EXISTS idx_screenshots_symbol_ts ON screenshots(symbol, ts);

                CREATE TABLE IF NOT EXISTS candle_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    symbol TEXT,
                    bars_json TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );

                -- replay_events is now a VIEW over the events table (no duplication)
                CREATE VIEW IF NOT EXISTS replay_events AS
                    SELECT id AS event_id, ts, type, payload_json
                    FROM events
                    WHERE type IN (
                        'cycle','classification','scan_hit','signal',
                        'trade_fill','trade_exit','mistake',
                        'market_context','market_regime','screenshot'
                    );
                """
            )
            self._conn.commit()

    @staticmethod
    def _has_column(cur: sqlite3.Cursor, table: str, column: str) -> bool:
        cur.execute("PRAGMA table_info({})".format(table))
        rows = cur.fetchall()
        return any(r[1] == column for r in rows)

    def _migrate_specialized_tables(self, cur: sqlite3.Cursor) -> None:
        # trades
        if self._has_column(cur, "trades", "payload_json"):
            cur.executescript(
                """
                CREATE TABLE trades_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    trade_type TEXT,
                    strategy TEXT,
                    quantity REAL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    reason TEXT,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );
                INSERT INTO trades_new(id, event_id, ts, symbol, side, trade_type, quantity, entry_price, exit_price, pnl, reason)
                SELECT id, event_id, ts, symbol, side, trade_type, quantity, entry_price, exit_price, pnl, reason
                FROM trades;
                DROP TABLE trades;
                ALTER TABLE trades_new RENAME TO trades;
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, ts);
                CREATE INDEX IF NOT EXISTS idx_trades_type ON trades(trade_type);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
                """
            )
        elif not self._has_column(cur, "trades", "strategy"):
            try:
                cur.execute("ALTER TABLE trades ADD COLUMN strategy TEXT")
            except Exception:
                pass

        # market_context
        if self._has_column(cur, "market_context", "payload_json"):
            cur.executescript(
                """
                CREATE TABLE market_context_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    cycle INTEGER,
                    phase TEXT,
                    symbols_scanned INTEGER,
                    scan_hits INTEGER,
                    signals INTEGER,
                    fills INTEGER,
                    exits INTEGER,
                    rejected INTEGER,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );
                INSERT INTO market_context_new(id, event_id, ts, cycle, phase, symbols_scanned, scan_hits, signals, fills, exits, rejected)
                SELECT id, event_id, ts, cycle, phase, symbols_scanned, scan_hits, signals, fills, exits, rejected
                FROM market_context;
                DROP TABLE market_context;
                ALTER TABLE market_context_new RENAME TO market_context;
                """
            )

        # classifications
        if self._has_column(cur, "classifications", "payload_json"):
            cur.executescript(
                """
                CREATE TABLE classifications_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    style TEXT,
                    confidence REAL,
                    trend_strength REAL,
                    relative_volume REAL,
                    spread_pct REAL,
                    volatility_pct REAL,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );
                INSERT INTO classifications_new(id, event_id, ts, symbol, style, confidence)
                SELECT id, event_id, ts, symbol, style, confidence
                FROM classifications;
                DROP TABLE classifications;
                ALTER TABLE classifications_new RENAME TO classifications;
                CREATE INDEX IF NOT EXISTS idx_classifications_symbol_ts ON classifications(symbol, ts);
                """
            )
        else:
            for col, dtype in [("trend_strength", "REAL"), ("relative_volume", "REAL"), ("spread_pct", "REAL"), ("volatility_pct", "REAL")]:
                if not self._has_column(cur, "classifications", col):
                    try:
                        cur.execute("ALTER TABLE classifications ADD COLUMN {} {}".format(col, dtype))
                    except Exception:
                        pass

        # mistakes
        if self._has_column(cur, "mistakes", "payload_json"):
            cur.executescript(
                """
                CREATE TABLE mistakes_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    symbol TEXT,
                    kind TEXT,
                    reason TEXT,
                    scanner TEXT,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );
                INSERT INTO mistakes_new(id, event_id, ts, symbol, kind, reason)
                SELECT id, event_id, ts, symbol, kind, reason
                FROM mistakes;
                DROP TABLE mistakes;
                ALTER TABLE mistakes_new RENAME TO mistakes;
                CREATE INDEX IF NOT EXISTS idx_mistakes_symbol_ts ON mistakes(symbol, ts);
                CREATE INDEX IF NOT EXISTS idx_mistakes_kind ON mistakes(kind);
                """
            )
        elif not self._has_column(cur, "mistakes", "scanner"):
            try:
                cur.execute("ALTER TABLE mistakes ADD COLUMN scanner TEXT")
            except Exception:
                pass

        # market_regime
        if self._has_column(cur, "market_regime", "payload_json"):
            cur.executescript(
                """
                CREATE TABLE market_regime_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    cycle INTEGER,
                    phase TEXT,
                    spy_change_pct REAL,
                    regime_label TEXT,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );
                INSERT INTO market_regime_new(id, event_id, ts, cycle, phase)
                SELECT id, event_id, ts, cycle, phase
                FROM market_regime;
                DROP TABLE market_regime;
                ALTER TABLE market_regime_new RENAME TO market_regime;
                """
            )
        else:
            for col, dtype in [("spy_change_pct", "REAL"), ("regime_label", "TEXT")]:
                if not self._has_column(cur, "market_regime", col):
                    try:
                        cur.execute("ALTER TABLE market_regime ADD COLUMN {} {}".format(col, dtype))
                    except Exception:
                        pass

        # screenshots
        if self._has_column(cur, "screenshots", "payload_json"):
            cur.executescript(
                """
                CREATE TABLE screenshots_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    path TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                );
                INSERT INTO screenshots_new(id, event_id, ts, symbol, path, context_json)
                SELECT id, event_id, ts, symbol, path, context_json
                FROM screenshots;
                DROP TABLE screenshots;
                ALTER TABLE screenshots_new RENAME TO screenshots;
                CREATE INDEX IF NOT EXISTS idx_screenshots_symbol_ts ON screenshots(symbol, ts);
                """
            )

        # Drop the replay_events TABLE if it exists (replaced by VIEW)
        cur.execute("SELECT type FROM sqlite_master WHERE name='replay_events'")
        row = cur.fetchone()
        if row and row[0] == "table":
            cur.execute("DROP TABLE replay_events")

    def _insert_specialized(self, event_id: int, event_type: str, ts_iso: str, payload: Dict[str, Any]) -> None:
        cur = self._conn.cursor()

        if event_type in {"trade_fill", "trade_exit"}:
            cur.execute(
                """
                INSERT INTO trades(event_id, ts, symbol, side, trade_type, strategy, quantity, entry_price, exit_price, pnl, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    ts_iso,
                    str(payload.get("symbol", "")),
                    payload.get("side"),
                    payload.get("trade_type"),
                    payload.get("strategy") or payload.get("pattern"),
                    payload.get("quantity"),
                    payload.get("entry_price") if "entry_price" in payload else payload.get("price"),
                    payload.get("exit_price"),
                    payload.get("pnl"),
                    payload.get("reason"),
                ),
            )
        elif event_type == "cycle":
            cur.execute(
                """
                INSERT INTO market_context(event_id, ts, cycle, phase, symbols_scanned, scan_hits, signals, fills, exits, rejected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    ts_iso,
                    payload.get("cycle"),
                    payload.get("phase"),
                    payload.get("symbols_scanned"),
                    payload.get("scan_hits"),
                    payload.get("signals"),
                    payload.get("fills"),
                    payload.get("exits"),
                    payload.get("rejected"),
                ),
            )
        elif event_type == "classification":
            cur.execute(
                """
                INSERT INTO classifications(event_id, ts, symbol, style, confidence, trend_strength, relative_volume, spread_pct, volatility_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    ts_iso,
                    str(payload.get("symbol", "")),
                    payload.get("style"),
                    payload.get("confidence"),
                    payload.get("trend_strength"),
                    payload.get("relative_volume"),
                    payload.get("spread_pct"),
                    payload.get("volatility_pct"),
                ),
            )
        elif event_type == "mistake":
            cur.execute(
                """
                INSERT INTO mistakes(event_id, ts, symbol, kind, reason, scanner)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    ts_iso,
                    payload.get("symbol"),
                    payload.get("kind"),
                    payload.get("reason"),
                    payload.get("scanner"),
                ),
            )
        elif event_type == "market_regime":
            cur.execute(
                """
                INSERT INTO market_regime(event_id, ts, cycle, phase, spy_change_pct, regime_label)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    ts_iso,
                    payload.get("cycle"),
                    payload.get("phase"),
                    payload.get("spy_change_pct"),
                    payload.get("regime_label"),
                ),
            )
        elif event_type == "screenshot":
            cur.execute(
                """
                INSERT INTO screenshots(event_id, ts, symbol, path, context_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    ts_iso,
                    str(payload.get("symbol", "")),
                    str(payload.get("path", "")),
                    json.dumps(_json_safe(payload.get("context", {})), separators=(",", ":"), ensure_ascii=True),
                ),
            )

        bars = payload.get("candle_snapshot")
        if isinstance(bars, list) and bars:
            cur.execute(
                """
                INSERT INTO candle_snapshots(event_id, ts, symbol, bars_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    event_id,
                    ts_iso,
                    str(payload.get("symbol", "")),
                    json.dumps(_json_safe(bars), separators=(",", ":"), ensure_ascii=True),
                ),
            )

    def record(self, event_type: str, payload: Dict[str, Any], ts: Optional[datetime] = None) -> None:
        t = ts or _utc_now()
        event = {
            "ts": t.isoformat(),
            "day": _day_key(t),
            "type": event_type,
            "payload": _json_safe(payload),
        }
        payload_json = json.dumps(event["payload"], separators=(",", ":"), ensure_ascii=True)
        line = json.dumps(event, separators=(",", ":"), ensure_ascii=True)  # legacy mirror
        path = os.path.join(self._event_dir, "{}.jsonl".format(event["day"]))
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO events(ts, day, type, payload_json) VALUES (?, ?, ?, ?)",
                (event["ts"], event["day"], event_type, payload_json),
            )
            event_id = int(cur.lastrowid)
            self._insert_specialized(event_id, event_type, event["ts"], event["payload"])
            self._conn.commit()
            # Keep JSONL mirror so old tools/scripts keep working.
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def candle_snapshot(self, bars: Sequence[Bar], limit: int = 30) -> List[Dict[str, Any]]:
        recent = list(bars[-limit:]) if bars else []
        return [
            {
                "symbol": b.symbol,
                "ts": b.ts.isoformat(),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "timeframe": getattr(b.timeframe, "value", str(b.timeframe)),
            }
            for b in recent
        ]

    def save_screenshot(
        self,
        symbol: str,
        *,
        image_b64: Optional[str] = None,
        source_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not image_b64 and not source_path:
            raise ValueError("image_b64 or source_path is required")

        now = _utc_now()
        day_dir = os.path.join(self._shot_dir, _day_key(now))
        os.makedirs(day_dir, exist_ok=True)
        filename = "{}_{}_{}.png".format(symbol.upper(), now.strftime("%H%M%S"), uuid.uuid4().hex[:8])
        out_path = os.path.join(day_dir, filename)

        if image_b64:
            data = image_b64.split(",", 1)[-1]
            raw = base64.b64decode(data)
            with open(out_path, "wb") as f:
                f.write(raw)
        else:
            shutil.copy2(str(source_path), out_path)

        meta = {
            "symbol": symbol.upper(),
            "path": out_path,
            "context": context or {},
        }
        self.record("screenshot", meta, ts=now)
        return meta

    def load_events(self, day: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        target_day = day or _day_key()
        query = "SELECT ts, day, type, payload_json FROM events WHERE day = ? ORDER BY id ASC"
        params: List[Any] = [target_day]
        if limit is not None and limit >= 0:
            query = "SELECT ts, day, type, payload_json FROM events WHERE day = ? ORDER BY id DESC LIMIT ?"
            params.append(limit)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
        if limit is not None and limit >= 0:
            rows = list(reversed(rows))
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                out.append(
                    {
                        "ts": r["ts"],
                        "day": r["day"],
                        "type": r["type"],
                        "payload": json.loads(r["payload_json"]),
                    }
                )
            except Exception:
                continue
        return out

    def replay_frames(self, day: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        target_day = day or _day_key()
        allowed = (
            "cycle",
            "classification",
            "scan_hit",
            "signal",
            "trade_fill",
            "trade_exit",
            "mistake",
            "market_context",
            "market_regime",
            "screenshot",
        )
        placeholders = ",".join("?" for _ in allowed)
        query = (
            "SELECT ts, type, payload_json "
            "FROM events "
            "WHERE day = ? AND type IN ({}) "
            "ORDER BY id ASC".format(placeholders)
        )
        params: List[Any] = [target_day] + list(allowed)
        if limit is not None and limit >= 0:
            query = (
                "SELECT * FROM ("
                "SELECT ts, type, payload_json "
                "FROM events "
                "WHERE day = ? AND type IN ({}) "
                "ORDER BY id DESC LIMIT ?"
                ") sub ORDER BY ts ASC".format(placeholders)
            )
            params = [target_day] + list(allowed) + [limit]

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
        frames: List[Dict[str, Any]] = []
        for r in rows:
            try:
                frames.append(
                    {
                        "ts": r["ts"],
                        "day": target_day,
                        "type": r["type"],
                        "payload": json.loads(r["payload_json"]),
                    }
                )
            except Exception:
                continue
        return frames

    def close(self) -> None:
        """Flush WAL and close the SQLite connection cleanly."""
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.close()
