"""Live data collector for ML training.

Logs every entry candidate (pass or reject) with features and metadata.
When a trade closes, the outcome (P&L) is backfilled into the log.

Data is stored as JSONL in data/ml/entry_candidates.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from daytrading.ml.features import FEATURE_NAMES, compute_entry_features
from daytrading.models import Bar

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "ml"
_CANDIDATES_FILE = _DATA_DIR / "entry_candidates.jsonl"

_lock = threading.Lock()
_pending_outcomes: Dict[str, str] = {}  # symbol -> line position marker


def _ensure_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def log_entry_candidate(
    *,
    symbol: str,
    price: float,
    score: int,
    passed: bool,
    reject_reason: Optional[str] = None,
    ml_prob: Optional[float] = None,
    breakdown: str = "",
    float_shares: Optional[float] = None,
    day_volume: float = 0.0,
    rel_vol: float = 0.0,
    bars: Optional[Sequence[Bar]] = None,
    session_high: float = 0.0,
    session_open: float = 0.0,
    prior_close: float = 0.0,
    minutes_since_open: int = 0,
) -> None:
    """Log a single entry candidate with features to JSONL."""
    try:
        features = compute_entry_features(
            price,
            float_shares=float_shares,
            day_volume=day_volume,
            rel_vol=rel_vol,
            session_high=session_high,
            session_open=session_open,
            prior_close=prior_close,
            bars=bars,
            minutes_since_open=minutes_since_open,
        )

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "price": round(price, 4),
            "score": score,
            "passed": passed,
            "reject_reason": reject_reason,
            "ml_prob": round(ml_prob, 4) if ml_prob is not None else None,
            "breakdown": breakdown,
            "float_shares": float_shares,
            "features": dict(zip(FEATURE_NAMES, features)),
            "outcome_pnl": None,
            "outcome_pct": None,
            "outcome_duration_s": None,
            "outcome_source": None,
        }

        _ensure_dir()
        with _lock:
            with open(_CANDIDATES_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")

    except Exception as exc:
        logger.debug("Data collector error: %s", exc)


def log_trade_outcome(
    *,
    symbol: str,
    entry_price: float,
    exit_price: float,
    entry_time: Optional[datetime] = None,
    exit_time: Optional[datetime] = None,
) -> None:
    """Backfill P&L outcome for the most recent entry candidate of this symbol."""
    if entry_price <= 0:
        return
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    duration_s = None
    if entry_time and exit_time:
        duration_s = int((exit_time - entry_time).total_seconds())

    try:
        _ensure_dir()
        if not _CANDIDATES_FILE.exists():
            return

        with _lock:
            lines = _CANDIDATES_FILE.read_text().splitlines()

            # Find the most recent passed entry for this symbol.  A shadow label
            # is only a provisional learning label; real closed-trade P&L must
            # replace it so training data prefers ground truth over future-price
            # inference.
            updated = False
            for i in range(len(lines) - 1, -1, -1):
                try:
                    rec = json.loads(lines[i])
                except json.JSONDecodeError:
                    continue
                if (
                    rec.get("symbol") == symbol
                    and rec.get("passed")
                    and (
                        rec.get("outcome_pnl") is None
                        or rec.get("outcome_source") == "shadow_future_price"
                    )
                ):
                    rec["outcome_pnl"] = round(pnl_pct, 4)
                    rec["outcome_pct"] = round(pnl_pct, 4)
                    rec["outcome_duration_s"] = duration_s
                    rec["outcome_source"] = "real_trade"
                    rec.pop("shadow_label", None)
                    rec.pop("labeled_at", None)
                    lines[i] = json.dumps(rec)
                    updated = True
                    break

            if updated:
                _CANDIDATES_FILE.write_text("\n".join(lines) + "\n")

    except Exception as exc:
        logger.debug("Outcome backfill error: %s", exc)


def update_deferred_entry_outcomes(
    bar_universe: Dict[str, Sequence[Bar]],
    *,
    wait_seconds: float = 180.0,
    min_move_pct: float = 1.5,
) -> int:
    """Label old entry candidates from same-day future price movement.

    Real fills still use ``log_trade_outcome``.  This shadow label gives the
    entry model more learning data for candidates that passed/rejected but did
    not become a completed trade.  Rows are only labeled against a latest bar
    from the same UTC day so old candidates are not accidentally labeled with a
    later trading day's price.
    """
    if not _CANDIDATES_FILE.exists():
        return 0

    latest: Dict[str, Bar] = {
        sym: bars[-1]
        for sym, bars in bar_universe.items()
        if bars and bars[-1].close > 0
    }
    if not latest:
        return 0

    now = datetime.now(timezone.utc)
    changed = 0
    try:
        _ensure_dir()
        with _lock:
            lines = _CANDIDATES_FILE.read_text().splitlines()
            for i, line in enumerate(lines):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("outcome_pnl") is not None:
                    continue
                # Passed entries may still become real trades.  Do not stamp a
                # shadow label onto a still-open/unknown real fill; otherwise
                # log_trade_outcome can lose the ground-truth P&L later.
                if rec.get("passed"):
                    continue
                sym = str(rec.get("symbol") or "")
                bar = latest.get(sym)
                if bar is None or bar.ts is None:
                    continue
                try:
                    rec_ts = datetime.fromisoformat(str(rec.get("ts")).replace("Z", "+00:00"))
                    if rec_ts.tzinfo is None:
                        rec_ts = rec_ts.replace(tzinfo=timezone.utc)
                    bar_ts = bar.ts
                    if bar_ts.tzinfo is None:
                        bar_ts = bar_ts.replace(tzinfo=timezone.utc)
                    rec_ts = rec_ts.astimezone(timezone.utc)
                    bar_ts = bar_ts.astimezone(timezone.utc)
                except Exception:
                    continue
                if rec_ts.date() != bar_ts.date():
                    continue
                if (now - rec_ts).total_seconds() < wait_seconds:
                    continue
                entry = float(rec.get("price") or 0.0)
                if entry <= 0:
                    continue
                move_pct = (float(bar.close) - entry) / entry * 100.0
                rec["outcome_pnl"] = round(move_pct, 4)
                rec["outcome_pct"] = round(move_pct, 4)
                rec["outcome_duration_s"] = int((bar_ts - rec_ts).total_seconds())
                rec["outcome_source"] = "shadow_future_price"
                rec["shadow_label"] = 1 if move_pct >= min_move_pct else 0
                rec["labeled_at"] = now.isoformat()
                lines[i] = json.dumps(rec)
                changed += 1
            if changed:
                _CANDIDATES_FILE.write_text("\n".join(lines) + "\n")
    except Exception as exc:
        logger.debug("Deferred entry outcome update error: %s", exc)
        return 0
    return changed


def load_training_data() -> List[dict]:
    """Load all candidates with outcomes for training."""
    if not _CANDIDATES_FILE.exists():
        return []

    records = []
    with open(_CANDIDATES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("outcome_pnl") is not None:
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def load_candidates_for(
    symbol: str,
    day: str,
    *,
    limit: int = 5000,
) -> List[dict]:
    """Return the live entry-score candidates for one symbol on one UTC date.

    The live/paper bot writes one record per entry-quality check (pass AND
    reject) from ``check_entry_quality`` — score, the full point breakdown, and
    rvol. This surfaces them so the dashboard can show what a name actually
    scored in paper next to a backtest of the same name/day (the paper-vs-
    backtest score gap). The file is large, so each line is string-prefiltered
    before JSON parsing. ``day`` is a ``YYYY-MM-DD`` UTC date.
    """
    if not _CANDIDATES_FILE.exists():
        return []
    sym = str(symbol or "").upper().strip()
    day = str(day or "").strip()
    if not sym or not day:
        return []
    sym_token = '"symbol": "{}"'.format(sym)
    records: List[dict] = []
    with open(_CANDIDATES_FILE) as f:
        for line in f:
            if sym_token not in line or day not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("symbol", "")).upper() != sym:
                continue
            ts = str(rec.get("ts", ""))
            if not ts.startswith(day):
                continue
            records.append({
                "ts": ts,
                "symbol": rec.get("symbol"),
                "price": rec.get("price"),
                "score": rec.get("score"),
                "passed": bool(rec.get("passed")),
                "reject_reason": rec.get("reject_reason"),
                "breakdown": rec.get("breakdown"),
                "rel_vol": rec.get("rel_vol"),
                "ml_prob": rec.get("ml_prob"),
            })
    records.sort(key=lambda r: r["ts"])
    if limit and len(records) > limit:
        records = records[-limit:]
    return records


def count_candidates() -> int:
    """Count total logged candidates."""
    if not _CANDIDATES_FILE.exists():
        return 0
    count = 0
    with open(_CANDIDATES_FILE) as f:
        for _ in f:
            count += 1
    return count


def count_labeled() -> int:
    """Count candidates with outcome labels."""
    if not _CANDIDATES_FILE.exists():
        return 0
    count = 0
    with open(_CANDIDATES_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                if rec.get("outcome_pnl") is not None:
                    count += 1
            except (json.JSONDecodeError, ValueError):
                continue
    return count
