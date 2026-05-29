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

            # Find the most recent passed entry for this symbol without an outcome
            updated = False
            for i in range(len(lines) - 1, -1, -1):
                try:
                    rec = json.loads(lines[i])
                except json.JSONDecodeError:
                    continue
                if rec.get("symbol") == symbol and rec.get("passed") and rec.get("outcome_pnl") is None:
                    rec["outcome_pnl"] = round(pnl_pct, 4)
                    rec["outcome_pct"] = round(pnl_pct, 4)
                    rec["outcome_duration_s"] = duration_s
                    lines[i] = json.dumps(rec)
                    updated = True
                    break

            if updated:
                _CANDIDATES_FILE.write_text("\n".join(lines) + "\n")

    except Exception as exc:
        logger.debug("Outcome backfill error: %s", exc)


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
