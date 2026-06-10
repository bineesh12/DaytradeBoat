"""Shadow ML data collection for scalping improvements.

These records are advisory-only.  They let the bot learn from missed
opportunities, pullback candidates, exit decisions, and execution quality
without changing live trade behavior.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from daytrading.ml.features import FEATURE_NAMES, compute_entry_features
from daytrading.models import Bar, Fill, Order, OrderStatus, Quote

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "ml"
MISSED_FILE = DATA_DIR / "missed_opportunities.jsonl"
PULLBACK_FILE = DATA_DIR / "pullback_candidates.jsonl"
EXIT_FILE = DATA_DIR / "exit_snapshots.jsonl"
EXECUTION_FILE = DATA_DIR / "execution_quality.jsonl"

_lock = threading.Lock()


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: Optional[datetime] = None) -> str:
    ts = ts or _utc_now()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("Ignoring malformed ML JSONL row in %s", path)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    _ensure_dir()
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _append(path: Path, record: dict) -> None:
    _ensure_dir()
    with _lock:
        with path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _latest_quote_features(quotes: Optional[Sequence[Quote]]) -> dict:
    if not quotes:
        return {
            "spread_pct": 0.0,
            "bid_size": 0.0,
            "ask_size": 0.0,
        }
    q = quotes[-1]
    return {
        "spread_pct": round(q.spread_pct, 4),
        "bid_size": float(q.bid_size or 0.0),
        "ask_size": float(q.ask_size or 0.0),
    }


def _feature_record(
    *,
    symbol: str,
    price: float,
    bars: Optional[Sequence[Bar]] = None,
    quotes: Optional[Sequence[Quote]] = None,
    float_shares: Optional[float] = None,
    rel_vol: float = 0.0,
) -> dict:
    bars = list(bars or [])
    today = bars
    session_high = max((b.high for b in today), default=price)
    session_open = today[0].open if today else price
    day_volume = float(sum(b.volume for b in today))
    prior_close = session_open
    if session_open > 0 and today:
        first = today[0]
        prior_close = first.open

    computed = compute_entry_features(
        price,
        float_shares=float_shares,
        day_volume=day_volume,
        rel_vol=rel_vol,
        session_high=session_high,
        session_open=session_open,
        prior_close=prior_close,
        bars=today,
        minutes_since_open=len(today),
    )
    features = dict(zip(FEATURE_NAMES, computed))
    features.update(_latest_quote_features(quotes))
    return {
        "symbol": symbol,
        "price": round(float(price), 4),
        "features": features,
        "bar_count": len(today),
        "session_high": round(float(session_high), 4),
        "day_volume": day_volume,
    }


def _future_outcome(
    row: dict,
    current_price: float,
    *,
    positive_pct: float,
    fail_pct: float,
) -> bool:
    entry = float(row.get("price") or 0.0)
    if entry <= 0 or current_price <= 0:
        return False
    move_pct = (current_price - entry) / entry * 100.0
    row["future_return_pct"] = round(move_pct, 4)
    row["label"] = 1 if move_pct >= positive_pct else 0
    row["failed_first"] = move_pct <= -abs(fail_pct)
    row["labeled_at"] = _iso()
    return True


def _label_pending_file(
    path: Path,
    latest_prices: Dict[str, float],
    *,
    wait_seconds: float,
    positive_pct: float,
    fail_pct: float,
) -> int:
    with _lock:
        rows = _read_jsonl(path)
        changed = 0
        now = _utc_now()
        for row in rows:
            if row.get("label") is not None:
                continue
            sym = str(row.get("symbol") or "")
            price = latest_prices.get(sym)
            if price is None:
                continue
            try:
                ts = datetime.fromisoformat(str(row.get("ts")).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if (now - ts.astimezone(timezone.utc)).total_seconds() < wait_seconds:
                continue
            if _future_outcome(row, price, positive_pct=positive_pct, fail_pct=fail_pct):
                changed += 1
        if changed:
            _write_jsonl(path, rows)
        return changed


def update_deferred_outcomes(bar_universe: Dict[str, Sequence[Bar]]) -> dict:
    """Label pending missed/pullback rows from latest prices."""
    latest = {
        sym: bars[-1].close
        for sym, bars in bar_universe.items()
        if bars and bars[-1].close > 0
    }
    if not latest:
        return {"missed": 0, "pullback": 0}
    missed = _label_pending_file(
        MISSED_FILE, latest,
        wait_seconds=180.0, positive_pct=2.0, fail_pct=1.5,
    )
    pullback = _label_pending_file(
        PULLBACK_FILE, latest,
        wait_seconds=180.0, positive_pct=1.5, fail_pct=1.2,
    )
    return {"missed": missed, "pullback": pullback}


def log_missed_opportunity(
    *,
    symbol: str,
    price: float,
    reason: str,
    scanner: str = "",
    bars: Optional[Sequence[Bar]] = None,
    quotes: Optional[Sequence[Quote]] = None,
    ml_prob: Optional[float] = None,
    scanner_score: Optional[float] = None,
    criteria: Optional[dict] = None,
) -> None:
    """Log a watchlist setup skipped or rejected by rules/ML."""
    try:
        base = _feature_record(symbol=symbol, price=price, bars=bars, quotes=quotes)
        base.update({
            "ts": _iso(),
            "kind": "missed_opportunity",
            "reason": reason,
            "scanner": scanner,
            "ml_prob": ml_prob,
            "scanner_score": scanner_score,
            "criteria": dict(criteria or {}),
            "label": None,
        })
        _append(MISSED_FILE, base)
    except Exception as exc:
        logger.debug("Missed-opportunity collector error: %s", exc)


def log_pullback_candidate(
    *,
    symbol: str,
    price: float,
    scanner: str,
    criteria: Optional[dict] = None,
    bars: Optional[Sequence[Bar]] = None,
    quotes: Optional[Sequence[Quote]] = None,
) -> None:
    """Log a breakout/pullback/reclaim candidate for shadow labeling."""
    try:
        base = _feature_record(symbol=symbol, price=price, bars=bars, quotes=quotes)
        base.update({
            "ts": _iso(),
            "kind": "pullback_candidate",
            "scanner": scanner,
            "criteria": dict(criteria or {}),
            "label": None,
        })
        _append(PULLBACK_FILE, base)
    except Exception as exc:
        logger.debug("Pullback collector error: %s", exc)


def log_exit_snapshot(
    *,
    symbol: str,
    price: float,
    entry_price: float,
    remaining_qty: float,
    sold_half: bool = False,
    breakeven_locked: bool = False,
    reason: str = "",
    entry_strategy: str = "",
    entry_pattern: str = "",
    entry_score: Optional[float] = None,
    bars: Optional[Sequence[Bar]] = None,
) -> None:
    """Log an open-position snapshot for hold/sell learning."""
    try:
        base = _feature_record(symbol=symbol, price=price, bars=bars)
        pnl_pct = (price - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0
        base.update({
            "ts": _iso(),
            "kind": "exit_snapshot",
            "entry_price": round(float(entry_price or 0.0), 4),
            "remaining_qty": float(remaining_qty or 0.0),
            "sold_half": bool(sold_half),
            "breakeven_locked": bool(breakeven_locked),
            "unrealized_pct": round(pnl_pct, 4),
            "reason": reason,
            "entry_strategy": entry_strategy,
            "entry_pattern": entry_pattern,
            "entry_score": round(float(entry_score), 4) if entry_score is not None else None,
            "label": None,
        })
        _append(EXIT_FILE, base)
    except Exception as exc:
        logger.debug("Exit snapshot collector error: %s", exc)


def label_exit_snapshots(symbol: str, exit_price: float) -> int:
    """Label pending exit snapshots for a symbol after the trade closes."""
    with _lock:
        rows = _read_jsonl(EXIT_FILE)
        changed = 0
        for row in rows:
            if row.get("symbol") != symbol or row.get("label") is not None:
                continue
            snap_price = float(row.get("price") or 0.0)
            if snap_price <= 0 or exit_price <= 0:
                continue
            move_after_snapshot = (exit_price - snap_price) / snap_price * 100.0
            row["future_return_pct"] = round(move_after_snapshot, 4)
            row["label"] = 1 if move_after_snapshot > 0.5 else 0
            row["labeled_at"] = _iso()
            changed += 1
        if changed:
            _write_jsonl(EXIT_FILE, rows)
        return changed


def log_execution_quality(
    *,
    order: Order,
    bar: Bar,
    status: OrderStatus,
    fill: Optional[Fill] = None,
    source: str = "",
    quote: Optional[Quote] = None,
    context: Optional[dict] = None,
) -> None:
    """Log order/fill quality for future slippage-risk learning."""
    try:
        intended = float(order.limit_price or bar.close or 0.0)
        fill_price = float(fill.price) if fill is not None else None
        slippage_pct = None
        if intended > 0 and fill_price is not None:
            if order.side.value == "buy":
                slippage_pct = (fill_price - intended) / intended * 100.0
            else:
                slippage_pct = (intended - fill_price) / intended * 100.0
        label = None
        if status is not OrderStatus.FILLED:
            label = 0
        elif slippage_pct is not None:
            label = 0 if slippage_pct > 0.75 else 1

        record = {
            "ts": _iso(fill.ts if fill else bar.ts),
            "kind": "execution_quality",
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": float(order.quantity),
            "intended_price": round(intended, 4),
            "fill_price": round(fill_price, 4) if fill_price is not None else None,
            "status": status.value,
            "source": source,
            "slippage_pct": round(slippage_pct, 4) if slippage_pct is not None else None,
            "label": label,
            "features": {
                "price": round(float(bar.close or intended or 0.0), 4),
                "bar_range_pct": (
                    round((bar.high - bar.low) / bar.low * 100.0, 4)
                    if bar.low > 0 else 0.0
                ),
                **_latest_quote_features([quote] if quote else None),
                **(context or {}),
            },
        }
        _append(EXECUTION_FILE, record)
    except Exception as exc:
        logger.debug("Execution-quality collector error: %s", exc)


def load_labeled(path: Path) -> List[dict]:
    return [row for row in _read_jsonl(path) if row.get("label") is not None]


def dataset_counts() -> dict:
    files = {
        "missed": MISSED_FILE,
        "pullback": PULLBACK_FILE,
        "exit": EXIT_FILE,
        "execution": EXECUTION_FILE,
    }
    counts = {}
    for name, path in files.items():
        rows = _read_jsonl(path)
        labeled = [r for r in rows if r.get("label") is not None]
        counts[name] = {"total": len(rows), "labeled": len(labeled)}
    return counts
