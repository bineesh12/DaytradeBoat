"""Train XGBoost entry model from historical Alpaca bar data.

Usage:
    PYTHONPATH=src python -m daytrading.ml.train

Requires: ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.
Output: data/models/entry_model.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "entry_model.json"

MIN_FLOAT = 500_000
MAX_FLOAT = 20_000_000
MIN_PRICE = 2.0
MAX_PRICE = 20.0
MIN_GAP_PCT = 10.0
STOP_LOSS_PCT = 3.0
TARGET_PCT = 2.0
LOOKAHEAD_BARS = 5


def _fetch_movers(api, start_date: str, end_date: str) -> List[str]:
    """Get symbols that had big moves in the date range using Alpaca screener."""
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.data.historical import StockHistoricalDataClient

    logger.info("Fetching active low-float movers from %s to %s...", start_date, end_date)
    symbols = set()
    try:
        from daytrading.data.watchlist_scanner import WatchlistScanner
        scanner = WatchlistScanner(
            min_price=MIN_PRICE,
            max_price=MAX_PRICE,
            min_change_pct=MIN_GAP_PCT,
        )
        candidates = scanner.scan()
        for c in candidates:
            symbols.add(c["symbol"])
    except Exception as exc:
        logger.warning("Scanner fallback: %s", exc)

    if len(symbols) < 20:
        fallback = [
            "AMSS", "ASTC", "CODX", "MTVA", "QTTB", "VCIG", "FGL",
            "SDOT", "AEMD", "CPSH", "IPWR", "BRCB", "TENX", "MNTS",
            "RELL", "DKI", "AKTX", "MED", "RYOJ", "GOAI",
        ]
        symbols.update(fallback)

    return list(symbols)[:100]


def _fetch_bars(api, symbol: str, start: datetime, end: datetime) -> List[dict]:
    """Fetch 1-minute bars for a symbol."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
        timeframe=TimeFrame.Minute,
    )
    try:
        result = api.get_stock_bars(request)
        data = result[symbol] if symbol in result.data else []
        return [
            {
                "ts": b.timestamp,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume),
            }
            for b in data
        ]
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", symbol, exc)
        return []


def _find_breakout_entries(bars: List[dict]) -> List[int]:
    """Find indices where price breaks to new session high with volume."""
    entries = []
    session_high = 0.0
    for i, b in enumerate(bars):
        if i < 10:
            session_high = max(session_high, b["high"])
            continue
        if b["high"] > session_high and b["volume"] > 0:
            avg_vol = sum(bars[j]["volume"] for j in range(max(0, i-10), i)) / 10
            if b["volume"] >= avg_vol * 1.5:
                entries.append(i)
        session_high = max(session_high, b["high"])
    return entries


def _label_entry(bars: List[dict], idx: int) -> int:
    """Label: 1 if trade hits target before stop, 0 otherwise."""
    entry_price = bars[idx]["close"]
    stop = entry_price * (1 - STOP_LOSS_PCT / 100)
    target = entry_price * (1 + TARGET_PCT / 100)

    for j in range(idx + 1, min(idx + LOOKAHEAD_BARS + 1, len(bars))):
        if bars[j]["low"] <= stop:
            return 0
        if bars[j]["high"] >= target:
            return 1
    return 0


def _compute_features(bars: List[dict], idx: int, prior_close: float) -> List[float]:
    """Compute features for a breakout at index idx."""
    from daytrading.ml.features import compute_entry_features
    from daytrading.models import Bar

    bar_objs = [
        Bar(
            symbol="X",
            ts=b["ts"] if isinstance(b["ts"], datetime) else None,
            open=b["open"],
            high=b["high"],
            low=b["low"],
            close=b["close"],
            volume=b["volume"],
        )
        for b in bars[max(0, idx - 20):idx + 1]
    ]

    price = bars[idx]["close"]
    session_high = max(b["high"] for b in bars[:idx + 1])
    session_open = bars[0]["open"]
    day_volume = sum(b["volume"] for b in bars[:idx + 1])

    avg_vol_10 = sum(bars[j]["volume"] for j in range(max(0, idx-10), idx)) / 10 if idx >= 10 else 1
    rel_vol = bars[idx]["volume"] / avg_vol_10 if avg_vol_10 > 0 else 1.0

    minutes_since_open = idx

    return compute_entry_features(
        price,
        float_shares=5_000_000,
        day_volume=day_volume,
        rel_vol=rel_vol,
        session_high=session_high,
        session_open=session_open,
        prior_close=prior_close,
        bars=bar_objs,
        minutes_since_open=minutes_since_open,
    )


def _load_live_data() -> Tuple[List[List[float]], List[int]]:
    """Load training data from live collection (data/ml/entry_candidates.jsonl)."""
    from daytrading.ml.data_collector import load_training_data
    from daytrading.ml.features import FEATURE_NAMES

    records = load_training_data()
    if not records:
        return [], []

    X = []
    y = []
    for rec in records:
        features = rec.get("features")
        outcome = rec.get("outcome_pnl")
        if features is None or outcome is None:
            continue
        feature_vec = [features.get(name, 0.0) for name in FEATURE_NAMES]
        label = 1 if outcome > 0 else 0
        X.append(feature_vec)
        y.append(label)

    return X, y


def train() -> None:
    """Main training pipeline."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost scikit-learn")
        sys.exit(1)

    # Phase 1: Try to use live collected data
    X_live, y_live = _load_live_data()
    if len(X_live) >= 50:
        logger.info("Using LIVE collected data: %d labeled samples (%.0f%% positive)",
                    len(X_live), sum(y_live) / len(y_live) * 100)
        X_all = X_live
        y_all = y_live
    else:
        logger.info("Live data insufficient (%d samples) — falling back to Alpaca historical",
                    len(X_live))
        X_all, y_all = _fetch_historical_data()
        # Merge any live data we do have
        if X_live:
            X_all.extend(X_live)
            y_all.extend(y_live)
            logger.info("Merged %d live samples with %d historical", len(X_live), len(X_all) - len(X_live))

    if len(X_all) < 50:
        logger.error("Not enough training data: %d samples (need at least 50)", len(X_all))
        sys.exit(1)

    logger.info("Total samples: %d (%.0f%% positive)", len(X_all), sum(y_all) / len(y_all) * 100)

    X = np.array(X_all)
    y = np.array(y_all)

    # Time-based split: last 20% for test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Class weight to handle imbalance
    pos_count = sum(y_train)
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Test accuracy: %.1f%%", acc * 100)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["loss", "profit"]))

    # Feature importance
    from daytrading.ml.features import FEATURE_NAMES
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    logger.info("Feature importance:")
    for name, imp in ranked:
        logger.info("  %s: %.3f", name, imp)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    logger.info("Model saved to %s", MODEL_PATH)


def _fetch_historical_data() -> Tuple[List[List[float]], List[int]]:
    """Fetch training data from Alpaca historical bars (fallback)."""
    from alpaca.data.historical import StockHistoricalDataClient

    api_key = (
        os.environ.get("ALPACA_API_KEY")
        or os.environ.get("APCA_API_KEY_ID")
        or os.environ.get("DAYTRADING_ALPACA_API_KEY")
    )
    secret_key = (
        os.environ.get("ALPACA_SECRET_KEY")
        or os.environ.get("APCA_API_SECRET_KEY")
        or os.environ.get("DAYTRADING_ALPACA_SECRET_KEY")
    )
    if not api_key or not secret_key:
        logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        sys.exit(1)

    client = StockHistoricalDataClient(api_key, secret_key)

    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=60)

    symbols = _fetch_movers(client, start_date.isoformat(), end_date.isoformat())
    logger.info("Training on %d symbols over %d days", len(symbols), 60)

    X_all: List[List[float]] = []
    y_all: List[int] = []

    for i, sym in enumerate(symbols):
        bars = _fetch_bars(client, sym, start_date, end_date)
        if len(bars) < 30:
            continue

        # Group bars by day
        days: Dict[str, List[dict]] = {}
        for b in bars:
            ts = b["ts"]
            day_key = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
            days.setdefault(day_key, []).append(b)

        for day_key, day_bars in days.items():
            if len(day_bars) < 20:
                continue
            prior_close = day_bars[0]["open"] * 0.9  # approximate
            entries = _find_breakout_entries(day_bars)
            for idx in entries[:5]:
                try:
                    features = _compute_features(day_bars, idx, prior_close)
                    label = _label_entry(day_bars, idx)
                    X_all.append(features)
                    y_all.append(label)
                except Exception:
                    continue

        if (i + 1) % 10 == 0:
            logger.info("  Processed %d/%d symbols, %d samples so far", i + 1, len(symbols), len(X_all))

    if len(X_all) < 50:
        logger.warning("Historical data insufficient: %d samples", len(X_all))

    return X_all, y_all


if __name__ == "__main__":
    train()
