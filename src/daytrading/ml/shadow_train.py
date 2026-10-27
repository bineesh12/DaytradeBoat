"""Train advisory shadow ML models from collected live JSONL datasets."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from daytrading.ml import shadow_collector as collector
from daytrading.ml.artifacts import archive_model, atomic_json, data_root
from daytrading.ml.validation import TrainingRow, chronological_split, parse_timestamp

logger = logging.getLogger(__name__)

MODEL_DIR = data_root() / "models"

DATASETS = {
    "missed_opportunity": (
        collector.MISSED_FILE,
        MODEL_DIR / "missed_opportunity_model.json",
    ),
    "pullback_entry": (
        collector.PULLBACK_FILE,
        MODEL_DIR / "pullback_entry_model.json",
    ),
    "exit_helper": (
        collector.EXIT_FILE,
        MODEL_DIR / "exit_helper_model.json",
    ),
    "execution_risk": (
        collector.EXECUTION_FILE,
        MODEL_DIR / "execution_risk_model.json",
    ),
}


def _flatten_features(record: dict) -> Dict[str, float]:
    features: Dict[str, float] = {}
    nested = record.get("features") or {}
    for key, value in nested.items():
        try:
            features[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    for key in (
        "price",
        "bar_count",
        "session_high",
        "day_volume",
        "entry_price",
        "remaining_qty",
        "unrealized_pct",
        "slippage_pct",
        "intended_price",
        "fill_price",
        "entry_score",
    ):
        if key not in record or record.get(key) is None:
            continue
        try:
            features[key] = float(record[key])
        except (TypeError, ValueError):
            continue
    for key in ("sold_half", "breakeven_locked", "failed_first"):
        if key in record:
            features[key] = 1.0 if record.get(key) else 0.0
    return features


def _matrix(records: List[dict], names=None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_maps = [_flatten_features(r) for r in records]
    names = names or sorted({k for fm in feature_maps for k in fm})
    if not names:
        return np.empty((0, 0)), np.empty((0,)), []
    X = np.array([[fm.get(name, 0.0) for name in names] for fm in feature_maps])
    y = np.array([int(r.get("label")) for r in records])
    return X, y, names


def train_shadow_model(name: str, *, min_samples: int = 50) -> bool:
    """Train one advisory model. Returns True if a model was written."""
    if name not in DATASETS:
        raise ValueError("unknown shadow model dataset: {}".format(name))
    data_path, model_path = DATASETS[name]
    records = collector.load_labeled(data_path)
    if len(records) < min_samples:
        logger.info(
            "ML SHADOW %s: skipped — only %d labeled samples (need %d)",
            name, len(records), min_samples,
        )
        return False

    lookup, dated = {}, []
    for record in records:
        try:
            ts = parse_timestamp(record["ts"])
            symbol = str(record.get("symbol", name)).upper()
            lookup[symbol, ts] = record
            dated.append(TrainingRow(symbol, ts, (0.0,), float(record["label"]), 0.0))
        except (ValueError, KeyError, TypeError):
            continue
    try:
        training, validation, test = chronological_split(dated, min_partition_rows=10)
    except ValueError as exc:
        logger.warning("ML SHADOW %s skipped: %s", name, exc)
        return False
    training_records = [lookup[r.symbol, r.ts] for r in training]
    testing_records = [lookup[r.symbol, r.ts] for r in test]
    X_train, y_train, feature_names = _matrix(training_records)
    X_test, y_test, _ = _matrix(testing_records, feature_names)
    if X_train.size == 0 or len(set(y_train.tolist())) < 2:
        logger.info("ML SHADOW %s: skipped — needs both training classes", name)
        return False

    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("ML SHADOW %s: xgboost unavailable", name)
        return False

    pos = int(sum(y_train))
    neg = len(y_train) - pos
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=(neg / pos if pos > 0 else 1.0),
        eval_metric="logloss",
        random_state=42, n_jobs=1,
    )
    model.fit(X_train, y_train, verbose=False)
    acc = float((model.predict(X_test) == y_test).mean()) if len(y_test) else 0.0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".json", dir=MODEL_DIR)
    os.close(fd)
    try:
        model.save_model(temporary)
        if model_path.exists():
            archive_model(model_path)
        os.replace(temporary, model_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    meta_path = model_path.with_suffix(".meta.json")
    atomic_json(meta_path, {
        "dataset": name,
        "advisory_only": True,
        "trained_through": max(r.session for r in training).isoformat(),
        "validation_start": min(r.session for r in validation).isoformat(),
        "test_start": min(r.session for r in test).isoformat(),
        "evaluated_through": max(r.session for r in test).isoformat(),
        "samples": len(records),
        "positive_rate": round(float(sum(r.label for r in dated)) / len(dated), 4),
        "test_accuracy": round(acc, 4),
        "feature_names": feature_names,
    })
    logger.info(
        "ML SHADOW %s: trained %d samples, acc %.1f%% → %s",
        name, len(records), acc * 100.0, model_path,
    )
    return True


def train_all_shadow_models(*, min_samples: int = 50) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for name in DATASETS:
        try:
            results[name] = train_shadow_model(name, min_samples=min_samples)
        except Exception as exc:
            logger.warning("ML SHADOW %s failed: %s", name, exc)
            results[name] = False
    return results
