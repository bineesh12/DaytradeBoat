"""Train only from recorded real outcomes, with forward session validation.

python -m daytrading.ml.train              # candidate + report, no promotion
python -m daytrading.ml.train --deploy     # promote only if validation passes
Historical bootstrap with invented float/prior-close features is unsupported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from daytrading.ml.artifacts import archive_model, atomic_json, data_root
from daytrading.ml.features import FEATURE_NAMES
from daytrading.ml.validation import (
    TrainingRow,
    chronological_split,
    parse_timestamp,
    promotion_reason,
    selected_metrics,
)

logger = logging.getLogger(__name__)
MODEL_DIR = data_root() / "models"
MODEL_PATH = MODEL_DIR / "entry_model.json"
MIN_LIVE_TRAIN_SAMPLES = int(os.getenv("DAYTRADING_ML_MIN_LIVE_SAMPLES", "200"))


def _load_rows():
    from daytrading.ml.data_collector import load_training_data

    rows = []
    cost_pct = float(os.getenv("DAYTRADING_ML_COST_BUFFER_PCT", "0.3"))
    if cost_pct < 0:
        raise ValueError("cost buffer must not be negative")
    for rec in load_training_data():
        if rec.get("outcome_source") != "real_trade" or not rec.get("trade_complete"):
            continue
        if rec.get("feed") != os.getenv("DAYTRADING_ALPACA_FEED", "iex"):
            continue
        try:
            feats = rec["features"]
            ts = parse_timestamp(rec["ts"])
            pnl = float(rec["outcome_pct"])
            after_cost = pnl - cost_pct
            rows.append(
                TrainingRow(
                    str(rec["symbol"]).upper(),
                    ts,
                    tuple(float(feats[name]) for name in FEATURE_NAMES),
                    float(after_cost > 0),
                    after_cost,
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return rows


def _matrix(rows):
    return np.asarray([r.features for r in rows]), np.asarray([r.label for r in rows])


def train(deploy: bool = False) -> bool:
    import xgboost as xgb

    try:
        rows = _load_rows()
        if len(rows) < MIN_LIVE_TRAIN_SAMPLES:
            raise ValueError(
                f"need {MIN_LIVE_TRAIN_SAMPLES} completed real-trade rows; found {len(rows)}"
            )
        training, validation, test = chronological_split(rows)
        X, y = _matrix(training)
        if len(set(y)) < 2:
            raise ValueError("training needs wins and losses")
    except ValueError as exc:
        logger.warning("Entry training skipped: %s", exc)
        return False
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y, verbose=False)
    # Choose the threshold only on validation dates; final dates are untouched.
    Xv, _ = _matrix(validation)
    pv = model.predict_proba(Xv)[:, 1]
    threshold = 0.30
    validation_metrics = selected_metrics(validation, pv, threshold)
    Xt, _ = _matrix(test)
    pt = model.predict_proba(Xt)[:, 1]
    test_metrics = selected_metrics(test, pt, threshold)
    incumbent_metrics = None
    incumbent_metadata = None
    if MODEL_PATH.exists():
        meta_path = MODEL_PATH.with_suffix(".meta.json")
        try:
            incumbent_metadata = (
                json.loads(meta_path.read_text()) if meta_path.exists() else {}
            )
            incumbent = xgb.Booster()
            incumbent.load_model(str(MODEL_PATH))
            incumbent_metrics = selected_metrics(
                test, incumbent.predict(xgb.DMatrix(Xt)), threshold
            )
        except (OSError, ValueError, TypeError, xgb.core.XGBoostError):
            incumbent_metadata = {}
    reason = promotion_reason(
        training, validation, test, test_metrics, incumbent_metadata, incumbent_metrics
    )
    if (
        validation_metrics["selected"] < 10
        or validation_metrics["mean_return_after_cost"] <= 0
    ):
        reason = "validation expectancy/count insufficient"
    report = {
        "samples": len(training) + len(validation) + len(test),
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "trained_through": max(r.session for r in training).isoformat(),
        "validation_start": min(r.session for r in validation).isoformat(),
        "test_start": min(r.session for r in test).isoformat(),
        "evaluated_through": max(r.session for r in test).isoformat(),
        "validation": validation_metrics,
        "final_holdout": test_metrics,
        "incumbent": incumbent_metrics,
        "promotion_allowed": reason is None,
        "promotion_reason": reason,
        "deployed": False,
        "cost_buffer_pct": float(os.getenv("DAYTRADING_ML_COST_BUFFER_PCT", "0.3")),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="entry-candidate-", suffix=".json", dir=MODEL_DIR
    )
    os.close(fd)
    candidate = Path(temporary)
    try:
        model.save_model(str(candidate))
        report["model_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        version = archive_model(candidate)
        if deploy and reason is None:
            if MODEL_PATH.exists():
                previous = archive_model(MODEL_PATH)
                report["previous_model"] = str(previous)
            os.replace(candidate, MODEL_PATH)
            report["deployed"] = True
            atomic_json(MODEL_PATH.with_suffix(".meta.json"), report)
        report["candidate_model"] = str(version)
        atomic_json(MODEL_DIR / "entry_training_report.json", report)
        logger.info("Entry validation: %s", json.dumps(report, sort_keys=True))
    finally:
        candidate.unlink(missing_ok=True)
    return True


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()
    return 0 if train(deploy=args.deploy) else 1


if __name__ == "__main__":
    raise SystemExit(main())
