"""Offline ignition candidates with deduplicated forward-date validation.

Promotion requires positive returns under a stated barrier/cost model and fresh
holdout dates. It does not establish future profitability or simulate the full
live exit strategy. Automatic nightly promotion is disabled by default.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from datetime import date
from pathlib import Path

import numpy as np

from daytrading.backtest.data_loader import fetch_alpaca_10s_bars_for_day
from daytrading.market_calendar import ET
from daytrading.ml.artifacts import atomic_json, data_root, promote_json
from daytrading.ml.validation import (
    TrainingRow,
    chronological_split,
    parse_timestamp,
    promotion_reason,
    selected_metrics,
)
from daytrading.strategy.warrior_ignition import (
    _MODEL_PATH,
    FEATS,
    detect_ignition,
    get_model,
    prior_day_high,
)
from daytrading.strategy.warrior_ignition_log import _DEFAULT_PATH as _LOG_PATH

HORIZON = 180


def _premarket(bars):
    return [
        b
        for b in bars
        if b.ts
        and 4 * 60
        <= b.ts.astimezone(ET).hour * 60 + b.ts.astimezone(ET).minute
        < 9 * 60 + 30
        and b.close > 0
    ]


def _runner_outcome(bars, i, entry, stop):
    risk = entry - stop
    if risk <= 0 or i >= len(bars):
        return None
    cost_r = (
        entry * float(os.getenv("DAYTRADING_ML_COST_BUFFER_PCT", "0.3")) / 100 / risk
    )
    if cost_r < 0:
        raise ValueError("negative execution cost buffer")
    # Entry is the OPEN of bar i. That bar already carries execution risk.
    for j in range(i, min(i + HORIZON, len(bars))):
        if float(bars[j].low) <= stop:
            return 0.0, -1.0 - cost_r
        if float(bars[j].high) >= entry + 2 * risk:
            return 1.0, 2.0 - cost_r
    if len(bars) - i < HORIZON:
        return None  # incomplete/censored trajectory is not a loss label
    return 0.0, (float(bars[i + HORIZON - 1].close) - entry) / risk - cost_r


def _runner_label(bars, i, entry, stop):
    outcome = _runner_outcome(bars, i, entry, stop)
    return None if outcome is None else outcome[0]


def _rows_from_cache(model):
    rows = []
    pattern = str(data_root() / "backtest_cache" / "*_10s_trades.json")
    for path in sorted(glob.glob(pattern)):
        match = re.match(r"(.+)_(\d{4}-\d{2}-\d{2})_10s", Path(path).name)
        if not match:
            continue
        symbol, day = match.group(1), date.fromisoformat(match.group(2))
        try:
            bars = _premarket(
                fetch_alpaca_10s_bars_for_day(symbol, day).get(symbol, [])
            )
            prior_high = prior_day_high(symbol, day)
        except Exception as exc:  # noqa: BLE001 -- optional cache fetch must not abort the complete dataset
            logging.getLogger(__name__).warning(
                "Skipping ignition cache %s: %s", symbol, type(exc).__name__
            )
            continue
        i = 24
        while i < len(bars) - 2:
            signal = detect_ignition(bars[:i], model, prior_high=prior_high)
            outcome = (
                _runner_outcome(bars, i, float(bars[i].open), signal.stop)
                if signal.detected
                else None
            )
            if outcome is not None:
                rows.append(
                    TrainingRow(
                        symbol,
                        parse_timestamp(bars[i - 1].ts),
                        tuple(float(signal.features[name]) for name in FEATS),
                        *outcome,
                    )
                )
                i += 18
            else:
                i += 1
    return rows


def _rows_from_log(model, day_offset=0):
    del model, day_offset  # identities use actual timestamps, never randomized hashes
    if not os.path.exists(_LOG_PATH):
        return []
    rows, cache = [], {}
    with open(_LOG_PATH) as file:
        for line in file:
            try:
                record = json.loads(line)
                if record.get("kind") != "candidate":
                    continue
                if record.get("feed") != os.getenv("DAYTRADING_ALPACA_FEED", "iex"):
                    continue
                ts = parse_timestamp(record["ts"])
                symbol = str(record["symbol"]).upper()
                day = ts.astimezone(ET).date()
                key = symbol, day
                if key not in cache:
                    cache[key] = _premarket(
                        fetch_alpaca_10s_bars_for_day(symbol, day).get(symbol, [])
                    )
                bars = cache[key]
                idx = next(
                    (j for j, b in enumerate(bars) if parse_timestamp(b.ts) == ts), None
                )
                if idx is None or idx + 1 >= len(bars):
                    continue
                outcome = _runner_outcome(
                    bars, idx + 1, float(bars[idx + 1].open), float(record["stop"])
                )
                if outcome is not None:
                    rows.append(
                        TrainingRow(
                            symbol,
                            ts,
                            tuple(float(record["features"][name]) for name in FEATS),
                            *outcome,
                        )
                    )
            except (ValueError, KeyError, TypeError, OSError):
                continue
    return rows


def _fit(X, y, l2=10.0, lr=0.1, it=2000):
    n, width = X.shape
    w, b = np.zeros(width), 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))
        error = p - y
        w -= lr * (X.T @ error / n + l2 * w / n)
        b -= lr * error.mean()
    return w, b


def _scores(rows, mean, std, weights, bias):
    X = np.asarray([r.features for r in rows])
    return 1 / (1 + np.exp(-np.clip((X - mean) / std @ weights + bias, -30, 30)))


def retrain(deploy=False):
    current = get_model()
    rows = _rows_from_cache(current) + _rows_from_log(current)
    try:
        train, validation, test = chronological_split(rows)
    except ValueError as exc:
        return {
            "status": "not_enough_data",
            "rows": len(rows),
            "reason": str(exc),
            "deployed": False,
        }
    if len({r.label for r in train}) < 2:
        return {
            "status": "not_enough_data",
            "rows": len(rows),
            "reason": "need both training classes",
            "deployed": False,
        }
    X = np.asarray([r.features for r in train])
    y = np.asarray([r.label for r in train])
    mean, std = X.mean(0), X.std(0) + 1e-9
    weights, bias = _fit((X - mean) / std, y)
    validation_scores = _scores(validation, mean, std, weights, bias)
    cutoff = float(np.quantile(validation_scores, 0.75))
    new_metrics = selected_metrics(
        test, _scores(test, mean, std, weights, bias), cutoff
    )
    current_metrics = selected_metrics(
        test,
        _scores(
            test,
            np.asarray(current.feat_mean),
            np.asarray(current.feat_std),
            np.asarray(current.weights),
            current.bias,
        ),
        current.cutoff,
    )
    # Missing incumbent provenance is an explicit promotion blocker.
    metadata = {}
    if Path(_MODEL_PATH).exists():
        metadata = json.loads(Path(_MODEL_PATH).read_text()).get("validation", {})
    reason = promotion_reason(
        train, validation, test, new_metrics, metadata, current_metrics
    )
    validation_metrics = selected_metrics(validation, validation_scores, cutoff)
    if (
        validation_metrics["selected"] < 10
        or validation_metrics["mean_return_after_cost"] <= 0
    ):
        reason = "validation expectancy/count insufficient"
    provenance = {
        "trained_through": max(r.session for r in train).isoformat(),
        "validation_start": min(r.session for r in validation).isoformat(),
        "test_start": min(r.session for r in test).isoformat(),
        "evaluated_through": max(r.session for r in test).isoformat(),
        "final_holdout": new_metrics,
        "return_model": "2R target / -1R stop, stop-first on ambiguous bars, execution-cost buffer",
    }
    artifact = {
        "model_type": "logistic_regression",
        "target": "premarket_ignition_runner(MFE>=2R)",
        "features": FEATS,
        "weights": weights.tolist(),
        "bias": float(bias),
        "feat_mean": mean.tolist(),
        "feat_std": std.tolist(),
        "candidate_gates": current.gates,
        "high_conviction_cutoff": cutoff,
        "validation": provenance,
    }
    candidate_path = data_root() / "models" / "warrior_ignition_candidate.json"
    atomic_json(candidate_path, artifact)
    result = {
        "status": "ok",
        "rows": len(train) + len(validation) + len(test),
        "runner_rate": float(np.mean([r.label for r in rows])),
        "current_oos": current_metrics["positive_rate"],
        "retrained_oos": new_metrics["positive_rate"],
        "current_holdout": current_metrics,
        "retrained_holdout": new_metrics,
        "improved": reason is None,
        "promotion_reason": reason,
        "deployed": False,
        **provenance,
    }
    if deploy and reason is None:
        promote_json(Path(_MODEL_PATH), artifact)
        get_model()  # explicit reload of the atomic persisted version
        result["deployed"] = True
    atomic_json(data_root() / "models" / "ignition_training_report.json", result)
    return result
