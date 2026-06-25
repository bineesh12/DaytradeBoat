"""Importable retrain for the Warrior ignition model — the offline half of the
adaptive loop. Runs on demand (CLI) or nightly after market close (runner hook).

Dataset = the cached bootstrap days PLUS the live-logged PAPER candidates (their
outcomes reconstructed from that day's bars). Retrains the logistic model,
validates out-of-sample, and DEPLOYS ONLY IF it beats the current model — a worse
or noise-chasing model can never reach trading.
"""
from __future__ import annotations

import glob
import json
import os
import re
from datetime import date, timezone
from typing import Dict, List, Tuple

import numpy as np

from daytrading.backtest.data_loader import fetch_alpaca_10s_bars_for_day
from daytrading.strategy.warrior_ignition import (
    detect_ignition, get_model, prior_day_high, _MODEL_PATH, FEATS,
)
from daytrading.strategy.warrior_ignition_log import _DEFAULT_PATH as _LOG_PATH

SESS = (8 * 60, 13 * 60 + 30)
HORIZON = 180


def _um(ts):
    t = ts.astimezone(timezone.utc)
    return t.hour * 60 + t.minute


def _runner_label(bars, i, entry, stop):
    risk = entry - stop
    if risk <= 0:
        return None
    for j in range(i + 1, min(i + 1 + HORIZON, len(bars))):
        if float(bars[j].high) >= entry + 2 * risk:
            return 1.0
        if float(bars[j].low) <= stop:
            return 0.0
    return 0.0


def _rows_from_cache(model) -> List[Tuple[int, list, float]]:
    pairs = []
    for f in sorted(glob.glob("data/backtest_cache/*_10s_trades.json")):
        m = re.match(r"(.+)_(\d{4})-(\d{2})-(\d{2})_10s", f.split("/")[-1])
        if m:
            pairs.append((m.group(1), date(int(m.group(2)), int(m.group(3)), int(m.group(4)))))
    rows = []
    for k, (sym, day) in enumerate(pairs):
        try:
            b = fetch_alpaca_10s_bars_for_day(sym, day).get(sym, [])
        except Exception:
            continue
        bars = [x for x in b if SESS[0] <= _um(x.ts) < SESS[1] and float(x.close) > 0]
        if len(bars) < 40:
            continue
        ph = prior_day_high(sym, day)
        i = 24
        while i < len(bars) - 2:
            sig = detect_ignition(bars[:i], model, prior_high=ph)
            if sig.detected:
                lab = _runner_label(bars, i, float(bars[i].open), sig.stop)
                if lab is not None:
                    rows.append((k, [sig.features[n] for n in FEATS], lab))
                i += 18
            else:
                i += 1
    return rows


def _rows_from_log(model, day_offset: int) -> List[Tuple[int, list, float]]:
    """Real paper candidates: use the logged features; reconstruct the outcome
    from that day's bars (fetched post-hoc, after close)."""
    if not os.path.exists(_LOG_PATH):
        return []
    rows = []
    for line in open(_LOG_PATH):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") != "candidate":
            continue
        try:
            ts = r["ts"]; sym = r["symbol"]
            d = date.fromisoformat(ts[:10])
            bars = fetch_alpaca_10s_bars_for_day(sym, d).get(sym, [])
            bars = [x for x in bars if SESS[0] <= _um(x.ts) < SESS[1] and float(x.close) > 0]
            idx = next((j for j, x in enumerate(bars) if x.ts.isoformat() == ts), None)
            if idx is None or idx + 2 >= len(bars):
                continue
            entry = float(bars[idx + 1].open); stop = float(r["stop"])
            lab = _runner_label(bars, idx + 1, entry, stop)
            if lab is None:
                continue
            feats = [float(r["features"][n]) for n in FEATS]
            rows.append((day_offset + hash(d.isoformat()) % 1000, feats, lab))  # distinct day index
        except Exception:
            continue
    return rows


def _fit(Xt, yt, l2=10.0, lr=0.1, it=6000):
    n, dd = Xt.shape; w = np.zeros(dd); b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(Xt @ w + b, -30, 30)))
        g = p - yt
        w -= lr * (Xt.T @ g / n + l2 * w / n); b -= lr * g.mean()
    return w, b


def retrain(deploy: bool = False) -> Dict[str, object]:
    """Retrain, validate OOS, and (if --deploy) overwrite the model only if it
    beats the current one. Returns a result summary."""
    cur = get_model()
    rows = _rows_from_cache(cur) + _rows_from_log(cur, day_offset=10_000)
    if len(rows) < 50:
        return {"status": "not_enough_data", "rows": len(rows)}
    X = np.array([r[1] for r in rows]); Y = np.array([r[2] for r in rows]); D = np.array([r[0] for r in rows])

    tr = D % 2 == 0
    if tr.sum() < 20 or (~tr).sum() < 20:
        return {"status": "not_enough_data", "rows": len(rows)}
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    w0, b0 = _fit(((X - mu) / sd)[tr], Y[tr])
    pt = 1 / (1 + np.exp(-np.clip(((X - mu) / sd)[~tr] @ w0 + b0, -30, 30)))
    new_topq = float(Y[~tr][pt >= np.quantile(pt, 0.75)].mean())
    if len(cur.weights) == X.shape[1]:
        # same feature set -> compare against the current model head-to-head
        pc = 1 / (1 + np.exp(-np.clip(
            ((X - np.array(cur.feat_mean)) / np.array(cur.feat_std))[~tr] @ np.array(cur.weights) + cur.bias,
            -30, 30)))
        cur_topq = float(Y[~tr][pc >= np.quantile(pc, 0.75)].mean())
    else:
        # feature set changed (e.g. added prior-day/1-min) -> the old model can't
        # be scored on the new vectors; gate against the base runner rate instead.
        cur_topq = float(Y[~tr].mean())

    result = {"status": "ok", "rows": int(len(rows)), "runner_rate": round(float(Y.mean()), 3),
              "current_oos": round(cur_topq, 3), "retrained_oos": round(new_topq, 3),
              "improved": bool(new_topq >= cur_topq), "deployed": False}
    if deploy and new_topq >= cur_topq:
        MU, SD = X.mean(0), X.std(0) + 1e-9
        W, B = _fit((X - MU) / SD, Y)
        scores = 1 / (1 + np.exp(-np.clip(((X - MU) / SD) @ W + B, -30, 30)))
        json.dump({
            "model_type": "logistic_regression", "target": "premarket_ignition_runner(MFE>=2R)",
            "features": FEATS, "weights": [round(float(x), 5) for x in W], "bias": round(float(B), 5),
            "feat_mean": [round(float(x), 6) for x in MU], "feat_std": [round(float(x), 6) for x in SD],
            "candidate_gates": cur.gates, "high_conviction_cutoff": round(float(np.quantile(scores, 0.75)), 4),
            "trained_on": {"candidates": int(len(Y)), "runner_rate": round(float(Y.mean()), 3)},
            "oos_validation": {"top_quartile_runner_rate": round(new_topq, 3)},
        }, open(_MODEL_PATH, "w"), indent=2)
        result["deployed"] = True
    return result
