"""Warrior premarket-ignition scorer.

Loads the pre-trained logistic model (``warrior_ignition_model.json``) and scores
a base-breakout ignition candidate from a window of bars. Pure-numpy-free, O(1)
per call beyond the fixed base window, so it runs per-tick in the live loop.

The model is trained OFFLINE (see scratchpad/train_export.py); at runtime this
module only computes 11 features and a single dot product -> conviction in [0,1].
There is no live learning and no warm-up.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import timezone
from typing import Dict, List, Optional, Sequence

from daytrading.models import Bar

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "warrior_ignition_model.json")
_PREMKT_OPEN_UTC_MIN = 8 * 60  # 04:00 ET


def _f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class IgnitionModel:
    features: List[str]
    weights: List[float]
    bias: float
    feat_mean: List[float]
    feat_std: List[float]
    gates: Dict[str, float]
    cutoff: float

    @classmethod
    def load(cls, path: str = _MODEL_PATH) -> "IgnitionModel":
        with open(path) as fh:
            m = json.load(fh)
        return cls(
            features=m["features"], weights=m["weights"], bias=m["bias"],
            feat_mean=m["feat_mean"], feat_std=m["feat_std"],
            gates=m["candidate_gates"], cutoff=float(m.get("high_conviction_cutoff", 0.31)),
        )


@dataclass
class IgnitionSignal:
    detected: bool                 # a valid base-breakout ignition (gates passed)
    conviction: float = 0.0        # P(runner) in [0,1] -> use for SIZING, not on/off
    entry_ref: float = 0.0         # ignition close (caller enters at next bar open)
    stop: float = 0.0              # base low
    base_high: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    reject: str = ""

    def size_factor(self, cutoff: float, *, floor: float = 0.35, cap: float = 1.0) -> float:
        """Conviction-scaled position size in [floor, cap]. A candidate at the
        model's high-conviction cutoff sizes ~1.0; weaker ones scale down (but
        still trade, so PLSM-type middling-conviction runners aren't excluded)."""
        if cutoff <= 0:
            return cap
        return max(floor, min(cap, self.conviction / cutoff))


def ignition_suppression_reason(
    signal: IgnitionSignal,
    *,
    failed_entries: int = 0,
    peak_price: float = 0.0,
    peak_day_move: float = 0.0,
) -> str:
    """Return a human-readable reason to skip post-peak chop re-ignitions.

    The first ignition on a symbol is untouched. After a losing ignition, require
    the next one to be a genuine fresh launch again, not a local bounce while the
    earlier move is fading. Two failed ignition attempts stop the symbol for the
    day.
    """
    failures = max(0, int(failed_entries or 0))
    if failures <= 0 or not signal.detected:
        return ""
    if failures >= 2:
        return "suppressed: 2 failed ignitions today"

    price = float(signal.entry_ref or 0.0)
    peak = float(peak_price or 0.0)
    day_move = float(signal.features.get("day_move", 0.0) or 0.0)
    peak_move = float(peak_day_move or 0.0)
    near_hod = float(signal.features.get("near_hod", 0.0) or 0.0)

    below_prior_peak = peak > 0 and price <= peak * 0.97
    collapsed_vs_peak = peak_move > 0 and day_move <= peak_move * 0.60
    weak_day_move = day_move <= 0.12
    local_bounce = near_hod >= 0.90
    if below_prior_peak and local_bounce and (collapsed_vs_peak or weak_day_move):
        return (
            "suppressed: post-peak chop after failed ignition "
            "(price ${:.2f} below peak ${:.2f}, day_move {:.1%})"
        ).format(price, peak, day_move)
    return ""


_MODEL_CACHE: Optional[IgnitionModel] = None


def get_model() -> IgnitionModel:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = IgnitionModel.load()
    return _MODEL_CACHE


def _utc_minutes(ts) -> float:
    t = ts.astimezone(timezone.utc)
    return t.hour * 60 + t.minute


FEATS = ["rvol_surge", "vol_z", "vol_buildup", "base_tight", "range_exp", "brk_strength",
         "accel", "day_move", "near_hod", "mins", "price",
         "min1_slope", "min1_hh", "pd_dist", "above_pdh"]

_PRIOR_HIGH_CACHE: Dict[tuple, float] = {}


def prior_day_high(symbol: str, session_date) -> float:
    """Prior trading day's session high (for the resistance feature). Fetched and
    cached identically in training, backtest, and live so the feature matches."""
    key = (symbol.upper(), session_date.isoformat())
    if key in _PRIOR_HIGH_CACHE:
        return _PRIOR_HIGH_CACHE[key]
    from datetime import timedelta
    from daytrading.backtest.data_loader import fetch_alpaca_bars_for_day
    ph = 0.0
    for back in range(1, 5):
        pd = session_date - timedelta(days=back)
        if pd.weekday() >= 5:
            continue
        try:
            b = fetch_alpaca_bars_for_day(symbol, pd).get(symbol, [])
            highs = [float(x.high) for x in b if float(x.high or 0.0) > 0]
            ph = max(highs) if highs else 0.0
        except Exception:
            ph = 0.0
        break
    _PRIOR_HIGH_CACHE[key] = ph
    return ph


def detect_ignition(
    history: Sequence[Bar],
    model: Optional[IgnitionModel] = None,
    *,
    prior_high: float = 0.0,
) -> IgnitionSignal:
    """Score the latest bar of ``history`` as a base-breakout ignition.

    ``history`` is oldest-first; its LAST bar is the just-closed ignition candle,
    and the caller enters on the NEXT bar's open. Feature extraction mirrors the
    training pipeline exactly.
    """
    model = model or get_model()
    base_n = int(model.gates["base_bars"])
    bars = [b for b in history if _f(b.close) > 0]
    if len(bars) < base_n + 4:
        return IgnitionSignal(False, reject="not enough bars")

    ig = bars[-1]
    base = bars[-base_n - 1:-1]
    bh = max(_f(b.high) for b in base)
    bl = min(_f(b.low) for b in base)
    bvols = [_f(b.volume) for b in base]
    avgv = sum(bvols) / len(bvols)
    branges = [_f(b.high) - _f(b.low) for b in base] or [1e-9]
    avgr = (sum(branges) / len(branges)) or 1e-9
    c, v = _f(ig.close), _f(ig.volume)
    igr = _f(ig.high) - _f(ig.low)
    if bl <= 0 or avgv <= 0:
        return IgnitionSignal(False, reject="bad base")
    g = model.gates
    if not (g["min_price"] <= c <= g["max_price"]):
        return IgnitionSignal(False, reject="price out of band")
    if not (c > bh and v >= g["min_vol_surge"] * avgv and (bh - bl) / bl <= g["max_base_range"]):
        return IgnitionSignal(False, reject="no ignition (gate)")

    sess_low = min(_f(b.low) for b in bars)
    sess_high = max(_f(b.high) for b in bars)
    half = len(bvols) // 2
    vstd = (sum((x - avgv) ** 2 for x in bvols) / len(bvols)) ** 0.5 or 1e-9
    recent = sum(bvols[half:]) / max(1, len(bvols) - half)
    earlier = (sum(bvols[:half]) / max(1, half)) or 1e-9
    # 1-minute structure (aggregate recent 10s closes into 1-min) — multi-timeframe
    mclose = [_f(b.close) for b in bars][-90:][::6]
    min1_slope = (mclose[-1] / mclose[0] - 1.0) if len(mclose) >= 2 and mclose[0] > 0 else 0.0
    min1_hh = 0.0
    for k in range(len(mclose) - 1, 0, -1):
        if mclose[k] > mclose[k - 1]:
            min1_hh += 1.0
        else:
            break
    # prior-day high = the key resistance: clearing yesterday's high distinguishes
    # a real breakout from a fakeout (validated as the strongest added feature).
    pd_dist = (c - prior_high) / c if prior_high > 0 else 0.0   # +ve = above prior-day high
    above_pdh = 1.0 if (prior_high > 0 and c > prior_high) else 0.0
    feats = [
        v / avgv,                                              # rvol_surge
        (v - avgv) / vstd,                                     # vol_z
        recent / earlier,                                      # vol_buildup
        (bh - bl) / bl,                                        # base_tight
        igr / avgr,                                            # range_exp
        (c - bh) / bh,                                         # brk_strength
        (c - _f(bars[-4].close)) / _f(bars[-4].close) if _f(bars[-4].close) else 0.0,  # accel
        c / sess_low - 1.0,                                    # day_move
        _f(ig.high) / sess_high if sess_high > 0 else 1.0,     # near_hod
        (_utc_minutes(ig.ts) - _PREMKT_OPEN_UTC_MIN) / 60.0,   # mins
        c,                                                     # price
        min1_slope,                                            # min1_slope
        min1_hh,                                               # min1_hh
        pd_dist,                                               # pd_dist
        above_pdh,                                             # above_pdh
    ]
    z = model.bias
    for x, mean, std, w in zip(feats, model.feat_mean, model.feat_std, model.weights):
        z += w * ((x - mean) / (std or 1e-9))
    z = max(-30.0, min(30.0, z))
    conviction = 1.0 / (1.0 + math.exp(-z))
    return IgnitionSignal(
        detected=True,                 # gates passed = a real ignition; size by conviction
        conviction=conviction, entry_ref=c, stop=bl, base_high=bh,
        features=dict(zip(FEATS, feats)),
    )


if __name__ == "__main__":
    import sys
    from datetime import date
    sys.path.insert(0, "src")
    from daytrading.backtest.data_loader import fetch_alpaca_10s_bars_for_day

    def um(ts):
        t = ts.astimezone(timezone.utc); return t.hour * 60 + t.minute

    bars = [b for b in fetch_alpaca_10s_bars_for_day("PLSM", date(2026, 6, 24))["PLSM"]
            if 8 * 60 <= um(b.ts) < 13 * 60 + 30 and _f(b.close) > 0]
    model = get_model()
    print(f"model cutoff (size~1.0 at conviction {model.cutoff})")
    fires = 0
    for i in range(20, len(bars) - 1):
        sig = detect_ignition(bars[:i], model)   # last bar = ignition candle
        if sig.detected:
            fires += 1
            t = bars[i - 1].ts.astimezone(timezone.utc)
            if 7 * 60 + 13 <= (t.hour * 60 + t.minute) - 4 * 60 <= 7 * 60 + 22:
                print(f"  {(t.hour-4)%24:02d}:{t.minute:02d}:{t.second:02d} ET  "
                      f"conviction={sig.conviction:.2f}  size={sig.size_factor(model.cutoff):.2f}  "
                      f"entry_ref={sig.entry_ref:.2f}  stop={sig.stop:.2f}")
    print(f"PLSM ignition entries (size-scaled): {fires}")
