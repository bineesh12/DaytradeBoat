"""Deterministic forward session splits; no symbol/hash/random row splits."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from daytrading.market_calendar import ET


def parse_timestamp(value) -> datetime:
    ts = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if ts.tzinfo is None:
        raise ValueError("training timestamps must include a timezone")
    return ts.astimezone(UTC)


@dataclass(frozen=True)
class TrainingRow:
    symbol: str
    ts: datetime
    features: tuple
    label: float
    return_after_cost: float

    @property
    def session(self):
        return self.ts.astimezone(ET).date()


def chronological_split(
    rows: Sequence[TrainingRow], *, min_sessions: int = 10, min_partition_rows: int = 20
):
    unique = {}
    for row in rows:
        if not all(
            math.isfinite(float(v))
            for v in (*row.features, row.label, row.return_after_cost)
        ):
            continue
        unique[(row.symbol.upper(), row.ts)] = row
    ordered = sorted(unique.values(), key=lambda r: (r.ts, r.symbol))
    days = sorted({r.session for r in ordered})
    if len(days) < min_sessions:
        raise ValueError(
            f"need at least {min_sessions} distinct sessions; found {len(days)}"
        )
    train_end, validation_end = int(len(days) * 0.6), int(len(days) * 0.8)
    train_days, val_days, test_days = (
        set(days[:train_end]),
        set(days[train_end:validation_end]),
        set(days[validation_end:]),
    )
    groups = tuple(
        [r for r in ordered if r.session in group]
        for group in (train_days, val_days, test_days)
    )
    if any(len(group) < min_partition_rows for group in groups):
        raise ValueError(
            f"need at least {min_partition_rows} candidates in each forward partition"
        )
    return groups


def selected_metrics(rows, probabilities, threshold: float):
    selected = np.asarray(probabilities) >= threshold
    returns = np.asarray([r.return_after_cost for r in rows], dtype=float)
    count = int(selected.sum())
    return {
        "selected": count,
        "mean_return_after_cost": float(returns[selected].mean()) if count else 0.0,
        "total_return_after_cost": float(returns[selected].sum()) if count else 0.0,
        "positive_rate": float((returns[selected] > 0).mean()) if count else 0.0,
    }


def promotion_reason(
    train,
    validation,
    test,
    candidate_metrics,
    incumbent_metadata=None,
    incumbent_metrics=None,
):
    if len({r.label for r in train}) < 2:
        return "training partition needs both outcome classes"
    if candidate_metrics["selected"] < 10:
        return "fewer than 10 final holdout selections"
    if candidate_metrics["mean_return_after_cost"] <= 0:
        return "final holdout expectancy after costs is not positive"
    if incumbent_metrics is not None:
        if incumbent_metrics["selected"] < 10:
            return "insufficient incumbent holdout selections"
        if (
            candidate_metrics["total_return_after_cost"]
            <= incumbent_metrics["total_return_after_cost"]
        ):
            return "candidate does not beat incumbent holdout total return"
        if (
            candidate_metrics["mean_return_after_cost"]
            <= incumbent_metrics["mean_return_after_cost"]
        ):
            return "candidate does not beat incumbent holdout expectancy"
    if incumbent_metadata is not None:
        through = incumbent_metadata.get("evaluated_through")
        if not through:
            return "incumbent has no validation provenance; use forward shadow comparison first"
        if min(r.session for r in validation + test).isoformat() <= through:
            return "holdout sessions overlap previous model evaluation"
    return None
