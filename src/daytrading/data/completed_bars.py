"""Validate completed event-bar context without importing strategy/execution code."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from math import isfinite
from zoneinfo import ZoneInfo

from daytrading.models import CompletedBar, Timeframe

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Rejection:
    code: str
    reason: str


def _aware(at: datetime) -> bool:
    return at.tzinfo is not None and at.utcoffset() is not None


def validate_micro_window(
    symbol: str,
    bars: Sequence[CompletedBar],
    *,
    now: datetime,
    max_completion_seconds: float = 15.0,
    max_confirmation_age_seconds: float = 15.0,
) -> Rejection | None:
    """Check completed-bar identity, timing and OHLCV without changing strategy thresholds."""
    if len(bars) < 6:
        return Rejection("history", "six completed 10-second bars required")
    window = bars[-6:]
    if not _aware(now) or any(
        not _aware(x.bar.ts) or not _aware(x.closed_at) for x in window
    ):
        return Rejection("timestamp", "timezone-aware timestamps required")
    session = now.astimezone(NY).date()
    for x in window:
        b = x.bar
        if b.symbol != symbol or b.timeframe != Timeframe.SEC_10:
            return Rejection("identity", "mixed symbol or non-10-second bars")
        if b.ts.astimezone(NY).date() != session:
            return Rejection("session", "micro window belongs to another session")
        if x.closed_at > now:
            return Rejection(
                "future", "micro candle is not completed as of observation"
            )
        duration = (x.closed_at - b.ts).total_seconds()
        if not 10 <= duration <= max_completion_seconds:
            return Rejection("sparse_bar", f"micro completion duration {duration:.3f}s")
        if not all(isfinite(v) for v in (b.open, b.high, b.low, b.close, b.volume)):
            return Rejection("ohlcv", "non-finite micro OHLCV")
        if (
            not (0 < b.low <= min(b.open, b.close) <= max(b.open, b.close) <= b.high)
            or b.volume <= 0
        ):
            return Rejection("ohlcv", "invalid micro OHLCV")
    for a, b in pairwise(window):
        if a.closed_at != b.bar.ts:
            return Rejection("gap", "micro bars are not contiguous at completion")
    age = (now - window[-1].closed_at).total_seconds()
    if age > max_confirmation_age_seconds:
        return Rejection("stale", f"micro confirmation age {age:.3f}s")
    return None
