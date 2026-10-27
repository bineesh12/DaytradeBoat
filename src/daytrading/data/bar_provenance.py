"""Explicit interval provenance; elapsed time never promotes a snapshot.

Old protocol-1 bars without provenance remain UNKNOWN. Timestamp alignment is
an additional integrity check, never sufficient evidence of canonical origin.
"""
from datetime import datetime, timedelta, timezone
from daytrading.models import BarSource, Timeframe


def canonical_minute(bar, now=None):
    stamp = getattr(bar, 'ts', None)
    return (getattr(bar, 'provenance', BarSource.UNKNOWN) == BarSource.CANONICAL_INTERVAL
        and getattr(bar, 'timeframe', None) == Timeframe.MIN_1
        and isinstance(stamp, datetime) and stamp.tzinfo is not None
        and stamp.utcoffset() is not None and stamp.second == 0 and stamp.microsecond == 0
        and (now is None or (now.tzinfo is not None and now.utcoffset() is not None
             and stamp + timedelta(minutes=1) <= now)))


def completed_minutes(bars, *, now=None):
    now = now if now is not None else datetime.now(timezone.utc)
    by_time = {}
    for bar in bars:
        if canonical_minute(bar, now):
            by_time[(bar.symbol, bar.ts)] = bar
    return sorted(by_time.values(), key=lambda bar: bar.ts)
