"""Session stats from intraday bars for HOD Momentum scanners."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dt_time
from typing import List, Optional, Sequence

from daytrading.market_calendar import ET
from daytrading.models import Bar
from daytrading.scanner.hod_momentum.prior_day import PriorDayStats


@dataclass
class SessionContext:
    session_high: float
    session_low: float
    session_open: float
    day_volume: float
    session_date: Optional[str]
    prior_day_close: Optional[float] = None
    prior_day_high: Optional[float] = None
    # True when today's bar list likely misses the real session open (truncated fetch).
    incomplete_history: bool = False


def _first_bar_at_or_after(
    bars: Sequence[Bar],
    *,
    hour: int,
    minute: int = 0,
) -> Optional[Bar]:
    """First bar at or after hour:minute ET on the bar's calendar day."""
    target = dt_time(hour, minute)
    for b in bars:
        if b.ts is None:
            continue
        try:
            t = b.ts.astimezone(ET)
            if t.time() >= target:
                return b
        except Exception:
            continue
    return None


def resolve_session_open(
    today: Sequence[Bar],
    *,
    rth_only: bool = False,
) -> tuple[float, bool]:
    """Return (session_open, incomplete_history).

    Uses 9:30 AM ET open for regular hours, else 4:00 AM ET for extended session.
    Falls back to first bar open; marks incomplete when the first bar is much later
    than the expected session start (typical when only the last N bars were fetched).
    """
    if not today:
        return 0.0, True

    anchor_hour, anchor_minute = (9, 30) if rth_only else (4, 0)
    anchor = _first_bar_at_or_after(today, hour=anchor_hour, minute=anchor_minute)
    session_open = today[0].open
    if anchor is not None and anchor.open > 0:
        session_open = anchor.open

    incomplete = False
    first = today[0]
    if first.ts is not None:
        try:
            first_et = first.ts.astimezone(ET)
            expected = first_et.replace(
                hour=anchor_hour, minute=anchor_minute, second=0, microsecond=0,
            )
            # Missing more than ~30 minutes of session start → open is unreliable.
            if first_et > expected and (first_et - expected).total_seconds() > 30 * 60:
                incomplete = True
        except Exception:
            incomplete = True

    if session_open <= 0:
        return 0.0, True
    return session_open, incomplete


def today_bars(
    bars: Sequence[Bar],
    *,
    rth_only: bool = False,
) -> List[Bar]:
    if not bars:
        return []
    latest = bars[-1]
    if latest.ts is None:
        return list(bars)
    try:
        today = latest.ts.date()
        out = [b for b in bars if b.ts is not None and b.ts.date() == today]
        if not rth_only:
            return out
        rth: List[Bar] = []
        for b in out:
            try:
                t = b.ts.astimezone(ET)
                if (t.hour > 9 or (t.hour == 9 and t.minute >= 30)) and t.hour < 16:
                    rth.append(b)
            except Exception:
                continue
        return rth if rth else out
    except Exception:
        return list(bars)


def session_context_from_bars(
    bars: Sequence[Bar],
    *,
    prior_day: Optional[PriorDayStats] = None,
    rth_only: bool = False,
) -> Optional[SessionContext]:
    """Build session high/low/open and total day volume from today's 1m bars."""
    today = today_bars(bars, rth_only=rth_only)
    if not today:
        return None
    session_open, incomplete = resolve_session_open(today, rth_only=rth_only)
    if session_open <= 0:
        return None
    session_date = None
    if today[-1].ts is not None:
        try:
            session_date = today[-1].ts.date().isoformat()
        except Exception:
            pass
    prior_close = prior_day.prior_close if prior_day else None
    prior_high = prior_day.prior_high if prior_day else None
    return SessionContext(
        session_high=max(b.high for b in today),
        session_low=min(b.low for b in today),
        session_open=session_open,
        day_volume=float(sum(b.volume for b in today)),
        session_date=session_date,
        prior_day_close=prior_close,
        prior_day_high=prior_high,
        incomplete_history=incomplete,
    )


def session_change_pct(
    price: float,
    ctx: SessionContext,
) -> float:
    """% change for display / filters — prefers prior close when history is truncated."""
    if ctx.incomplete_history and ctx.prior_day_close and ctx.prior_day_close > 0:
        return (price - ctx.prior_day_close) / ctx.prior_day_close * 100
    if ctx.session_open > 0:
        return (price - ctx.session_open) / ctx.session_open * 100
    return 0.0
