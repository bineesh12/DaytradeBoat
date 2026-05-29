"""Shared HOD breakout classification (today vs prior day)."""

from __future__ import annotations

from typing import Optional, Tuple


def breaks_today_hod(
    latest_high: float,
    prior_today_hod: float,
    latest_close: float,
    latest_open: float,
) -> bool:
    return (
        prior_today_hod > 0
        and latest_high > prior_today_hod
        and latest_close > latest_open
    )


def classify_hod_breakout_alerts(
    latest_high: float,
    prior_today_hod: float,
    latest_close: float,
    latest_open: float,
    prior_day_high: Optional[float],
    *,
    require_break_prior_day: bool = True,
) -> Tuple[bool, bool]:
    """Return (include_new_hod_breakout, include_today_hod_breakout)."""
    if not breaks_today_hod(latest_high, prior_today_hod, latest_close, latest_open):
        return False, False

    if prior_day_high is None or prior_day_high <= 0:
        return True, False

    if latest_high > prior_day_high:
        return True, False

    if require_break_prior_day:
        return False, True

    return True, False


def classify_hod_reclaim(
    reclaim_detected: bool,
    latest_high: float,
    prior_today_hod: float,
    reclaim_hod: float,
    prior_day_high: Optional[float],
    *,
    require_break_prior_day: bool = True,
) -> bool:
    if not reclaim_detected or prior_today_hod <= 0:
        return False
    at_session_hod = reclaim_hod >= prior_today_hod * 0.998
    breaks_today = latest_high > prior_today_hod
    if not (at_session_hod and breaks_today):
        return False
    if prior_day_high is None or prior_day_high <= 0:
        return True
    if require_break_prior_day:
        return latest_high > prior_day_high
    return True
