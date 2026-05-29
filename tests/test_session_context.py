"""Tests for session open / day-change calculation."""

from datetime import datetime

from daytrading.market_calendar import ET
from daytrading.models import Bar
from daytrading.scanner.hod_momentum.prior_day import PriorDayStats
from daytrading.scanner.hod_momentum.session_context import (
    resolve_session_open,
    session_change_pct,
    session_context_from_bars,
)


def _bar_et(h: int, m: int, open_: float, close: float, sym: str = "TOYO") -> Bar:
    return Bar(
        symbol=sym,
        ts=datetime(2026, 5, 18, h, m, 0, tzinfo=ET),
        open=open_,
        high=max(open_, close) + 0.1,
        low=min(open_, close) - 0.1,
        close=close,
        volume=50_000,
    )


def test_resolve_session_open_uses_4am_for_extended() -> None:
    bars = [
        _bar_et(4, 0, 4.0, 4.1),
        _bar_et(9, 30, 5.0, 5.2),
        _bar_et(14, 0, 5.5, 5.8),
    ]
    open_px, incomplete = resolve_session_open(bars, rth_only=False)
    assert open_px == 4.0
    assert incomplete is False


def test_truncated_history_uses_prior_close_for_change_pct() -> None:
    bars = [
        _bar_et(14, 0, 10.0, 10.2, sym="GPRK"),
        _bar_et(15, 0, 10.2, 10.5, sym="GPRK"),
    ]
    prior = PriorDayStats(prior_close=9.0, prior_high=10.5)
    ctx = session_context_from_bars(bars, prior_day=prior, rth_only=False)
    assert ctx is not None
    assert ctx.incomplete_history is True
    # Truncated open would show 5% (10.5 vs 10.0); vs prior close 9.0 → ~16.7%
    chg = session_change_pct(10.5, ctx)
    assert abs(chg - 16.67) < 0.1
