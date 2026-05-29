"""US equity market calendar — weekends, holidays, and Eastern time helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

US_MARKET_HOLIDAYS_2025_2026 = {
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
}


def now_et() -> datetime:
    """Current wall-clock time in US/Eastern (DST-aware)."""
    return datetime.now(timezone.utc).astimezone(ET)


def is_us_market_holiday(d: date) -> bool:
    """True on weekends or NYSE full closures in our holiday table."""
    if d.weekday() >= 5:
        return True
    return d in US_MARKET_HOLIDAYS_2025_2026


def is_us_trading_day(d: date) -> bool:
    """True on a weekday that is not a US market holiday."""
    return not is_us_market_holiday(d)
