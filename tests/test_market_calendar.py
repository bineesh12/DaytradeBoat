"""US market calendar only — no local/Swedish holidays."""

from __future__ import annotations

from datetime import date

from daytrading.market_calendar import is_us_market_holiday, is_us_trading_day


def test_memorial_day_2026_not_trading_day() -> None:
    assert is_us_market_holiday(date(2026, 5, 25)) is True
    assert is_us_trading_day(date(2026, 5, 25)) is False


def test_regular_weekday_is_trading_day() -> None:
    assert is_us_trading_day(date(2026, 5, 15)) is True
