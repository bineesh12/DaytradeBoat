"""Shared entry guard — Warrior Trading momentum filter.

Criteria (``check_entry_quality``):

  - Price $2–$20
  - Bar age ≤ 120s
  - Relative volume ≥ 2× (day-level vs historical average)
  - Price above VWAP (buyers in control)
  - Candle body strength (reject dojis)
  - Momentum quality score (green bars, higher closes, acceleration)

Float filtering is done at the watchlist level (startup + RT scanner).

Every verifier should call ``check_entry_quality`` before generating a signal.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional, Sequence

from daytrading.indicators.core import relative_volume, vwap
from daytrading.models import Bar

logger = logging.getLogger(__name__)


def check_entry_quality(
    bars: Sequence[Bar],
    *,
    symbol: str = "",
    min_price: float = 2.0,
    max_price: float = 20.0,
    min_rvol: float = 2.0,
    max_bar_age_seconds: int = 120,
    min_momentum_quality: int = 40,
    avg_daily_volume: Optional[float] = None,
) -> Optional[str]:
    """Return a rejection reason string, or ``None`` if the setup is OK.

    Checks (Warrior Trading momentum criteria):
    1. Price between $2 and $20
    2. Staleness — bar not older than 120s
    3. Relative volume ≥ 3× (day-level: today's vol vs historical avg)
    4. Price above VWAP (buyers in control)
    5. Candle body not tiny vs range (reject doji)
    6. Momentum quality score (green bars, higher closes, acceleration)
    """
    if not bars or len(bars) < 3:
        return "insufficient bars"

    latest = bars[-1]
    price = latest.close
    if price <= 0:
        return "invalid price"

    # 1. Price range $2–$20
    if price < min_price:
        return "price ${:.2f} below ${:.2f}".format(price, min_price)
    if price > max_price:
        return "price ${:.2f} above ${:.2f}".format(price, max_price)

    # 2. Staleness check
    if latest.ts is not None:
        try:
            bar_time = latest.ts if latest.ts.tzinfo else latest.ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - bar_time).total_seconds()
            if age > max_bar_age_seconds:
                return "stale data ({:.0f}s old, max={}s)".format(age, max_bar_age_seconds)
        except Exception:
            pass

    # Split bars into today vs historical
    today_bars = bars
    hist_bars: list = []
    if latest.ts is not None:
        try:
            today_date = latest.ts.date()
            today_bars = [b for b in bars if b.ts is not None and b.ts.date() == today_date]
            hist_bars = [b for b in bars if b.ts is not None and b.ts.date() != today_date]
        except Exception:
            pass

    # 4. Day-level RVOL: is today's volume unusual compared to history?
    rvol_passed = False
    if avg_daily_volume is not None and avg_daily_volume > 0 and len(today_bars) >= 3:
        today_total_vol = sum(b.volume for b in today_bars)
        minutes_elapsed = max(len(today_bars), 1)
        expected_vol_so_far = (avg_daily_volume / 390.0) * minutes_elapsed
        day_rvol = today_total_vol / expected_vol_so_far if expected_vol_so_far > 0 else 0.0
        if day_rvol >= min_rvol:
            rvol_passed = True
        else:
            return "day rvol {:.1f}x < {:.1f}x (today {:.0f} vs expected {:.0f})".format(
                day_rvol, min_rvol, today_total_vol, expected_vol_so_far)
    elif hist_bars and len(today_bars) >= 3:
        hist_avg_vol = sum(b.volume for b in hist_bars) / len(hist_bars)
        today_avg_vol = sum(b.volume for b in today_bars) / len(today_bars)
        day_rvol = today_avg_vol / hist_avg_vol if hist_avg_vol > 0 else 0.0
        if day_rvol >= min_rvol:
            rvol_passed = True
        else:
            return "day rvol {:.1f}x < {:.1f}x (today avg {:.0f} vs hist avg {:.0f})".format(
                day_rvol, min_rvol, today_avg_vol, hist_avg_vol)
    elif len(today_bars) >= 4:
        rvol_period = min(20, len(today_bars) - 1)
        rv = relative_volume(today_bars, period=rvol_period)
        last_rv = rv[-1] if rv else 0.0
        if not math.isnan(last_rv) and last_rv < min_rvol:
            return "rvol {:.1f}x < {:.1f}x".format(last_rv, min_rvol)

    # 5. Price > VWAP (buyers in control) — use today's bars only
    if len(today_bars) >= 3:
        vwap_vals = vwap(today_bars)
        last_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if not math.isnan(last_vwap) and last_vwap > 0 and price < last_vwap:
            return "below VWAP ({:.2f} < {:.2f})".format(price, last_vwap)

    # 6. Body vs range (reject doji / weak conviction)
    if latest.close > 0 and latest.high > latest.low:
        body = abs(latest.close - latest.open)
        full_range = latest.high - latest.low
        if full_range > 0 and body / full_range < 0.15:
            return "weak candle body ({:.0f}% of range)".format(body / full_range * 100)

    # 7. Momentum quality: green bars + higher closes + acceleration (today's bars)
    # Skip if fewer than 3 today-bars — not enough data to judge momentum
    if len(today_bars) >= 3:
        mq_score, mq_detail = _momentum_quality(today_bars)
        if mq_score < min_momentum_quality:
            return "low momentum quality {}/100 ({}) min={}".format(
                mq_score, mq_detail, min_momentum_quality)

    # 8. Dead cat bounce filter: reject if price is >15% below today's high
    if len(today_bars) >= 5:
        today_high = max(b.high for b in today_bars)
        if today_high > 0:
            drop_from_high = (today_high - price) / today_high
            if drop_from_high > 0.15:
                return "dead cat bounce: price {:.2f} is {:.0f}% below HOD {:.2f}".format(
                    price, drop_from_high * 100, today_high)

    # 9. Overextension filter: don't buy at the tip of a spike
    # If the last 3 bars are all green and the total move is >5%,
    # the stock is overextended — wait for a pullback instead of chasing.
    if len(today_bars) >= 4:
        last3 = list(today_bars[-3:])
        prev_bar = today_bars[-4]
        all_green = all(b.close > b.open for b in last3)
        if all_green and prev_bar.close > 0:
            run_pct = (last3[-1].close - prev_bar.close) / prev_bar.close
            if run_pct > 0.05:
                return "overextended: {:.1f}% spike in last 3 bars (wait for pullback)".format(
                    run_pct * 100)

        # Also check: is the latest candle an extension bar (huge body)?
        # Extension bars = selling opportunity, not buying opportunity
        last = last3[-1]
        if last.close > last.open and last.high > last.low:
            body = last.close - last.open
            avg_body = sum(abs(b.close - b.open) for b in today_bars[-6:-1]) / min(5, len(today_bars) - 1) if len(today_bars) > 1 else body
            if avg_body > 0 and body > avg_body * 3:
                return "extension bar: body ${:.2f} is {:.0f}x avg (sell into spike, don't buy)".format(
                    body, body / avg_body)

    return None


def _momentum_quality(bars: Sequence[Bar], lookback: int = 5) -> tuple:
    """Compute a momentum quality score from recent bars.

    Returns (score 0-100, reason_str).
    Score components:
      - Green bar ratio (3/5+ green bars = bullish)   0-30 pts
      - Higher-close streak (closes stepping up)      0-25 pts
      - Tick acceleration (recent move > early move)   0-25 pts
      - Buyer pressure (close near high of bar)        0-20 pts
    """
    n = min(lookback, len(bars))
    if n < 3:
        return (0, "too few bars")

    recent = list(bars[-n:])
    score = 0
    details = []

    # 1. Green bar ratio
    green = sum(1 for b in recent if b.close > b.open)
    ratio = green / n
    pts = int(ratio * 30)
    score += pts
    details.append("{}/{} green".format(green, n))

    # 2. Higher-close streak
    streak = 0
    for i in range(len(recent) - 1, 0, -1):
        if recent[i].close > recent[i - 1].close:
            streak += 1
        else:
            break
    pts = min(25, streak * 8)
    score += pts
    details.append("streak={}".format(streak))

    if streak < 2:
        score = min(score, 35)
        details.append("weak-streak")

    # 3. Tick acceleration
    mid = n // 2
    first_half = recent[:mid]
    second_half = recent[mid:]
    if first_half and second_half and first_half[0].close > 0:
        early_move = (first_half[-1].close - first_half[0].close) / first_half[0].close
        late_move = (second_half[-1].close - second_half[0].close) / second_half[0].close
        if late_move > early_move and late_move > 0:
            score += 25
            details.append("accelerating")
        elif late_move > 0:
            score += 12
            details.append("steady")
        else:
            details.append("decelerating")

    # 4. Buyer pressure: close in upper 60% of bar range
    latest = recent[-1]
    if latest.high > latest.low:
        position = (latest.close - latest.low) / (latest.high - latest.low)
        pts = int(position * 20)
        score += pts
        details.append("close@{:.0f}%".format(position * 100))

    return (score, ", ".join(details))
