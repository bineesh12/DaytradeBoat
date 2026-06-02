"""Shared entry guard — scoring-based momentum filter.

Uses a hybrid approach:
  1. Hard rejects: 4 absolute deal-breakers (immediate rejection)
  2. Scoring: 7 weighted conditions scored 0-100
  3. Penalty: S8 volume exhaustion subtracts up to -30 for declining-volume green bars
  4. Threshold: score >= 60 to trade
  5. ML model: optional XGBoost probability check (if model file exists)

This allows strong setups (VWAP break + volume) to pass even if
some minor conditions are imperfect — matching how real traders decide.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence, List

from daytrading.indicators.core import relative_volume, vwap
from daytrading.models import Bar, Quote, Tick

logger = logging.getLogger(__name__)

ENTRY_SCORE_THRESHOLD = 65

# ML Monitor — singleton instance for tracking model performance
_ml_monitor = None
try:
    from daytrading.ml.monitor import MLMonitor
    _ml_monitor = MLMonitor()
except Exception:
    pass

# XGBoost model — loaded once at import time, None if unavailable
_xgb_model = None
_XGB_THRESHOLD = 0.30
_ML_SOFT_PASS_SCORE = 80
try:
    import xgboost as xgb
    _model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "models", "entry_model.json",
    )
    _model_path = os.path.normpath(_model_path)
    if os.path.exists(_model_path):
        _xgb_model = xgb.Booster()
        _xgb_model.load_model(_model_path)
        logger.info("XGBoost entry model loaded from %s", _model_path)
except Exception:
    pass


def get_ml_monitor():
    """Get the global ML monitor instance."""
    return _ml_monitor


def record_rule_rejection(
    symbol: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    """Record a non-ML rule rejection for dashboard visibility."""
    if _ml_monitor:
        try:
            _ml_monitor.record_rule_rejection(symbol=symbol, reason=reason)
        except TypeError:
            _ml_monitor.record_rule_rejection()


def _recent_volume_stats(today_bars: Sequence[Bar]) -> tuple[float, float, float]:
    """Return (day volume, recent average volume, recent/earlier RVOL)."""
    if not today_bars:
        return 0.0, 0.0, 0.0

    day_volume = float(sum(b.volume for b in today_bars))
    recent_n = min(5, len(today_bars))
    recent = list(today_bars[-recent_n:])
    recent_avg = float(sum(b.volume for b in recent) / recent_n) if recent_n else 0.0
    earlier = list(today_bars[:-recent_n])
    if not earlier:
        return day_volume, recent_avg, 0.0
    earlier_avg = float(sum(b.volume for b in earlier) / len(earlier))
    bar_rvol = recent_avg / earlier_avg if earlier_avg > 0 else 0.0
    return day_volume, recent_avg, bar_rvol


def _liquidity_score(
    price: float,
    today_bars: Sequence[Bar],
    quotes: Optional[Sequence[Quote]] = None,
    avg_daily_volume: Optional[float] = None,
) -> tuple[int, str]:
    """Score executable liquidity from volume, recent tape, spread, and size."""
    day_volume, recent_avg, bar_rvol = _recent_volume_stats(today_bars)
    daily_rvol = (
        day_volume / float(avg_daily_volume)
        if avg_daily_volume and avg_daily_volume > 0
        else 0.0
    )
    effective_rvol = max(bar_rvol, daily_rvol)
    score = 0
    parts: list[str] = []

    if day_volume >= 2_000_000:
        score += 30
        parts.append("dayVol=30")
    elif day_volume >= 1_000_000:
        score += 25
        parts.append("dayVol=25")
    elif day_volume >= 500_000:
        score += 20
        parts.append("dayVol=20")
    elif day_volume >= 200_000:
        score += 12
        parts.append("dayVol=12")
    else:
        parts.append("dayVol=0")

    if recent_avg >= 75_000:
        score += 25
        parts.append("recentVol=25")
    elif recent_avg >= 50_000:
        score += 20
        parts.append("recentVol=20")
    elif recent_avg >= 25_000:
        score += 14
        parts.append("recentVol=14")
    elif recent_avg >= 10_000:
        score += 8
        parts.append("recentVol=8")
    else:
        parts.append("recentVol=0")

    if effective_rvol >= 3.0:
        score += 20
        parts.append("rvol{:.1f}x=20".format(effective_rvol))
    elif effective_rvol >= 2.0:
        score += 18
        parts.append("rvol{:.1f}x=18".format(effective_rvol))
    elif effective_rvol >= 1.5:
        score += 15
        parts.append("rvol{:.1f}x=15".format(effective_rvol))
    elif effective_rvol >= 1.0:
        score += 10
        parts.append("rvol{:.1f}x=10".format(effective_rvol))
    elif effective_rvol > 0:
        score += 4
        parts.append("rvol{:.1f}x=4".format(effective_rvol))
    elif recent_avg >= 50_000:
        score += 10
        parts.append("rvol=10")
    else:
        score += 5
        parts.append("rvol=5")

    valid_quotes = [q for q in list(quotes or [])[-5:] if q.ask > q.bid > 0]
    if valid_quotes and price > 0:
        avg_spread_pct = (
            sum((q.ask - q.bid) / price for q in valid_quotes) / len(valid_quotes)
        ) * 100
        if avg_spread_pct <= 0.15:
            score += 15
            parts.append("spread=15")
        elif avg_spread_pct <= 0.30:
            score += 10
            parts.append("spread=10")
        elif avg_spread_pct <= 0.50:
            score += 5
            parts.append("spread=5")
        else:
            parts.append("spread=0")

        avg_size = sum(min(q.bid_size, q.ask_size) for q in valid_quotes) / len(valid_quotes)
        if avg_size >= 2_000:
            score += 10
            parts.append("size=10")
        elif avg_size >= 500:
            score += 7
            parts.append("size=7")
        elif avg_size >= 100:
            score += 4
            parts.append("size=4")
        else:
            score += 1
            parts.append("size=1")
    else:
        score += 16
        parts.append("quote=neutral")

    return min(100, int(score)), ",".join(parts)


def _log_candidate(
    symbol: str, price: float, score: int, passed: bool,
    reject_reason: Optional[str], ml_prob: Optional[float],
    breakdown: str, float_shares: Optional[float],
    today_bars: Sequence[Bar], rel_vol: float,
    session_high: float, session_open: float, prior_close: float,
) -> None:
    """Fire-and-forget log to ML data collector."""
    try:
        from daytrading.ml.data_collector import log_entry_candidate
        log_entry_candidate(
            symbol=symbol,
            price=price,
            score=score,
            passed=passed,
            reject_reason=reject_reason,
            ml_prob=ml_prob,
            breakdown=breakdown,
            float_shares=float_shares,
            day_volume=float(sum(b.volume for b in today_bars)),
            rel_vol=rel_vol,
            bars=list(today_bars),
            session_high=session_high,
            session_open=session_open,
            prior_close=prior_close,
            minutes_since_open=len(today_bars),
        )
    except Exception:
        pass


@dataclass(frozen=True)
class ScannerProfile:
    """Parameter set for a Warrior Trading style scanner."""
    name: str
    min_price: float
    max_price: float
    min_day_change_pct: float
    min_bar_volume: float
    min_today_volume: float
    require_volume_surge: bool


LOW_FLOAT_RUNNER = ScannerProfile(
    name="Low Float Runner",
    min_price=2.0,
    max_price=20.0,
    min_day_change_pct=5.0,
    min_bar_volume=10_000,
    min_today_volume=200_000,
    require_volume_surge=True,
)

MEDIUM_FLOAT_SQUEEZE = ScannerProfile(
    name="Medium Float Squeeze",
    min_price=5.0,
    max_price=50.0,
    min_day_change_pct=5.0,
    min_bar_volume=30_000,
    min_today_volume=500_000,
    require_volume_surge=True,
)

FORMER_MOMO = ScannerProfile(
    name="Former Momo $20+",
    min_price=20.0,
    max_price=500.0,
    min_day_change_pct=3.0,
    min_bar_volume=50_000,
    min_today_volume=1_000_000,
    require_volume_surge=False,
)

ALL_PROFILES: List[ScannerProfile] = [LOW_FLOAT_RUNNER, MEDIUM_FLOAT_SQUEEZE, FORMER_MOMO]


def check_entry_quality(
    bars: Sequence[Bar],
    *,
    symbol: str = "",
    min_price: float = 2.0,
    max_price: float = 20.0,
    min_rvol: float = 1.5,
    max_bar_age_seconds: int = 300,
    min_momentum_quality: int = 40,
    min_day_change_pct: float = 5.0,
    avg_daily_volume: Optional[float] = None,
    bars_5m: Optional[Sequence[Bar]] = None,
    float_shares: Optional[float] = None,
    ticks: Optional[Sequence[Tick]] = None,
    quotes: Optional[Sequence[Quote]] = None,
) -> Optional[str]:
    """Return a rejection reason string, or ``None`` if the setup is OK.

    Uses a scoring system: hard rejects first, then score 7 conditions.
    Score >= 60/100 passes.
    """
    def _rule_reject(reason: str) -> str:
        record_rule_rejection(symbol=symbol, reason=reason)
        return reason

    if not bars or len(bars) < 3:
        return _rule_reject("insufficient bars")

    latest = bars[-1]
    price = latest.close
    if price <= 0:
        return _rule_reject("invalid price")

    # --- Split bars into today vs historical ---
    today_bars: Sequence[Bar] = bars
    if latest.ts is not None:
        try:
            today_date = latest.ts.date()
            today_bars = [b for b in bars if b.ts is not None and b.ts.date() == today_date]
        except Exception:
            pass

    if len(today_bars) < 3:
        today_bars = bars

    # =================================================================
    # HARD REJECTS — absolute deal-breakers, no scoring possible
    # =================================================================

    # 1. Price must be in a tradeable range
    if price < min_price or price > max_price:
        return _rule_reject(
            "price ${:.2f} outside range ${:.2f}-${:.2f}".format(price, min_price, max_price)
        )

    # 2. Staleness — data too old to act on (5 minutes)
    #    Exception: if stock was running hot before going quiet, it's likely
    #    halted (LULD circuit breaker) — don't reject, it may resume with continuation
    if latest.ts is not None:
        try:
            bar_time = latest.ts if latest.ts.tzinfo else latest.ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - bar_time).total_seconds()
            if age > max_bar_age_seconds:
                # Check if this looks like a halt (high volume spike before silence)
                is_likely_halt = False
                if len(today_bars) >= 3:
                    recent_bars = today_bars[-5:] if len(today_bars) >= 5 else today_bars[-3:]
                    avg_vol = sum(b.volume for b in today_bars) / len(today_bars)
                    last_vol = recent_bars[-1].volume
                    last_change = abs(recent_bars[-1].close - recent_bars[-1].open) / recent_bars[-1].open * 100 if recent_bars[-1].open > 0 else 0
                    # Halt signature: last bar had above-average volume + big move
                    if last_vol > avg_vol * 1.5 and last_change > 2.0:
                        is_likely_halt = True
                    # Also halt if recent run was very strong (multiple big green bars)
                    if len(recent_bars) >= 3:
                        green_count = sum(1 for b in recent_bars if b.close > b.open)
                        total_move = (recent_bars[-1].close - recent_bars[0].open) / recent_bars[0].open * 100 if recent_bars[0].open > 0 else 0
                        if green_count >= 2 and total_move > 5.0:
                            is_likely_halt = True

                if not is_likely_halt:
                    return _rule_reject(
                        "stale data ({:.0f}s old, max={}s)".format(age, max_bar_age_seconds)
                    )
                # Halted stock: allow through but cap staleness at 15 min
                if age > 900:
                    return _rule_reject(
                        "stale data ({:.0f}s old, likely halted too long)".format(age)
                    )
        except Exception:
            pass

    # 3. Below VWAP — buying into sellers is always wrong for momentum
    last_vwap = 0.0
    if len(today_bars) >= 3:
        vwap_vals = vwap(today_bars)
        last_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if not math.isnan(last_vwap) and last_vwap > 0:
            if price < last_vwap * 0.995:
                return _rule_reject("below VWAP ({:.2f} < {:.2f})".format(price, last_vwap))

    # 4. Dead cat bounce — price crashed too far from session HOD
    if len(today_bars) >= 5:
        today_high = max(b.high for b in today_bars)
        session_open_price = today_bars[0].open
        if today_high > 0 and session_open_price > 0:
            drop_from_high = (today_high - price) / today_high
            if drop_from_high > 0.20:
                return _rule_reject(
                    "dead cat bounce: price {:.2f} is {:.0f}% below HOD {:.2f}".format(
                        price, drop_from_high * 100, today_high)
                )

    # 5. Minimum upward movement — avoid flat names with no momentum edge
    if len(today_bars) >= 3:
        session_open_price = today_bars[0].open
        if session_open_price > 0:
            day_change_pct_hard = ((price - session_open_price) / session_open_price) * 100
            if day_change_pct_hard < min_day_change_pct:
                return _rule_reject(
                    "not enough movement: day change {:.1f}% (need {:.1f}%+)".format(
                        day_change_pct_hard, min_day_change_pct)
                )

    # 6. Minimum day volume — low-liquidity stocks produce unreliable breakouts
    day_volume = sum(b.volume for b in today_bars)
    _, recent_avg_volume, hard_gate_rvol = _recent_volume_stats(today_bars)
    daily_rvol = (
        day_volume / float(avg_daily_volume)
        if avg_daily_volume and avg_daily_volume > 0
        else 0.0
    )
    sub5_has_relative_momentum = (
        price < 5.0
        and day_volume >= 200_000
        and recent_avg_volume >= 20_000
        and max(daily_rvol, hard_gate_rvol) >= max(min_rvol, 1.8)
    )
    if day_volume < 200_000:
        return _rule_reject(
            "low day volume {:.0f} for ${:.2f} stock (need 200K+)".format(day_volume, price)
        )
    if price < 5.0 and day_volume < 500_000 and not sub5_has_relative_momentum:
        return _rule_reject(
            "thin sub-$5 liquidity {:.0f} volume, RVOL {:.1f}x (need 500K+ or strong RVOL before ML)".format(
                day_volume, max(daily_rvol, hard_gate_rvol)
            )
        )
    if price >= 20.0 and day_volume < 1_000_000:
        return _rule_reject(
            "low day volume {:.0f} for ${:.2f} stock (need 1M+)".format(day_volume, price)
        )
    elif price >= 10.0 and day_volume < 500_000:
        return _rule_reject(
            "low day volume {:.0f} for ${:.2f} stock (need 500K+)".format(day_volume, price)
        )

    # 7. Spread filter — wide spread means illiquid stock, bad fills
    if quotes and len(quotes) >= 3:
        recent_quotes = list(quotes[-5:])
        avg_spread = sum(q.ask - q.bid for q in recent_quotes if q.ask > q.bid > 0) / len(recent_quotes)
        if price > 0 and avg_spread / price > 0.005:
            return _rule_reject(
                "spread too wide ({:.2f}c = {:.2f}% of ${:.2f})".format(
                    avg_spread * 100, (avg_spread / price) * 100, price)
            )

    # 8. Full liquidity score — avoid weak chop even when simple volume passes
    liquidity_score, liquidity_parts = _liquidity_score(
        price, today_bars, quotes, avg_daily_volume=avg_daily_volume,
    )
    if price < 5.0 and liquidity_score < 65 and not (
        sub5_has_relative_momentum and liquidity_score >= 55
    ):
        return _rule_reject(
            "thin liquidity score {}/100 for sub-$5 stock ({})".format(
                liquidity_score, liquidity_parts)
        )
    if liquidity_score < 50:
        return _rule_reject(
            "watch-only liquidity score {}/100 ({})".format(
                liquidity_score, liquidity_parts)
        )

    # 9. Tape confirmation — sellers dominating means breakout is failing
    if ticks and len(ticks) >= 20:
        from daytrading.indicators.scalping import order_flow_imbalance
        imb_values = order_flow_imbalance(list(ticks), window=min(30, len(ticks)))
        current_imb = imb_values[-1] if imb_values else 0.0
        if current_imb <= -0.3:
            return _rule_reject(
                "tape shows selling pressure (imbalance={:.2f}, need >-0.3)".format(current_imb)
            )

    # =================================================================
    # SCORING SYSTEM — weighted conditions, threshold 60/100
    # =================================================================
    score = 0
    breakdown: List[str] = []

    # --- S1: Day Change (max 20 pts) ---
    day_change_pct = 0.0
    if len(today_bars) >= 3:
        session_open = today_bars[0].open
        if session_open > 0:
            day_change_pct = ((price - session_open) / session_open) * 100

    if day_change_pct >= 10.0:
        score += 20
        breakdown.append("day+{:.0f}%=20".format(day_change_pct))
    elif day_change_pct >= 5.0:
        score += 15
        breakdown.append("day+{:.0f}%=15".format(day_change_pct))
    elif day_change_pct >= 3.0:
        score += 10
        breakdown.append("day+{:.0f}%=10".format(day_change_pct))
    elif day_change_pct >= 1.5:
        score += 5
        breakdown.append("day+{:.1f}%=5".format(day_change_pct))
    else:
        breakdown.append("day+{:.1f}%=0".format(day_change_pct))

    # --- S2: Volume - Total Today (max 15 pts) ---
    today_total_vol = sum(b.volume for b in today_bars) if today_bars else 0
    if today_total_vol >= 500_000:
        score += 15
        breakdown.append("vol{:.0f}K=15".format(today_total_vol / 1000))
    elif today_total_vol >= 200_000:
        score += 10
        breakdown.append("vol{:.0f}K=10".format(today_total_vol / 1000))
    elif today_total_vol >= 100_000:
        score += 5
        breakdown.append("vol{:.0f}K=5".format(today_total_vol / 1000))
    else:
        breakdown.append("vol{:.0f}K=0".format(today_total_vol / 1000))

    # --- S3: Volume Surge — recent bars vs earlier (max 15 pts) ---
    bar_rvol = 0.0  # computed here, also used by S9 (low rvol penalty)
    if len(today_bars) >= 5:
        recent_5 = list(today_bars[-5:])
        recent_avg = sum(b.volume for b in recent_5) / 5
        if len(today_bars) >= 10:
            earlier = list(today_bars[:-5])
            earlier_avg = sum(b.volume for b in earlier) / len(earlier)
            bar_rvol = recent_avg / earlier_avg if earlier_avg > 0 else 0
            if bar_rvol >= 2.0 or recent_avg >= 50_000:
                score += 15
                breakdown.append("surge{:.1f}x=15".format(bar_rvol))
            elif bar_rvol >= 1.5 or recent_avg >= 30_000:
                score += 10
                breakdown.append("surge{:.1f}x=10".format(bar_rvol))
            elif bar_rvol >= 1.0 or recent_avg >= 10_000:
                score += 5
                breakdown.append("surge{:.1f}x=5".format(bar_rvol))
            else:
                breakdown.append("surge{:.1f}x=0".format(bar_rvol))
        else:
            if recent_avg >= 30_000:
                score += 15
                breakdown.append("barvol{:.0f}K=15".format(recent_avg / 1000))
            elif recent_avg >= 10_000:
                score += 10
                breakdown.append("barvol{:.0f}K=10".format(recent_avg / 1000))
            else:
                score += 5
                breakdown.append("barvol{:.0f}K=5".format(recent_avg / 1000))
    elif len(today_bars) >= 3:
        recent_avg = sum(b.volume for b in today_bars[-3:]) / 3
        if recent_avg >= 10_000:
            score += 10
            breakdown.append("barvol{:.0f}K=10".format(recent_avg / 1000))
        else:
            score += 3
            breakdown.append("barvol{:.0f}K=3".format(recent_avg / 1000))

    # --- S4: Momentum Quality (max 15 pts) ---
    mq_score = 0
    if len(today_bars) >= 3:
        mq_score, _ = _momentum_quality(today_bars)
    if mq_score >= 70:
        score += 15
        breakdown.append("mq{}=15".format(mq_score))
    elif mq_score >= 50:
        score += 10
        breakdown.append("mq{}=10".format(mq_score))
    elif mq_score >= 30:
        score += 5
        breakdown.append("mq{}=5".format(mq_score))
    else:
        breakdown.append("mq{}=0".format(mq_score))

    # --- S5: Near HOD — not fading (max 15 pts) ---
    if len(today_bars) >= 5:
        lookback = min(10, len(today_bars))
        recent_high = max(b.high for b in today_bars[-lookback:])
        if recent_high > 0:
            pullback_pct = (recent_high - price) / recent_high
            if pullback_pct <= 0.02:
                score += 15
                breakdown.append("nearHOD{:.1f}%=15".format(pullback_pct * 100))
            elif pullback_pct <= 0.05:
                score += 10
                breakdown.append("nearHOD{:.1f}%=10".format(pullback_pct * 100))
            elif pullback_pct <= 0.08:
                score += 5
                breakdown.append("nearHOD{:.1f}%=5".format(pullback_pct * 100))
            else:
                breakdown.append("fading{:.1f}%=0".format(pullback_pct * 100))
        else:
            score += 5
            breakdown.append("noHOD=5")
    else:
        score += 5
        breakdown.append("fewBars=5")

    # --- S6: Candle Strength (max 10 pts) ---
    if latest.high > latest.low:
        body = abs(latest.close - latest.open)
        full_range = latest.high - latest.low
        body_ratio = body / full_range if full_range > 0 else 0
        is_green = latest.close >= latest.open
        if body_ratio >= 0.5 and is_green:
            score += 10
            breakdown.append("candle{:.0f}%G=10".format(body_ratio * 100))
        elif body_ratio >= 0.3 and is_green:
            score += 7
            breakdown.append("candle{:.0f}%G=7".format(body_ratio * 100))
        elif body_ratio >= 0.15:
            score += 4
            breakdown.append("candle{:.0f}%=4".format(body_ratio * 100))
        else:
            breakdown.append("doji{:.0f}%=0".format(body_ratio * 100))
    else:
        score += 3
        breakdown.append("noRange=3")

    # --- S7: 5-min Trend Support (max 10 pts) ---
    if bars_5m and len(bars_5m) >= 3:
        recent_5m = list(bars_5m[-3:])
        closes_5m = [b.close for b in recent_5m]
        green_5m = sum(1 for b in recent_5m if b.close >= b.open)

        if closes_5m[-1] > closes_5m[0] and green_5m >= 2:
            score += 10
            breakdown.append("5mUp=10")
        elif closes_5m[-1] >= closes_5m[0] or green_5m >= 2:
            score += 6
            breakdown.append("5mFlat=6")
        elif green_5m >= 1:
            score += 3
            breakdown.append("5mMixed=3")
        else:
            breakdown.append("5mDown=0")
    elif bars_5m and len(bars_5m) >= 1:
        last_5m = bars_5m[-1]
        if last_5m.close >= last_5m.open:
            score += 6
            breakdown.append("5mGreen=6")
        else:
            score += 2
            breakdown.append("5mRed=2")
    else:
        score += 5
        breakdown.append("no5m=5")

    # --- S8: Volume Exhaustion Penalty (0 to -30 pts) ---
    exhaust_penalty = _volume_exhaustion_penalty(today_bars)
    if exhaust_penalty < 0:
        score += exhaust_penalty
        breakdown.append("exhaust={}".format(exhaust_penalty))

    # --- S9: Low Relative Volume Penalty (0 to -25 pts) ---
    if bar_rvol > 0 and len(today_bars) >= 10:
        if bar_rvol < 0.5:
            score -= 25
            breakdown.append("rvol{:.1f}x=-25".format(bar_rvol))
        elif bar_rvol < 1.0:
            score -= 20
            breakdown.append("rvol{:.1f}x=-20".format(bar_rvol))
        elif bar_rvol < 2.0:
            score -= 5
            breakdown.append("rvol{:.1f}x=-5".format(bar_rvol))

    # =================================================================
    # FINAL DECISION
    # =================================================================

    # =================================================================
    # FINAL DECISION
    # Score is computed for data collection and ML training, but does NOT
    # block entries. Only ML (when trained) can reject based on quality.
    # Safety checks above (stale, VWAP, spread, tape) still block.
    # =================================================================

    # Compute values needed for data collection
    _session_high = max(b.high for b in today_bars) if today_bars else price
    _session_open = today_bars[0].open if today_bars else price
    _prior_close_est = _session_open / (1 + day_change_pct / 100) if day_change_pct > 0 else _session_open

    ml_prob_val = None
    # XGBoost ML check (optional — skipped if model not loaded or monitor disabled it)
    ml_active = _xgb_model is not None and (
        _ml_monitor is None or _ml_monitor.is_model_enabled
    )
    if ml_active:
        try:
            from daytrading.ml.features import compute_entry_features
            import xgboost as xgb
            import numpy as np

            features = compute_entry_features(
                price,
                float_shares=float_shares,
                day_volume=float(sum(b.volume for b in today_bars)),
                rel_vol=bar_rvol,
                session_high=_session_high,
                session_open=_session_open,
                prior_close=_prior_close_est,
                bars=list(today_bars),
                minutes_since_open=len(today_bars),
            )
            dmat = xgb.DMatrix([features])
            ml_prob_val = float(_xgb_model.predict(dmat)[0])
            if ml_prob_val < _XGB_THRESHOLD:
                if score >= _ML_SOFT_PASS_SCORE:
                    logger.info(
                        "ENTRY GUARD ML SOFT PASS %s: prob=%.0f%% < %.0f%% but rule score=%d >= %d",
                        symbol, ml_prob_val * 100, _XGB_THRESHOLD * 100,
                        score, _ML_SOFT_PASS_SCORE,
                    )
                    if _ml_monitor:
                        try:
                            _ml_monitor.record_ml_rejection(
                                symbol, price, ml_prob_val, score, counted=False,
                            )
                        except TypeError:
                            _ml_monitor.record_ml_rejection(symbol, price, ml_prob_val, score)
                else:
                    logger.info("ENTRY GUARD ML REJECT %s: prob=%.0f%% < %.0f%% [score=%d]",
                                symbol, ml_prob_val * 100, _XGB_THRESHOLD * 100, score)
                    if _ml_monitor:
                        _ml_monitor.record_ml_rejection(symbol, price, ml_prob_val, score)
                    _log_candidate(symbol, price, score, False,
                                   "ML low confidence ({:.0f}%)".format(ml_prob_val * 100),
                                   ml_prob_val, ", ".join(breakdown),
                                   float_shares, today_bars, bar_rvol,
                                   _session_high, _session_open, _prior_close_est)
                    return "ML model low confidence ({:.0f}%, need {:.0f}%)".format(
                        ml_prob_val * 100, _XGB_THRESHOLD * 100)
        except Exception as exc:
            logger.debug("ML scoring skipped for %s: %s", symbol, exc)

    # PASS — log score for data collection (score no longer blocks)
    if _ml_monitor:
        _ml_monitor.record_entry_passed()

    logger.info("ENTRY GUARD SCORE %s: %d/100 PASS [%s]",
                symbol, score, ", ".join(breakdown))
    _log_candidate(symbol, price, score, True, None, ml_prob_val,
                   ", ".join(breakdown), float_shares, today_bars, bar_rvol,
                   _session_high, _session_open, _prior_close_est)
    return None  # PASS


def _volume_exhaustion_penalty(today_bars: Sequence[Bar]) -> int:
    """Return negative points (0 to -30) if recent green bars show declining volume."""
    if len(today_bars) < 3:
        return 0

    recent = list(today_bars[-5:])

    declining_streak = 0
    for i in range(len(recent) - 1, 0, -1):
        bar = recent[i]
        prev = recent[i - 1]
        is_green = bar.close > bar.open
        vol_declining = bar.volume < prev.volume
        if is_green and vol_declining:
            declining_streak += 1
        else:
            break

    if declining_streak >= 4:
        return -30
    elif declining_streak >= 3:
        return -20
    elif declining_streak >= 2:
        return -10
    return 0


def _select_profiles(price: float, float_shares: Optional[float]) -> List[ScannerProfile]:
    """Return profiles that match the stock's price range and float."""
    profiles = []
    for p in ALL_PROFILES:
        if p.min_price <= price <= p.max_price:
            if float_shares is not None:
                if p.name == "Low Float Runner" and float_shares <= 20_000_000:
                    profiles.append(p)
                elif p.name == "Medium Float Squeeze" and 5_000_000 <= float_shares <= 100_000_000:
                    profiles.append(p)
                elif p.name == "Former Momo $20+" and float_shares >= 10_000_000:
                    profiles.append(p)
            else:
                profiles.append(p)
    return profiles


def _momentum_quality(bars: Sequence[Bar], lookback: int = 5) -> tuple:
    """Compute a momentum quality score from recent bars.

    Returns (score 0-100, reason_str).
    """
    n = min(lookback, len(bars))
    if n < 3:
        return (0, "too few bars")

    recent = list(bars[-n:])
    score = 0
    details = []

    green = sum(1 for b in recent if b.close > b.open)
    ratio = green / n
    pts = int(ratio * 30)
    score += pts
    details.append("{}/{} green".format(green, n))

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

    latest = recent[-1]
    if latest.high > latest.low:
        position = (latest.close - latest.low) / (latest.high - latest.low)
        pts = int(position * 20)
        score += pts
        details.append("close@{:.0f}%".format(position * 100))

    return (score, ", ".join(details))
