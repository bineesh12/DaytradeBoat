from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from daytrading.models import Bar


def momentum_burst_level_context(previous_high: float, current_high: float) -> Dict[str, Any]:
    try:
        prev = float(previous_high or 0.0)
        high = float(current_high or 0.0)
    except (TypeError, ValueError):
        return {}
    if high <= 0 or high <= prev:
        return {}
    step = 0.5 if high < 10.0 else 1.0
    next_level = (int(prev / step) + 1) * step
    if next_level <= high and next_level >= 1.0:
        return {
            "psych_level": round(next_level, 2),
            "entry_trigger": "psych_level_break",
        }
    return {}


def first_starter_has_proof_hold(history: Sequence[Bar], proof_level: float) -> bool:
    if proof_level <= 0:
        return False
    recent = list(history)[-6:]
    if len(recent) < 3:
        return False
    for bar in recent:
        close = float(bar.close or 0.0)
        if close <= 0:
            return False
        range_pct = (float(bar.high or 0.0) - float(bar.low or 0.0)) / close
        if range_pct > 0.20:
            return False
    hold_floor = proof_level * 0.99
    hold_bars = [
        bar for bar in recent[-3:]
        if float(bar.close or 0.0) >= proof_level
        and float(bar.low or 0.0) >= hold_floor
    ]
    return len(hold_bars) >= 2


def warrior_squeeze_pullaway_context(
    latest_bar: Bar,
    pending_breakout: Dict[str, Any],
    *,
    history: Sequence[Bar],
    reject_high: float,
    rejection_reason: Optional[str],
    reentry_count: int,
    min_reclaim_price: float,
    reward_risk_value: float,
    add_reward_risk_value: float,
) -> Optional[Dict[str, Any]]:
    reject_high = float(reject_high or 0.0)
    if reject_high <= 0:
        return None
    min_price = max(0.0, float(min_reclaim_price or 0.0))
    if reentry_count > 0:
        step = 0.5 if float(latest_bar.high or 0.0) < 10.0 else 1.0
        proof_level = (int(float(latest_bar.close or 0.0) / step)) * step
        proof_level = max(proof_level, min_price)
    else:
        proof_level = max(reject_high * 1.03, min_price)
        warrior_reclaim_trigger = pending_breakout.get("entry_trigger") in {
            "warrior_a_plus_reclaim",
            "psych_level_break",
        }
        stale_max_pay = round(proof_level * 1.06, 4)
        if warrior_reclaim_trigger and float(latest_bar.low or 0.0) > stale_max_pay:
            step = 0.5 if float(latest_bar.close or 0.0) < 10.0 else 1.0
            proof_level = max(
                proof_level,
                (int(float(latest_bar.close or 0.0) / step)) * step,
            )
    if proof_level <= 0:
        return None
    open_ = float(latest_bar.open or 0.0)
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if high < proof_level or close < proof_level:
        return None
    if close <= open_:
        return None
    rng = max(high - low, 0.0)
    close_location = (close - low) / rng if rng > 0 else 0.0
    range_pct = rng / close if close > 0 else 0.0
    clwt_fast_pullaway = (
        reentry_count == 0
        and proof_level < 5.0
        and rejection_reason in {
            "high-volume shooting-star rejection",
            "first explosive 10s spike",
        }
        and high >= proof_level * 1.14
        and close >= proof_level * 1.08
        and close_location >= 0.62
    )
    if proof_level >= 5.0 and range_pct < 0.15 and close_location < 0.72:
        return None
    warrior_reclaim_trigger = pending_breakout.get("entry_trigger") in {
        "warrior_a_plus_reclaim",
        "psych_level_break",
    }
    min_close_location = 0.30 if warrior_reclaim_trigger else 0.55
    if close_location < min_close_location:
        return None
    if (
        reentry_count == 0
        and proof_level < 5.0
        and not clwt_fast_pullaway
        and not first_starter_has_proof_hold(history, proof_level)
    ):
        return None
    breakout_volume = float(pending_breakout.get("breakout_volume") or 0.0)
    min_ratio = 0.20 if warrior_reclaim_trigger else 0.75
    min_volume_floor = 75_000.0 if warrior_reclaim_trigger else 150_000.0
    if volume < max(min_volume_floor, breakout_volume * min_ratio):
        return None
    max_pay = round(proof_level * (1.15 if clwt_fast_pullaway else 1.06), 4)
    if low > max_pay:
        return None
    entry_price = min(close, max_pay)
    stop_price = round(max(0.01, proof_level - max(0.08, proof_level * 0.03)), 4)
    risk = max(entry_price - stop_price, entry_price * 0.015, 0.06)
    if warrior_reclaim_trigger:
        risk = min(risk, entry_price * 0.058)
    stop_price = round(entry_price - risk, 4)
    reward_risk = max(1.0, float(reward_risk_value or 3.0))
    if reentry_count > 0:
        add_reward_risk = max(0.2, float(add_reward_risk_value or 1.0))
        step = 0.5 if entry_price < 10.0 else 1.0
        level_target = round((int(entry_price / step) + 1) * step, 4)
        rr_target = round(entry_price + risk * add_reward_risk, 4)
        target_price = min(level_target, rr_target) if level_target > entry_price else rr_target
        if target_price <= entry_price + max(0.04, risk * 0.25):
            target_price = round(entry_price + risk * add_reward_risk, 4)
    else:
        target_price = round(entry_price + risk * reward_risk, 4)
    if (
        reentry_count == 0
        and proof_level >= 5.0
        and not clwt_fast_pullaway
        and rejection_reason in {
            "high-volume shooting-star rejection",
            "first explosive 10s spike",
        }
    ):
        return None
    return {
        "entry_trigger": "warrior_level_pullaway",
        "variant_override": (
            "warrior_clwt_fast_pullaway"
            if clwt_fast_pullaway else
            "warrior_proof_pullback_hold"
            if proof_level < 5.0
            else "warrior_level_pullaway_starter"
        ),
        "psych_level": round(proof_level, 4),
        "pullaway_level": round(proof_level, 4),
        "max_pay": max_pay,
        "entry_price_override": round(entry_price, 4),
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "max_hold_seconds_override": 180.0 if proof_level < 5.0 else None,
        "reward_risk": round(reward_risk, 2),
        "rr_note_override": (
            "warrior level pull-away starter "
            "level=${:.2f} cap=${:.2f} risk={:.1f}% target=${:.2f}"
        ).format(
            proof_level,
            max_pay,
            risk / entry_price * 100.0 if entry_price else 0.0,
            target_price,
        ),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_squeeze_equal_high_pullaway_context(
    latest_bar: Bar,
    pending_breakout: Dict[str, Any],
    *,
    history: Sequence[Bar],
    window_high: float,
    reject_high: float,
    rejection_reason: Optional[str],
    reentry_count: int,
    min_reclaim_price: float,
) -> Optional[Dict[str, Any]]:
    if int(reentry_count or 0) > 0:
        return None
    reject_high = float(reject_high or 0.0)
    if reject_high <= 0:
        return None
    if rejection_reason not in {
        "high-volume shooting-star rejection",
        "first explosive 10s spike",
    }:
        return None
    min_price = max(0.0, float(min_reclaim_price or 0.0))
    proof_level = max(reject_high * 1.03, min_price)
    if proof_level >= 5.0:
        return None
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= open_ or close <= 0 or high <= 0:
        return None
    if close < proof_level * 0.995 or high < proof_level:
        return None
    if window_high > 0 and high < window_high * 0.985:
        return None
    rng = max(high - low, 0.0)
    close_location = (close - low) / rng if rng > 0 else 0.0
    range_pct = rng / close if close > 0 else 0.0
    fast_pullaway = (
        high >= proof_level * 1.14
        and close >= proof_level * 1.08
        and close_location >= 0.62
    )
    if close_location < 0.66 and not fast_pullaway:
        return None
    if range_pct > 0.22:
        return None
    history = [bar for bar in history if float(bar.close or 0.0) > 0]
    prior = history[:-1]
    prior_high = max((float(bar.high or 0.0) for bar in prior), default=0.0)
    if not fast_pullaway and prior_high >= proof_level * 1.18 and high < prior_high * 0.94:
        return None
    recent = history[-6:]
    if len(recent) < 3:
        return None
    proof_hold = [
        bar for bar in recent[-4:]
        if float(bar.close or 0.0) >= proof_level * 0.985
        and float(bar.low or 0.0) >= proof_level * 0.92
    ]
    if len(proof_hold) < 1 and not fast_pullaway:
        return None
    prior = recent[:-1]
    avg_recent_volume = (
        sum(float(bar.volume or 0.0) for bar in prior) / len(prior)
        if prior else 0.0
    )
    breakout_volume = float(pending_breakout.get("breakout_volume") or 0.0)
    if volume < max(100_000.0, avg_recent_volume * 0.70, breakout_volume * 0.35):
        return None
    topping_tail = (
        high > close
        and (high - close) / close > 0.035
        and close_location < 0.72
    )
    if topping_tail:
        return None
    max_pay = round(proof_level * (1.15 if fast_pullaway else 1.045), 4)
    if low > max_pay:
        return None
    if fast_pullaway and close > max_pay:
        return None
    entry_price = close if fast_pullaway else min(close, max_pay)
    risk = max(entry_price * 0.025, 0.08)
    risk = min(risk, entry_price * 0.058)
    stop_price = round(max(0.01, entry_price - risk), 4)
    target_price = round(entry_price + risk * 2.4, 4)
    return {
        "entry_trigger": "warrior_equal_high_pullaway",
        "variant_override": "warrior_equal_high_pullaway",
        "psych_level": round(proof_level, 4),
        "pullaway_level": round(proof_level, 4),
        "max_pay": max_pay,
        "entry_price_override": round(entry_price, 4),
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "max_hold_seconds_override": 180.0,
        "reward_risk": 2.4,
        "rr_note_override": (
            "warrior equal-high pull-away starter "
            "level=${:.2f} cap=${:.2f} risk={:.1f}% target=${:.2f}"
        ).format(
            proof_level,
            max_pay,
            risk / entry_price * 100.0 if entry_price else 0.0,
            target_price,
        ),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_squeeze_second_leg_reclaim_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    history = [bar for bar in history if float(bar.close or 0.0) > 0]
    if len(history) < 18:
        return None
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= open_ or close <= 0 or high <= 0:
        return None
    prior = history[:-1]
    if not prior:
        return None
    prior_high = max(float(bar.high or 0.0) for bar in prior)
    squeeze_high = max(float(window_high or 0.0), prior_high)
    if squeeze_high < max(10.0, close * 1.20):
        return None
    recent = history[-12:]
    base_bars = history[-10:-1]
    if len(base_bars) < 6:
        return None
    base_high = max(float(bar.high or 0.0) for bar in base_bars)
    base_low = min(float(bar.low or 0.0) for bar in base_bars)
    if base_high <= 0 or base_low <= 0:
        return None
    washout_pct = (squeeze_high - base_low) / squeeze_high
    if washout_pct < 0.25 or washout_pct > 0.60:
        return None
    if close > squeeze_high * 0.92:
        return None
    base_range_pct = (base_high - base_low) / close if close > 0 else 999.0
    if base_range_pct > 0.22:
        return None
    if high < base_high or close < base_high * 0.995:
        return None
    rng = max(high - low, 0.0)
    close_location = (close - low) / rng if rng > 0 else 0.0
    if close_location < 0.58:
        return None
    recent_volume = sum(float(bar.volume or 0.0) for bar in recent[-3:])
    avg_base_volume = sum(float(bar.volume or 0.0) for bar in base_bars) / len(base_bars)
    if volume < max(75_000.0, avg_base_volume * 0.80) or recent_volume < max(180_000.0, avg_base_volume * 1.75):
        return None
    heavy_red = any(
        float(bar.close or 0.0) < float(bar.open or 0.0)
        and (float(bar.open or 0.0) - float(bar.close or 0.0)) / float(bar.close or 1.0) > 0.07
        and float(bar.volume or 0.0) >= avg_base_volume * 1.25
        for bar in history[-4:-1]
    )
    if heavy_red:
        return None
    step = 0.5 if close < 10.0 else 1.0
    reclaim_level = max(base_high, (int(close / step)) * step)
    max_pay = round(reclaim_level * 1.04, 4)
    if low > max_pay:
        return None
    entry_price = min(close, max_pay)
    stop_anchor = max(base_low, reclaim_level * 0.92)
    stop_price = round(max(0.01, stop_anchor - max(0.04, entry_price * 0.004)), 4)
    risk = max(entry_price - stop_price, entry_price * 0.018, 0.10)
    if risk > entry_price * 0.105:
        return None
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.8, 4)
    return {
        "entry_trigger": "warrior_second_leg_reclaim",
        "variant_override": "warrior_second_leg_reclaim",
        "psych_level": round(reclaim_level, 4),
        "pullaway_level": round(reclaim_level, 4),
        "base_high": round(base_high, 4),
        "base_low": round(base_low, 4),
        "washout_pct": round(washout_pct * 100.0, 2),
        "max_pay": max_pay,
        "entry_price_override": round(entry_price, 4),
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "max_hold_seconds_override": 240.0,
        "reward_risk": 1.8,
        "rr_note_override": (
            "warrior second-leg reclaim base=${:.2f}-${:.2f} "
            "washout={:.1f}% cap=${:.2f} target=${:.2f}"
        ).format(base_low, base_high, washout_pct * 100.0, max_pay, target_price),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_squeeze_curl_reclaim_context(
    latest_bar: Bar,
    pending_breakout: Dict[str, Any],
    *,
    history: Sequence[Bar],
    window_high: float,
    reentry_count: int,
    min_reclaim_price: float,
) -> Optional[Dict[str, Any]]:
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= open_ or close <= 0 or high <= 0:
        return None
    min_price = max(0.0, float(min_reclaim_price or 0.0))
    if close < min_price:
        return None
    rng = max(high - low, 0.0)
    close_location = (close - low) / rng if rng > 0 else 0.0
    if close_location < 0.58:
        return None
    step = 0.5 if close < 10.0 else 1.0
    reclaim_level = max(min_price, (int(close / step)) * step)
    if high < reclaim_level or close < reclaim_level:
        return None
    range_pct = rng / close if close > 0 else 0.0
    if reentry_count == 0 and reclaim_level >= 7.5:
        return None
    if reclaim_level >= 5.0 and range_pct < 0.15 and close_location < 0.72:
        strong_first_level_break = (
            reentry_count == 0
            and reclaim_level < 7.5
            and close_location >= 0.52
            and volume >= 250_000
        )
        if not strong_first_level_break:
            return None
    if (
        reentry_count == 0
        and reclaim_level < 5.0
        and not first_starter_has_proof_hold(history, reclaim_level)
    ):
        return None
    prior_high = max(
        float(window_high or 0.0),
        float(pending_breakout.get("breakout_high") or 0.0),
    )
    if prior_high > 0 and close < prior_high * 0.90:
        return None
    breakout_volume = float(pending_breakout.get("breakout_volume") or 0.0)
    if volume < max(100_000.0, breakout_volume * 0.30):
        return None
    max_pay = round(reclaim_level * 1.05, 4)
    if low > max_pay:
        return None
    entry_price = min(close, max_pay)
    stop_price = round(max(0.01, reclaim_level - max(0.10, reclaim_level * 0.035)), 4)
    risk = max(entry_price - stop_price, entry_price * 0.015, 0.08)
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.5, 4)
    return {
        "entry_trigger": "warrior_curl_reclaim",
        "psych_level": round(reclaim_level, 4),
        "pullaway_level": round(reclaim_level, 4),
        "max_pay": max_pay,
        "entry_price_override": round(entry_price, 4),
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "reward_risk": 1.5,
        "rr_note_override": (
            "warrior curl reclaim starter "
            "level=${:.2f} cap=${:.2f} risk={:.1f}% target=${:.2f}"
        ).format(
            reclaim_level,
            max_pay,
            risk / entry_price * 100.0 if entry_price else 0.0,
            target_price,
        ),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_trend_pullback_reclaim_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    history = [bar for bar in history if float(bar.close or 0.0) > 0][-30:]
    if len(history) < 10:
        return None
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= open_ or close <= 0 or high <= 0:
        return None
    prior = history[:-1]
    prior_high = max((float(bar.high or 0.0) for bar in prior), default=0.0)
    trend_high = max(float(window_high or 0.0), prior_high)
    if trend_high < max(4.0, close * 1.025):
        return None
    base_bars = history[-7:-1]
    if len(base_bars) < 5:
        return None
    base_high = max(float(bar.high or 0.0) for bar in base_bars)
    base_low = min(float(bar.low or 0.0) for bar in base_bars)
    if base_high <= 0 or base_low <= 0:
        return None
    pullback_pct = (trend_high - base_low) / trend_high if trend_high > 0 else 0.0
    if pullback_pct < 0.025 or pullback_pct > 0.22:
        return None
    base_range_pct = (base_high - base_low) / close if close > 0 else 999.0
    if base_range_pct > 0.16:
        return None
    total_volume = sum(float(bar.volume or 0.0) for bar in history)
    if total_volume <= 0:
        return None
    vwap = sum(
        ((float(bar.high or 0.0) + float(bar.low or 0.0) + float(bar.close or 0.0)) / 3.0)
        * float(bar.volume or 0.0)
        for bar in history
    ) / total_volume
    ema = float(history[0].close or 0.0)
    alpha = 2.0 / 10.0
    for bar in history[1:]:
        ema = float(bar.close or 0.0) * alpha + ema * (1.0 - alpha)
    if close < vwap * 0.98 or base_low < vwap * 0.88:
        return None
    if base_low > max(vwap, ema) * 1.06 and close > trend_high * 0.965:
        return None
    if high < base_high or close < base_high * 0.995:
        return None
    rng = max(high - low, 0.0)
    close_location = (close - low) / rng if rng > 0 else 0.0
    if close_location < 0.56:
        return None
    avg_base_volume = sum(float(bar.volume or 0.0) for bar in base_bars) / len(base_bars)
    recent_volume = sum(float(bar.volume or 0.0) for bar in history[-3:])
    if volume < max(75_000.0, avg_base_volume * 0.80):
        return None
    if recent_volume < max(180_000.0, avg_base_volume * 1.35):
        return None
    heavy_red = any(
        float(bar.close or 0.0) < float(bar.open or 0.0)
        and (float(bar.open or 0.0) - float(bar.close or 0.0)) / float(bar.close or 1.0) > 0.055
        and float(bar.volume or 0.0) >= avg_base_volume * 1.30
        for bar in history[-5:-1]
    )
    if heavy_red:
        return None
    reclaim_level = base_high
    max_pay = round(reclaim_level * 1.025, 4)
    if low > max_pay:
        return None
    entry_price = min(close, max_pay)
    stop_anchor = max(base_low, vwap * 0.96, ema * 0.94)
    stop_price = round(max(0.01, stop_anchor - max(0.04, entry_price * 0.004)), 4)
    risk = max(entry_price - stop_price, entry_price * 0.015, 0.08)
    if risk > entry_price * 0.095:
        return None
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.7, 4)
    return {
        "entry_trigger": "warrior_trend_pullback_reclaim",
        "variant_override": "warrior_trend_pullback_reclaim",
        "psych_level": round(reclaim_level, 4),
        "pullaway_level": round(reclaim_level, 4),
        "base_high": round(base_high, 4),
        "base_low": round(base_low, 4),
        "pullback_pct": round(pullback_pct * 100.0, 2),
        "vwap": round(vwap, 4),
        "ema9": round(ema, 4),
        "max_pay": max_pay,
        "entry_price_override": round(entry_price, 4),
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "max_hold_seconds_override": 210.0,
        "reward_risk": 1.7,
        "rr_note_override": (
            "warrior trend pullback reclaim base=${:.2f}-${:.2f} "
            "pullback={:.1f}% vwap=${:.2f} cap=${:.2f} target=${:.2f}"
        ).format(base_low, base_high, pullback_pct * 100.0, vwap, max_pay, target_price),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_news_continuation_pullback_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    history = [bar for bar in history if float(bar.close or 0.0) > 0][-36:]
    if len(history) < 12:
        return None
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= open_ or close <= 0 or high <= 0:
        return None
    prior_high = max((float(bar.high or 0.0) for bar in history[:-1]), default=0.0)
    squeeze_high = max(float(window_high or 0.0), prior_high)
    if squeeze_high < max(7.0, close * 1.08):
        return None
    base_bars = history[-7:-1]
    if len(base_bars) < 5:
        return None
    base_high = max(float(bar.high or 0.0) for bar in base_bars)
    base_low = min(float(bar.low or 0.0) for bar in base_bars)
    if base_high <= 0 or base_low <= 0:
        return None
    pullback_pct = (squeeze_high - base_low) / squeeze_high
    if pullback_pct < 0.06 or pullback_pct > 0.28:
        return None
    base_range_pct = (base_high - base_low) / close if close > 0 else 999.0
    if base_range_pct > 0.14:
        return None
    if high < base_high or close < base_high * 0.995:
        return None
    rng = max(high - low, 0.0)
    close_location = (close - low) / rng if rng > 0 else 0.0
    if close_location < 0.58:
        return None
    total_volume = sum(float(bar.volume or 0.0) for bar in history)
    if total_volume <= 0:
        return None
    vwap = sum(
        ((float(bar.high or 0.0) + float(bar.low or 0.0) + float(bar.close or 0.0)) / 3.0)
        * float(bar.volume or 0.0)
        for bar in history
    ) / total_volume
    if close < vwap * 0.98 or base_low < vwap * 0.88:
        return None
    avg_base_volume = sum(float(bar.volume or 0.0) for bar in base_bars) / len(base_bars)
    recent_volume = sum(float(bar.volume or 0.0) for bar in history[-3:])
    if volume < max(100_000.0, avg_base_volume * 0.85):
        return None
    if recent_volume < max(250_000.0, avg_base_volume * 1.75):
        return None
    heavy_red = any(
        float(bar.close or 0.0) < float(bar.open or 0.0)
        and (float(bar.open or 0.0) - float(bar.close or 0.0)) / float(bar.close or 1.0) > 0.07
        and float(bar.volume or 0.0) >= avg_base_volume * 1.20
        for bar in history[-4:-1]
    )
    if heavy_red:
        return None
    reclaim_level = base_high
    max_pay = round(reclaim_level * 1.025, 4)
    if low > max_pay:
        return None
    entry_price = min(close, max_pay)
    stop_price = round(max(0.01, base_low - max(0.04, entry_price * 0.004)), 4)
    risk = max(entry_price - stop_price, entry_price * 0.015, 0.08)
    if risk > entry_price * 0.09:
        return None
    risk = min(risk, entry_price * 0.059)
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.5, 4)
    return {
        "entry_trigger": "warrior_news_continuation_pullback",
        "variant_override": "warrior_news_continuation_pullback",
        "psych_level": round(reclaim_level, 4),
        "pullaway_level": round(reclaim_level, 4),
        "base_high": round(base_high, 4),
        "base_low": round(base_low, 4),
        "pullback_pct": round(pullback_pct * 100.0, 2),
        "vwap": round(vwap, 4),
        "max_pay": max_pay,
        "entry_price_override": round(entry_price, 4),
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "max_hold_seconds_override": 180.0,
        "reward_risk": 1.5,
        "rr_note_override": (
            "warrior news-continuation pullback "
            "base=${:.2f}-${:.2f} pullback={:.1f}% vwap=${:.2f} target=${:.2f}"
        ).format(base_low, base_high, pullback_pct * 100.0, vwap, target_price),
        "skip_unstable_confirm_stop_check": True,
    }
