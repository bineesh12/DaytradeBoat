from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from daytrading.models import Bar


WARRIOR_ENTRY_TRIGGERS = {
    "warrior_level_pullaway",
    "warrior_curl_reclaim",
    "warrior_second_leg_reclaim",
    "warrior_prior_runner_continuation_pullback",
    "warrior_first_pullback_reclaim",
    "warrior_first_impulse_scalp",
    "warrior_high_base_reclaim",
    "warrior_stair_step_runner",
    "warrior_smooth_10s_pullback_continuation",
    "warrior_smooth_hod_reclaim",
    "warrior_post_target_pullback_reclaim",
    "warrior_trend_pullback_reclaim",
    "warrior_equal_high_pullaway",
    "warrior_level_break_starter",
    "warrior_failed_burst_recovery",
    "warrior_failed_spike_vwap_reclaim",
}


WARRIOR_LATE_REENTRY_TRIGGERS = {
    "warrior_level_pullaway",
    "warrior_curl_reclaim",
    "warrior_second_leg_reclaim",
    "warrior_prior_runner_continuation_pullback",
    "warrior_first_pullback_reclaim",
    "warrior_first_impulse_scalp",
    "warrior_high_base_reclaim",
    "warrior_stair_step_runner",
    "warrior_smooth_10s_pullback_continuation",
    "warrior_smooth_hod_reclaim",
    "warrior_post_target_pullback_reclaim",
    "warrior_trend_pullback_reclaim",
    "warrior_equal_high_pullaway",
    "warrior_failed_spike_vwap_reclaim",
}


WARRIOR_INITIAL_STARTER_TRIGGERS = {
    "warrior_level_break_starter",
    "warrior_first_pullback_reclaim",
    "warrior_high_base_reclaim",
    "warrior_smooth_10s_pullback_continuation",
    "warrior_smooth_hod_reclaim",
    "warrior_failed_spike_vwap_reclaim",
}

WARRIOR_HIGH_BASE_CONFIRM_TRIGGERS = {
    "warrior_first_pullback_reclaim",
    "warrior_high_base_reclaim",
    "warrior_smooth_10s_pullback_continuation",
    "warrior_smooth_hod_reclaim",
    "warrior_failed_spike_vwap_reclaim",
}

WARRIOR_FRESH_RECLAIM_TRIGGERS = {
    "warrior_first_pullback_reclaim",
    "warrior_first_impulse_scalp",
    "warrior_high_base_reclaim",
    "warrior_stair_step_runner",
    "warrior_smooth_10s_pullback_continuation",
    "warrior_smooth_hod_reclaim",
    "warrior_post_target_pullback_reclaim",
    "warrior_failed_spike_vwap_reclaim",
}


def is_warrior_entry_trigger(value: object) -> bool:
    return str(value or "") in WARRIOR_ENTRY_TRIGGERS


def is_warrior_initial_starter_trigger(value: object) -> bool:
    return str(value or "") in WARRIOR_INITIAL_STARTER_TRIGGERS


def is_warrior_high_base_confirm_trigger(value: object) -> bool:
    return str(value or "") in WARRIOR_HIGH_BASE_CONFIRM_TRIGGERS


def is_warrior_fresh_reclaim_trigger(value: object) -> bool:
    return str(value or "") in WARRIOR_FRESH_RECLAIM_TRIGGERS


def warrior_variant_for_entry_trigger(value: object) -> str:
    trigger = str(value or "")
    if trigger == "warrior_level_pullaway":
        return "warrior_level_pullaway_starter"
    if trigger == "warrior_curl_reclaim":
        return "warrior_curl_reclaim_starter"
    if trigger in WARRIOR_ENTRY_TRIGGERS:
        return trigger
    return "warrior_reclaim_starter"


def warrior_recent_rejection_high(history: Sequence[Bar]) -> float:
    """Return a fresh rejection high that must be reclaimed before entry.

    EHGO showed the failure mode this protects against: a hard first spike,
    a heavy red dump, then a top-wick rejection near HOD. Buying the next small
    green 10s candle while it is still below that rejection high is not a
    Warrior continuation; it is chop under supply.
    """
    bars = [bar for bar in history if float(bar.close or 0.0) > 0][-6:]
    if len(bars) < 3:
        return 0.0

    rejection_high = 0.0
    saw_dump = False
    for bar in bars[:-1]:
        high = float(bar.high or 0.0)
        low = float(bar.low or 0.0)
        open_ = float(bar.open or 0.0)
        close = float(bar.close or 0.0)
        volume = float(bar.volume or 0.0)
        if close <= 0 or high <= low:
            continue
        rng = high - low
        range_pct = rng / close
        body_pct = abs(close - open_) / close
        close_loc = (close - low) / rng

        is_heavy_red_dump = (
            close < open_
            and body_pct >= 0.075
            and range_pct >= 0.095
            and close_loc <= 0.18
            and volume >= 20_000.0
        )
        if is_heavy_red_dump:
            saw_dump = True
            rejection_high = max(rejection_high, high)
            continue

        is_top_wick_rejection = (
            saw_dump
            and close_loc <= 0.18
            and range_pct >= 0.055
            and high >= rejection_high * 1.04
            and volume >= 30_000.0
        )
        if is_top_wick_rejection:
            rejection_high = max(rejection_high, high)

    latest = bars[-1]
    latest_high = float(latest.high or 0.0)
    latest_low = float(latest.low or 0.0)
    latest_close = float(latest.close or 0.0)
    if rejection_high <= 0 or latest_close <= 0:
        return 0.0
    latest_rng = latest_high - latest_low
    latest_close_loc = (
        (latest_close - latest_low) / latest_rng
        if latest_rng > 0 else 0.0
    )
    reclaimed = (
        latest_high >= rejection_high * 1.002
        and latest_close >= rejection_high * 0.995
        and latest_close_loc >= 0.55
    )
    return 0.0 if reclaimed else rejection_high


def classify_warrior_trend_lane(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> str:
    """Pick exactly one trend/pullback Warrior lane for the current 10s bar.

    The individual lane functions still own the detailed safety checks. This
    classifier is intentionally coarse: it prevents a setup from silently falling
    through several unrelated lane families and makes the playbook easier to
    reason about when we tune symbols such as CLWT, BJDX, SPRC, or LABT.
    """
    bars = [bar for bar in history if float(bar.close or 0.0) > 0][-30:]
    if len(bars) < 6:
        return ""
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    if close <= open_ or close <= 0 or high <= 0 or low <= 0:
        return ""

    prior = bars[:-1]
    prior_high = max((float(bar.high or 0.0) for bar in prior), default=0.0)
    local_low = min((float(bar.low or 0.0) for bar in bars), default=0.0)
    runner_high = max(float(window_high or 0.0), prior_high, high)
    if prior_high <= 0 or local_low <= 0 or runner_high <= 0:
        return ""

    local_move = runner_high / local_low - 1.0
    near_hod = close >= runner_high * 0.86
    tight_near_hod = close >= runner_high * 0.94
    price = close

    if price >= 3.25 and high >= 4.0 and local_move >= 0.45 and tight_near_hod:
        return "warrior_first_impulse_scalp"
    if price >= 8.0 and local_move >= 0.08 and tight_near_hod:
        return "warrior_stair_step_runner"
    if 1.50 <= price <= 3.50 and local_move >= 0.25 and near_hod:
        return "warrior_first_pullback_reclaim"
    if price >= 7.0 and tight_near_hod:
        return "warrior_high_base_reclaim"
    if 3.50 <= price <= 12.50 and local_move >= 0.45 and near_hod:
        return "warrior_smooth_10s_pullback_continuation"
    if 3.0 <= price <= 7.25 and near_hod:
        return "warrior_smooth_hod_reclaim"
    if price >= 4.0:
        return "warrior_trend_pullback_reclaim"
    return ""


def warrior_trend_playbook_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    """Dispatch a trend/pullback Warrior setup through compatible lanes.

    The classifier picks the primary lane first, but some fast runners can look
    like one lane before the detailed checks prove a different, compatible lane.
    Keep cheap first-pullback isolated; allow higher-price runner lanes to fall
    through only within the same continuation family.
    """
    bars = [bar for bar in history if float(bar.close or 0.0) > 0][-30:]
    failed_spike_reclaim = warrior_failed_spike_vwap_reclaim_context(
        latest_bar,
        history=bars,
        window_high=window_high,
    )
    if failed_spike_reclaim is not None:
        return failed_spike_reclaim
    if warrior_recent_rejection_high(bars) > 0:
        return None
    lane = classify_warrior_trend_lane(
        latest_bar,
        history=bars,
        window_high=window_high,
    )
    detectors = {
        "warrior_high_base_reclaim": warrior_high_base_reclaim_context,
        "warrior_stair_step_runner": warrior_stair_step_runner_context,
        "warrior_smooth_10s_pullback_continuation": (
            warrior_smooth_10s_pullback_continuation_context
        ),
        "warrior_smooth_hod_reclaim": warrior_smooth_hod_reclaim_context,
        "warrior_failed_spike_vwap_reclaim": (
            warrior_failed_spike_vwap_reclaim_context
        ),
        "warrior_first_impulse_scalp": warrior_first_impulse_scalp_context,
        "warrior_first_pullback_reclaim": warrior_first_pullback_reclaim_context,
        "warrior_trend_pullback_reclaim": warrior_trend_pullback_reclaim_context,
    }
    fallback_order = {
        "warrior_first_pullback_reclaim": ("warrior_first_pullback_reclaim",),
        "warrior_first_impulse_scalp": (
            "warrior_first_impulse_scalp",
            "warrior_smooth_hod_reclaim",
            "warrior_stair_step_runner",
            "warrior_high_base_reclaim",
            "warrior_trend_pullback_reclaim",
        ),
        "warrior_high_base_reclaim": (
            "warrior_high_base_reclaim",
            "warrior_stair_step_runner",
            "warrior_trend_pullback_reclaim",
        ),
        "warrior_stair_step_runner": (
            "warrior_stair_step_runner",
            "warrior_smooth_10s_pullback_continuation",
            "warrior_high_base_reclaim",
            "warrior_trend_pullback_reclaim",
        ),
        "warrior_smooth_10s_pullback_continuation": (
            "warrior_smooth_10s_pullback_continuation",
            "warrior_smooth_hod_reclaim",
            "warrior_trend_pullback_reclaim",
        ),
        "warrior_smooth_hod_reclaim": (
            "warrior_smooth_hod_reclaim",
            "warrior_failed_spike_vwap_reclaim",
            "warrior_smooth_10s_pullback_continuation",
            "warrior_trend_pullback_reclaim",
        ),
        "warrior_trend_pullback_reclaim": ("warrior_trend_pullback_reclaim",),
    }
    for candidate in fallback_order.get(lane, ()):
        detector = detectors[candidate]
        context = detector(latest_bar, history=bars, window_high=window_high)
        if context is not None:
            return context
    return None


def warrior_late_reentry_reject(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
    reentry_count: int,
    target_wins: int,
    entry_trigger: Optional[str],
) -> Optional[str]:
    """Block late Warrior adds when the 10s tape has turned distributive.

    The first Warrior entry is allowed to be aggressive. After the stock has
    already paid or the bot has already taken multiple shots, new entries need
    cleaner tape so the playbook does not keep buying every violent HOD pop.
    """
    if int(target_wins or 0) < 1 and int(reentry_count or 0) < 2:
        return None
    if entry_trigger != "warrior_stair_step_runner":
        return None

    bars = [bar for bar in history if float(bar.close or 0.0) > 0]
    current_ts = getattr(latest_bar, "ts", None)
    if current_ts is not None:
        bars = [
            bar for bar in bars
            if getattr(bar, "ts", None) is None or bar.ts <= current_ts
        ]
    if not bars or bars[-1] is not latest_bar:
        bars = bars + [latest_bar]
    recent = bars[-8:]
    if len(recent) < 4:
        return None

    close = float(latest_bar.close or 0.0)
    high = float(latest_bar.high or 0.0)
    if close <= 0 or high <= 0:
        return "late Warrior re-entry blocked: invalid 10s price"

    ranges = sorted(
        (float(bar.high or 0.0) - float(bar.low or 0.0)) / float(bar.close or 1.0)
        for bar in recent[-6:]
        if float(bar.close or 0.0) > 0
    )
    if ranges:
        median_range = ranges[len(ranges) // 2]
        if int(reentry_count or 0) >= 2 and median_range > 0.038:
            return (
                "late Warrior re-entry blocked: 10s tape too wide/choppy "
                "({:.1f}% median range)"
            ).format(median_range * 100.0)

    prior = recent[:-1]
    prior_volumes = [float(bar.volume or 0.0) for bar in prior[-5:]]
    avg_prior_volume = (
        sum(prior_volumes) / len(prior_volumes) if prior_volumes else 0.0
    )
    for bar in prior[-5:]:
        bar_close = float(bar.close or 0.0)
        bar_open = float(bar.open or 0.0)
        if bar_close <= 0 or bar_close >= bar_open:
            continue
        bar_range = float(bar.high or 0.0) - float(bar.low or 0.0)
        body_pct = (bar_open - bar_close) / bar_close
        range_pct = bar_range / bar_close if bar_close > 0 else 0.0
        close_location = (
            (bar_close - float(bar.low or 0.0)) / bar_range
            if bar_range > 0 else 0.0
        )
        volume = float(bar.volume or 0.0)
        if (
            body_pct >= 0.030
            and range_pct >= 0.040
            and close_location <= 0.45
            and volume >= max(30_000.0, avg_prior_volume * 0.70)
        ):
            return "late Warrior re-entry blocked: recent red distribution candle"

    # After a target win, do not buy a lower bounce unless a dedicated lane
    # has proven a fresh reclaim. This catches CAST-style attempts that are
    # still below the old high after the stock has started topping.
    prior_high = max((float(bar.high or 0.0) for bar in bars[:-1]), default=0.0)
    reference_high = max(float(window_high or 0.0), prior_high)
    if (
        int(target_wins or 0) >= 1
        and reference_high > 0
        and high < reference_high * 0.985
        and close < reference_high * 0.965
        and entry_trigger not in {
            "warrior_second_leg_reclaim",
            "warrior_prior_runner_continuation_pullback",
        }
    ):
        return (
            "late Warrior re-entry blocked: bounce below prior high "
            "({:.2f} < {:.2f})"
        ).format(close, reference_high * 0.965)

    return None


def warrior_violent_liquid_reject(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    target_wins: int,
    entry_trigger: Optional[str],
) -> Optional[str]:
    """Require extra proof before trading violent-liquid Warrior pullbacks.

    Violent liquidity is not automatically good liquidity. EDHL-style tapes can
    print huge volume and still fail every reclaim, so the first Warrior shot in
    this lane needs a cleaner micro-base than normal Warrior setups.
    """
    if int(target_wins or 0) >= 1:
        return None
    if entry_trigger not in {
        "warrior_trend_pullback_reclaim",
        "warrior_stair_step_runner",
    }:
        return None

    bars = [bar for bar in history if float(bar.close or 0.0) > 0]
    current_ts = getattr(latest_bar, "ts", None)
    if current_ts is not None:
        bars = [
            bar for bar in bars
            if getattr(bar, "ts", None) is None or bar.ts <= current_ts
        ]
    if not bars or bars[-1] is not latest_bar:
        bars = bars + [latest_bar]
    recent = bars[-8:]
    if len(recent) < 5:
        return "violent-liquid Warrior blocked: not enough clean 10s proof"

    close = float(latest_bar.close or 0.0)
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    if close <= 0 or high <= 0 or close <= open_:
        return "violent-liquid Warrior blocked: confirm bar not green"

    rng = high - low
    close_location = (close - low) / rng if rng > 0 else 0.0
    if close_location < 0.72:
        return (
            "violent-liquid Warrior blocked: confirm close not strong "
            "({:.0%} of range)"
        ).format(close_location)

    ranges = sorted(
        (float(bar.high or 0.0) - float(bar.low or 0.0)) / float(bar.close or 1.0)
        for bar in recent[-6:]
        if float(bar.close or 0.0) > 0
    )
    if not ranges:
        return "violent-liquid Warrior blocked: no range proof"
    median_range = ranges[len(ranges) // 2]
    if median_range > 0.030:
        return (
            "violent-liquid Warrior blocked: 10s tape too wide "
            "({:.1f}% median range)"
        ).format(median_range * 100.0)

    prior = recent[:-1]
    prior_volumes = [float(bar.volume or 0.0) for bar in prior[-5:]]
    avg_prior_volume = (
        sum(prior_volumes) / len(prior_volumes) if prior_volumes else 0.0
    )
    for bar in prior[-5:]:
        bar_close = float(bar.close or 0.0)
        bar_open = float(bar.open or 0.0)
        if bar_close <= 0 or bar_close >= bar_open:
            continue
        bar_range = float(bar.high or 0.0) - float(bar.low or 0.0)
        body_pct = (bar_open - bar_close) / bar_close
        range_pct = bar_range / bar_close if bar_close > 0 else 0.0
        volume = float(bar.volume or 0.0)
        if (
            body_pct >= 0.025
            and range_pct >= 0.035
            and volume >= max(30_000.0, avg_prior_volume * 0.75)
        ):
            return "violent-liquid Warrior blocked: recent red distribution"

    prior_high = max((float(bar.high or 0.0) for bar in prior[-5:]), default=0.0)
    if prior_high > 0 and high < prior_high * 1.002 and close < prior_high:
        return "violent-liquid Warrior blocked: no fresh 10s reclaim"

    return None


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


def recent_wide_red_fakeout(history: Sequence[Bar]) -> bool:
    bars = [bar for bar in history if float(bar.close or 0.0) > 0]
    if len(bars) < 4:
        return False
    prior = bars[:-1]
    start = max(0, len(prior) - 4)
    for idx in range(start, len(prior)):
        bar = prior[idx]
        bar_close = float(bar.close or 0.0)
        bar_open = float(bar.open or 0.0)
        if bar_close <= 0 or bar_close >= bar_open:
            continue
        preceding = prior[max(0, idx - 3):idx]
        preceding_volumes = [float(prev.volume or 0.0) for prev in preceding]
        avg_prior_volume = (
            sum(preceding_volumes) / len(preceding_volumes)
            if preceding_volumes else 0.0
        )
        bar_range = float(bar.high or 0.0) - float(bar.low or 0.0)
        if bar_range <= 0:
            continue
        body_pct = (bar_open - bar_close) / bar_close
        range_pct = bar_range / bar_close
        close_loc = (bar_close - float(bar.low or 0.0)) / bar_range
        volume_gate = max(1.0, avg_prior_volume * 1.20)
        if (
            body_pct > 0.045
            and range_pct > 0.12
            and close_loc < 0.35
            and float(bar.volume or 0.0) >= volume_gate
        ):
            return True
    return False


def warrior_failed_burst_recovery_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    failed_high: float,
) -> Optional[Dict[str, Any]]:
    """Allow one reduced Warrior retry only after a stopped burst proves strength again."""
    failed_high = float(failed_high or 0.0)
    if failed_high <= 0:
        return None
    open_ = float(latest_bar.open or 0.0)
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if open_ <= 0 or high <= 0 or close <= 0:
        return None
    if close <= open_:
        return None
    rng = high - low
    if rng <= 0:
        return None
    close_location = (close - low) / rng
    if close_location < 0.64:
        return None
    recovery_level = failed_high * 1.035
    if high < recovery_level or close < failed_high * 1.01:
        return None
    history = [bar for bar in history if float(bar.close or 0.0) > 0]
    if len(history) < 6:
        return None
    if recent_wide_red_fakeout(history[-6:]):
        return None
    prior = history[-7:-1]
    avg_prior_volume = (
        sum(float(bar.volume or 0.0) for bar in prior) / len(prior)
        if prior else 0.0
    )
    if volume < max(90_000.0, avg_prior_volume * 0.85):
        return None
    recent_lows = [float(bar.low or 0.0) for bar in history[-5:-1] if float(bar.low or 0.0) > 0]
    structural_low = min(recent_lows, default=low)
    entry_price = close
    max_pay = round(recovery_level * 1.035, 4)
    if low > max_pay:
        return None
    entry_price = min(entry_price, max_pay)
    raw_risk = max(entry_price - structural_low, entry_price * 0.018, 0.08)
    risk = min(raw_risk, entry_price * 0.055)
    if risk <= 0 or risk > entry_price * 0.06:
        return None
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.35, 4)
    return {
        "entry_trigger": "warrior_failed_burst_recovery",
        "variant_override": "warrior_failed_burst_recovery",
        "psych_level": round(recovery_level, 4),
        "pullaway_level": round(recovery_level, 4),
        "failed_high": round(failed_high, 4),
        "max_pay": max_pay,
        "entry_price_override": round(entry_price, 4),
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "max_hold_seconds_override": 90.0,
        "reward_risk": 1.35,
        "rr_note_override": (
            "warrior failed-burst recovery fresh high=${:.2f} "
            "failed_high=${:.2f} risk={:.1f}% target=${:.2f}"
        ).format(
            recovery_level,
            failed_high,
            risk / entry_price * 100.0 if entry_price else 0.0,
            target_price,
        ),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_failed_burst_watch_reason(latest_bar: Bar) -> Optional[str]:
    """Mark a blocked giant Warrior spike as a recovery-watch candidate."""
    open_ = float(latest_bar.open or 0.0)
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if open_ <= 0 or high <= 0 or low <= 0 or close <= 0:
        return None
    if close <= open_:
        return None
    rng = high - low
    if rng <= 0:
        return None
    range_pct = rng / close
    close_location = (close - low) / rng
    if (
        close >= 4.0
        and range_pct >= 0.18
        and close_location >= 0.65
        and volume >= 100_000.0
    ):
        return (
            "blocked giant Warrior spike {:.1f}% range; "
            "watching for failed-burst recovery"
        ).format(range_pct * 100.0)
    return None


def warrior_level_break_starter_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
    min_reclaim_price: float,
) -> Optional[Dict[str, Any]]:
    """Starter for the first clean Warrior break through a major level.

    This catches the Warrior-style $3.50/$4.00 break early, but only after a
    strong 10s close and real volume. It is intentionally separate from the
    generic pending-confirm path, which can enter one 10s bar late.
    """
    open_ = float(latest_bar.open or 0.0)
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if open_ <= 0 or high <= 0 or close <= 0:
        return None
    if close <= open_:
        return None
    rng = high - low
    if rng <= 0:
        return None
    close_location = (close - low) / rng
    if close_location < 0.64:
        return None
    if volume < 100_000:
        return None
    step = 0.5 if close < 10.0 else 1.0
    min_level = float(min_reclaim_price or 0.0)
    level_reclaim = (
        min_level > 0
        and float(window_high or 0.0) <= min_level * 1.01
        and high >= min_level
    )
    if level_reclaim:
        crossed_level = min_level
    else:
        crossed_level = (int(float(window_high or 0.0) / step) + 1) * step
        crossed_level = max(crossed_level, min_level)
    if crossed_level <= 0:
        return None
    if not level_reclaim and float(window_high or 0.0) >= crossed_level:
        return None
    if high < crossed_level or close < crossed_level * 0.995:
        return None
    history = [bar for bar in history if float(bar.close or 0.0) > 0]
    if len(history) < 6:
        return None
    if recent_wide_red_fakeout(history[-6:]):
        return None
    prior = history[-7:-1]
    avg_prior_volume = (
        sum(float(bar.volume or 0.0) for bar in prior) / len(prior)
        if prior else 0.0
    )
    if volume < max(100_000.0, avg_prior_volume * 0.75):
        return None
    session_lows = [
        float(bar.low or 0.0)
        for bar in history
        if float(bar.low or 0.0) > 0
    ]
    session_low = min(session_lows, default=low)
    extended_from_session_low = (
        close / session_low - 1.0 if session_low > 0 else 0.0
    )
    if extended_from_session_low > 0.75 and volume < 300_000:
        return None
    recent_lows = [float(bar.low or 0.0) for bar in history[-5:-1] if float(bar.low or 0.0) > 0]
    structural_low = min(recent_lows, default=low)
    max_pay = round(crossed_level * 1.035, 4)
    if low > max_pay:
        return None
    entry_price = min(close, max_pay)
    raw_risk = max(entry_price - structural_low, entry_price * 0.018, 0.08)
    risk = min(raw_risk, entry_price * 0.055)
    if risk <= 0 or risk > entry_price * 0.06:
        return None
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.35, 4)
    return {
        "entry_trigger": "warrior_level_break_starter",
        "variant_override": "warrior_level_break_starter",
        "psych_level": round(crossed_level, 4),
        "pullaway_level": round(crossed_level, 4),
        "max_pay": max_pay,
        "entry_price_override": round(entry_price, 4),
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "max_hold_seconds_override": 90.0,
        "reward_risk": 1.35,
        "rr_note_override": (
            "warrior level-break starter level=${:.2f} "
            "cap=${:.2f} risk={:.1f}% target=${:.2f}"
        ).format(
            crossed_level,
            max_pay,
            risk / entry_price * 100.0 if entry_price else 0.0,
            target_price,
        ),
        "skip_unstable_confirm_stop_check": True,
    }


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
    if warrior_recent_rejection_high(history) > 0:
        return None
    prior_high = max(
        (float(bar.high or 0.0) for bar in history[:-1]),
        default=0.0,
    )
    if (
        reentry_count == 0
        and rejection_reason in {
            "high-volume shooting-star rejection",
            "first explosive 10s spike",
        }
        and prior_high > proof_level * 1.08
        and high >= prior_high * 0.998
        and close < prior_high * 0.995
    ):
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
    if (
        clwt_fast_pullaway
        and rejection_reason == "first explosive 10s spike"
        and recent_wide_red_fakeout(history)
    ):
        return None
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
    if reclaim_level >= 5.0:
        if range_pct >= 0.18:
            return None
        if range_pct < 0.15 and close_location < 0.72:
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


def warrior_first_impulse_scalp_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    """Detect NCT-style first impulse scalps.

    This is intentionally not a pullback/reclaim lane. It is for the first
    explosive momentum burst after a stock is no longer too cheap, with a
    reduced-size stop-limit style starter and a fast 1R target. The entry still
    waits for the existing next-10s confirmation path before execution.
    """
    history = [bar for bar in history if float(bar.close or 0.0) > 0][-16:]
    if len(history) < 6:
        return None

    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= 0 or high <= 0 or low <= 0 or close <= open_:
        return None
    if close < 3.25 or high < 4.0:
        return None

    prior = history[:-1]
    prior_high = max((float(bar.high or 0.0) for bar in prior[-8:]), default=0.0)
    prior_low = min((float(bar.low or 0.0) for bar in prior[-8:]), default=0.0)
    prior_close = float(prior[-1].close or 0.0) if prior else 0.0
    if prior_high <= 0 or prior_low <= 0 or prior_close <= 0:
        return None
    if float(window_high or 0.0) > 0 and high <= float(window_high or 0.0) * 1.003:
        return None

    impulse_from_prior = (high - prior_high) / prior_high
    close_thrust = (close - prior_close) / prior_close
    day_move = (close - prior_low) / prior_low
    first_spike = impulse_from_prior >= 0.12 and close_thrust >= 0.10 and day_move >= 0.35
    second_push = impulse_from_prior >= 0.05 and close_thrust >= 0.02 and day_move >= 0.65
    if not (first_spike or second_push):
        return None

    rng = high - low
    if rng <= 0:
        return None
    range_pct = rng / close
    close_location = (close - low) / rng
    min_close_location = 0.62 if first_spike else 0.42
    if range_pct > 0.32 or close_location < min_close_location:
        return None

    prior_volumes = [float(bar.volume or 0.0) for bar in prior[-6:]]
    avg_prior_volume = sum(prior_volumes) / len(prior_volumes) if prior_volumes else 0.0
    recent_volume = sum(float(bar.volume or 0.0) for bar in history[-3:])
    volume_floor = avg_prior_volume * (2.2 if first_spike else 1.1)
    recent_floor = avg_prior_volume * (3.0 if first_spike else 3.0)
    if volume < max(85_000.0, volume_floor):
        return None
    if recent_volume < max(175_000.0, recent_floor):
        return None

    # Do not arm if the stock already printed a heavy red rejection before the
    # impulse; that is the cheap/choppy warning the Warrior playbook avoids.
    # Look beyond the immediate pause so a dump-then-ramp sequence cannot sneak
    # in just because the last few bars curled.
    for bar in prior[-6:]:
        bar_close = float(bar.close or 0.0)
        bar_open = float(bar.open or 0.0)
        if bar_close <= 0 or bar_close >= bar_open:
            continue
        bar_range = float(bar.high or 0.0) - float(bar.low or 0.0)
        body_pct = (bar_open - bar_close) / bar_close
        bar_range_pct = bar_range / bar_close if bar_close > 0 else 0.0
        close_loc = (
            (bar_close - float(bar.low or 0.0)) / bar_range
            if bar_range > 0 else 0.0
        )
        volume_gate = max(20_000.0, avg_prior_volume * 0.45)
        if body_pct > 0.055 and close_loc < 0.45 and float(bar.volume or 0.0) >= volume_gate:
            return None
        if (
            body_pct > 0.045
            and bar_range_pct > 0.12
            and close_loc < 0.35
            and float(bar.volume or 0.0) >= avg_prior_volume * 1.20
        ):
            return None

    entry_price = round(close, 4)
    risk = max(entry_price * 0.026, 0.12)
    # If the impulse bar itself is too stretched for a tactical starter, skip.
    if risk > entry_price * 0.04:
        return None
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk, 4)
    return {
        "entry_trigger": "warrior_first_impulse_scalp",
        "variant_override": "warrior_first_impulse_scalp",
        "psych_level": round(max(prior_high, entry_price), 4),
        "pullaway_level": round(prior_high, 4),
        "impulse_high": round(high, 4),
        "impulse_from_prior_pct": round(impulse_from_prior * 100.0, 2),
        "day_move_pct": round(day_move * 100.0, 2),
        "max_pay": round(entry_price * 1.03, 4),
        "entry_price_override": entry_price,
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "max_hold_seconds_override": 45.0,
        "reward_risk": 1.0,
        "size_factor": 0.20,
        "rr_note_override": (
            "warrior first-impulse scalp prior_high=${:.2f} "
            "impulse={:.1f}% cap=${:.2f} target=${:.2f}"
        ).format(prior_high, impulse_from_prior * 100.0, entry_price * 1.03, target_price),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_first_pullback_reclaim_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    """Detect the first controlled pullback/reclaim after a cheap-name burst.

    This is not a "cheap stock" strategy. It is the Warrior first-pullback
    pattern with stricter risk/liquidity guardrails for $2.50-$3.50 names:
    ignore the first spike, require a real burst, then buy only when the first
    10s base reclaims on returning volume while still holding VWAP/EMA support.
    """
    history = [bar for bar in history if float(bar.close or 0.0) > 0][-24:]
    if len(history) < 10:
        return None

    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= 0 or high <= 0 or low <= 0 or close <= open_:
        return None
    if close < 1.50 or close > 3.50:
        return None

    prior = history[:-1]
    prior_high = max((float(bar.high or 0.0) for bar in prior), default=0.0)
    day_low = min((float(bar.low or 0.0) for bar in history), default=0.0)
    burst_high = max(float(window_high or 0.0), prior_high, high)
    if prior_high <= 0 or day_low <= 0 or burst_high <= 0:
        return None
    # Keep cheap first legs out of this lane. In the Warrior playbook these
    # names can stay on watch, but sub-$2.50 first reclaims are still too close
    # to the low-priced/choppy zone that produces false pops.
    if close < 2.50:
        return None

    burst_move = burst_high / day_low - 1.0
    if burst_move < 0.35:
        return None

    base_bars = history[-7:-1]
    if len(base_bars) < 5:
        return None
    base_high = max(float(bar.high or 0.0) for bar in base_bars)
    base_low = min(float(bar.low or 0.0) for bar in base_bars)
    if base_high <= 0 or base_low <= 0:
        return None
    base_high_index = max(
        range(len(base_bars)),
        key=lambda idx: float(base_bars[idx].high or 0.0),
    )
    base_low_index = min(
        range(len(base_bars)),
        key=lambda idx: float(base_bars[idx].low or 0.0),
    )
    # This lane is specifically a pullback reclaim. If the base low came before
    # the base high, the tape is still stair-stepping up and the latest bar is
    # an extension buy, not a reclaim after a controlled dip.
    if base_low_index <= base_high_index:
        return None

    pullback_pct = (burst_high - base_low) / burst_high
    if pullback_pct < 0.045 or pullback_pct > 0.32:
        return None
    base_range_pct = (base_high - base_low) / close
    if base_range_pct > 0.22:
        return None
    if high < base_high or close < base_high * 0.995:
        return None

    rng = high - low
    if rng <= 0:
        return None
    close_location = (close - low) / rng
    range_pct = rng / close
    if close_location < 0.58 or range_pct > 0.16:
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
    if close < max(vwap * 0.99, ema * 0.985):
        return None
    if base_low < vwap * 0.88:
        return None

    avg_base_volume = sum(float(bar.volume or 0.0) for bar in base_bars) / len(base_bars)
    recent_volume = sum(float(bar.volume or 0.0) for bar in history[-3:])
    if volume < max(55_000.0, avg_base_volume * 0.75):
        return None
    if recent_volume < max(150_000.0, avg_base_volume * 1.25):
        return None

    heavy_red = any(
        float(bar.close or 0.0) < float(bar.open or 0.0)
        and (float(bar.open or 0.0) - float(bar.close or 0.0)) / float(bar.close or 1.0) > 0.065
        and float(bar.volume or 0.0) >= avg_base_volume * 1.25
        for bar in history[-6:-1]
    )
    if heavy_red:
        return None

    reclaim_level = base_high
    max_pay = round(reclaim_level * 1.025, 4)
    if low > max_pay:
        return None
    entry_price = round(min(close, max_pay), 4)
    stop_anchor = max(base_low, vwap * 0.94, ema * 0.94)
    risk = max(entry_price - (stop_anchor - max(0.025, entry_price * 0.006)), entry_price * 0.022, 0.07)
    risk = min(risk, entry_price * 0.058)
    if risk <= 0:
        return None
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.35, 4)
    return {
        "entry_trigger": "warrior_first_pullback_reclaim",
        "variant_override": "warrior_first_pullback_reclaim",
        "psych_level": round(reclaim_level, 4),
        "pullaway_level": round(reclaim_level, 4),
        "base_high": round(base_high, 4),
        "base_low": round(base_low, 4),
        "burst_high": round(burst_high, 4),
        "burst_move_pct": round(burst_move * 100.0, 2),
        "pullback_pct": round(pullback_pct * 100.0, 2),
        "vwap": round(vwap, 4),
        "ema9": round(ema, 4),
        "max_pay": max_pay,
        "entry_price_override": entry_price,
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "max_hold_seconds_override": 90.0,
        "reward_risk": 1.35,
        "size_factor": 0.20,
        "rr_note_override": (
            "warrior first pullback reclaim base=${:.2f}-${:.2f} "
            "burst={:.1f}% pullback={:.1f}% cap=${:.2f} target=${:.2f}"
        ).format(base_low, base_high, burst_move * 100.0, pullback_pct * 100.0, max_pay, target_price),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_high_base_reclaim_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    """Detect SPRC-style violent high-base reclaims.

    This lane is for a stock that already made a large impulse, then held a
    high, choppy base near HOD and reclaimed the base. It is intentionally more
    permissive on base width than the stair-step lane, but compensates with
    stronger volume/close requirements and reduced size.
    """
    history = [bar for bar in history if float(bar.close or 0.0) > 0][-30:]
    if len(history) < 12:
        return None
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= open_ or close <= 0 or high <= 0 or low <= 0:
        return None

    prior = history[:-1]
    prior_high = max((float(bar.high or 0.0) for bar in prior), default=0.0)
    runner_high = max(float(window_high or 0.0), prior_high, high)
    if runner_high < max(7.0, close * 0.94):
        return None
    prior_high_index = max(
        range(len(prior)),
        key=lambda idx: float(prior[idx].high or 0.0),
        default=-1,
    )
    bars_since_prior_high = (
        len(prior) - 1 - prior_high_index if prior_high_index >= 0 else 0
    )
    # Do not treat the first vertical expansion as a high-base reclaim. This
    # lane is for the later SPRC-style base after the first top is established.
    if bars_since_prior_high < 18:
        return None

    lookback_low = min((float(bar.low or 0.0) for bar in history[:-2]), default=0.0)
    if lookback_low <= 0 or runner_high < lookback_low * 1.18:
        return None

    base_bars = history[-10:-1]
    if len(base_bars) < 7:
        return None
    base_high = max(float(bar.high or 0.0) for bar in base_bars)
    base_low = min(float(bar.low or 0.0) for bar in base_bars)
    if base_high <= 0 or base_low <= 0:
        return None

    pullback_pct = (runner_high - base_low) / runner_high if runner_high > 0 else 0.0
    if pullback_pct < 0.06 or pullback_pct > 0.24:
        return None
    base_range_pct = (base_high - base_low) / close if close > 0 else 999.0
    if base_range_pct < 0.035 or base_range_pct > 0.23:
        return None
    # Keep this lane for high bases, not deep washouts.
    if base_low < runner_high * 0.72:
        return None
    if high < base_high or close < base_high * 0.985:
        return None

    rng = max(high - low, 0.0)
    close_location = (close - low) / rng if rng > 0 else 0.0
    if close_location < 0.62:
        return None

    total_volume = sum(float(bar.volume or 0.0) for bar in history)
    if total_volume <= 0:
        return None
    vwap = sum(
        ((float(bar.high or 0.0) + float(bar.low or 0.0) + float(bar.close or 0.0)) / 3.0)
        * float(bar.volume or 0.0)
        for bar in history
    ) / total_volume
    if close < vwap * 0.99 or base_low < vwap * 0.86:
        return None

    avg_base_volume = sum(float(bar.volume or 0.0) for bar in base_bars) / len(base_bars)
    recent_volume = sum(float(bar.volume or 0.0) for bar in history[-3:])
    # EDHL-style violent tapes often print several large candles in a row and
    # then rug the reclaim. SPRC's good setup was different: a high-base reclaim
    # on a fresh volume expansion, not a second/third chase candle. Require the
    # reclaim bar to stand out versus the base and avoid buying immediately
    # after another blow-off 10s body.
    prev_bar = history[-2] if len(history) >= 2 else None
    if prev_bar is not None:
        prev_close = float(prev_bar.close or 0.0)
        prev_open = float(prev_bar.open or 0.0)
        if prev_close > 0 and prev_close > prev_open:
            prev_body_pct = (prev_close - prev_open) / prev_close
            if prev_body_pct > 0.06:
                return None
    if close_location < 0.72:
        return None
    if volume < max(140_000.0, avg_base_volume * 2.5):
        return None
    if recent_volume < max(300_000.0, avg_base_volume * 1.65):
        return None

    heavy_red_count = 0
    for bar in history[-6:-1]:
        bar_close = float(bar.close or 0.0)
        bar_open = float(bar.open or 0.0)
        if bar_close <= 0 or bar_close >= bar_open:
            continue
        body_pct = (bar_open - bar_close) / bar_close
        bar_range = float(bar.high or 0.0) - float(bar.low or 0.0)
        close_loc = (
            (bar_close - float(bar.low or 0.0)) / bar_range
            if bar_range > 0 else 0.0
        )
        if (
            body_pct > 0.065
            and close_loc < 0.45
            and float(bar.volume or 0.0) >= avg_base_volume * 1.15
        ):
            heavy_red_count += 1
    if heavy_red_count >= 2:
        return None

    reclaim_level = base_high
    max_pay = round(reclaim_level * 1.045, 4)
    if low > max_pay:
        return None
    entry_price = min(close, max_pay)
    stop_anchor = max(base_low, vwap * 0.955)
    structural_stop = max(0.01, stop_anchor - max(0.05, entry_price * 0.005))
    risk = max(entry_price - structural_stop, entry_price * 0.020, 0.12)
    # This lane is reduced-size and uses a capped tactical stop so it can pass
    # the shared final guard without pretending the full base risk is small.
    risk = min(risk, entry_price * 0.059)
    if risk <= 0:
        return None
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.0, 4)
    return {
        "entry_trigger": "warrior_high_base_reclaim",
        "variant_override": "warrior_high_base_reclaim",
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
        "max_hold_seconds_override": 150.0,
        "reward_risk": 1.0,
        "size_factor": 0.25,
        "rr_note_override": (
            "warrior high-base reclaim base=${:.2f}-${:.2f} "
            "pullback={:.1f}% vwap=${:.2f} cap=${:.2f} target=${:.2f}"
        ).format(base_low, base_high, pullback_pct * 100.0, vwap, max_pay, target_price),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_stair_step_runner_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    """Detect STI-style stair-step runners after the first impulse.

    This is intentionally separate from the CLWT/UTSI pull-away lanes. It looks
    for a strong runner that pauses in a tight higher-low base, then reclaims
    that base instead of buying the first spike candle.
    """
    history = [bar for bar in history if float(bar.close or 0.0) > 0][-30:]
    if len(history) < 12:
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
    runner_high = max(float(window_high or 0.0), prior_high)
    if runner_high < max(8.0, close * 0.96):
        return None
    lookback_low = min((float(bar.low or 0.0) for bar in history[:-2]), default=0.0)
    if lookback_low <= 0 or runner_high < lookback_low * 1.08:
        return None

    base_bars = history[-8:-1]
    if len(base_bars) < 6:
        return None
    base_high = max(float(bar.high or 0.0) for bar in base_bars)
    base_low = min(float(bar.low or 0.0) for bar in base_bars)
    if base_high <= 0 or base_low <= 0:
        return None
    pullback_pct = (runner_high - base_low) / runner_high if runner_high > 0 else 0.0
    if pullback_pct < 0.025 or pullback_pct > 0.16:
        return None
    base_range_pct = (base_high - base_low) / close if close > 0 else 999.0
    if base_range_pct > 0.105:
        return None
    if high < base_high or close < base_high * 0.995:
        return None

    rng = max(high - low, 0.0)
    close_location = (close - low) / rng if rng > 0 else 0.0
    if close_location < 0.54:
        return None

    total_volume = sum(float(bar.volume or 0.0) for bar in history)
    if total_volume <= 0:
        return None
    vwap = sum(
        ((float(bar.high or 0.0) + float(bar.low or 0.0) + float(bar.close or 0.0)) / 3.0)
        * float(bar.volume or 0.0)
        for bar in history
    ) / total_volume
    if close < vwap * 1.015 or base_low < vwap * 0.94:
        return None

    avg_base_volume = sum(float(bar.volume or 0.0) for bar in base_bars) / len(base_bars)
    recent_volume = sum(float(bar.volume or 0.0) for bar in history[-3:])
    if volume < max(30_000.0, avg_base_volume * 0.65):
        return None
    if recent_volume < max(100_000.0, avg_base_volume * 1.15):
        return None

    heavy_red = any(
        float(bar.close or 0.0) < float(bar.open or 0.0)
        and (float(bar.open or 0.0) - float(bar.close or 0.0)) / float(bar.close or 1.0) > 0.065
        and float(bar.volume or 0.0) >= avg_base_volume * 1.35
        for bar in history[-5:-1]
    )
    if heavy_red:
        return None

    reclaim_level = base_high
    max_pay = round(reclaim_level * 1.035, 4)
    if low > max_pay:
        return None
    entry_price = min(close, max_pay)
    stop_anchor = max(base_low, vwap * 0.965)
    structural_risk = entry_price - max(0.01, stop_anchor - max(0.03, entry_price * 0.0035))
    risk = max(structural_risk, entry_price * 0.018, 0.10)
    # Keep this lane inside the shared final-guard risk cap while preserving a
    # real stop outside normal 10s noise.
    risk = min(risk, entry_price * 0.059)
    if risk <= 0:
        return None
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.8, 4)
    return {
        "entry_trigger": "warrior_stair_step_runner",
        "variant_override": "warrior_stair_step_runner",
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
        "reward_risk": 1.8,
        "rr_note_override": (
            "warrior stair-step runner base=${:.2f}-${:.2f} "
            "pullback={:.1f}% vwap=${:.2f} cap=${:.2f} target=${:.2f}"
        ).format(base_low, base_high, pullback_pct * 100.0, vwap, max_pay, target_price),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_smooth_10s_pullback_continuation_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    """Detect WNW-style smooth 10s pullback continuation.

    This lane is for the first controlled pullback after a violent premarket
    squeeze. It deliberately skips the first spike candle, waits for the stock
    to pull back and rebuild on 10s candles, then buys the reclaim before a new
    extension. It is not a generic "deep pullback" relaxer.
    """
    history = [bar for bar in history if float(bar.close or 0.0) > 0][-32:]
    if len(history) < 12:
        return None
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= open_ or close <= 0 or high <= 0 or low <= 0:
        return None

    prior = history[:-1]
    prior_high = max((float(bar.high or 0.0) for bar in prior), default=0.0)
    runner_high = max(float(window_high or 0.0), prior_high)
    day_low = min((float(bar.low or 0.0) for bar in history), default=0.0)
    if prior_high <= 0 or runner_high <= 0 or day_low <= 0:
        return None
    if close < 3.50 or close > 12.50:
        return None
    if runner_high < day_low * 1.45:
        return None

    prior_high_index = max(
        range(len(prior)),
        key=lambda idx: float(prior[idx].high or 0.0),
        default=-1,
    )
    bars_since_prior_high = (
        len(prior) - 1 - prior_high_index if prior_high_index >= 0 else 0
    )
    # Require the first spike to have already happened. This lane should not
    # buy the initial vertical candle.
    if bars_since_prior_high < 4:
        return None

    base_bars = history[-10:-1]
    if len(base_bars) < 7:
        return None
    base_high = max(float(bar.high or 0.0) for bar in base_bars)
    base_low = min(float(bar.low or 0.0) for bar in base_bars)
    if base_high <= 0 or base_low <= 0:
        return None
    pullback_pct = (runner_high - base_low) / runner_high
    # WNW-style can pull back deeper than high-base/stair-step, but it must not
    # be a full trend break.
    if pullback_pct < 0.08 or pullback_pct > 0.34:
        return None
    if base_low < day_low * 1.35:
        return None

    base_range_pct = (base_high - base_low) / close
    if base_range_pct > 0.18:
        return None
    if high < base_high or close < base_high * 0.985:
        return None

    rng = high - low
    if rng <= 0:
        return None
    close_location = (close - low) / rng
    range_pct = rng / close
    if close_location < 0.58 or range_pct > 0.105:
        return None

    total_volume = sum(float(bar.volume or 0.0) for bar in history)
    if total_volume <= 0:
        return None
    vwap = sum(
        ((float(bar.high or 0.0) + float(bar.low or 0.0) + float(bar.close or 0.0)) / 3.0)
        * float(bar.volume or 0.0)
        for bar in history
    ) / total_volume
    if close < vwap * 1.01 or base_low < vwap * 0.90:
        return None

    avg_base_volume = sum(float(bar.volume or 0.0) for bar in base_bars) / len(base_bars)
    recent_volume = sum(float(bar.volume or 0.0) for bar in history[-3:])
    if volume < max(22_000.0, avg_base_volume * 0.70):
        return None
    if recent_volume < max(75_000.0, avg_base_volume * 1.05):
        return None

    heavy_red = 0
    for bar in history[-6:-1]:
        bar_close = float(bar.close or 0.0)
        bar_open = float(bar.open or 0.0)
        if bar_close <= 0 or bar_close >= bar_open:
            continue
        bar_range = float(bar.high or 0.0) - float(bar.low or 0.0)
        if bar_range <= 0:
            continue
        body_pct = (bar_open - bar_close) / bar_close
        range_pct = bar_range / bar_close
        close_loc = (bar_close - float(bar.low or 0.0)) / bar_range
        if (
            body_pct >= 0.070
            and range_pct >= 0.095
            and close_loc <= 0.38
            and float(bar.volume or 0.0) >= avg_base_volume * 1.15
        ):
            heavy_red += 1
    if heavy_red >= 2:
        return None

    reclaim_level = base_high
    max_pay = round(reclaim_level * 1.032, 4)
    if low > max_pay:
        return None
    entry_price = min(close, max_pay)
    stop_anchor = max(base_low, vwap * 0.94)
    structural_stop = max(0.01, stop_anchor - max(0.04, entry_price * 0.004))
    risk = max(entry_price - structural_stop, entry_price * 0.018, 0.10)
    # Keep the tactical stop compatible with the shared final entry guard.
    risk = min(risk, entry_price * 0.058)
    if risk <= 0:
        return None
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.45, 4)
    return {
        "entry_trigger": "warrior_smooth_10s_pullback_continuation",
        "variant_override": "warrior_smooth_10s_pullback_continuation",
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
        "max_hold_seconds_override": 150.0,
        "reward_risk": 1.45,
        "size_factor": 0.30,
        "rr_note_override": (
            "warrior smooth 10s pullback continuation base=${:.2f}-${:.2f} "
            "pullback={:.1f}% vwap=${:.2f} cap=${:.2f} target=${:.2f}"
        ).format(base_low, base_high, pullback_pct * 100.0, vwap, max_pay, target_price),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_failed_spike_vwap_reclaim_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    """Detect NXTS-style failed spike, VWAP/base rebuild, then breakout.

    This is different from a first-spike scalp. It first requires evidence that
    an earlier vertical move failed, then waits for a controlled 10s base above
    VWAP/EMA-like support and buys only the reclaim through that fresh base.
    """
    history = [bar for bar in history if float(bar.close or 0.0) > 0][-42:]
    if len(history) < 16:
        return None
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= open_ or close <= 0 or high <= 0 or low <= 0:
        return None
    if close < 4.0 or close > 18.0:
        return None

    prior = history[:-1]
    if len(prior) < 12:
        return None
    earlier_high = max(float(bar.high or 0.0) for bar in prior)
    local_low = min(float(bar.low or 0.0) for bar in history)
    runner_high = max(float(window_high or 0.0), earlier_high, high)
    if earlier_high <= 0 or local_low <= 0 or runner_high < local_low * 1.45:
        return None

    failed_idx = -1
    failed_high = 0.0
    for idx, bar in enumerate(prior[:-4]):
        bar_high = float(bar.high or 0.0)
        bar_low = float(bar.low or 0.0)
        bar_open = float(bar.open or 0.0)
        bar_close = float(bar.close or 0.0)
        bar_volume = float(bar.volume or 0.0)
        if bar_close <= 0 or bar_high <= bar_low:
            continue
        bar_range = bar_high - bar_low
        close_loc = (bar_close - bar_low) / bar_range
        body_pct = abs(bar_close - bar_open) / bar_close
        range_pct = bar_range / bar_close
        dump_after = any(
            float(next_bar.close or 0.0) < float(next_bar.open or 0.0)
            and float(next_bar.close or 0.0) > 0
            and (
                float(next_bar.open or 0.0) - float(next_bar.close or 0.0)
            ) / float(next_bar.close or 1.0) >= 0.045
            and (
                float(next_bar.high or 0.0) - float(next_bar.low or 0.0)
            ) / float(next_bar.close or 1.0) >= 0.070
            and float(next_bar.volume or 0.0) >= max(20_000.0, bar_volume * 0.55)
            for next_bar in prior[idx + 1: min(len(prior), idx + 5)]
        )
        first_failed_spike = (
            bar_high >= local_low * 1.35
            and range_pct >= 0.065
            and (
                (bar_close < bar_open and close_loc <= 0.45)
                or body_pct >= 0.060
            )
            and bar_volume >= 25_000.0
            and dump_after
        )
        if first_failed_spike and bar_high >= failed_high:
            failed_idx = idx
            failed_high = bar_high
    if failed_idx < 0 or failed_high <= 0:
        return None

    bars_after_failure = history[failed_idx + 1:]
    if len(bars_after_failure) < 8:
        return None
    base_bars = history[-10:-1]
    if len(base_bars) < 7:
        return None
    base_high = max(float(bar.high or 0.0) for bar in base_bars)
    base_low = min(float(bar.low or 0.0) for bar in base_bars)
    if base_high <= 0 or base_low <= 0:
        return None
    base_range_pct = (base_high - base_low) / close
    if base_range_pct > 0.145:
        return None
    if high < base_high * 1.002 or close < base_high * 0.995:
        return None

    total_volume = sum(float(bar.volume or 0.0) for bar in history)
    if total_volume <= 0:
        return None
    vwap = sum(
        ((float(bar.high or 0.0) + float(bar.low or 0.0) + float(bar.close or 0.0)) / 3.0)
        * float(bar.volume or 0.0)
        for bar in history
    ) / total_volume
    ema_like = sum(float(bar.close or 0.0) for bar in history[-6:]) / 6.0
    if close < vwap * 1.005 or close < ema_like * 0.995:
        return None
    if base_low < vwap * 0.90:
        return None

    rng = high - low
    if rng <= 0:
        return None
    close_location = (close - low) / rng
    range_pct = rng / close
    if close_location < 0.62 or range_pct > 0.115:
        return None
    # CUPR-style failed-spike reclaims can look strong on volume but still be
    # one huge, unstable 10s candle. If the reclaim candle itself is this wide,
    # require a near-high close; otherwise wait for a tighter micro-base.
    if range_pct >= 0.09 and close_location < 0.80:
        return None

    avg_base_volume = sum(float(bar.volume or 0.0) for bar in base_bars) / len(base_bars)
    recent_volume = sum(float(bar.volume or 0.0) for bar in history[-3:])
    if volume < max(35_000.0, avg_base_volume * 1.10):
        return None
    if recent_volume < max(120_000.0, avg_base_volume * 2.25):
        return None

    heavy_red = 0
    for bar in history[-5:-1]:
        bar_close = float(bar.close or 0.0)
        bar_open = float(bar.open or 0.0)
        if bar_close <= 0 or bar_close >= bar_open:
            continue
        bar_range = float(bar.high or 0.0) - float(bar.low or 0.0)
        if bar_range <= 0:
            continue
        body_pct = (bar_open - bar_close) / bar_close
        bar_range_pct = bar_range / bar_close
        close_loc = (bar_close - float(bar.low or 0.0)) / bar_range
        if (
            body_pct >= 0.065
            and bar_range_pct >= 0.090
            and close_loc <= 0.35
            and float(bar.volume or 0.0) >= avg_base_volume * 1.15
        ):
            heavy_red += 1
    if heavy_red > 0:
        return None

    reclaim_level = base_high
    max_pay = round(reclaim_level * 1.035, 4)
    if low > max_pay:
        return None
    entry_price = min(close, max_pay)
    stop_anchor = max(base_low, vwap * 0.94, ema_like * 0.955)
    structural_stop = max(0.01, stop_anchor - max(0.05, entry_price * 0.005))
    risk = max(entry_price - structural_stop, entry_price * 0.020, 0.12)
    risk = min(risk, entry_price * 0.060)
    if risk <= 0:
        return None
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.55, 4)
    return {
        "entry_trigger": "warrior_failed_spike_vwap_reclaim",
        "variant_override": "warrior_failed_spike_vwap_reclaim",
        "psych_level": round(reclaim_level, 4),
        "pullaway_level": round(reclaim_level, 4),
        "base_high": round(base_high, 4),
        "base_low": round(base_low, 4),
        "failed_spike_high": round(failed_high, 4),
        "vwap": round(vwap, 4),
        "ema_like": round(ema_like, 4),
        "max_pay": max_pay,
        "entry_price_override": round(entry_price, 4),
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "max_hold_seconds_override": 150.0,
        "reward_risk": 1.55,
        "size_factor": 0.25,
        "rr_note_override": (
            "warrior failed-spike VWAP reclaim base=${:.2f}-${:.2f} "
            "failed_high=${:.2f} vwap=${:.2f} cap=${:.2f} target=${:.2f}"
        ).format(base_low, base_high, failed_high, vwap, max_pay, target_price),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_smooth_hod_reclaim_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    """Detect BJDX-style smooth HOD reclaims.

    This lane is for a steady premarket grinder that holds above VWAP/EMA-like
    support and reclaims HOD after controlled pullbacks. It is intentionally
    less violent than the CLWT/UTSI squeeze lanes and should not trade a topping
    wick or wide dump.
    """
    history = [bar for bar in history if float(bar.close or 0.0) > 0][-30:]
    if len(history) < 14:
        return None
    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= open_ or close <= 0 or high <= 0:
        return None

    prior = history[:-1]
    prior_high = max((float(bar.high or 0.0) for bar in prior[-18:]), default=0.0)
    day_low = min((float(bar.low or 0.0) for bar in history), default=0.0)
    hod = max(float(window_high or 0.0), prior_high)
    if prior_high <= 0 or day_low <= 0 or hod <= 0:
        return None
    # Keep this lane scoped to lower-price smooth grinders. Higher-priced
    # squeezes such as CAST have their own Warrior paths; using this gentler
    # HOD-reclaim lane there over-trades the trend and gives back profit.
    if close < 3.0 or close > 7.25:
        return None
    # The first impulse can be older than the 30-bar context on smooth grinders.
    # Require a meaningful local trend, but do not demand that the full initial
    # launch is still inside this short confirmation window.
    if hod < day_low * 1.12:
        return None

    base_bars = history[-9:-1]
    if len(base_bars) < 6:
        return None
    base_high = max(float(bar.high or 0.0) for bar in base_bars)
    base_low = min(float(bar.low or 0.0) for bar in base_bars)
    if base_high <= 0 or base_low <= 0:
        return None

    # Smooth HOD reclaim means the stock has been digesting near its highs, not
    # buying the first vertical print or a deep washout bounce.
    if base_low < hod * 0.86:
        return None
    base_range_pct = (base_high - base_low) / close if close > 0 else 999.0
    if base_range_pct > 0.115:
        return None
    if high < prior_high * 0.998 or close < base_high * 0.995:
        return None
    if high >= hod * 0.998 and close < hod * 0.995:
        return None

    rng = high - low
    if rng <= 0:
        return None
    close_location = (close - low) / rng
    range_pct = rng / close
    if close_location < 0.58 or range_pct > 0.085:
        return None

    total_volume = sum(float(bar.volume or 0.0) for bar in history)
    if total_volume <= 0:
        return None
    vwap = sum(
        ((float(bar.high or 0.0) + float(bar.low or 0.0) + float(bar.close or 0.0)) / 3.0)
        * float(bar.volume or 0.0)
        for bar in history
    ) / total_volume
    if close < vwap * 1.01 or base_low < vwap * 0.96:
        return None

    avg_base_volume = sum(float(bar.volume or 0.0) for bar in base_bars) / len(base_bars)
    recent_volume = sum(float(bar.volume or 0.0) for bar in history[-3:])
    if volume < max(45_000.0, avg_base_volume * 0.70):
        return None
    if recent_volume < max(150_000.0, avg_base_volume * 1.35):
        return None
    prior_hod_attempts = [
        bar for bar in prior[-8:]
        if float(bar.high or 0.0) >= hod * 0.96
        and float(bar.close or 0.0) < hod * 0.985
    ]
    if len(prior_hod_attempts) >= 3:
        max_attempt_volume = max(
            (float(bar.volume or 0.0) for bar in prior_hod_attempts),
            default=0.0,
        )
        if volume < max(90_000.0, max_attempt_volume * 1.05):
            return None

    red_distribution = any(
        float(bar.close or 0.0) < float(bar.open or 0.0)
        and (float(bar.open or 0.0) - float(bar.close or 0.0)) / float(bar.close or 1.0) > 0.045
        and (float(bar.high or 0.0) - float(bar.low or 0.0)) / float(bar.close or 1.0) > 0.055
        and float(bar.volume or 0.0) >= avg_base_volume * 1.25
        for bar in history[-5:-1]
    )
    if red_distribution:
        return None

    reclaim_level = max(prior_high, base_high)
    max_pay = round(reclaim_level * 1.025, 4)
    if low > max_pay:
        return None
    entry_price = min(close, max_pay)
    stop_anchor = max(base_low, vwap * 0.975)
    risk = max(entry_price - (stop_anchor - max(0.025, entry_price * 0.003)), entry_price * 0.016, 0.08)
    risk = min(risk, entry_price * 0.058)
    if risk <= 0:
        return None
    stop_price = round(entry_price - risk, 4)
    target_price = round(entry_price + risk * 1.35, 4)
    return {
        "entry_trigger": "warrior_smooth_hod_reclaim",
        "variant_override": "warrior_smooth_hod_reclaim",
        "psych_level": round(reclaim_level, 4),
        "pullaway_level": round(reclaim_level, 4),
        "base_high": round(base_high, 4),
        "base_low": round(base_low, 4),
        "vwap": round(vwap, 4),
        "max_pay": max_pay,
        "entry_price_override": round(entry_price, 4),
        "stop_price_override": stop_price,
        "target_price_override": target_price,
        "max_hold_seconds_override": 180.0,
        "reward_risk": 1.35,
        "size_factor": 0.35,
        "rr_note_override": (
            "warrior smooth HOD reclaim base=${:.2f}-${:.2f} "
            "hod=${:.2f} vwap=${:.2f} cap=${:.2f} target=${:.2f}"
        ).format(base_low, base_high, hod, vwap, max_pay, target_price),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_prior_runner_continuation_pullback_context(
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
        "entry_trigger": "warrior_prior_runner_continuation_pullback",
        "variant_override": "warrior_prior_runner_continuation_pullback",
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
            "warrior prior-runner continuation pullback "
            "base=${:.2f}-${:.2f} pullback={:.1f}% vwap=${:.2f} target=${:.2f}"
        ).format(base_low, base_high, pullback_pct * 100.0, vwap, target_price),
        "skip_unstable_confirm_stop_check": True,
    }


def warrior_post_target_pullback_reclaim_context(
    latest_bar: Bar,
    *,
    history: Sequence[Bar],
    window_high: float,
) -> Optional[Dict[str, Any]]:
    """One extra Warrior attempt after a banked target and tiny runner shakeout.

    This is not a generic add. It only wraps a clean existing Warrior pullback
    context and makes it smaller/tighter so a target winner can re-engage a
    controlled reclaim without reopening blind post-target chasing.
    """

    history = [bar for bar in history if float(bar.close or 0.0) > 0][-28:]
    if len(history) < 14:
        return None

    high = float(latest_bar.high or 0.0)
    low = float(latest_bar.low or 0.0)
    open_ = float(latest_bar.open or 0.0)
    close = float(latest_bar.close or 0.0)
    volume = float(latest_bar.volume or 0.0)
    if close <= open_ or close <= 0 or high <= 0 or low <= 0:
        return None
    if close < 3.50 or close > 12.50:
        return None

    prior = history[:-1]
    recent_peak = max((float(bar.high or 0.0) for bar in prior), default=0.0)
    runner_high = max(float(window_high or 0.0), recent_peak)
    day_low = min((float(bar.low or 0.0) for bar in history), default=0.0)
    if runner_high <= 0 or day_low <= 0 or runner_high < day_low * 1.20:
        return None

    base_bars = history[-8:-1]
    if len(base_bars) < 6:
        return None
    base_high = max(float(bar.high or 0.0) for bar in base_bars)
    base_low = min(float(bar.low or 0.0) for bar in base_bars)
    if base_high <= 0 or base_low <= 0:
        return None
    pullback_pct = (runner_high - base_low) / runner_high
    if pullback_pct < 0.06 or pullback_pct > 0.36:
        return None
    base_range_pct = (base_high - base_low) / close
    if base_range_pct > 0.115:
        return None
    if max(high, close) < base_high * 0.995:
        return None

    rng = high - low
    close_location = (close - low) / rng if rng > 0 else 0.0
    if close_location < 0.55 or rng / close > 0.08:
        return None

    total_volume = sum(float(bar.volume or 0.0) for bar in history)
    if total_volume <= 0:
        return None
    vwap = sum(
        ((float(bar.high or 0.0) + float(bar.low or 0.0) + float(bar.close or 0.0)) / 3.0)
        * float(bar.volume or 0.0)
        for bar in history
    ) / total_volume
    if close < vwap * 0.98 or base_low < vwap * 0.90:
        return None

    avg_base_volume = sum(float(bar.volume or 0.0) for bar in base_bars) / len(base_bars)
    recent_volume = sum(float(bar.volume or 0.0) for bar in history[-3:])
    if volume < max(35_000.0, avg_base_volume * 0.75):
        return None
    if recent_volume < max(140_000.0, avg_base_volume * 1.05):
        return None

    heavy_red = any(
        float(bar.close or 0.0) < float(bar.open or 0.0)
        and (float(bar.open or 0.0) - float(bar.close or 0.0)) / float(bar.close or 1.0) > 0.065
        and float(bar.volume or 0.0) >= avg_base_volume * 1.25
        for bar in history[-5:-1]
    )
    if heavy_red:
        return None

    reclaim_level = base_high
    max_pay = round(reclaim_level * 1.025, 4)
    if low > max_pay:
        return None
    entry = min(close, max_pay)
    stop_anchor = max(base_low, vwap * 0.94)
    stop = max(0.01, stop_anchor - max(0.035, entry * 0.004))
    risk = max(entry - stop, entry * 0.018, 0.08)
    risk = min(risk, entry * 0.055)
    stop = entry - risk
    target = entry + risk * 1.25
    return {
        "entry_trigger": "warrior_post_target_pullback_reclaim",
        "variant_override": "warrior_post_target_pullback_reclaim",
        "psych_level": round(reclaim_level, 4),
        "pullaway_level": round(reclaim_level, 4),
        "base_high": round(base_high, 4),
        "base_low": round(base_low, 4),
        "pullback_pct": round(pullback_pct * 100.0, 2),
        "vwap": round(vwap, 4),
        "max_pay": max_pay,
        "entry_price_override": round(entry, 4),
        "stop_price_override": round(stop, 4),
        "target_price_override": round(target, 4),
        "max_hold_seconds_override": 130.0,
        "reward_risk": 1.25,
        "size_factor": 0.25,
        "rr_note_override": (
            "warrior post-target pullback reclaim "
            "base=${:.2f}-${:.2f} pullback={:.1f}% target=${:.2f}"
        ).format(base_low, base_high, pullback_pct * 100.0, target),
        "skip_unstable_confirm_stop_check": True,
    }
