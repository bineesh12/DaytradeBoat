"""Reusable micro-pullback rules; automatic execution is explicitly PAPER-only.

No ticker, data client, wall clock or replay state lives here. Callers supply
completed bars and their actual completion times. A detected candle is evidence
of confirmation, not permission to bypass quality, freshness or portfolio gates.
The primary 65% and exploratory 50% close thresholds are separate hypotheses;
neither changes the existing HOD confirmation rule.
"""

from __future__ import annotations
from daytrading.data.bar_provenance import canonical_minute

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import floor, isfinite

from daytrading.data.completed_bars import NY, Rejection, validate_micro_window
from daytrading.models import Bar, Timeframe
from daytrading.models import CompletedBar as CompletedMicroBar
from daytrading.models import Quote

NAME = "micro_pullback_reclaim"


@dataclass(frozen=True)
class MicroPullbackRules:
    close_floor: float = 0.65
    min_impulse_pct: float = 5.0
    min_pullback_pct: float = 1.0
    max_pullback_pct: float = 12.0
    stop_buffer: float = 0.01
    max_completion_seconds: float = 15.0
    max_confirmation_age_seconds: float = 15.0

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if not all(isfinite(v) for v in values):
            raise ValueError("micro rules must be finite")
        if not 0 <= self.close_floor <= 1:
            raise ValueError("close_floor must be between zero and one")
        if not 0 < self.min_pullback_pct <= self.max_pullback_pct < 100:
            raise ValueError("invalid pullback bounds")
        if self.min_impulse_pct <= 0 or self.stop_buffer <= 0:
            raise ValueError("impulse and stop buffer must be positive")
        if self.max_completion_seconds < 10 or self.max_confirmation_age_seconds < 0:
            raise ValueError("invalid completion/freshness bounds")


@dataclass(frozen=True)
class MicroCandidate:
    symbol: str
    bar_start: datetime
    confirmed_at: datetime
    anchor: float
    technical_stop: float
    base_high: float
    close_location: float
    volume: float
    score: float

    @property
    def setup_id(self) -> str:
        # Multiple completed micro triggers inside one minute remain distinct.
        return f"{self.symbol}|{NAME}|{self.bar_start.isoformat()}"


def _aware(at: datetime) -> bool:
    return at.tzinfo is not None and at.utcoffset() is not None


DEFAULT_RULES = MicroPullbackRules()


def earlier_impulse(symbol, minute_bars, *, now, before,
                    lookback_minutes=10, peak_age_minutes=5, minimum_pct=5):
    """Causal, distinct-minute trough -> peak context, frozen in TCRT research.

    A peak must complete before the three-candle micro base begins. Its expiry
    is measured from completion, never refreshed by a new trigger or quote.
    """
    cutoff = now.replace(second=0, microsecond=0) - timedelta(minutes=lookback_minutes)
    rows = sorted((b for b in minute_bars
        if canonical_minute(b, min(now, before)) and b.symbol == symbol and b.timeframe == Timeframe.MIN_1
        and b.ts is not None and _aware(b.ts)
        and b.ts.astimezone(NY).date() == now.astimezone(NY).date()
        and cutoff <= b.ts and b.ts + timedelta(minutes=1) <= min(now, before)
        and all(isfinite(v) for v in (b.open, b.high, b.low, b.close, b.volume))
        and 0 < b.low <= min(b.open, b.close) <= max(b.open, b.close) <= b.high
        and b.volume > 0), key=lambda b: b.ts)
    eligible = []
    for i, peak in enumerate(rows):
        prior = [b for b in rows[:i] if b.ts < peak.ts]
        if not prior:
            continue
        trough = min(prior, key=lambda b: (b.low, b.ts))
        strength = (peak.high / trough.low - 1) * 100
        age = (now - (peak.ts + timedelta(minutes=1))).total_seconds()
        if strength >= minimum_pct and 0 <= age <= peak_age_minutes * 60:
            eligible.append((peak.high, peak.ts, trough, peak, strength, age))
    if not eligible:
        return Rejection('impulse_context', 'no >=5% ordered completed-minute impulse with peak age <=5m in prior 10m')
    _, _, low, high, strength, age = max(eligible, key=lambda x: (x[0], x[1]))
    return dict(trough_start=low.ts.isoformat(), trough=low.low,
                peak_start=high.ts.isoformat(), peak=high.high,
                peak_completed_at=(high.ts + timedelta(minutes=1)).isoformat(),
                impulse_pct=strength, peak_age_seconds=age)


def evaluate_micro_pullback(
    symbol: str,
    bars: Sequence[CompletedMicroBar],
    *,
    now: datetime,
    rules: MicroPullbackRules = DEFAULT_RULES,
    minute_bars: Sequence[Bar] | None = None,
) -> MicroCandidate | Rejection:
    """Evaluate one six-bar window; callers deduplicate using ``setup_id``."""
    invalid = validate_micro_window(
        symbol,
        bars,
        now=now,
        max_completion_seconds=rules.max_completion_seconds,
        max_confirmation_age_seconds=rules.max_confirmation_age_seconds,
    )
    if invalid is not None:
        return invalid
    window = bars[-6:]
    plain = [x.bar for x in window]
    a, b, z = plain[-3:]
    loc = (z.close - z.low) / (z.high - z.low) if z.high > z.low else 0
    if z.close <= z.open or z.close <= b.high:
        return Rejection("reclaim", "green candle must close above previous micro high")
    if not z.low >= b.low >= a.low:
        return Rejection("lows", "last three micro lows are not nondecreasing")
    if loc + 1e-12 < rules.close_floor:
        return Rejection(
            "close_location", f"micro close {loc:.1%} < {rules.close_floor:.1%}"
        )
    high = max(x.high for x in plain[:-1])
    low = min(x.low for x in plain[:-1])
    pull = (high - b.low) / high * 100
    impulse = (high / low - 1) * 100
    if not rules.min_pullback_pct <= pull <= rules.max_pullback_pct:
        return Rejection(
            "pullback", f"micro pullback {pull:.2f}% outside configured bounds"
        )
    if minute_bars is not None:
        context = earlier_impulse(symbol, minute_bars, now=now, before=a.ts,
                                  minimum_pct=rules.min_impulse_pct)
        if isinstance(context, Rejection):
            return context
        impulse = context['impulse_pct']
    if impulse < rules.min_impulse_pct:
        return Rejection(
            "impulse", f"micro impulse {impulse:.2f}% < {rules.min_impulse_pct:.2f}%"
        )
    if not any(x.close < x.open for x in plain[-5:-1]):
        return Rejection("pullback_candle", "no preceding red micro candle")
    reclaim = (z.close / b.high - 1) * 100
    return MicroCandidate(
        symbol,
        z.ts,
        window[-1].closed_at,
        z.close,
        round(z.low - rules.stop_buffer, 4),
        b.high,
        loc,
        sum(x.volume for x in plain[-3:]),
        round(impulse * 0.8 + pull * 2 + reclaim * 5, 3),
    )


@dataclass(frozen=True)
class MicroRiskLimits:
    min_price: float = 1.0
    max_price: float = 20.0
    max_dollar_risk: float = 50.0
    max_order_shares: int = 500
    max_stop_pct: float = 0.06
    min_technical_stop_pct: float = 0.005
    max_chase_pct: float = 0.025
    max_quote_age_seconds: float = 10.0
    max_confirmation_age_seconds: float = 15.0
    # Match the existing ExitManager minimum room when estimating filled risk.
    min_exit_risk_dollars: float = 0.05
    min_exit_risk_pct: float = 0.03

    def __post_init__(self) -> None:
        if not all(isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError("risk limits must be finite and positive")
        if self.min_price > self.max_price or self.max_stop_pct >= 1:
            raise ValueError("invalid price or stop range")
        if self.min_technical_stop_pct > self.max_stop_pct:
            raise ValueError("minimum stop exceeds maximum")
        if int(self.max_order_shares) != self.max_order_shares:
            raise ValueError("whole-share order cap required")


@dataclass(frozen=True)
class MicroEntryPlan:
    price: float
    technical_stop: float
    estimated_effective_stop: float
    quantity: int
    planned_risk: float
    technical_2r_target: float


DEFAULT_LIMITS = MicroRiskLimits()


def assess_micro_entry(
    candidate: MicroCandidate,
    quote: Quote | None,
    *,
    now: datetime,
    limits: MicroRiskLimits = DEFAULT_LIMITS,
) -> MicroEntryPlan | Rejection:
    """Only structural sizing; shared quality/portfolio/final gates still apply.

    Target retains the prototype's 2x technical gap. Effective filled risk can be
    larger owing to ExitManager's minimum room, and is explicitly budgeted here.
    Recompute/check protection against actual fills before accepting exposure.
    """
    if quote is None or not all(
        _aware(t) for t in (now, quote.ts, candidate.confirmed_at)
    ):
        return Rejection("quote", "fresh timezone-aware quote required")
    if quote.symbol != candidate.symbol:
        return Rejection("quote", "quote symbol does not match setup")
    age = (now - quote.ts).total_seconds()
    confirmation_age = (now - candidate.confirmed_at).total_seconds()
    if not 0 <= age <= limits.max_quote_age_seconds:
        return Rejection("quote_age", f"quote age {age:.3f}s outside freshness bounds")
    if not 0 <= confirmation_age <= limits.max_confirmation_age_seconds:
        return Rejection(
            "confirmation_age", f"confirmation age {confirmation_age:.3f}s"
        )
    if not all(
        isfinite(v)
        for v in (
            quote.bid,
            quote.ask,
            quote.bid_size,
            quote.ask_size,
            candidate.anchor,
            candidate.technical_stop,
        )
    ):
        return Rejection("quote", "non-finite quote or setup price")
    if not 0 < quote.bid <= quote.ask or min(quote.bid_size, quote.ask_size) <= 0:
        return Rejection("quote", "valid two-sided executable quote required")
    price, stop = quote.ask, candidate.technical_stop
    if not limits.min_price <= price <= limits.max_price:
        return Rejection("price", "entry price outside configured universe")
    if candidate.anchor <= 0 or price > candidate.anchor * (1 + limits.max_chase_pct):
        return Rejection("chase", "entry beyond original setup chase ceiling")
    gap = price - stop
    if not 0 < stop < price or gap / price > limits.max_stop_pct + 1e-12:
        return Rejection(
            "stop", "structural stop invalid or wider than maximum; never clamped"
        )
    if gap / price < limits.min_technical_stop_pct:
        return Rejection("stop_tight", "technical stop below minimum width")
    risk = max(gap, limits.min_exit_risk_dollars, price * limits.min_exit_risk_pct)
    if risk / price > limits.max_stop_pct + 1e-12:
        return Rejection(
            "effective_stop", "minimum exit room exceeds maximum stop width"
        )
    # Depth is shares, not 100-share lots. The later fill model must cap it again.
    qty = min(
        int(limits.max_order_shares),
        floor(limits.max_dollar_risk / risk),
        floor(quote.ask_size),
    )
    if qty < 1:
        return Rejection("quantity", "no shares fit risk and displayed depth")
    return MicroEntryPlan(price, stop, price - risk, qty, qty * risk, price + 2 * gap)


def partial_runner_stop(
    *,
    entry: float,
    initial_stop: float,
    previous_stop: float,
    highest_observed_bid: float,
    partial_filled_shares: int,
) -> float:
    """Ratchet only after actual partial fills; an order submission is not a fill."""
    if not all(
        isfinite(v) for v in (entry, initial_stop, previous_stop, highest_observed_bid)
    ):
        raise ValueError("finite position prices required")
    if not 0 < initial_stop < entry or partial_filled_shares < 0:
        raise ValueError("invalid initial protection or filled quantity")
    if partial_filled_shares == 0:
        return max(initial_stop, previous_stop)
    return max(
        initial_stop,
        previous_stop,
        entry,
        highest_observed_bid - (entry - initial_stop),
    )
