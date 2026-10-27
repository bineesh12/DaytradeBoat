"""Causal NBBO limit fills, with explicitly separate paper and depth assumptions.

No broker fills, order history, trade outcome, or future quote is an input.
Latency/expiry belong to the caller. Paper mode approximates Alpaca's published
10% random partial-fill rule; its internal cadence/distribution is unknown.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from random import Random
from daytrading.models import Quote, Side
from .matching_profile import MatchingProfile

@dataclass(frozen=True)
class NBBOExecutionPolicy:
    mode: str = 'depth'
    seed: int = 0
    max_quote_age_seconds: float = 10.
    paper_recheck_seconds: float = .15
    matching_profile: MatchingProfile | None = None

    def __post_init__(self):
        if self.matching_profile is not None and not isinstance(self.matching_profile,MatchingProfile):
            raise ValueError('Matching profile must use the aggregate schema')
        if self.mode not in ('depth', 'paper'):
            raise ValueError('Unknown NBBO execution mode')
        if not isfinite(self.max_quote_age_seconds) or self.max_quote_age_seconds <= 0:
            raise ValueError('Invalid quote age limit')
        if not isfinite(self.paper_recheck_seconds) or self.paper_recheck_seconds <= 0:
            raise ValueError('Invalid paper matching interval')

class NBBOFillSimulator:
    def __init__(self, policy=NBBOExecutionPolicy()):
        self.policy = policy
        self.random = Random(policy.seed)
        self._last_time = None
        self._seen = {}
        self._depth = {}
        self._paper_attempts = {}
        self._match_due = {}

    def next_match_time(self,order_id):
        if self.policy.matching_profile is not None:return self._match_due.get(order_id)
        prior=self._paper_attempts.get(order_id)
        return prior+timedelta(seconds=self.policy.paper_recheck_seconds) if prior is not None else None

    def take(self, order_id: str, symbol: str, side: Side, remaining: int,
             limit: float, quote: Quote | None, *, now: datetime,
             arrived_at: datetime, expires_at: datetime | None = None,
             quote_now: datetime | None = None):
        """Return (quantity, price), or None. Depth: one observation per quote/order. PAPER: paced matching attempts.

        A marketable SELL receives the contemporaneous bid, including improvement
        over its limit; it never receives a historical decision-price clip.
        """
        if now.tzinfo is None or arrived_at.tzinfo is None:
            raise ValueError('Timezone-aware execution times required')
        if self._last_time is not None and now < self._last_time:
            raise ValueError('Execution clock moved backwards')
        self._last_time = now
        if not order_id or remaining <= 0 or int(remaining) != remaining or not isfinite(limit) or limit <= 0:
            raise ValueError('Invalid order intent')
        if self.policy.mode=='paper' and self.policy.matching_profile is not None:
            self.policy.matching_profile.assert_available(arrived_at)
        if now < arrived_at or (expires_at is not None and now >= expires_at):
            return None
        if quote is None or quote.symbol != symbol or quote.ts.tzinfo is None:
            return None
        # Arrival/deadline cadence uses elapsed receipt time. Quote age uses
        # exchange UTC, which can differ from the recording host's wall clock.
        quote_now = now if quote_now is None else quote_now
        if quote_now.tzinfo is None:
            raise ValueError('Timezone-aware quote reference required')
        if not 0 <= (quote_now-quote.ts).total_seconds() <= self.policy.max_quote_age_seconds:
            return None
        if not all(isfinite(x) and x > 0 for x in (quote.bid, quote.ask)) or quote.bid > quote.ask:
            return None
        # A locked NBBO still has an executable best bid/ask for an already
        # accepted order. Crossed quotes remain invalid. Entry admission is a
        # separate strategy decision and retains its existing spread guards.
        price = quote.ask if side is Side.BUY else quote.bid
        size = quote.ask_size if side is Side.BUY else quote.bid_size
        if (side is Side.BUY and price > limit+1e-12) or (side is Side.SELL and price < limit-1e-12):
            return None
        stamp = (quote.ts, quote.bid, quote.ask, quote.bid_size, quote.ask_size)
        if self.policy.mode == 'paper':
            profile=self.policy.matching_profile
            if profile is not None:
                due=self._match_due.get(order_id)
                if due is None:
                    self._match_due[order_id]=now+timedelta(seconds=profile.initial_delay.draw(self.random))
                    return None
                if now<due:return None
            else:
                # Legacy documented-spec approximation remains available for
                # explicit baseline comparisons; its cadence is an assumption.
                prior = self._paper_attempts.get(order_id)
                if prior is not None and (now - prior).total_seconds() < self.policy.paper_recheck_seconds - 1e-9:
                    return None
            self._paper_attempts[order_id] = now
            quantity = remaining
            probability=profile.partial_probability if profile is not None else .1
            if remaining > 1 and self.random.random() < probability:
                quantity = self.random.randint(1, remaining-1)
            if profile is not None:
                if quantity<remaining:self._match_due[order_id]=now+timedelta(seconds=profile.remainder_delay.draw(self.random))
                else:self._match_due.pop(order_id,None)
        else:
            if self._seen.get(order_id) == stamp:
                return None
            self._seen[order_id] = stamp
            if not isfinite(size) or size < 0:
                return None
            key = (symbol, side)
            prior = self._depth.get(key)
            if prior is None or prior[0] != price:
                available = int(size)
            else:
                # Bid-only changes cannot replenish consumed ask depth (and vice
                # versa). Only an observed size increase adds capacity at a price.
                available = min(prior[2], int(size)) + max(0, int(size)-prior[1])
            quantity = min(remaining, available)
            self._depth[key] = (price, int(size), available-quantity)
        return (quantity, price) if quantity > 0 else None


def dollar_capped_stop(entry: float, quantity: int, stop: float, budget: float) -> float:
    """Enforce the existing dollar budget after cumulative weighted partial fills.

    Round protection upwards, never round the dollar risk above its budget.
    This is a simulation invariant, not a change to an entry threshold.
    """
    from decimal import Decimal, ROUND_CEILING
    if not all(isfinite(x) and x > 0 for x in (entry, quantity, stop, budget)) or stop >= entry:
        raise ValueError('Invalid long protection inputs')
    floor = Decimal(str(entry)) - Decimal(str(budget)) / Decimal(str(quantity))
    bounded = max(Decimal(str(stop)), floor.quantize(Decimal('.0001'), rounding=ROUND_CEILING))
    return float(bounded)
