"""Explicit entry spread policy; baseline reproduces existing numeric limits.

Normal eligibility shares one percentage/tick floor. Historical cost and
instantaneous execution remain separate measurements. The liquidity trap uses
a raw percentage with no tick floor and a completed-minute participation test;
the broker safety ceiling is independently wider. These are named settings,
never hidden overrides of the normal rule. No exit pricing is configured here.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
import os
from daytrading.strategy.entry_checks import record


@dataclass(frozen=True)
class SpreadPolicy:
    normal_pct: float = .005
    normal_tick_floor: float = .01
    quote_max_age_seconds: float = 10.
    micro_history_seconds: float = 10.
    micro_min_coverage_seconds: float = 2.
    execution_recent_samples: int = 3
    quality_recent_samples: int = 5
    trap_pct: float = .005
    trap_tick_floor: float = 0.
    trap_minute_volume_ratio: float = 1.2
    broker_safety_pct: float = .01

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid spread policy: '+name)
        if not (0 < self.normal_pct <= .1 and 0 < self.trap_pct <= .1 and 0 < self.broker_safety_pct <= .1):
            raise ValueError('Spread percentage must be a positive fraction')
        if not (0 < self.quote_max_age_seconds <= 10 and 0 < self.micro_min_coverage_seconds <= self.micro_history_seconds <= 10):
            raise ValueError('Quote age/window exceeds retained causal history')
        for name in ('execution_recent_samples','quality_recent_samples'):
            value=getattr(self,name)
            if not isinstance(value,int) or not 1<=value<=200:
                raise ValueError('Invalid sample count: '+name)
        if self.trap_minute_volume_ratio <= 0:
            raise ValueError('Invalid minute volume ratio')

    def limit(self, price, kind='normal'):
        if kind=='trap': return max(self.trap_tick_floor, price*self.trap_pct)
        if kind=='broker': return price*self.broker_safety_pct
        return max(self.normal_tick_floor, price*self.normal_pct)


_OVERRIDE=ContextVar('entry_spread_policy', default=None)


@lru_cache(maxsize=8)
def _configured(raw):
    values=json.loads(raw) if raw else {}
    if not isinstance(values,dict):raise ValueError('Entry spread policy must be an object')
    return SpreadPolicy(**values)


def get_policy():
    return _OVERRIDE.get() or _configured(os.getenv('DAYTRADING_ENTRY_SPREAD_POLICY',''))


@contextmanager
def configured(policy):
    """Explicit scoped configuration for tests/replay; no source monkeypatch."""
    if not isinstance(policy,SpreadPolicy):raise TypeError('Typed spread policy required')
    token=_OVERRIDE.set(policy)
    try: yield policy
    finally: _OVERRIDE.reset(token)


def check(target,name,*,spread,price,now,quote,window,kind='normal'):
    policy=get_policy();limit=policy.limit(price,kind)
    valid=math.isfinite(spread) and math.isfinite(price) and price>0 and spread>=0
    ok=valid and spread<=limit+(0 if kind in ('trap','broker') else 1e-6)
    if target is not None:target['entry_spread_policy']=asdict(policy)
    return record(target,name,now=now,measured=spread if math.isfinite(spread) else None,
        threshold=limit,units='USD/share',passed=ok,window=window,quote=quote,
        denominator_price=price,policy_rule=kind)


def current(target,name,quote,now,*,kind='normal'):
    age=(now-quote.ts).total_seconds() if quote is not None and quote.ts.tzinfo is not None and now is not None and now.tzinfo is not None else None
    policy=get_policy()
    valid=quote is not None and age is not None and 0<=age<=policy.quote_max_age_seconds
    valid=valid and all(math.isfinite(x) and x>0 for x in (quote.bid,quote.ask,quote.bid_size,quote.ask_size)) and quote.ask>quote.bid
    record(target,name+'.quote_age',now=now,measured=age,threshold=policy.quote_max_age_seconds,
        units='seconds',comparison='0 <= age <=',passed=bool(valid),window={'source':'current quote'},quote=quote)
    if not valid:return False
    denominator=quote.mid if kind in ('trap','broker') else quote.ask
    return check(target,name,spread=quote.spread,price=denominator,now=now,quote=quote,
                 window={'source':'current quote','start':quote.ts.isoformat(),'end':now.isoformat()},kind=kind)
