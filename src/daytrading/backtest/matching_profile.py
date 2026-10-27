"""Aggregate, frozen matching parameters. Per-order outcomes are not a schema.

Fitting happens offline on independent earlier sessions. Prediction receives no
order IDs, instruments, fill prices, quantities, times, or recorded trajectories.
"""
from dataclasses import dataclass
from datetime import date,datetime
from math import isfinite,exp

@dataclass(frozen=True)
class DelayDistribution:
    mu: float
    sigma: float
    def __post_init__(self):
        if not isfinite(self.mu) or not isfinite(self.sigma) or self.sigma<0:
            raise ValueError('Invalid delay distribution')
    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or set(value)!={'mu','sigma'}:
            raise ValueError('Delay profile permits aggregate parameters only')
        return cls(float(value['mu']),float(value['sigma']))
    def draw(self,rng):
        try:value=exp(self.mu+self.sigma*rng.gauss(0.,1.))
        except OverflowError:raise ValueError('Delay sample overflow')
        if not isfinite(value) or value<=0:raise ValueError('Invalid sampled delay')
        return value

@dataclass(frozen=True)
class MatchingProfile:
    schema_version: int
    version: str
    partial_probability: float
    initial_delay: DelayDistribution
    remainder_delay: DelayDistribution
    size_model: str
    training_end_exclusive: str
    training_order_count: int
    initial_sample_count: int
    remainder_sample_count: int
    def __post_init__(self):
        if self.schema_version!=1 or not self.version or not isfinite(self.partial_probability) or not 0<=self.partial_probability<=1:
            raise ValueError('Invalid matching profile')
        if self.size_model!='uniform_integer_remaining':raise ValueError('Unapproved partial size model')
        date.fromisoformat(self.training_end_exclusive)
        if any(type(v) is not int or v<1 for v in (self.training_order_count,self.initial_sample_count,self.remainder_sample_count)):
            raise ValueError('Missing calibration sample counts')
    @classmethod
    def from_dict(cls,value):
        fields=set(cls.__dataclass_fields__)
        if not isinstance(value,dict) or set(value)!=fields:
            raise ValueError('Matching profile permits aggregate parameters only')
        data=dict(value)
        for name in ('initial_delay','remainder_delay'):data[name]=DelayDistribution.from_dict(data[name])
        return cls(**data)
    def assert_available(self,stamp):
        if stamp.tzinfo is None or stamp.date()<date.fromisoformat(self.training_end_exclusive):
            raise ValueError('Matching calibration is newer than the simulated order')
    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)


# Frozen aggregate fit: independent earlier non-NCPL limit-order executions.
# This is execution-model calibration, never a strategy/risk configuration.
_DEFAULT_PROFILE_DATA = {'initial_delay': {'mu': -0.7221637127826194, 'sigma': 0.6945525084296982},
 'initial_sample_count': 106,
 'partial_probability': 0.6320406278855032,
 'remainder_delay': {'mu': -0.576563521152713, 'sigma': 0.5431314048538747},
 'remainder_sample_count': 134,
 'schema_version': 1,
 'size_model': 'uniform_integer_remaining',
 'training_end_exclusive': '2026-06-17',
 'training_order_count': 879,
 'version': 'independent-limit-matching-v1'}

def default_matching_profile():
    return MatchingProfile.from_dict(_DEFAULT_PROFILE_DATA)
