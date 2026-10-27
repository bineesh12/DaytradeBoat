"""Opt-in immutable micro risk plan and fill-aware completed-10s runner exit.

The stop is approved BEFORE submission. Actual fills may reduce target distance,
never widen the stop, increase quantity or enlarge the reserved dollar budget.
One fixed candidate: half at 1R; then breakeven plus the lowest of the two most
recent completed, post-entry 10s candles minus one cent. No stock exceptions.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from math import isfinite, floor
from decimal import Decimal

VERSION = 'micro-risk-plan-v1'
POLICIES = ('coordinated_tick', 'coordinated_structure')


@dataclass(frozen=True)
class MicroRiskPlan:
    symbol: str
    cap: float
    stop: float
    quantity: int
    budget: float
    policy: str = 'coordinated_structure'
    target_r: float = 1.0
    max_hold_seconds: int = 60
    version: str = VERSION

    def __post_init__(self):
        if self.version != VERSION or self.policy not in POLICIES:
            raise ValueError('unknown coordinated micro plan')
        if not self.symbol or not all(isfinite(x) for x in
                (self.cap, self.stop, self.quantity, self.budget, self.target_r)):
            raise ValueError('finite identified plan required')
        if not 1 <= self.cap <= 20 or not 0 < self.stop < self.cap:
            raise ValueError('invalid cap/stop')
        if not 1 <= self.quantity <= 500 or int(self.quantity) != self.quantity:
            raise ValueError('whole-share quantity within unchanged cap required')
        if not 0 < self.budget <= 50 or (self.cap-self.stop)/self.cap > .06+1e-10:
            raise ValueError('unchanged risk caps exceeded')
        if self.quantity*(self.cap-self.stop) > self.budget+1e-8:
            raise ValueError('approved stop exceeds reserved budget')
        if self.target_r != 1 or self.max_hold_seconds != 60:
            raise ValueError('frozen 1R / 60s candidate required')

    def metadata(self):
        return asdict(self)

    def validate_fill(self, symbol, price, quantity):
        if symbol != self.symbol or not all(isfinite(x) for x in (price, quantity)):
            raise ValueError('invalid fill identity')
        if not self.stop < price <= self.cap+1e-10 or not 0 < quantity <= self.quantity:
            raise ValueError('fill outside approved plan; reconcile exposure')
        if quantity*(price-self.stop) > self.budget+1e-8:
            raise ValueError('fill risk exceeds reserved budget')


def approve(signal, cap, quantity, budget, policy='coordinated_structure'):
    """Resolve legacy minimum stop room once, before the order can be sent.

    Existing verifier/price caps and conservative risk sizing run first. This
    fixes downstream reinterpretation; it does not move the original cap.
    """
    gap=max(cap-signal.stop_loss, .05, cap*.03)
    stop=round(cap-gap, 4)
    plan=MicroRiskPlan(signal.symbol, cap, stop, int(quantity), budget, policy)
    signal.scan_result.criteria['micro_risk_plan']=plan.metadata()
    signal.scan_result.criteria['_approved_micro_risk_plan']=plan
    signal.scan_result.criteria['micro_original_technical_stop']=signal.stop_loss
    # TradeSignal is immutable and retains the detector's technical proposal.
    # ExitManager consumes this authoritative approved plan, never that proposal.
    signal.scan_result.criteria['micro_planned_first_target']=round(cap+(cap-plan.stop)*plan.target_r,6)
    return plan


def from_signal(signal):
    scan=signal.scan_result
    raw=scan.criteria.get('micro_risk_plan') if scan is not None else None
    plan=MicroRiskPlan(**raw) if raw is not None else None
    typed=scan.criteria.get('_approved_micro_risk_plan') if scan is not None else None
    if typed is not None and (plan is None or typed.metadata()!=plan.metadata()):
        raise ValueError('approved micro risk plan changed')
    if plan is not None and plan.symbol!=signal.symbol:
        raise ValueError('approved plan symbol changed')
    return plan


@dataclass
class MicroExitState:
    plan: MicroRiskPlan
    entry: float
    quantity: float
    entered_at: datetime
    stop: float
    sold: float = 0
    target_filled: float = 0
    closed_bars: list = field(default_factory=list)
    seen_fills: set = field(default_factory=set)
    pending_reason: str = ''
    stop_trace: list = field(default_factory=list)
    plan_violation: str = ''

    @classmethod
    def open(cls, plan, price, quantity, at):
        plan.validate_fill(plan.symbol,price,quantity)
        if at.tzinfo is None:raise ValueError('aware fill time required')
        return cls(plan,price,quantity,at,plan.stop)

    @classmethod
    def from_observed_fill(cls, plan, price, quantity, at):
        """A real broker fill cannot disappear when its approved plan fails.

        Keep the original stop and immediately latch liquidation. This is an
        exposure reconciliation path, never permission to enter outside a plan.
        """
        if not all(isfinite(x) and x>0 for x in (price,quantity)) or at.tzinfo is None:
            raise ValueError('invalid observed broker fill')
        try:
            return cls.open(plan,price,quantity,at)
        except ValueError as exc:
            return cls(plan,price,quantity,at,plan.stop,
                       pending_reason='stop_loss',plan_violation=str(exc))

    @property
    def target(self):
        return float(Decimal(str(self.entry)) +
            (Decimal(str(self.entry))-Decimal(str(self.plan.stop)))*Decimal(str(self.plan.target_r)))

    @property
    def partial_quantity(self):
        return max(1, floor(self.quantity/2)) if self.quantity>1 else 1

    @property
    def remaining(self):
        return max(0.,self.quantity-self.sold)

    def synchronize_entry(self, price, quantity, *, observed=False):
        """Cumulative/late entry fills retain the original cap, stop and budget."""
        if quantity < self.quantity or self.sold:
            raise ValueError('entry correction after reduction requires reconciliation')
        try:
            self.plan.validate_fill(self.plan.symbol,price,quantity)
        except ValueError as exc:
            if not observed or not all(isfinite(x) and x>0 for x in (price,quantity)):
                raise
            self.pending_reason='stop_loss';self.plan_violation=str(exc)
        self.entry,self.quantity=price,quantity

    def on_completed_bar(self, bar, *, observed_at, closed_at=None):
        from daytrading.models import Timeframe
        closed=closed_at or bar.ts+timedelta(seconds=10)
        if (bar.symbol!=self.plan.symbol or bar.ts.tzinfo is None
                or bar.timeframe is not Timeframe.SEC_10
                or observed_at.tzinfo is None or closed>observed_at
                or not 10 <= (closed-bar.ts).total_seconds() <= 20
                or bar.ts<self.entered_at
                or not all(isfinite(x) and x>0 for x in (bar.open,bar.high,bar.low,bar.close))
                or not bar.low<=min(bar.open,bar.close)<=max(bar.open,bar.close)<=bar.high):
            return False
        if self.closed_bars and bar.ts<=self.closed_bars[-1][0]:return False
        # Gaps are not two consecutive candles; never manufacture empty bars.
        if self.closed_bars and bar.ts!=self.closed_bars[-1][2]:
            self.closed_bars=[]
        self.closed_bars=(self.closed_bars+[(bar.ts,bar.low,closed)])[-2:]
        if self.target_filled>=self.partial_quantity and len(self.closed_bars)==2:
            self._tighten(max(self.entry,min(x[1] for x in self.closed_bars)-.01),observed_at,'completed_10s_structure')
        return True

    def _tighten(self, value, at, reason):
        if value>self.stop:
            self.stop=round(value,6)
            self.stop_trace.append(dict(at=at.isoformat(),stop=self.stop,reason=reason))

    def decide(self, price, now):
        if not isfinite(price) or price<=0 or now<self.entered_at or self.remaining<=0:return None
        if self.pending_reason in ('stop_loss','time_exit'):
            return self.remaining,self.pending_reason
        if price<=self.stop:
            self.pending_reason='stop_loss'
            return self.remaining,self.pending_reason
        if (now-self.entered_at).total_seconds()>=self.plan.max_hold_seconds:
            self.pending_reason='time_exit'
            return self.remaining,self.pending_reason
        if self.target_filled<self.partial_quantity and price>=self.target:
            self.pending_reason='partial_1r'
            return min(self.remaining,self.partial_quantity-self.target_filled),self.pending_reason
        return None

    def commit_fill(self, fill):
        from daytrading.execution.fill_identity import identity
        key=identity(fill)
        if key in self.seen_fills:return False
        if fill.symbol!=self.plan.symbol or getattr(fill.side,'value',fill.side)!='sell':
            raise ValueError('wrong exit fill')
        if not isfinite(fill.quantity) or not 0<fill.quantity<=self.remaining:
            raise ValueError('exit fill exceeds actual inventory')
        self.seen_fills.add(key);self.sold+=fill.quantity
        if self.pending_reason=='partial_1r':
            self.target_filled+=fill.quantity
            if self.target_filled>=self.partial_quantity:
                self._tighten(self.entry,fill.ts,'filled_partial_breakeven')
        # A cancelled/unfilled request never calls here; protective intents stay
        # latched through partial fills, late fills, quote bounces and retries.
        if self.pending_reason=='partial_1r' and self.target_filled>=self.partial_quantity:
            self.pending_reason=''
        return True
