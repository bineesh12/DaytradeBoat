"""Recorded execution calibration, explicitly not a predictive fill model.

Historical quotes do not determine broker queue, improvement or weighted fills.
Recorded events may be consumed only by a compatible generated order intent.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Iterable, Mapping, Optional
from daytrading.models import Fill, Order, Side


def _time(value: str | datetime) -> datetime:
    if not isinstance(value,(str,datetime)):
        raise ValueError('Missing execution timestamp')
    stamp=value if isinstance(value,datetime) else datetime.fromisoformat(value.replace('Z','+00:00'))
    if stamp.tzinfo is None:raise ValueError('Execution timestamps must be timezone-aware')
    return stamp


@dataclass(frozen=True)
class RecordedExecution:
    order_id: str
    symbol: str
    side: Side
    requested_quantity: int
    limit_price: float
    submitted_at: datetime
    fills: tuple[Fill,...]
    granularity: str='individual_fills'

    def __post_init__(self):
        _time(self.submitted_at)
        if not self.order_id or not self.symbol or self.requested_quantity<=0:raise ValueError('Invalid recorded order identity/quantity')
        if not math.isfinite(self.limit_price) or self.limit_price<=0:raise ValueError('Invalid recorded limit')
        total=0.;last=self.submitted_at
        for fill in self.fills:
            _time(fill.ts)
            if fill.symbol!=self.symbol or fill.side is not self.side or fill.ts<last:raise ValueError('Recorded fill identity/time mismatch')
            if not all(math.isfinite(x) and x>0 for x in (fill.quantity,fill.price)):raise ValueError('Invalid recorded fill')
            if ((self.side is Side.BUY and fill.price>self.limit_price+1e-9) or (self.side is Side.SELL and fill.price<self.limit_price-1e-9)):raise ValueError('Recorded fill violates recorded limit')
            total+=fill.quantity;last=fill.ts
        if total>self.requested_quantity+1e-9:raise ValueError('Recorded order overfilled')

    @classmethod
    def from_order_summary(cls,row:Mapping)->Optional['RecordedExecution']:
        """An average fill is one aggregate event, never invented partials."""
        if row.get('type')!='limit' or float(row.get('filled_qty') or 0)<=0:return None
        side=Side(str(row['side']))
        fill=Fill(str(row['symbol']),side,float(row['filled_qty']),float(row['filled_avg_price']),_time(row['filled_at']))
        return cls(str(row['id']),str(row['symbol']),side,int(float(row['qty'])),float(row['limit_price']),_time(row['submitted_at']),(fill,),'order_aggregate')


class RecordedExecutionReplay:
    """Intent binding and one-time recorded fills. No simulated fill fallback."""
    mode='recorded_execution_calibration'
    predictive=False
    def __init__(self,records:Iterable[RecordedExecution],*,max_submit_delay:float=2.):
        self.records=tuple(records)
        if len({r.order_id for r in self.records})!=len(self.records):raise ValueError('Duplicate recorded order ID')
        if not math.isfinite(max_submit_delay) or max_submit_delay<0:raise ValueError('Invalid submission observation window')
        self.max_submit_delay=timedelta(seconds=max_submit_delay)
        self.bound:dict[str,int]={};self.audit:list[dict]=[];self.last_observation:Optional[datetime]=None

    def bind(self,order:Order,*,now:datetime,limit_floor:Optional[float]=None,limit_ceiling:Optional[float]=None)->RecordedExecution:
        _time(now)
        if self.last_observation is not None and now < self.last_observation:
            raise ValueError('Cannot bind an order in the observed past')
        if order.hard_limit_price is not None and (not math.isfinite(order.hard_limit_price) or order.hard_limit_price<=0):
            raise ValueError('Invalid hard execution limit')
        if not math.isfinite(order.quantity) or order.quantity<=0 or int(order.quantity)!=order.quantity:raise ValueError('Invalid generated quantity')
        # Exact request by default. Repricing bounds must be explicit and
        # cannot override a hard execution limit.
        floor=order.limit_price if limit_floor is None else limit_floor
        ceiling=order.limit_price if limit_ceiling is None else limit_ceiling
        if floor is None or ceiling is None or not 0<floor<=ceiling or not math.isfinite(floor) or not math.isfinite(ceiling):raise ValueError('A finite generated request price range is required')
        matches=[]
        for row in self.records:
            delay=row.submitted_at-now
            if (row.order_id in self.bound or row.symbol!=order.symbol or row.side is not order.side or row.requested_quantity!=int(order.quantity) or not timedelta(0)<=delay<=self.max_submit_delay or not floor-1e-9<=row.limit_price<=ceiling+1e-9):continue
            if order.hard_limit_price is not None and ((order.side is Side.BUY and row.limit_price>order.hard_limit_price+1e-9) or (order.side is Side.SELL and row.limit_price<order.hard_limit_price-1e-9)):continue
            matches.append(row)
        if len(matches)!=1:raise ValueError('Recorded order intent mismatch: expected one match, found {}'.format(len(matches)))
        row=matches[0];self.bound[row.order_id]=0
        self.audit.append(dict(mode=self.mode,predictive=False,order_id=row.order_id,symbol=row.symbol,side=row.side.value,quantity=row.requested_quantity,request_observed_at=now.isoformat(),recorded_submitted_at=row.submitted_at.isoformat(),limit_price=row.limit_price,granularity=row.granularity))
        return row

    def poll(self,now:datetime)->list[tuple[str,Fill]]:
        _time(now)
        if self.last_observation is not None and now<self.last_observation:raise ValueError('Execution replay clock moved backwards')
        self.last_observation=now;out=[]
        for row in self.records:
            if row.order_id not in self.bound:continue
            index=self.bound[row.order_id]
            while index<len(row.fills) and row.fills[index].ts<=now:
                out.append((row.order_id,row.fills[index]));index+=1
            self.bound[row.order_id]=index
        return sorted(out,key=lambda item:(item[1].ts,item[0]))

    def remaining_recorded_orders(self)->list[str]:
        return [r.order_id for r in self.records if r.order_id not in self.bound]


def quote_request_limit(order: Order, bar, quote, *, now: datetime) -> float:
    """Use the live broker's smart request price; this is not the fill price."""
    from daytrading.execution.pricing import marketable_limit
    from daytrading.risk.guards import SlippageGuard
    _time(now)
    if quote is None or quote.ts is None or quote.ts.tzinfo is None:
        raise ValueError('Missing timestamped execution quote')
    if quote.symbol != order.symbol or bar.symbol != order.symbol:
        raise ValueError('Execution quote symbol mismatch')
    if not all(math.isfinite(v) and v > 0 for v in (quote.bid, quote.ask)) or quote.bid >= quote.ask:
        raise ValueError('Execution quote must be finite and unlocked')
    if not 0 <= (now - quote.ts).total_seconds() <= 10:
        raise ValueError('Execution quote stale or from the future')
    guard = SlippageGuard()
    guard.update_quote(quote)
    smart = guard.get_limit_price(order.symbol, order.side.value)
    return marketable_limit(order, bar, quote_price=smart)
