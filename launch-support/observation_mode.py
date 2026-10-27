"""Process-latched PAPER entry prohibition plus bounded observational timings.

Deployment support only: no change to the frozen daytrading source. Closing
orders still traverse every native broker, reconciliation and protection check.
"""
from contextvars import ContextVar
from functools import wraps
import math
import threading
import time

REASON='paper_observation_only'
_closing=ContextVar('paper_observation_closing',default=None)

class LatencyProbe:
    def __init__(self,capacity=25000):
        self.capacity=capacity;self.lock=threading.Lock();self.pending={}
        self.rows={k:dict(count=0,total_ns=0,max_ns=0,buckets=[0]*64) for k in ['QuoteEvent','TradeEvent']}
        self.missing=0;self.overflow=0;self.rejected_entries=0

    def receipt(self,event):
        kind=type(event).__name__
        if kind not in self.rows:return
        now=time.perf_counter_ns()
        with self.lock:
            if len(self.pending)>=self.capacity:self.overflow+=1
            else:self.pending[id(event)]=(now,kind)

    def remove(self,event):
        with self.lock:self.pending.pop(id(event),None)

    def consume(self,event):
        if type(event).__name__ not in self.rows:return
        now=time.perf_counter_ns()
        with self.lock:
            record=self.pending.pop(id(event),None)
            if record is None:self.missing+=1;return
            stamp,kind=record;elapsed=max(0,now-stamp);r=self.rows[kind]
            r['count']+=1;r['total_ns']+=elapsed;r['max_ns']=max(r['max_ns'],elapsed)
            r['buckets'][min(63,elapsed.bit_length())]+=1

    def snapshot(self):
        with self.lock:
            rows={kind:dict(count=x['count'],total_ns=x['total_ns'],max_ns=x['max_ns'],histogram={str(i):n for i,n in enumerate(x['buckets']) if n}) for kind,x in self.rows.items()}
            return dict(mode=REASON,automatic_entries=False,latched_for_process=True,
                metric='native enqueue callback entry to native event-processing entry; same perf_counter clock',
                percentile_kind='power-of-two nanosecond histogram upper bounds',receipt_to_processing=rows,
                unmatched_consumptions=self.missing,probe_overflow=self.overflow,pending_probe_records=len(self.pending),
                probe_capacity=self.capacity,rejected_opening_calls=self.rejected_entries,
                protection='native exit, stop, cancellation and reconciliation paths retained')


def reducing(order,portfolio):
    pos=portfolio.positions.get(order.symbol)
    qty=float(pos.quantity) if pos is not None else 0.
    amount=float(order.quantity);side=getattr(order.side,'value',order.side)
    return (math.isfinite(qty) and math.isfinite(amount) and amount>0 and
            ((side=='sell' and qty>0 and amount<=qty) or (side=='buy' and qty<0 and amount<=-qty)))


def install(runner):
    broker=vars(runner._broker).get('_target',runner._broker)
    if getattr(broker,'_paper',False) is not True:raise ValueError('Observation launcher requires PAPER')
    if getattr(runner._entry_window_checker,'_paper_observation_latch',False):raise ValueError('Observation latch already installed')
    probe=LatencyProbe()
    old_window=runner._entry_window_checker;old_reason=getattr(runner,'_entry_block_reason',lambda *args:None)
    old_admission=getattr(broker,'_strategy_admission',None)

    def window():return bool(old_window()) if _closing.get() is not None else False
    window._paper_observation_latch=True
    def reason(symbol,source):
        return old_reason(symbol,source) if _closing.get() is not None else REASON
    def admission(signal,order,price):
        if _closing.get() is not None and old_admission is not None:return old_admission(signal,order,price)
        return REASON
    runner._entry_window_checker=window;runner._entry_block_reason=reason
    broker._strategy_admission=admission
    runner._pipeline.set_scan_only(True)

    for name in ['submit','submit_urgent_exit']:
        original=getattr(broker,name)
        def wrap_submit(original):
            @wraps(original)
            def guarded(order,bar,portfolio,*args,**kwargs):
                if not reducing(order,portfolio):
                    from daytrading.execution import origin
                    from daytrading.models import OrderStatus
                    # The rollback build predates typed attempt outcomes. Both
                    # builds refuse here, before invoking native submission.
                    refused=getattr(origin,'entry_refused',None)
                    if refused is not None:refused(REASON)
                    with probe.lock:probe.rejected_entries+=1
                    return None,OrderStatus.REJECTED
                token=_closing.set(order.symbol)
                try:return original(order,bar,portfolio,*args,**kwargs)
                finally:_closing.reset(token)
            return guarded
        setattr(broker,name,wrap_submit(original))

    enqueue=runner._enqueue_market_event;process=runner._process_market_event
    @wraps(enqueue)
    def received(event):
        probe.receipt(event)
        try:
            accepted=enqueue(event)
            if not accepted:probe.remove(event)
            return accepted
        except BaseException:
            probe.remove(event);raise
    @wraps(process)
    def consumed(event):
        probe.consume(event)
        return process(event)
    runner._enqueue_market_event=received;runner._process_market_event=consumed
    publish=runner._publish_runtime_metrics;cached=[0.,None]
    @wraps(publish)
    def publish_metrics():
        value=publish();now=time.perf_counter()
        if cached[1] is None or now-cached[0]>=1.:
            cached[:]=[now,probe.snapshot()]
        runner._hub.runtime_metrics['paper_observation']=cached[1]
        runner._hub.baseline['entry_window_open']=False
        return value
    runner._publish_runtime_metrics=publish_metrics
    # Leave the normal cycle active; UI pause/resume cannot remove the latch.
    runner._hub.trading_paused=False
    return probe
