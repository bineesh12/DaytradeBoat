"""Opt-in coordinated micro evidence. Data preparation never grants entry eligibility.

One-minute/session context, completed ten-second confirmation and quote costs are
separate typed inputs. No broker, I/O, clock reads or ticker-specific rules.
"""
from daytrading.data.bar_provenance import canonical_minute
from collections import OrderedDict,Counter,deque
from dataclasses import dataclass
from datetime import datetime,timedelta,timezone
from math import isfinite,fsum
from threading import RLock
from daytrading.models import Bar,Quote,Tick,ConfirmationWindow,Timeframe
from daytrading.data.completed_bars import Rejection,validate_micro_window,NY
from daytrading.data.session_vwap import session_vwap_context
from daytrading.strategy.scalping.micro_pullback import earlier_impulse

@dataclass(frozen=True)
class QuoteCost:
    observed_at:datetime
    last_quote:Quote
    samples:tuple
    coverage_seconds:float
    mean_spread:float
    maximum_spread:float
    current_spread:float
    mean_depth:float
    invalid_seconds:float
    missing_seconds:float
    window_seconds:float

@dataclass(frozen=True)
class MicroFlowEvidence:
    symbol:str
    observed_at:datetime
    session_minutes:tuple
    confirmation:ConfirmationWindow
    quote_cost:QuoteCost
    last_trade:Tick
    session_vwap:float
    def metadata(self):
        w=self.confirmation.bars
        return dict(flow_version='coordinated-quote-recovery-v2b',context_timeframe='completed 1m/session',
            context_first=self.session_minutes[0].ts.isoformat(),context_last=self.session_minutes[-1].ts.isoformat(),
            context_count=len(self.session_minutes),confirmation_timeframe='completed 10s',
            confirmation_start=w[-1].bar.ts.isoformat(),confirmation_closed_at=w[-1].closed_at.isoformat(),
            confirmation_volume=w[-1].bar.volume,confirmation_prior_five_volume=sum(x.bar.volume for x in w[:-1])/5,
            spread_reference='valid-duration-weighted causal quote samples over at most 10s; invalid/stale intervals remain missing coverage; current ask denominator; instantaneous spread must also pass',
            quote_invalid_seconds=self.quote_cost.invalid_seconds,quote_missing_seconds=self.quote_cost.missing_seconds,quote_window_seconds=self.quote_cost.window_seconds,
            spread_price=self.quote_cost.last_quote.ask,spread_current=self.quote_cost.current_spread,
            spread_time_weighted=self.quote_cost.mean_spread,spread_coverage_seconds=self.quote_cost.coverage_seconds,
            spread_sample_count=len(self.quote_cost.samples),quote_depth_time_weighted=self.quote_cost.mean_depth,last_trade_ts=self.last_trade.ts.isoformat(),
            last_quote_ts=self.quote_cost.last_quote.ts.isoformat(),observed_at=self.observed_at.isoformat())

def aware(t):return t is not None and t.tzinfo is not None and t.utcoffset() is not None

def quote_cost(symbol,samples,now):
    if not aware(now):return Rejection('micro_flow_evidence','aware observation time required')
    rows=sorted((q for q in samples if q.symbol==symbol and aware(q.ts) and q.ts<=now),key=lambda q:q.ts)
    if not rows:return Rejection('micro_flow_evidence','missing causal quote history')
    from daytrading.strategy.entry_spread import get_policy
    policy=get_policy()
    current=rows[-1]
    if not 0<=(now-current.ts).total_seconds()<=policy.quote_max_age_seconds:return Rejection('micro_flow_evidence','stale current quote')
    if not all(isfinite(v) and v>0 for v in (current.bid,current.ask,current.bid_size,current.ask_size)) or current.bid>=current.ask:
        return Rejection('micro_flow_evidence','unlocked positive-depth current quote required')
    start=now-timedelta(seconds=policy.micro_history_seconds)
    spread_terms=[];depth_terms=[];valid_spans=[];invalid_spans=[];maximum=0.
    for j,q in enumerate(rows):
        end=rows[j+1].ts if j+1<len(rows) else now
        a=max(start,q.ts);b=min(now,end,q.ts+timedelta(seconds=policy.quote_max_age_seconds))
        if b<=a:continue
        span=(b-a).total_seconds()
        if not all(isfinite(v) and v>0 for v in (q.bid,q.ask,q.bid_size,q.ask_size)) or q.bid>=q.ask:
            # The invalid quote terminates the preceding valid interval. It
            # never contributes a zero spread, depth, or valid duration, and
            # the preceding good quote is never carried across this gap.
            invalid_spans.append(span)
            continue
        spread=q.ask-q.bid;spread_terms.append(spread*span)
        depth_terms.append(min(q.bid_size,q.ask_size)*span)
        valid_spans.append(span);maximum=max(maximum,spread)
    duration=fsum(valid_spans);invalid=fsum(invalid_spans)
    if duration<policy.micro_min_coverage_seconds:
        return Rejection('micro_flow_evidence',
            'need at least 2s valid quote coverage in original setup deadline (valid {:.3f}s, invalid {:.3f}s, missing {:.3f}s)'.format(duration,invalid,max(0.,policy.micro_history_seconds-duration-invalid)))
    selected=tuple(q for q in rows if q.ts>=start-timedelta(seconds=.1))
    return QuoteCost(now,current,selected,duration,fsum(spread_terms)/duration,
        maximum,current.ask-current.bid,fsum(depth_terms)/duration,
        invalid,max(0.,policy.micro_history_seconds-duration-invalid),policy.micro_history_seconds)

def build_evidence(symbol,minutes,window,ticks,samples,now):
    if not aware(now):return Rejection('micro_flow_evidence','aware observation time required')
    if window is None:return Rejection('micro_flow_evidence','original completed micro confirmation required')
    invalid=validate_micro_window(symbol,window.bars,now=now,max_completion_seconds=15,max_confirmation_age_seconds=15)
    if invalid:return Rejection('micro_flow_evidence',invalid.reason)
    complete=tuple(sorted((b for b in minutes if canonical_minute(b, now) and b.symbol==symbol and b.timeframe==Timeframe.MIN_1 and aware(b.ts)
        and b.ts.astimezone(NY).date()==now.astimezone(NY).date() and b.ts+timedelta(minutes=1)<=now),key=lambda b:b.ts))
    if len(complete)<3:return Rejection('micro_flow_evidence','completed same-session minute context missing')
    context=session_vwap_context(complete,symbol=symbol,now=now)
    if isinstance(context,Rejection):return Rejection('micro_flow_evidence',context.reason)
    cost=quote_cost(symbol,samples,now)
    if isinstance(cost,Rejection):return cost
    last=ticks[-1] if ticks else None
    if last is None or last.symbol!=symbol or not aware(last.ts) or not 0<=(now-last.ts).total_seconds()<=1:
        return Rejection('micro_flow_evidence','fresh same-symbol trade <=1s required')
    if not all(isfinite(v) and v>0 for v in (last.price,last.size)):return Rejection('micro_flow_evidence','invalid latest trade')
    return MicroFlowEvidence(symbol,now,tuple(context['bars']),window,cost,last,context['metadata']['vwap_reference'])

class MicroFlowCollector:
    """64 data-only preparations; 102 coalesced 100ms quote buckets per symbol.

    Original market events remain in the recorder. Coalescing only bounds this
    derived cost view; counters expose late observations and requested coverage.
    Subscription requests are explicit and never promote the trading watchlist.
    """
    def __init__(self,max_symbols=64):
        self.max_symbols=max_symbols;self.prepared=OrderedDict();self.expiry={};self.prepared_date={};self.quotes={};self.last_minute=OrderedDict();self.counts=Counter();self._lock=RLock()
    def prepare(self,symbol,minutes,now):
        if not aware(now):return False
        marker=(now.replace(second=0,microsecond=0),tuple(minutes[-12:]))
        with self._lock:
            if self.last_minute.get(symbol)==marker:return self.is_prepared(symbol,now)
        complete=[b for b in minutes if canonical_minute(b, now) and b.symbol==symbol and b.timeframe==Timeframe.MIN_1 and aware(b.ts) and b.ts+timedelta(minutes=1)<=now]
        if not complete:return False
        last=max(complete,key=lambda b:b.ts)
        with self._lock:
            self.prune(now)
            complete.sort(key=lambda b:b.ts)
            context_key=marker
            if self.last_minute.get(symbol)==context_key:return symbol in self.prepared
            self.last_minute[symbol]=context_key;self.last_minute.move_to_end(symbol)
            while len(self.last_minute)>256:self.last_minute.popitem(last=False)
            impulse=earlier_impulse(symbol,complete,now=now,before=now)
            if isinstance(impulse,Rejection) or not 1<=last.close<=20:return False
            expiry=datetime.fromisoformat(impulse['peak_completed_at'])+timedelta(minutes=5)
            self.prepared[symbol]=dict(prepared_at=now.isoformat(),expires_at=expiry.isoformat(),impulse=impulse,data_only=True);self.prepared.move_to_end(symbol)
            self.expiry[symbol]=expiry;self.prepared_date[symbol]=now.astimezone(NY).date()
            while len(self.prepared)>self.max_symbols:
                old,_=self.prepared.popitem(last=False);self.quotes.pop(old,None);self.last_minute.pop(old,None);self.expiry.pop(old,None);self.prepared_date.pop(old,None);self.counts['bounded_evictions']+=1
            self.counts['preparations']+=1;return True
    def prune(self,now):
        with self._lock:
            for symbol,row in list(self.prepared.items()):
                if now>=self.expiry[symbol] or now.astimezone(NY).date()!=self.prepared_date[symbol]:
                    self.prepared.pop(symbol);self.quotes.pop(symbol,None);self.last_minute.pop(symbol,None);self.expiry.pop(symbol,None);self.prepared_date.pop(symbol,None);self.counts['expired']+=1
    def is_prepared(self,symbol,now):
        with self._lock:
            if symbol not in self.prepared:return False
            if now>=self.expiry[symbol] or now.astimezone(NY).date()!=self.prepared_date[symbol]:
                self.prepared.pop(symbol);self.quotes.pop(symbol,None);self.last_minute.pop(symbol,None);self.expiry.pop(symbol,None);self.prepared_date.pop(symbol,None);self.counts['expired']+=1;return False
            return True
    def observe(self,event,*,observed_at=None):
        if not isinstance(event,Quote) or not aware(event.ts):return
        if observed_at is not None and (not aware(observed_at) or event.ts>observed_at):
            self.counts['future_quotes_ignored_in_derived_window']+=1;return
        with self._lock:
            reference=observed_at or event.ts
            if not self.is_prepared(event.symbol,reference):return
            if event.ts<datetime.fromisoformat(self.prepared[event.symbol]['prepared_at']):
                self.counts['preparation_predates_quote_ignored']+=1;return
            buf=self.quotes.setdefault(event.symbol,deque(maxlen=102))
            if buf and event.ts<buf[-1].ts:self.counts['late_quotes_ignored_in_derived_window']+=1;return
            bucket=lambda q:int(q.ts.timestamp()*10)
            if buf and bucket(buf[-1])==bucket(event):buf[-1]=event
            else:buf.append(event)
            while len(buf)>1 and buf[1].ts<event.ts-timedelta(seconds=10.1):buf.popleft()
    def quote_samples(self,symbol,now):
        with self._lock:return tuple(q for q in self.quotes.get(symbol,()) if q.ts<=now)
    def requested_symbols(self,now):
        self.prune(now)
        with self._lock:return tuple(self.prepared)
    def status(self):
        with self._lock:return dict(version='one-minute-ten-second-v1',data_only=True,prepared=dict(self.prepared),counts=dict(self.counts),symbols_with_quote_history=len(self.quotes),quote_buckets=sum(len(v) for v in self.quotes.values()))


def transient_reason(reason):
    from daytrading.strategy.scalping.micro_failure import temporary
    return temporary(reason)

from daytrading.strategy.scalping.micro_vwap import MicroVWAPRetries
_UNSET=object()
class CoordinatedMicroRetries(MicroVWAPRetries):
    def __init__(self,flow):
        super().__init__(retry_reason=transient_reason);self.flow=flow
    def seconds_until_due(self,now):
        if not aware(now) or not self.rows:return None
        values=[]
        from daytrading.strategy.scalping.micro_vwap import deadline
        for symbol,hit in self.rows.items():
            until=max(0.,(deadline(hit)-now).total_seconds());values.append(until)
            if symbol in self.dirty and symbol not in self.blocked:
                previous=self.checked.get(symbol)
                values.append(max(0.,.25-(now-previous).total_seconds()) if previous is not None else 0.)
        return min(values)
    def ready(self,now):
        due=self.seconds_until_due(now)
        return due is not None and due<=0
    def mark(self,event,*,now=_UNSET):
        now=event.ts if now is _UNSET else now
        if not aware(now) or not aware(event.ts) or event.ts>now:
            hit=self.rows.get(event.symbol)
            if not aware(now) and isinstance(event,Tick) and hit is not None:
                from daytrading.strategy.scalping.micro_vwap import deadline
                # Retained event/native paths are uncertain, never reanchored.
                if not aware(event.ts) or hit.ts<=event.ts<deadline(hit):
                    self.cancel(event.symbol,'original_path_uncertain',now=None)
            self.flow.counts['future_or_unaligned_event_ignored']+=1;return
        self.flow.observe(event,observed_at=now)
        return super().mark(event)


@dataclass(frozen=True)
class EventFlowEvidence:
    symbol:str
    observed_at:datetime
    session_minutes:tuple
    quote_cost:QuoteCost
    last_trade:Tick
    event:object
    def metadata(self):
        e=self.event
        return dict(flow_version='unified-micro-profile-v1',context_timeframe='completed 1m/session',
            context_first=self.session_minutes[0].ts.isoformat(),context_last=self.session_minutes[-1].ts.isoformat(),context_count=len(self.session_minutes),
            confirmation_timeframe='10s event-aligned; not native completed',confirmation_start=e.windows[-1].start.isoformat(),
            confirmation_closed_at=e.observed_at.isoformat(),confirmation_volume=e.windows[-1].bar.volume,
            confirmation_prior_five_volume=sum(w.bar.volume for w in e.windows[:-1])/5,
            spread_reference='valid-duration-weighted causal quote view; instantaneous current spread must pass',
            spread_price=self.quote_cost.last_quote.ask,spread_current=self.quote_cost.current_spread,spread_time_weighted=self.quote_cost.mean_spread,
            spread_coverage_seconds=self.quote_cost.coverage_seconds,quote_invalid_seconds=self.quote_cost.invalid_seconds,
            quote_missing_seconds=self.quote_cost.missing_seconds,quote_window_seconds=self.quote_cost.window_seconds,
            last_quote_ts=self.quote_cost.last_quote.ts.isoformat(),last_trade_ts=self.last_trade.ts.isoformat(),observed_at=self.observed_at.isoformat())

def build_hit_evidence(hit,minutes,ticks,samples,now):
    e=hit.criteria.get('_event_evidence')
    if e is None:return build_evidence(hit.symbol,minutes,hit.confirmation_window,ticks,samples,now)
    from daytrading.strategy.scalping.event_reclaim import candidate_from_event
    invalid=candidate_from_event(e,now)
    if isinstance(invalid,Rejection):return invalid
    complete=tuple(sorted((b for b in minutes if canonical_minute(b, now) and b.symbol==hit.symbol and b.timeframe==Timeframe.MIN_1 and aware(b.ts)
        and b.ts.astimezone(NY).date()==now.astimezone(NY).date() and b.ts+timedelta(minutes=1)<=now),key=lambda b:b.ts))
    if len(complete)<3:return Rejection('micro_flow_evidence','completed session context missing')
    cost=quote_cost(hit.symbol,samples,now)
    if isinstance(cost,Rejection):return cost
    last=ticks[-1] if ticks else None
    if (last is None or last.symbol!=hit.symbol or not aware(last.ts) or not 0<=(now-last.ts).total_seconds()<=1
            or last.price<e.peak+.01 or cost.last_quote.bid<e.peak):
        return Rejection('micro_flow_evidence','fresh current trade and quote must still hold original event reclaim')
    return EventFlowEvidence(hit.symbol,now,complete,cost,last,e)
