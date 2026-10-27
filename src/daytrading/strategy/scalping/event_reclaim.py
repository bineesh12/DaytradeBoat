from daytrading.models import BarSource
"""Explicitly typed trade/quote reclaim candidate. No broker or clock.

The existing native completed-bar detector remains separate. Trailing windows
below end at the observed event; they are never called native completed bars.
"""
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import isfinite, isclose

from daytrading.models import Bar, Tick, Timeframe, CompletedBar, Quote
from daytrading.data.completed_bars import NY, Rejection, validate_micro_window
from daytrading.strategy.scalping.micro_pullback import MicroCandidate, earlier_impulse
from daytrading.strategy.scalping.micro_pullback_runtime import MicroPullbackVerifier

NAME = 'prepared_reclaim_tick'

class ObservedInputs:
    """Only already-processed inputs. Equal timestamps never reveal later rows."""
    def __init__(self,size=200):
        self.quotes=deque(maxlen=size);self.ticks=deque(maxlen=size)
    def quote(self,q):self.quotes.append(q)
    def tick(self,t):self.ticks.append(t)
    def snapshot(self):return tuple(self.quotes),tuple(self.ticks)

def aware(at):
    return isinstance(at,datetime) and at.tzinfo is not None and at.utcoffset() is not None

@dataclass(frozen=True)
class TrailingTradeWindow:
    start: datetime
    end: datetime
    bar: Bar

@dataclass(frozen=True)
class EventTriggerEvidence:
    symbol: str
    setup_id: str
    observed_at: datetime
    trough: float
    trough_at: datetime
    peak: float
    peak_at: datetime
    support: float
    armed_at: datetime
    reclaimed_at: datetime
    deadline: datetime
    anchor: float
    pullback_prints: int
    ticks: tuple[Tick,...]
    native_context: tuple[CompletedBar,...]
    minute_context: tuple[Bar,...]
    quote: Quote
    windows: tuple[TrailingTradeWindow,...]

def trailing_windows(symbol,ticks,now):
    """Six equal, past-only 10s windows, explicitly event-aligned."""
    result=[]
    for i in range(6):
        start=now-timedelta(seconds=60-i*10);end=start+timedelta(seconds=10)
        rows=[t for t in ticks if t.symbol==symbol and (start<=t.ts<=end if i==5 else start<=t.ts<end)]
        if not rows:return Rejection('event_history','empty trailing 10-second window; no candle fallback')
        bar=Bar(symbol,start,rows[0].price,max(t.price for t in rows),min(t.price for t in rows),rows[-1].price,sum(t.size for t in rows),Timeframe.SEC_10, provenance=BarSource.CANONICAL_INTERVAL)
        result.append(TrailingTradeWindow(start,end,bar))
    return tuple(result)

def candidate_from_event(evidence,now):
    if not isinstance(evidence,EventTriggerEvidence):return Rejection('event_type','typed event-trigger evidence required')
    e=evidence
    times=(now,e.observed_at,e.trough_at,e.peak_at,e.armed_at,e.reclaimed_at,e.deadline,e.quote.ts)
    if not all(aware(t) for t in times):return Rejection('event_time','aware event/structure/quote timestamps required')
    if not e.trough_at<=e.peak_at<e.armed_at<e.reclaimed_at<=e.observed_at<=now<e.deadline:
        return Rejection('event_order','ordered trough/peak/dip/reclaim and original deadline required')
    if e.deadline!=min(e.armed_at+timedelta(seconds=30),e.reclaimed_at+timedelta(seconds=15)):
        return Rejection('event_deadline','original preparation/confirmation deadline changed')
    invalid=validate_micro_window(e.symbol,e.native_context,now=e.observed_at)
    if invalid:return invalid
    if (now-e.observed_at).total_seconds()>15:return Rejection('event_stale','original event confirmation expired')
    rows=e.ticks
    if not rows or any(t.symbol!=e.symbol or not aware(t.ts) or t.ts>e.observed_at or
                       t.ts.astimezone(NY).date()!=now.astimezone(NY).date() or
                       not all(isfinite(x) and x>0 for x in (t.price,t.size)) for t in rows):
        return Rejection('event_ticks','positive same-session past-only trade evidence required')
    if any(a.ts>b.ts for a,b in zip(rows,rows[1:])):return Rejection('event_ticks','unordered event trades')
    if not any(t.ts==e.peak_at and t.price==e.peak for t in rows):return Rejection('event_peak','original peak trade is missing')
    if not any(t.ts==e.trough_at and t.price==e.trough for t in rows):return Rejection('event_trough','original trough trade is missing')
    base=[t for t in rows if e.peak_at<t.ts<e.reclaimed_at]
    reclaim=next((t for t in rows if t.ts==e.reclaimed_at and t.price==e.anchor),None)
    if not base or reclaim is None or not isclose(min(t.price for t in base),e.support,abs_tol=1e-9):
        return Rejection('event_support','structural support/reclaim does not match original trades')
    if any(t.price<e.support for t in rows if t.ts>=e.reclaimed_at):
        return Rejection('event_support','original support broke after reclaim')
    pull=(e.peak-e.support)/e.peak*100
    if not (e.peak/e.trough>=1.05 and 1<=pull<=12 and e.pullback_prints>=3 and
            (e.reclaimed_at-e.armed_at).total_seconds()>=1 and e.anchor>=e.peak+.01):
        return Rejection('event_structure','impulse/pullback/maturity/reclaim rules failed')
    context=earlier_impulse(e.symbol,e.minute_context,now=e.armed_at,before=e.peak_at)
    if isinstance(context,Rejection):return context
    q=e.quote
    if (q.symbol!=e.symbol or not all(isfinite(x) for x in (q.bid,q.ask,q.bid_size,q.ask_size))
            or not 0<q.bid<q.ask or min(q.bid_size,q.ask_size)<=0
            or not 0<=(e.observed_at-q.ts).total_seconds()<=10 or q.bid<e.peak):
        return Rejection('event_quote','fresh unlocked quote holding original reclaim required')
    latest=rows[-1]
    if (e.observed_at-latest.ts).total_seconds()>1 or latest.price<e.peak+.01:
        return Rejection('event_tape','fresh reclaim trade required, not quote-only recovery')
    windows=trailing_windows(e.symbol,rows,e.observed_at)
    if isinstance(windows,Rejection):return windows
    if windows!=e.windows:return Rejection('event_windows','trailing evidence differs from original trades')
    z=windows[-1].bar
    loc=(z.close-z.low)/(z.high-z.low) if z.high>z.low else 0
    if z.close<=z.open or loc+1e-12<.65:return Rejection('event_close','trailing trigger must be green and close in top 35%')
    return MicroCandidate(e.symbol,e.reclaimed_at,e.reclaimed_at,e.anchor,round(e.support-.01,4),e.peak,loc,
                          sum(w.bar.volume for w in windows[-3:]),round(context['impulse_pct']*.8+pull*2,3))

def validate_event_guard(signal,evidence,now):
    result=candidate_from_event(evidence,now)
    if isinstance(result,Rejection):return result
    criteria=signal.scan_result.criteria
    if (signal.symbol!=evidence.symbol or criteria.get('event_setup_id')!=evidence.setup_id
            or criteria.get('setup_anchor')!=result.anchor or signal.stop_loss!=result.technical_stop):
        return Rejection('event_identity','signal does not match original event/stop/anchor')
    return [w.bar for w in evidence.windows]

class EventReclaimVerifier(MicroPullbackVerifier):
    name=NAME
    def _candidate(self,hit):
        e=hit.criteria.get('_event_evidence')
        return candidate_from_event(e,e.observed_at if isinstance(e,EventTriggerEvidence) else hit.ts)
    def verify(self,hit,portfolio,*,now=None,session_bars=None):
        e=hit.criteria.get('_event_evidence')
        invalid=candidate_from_event(e,now or hit.ts)
        if isinstance(invalid,Rejection):
            self._last_reject=invalid.code+': '+invalid.reason;return None
        quotes=self._quote_buffer.get(hit.symbol,())
        ticks=self._tick_buffer.get(hit.symbol,())
        if not quotes or quotes[-1].bid<e.peak:
            self._last_reject='event_live_quote: original reclaim is no longer held';return None
        if (not ticks or not aware(ticks[-1].ts) or
                not 0<=((now or hit.ts)-ticks[-1].ts).total_seconds()<=1 or
                ticks[-1].price<e.peak+.01):
            self._last_reject='event_live_trade: current trade no longer confirms reclaim';return None
        signal=super().verify(hit,portfolio,now=now,session_bars=session_bars)
        return replace(signal,reason='PAPER prepared trade/quote reclaim') if signal else None

class FirstExecutableReclaim:
    """Bounded, causal setup memory. Emits evidence only; cannot place an order."""
    def __init__(self,symbol,max_trades=50000):
        self.symbol=symbol;self.max_trades=max_trades;self.ticks=deque();self.quote=None
        self.last_tick=None;self.date=None;self.blocked_until=None;self.stats=Counter();self.events=deque(maxlen=100)
        self.active=None;self.low=None;self.low_at=None;self.peak=None;self.peak_at=None
        self.observed_at=None
    def reset_leg(self,tick):
        self.low=self.peak=tick.price;self.low_at=self.peak_at=tick.ts
    def close(self,reason):
        if self.active:self.events.append(dict(setup_id=self.active['id'],status=reason,time=self.observed_at.isoformat() if self.observed_at else None))
        self.active=None;self.stats[reason]+=1
        if self.last_tick:self.reset_leg(self.last_tick)
    def on_quote(self,q):
        if q.symbol!=self.symbol or not aware(q.ts):return
        if self.quote is not None and q.ts<self.quote.ts:
            self.stats['out_of_order_quote']+=1;return
        self.quote=q
        self.observed_at=q.ts
    def on_tick(self,tick,minutes):
        if tick.symbol!=self.symbol or not aware(tick.ts) or not all(isfinite(x) and x>0 for x in (tick.price,tick.size)):
            self.stats['invalid_tick']+=1;return
        if self.last_tick and tick.ts<self.last_tick.ts:
            self.close('out_of_order_tick');self.blocked_until=self.last_tick.ts+timedelta(seconds=70);return
        date=tick.ts.astimezone(NY).date()
        if self.date!=date:
            self.ticks.clear();self.active=None;self.quote=None;self.date=date;self.reset_leg(tick)
        self.last_tick=tick;self.ticks.append(tick)
        self.observed_at=tick.ts
        while self.ticks and self.ticks[0].ts<tick.ts-timedelta(seconds=70):self.ticks.popleft()
        if len(self.ticks)>self.max_trades:
            self.ticks.popleft();self.close('trade_buffer_overflow');self.blocked_until=tick.ts+timedelta(seconds=70);return
        if self.blocked_until and tick.ts<self.blocked_until:return
        if self.low_at is None or (tick.ts-self.low_at).total_seconds()>60:self.reset_leg(tick)
        if self.active:
            a=self.active
            if tick.ts>=a['deadline']:self.close('original_deadline_expired');return
            if a['reclaimed_at'] is not None:
                if tick.price<a['support']:self.close('original_support_broken')
                elif tick.price>a['anchor']*1.025:self.close('original_chase_cap_exceeded')
                return
            a['support']=min(a['support'],tick.price);a['count']+=1
            if (a['peak']-a['support'])/a['peak']*100>12:self.close('pullback_too_deep');return
            if tick.price>=a['peak']+.01:
                if a['count']>=3 and (tick.ts-a['armed_at']).total_seconds()>=1:
                    a['reclaimed_at']=tick.ts;a['anchor']=tick.price
                    a['deadline']=min(a['deadline'],tick.ts+timedelta(seconds=15));self.stats['reclaimed']+=1
                    self.events.append(dict(setup_id=a['id'],status='reclaimed',time=tick.ts.isoformat(),anchor=tick.price,support=a['support'],deadline=a['deadline'].isoformat()))
                else:self.close('reclaim_before_mature_base')
            return
        if tick.price<self.low:
            self.reset_leg(tick);return
        if tick.price>self.peak:self.peak=tick.price;self.peak_at=tick.ts
        pull=(self.peak-tick.price)/self.peak*100
        if pull>12:self.reset_leg(tick);return
        if self.peak/self.low<1.05 or pull<1:return
        context=earlier_impulse(self.symbol,minutes,now=tick.ts,before=self.peak_at)
        if isinstance(context,Rejection):self.stats[context.code]+=1;return
        sid=f'{self.symbol}|{NAME}|{self.peak_at.isoformat()}|{tick.ts.isoformat()}'
        self.active=dict(id=sid,trough=self.low,trough_at=self.low_at,peak=self.peak,peak_at=self.peak_at,
            support=tick.price,armed_at=tick.ts,reclaimed_at=None,deadline=tick.ts+timedelta(seconds=30),
            anchor=None,count=1,emitted=False)
        self.events.append(dict(setup_id=sid,status='armed',time=tick.ts.isoformat(),peak=self.peak,support=tick.price))
        self.stats['armed']+=1
    def ready(self,now,native_context,minutes):
        a=self.active;q=self.quote
        if not a or a['reclaimed_at'] is None or a['emitted'] or q is None:return None
        if not aware(now):return None
        if now>=a['deadline']:self.close('original_deadline_expired');return None
        if (not aware(now) or not aware(q.ts) or not all(isfinite(x) for x in (q.bid,q.ask,q.bid_size,q.ask_size))
                or not 0<q.bid<q.ask or min(q.bid_size,q.ask_size)<=0
                or not 0<=(now-q.ts).total_seconds()<=10 or q.bid<a['peak']):return None
        if not self.last_tick or not 0<=(now-self.last_tick.ts).total_seconds()<=1 or self.last_tick.price<a['peak']+.01:return None
        if not a.get('first_quote_at'):
            a['first_quote_at']=now
            self.events.append(dict(setup_id=a['id'],status='first_executable_quote',time=now.isoformat(),bid=q.bid,ask=q.ask))
        invalid=validate_micro_window(self.symbol,native_context,now=now)
        if invalid:
            self.stats['context_'+invalid.code]+=1;return None
        rows=tuple(self.ticks);windows=trailing_windows(self.symbol,rows,now)
        if isinstance(windows,Rejection):self.stats[windows.code]+=1;return None
        e=EventTriggerEvidence(self.symbol,a['id'],now,a['trough'],a['trough_at'],a['peak'],a['peak_at'],
            a['support'],a['armed_at'],a['reclaimed_at'],a['deadline'],a['anchor'],a['count'],rows,
            tuple(native_context[-6:]),tuple(minutes),q,windows)
        candidate=candidate_from_event(e,now)
        if isinstance(candidate,Rejection):
            self.stats[candidate.code]+=1;return None
        a['emitted']=True;self.stats['proposals']+=1
        return e


def submission_reclaim_recheck(signal, quote, ticks, now):
    """Repeat the existing event contract without resizing or approving a plan."""
    e=signal.scan_result.criteria.get('_event_evidence')
    invalid=validate_event_guard(signal,e,now)
    if isinstance(invalid,Rejection):return invalid
    if (quote is None or quote.bid<e.peak or not ticks or
            not aware(ticks[-1].ts) or not 0<=(now-ticks[-1].ts).total_seconds()<=1 or
            ticks[-1].price<e.peak+.01):
        return Rejection('event_execution','fresh trade and quote must still hold original reclaim')
    if any(t.price<e.support or t.price>e.anchor*1.025 for t in ticks if e.reclaimed_at<=t.ts<=now):
        return Rejection('event_execution','original support or chase limit invalidated')
    return None


def bounded_event_order(signal, quote, bar, *, ticks, now, limits):
    """Final causal recheck followed by the same hard-cap/risk reservation."""
    from daytrading.strategy.scalping.micro_pullback_runtime import bounded_candidate_order
    invalid=submission_reclaim_recheck(signal,quote,ticks,now)
    if invalid:return invalid
    e=signal.scan_result.criteria['_event_evidence']
    c=candidate_from_event(e,now)
    order=bounded_candidate_order(signal,c,quote,bar,now=now,limits=limits)
    if not isinstance(order,Rejection) and signal.scan_result.criteria.get("_unified_micro_entry"):
        order=replace(order,expires_at=min(e.deadline,order.expires_at or e.deadline))
        if "execution_inputs" in signal.scan_result.criteria:signal.scan_result.criteria["execution_inputs"]["original_deadline"]=order.expires_at.isoformat()
    return order


class EventReclaimScanner:
    """Main-thread input adapter. No broker, timer, discovery or guard bypass."""
    name=NAME
    accepts_cycle_time=True
    def __init__(self, *, flow=None):
        self.flow=flow
        self._now=lambda:datetime.now().astimezone()
        self._bar_aggregator=None
        self._clock_ready=lambda:False
        self.models={};self.pending={};self.rejections=Counter();self.last_rejection={}
        self.candidates=0;self.max_symbols=64
        from daytrading.strategy.scalping.micro_vwap import MicroVWAPRetries
        if flow is None:self.vwap_retries=MicroVWAPRetries()
        else:
            from daytrading.strategy.scalping.micro_flow import CoordinatedMicroRetries
            self.vwap_retries=CoordinatedMicroRetries(flow)
    @property
    def needs_recheck(self):
        if self.pending:
            return True
        if self.flow is None:
            return bool(self.vwap_retries.dirty)
        from daytrading.strategy.scalping.micro_flow import CoordinatedMicroRetries
        book = self.vwap_retries
        # Native empty retry books cannot be due. Do not observe a clock whose
        # value cannot change this result. Unknown adapters retain their path.
        if type(book) is CoordinatedMicroRetries and type(book.rows) is dict and not book.rows:
            return False
        return book.ready(self._now())
    def observe(self,event,minutes,*,eligible_symbols):
        # Bound memory to actual admitted watchlist; never subscribe a market-wide
        # candidate to execution just because its tape was visible to discovery.
        for symbol in tuple(self.models):
            if symbol not in eligible_symbols:
                self.models.pop(symbol);self.pending.pop(symbol,None)
                self.last_rejection.pop(symbol,None)
        if event.symbol not in eligible_symbols or not self._clock_ready():return
        if event.symbol not in self.models:
            if len(self.models)>=self.max_symbols:
                self.rejections['symbol_capacity']+=1;return
            self.models[event.symbol]=FirstExecutableReclaim(event.symbol)
        m=self.models[event.symbol]
        if isinstance(event,Tick):m.on_tick(event,minutes)
        elif isinstance(event,Quote):m.on_quote(event)
        else:return
        e=self.pending.get(event.symbol)
        if e and (m.active is None or m.active['id']!=e.setup_id or event.ts>=e.deadline):
            self.pending.pop(event.symbol,None)
            self.rejections['original_setup_cancelled']+=1
            self.last_rejection[event.symbol]=dict(reason='original setup cancelled by support, chase or deadline',rule='original_setup_cancelled',time=event.ts.isoformat())
        if self._bar_aggregator is None:return
        e=m.ready(event.ts,self._bar_aggregator.get_completed_10s(event.symbol,6),minutes)
        if e is not None:
            self.pending[event.symbol]=e;self.candidates+=1
    def scan(self,universe,*,now=None):
        from daytrading.models import ScanResult
        hits=[]
        if now is None or not self._clock_ready():return hits
        hits=self.vwap_retries.take(universe,now)
        for symbol,e in tuple(self.pending.items()):
            if symbol not in universe or symbol in self.vwap_retries.rows:continue
            self.pending.pop(symbol,None)
            c=candidate_from_event(e,now)
            if isinstance(c,Rejection):
                self.rejections[c.code]+=1
                self.last_rejection[symbol]=dict(reason=c.reason,rule=c.code,time=now.isoformat())
                continue
            hits.append(ScanResult(symbol,NAME,now,c.score,criteria=dict(
                pattern='micro_pullback_reclaim',entry_trigger=NAME,event_setup_id=e.setup_id,
                setup_anchor=c.anchor,stop_price=c.technical_stop,close=c.anchor,
                execution_confirmation='trade_quote_event',paper_trial=True,
                event_observed_at=e.observed_at.isoformat(),event_deadline=e.deadline.isoformat(),
                event_reclaimed_at=e.reclaimed_at.isoformat(),event_support=e.support,event_peak=e.peak,
                event_trade_count=len(e.ticks),
                _event_evidence=e,**({'_coordinated_micro_flow':True,'_unified_micro_entry':True} if self.flow is not None else {})),bars=list(e.minute_context)))
        return hits
    def status(self):
        return dict(enabled=True,paper_only=True,confirmation='first executable trade/quote reclaim',
                    candidates=self.candidates,admitted_symbols=sorted(self.models),pending=len(self.pending),
                    detector_rejections=dict(self.rejections),last_rejections=dict(self.last_rejection),
                    structure_counts={s:dict(m.stats) for s,m in self.models.items()},
                    recent_events={s:list(m.events)[-5:] for s,m in self.models.items()},
                    vwap_rechecks=self.vwap_retries.status(),
                    automatic_orders=True,exits='existing protected micro workflow',
                    validation='experimental PAPER trial; profitability not established')
