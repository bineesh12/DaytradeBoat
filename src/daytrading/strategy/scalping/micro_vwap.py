"""Fresh micro VWAP evidence and bounded retries; never submits or sizes orders."""
from collections import Counter,OrderedDict,deque
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timedelta
from math import isfinite
from daytrading.models import Bar,Tick,Quote,Timeframe
from daytrading.data.completed_bars import NY,Rejection
from daytrading.indicators.core import vwap

MICRO_NAMES=frozenset(('micro_pullback_reclaim','prepared_reclaim_tick'))
TEMPORARY_ENTRY_BLOCKS=frozenset(('broker_position_sync_stale','broker_io_busy','event_queue_stale',
    'broker_safety_pending','execution_recovery_pending','fill_reconciliation_pending',
    'protection_pending','partial_entry_pending','residual_confirmation_pending','subscription_error'))
def aware(t):return t is not None and t.tzinfo is not None and t.utcoffset() is not None

def fresh_micro_vwap(bars,ticks,quotes,*,symbol,now):
    if not aware(now):return Rejection('vwap_evidence','fresh broker-aligned observation time required')
    t=ticks[-1] if ticks else None;q=quotes[-1] if quotes else None
    if (t is None or q is None or t.symbol!=symbol or q.symbol!=symbol
            or not aware(t.ts) or not aware(q.ts)
            or not 0<=(now-t.ts).total_seconds()<=1
            or not 0<=(now-q.ts).total_seconds()<=10
            or not all(isfinite(x) and x>0 for x in (t.price,t.size,q.bid,q.ask,q.bid_size,q.ask_size))
            or q.bid>=q.ask):
        return Rejection('vwap_evidence','fresh same-symbol trade <=1s and unlocked two-sided quote <=10s required')
    from daytrading.data.session_vwap import session_vwap_context
    context=session_vwap_context(bars,symbol=symbol,now=now)
    if isinstance(context,Rejection):return context
    return dict(**context['metadata'],vwap_threshold=context['metadata']['vwap_reference']*.995,
        vwap_comparison_price=min(t.price,q.bid),vwap_trade_price=t.price,vwap_bid=q.bid,vwap_ask=q.ask,
        vwap_trade_ts=t.ts.isoformat(),vwap_quote_ts=q.ts.isoformat(),
        vwap_price_source='min(fresh trade, executable bid)')



def retryable(reason):
    text=str(reason or '').lower()
    return text.startswith(('below vwap (','vwap_evidence:'))

def deadline(hit):
    if hit.scanner_name=='prepared_reclaim_tick':
        e=hit.criteria.get('_event_evidence');return e.deadline if e is not None else None
    w=hit.confirmation_window
    return w.bars[-1].closed_at+timedelta(seconds=15) if w and w.bars else None

def identity(hit):
    return hit.criteria.get('event_setup_id') or hit.criteria.get('micro_setup_id')

class MicroVWAPRetries:
    def __init__(self, retry_reason=None):
        from daytrading.strategy.scalping.micro_failure import temporary
        self.retry_reason=retry_reason or temporary
        self.blocked={};self.terminal=OrderedDict();self.transitions=deque();self.rows={};self.dirty=set();self.latest={};self.checked={};self.stats=Counter();self.last=OrderedDict();self.max_setups=64
    def prune_terminal(self,now):
        if aware(now):
            for key,end in tuple(self.terminal.items()):
                if now>=end:self.terminal.pop(key,None)
    def transition(self,hit,state,reason,now):
        # Main-thread outbox, bounded by setup capacity and original <=15s life.
        # A full audit outbox fails closed for additional retained setups.
        if len(self.transitions)>=1024:return False
        self.transitions.append(dict(symbol=hit.symbol,route=hit.scanner_name,setup_id=identity(hit),
            state=state,reason=reason,at=now.isoformat() if aware(now) else None,
            deadline=deadline(hit).isoformat() if aware(deadline(hit)) else None))
        return True
    def hold_confirmed(self,hit,blocker,now):
        key=identity(hit);end=deadline(hit)
        invalid = hit.criteria.get('_native_setup_state', {}).get('reason')
        if invalid:
            self.finish(hit, 'native_invalidation: '+invalid, now)
            return False
        self.prune_terminal(now)
        if (not key or not aware(now) or not aware(end) or now>=end or key in self.terminal):
            self.transition(hit,'not_retained','expired_or_terminal',now);return False
        existing=self.rows.get(hit.symbol)
        if existing is not None and identity(existing)!=key:
            self.transition(hit,'not_retained','another_original_setup_pending',now);return False
        if (existing is None and len(self.rows)>=self.max_setups) or len(self.transitions)>=1024:
            self.stats['confirmed_capacity_rejected']+=1;return False
        if self.blocked.get(hit.symbol)!=blocker:
            self.transition(hit,'confirmed_waiting',blocker,now)
        if existing is None:self.rows[hit.symbol]=self.clone(hit)
        self.blocked[hit.symbol]=blocker;self.dirty.discard(hit.symbol)
        self.checked[hit.symbol]=now
        return True
    def wake_confirmed(self,blocker,now):
        self.prune_terminal(now)
        for symbol in tuple(self.blocked):
            hit=self.rows.get(symbol)
            if hit is None:self.blocked.pop(symbol,None);continue
            if now>=deadline(hit):self.cancel(symbol,'original_deadline_expired',now=now);continue
            if blocker:
                if self.blocked[symbol]!=blocker:
                    self.blocked[symbol]=blocker;self.transition(hit,'confirmed_waiting',blocker,now)
                continue
            self.blocked.pop(symbol,None);self.dirty.add(symbol)
            self.checked.pop(symbol,None)  # Gate-clear wakeup is immediate, not a new deadline.
            self.transition(hit,'revalidation_due','global_block_cleared',now)
    def finish(self,hit,reason,now):
        key=identity(hit);end=deadline(hit)
        if key and aware(end):self.terminal[key]=end
        if hit.symbol in self.rows and identity(self.rows[hit.symbol])==key:
            self.cancel(hit.symbol,reason,now=now)
        else:self.transition(hit,'terminal',reason,now)
    def note(self,symbol,value):
        self.last[symbol]=value;self.last.move_to_end(symbol)
        while len(self.last)>128:self.last.popitem(last=False)
    def clone(self,hit):
        # Market models/evidence are frozen. Only criteria metadata is mutable.
        criteria={k:(v if k in ("_event_evidence", "_native_setup_state") else deepcopy(v)) for k,v in hit.criteria.items()}
        return replace(hit,criteria=criteria,bars=list(hit.bars))
    def decide(self,hit,reason,now):
        if hit is None or hit.scanner_name not in MICRO_NAMES:return
        if hit.criteria.get('_native_setup_state',{}).get('reason'):
            self.finish(hit,'native_invalidation: '+hit.criteria['_native_setup_state']['reason'],now)
            return
        self.prune_terminal(now)
        key=identity(hit);end=deadline(hit)
        if key in self.terminal:return
        if not reason or not self.retry_reason(reason):
            if reason and key and aware(end):self.terminal[key]=end
            if hit.symbol in self.rows and identity(self.rows[hit.symbol])==key:
                self.rows.pop(hit.symbol);self.dirty.discard(hit.symbol);self.blocked.pop(hit.symbol,None)
                self.transition(hit,"verified" if not reason else "rejected",reason or "all verifier checks passed",now)
                self.latest.pop(hit.symbol,None);self.checked.pop(hit.symbol,None)
                self.note(hit.symbol,dict(status="quality_passed" if not reason else "cancelled",reason=reason or "VWAP quality passed; execution gates remain"))
            return
        if not key or not aware(now) or not aware(end) or now>=end:return
        if hit.symbol not in self.rows and len(self.rows)>=self.max_setups:return
        # Preserve the first setup, its evidence, support, price cap and deadline.
        if hit.symbol not in self.rows:
            self.rows[hit.symbol]=self.clone(hit);self.stats['retained']+=1
        if 'quality_inputs' in hit.criteria:
            self.rows[hit.symbol].criteria['quality_inputs']=deepcopy(hit.criteria['quality_inputs'])
        self.checked[hit.symbol]=now
        info=hit.criteria.get('quality_inputs',{})
        if 'vwap_trade_ts' in info and 'vwap_quote_ts' in info:
            self.latest[hit.symbol]=dict(trade=(datetime.fromisoformat(info['vwap_trade_ts']),info['vwap_trade_price']),quote=(datetime.fromisoformat(info['vwap_quote_ts']),info['vwap_bid']))
        self.note(hit.symbol,dict(reason=reason,deadline=deadline(self.rows[hit.symbol]).isoformat(),status='waiting_for_fresh_evidence'))
    def mark(self,event,*,now=None):
        hit=self.rows.get(event.symbol)
        if hit is None or not aware(event.ts):return
        end=deadline(hit)
        stamp=event.ts+timedelta(minutes=1) if isinstance(event,Bar) else event.ts
        if stamp<hit.ts:return
        if stamp>=end:
            self.cancel(event.symbol,'original_deadline_expired',now=now or stamp);return
        if isinstance(event,Tick):
            e=hit.criteria.get('_event_evidence')
            support=e.support if e is not None else float(hit.criteria.get('base_low') or hit.criteria.get('stop_price') or 0)
            anchor=float(hit.criteria.get('setup_anchor') or 0)
            if event.price<support or (anchor>0 and event.price>anchor*1.025):
                self.cancel(event.symbol,'original_support_or_chase_broken',now=now or stamp);return
        observed=self.latest.setdefault(event.symbol,{})
        if isinstance(event,Tick):observed['trade']=(event.ts,event.price)
        if isinstance(event,Quote):
            observed['quote']=(event.ts,event.bid) if 0<event.bid<event.ask and min(event.bid_size,event.ask_size)>0 else None
        if isinstance(event,(Tick,Quote)):
            threshold=hit.criteria.get('quality_inputs',{}).get('vwap_threshold')
            trade=observed.get('trade');quote=observed.get('quote')
            if threshold and (not trade or not quote or min(trade[1],quote[1])<threshold):
                self.dirty.discard(event.symbol);self.stats['still_below_or_waiting']+=1;return
            self.dirty.add(event.symbol)
        else:
            # A newly completed minute can update the VWAP reference itself.
            self.dirty.add(event.symbol)
    def cancel(self,symbol,reason,*,now=None):
        hit=self.rows.get(symbol)
        if hit is not None:
            end=deadline(hit);key=identity(hit)
            if key and aware(end):self.terminal[key]=end
            self.transition(hit,'cancelled',reason,now)
        self.blocked.pop(symbol,None)
        self.rows.pop(symbol,None);self.dirty.discard(symbol);self.latest.pop(symbol,None);self.checked.pop(symbol,None);self.stats[reason]+=1
        self.note(symbol,dict(status='cancelled',reason=reason))
    def take(self,universe,now):
        hits=[]
        for symbol,hit in tuple(self.rows.items()):
            if now>=deadline(hit):self.cancel(symbol,'original_deadline_expired',now=now);continue
            if symbol in self.blocked:continue
            if symbol not in universe or symbol not in self.dirty:continue
            if symbol in self.checked and (now-self.checked[symbol]).total_seconds()<.25:continue
            self.dirty.discard(symbol);self.stats['rechecks']+=1
            # Do not alter hit.ts, original confirmation or price/stop anchors.
            self.checked[symbol]=now
            clone=self.clone(hit);clone.criteria['_micro_vwap_recheck']=True;hits.append(clone)
            self.transition(hit,'rechecking','original_setup_freshness_and_all_guards',now)
        return hits
    def status(self):
        return dict(waiting=len(self.rows),global_blocked=dict(self.blocked),counts=dict(self.stats),last=dict(self.last))
