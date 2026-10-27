"""Candidate-only bounded BUY allowance. Activated only by coordinated micro evidence.

Inputs are a previously eligible intent and causal quotes, never fill outcomes.
All strategies retain original signal/chase/stop ceilings and reserved budget.
"""
from dataclasses import dataclass
from datetime import datetime,timedelta
from decimal import Decimal,ROUND_FLOOR
from math import ceil,floor,isfinite
from bisect import bisect_right

@dataclass(frozen=True)
class Intent:
 symbol:str
 at:datetime
 confirmed_at:datetime
 original_cap:float
 original_quantity:int
 technical_stop:float
 target:float
 anchor:float
 signal_window_ceiling:float
 original_budget:float
 guards_rejection:str|None=None
 max_chase_pct:float=.025
 max_stop_pct:float=.06
 min_price:float=1.
 max_price:float=20.
 max_quote_age_seconds:float=10.
 @property
 def expires_at(self):return self.confirmed_at+timedelta(seconds=15)
 def __post_init__(self):
  values=(self.original_cap,self.original_quantity,self.technical_stop,self.target,self.anchor,self.signal_window_ceiling,self.original_budget)
  if any(not isfinite(v) or v<=0 for v in values) or not 0<self.original_quantity<=500 or int(self.original_quantity)!=self.original_quantity:raise ValueError('Invalid intent')
  if not 0<self.original_budget<=50 or self.technical_stop>=self.original_cap:raise ValueError('Invalid original risk budget/stop')
  if (not all(isfinite(v) for v in (self.max_chase_pct,self.max_stop_pct,self.min_price,self.max_price)) or not 0<self.max_chase_pct<1 or not 0<self.max_stop_pct<1 or not 1<=self.min_price<=self.max_price<=20):raise ValueError('Invalid configured price/extension/stop ceilings')
  if not isfinite(self.max_quote_age_seconds) or not 0<self.max_quote_age_seconds<=10:raise ValueError('Invalid configured current quote age')
  if self.at.tzinfo is None or self.confirmed_at.tzinfo is None:raise ValueError('Aware times required')

@dataclass(frozen=True)
class BufferPlan:
 mode:str
 cap:float
 quantity:int
 reserved_risk:float
 extra_allowance:float
 requested_allowance:float
 deadline:datetime
 limiting_ceiling:str
 evidence:dict

@dataclass(frozen=True)
class Rejected:
 reason:str

def tick_floor(v):return float(Decimal(str(v)).quantize(Decimal('.01'),rounding=ROUND_FLOOR))
def percentile(values,p):
 if not values:return 0.
 return sorted(values)[max(0,ceil(len(values)*p)-1)]
def quote_rejection(q,symbol,now,*,max_age_seconds=10.):
 if q is None or q.symbol!=symbol:return 'missing or wrong-symbol quote'
 if q.ts.tzinfo is None or now.tzinfo is None:return 'timezone-aware quote required'
 if not 0<q.bid<q.ask or not all(isfinite(v) and v>0 for v in (q.bid,q.ask,q.bid_size,q.ask_size)):return 'fresh unlocked positive-size quote required'
 if not 0<=(now-q.ts).total_seconds()<=max_age_seconds:return 'stale or future quote'
 return None

def past_move_estimate(history,symbol,now,horizon):
 """Fixed 100ms sampling avoids weighting a busy quote burst more heavily.

 Both ends of every interval are before plan time; new data cannot alter a cap.
 """
 rows=sorted([q for q in history if q.symbol==symbol and not quote_rejection(q,symbol,now) and now-timedelta(seconds=10)<=q.ts<=now],key=lambda q:q.ts)
 if not rows or (rows[-1].ts-rows[0].ts).total_seconds()<2:return dict(upward_p90=0.,pairs=0,insufficient_history=True,horizon_seconds=horizon)
 times=[q.ts for q in rows];moves=[];point=rows[0].ts+timedelta(seconds=horizon)
 while point<=now:
  j=bisect_right(times,point)-1;k=bisect_right(times,point-timedelta(seconds=horizon))-1
  if k>=0 and j>=0:moves.append(max(0.,rows[j].ask-rows[k].ask))
  point+=timedelta(seconds=.1)
 return dict(upward_p90=percentile(moves,.90),pairs=len(moves),insufficient_history=len(moves)<5,horizon_seconds=horizon,last_input=rows[-1].ts.isoformat())

def plan(intent,q,history,mode,*,now,horizon):
 if mode not in ('current','fixed_5c','adaptive'):raise ValueError('Unknown frozen variant')
 if not isfinite(horizon) or horizon<=0:raise ValueError('Invalid causal matching horizon')
 if intent.guards_rejection:return Rejected('original guard: '+intent.guards_rejection)
 why=quote_rejection(q,intent.symbol,now,max_age_seconds=intent.max_quote_age_seconds)
 if why:return Rejected(why)
 if not intent.confirmed_at<=now<intent.expires_at:return Rejected('original setup expired or not confirmed')
 if q.ask<intent.technical_stop:return Rejected('quote below technical stop')
 evidence={};requested=0.
 if mode=='fixed_5c':requested=.05
 elif mode=='adaptive':
  evidence=past_move_estimate(history,intent.symbol,now,horizon)
  if not evidence['insufficient_history']:
   wanted=max(.01,(q.ask-q.bid)*.5,evidence['upward_p90'])
   requested=min(.05,q.ask*.01,max(0.,q.ask-intent.technical_stop)*.25,wanted)
 ceilings={'signal_window':intent.signal_window_ceiling,'configured_chase':intent.anchor*(1+min(.025,intent.max_chase_pct)),'configured_stop_width':intent.technical_stop/(1-min(.06,intent.max_stop_pct)),'configured_max_price':intent.max_price,'target':intent.target-.01,'requested_buffer':intent.original_cap+requested}
 limiting=min(ceilings,key=ceilings.get);cap=tick_floor(ceilings[limiting])
 # Preserve exactly the existing priced cap in the control arm.
 if mode=='current':cap=min(intent.original_cap,cap)
 if not intent.min_price<=cap<=intent.max_price or q.ask>cap+1e-12:return Rejected('current ask above bounded ceiling')
 gap=max(cap-intent.technical_stop,.05,cap*.03)
 if gap/cap>min(.06,intent.max_stop_pct)+1e-12:return Rejected('original maximum stop width')
 # Native stop calculation rounds to 0.0001: a half-step can add
 # at most $0.00005/share. Reserve that analytically for both research
 # buffer variants. Preserve the original control arm for an honest comparison.
 sizing_gap=gap+(0.00005 if mode!='current' else 0.)
 evidence['rounding_reserve_per_share']=sizing_gap-gap
 qty=min(intent.original_quantity,500,floor(intent.original_budget/sizing_gap+1e-10),floor(q.ask_size))
 if qty<1:return Rejected('no quantity within original reserved budget')
 reserve=qty*sizing_gap
 if reserve>intent.original_budget+1e-8 or reserve>50+1e-8:raise AssertionError('Risk reservation increased')
 return BufferPlan(mode,cap,qty,reserve,max(0.,cap-intent.original_cap),requested,intent.expires_at,limiting,evidence)

def refresh(existing,intent,q,*,now,entry_guard_rejection=None):
 """No sleep, quantity increase, cap ratchet or original-deadline extension.

 A production caller would have to supply its fresh native guard result. This
 research adapter is not wired to automatic orders.
 """
 if entry_guard_rejection:return Rejected('refreshed guard: '+entry_guard_rejection)
 why=quote_rejection(q,intent.symbol,now,max_age_seconds=intent.max_quote_age_seconds)
 if why:return Rejected(why)
 if now>=existing.deadline:return Rejected('original deadline expired')
 if q.ask>existing.cap+1e-12:return Rejected('fresh ask exceeds original approved cap')
 return existing
