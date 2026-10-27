"""Exact native decision audit using recorded external observations.

This lane intentionally consumes broker updates. It NEVER claims predictive
fill accuracy. The predictive lane has a different broker and input view.
"""
from __future__ import annotations
from collections import deque,defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
import contextlib
import logging
import os
from .codec import pack,unpack
from .engine import ACTIVE,OPERATIONS,EXTERNAL_NAMES,EXCLUDED_STATE,NativeSessionEngine,install_scoped_clocks
from .store import digest

class Divergence(ValueError):
 def __init__(self,kind,message,expected=None,actual=None):
  super().__init__(message);self.kind=kind;self.expected=expected;self.actual=actual
 def payload(self):return dict(category=self.kind,reason=str(self),paper=self.expected,replay=self.actual)

class AuditOperation:
 def __init__(self,owner,rows):self.owner=owner;self.rows=deque(rows);self.decisions=[];self.last=None;self.first_error=None
 def fail(self,kind,message,expected=None,actual=None):
  error=Divergence(kind,message,expected,actual)
  if self.first_error is None:self.first_error=error
  raise self.first_error
 def take(self,kind):
  if self.first_error is not None:raise self.first_error
  if not self.rows:self.fail('decision','Replay produced an extra '+kind)
  row=self.rows[0];self.last=row
  if row['kind']!=kind:self.fail('decision','Recorded/replay operation ordering differs',row['kind'],kind)
  self.rows.popleft();return row['data']
 def clock(self,name,arg):
  data=self.take('clock')
  if data['name']!=name:self.fail('decision','Clock-read order differs',data['name'],name)
  return unpack(data['value'])
 def attribute(self,role,name):
  data=self.take('external_attribute')
  if (data['role'],data['name'])!=(role,name):self.fail('input','External attribute order differs',[data['role'],data['name']],[role,name])
  return unpack(data['value'])
 def external(self,role,name,args,kwargs):
  data=self.take('external_request');actual=dict(role=role,method=name,args=pack(args),kwargs=pack(kwargs))
  if data!=actual:self.fail('decision','External request differs',data,actual)
  # A broker method may contain SDK calls, clock reads and observed callbacks.
  # It is an external boundary here; the native caller still recomputes its
  # order/protection/exit request. Separate broker tests verify lifecycle logic.
  depth=1
  while self.rows:
   row=self.rows.popleft();d=row['data']
   if row['kind']=='external_request':depth+=1
   if row['kind']=='external_response':
    depth-=1
    if depth==0:
     if (d['role'],d['method'])!=(role,name):raise Divergence('capture','External response identity mismatch')
     if d.get('error'):raise RuntimeError('Recorded external failure: '+d['error'])
     return unpack(d['value'])
  raise Divergence('capture','Missing external response')
 def decision(self,kind,payload,ts=None):
  from .engine import DECISION_TYPES
  if kind not in DECISION_TYPES:return
  actual=dict(type=kind,payload=pack(payload),ts=pack(ts));expected=self.take('decision')
  if actual!=expected:self.fail('decision','Decision payload differs',expected,actual)
  self.decisions.append(actual)

class AuditDependency:
 def __init__(self,role):self.role=role
 def __bool__(self):return True
 def __getattr__(self,name):
  context=ACTIVE.get()
  if name=='client' and self.role=='_broker':return AuditDependency('broker_client')
  if name=='_client' and self.role=='_hist':return AuditDependency('hist_client')
  if context and isinstance(context,AuditOperation):
   if (context.next_kind() if hasattr(context,'next_kind') else context.rows[0]['kind'] if context.rows else None)=='external_missing':
    data=context.take('external_missing')
    if (data['role'],data['name'])!=(self.role,name):raise Divergence('input','Missing external attribute differs',data,[self.role,name])
    raise AttributeError(name)
   if (context.next_kind() if hasattr(context,'next_kind') else context.rows[0]['kind'] if context.rows else None)=='external_attribute':return context.attribute(self.role,name)
   return lambda *a,**kw:context.external(self.role,name,a,kw)
  if name in self.__dict__.get('_state',{}):return self.__dict__['_state'][name]
  if name=='_paper':return True
  if name=='is_running':return False
  if name in ('_safety_blocked','_position_sync_stale'):return False
  return lambda *a,**kw:None
 def __setattr__(self,name,value):
  if name=='role':object.__setattr__(self,name,value)
  else:self.__dict__.setdefault('_state',{})[name]=value

class AuditJournal:
 def record(self,kind,payload,ts=None):
  context=ACTIVE.get()
  if context:context.decision(kind,payload,ts)
 def candle_snapshot(self,bars,limit=30):
  from daytrading.journal.store import TradingJournal
  return TradingJournal.candle_snapshot(self,bars,limit)
 def __getattr__(self,name):return lambda *a,**kw:[]

@contextlib.contextmanager
def offline_only():
 with contextlib.ExitStack() as stack:
  stack.enter_context(patch('requests.sessions.Session.request',side_effect=AssertionError('Replay cannot contact a network provider')))
  stack.enter_context(patch('socket.create_connection',side_effect=AssertionError('Replay network access prohibited')))
  stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('Replay network access prohibited')))
  stack.enter_context(patch('socket.socket.connect_ex',side_effect=AssertionError('Replay network access prohibited')))
  # Learning writes are separate from trading decisions and never replay into
  # production datasets. Workers additionally use an isolated working directory.
  for name in ('daytrading.ml.data_collector.log_trade_outcome','daytrading.ml.shadow_collector.log_execution_quality'):
   try:stack.enter_context(patch(name))
   except (ImportError,AttributeError):pass
  yield

def construct_runner(manifest,bootstrap,dependencies=None):
 from daytrading.runner import AlpacaRunner
 from daytrading.replay.engine import shared_pipeline
 from daytrading.config import Settings
 if bootstrap.get('module_state'):
  import daytrading.runner as module
  module._FLOAT_SPLIT_SANITY_LAST_CHECK=unpack(bootstrap['module_state'])['float_split_sanity']
 cfg=Settings();roles=bootstrap.get('external_roles',[])
 deps={name:AuditDependency(name) for name in roles}
 if dependencies:deps.update(dependencies)
 recipe=dict(bootstrap.get('recipe') or {});recipe['broker']=deps.get('_broker',AuditDependency('_broker'))
 if '_float_checker' in deps:recipe['float_checker']=deps['_float_checker']
 pipeline=shared_pipeline(**recipe)
 runner=AlpacaRunner(recipe['broker'],deps.get('_hist',AuditDependency('_hist')),deps.get('_stream',AuditDependency('_stream')),pipeline,bootstrap['watchlist'],dashboard_port=0)
 runner._journal=AuditJournal();runner._hub.journal=runner._journal
 runner._sip_feed=bootstrap.get('sip_feed',False)
 for name,value in deps.items():setattr(runner,name,value)
 if hasattr(runner,'_float_checker') and runner._float_checker is not None:pipeline._float_checker=runner._float_checker
 if runner._news_checker is not None:pipeline.set_news_checker(runner._news_checker)
 # Recreate callback wiring from native code, never deserialize functions.
 if '_hod_entry_gate_alerts' in str(bootstrap['state']):runner._setup_hod_momentum(cfg,pipeline)
 external={name:getattr(runner,name,None) for name in EXTERNAL_NAMES+EXCLUDED_STATE if name!='_session_engine'}
 external['_session_engine']=None
 restored=unpack(bootstrap['state'],external,existing=dict(vars(runner)));runner.__dict__.update(restored)
 # Rebind source-defined callbacks following graph restoration.
 runner._entry_executor.set_recorder(runner._record_entry_decision)
 runner._entry_executor.set_policy(runner._entry_policy)
 runner._pipeline._entry_policy=runner._entry_policy
 runner._micro_market_now=lambda:ACTIVE.get().external('market_clock','now',(),{}) if ACTIVE.get() else None
 runner._journal=AuditJournal()
 if getattr(runner._pipeline,'_execution_timer',None) is not None:runner._wire_session_context()
 install_scoped_clocks()
 return runner

def operation_groups(events):
 groups=[];current=None
 for row in events:
  if row['kind']=='operation_start':
   if current is not None:raise Divergence('capture','Overlapping native operations')
   current=[row,[]]
  elif row['kind']=='operation_end':
   if current is None or row['operation_id']!=current[0]['operation_id']:raise Divergence('capture','Unpaired native operation')
   groups.append((current[0],current[1],row));current=None
  elif row.get('operation_id'):
   if current is None or row['operation_id']!=current[0]['operation_id']:raise Divergence('capture','Child event outside its native operation')
   current[1].append(row)
 # An active prefix can finish on a receipt or unfinished operation; don't
 # silently treat an unfinished operation as validated.
 if current is not None:raise Divergence('capture','Recorded prefix ends inside a native operation')
 return groups

def audit_session(manifest,events):
 bootstrap=next((e['data'] for e in events if e['kind']=='bootstrap'),None)
 if not bootstrap:return dict(status='blocked',first_divergence=dict(category='capture',reason='Missing initial session state'),rows=[])
 rows=[];count=0
 try:
  with offline_only():
   runner=construct_runner(manifest,bootstrap)
   engine=NativeSessionEngine(runner,None)
   for start,children,end in operation_groups(events):
    d=start['data'];method=d['method']
    if method not in OPERATIONS:raise Divergence('capture','Unsupported native operation '+method)
    context=AuditOperation(engine,children);runner._hub.trading_paused=d['paused'];runner._event_entry_lag_sec=d.get('event_lag',0.)
    if d.get('has_market_clock'):runner._market_clock=SimpleNamespace(status=lambda:dict(synchronized=True,error=None))
    token=ACTIVE.set(context)
    try:
     args=unpack(d['args'],engine.externals);kwargs=unpack(d['kwargs'],engine.externals)
     try:value=getattr(runner,method)(*args,**kwargs)
     except Divergence:raise
     except Exception as exc:
      if end['data'].get('error')!=type(exc).__name__:raise Divergence('decision','Native operation failed differently',end['data'].get('error'),type(exc).__name__)
      value=None
     if context.first_error is not None:raise context.first_error
     actual=dict(result=pack(value,engine.externals),state_digest=engine.state_digest())
     if not end['data'].get('error') and actual!=end['data']:raise Divergence('state','Native state/result differs',end['data'],actual)
     if context.rows:raise Divergence('decision','Replay omitted recorded operations',context.rows[0]['kind'],None)
     count+=1;rows.append(dict(sequence=start['sequence'],time=start['received_at'],stage=method,match=True,paper='recorded native operation',replay='exact result, decisions and risk-state match'))
    finally:ACTIVE.reset(token)
 except Divergence as exc:
  mismatch=exc.payload();mismatch['after_matched_operations']=count
  rows.append(dict(sequence=start['sequence'] if 'start' in locals() else None,time=start['received_at'] if 'start' in locals() else None,stage=method if 'method' in locals() else 'initialization',match=False,paper=exc.expected,replay=exc.actual,reason=str(exc)))
  return dict(status='failed',matched_operations=count,first_divergence=mismatch,rows=rows)
 except Exception as exc:
  return dict(status='blocked',matched_operations=count,first_divergence=dict(category='capture',reason='Replay initialization/coverage failure: '+type(exc).__name__+': '+str(exc)),rows=rows)
 return dict(status='matched',matched_operations=count,first_divergence=None,rows=rows,uses_recorded_fills=True,predictive=False)


class ConcurrentTimeline:
 """Cooperative native actors, ordered by recorded I/O yield/resume boundaries."""
 def __init__(self,events):
  import threading
  self.events=[dict(e) for e in events if e.get('operation_id') or e['kind']=='receipt'];self.index=0;self.raw_inputs=defaultdict(int);self.generated_inputs=defaultdict(list);self.verify_ingress=False;self.strict_receipts=False
  self.condition=threading.Condition();self.execution=threading.RLock();self.error=None
 def abort(self,error):
  with self.condition:
   if self.error is None:self.error=error
   self.condition.notify_all()
 def peek(self,op):
  import time
  deadline=time.monotonic()+30
  with self.condition:
   while True:
    if self.error:raise self.error
    if self.index>=len(self.events):raise Divergence('capture','Transcript ended before native operation')
    row=self.events[self.index]
    if row['operation_id']==op:return row
    if time.monotonic()>deadline:raise Divergence('capture','Replay actor stalled waiting for transcript ownership')
    self.condition.wait(.1)
 def take(self,op,kind):
  row=self.peek(op)
  if row['kind']!=kind:
   error=Divergence('decision','Recorded/replay operation ordering differs',row['kind'],kind);self.abort(error);raise error
  with self.condition:
   self.index+=1;self.last=row;self.condition.notify_all()
  return row['data']

class ConcurrentAuditOperation(AuditOperation):
 def __init__(self,owner,timeline,op):
  super().__init__(owner,[]);self.timeline=timeline;self.op=op;self.held=True
 def fail(self,kind,message,expected=None,actual=None):
  error=Divergence(kind,message,expected,actual);self.first_error=self.first_error or error;self.timeline.abort(self.first_error);raise self.first_error
 def take(self,kind):return self.timeline.take(self.op,kind)
 def clock(self,name,arg):
  while self.next_kind() in ('io_yield','io_resume'):
   kind=self.next_kind()
   if kind=='io_yield':self.take(kind);self.timeline.execution.release();self.held=False
   else:self.timeline.execution.acquire();self.held=True;self.take(kind)
  return super().clock(name,arg)
 def next_kind(self):return self.timeline.peek(self.op)['kind']
 def attribute(self,role,name):return super().attribute(role,name)
 def external(self,role,name,args,kwargs):
  expected=self.take('external_request');actual=dict(role=role,method=name,args=pack(args),kwargs=pack(kwargs))
  if 'opportunity_id' in expected:
   from daytrading.execution.origin import EXECUTION_ORIGIN
   order=args[0] if args else kwargs.get('order_data');position=self.owner.runner._pipeline.portfolio.positions.get(order.symbol)
   actual['opportunity_id']=(self.owner.order_origins.get(order.symbol) if position is not None and not position.is_flat else None) or EXECUTION_ORIGIN.get() or self.owner.order_origins.get(order.symbol)
  if actual!=expected:self.fail('decision','External request differs',expected,actual)
  depth=1
  while True:
   kind=self.next_kind()
   if kind=='io_yield':
    self.take(kind);self.timeline.execution.release();self.held=False
   elif kind=='io_resume':
    self.timeline.execution.acquire();self.held=True;self.take(kind)
   elif kind=='broker_callback':
    data=self.take(kind);getattr(self.owner.runner,data['method'])(*unpack(data['args']),**unpack(data['kwargs']))
   else:
    data=self.take(kind)
    if kind=='external_request':depth+=1
    elif kind=='external_response':
     depth-=1
     if depth==0:
      if (data['role'],data['method'])!=(role,name):self.fail('capture','External response identity mismatch')
      if data.get('side_state'):
       for key,value in unpack(data['side_state']).items():setattr(self.owner.runner._broker,key,value)
      if data.get('error'):raise RuntimeError('Recorded external failure: '+data['error'])
      value=unpack(data['value'])
      if role=='market_ingress' and name=='pending_entry_batch' and self.timeline.strict_receipts:
       for event,lag in value:
        key=digest(pack(event))
        if self.timeline.raw_inputs[key]:self.timeline.raw_inputs[key]-=1
        elif self.timeline.generated_inputs[key]:self.timeline.generated_inputs[key].pop(0)
        else:self.fail('input','Pending-entry event lacks a recorded receipt')
      if role=='_broker' and name in ('submit','submit_urgent_exit') and value and value[0] is not None and getattr(getattr(args[0],'side',None),'value',None)=='buy':self.owner.order_origins[args[0].symbol]=expected.get('opportunity_id')
      return value

class AuditEventQueue(__import__('queue').Queue):
 def __init__(self,timeline,maxsize):super().__init__(maxsize);self.timeline=timeline
 def put(self,item,block=True,timeout=None):
  context=ACTIVE.get()
  if self.timeline.verify_ingress and isinstance(context,AuditOperation):
   expected=context.take('produced_event');actual=dict(event=pack(item))
   if actual!=expected:context.fail('decision','Generated discovery/hydration event differs',expected,actual)
   self.timeline.generated_inputs[digest(actual['event'])].append(item)
   return None
  return super().put(item,block,timeout)

# Preserve the simple audit implementation for debugging old, non-interleaved
# fixtures. New sessions always use the actor-aware audit below.
sequential_audit_session=audit_session

def audit_session(manifest,events):
 import queue,threading,time
 bootstrap=next((e['data'] for e in events if e['kind']=='bootstrap'),None)
 if not bootstrap:return dict(status='blocked',first_divergence=dict(category='capture',reason='Missing initial session state'),rows=[])
 rows=[];matched=0;workers={};queues={};timeline=ConcurrentTimeline(events)
 try:
  with offline_only():
   runner=construct_runner(manifest,bootstrap);engine=NativeSessionEngine(runner,None)
   timeline.verify_ingress=bootstrap.get('verified_ingress',False);timeline.strict_receipts=bootstrap.get('strict_receipts',False)
   runner._event_queue=AuditEventQueue(timeline,runner._event_queue.maxsize)
   def actor(inbox):
    nonlocal matched
    while True:
     start=inbox.get()
     if start is None:return
     context=None;token=None
     try:
      timeline.execution.acquire();context=ConcurrentAuditOperation(engine,timeline,start['operation_id'])
      d=context.take('operation_start');method=d['method']
      if method not in OPERATIONS:raise Divergence('capture','Unsupported native operation '+method)
      runner._hub.trading_paused=d['paused'];runner._event_entry_lag_sec=d.get('event_lag',0.)
      if d.get('has_market_clock'):runner._market_clock=SimpleNamespace(status=lambda:dict(synchronized=True,error=None))
      token=ACTIVE.set(context);args=unpack(d['args'],engine.externals);kwargs=unpack(d['kwargs'],engine.externals)
      if timeline.verify_ingress and method in ('_process_market_event','_process_spy_bar'):
       from daytrading.runner import BarEvent,QuoteEvent,TradeEvent
       event=args[0] if method=='_process_market_event' else BarEvent(args[0].symbol,args[0]);key=digest(pack(event))
       if isinstance(event,(BarEvent,QuoteEvent,TradeEvent)):
        if timeline.raw_inputs[key]:timeline.raw_inputs[key]-=1
        elif timeline.strict_receipts:raise Divergence('input','Processed market event lacks a recorded receipt')
       elif timeline.generated_inputs[key]:args=(timeline.generated_inputs[key].pop(0),*args[1:])
       elif timeline.strict_receipts:raise Divergence('decision','Processed derived event was not generated by the shared engine')
      error=None
      try:value=getattr(runner,method)(*args,**kwargs)
      except Divergence:raise
      except Exception as exc:error=type(exc).__name__;value=None
      if context.first_error:raise context.first_error
      actual=dict(error=error) if error else dict(result=pack(value,engine.externals),state_digest=engine.state_digest())
      expected=context.take('operation_end')
      if actual!=expected:raise Divergence('state','Native state/result differs',expected,actual)
      if error:raise Divergence('decision','Recorded native operation failed: '+error)
      with timeline.condition:
       matched+=1
       if len(rows)<2000:rows.append(dict(sequence=start['sequence'],time=start['received_at'],stage=method,match=True,paper='recorded native operation',replay='exact result, decisions and risk-state match'))
     except Exception as exc:timeline.abort(exc if isinstance(exc,Divergence) else Divergence('capture',type(exc).__name__+': '+str(exc)))
     finally:
      if token is not None:ACTIVE.reset(token)
      if context is None or context.held:
       try:timeline.execution.release()
       except RuntimeError:pass
   last_progress=time.monotonic();previous=-1
   while timeline.index<len(timeline.events) and timeline.error is None:
    with timeline.condition:
     if timeline.index!=previous:previous=timeline.index;last_progress=time.monotonic()
     if time.monotonic()-last_progress>30:timeline.abort(Divergence('capture','Native replay made no progress'));break
     row=timeline.events[timeline.index]
     if row['kind']=='receipt':
      if not row['data'].get('derived'):timeline.raw_inputs[digest(row['data']['event'])]+=1
      timeline.index+=1;timeline.condition.notify_all();continue
     if row['kind']=='operation_start':
      actor_id=row['data'].get('actor','main')
      if actor_id not in queues:
       queues[actor_id]=queue.Queue();workers[actor_id]=threading.Thread(target=actor,args=(queues[actor_id],),daemon=True);workers[actor_id].start()
      # Dispatch each start once. Its actor consumes the header when it owns
      # the execution lock; other actors advance only at recorded yields.
      if not row.get('_dispatched'):
       row['_dispatched']=True;queues[actor_id].put(row)
     timeline.condition.wait(.01)
   for inbox in queues.values():inbox.put(None)
   for worker in workers.values():worker.join(2)
   if timeline.error:raise timeline.error
   if any(worker.is_alive() for worker in workers.values()):raise Divergence('capture','Native replay actor did not finish')
 except Exception as exc:
  failure=exc if isinstance(exc,Divergence) else Divergence('capture',type(exc).__name__+': '+str(exc))
  return dict(status='failed' if isinstance(exc,Divergence) else 'blocked',matched_operations=matched,first_divergence=dict(**failure.payload(),after_matched_operations=matched,sequence=(getattr(timeline,'last',{}) or {}).get('sequence'),time=(getattr(timeline,'last',{}) or {}).get('received_at')),rows=sorted(rows,key=lambda x:x['sequence']),uses_recorded_fills=True,predictive=False)
 return dict(status='matched',matched_operations=matched,first_divergence=None,rows=sorted(rows,key=lambda x:x['sequence']),omitted_matching_operations=max(0,matched-len(rows)),uses_recorded_fills=True,predictive=False)
