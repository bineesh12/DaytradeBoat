"""Capture and replay the native paper runner's operations, not copied rules.

The recorder is observational: on any capture failure trading continues, and the
session is marked incomplete. Native methods remain the shared execution kernel.
"""
from __future__ import annotations
import contextvars
import concurrent.futures
from daytrading.reliability_metrics import timed_lock
from contextlib import contextmanager, nullcontext
from datetime import datetime as RealDatetime, timezone
import functools
import hashlib
import importlib
import json
import queue
import logging
import sys
import threading
import time as real_time
import types
from pathlib import Path
from .codec import pack, unpack, SECRET, StateError
from .store import SessionRecorder, session_root, digest

log=logging.getLogger(__name__)
ACTIVE=contextvars.ContextVar('paper_replay_operation',default=None)
OPERATIONS=(
 '_process_market_turn','_drain_broker_io_events',
 '_drain_events',
 '_candidate_hydration_step','_session_iteration_context','_hod_seed_step','_enforce_session_window','_initialize_session_loop','_session_step','_run_fast_scan','_run_pool_refresh','_process_candidate_hydration_batch','_process_hod_seed_batch','_scan_discovery','_process_market_event','_run_one_cycle','_execute_timed_signal','_check_exits_only',
 '_load_history','_hydrate_hod_bar_pool','_handle_fast_scan_movers',
 '_handle_market_phase_transition','_maybe_daily_session_reset',
 '_on_conflicting_stop_fill','_on_entry_wait','_on_entry_progress','_on_entry_recovered','_check_new_fills','_sync_alpaca_positions','_push_positions_from_alpaca','_sync_trade_history',
 '_retry_pipeline_protection','_on_hod_alerts_changed','_promote_hot_watch',
 '_process_breakout_scalps','_process_momentum_burst_scalps',
 '_shutdown_gracefully','_pending_entry_recheck_due','_dispatch_next_timed_signal','_check_execution_timeouts','_wire_session_context','_process_spy_bar','_initialize_trade_analysis','_run_trade_analysis',
)
EXTERNAL_NAMES=('_broker','_hist','_stream','_float_checker','_news_checker','_market_clock','_scanner_data','_scanner_trading')
EXCLUDED_STATE=('_stage_metrics','_execution_store','_hub','_journal','_session_engine','_replay_clock','_event_queue',
                '_pos_sync_thread','_hod_seed_thread','_pool_refresh_thread','_fast_scan_thread',
                '_candidate_hydrate_thread')
DECISION_TYPES=frozenset(('micro_entry_attempt','position_refresh','entry_global_control','micro_retry_transition','order_lineage','classification','scan_hit','signal','entry_decision','trade_fill','trade_exit',
                         'entry_submission_checks','entry_fill_progress','execution_timer_bar','hot_watch','mistake','market_regime'))

class _ClockDateMeta(type):
 def __instancecheck__(cls,value):return isinstance(value,RealDatetime)

class ClockDatetime(RealDatetime,metaclass=_ClockDateMeta):
 @classmethod
 def now(cls,tz=None):
  context=ACTIVE.get()
  if context:return context.clock('datetime.now',tz)
  return RealDatetime.now(tz)
 @classmethod
 def utcnow(cls):
  context=ACTIVE.get()
  if context:return context.clock('datetime.utcnow',None)
  return RealDatetime.utcnow()

class TimeProxy:
 def __getattr__(self,name):return getattr(real_time,name)
 def time(self):return self._read('time.time')
 def monotonic(self):return self._read('time.monotonic')
 def monotonic_ns(self):return self._read('time.monotonic_ns')
 def sleep(self,seconds):
  context=ACTIVE.get()
  if context:return context.clock('time.sleep',seconds)
  return real_time.sleep(seconds)
 def _read(self,name):
  context=ACTIVE.get()
  if context:return context.clock(name,None)
  return getattr(real_time,name.split('.')[-1])()

_CLOCK_MODULE_PREFIXES=('daytrading.runner','daytrading.pipeline.','daytrading.strategy.',
 'daytrading.risk.','daytrading.exits.','daytrading.execution.','daytrading.data.bar_aggregator',
 'daytrading.analytics.missed_a_plus','daytrading.analytics.trade_analyzer','daytrading.scanner.',
 'daytrading.data.market_data_service','daytrading.data.watchlist_scanner','daytrading.market_calendar')
def install_scoped_clocks():
 # Module-local proxies, not global datetime/time monkeypatches. ContextVars
 # isolate live runner, web requests and replay workers from each other.
 for name,module in list(sys.modules.items()):
  if name.startswith(('daytrading.dashboard.','daytrading.journal.','daytrading.reporting','daytrading.ml.')):
   if getattr(module,'datetime',None) is ClockDatetime:module.datetime=RealDatetime
   if isinstance(getattr(module,'time',None),TimeProxy):module.time=real_time
  if name.startswith(_CLOCK_MODULE_PREFIXES):
   if getattr(module,'datetime',None) is RealDatetime:module.datetime=ClockDatetime
   if getattr(module,'time',None) is real_time:module.time=TimeProxy()
   for attr in ('monotonic','monotonic_ns','sleep'):
    if getattr(module,attr,None) is getattr(real_time,attr):setattr(module,attr,getattr(TimeProxy(),attr))

class ExternalProxy:
 def __init__(self,target,role,owner):object.__setattr__(self,'_target',target);object.__setattr__(self,'_role',role);object.__setattr__(self,'_owner',owner)
 def __getattr__(self,name):
  target=object.__getattribute__(self,'_target')
  try:value=getattr(target,name)
  except AttributeError:
   context=ACTIVE.get()
   if context and context.owner is self._owner:context.emit('external_missing',dict(role=self._role,name=name))
   raise
  if isinstance(value,ExternalProxy):return value
  if not callable(value):
   context=ACTIVE.get()
   if context and context.owner is self._owner:
    try:context.emit('external_attribute',dict(role=self._role,name=name,value=pack(value)))
    except StateError:return ExternalProxy(value,self._role+'.'+name,self._owner)
   return value
  @functools.wraps(value)
  def call(*args,**kwargs):
   context=ACTIVE.get()
   if context and context.owner is self._owner:return context.external(self._role,name,args,kwargs,lambda:value(*args,**kwargs))
   # External observations outside the decision kernel remain evidence, but
   # aren't substituted for future strategy inputs during prediction.
   start=real_time.monotonic_ns()
   self._owner.capture_fact('background_request',dict(role=self._role,method=name,args=args,kwargs=kwargs))
   try:result=value(*args,**kwargs)
   except Exception:
    self._owner.capture_fact('background_error',dict(role=self._role,method=name));raise
   self._owner.capture_fact('background_response',dict(role=self._role,method=name,value=result,duration_ns=real_time.monotonic_ns()-start))
   return result
  return call
 def __setattr__(self,name,value):setattr(self._target,name,value)

class LiveOperation:
 def __init__(self,owner,op_id,method=None):self.owner=owner;self.op_id=op_id;self.method=method
 def emit(self,kind,data):
  try:return self.owner.recorder.record(kind,data,operation_id=self.op_id)
  except Exception as exc:self.owner.recorder.fail('capture '+type(exc).__name__);return None
 def clock(self,name,arg):
  if name=='datetime.now':value=RealDatetime.now(arg)
  elif name=='datetime.utcnow':value=RealDatetime.utcnow()
  elif name=='time.sleep':
   self.emit('io_yield',dict(role='clock',method='sleep'));self.owner.lock.release()
   try:value=real_time.sleep(arg)
   finally:self.owner.lock.acquire();self.emit('io_resume',dict(role='clock',method='sleep'))
  else:value=getattr(real_time,name.split('.')[-1])()
  self.emit('clock',dict(name=name,value=pack(value)));return value
 def external(self,role,name,args,kwargs,invoke):
  origin=None;order=args[0] if args else kwargs.get('order_data')
  if role in ('_broker','broker_client') and getattr(order,'symbol',None):
   from daytrading.execution.origin import EXECUTION_ORIGIN
   symbol=order.symbol;position=self.owner.runner._pipeline.portfolio.positions.get(symbol)
   origin=(self.owner.order_origins.get(symbol) if position is not None and not position.is_flat else None) or EXECUTION_ORIGIN.get() or self.owner.order_origins.get(symbol)
  try:
   request=dict(role=role,method=name,args=pack(args),kwargs=pack(kwargs))
   if role in ('_broker','broker_client') and getattr(order,'symbol',None):request['opportunity_id']=origin
   self.emit('external_request',request)
  except Exception as exc:self.owner.recorder.fail('external request '+type(exc).__name__)

  yielded=(role in ('broker_client','hist_client','_hist','_scanner_data','_scanner_trading','_stream','_float_checker','_news_checker')
           or (role=='_broker' and name=='get_positions' and self.method=='_push_positions_from_alpaca'
               and getattr(self.owner.runner,'_flat_tape_collection_safe',lambda:False)() is True))
  # The native broker has its own position-request/cache safety lock. Waiting
  # for a fully flat background read need not freeze tape. Open/unsettled
  # exposure and all main order/protection reads retain their boundaries. A
  # nested SDK call inside already-yielded I/O must not release twice.
  released=yielded and getattr(self.owner.lock,'_is_owned',lambda:True)()
  if released:
   self.emit('io_yield',dict(role=role,method=name));self.owner.lock.release()
  call_started_ns=real_time.monotonic_ns()
  error=None;prepared_value=None
  try:
   if role=='broker_client' and released:
    # The worker performs only SDK I/O. All result application and market/risk
    # mutation remain on the original decision thread.
    worker=getattr(self.owner,'sdk_worker',None)
    if worker is None:
     worker=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='paper-broker-io')
     self.owner.sdk_worker=worker
    future=worker.submit(invoke)
    while True:
     try:value=future.result(timeout=.02);break
     except concurrent.futures.TimeoutError:
      if future.done():value=future.result();break
      self.owner.lock.acquire();self.emit('io_resume',dict(role=role,method=name))
      try:
       self.emit('broker_callback',dict(method='_drain_broker_io_events',args=pack(()),kwargs=pack({})))
       self.owner.runner._drain_broker_io_events()
      finally:
       self.emit('io_yield',dict(role=role,method=name));self.owner.lock.release()
   else:value=invoke()
  except Exception as exc:error=exc
  finally:
   call_finished_ns=real_time.monotonic_ns();call_finished_at=RealDatetime.now(timezone.utc)
   if released:
    # These SDK snapshot responses are newly returned private objects, not
    # shared broker/risk state. Snapshot them before reacquiring the global
    # recorder lock so large discovery responses cannot stall native turns.
    # All graph values and their eventual response/processing order remain
    # recorded; transport time still ends when invoke() actually returned.
    if error is None and role=='_scanner_data' and name=='get_stock_snapshot':
     try:prepared_value=pack(value)
     except Exception as exc:self.owner.recorder.fail('external response '+type(exc).__name__)
    self.owner.lock.acquire();self.emit('io_resume',dict(role=role,method=name))
  timing=dict(duration_ns=max(0,call_finished_ns-call_started_ns),
              response_received_at=call_finished_at.isoformat(),response_received_monotonic_ns=call_finished_ns,
              processing_delay_ns=max(0,real_time.monotonic_ns()-call_finished_ns))
  if error is not None:
   self.emit('external_response',dict(role=role,method=name,error=type(error).__name__,**timing));raise error
  try:
   response=dict(role=role,method=name,value=(prepared_value if prepared_value is not None else pack(value)),**timing)
   if role=='_broker':
    target=object.__getattribute__(self.owner.runner._broker,'_target')
    response['side_state']=pack({key:getattr(target,key) for key in ('_safety_blocked','_position_sync_stale','_cancel_unconfirmed','_last_order_latency_sec') if hasattr(target,key)})
   if role=='_broker' and name in ('submit','submit_urgent_exit') and value and value[0] is not None and getattr(getattr(order,'side',None),'value',None)=='buy':self.owner.order_origins[order.symbol]=origin
   self.emit('external_response',response)
  except Exception as exc:self.owner.recorder.fail('external response '+type(exc).__name__)
  return value

class RecordedEventQueue(queue.Queue):
 def __init__(self,owner,maxsize):super().__init__(maxsize);self.owner=owner
 def put(self,item,block=True,timeout=None):
  if self.owner.recorder._failed:return super().put(item,block,timeout)
  timing=(RealDatetime.now(timezone.utc),real_time.monotonic_ns())
  with timed_lock(self.owner.capture_lock,self.owner.recorder._timings,'capture_ingress'):
   try:value=super().put(item,False)
   except queue.Full:self.owner.recorder.fail('paper event queue saturation')
   else:
    self.owner.capture_receipt(item,timing=timing)
    context=ACTIVE.get()
    if context and context.owner is self.owner and not self.owner.recorder._failed:
     try:context.emit('produced_event',dict(event=pack(item)))
     except Exception as exc:self.owner.recorder.fail('generated event capture '+type(exc).__name__)
    return value
  # Revert to the native queue's semantics after marking capture incomplete.
  # The failed-recorder path bypasses the engine lock, so exits still drain.
  return super().put(item,block,timeout)

class NativeSessionEngine:
 def __init__(self,runner,recorder):
  from .graph_digest import CanonicalGraphDigest
  self._state_encoder=CanonicalGraphDigest()
  self.runner=runner;self.recorder=recorder;self.lock=threading.RLock();self.capture_lock=threading.RLock();self.spy_inflight=0;self.bootstrapped=False;self.counter=0;self.order_origins={}
  self.externals={name:getattr(runner,name,None) for name in EXTERNAL_NAMES+EXCLUDED_STATE if name!='_session_engine'}
  self.externals['_session_engine']=self
 def restart_boundary(self):
  """Start a new capture at the decision-owner boundary, retaining the gap."""
  if ACTIVE.get() is not None:raise ValueError('Capture boundary requires an idle native operation')
  if not self.recorder._failed:raise ValueError('Capture is already healthy')
  with self.lock,self.capture_lock:
   if getattr(self,'native_inflight',0):raise ValueError('Native operation remains in flight')
   old=self.recorder
   with self.runner._event_queue.mutex:queued=list(self.runner._event_queue.queue)
   boundary=dict(previous_session_id=old.session_id,previous_committed_sequence=old.manifest['committed_sequence'],
       previous_coverage_end=old.manifest.get('coverage_end'),failures=list(old.manifest['failures']),
       gap=True,queued_inputs=pack(queued),unsettled_orders=pack(getattr(self.runner._broker,'_unsettled_orders',{})))
   new=SessionRecorder(old.root,metadata=dict(old.manifest['metadata'],capture_boundary=boundary))
   new.record('capture_boundary',boundary)
   self.recorder=new;self.bootstrapped=False;self.counter=0
   # These are boundary snapshots, explicitly identified as such. They are
   # never presented as original receipt times or backfilled old coverage.
   for event in queued:
    new.record('receipt',dict(event=pack(event),derived=False,origin='capture_boundary_snapshot'))
   old.close(timeout=.1)
   return new.session_id
 def snapshot(self):
  # Snapshot all non-I/O state; factory recreates runtime locks/callbacks.
  return pack({k:v for k,v in vars(self.runner).items() if k not in EXCLUDED_STATE and k not in EXTERNAL_NAMES and not callable(v)},self.externals)
 def state_digest(self):
  p=self.runner._pipeline
  return self._state_encoder.digest(pack(dict(portfolio=p.portfolio,tracked=p.exit_manager._positions,
      pending_entry_protection=getattr(self.runner,'_partial_entries',{}),
      fill_projections=dict(runner=getattr(self.runner,'_fill_projections',{}),
          pipeline=getattr(p,'_fill_projections',{}),portfolio=getattr(p.portfolio,'_fill_projections',{}),
          manager=getattr(p.exit_manager,'_fill_projections',{}),legs=getattr(p,'_realized_trade_legs',{}),
          realized=getattr(p,'_realized_exit_fills',{}),reported=getattr(self.runner,'_recorded_exit_fill_keys',set())),
      risk={k:getattr(p,k,None) for k in ('_daily_pnl','_symbol_entry_counts','_daily_loss_counts','_circuit_breaker_tripped')},
      prior_day_stats=getattr(self.runner,'_prior_day_stats',{}),
      trade_analysis=dict(analyzer=getattr(self.runner,'_trade_analyzer',None),
          cooldown_seconds=getattr(p,'_cooldown_seconds',None),
          exit_cooldowns=getattr(p,'_exit_cooldowns',{})),
      scanner_state={k:v for k,v in vars(self.runner._scanner).items() if k in ('_candidate_symbols','_all_symbols','_known_watchlist','_symbols_loaded_at')} if getattr(self.runner,'_scanner',None) is not None else None,
      inputs={name:{symbol:(len(rows),rows[-1] if rows else None) for symbol,rows in getattr(self.runner,name,{}).items()} for name in ('_bar_buffer','_quote_buffer','_tick_buffer')},
      watchlist=sorted(self.runner._watchlist_set),pool=list(self.runner._hod_bar_pool),
      pending=list(self.runner._exec_timer.pending_symbols),timed=list(self.runner._timed_signal_queue))))
 def invoke(self,name,original,args,kwargs):
  if ACTIVE.get() is not None:return original(*args,**kwargs)
  if self.recorder._failed:return original(*args,**kwargs)
  with self.lock:
   from zoneinfo import ZoneInfo
   from datetime import timedelta
   now=RealDatetime.now(ZoneInfo('America/New_York'))
   day=(now.date() if now.hour>=4 else (now-timedelta(days=1)).date()).isoformat()
   if (self.recorder.manifest['session_date']!=day and not self.recorder._active_ops
       and name in ('_initialize_session_loop','_session_iteration_context','_session_step','_maybe_daily_session_reset')):
    with self.capture_lock:
     if self.runner._event_queue.empty() and self.spy_inflight==0:
      old=self.recorder
      try:
       metadata=dict(old.manifest['metadata']);metadata['initial_cash']=self.runner._pipeline.portfolio.cash
       new=SessionRecorder(old.root,metadata=metadata);new.manifest['session_date']=day
       self.recorder=new;self.bootstrapped=False;self.counter=0
       old.close()
      except Exception as exc:old.fail('session rollover '+type(exc).__name__)
   if self.recorder._failed:return original(*args,**kwargs)
   if name!='_process_market_event':install_scoped_clocks()
   if not self.bootstrapped:
    try:
     clock=getattr(self.runner,'_market_clock',None)
     if clock is not None and hasattr(clock,'status'):self.capture_fact('clock_calibration',clock.status())
    except Exception as exc:
     self.recorder.fail('initial clock capture '+type(exc).__name__);return original(*args,**kwargs)
    try:self.recorder.record('bootstrap',dict(verified_ingress=True,strict_receipts=type(object.__getattribute__(self.runner._broker,'_target')).__module__=='daytrading.execution.alpaca_broker',module_state=pack(dict(float_split_sanity=vars(sys.modules['daytrading.runner']).get('_FLOAT_SPLIT_SANITY_LAST_CHECK',{}))),state=self.snapshot(),watchlist=list(self.runner._watchlist),recipe=getattr(self.runner._pipeline,'_session_recipe',{}),sip_feed=self.runner._sip_feed,external_roles=[name for name in EXTERNAL_NAMES if getattr(self.runner,name,None) is not None]))
    except Exception as exc:self.recorder.fail('bootstrap '+type(exc).__name__+': '+str(exc));return original(*args,**kwargs)
    self.bootstrapped=True
   self.counter+=1;op=f'op-{self.counter}'
   try:
    with self.capture_lock:self.recorder.record('operation_start',dict(actor=threading.current_thread().name,method=name,args=pack(args,self.externals),kwargs=pack(kwargs,self.externals),paused=bool(self.runner._hub.trading_paused),event_lag=getattr(self.runner,'_event_entry_lag_sec',0.),has_market_clock=getattr(self.runner,'_market_clock',None) is not None),operation_id=op)
   except Exception as exc:self.recorder.fail('operation input '+type(exc).__name__);return original(*args,**kwargs)
   token=ACTIVE.set(LiveOperation(self,op,name))
   self.native_inflight=getattr(self,'native_inflight',0)+1
   try:
    value=original(*args,**kwargs)
    try:
     digest_started=real_time.perf_counter_ns()
     state_digest=self.state_digest()
     self.recorder._timings.add('state_digest',real_time.perf_counter_ns()-digest_started)
     self.recorder.record('operation_end',dict(result=pack(value,self.externals),state_digest=state_digest),operation_id=op)
    except Exception as exc:self.recorder.fail('operation output '+type(exc).__name__+': '+str(exc))
    return value
   except Exception as exc:
    self.recorder.record('operation_end',dict(error=type(exc).__name__),operation_id=op);raise
   finally:
    self.native_inflight-=1
    ACTIVE.reset(token)
    if name=='_process_spy_bar':
     with self.capture_lock:self.spy_inflight=max(0,self.spy_inflight-1)
 def capture_decision(self,event_type,payload,ts):
  context=ACTIVE.get()
  if context and context.owner is self and event_type in DECISION_TYPES:
   try:context.emit('decision',dict(type=event_type,payload=pack(payload),ts=pack(ts)))
   except Exception as exc:self.recorder.fail('decision '+type(exc).__name__)
 def capture_fact(self,kind,value):
  if self.recorder._failed:return
  with self.capture_lock:
   try:self.recorder.record(kind,dict(value=pack(value)))
   except Exception as exc:self.recorder.fail('capture fact '+type(exc).__name__)
 def capture_receipt(self,event,timing=None):
  if self.recorder._failed:return
  with self.capture_lock:
   try:
    payload=getattr(event,'bar',getattr(event,'quote',getattr(event,'tick',None)))
    context=ACTIVE.get();derived=bool(context and context.owner is self)
    self.recorder.record('receipt',dict(event=pack(event),derived=derived),exchange_ts=getattr(payload,'ts',None),received_at=timing[0] if timing else None,monotonic_ns=timing[1] if timing else None)
   except Exception as exc:self.recorder.fail('receipt '+type(exc).__name__)


def shared_pipeline(**kwargs):
 from daytrading.pipeline.factory import create_scalping_pipeline
 pipeline=create_scalping_pipeline(**kwargs)
 pipeline._session_recipe={k:v for k,v in kwargs.items() if k not in ('broker','portfolio','float_checker')}
 return pipeline

def recorded_analysis_trades(runner):
 """Capture dashboard-owned analytical inputs without replacing its rules.

 The hub is intentionally outside the native state graph. Its trade list can
 influence native cooldown decisions, so record that observation explicitly.
 """
 context=ACTIVE.get()
 if isinstance(context,LiveOperation):
  return context.external('analysis_inputs','recent_trades',(),{},lambda:list(runner._hub.trades))
 if context is not None:
  return context.external('analysis_inputs','recent_trades',(),{})
 return list(runner._hub.trades)

def recorded_market_dequeue(runner):
 """Capture every FIFO choice inside a bounded native drain transaction.

 The full state checksum is retained at the drain boundary, before the native
 protection/entry heartbeat. Each input, processing order, clock and decision
 is still recorded. Prediction consumes its own causal FIFO, never these audit
 choices. No events are combined, filtered or discarded.
 """
 context=ACTIVE.get()
 if context is not None and not isinstance(context,LiveOperation) and hasattr(context,'take'):
  data=context.take('market_dequeue')
  if data['empty']:raise queue.Empty
  event=unpack(data['event']);timeline=getattr(context,'timeline',None)
  if timeline is not None and timeline.verify_ingress:
   from daytrading.runner import BarEvent,QuoteEvent,TradeEvent
   key=digest(pack(event))
   if isinstance(event,(BarEvent,QuoteEvent,TradeEvent)):
    if timeline.raw_inputs[key]:timeline.raw_inputs[key]-=1
    elif timeline.strict_receipts:context.fail('input','Drained market event lacks a recorded receipt')
   elif timeline.generated_inputs[key]:event=timeline.generated_inputs[key].pop(0)
   elif timeline.strict_receipts:context.fail('input','Drained derived event was not generated by the shared engine')
  return event,data['enqueued_at']
 # RecordedEventQueue publishes the FIFO item and its receipt under this
 # same lock. Do not let a consumer record/act on it halfway through that
 # publication, before its ingress evidence exists. Lock only the selection,
 # never the subsequent trading/protection work or a blocking queue wait.
 with timed_lock(context.owner.capture_lock,context.owner.recorder._timings,'capture_dequeue') if isinstance(context,LiveOperation) else nullcontext():
  try:event=runner._event_queue.get_nowait()
  except queue.Empty:
   if isinstance(context,LiveOperation):context.emit('market_dequeue',dict(empty=True))
   raise
  enqueued=getattr(runner,'_event_enqueued_at',{}).pop(id(event),None)
  if isinstance(context,LiveOperation):
   try:context.emit('market_dequeue',dict(empty=False,event=pack(event),enqueued_at=enqueued))
   except Exception as exc:context.owner.recorder.fail('market dequeue '+type(exc).__name__)
 return event,enqueued

def attach_capture(runner,settings=None):
 if getattr(runner._broker,'_paper',False) is not True:return None
 runner._capture_required=True
 try:
  if str(__import__('os').environ.get('DAYTRADING_SESSION_RECORDING','true')).lower() in ('false','0','no'):return None
  settings_data={k:v for k,v in vars(settings).items() if not SECRET.search(k)} if settings else {}
  recorder=SessionRecorder(session_root(),metadata=dict(paper=True,python_hash_seed=__import__('os').environ.get('PYTHONHASHSEED'),settings=pack(settings_data),environment={k:v for k,v in __import__('os').environ.items() if k.startswith('DAYTRADING_') and not SECRET.search(k) and k not in ('DAYTRADING_DATA_DIR','DAYTRADING_JOURNAL_DIR','DAYTRADING_JOURNAL_DB','DAYTRADING_SCREENSHOT_DIR')},initial_cash=runner._pipeline.portfolio.cash,scope='complete native operations; source-pinned, full-universe'))
  scanner=getattr(runner,'_scanner',None)
  if scanner is not None:
   runner._scanner_data=scanner._data;runner._scanner_trading=scanner._trading
  engine=NativeSessionEngine(runner,recorder);runner._session_engine=engine
  # Native broker remains native. Capture SDK calls separately as well as
  # high-level broker methods, including cancellation/protective operations.
  for name in EXTERNAL_NAMES:
   target=getattr(runner,name,None)
   if target is None:continue
   if name in ('_broker','_hist') and hasattr(target,'_client'):
    target._client=ExternalProxy(target._client,'broker_client' if name=='_broker' else 'hist_client',engine)
   proxy=ExternalProxy(target,name,engine);setattr(runner,name,proxy)
   engine.externals[name]=proxy
  if scanner is not None:
   scanner._data=runner._scanner_data;scanner._trading=runner._scanner_trading
  runner._hub.recording_control=lambda paused:engine.capture_fact('trading_control',dict(paused=bool(paused)))
  old_queue=runner._event_queue
  runner._event_queue=RecordedEventQueue(engine,old_queue.maxsize)
  if not old_queue.empty():recorder.fail('recorder attached after events were already queued')
  runner._pipeline._broker=runner._broker
  micro_now=runner._micro_market_now
  def recorded_market_now():
   context=ACTIVE.get()
   if context and context.owner is engine:return context.external('market_clock','now',(),{},micro_now)
   return micro_now()
  runner._micro_market_now=recorded_market_now
  if runner._float_checker is not None:runner._pipeline._float_checker=runner._float_checker
  if runner._news_checker is not None:runner._pipeline.set_news_checker(runner._news_checker)
  record=runner._journal.record
  runner._journal._execution_delivery=record
  def journal_record(event_type,payload,ts=None):
   engine.capture_decision(event_type,payload,ts)
   return record(event_type,payload,ts=ts)
  runner._journal.record=journal_record
  for name in OPERATIONS:
   original=getattr(runner,name,None)
   if original is None:continue
   def wrapped(*args,_name=name,_original=original,**kwargs):return engine.invoke(_name,_original,args,kwargs)
   setattr(runner,name,wrapped)
  target_broker=object.__getattribute__(runner._broker,'_target')
  if hasattr(target_broker,'set_execution_callbacks'):
   callback=runner._on_conflicting_stop_fill
   def recorded_callback(*args,**kwargs):
    context=ACTIVE.get()
    if context and context.owner is engine:context.emit('broker_callback',dict(method='_on_conflicting_stop_fill',args=pack(args),kwargs=pack(kwargs)))
    return callback(*args,**kwargs)
   target_broker.set_execution_callbacks(recorded_callback,getattr(target_broker,'_entry_blocker',lambda symbol:False))
  if hasattr(target_broker,'set_entry_lifecycle_callbacks'):
   def lifecycle_callback(method):
    callback=getattr(runner,method)
    def recorded(*args,**kwargs):
     context=ACTIVE.get()
     if context and context.owner is engine:context.emit('broker_callback',dict(method=method,args=pack(args),kwargs=pack(kwargs)))
     return callback(*args,**kwargs)
    return recorded
   target_broker.set_entry_lifecycle_callbacks(*(lifecycle_callback(name) for name in
       ('_on_entry_progress','_on_entry_wait','_on_entry_recovered')))
  original_shutdown=runner._shutdown_gracefully
  def shutdown(*args,**kwargs):
   try:return original_shutdown(*args,**kwargs)
   finally:
    engine.recorder.close()
    worker=getattr(engine,'sdk_worker',None)
    if worker is not None:worker.shutdown(wait=False)
  runner._shutdown_gracefully=shutdown
  install_scoped_clocks()
  return engine
 except Exception:
  log.exception('Session recorder unavailable; new entries blocked, protection continues')
  return None
