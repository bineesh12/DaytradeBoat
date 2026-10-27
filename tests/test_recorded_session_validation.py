from daytrading.models import BarSource
from datetime import datetime,timezone,timedelta
from pathlib import Path
from types import SimpleNamespace
import gzip,json,time
import pytest
from daytrading.models import Bar,Quote,Tick,Side,PortfolioState
from daytrading.replay.codec import pack,unpack,StateError
from daytrading.replay.store import SessionRecorder,read_session,safe_session


def test_codec_graph_identity_and_model_values():
 q=Quote('NCPL',datetime.now(timezone.utc),2.,2.01,100,100)
 original={'q':q,'alias':q,'side':Side.BUY};original['self']=original
 restored=unpack(pack(original))
 assert restored['self'] is restored and restored['q'] is restored['alias']
 assert restored['q']==q and restored['side'] is Side.BUY


def test_codec_rejects_unapproved_types_and_removes_credentials():
 with pytest.raises(StateError):pack(object())
 assert 'never-store' not in json.dumps(pack({'api_key':'never-store','alpaca_secret_key':'never-store','DAYTRADING_DASHBOARD_TOKEN':'never-store','x':1}))
 with pytest.raises(StateError):unpack({'root':{'ref':0},'nodes':[dict(kind='object',type='os:system',fields={})]})


def test_recorder_roundtrip_integrity_and_paths(tmp_path):
 r=SessionRecorder(tmp_path,metadata={'paper':True},min_free_bytes=0,chunk_size=2)
 for i in range(5):r.record('receipt',{'value':i})
 r.close();m,events=read_session(tmp_path,r.session_id)
 assert m['state']=='sealed' and [e['sequence'] for e in events]==list(range(1,6))
 with pytest.raises(ValueError):safe_session(tmp_path,'../escape')
 p=r.path/m['segments'][0]['file'];p.write_bytes(b'tampered')
 with pytest.raises(ValueError):read_session(tmp_path,r.session_id)


def test_recorder_serializes_before_later_mutation(tmp_path):
 r=SessionRecorder(tmp_path,metadata={},min_free_bytes=0);value={'x':[1]}
 r.record('receipt',value);value['x'].append(2);r.close()
 assert read_session(tmp_path,r.session_id)[1][0]['data']=={'x':[1]}


def test_bad_recording_marks_incomplete_without_raising(tmp_path):
 r=SessionRecorder(tmp_path,metadata={},min_free_bytes=0)
 assert r.record('bad',{'value':object()}) is None
 r.close();assert r.status()['state']=='incomplete'

class PaperFake:
 _paper=True
 _safety_blocked=False
 def set_execution_callbacks(self,*a):pass
 def get_positions(self):return {}
 def get_account(self):return dict(cash=25000.,equity=25000.,buying_power=25000.)

class HistFake:
 def get_bars(self,*a,**kw):return {}
class StreamFake:
 is_running=False
 _feed_str='sip'
 def subscribe(self,*a,**kw):pass
 def unsubscribe(self,*a,**kw):pass

def make_runner(tmp_path,monkeypatch):
 monkeypatch.setenv('DAYTRADING_DATA_DIR',str(tmp_path))
 monkeypatch.setenv('DAYTRADING_JOURNAL_DIR',str(tmp_path/'journal'))
 monkeypatch.setenv('DAYTRADING_SESSION_RECORDING','true')
 from daytrading.runner import AlpacaRunner
 from daytrading.replay.engine import shared_pipeline,attach_capture
 broker=PaperFake();p=shared_pipeline(broker=broker,initial_cash=25000,min_avg_volume=10000,high_liquidity_volume=50000,entry_patterns=('vwap_pullback','hod_reclaim','micro_pullback_reclaim'))
 runner=AlpacaRunner(broker,HistFake(),StreamFake(),p,['NCPL','SPY'],dashboard_port=0)
 runner._entry_window_checker=runner._is_in_trading_window
 engine=attach_capture(runner)
 assert engine is not None
 return runner,engine


def test_native_event_audit_recomputes_and_detects_changed_inputs(tmp_path,monkeypatch):
 from daytrading.runner import QuoteEvent,TradeEvent,BarEvent
 from daytrading.replay.audit import audit_session
 runner,engine=make_runner(tmp_path,monkeypatch)
 at=datetime.now(timezone.utc)
 runner._process_market_event(QuoteEvent('NCPL',Quote('NCPL',at,2.,2.01,1000,1000)))
 runner._process_market_event(TradeEvent('NCPL',Tick('NCPL',at,2.01,100,Side.BUY)))
 runner._process_market_event(BarEvent('NCPL',Bar('NCPL',at,2.,2.1,1.99,2.05,20000, provenance=BarSource.CANONICAL_INTERVAL)))
 engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 assert m['state']=='sealed',m['failures']
 result=audit_session(m,events)
 assert result['status']=='matched',result
 assert result['matched_operations']==3
 # A different close must change recomputed state/decisions, never be accepted
 # merely because expected journal rows were copied.
 end=next(e for e in events if e['kind']=='operation_end')
 end['data']['state_digest']='wrong'
 assert audit_session(m,events)['status']=='failed'


def test_capture_failure_does_not_stop_native_processing(tmp_path,monkeypatch):
 from daytrading.runner import QuoteEvent
 runner,engine=make_runner(tmp_path,monkeypatch)
 engine.recorder.fail('simulated disk problem')
 q=Quote('NCPL',datetime.now(timezone.utc),2.,2.01,100,100)
 runner._process_market_event(QuoteEvent('NCPL',q))
 assert runner._quote_buffer['NCPL'][-1]==q
 engine.recorder.close()


def test_shared_heartbeat_and_full_pipeline_audit(tmp_path,monkeypatch):
 from daytrading.runner import BarEvent,QuoteEvent
 from daytrading.replay.audit import audit_session
 from daytrading.replay.predict import PredictiveInputs,predict_session
 runner,engine=make_runner(tmp_path,monkeypatch)
 now=datetime.now(timezone.utc)
 runner._initialize_session_loop()
 for i in range(8):
  at=now-timedelta(minutes=8-i)
  runner._process_market_event(BarEvent('NCPL',Bar('NCPL',at,2.,2.02,1.99,2.,20000, provenance=BarSource.CANONICAL_INTERVAL)))
 runner._process_market_event(QuoteEvent('NCPL',Quote('NCPL',now,1.99,2.,1000,1000)))
 runner._session_step(time.time(),True,True,False,True)
 engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 assert m['state']=='sealed',m['failures']
 result=audit_session(m,events)
 assert result['status']=='matched',result
 inputs=PredictiveInputs.from_recording(events)
 assert all(not x.get('request',{}).get('role','').startswith(('_broker','broker_client')) for x in inputs.market_observations)
 predicted=predict_session(m,inputs)
 assert predicted['status']=='simulated',predicted
 assert predicted['known_fill_inputs'] is False


def test_predictive_view_excludes_broker_outcomes_and_signals():
 from daytrading.replay.predict import PredictiveInputs
 base=[dict(kind='bootstrap',data=dict(state={},watchlist=[],recipe={})),dict(kind='external_response',operation_id='x',received_at='2026-10-16T12:00:00+00:00',data=dict(role='broker_client',method='submit_order',duration_ns=100,value=pack({'filled_avg_price':99,'secret':'bad'})))]
 view=PredictiveInputs.from_recording(base)
 assert view.market_observations==() and view.transport_samples[0]==dict(available_at='2026-10-16T12:00:00+00:00',method='submit_order',duration_ns=100,role='broker_client',timing_basis='legacy_combined_io_and_processing')
 assert 'filled_avg_price' not in repr(view)
 base.append(dict(kind='operation_start',data={'method':'_execute_timed_signal'}))
 with pytest.raises(ValueError):PredictiveInputs.from_recording(base)


def test_future_quotes_are_unavailable_to_simulated_broker():
 from daytrading.replay.predict import PredictiveInputs,PredictClock,MarketView
 from daytrading.runner import QuoteEvent
 now=datetime(2026,10,16,12,tzinfo=timezone.utc)
 rows=[dict(received_at=(now+timedelta(seconds=i)).isoformat(),data=dict(event=pack(QuoteEvent('X',Quote('X',now+timedelta(seconds=i),2+i,2.01+i,100,100))))) for i in (0,1)]
 clock=PredictClock(now.isoformat(),0);view=MarketView(PredictiveInputs({},(),(),tuple(rows),()),clock)
 assert view.quote('X').ask==2.01
 clock.advance(now+timedelta(seconds=1));assert view.quote('X').ask==3.01


def test_missing_recording_never_calls_historical_loader(monkeypatch):
 from daytrading.backtest.service import run_backtest
 monkeypatch.setattr('daytrading.backtest.service.fetch_alpaca_bars_for_day',lambda *a,**kw:pytest.fail('invisible candle fallback'))
 result=run_backtest('NCPL','2026-10-16',mode='paper_validation')
 assert result['capture_completeness']=='blocked' and result['validated'] is False
 with pytest.raises(ValueError):run_backtest('NCPL','2026-10-16',mode='paper_validation',flags={'execution_timer_10s':True})


def test_dashboard_defaults_to_validation_and_lists_sessions(tmp_path,monkeypatch):
 monkeypatch.setenv('DAYTRADING_DATA_DIR',str(tmp_path))
 from daytrading.dashboard.hub import DashboardHub
 from daytrading.dashboard.server import create_app
 app=create_app(DashboardHub());client=app.test_client();html=client.get('/').get_data(as_text=True)
 assert 'value="paper_validation" selected' in html and 'Predictive execution' in html
 assert client.get('/api/backtest/sessions').json==dict(ok=True,sessions=[])
 result=client.post('/api/backtest',json=dict(symbol='NCPL',date='2026-10-16',mode='paper_validation')).json
 assert result['capture_completeness']=='blocked'
 assert client.get('/api/backtest/jobs/invalid').status_code==400


def test_worker_write_barrier_cannot_change_original_state(tmp_path):
 from daytrading.replay.isolation import worker_write_barrier
 original=tmp_path/'original';original.mkdir();outside=original/'risk.txt';outside.write_text('unchanged');job=tmp_path/'job';job.mkdir()
 with worker_write_barrier(job):
  (job/'allowed').write_text('ok')
  with pytest.raises(PermissionError):outside.write_text('corrupted')
  assert outside.read_text()=='unchanged'


def test_acceptance_requires_independent_held_out_coverage():
 from daytrading.replay.service import acceptance_gate
 assert acceptance_gate([])['status']=='collecting'
 def session(i):
  trades=[dict(entry=2.,exit=1.98 if j<2 else 2.02,paper_entry=2.,paper_exit=1.98 if j<2 else 2.02,pnl=-10 if j<2 else 10,paper_pnl=-10 if j<2 else 10,risk_budget=50.) for j in range(10)]
  return dict(session_id=str(i),role='held_out',complete_session=True,decision_parity='matched',capture_completeness='complete',model_sha256='frozen',matched_trades=trades,matched_intents=[dict(filled_qty=0,paper_filled_qty=0,cancelled=True),dict(filled_qty=0,paper_filled_qty=0,cancelled=True),dict(filled_qty=500,paper_filled_qty=500)])
 rows=[session(i) for i in range(3)]
 assert acceptance_gate(rows)['status']=='passed'
 assert acceptance_gate(rows+[rows[0]])['metrics']['trades']==30
 rows[0]['matched_trades'][0].update(pnl=40,exit=2.08)
 assert acceptance_gate(rows)['status']=='failed'


def test_source_pinned_worker_runs_offline(tmp_path,monkeypatch):
 from daytrading.runner import QuoteEvent
 from daytrading.replay.service import start_validation,job_status
 runner,engine=make_runner(tmp_path,monkeypatch);now=datetime.now(timezone.utc)
 runner._process_market_event(QuoteEvent('NCPL',Quote('NCPL',now,2.,2.01,100,100)));engine.recorder.close()
 m,_=read_session(engine.recorder.root,engine.recorder.session_id)
 job=start_validation('NCPL',m['session_date'],engine.recorder.session_id,root=engine.recorder.root)
 assert job.get('job_id'),job
 deadline=time.monotonic()+30
 while time.monotonic()<deadline:
  result=job_status(job['job_id'])
  if result['status']!='running':break
  time.sleep(.1)
 assert result['status']=='complete',result
 assert result['result']['decision_parity']=='matched',result
 assert result['result']['validated'] is False
 assert result['result']['predictions']['paper']['known_fill_inputs'] is False


def test_native_configured_runner_bootstrap_is_recordable(tmp_path,monkeypatch):
 monkeypatch.setenv('DAYTRADING_DATA_DIR',str(tmp_path));monkeypatch.setenv('DAYTRADING_JOURNAL_DIR',str(tmp_path/'journal'));monkeypatch.setenv('DAYTRADING_SESSION_RECORDING','true');monkeypatch.setenv('DAYTRADING_ENTRY_PATTERNS','vwap_pullback,first_pullback_reclaim,hod_reclaim,micro_pullback_reclaim')
 from daytrading import runner as module
 from daytrading.config import Settings
 from daytrading.runner import QuoteEvent
 class Broker(PaperFake):
  def __init__(self,*a,**kw):pass
 class Hist(HistFake):
  def __init__(self,*a,**kw):pass
 class Stream(StreamFake):
  def __init__(self,*a,**kw):pass
 monkeypatch.setattr(module,'AlpacaBroker',Broker);monkeypatch.setattr(module,'AlpacaHistoricalFeed',Hist);monkeypatch.setattr(module,'AlpacaStreamFeed',Stream)
 cfg=Settings();cfg.alpaca_api_key='fake-only-not-stored';cfg.alpaca_secret_key='fake-secret-not-stored';cfg.alpaca_feed='sip';cfg.alpaca_paper=True
 runner=module.AlpacaRunner.from_env(settings=cfg)
 engine=runner._session_engine;now=datetime.now(timezone.utc)
 runner._process_market_event(QuoteEvent('NCPL',Quote('NCPL',now,2.,2.01,100,100)));engine.recorder.close()
 m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 assert m['state']=='sealed',m['failures']
 assert 'fake-only-not-stored' not in json.dumps(m) and 'fake-secret-not-stored' not in json.dumps(events)
 from daytrading.replay.audit import audit_session
 result=audit_session(m,events)
 assert result['status']=='matched',result


def test_runtime_queues_locks_and_events_preserve_warmup():
 import queue,threading
 q=queue.PriorityQueue(3);q.put((1,{'symbol':'A'}));event=threading.Event();event.set()
 state=unpack(pack(dict(queue=q,lock=threading.Lock(),event=event)))
 with state['lock']:assert state['queue'].get_nowait()==(1,{'symbol':'A'})
 assert state['event'].is_set()


def test_virtual_scheduler_allows_crossing_io_and_censors_future():
 from daytrading.replay.predict import PredictClock
 from daytrading.replay.scheduler import VirtualScheduler
 now=datetime(2026,10,16,12,tzinfo=timezone.utc);clock=PredictClock(now.isoformat(),0);scheduler=VirtualScheduler(clock,now+timedelta(seconds=3));seen=[]
 def actor(name,delay):
  seen.append((name,'start',clock.now));clock.advance(clock.now+timedelta(seconds=delay));seen.append((name,'end',clock.now))
 scheduler.spawn('a',now,lambda:actor('a',2))
 scheduler.spawn('b',now+timedelta(seconds=.5),lambda:actor('b',.5))
 scheduler.spawn('future',now+timedelta(seconds=4),lambda:pytest.fail('future ran'))
 scheduler.run()
 assert [(a,b) for a,b,_ in seen]==[('a','start'),('b','start'),('b','end'),('a','end')]
 assert seen[2][2]==now+timedelta(seconds=1)


@pytest.fixture
def history_session_clock(monkeypatch):
 # Record the fixture clock through the real capture seam so replay consumes
 # the same clock values, even when the test host is before 04:00 New York.
 from daytrading.replay.engine import LiveOperation
 original=LiveOperation.clock
 at=datetime.now(timezone.utc).replace(hour=12,minute=0,second=0,microsecond=0)
 def clock(self,name,arg):
  if name=='datetime.now':
   value=at.astimezone(arg) if arg is not None else at.replace(tzinfo=None)
   self.emit('clock',dict(name=name,value=pack(value)))
   return value
  return original(self,name,arg)
 monkeypatch.setattr(LiveOperation,'clock',clock)


def test_delayed_concurrent_native_request_keeps_stream_processing(tmp_path,monkeypatch,history_session_clock):
 # These exercise historical I/O, not host-date calendar eligibility.
 monkeypatch.setattr('daytrading.runner.is_us_trading_day',lambda day:True)
 import threading
 from daytrading.runner import QuoteEvent
 from daytrading.replay.audit import audit_session
 runner,engine=make_runner(tmp_path,monkeypatch)
 entered=threading.Event();release=threading.Event()
 hist=object.__getattribute__(runner._hist,'_target')
 def blocked(*args,**kwargs):entered.set();assert release.wait(5);return {}
 hist.get_bars=blocked
 thread=threading.Thread(target=runner._load_history,name='history-test');thread.start()
 assert entered.wait(5)
 at=datetime.now(timezone.utc);runner._process_market_event(QuoteEvent('NCPL',Quote('NCPL',at,2.,2.01,100,100)))
 assert runner._quote_buffer['NCPL'][-1].ask==2.01
 release.set();thread.join(5);assert not thread.is_alive()
 engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 assert [e['kind'] for e in events].count('io_yield')==1
 assert audit_session(m,events)['status']=='matched'


def test_changed_quote_is_first_state_divergence(tmp_path,monkeypatch):
 from daytrading.runner import QuoteEvent
 from daytrading.replay.audit import audit_session
 runner,engine=make_runner(tmp_path,monkeypatch);now=datetime.now(timezone.utc)
 runner._process_market_event(QuoteEvent('NCPL',Quote('NCPL',now,2.,2.01,100,100)));engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 start=next(e for e in events if e['kind']=='operation_start');event=unpack(start['data']['args'])[0]
 start['data']['args']=pack((QuoteEvent('NCPL',Quote('NCPL',now,2.,2.03,100,100)),))
 result=audit_session(m,events)
 assert result['status']=='failed' and result['matched_operations']==0


def test_stale_and_out_of_order_market_events_still_audit(tmp_path,monkeypatch):
 from daytrading.runner import TradeEvent,QuoteEvent
 from daytrading.replay.audit import audit_session
 runner,engine=make_runner(tmp_path,monkeypatch);now=datetime.now(timezone.utc)
 for second in (0,-30,-1):
  at=now+timedelta(seconds=second);runner._process_market_event(TradeEvent('NCPL',Tick('NCPL',at,2.01,100,Side.BUY)))
 runner._process_market_event(QuoteEvent('NCPL',Quote('NCPL',now-timedelta(minutes=5),2.,2.01,100,100)))
 engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 assert audit_session(m,events)['status']=='matched'


def test_prediction_ignores_mutated_known_fill_outcomes():
 import copy
 from daytrading.replay.predict import PredictiveInputs
 event=dict(kind='external_response',operation_id='x',received_at='2026-10-16T12:00:00+00:00',data=dict(role='broker_client',method='submit_order',duration_ns=10,value=pack({'filled_qty':100,'filled_avg_price':2})))
 rows=[dict(kind='bootstrap',data={'state':{},'watchlist':[],'recipe':{}}),event]
 first=PredictiveInputs.from_recording(rows)
 altered=copy.deepcopy(rows);altered[-1]['data']['value']=pack({'filled_qty':9999,'filled_avg_price':5000})
 assert first==PredictiveInputs.from_recording(altered)


def test_paper_and_depth_partial_fills_are_distinct_and_causal():
 from daytrading.replay.predict import PredictiveInputs,PredictClock,MarketView,SimulatedBroker
 from daytrading.runner import QuoteEvent
 from daytrading.models import Order,OrderStatus
 now=datetime(2026,10,16,12,tzinfo=timezone.utc)
 def broker(mode):
  rows=tuple(dict(received_at=(now+timedelta(seconds=i)).isoformat(),data=dict(event=pack(QuoteEvent('X',Quote('X',now+timedelta(seconds=i),1.99,2.,100,100))))) for i in (0,1))
  clock=PredictClock(now.isoformat(),0);view=MarketView(PredictiveInputs({},(),(),rows,()),clock)
  return SimulatedBroker(view,mode,0,10000.)
 order=Order('X',Side.BUY,250);bar=Bar('X',now,2.,2.,2.,2.,1000, provenance=BarSource.CANONICAL_INTERVAL);portfolio=PortfolioState(cash=10000.)
 paper=broker('paper');fill,status=paper.submit(order,bar,portfolio)
 assert fill.quantity==250 and fill.price==2. and status is OrderStatus.FILLED
 depth=broker('depth');fill,status=depth.submit(order,bar,portfolio)
 assert fill.quantity==100 and any(e['kind']=='cancel' and e['remaining']==150 for e in depth.events)
 assert depth.positions['X']['qty']==100


def test_market_clock_calibration_does_not_use_tape_timestamps():
 from daytrading.replay.predict import PredictiveInputs,PredictClock,MarketView
 now=datetime(2026,10,16,12,tzinfo=timezone.utc)
 calibration=dict(received_at=now.isoformat(),data=dict(value=pack(dict(synchronized=True,offset_seconds=.02,sample_age_seconds=1))))
 clock=PredictClock(now.isoformat(),0);view=MarketView(PredictiveInputs({},(),(),(),(),None,(calibration,)),clock)
 assert view.market_now()==now+timedelta(seconds=.02)
 clock.advance(now+timedelta(seconds=181));assert view.market_now() is None


def test_recorder_disk_failure_is_explicit_and_native_keeps_running(tmp_path,monkeypatch):
 r=SessionRecorder(tmp_path,metadata={},min_free_bytes=10**30,chunk_size=1)
 r.record('receipt',{'x':1});r.close()
 assert r.manifest['state']=='incomplete' and 'disk reserve' in repr(r.manifest['failures'])


def test_acceptance_does_not_count_restarts_as_new_session_dates():
 from daytrading.replay.service import acceptance_gate
 rows=[dict(session_id=str(i),session_date='2026-10-16',role='held_out',complete_session=True) for i in range(3)]
 assert acceptance_gate(rows)['metrics']['sessions']==1


def test_stop_fill_during_cancel_prevents_new_entry_and_oversell():
 from daytrading.replay.predict import PredictiveInputs,PredictClock,MarketView,SimulatedBroker
 from daytrading.runner import QuoteEvent
 from daytrading.models import Order,OrderStatus
 now=datetime(2026,10,16,14,tzinfo=timezone.utc)
 rows=tuple(dict(received_at=(now+timedelta(seconds=i)).isoformat(),data=dict(event=pack(QuoteEvent('X',Quote('X',now+timedelta(seconds=i),bid,bid+.01,1000,1000))))) for i,bid in ((0,2.),(1,1.8)))
 clock=PredictClock(now.isoformat(),0);view=MarketView(PredictiveInputs({},(),(),rows,()),clock);broker=SimulatedBroker(view,'paper',0,10000.)
 broker.positions['X']=dict(qty=100.,avg_entry=2.);oid=broker.place_protective_stop('X',100,1.9)
 captured=[];broker.set_execution_callbacks(lambda *args:captured.append(args),lambda s:False)
 # Get-orders and cancel latency make the lower quote arrive while canceling.
 fill,status=broker.submit(Order('X',Side.BUY,100),Bar('X',now,2.,2.,2.,2.,1000, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=10000.))
 assert fill is None and status is OrderStatus.REJECTED and broker.positions=={}
 assert len(captured)==1 and broker.orders[oid]['status']=='filled'


def test_partial_exit_restores_remaining_protection():
 from daytrading.replay.predict import PredictiveInputs,PredictClock,MarketView,SimulatedBroker
 from daytrading.runner import QuoteEvent
 from daytrading.models import Order
 now=datetime(2026,10,16,14,tzinfo=timezone.utc)
 row=dict(received_at=now.isoformat(),data=dict(event=pack(QuoteEvent('X',Quote('X',now,2.,2.01,40,1000)))))
 clock=PredictClock(now.isoformat(),0);view=MarketView(PredictiveInputs({},(),(),(row,),()),clock);broker=SimulatedBroker(view,'depth',0,10000.)
 broker.positions['X']=dict(qty=100.,avg_entry=2.);broker.place_protective_stop('X',100,1.9)
 fill,_=broker.submit(Order('X',Side.SELL,100),Bar('X',now,2.,2.,2.,2.,1000, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=10000.))
 assert fill.quantity==40 and broker.positions['X']['qty']==60
 oid=broker.get_protective_stop_id('X');assert broker.orders[oid]['qty']==60


def test_cumulative_stop_evidence_is_not_double_counted():
 from daytrading.replay.comparison import protection_evidence
 now=datetime(2026,10,16,14,tzinfo=timezone.utc)
 def event(i,oid,side,qty,filled,status,stop=None):
  value=SimpleNamespace(id=oid,symbol='X',side=side,qty=qty,filled_qty=filled,status=status,stop_price=stop)
  return dict(kind='external_response',received_at=(now+timedelta(seconds=i)).isoformat(),data=dict(role='broker_client',value=pack(value)))
 rows=[event(0,'buy','buy',100,40,'partially_filled'),event(.1,'buy','buy',100,100,'filled'),event(.2,'stop','sell',100,0,'new',1.9),event(.3,'buy','buy',100,100,'filled'),event(1,'stop','sell',100,100,'filled',1.9)]
 evidence=protection_evidence(rows)
 assert evidence['status']=='observed' and not evidence['failures']


@pytest.mark.parametrize('exit_price',[2.10,1.90])
def test_native_broker_callback_audits_winner_and_loss(tmp_path,monkeypatch,exit_price):
 from daytrading.models import Fill,Position
 from daytrading.exits.manager import TrackedPosition
 from daytrading.replay.audit import audit_session
 runner,engine=make_runner(tmp_path,monkeypatch)
 runner._pipeline.portfolio.positions['NCPL']=Position('NCPL',100,2.)
 runner._pipeline.exit_manager.track(TrackedPosition(symbol='NCPL',side=Side.BUY,quantity=100,entry_price=2.,stop_loss=1.9))
 now=datetime.now(timezone.utc)
 runner._on_conflicting_stop_fill(Fill('NCPL',Side.SELL,100,exit_price,now),'broker-stop',0.)
 engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 result=audit_session(m,events)
 assert result['status']=='matched',result
 assert runner._pipeline._daily_pnl==pytest.approx((exit_price-2)*100)
 assert any(e['kind']=='decision' and e['data']['type']=='trade_exit' for e in events)


def test_queue_overflow_marks_gap_and_does_not_raise_into_trading(tmp_path,monkeypatch):
 import threading
 ready=threading.Event();writer=SessionRecorder._writer
 def delayed(self):ready.wait(5);writer(self)
 monkeypatch.setattr(SessionRecorder,'_writer',delayed)
 r=SessionRecorder(tmp_path,metadata={},min_free_bytes=0,max_queue=1)
 assert r.record('receipt',{'x':1})==1
 assert r.record('receipt',{'x':2}) is None
 ready.set();r.close()
 assert r.manifest['state']=='incomplete' and 'queue overflow' in repr(r.manifest['failures'])


def test_validation_does_not_take_an_overlapping_unfinished_prefix(tmp_path):
 from daytrading.replay.service import start_validation
 r=SessionRecorder(tmp_path,metadata={},min_free_bytes=0)
 r.record('operation_start',{},operation_id='a');r.record('operation_start',{},operation_id='b');r.record('operation_end',{},operation_id='a');r.close()
 result=start_validation('X',r.manifest['session_date'],r.session_id,root=tmp_path)
 assert result['capture_completeness']=='blocked'


def test_posthoc_comparison_includes_scale_outs_and_stop_fills():
 from daytrading.replay.comparison import compare_execution
 from daytrading.models import Order,Fill,OrderStatus
 now=datetime(2026,10,16,14,tzinfo=timezone.utc)
 def row(kind,data,i,op='entry'):
  return dict(kind=kind,data=data,operation_id=op,received_at=(now+timedelta(seconds=i)).isoformat())
 events=[row('bootstrap',{},0),row('external_request',dict(role='_broker',method='submit',args=pack((Order('X',Side.BUY,100),)),kwargs=pack({}),opportunity_id='setup'),.1),row('external_response',dict(role='_broker',method='submit',value=pack((Fill('X',Side.BUY,100,2.,now),OrderStatus.FILLED))),.2)]
 stop=SimpleNamespace(id='s',symbol='X',side='sell',type='stop',qty=100,filled_qty=100,filled_avg_price=1.9,submitted_at=now+timedelta(seconds=1),filled_at=now+timedelta(seconds=2))
 events.append(row('external_response',dict(role='broker_client',method='get_order_by_id',value=pack(stop)),2))
 prediction={'events':[dict(kind='order_request',opportunity_id='setup',id='b',symbol='X',side='buy',qty=100,limit=2.,time=now.isoformat()),dict(kind='fill',id='b',symbol='X',side='buy',qty=100,price=2.,time=now.isoformat()),dict(kind='fill',opportunity_id='setup',id='s1',symbol='X',side='sell',qty=100,price=1.9,time=(now+timedelta(seconds=2)).isoformat(),reason='protective_stop')]}
 result=compare_execution(events,prediction,risk_budget=50.)
 assert not result['unmatched_intents'] and len(result['matched_trades'])==1
 assert result['matched_trades'][0]['paper_pnl']==pytest.approx(-10.)


def test_clock_telemetry_does_not_short_circuit_native_heartbeat(tmp_path,monkeypatch):
 from daytrading.replay.audit import audit_session
 runner,engine=make_runner(tmp_path,monkeypatch)
 runner._market_clock=SimpleNamespace(status=lambda:dict(synchronized=True,error=None),now=lambda:datetime.now(timezone.utc))
 runner._initialize_session_loop();runner._session_step(time.time(),False,False,False,False)
 engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 result=audit_session(m,events)
 assert result['status']=='matched',result


def test_unfinished_native_actor_cannot_seal_as_complete(tmp_path):
 r=SessionRecorder(tmp_path,metadata={},min_free_bytes=0)
 r.record('operation_start',{'method':'_load_history'},operation_id='pending');r.close()
 assert r.manifest['state']=='incomplete'


def test_sdk_daily_bar_reference_is_recorded_and_recomputed(tmp_path,monkeypatch,history_session_clock):
 # These exercise historical I/O, not host-date calendar eligibility.
 monkeypatch.setattr('daytrading.runner.is_us_trading_day',lambda day:True)
 from alpaca.data.models import BarSet
 from alpaca.data.timeframe import TimeFrame
 from daytrading.replay.audit import audit_session
 class Client:
  def get_stock_bars(self,request):
   now=datetime.now(timezone.utc)-timedelta(days=1)
   return BarSet({'NCPL':[{'t':now.isoformat(),'o':2.,'h':2.3,'l':1.9,'c':2.1,'v':10000,'n':100,'vw':2.}]})
 monkeypatch.setattr(HistFake,'_client',Client(),raising=False);monkeypatch.setattr(HistFake,'_feed','sip',raising=False)
 runner,engine=make_runner(tmp_path,monkeypatch);runner._load_history();engine.recorder.close()
 m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 assert runner._prior_day_stats['NCPL'].prior_close==2.1
 assert any(e['kind']=='external_response' and e['data']['role']=='hist_client' for e in events)
 result=audit_session(m,events)
 assert result['status']=='matched',result
 restored=unpack(pack(TimeFrame.Day));assert (restored.amount,restored.unit)==(TimeFrame.Day.amount,TimeFrame.Day.unit)


def test_byte_budget_is_bounded_and_marks_incomplete(tmp_path):
 r=SessionRecorder(tmp_path,metadata={},min_free_bytes=0,max_queue_bytes=64)
 assert r.record('receipt',{'blob':'x'*200}) is None
 r.close();assert r.manifest['state']=='incomplete' and 'byte capacity' in repr(r.manifest['failures'])


def test_native_discovery_cache_clock_and_sdk_inputs_replay(tmp_path,monkeypatch):
 from daytrading.data.watchlist_scanner import WatchlistScanner
 from daytrading.replay import engine as module
 from daytrading.replay.audit import audit_session
 original=module.attach_capture
 class Assets:
  def get_all_assets(self,request):return [SimpleNamespace(symbol='NCPL',exchange='NASDAQ',tradable=True,name='Netcapital common stock')]
 class Data:
  def get_stock_snapshot(self,request):return {}
 def attach(runner,*args,**kwargs):
  scanner=WatchlistScanner('fake-key-only','fake-secret-only',feed='sip');scanner._trading=Assets();scanner._data=Data();runner._scanner=scanner
  return original(runner,*args,**kwargs)
 monkeypatch.setattr(module,'attach_capture',attach)
 runner,engine=make_runner(tmp_path,monkeypatch);runner._scan_discovery('scan');engine.recorder.close()
 m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 assert m['state']=='sealed',m['failures']
 assert runner._scanner._all_symbols==['NCPL']
 result=audit_session(m,events)
 assert result['status']=='matched',result


def test_predictive_sdk_matching_normalizes_enums_without_future_bars():
 from alpaca.data.requests import StockBarsRequest
 from alpaca.data.timeframe import TimeFrame
 from alpaca.data.enums import DataFeed
 from alpaca.data.models import BarSet
 from daytrading.replay.predict import PredictiveInputs,PredictClock,MarketView
 now=datetime(2026,10,16,12,tzinfo=timezone.utc)
 request=StockBarsRequest(symbol_or_symbols=['X'],timeframe=TimeFrame.Day,feed=DataFeed.SIP,start=now-timedelta(days=5),end=now)
 response=BarSet({'X':[{'t':(now+timedelta(seconds=i)).isoformat(),'o':2.,'h':2.3,'l':1.9,'c':2.1,'v':10000,'n':100,'vw':2.} for i in (-60,60)]})
 observation=dict(operation_id='warmup',available_at=now.isoformat(),request=dict(role='hist_client',method='get_stock_bars',args=pack((request,)),kwargs=pack({})),value=pack(response),error=None)
 inputs=PredictiveInputs({'sip_feed':True},(),(observation,),(),())
 view=MarketView(inputs,PredictClock(now.isoformat(),0))
 result=view.external('hist_client','get_stock_bars',(request,),{})
 assert len(result.data['X'])==1 and result.data['X'][0].timestamp==now-timedelta(seconds=60)
 assert not view.failures


def test_day_rotation_resumes_native_loop_without_observed_signals(tmp_path,monkeypatch):
 from daytrading.replay.store import SessionRecorder
 from daytrading.replay.predict import PredictiveInputs,predict_session
 from daytrading.replay.audit import audit_session
 runner,engine=make_runner(tmp_path,monkeypatch)
 runner._wire_session_context();runner._initialize_session_loop()
 old=engine.recorder;old.close()
 engine.recorder=SessionRecorder(old.root,metadata=old.manifest['metadata'],min_free_bytes=0)
 engine.bootstrapped=False;engine.counter=0
 runner._session_step(time.time(),False,False,False,False)
 engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 assert not any(e['kind']=='operation_start' and e['data']['method']=='_initialize_session_loop' for e in events)
 assert audit_session(m,events)['status']=='matched'
 result=predict_session(m,PredictiveInputs.from_recording(events))
 assert result['status']=='simulated',result
 assert result['initialization']=='resumed native loop snapshot' and result['processed_operations']>0


def test_new_segment_captures_existing_clock_anchor(tmp_path,monkeypatch):
 from daytrading.replay.predict import PredictiveInputs,PredictClock,MarketView
 runner,engine=make_runner(tmp_path,monkeypatch)
 runner._market_clock=SimpleNamespace(status=lambda:dict(synchronized=True,offset_seconds=.02,sample_age_seconds=1,error=None))
 runner._initialize_session_loop();engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 view=PredictiveInputs.from_recording(events)
 first=view.operations[0];clock=PredictClock(first['received_at'],first['monotonic_ns'])
 assert view.calibrations
 # Monotonic and wall timestamps are sampled separately; their microsecond
 # sampling skew is preserved instead of silently treating both as one clock.
 assert abs(MarketView(view,clock).market_now()-(clock.now+timedelta(seconds=.02)))<timedelta(milliseconds=1)


def test_failed_recorder_does_not_lock_native_protection_path(tmp_path,monkeypatch):
 import threading
 from daytrading.runner import QuoteEvent
 runner,engine=make_runner(tmp_path,monkeypatch);engine.recorder.fail('disk full')
 held=threading.Event();release=threading.Event();done=threading.Event()
 def holder():
  with engine.lock:held.set();release.wait(5)
 lock_thread=threading.Thread(target=holder);lock_thread.start();assert held.wait(2)
 now=datetime.now(timezone.utc)
 worker=threading.Thread(target=lambda:(runner._process_market_event(QuoteEvent('NCPL',Quote('NCPL',now,2.,2.01,100,100))),done.set()));worker.start()
 try:assert done.wait(1),'capture failure serialized native processing behind recorder lock'
 finally:release.set();lock_thread.join(2);worker.join(2);engine.recorder.close()


def test_late_cumulative_limit_fill_overrides_initial_unknown_outcome():
 from daytrading.replay.comparison import observed_intents
 from daytrading.models import Order,OrderStatus
 from alpaca.trading.requests import LimitOrderRequest
 now=datetime(2026,10,16,14,tzinfo=timezone.utc)
 def event(kind,data,i):return dict(kind=kind,data=data,operation_id='entry',received_at=(now+timedelta(seconds=i)).isoformat())
 request=LimitOrderRequest(symbol='X',qty=100,side='buy',time_in_force='day',limit_price=2.,client_order_id='causal-request')
 order=SimpleNamespace(id='sdk-id',client_order_id='causal-request',symbol='X',side='buy',type='limit',status='filled',qty=100,filled_qty=100,filled_avg_price=1.99,submitted_at=now,filled_at=now+timedelta(seconds=3))
 events=[event('external_request',dict(role='_broker',method='submit',args=pack((Order('X',Side.BUY,100),)),kwargs=pack({}),opportunity_id='setup'),0),event('external_request',dict(role='broker_client',method='submit_order',args=pack(()),kwargs=pack(dict(order_data=request))),.1),event('external_response',dict(role='_broker',method='submit',value=pack((None,OrderStatus.NEW))),1),event('external_response',dict(role='broker_client',method='get_order_by_client_id',value=pack(order)),3)]
 rows=observed_intents(events)
 assert len(rows)==1 and rows[0]['filled_qty']==100 and rows[0]['price']==1.99 and rows[0]['broker_id']=='sdk-id'


def test_native_virtual_event_wakes_worker_before_periodic_timeout():
 from daytrading.replay.scheduler import VirtualScheduler,VirtualEvent
 from daytrading.replay.predict import PredictClock
 now=datetime(2026,10,16,12,tzinfo=timezone.utc);clock=PredictClock(now.isoformat(),0);scheduler=VirtualScheduler(clock,now+timedelta(seconds=3));event=VirtualEvent(scheduler);observed=[]
 def worker():
  assert event.wait(600);observed.append(clock.now);event.clear()
 def source():clock.advance(now+timedelta(seconds=1));event.set()
 scheduler.spawn('pool',now,worker);scheduler.spawn('market',now,source);scheduler.run()
 assert observed==[now+timedelta(seconds=1)]


def test_historical_orders_do_not_claim_new_session_protection():
 from daytrading.replay.comparison import protection_evidence
 now=datetime(2026,10,16,21,tzinfo=timezone.utc)
 order=SimpleNamespace(id='historical',symbol='NCPL',side='buy',status='filled',qty=100,filled_qty=100,stop_price=None,submitted_at=now-timedelta(hours=5))
 events=[dict(kind='external_response',received_at=now.isoformat(),data=dict(role='broker_client',value=pack([order])))]
 result=protection_evidence(events)
 assert result['status']=='unverified' and result['failures']==[]


def test_generated_hydration_is_recomputed_not_reinjected(tmp_path,monkeypatch):
 from daytrading.replay.audit import audit_session
 now=datetime.now(timezone.utc)
 monkeypatch.setattr(HistFake,'get_bars',lambda self,*a,**kw:{'BENF':[Bar('BENF',now,2.,2.1,1.99,2.05,12345, provenance=BarSource.CANONICAL_INTERVAL)]})
 monkeypatch.setattr(HistFake,'last_fetch_failures',0,raising=False)
 runner,engine=make_runner(tmp_path,monkeypatch)
 runner._process_hod_seed_batch(['BENF'])
 event=runner._event_queue.get_nowait();runner._process_market_event(event)
 engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 produced=next(e for e in events if e['kind']=='produced_event')
 assert audit_session(m,events)['status']=='matched'
 produced['data']['event']=pack({'tampered':'hydration output'})
 result=audit_session(m,events)
 assert result['status']=='failed' and result['first_divergence']['category']=='decision',result


def test_strict_audit_requires_actual_raw_receipt(tmp_path,monkeypatch):
 from daytrading.runner import QuoteEvent
 from daytrading.replay.audit import audit_session
 runner,engine=make_runner(tmp_path,monkeypatch)
 event=QuoteEvent('NCPL',Quote('NCPL',datetime.now(timezone.utc),2.,2.01,100,100))
 runner._event_queue.put_nowait(event);runner._process_market_event(runner._event_queue.get_nowait())
 engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 next(e for e in events if e['kind']=='bootstrap')['data']['strict_receipts']=True
 assert audit_session(m,events)['status']=='matched'
 events=[e for e in events if e['kind']!='receipt']
 result=audit_session(m,events)
 assert result['status']=='failed' and result['first_divergence']['category']=='input',result


def test_rollover_failure_keeps_native_session_running(tmp_path,monkeypatch):
 from daytrading.replay import engine as module
 runner,engine=make_runner(tmp_path,monkeypatch)
 old=engine.recorder;old.manifest['session_date']='2000-01-01'
 def unavailable(*a,**kw):raise OSError('disk unavailable')
 monkeypatch.setattr(module,'SessionRecorder',unavailable)
 runner._initialize_session_loop()
 assert old._failed and old.manifest['state']=='incomplete'
 assert hasattr(runner,'_session_cycle_count')
 old.close()


def test_spy_receipt_tracks_inflight_and_matches_audit(tmp_path,monkeypatch):
 from daytrading.replay.audit import audit_session
 runner,engine=make_runner(tmp_path,monkeypatch)
 runner._on_bar(Bar('SPY',datetime.now(timezone.utc),600.,601.,599.,600.5,100000, provenance=BarSource.CANONICAL_INTERVAL))
 runner._drain_events()
 assert engine.spy_inflight==0
 engine.recorder.close();m,events=read_session(engine.recorder.root,engine.recorder.session_id)
 next(e for e in events if e['kind']=='bootstrap')['data']['strict_receipts']=True
 result=audit_session(m,events)
 assert result['status']=='matched',result


def test_unrecordable_generated_event_does_not_change_native_queue(tmp_path,monkeypatch):
 runner,engine=make_runner(tmp_path,monkeypatch)
 item=object()
 engine.invoke('_initialize_session_loop',lambda:runner._event_queue.put_nowait(item),(),{})
 assert runner._event_queue.get_nowait() is item
 assert engine.recorder._failed and engine.recorder.manifest['state']=='incomplete'
 engine.recorder.close()


def test_acceptance_fingerprint_includes_discovery_and_data_build():
 from daytrading.replay.service import prediction_fingerprint
 import copy
 manifest={'source':{'source_files':{'daytrading/data/watchlist_scanner.py':'a','daytrading/replay/predict.py':'x'}},'metadata':{}}
 changed=copy.deepcopy(manifest);changed['source']['source_files']['daytrading/data/watchlist_scanner.py']='b'
 assert prediction_fingerprint(manifest)!=prediction_fingerprint(changed)
