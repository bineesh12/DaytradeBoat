"""Source-pinned offline replay jobs and explicit validation status."""
from __future__ import annotations
import hashlib,json,os,re,subprocess,sys,threading,time,uuid,zipfile
from pathlib import Path
from .store import session_root,safe_session,read_session,list_sessions,atomic,digest,dependencies

_JOBS={};_LOCK=threading.Lock()

def blocked(symbol,day,reason,*,category='capture'):
 return dict(ok=True,mode='paper_validation',symbol=symbol,date=day,label='Paper validation — blocked',capture_completeness='blocked',decision_parity='not_run',execution_validation='not_run',validated=False,first_divergence=dict(category=category,reason=reason),comparison_rows=[],bars_data=[],charts_10s=[],round_trips=[],trades=[],scorecard={},message=reason)

def start_validation(symbol,day,session_id,*,root=None):
 root=Path(root or session_root())
 if not session_id:return blocked(symbol,day,'Select a recorded paper session; historical candles cannot substitute for missing recordings.')
 try:
  path=safe_session(root,session_id);manifest=json.loads((path/'manifest.json').read_text())
  if manifest.get('schema_version')!=1:raise ValueError('Unsupported recorded session schema')
 except (OSError,ValueError,KeyError) as exc:return blocked(symbol,day,'Recorded session unavailable or invalid: '+str(exc))
 if manifest['session_date']!=day:return blocked(symbol,day,'Selected recording belongs to a different session date.')
 if manifest['state']=='incomplete' or manifest.get('failures'):return blocked(symbol,day,'Recording is incomplete: '+'; '.join(manifest.get('failures') or ['capture failed']))
 end=manifest.get('committed_operation_sequence')
 if end is None:
  # Compatibility only for early capture schema; never substitute candles.
  _,events=read_session(root,session_id,manifest=manifest);active=set();end=0
  for event in events:
   if event['kind']=='operation_start':active.add(event['operation_id'])
   elif event['kind']=='operation_end':
    active.discard(event['operation_id'])
    if not active:end=event['sequence']
 if not end:return blocked(symbol,day,'Recording has no completed quiescent native operation yet.')
 artifact=root/'artifacts'/manifest['source']['artifact']
 if artifact.name!=manifest['source']['build_id']+'.zip' or artifact.is_symlink() or not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest()!=manifest['source']['artifact_sha256']:
  return blocked(symbol,day,'Frozen source artifact is missing or its checksum differs.',category='version')
 if manifest.get('dependencies')!=dependencies():return blocked(symbol,day,'Frozen session dependencies differ from the available worker environment.',category='version')
 with _LOCK:
  if any(j['status']=='running' for j in _JOBS.values()):return dict(ok=False,error='Another recorded-session replay is running')
  job_id=uuid.uuid4().hex;jobdir=root/'jobs'/job_id;jobdir.mkdir(parents=True)
  atomic(jobdir/'request.json',dict(symbol=symbol,date=day,session_id=session_id,manifest=manifest,max_sequence=end))
  _JOBS[job_id]=dict(ok=True,job_id=job_id,status='running',mode='paper_validation',session_id=session_id,started_at=time.time())
 thread=threading.Thread(target=_run_job,args=(job_id,jobdir,root,artifact,manifest),daemon=True);thread.start()
 return dict(_JOBS[job_id])

def _run_job(job_id,jobdir,root,artifact,manifest):
 try:_run_job_impl(job_id,jobdir,root,artifact,manifest)
 except Exception as exc:
  with _LOCK:_JOBS[job_id].update(status='failed',error=str(exc))

def _run_job_impl(job_id,jobdir,root,artifact,manifest):
 # Only sanitized strategy settings enter this process. Credentials, .env files,
 # mutable live state and provider clients are never inherited.
 env={k:v for k,v in os.environ.items() if k in ('PATH','LANG','LC_ALL','SYSTEMROOT')}
 env.update({k:str(v) for k,v in manifest['metadata'].get('environment',{}).items() if not re.search(r'(KEY|SECRET|TOKEN|PASSWORD|AUTH|_DIR$|_DB$|_PATH$|_FILE$)',k,re.I)})
 source=jobdir/'source';source.mkdir()
 with zipfile.ZipFile(artifact) as archive:
  assert set(archive.namelist())==set(manifest['source']['source_files']),'Frozen source member mismatch'
  for name in archive.namelist():
   target=source/name
   if target.resolve().is_relative_to(source.resolve()) is not True:raise ValueError('Invalid source archive path')
   data=archive.read(name)
   if hashlib.sha256(data).hexdigest()!=manifest['source']['source_files'][name]:raise ValueError('Frozen source file checksum differs')
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 env.update(PYTHONPATH=str(source.resolve()),PYTHONHASHSEED=str(manifest['metadata'].get('python_hash_seed') or '0'),DAYTRADING_DATA_DIR=str((jobdir/'data').resolve()),DAYTRADING_CONFIG_LOADED='true',DAYTRADING_REPLAY_ISOLATED='true',TMPDIR=str(jobdir.resolve()))
 try:
  with (jobdir/'worker.log').open('w') as log:
   run=subprocess.run([sys.executable,'-m','daytrading.replay.worker','--root',str(root.resolve()),'--request',str((jobdir/'request.json').resolve()),'--output',str((jobdir/'result.json').resolve())],cwd=jobdir,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  if run.returncode or not (jobdir/'result.json').exists():raise RuntimeError('Offline worker failed; see the local replay job log.')
  result=json.loads((jobdir/'result.json').read_text())
  result=publish_evaluation(root,manifest,result);atomic(jobdir/'result.json',result)
  with _LOCK:_JOBS[job_id].update(status='complete',result_path=str(jobdir/'result.json'))
 except Exception as exc:
  with _LOCK:_JOBS[job_id].update(status='failed',error=str(exc))

def job_status(job_id):
 if not re.fullmatch(r'[a-f0-9]{32}',str(job_id)):raise ValueError('Invalid replay job ID')
 with _LOCK:
  if job_id not in _JOBS:return dict(ok=False,error='Replay job not found')
  row=dict(_JOBS[job_id])
 if row.get('status')=='complete' and row.get('result_path'):
  try:row['result']=json.loads(Path(row.pop('result_path')).read_text())
  except (OSError,ValueError):row.update(status='failed',error='Replay result artifact is unavailable')
 return row


def percentile(values,p):
 if not values:return None
 ordered=sorted(values);return ordered[max(0,__import__('math').ceil(len(ordered)*p)-1)]

def acceptance_gate(records):
 """Frozen engineering gates; count distinct held-out sessions, not variants."""
 selected={r.get('session_date',r['session_id']):r for r in records if r.get('role')=='held_out' and r.get('complete_session')}
 rows=list(selected.values());trades=[t for r in rows for t in r.get('matched_trades',[])];intents=[i for r in rows for i in r.get('matched_intents',[])]
 losses=sum(t['paper_pnl']<0 for t in trades);unfilled=sum(bool(i['paper_filled_qty']==0 or i.get('cancelled')) for i in intents)
 missing=[]
 if len(rows)<3:missing.append('at least three complete held-out sessions')
 if len(trades)<30:missing.append('at least 30 completed matched trades')
 if losses<5:missing.append('at least five losses')
 if unfilled<5:missing.append('at least five unfilled/cancelled intents')
 if any(r.get('decision_parity')!='matched' or r.get('capture_completeness')!='complete' or r.get('prediction_status','simulated')!='simulated' for r in rows):missing.append('exact decision parity and complete recordings')
 if len({r.get('model_sha256') for r in rows})>1:missing.append('one frozen predictive model')
 prices=[abs(t[k]-t['paper_'+k]) for t in trades for k in ('entry','exit')]
 pnl=[abs(t['pnl']-t['paper_pnl'])/t['risk_budget'] for t in trades]
 quantity=max(0.,1.-sum(abs(i['filled_qty']-i['paper_filled_qty']) for i in intents)/sum(max(i['filled_qty'],i['paper_filled_qty']) for i in intents)) if any(max(i['filled_qty'],i['paper_filled_qty'])>0 for i in intents) else (1. if intents else None)
 reversals=[t for t in trades if t['pnl']*t['paper_pnl']<0 and abs(t['pnl']-t['paper_pnl'])>.25*t['risk_budget']]
 metrics=dict(fill_price_error_p95=percentile(prices,.95),final_quantity_agreement=quantity,mean_absolute_pnl_error_R=sum(pnl)/len(pnl) if pnl else None,outcome_reversals=len(reversals),sessions=len(rows),trades=len(trades),losses=losses,unfilled_cancelled=unfilled)
 failures=[]
 if metrics['fill_price_error_p95'] is not None and metrics['fill_price_error_p95']>.01+1e-9:failures.append('fill-price error exceeds $0.01/share')
 if quantity is not None and quantity<.95:failures.append('final filled-quantity agreement below 95%')
 if pnl and sum(pnl)/len(pnl)>.25:failures.append('mean trade P&L error exceeds 0.25R')
 if reversals:failures.append('unexplained outcome reversal exceeds 0.25R')
 if any(r.get('unmatched_intents') or r.get('protection_failures') or r.get('right_censored_orders') for r in rows):failures.append('unmatched order intent or protection failure')
 return dict(status='collecting' if missing else 'failed' if failures else 'passed',requirements=missing,failures=failures,metrics=metrics)


def prediction_fingerprint(manifest):
 files=manifest['source']['source_files']
 code=dict(files) # Full frozen build: discovery/data changes also start a new baseline.}
 return digest(dict(protocol_schema=1,code=code,settings=manifest['metadata'].get('settings'),environment={k:v for k,v in manifest['metadata'].get('environment',{}).items() if not re.search(r'(_DIR|_DB|_PATH|_FILE)$',k)}))


def publish_evaluation(root,manifest,result):
 """Roles are determined chronologically from capture coverage, never P&L."""
 from .comparison import complete_session
 root=Path(root);fingerprint=prediction_fingerprint(manifest);directory=root/'evaluations'/fingerprint;directory.mkdir(parents=True,exist_ok=True)
 eligible=[]
 for session in list_sessions(root):
  try:
   m=json.loads((safe_session(root,session['session_id'])/'manifest.json').read_text())
   coverage=[dict(received_at=m['coverage_start']),dict(received_at=m['coverage_end'])] if m.get('coverage_start') and m.get('coverage_end') else []
   if complete_session(m,coverage) and prediction_fingerprint(m)==fingerprint:eligible.append(m['session_date'])
  except (OSError,ValueError,KeyError):continue
 dates=sorted(set(eligible));role='held_out' if manifest['session_date'] in dates[2:] else 'calibration'
 comparison=result.get('execution_comparison',{}).get('paper',{})
 row=dict(session_id=manifest['session_id'],session_date=manifest['session_date'],role=role,complete_session=result.get('complete_session',False),capture_completeness=result.get('capture_completeness'),decision_parity=result.get('decision_parity'),model_sha256=fingerprint,matched_intents=comparison.get('matched_intents',[]),matched_trades=comparison.get('matched_trades',[]),unmatched_intents=comparison.get('unmatched_intents',[]),protection_failures=result.get('protection',{}).get('failures',[]),prediction_status=result.get('predictions',{}).get('paper',{}).get('status'),right_censored_orders=result.get('predictions',{}).get('paper',{}).get('right_censored_orders',[]))
 atomic(directory/(manifest['session_id']+'.json'),row)
 records=[json.loads(p.read_text()) for p in directory.glob('*.json') if p.name!='protocol.json']
 protocol=dict(model_sha256=fingerprint,calibration_dates=dates[:2],held_out_dates=dates[2:],selection='first two fully captured trading dates calibrate; subsequent dates held out; no outcome-based selection',thresholds=dict(fill_price_p95_dollars=.01,final_quantity_agreement=.95,mean_absolute_pnl_error_R=.25,maximum_unexplained_reversal_R=.25),minimums=dict(held_out_sessions=3,completed_trades=30,losses=5,unfilled_cancelled_intents=5))
 atomic(directory/'protocol.json',protocol)
 gate=acceptance_gate(records)
 result.update(evaluation_role=role,acceptance=gate,frozen_protocol=protocol,execution_validation=gate['status'],validated=(gate['status']=='passed' and result.get('complete_session') is True and result.get('decision_parity')=='matched' and result.get('capture_completeness')=='complete' and result.get('predictions',{}).get('paper',{}).get('status')=='simulated'))
 if not result['validated'] and result['execution_validation']=='passed':result['execution_validation']='unvalidated_current_session'
 return result
