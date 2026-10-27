"""Isolated, offline source-pinned replay worker. Never imported by live trading."""
from __future__ import annotations
import argparse,json,hashlib,sys,os
from datetime import datetime
from pathlib import Path
from .store import read_session,digest,atomic

def build_report(manifest,events,symbol,day):
 from .audit import audit_session
 from .predict import PredictiveInputs,predict_session,MODEL
 from .codec import unpack
 audit=audit_session(manifest,events)
 try:
  inputs=PredictiveInputs.from_recording(events)
  predictions={mode:predict_session(manifest,inputs,mode=mode,seed=MODEL['seed'],report_symbol=symbol) for mode in ('paper','depth')}
 except Exception as exc:predictions={mode:dict(status='blocked',reason=str(exc),known_fill_inputs=False) for mode in ('paper','depth')}
 from .comparison import compare_execution,complete_session,protection_evidence,decision_comparison_rows,observed_intents
 risk_budget=float(manifest['metadata'].get('environment',{}).get('DAYTRADING_MAX_DOLLAR_RISK_PER_TRADE','50'))
 comparisons={name:compare_execution(events,prediction,risk_budget=risk_budget) for name,prediction in predictions.items()}
 protection=protection_evidence(events)
 prefix=manifest['state']!='sealed'
 capture='recording_prefix' if prefix else 'complete'
 bars=[];ticks=[];paper=[];decisions=[]
 for e in events:
  if e['kind']=='receipt':
   event=unpack(e['data']['event']);bar=getattr(event,'bar',None);tick=getattr(event,'tick',None)
   if bar and bar.symbol==symbol:bars.append(dict(symbol=symbol,ts=bar.ts.isoformat(),open=bar.open,high=bar.high,low=bar.low,close=bar.close,volume=bar.volume))
   if tick and tick.symbol==symbol:ticks.append(tick)
  if e['kind']=='decision':
   payload=unpack(e['data']['payload'])
   if payload.get('symbol')==symbol:decisions.append(dict(time=e['received_at'],stage=e['data']['type'],paper=payload))
  if e['kind']=='external_response' and e['data']['role']=='_broker' and e['data']['method'] in ('submit','submit_urgent_exit') and 'value' in e['data']:
   value=unpack(e['data']['value'])
   if value and value[0] is not None and value[0].symbol==symbol:
    f=value[0];paper.append(dict(time=f.ts.isoformat(),side=f.side.value,price=f.price,qty=f.quantity))
 paper=[dict(time=row.get('fill_time',row['time']),side=row['side'],price=row['price'],qty=row['filled_qty'],evidence='weighted cumulative broker fill') for row in observed_intents(events) if row['symbol']==symbol and row['filled_qty']>0]
 ten={}
 for tick in ticks:
  ts=tick.ts.replace(microsecond=0,second=tick.ts.second//10*10);key=ts.isoformat();b=ten.setdefault(key,dict(symbol=symbol,ts=key,open=tick.price,high=tick.price,low=tick.price,close=tick.price,volume=0));b.update(high=max(b['high'],tick.price),low=min(b['low'],tick.price),close=tick.price,volume=b['volume']+tick.size)
 predicted=[e for e in predictions['paper'].get('events',[]) if e['kind']=='fill' and e['symbol']==symbol]
 fillrows=[]
 for row in comparisons['paper']['matched_intents']:
  if row['symbol']!=symbol:continue
  fillrows.append(dict(stage='weighted fill',time=row['paper'].get('fill_time',row['paper']['time']),paper=dict(qty=row['paper_filled_qty'],price=row['paper_price']),replay=dict(qty=row['filled_qty'],price=row['price']),match=bool(row['paper_filled_qty']==row['filled_qty'] and (row['price']==row['paper_price'] or row['price'] is not None and row['paper_price'] is not None and abs(row['price']-row['paper_price'])<1e-8))))
 for row in comparisons['paper']['unmatched_intents']:
  actual=row.get('paper');estimated=row.get('replay')
  if (actual or estimated or {}).get('symbol')==symbol:fillrows.append(dict(stage='order / fill',time=(actual or estimated).get('time'),paper=actual,replay=estimated,match=False,reason=row.get('reason','Unmatched intent')))
 detailed=decision_comparison_rows(events,predictions['paper'],symbol,audit['status']=='matched')
 first=audit.get('first_divergence')
 if first is None:
  differing=next((row for row in fillrows if not row['match']),None)
  if differing:first=dict(category='execution',reason='Independent execution differs; later decisions may follow from different position state.',**differing)
 if first is None and any(p['status']=='blocked' for p in predictions.values()):first=dict(category='input',reason='Predictive market input coverage is insufficient; see execution lanes.')
 return dict(ok=True,mode='paper_validation',label='Recorded PAPER validation',symbol=symbol,date=day,session_id=manifest['session_id'],capture_completeness=capture,decision_parity=audit['status'],execution_validation='unvalidated',validated=False,first_divergence=first,comparison_rows=sorted(detailed+fillrows,key=lambda row:row.get('time') or ''),native_operation_rows=audit.get('rows',[]),observed_decisions=decisions,audit=audit,predictions=predictions,execution_comparison=comparisons,protection=protection,complete_session=complete_session(manifest,events),manifest=dict(source=manifest['source'],dependencies=manifest['dependencies'],coverage_start=events[0]['received_at'] if events else None,coverage_end=events[-1]['received_at'] if events else None,capture_state=manifest['state'],model_sha256=digest(MODEL)),bars_data=bars,charts_10s=list(ten.values()),paper_fills=paper,predicted_fills=predicted,round_trips=[],trades=[],scorecard={},message='Decision audit uses recorded broker observations. Predictive execution does not. Collect held-out sessions before claiming execution validation.')

def main():
 cli=argparse.ArgumentParser();cli.add_argument('--root',required=True);cli.add_argument('--request',required=True);cli.add_argument('--output',required=True);a=cli.parse_args();request=json.loads(Path(a.request).read_text())
 manifest,events=read_session(Path(a.root),request['session_id'],manifest=request['manifest']);events=[e for e in events if e['sequence']<=request['max_sequence']]
 logging=__import__('logging');logging.disable(logging.CRITICAL)
 from .isolation import worker_write_barrier
 try:os.nice(10)
 except OSError:pass
 with worker_write_barrier(Path.cwd()):
  result=build_report(manifest,events,request['symbol'],request['date']);atomic(Path(a.output),result)
if __name__=='__main__':main()
