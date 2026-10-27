"""Synthetic execution evidence only; no historical fills or profit targets."""
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from unittest.mock import patch
import threading
import pytest
from daytrading.models import Quote
from daytrading.runner import QuoteEvent
from daytrading.replay.codec import pack
from daytrading.replay.predict import PredictiveInputs,PredictClock,MarketView,SimulatedBroker
T=datetime(2026,1,5,12,tzinfo=timezone.utc)
def row(n,wall,mono,ask):
 return dict(sequence=n,received_at=(T+timedelta(seconds=wall)).isoformat(),monotonic_ns=int(mono*1e9),data=dict(event=pack(QuoteEvent('X',Quote('X',T,ask-.01,ask,100,100)))))
def make(rows):
 clock=PredictClock(T.isoformat(),0);v=MarketView(PredictiveInputs({},(),(),tuple(rows),()),clock);v.use_delivery_order();return v

def test_undelivered_tied_quote_cannot_fill_an_order():
 rows=[row(1,0,0,2.1),row(2,0,0,1.9)];v=make(rows)
 assert v.quote('X') is None
 v.deliver_receipt(rows[0]);assert v.quote('X').ask==2.1
 b=SimulatedBroker(v,'paper',0,10000)
 b.orders['a']=dict(id='a',symbol='X',side='buy',type='limit',qty=100,limit_price=2.,filled_qty=0,filled_avg_price=None,status='new',submitted_at=T,filled_at=None)
 b._match_limits();assert b.orders['a']['filled_qty']==0
 v.deliver_receipt(rows[1]);b._match_limits();assert b.orders['a']['filled_avg_price']==1.9

def test_receipt_order_cannot_rewind_or_reprocess():
 rows=[row(1,0,0,2.),row(2,1,1,2.1)];v=make(rows)
 v.deliver_receipt(rows[0]);v.deliver_receipt(rows[1])
 with pytest.raises(ValueError):v.deliver_receipt(rows[0])

def test_monotonic_arrival_time_survives_backward_wall_clock_adjustment():
 from daytrading.replay.predict import receipt_schedule
 rows=[row(1,0,10,2.),row(2,-2,11,2.1),row(3,-1,12,2.2)]
 schedule=receipt_schedule(rows,rows[0]['received_at'],rows[0]['monotonic_ns'])
 assert [stamp for stamp,_ in schedule]==[T,T+timedelta(seconds=1),T+timedelta(seconds=2)]
 assert [r['sequence'] for _,r in schedule]==[1,2,3]

def test_partial_monotonic_capture_is_blocked_instead_of_mixed_clocks():
 from daytrading.replay.predict import receipt_schedule
 rows=[row(1,0,10,2.),row(2,1,11,2.1)];del rows[1]['monotonic_ns']
 with pytest.raises(ValueError,match='monotonic'):receipt_schedule(rows,T.isoformat(),int(10e9))

def test_equal_monotonic_receipts_preserve_recorded_order():
 from daytrading.replay.predict import receipt_schedule
 rows=[row(3,0,10,2.2),row(4,0,10,2.1)]
 assert [r['sequence'] for _,r in receipt_schedule(rows,T.isoformat(),int(10e9))]==[3,4]

def test_transport_duration_excludes_wait_to_reacquire_engine_lock():
 from daytrading.replay.engine import LiveOperation
 class Lock:
  def release(self):pass
  def acquire(self):elapsed[0]+=3_000_000_000
 class Recorder:
  def record(self,kind,data,**kw):facts.append((kind,data));return 1
  def fail(self,*a):raise AssertionError(a)
 elapsed=[0];facts=[];owner=SimpleNamespace(lock=Lock(),recorder=Recorder(),runner=SimpleNamespace())
 op=LiveOperation(owner,'op')
 def call():elapsed[0]+=200_000_000;return 'ok'
 with patch('daytrading.replay.engine.real_time.monotonic_ns',side_effect=lambda:elapsed[0]):
  assert op.external('broker_client','get_clock',(),{},call)=='ok'
 response=next(d for kind,d in facts if kind=='external_response')
 assert response['duration_ns']==200_000_000
 assert response['processing_delay_ns']==3_000_000_000
 assert 'response_received_at' in response and 'response_received_monotonic_ns' in response

def test_old_timing_sample_remains_explicitly_legacy():
 from daytrading.replay.predict import transport_sample
 e=dict(received_at=T.isoformat(),data=dict(role='broker_client',method='get_clock',duration_ns=123))
 s=transport_sample(e)
 assert s['duration_ns']==123 and s['timing_basis']=='legacy_combined_io_and_processing'
 assert 'filled_qty' not in s

def test_new_timing_sample_has_no_broker_outcome_fields():
 from daytrading.replay.predict import transport_sample
 e=dict(received_at=T.isoformat(),data=dict(role='broker_client',method='get_clock',duration_ns=123,processing_delay_ns=456,response_received_at=T.isoformat(),response_received_monotonic_ns=10,value={'filled_qty':100,'filled_avg_price':2.}))
 s=transport_sample(e)
 assert s['duration_ns']==123 and s['timing_basis']=='measured_sdk_call'
 assert set(s)<= {'available_at','method','duration_ns','timing_basis','role'}

def test_derived_inputs_are_not_injected_twice():
 rows=[dict(kind='bootstrap',data={}),dict(row(1,0,0,2.),kind='receipt'),dict(row(2,1,1,2.1),kind='receipt')]
 rows[2]['data']['derived']=True
 p=PredictiveInputs.from_recording(rows)
 assert [r['sequence'] for r in p.receipts]==[1]

def test_calibrated_market_clock_does_not_double_apply_host_clock_step():
 from daytrading.replay.codec import pack
 clock=PredictClock(T.isoformat(),int(10e9))
 c=dict(received_at=(T-timedelta(seconds=2)).isoformat(),monotonic_ns=int(11e9),data=dict(value=pack(dict(synchronized=True,offset_seconds=3.,sample_age_seconds=0))))
 v=MarketView(PredictiveInputs({},(),(),(),(),calibrations=(c,)),clock)
 clock.advance(T+timedelta(seconds=2))
 assert v.market_now()==T+timedelta(seconds=2)

def test_backward_clock_step_cannot_reveal_future_network_sample():
 clock=PredictClock(T.isoformat(),int(10e9))
 sample=dict(available_at=(T-timedelta(seconds=2)).isoformat(),available_monotonic_ns=int(12e9),method='submit_order',duration_ns=int(4e9),role='broker_client',timing_basis='measured_sdk_call')
 v=MarketView(PredictiveInputs({},(),(),(),(sample,)),clock)
 assert v.latency('submit_order',False)==.5
 clock.advance(T+timedelta(seconds=2));assert v.latency('submit_order',False)==4.
