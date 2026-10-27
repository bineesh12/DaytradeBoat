from daytrading.models import BarSource
"""Causal execution mechanics; fixtures contain no historical broker fills."""
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from daytrading.models import Quote,Side,Order,Bar,PortfolioState,OrderStatus
from daytrading.backtest.nbbo import NBBOExecutionPolicy,NBBOFillSimulator
from daytrading.replay.predict import PredictClock,PredictiveInputs,MarketView,SimulatedBroker,MODEL
from daytrading.replay.codec import pack
from daytrading.runner import QuoteEvent
T=datetime(2026,10,16,12,tzinfo=timezone.utc)

def take(s,now=T,remaining=100,quote=None,**kwargs):
 return s.take('entry','X',Side.BUY,remaining,2.02,quote or Quote('X',T,2,2.01,100,100),now=now,arrived_at=T,**kwargs)

def test_paper_remainder_rechecks_unchanged_marketable_quote():
 s=NBBOFillSimulator(NBBOExecutionPolicy('paper',31));first=take(s)
 assert 0<first[0]<100
 assert take(s,now=T+timedelta(seconds=.149),remaining=100-first[0]) is None
 second=take(s,now=T+timedelta(seconds=.15),remaining=100-first[0])
 assert first[0]+second[0]==100 and second[1]==2.01

def test_faster_quote_arrivals_do_not_accelerate_paper_attempts():
 s=NBBOFillSimulator(NBBOExecutionPolicy('paper',31));qty,_=take(s)
 assert take(s,now=T+timedelta(seconds=.01),remaining=100-qty,quote=Quote('X',T+timedelta(seconds=.01),1.99,2,100,100)) is None
 assert take(s,now=T+timedelta(seconds=.15),remaining=100-qty,quote=Quote('X',T+timedelta(seconds=.15),1.98,1.99,100,100))[1]==1.99

def test_depth_does_not_regenerate_displayed_size_with_clock_ticks():
 s=NBBOFillSimulator(NBBOExecutionPolicy('depth'));assert take(s,remaining=200)==(100,2.01)
 assert take(s,now=T+timedelta(seconds=1),remaining=100) is None

def test_future_or_stale_quote_cannot_complete_paper_remainder():
 s=NBBOFillSimulator(NBBOExecutionPolicy('paper',31));qty,_=take(s)
 assert take(s,now=T+timedelta(seconds=.15),remaining=100-qty,quote=Quote('X',T+timedelta(seconds=.16),1.99,2,100,100)) is None
 assert take(s,now=T+timedelta(seconds=11),remaining=100-qty) is None
 assert take(s,now=T+timedelta(seconds=11),remaining=100-qty,quote=Quote('X',T+timedelta(seconds=11),2.02,2.03,100,100)) is None

@pytest.mark.parametrize('cadence',[0,-1,float('nan'),float('inf')])
def test_invalid_matching_cadence_rejected(cadence):
 with pytest.raises(ValueError):NBBOExecutionPolicy('paper',paper_recheck_seconds=cadence)

def broker(quotes,seed=0,mode='paper'):
 rows=tuple(dict(received_at=q.ts.isoformat(),data=dict(event=pack(QuoteEvent('X',q)))) for q in quotes)
 clock=PredictClock(T.isoformat(),0);view=MarketView(PredictiveInputs({},(),(),rows,()),clock)
 return SimulatedBroker(view,mode,seed,10000)

def row():return dict(id='manual',symbol='X',side='buy',type='limit',qty=100,limit_price=2.,filled_qty=0,filled_avg_price=None,status='new',submitted_at=T,filled_at=None)

def test_fill_during_response_transit_cannot_rewrite_earlier_observation():
 b=broker([Quote('X',T,2.01,2.02,1000,1000),Quote('X',T+timedelta(seconds=.35),1.98,1.99,1000,1000)])
 b.orders['manual']=row();observed=b._observe_order('manual')
 assert observed['status']=='new' and observed['filled_qty']==0
 assert b.orders['manual']['status']=='filled' and b.orders['manual']['filled_qty']==100
 assert b.clock.now==T+timedelta(seconds=.5)

def test_fill_venue_time_ack_and_client_return_are_separate():
 b=broker([Quote('X',T,1.99,2,1000,1000)]);f,status=b.submit(Order('X',Side.BUY,100),Bar('X',T,2,2,2,2,1000, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=10000))
 assert status is OrderStatus.FILLED and f.quantity==100
 fill=next(e for e in b.events if e['kind']=='fill');ack=next(e for e in b.events if e['kind']=='order_ack');ret=next(e for e in b.events if e['kind']=='submit_return')
 assert fill['time']<ack['time']<ret['time'] and f.ts.isoformat()==fill['time']
 assert b._last_order_latency_sec==pytest.approx(1.5)

class AlwaysPartial:
 def random(self):return 0.
 def randint(self,a,b):return 1

def test_timeout_cancels_only_remainder_and_no_fills_after_ack():
 b=broker([Quote('X',T,1.99,2,1000,1000)]);b.model.random=AlwaysPartial();b._max_wait=.1
 f,status=b.submit(Order('X',Side.BUY,100),Bar('X',T,2,2,2,2,1000, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=10000))
 assert status is OrderStatus.FILLED and 0<f.quantity<100
 cancel=next(e for e in b.events if e['kind']=='cancel');assert cancel['remaining']==100-f.quantity
 assert not [e for e in b.events if e['kind']=='fill' and e['time']>cancel['time']]
 assert b.positions['X']['qty']==f.quantity and b.cash==pytest.approx(10000-2*f.quantity)

@pytest.mark.parametrize('last_price',[1.5,2.5])
def test_future_quote_suffix_does_not_change_completed_entry(last_price):
 b=broker([Quote('X',T,1.99,2,1000,1000),Quote('X',T+timedelta(seconds=5),last_price-.01,last_price,1000,1000)])
 f,status=b.submit(Order('X',Side.BUY,100),Bar('X',T,2,2,2,2,1000, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=10000))
 assert f.price==2 and f.quantity==100

def test_native_adapter_only_exposes_final_cumulative_entry():
 from daytrading.execution.alpaca_broker import AlpacaBroker
 import daytrading.execution.alpaca_broker as module
 class Clock:
  now=0.
  def monotonic(self):return self.now
  def sleep(self,x):self.now+=x
 clock=Clock();observations=[]
 def get_order(oid):
  q=20 if clock.now<.3 else 100
  observations.append((clock.now,q))
  return SimpleNamespace(id='generic-entry',status='partially_filled' if q<100 else 'filled',filled_qty=q,filled_avg_price=2,filled_at=T+timedelta(seconds=clock.now))
 b=AlpacaBroker.__new__(AlpacaBroker);b._client=SimpleNamespace(get_order_by_id=get_order);b._slippage_guard=None;b._max_wait=5;b._poll_interval=.15;b._safety_state()
 with patch.object(module,'time',clock):f,status=b._wait_for_fill('generic-entry',Order('X',Side.BUY,100))
 assert observations[0]==(0,20) and f.quantity==100 and status is OrderStatus.FILLED
 assert clock.now>=.3
 # Demonstrates the live handoff gap; test does not claim first-partial protection.
 assert len([q for _,q in observations if q<100])==2

def test_broker_stop_not_treated_as_premarket_protection():
 b=broker([Quote('X',T,1.79,1.8,1000,1000)]);b.positions['X']=dict(qty=100,avg_entry=2)
 oid=b.place_protective_stop('X',100,1.9);b._trigger_stops()
 assert b.orders[oid]['status']=='new' and b.positions['X']['qty']==100
