from daytrading.models import BarSource
"""Synthetic profile checks; no NCPL or observed fills are test inputs."""
from datetime import datetime,timedelta,timezone
from math import log
from unittest.mock import patch
from types import SimpleNamespace
import pytest
from daytrading.backtest.matching_profile import MatchingProfile
from daytrading.backtest.nbbo import NBBOFillSimulator,NBBOExecutionPolicy
from daytrading.models import Quote,Side,Order,Bar,PortfolioState,OrderStatus
from daytrading.runner import QuoteEvent
from daytrading.replay.predict import PredictClock,PredictiveInputs,MarketView,SimulatedBroker
from daytrading.replay.codec import pack
from daytrading.replay.execution import NativePredictiveBroker
T=datetime(2026,1,5,12,tzinfo=timezone.utc)
def profile(p=1.,initial=.2,gap=.5):
 return MatchingProfile.from_dict(dict(schema_version=1,version='synthetic',partial_probability=p,initial_delay=dict(mu=log(initial),sigma=0.),remainder_delay=dict(mu=log(gap),sigma=0.),size_model='uniform_integer_remaining',training_end_exclusive='2025-01-01',training_order_count=50,initial_sample_count=30,remainder_sample_count=30))
def broker(points=((0,1.99,2.),),p=1.):
 rows=tuple(dict(received_at=(T+timedelta(seconds=s)).isoformat(),data=dict(event=pack(QuoteEvent('X',Quote('X',T+timedelta(seconds=s),bid,ask,100,100))))) for s,bid,ask in points)
 clock=PredictClock(T.isoformat(),0);view=MarketView(PredictiveInputs({},(),(),rows,()),clock)
 return SimulatedBroker(view,'paper',0,10000,matching_profile=profile(p))
def buy(b,qty=3):
 b.orders['a']=dict(id='a',symbol='X',side='buy',type='limit',qty=qty,limit_price=2.,filled_qty=0,filled_avg_price=None,status='new',submitted_at=T,filled_at=None)

def test_pending_initial_delay_is_not_an_immediate_fill():
 b=broker(p=0.);buy(b);b._advance_venue(T+timedelta(seconds=.199))
 assert b.orders['a']['filled_qty']==0
 b._advance_venue(T+timedelta(seconds=.2));assert b.orders['a']['filled_qty']==3
 assert b.orders['a']['filled_at']==T+timedelta(seconds=.2)

def test_partial_remainder_has_independent_sampled_deadline():
 b=broker();buy(b,3);b._advance_venue(T+timedelta(seconds=.2))
 assert 0<b.orders['a']['filled_qty']<3
 first=b.orders['a']['filled_qty'];b._advance_venue(T+timedelta(seconds=.699));assert b.orders['a']['filled_qty']==first
 b._advance_venue(T+timedelta(seconds=2));assert b.orders['a']['filled_qty']==3

def test_future_quote_cannot_affect_first_matching_attempt():
 def run(future):
  b=broker(((0,1.99,2.),(1,future,future+.01)),p=0);buy(b);b._advance_venue(T+timedelta(seconds=.5));return b.events
 assert run(1.)==run(10.)

def test_wait_partition_cannot_change_matching_sequence():
 def run(parts):
  b=broker();buy(b,9)
  for t in parts:b._advance_venue(T+timedelta(seconds=t))
  return b.events
 assert run([2.])==run([.1,.21,.35,.8,1.5,2.])

def test_cancellation_before_initial_deadline_has_no_fill():
 b=broker(p=0.);buy(b);b.view.latency=lambda *a:0.
 b.cancel_order_by_id('a');b._advance_venue(T+timedelta(seconds=2))
 assert b.orders['a']['status']=='canceled' and b.orders['a']['filled_qty']==0

def test_marketability_is_rechecked_at_sampled_execution_time():
 b=broker(((0,1.99,2.),(.1,2.01,2.02),(.7,1.98,1.99)),p=0);buy(b)
 b._advance_venue(T+timedelta(seconds=.6));assert b.orders['a']['filled_qty']==0
 b._advance_venue(T+timedelta(seconds=.7));assert b.orders['a']['filled_avg_price']==1.99

@pytest.mark.parametrize('key,value',[('known_fills',[]),('order_id','actual'),('symbol','X'),('entry_price',2.),('fill_times',[])])
def test_profile_rejects_per_order_outcomes(key,value):
 data=profile().as_dict();data[key]=value
 with pytest.raises(ValueError,match='aggregate'):MatchingProfile.from_dict(data)

def test_profile_cannot_be_used_before_its_calibration_window():
 with pytest.raises(ValueError,match='newer'):profile().assert_available(T.replace(year=2024))

def test_native_broker_partial_callbacks_and_finalization_still_work():
 b=broker();view=b.view
 native=NativePredictiveBroker(view,'paper',0,10000,matching_profile=profile());native.clock.venue=native.venue;seen=[]
 native.set_entry_lifecycle_callbacks(lambda oid,order,fill:seen.append((native.clock.now,fill.quantity)) if fill is not None else None,lambda s:False,lambda s:None)
 timing=SimpleNamespace(monotonic=lambda:(native.clock.now-T).total_seconds(),sleep=lambda seconds:native.clock.clock('time.sleep',seconds))
 with patch('daytrading.execution.alpaca_broker.time',timing):
  fill,status=native.submit(Order('X',Side.BUY,3,limit_price=2.),Bar('X',T,2,2,2,2,1000, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=10000))
 assert status is OrderStatus.FILLED and fill.quantity==3
 assert seen[0][1]<3 and seen[0][0]<seen[-1][0]
 assert not native._unsettled_orders and not native._safety_blocked

def test_new_flat_entry_does_not_inherit_old_opportunity_id():
 from alpaca.trading.requests import LimitOrderRequest
 from alpaca.trading.enums import OrderSide,TimeInForce
 from daytrading.execution.origin import EXECUTION_ORIGIN
 b=broker(p=0);native=NativePredictiveBroker(b.view,'paper',0,10000)
 native.venue.order_origins['X']='old-setup'
 token=EXECUTION_ORIGIN.set('new-setup')
 try:native.client.submit_order(order_data=LimitOrderRequest(symbol='X',qty=3,side=OrderSide.BUY,type='limit',limit_price=2.,time_in_force=TimeInForce.DAY,extended_hours=True))
 finally:EXECUTION_ORIGIN.reset(token)
 request=next(e for e in native.events if e['kind']=='order_request')
 assert request['opportunity_id']=='new-setup'

def test_model_metadata_distinguishes_default_and_legacy():
 from daytrading.backtest.matching_profile import default_matching_profile
 from daytrading.replay.predict import execution_model
 current=execution_model(default_matching_profile());legacy=execution_model(None)
 assert current['matching_profile']['training_order_count']>=30
 assert current['paper_recheck_seconds'] is None and legacy['paper_recheck_seconds']==.15
 assert current['matching_profile'] is not None and legacy['matching_profile'] is None
 assert current['known_fill_inputs'] is False

def test_cancel_transit_preserves_matching_before_cancel_arrives():
 b=broker(p=0);buy(b);b.view.latency=lambda *args:.25
 b.cancel_order_by_id('a')
 assert b.orders['a']['status']=='filled' and b.positions['X']['qty']==3
 assert b.orders['a']['filled_at']==T+timedelta(seconds=.2)
