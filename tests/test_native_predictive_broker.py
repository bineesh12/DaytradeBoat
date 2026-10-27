from daytrading.models import BarSource
"""The independent matcher must exercise the actual native order lifecycle."""
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
from types import SimpleNamespace
import pytest
from daytrading.models import Order,Side,Bar,PortfolioState,Quote,OrderStatus
from daytrading.runner import QuoteEvent
from daytrading.replay.codec import pack
from daytrading.replay.predict import PredictClock,MarketView,PredictiveInputs
from daytrading.replay.execution import NativePredictiveBroker
from daytrading.execution.alpaca_broker import AlpacaBroker
T=datetime(2026,1,5,12,tzinfo=timezone.utc)
def make(seed=31,points=((0,1.99,2.),)):
 receipts=tuple(dict(received_at=(T+timedelta(seconds=s)).isoformat(),data=dict(event=pack(QuoteEvent('X',Quote('X',T+timedelta(seconds=s),bid,ask,1000,1000))))) for s,bid,ask in points)
 clock=PredictClock(T.isoformat(),0);view=MarketView(PredictiveInputs({},(),(),receipts,()),clock)
 with patch('daytrading.execution.alpaca_broker.TradingClient',side_effect=AssertionError('No live client')):b=NativePredictiveBroker(view,'paper',seed,10000)
 clock.venue=b.venue
 time=SimpleNamespace(monotonic=lambda:(clock.now-T).total_seconds(),sleep=lambda t:clock.clock('time.sleep',t))
 return b,time

def test_factory_preserves_native_methods_and_never_constructs_network_client():
 b,_=make()
 assert isinstance(b,AlpacaBroker)
 for name in ('submit','_wait_for_fill','_wait_for_cancel_fill','place_protective_stop','replace_protective_stop','submit_urgent_exit'):
  assert getattr(type(b),name) is getattr(AlpacaBroker,name),name

def test_native_partial_progress_callback_precedes_complete_observation():
 b,time=make();seen=[]
 b.set_entry_lifecycle_callbacks(lambda oid,order,fill:seen.append((b.clock.now,fill.quantity)),lambda s:False,lambda s:None)
 with patch('daytrading.execution.alpaca_broker.time',time):
  fill,status=b.submit(Order('X',Side.BUY,100,limit_price=2.),Bar('X',T,2,2,2,2,1000, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=10000))
 assert status is OrderStatus.FILLED and fill.quantity==100
 assert seen[0][1]<100 and seen[0][0]<next(at for at,q in seen if q==100)
 assert b._unsettled_orders=={} and not b._safety_blocked

def test_native_position_cache_and_protective_stop_preserve_inventory():
 b,time=make(seed=0)
 with patch('daytrading.execution.alpaca_broker.time',time):
  fill,status=b.submit(Order('X',Side.BUY,100,limit_price=2.),Bar('X',T,2,2,2,2,1000, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=10000))
  stop=b.place_protective_stop('X',fill.quantity,1.9)
  assert b.get_positions()['X']['qty']==100
  replaced=b.replace_protective_stop('X',stop,100,1.95)
 assert b.orders[stop]['status']=='canceled' and replaced!=stop
 assert b.orders[replaced]['qty']==100 and b.orders[replaced]['stop_price']==1.95
 assert not [e for e in b.events if e['kind']=='order_request' and e['type']=='stop']

def test_terminal_native_cancel_reconciles_partial_inventory():
 b,time=make(seed=0,points=((0,2.01,2.02),));b._max_wait=.1
 with patch('daytrading.execution.alpaca_broker.time',time):
  fill,status=b.submit(Order('X',Side.BUY,100,limit_price=2.),Bar('X',T,2,2,2,2,1000, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=10000))
 assert fill is None and status is OrderStatus.CANCELLED and b._unsettled_orders=={} and not b.positions

def test_native_monitor_failure_keeps_unsettled_state_instead_of_claiming_success():
 b,time=make();seen=[]
 def failed(symbol):raise RuntimeError('test monitor failure')
 b.set_entry_lifecycle_callbacks(lambda *a:seen.append(a),failed,lambda s:None)
 with patch('daytrading.execution.alpaca_broker.time',time):
  fill,status=b.submit(Order('X',Side.BUY,100,limit_price=2.),Bar('X',T,2,2,2,2,1000, provenance=BarSource.CANONICAL_INTERVAL),PortfolioState(cash=10000))
 assert fill is None and status is OrderStatus.NEW and b._safety_blocked and b._unsettled_orders and seen
