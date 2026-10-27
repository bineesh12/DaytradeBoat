"""Synthetic market evidence only: no NCPL fills or target P&L."""
from datetime import datetime,timedelta,timezone
import pytest
from daytrading.models import Quote
from daytrading.runner import QuoteEvent
from daytrading.replay.codec import pack
from daytrading.replay.predict import PredictClock,PredictiveInputs,MarketView,SimulatedBroker
T=datetime(2026,1,5,15,tzinfo=timezone.utc)
def broker(points,mode='paper',seed=0):
 rows=tuple(dict(received_at=(T+timedelta(seconds=s)).isoformat(),data=dict(event=pack(QuoteEvent('X',Quote('X',T+timedelta(seconds=s),bid,ask,100,100))))) for s,bid,ask in points)
 clock=PredictClock(T.isoformat(),0);view=MarketView(PredictiveInputs({},(),(),rows,()),clock)
 return SimulatedBroker(view,mode,seed,10000)
def buy(b,qty=100,limit=2.):
 b.orders['buy']=dict(id='buy',symbol='X',side='buy',type='limit',qty=qty,limit_price=limit,filled_qty=0,filled_avg_price=None,status='new',submitted_at=T,filled_at=None)
@pytest.mark.parametrize('mode',['paper','depth'])
def test_brief_quote_is_not_skipped_during_http_wait(mode):
 b=broker([(0,2.01,2.02),(.05,1.98,1.99),(.10,2.01,2.02)],mode);buy(b)
 b._advance_venue(T+timedelta(seconds=.5))
 assert b.orders['buy']['filled_qty']==100
 assert b.orders['buy']['filled_at']==T+timedelta(seconds=.05)
 assert b.orders['buy']['filled_avg_price']==1.99

def test_paper_recheck_is_independent_of_client_wait_partition():
 def run(parts):
  b=broker([(0,1.99,2.)],seed=31);buy(b)
  for t in parts:b._advance_venue(T+timedelta(seconds=t))
  return [(e['time'],e['qty'],e['price']) for e in b.events if e['kind']=='fill']
 assert run([.5])==run([.13,.26,.5])
 assert len(run([.5]))==2
 assert run([.5])[1][0]==(T+timedelta(seconds=.15)).isoformat()

def test_cancel_transit_retains_brief_late_fill():
 b=broker([(0,2.01,2.02),(.05,1.98,1.99),(.10,2.01,2.02)]);buy(b)
 b.cancel_order_by_id('buy')
 assert b.orders['buy']['status']=='filled' and b.orders['buy']['filled_qty']==100
 assert not [e for e in b.events if e['kind']=='cancel']

def test_stop_gap_during_http_wait_cannot_be_skipped():
 b=broker([(0,2.,2.01),(.05,1.8,1.81),(.10,2.,2.01)])
 b.positions['X']=dict(qty=100,avg_entry=2.)
 b.orders['stop']=dict(id='stop',symbol='X',side='sell',type='stop',qty=100,stop_price=1.9,filled_qty=0,filled_avg_price=None,status='new',submitted_at=T,filled_at=None)
 b._advance_venue(T+timedelta(seconds=.5))
 assert b.orders['stop']['filled_avg_price']==1.8 and not b.positions
 assert b.orders['stop']['filled_at']==T+timedelta(seconds=.05)

def test_future_quote_suffix_cannot_change_prefix_execution():
 def run(last):
  b=broker([(0,1.99,2.),(1,last,last+.01)]);buy(b);b._advance_venue(T+timedelta(seconds=.5));return b.events
 assert run(1.)==run(10.)
