"""Pending-order execution evidence only; entry admission rules are untouched."""
from datetime import datetime,timezone,timedelta
import pytest
from daytrading.backtest.nbbo import NBBOExecutionPolicy,NBBOFillSimulator
from daytrading.models import Quote,Side
from daytrading.runner import QuoteEvent
from daytrading.replay.codec import pack
from daytrading.replay.predict import PredictiveInputs,PredictClock,MarketView,SimulatedBroker
T=datetime(2026,1,5,15,tzinfo=timezone.utc)
@pytest.mark.parametrize('mode',['paper','depth'])
@pytest.mark.parametrize('side',[Side.BUY,Side.SELL])
def test_locked_nbbo_executes_existing_marketable_order(mode,side):
 m=NBBOFillSimulator(NBBOExecutionPolicy(mode,0));q=Quote('X',T,5.,5.,100,100)
 assert m.take('a','X',side,100,5.,q,now=T,arrived_at=T)==(100,5.)
@pytest.mark.parametrize('mode',['paper','depth'])
def test_crossed_quote_still_cannot_execute(mode):
 m=NBBOFillSimulator(NBBOExecutionPolicy(mode,0));q=Quote('X',T,5.01,5.,100,100)
 assert m.take('a','X',Side.BUY,100,5.01,q,now=T,arrived_at=T) is None

def test_pending_stop_cannot_ignore_locked_quote_below_stop():
 q=Quote('X',T,4.,4.,100,100);row=dict(received_at=T.isoformat(),data=dict(event=pack(QuoteEvent('X',q))))
 clock=PredictClock(T.isoformat(),0);view=MarketView(PredictiveInputs({},(),(),(row,),()),clock);b=SimulatedBroker(view,'paper',0,10000)
 b.positions['X']=dict(qty=100,avg_entry=5.)
 b.orders['s']=dict(id='s',symbol='X',side='sell',type='stop',qty=100,stop_price=4.5,filled_qty=0,filled_avg_price=None,status='new',submitted_at=T,filled_at=None)
 b._trigger_stops();assert b.orders['s']['status']=='filled' and b.orders['s']['filled_avg_price']==4.
