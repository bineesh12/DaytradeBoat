"""Receipt order/elapsed timing and exchange UTC are different clock domains."""
from datetime import datetime, timedelta, timezone
from daytrading.models import Quote
from daytrading.runner import QuoteEvent
from daytrading.replay.codec import pack
from daytrading.replay.predict import PredictClock, PredictiveInputs, MarketView, SimulatedBroker

T = datetime(2026, 10, 16, 14, tzinfo=timezone.utc)

def venue(offset=.7, quote_age=0., at=T):
    q = Quote('X', at+timedelta(seconds=offset-quote_age), 1.99, 2., 100, 100)
    receipt = dict(received_at=at.isoformat(),data=dict(event=pack(QuoteEvent('X',q))))
    calibration = dict(received_at=at.isoformat(),data=dict(value=pack(dict(synchronized=True,offset_seconds=offset,sample_age_seconds=0))))
    clock=PredictClock(at.isoformat(),0)
    view=MarketView(PredictiveInputs({},(),(),(receipt,),(),calibrations=(calibration,)),clock)
    b=SimulatedBroker(view,'paper',0,10000)
    b.orders['buy']=dict(id='buy',symbol='X',side='buy',type='limit',qty=100,limit_price=2.,filled_qty=0,filled_avg_price=None,status='new',submitted_at=at,filled_at=None)
    return b

def test_delivered_current_quote_is_not_rejected_as_future_due_to_host_clock():
    b=venue()
    b._match_limits()
    assert b.orders['buy']['filled_qty']==100

def test_receipt_clock_ahead_does_not_make_exchange_stale_quote_fresh():
    b=venue(offset=1.,quote_age=10.5)
    b._match_limits()
    assert b.orders['buy']['filled_qty']==0

def test_fill_timestamp_uses_exchange_utc_without_moving_receipt_clock():
    b=venue(offset=-.7)
    b._match_limits()
    assert b.orders['buy']['filled_at']==T-timedelta(seconds=.7)
    assert b.clock.now==T
    assert b.events[-1]['time']==T.isoformat()
    assert b.events[-1]['exchange_time']==(T-timedelta(seconds=.7)).isoformat()

def test_stop_uses_exchange_regular_session_boundary():
    at=T.replace(hour=13,minute=29,second=59,microsecond=500000)
    b=venue(offset=1.,at=at)
    b.orders.clear();b.positions['X']=dict(qty=100,avg_entry=2.)
    b.orders['stop']=dict(id='stop',symbol='X',side='sell',type='stop',qty=100,stop_price=2.,filled_qty=0,filled_avg_price=None,status='new',submitted_at=at,filled_at=None)
    b._trigger_stops()
    assert b.orders['stop']['status']=='filled'

def test_future_calibration_cannot_change_a_prior_fill():
    b=venue(offset=0.)
    b.view.inputs=PredictiveInputs({},(),(),b.view.inputs.receipts,(),calibrations=b.view.inputs.calibrations+(dict(received_at=(T+timedelta(seconds=1)).isoformat(),data=dict(value=pack(dict(synchronized=True,offset_seconds=500.,sample_age_seconds=0)))),))
    b._match_limits()
    assert b.orders['buy']['filled_at']==T

def test_sdk_reports_exchange_submission_but_keeps_internal_arrival_elapsed():
    from types import SimpleNamespace
    from daytrading.replay.execution import OfflineTradingSDK
    b=venue();b.orders.clear()
    sdk=OfflineTradingSDK(b)
    observed=sdk.submit_order(order_data=SimpleNamespace(symbol='X',side='buy',type='limit',qty=100,limit_price=2.,client_order_id='synthetic'))
    assert observed.submitted_at==T+timedelta(seconds=.95)
    assert b.orders[observed.id]['submitted_at']==T+timedelta(seconds=.25)
    assert observed.filled_at==T+timedelta(seconds=.95)
    assert b.clock.now==T+timedelta(seconds=.5)

def test_exchange_time_query_does_not_filter_out_an_accepted_order():
    from types import SimpleNamespace
    b=venue()
    b.orders['buy']['_exchange_submitted_at']=T+timedelta(seconds=.7)
    rows=b.get_orders(filter=SimpleNamespace(status='all',after=T+timedelta(seconds=.5),limit=10))
    assert len(rows)==1
