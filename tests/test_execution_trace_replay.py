from daytrading.models import BarSource
from dataclasses import replace
from datetime import datetime,timedelta,timezone
import pytest
from daytrading.backtest.execution import RecordedExecution,RecordedExecutionReplay
from daytrading.models import Fill,Order,Side
AT=datetime(2026,10,16,12,tzinfo=timezone.utc)
def record(**kw):return replace(RecordedExecution('one','TEST',Side.BUY,500,2.60,AT+timedelta(seconds=1),(Fill('TEST',Side.BUY,200,2.58,AT+timedelta(seconds=2)),Fill('TEST',Side.BUY,300,2.59,AT+timedelta(seconds=3)))),**kw)
def order(**kw):return replace(Order('TEST',Side.BUY,500,2.60),**kw)
def test_partials_arrive_once_at_recorded_times_with_weighted_price():
    t=RecordedExecutionReplay([record()]);t.bind(order(),now=AT)
    assert t.poll(AT+timedelta(seconds=1.9))==[]
    one=t.poll(AT+timedelta(seconds=2));assert one[0][1].quantity==200
    assert t.poll(AT+timedelta(seconds=2))==[]
    two=t.poll(AT+timedelta(seconds=3));assert two[0][1].quantity==300
    assert sum(f.price*f.quantity for _,f in one+two)/500==pytest.approx(2.586)
    assert t.poll(AT+timedelta(seconds=4))==[] and t.predictive is False
@pytest.mark.parametrize('changes',[dict(symbol='WRONG'),dict(side=Side.SELL),dict(quantity=499),dict(limit_price=2.59),dict(hard_limit_price=2.59)])
def test_mismatch_cannot_borrow_a_known_fill(changes):
    t=RecordedExecutionReplay([record()])
    with pytest.raises(ValueError,match='intent mismatch'):t.bind(order(**changes),now=AT)
    assert t.poll(AT+timedelta(seconds=4))==[]
def test_late_or_future_request_and_reused_order_fail():
    t=RecordedExecutionReplay([record()])
    for at in (AT-timedelta(seconds=2),AT+timedelta(seconds=2)):
        with pytest.raises(ValueError):t.bind(order(),now=at)
    t.bind(order(),now=AT)
    with pytest.raises(ValueError):t.bind(order(),now=AT)
def test_no_quote_or_fake_fill_fallback():
    t=RecordedExecutionReplay([])
    with pytest.raises(ValueError):t.bind(order(),now=AT)
    assert t.poll(AT+timedelta(days=1))==[]
def test_summary_honestly_marked_aggregate():
    row=dict(id='id',symbol='TEST',side='buy',type='limit',qty='500',filled_qty='500',filled_avg_price='2.5847',limit_price='2.60',submitted_at=(AT+timedelta(seconds=1)).isoformat(),filled_at=(AT+timedelta(seconds=3)).isoformat())
    r=RecordedExecution.from_order_summary(row)
    assert r.granularity=='order_aggregate' and len(r.fills)==1 and r.fills[0].price==2.5847
    assert RecordedExecution.from_order_summary(dict(row,filled_qty='0')) is None
@pytest.mark.parametrize('fill',[Fill('TEST',Side.BUY,501,2.58,AT+timedelta(seconds=2)),Fill('TEST',Side.BUY,500,2.61,AT+timedelta(seconds=2)),Fill('WRONG',Side.BUY,500,2.58,AT+timedelta(seconds=2)),Fill('TEST',Side.BUY,500,2.58,AT)])
def test_bad_recorded_evidence_rejected(fill):
    with pytest.raises(ValueError):record(fills=(fill,))
def test_clock_backwards_and_duplicate_ids_fail():
    with pytest.raises(ValueError):RecordedExecutionReplay([record(),record()])
    t=RecordedExecutionReplay([record()]);t.poll(AT)
    with pytest.raises(ValueError):t.poll(AT-timedelta(seconds=1))
def test_sell_repricing_range_cannot_override_hard_limit():
    r=RecordedExecution('sell','TEST',Side.SELL,500,2.58,AT+timedelta(seconds=1),(Fill('TEST',Side.SELL,500,2.59,AT+timedelta(seconds=2)),))
    t=RecordedExecutionReplay([r])
    with pytest.raises(ValueError):t.bind(Order('TEST',Side.SELL,500,2.59,hard_limit_price=2.59),now=AT,limit_floor=2.57,limit_ceiling=2.59)
    t.bind(Order('TEST',Side.SELL,500,2.59),now=AT,limit_floor=2.57,limit_ceiling=2.59)
    assert t.poll(AT+timedelta(seconds=2))[0][1].price==2.59

def test_request_limit_matches_live_smart_pricing_not_fill_price():
    from daytrading.backtest.execution import quote_request_limit
    from daytrading.models import Bar,Quote
    from daytrading.risk.guards import SlippageGuard
    from daytrading.execution.pricing import marketable_limit
    bar=Bar('TEST',AT,2.55,2.59,2.51,2.5812,144596, provenance=BarSource.CANONICAL_INTERVAL)
    q=Quote('TEST',AT,2.59,2.60,1000,1000)
    guard=SlippageGuard();guard.update_quote(q)
    o=Order('TEST',Side.BUY,500,2.5812)
    assert quote_request_limit(o,bar,q,now=AT)==marketable_limit(o,bar,quote_price=guard.get_limit_price('TEST','buy'))==2.61
    assert quote_request_limit(Order('TEST',Side.SELL,500,2.59),bar,q,now=AT)==2.58
    with pytest.raises(ValueError):quote_request_limit(replace(o,hard_limit_price=2.60),bar,q,now=AT)
    with pytest.raises(ValueError):quote_request_limit(o,bar,replace(q,ts=AT+timedelta(seconds=1)),now=AT)
    with pytest.raises(ValueError):quote_request_limit(o,bar,replace(q,ts=AT-timedelta(seconds=11)),now=AT)

def test_quote_estimate_separates_request_cap_fill_and_partial_depth():
    from daytrading.backtest.broker import BacktestBroker
    from daytrading.models import Bar,Quote,PortfolioState
    broker=BacktestBroker();bar=Bar('TEST',AT,2.55,2.59,2.51,2.5812,144596, provenance=BarSource.CANONICAL_INTERVAL)
    q=Quote('TEST',AT,2.59,2.60,1000,200)
    order=Order('TEST',Side.BUY,500,2.5812);p=PortfolioState(cash=25000)
    f,status=broker.submit_at_quote(order,bar,p,quote=q,now=AT)
    assert f.quantity==200 and f.price==2.60 and f.ts==AT
    assert broker.submit_at_quote(order,bar,p,quote=q,now=AT)[0] is None
    assert broker.submit_at_quote(order,bar,p,quote=replace(q,ts=AT+timedelta(seconds=1)),now=AT)[0] is None
    assert broker.submit_at_quote(order,bar,p,quote=replace(q,bid=q.ask),now=AT)[0] is None
