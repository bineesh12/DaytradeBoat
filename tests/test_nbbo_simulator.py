from datetime import datetime, timedelta, timezone
import pytest
from daytrading.models import Quote, Side
from daytrading.backtest.nbbo import NBBOExecutionPolicy, NBBOFillSimulator
T=datetime(2026,10,16,12,tzinfo=timezone.utc)
def q(t=T,bid=2.,ask=2.01,bs=100,asz=100): return Quote('X',t,bid,ask,bs,asz)
def take(sim,quote=None,**kw):
    args=dict(order_id='a',symbol='X',side=Side.BUY,remaining=500,limit=2.02,quote=quote or q(),now=T,arrived_at=T)
    args.update(kw);return sim.take(**args)
def test_current_price_and_marketability():
    s=NBBOFillSimulator();assert take(s)==(100,2.01)
    assert take(s,q(T+timedelta(seconds=1),ask=2.03),now=T+timedelta(seconds=1)) is None

def test_sell_improvement_is_not_clipped_to_decision():
    assert take(NBBOFillSimulator(),side=Side.SELL,limit=1.99)==(100,2.)

@pytest.mark.parametrize('quote,now,arrival,expiry',[
    (q(),T,T+timedelta(seconds=1),None),
    (q(T+timedelta(microseconds=1)),T,T,None),
    (q(),T+timedelta(seconds=11),T,None),
    (q(bid=2.02),T,T,None),  # Crossed book: bid exceeds ask.
    (q(),T,T,T),
    (Quote('Y',T,2.,2.01,100,100),T,T,None),
])
def test_ineligible_evidence_never_fills(quote,now,arrival,expiry):
    assert take(NBBOFillSimulator(),quote,now=now,arrived_at=arrival,expires_at=expiry) is None

@pytest.mark.parametrize('mode',['depth','paper'])
@pytest.mark.parametrize('side',[Side.BUY,Side.SELL])
def test_locked_book_can_fill_an_already_accepted_order(mode,side):
    # Order matching and the strategy's unlocked entry-book gate are separate.
    result=take(NBBOFillSimulator(NBBOExecutionPolicy(mode,0)),q(bid=2.01),
                side=side,limit=2.01)
    assert result==(100 if mode=='depth' else 500,2.01)

def test_depth_shared_across_orders_no_reuse():
    s=NBBOFillSimulator();assert take(s)==(100,2.01)
    assert take(s,order_id='b') is None
    assert take(s,q(T+timedelta(seconds=1),bid=1.99),order_id='c',now=T+timedelta(seconds=1)) is None
    assert take(s,q(T+timedelta(seconds=2),asz=140),order_id='d',now=T+timedelta(seconds=2))==(40,2.01)

def test_partial_remainder_uses_later_quote_not_future_average():
    s=NBBOFillSimulator();assert take(s)==(100,2.01)
    assert take(s,remaining=400) is None
    assert take(s,q(T+timedelta(seconds=1),ask=2.00,bid=1.99,asz=1000),remaining=400,now=T+timedelta(seconds=1))==(400,2.)

def test_paper_no_displayed_depth_limit_and_seed_reproducible():
    assert take(NBBOFillSimulator(NBBOExecutionPolicy('paper',0)),q(asz=1))==(500,2.01)
    results=[]
    for _ in range(2):
        s=NBBOFillSimulator(NBBOExecutionPolicy('paper',31))
        first=take(s,q(asz=1));assert 0<first[0]<500
        second=take(s,q(T+timedelta(seconds=1),ask=2.,bid=1.99),remaining=500-first[0],now=T+timedelta(seconds=1))
        results.append((first,second))
    assert results[0]==results[1]
    assert sum(x[0] for x in results[0])==500

def test_future_suffix_does_not_change_prefix():
    def prefix(future):
        s=NBBOFillSimulator();a=take(s)
        b=take(s,future,now=T+timedelta(seconds=1),remaining=400)
        return a,b
    assert prefix(q(T+timedelta(seconds=1),ask=2.,bid=1.99))[0]==prefix(q(T+timedelta(seconds=1),ask=3.,bid=2.99))[0]

def test_clock_cannot_go_backwards():
    s=NBBOFillSimulator();take(s,now=T+timedelta(seconds=1))
    with pytest.raises(ValueError):take(s)

def test_weighted_partial_entry_cannot_exceed_dollar_risk_by_small_adjustment():
    from daytrading.backtest.nbbo import dollar_capped_stop
    # Generic weighted fills, not recorded broker fills: 44 at 2.60, 456 at 2.59.
    entry=(44*2.6+456*2.59)/500
    stop=dollar_capped_stop(entry,500,2.49,50.)
    assert stop==2.4909
    assert 0<(entry-stop)*500<=50.
    assert dollar_capped_stop(2.48,384,2.36,50.)==2.36

def test_capped_stop_never_lowers_existing_protection():
    from daytrading.backtest.nbbo import dollar_capped_stop
    assert dollar_capped_stop(2.59,500,2.55,50.)==2.55

def test_losing_gap_fills_at_executable_bid_not_optimistic_stop():
    s=NBBOFillSimulator(NBBOExecutionPolicy('paper',0))
    entry=take(s,q(ask=2.60,bid=2.59,asz=1000),limit=2.61)
    assert entry==(500,2.60)
    # A $2.50 stop is a trigger, not a guaranteed fill during a falling tape.
    later=T+timedelta(seconds=1)
    exit_fill=take(s,q(later,bid=2.49,ask=2.50,bs=1000),order_id='exit',side=Side.SELL,limit=2.48,now=later,arrived_at=later)
    assert exit_fill==(500,2.49)
    assert (exit_fill[1]-entry[1])*500==pytest.approx(-55.)
