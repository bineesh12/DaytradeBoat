from daytrading.models import BarSource
"""Count completed positions once; retain immediate dollar protection."""
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from collections import deque
from unittest.mock import Mock
import pytest
from daytrading.execution.broker import apply_fill
from daytrading.exits.manager import TrackedPosition
from daytrading.models import Fill, Side, Position
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.runner import AlpacaRunner
from daytrading.runtime_state import capture_state, restore_state, encode, digest
from test_runtime_state import runner as checkpoint_runner, NOW

def pipeline():
    return create_scalping_pipeline(initial_cash=10_000,enable_daily_loser_blacklist=True)

def leg(p,qty,price,left,*,stamp=NOW,symbol='NCPL',entry=2.64):
    fill=Fill(symbol,Side.SELL,qty,price,stamp)
    return p.record_realized_exit(symbol,entry,price,qty,now=stamp,remaining_quantity=left,fill_key=p.exit_fill_key(fill))

def test_ncpl_net_winner_does_not_count_partial_losses():
    p=pipeline();p._daily_loss_counts['NCPL']=1;counts=[]
    for i,(qty,px,left) in enumerate([(200,2.69,300),(50,2.68,250),(100,2.63,150),(100,2.63,50),(50,2.64,0)]):
        leg(p,qty,px,left,stamp=NOW+timedelta(seconds=i));counts.append(p._daily_loss_counts['NCPL'])
        assert 'NCPL' not in p._daily_losers
    assert counts==[1,1,1,1,0]
    assert p._daily_pnl==pytest.approx(10) and not p._realized_trade_legs

def test_two_whole_losses_count_once_each_and_ban_remains_latched():
    p=pipeline()
    for trade in range(2):
        for i,(qty,px,left) in enumerate([(50,2.66,150),(100,2.62,50),(50,2.63,0)]):
            leg(p,qty,px,left,stamp=NOW+timedelta(seconds=trade*10+i))
            assert p._daily_loss_counts.get('NCPL',0)==trade+(left==0)
        assert ('NCPL' in p._daily_losers)==(trade==1)
    leg(p,100,2.70,0,stamp=NOW+timedelta(seconds=30))
    assert p._daily_loss_counts['NCPL']==0 and 'NCPL' in p._daily_losers

def test_dollar_guard_and_circuit_breaker_act_before_close():
    p=pipeline();p._max_daily_loss=-50
    leg(p,100,2.34,400)
    assert not p._daily_loss_counts and not p._daily_losers
    leg(p,100,2.34,300,stamp=NOW+timedelta(seconds=1))
    assert p._daily_pnl==pytest.approx(-60) and p._circuit_breaker_tripped
    assert 'NCPL' in p._daily_losers and not p._daily_loss_counts

def test_large_negative_leg_still_blacklists_after_profitable_partial():
    p=pipeline();leg(p,100,3.64,200);leg(p,100,2.04,100,stamp=NOW+timedelta(seconds=1))
    assert p._daily_pnl==pytest.approx(40) and 'NCPL' in p._daily_losers
    assert not p._daily_loss_counts

def test_partial_win_does_not_reset_streak_or_shorten_cooldown():
    p=pipeline();p._daily_loss_counts['NCPL']=1;p._exit_cooldowns['NCPL']=NOW
    leg(p,50,2.70,50)
    assert p._daily_loss_counts['NCPL']==1 and p._exit_cooldowns['NCPL']==NOW
    leg(p,50,2.50,0,stamp=NOW+timedelta(seconds=1))
    assert p._daily_loss_counts['NCPL']==2

def test_breakeven_close_preserves_streak():
    p=pipeline();p._daily_loss_counts['NCPL']=1
    leg(p,50,2.65,50);leg(p,50,2.63,0,stamp=NOW+timedelta(seconds=1))
    assert p._daily_loss_counts['NCPL']==1

def test_actual_remaining_overrides_exit_intent_and_removed_tracker():
    p=pipeline();pos=TrackedPosition(symbol='NCPL',side=Side.BUY,quantity=100,entry_price=2.64,remaining_qty=0);p.exit_manager.track(pos);pos.remaining_qty=0
    leg(p,40,2.60,60)
    assert not p._daily_loss_counts and 'NCPL' in p._realized_trade_legs
    p.exit_manager.untrack('NCPL');leg(p,60,2.68,0,stamp=NOW+timedelta(seconds=1))
    assert p._daily_loss_counts['NCPL']==0 and p._daily_pnl==pytest.approx(.8)

def test_changed_average_entry_and_competing_symbol_remain_separate():
    p=pipeline();leg(p,50,2.70,50);leg(p,10,9,0,symbol='OTHER',entry=10)
    leg(p,50,2.65,0,entry=2.70,stamp=NOW+timedelta(seconds=1))
    assert p._daily_pnl==pytest.approx(-9.5) and p._daily_loss_counts=={'NCPL':0,'OTHER':1}

def test_concurrent_duplicate_is_booked_once():
    p=pipeline()
    with ThreadPoolExecutor(max_workers=8) as pool:values=list(pool.map(lambda _:leg(p,100,2.60,0),range(20)))
    assert sum(values)==pytest.approx(-4)
    assert p._daily_pnl==pytest.approx(-4) and p._daily_loss_counts['NCPL']==1

def test_checkpoint_preserves_pending_totals_and_deduplication():
    source=checkpoint_runner();p=source._pipeline;leg(p,50,2.70,50)
    cp=capture_state(source,NOW);target=checkpoint_runner();restore_state(target,cp,NOW+timedelta(seconds=1));q=target._pipeline
    assert q._realized_trade_legs==p._realized_trade_legs and q._realized_exit_fills==p._realized_exit_fills
    before=q._daily_pnl;assert leg(q,50,2.70,50)==0 and q._daily_pnl==before
    leg(q,50,2.63,0,stamp=NOW+timedelta(seconds=2));assert q._daily_loss_counts['NCPL']==0

def test_old_flat_checkpoint_is_compatible():
    source=checkpoint_runner();cp=capture_state(source,NOW);del cp['guards']['exit_accounting'];cp['guards_sha256']=digest(cp['guards'])
    target=checkpoint_runner();restore_state(target,cp,NOW+timedelta(seconds=1))
    assert target._pipeline._daily_pnl==source._pipeline._daily_pnl
    assert target._pipeline._daily_loss_counts==source._pipeline._daily_loss_counts
    assert target._pipeline._realized_trade_legs=={} and target._pipeline._realized_exit_fills=={}

@pytest.mark.parametrize('bad',[{'legs':{},'fills':[['wrong',0]]},{'legs':{'NCPL':(1,-1,1)},'fills':[]},{'legs':{},'fills':[(('NCPL','sell',1,1,'bad'),1)]}])
def test_invalid_accounting_checkpoint_is_atomic(bad):
    cp=capture_state(checkpoint_runner(),NOW);cp['guards']['exit_accounting']=encode(bad);cp['guards_sha256']=digest(cp['guards'])
    target=checkpoint_runner();target._pipeline._daily_pnl=123
    with pytest.raises(ValueError):restore_state(target,cp,NOW)
    assert target._pipeline._daily_pnl==123 and not target._pipeline._realized_exit_fills

def reporting_runner(p):
    r=object.__new__(AlpacaRunner);r._pipeline=p;r._hub=SimpleNamespace(on_exit_fill=Mock(),add_log=Mock())
    r._journal=SimpleNamespace(record=Mock());r._exec_timer=SimpleNamespace(cancel=Mock());r._timed_signal_queue=deque();r._market_phase=lambda:'TEST'
    r._clear_broker_stop=Mock();r._refresh_broker_stop=Mock()
    return r

def test_pipeline_and_runner_share_fill_accounting():
    p=pipeline();f=Fill('NCPL',Side.SELL,100,2.60,NOW)
    p.record_realized_exit('NCPL',2.64,2.60,100,remaining_quantity=0,fill_key=p.exit_fill_key(f))
    r=reporting_runner(p);assert r._record_trade_exit(f,2.64,'stop')==pytest.approx(-4)
    assert r._record_trade_exit(f,2.64,'broker_fill_sync')==0
    assert p._daily_pnl==pytest.approx(-4) and p._daily_loss_counts['NCPL']==1 and r._hub.on_exit_fill.call_count==1

def test_runner_uses_filled_portfolio_even_if_tracker_zeroed_early():
    p=pipeline();p.portfolio.positions['NCPL']=Position('NCPL',100,2.64)
    p.exit_manager.track(TrackedPosition(symbol='NCPL',side=Side.BUY,quantity=100,entry_price=2.64,remaining_qty=0));p.exit_manager.tracked['NCPL'].remaining_qty=0;r=reporting_runner(p)
    f=Fill('NCPL',Side.SELL,40,2.60,NOW);apply_fill(p.portfolio,f);r._record_trade_exit(f,2.64,'partial')
    assert not p._daily_loss_counts and 'NCPL' in p._realized_trade_legs
    f=Fill('NCPL',Side.SELL,60,2.68,NOW+timedelta(seconds=1));apply_fill(p.portfolio,f);r._record_trade_exit(f,2.64,'close')
    assert p._daily_pnl==pytest.approx(.8) and p._daily_loss_counts['NCPL']==0

def test_pipeline_full_stop_books_entry_cost_after_manager_removes_tracker():
    from daytrading.execution.broker import PaperBroker
    from daytrading.pipeline.engine import TradingPipeline
    from daytrading.models import PortfolioState, Bar
    p=TradingPipeline([],{},PaperBroker(),PortfolioState(cash=10000));p._enable_daily_loser_blacklist=True
    p.portfolio.positions['TEST']=Position('TEST',100,10)
    p.exit_manager.track(TrackedPosition(symbol='TEST',side=Side.BUY,quantity=100,remaining_qty=100,entry_price=10,stop_loss=9.5,entry_ts=NOW-timedelta(minutes=5)))
    b=Bar(symbol='TEST',open=9,high=9,low=9,close=9,volume=10000,ts=NOW, provenance=BarSource.CANONICAL_INTERVAL)
    result=p.run_cycle({'TEST':[b]},now=NOW,allow_entries=False)
    assert len(result.exit_fills)==1 and p._daily_pnl==pytest.approx(-100)
    assert p._daily_loss_counts['TEST']==1 and not p._realized_trade_legs


def test_daily_reset_keeps_pending_trade_totals_and_exit_identity(monkeypatch):
    from test_daily_session_reset import _runner
    from datetime import datetime
    r=_runner();p=pipeline();r._pipeline=p;leg(p,50,2.70,50)
    before=dict(p._realized_exit_fills)
    monkeypatch.setattr(AlpacaRunner,'_is_trading_day',classmethod(lambda cls,when=None:True))
    assert r._maybe_daily_session_reset(datetime(2026,10,19,4,0),'PRE-MARKET')
    assert p._daily_pnl==0 and not p._daily_loss_counts
    assert p._realized_exit_fills==before and p._realized_trade_legs['NCPL'][0]==pytest.approx(3)
    assert leg(p,50,2.70,50)==0 and p._daily_pnl==0


@pytest.mark.parametrize('method',['_instant_trail_exit','_tape_pressure_exit'])
def test_partial_protective_exit_retains_remaining_position_and_stop(method):
    p=pipeline();p.portfolio.positions['NCPL']=Position('NCPL',100,2.64)
    p.exit_manager.track(TrackedPosition(symbol='NCPL',side=Side.BUY,quantity=100,remaining_qty=100,entry_price=2.64,stop_loss=2.60))
    r=reporting_runner(p);r._submit_protective_exit_order=lambda *a:(Fill('NCPL',Side.SELL,40,2.60,NOW),None)
    r._seed_recent_order_ids=Mock();r._push_positions_from_alpaca=Mock()
    getattr(r,method)('NCPL',2.60)
    assert p.portfolio.positions['NCPL'].quantity==60
    assert p.exit_manager.tracked['NCPL'].remaining_qty==60
    assert not p._daily_loss_counts and p._daily_pnl==pytest.approx(-1.6)
    r._clear_broker_stop.assert_not_called();r._refresh_broker_stop.assert_called_once_with('NCPL')


def test_delayed_broker_exit_batch_counts_one_winning_position(monkeypatch):
    from test_trade_history_sync import _order
    p=pipeline();r=reporting_runner(p);r._last_synced_order_ids={'buy'}
    r._now_et=lambda:NOW;r._push_positions_from_alpaca=Mock();r._lookup_entry_price_from_orders=lambda symbol:2.64
    orders=[_order(order_id='buy',symbol='NCPL',side='buy',status='filled',qty=500,price=2.64,ts=NOW)]
    for i,(qty,px) in enumerate([(200,2.69),(50,2.68),(100,2.63),(100,2.63),(50,2.64)]):
        orders.append(_order(order_id=str(i),symbol='NCPL',side='sell',status='filled',qty=qty,price=px,ts=NOW+timedelta(seconds=i+1)))
    r._broker=SimpleNamespace(get_positions=lambda:{},client=SimpleNamespace(get_orders=lambda **kw:list(reversed(orders))))
    r._check_new_fills()
    assert p._daily_pnl==pytest.approx(10) and p._daily_loss_counts['NCPL']==0
    assert not p._daily_losers and not p._realized_trade_legs
    assert len(r._last_synced_order_ids)==6
    r._check_new_fills();assert p._daily_pnl==pytest.approx(10) and r._hub.on_exit_fill.call_count==5


def test_broker_batch_keeps_two_separate_losing_positions_separate():
    from test_trade_history_sync import _order
    p=pipeline();r=reporting_runner(p);r._last_synced_order_ids={'buy0','buy1'}
    r._now_et=lambda:NOW;r._push_positions_from_alpaca=Mock();r._lookup_entry_price_from_orders=lambda symbol:2.64
    orders=[]
    for i in range(2):
        for j,(oid,side,qty,price) in enumerate([(f'buy{i}','buy',100,2.64),(f'part{i}','sell',40,2.63),(f'close{i}','sell',60,2.62)]):
            orders.append(_order(order_id=oid,symbol='NCPL',side=side,status='filled',qty=qty,price=price,ts=NOW+timedelta(seconds=i*10+j)))
    r._broker=SimpleNamespace(get_positions=lambda:{},client=SimpleNamespace(get_orders=lambda **kw:list(reversed(orders))))
    r._check_new_fills()
    assert p._daily_pnl==pytest.approx(-3.2) and p._daily_loss_counts['NCPL']==2
    assert 'NCPL' in p._daily_losers and not p._realized_trade_legs


def test_broker_batch_uses_each_round_trip_cost_not_today_average():
    from test_trade_history_sync import _order
    p=pipeline();r=reporting_runner(p);r._last_synced_order_ids={'buy0','buy1'};r._now_et=lambda:NOW
    r._push_positions_from_alpaca=Mock();r._lookup_entry_price_from_orders=lambda symbol:999
    orders=[]
    for i,(entry,exit_price) in enumerate([(2.0,2.1),(3.0,2.9)]):
        for j,(oid,side,price) in enumerate([(f'buy{i}','buy',entry),(f'close{i}','sell',exit_price)]):
            orders.append(_order(order_id=oid,symbol='NCPL',side=side,status='filled',qty=100,price=price,ts=NOW+timedelta(seconds=i*10+j)))
    r._broker=SimpleNamespace(get_positions=lambda:{},client=SimpleNamespace(get_orders=lambda **kw:list(reversed(orders))))
    r._check_new_fills()
    assert p._daily_pnl==pytest.approx(0) and p._daily_loss_counts['NCPL']==1
    assert not p._daily_losers
