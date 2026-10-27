"""Separate original micro wick evidence from minute/session context."""
from dataclasses import replace
from unittest.mock import Mock
import pytest
from daytrading.models import Timeframe,Quote
from daytrading.risk.guards import LiquidityTrapDetector,TradeGuard
from test_micro_confirmation_context import context

def configured(wick_bad=False):
 s,w,now,minutes=context();b=replace(w.bars[-1].bar,open=2.17,high=2.25 if wick_bad else 2.19,low=2.16,close=2.18,volume=10000)
 w=replace(w,bars=w.bars[:-1]+(replace(w.bars[-1],bar=b),));s=replace(s,scan_result=replace(s.scan_result,confirmation_window=w));g=TradeGuard();g.false_breakout.check=Mock(return_value=None)
 return g,s,w,now,minutes

def test_bad_minute_wick_does_not_replace_valid_micro_trigger():
 g,s,w,now,minutes=configured();minutes[-1]=replace(minutes[-1],open=2.1776,high=2.1896,low=2.14,close=2.18)
 assert 'spike-and-fade' in LiquidityTrapDetector().check(minutes)
 assert g.check_entry(s,bars=minutes,breakout_context=w,now=now)==(True,None)

@pytest.mark.parametrize('count',[0,3,5,6])
def test_bad_micro_wick_rejects_clean_or_short_minutes(count):
 g,s,w,now,minutes=configured(True);minutes=minutes[-count:] if count else []
 if minutes:minutes[-1]=replace(minutes[-1],open=2.17,high=2.19,low=2.16,close=2.18)
 ok,why=g.check_entry(s,bars=minutes,breakout_context=w,now=now)
 assert not ok and '10s spike-and-fade' in why and '0.070000' in why and '0.010000' in why

def test_flat_minute_cannot_hide_bad_micro_wick():
 g,s,w,now,minutes=configured(True);minutes[-1]=replace(minutes[-1],open=2.18,high=2.18,low=2.18,close=2.18)
 assert not g.check_entry(s,bars=minutes,breakout_context=w,now=now)[0]

def test_spread_still_uses_minute_context():
 g,s,w,now,minutes=configured();minutes[-1]=replace(minutes[-1],open=2.17,high=2.19,low=2.16,close=2.18,volume=1)
 q=Quote(s.symbol,now,2.17,2.19,100,100);ok,why=g.check_entry(s,bars=minutes,quotes=[q],breakout_context=w,now=now)
 assert not ok and 'spread' in why

def test_minute_gap_reversal_still_rejects():
 g,s,w,now,minutes=configured();minutes[-1]=replace(minutes[-1],open=2.20,high=2.21,low=2.11,close=2.12)
 ok,why=g.check_entry(s,bars=minutes,breakout_context=w,now=now)
 assert not ok and 'gap-up reversal' in why

def test_exact_original_candle_routed_with_minutes_unchanged():
 g,s,w,now,minutes=configured();g.liquidity_trap.check=Mock(return_value=None)
 assert g.check_entry(s,bars=minutes,breakout_context=w,now=now)==(True,None)
 g.liquidity_trap.check.assert_called_once_with(minutes,quotes=None,symbol=s.symbol,wick_bar=w.bars[-1].bar,
     now=now,diagnostics=s.scan_result.criteria['quality_inputs'])
 assert g.liquidity_trap.check.call_args.kwargs['wick_bar'] is w.bars[-1].bar

@pytest.mark.parametrize('fault',['symbol','timeframe','naive','nan','zero_volume','range'])
def test_invalid_explicit_wick_fails_closed(fault):
 _,s,w,now,minutes=configured();b=w.bars[-1].bar
 if fault=='symbol':b=replace(b,symbol='BAD')
 if fault=='timeframe':b=replace(b,timeframe=Timeframe.MIN_1)
 if fault=='naive':b=replace(b,ts=b.ts.replace(tzinfo=None))
 if fault=='nan':b=replace(b,high=float('nan'))
 if fault=='zero_volume':b=replace(b,volume=0)
 if fault=='range':b=replace(b,high=b.low-.01)
 assert 'invalid original 10s' in LiquidityTrapDetector().check(minutes,symbol=s.symbol,wick_bar=b)

@pytest.mark.parametrize('high,blocked',[(13,False),(13.001,True)])
def test_same_two_x_rule(high,blocked):
 _,s,w,now,minutes=configured();b=replace(w.bars[-1].bar,open=10,close=11,high=high,low=9)
 assert bool(LiquidityTrapDetector().check([],symbol=s.symbol,wick_bar=b)) is blocked

def test_non_micro_and_legacy_call_unchanged():
 _,_,_,_,minutes=configured();minutes[-1]=replace(minutes[-1],open=2.1776,high=2.1896,low=2.14,close=2.18)
 assert 'spike-and-fade' in LiquidityTrapDetector().check(minutes)
 assert LiquidityTrapDetector().check([]) is None
