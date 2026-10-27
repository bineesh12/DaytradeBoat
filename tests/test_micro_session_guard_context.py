from daytrading.models import BarSource
from dataclasses import replace
from datetime import timedelta
from unittest.mock import Mock
import pytest
from test_micro_paper_runtime import AT,hit,signal,window,minutes
from daytrading.models import Bar,ConfirmationWindow,Timeframe
from daytrading.risk.guards import FalseBreakoutDetector,TradeGuard,_completed_session_context


def micro():
 w=list(window());w[-1]=replace(w[-1],bar=replace(w[-1].bar,volume=7000))
 return tuple(w)


def context(volume=100000):
 return [Bar('TEST',AT.replace(second=0,microsecond=0)-timedelta(minutes=i),2,2.3,1.9,2.2,volume, provenance=BarSource.CANONICAL_INTERVAL) for i in range(7,1,-1)]


def test_existing_exception_uses_session_volume_vwap_not_six_micro_bars():
 bs=[x.bar for x in micro()];guard=FalseBreakoutDetector()
 assert 'volume declining' in guard.check(bs,symbol='TEST')
 assert guard.check(bs,symbol='TEST',session_bars=context(),now=AT) is None
 # Threshold stays 500K; a 480K session does not qualify.
 assert 'volume declining' in guard.check(bs,symbol='TEST',session_bars=context(80000),now=AT)


def test_original_completed_micro_ratio_remains_the_volume_evidence():
 bs=[x.bar for x in micro()];assert bs[-1].volume/(sum(b.volume for b in bs[:-1])/5)==.7
 # Good one-minute volume cannot excuse lack of VWAP reclaim or proximity to HOD.
 bad=[replace(b,open=2.6,high=2.8,low=2.5,close=2.7) for b in context()]
 assert 'volume declining' in FalseBreakoutDetector().check(bs,symbol='TEST',session_bars=bad,now=AT)


def test_trade_guard_routes_both_timeframes_without_changing_confirmation():
 w=micro();h=hit();h=replace(h,confirmation_window=ConfirmationWindow(w,AT))
 sig=replace(signal(),scan_result=h);g=TradeGuard();g.liquidity_trap.check=Mock(return_value=None)
 ok,reason=g.check_entry(sig,bars=context(),breakout_context=h.confirmation_window,now=AT)
 assert ok,reason
 assert g.liquidity_trap.check.call_args.kwargs['wick_bar']==w[-1].bar


@pytest.mark.parametrize('kind',['future','partial','wrong_day','wrong_symbol','micro_timeframe'])
def test_unavailable_or_unrelated_context_cannot_grant_exception(kind):
 bs=context(50000);extra=replace(bs[0],volume=1000000)
 if kind=='future':extra=replace(extra,ts=AT+timedelta(minutes=1))
 if kind=='partial':extra=replace(extra,ts=AT.replace(second=0,microsecond=0))
 if kind=='wrong_day':extra=replace(extra,ts=extra.ts-timedelta(days=1))
 if kind=='wrong_symbol':extra=replace(extra,symbol='OTHER')
 if kind=='micro_timeframe':extra=replace(extra,timeframe=Timeframe.SEC_10)
 assert len(_completed_session_context(bs+[extra],'TEST',AT))==6
 assert 'volume declining' in FalseBreakoutDetector().check([x.bar for x in micro()],symbol='TEST',session_bars=bs+[extra],now=AT)


def test_duplicates_cannot_multiply_volume_and_conflicts_disable_exception():
 bs=context(50000);assert len(_completed_session_context(bs+bs,'TEST',AT))==6
 assert 'volume declining' in FalseBreakoutDetector().check([x.bar for x in micro()],symbol='TEST',session_bars=bs+bs,now=AT)
 assert _completed_session_context(context()+[replace(context()[0],volume=1)],'TEST',AT)==[]


@pytest.mark.parametrize('change',[{'volume':float('nan')},{'low':3.0},{'volume':-1}])
def test_corrupt_eligible_session_context_fails_closed(change):
 bs=context();bs[0]=replace(bs[0],**change)
 assert _completed_session_context(bs,'TEST',AT)==[]
 assert 'volume declining' in FalseBreakoutDetector().check([x.bar for x in micro()],symbol='TEST',session_bars=bs,now=AT)


def test_absent_clock_or_session_context_does_not_invent_exception():
 bs=[x.bar for x in micro()]
 assert 'volume declining' in FalseBreakoutDetector().check(bs,symbol='TEST',session_bars=[],now=AT)
 assert 'volume declining' in FalseBreakoutDetector().check(bs,symbol='TEST',session_bars=context())
