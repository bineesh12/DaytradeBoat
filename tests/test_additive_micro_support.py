from dataclasses import replace
import pytest
from test_micro_paper_runtime import AT,window,minutes,quote
from test_micro_support_study import base
from daytrading.models import ConfirmationWindow,ScanResult,PortfolioState
from daytradeboat_studies.micro_pullback import MicroPullbackRules,Rejection,evaluate_micro_pullback,micro_technical_stop,effective_base_mode
from daytradeboat_studies.micro_pullback_adapter import NAME,MicroPullbackVerifier,bounded_micro_order,candidate_from_hit

def run(w,mode):
 return evaluate_micro_pullback('TEST',w,now=AT,minute_bars=minutes(),rules=MicroPullbackRules(base_mode=mode))

@pytest.mark.parametrize('floor',[2.31,2.32,2.33])
def test_existing_strict_evidence_is_identical(floor):
 w=list(window());w[-3]=replace(w[-3],bar=replace(w[-3].bar,low=floor))
 a=run(w,'rising_lows');b=run(w,'support_fallback')
 assert not isinstance(a,Rejection) and a==b
 assert effective_base_mode(w,'support_fallback')=='rising_lows'
 assert micro_technical_stop(w,base_mode='support_fallback')==micro_technical_stop(w)


def test_only_failed_low_shape_uses_support():
 w=base();assert run(w,'rising_lows').code=='lows'
 assert run(w,'support_fallback')==run(w,'support_reclaim')
 assert effective_base_mode(w,'support_fallback')=='support_reclaim'
 w=list(w);w[-1]=replace(w[-1],bar=replace(w[-1].bar,low=2.32))
 assert run(w,'support_fallback').code=='support_break'

@pytest.mark.parametrize('change',['weak','reclaim','shallow'])
def test_no_fallback_for_other_strict_failures(change):
 w=list(window())
 if change=='weak':w[-1]=replace(w[-1],bar=replace(w[-1].bar,high=2.60))
 if change=='reclaim':w[-1]=replace(w[-1],bar=replace(w[-1].bar,close=2.38))
 if change=='shallow':w[-2]=replace(w[-2],bar=replace(w[-2].bar,low=2.34));w[-1]=replace(w[-1],bar=replace(w[-1].bar,low=2.34))
 assert run(w,'rising_lows')==run(w,'support_fallback')

@pytest.mark.parametrize('w',[window(),base()])
def test_adapter_and_bounded_order_use_same_branch(w,monkeypatch):
 import daytradeboat_studies.micro_pullback_adapter as rt
 monkeypatch.setattr(rt,'check_entry_quality',lambda *a,**kw:None)
 sigs=[];orders=[]
 for mode in ('support_fallback',effective_base_mode(w,'support_fallback')):
  c=run(w,mode);assert not isinstance(c,Rejection)
  h=ScanResult('TEST',NAME,AT,c.score,criteria=dict(pattern=NAME,setup_anchor=c.anchor,stop_price=c.technical_stop,micro_base_mode=mode,micro_stop_buffer=.01,
      micro_setup_id=c.setup_id,micro_bar_start=c.bar_start.isoformat(),micro_confirmed_at=c.confirmed_at.isoformat()),bars=minutes(),confirmation_window=ConfirmationWindow(w,AT))
  assert candidate_from_hit(h).technical_stop==c.technical_stop
  v=MicroPullbackVerifier(rt.MicroRiskLimits());v._quote_buffer={'TEST':[quote()]}
  sig=v.verify(h,PortfolioState(cash=25000),now=AT);sigs.append(sig)
  orders.append(bounded_micro_order(sig,quote(),minutes()[-1],now=AT,limits=rt.MicroRiskLimits()))
 assert sigs[0].stop_loss==sigs[1].stop_loss and sigs[0].quantity==sigs[1].quantity
 assert orders[0]==orders[1] and not isinstance(orders[0],Rejection)
