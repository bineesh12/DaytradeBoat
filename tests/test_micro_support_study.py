from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from test_micro_paper_runtime import AT,minutes,window,quote
from daytrading.models import ConfirmationWindow,ScanResult,TradeSignal,SignalAction,PortfolioState
from daytradeboat_studies.micro_pullback import MicroPullbackRules,MicroRiskLimits,Rejection,evaluate_micro_pullback
from daytradeboat_studies.micro_pullback_adapter import NAME,MicroPullbackVerifier,candidate_from_hit,bounded_micro_order


def base(symbol='TEST'):
 w=list(window(symbol));w[-3]=replace(w[-3],bar=replace(w[-3].bar,low=2.34))
 return tuple(w)


def evaluate(w=None,**kw):
 return evaluate_micro_pullback('TEST',w or base(),now=kw.pop('now',AT),minute_bars=kw.pop('minute_bars',minutes()),rules=MicroPullbackRules(base_mode='support_reclaim',**kw))


def make_hit(w=None):
 w=w or base();c=evaluate(w)
 return ScanResult('TEST',NAME,AT,c.score,criteria=dict(pattern=NAME,setup_anchor=c.anchor,stop_price=c.technical_stop,micro_base_mode='support_reclaim',micro_stop_buffer=.01,
     micro_setup_id=c.setup_id,micro_bar_start=c.bar_start.isoformat(),micro_confirmed_at=c.confirmed_at.isoformat()),bars=minutes(),confirmation_window=ConfirmationWindow(w,AT))


def test_default_retains_old_lower_low_rejection():
 assert MicroPullbackRules().base_mode=='rising_lows'
 assert evaluate_micro_pullback('TEST',base(),now=AT,minute_bars=minutes()).code=='lows'


@pytest.mark.parametrize('symbol',['TEST','MEDS','NCPL','BENF','GENK','OTHER'])
def test_generic_support_recovery_and_structural_stop(symbol):
 w=base(symbol);c=evaluate_micro_pullback(symbol,w,now=AT,minute_bars=minutes(symbol),rules=MicroPullbackRules(base_mode='support_reclaim'))
 assert not isinstance(c,Rejection)
 assert c.technical_stop==2.32 and c.technical_stop<min(w[-3].bar.low,w[-2].bar.low)
 assert c.technical_stop<w[-1].bar.low-.01


def test_trigger_must_hold_support_even_if_it_closes_strong():
 w=list(base());w[-1]=replace(w[-1],bar=replace(w[-1].bar,low=2.3299))
 assert evaluate(w).code=='support_break'


@pytest.mark.parametrize('kind,code',[('weak_close','close_location'),('no_reclaim','reclaim'),('sparse','sparse_bar'),('future','future'),('stale','stale'),('no_impulse','impulse_context')])
def test_all_other_confirmation_and_context_gates_remain(kind,code):
 w=list(base());kw={}
 if kind=='weak_close':w[-1]=replace(w[-1],bar=replace(w[-1].bar,close=2.39,high=2.50))
 if kind=='no_reclaim':w[-1]=replace(w[-1],bar=replace(w[-1].bar,close=2.38))
 if kind=='sparse':w[0]=replace(w[0],closed_at=w[0].closed_at+timedelta(seconds=6))
 if kind=='future':kw['now']=w[-1].closed_at-timedelta(seconds=1)
 if kind=='stale':kw['now']=AT+timedelta(seconds=16)
 if kind=='no_impulse':kw['minute_bars']=[]
 assert evaluate(w,**kw).code==code


def test_base_stop_survives_adapter_verifier_and_bounded_order(monkeypatch):
 h=make_hit();c=candidate_from_hit(h);assert c.technical_stop==2.32
 import daytradeboat_studies.micro_pullback_adapter as rt
 guard=Mock(return_value=None);monkeypatch.setattr(rt,'check_entry_quality',guard)
 v=MicroPullbackVerifier(MicroRiskLimits());v._quote_buffer={'TEST':[quote()]}
 sig=v.verify(h,PortfolioState(cash=25000),now=AT);assert sig.stop_loss==2.32
 o=bounded_micro_order(sig,quote(),minutes()[-1],now=AT,limits=MicroRiskLimits())
 assert not isinstance(o,Rejection) and o.quantity<=500
 assert o.quantity*max(o.limit_price-sig.stop_loss,.05,o.limit_price*.03)<=50+1e-8
 assert guard.called
 changed=replace(sig,stop_loss=2.33)
 assert bounded_micro_order(changed,quote(),minutes()[-1],now=AT,limits=MicroRiskLimits()).code=='stop'


def test_declared_stop_cannot_disagree_with_completed_base_evidence():
 h=make_hit();h.criteria['stop_price']=2.34
 assert candidate_from_hit(h).code=='stop_evidence'
 h=make_hit();h.criteria['micro_base_mode']='unknown'
 assert candidate_from_hit(h).code=='stop_evidence'


def test_shared_spread_guard_still_rejects_support_variant(monkeypatch):
 import daytradeboat_studies.micro_pullback_adapter as rt
 monkeypatch.setattr(rt,'check_entry_quality',Mock(return_value='spread blocked'))
 v=MicroPullbackVerifier(MicroRiskLimits());v._quote_buffer={'TEST':[quote()]}
 assert v.verify(make_hit(),PortfolioState(cash=25000),now=AT) is None
 assert v._last_reject=='spread blocked'


def test_base_stop_never_clamped_to_force_wide_trade():
 h=make_hit();sig=TradeSignal('TEST',SignalAction.ENTER_LONG,500,2.41,2.32,2.59,scan_result=h)
 assert bounded_micro_order(sig,replace(quote(),ask=2.6),minutes()[-1],now=AT,limits=MicroRiskLimits()).code=='chase'


def test_invalid_mode_rejected():
 with pytest.raises(ValueError,match='base mode'):MicroPullbackRules(base_mode='stock_specific')


def test_same_base_floor_measures_depth_and_stop_without_lowering_minimum():
 # Generic recovery: preceding candle alone dipped <1%, full base dipped >3%.
 w=list(window());values=[(4.10,4.20,4.10,4.18),(4.18,4.24,4.15,4.22),
  (4.22,4.24,4.12,4.15),(4.14,4.25,4.12,4.23),
  (4.21,4.25,4.21,4.2331),(4.2417,4.33,4.20,4.33)]
 for i,v in enumerate(values):w[i]=replace(w[i],bar=replace(w[i].bar,open=v[0],high=v[1],low=v[2],close=v[3]))
 c=evaluate(w)
 assert not isinstance(c,Rejection)
 assert c.technical_stop==4.11
 # The new pattern still needs a full 1% pullback from its actual base floor.
 flat=[replace(x,bar=replace(x.bar,open=4.22,high=4.25,low=4.22,close=4.23)) for x in w]
 flat[-1]=replace(flat[-1],bar=replace(flat[-1].bar,open=4.24,high=4.33,low=4.22,close=4.33))
 assert evaluate(flat).code=='pullback'
