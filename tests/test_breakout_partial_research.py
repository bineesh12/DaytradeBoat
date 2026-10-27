from daytrading.models import BarSource
from dataclasses import replace
from datetime import timedelta
import pytest
from test_micro_paper_runtime import window,minutes,AT,quote
from daytrading.models import Tick,Side,CompletedBar,Bar,Timeframe
from daytradeboat_studies.micro_pullback import evaluate_micro_pullback,MicroPullbackRules,Rejection
from daytradeboat_studies.breakout_volume import participation,participation_reject
from daytradeboat_studies.partial_structure import decide

def tape(w=None,kind='buy'):
 w=w or window();bar=w[-1];ticks=[];quotes=[]
 for i in range(20):
  at=bar.bar.ts+timedelta(seconds=.1+i*.4)
  ticks.append(Tick('TEST',at,2.41 if kind=='buy' else 2.40,500,Side.BUY))
  quotes.append(replace(quote(),ts=at,bid=2.39,ask=2.41))
 return ticks,quotes

def test_no_sticky_side_inside_spread_or_stale_or_future_quotes():
 w=window();ticks,qs=tape(kind='unknown');e=participation('TEST',w,ticks,qs,now=AT)
 assert e['unknown_volume']==10000 and e['classified_fraction']==0
 assert participation_reject(e).code=='volume_expansion'
 qs=[replace(q,ts=q.ts+timedelta(seconds=20)) for q in qs]
 assert participation('TEST',w,ticks,qs,now=AT)['unknown_volume']==10000


def test_exact_volume_window_excludes_closing_trade_and_future_ticks():
 w=window();t,q=tape();e=participation('TEST',w,t+[Tick('TEST',w[-1].closed_at,2.5,5000,Side.BUY)],q,now=AT)
 assert e['total']==10000 and e['volume_reconciles'] and e['buy_fraction']==1


def test_baseline_candidate_unchanged_when_entry_enabled():
 w=window();a=evaluate_micro_pullback('TEST',w,now=AT,minute_bars=minutes())
 b=evaluate_micro_pullback('TEST',w,now=AT,minute_bars=minutes(),rules=MicroPullbackRules(entry_mode='body_volume'))
 assert a==b and not isinstance(a,Rejection)


def body_window():
 w=list(window());w[-2]=replace(w[-2],bar=replace(w[-2].bar,high=2.43))
 w[-1]=replace(w[-1],bar=replace(w[-1].bar,volume=20000))
 return tuple(w)


def test_body_entry_needs_full_causal_volume_and_all_shared_shape_checks():
 w=body_window();t,q=tape(w);t=[replace(x,size=1000) for x in t]
 assert evaluate_micro_pullback('TEST',w,now=AT,minute_bars=minutes()).code=='reclaim'
 result=evaluate_micro_pullback('TEST',w,now=AT,minute_bars=minutes(),ticks=t,quotes=q,rules=MicroPullbackRules(entry_mode='body_volume'))
 assert not isinstance(result,Rejection) and result.technical_stop==2.33
 for tt,qq,code in [(t[:2],q,'volume_capture'),(t,[],'volume_unknown')]:
  assert evaluate_micro_pullback('TEST',w,now=AT,minute_bars=minutes(),ticks=tt,quotes=qq,rules=MicroPullbackRules(entry_mode='body_volume')).code==code
 w=list(w);w[-1]=replace(w[-1],bar=replace(w[-1].bar,high=2.60))
 assert evaluate_micro_pullback('TEST',w,now=AT,minute_bars=minutes(),ticks=t,quotes=q,rules=MicroPullbackRules(entry_mode='body_volume')).code=='close_location'


def position():return dict(symbol='TEST',entry=2.,initial_stop=1.9,qty=100,remaining=100,entry_at=AT)
def exit_at(seconds=20,bid=2.11,partial=0,pending='',completed=(),previous=1.9,remainder=False,r=None):
 now=AT+timedelta(seconds=seconds)
 return decide(r or position(),replace(quote(),ts=now,bid=bid,ask=bid+.01),completed,now=now,previous_stop=previous,partial_filled=partial,pending_reason=pending,entry_remainder=remainder)

def test_partial_submission_cannot_move_stop_or_duplicate_order():
 d=exit_at();assert d.quantity==50 and d.stop==1.9 and d.reason=='half_at_1R'
 assert exit_at(partial=49,pending='half_at_1R').stop==1.9
 assert exit_at(pending='half_at_1R').quantity==0
 assert exit_at(partial=50).stop==2
 assert exit_at(partial=50,remainder=True).stop==1.9


def test_stop_supersedes_pending_half_on_actual_remaining_quantity():
 d=exit_at(bid=1.89,partial=20,pending='half_at_1R',r=dict(position(),remaining=80))
 assert d.quantity==80 and d.reason=='structure_stop'
 assert exit_at(bid=1.88,partial=20,pending='structure_stop').quantity==0


def pivot():
 start=AT+timedelta(seconds=1);vals=[(2.05,2.10,2.03,2.08)]*3+[(2.10,2.15,2.05,2.13),(2.12,2.16,2.03,2.08),(2.08,2.20,2.06,2.19)]
 return [CompletedBar(Bar('TEST',start+timedelta(seconds=i*10),*v,10000,Timeframe.SEC_10, provenance=BarSource.CANONICAL_INTERVAL),start+timedelta(seconds=(i+1)*10)) for i,v in enumerate(vals)]

def test_pivot_requires_right_bar_completion_and_never_widens():
 w=pivot()
 assert exit_at(seconds=61,partial=50,completed=w,bid=2.2).stop==pytest.approx(2.02)
 assert exit_at(seconds=60,partial=50,completed=w,bid=2.2).stop==2
 assert exit_at(seconds=61,partial=50,completed=w,bid=2.2,previous=2.06).stop==2.06
 assert exit_at(seconds=61,partial=49,completed=w,bid=2.2).stop==1.9


def test_timeouts_and_unfilled_remainder_are_explicit():
 assert exit_at(seconds=60,bid=2).reason=='no_partial_timeout_60s'
 assert exit_at(seconds=179,bid=2.1,partial=50).quantity==0
 assert exit_at(seconds=180,bid=2.1,partial=50).reason=='runner_timeout_180s'


def test_stale_quote_and_other_symbol_cannot_trigger_exit():
 now=AT+timedelta(seconds=20)
 d=decide(position(),replace(quote(),ts=AT),[],now=now,previous_stop=1.9,partial_filled=0)
 assert not d.quantity
 for bars in [[replace(x,bar=replace(x.bar,symbol='OTHER')) for x in pivot()]]:
  assert exit_at(seconds=61,partial=50,completed=bars,bid=2.2).stop==2
