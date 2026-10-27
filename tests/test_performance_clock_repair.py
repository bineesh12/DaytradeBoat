from daytrading.models import BarSource
"""Exact evidence/risk semantics, with the currently deployed Project A as oracle."""
import importlib.util,json,math,random,sys,queue
from pathlib import Path
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace as NS
from unittest.mock import Mock
from zoneinfo import ZoneInfo
import pytest
from daytrading.models import Bar,Quote,Tick,CompletedBar,Side
from daytrading.replay.codec import pack
from daytrading.reliability_metrics import StageMetrics
from daytrading.indicators.core import relative_volume,relative_volume_last
from daytrading.data.market_clock import MarketClock
from daytrading.strategy.scalping.micro_pullback_runtime import MicroPullbackScanner
from daytrading.strategy.scalping.micro_flow import MicroFlowCollector
from daytrading.runner import AlpacaRunner

BASE=Path(__file__).resolve().parent/'reference/performance_core'
AT=datetime(2026,10,23,13,tzinfo=timezone.utc)
def old(name):
    spec=importlib.util.spec_from_file_location('oracle_'+name.replace('/','_'),BASE/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod

def test_encoding_bytes_aliases_cycles_mutations_and_external_precedence():
    oracle=old('replay/codec');rng=random.Random(826)
    for i in range(100):
        b=Bar('TEST',AT,2.,2.1,1.9,2.+rng.random()/100,100+i, provenance=BarSource.CANONICAL_INTERVAL)
        q=Quote('TEST',AT,2.,2.01,100,100);t=Tick('TEST',AT,2.01,100,Side.BUY)
        x=dict(values=[b,q,t,CompletedBar(b,AT)],same=b,strings=['A','B'],mixed=['A',Side.BUY,1],floats=[float('nan'),float('inf'),-0.0])
        x['cycle']=x;object.__setattr__(b,'extra',x);object.__setattr__(b,'secret','hidden')
        externals={'q':q,'primitive':x['strings'][0],'timestamp':AT} if i%2 else None
        for phase in range(2):
            assert json.dumps(pack(x,externals),separators=(',',':'))==json.dumps(oracle.pack(x,externals),separators=(',',':'))
            object.__setattr__(b,'correction',phase+1)

def test_metrics_counters_and_histograms_exact_including_negative_durations():
    oracle=old('reliability_metrics').StageMetrics();actual=StageMetrics();rng=random.Random(1101)
    for i in range(10000):
        name='stage'+str(i%7);duration=rng.randrange(-10,10**8)
        oracle.add(name,duration);actual.add(name,duration)
    assert actual.rows==oracle.rows and actual.status()==oracle.status()

def test_metrics_allocate_only_for_new_stage(monkeypatch):
    import daytrading.reliability_metrics as mod
    calls=[];real=dict
    monkeypatch.setattr(mod,'dict',lambda *a,**k:calls.append(1) or real(*a,**k),raising=False)
    m=StageMetrics()
    for i in range(10000):m.add('stage',i)
    assert len(calls)==1 and m.rows['stage']['count']==10000

@pytest.mark.parametrize('size',[2,3,20,21,30,120,300])
def test_latest_volume_is_bit_identical_and_not_cached(size):
    rng=random.Random(size);bars=[Bar('X',AT,2,2,2,2,rng.random()*10000, provenance=BarSource.CANONICAL_INTERVAL) for _ in range(size)]
    for p in (1,min(20,size-1),size,size+1):
        expected=relative_volume(bars,p)[-1];actual=relative_volume_last(bars,p)
        assert math.isnan(actual) if math.isnan(expected) else actual.hex()==expected.hex()
    object.__setattr__(bars[-2],'volume',13.);assert relative_volume_last(bars,1).hex()==relative_volume(bars,1)[-1].hex()

@pytest.mark.parametrize('value',['invalid',10**400,True])
def test_latest_volume_keeps_legacy_unusual_input_behavior(value):
    bars=[Bar('X',AT,2,2,2,2,100, provenance=BarSource.CANONICAL_INTERVAL) for _ in range(30)];object.__setattr__(bars[0],'volume',value)
    try:expected=relative_volume(bars,2)[-1]
    except Exception as exc:
        with pytest.raises(type(exc)):relative_volume_last(bars,2)
    else:assert relative_volume_last(bars,2)==expected

def test_empty_retry_state_has_no_clock_observation_but_active_state_still_does():
    s=MicroPullbackScanner(flow=MicroFlowCollector());s._now=Mock(side_effect=AssertionError('unused clock'))
    assert s.needs_recheck is False;s._now.assert_not_called()
    r=AlpacaRunner.__new__(AlpacaRunner);r._event_queue=queue.Queue();r._cycle_interval=1.
    r._broker=NS();r._exec_timer=NS(seconds_until_next_timeout=lambda:None)
    r._hub=NS(trading_paused=False);r._pipeline=NS(_coordinated_micro_flow=s.flow,_scanners=[s]);r._micro_market_now=Mock(side_effect=AssertionError('unused clock'))
    assert r._event_wait_timeout()==1.;r._micro_market_now.assert_not_called()
    from test_micro_paper_runtime import hit,AT as T
    s.vwap_retries.rows['TEST']=hit();s.vwap_retries.dirty.add('TEST');s._now=Mock(return_value=T)
    assert s.needs_recheck;s._now.assert_called_once()
    r._micro_market_now=Mock(return_value=T);r._is_in_pattern_scan_window=lambda _:True
    assert r._event_wait_timeout()==.001;r._micro_market_now.assert_called_once()

def test_clock_rejection_keeps_measured_values_without_relaxing_bounds():
    mono=[100.];wall=lambda:AT+timedelta(seconds=mono[0]-100)
    def read():
        server=wall()+timedelta(seconds=5.198+.05);mono[0]+=.1
        return NS(timestamp=server.astimezone(ZoneInfo('America/New_York')))
    c=MarketClock(wall=wall,monotonic=lambda:mono[0]);assert not c.synchronize(read);assert c.now() is None
    s=c.status();assert s['anchor_state']=='missing' and not s['synchronized']
    d=s['last_calibration'];assert d['accepted_samples']==0 and d['limits']==dict(max_age_seconds=180.,max_rtt_seconds=1.,max_offset_seconds=5.)
    assert all(x['offset_seconds']==pytest.approx(5.198) for x in d['samples'])
    assert all(x['reason']=='broker clock offset exceeds bound' for x in d['samples'])
    d['samples'][0]['offset_seconds']=0;d['limits']['max_offset_seconds']=100
    assert c.status()['last_calibration']['samples'][0]['offset_seconds']==pytest.approx(5.198)
    assert c.status()['last_calibration']['limits']['max_offset_seconds']==5.

def test_clock_policy_and_timezone_conversion_match_deployed_for_all_edges():
    oracle=old('data/market_clock')
    for offset,rtt in [(0,.1),(.844,.1),(4.999,.1),(5.001,.1),(-5.001,.1),(.8,1.01)]:
        times=[[100.],[100.]];clocks=[]
        for cls,m in zip((oracle.MarketClock,MarketClock),times):
            wall=lambda m=m:AT+timedelta(seconds=m[0]-100)
            def reader(m=m,wall=wall):
                stamp=wall()+timedelta(seconds=offset+rtt/2);m[0]+=rtt
                return NS(timestamp=stamp.astimezone(ZoneInfo('Europe/Stockholm')))
            c=cls(wall=wall,monotonic=lambda m=m:m[0]);ok=c.synchronize(reader);clocks.append(c)
            assert ok==(abs(offset)<=5 and rtt<=1)
        for step in (0,100,81,-400):
            for m in times:m[0]+=step
            assert clocks[0].now()==clocks[1].now()
            assert clocks[0].status()['synchronized']==clocks[1].status()['synchronized']
