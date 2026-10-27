"""Market-time calibration must not trust tape, widen freshness, or move OS time."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import pytest
from daytrading.data.market_clock import MarketClock

AT = datetime(2026,10,16,11,2,5,tzinfo=timezone.utc)

class Time:
    def __init__(self):self.mono=100.;self.wall_jump=0
    def wall(self):return AT+timedelta(seconds=self.mono-100+self.wall_jump)
    def monotonic(self):return self.mono
    def advance(self,s):self.mono+=s
    def reader(self,offset=.844,rtt=.12):
        def read():
            midpoint=self.wall()+timedelta(seconds=offset+rtt/2)
            self.advance(rtt)
            return SimpleNamespace(timestamp=midpoint)
        return read

def test_broker_ahead_alignment_and_monotonic_holdover():
    t=Time();c=MarketClock(wall=t.wall,monotonic=t.monotonic)
    assert c.now() is None
    assert c.synchronize(t.reader())
    assert (c.now()-t.wall()).total_seconds()==pytest.approx(.844)
    before=c.now();t.wall_jump=-30;t.advance(2)
    assert (c.now()-before).total_seconds()==2
    assert c.status()['offset_seconds']==pytest.approx(.844)

@pytest.mark.parametrize('offset,rtt',[(6,.12),(-6,.12),(.844,1.1)])
def test_unbounded_offset_and_slow_response_fail_closed(offset,rtt):
    t=Time();c=MarketClock(wall=t.wall,monotonic=t.monotonic)
    assert not c.synchronize(t.reader(offset,rtt)) and c.now() is None

def test_failed_refresh_retains_only_a_fresh_previously_valid_anchor():
    t=Time();c=MarketClock(wall=t.wall,monotonic=t.monotonic)
    assert c.synchronize(t.reader())
    fail=lambda:(_ for _ in ()).throw(RuntimeError('down'))
    t.advance(100);assert not c.synchronize(fail) and c.now() is not None
    t.advance(81);assert c.now() is None and not c.status()['synchronized']

def test_utc_identity_and_wall_jump_during_request_rejected():
    t=Time();c=MarketClock(wall=t.wall,monotonic=t.monotonic)
    def bad():t.advance(.1);return SimpleNamespace(timestamp=AT.replace(tzinfo=None))
    assert not c.synchronize(bad) and c.now() is None
    def jump():
        server=t.wall();t.wall_jump+=1;t.advance(.1)
        return SimpleNamespace(timestamp=server)
    assert not c.synchronize(jump) and c.now() is None

def test_monotonic_rollback_invalidates_anchor():
    t=Time();c=MarketClock(wall=t.wall,monotonic=t.monotonic)
    assert c.synchronize(t.reader());t.mono-=10
    assert c.now() is None

def test_disagreeing_broker_samples_cannot_select_an_outlier():
    t=Time();c=MarketClock(wall=t.wall,monotonic=t.monotonic)
    reads=iter([t.reader(4,.01),t.reader(.844,.12),t.reader(.844,.12)])
    assert not c.synchronize(lambda:next(reads)()) and c.now() is None
