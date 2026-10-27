"""Native retry semantics. Run only with the offline guard installed first."""
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from daytrading.strategy.scalping.event_reclaim import EventReclaimScanner
from daytrading.strategy.scalping.micro_flow import CoordinatedMicroRetries
from daytrading.strategy.scalping.micro_vwap import deadline, identity
from daytrading.replay.codec import pack
from daytrading.replay.engine import LiveOperation
from daytrading.runner import QuoteEvent, TradeEvent
from daytrading.models import Quote, Tick, Side
from test_micro_paper_runtime import AT, hit, minutes


def scanner():
    return EventReclaimScanner(flow=NS(counts={},observe=lambda *a,**k:None))


def test_native_empty_book_avoids_an_unused_failing_clock():
    s=scanner();s._now=Mock(side_effect=AssertionError('unused clock observed'))
    before=pack((s.pending,s.vwap_retries))
    assert s.needs_recheck is False
    s._now.assert_not_called()
    assert pack((s.pending,s.vwap_retries))==before


@pytest.mark.parametrize('pending',[False,True])
def test_pending_evidence_keeps_priority_over_retry_and_clock(pending):
    s=scanner();s._now=Mock(return_value=AT)
    if pending:s.pending['TEST']=object()
    assert s.needs_recheck is pending
    s._now.assert_not_called()


@pytest.mark.parametrize('dirty',[False,True])
def test_noncoordinated_path_keeps_dirty_semantics(dirty):
    s=EventReclaimScanner();s._now=Mock(side_effect=AssertionError('nonflow clock'))
    if dirty:s.vwap_retries.dirty.add('TEST')
    assert s.needs_recheck is dirty
    s._now.assert_not_called()


@pytest.mark.parametrize('subclass',[False,True])
def test_unknown_adapter_and_subclass_preserve_original_clock_path(subclass):
    s=scanner();s._now=Mock(return_value=AT)
    if subclass:
        class DifferentBook(CoordinatedMicroRetries):
            def ready(self,now):return now is AT
        s.vwap_retries=DifferentBook(s.flow)
    else:s.vwap_retries=NS(rows={},ready=Mock(return_value=True))
    assert s.needs_recheck is True
    s._now.assert_called_once_with()
    if not subclass:s.vwap_retries.ready.assert_called_once_with(AT)


@pytest.mark.parametrize('clock_state',['fresh','missing','future','expired'])
@pytest.mark.parametrize('blocked',[False,True])
def test_nonempty_book_preserves_fresh_clock_blocked_and_expiry_semantics(clock_state,blocked):
    s=scanner();h=hit();b=s.vwap_retries
    b.rows[h.symbol]=h;b.dirty.add(h.symbol)
    if blocked:b.blocked[h.symbol]='broker_position_sync_stale'
    now=None if clock_state=='missing' else deadline(h) if clock_state=='expired' else AT-timedelta(seconds=1) if clock_state=='future' else AT
    s._now=Mock(return_value=now)
    before=pack((s.pending,b))
    expected=b.ready(now)
    assert s.needs_recheck is expected
    s._now.assert_called_once_with()
    assert pack((s.pending,b))==before
    assert identity(b.rows[h.symbol])==identity(h) and deadline(b.rows[h.symbol])==deadline(h)


def test_retry_lifecycle_wakes_and_expires_with_original_deadline():
    s=scanner();h=hit();b=s.vwap_retries;end=deadline(h)
    assert b.hold_confirmed(h,'broker_position_sync_stale',AT)
    s._now=Mock(return_value=AT)
    assert s.needs_recheck is False and s._now.call_count==1
    before=end-timedelta(microseconds=1);b.wake_confirmed(None,before)
    s._now=Mock(return_value=before)
    assert s.needs_recheck is True and s._now.call_count==1
    rows=b.take({'TEST':minutes()},before)
    assert len(rows)==1 and deadline(rows[0])==end and identity(rows[0])==identity(h)
    b.finish(h,'submission_started',before)
    s._now=Mock(side_effect=AssertionError('terminal empty clock'))
    assert s.needs_recheck is False and not b.rows
    assert not b.hold_confirmed(h,'broker_position_sync_stale',before)


def test_clock_evidence_omission_is_explicit_but_active_observation_is_unchanged():
    rows=[]
    class Recorder:
        def record(self,kind,data,operation_id=None):rows.append(dict(kind=kind,data=data,operation_id=operation_id))
        def fail(self,*args):raise AssertionError(args)
    owner=NS(recorder=Recorder());op=LiveOperation(owner,'diagnostic-native-retry')
    def clock():return op.external('market_clock','now',(),{},lambda:AT)
    s=scanner();s._now=clock
    assert s.needs_recheck is False and rows==[]
    s.vwap_retries.rows['TEST']=hit();s.vwap_retries.dirty.add('TEST')
    assert s.needs_recheck is True
    assert [x['kind'] for x in rows]==['external_request','external_response']
    assert rows[0]['data']==dict(role='market_clock',method='now',args=pack(()),kwargs=pack({}))
    assert rows[1]['data']['value']==pack(AT)
    assert all(x['operation_id']=='diagnostic-native-retry' for x in rows)


def test_readiness_does_not_mutate_market_source_packets():
    s=scanner();s._now=Mock(return_value=AT)
    packets=(QuoteEvent('TEST',Quote('TEST',AT,2.4,2.41,1000,1000)),TradeEvent('TEST',Tick('TEST',AT,2.41,100,Side.BUY)))
    before=pack(packets)
    assert s.needs_recheck is False
    assert pack(packets)==before


@pytest.mark.parametrize('subclass',[False,True])
def test_custom_rows_mapping_retains_original_clock_before_mapping_semantics(subclass):
    calls=[]
    class CustomRows(dict):
        def __bool__(self):
            calls.append('rows_bool')
            return False
    s=scanner();s.vwap_retries.rows=CustomRows() if subclass else NS()
    s._now=lambda:calls.append('clock') or None
    # Original ready(None) short-circuits before inspecting rows. An exotic
    # mapping must not gain a new pre-clock bool or silently become native.
    assert s.needs_recheck is False
    assert calls==['clock']
