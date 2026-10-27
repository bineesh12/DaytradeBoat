"""Approved fresh qualification renews eligibility, not notifications."""
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import pytest
from daytrading.models import Bar,BarSource
from daytrading.scanner.hod_momentum.alert_store import HODAlertStore
from daytrading.scanner.hod_momentum.bar_scanner import HODMomentumScanner
from daytrading.runner import AlpacaRunner
AT=datetime(2026,10,22,12,0,tzinfo=timezone.utc)
def bars(now,source=BarSource.CANONICAL_INTERVAL):
 return [Bar('PROBE',now-timedelta(minutes=20-i),3+i*.03,3.04+i*.03,2.98+i*.03,3.03+i*.03,100000,provenance=source) for i in range(20)]
def scanner(cb):
 return HODMomentumScanner(HODAlertStore(ttl_minutes=15),float_checker=NS(get_float_cached=lambda s:6000000),on_qualification=cb)

def test_fresh_requalification_renews_after_ttl_without_repeating_notifications():
 rows=[];s=scanner(rows.append)
 first=s.scan({'PROBE':bars(AT)},rel_vols={'PROBE':5},now=AT)
 count=len(rows); assert count and first['_new_alerts']>0
 later=AT+timedelta(minutes=16)
 second=s.scan({'PROBE':bars(later)},rel_vols={'PROBE':5},now=later)
 assert len(rows)==2*count and second['_new_alerts']==0
 assert all(r['qualified_at']==later.isoformat() and r['notification_suppressed'] for r in rows[count:])

@pytest.mark.parametrize('offset',[0,.1,30,59,61,900,960])
def test_same_candle_or_timer_never_renews_qualification(offset):
 rows=[];s=scanner(rows.append);old=bars(AT)
 s.scan({'PROBE':old},rel_vols={'PROBE':5},now=AT);count=len(rows)
 s.scan({'PROBE':old},rel_vols={'PROBE':5},now=AT+timedelta(seconds=offset))
 assert len(rows)==count

@pytest.mark.parametrize('kind',['synthetic','unknown','partial','low_volume','missing_float','weak_change'])
def test_failed_or_uncertain_qualification_does_not_renew(kind):
 rows=[];s=scanner(rows.append);s.scan({'PROBE':bars(AT)},rel_vols={'PROBE':5},now=AT);count=len(rows)
 later=AT+timedelta(minutes=16);new=bars(later)
 if kind in ('synthetic','unknown'):
  from dataclasses import replace
  new=[replace(b,provenance=BarSource.ROLLING_SNAPSHOT if kind=='synthetic' else BarSource.UNKNOWN) for b in new]
 if kind=='partial':new=bars(later+timedelta(minutes=20))
 if kind=='low_volume':new=[Bar(b.symbol,b.ts,b.open,b.high,b.low,b.close,1,provenance=b.provenance) for b in new]
 if kind=='missing_float':s._float_checker=NS(get_float_cached=lambda sym:None)
 if kind=='weak_change':new=[Bar(b.symbol,b.ts,3,3.01,2.99,3,100000,provenance=b.provenance) for b in new]
 s.scan({'PROBE':new},rel_vols={'PROBE':0 if kind=='weak_change' else 5},now=later)
 assert len(rows)==count

def test_runner_qualification_updates_only_mandatory_approved_label():
 o=AlpacaRunner.__new__(AlpacaRunner);o._hod_active={};o._hod_entry_gate_alerts={'HOD Breakout'};o._journal=NS(record=Mock());o._hod_tradeable_alert_at={};o._is_tradeable_hod_watchlist_alert=Mock(return_value=True)
 row=dict(symbol='PROBE',alert_name='Other',qualified_at=AT.isoformat())
 o._on_hod_qualification(row);assert not o._hod_active
 row['alert_name']='HOD Breakout';o._on_hod_qualification(row)
 assert o._hod_active['PROBE']==AT and o._hod_tradeable_alert_at['PROBE']==AT
 o._hod_alert_ttl_minutes=15
 class Clock(datetime):
  value=AT
  @classmethod
  def now(cls,tz=None):return cls.value
 with patch('daytrading.runner.datetime',Clock):
  assert o.is_hod_active('PROBE');Clock.value+=timedelta(minutes=15)
  assert not o.is_hod_active('PROBE')
 assert o._journal.record.call_count==1

def test_watermarks_survive_guard_checkpoint_and_no_replay_renewal():
 from test_runtime_state import runner,NOW
 from daytrading.runtime_state import capture_state,restore_state
 source=runner();source._hod_active={'PROBE':NOW};source._hod_bar_scanner=scanner(Mock())
 source._hod_bar_scanner._qualification_watermarks={'PROBE|HOD Breakout':NOW-timedelta(minutes=1)}
 source._hod_bar_scanner._bar_cooldown_keys={'PROBE|HOD Breakout'}
 state=capture_state(source,NOW)
 target=runner();target._hod_bar_scanner=scanner(Mock())
 restore_state(target,state,NOW+timedelta(seconds=1))
 assert target._hod_active==source._hod_active
 assert target._hod_bar_scanner._qualification_watermarks==source._hod_bar_scanner._qualification_watermarks
 assert target._hod_bar_scanner._bar_cooldown_keys==source._hod_bar_scanner._bar_cooldown_keys
