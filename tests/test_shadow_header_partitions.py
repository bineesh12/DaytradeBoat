"""Exact-output and bounded immutable-header reuse; no broker or live files."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from hashlib import sha256
import importlib.util,json,os
from pathlib import Path
from unittest.mock import patch

import pytest
from daytrading.ml import shadow_collector as candidate
from daytrading.models import Bar
from daytrading.replay.codec import pack

ORIGINAL=Path(__file__).resolve().parent/'reference/shadow_collector.py'
spec=importlib.util.spec_from_file_location('reviewed_phase2_original_shadow',ORIGINAL)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
NOW=datetime(2026,10,27,14,33,19,tzinfo=timezone.utc)

@pytest.fixture(autouse=True)
def clear_headers(monkeypatch,tmp_path):
    monkeypatch.setattr(original,'DATA_DIR',tmp_path/'reference-ml')
    candidate._row_headers.clear();original._row_headers.clear()
    yield
    candidate._row_headers.clear();original._row_headers.clear()

@pytest.fixture
def active():
    return original if os.getenv('DAYTRADING_HEADER_PROBE_ORIGINAL')=='1' else candidate

def final_row(number):
    return json.dumps({'label':1,'symbol':'BKYI','ts':'2026-10-27T14:00:00+00:00','unique':number})

def label(mod,path,prices=None):
    return mod._label_pending_file(path,{} if prices is None else prices,wait_seconds=180.,positive_pct=2.,fail_pct=1.5)

def test_two_files_33882_headers_do_not_thrash(tmp_path,active):
    missed=tmp_path/'missed.jsonl';pullback=tmp_path/'pullback.jsonl'
    missed.write_text('\n'.join(final_row(i) for i in range(22983))+'\n')
    pullback.write_text('\n'.join(final_row(i) for i in range(22983,33882))+'\n')
    loads=json.loads
    with patch.object(active.json,'loads',wraps=loads) as calls:
        assert label(active,missed)==label(active,pullback)==0
        assert calls.call_count==33882
        calls.reset_mock()
        assert label(active,missed)==label(active,pullback)==0
        assert calls.call_count==0
    assert len(active._row_headers)<=2
    assert sorted(len(rows) for rows in active._row_headers.values())==[10899,22983]

def test_partition_count_and_per_file_header_capacity_are_bounded(tmp_path):
    for index in range(3):
        path=tmp_path/(str(index)+'.jsonl')
        path.write_text(final_row(index)+'\n');assert label(candidate,path)==0
    assert list(candidate._row_headers)==[tmp_path/'1.jsonl',tmp_path/'2.jsonl']
    large=tmp_path/'large.jsonl';large.write_text('\n'.join(final_row(i) for i in range(32769))+'\n')
    before=large.read_bytes();assert label(candidate,large)==label(candidate,large)==0
    assert len(candidate._row_headers)==2
    assert len(candidate._row_headers[large])==32768
    assert large.read_bytes()==before

def test_evicted_partition_is_reparsed_without_output_change(tmp_path):
    paths=[tmp_path/(str(i)+'.jsonl') for i in range(3)]
    for i,p in enumerate(paths):p.write_text(final_row(i)+'\n');label(candidate,p)
    loads=json.loads
    with patch.object(candidate.json,'loads',wraps=loads) as calls:
        label(candidate,paths[0]);assert calls.call_count==1
    assert len(candidate._row_headers)==2

def test_append_correction_and_same_length_edit_use_current_bytes(tmp_path):
    p=tmp_path/'file.jsonl';p.write_text(final_row(1)+'\n');label(candidate,p)
    p.write_text(final_row(1)+'\n'+final_row(2)+'\n')
    loads=json.loads
    with patch.object(candidate.json,'loads',wraps=loads) as calls:
        label(candidate,p);assert calls.call_count==1
        p.write_text(final_row(1).replace('"label": 1','"label": 0')+'\n'+final_row(2)+'\n')
        calls.reset_mock();label(candidate,p);assert calls.call_count==1
    changed=final_row(1).replace('"label": 1','"label": 0').strip()
    assert sha256(changed.encode()).digest() in candidate._row_headers[p]

def test_current_price_and_time_are_not_cached(tmp_path,monkeypatch):
    p=tmp_path/'file.jsonl'
    pending={'label':None,'symbol':'BKYI','price':10.,'ts':(NOW-timedelta(seconds=60)).isoformat()}
    p.write_text(json.dumps(pending)+'\n');monkeypatch.setattr(candidate,'_utc_now',lambda:NOW)
    assert label(candidate,p,{'BKYI':10.5})==0
    monkeypatch.setattr(candidate,'_utc_now',lambda:NOW+timedelta(seconds=200))
    assert label(candidate,p,{'BKYI':9.5})==1
    row=json.loads(p.read_text());assert row['future_return_pct']==-5. and row['label']==0 and row['failed_first'] is True
    assert row['labeled_at']==(NOW+timedelta(seconds=200)).isoformat()

def test_changed_symbol_timestamp_and_finalization_get_distinct_keys(tmp_path):
    cache=candidate._shadow_header_cache(tmp_path/'file')
    lines=[json.dumps(v) for v in [dict(label=None,symbol='BKYI',ts='a'),dict(label=None,symbol='STAK',ts='a'),dict(label=None,symbol='STAK',ts='b'),dict(label=0,symbol='STAK',ts='b')]]
    assert [candidate._shadow_row_header(line,cache) for line in lines]==[(False,'BKYI','a'),(False,'STAK','a'),(False,'STAK','b'),(True,'',None)]
    assert len(cache)==4

@pytest.mark.parametrize('line',[json.dumps({'label':None,'symbol':'x'*33,'ts':'a'}),json.dumps({'label':None,'symbol':'BKYI','ts':'x'*65}),json.dumps({'label':None,'symbol':'BKYI','ts':None}),'not json'])
def test_malformed_or_unbounded_headers_preserve_original_and_remain_uncached(line):
    loads=json.loads
    with patch.object(candidate.json,'loads',wraps=loads) as calls:
        a=candidate._shadow_row_header(line);b=candidate._shadow_row_header(line)
        assert calls.call_count==2
    assert a==b==original._shadow_row_header(line)
    assert all(not rows for rows in candidate._row_headers.values())

@pytest.mark.parametrize('line',['[]','null','1','"text"'])
def test_original_header_exception_type_and_message_preserved(line):
    errors=[]
    for mod in (original,candidate):
        with pytest.raises(Exception) as exc:mod._shadow_row_header(line)
        errors.append((type(exc.value),str(exc.value)))
    assert errors[0]==errors[1]

def test_direct_header_call_and_legacy_clear_clear_all_partitions(tmp_path):
    assert candidate._shadow_row_header(final_row(1))==(True,'',None)
    candidate._shadow_header_cache(tmp_path/'file')
    assert len(candidate._row_headers)==2
    candidate._row_headers.clear();assert len(candidate._row_headers)==0

class UnknownPathAdapter:
    __hash__=None
    def __str__(self):raise AssertionError('No new adapter string conversion')

def test_unknown_path_adapter_does_not_add_new_hash_or_string_behavior():
    assert candidate._shadow_header_cache(UnknownPathAdapter()) is candidate._shadow_header_cache()

@pytest.mark.parametrize('row',[{'label':1,'symbol':'BKYI','ts':'bad','price':10.}, {'label':None,'symbol':'BKYI','ts':(NOW-timedelta(seconds=300)).isoformat(),'price':10.}, {'label':None,'symbol':'BKYI','ts':(NOW-timedelta(seconds=20)).isoformat(),'price':10.}, {'label':None,'symbol':'BKYI','ts':'malformed','price':10.}, {'label':None,'symbol':'OTHER','ts':(NOW-timedelta(seconds=300)).isoformat(),'price':10.}])
def test_whole_file_bytes_exact_against_original_fixed_clock(tmp_path,monkeypatch,row):
    before='\n'+json.dumps(row,separators=(',',':'))+'\nnot json\n'+final_row(99)+'\n'
    paths=[tmp_path/'old.jsonl',tmp_path/'new.jsonl']
    prices={'BKYI':10.4};raw_prices=pack(prices)
    results=[]
    for mod,path in zip((original,candidate),paths):
        monkeypatch.setattr(mod,'_utc_now',lambda:NOW)
        path.write_text(before);results.append(label(mod,path,prices))
    assert results[0]==results[1] and paths[0].read_bytes()==paths[1].read_bytes()
    assert pack(prices)==raw_prices
    assert pack((results[0],paths[0].read_text()))==pack((results[1],paths[1].read_text()))

def test_actual_update_pair_preserves_native_bars_and_protocol1_encoded_output(tmp_path,monkeypatch):
    bars={'BKYI':[Bar(symbol='BKYI',ts=NOW,open=10.,high=10.4,low=10.,close=10.4,volume=1000)]}
    before=pack(bars);results=[];contents=[]
    row=json.dumps({'label':None,'symbol':'BKYI','ts':(NOW-timedelta(seconds=300)).isoformat(),'price':10.})+'\n'
    for index,mod in enumerate((original,candidate)):
        folder=tmp_path/str(index);folder.mkdir();missed=folder/'missed.jsonl';pullback=folder/'pullback.jsonl'
        missed.write_text(row);pullback.write_text(row)
        monkeypatch.setattr(mod,'MISSED_FILE',missed);monkeypatch.setattr(mod,'PULLBACK_FILE',pullback);monkeypatch.setattr(mod,'_utc_now',lambda:NOW)
        results.append(mod.update_deferred_outcomes(bars));contents.append((missed.read_bytes(),pullback.read_bytes()))
        assert pack(bars)==before
    assert results[0]==results[1] and contents[0]==contents[1]
    assert pack((bars,results[0],tuple(x.decode('utf-8') for x in contents[0])))==pack((bars,results[1],tuple(x.decode('utf-8') for x in contents[1])))
