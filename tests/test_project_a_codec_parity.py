from daytrading.models import BarSource
"""Protocol-1 output parity, not a protocol-2 recorder or replay feature."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import random

import pytest
from daytrading.models import Bar, Quote, Tick, CompletedBar, Side
from daytrading.replay.codec import pack, unpack, StateError


def baseline():
    path=Path(__file__).parent/'fixtures/deployed_protocol1_codec.py'
    spec=importlib.util.spec_from_file_location('frozen_deployed_codec',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_native_value_encoding_preserves_complete_graph_and_redaction():
    old=baseline();at=datetime(2026,10,23,8,tzinfo=timezone.utc)
    b=Bar('TEST',at,2.,2.2,1.9,2.1,1000, provenance=BarSource.CANONICAL_INTERVAL)
    q=Quote('TEST',at,2.,2.01,100,100);t=Tick('TEST',at,2.01,100,Side.BUY)
    graph=dict(bar=b,again=b,quote=q,tick=t,completed=CompletedBar(b,at),special=[float('nan'),float('inf')])
    graph['self']=graph;object.__setattr__(b,'extra',graph);object.__setattr__(b,'api_key','dummy-must-not-leak')
    actual=pack(graph);expected=old.pack(graph)
    assert json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True)
    assert 'dummy-must-not-leak' not in json.dumps(actual)
    restored=unpack(actual)
    assert restored['bar'] is restored['again'] and restored['self'] is restored
    assert restored['bar'].extra is restored
    # No per-object cache can hide subsequent fields/mutations.
    object.__setattr__(b,'correction',1)
    assert pack(graph)==old.pack(graph)


def test_native_subclasses_keep_existing_unsupported_type_behavior():
    class LocalBar(Bar):
        pass
    b=LocalBar('TEST',datetime.now(timezone.utc),2,2,2,2,100)
    with pytest.raises(StateError):pack(b)
    old=baseline()
    with pytest.raises(old.StateError):old.pack(b)


def test_deterministic_varied_native_graphs_match_deployed_codec_byte_for_byte():
    old=baseline();rng=random.Random(324);at=datetime(2026,10,23,8,tzinfo=timezone.utc)
    for i in range(100):
        b=Bar('S'+str(i),at,2.,2.5,1.5,2.+rng.random()/10,rng.randrange(1,100000), provenance=BarSource.CANONICAL_INTERVAL)
        q=Quote(b.symbol,at,2.,2.01,100,100)
        data=[b,q,CompletedBar(b,at),dict(alias=b,quote=q,metadata=[at,b.symbol,False,None])]
        assert json.dumps(pack(data),separators=(',',':'))==json.dumps(old.pack(data),separators=(',',':'))
