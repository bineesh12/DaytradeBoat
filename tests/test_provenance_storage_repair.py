from dataclasses import replace
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace as NS
import pytest
from daytrading.models import Bar,BarSource,Timeframe
from daytrading.journal.store import TradingJournal
from daytrading.backtest.data_loader import _bar_to_dict,_bar_from_dict
from daytrading.backtest.replay import _bars_from_payload

@pytest.mark.parametrize('source',list(BarSource))
def test_journal_and_research_roundtrip_preserves_source(source):
    b=Bar('X',datetime(2026,10,22,12,tzinfo=timezone.utc),2,2.2,1.9,2.1,1000,provenance=source)
    j=TradingJournal.__new__(TradingJournal)
    row=j.candle_snapshot([b])[0]
    assert row['provenance']==source.value
    assert _bar_from_dict(_bar_to_dict(b))==b
    assert _bars_from_payload({'candle_snapshot':[row]},'X')==[b]
    row.pop('provenance')
    assert _bars_from_payload({'candle_snapshot':[row]},'X')[0].provenance is BarSource.UNKNOWN
