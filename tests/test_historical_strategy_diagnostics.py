from daytrading.models import BarSource
from datetime import datetime, timedelta, timezone
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from daytrading.models import Bar, Tick, Side, Timeframe, PortfolioState, ScanResult
from daytrading.backtest.driver import PipelineBacktestDriver
from daytrading.backtest.diagnostics import micro_diagnostics, strategy_breakdown

AT = datetime(2026, 10, 13, 8, tzinfo=timezone.utc)


def minute(i, price=3):
    return Bar('MEDS', AT+timedelta(minutes=i), price, price+.05, price-.05, price, 10000, provenance=BarSource.CANONICAL_INTERVAL)


def test_session_quality_keeps_original_open_after_120_bars_and_excludes_other_days():
    rows = [minute(i) for i in range(160)]
    rows[0] = replace(rows[0], open=2.9)
    source = [minute(-24*60), *rows, replace(minute(1), symbol='OTHER')]
    driver = PipelineBacktestDriver({'MEDS':source}, max_bars_per_symbol=120)
    actual = driver._session_universe(rows[140].ts)['MEDS']
    assert actual == rows[:141] and actual[0].open == 2.9
    assert driver._session_universe(rows[119].ts)['MEDS'] == rows[:120]


def test_intraminute_context_has_exact_visible_partial_not_future_final_candle():
    rows = [minute(i) for i in range(150)]
    partial = replace(rows[140], close=3.01, high=3.02, volume=500)
    driver = PipelineBacktestDriver({'MEDS':rows})
    ctx = driver._session_universe(partial.ts+timedelta(seconds=20),
                                  visible_universe={'MEDS':rows[21:140]+[partial]})
    assert len(ctx['MEDS']) == 141 and ctx['MEDS'][-1] is partial
    assert rows[140] not in ctx['MEDS'] and rows[141] not in ctx['MEDS']


def test_pipeline_routes_full_session_context_without_rewriting_pattern_window():
    from daytrading.pipeline.engine import TradingPipeline
    rows = [minute(i) for i in range(145)]
    hit = ScanResult('MEDS','vwap_pullback',rows[-1].ts,10,
                     criteria={'stop_price':2.9},bars=rows[-120:])
    scanner = Mock();scanner.name='vwap_pullback';scanner.scan.return_value=[hit]
    seen = {}
    class Verifier:
        _last_reject='test measured rejection'
        def verify(self, h, p, *, now=None, session_bars=None):
            seen['context'] = session_bars
            return None
    broker=Mock()
    pipeline=TradingPipeline(scanners=[scanner],verifiers={'vwap_pullback':Verifier()},
        portfolio=PortfolioState(cash=25000),broker=broker)
    pipeline.run_cycle({'MEDS':rows[-120:]},now=rows[-1].ts,session_universe={'MEDS':rows})
    assert seen['context'] is rows
    assert hit.bars == rows[-120:] and hit.criteria['stop_price'] == 2.9
    broker.submit.assert_not_called()


def test_micro_missing_raw_trades_never_uses_fixed_grid_candle_cache(tmp_path):
    (tmp_path/'MEDS_2026-10-13_10s_trades.json').write_text(json.dumps({'bars':[1,2,3],'complete':True}))
    result=micro_diagnostics('MEDS','2026-10-13',[],cache_dir=tmp_path)
    assert result['status']=='missing_trade_recording' and result['orders_simulated']==0
    assert not result['validated']


@pytest.mark.parametrize('change',[{'complete':False},{'feed':'iex'},{'date':'2026-10-12'}, {'schema':'wrong'}])
def test_incomplete_or_wrong_raw_cache_blocks_micro_lane(tmp_path,change):
    data=dict(schema='exchange-trades-v1',symbol='MEDS',date='2026-10-13',feed='sip',complete=True,trades=[])
    data.update(change)
    (tmp_path/'MEDS_2026-10-13_event_trades.json').write_text(json.dumps(data))
    assert micro_diagnostics('MEDS','2026-10-13',[],cache_dir=tmp_path)['status']=='invalid_trade_recording'


def test_micro_uses_native_event_anchored_completion_and_causal_minutes(monkeypatch):
    from daytrading.strategy.scalping.micro_pullback_runtime import MicroPullbackScanner
    original=MicroPullbackScanner.scan
    observed=[]
    def scan(self, universe, *, now=None):
        assert all(b.ts+timedelta(minutes=1)<=now for b in universe['MEDS'])
        observed.append(now)
        return original(self,universe,now=now)
    monkeypatch.setattr(MicroPullbackScanner,'scan',scan)
    start=AT+timedelta(seconds=.2)
    ticks=[Tick('MEDS',start+timedelta(seconds=10*i),3+i*.01,100,Side.BUY) for i in range(9)]
    result=micro_diagnostics('MEDS','2026-10-13',[minute(i) for i in range(5)],ticks=ticks)
    assert result['status']=='detector_only' and result['windows']==8
    assert result['events'][0]['trigger']==start.isoformat()
    assert observed[0]==start+timedelta(seconds=10)
    assert result['orders_simulated']==0 and result['signal_status']=='not_verified'
    assert not result['validated']
    prefix=micro_diagnostics('MEDS','2026-10-13',[minute(i) for i in range(5)],ticks=ticks[:7])
    assert result['events'][:6]==prefix['events']


@pytest.mark.parametrize('bad',[float('nan'),-1,0])
def test_invalid_ticks_fail_closed(bad):
    tick=Tick('MEDS',AT,bad,100,Side.BUY)
    assert micro_diagnostics('MEDS','2026-10-13',[],ticks=[tick])['status']=='invalid_trade_recording'


def test_strategy_breakdown_preserves_measured_layer_and_reason():
    scans=[{'scanner':'vwap_pullback','ts':AT.isoformat()}]*2
    rejects=[dict(scanner='vwap_pullback',passed=False,blocked_layer='verifier',reason='52,396 < 75,000')]*2
    row=strategy_breakdown(scans,rejects)[0]
    assert row['hits']==2 and row['blocked_by_stage']=={'verifier':2}
    assert row['top_reasons']==[{'reason':'52,396 < 75,000','count':2}]


def test_historical_worker_isolates_writes_and_receives_no_fill_outcomes(tmp_path,monkeypatch):
    import daytrading.backtest.historical_worker as worker
    monkeypatch.setenv('DAYTRADING_DATA_DIR',str(tmp_path/'live'))
    monkeypatch.setenv('DAYTRADING_BACKTEST_CACHE_DIR',str(tmp_path/'cache'))
    def run(args,**kw):
        assert kw['env']['DAYTRADING_DATA_DIR'] != str(tmp_path/'live')
        assert kw['env']['DAYTRADING_BACKTEST_CACHE_DIR'] == str(tmp_path/'cache')
        assert set(json.loads(kw['input'])) == {'symbol','session_date','flags','start_time'}
        assert args[-1]=='daytrading.backtest.historical_worker'
        return SimpleNamespace(returncode=0,stdout='{"ok":true}')
    monkeypatch.setattr(worker.subprocess,'run',run)
    assert worker.run_isolated_historical('MEDS','2026-10-13') == {'ok':True}


@pytest.mark.parametrize('flags,expected',[
    ({'execution_timer_10s':True}, ('vwap_pullback','hod_reclaim')),
    ({'momentum_burst_live':True}, None),
])
def test_historical_baseline_uses_paper_patterns_and_labels_explicit_experiments(tmp_path,monkeypatch,flags,expected):
    import daytrading.backtest.service as service
    monkeypatch.setenv('DAYTRADING_ENTRY_PATTERNS','vwap_pullback,hod_reclaim,micro_pullback_reclaim')
    monkeypatch.setenv('DAYTRADING_BACKTEST_CACHE_DIR',str(tmp_path))
    factory=service.create_scalping_pipeline;seen={}
    def build(**kwargs):
        seen['patterns']=kwargs['entry_patterns']
        return factory(**kwargs)
    monkeypatch.setattr(service,'create_scalping_pipeline',build)
    r=service.run_backtest('MEDS','2026-10-13',flags=flags,bars_by_symbol={'MEDS':[minute(i) for i in range(3)]})
    assert seen['patterns']==expected
    assert r['micro_diagnostics']['status']=='missing_trade_recording'
    assert r['research_coverage']['minute_entry_patterns']==expected
    assert r['research_coverage']['entry_policy']==('explicit_research_experiment' if expected is None else 'configured_paper_patterns')
    assert r['fills']==0 and not r['validated']
