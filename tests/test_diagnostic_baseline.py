from daytrading.models import BarSource
"""Failure injection, setup identity and data provenance for paper diagnostics."""
from collections import defaultdict, deque
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
import pytest
from daytrading.dashboard.hub import DashboardHub
from daytrading.dashboard.server import create_app
from daytrading.data.float_checker import FloatChecker
from daytrading.data.float_store import FloatStore
from daytrading.data.watchlist_scanner import WatchlistScanner
from daytrading.execution.broker import apply_fill
from daytrading.exits.manager import ExitManager
from daytrading.models import Bar, Fill, PortfolioState, ScanResult, Side, SignalAction, TradeSignal, TradingStyle
from daytrading.opportunities import hit_identity
from daytrading.pipeline.engine import PipelineResult
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.reporting import report_safely, reporting_failures
from daytrading.runner import AlpacaRunner
from daytrading.strategy.entry_policy import EntryPolicy
PATTERNS = ('vwap_pullback', 'first_pullback_reclaim', 'hod_reclaim')
NOW = datetime(2026, 10, 14, 15, tzinfo=timezone.utc)
def bar(ts=NOW):
    return Bar(symbol='TEST', ts=ts, open=1.4, high=1.41, low=1.39, close=1.4, volume=100000, provenance=BarSource.CANONICAL_INTERVAL)
def signal(pattern='vwap_pullback', ts=NOW):
    return TradeSignal(symbol='TEST', action=SignalAction.ENTER_LONG, quantity=100, entry_price=1.4, stop_loss=1.3,
        scan_result=ScanResult(symbol='TEST', scanner_name=pattern, ts=ts, score=90, criteria={}, bars=[bar()]))
def test_baseline_restricts_auxiliary_buy_paths_and_shares_bounds():
    pipeline = create_scalping_pipeline(min_price=1, max_price=20, entry_patterns=PATTERNS)
    assert set(pipeline._verifiers) == set(PATTERNS)
    assert len(pipeline._scanners) == 14
    assert pipeline._scaler is None and pipeline._reentry is None
    guard = Mock(return_value=None); pipeline._entry_policy._guard = guard
    assert pipeline._entry_policy.evaluate(signal(), bars=[bar()], stage='timed_entry_final_guard').passed
    assert guard.call_args.kwargs['min_price'] == 1 and guard.call_args.kwargs['max_price'] == 20
    assert not pipeline._entry_policy.evaluate(signal('breakout_scalp'), bars=[bar()], stage='timed_entry_final_guard').passed
    assert guard.call_count == 1
    with pytest.raises(ValueError, match='Unknown entry patterns'):
        create_scalping_pipeline(entry_patterns=('typo',))
def test_identity_uses_candle_not_refresh_clock():
    first = signal().scan_result; refresh = signal(ts=NOW+timedelta(seconds=10)).scan_result
    assert hit_identity(first) == hit_identity(refresh)
    refresh = replace(refresh, bars=[bar(NOW+timedelta(minutes=1))])
    assert hit_identity(first) != hit_identity(refresh)
def test_funnel_dedupes_refreshes_and_partial_exit_remains_open():
    hub = DashboardHub(); sig = signal()
    for _ in range(3):
        hub.on_scan_hit(sig.scan_result)
        hub.on_entry_decision(EntryPolicy().decision(symbol='TEST', stage='deferred_to_timer', passed=True, signal=sig).to_payload())
    hub.on_opportunity_fill(sig, Fill(symbol='TEST', side=Side.BUY, quantity=40, price=1.4, ts=NOW))
    hub.on_opportunity_exit('TEST', False)
    assert hub.funnel_snapshot()['counts'] == dict(pattern=1, qualified=1, submitted=1, filled=1, closed=0)
    hub.on_opportunity_exit('TEST', True)
    assert hub.funnel_snapshot()['counts']['closed'] == 1
    hub.reset_daily_overview(); assert not any(hub.funnel_snapshot()['counts'].values())
def test_rejected_order_does_not_invent_submission():
    hub = DashboardHub()
    hub.on_entry_decision(EntryPolicy().decision(symbol='TEST', signal=signal(), stage='order', passed=False,
                          reason='risk block', metadata={'status':'rejected'}).to_payload())
    assert hub.funnel_snapshot()['counts']['submitted'] == hub.funnel_snapshot()['counts']['filled'] == 0
def test_stop_precedes_reporting_failure_and_cycle_continues():
    sig = signal(); fill = Fill(symbol='TEST', side=Side.BUY, quantity=40, price=1.4, ts=NOW)
    manager = ExitManager(); manager.register_from_signal(sig, NOW, fill_price=1.4, fill_quantity=40)
    portfolio = PortfolioState(cash=25000); apply_fill(portfolio, fill)
    runner = MagicMock(); runner._pipeline.exit_manager = manager; runner._pipeline.portfolio = portfolio
    runner._pipeline.scan_rejections = {}; runner._bar_buffer = {'TEST':deque([bar()])}
    runner._skip_counts = defaultdict(int); runner._CLEANUP_EVERY = 10
    runner._broker.get_positions.return_value = {'TEST':{'qty':40,'current_price':1.4}}
    events = []; runner._arm_broker_protection.side_effect = lambda *args: events.append(('stop',args))
    def fail(*args, **kwargs):
        events.append(('report',args)); raise RuntimeError('reporting unavailable')
    runner._hub.on_classification.side_effect = fail; runner._journal.record.side_effect = fail
    regime = SimpleNamespace(style=TradingStyle.SCALPING, confidence=80, volatility_pct=1, spread_pct=.1,
                              relative_volume=2, trend_strength=.7, liquidity_score=.8, reasons=[])
    AlpacaRunner._process_result(runner, PipelineResult(fills=[fill], regimes={'TEST':regime}), 2, {'TEST':[bar()]})
    assert events[0] == ('stop',('TEST',40,manager.tracked['TEST'].stop_loss))
    runner._hub.on_cycle_complete.assert_called_once(); runner._push_positions_from_alpaca.assert_called_once()
def test_reporting_failure_does_not_interrupt_exit_accounting_or_stop_resize():
    from test_fast_exit_safety import _runner_for_exit_recording
    manager = ExitManager(); manager.register_from_signal(signal(), NOW, fill_quantity=40, fill_price=1.4)
    manager.tracked['TEST'].remaining_qty = 40; runner = _runner_for_exit_recording(manager)
    runner._refresh_broker_stop = Mock(); runner._clear_broker_stop = Mock()
    runner._pipeline.record_realized_exit = Mock(return_value=-2)
    runner._hub.on_exit_fill = Mock(side_effect=RuntimeError('UI failed'))
    runner._journal.record = Mock(side_effect=RuntimeError('disk failed'))
    fill = Fill(symbol='TEST',side=Side.SELL,quantity=20,price=1.3,ts=NOW)
    assert runner._record_trade_exit(fill,1.4,'partial') == -2
    runner._refresh_broker_stop.assert_called_once_with('TEST'); runner._clear_broker_stop.assert_not_called()
    runner._record_trade_exit(fill,1.4,'partial'); assert runner._pipeline.record_realized_exit.call_count == 1
def test_reporting_failure_counter():
    before = reporting_failures(); report_safely(Mock(side_effect=RuntimeError('failed')))
    assert reporting_failures() == before+1
@pytest.mark.parametrize('name',['GraniteShares 2x Short PLTR Daily ETF','Direxion Daily Bear 3X Shares','ProShares UltraShort QQQ'])
def test_stock_policy_uses_asset_metadata(name):
    assert WatchlistScanner.asset_rejection(SimpleNamespace(symbol='FUND',name=name))
    assert WatchlistScanner.asset_rejection(SimpleNamespace(symbol='TEST',name='Test Technologies Inc')) is None
def test_float_memory_preserves_age_and_expires(tmp_path):
    store = FloatStore(str(tmp_path/'float.db')); fetched = datetime.now(timezone.utc)-timedelta(days=6)
    store.upsert('TEST',5000000,fetched_at=fetched); checker = FloatChecker(store=store); checker.warm_from_store(['TEST'])
    meta = checker.get_float_metadata('TEST'); assert meta['source']=='yfinance' and meta['fresh'] and meta['age_hours']>=144
    stale = datetime.now(timezone.utc)-timedelta(days=8)
    checker._provenance['TEST']['fetched_at'] = stale; store.upsert('TEST',5000000,fetched_at=stale)
    assert checker.get_float_cached('TEST') is None and checker.needs_yahoo_refresh('TEST')
    store.close(); checker._executor.shutdown(wait=False)
def test_ui_asset_edit_only_requires_refresh(tmp_path,monkeypatch):
    import daytrading.dashboard.server as server
    package = tmp_path/'dashboard'; assets = package/'assets'; assets.mkdir(parents=True)
    (assets/'index.html').write_text('__DASHBOARD_CSS__ __DASHBOARD_JS__')
    (assets/'dashboard.css').write_text('style'); (assets/'dashboard.js').write_text('first script')
    monkeypatch.setattr(server,'__file__',str(package/'server.py')); client=create_app(DashboardHub()).test_client()
    assert 'first script' in client.get('/').get_data(as_text=True)
    (assets/'dashboard.js').write_text('updated script')
    assert 'updated script' in client.get('/').get_data(as_text=True)

def test_protection_retry_uses_current_broker_quantity():
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._pipeline_protection_pending = {'TEST':(100,1.3)}
    runner._pipeline = SimpleNamespace(exit_manager=SimpleNamespace(tracked={}))
    runner._broker = SimpleNamespace(get_positions=Mock(return_value={'TEST':{'qty':20}}),
                                     get_protective_stop_id=Mock(return_value='stop'), _position_sync_stale=False)
    runner._arm_broker_protection = Mock()
    runner._retry_pipeline_protection()
    runner._arm_broker_protection.assert_called_once_with('TEST',20,1.3)
    assert not runner._pipeline_protection_pending

def test_adjusted_prior_day_reference_excludes_current_day(monkeypatch):
    from daytrading.market_calendar import ET
    from daytrading.scanner.hod_momentum.prior_day import fetch_prior_day_stats
    now = datetime.now(ET)
    previous = SimpleNamespace(timestamp=now-timedelta(days=1),close=10.8,high=11,volume=10000)
    current = SimpleNamespace(timestamp=now,close=12,high=13,volume=10000)
    client = SimpleNamespace(get_stock_bars=Mock(return_value=SimpleNamespace(data={'TEST':[previous,current]})))
    result = fetch_prior_day_stats(client,['TEST'],feed='sip')
    assert result['TEST'].prior_close == 10.8 and result['TEST'].prior_high == 11
    req = client.get_stock_bars.call_args.args[0]
    assert req.adjustment.value == 'split'


def test_delayed_entry_is_blocked_after_trading_window_closes():
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._entry_window_checker = lambda: False
    assert runner._new_entries_blocked('TEST', 'TIMED ENTRY') is True
