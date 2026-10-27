from daytrading.models import BarSource
"""After-cutoff pattern observation cannot enqueue or submit fresh exposure."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from daytrading.market_calendar import ET
from daytrading.models import Bar, Fill, OrderStatus, PortfolioState, ScanResult, Side, SignalAction, TradeSignal
from daytrading.pipeline.engine import TradingPipeline
from daytrading.execution.broker import apply_fill
from daytrading.exits.manager import ExitManager
from daytrading.runner import AlpacaRunner

@pytest.mark.parametrize('hour,minute,entries,scanning', [(15,29,True,True),(15,45,False,True),(16,0,False,False),(16,1,False,False)])
def test_pattern_and_entry_windows_are_independent(hour,minute,entries,scanning):
    runner = AlpacaRunner.__new__(AlpacaRunner); runner._after_hours_enabled = False
    now = datetime(2026,10,14,hour,minute,tzinfo=ET)
    assert runner._is_in_trading_window(now) is entries
    assert runner._is_in_pattern_scan_window(now) is scanning

def test_observation_cycle_detects_and_verifies_without_new_orders():
    now=datetime(2026,10,14,15,45,tzinfo=ET)
    bar=Bar(symbol='TEST',ts=now,open=5,high=5.1,low=4.9,close=5,volume=100000, provenance=BarSource.CANONICAL_INTERVAL)
    hit=ScanResult(symbol='TEST',scanner_name='vwap_pullback',ts=now,score=90,criteria={},bars=[bar])
    signal=TradeSignal(symbol='TEST',action=SignalAction.ENTER_LONG,quantity=10,entry_price=5,stop_loss=4.9,scan_result=hit)
    scanner=SimpleNamespace(name='vwap_pullback',scan=Mock(return_value=[hit]))
    verifier=SimpleNamespace(verify=Mock(return_value=signal))
    broker=SimpleNamespace(submit=Mock())
    scaler=SimpleNamespace(check_scale_ups=Mock())
    reentry=SimpleNamespace(check_reentries=Mock())
    pipeline=TradingPipeline(scanners=[scanner],verifiers={'vwap_pullback':verifier},broker=broker,
                             portfolio=PortfolioState(cash=10000),scaler=scaler,reentry_detector=reentry)
    pipeline._execution_timer=Mock()
    result=pipeline.run_cycle({'TEST':[bar]},now=now,allow_entries=False)
    assert len(result.scan_hits)==len(result.signals)==1
    verifier.verify.assert_called_once()
    assert result.entry_decisions[-1]['stage']=='observation_only'
    assert not result.fills and not result.deferred_signals and not result.scale_up_fills and not result.reentry_fills
    broker.submit.assert_not_called();scaler.check_scale_ups.assert_not_called();reentry.check_reentries.assert_not_called()
    pipeline._execution_timer.queue.assert_not_called()

def test_observation_still_executes_exits():
    now=datetime(2026,10,14,15,45,tzinfo=ET)
    portfolio=PortfolioState(cash=10000)
    apply_fill(portfolio,Fill(symbol='TEST',side=Side.BUY,quantity=10,price=5,ts=now))
    manager=ExitManager()
    manager.register_from_signal(TradeSignal(symbol='TEST',action=SignalAction.ENTER_LONG,quantity=10,entry_price=5,stop_loss=4.9),now,fill_quantity=10,fill_price=5)
    fill=Fill(symbol='TEST',side=Side.SELL,quantity=10,price=4.8,ts=now)
    broker=SimpleNamespace(submit=Mock(return_value=(fill,OrderStatus.FILLED)))
    pipeline=TradingPipeline(scanners=[],verifiers={},broker=broker,portfolio=portfolio,exit_manager=manager)
    bar=Bar(symbol='TEST',ts=now,open=4.8,high=4.8,low=4.8,close=4.8,volume=10000, provenance=BarSource.CANONICAL_INTERVAL)
    result=pipeline.run_cycle({'TEST':[bar]},now=now,allow_entries=False)
    assert result.exit_fills==[fill]
    assert 'TEST' not in portfolio.positions or portfolio.positions['TEST'].is_flat
    assert all(call.args[0].side is Side.SELL for call in broker.submit.call_args_list)
