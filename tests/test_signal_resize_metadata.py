from daytrading.models import BarSource
"""Risk sizing must not reset the signal used by exit management."""
from dataclasses import fields
from datetime import datetime,timezone
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from daytrading.models import Bar,ScanResult,TradeSignal,SignalAction,PortfolioState
from daytrading.pipeline.engine import TradingPipeline
from daytrading.execution.broker import PaperBroker
from daytrading.strategy.execution_timer import ExecutionTimer
AT=datetime(2026,10,16,12,tzinfo=timezone.utc)
@pytest.mark.parametrize('qty,risk,cap,expected',[(548,.0912,500,500),(900,.125,1000,400),(900,.125,200,200)])
def test_caps_preserve_every_other_signal_field(monkeypatch,qty,risk,cap,expected):
    bar=Bar('TEST',AT,2.55,2.65,2.5,2.6,500000, provenance=BarSource.CANONICAL_INTERVAL)
    hit=ScanResult('TEST','vwap_pullback',AT,90,{'pattern':'vwap_pullback'},[bar])
    original=TradeSignal('TEST',SignalAction.ENTER_LONG,qty,2.6,2.6-risk,2.6+2*risk,.07,321,'metadata must survive',hit,.91)
    scanner=SimpleNamespace(name='vwap_pullback',scan=lambda universe:[hit]);verifier=SimpleNamespace(verify=lambda *args,**kwargs:original)
    p=TradingPipeline(scanners=[scanner],verifiers={'vwap_pullback':verifier},broker=PaperBroker(),portfolio=PortfolioState(cash=25000),max_order_shares=cap,max_position_shares=1000)
    p._execution_timer=ExecutionTimer(enabled=True);p._max_dollar_risk_per_trade=50
    p._final_entry_quality_reject=Mock(return_value=None);p._entry_policy._guard=Mock(return_value=None);p.trade_guard.check_entry=Mock(return_value=(True,None))
    p._log_shadow_pullback=Mock();p._log_shadow_missed=Mock()
    r=p.run_cycle({'TEST':[bar]},now=AT)
    assert len(r.deferred_signals)==1,r.entry_decisions
    resized=r.deferred_signals[0]
    assert resized.quantity==expected and resized.quantity*risk<=50+1e-7
    for f in fields(TradeSignal):
        if f.name!='quantity':assert getattr(resized,f.name)==getattr(original,f.name),f.name
    assert resized.scan_result is hit
