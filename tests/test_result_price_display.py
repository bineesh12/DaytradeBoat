from daytrading.models import BarSource
"""Exercise result processing through cycle completion, not just the hub API."""
from collections import defaultdict, deque
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from daytrading.dashboard.hub import DashboardHub
from daytrading.models import Bar, PortfolioState, TradingStyle
from daytrading.pipeline.engine import PipelineResult
from daytrading.runner import AlpacaRunner


def test_classification_price_update_does_not_abort_result_processing():
    bar = Bar(symbol='TEST', open=4.4, high=4.5, low=4.3, close=4.45,
              volume=10000, ts=datetime.now(timezone.utc), provenance=BarSource.CANONICAL_INTERVAL)
    regime = SimpleNamespace(style=TradingStyle.SCALPING, confidence=60,
                             volatility_pct=.5, spread_pct=.2, relative_volume=2,
                             trend_strength=.7, liquidity_score=.8, reasons=[])
    result = PipelineResult(regimes={'TEST':regime})
    runner = MagicMock()
    runner._hub = DashboardHub()
    runner._journal = MagicMock()
    runner._pipeline.portfolio = PortfolioState(cash=25000)
    runner._pipeline.scan_rejections = {}
    runner._bar_buffer = {'TEST':deque([bar])}
    runner._skip_counts = defaultdict(int)
    runner._CLEANUP_EVERY = 10
    runner._market_phase.return_value = 'OPEN'
    runner._live_prices.return_value = {}
    runner._broker.get_positions.return_value = {}
    runner._pipeline.missed_a_plus_report.return_value = {}
    runner._pipeline.scanner_near_miss_summary.return_value = {}

    AlpacaRunner._process_result(runner, result, 2, {'TEST':[bar]}, hod_bar_count=1)

    snapshot = runner._hub.snapshot()
    assert snapshot['symbols']['TEST']['price'] == 4.45
    assert snapshot['stats']['cycle_count'] == 2
    assert any(call.args[0]=='cycle' for call in runner._journal.record.call_args_list)
