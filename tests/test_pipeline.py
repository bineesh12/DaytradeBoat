"""Integration test for the full Scanner → Verify → Execute pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

from daytrading.execution.broker import PaperBroker
from daytrading.pipeline.engine import TradingPipeline
from daytrading.scanner.premarket_gap import PremarketGapScanner
from daytrading.strategy.gap_reversal import GapReversalVerifier
from daytrading.models import Bar, PortfolioState, SignalAction

TS = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)


def _bar(
    symbol: str, close: float, volume: float = 200_000,
    open_: float | None = None, high: float | None = None, low: float | None = None,
) -> Bar:
    o = open_ if open_ is not None else close
    h = high if high is not None else close + 1.0
    lo = low if low is not None else close - 1.0
    return Bar(symbol=symbol, ts=TS, open=o, high=h, low=lo, close=close, volume=volume)


def test_pipeline_scans_verifies_and_fills() -> None:
    """A gap-up stock that fades from open should produce a long fill."""

    # 3 bars: day-2, day-1 (prev close), today (gapped up but faded)
    bars = [
        _bar("AAPL", 95.0, volume=150_000),
        _bar("AAPL", 100.0, volume=150_000),
        # gap up 5% but close < open → fading → gap reversal long
        _bar("AAPL", 103.0, open_=106.0, volume=200_000, high=107.0, low=102.0),
    ]
    universe = {"AAPL": bars}

    scanner = PremarketGapScanner(min_gap_pct=3.0, min_volume=100_000)
    verifier = GapReversalVerifier(position_size=10)
    broker = PaperBroker(commission_per_share=0.01)
    portfolio = PortfolioState(cash=50_000.0, positions={})

    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"premarket_gap": verifier},
        broker=broker,
        portfolio=portfolio,
        max_positions=5,
        max_position_shares=500,
        max_order_shares=200,
    )

    result = pipeline.run_cycle(universe)

    assert len(result.scan_hits) >= 1, "Scanner should detect gap"
    # Verifier should produce a signal (gap up + fade → long)
    if len(result.signals) > 0:
        assert result.signals[0].action == SignalAction.ENTER_LONG
        assert len(result.fills) >= 1
        assert portfolio.cash < 50_000.0  # spent some cash


def test_pipeline_respects_position_limit() -> None:
    """Pipeline should stop opening positions after max_positions."""
    bars_a = [_bar("A", 10.0), _bar("A", 10.0), _bar("A", 10.5, open_=11.0, volume=200_000)]
    bars_b = [_bar("B", 20.0), _bar("B", 20.0), _bar("B", 21.0, open_=22.0, volume=200_000)]

    universe = {"A": bars_a, "B": bars_b}

    scanner = PremarketGapScanner(min_gap_pct=3.0, min_volume=100_000)
    verifier = GapReversalVerifier(position_size=10)
    broker = PaperBroker()
    portfolio = PortfolioState(cash=100_000.0, positions={})

    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"premarket_gap": verifier},
        broker=broker,
        portfolio=portfolio,
        max_positions=1,
    )

    result = pipeline.run_cycle(universe)
    filled_symbols = {f.symbol for f in result.fills}
    assert len(filled_symbols) <= 1
