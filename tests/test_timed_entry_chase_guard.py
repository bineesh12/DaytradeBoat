from __future__ import annotations

from datetime import datetime, timezone

from daytrading.models import Bar, ScanResult, SignalAction, Timeframe, TradeSignal
from daytrading.runner import AlpacaRunner


class _Agg:
    def __init__(self, bars):
        self._bars = bars

    def get_latest_10s(self, symbol: str, count: int = 1):
        return self._bars[-count:]


def _bar(symbol: str = "DXST", close: float = 5.00) -> Bar:
    return Bar(
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000,
        ts=datetime.now(timezone.utc),
    )


def _signal(symbol: str = "DXST", price: float = 5.00) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        reason="ABC continuation",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="abc_continuation",
            ts=datetime.now(timezone.utc),
            score=8.0,
            criteria={
                "pattern": "abc_continuation",
                "close": price,
            },
        ),
    )


def _runner(live_price: float) -> AlpacaRunner:
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._live_prices = lambda symbols: {symbols[0]: live_price}
    runner._latest_price = lambda symbol: live_price
    runner._quote_buffer = {}
    runner._bar_aggregator = None
    return runner


def test_timed_entry_chase_guard_rejects_extended_hot_signal() -> None:
    runner = _runner(5.20)

    reason = runner._timed_entry_chase_reject(_signal(price=5.00), _bar(close=5.20))

    assert reason is not None
    assert "ran 4.0%" in reason


def test_timed_entry_chase_guard_allows_near_signal_price() -> None:
    runner = _runner(5.08)

    reason = runner._timed_entry_chase_reject(_signal(price=5.00), _bar(close=5.08))

    assert reason is None


def test_timed_entry_chase_guard_rejects_red_10s_release() -> None:
    runner = _runner(5.05)
    red_10s = Bar(
        symbol="DXST",
        open=5.08,
        high=5.09,
        low=5.04,
        close=5.05,
        volume=1000,
        ts=datetime.now(timezone.utc),
        timeframe=Timeframe.SEC_10,
    )
    runner._bar_aggregator = _Agg([red_10s])

    reason = runner._timed_entry_chase_reject(_signal(price=5.00), _bar(close=5.05))

    assert reason == "latest 10s candle turned red during entry wait"
