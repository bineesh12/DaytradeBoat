from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Sequence

import pytest

from daytrading.backtest.broker import BacktestBroker, FillModel
from daytrading.backtest.data_loader import load_bars_csv
from daytrading.backtest.data_loader import fetch_alpaca_bars_for_day
from daytrading.backtest.driver import PipelineBacktestDriver, PipelineBacktestResult
from daytrading.backtest.report import BacktestLedger
from daytrading.execution.broker import apply_fill
from daytrading.backtest.service import (
    normalize_flags,
    normalize_session_date,
    normalize_start_time,
    run_backtest,
    run_backtest_sweep,
)
from daytrading.config import Settings, StrategyConfig
from daytrading.exits.manager import ExitManager
from daytrading.backtest.service import normalize_symbol
from daytrading.models import (
    Bar,
    Fill,
    Order,
    PortfolioState,
    ScanResult,
    Side,
    SignalAction,
    Timeframe,
    TradeSignal,
)
from daytrading.pipeline.engine import PipelineResult, TradingPipeline
from daytrading.strategy.entry_guard import check_entry_quality
from daytrading.strategy import warrior_lanes


_BASE_TS = datetime(2026, 6, 1, 13, 30, tzinfo=timezone.utc)


def _bar(symbol: str, minute: int, close: float, *, volume: float = 250_000) -> Bar:
    ts = _BASE_TS + timedelta(minutes=minute)
    return Bar(
        symbol=symbol,
        ts=ts,
        open=close - 0.02,
        high=close + 0.04,
        low=close - 0.04,
        close=close,
        volume=volume,
        timeframe=Timeframe.MIN_1,
    )


class _OnceScanner:
    name = "first_pullback_reclaim"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        bars = list(universe.get("TEST", []))
        if len(bars) == 6:
            return [
                ScanResult(
                    symbol="TEST",
                    scanner_name=self.name,
                    ts=bars[-1].ts,
                    score=100.0,
                    criteria={"setup_tier": "A+ setup"},
                    bars=bars,
                )
            ]
        return []


class _ClockAwareVerifier:
    name = "unit_verifier"

    def __init__(self) -> None:
        self.now_seen = None
        self.reject_seen = ""

    def verify(
        self,
        hit: ScanResult,
        portfolio: PortfolioState,
        *,
        now: datetime | None = None,
    ) -> TradeSignal | None:
        self.now_seen = now
        self.reject_seen = check_entry_quality(
            hit.bars,
            symbol=hit.symbol,
            quotes=[],
            entry_pattern=hit.scanner_name,
            setup_tier=str(hit.criteria.get("setup_tier") or ""),
            now=now,
        ) or ""
        if self.reject_seen:
            return None
        price = hit.bars[-1].close
        return TradeSignal(
            symbol=hit.symbol,
            action=SignalAction.ENTER_LONG,
            quantity=10,
            entry_price=price,
            stop_loss=price - 0.10,
            take_profit=price + 0.25,
            reason="unit breakout",
            scan_result=hit,
        )


class _RejectingVerifier:
    name = "rejecting_verifier"

    def __init__(self) -> None:
        self._last_reject = ""

    def verify(
        self,
        hit: ScanResult,
        portfolio: PortfolioState,
        *,
        now: datetime | None = None,
    ) -> None:
        self._last_reject = "spread too wide (8.40c = 1.08% of $7.75)"
        object.__setattr__(hit, "_reject_reason", self._last_reject)
        return None


def test_backtest_broker_buys_at_ask_and_sells_at_bid() -> None:
    broker = BacktestBroker(FillModel(min_spread_cents=0.02, spread_pct_of_range=0.0))
    portfolio = PortfolioState(cash=1_000)
    bar = _bar("TEST", 0, 10.0)

    buy, status = broker.submit(Order("TEST", Side.BUY, 1, limit_price=10.0), bar, portfolio)
    sell, sell_status = broker.submit(Order("TEST", Side.SELL, 1, limit_price=10.0), bar, portfolio)

    assert status.value == "filled"
    assert sell_status.value == "filled"
    assert buy is not None and buy.price == pytest.approx(10.01)
    assert sell is not None and sell.price == pytest.approx(9.99)


def test_backtest_broker_can_fill_touched_buy_limit_below_close() -> None:
    broker = BacktestBroker(FillModel(min_spread_cents=0.02, spread_pct_of_range=0.0))
    portfolio = PortfolioState(cash=1_000)
    bar = Bar(
        symbol="TEST",
        ts=_BASE_TS,
        open=4.20,
        high=5.08,
        low=4.11,
        close=4.70,
        volume=750_000,
        timeframe=Timeframe.MIN_1,
    )

    fill, status = broker.submit(Order("TEST", Side.BUY, 10, limit_price=4.49), bar, portfolio)

    assert status.value == "filled"
    assert fill is not None
    assert fill.price == pytest.approx(4.50)


def test_level_capped_entry_reprices_late_stair_scout_near_base() -> None:
    portfolio = PortfolioState(cash=10_000)
    pipeline = TradingPipeline(
        scanners=[],
        verifiers={},
        broker=BacktestBroker(FillModel(min_spread_cents=0.01, spread_pct_of_range=0.0)),
        portfolio=portfolio,
        exit_manager=ExitManager(max_unrealized_loss=50.0),
        level_capped_entry_enabled=True,
    )
    bars = [
        Bar(
            symbol="CUPR",
            ts=_BASE_TS,
            open=4.20,
            high=5.08,
            low=4.11,
            close=4.6987,
            volume=750_000,
            timeframe=Timeframe.MIN_1,
        )
    ]
    hit = ScanResult(
        symbol="CUPR",
        scanner_name="shallow_stair_continuation",
        ts=bars[-1].ts,
        score=135.0,
        criteria={
            "pattern": "shallow_stair_continuation",
            "setup_tier": "A+ setup",
            "entry_tier": "stair_scout",
            "base_high": 4.4499,
            "stop_price": 4.4168,
        },
        bars=bars,
    )
    signal = TradeSignal(
        symbol="CUPR",
        action=SignalAction.ENTER_LONG,
        quantity=62,
        entry_price=4.6987,
        stop_loss=4.4168,
        take_profit=5.26,
        reason="stair scout",
        scan_result=hit,
    )

    capped = pipeline._maybe_apply_level_capped_entry(signal, {"CUPR": bars})

    assert capped.entry_price == pytest.approx(4.4944)
    assert hit.criteria["entry_mode"] == "level_capped_scout"
    assert hit.criteria["uncapped_entry_price"] == pytest.approx(4.6987)
    assert pipeline._normal_entry_chase_reject(capped, universe={"CUPR": bars}, now=_BASE_TS) is None


def test_pipeline_backtest_driver_replays_real_pipeline_and_scorecard() -> None:
    broker = BacktestBroker(FillModel(min_spread_cents=0.01, spread_pct_of_range=0.0))
    portfolio = PortfolioState(cash=10_000)
    scanner = _OnceScanner()
    verifier = _ClockAwareVerifier()
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={scanner.name: verifier},
        broker=broker,
        portfolio=portfolio,
        exit_manager=ExitManager(max_unrealized_loss=50.0),
        max_positions=3,
        max_position_shares=1_000,
        max_order_shares=500,
        max_dollar_risk_per_trade=50.0,
    )
    bars = {
        "TEST": [
            _bar("TEST", 0, 8.20),
            _bar("TEST", 1, 8.45),
            _bar("TEST", 2, 8.80),
            _bar("TEST", 3, 9.20),
            _bar("TEST", 4, 9.55),
            _bar("TEST", 5, 10.10, volume=500_000),
            _bar("TEST", 6, 10.40),
        ]
    }

    result = PipelineBacktestDriver(
        bars,
        pipeline=pipeline,
        initial_cash=10_000,
        max_bars_per_symbol=20,
    ).run()

    assert result.cycles == 7
    assert result.scan_hits == 1
    assert result.signals == 1
    assert verifier.now_seen == bars["TEST"][5].ts
    assert "stale data" not in verifier.reject_seen
    assert len(result.fills) == 2
    assert result.scan_events
    assert result.scan_events[0]["symbol"] == "TEST"
    assert result.scan_events[0]["scanner"] == "first_pullback_reclaim"
    assert result.scan_events[0]["a_plus"] is True
    assert result.entry_decisions
    assert result.entry_decisions[0]["ts"] == bars["TEST"][5].ts.isoformat()
    assert result.scorecard["trades_taken"] == 1
    assert result.scorecard["closed_trades"] == 1
    assert result.scorecard["total_pnl"] > 0


def test_pipeline_backtest_reports_rejections_by_layer_and_reason() -> None:
    portfolio = PortfolioState(cash=10_000)
    scanner = _OnceScanner()
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={scanner.name: _RejectingVerifier()},
        broker=BacktestBroker(FillModel(min_spread_cents=0.01, spread_pct_of_range=0.0)),
        portfolio=portfolio,
        exit_manager=ExitManager(max_unrealized_loss=50.0),
    )
    bars = {
        "TEST": [
            _bar("TEST", 0, 8.20),
            _bar("TEST", 1, 8.45),
            _bar("TEST", 2, 8.80),
            _bar("TEST", 3, 9.20),
            _bar("TEST", 4, 9.55),
            _bar("TEST", 5, 10.10, volume=500_000),
        ]
    }

    result = PipelineBacktestDriver(
        bars,
        pipeline=pipeline,
        initial_cash=10_000,
        max_bars_per_symbol=20,
    ).run()

    assert result.rejected == 1
    assert result.rejection_details[0]["blocked_layer"] == "verifier"
    assert result.rejected_by_layer == {"verifier": 1}
    assert result.scorecard["funnel"]["rejected"] == 1
    assert result.scorecard["funnel"]["rejected_by_layer"] == {"verifier": 1}
    reasons = result.scorecard["funnel"]["top_reject_reasons_by_layer"]["verifier"]
    assert reasons == [{"reason": "spread too wide (8.40c = 1.08% of $7.75)", "count": 1}]


def test_10s_breakout_scout_requires_a_plus_context() -> None:
    bars = {"TEST": [_bar("TEST", idx, 9.40 + idx * 0.10, volume=200_000) for idx in range(7)]}
    now = bars["TEST"][6].ts
    ten_sec = Bar(
        symbol="TEST",
        ts=now + timedelta(seconds=20),
        open=10.01,
        high=10.16,
        low=9.99,
        close=10.12,
        volume=75_000,
        timeframe=Timeframe.SEC_10,
    )
    pipeline = TradingPipeline(
        scanners=[],
        verifiers={},
        broker=BacktestBroker(FillModel(min_spread_cents=0.01, spread_pct_of_range=0.0)),
        portfolio=PortfolioState(cash=10_000),
        exit_manager=ExitManager(max_unrealized_loss=50.0),
    )
    driver = PipelineBacktestDriver(
        bars,
        pipeline=pipeline,
        timer_bars_by_symbol={"TEST": [ten_sec]},
        use_micro_breakout_scout=True,
    )
    pipeline._final_entry_quality_reject = lambda *args, **kwargs: pytest.fail(  # type: ignore[method-assign]
        "raw 10s breakout must not reach final guard without A+ context"
    )
    result = driver.run(start=now, end=now)

    assert result.micro_opportunities
    assert result.micro_opportunities[0]["tradeable_context"] is False
    assert result.fills == []
    assert result.trades == []


def test_10s_breakout_scout_can_enter_for_matching_a_plus_context() -> None:
    bars = {"TEST": [_bar("TEST", idx, 9.40 + idx * 0.10, volume=250_000) for idx in range(7)]}
    now = bars["TEST"][6].ts
    ten_sec = Bar(
        symbol="TEST",
        ts=now + timedelta(seconds=20),
        open=10.01,
        high=10.16,
        low=9.99,
        close=10.12,
        volume=75_000,
        timeframe=Timeframe.SEC_10,
    )
    portfolio = PortfolioState(cash=10_000)
    pipeline = TradingPipeline(
        scanners=[],
        verifiers={},
        broker=BacktestBroker(FillModel(min_spread_cents=0.01, spread_pct_of_range=0.0)),
        portfolio=portfolio,
        exit_manager=ExitManager(max_unrealized_loss=50.0),
    )
    driver = PipelineBacktestDriver(
        bars,
        pipeline=pipeline,
        timer_bars_by_symbol={"TEST": [ten_sec]},
        use_micro_breakout_scout=True,
    )
    pipeline._final_entry_quality_reject = lambda *args, **kwargs: None  # type: ignore[method-assign]
    cycle = PipelineResult()
    hit = ScanResult(
        symbol="TEST",
        scanner_name="hod_reclaim",
        ts=now,
        score=120.0,
        criteria={
            "pattern": "hod_reclaim",
            "setup_tier": "A+ setup",
            "base_high": 9.94,
            "close": 10.0,
        },
        bars=list(bars["TEST"]),
    )
    cycle.scan_hits.append(hit)
    cycle.deferred_signals.append(TradeSignal(
        symbol="TEST",
        action=SignalAction.ENTER_LONG,
        quantity=10,
        entry_price=10.0,
        stop_loss=9.70,
        take_profit=10.60,
        reason="accepted hod reclaim",
        scan_result=hit,
    ))
    result = driver.run(start=now, end=now)

    # The public run has no scanner in this unit pipeline, so call the micro
    # hook directly with the A+ context to verify the gated path.
    direct = type(result)(final_portfolio=portfolio)
    driver._record_10s_opportunities(
        direct,
        BacktestLedger(),
        universe={"TEST": bars["TEST"]},
        now=now,
        cycle=cycle,
    )

    assert direct.micro_opportunities
    assert direct.micro_opportunities[0]["tradeable_context"] is True
    assert direct.micro_opportunities[0]["context_scanner"] == "hod_reclaim"
    assert direct.fills
    assert direct.entry_decisions[0]["metadata"]["context_pattern"] == "hod_reclaim"


def _momentum_burst_backtest_driver(ten_sec: Sequence[Bar]) -> PipelineBacktestDriver:
    bars = {"MBUR": [_bar("MBUR", idx, 2.00 + idx * 0.02, volume=100_000) for idx in range(12)]}
    pipeline = TradingPipeline(
        scanners=[],
        verifiers={},
        broker=BacktestBroker(FillModel(min_spread_cents=0.01, spread_pct_of_range=0.0)),
        portfolio=PortfolioState(cash=10_000),
        exit_manager=ExitManager(max_unrealized_loss=50.0),
    )
    return PipelineBacktestDriver(
        bars,
        pipeline=pipeline,
        timer_bars_by_symbol={"MBUR": list(ten_sec)},
        use_momentum_burst_replay=True,
    )


def _ten_bar(second: int, close: float, *, width_pct: float, volume: float = 5_000) -> Bar:
    width = close * width_pct
    return Bar(
        symbol="MBUR",
        ts=_BASE_TS + timedelta(seconds=second),
        open=close - width * 0.15,
        high=close + width * 0.5,
        low=close - width * 0.5,
        close=close,
        volume=volume,
        timeframe=Timeframe.SEC_10,
    )


def test_momentum_burst_replay_allows_smooth_10s_tape() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)

    smooth, median_range = driver._momentum_burst_10s_tape_is_smooth("MBUR", ten_sec[-1].ts)
    signal = driver._momentum_burst_replay_signal("MBUR", ten_sec[-1], driver._bars_by_symbol["MBUR"])

    assert smooth is True
    assert median_range <= 2.0
    assert signal is not None
    assert signal.scan_result is not None
    assert signal.scan_result.criteria["median_10s_range_pct"] <= 2.0


def test_momentum_burst_hit_run_signal_uses_1r_and_short_hold() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_momentum_burst_hit_run = True

    signal = driver._momentum_burst_replay_signal(
        "MBUR",
        ten_sec[-1],
        driver._bars_by_symbol["MBUR"],
        hit_run=True,
    )

    assert signal is not None
    assert signal.max_hold_seconds == 45.0
    assert signal.scan_result.scanner_name == "momentum_burst_hit_run"
    assert signal.scan_result.criteria["entry_mode"] == "momentum_burst_hit_run"
    assert signal.take_profit == round(
        signal.entry_price + (signal.entry_price - signal.stop_loss),
        2,
    )


def test_warrior_squeeze_backtest_signal_requires_named_playbook_trigger() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(10)]
    ten_sec.extend([
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 3.46, 3.56, 3.45, 3.53, 260_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 3.54, 3.62, 3.49, 3.58, 320_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=120), 3.58, 3.70, 3.50, 3.64, 760_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=130), 3.65, 4.08, 3.56, 3.97, 845_000, Timeframe.SEC_10),
    ])
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_starter_size_factor = 0.35
    driver._warrior_squeeze_rejection_high["MBUR"] = 2.25
    pending = {
        "ts": ten_sec[-2].ts,
        "breakout_close": 3.64,
        "breakout_high": 3.50,
        "breakout_volume": 760_000,
    }
    context = driver._warrior_squeeze_pullaway_context("MBUR", ten_sec[-1], pending)

    generic = driver._momentum_burst_replay_signal(
        "MBUR",
        ten_sec[-1],
        driver._bars_by_symbol["MBUR"],
        hit_run=True,
        strategy_override="warrior_squeeze_playbook",
        size_factor_override=0.35,
    )
    signal = driver._momentum_burst_replay_signal(
        "MBUR",
        ten_sec[-1],
        driver._bars_by_symbol["MBUR"],
        hit_run=True,
        entry_context={**pending, **(context or {})},
        strategy_override="warrior_squeeze_playbook",
        size_factor_override=0.35,
    )

    assert context is not None
    assert generic is None
    assert signal is not None
    assert signal.scan_result.scanner_name == "warrior_squeeze_playbook"
    assert signal.scan_result.criteria["entry_mode"] == "warrior_squeeze_playbook"
    assert signal.scan_result.criteria["entry_trigger"] == "warrior_level_pullaway"
    assert signal.scan_result.criteria["size_factor"] == 0.35
    assert signal.quantity * signal.entry_price <= 2000.0 + signal.entry_price
    assert signal.quantity * (signal.entry_price - signal.stop_loss) <= 150.0


def test_warrior_squeeze_backtest_pullaway_uses_capped_level_price() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(10)]
    ten_sec.extend([
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 3.46, 3.56, 3.45, 3.53, 260_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 3.54, 3.62, 3.49, 3.58, 320_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=120), 3.58, 3.70, 3.50, 3.64, 760_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=130), 3.65, 4.08, 3.56, 3.97, 845_000, Timeframe.SEC_10),
    ])
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_starter_size_factor = 0.35
    driver._warrior_squeeze_rejection_high["MBUR"] = 2.25
    pending = {
        "ts": ten_sec[-2].ts,
        "breakout_close": 3.64,
        "breakout_high": 3.50,
        "breakout_volume": 760_000,
    }

    context = driver._warrior_squeeze_pullaway_context("MBUR", ten_sec[-1], pending)
    assert context is not None
    signal = driver._momentum_burst_replay_signal(
        "MBUR",
        ten_sec[-1],
        driver._bars_by_symbol["MBUR"],
        hit_run=True,
        entry_context={**pending, **context},
        strategy_override="warrior_squeeze_playbook",
        size_factor_override=0.35,
    )

    assert signal is not None
    assert signal.scan_result.scanner_name == "warrior_squeeze_playbook"
    assert signal.scan_result.criteria["variant"] == "warrior_proof_pullback_hold"
    assert signal.scan_result.criteria["entry_trigger"] == "warrior_level_pullaway"
    assert signal.scan_result.criteria["pullaway_level"] == 3.5
    assert signal.scan_result.criteria["max_pay"] == 3.71
    assert signal.entry_price == 3.71
    assert signal.entry_price < 3.97
    assert signal.take_profit > signal.entry_price + (signal.entry_price - signal.stop_loss) * 2.5
    assert signal.max_hold_seconds == 180.0
    assert signal.quantity * signal.entry_price <= 2000.0 + signal.entry_price
    assert signal.quantity * (signal.entry_price - signal.stop_loss) <= 150.0


def test_warrior_squeeze_backtest_clwt_fast_pullaway_without_slow_proof_hold() -> None:
    ten_sec = [_ten_bar(i * 10, 1.85 + i * 0.01, width_pct=0.006, volume=8_000) for i in range(10)]
    ten_sec.extend([
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 1.20, 1.60, 1.18, 1.48, 320_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 2.00, 2.25, 1.84, 1.92, 900_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=120), 2.70, 3.50, 2.70, 3.16, 760_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=130), 3.17, 4.08, 3.10, 3.9674, 845_000, Timeframe.SEC_10),
    ])
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_rejection_high["MBUR"] = 2.25
    driver._warrior_squeeze_rejection_reason["MBUR"] = "high-volume shooting-star rejection"
    pending = {
        "ts": ten_sec[-2].ts,
        "breakout_close": 3.16,
        "breakout_high": 3.50,
        "breakout_volume": 760_000,
        "entry_trigger": "warrior_a_plus_reclaim",
    }

    context = driver._warrior_squeeze_pullaway_context("MBUR", ten_sec[-1], pending)
    assert context is not None
    signal = driver._momentum_burst_replay_signal(
        "MBUR",
        ten_sec[-1],
        driver._bars_by_symbol["MBUR"],
        hit_run=True,
        entry_context={**pending, **context},
        strategy_override="warrior_squeeze_playbook",
        size_factor_override=0.35,
    )

    assert signal is not None
    assert signal.scan_result.criteria["entry_trigger"] == "warrior_level_pullaway"
    assert signal.scan_result.criteria["variant"] == "warrior_clwt_fast_pullaway"
    assert signal.scan_result.criteria["max_pay"] == 4.025
    assert signal.entry_price == 3.9674
    assert signal.take_profit > signal.entry_price


def test_warrior_squeeze_backtest_fast_pullaway_rejects_immediate_wide_red_fakeout() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 3.24, 3.57, 3.24, 3.536, 14_099, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 3.56, 3.7087, 3.55, 3.70, 6_998, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 3.56, 3.56, 3.54, 3.54, 500, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 3.55, 4.00, 3.51, 4.00, 11_220, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 4.03, 4.29, 3.53, 3.7233, 18_356, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 3.7225, 3.74, 3.52, 3.5395, 14_195, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 3.53, 4.38, 3.51, 4.30, 38_079, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 4.24, 5.55, 4.20, 5.375, 127_967, Timeframe.SEC_10),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_rejection_high["MBUR"] = 4.29
    driver._warrior_squeeze_rejection_reason["MBUR"] = "first explosive 10s spike"
    pending = {
        "ts": ten_sec[-2].ts,
        "breakout_close": 4.30,
        "breakout_high": 4.38,
        "breakout_volume": 38_079,
        "entry_trigger": "warrior_a_plus_reclaim",
    }

    context = driver._warrior_squeeze_pullaway_context("MBUR", ten_sec[-1], pending)

    assert context is None


def test_warrior_level_break_rejects_late_extended_cast_style_break() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 4.83, 6.50, 4.74, 6.40, 74_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 6.45, 7.20, 6.30, 7.00, 95_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 7.10, 8.00, 7.00, 7.80, 90_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 8.32, 8.65, 8.30, 8.53, 98_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 8.44, 8.78, 8.44, 8.74, 87_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 8.72, 8.97, 8.67, 8.91, 84_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 8.94, 9.45, 8.87, 9.39, 112_000, Timeframe.SEC_10),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50

    context = driver._warrior_level_break_starter_context(
        "MBUR",
        ten_sec[-1],
        window_high=8.97,
    )

    assert context is None


def test_warrior_level_break_allows_extended_exceptional_volume_break() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=-20), 1.18, 1.22, 1.18, 1.20, 40_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=-10), 1.20, 1.24, 1.19, 1.22, 45_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 1.20, 1.60, 1.18, 1.48, 320_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 1.48, 2.25, 1.46, 2.20, 900_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 2.20, 3.50, 2.18, 3.16, 760_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 3.17, 4.08, 3.10, 3.9674, 845_000, Timeframe.SEC_10),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50

    context = driver._warrior_level_break_starter_context(
        "MBUR",
        ten_sec[-1],
        window_high=3.50,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_level_break_starter"


def test_warrior_squeeze_backtest_equal_high_pullaway_allows_clwt_style_hold() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(10)]
    ten_sec.extend([
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 3.42, 3.50, 3.36, 3.47, 180_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 3.48, 3.54, 3.42, 3.52, 230_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=120), 3.51, 3.58, 3.45, 3.55, 260_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=130), 3.54, 3.59, 3.49, 3.57, 280_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=140), 3.56, 3.60, 3.50, 3.59, 310_000, Timeframe.SEC_10),
    ])
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_rejection_high["MBUR"] = 2.25
    driver._warrior_squeeze_rejection_reason["MBUR"] = "high-volume shooting-star rejection"
    pending = {
        "ts": ten_sec[-2].ts,
        "breakout_close": 3.57,
        "breakout_high": 3.60,
        "breakout_volume": 280_000,
    }

    context = driver._warrior_squeeze_equal_high_pullaway_context(
        "MBUR",
        ten_sec[-1],
        pending,
        window_high=3.60,
    )
    assert context is not None
    signal = driver._momentum_burst_replay_signal(
        "MBUR",
        ten_sec[-1],
        driver._bars_by_symbol["MBUR"],
        hit_run=True,
        entry_context={**pending, **context},
        strategy_override="warrior_squeeze_playbook",
        size_factor_override=0.35,
    )

    assert signal is not None
    assert signal.scan_result.criteria["entry_trigger"] == "warrior_equal_high_pullaway"
    assert signal.scan_result.criteria["variant"] == "warrior_equal_high_pullaway"
    assert signal.entry_price == 3.59
    assert signal.take_profit > signal.entry_price
    assert signal.quantity * signal.entry_price <= 2000.0 + signal.entry_price
    assert signal.quantity * (signal.entry_price - signal.stop_loss) <= 150.0


def test_warrior_squeeze_backtest_equal_high_pullaway_rejects_above_five() -> None:
    ten_sec = [_ten_bar(i * 10, 6.80 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(10)]
    ten_sec.extend([
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 7.10, 7.45, 7.00, 7.34, 250_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 7.33, 7.50, 7.18, 7.42, 320_000, Timeframe.SEC_10),
    ])
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_rejection_high["MBUR"] = 6.95
    driver._warrior_squeeze_rejection_reason["MBUR"] = "high-volume shooting-star rejection"
    pending = {
        "ts": ten_sec[-2].ts,
        "breakout_close": 7.34,
        "breakout_high": 7.45,
        "breakout_volume": 250_000,
    }

    context = driver._warrior_squeeze_equal_high_pullaway_context(
        "MBUR",
        ten_sec[-1],
        pending,
        window_high=7.50,
    )

    assert context is None


def test_warrior_squeeze_backtest_curl_reclaim_allows_level_starter_without_new_high() -> None:
    ten_sec = [_ten_bar(i * 10, 5.80 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(10)]
    ten_sec.extend([
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 6.30, 7.00, 6.24, 6.88, 900_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 6.28, 6.60, 6.20, 6.52, 450_000, Timeframe.SEC_10),
    ])
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_starter_size_factor = 0.35
    pending = {
        "ts": ten_sec[-2].ts,
        "breakout_close": 6.88,
        "breakout_high": 7.00,
        "breakout_volume": 900_000,
    }

    context = driver._warrior_squeeze_curl_reclaim_context(
        "MBUR",
        ten_sec[-1],
        pending,
        window_high=7.00,
    )
    assert context is not None
    signal = driver._momentum_burst_replay_signal(
        "MBUR",
        ten_sec[-1],
        driver._bars_by_symbol["MBUR"],
        hit_run=True,
        entry_context={**pending, **context},
        strategy_override="warrior_squeeze_playbook",
        size_factor_override=0.35,
    )

    assert signal is not None
    assert signal.scan_result.scanner_name == "warrior_squeeze_playbook"
    assert signal.scan_result.criteria["variant"] == "warrior_curl_reclaim_starter"
    assert signal.scan_result.criteria["entry_trigger"] == "warrior_curl_reclaim"
    assert signal.scan_result.criteria["pullaway_level"] == 6.5
    assert signal.entry_price == 6.52
    assert signal.entry_price < pending["breakout_high"]
    assert signal.take_profit > signal.entry_price
    assert signal.quantity * signal.entry_price <= 2000.0 + signal.entry_price
    assert signal.quantity * (signal.entry_price - signal.stop_loss) <= 150.0


def test_warrior_squeeze_curl_reclaim_rejects_wgrx_giant_expansion_bar() -> None:
    ten_sec = [
        Bar("WGRX", _BASE_TS + timedelta(seconds=0), 3.24, 3.57, 3.24, 3.536, 14_099, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=10), 3.56, 3.7087, 3.55, 3.70, 6_998, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=20), 3.56, 3.56, 3.54, 3.54, 500, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=30), 3.55, 4.00, 3.51, 4.00, 11_220, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=40), 4.03, 4.29, 3.53, 3.7233, 18_356, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=50), 3.7225, 3.74, 3.52, 3.5395, 14_195, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=60), 3.53, 4.38, 3.51, 4.30, 38_079, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=70), 4.24, 5.55, 4.20, 5.375, 127_967, Timeframe.SEC_10),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    pending = {
        "ts": ten_sec[-2].ts,
        "breakout_close": 4.30,
        "breakout_high": 4.38,
        "breakout_volume": 38_079,
    }

    context = driver._warrior_squeeze_curl_reclaim_context(
        "WGRX",
        ten_sec[-1],
        pending,
        window_high=4.38,
    )

    assert context is None
    reason = warrior_lanes.warrior_failed_burst_watch_reason(ten_sec[-1])
    assert reason is not None
    assert "blocked giant Warrior spike" in reason


def test_warrior_failed_burst_recovery_allows_wgrx_fresh_reclaim_after_spike() -> None:
    ten_sec = [
        Bar("WGRX", _BASE_TS + timedelta(seconds=0), 3.24, 3.57, 3.24, 3.536, 14_099, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=10), 3.56, 3.7087, 3.55, 3.70, 6_998, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=20), 3.56, 3.56, 3.54, 3.54, 500, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=30), 3.55, 4.00, 3.51, 4.00, 11_220, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=40), 4.03, 4.29, 3.53, 3.7233, 18_356, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=50), 3.7225, 3.74, 3.52, 3.5395, 14_195, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=60), 3.53, 4.38, 3.51, 4.30, 38_079, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=70), 4.24, 5.55, 4.20, 5.375, 127_967, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=80), 5.35, 6.04, 4.94, 5.10, 151_609, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=90), 5.05, 5.13, 4.50, 4.515, 74_410, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=100), 4.51, 4.98, 4.24, 4.95, 74_497, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=110), 4.9431, 5.3913, 4.8403, 5.38, 102_828, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=120), 5.3556, 6.17, 5.08, 5.70, 170_765, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=130), 5.67, 5.74, 5.07, 5.11, 126_551, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=140), 5.10, 5.38, 5.00, 5.31, 65_476, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=150), 5.31, 5.55, 5.2479, 5.285, 75_908, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=160), 5.30, 5.65, 5.25, 5.54, 53_327, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=170), 5.5021, 6.12, 5.47, 5.89, 132_883, Timeframe.SEC_10),
        Bar("WGRX", _BASE_TS + timedelta(seconds=180), 5.9088, 6.49, 5.75, 6.35, 169_573, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_failed_burst_recovery_context(
        ten_sec[-2],
        history=ten_sec,
        failed_high=5.55,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_failed_burst_recovery"
    assert context["entry_price_override"] < ten_sec[-1].close
    assert context["target_price_override"] > context["entry_price_override"]


def test_warrior_squeeze_backtest_reentry_uses_current_level_pullaway() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(10)]
    ten_sec.extend([
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 3.96, 5.00, 3.94, 4.68, 792_000, Timeframe.SEC_10),
    ])
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_rejection_high["MBUR"] = 2.25
    driver._mb_hit_run_counts["MBUR"] = 1
    pending = {
        "ts": ten_sec[-2].ts,
        "breakout_close": 4.30,
        "breakout_high": 4.50,
        "breakout_volume": 600_000,
    }

    context = driver._warrior_squeeze_pullaway_context("MBUR", ten_sec[-1], pending)

    assert context is not None
    assert context["pullaway_level"] == 4.5
    assert context["max_pay"] == 4.77
    assert context["entry_price_override"] == 4.68
    assert context["stop_price_override"] < 4.5
    assert 4.99 <= context["target_price_override"] <= 5.0


def test_warrior_squeeze_backtest_a_plus_reclaim_scout_does_not_arm_playbook() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_rejection_high["MBUR"] = 2.25
    hit = ScanResult(
        symbol="MBUR",
        scanner_name="vwap_pullback",
        ts=ten_sec[-1].ts,
        score=300.0,
        criteria={
            "pattern": "vwap_pullback",
            "setup_tier": "A+ setup",
            "entry_tier": "a_plus_reclaim_scout",
            "close": 3.97,
            "volume": 4_000_000,
        },
        bars=[
            Bar("MBUR", _BASE_TS, 3.16, 3.50, 2.70, 3.16, 760_000, Timeframe.SEC_10),
            Bar("MBUR", _BASE_TS + timedelta(seconds=10), 3.17, 4.08, 3.10, 3.97, 845_000, Timeframe.SEC_10),
        ],
    )

    driver._arm_momentum_burst_from_cycle(PipelineResult(scan_hits=[hit]), ten_sec[-1].ts)

    assert "MBUR" not in driver._momentum_burst_pending
    assert "MBUR" not in driver._momentum_burst_armed


def test_warrior_squeeze_backtest_generic_a_plus_does_not_arm_playbook() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    hit = ScanResult(
        symbol="MBUR",
        scanner_name="vwap_pullback",
        ts=ten_sec[-1].ts,
        score=300.0,
        criteria={
            "pattern": "vwap_pullback",
            "setup_tier": "A+ setup",
            "entry_tier": "deep_runner_scout",
            "close": 3.97,
            "volume": 4_000_000,
        },
        bars=[
            Bar("MBUR", _BASE_TS, 3.16, 3.50, 2.70, 3.16, 760_000, Timeframe.SEC_10),
            Bar("MBUR", _BASE_TS + timedelta(seconds=10), 3.17, 4.08, 3.10, 3.97, 845_000, Timeframe.SEC_10),
        ],
    )

    driver._arm_momentum_burst_from_cycle(PipelineResult(scan_hits=[hit]), ten_sec[-1].ts)

    assert "MBUR" not in driver._momentum_burst_pending
    assert "MBUR" not in driver._momentum_burst_armed


def test_warrior_squeeze_backtest_near_hod_a_plus_does_not_seed_playbook() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    hit = ScanResult(
        symbol="MBUR",
        scanner_name="vwap_pullback",
        ts=ten_sec[-1].ts,
        score=300.0,
        criteria={
            "pattern": "vwap_pullback",
            "setup_tier": "A+ setup",
            "close": 8.58,
            "session_high": 9.99,
            "volume": 900_000,
        },
        bars=[
            Bar("MBUR", _BASE_TS, 7.60, 8.70, 7.60, 8.52, 290_000, Timeframe.SEC_10),
            Bar("MBUR", _BASE_TS + timedelta(seconds=10), 8.50, 9.74, 8.21, 8.67, 291_000, Timeframe.SEC_10),
        ],
    )

    driver._arm_momentum_burst_from_cycle(PipelineResult(scan_hits=[hit]), ten_sec[-1].ts)

    assert "MBUR" not in driver._momentum_burst_pending
    assert "MBUR" not in driver._momentum_burst_armed


def test_warrior_squeeze_backtest_a_plus_reclaim_stop_stays_inside_final_guard() -> None:
    ten_sec = [
        _ten_bar(0, 7.90, width_pct=0.012, volume=120_000),
        _ten_bar(10, 8.10, width_pct=0.012, volume=160_000),
        Bar(
            "MBUR",
            _BASE_TS + timedelta(seconds=20),
            8.55,
            9.10,
            8.35,
            9.00,
            220_000,
            Timeframe.SEC_10,
        ),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_rejection_high["MBUR"] = 4.00
    pending = {
        "ts": ten_sec[-2].ts,
        "breakout_close": 8.10,
        "breakout_high": 8.25,
        "breakout_volume": 180_000,
        "entry_trigger": "warrior_a_plus_reclaim",
    }

    context = driver._warrior_squeeze_pullaway_context("MBUR", ten_sec[-1], pending)

    assert context is not None
    entry = context["entry_price_override"]
    stop = context["stop_price_override"]
    assert (entry - stop) / entry <= 0.06


def test_momentum_burst_hit_run_backtest_defaults_to_one_entry() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)

    assert driver._mb_hit_run_max_entries == 1


def test_momentum_burst_hit_run_backtest_giveback_blocks_symbol() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._mb_hit_run_max_giveback = 20.0

    assert driver._record_mb_hit_run_pnl("MBUR", 55.0) == ""
    reason = driver._record_mb_hit_run_pnl("MBUR", -25.0)

    assert "gave back" in reason
    assert "MBUR" in driver._mb_hit_run_day_blocked


def test_momentum_burst_hit_run_backtest_daily_loss_works_when_giveback_disabled() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._mb_hit_run_stop_after_giveback = False
    driver._mb_hit_run_daily_loss_stop = 20.0

    reason = driver._record_mb_hit_run_pnl("MBUR", -22.0)

    assert "daily hit-run loss" in reason
    assert "MBUR" in driver._mb_hit_run_day_blocked


def test_momentum_burst_hit_run_backtest_giveback_can_be_disabled() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._mb_hit_run_stop_after_giveback = False
    driver._mb_hit_run_max_giveback = 20.0

    assert driver._record_mb_hit_run_pnl("MBUR", 55.0) == ""
    assert driver._record_mb_hit_run_pnl("MBUR", -25.0) == ""
    assert "MBUR" not in driver._mb_hit_run_day_blocked


def test_backtest_momentum_burst_respects_global_active_scalp_latch() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=80_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_momentum_burst_hit_run = True
    driver._momentum_burst_armed["MBUR"] = ten_sec[-1].ts
    driver._momentum_burst_window_high["MBUR"] = 2.20
    driver._mb_bracket["OTHER"] = {
        "stop": 3.90,
        "target": 4.30,
        "qty": 10,
        "entry": 4.10,
        "ts": ten_sec[-1].ts,
        "max_hold": 45,
        "strategy": "warrior_squeeze_playbook",
    }
    result = PipelineBacktestResult()

    driver._maybe_execute_momentum_burst_replay(
        result,
        BacktestLedger(),
        universe=driver._bars_by_symbol,
        quotes={},
        now=ten_sec[-1].ts,
    )

    assert not result.fills
    assert driver._has_active_replay_scalp() is True


def test_warrior_backtest_ambiguous_green_bar_takes_target_before_stop() -> None:
    ten_sec = [_ten_bar(i * 10, 8.50 + i * 0.01, width_pct=0.006, volume=80_000) for i in range(12)]
    bar = Bar(
        "MBUR",
        ten_sec[-1].ts,
        8.50,
        9.75,
        8.21,
        8.67,
        290_000,
        Timeframe.SEC_10,
    )
    ten_sec[-1] = bar
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._mb_bracket["MBUR"] = {
        "stop": 8.24,
        "target": 9.35,
        "qty": 63,
        "entry": 8.58,
        "ts": bar.ts - timedelta(seconds=10),
        "max_hold": 45,
        "strategy": "warrior_squeeze_playbook",
    }
    apply_fill(
        driver._pipeline.portfolio,
        Fill("MBUR", Side.BUY, 63, 8.58, bar.ts - timedelta(seconds=10)),
    )
    result = PipelineBacktestResult()
    ledger = BacktestLedger()

    driver._process_mb_brackets(result, ledger, bar.ts)

    assert ledger.trades
    assert ledger.trades[-1]["exit_reason"] == "mb_bracket_target: Warrior Squeeze"
    assert ledger.trades[-1]["exit_price"] == pytest.approx(9.345)
    assert ledger.trades[-1]["exit_price"] > 8.58
    assert driver._mb_bracket["MBUR"]["partial_taken"] is True
    assert driver._mb_bracket["MBUR"]["qty"] == 42


def test_warrior_backtest_emergency_dump_exits_at_close() -> None:
    bar = Bar(
        "MBUR",
        _BASE_TS + timedelta(seconds=120),
        10.20,
        10.30,
        9.48,
        9.50,
        210_000,
        Timeframe.SEC_10,
    )
    ten_sec = [_ten_bar(i * 10, 10.00 + i * 0.01, width_pct=0.006, volume=80_000) for i in range(12)]
    ten_sec[-1] = bar
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._mb_bracket["MBUR"] = {
        "stop": 9.20,
        "target": 10.90,
        "qty": 63,
        "entry": 10.00,
        "ts": bar.ts - timedelta(seconds=10),
        "max_hold": 45,
        "strategy": "warrior_squeeze_playbook",
    }
    apply_fill(
        driver._pipeline.portfolio,
        Fill("MBUR", Side.BUY, 63, 10.00, bar.ts - timedelta(seconds=10)),
    )
    result = PipelineBacktestResult()
    ledger = BacktestLedger()

    driver._process_mb_brackets(result, ledger, bar.ts)

    assert ledger.trades
    assert ledger.trades[-1]["exit_reason"] == "mb_bracket_dump: Warrior Squeeze"
    assert ledger.trades[-1]["exit_price"] == pytest.approx(9.495)
    assert "MBUR" not in driver._mb_bracket


def test_warrior_squeeze_profit_lock_blocks_reentry_until_fresh_high() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=80_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_target_wins["MBUR"] = 2
    driver._momentum_burst_armed["MBUR"] = ten_sec[-1].ts
    driver._momentum_burst_window_high["MBUR"] = 3.00
    result = PipelineBacktestResult()

    driver._maybe_execute_momentum_burst_replay(
        result,
        BacktestLedger(),
        universe=driver._bars_by_symbol,
        quotes={},
        now=ten_sec[-1].ts,
    )

    assert not result.fills
    assert result.rejection_details
    assert result.rejection_details[-1]["blocked_layer"] == "warrior_squeeze_playbook_profit_lock"
    assert "target win banked" in result.rejection_details[-1]["reason"]


def test_warrior_profit_lock_does_not_reenter_on_generic_trend_pullback() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=80_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_target_wins["MBUR"] = 1
    driver._momentum_burst_armed["MBUR"] = ten_sec[-1].ts
    driver._momentum_burst_window_high["MBUR"] = 3.00
    driver._warrior_prior_runner_continuation_pullback_context = lambda *args, **kwargs: None
    driver._warrior_squeeze_second_leg_reclaim_context = lambda *args, **kwargs: None

    def _generic_trend_pullback_should_not_be_used(*args, **kwargs):
        raise AssertionError("generic trend pullback must stay locked after a Warrior target win")

    driver._warrior_trend_pullback_reclaim_context = _generic_trend_pullback_should_not_be_used
    result = PipelineBacktestResult()

    driver._maybe_execute_momentum_burst_replay(
        result,
        BacktestLedger(),
        universe=driver._bars_by_symbol,
        quotes={},
        now=ten_sec[-1].ts,
    )

    assert not result.fills
    assert "MBUR" not in driver._momentum_burst_pending
    assert result.rejection_details
    assert result.rejection_details[-1]["blocked_layer"] == "warrior_squeeze_playbook_profit_lock"


def test_backtest_timed_release_chase_uses_queued_price_not_stale_anchor() -> None:
    driver = _momentum_burst_backtest_driver([])
    hit = ScanResult(
        symbol="CAST",
        scanner_name="shallow_stair_continuation",
        ts=_BASE_TS,
        score=95.0,
        criteria={
            "pattern": "shallow_stair_continuation",
            "setup_tier": "A+ setup",
            "setup_anchor": 3.12,
            "queued_entry_price": 3.63,
        },
        bars=[],
    )
    signal = TradeSignal(
        symbol="CAST",
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=3.63,
        stop_loss=3.41,
        take_profit=4.06,
        reason="queued CAST stair scout",
        scan_result=hit,
    )
    release = Bar(
        symbol="CAST",
        ts=_BASE_TS + timedelta(seconds=10),
        open=3.62,
        high=3.66,
        low=3.60,
        close=3.635,
        volume=100_000,
        timeframe=Timeframe.SEC_10,
    )
    assert driver._timed_release_chase_reject(signal, release) is None

    chased = Bar(
        symbol="CAST",
        ts=_BASE_TS + timedelta(seconds=20),
        open=3.80,
        high=3.88,
        low=3.79,
        close=3.86,
        volume=100_000,
        timeframe=Timeframe.SEC_10,
    )
    reason = driver._timed_release_chase_reject(signal, chased)
    assert reason is not None
    assert "queued setup $3.6300" in reason


def test_warrior_squeeze_second_leg_reclaim_after_deep_washout() -> None:
    closes = [
        3.90, 4.20, 6.50, 9.80, 12.40, 14.20, 13.70, 11.20,
        9.30, 8.20, 7.80, 8.10, 8.25, 8.35, 8.20, 8.45,
        8.70, 9.05, 9.60,
    ]
    volumes = [
        50_000, 80_000, 220_000, 420_000, 650_000, 720_000, 500_000, 430_000,
        260_000, 210_000, 180_000, 190_000, 170_000, 185_000, 160_000, 190_000,
        220_000, 260_000, 420_000,
    ]
    ten_sec = [
        _ten_bar(i * 10, close, width_pct=0.025, volume=volumes[i])
        for i, close in enumerate(closes)
    ]
    ten_sec[-1] = Bar(
        symbol="MBUR",
        ts=ten_sec[-1].ts,
        open=9.22,
        high=9.72,
        low=9.28,
        close=9.62,
        volume=420_000,
        timeframe=Timeframe.SEC_10,
    )
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_target_wins["MBUR"] = 1
    driver._mb_hit_run_counts["MBUR"] = 1

    context = driver._warrior_squeeze_second_leg_reclaim_context(
        "MBUR",
        ten_sec[-1],
        {
            "breakout_close": ten_sec[-1].close,
            "breakout_high": ten_sec[-1].high,
            "breakout_volume": ten_sec[-1].volume,
        },
        window_high=14.78,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_second_leg_reclaim"
    assert context["variant_override"] == "warrior_second_leg_reclaim"
    assert context["washout_pct"] >= 25.0
    assert context["entry_price_override"] <= context["max_pay"]


def test_warrior_prior_runner_continuation_pullback_context_allows_controlled_reclaim() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 6.10, 6.55, 6.05, 6.45, 120_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 6.45, 7.20, 6.40, 7.08, 210_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 7.05, 7.95, 7.00, 7.72, 340_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 7.70, 8.80, 7.65, 8.45, 520_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 8.42, 8.62, 8.08, 8.18, 260_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 8.10, 8.15, 7.78, 7.92, 190_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 7.92, 8.04, 7.72, 7.86, 160_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 7.86, 7.98, 7.76, 7.90, 145_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 7.90, 8.06, 7.82, 8.00, 150_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 8.00, 8.10, 7.88, 8.04, 170_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 8.04, 8.12, 7.90, 8.06, 180_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 8.06, 8.20, 8.00, 8.14, 260_000, Timeframe.SEC_10),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True

    context = driver._warrior_prior_runner_continuation_pullback_context(
        "MBUR",
        ten_sec[-1],
        window_high=8.80,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_prior_runner_continuation_pullback"
    assert context["variant_override"] == "warrior_prior_runner_continuation_pullback"
    assert context["entry_price_override"] <= context["max_pay"]
    assert context["stop_price_override"] < context["base_low"]
    assert context["target_price_override"] > context["entry_price_override"]
    assert 6.0 <= context["pullback_pct"] <= 28.0


def test_warrior_prior_runner_continuation_pullback_rejects_dump_base() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 6.10, 6.55, 6.05, 6.45, 120_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 6.45, 7.20, 6.40, 7.08, 210_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 7.05, 7.95, 7.00, 7.72, 340_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 7.70, 8.80, 7.65, 8.45, 520_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 8.42, 8.62, 8.08, 8.18, 260_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 8.10, 8.15, 7.78, 7.92, 190_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 7.92, 8.04, 7.72, 7.86, 160_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 8.35, 8.42, 7.76, 7.70, 360_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 7.70, 8.06, 7.82, 8.00, 150_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 8.00, 8.10, 7.88, 8.04, 170_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 8.04, 8.12, 7.90, 8.06, 180_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 8.06, 8.20, 8.00, 8.14, 260_000, Timeframe.SEC_10),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True

    context = driver._warrior_prior_runner_continuation_pullback_context(
        "MBUR",
        ten_sec[-1],
        window_high=8.80,
    )

    assert context is None


def test_warrior_prior_runner_continuation_pullback_rejects_weak_reclaim_volume() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 6.10, 6.55, 6.05, 6.45, 120_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 6.45, 7.20, 6.40, 7.08, 210_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 7.05, 7.95, 7.00, 7.72, 340_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 7.70, 8.80, 7.65, 8.45, 520_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 8.42, 8.62, 8.08, 8.18, 260_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 8.10, 8.15, 7.78, 7.92, 190_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 7.92, 8.04, 7.72, 7.86, 160_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 7.86, 7.98, 7.76, 7.90, 80_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 7.90, 8.06, 7.82, 8.00, 45_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 8.00, 8.10, 7.88, 8.04, 50_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 8.04, 8.12, 7.90, 8.06, 55_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 8.06, 8.20, 8.00, 8.14, 85_000, Timeframe.SEC_10),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True

    context = driver._warrior_prior_runner_continuation_pullback_context(
        "MBUR",
        ten_sec[-1],
        window_high=8.80,
    )

    assert context is None


def test_warrior_trend_pullback_reclaim_context_allows_cast_style_base() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 6.80, 7.30, 6.76, 7.22, 180_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 7.24, 7.85, 7.20, 7.78, 260_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 7.80, 8.55, 7.74, 8.42, 330_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 8.42, 9.35, 8.35, 9.12, 440_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 9.12, 10.20, 9.00, 9.82, 620_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 9.78, 10.34, 9.62, 9.90, 520_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 9.86, 9.98, 9.18, 9.30, 240_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 9.30, 9.42, 8.82, 8.96, 170_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 8.96, 9.10, 8.54, 8.76, 145_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 8.76, 8.98, 8.46, 8.84, 135_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 8.84, 9.08, 8.58, 8.98, 150_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 8.98, 9.16, 8.70, 9.06, 165_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=120), 9.08, 9.24, 8.86, 9.18, 185_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=130), 9.18, 9.45, 9.05, 9.39, 320_000, Timeframe.SEC_10),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True

    context = driver._warrior_trend_pullback_reclaim_context(
        "MBUR",
        ten_sec[-1],
        window_high=10.34,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_trend_pullback_reclaim"
    assert context["variant_override"] == "warrior_trend_pullback_reclaim"
    assert context["entry_price_override"] <= context["max_pay"]
    assert context["stop_price_override"] < context["entry_price_override"]
    assert context["target_price_override"] > context["entry_price_override"]
    assert 2.5 <= context["pullback_pct"] <= 22.0


def test_warrior_high_base_reclaim_allows_sprc_style_volume_reclaim() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 7.20, 10.40, 7.10, 9.80, 480_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 9.78, 10.55, 9.40, 9.70, 220_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 9.70, 9.95, 9.30, 9.45, 110_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 9.45, 9.65, 9.10, 9.30, 92_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 9.30, 9.50, 9.05, 9.25, 75_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 9.24, 9.40, 8.92, 9.02, 68_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 9.02, 9.20, 8.74, 8.92, 64_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 8.92, 9.08, 8.62, 8.86, 58_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 8.86, 9.02, 8.58, 8.78, 52_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 8.78, 8.96, 8.45, 8.60, 48_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 8.60, 8.86, 8.42, 8.47, 51_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 8.49, 8.52, 8.29, 8.43, 69_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=120), 8.43, 8.66, 8.42, 8.61, 35_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=130), 8.59, 8.61, 8.49, 8.55, 32_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=140), 8.57, 8.78, 8.54, 8.77, 27_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=150), 8.74, 8.80, 8.62, 8.67, 26_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=160), 8.66, 8.70, 8.62, 8.62, 10_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=170), 8.62, 8.96, 8.62, 8.93, 52_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=180), 8.89, 9.15, 8.76, 9.11, 59_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=190), 9.12, 9.39, 9.02, 9.39, 64_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=200), 9.39, 10.66, 9.25, 10.38, 356_000, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_high_base_reclaim_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=10.55,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_high_base_reclaim"
    assert context["size_factor"] == pytest.approx(0.25)


def test_warrior_high_base_reclaim_rejects_second_blowoff_candle() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 7.20, 10.40, 7.10, 9.80, 480_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 9.78, 10.55, 9.40, 9.70, 220_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 9.70, 9.95, 9.30, 9.45, 110_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 9.45, 9.65, 9.10, 9.30, 92_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 9.30, 9.50, 9.05, 9.25, 75_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 9.24, 9.40, 8.92, 9.02, 68_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 9.02, 9.20, 8.74, 8.92, 64_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 8.92, 9.08, 8.62, 8.86, 58_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 8.86, 9.02, 8.58, 8.78, 52_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 8.78, 8.96, 8.45, 8.60, 48_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 8.60, 8.86, 8.42, 8.47, 51_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 8.49, 8.52, 8.29, 8.43, 69_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=120), 8.43, 8.66, 8.42, 8.61, 35_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=130), 8.59, 8.61, 8.49, 8.55, 32_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=140), 8.57, 8.78, 8.54, 8.77, 27_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=150), 8.74, 8.80, 8.62, 8.67, 26_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=160), 8.66, 8.70, 8.62, 8.62, 10_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=170), 8.62, 8.96, 8.62, 8.93, 52_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=180), 8.89, 9.15, 8.76, 9.11, 59_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=190), 9.12, 10.05, 9.02, 9.95, 226_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=200), 9.95, 10.75, 9.73, 10.47, 277_000, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_high_base_reclaim_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=10.55,
    )

    assert context is None


def test_warrior_first_impulse_scalp_allows_clean_nct_style_second_push() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 2.88, 3.25, 2.80, 3.10, 26_588, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 3.21, 3.82, 3.0799, 3.81, 74_968, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 3.84, 4.38, 3.53, 4.02, 96_448, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 4.03, 4.41, 3.96, 4.38, 98_476, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 4.38, 4.41, 4.10, 4.23, 91_192, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 4.19, 4.40, 4.00, 4.36, 67_646, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 4.34, 5.50, 4.26, 5.46, 219_123, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 5.4403, 5.58, 5.09, 5.43, 155_415, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 5.44, 5.90, 5.29, 5.57, 143_194, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_first_impulse_scalp_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=5.58,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_first_impulse_scalp"
    assert context["size_factor"] == pytest.approx(0.20)
    assert context["entry_price_override"] == pytest.approx(5.57)

    lane = warrior_lanes.classify_warrior_trend_lane(
        ten_sec[-1],
        history=ten_sec,
        window_high=5.58,
    )
    dispatched = warrior_lanes.warrior_trend_playbook_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=5.58,
    )

    assert lane == "warrior_first_impulse_scalp"
    assert dispatched is not None
    assert dispatched["entry_trigger"] == "warrior_first_impulse_scalp"


def test_warrior_first_impulse_scalp_rejects_prior_red_breakdown() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 3.95, 4.20, 3.91, 4.05, 12_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 4.10, 4.45, 4.02, 4.32, 18_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 4.82, 4.876, 4.18, 4.34, 41_856, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 4.39, 4.4291, 4.15, 4.154, 25_674, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 4.2513, 4.42, 4.00, 4.41, 31_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 4.47, 4.48, 4.32, 4.40, 32_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 4.43, 5.25, 4.40, 5.20, 72_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 5.194, 5.65, 4.92, 5.4739, 80_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 5.47, 6.57, 5.42, 6.52, 183_000, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_first_impulse_scalp_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=5.65,
    )

    assert context is None


def test_warrior_first_impulse_scalp_rejects_prior_wide_red_fakeout() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 3.24, 3.57, 3.24, 3.536, 14_099, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 3.56, 3.7087, 3.55, 3.70, 6_998, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 3.56, 3.56, 3.54, 3.54, 500, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 3.55, 4.00, 3.51, 4.00, 11_220, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 4.03, 4.29, 3.53, 3.7233, 18_356, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 3.7225, 3.74, 3.52, 3.5395, 14_195, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 3.53, 4.38, 3.51, 4.30, 38_079, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_first_impulse_scalp_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=4.29,
    )

    assert context is None


def test_warrior_playbook_rejects_ehgo_style_under_rejection_high() -> None:
    ten_sec = [
        Bar("EHGO", _BASE_TS + timedelta(seconds=0), 3.73, 4.60, 3.73, 4.15, 146_823, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=10), 4.21, 4.70, 4.00, 4.50, 99_329, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=20), 4.50, 4.80, 4.46, 4.57, 84_675, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=30), 4.56, 4.86, 4.54, 4.68, 86_313, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=40), 4.74, 4.85, 4.58, 4.59, 88_875, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=50), 4.58, 4.59, 4.06, 4.11, 66_406, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=60), 4.11, 4.21, 3.98, 4.16, 63_971, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=70), 4.16, 4.86, 4.14, 4.81, 141_220, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=80), 4.81, 5.10, 4.77, 4.78, 114_421, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=90), 4.77, 4.98, 4.72, 4.80, 77_109, Timeframe.SEC_10),
    ]

    assert warrior_lanes.warrior_recent_rejection_high(ten_sec) == pytest.approx(5.10)
    context = warrior_lanes.warrior_trend_playbook_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=5.10,
    )

    assert context is None


def test_warrior_pullaway_rejects_ehgo_style_under_rejection_high() -> None:
    ten_sec = [
        Bar("EHGO", _BASE_TS + timedelta(seconds=0), 4.50, 4.80, 4.46, 4.57, 84_675, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=10), 4.56, 4.86, 4.54, 4.68, 86_313, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=20), 4.74, 4.85, 4.58, 4.59, 88_875, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=30), 4.58, 4.59, 4.06, 4.11, 66_406, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=40), 4.11, 4.21, 3.98, 4.16, 63_971, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=50), 4.16, 4.86, 4.14, 4.81, 141_220, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=60), 4.81, 5.10, 4.77, 4.78, 114_421, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=70), 4.77, 4.98, 4.72, 4.80, 77_109, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_squeeze_pullaway_context(
        ten_sec[-1],
        {
            "breakout_volume": 100_000,
            "entry_trigger": "momentum_burst",
        },
        history=ten_sec,
        reject_high=4.37,
        rejection_reason="first explosive 10s spike",
        reentry_count=0,
        min_reclaim_price=3.50,
        reward_risk_value=3.0,
        add_reward_risk_value=1.0,
    )

    assert context is None


def test_warrior_pullaway_rejects_wick_through_prior_high_without_close_reclaim() -> None:
    ten_sec = [
        Bar("EHGO", _BASE_TS + timedelta(seconds=0), 4.51, 4.88, 4.46, 4.86, 86_999, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=10), 4.86, 4.90, 4.66, 4.79, 104_505, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=20), 4.80, 4.88, 4.69, 4.70, 51_929, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=30), 4.70, 4.85, 4.64, 4.84, 42_254, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=40), 4.82, 4.90, 4.70, 4.71, 80_899, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=50), 4.71, 4.80, 4.70, 4.76, 26_370, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=60), 4.76, 4.77, 4.65, 4.65, 36_805, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=70), 4.66, 4.66, 4.27, 4.39, 39_615, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=80), 4.43, 4.82, 4.40, 4.82, 41_541, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=90), 4.82, 4.91, 4.69, 4.90, 89_749, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=100), 4.89, 4.92, 4.78, 4.80, 48_565, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=110), 4.80, 4.85, 4.71, 4.84, 30_818, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=120), 4.84, 4.88, 4.69, 4.72, 48_145, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=130), 4.68, 4.83, 4.68, 4.83, 22_726, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=140), 4.83, 4.94, 4.70, 4.78, 71_162, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=150), 4.81, 5.11, 4.81, 5.01, 113_316, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_squeeze_pullaway_context(
        ten_sec[-1],
        {
            "breakout_volume": 100_000,
            "entry_trigger": "momentum_burst",
        },
        history=ten_sec,
        reject_high=4.37,
        rejection_reason="first explosive 10s spike",
        reentry_count=0,
        min_reclaim_price=3.50,
        reward_risk_value=3.0,
        add_reward_risk_value=1.0,
    )

    assert context is None


def test_warrior_playbook_allows_after_reclaiming_rejection_high() -> None:
    ten_sec = [
        Bar("EHGO", _BASE_TS + timedelta(seconds=0), 4.50, 4.80, 4.46, 4.57, 84_675, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=10), 4.56, 4.86, 4.54, 4.68, 86_313, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=20), 4.74, 4.85, 4.58, 4.59, 88_875, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=30), 4.58, 4.59, 4.06, 4.11, 66_406, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=40), 4.11, 4.21, 3.98, 4.16, 63_971, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=50), 4.16, 4.86, 4.14, 4.81, 141_220, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=60), 4.81, 5.10, 4.77, 4.78, 114_421, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=70), 4.78, 5.12, 4.76, 4.97, 80_000, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=80), 4.95, 5.18, 4.92, 5.14, 95_000, Timeframe.SEC_10),
    ]

    assert warrior_lanes.warrior_recent_rejection_high(ten_sec) == 0.0


def test_warrior_first_pullback_reclaim_allows_labt_style_low_price_base() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 2.145, 2.19, 2.05, 2.18, 129_531, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 2.1796, 2.2499, 2.13, 2.1896, 165_279, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 2.1879, 2.71, 2.18, 2.60, 296_442, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 2.61, 2.64, 2.40, 2.45, 289_954, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 2.46, 2.5599, 2.42, 2.4405, 212_667, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 2.45, 2.54, 2.42, 2.44, 151_810, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 2.45, 2.50, 2.43, 2.4513, 85_321, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 2.46, 2.51, 2.36, 2.40, 205_073, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 2.40, 2.64, 2.3805, 2.57, 251_486, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 2.56, 2.93, 2.56, 2.91, 483_408, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_trend_playbook_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=2.93,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_first_pullback_reclaim"
    assert context["variant_override"] == "warrior_first_pullback_reclaim"
    assert context["size_factor"] == pytest.approx(0.20)
    assert context["entry_price_override"] <= context["max_pay"]
    assert context["stop_price_override"] < context["entry_price_override"]
    assert context["target_price_override"] > context["entry_price_override"]
    assert context["pullback_pct"] >= 4.5


def test_warrior_first_pullback_reclaim_rejects_red_distribution_base() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 1.55, 1.62, 1.52, 1.60, 15_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 1.60, 1.78, 1.58, 1.74, 35_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 1.74, 2.12, 1.70, 2.02, 110_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 2.03, 2.38, 1.95, 2.10, 260_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 2.14, 2.30, 1.86, 1.91, 240_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 1.92, 2.09, 1.88, 1.98, 90_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 1.99, 2.08, 1.96, 2.05, 85_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 2.05, 2.18, 2.00, 2.15, 95_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 2.14, 2.28, 2.07, 2.25, 125_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 2.25, 2.42, 2.18, 2.39, 165_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 2.39, 2.58, 2.34, 2.52, 220_000, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_trend_playbook_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=2.42,
    )

    assert context is None


def test_warrior_first_pullback_reclaim_rejects_codx_style_cheap_violent_tape() -> None:
    ten_sec = [
        Bar("CODX", _BASE_TS + timedelta(seconds=0), 1.84, 1.85, 1.77, 1.8497, 173_721, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=10), 1.83, 1.99, 1.83, 1.9196, 305_468, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=20), 1.92, 2.00, 1.90, 1.9696, 203_534, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=30), 1.97, 2.0999, 1.97, 2.0087, 243_320, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=40), 2.00, 2.01, 1.94, 1.97, 156_445, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=50), 1.97, 2.11, 1.97, 2.08, 283_333, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=60), 2.075, 2.19, 2.03, 2.14, 444_299, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=70), 2.14, 2.25, 2.12, 2.18, 277_712, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=80), 2.18, 2.19, 2.10, 2.11, 197_227, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=90), 2.1097, 2.1801, 2.05, 2.1596, 275_898, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=100), 2.16, 2.26, 2.14, 2.22, 193_633, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=110), 2.22, 2.2204, 2.15, 2.1774, 175_846, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=120), 2.17, 2.31, 2.1604, 2.18, 275_026, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=130), 2.17, 2.20, 2.08, 2.1495, 247_692, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=140), 2.1497, 2.25, 2.12, 2.19, 174_048, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=150), 2.1899, 2.21, 2.16, 2.17, 71_515, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=160), 2.1772, 2.24, 2.17, 2.2196, 148_437, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=170), 2.20, 2.25, 2.20, 2.2401, 139_454, Timeframe.SEC_10),
        Bar("CODX", _BASE_TS + timedelta(seconds=180), 2.24, 2.41, 2.24, 2.35, 395_028, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_trend_playbook_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=2.44,
    )

    assert context is None


def test_warrior_first_pullback_reclaim_rejects_stair_up_without_pullback() -> None:
    ten_sec = [
        Bar("SBFM", _BASE_TS + timedelta(seconds=0), 1.9998, 1.9999, 1.97, 1.98, 189_000, Timeframe.SEC_10),
        Bar("SBFM", _BASE_TS + timedelta(seconds=10), 1.9703, 2.09, 1.97, 2.07, 560_761, Timeframe.SEC_10),
        Bar("SBFM", _BASE_TS + timedelta(seconds=20), 2.07, 2.19, 2.05, 2.1686, 824_780, Timeframe.SEC_10),
        Bar("SBFM", _BASE_TS + timedelta(seconds=30), 2.17, 2.22, 2.16, 2.19, 648_427, Timeframe.SEC_10),
        Bar("SBFM", _BASE_TS + timedelta(seconds=40), 2.1804, 2.26, 2.17, 2.25, 633_547, Timeframe.SEC_10),
        Bar("SBFM", _BASE_TS + timedelta(seconds=50), 2.2498, 2.34, 2.23, 2.32, 580_510, Timeframe.SEC_10),
        Bar("SBFM", _BASE_TS + timedelta(seconds=60), 2.3299, 2.37, 2.30, 2.3501, 721_054, Timeframe.SEC_10),
        Bar("SBFM", _BASE_TS + timedelta(seconds=70), 2.3598, 2.40, 2.34, 2.38, 548_203, Timeframe.SEC_10),
        Bar("SBFM", _BASE_TS + timedelta(seconds=80), 2.37, 2.45, 2.37, 2.4315, 709_329, Timeframe.SEC_10),
        Bar("SBFM", _BASE_TS + timedelta(seconds=90), 2.43, 2.55, 2.42, 2.53, 547_063, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_trend_playbook_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=2.55,
    )

    assert context is None


def test_warrior_first_pullback_reclaim_ignores_clwt_style_cheap_first_leg() -> None:
    ten_sec = [
        Bar("CLWT", _BASE_TS + timedelta(seconds=0), 1.05, 1.12, 1.02, 1.10, 18_000, Timeframe.SEC_10),
        Bar("CLWT", _BASE_TS + timedelta(seconds=10), 1.10, 1.35, 1.08, 1.30, 42_000, Timeframe.SEC_10),
        Bar("CLWT", _BASE_TS + timedelta(seconds=20), 1.30, 1.62, 1.25, 1.55, 125_000, Timeframe.SEC_10),
        Bar("CLWT", _BASE_TS + timedelta(seconds=30), 1.55, 1.95, 1.45, 1.70, 260_000, Timeframe.SEC_10),
        Bar("CLWT", _BASE_TS + timedelta(seconds=40), 1.68, 1.78, 1.52, 1.56, 95_000, Timeframe.SEC_10),
        Bar("CLWT", _BASE_TS + timedelta(seconds=50), 1.56, 1.66, 1.50, 1.58, 82_000, Timeframe.SEC_10),
        Bar("CLWT", _BASE_TS + timedelta(seconds=60), 1.57, 1.68, 1.54, 1.64, 78_000, Timeframe.SEC_10),
        Bar("CLWT", _BASE_TS + timedelta(seconds=70), 1.63, 1.73, 1.59, 1.70, 92_000, Timeframe.SEC_10),
        Bar("CLWT", _BASE_TS + timedelta(seconds=80), 1.70, 1.78, 1.64, 1.74, 105_000, Timeframe.SEC_10),
        Bar("CLWT", _BASE_TS + timedelta(seconds=90), 1.73, 1.84, 1.69, 1.80, 130_000, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_trend_playbook_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=1.95,
    )

    assert context is None


def test_backtest_timed_release_rejects_late_vwap_pullback_from_base() -> None:
    driver = PipelineBacktestDriver(
        {"VWAP": []},
        use_execution_timer=True,
    )
    signal = TradeSignal(
        symbol="VWAP",
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=4.08,
        stop_loss=3.88,
        reason="VWAP Pullback",
        scan_result=ScanResult(
            symbol="VWAP",
            scanner_name="vwap_pullback",
            ts=_BASE_TS,
            score=90.0,
            criteria={
                "pattern": "vwap_pullback",
                "close": 4.08,
                "queued_entry_price": 4.08,
                "vwap": 4.00,
                "base_high": 4.05,
                "pullback_low": 3.90,
                "stop_price": 3.88,
            },
        ),
    )

    reason = driver._timed_release_chase_reject(
        signal,
        Bar("VWAP", _BASE_TS, 4.16, 4.20, 4.14, 4.18, 20_000, Timeframe.SEC_10),
    )

    assert reason == "late VWAP pullback release: $4.1800 too extended from base $4.0500 (max 2.5%)"


def test_warrior_reject_history_blocks_weak_normal_fallback() -> None:
    driver = PipelineBacktestDriver(
        {"UBXG": []},
        use_warrior_squeeze_playbook=True,
    )
    result = PipelineBacktestResult()
    for idx in range(3):
        row = {
            "ts": (_BASE_TS + timedelta(seconds=idx)).isoformat(),
            "symbol": "UBXG",
            "blocked_layer": "warrior_squeeze_playbook_unconfirmed",
            "reason": "warrior setup not confirmed by playbook pattern",
        }
        driver._append_rejection(result, row)
        driver._track_warrior_normal_fallback_state(row)

    weak = TradeSignal(
        symbol="UBXG",
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=8.50,
        stop_loss=8.00,
        scan_result=ScanResult(
            symbol="UBXG",
            scanner_name="pullback_base",
            ts=_BASE_TS,
            score=78.0,
            criteria={"pattern": "pullback_base", "setup_tier": "A+ setup"},
        ),
    )
    elite = TradeSignal(
        symbol="UBXG",
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=8.50,
        stop_loss=8.00,
        scan_result=ScanResult(
            symbol="UBXG",
            scanner_name="pullback_base",
            ts=_BASE_TS,
            score=94.0,
            criteria={"pattern": "pullback_base", "setup_tier": "A+ setup"},
        ),
    )

    assert "Warrior watched UBXG" in str(driver._warrior_normal_fallback_reject(weak))
    assert driver._warrior_normal_fallback_reject(elite) is None


def test_backtest_rejection_sink_does_not_overcount_warrior_fallback_memory() -> None:
    driver = PipelineBacktestDriver(
        {"UBXG": []},
        use_warrior_squeeze_playbook=True,
    )
    result = PipelineBacktestResult()
    for idx in range(3):
        driver._append_rejection(result, {
            "ts": (_BASE_TS + timedelta(seconds=idx)).isoformat(),
            "symbol": "UBXG",
            "blocked_layer": "warrior_squeeze_playbook_time_window",
            "reason": "warrior window expired",
        })

    weak = TradeSignal(
        symbol="UBXG",
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=8.50,
        stop_loss=8.00,
        scan_result=ScanResult(
            symbol="UBXG",
            scanner_name="pullback_base",
            ts=_BASE_TS,
            score=78.0,
            criteria={"pattern": "pullback_base", "setup_tier": "A+ setup"},
        ),
    )

    assert driver._warrior_normal_fallback_reject(weak) is None


def test_backtest_remembers_recent_normal_bad_tape_for_warrior_wait() -> None:
    driver = PipelineBacktestDriver(
        {"EHGO": []},
        use_warrior_squeeze_playbook=True,
    )
    result = PipelineBacktestResult()
    driver._append_rejection(result, {
        "ts": _BASE_TS.isoformat(),
        "symbol": "EHGO",
        "blocked_layer": "verifier",
        "reason": "tape shows selling pressure (imbalance=-0.56, need >-0.3)",
    })

    reason = driver._warrior_recent_bad_tape_reject("EHGO", _BASE_TS + timedelta(seconds=30))

    assert reason is not None
    assert "selling pressure" in reason
    assert driver._warrior_recent_bad_tape_reject("EHGO", _BASE_TS + timedelta(seconds=90)) is None


def test_backtest_does_not_treat_unreclaimed_level_watch_as_bad_tape() -> None:
    driver = PipelineBacktestDriver(
        {"NXTS": []},
        use_warrior_squeeze_playbook=True,
    )
    result = PipelineBacktestResult()
    driver._append_rejection(result, {
        "ts": _BASE_TS.isoformat(),
        "symbol": "NXTS",
        "blocked_layer": "verifier",
        "reason": "watch only: level breakout has not reclaimed with a clean close",
    })

    reason = driver._warrior_recent_bad_tape_reject("NXTS", _BASE_TS + timedelta(seconds=30))

    assert reason is None


def test_backtest_rebase_preserves_named_warrior_lane_after_bad_tape() -> None:
    driver = PipelineBacktestDriver(
        {"EHGO": []},
        use_warrior_squeeze_playbook=True,
    )
    driver._momentum_burst_continuation_base_ok = lambda symbol, now: (  # type: ignore[method-assign]
        True,
        "fresh continuation base",
        {"base_high": 5.20, "base_low": 5.00},
    )
    pending = {
        "ts": _BASE_TS,
        "breakout_close": 5.10,
        "breakout_high": 5.25,
        "breakout_volume": 100_000,
        "entry_trigger": "warrior_smooth_10s_pullback_continuation",
        "variant_override": "warrior_smooth_10s_pullback_continuation",
    }
    ten_sec = Bar("EHGO", _BASE_TS + timedelta(seconds=10), 5.18, 5.32, 5.12, 5.30, 120_000, Timeframe.SEC_10)

    assert driver._momentum_burst_rebase_pending_after_reject(
        "EHGO",
        ten_sec,
        pending,
        "volume too light after recent normal entry reject: tape shows selling pressure",
        _BASE_TS + timedelta(seconds=10),
        hit_run=True,
    )
    rebased = driver._momentum_burst_pending["EHGO"]
    assert rebased["entry_trigger"] == "warrior_smooth_10s_pullback_continuation"
    assert rebased["variant_override"] == "warrior_smooth_10s_pullback_continuation"


def test_normal_fallback_third_attempt_needs_elite_score() -> None:
    driver = PipelineBacktestDriver({"CODX": []})
    driver._pipeline._symbol_entry_counts["CODX"] = 2

    weak = TradeSignal(
        symbol="CODX",
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=2.50,
        stop_loss=2.25,
        scan_result=ScanResult(
            symbol="CODX",
            scanner_name="level_breakout_reclaim",
            ts=_BASE_TS,
            score=81.0,
            criteria={"pattern": "level_breakout_reclaim", "setup_tier": "A+ setup"},
        ),
    )
    elite = TradeSignal(
        symbol="CODX",
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=2.50,
        stop_loss=2.25,
        scan_result=ScanResult(
            symbol="CODX",
            scanner_name="abc_continuation",
            ts=_BASE_TS,
            score=95.0,
            criteria={"pattern": "abc_continuation", "setup_tier": "A+ setup"},
        ),
    )

    assert "normal level_breakout_reclaim overtrade" in str(
        driver._normal_fallback_overtrade_reject(weak)
    )
    assert driver._normal_fallback_overtrade_reject(elite) is None


def test_warrior_stair_step_runner_context_allows_sti_style_eight_base() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 6.95, 7.08, 6.80, 7.0778, 39_121, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 7.17, 7.39, 7.11, 7.2868, 38_786, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 7.28, 7.65, 7.26, 7.50, 64_857, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 7.45, 7.85, 7.45, 7.78, 46_763, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 7.76, 8.13, 7.73, 7.98, 95_488, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 7.98, 8.05, 7.84, 7.89, 63_975, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 7.86, 8.02, 7.70, 7.9855, 51_719, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 7.98, 8.03, 7.8303, 7.88, 43_498, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 7.87, 8.07, 7.82, 8.04, 33_935, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 8.07, 8.23, 7.96, 7.98, 62_624, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 8.16, 8.39, 8.06, 8.23, 67_057, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 8.26, 8.35, 8.18, 8.27, 39_033, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=120), 8.00, 8.27, 7.90, 8.23, 47_045, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=130), 8.23, 8.34, 8.21, 8.26, 30_874, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=140), 8.10, 8.244, 8.0686, 8.211, 20_979, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=150), 8.08, 8.31, 8.07, 8.24, 31_315, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=160), 8.21, 8.45, 8.15, 8.38, 63_354, Timeframe.SEC_10),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True

    context = driver._warrior_trend_pullback_reclaim_context(
        "MBUR",
        ten_sec[-1],
        window_high=8.39,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_stair_step_runner"
    assert context["variant_override"] == "warrior_stair_step_runner"
    assert context["entry_price_override"] <= context["max_pay"]
    assert context["stop_price_override"] < context["entry_price_override"]
    assert context["target_price_override"] > context["entry_price_override"]
    assert (context["entry_price_override"] - context["stop_price_override"]) / context["entry_price_override"] <= 0.06

    lane = warrior_lanes.classify_warrior_trend_lane(
        ten_sec[-1],
        history=ten_sec,
        window_high=8.39,
    )
    assert lane == "warrior_stair_step_runner"


def test_warrior_smooth_10s_pullback_continuation_allows_wnw_style_reclaim() -> None:
    ten_sec = [
        Bar("WNW", _BASE_TS + timedelta(seconds=0), 2.99, 3.76, 2.94, 3.58, 25_367, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=10), 3.50, 3.58, 3.38, 3.58, 24_243, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=20), 3.57, 3.99, 3.45, 3.92, 24_168, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=30), 3.88, 4.30, 3.82, 4.22, 30_110, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=40), 4.25, 4.76, 4.16, 4.52, 40_560, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=50), 4.51, 4.71, 4.40, 4.49, 40_239, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=60), 4.54, 4.90, 4.49, 4.81, 38_729, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=70), 4.80, 5.28, 4.78, 5.26, 40_680, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=80), 5.26, 6.94, 5.23, 6.88, 57_582, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=90), 6.88, 7.00, 5.51, 5.73, 60_102, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=100), 5.72, 5.86, 5.25, 5.29, 40_244, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=110), 5.25, 5.68, 5.00, 5.62, 41_437, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=120), 5.62, 6.41, 5.57, 5.78, 33_596, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=130), 5.91, 6.13, 5.69, 5.70, 30_621, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=140), 5.70, 5.70, 5.33, 5.51, 31_433, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=150), 5.51, 5.86, 5.51, 5.71, 22_168, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=160), 5.71, 5.86, 5.52, 5.63, 20_968, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=170), 5.63, 5.79, 5.50, 5.79, 14_118, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=180), 5.74, 6.10, 5.73, 6.04, 32_296, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=190), 6.04, 6.17, 5.85, 5.85, 26_455, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=200), 5.86, 6.17, 5.67, 6.09, 22_684, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=210), 6.09, 6.44, 5.86, 6.35, 38_743, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_trend_playbook_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=7.00,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_smooth_10s_pullback_continuation"
    assert context["variant_override"] == "warrior_smooth_10s_pullback_continuation"
    assert context["size_factor"] == pytest.approx(0.30)
    assert context["entry_price_override"] <= context["max_pay"]
    assert context["stop_price_override"] < context["entry_price_override"]
    assert context["target_price_override"] > context["entry_price_override"]
    assert 8.0 <= context["pullback_pct"] <= 34.0

    lane = warrior_lanes.classify_warrior_trend_lane(
        ten_sec[-1],
        history=ten_sec,
        window_high=7.00,
    )
    assert lane == "warrior_smooth_10s_pullback_continuation"


def test_warrior_smooth_10s_pullback_continuation_rejects_first_spike() -> None:
    ten_sec = [
        Bar("WNW", _BASE_TS + timedelta(seconds=0), 2.99, 3.76, 2.94, 3.58, 25_367, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=10), 3.50, 3.58, 3.38, 3.58, 24_243, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=20), 3.57, 3.99, 3.45, 3.92, 24_168, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=30), 3.88, 4.30, 3.82, 4.22, 30_110, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=40), 4.25, 4.76, 4.16, 4.52, 40_560, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=50), 4.51, 4.71, 4.40, 4.49, 40_239, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=60), 4.54, 4.90, 4.49, 4.81, 38_729, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=70), 4.80, 5.28, 4.78, 5.26, 40_680, Timeframe.SEC_10),
        Bar("WNW", _BASE_TS + timedelta(seconds=80), 5.26, 6.94, 5.23, 6.88, 57_582, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_smooth_10s_pullback_continuation_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=6.94,
    )

    assert context is None


def test_warrior_failed_spike_vwap_reclaim_allows_nxts_style_reclaim() -> None:
    ten_sec = [
        Bar("NXTS", _BASE_TS + timedelta(seconds=0), 4.80, 5.10, 4.70, 5.00, 30_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=10), 5.00, 6.80, 4.95, 6.60, 70_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=20), 6.60, 8.90, 6.50, 8.40, 120_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=30), 8.35, 8.55, 7.10, 7.40, 100_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=40), 7.40, 7.70, 6.80, 7.00, 80_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=50), 7.05, 7.80, 7.00, 7.60, 60_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=60), 7.70, 8.20, 7.50, 8.10, 45_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=70), 8.10, 8.40, 7.90, 8.25, 50_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=80), 8.20, 8.35, 7.95, 8.05, 35_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=90), 8.05, 8.30, 7.90, 8.20, 32_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=100), 8.20, 8.45, 8.05, 8.36, 45_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=110), 8.35, 8.50, 8.10, 8.20, 40_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=120), 8.20, 8.55, 8.15, 8.42, 50_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=130), 8.40, 8.58, 8.25, 8.50, 55_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=140), 8.50, 8.62, 8.30, 8.56, 60_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=150), 8.62, 9.15, 8.55, 9.02, 160_000, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_trend_playbook_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=8.90,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_failed_spike_vwap_reclaim"
    assert context["variant_override"] == "warrior_failed_spike_vwap_reclaim"
    assert context["size_factor"] == pytest.approx(0.25)
    assert context["failed_spike_high"] == pytest.approx(8.90)
    assert context["entry_price_override"] <= context["max_pay"]
    assert context["stop_price_override"] < context["entry_price_override"]
    assert context["target_price_override"] > context["entry_price_override"]


def test_warrior_failed_spike_vwap_reclaim_rejects_wide_unstable_reclaim() -> None:
    ten_sec = [
        Bar("CUPR", _BASE_TS + timedelta(seconds=0), 4.80, 5.10, 4.70, 5.00, 30_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=10), 5.00, 6.80, 4.95, 6.60, 70_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=20), 6.60, 8.90, 6.50, 8.40, 120_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=30), 8.35, 8.55, 7.10, 7.40, 100_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=40), 7.40, 7.70, 6.80, 7.00, 80_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=50), 7.05, 7.80, 7.00, 7.60, 60_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=60), 7.70, 8.20, 7.50, 8.10, 45_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=70), 8.10, 8.40, 7.90, 8.25, 50_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=80), 8.20, 8.35, 7.95, 8.05, 35_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=90), 8.05, 8.30, 7.90, 8.20, 32_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=100), 8.20, 8.45, 8.05, 8.36, 45_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=110), 8.35, 8.50, 8.10, 8.20, 40_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=120), 8.20, 8.55, 8.15, 8.42, 50_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=130), 8.40, 8.58, 8.25, 8.50, 55_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=140), 8.50, 8.62, 8.30, 8.56, 60_000, Timeframe.SEC_10),
        Bar("CUPR", _BASE_TS + timedelta(seconds=150), 8.62, 9.40, 8.12, 9.04, 275_000, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_failed_spike_vwap_reclaim_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=8.90,
    )

    assert context is None


def test_warrior_backtest_arms_failed_spike_vwap_reclaim_below_old_high() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 4.80, 5.10, 4.70, 5.00, 30_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 5.00, 6.80, 4.95, 6.60, 70_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 6.60, 8.90, 6.50, 8.40, 120_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 8.35, 8.55, 7.10, 7.40, 100_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 7.40, 7.70, 6.80, 7.00, 80_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 7.05, 7.80, 7.00, 7.60, 60_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 7.70, 8.20, 7.50, 8.10, 45_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 8.10, 8.40, 7.90, 8.25, 50_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 8.20, 8.35, 7.95, 8.05, 35_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 8.05, 8.30, 7.90, 8.20, 32_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 8.20, 8.45, 8.05, 8.36, 45_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 8.35, 8.50, 8.10, 8.20, 40_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=120), 8.20, 8.55, 8.15, 8.42, 50_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=130), 8.40, 8.58, 8.25, 8.50, 55_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=140), 8.50, 8.62, 8.30, 8.56, 60_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=150), 8.62, 9.15, 8.55, 9.02, 160_000, Timeframe.SEC_10),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_rejection_high["MBUR"] = 8.90

    driver._maybe_arm_warrior_squeeze_from_10s("MBUR", ten_sec[-1], ten_sec[-1].ts)

    pending = driver._momentum_burst_pending["MBUR"]
    assert pending["entry_trigger"] == "warrior_failed_spike_vwap_reclaim"
    assert pending["breakout_high"] == ten_sec[-1].high
    assert driver._momentum_burst_window_high["MBUR"] == pytest.approx(ten_sec[-1].high)


def test_warrior_failed_spike_vwap_reclaim_rejects_weak_reclaim_volume() -> None:
    ten_sec = [
        Bar("NXTS", _BASE_TS + timedelta(seconds=0), 4.80, 5.10, 4.70, 5.00, 30_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=10), 5.00, 6.80, 4.95, 6.60, 70_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=20), 6.60, 8.90, 6.50, 8.40, 120_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=30), 8.35, 8.55, 7.10, 7.40, 100_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=40), 7.40, 7.70, 6.80, 7.00, 80_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=50), 7.05, 7.80, 7.00, 7.60, 60_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=60), 7.70, 8.20, 7.50, 8.10, 45_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=70), 8.10, 8.40, 7.90, 8.25, 50_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=80), 8.20, 8.35, 7.95, 8.05, 35_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=90), 8.05, 8.30, 7.90, 8.20, 32_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=100), 8.20, 8.45, 8.05, 8.36, 45_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=110), 8.35, 8.50, 8.10, 8.20, 40_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=120), 8.20, 8.55, 8.15, 8.42, 20_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=130), 8.40, 8.58, 8.25, 8.50, 18_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=140), 8.50, 8.62, 8.30, 8.56, 19_000, Timeframe.SEC_10),
        Bar("NXTS", _BASE_TS + timedelta(seconds=150), 8.62, 9.15, 8.55, 9.02, 24_000, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_failed_spike_vwap_reclaim_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=8.90,
    )

    assert context is None


def test_warrior_smooth_hod_reclaim_allows_bjdx_style_grinder() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 4.82, 5.18, 4.80, 5.1025, 100_470, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 5.11, 5.29, 5.06, 5.22, 127_415, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 5.23, 5.35, 5.16, 5.17, 143_424, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 5.167, 5.19, 5.04, 5.10, 69_406, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 5.0975, 5.16, 5.05, 5.083, 47_884, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 5.08, 5.22, 5.0503, 5.21, 49_502, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 5.2005, 5.26, 5.12, 5.17, 55_193, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 5.1646, 5.40, 5.16, 5.30, 84_325, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 5.30, 5.37, 5.19, 5.23, 72_382, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 5.22, 5.33, 5.19, 5.306, 40_744, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 5.29, 5.35, 5.20, 5.2503, 62_404, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 5.2505, 5.3499, 5.22, 5.2603, 31_375, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=120), 5.2627, 5.30, 5.25, 5.30, 26_503, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=130), 5.26, 5.26, 4.97, 5.05, 103_363, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=140), 5.03, 5.24, 5.02, 5.1903, 45_316, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=150), 5.2, 5.2312, 5.09, 5.10, 28_650, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=160), 5.103, 5.14, 5.05, 5.05, 25_617, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=170), 5.067, 5.25, 5.0419, 5.1753, 40_811, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=180), 5.1759, 5.22, 5.13, 5.15, 30_030, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=190), 5.15, 5.1984, 5.13, 5.17, 15_964, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=200), 5.17, 5.26, 5.1416, 5.2042, 41_108, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=210), 5.21, 5.23, 5.16, 5.2013, 13_686, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=220), 5.2016, 5.267, 5.1618, 5.2467, 31_801, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=230), 5.23, 5.25, 5.1456, 5.151, 23_315, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=240), 5.17, 5.32, 5.1523, 5.20, 57_041, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=250), 5.2268, 5.29, 5.1701, 5.2779, 24_583, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=260), 5.27, 5.30, 5.20, 5.29, 36_932, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=270), 5.30, 5.4899, 5.2432, 5.38, 80_626, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=280), 5.39, 5.6222, 5.36, 5.52, 135_804, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=290), 5.5384, 5.78, 5.50, 5.72, 162_354, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_smooth_hod_reclaim_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=5.62,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_smooth_hod_reclaim"
    assert context["variant_override"] == "warrior_smooth_hod_reclaim"
    assert context["size_factor"] == pytest.approx(0.35)
    assert context["entry_price_override"] <= context["max_pay"]
    assert context["stop_price_override"] < context["entry_price_override"]
    assert context["target_price_override"] > context["entry_price_override"]

    lane = warrior_lanes.classify_warrior_trend_lane(
        ten_sec[-1],
        history=ten_sec,
        window_high=5.62,
    )
    dispatched = warrior_lanes.warrior_trend_playbook_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=5.62,
    )

    assert lane == "warrior_smooth_hod_reclaim"
    assert dispatched is not None
    assert dispatched["entry_trigger"] == "warrior_smooth_hod_reclaim"


def test_warrior_smooth_hod_reclaim_rejects_wick_through_hod_without_close() -> None:
    ten_sec = [
        Bar("EHGO", _BASE_TS + timedelta(seconds=0), 4.47, 4.79, 4.44, 4.57, 88_583, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=10), 4.57, 4.83, 4.56, 4.78, 73_172, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=20), 4.79, 4.80, 4.56, 4.69, 66_484, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=30), 4.68, 4.70, 4.56, 4.58, 40_885, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=40), 4.58, 4.58, 4.45, 4.51, 33_045, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=50), 4.51, 4.88, 4.46, 4.86, 86_999, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=60), 4.86, 4.90, 4.66, 4.79, 104_505, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=70), 4.80, 4.88, 4.69, 4.70, 51_929, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=80), 4.70, 4.85, 4.64, 4.84, 42_254, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=90), 4.82, 4.90, 4.70, 4.71, 80_899, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=100), 4.71, 4.80, 4.70, 4.76, 26_370, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=110), 4.76, 4.77, 4.65, 4.65, 36_805, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=120), 4.66, 4.66, 4.27, 4.39, 39_615, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=130), 4.43, 4.82, 4.40, 4.82, 41_541, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=140), 4.82, 4.91, 4.69, 4.90, 89_749, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=150), 4.89, 4.92, 4.78, 4.80, 48_565, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=160), 4.80, 4.85, 4.71, 4.84, 30_818, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=170), 4.84, 4.88, 4.69, 4.72, 48_145, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=180), 4.68, 4.83, 4.68, 4.83, 22_726, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=190), 4.83, 4.94, 4.70, 4.78, 71_162, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=200), 4.81, 5.11, 4.81, 5.01, 113_316, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_smooth_hod_reclaim_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=5.10,
    )

    assert context is None


def test_warrior_smooth_hod_reclaim_rejects_flat_top_without_volume_expansion() -> None:
    ten_sec = [
        Bar("EHGO", _BASE_TS + timedelta(seconds=0), 4.02, 4.15, 4.00, 4.14, 13_481, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=10), 4.15, 4.16, 4.06, 4.09, 23_432, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=20), 4.09, 4.23, 4.06, 4.21, 22_290, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=30), 4.24, 4.38, 4.21, 4.32, 47_456, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=40), 4.35, 4.45, 4.29, 4.42, 42_246, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=50), 4.45, 4.46, 4.40, 4.41, 38_724, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=60), 4.41, 4.97, 4.40, 4.95, 141_998, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=70), 4.97, 5.10, 4.90, 4.92, 160_503, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=80), 4.92, 4.95, 4.80, 4.94, 87_767, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=90), 4.95, 4.95, 4.86, 4.86, 52_206, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=100), 4.87, 5.19, 4.84, 5.11, 93_883, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=110), 5.10, 5.25, 5.09, 5.10, 137_820, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=120), 5.09, 5.20, 5.09, 5.10, 92_838, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=130), 5.09, 5.15, 5.09, 5.10, 62_845, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=140), 5.10, 5.20, 5.09, 5.11, 61_092, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=150), 5.13, 5.13, 5.09, 5.09, 46_258, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=160), 5.09, 5.30, 5.09, 5.24, 94_420, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=170), 5.24, 5.26, 5.19, 5.19, 48_276, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=180), 5.19, 5.21, 5.14, 5.19, 77_251, Timeframe.SEC_10),
        Bar("EHGO", _BASE_TS + timedelta(seconds=190), 5.20, 5.30, 5.19, 5.28, 79_339, Timeframe.SEC_10),
    ]

    context = warrior_lanes.warrior_smooth_hod_reclaim_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=5.30,
    )

    assert context is None


def test_warrior_late_reentry_rejects_choppy_stair_step_after_win() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 13.80, 14.20, 13.60, 14.05, 90_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 14.05, 14.70, 13.88, 14.48, 110_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 14.49, 14.76, 13.89, 13.90, 120_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 13.92, 14.62, 13.86, 14.50, 95_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 14.50, 15.02, 14.19, 14.31, 130_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 14.31, 14.95, 14.23, 14.89, 81_083, Timeframe.SEC_10),
    ]

    reason = warrior_lanes.warrior_late_reentry_reject(
        ten_sec[-1],
        history=ten_sec,
        window_high=14.95,
        reentry_count=2,
        target_wins=1,
        entry_trigger="warrior_stair_step_runner",
    )

    assert reason is not None
    assert "late Warrior re-entry blocked" in reason


def test_warrior_post_target_pullback_reclaim_allows_rubi_style_reclaim() -> None:
    prices = [
        (4.07, 4.64, 4.05, 4.59, 216_000),
        (4.60, 4.69, 4.45, 4.59, 235_000),
        (4.58, 4.66, 4.51, 4.52, 145_000),
        (4.51, 4.54, 4.20, 4.42, 171_000),
        (4.41, 4.59, 4.34, 4.58, 81_000),
        (4.57, 4.85, 4.56, 4.84, 236_000),
        (4.84, 4.98, 4.71, 4.71, 290_000),
        (4.72, 4.78, 4.41, 4.46, 183_000),
        (4.46, 4.50, 4.36, 4.49, 96_000),
        (4.49, 4.50, 4.45, 4.45, 44_000),
        (4.46, 4.51, 4.44, 4.45, 58_000),
        (4.45, 4.45, 4.24, 4.30, 90_000),
        (4.31, 4.50, 4.29, 4.42, 118_000),
        (4.41, 4.62, 4.39, 4.54, 78_000),
        (4.52, 4.62, 4.52, 4.56, 71_000),
        (4.54, 4.60, 4.52, 4.58, 59_000),
        (4.56, 4.58, 4.49, 4.50, 46_000),
        (4.49, 4.50, 4.42, 4.50, 48_000),
        (4.50, 4.70, 4.49, 4.58, 102_000),
        (4.58, 4.73, 4.57, 4.65, 86_000),
        (4.65, 4.70, 4.62, 4.63, 63_000),
        (4.64, 4.68, 4.59, 4.63, 59_000),
        (4.64, 4.64, 4.48, 4.50, 37_000),
        (4.49, 4.71, 4.40, 4.7075, 86_000),
    ]
    ten_sec = [
        Bar(
            "RUBI",
            _BASE_TS + timedelta(seconds=i * 10),
            open_,
            high,
            low,
            close,
            volume,
            Timeframe.SEC_10,
        )
        for i, (open_, high, low, close, volume) in enumerate(prices)
    ]

    context = warrior_lanes.warrior_post_target_pullback_reclaim_context(
        ten_sec[-1],
        history=ten_sec,
        window_high=4.89,
    )

    assert context is not None
    assert context["entry_trigger"] == "warrior_post_target_pullback_reclaim"
    assert context["size_factor"] == pytest.approx(0.25)
    assert context["entry_price_override"] == pytest.approx(4.7075)
    assert context["stop_price_override"] < context["entry_price_override"]
    assert context["target_price_override"] > context["entry_price_override"]


def test_warrior_late_reentry_leaves_non_stair_lanes_alone() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 7.00, 7.40, 6.90, 7.35, 120_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 7.35, 7.80, 7.20, 7.60, 130_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 7.60, 8.10, 7.30, 7.50, 150_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 7.50, 7.90, 7.10, 7.20, 160_000, Timeframe.SEC_10),
    ]

    reason = warrior_lanes.warrior_late_reentry_reject(
        ten_sec[-1],
        history=ten_sec,
        window_high=8.10,
        reentry_count=2,
        target_wins=1,
        entry_trigger="warrior_prior_runner_continuation_pullback",
    )

    assert reason is None


def test_warrior_violent_liquid_blocks_unproven_trend_reclaim() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 9.10, 9.72, 9.00, 9.55, 160_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 9.56, 10.10, 9.40, 9.82, 180_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 9.82, 10.26, 9.35, 9.48, 220_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 9.48, 9.92, 9.10, 9.30, 170_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 9.30, 10.05, 9.20, 9.88, 190_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 9.88, 10.30, 9.62, 10.20, 210_000, Timeframe.SEC_10),
    ]

    reason = warrior_lanes.warrior_violent_liquid_reject(
        ten_sec[-1],
        history=ten_sec,
        target_wins=0,
        entry_trigger="warrior_trend_pullback_reclaim",
    )

    assert reason is not None
    assert "violent-liquid Warrior blocked" in reason

    stair_reason = warrior_lanes.warrior_violent_liquid_reject(
        ten_sec[-1],
        history=ten_sec,
        target_wins=0,
        entry_trigger="warrior_stair_step_runner",
    )

    assert stair_reason is not None
    assert "violent-liquid Warrior blocked" in stair_reason


def test_warrior_violent_liquid_allows_after_target_win() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 9.10, 9.72, 9.00, 9.55, 160_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 9.56, 10.10, 9.40, 9.82, 180_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 9.82, 10.26, 9.35, 9.48, 220_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 9.48, 9.92, 9.10, 9.30, 170_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 9.30, 10.05, 9.20, 9.88, 190_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 9.88, 10.30, 9.62, 10.20, 210_000, Timeframe.SEC_10),
    ]

    reason = warrior_lanes.warrior_violent_liquid_reject(
        ten_sec[-1],
        history=ten_sec,
        target_wins=1,
        entry_trigger="warrior_trend_pullback_reclaim",
    )

    assert reason is None


def test_warrior_trend_pullback_reclaim_rejects_dump_base() -> None:
    ten_sec = [
        Bar("MBUR", _BASE_TS + timedelta(seconds=0), 6.80, 7.30, 6.76, 7.22, 180_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=10), 7.24, 7.85, 7.20, 7.78, 260_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=20), 7.80, 8.55, 7.74, 8.42, 330_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=30), 8.42, 9.35, 8.35, 9.12, 440_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=40), 9.12, 10.20, 9.00, 9.82, 620_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=50), 9.78, 10.34, 9.62, 9.90, 520_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=60), 9.86, 9.98, 9.18, 9.30, 240_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=70), 9.30, 9.42, 8.82, 8.96, 170_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=80), 9.30, 9.38, 8.42, 8.58, 390_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=90), 8.76, 8.98, 8.46, 8.84, 135_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=100), 8.84, 9.08, 8.58, 8.98, 150_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=110), 9.35, 9.42, 8.62, 8.78, 420_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=120), 9.08, 9.24, 8.86, 9.18, 185_000, Timeframe.SEC_10),
        Bar("MBUR", _BASE_TS + timedelta(seconds=130), 9.18, 9.45, 9.05, 9.39, 320_000, Timeframe.SEC_10),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True

    context = driver._warrior_trend_pullback_reclaim_context(
        "MBUR",
        ten_sec[-1],
        window_high=10.34,
    )

    assert context is None


def test_warrior_squeeze_rejects_midrange_weak_close_above_five() -> None:
    ten_sec = [
        _ten_bar(0, 5.74, width_pct=0.04, volume=140_000),
        _ten_bar(10, 5.90, width_pct=0.08, volume=185_000),
        Bar(
            symbol="MBUR",
            ts=_BASE_TS + timedelta(seconds=20),
            open=5.93,
            high=6.19,
            low=5.89,
            close=6.08,
            volume=104_000,
            timeframe=Timeframe.SEC_10,
        ),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_rejection_high["MBUR"] = 3.85
    pending = {
        "breakout_close": 5.91,
        "breakout_high": 6.19,
        "breakout_volume": 185_000,
        "entry_trigger": "warrior_a_plus_reclaim",
    }

    context = driver._warrior_squeeze_pullaway_context("MBUR", ten_sec[-1], pending)

    assert context is None


def test_warrior_squeeze_blocks_first_starter_without_proof_hold() -> None:
    ten_sec = [
        _ten_bar(0, 3.38, width_pct=0.04, volume=75_000),
        _ten_bar(10, 3.46, width_pct=0.07, volume=260_000),
        _ten_bar(20, 3.52, width_pct=0.05, volume=180_000),
        _ten_bar(30, 3.53, width_pct=0.03, volume=125_000),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_rejection_high["MBUR"] = 2.24
    pending = {
        "breakout_close": 3.52,
        "breakout_high": 3.60,
        "breakout_volume": 180_000,
        "entry_trigger": "warrior_a_plus_reclaim",
    }

    context = driver._warrior_squeeze_pullaway_context("MBUR", ten_sec[-1], pending)

    assert context is None


def test_warrior_squeeze_allows_first_starter_after_proof_hold() -> None:
    ten_sec = [
        _ten_bar(0, 3.48, width_pct=0.012, volume=80_000),
        _ten_bar(10, 3.53, width_pct=0.012, volume=150_000),
        _ten_bar(20, 3.56, width_pct=0.012, volume=160_000),
        _ten_bar(30, 3.62, width_pct=0.012, volume=180_000),
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_warrior_squeeze_playbook = True
    driver._warrior_squeeze_min_reclaim_price = 3.50
    driver._warrior_squeeze_rejection_high["MBUR"] = 2.24
    pending = {
        "breakout_close": 3.56,
        "breakout_high": 3.60,
        "breakout_volume": 150_000,
        "entry_trigger": "warrior_a_plus_reclaim",
    }

    context = driver._warrior_squeeze_pullaway_context("MBUR", ten_sec[-1], pending)

    assert context is not None
    assert context["entry_trigger"] == "warrior_level_pullaway"
    assert context["pullaway_level"] == pytest.approx(3.5)


def test_post_blowoff_micro_base_scout_signal_is_reduced_and_tagged() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=80_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_momentum_burst_hit_run = True

    normal = driver._momentum_burst_replay_signal(
        "MBUR",
        ten_sec[-1],
        driver._bars_by_symbol["MBUR"],
        hit_run=True,
    )
    scout = driver._momentum_burst_replay_signal(
        "MBUR",
        ten_sec[-1],
        driver._bars_by_symbol["MBUR"],
        hit_run=True,
        post_blowoff_micro_base=True,
    )

    assert normal is not None
    assert scout is not None
    assert scout.scan_result.scanner_name == "post_blowoff_micro_base_scout"
    assert scout.scan_result.criteria["entry_mode"] == "post_blowoff_micro_base_scout"
    assert scout.scan_result.criteria["variant"] == "post_blowoff_micro_base"
    assert scout.scan_result.criteria["size_factor"] == 0.35
    assert scout.quantity < normal.quantity
    assert scout.stop_loss == round(scout.entry_price - max(scout.entry_price * 0.015, 0.06), 2)
    assert scout.take_profit == round(scout.entry_price + (scout.entry_price - scout.stop_loss), 2)


def test_momentum_burst_hit_run_rejects_confirm_bar_that_sweeps_stop() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=60_000) for i in range(12)]
    unstable = Bar(
        symbol="MBUR",
        ts=ten_sec[-1].ts + timedelta(seconds=10),
        open=2.46,
        high=2.62,
        low=2.43,
        close=2.50,
        volume=120_000,
        timeframe=Timeframe.SEC_10,
    )
    ten_sec.append(unstable)
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._use_momentum_burst_hit_run = True

    signal = driver._momentum_burst_replay_signal(
        "MBUR",
        unstable,
        driver._bars_by_symbol["MBUR"],
        hit_run=True,
        violent_liquid=True,
    )

    assert signal is None


def test_momentum_burst_hit_run_backtest_time_window_blocks_afternoon_et() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._mb_hit_run_end_et = "11:30"

    assert driver._momentum_burst_hit_run_time_allowed(
        datetime(2026, 6, 8, 14, 30, tzinfo=timezone.utc)
    ) is True
    assert driver._momentum_burst_hit_run_time_allowed(
        datetime(2026, 6, 8, 18, 0, tzinfo=timezone.utc)
    ) is False


def test_momentum_burst_hit_run_backtest_time_window_blank_allows_all_day() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._mb_hit_run_end_et = ""

    assert driver._momentum_burst_hit_run_time_allowed(
        datetime(2026, 6, 8, 18, 0, tzinfo=timezone.utc)
    ) is True


def test_momentum_burst_hit_run_backtest_rebase_preserves_original_anchor() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=80_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._momentum_burst_continuation_base_ok = (
        lambda symbol, now: (True, "fresh micro-base reclaim", {"base_high": 2.55, "base_low": 2.50})
    )
    now = ten_sec[-1].ts + timedelta(seconds=10)
    pending = {
        "ts": ten_sec[-2].ts,
        "breakout_close": 2.50,
        "breakout_high": 2.55,
        "breakout_volume": 120_000,
        "reset_from_stale_high": 2.90,
    }

    rebased = driver._momentum_burst_rebase_pending_after_reject(
        "MBUR",
        Bar("MBUR", now, 2.52, 2.55, 2.50, 2.54, 130_000, Timeframe.SEC_10),
        pending,
        "confirm bar did not break continuation high (2.55 <= 2.55)",
        now,
        hit_run=True,
    )

    assert rebased is True
    assert driver._momentum_burst_pending["MBUR"]["original_ts"] == ten_sec[-2].ts
    assert driver._momentum_burst_pending["MBUR"]["original_breakout_close"] == 2.50
    assert driver._momentum_burst_pending["MBUR"]["reset_from_stale_high"] == 2.90
    assert driver._momentum_burst_pending["MBUR"]["rebase_count"] == 1


def test_momentum_burst_hit_run_backtest_rebase_is_capped() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.006, volume=80_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)
    driver._momentum_burst_continuation_base_ok = (
        lambda symbol, now: (True, "fresh micro-base reclaim", {"base_high": 2.55, "base_low": 2.50})
    )
    now = ten_sec[-1].ts + timedelta(seconds=10)
    pending = {
        "ts": ten_sec[-1].ts,
        "original_ts": ten_sec[-2].ts,
        "breakout_close": 2.54,
        "breakout_high": 2.55,
        "original_breakout_close": 2.50,
        "original_breakout_high": 2.55,
        "breakout_volume": 120_000,
        "rebase_count": 1,
    }

    rebased = driver._momentum_burst_rebase_pending_after_reject(
        "MBUR",
        Bar("MBUR", now, 2.54, 2.56, 2.52, 2.55, 130_000, Timeframe.SEC_10),
        pending,
        "confirm bar did not break continuation high (2.56 <= 2.56)",
        now,
        hit_run=True,
    )

    assert rebased is False
    assert "MBUR" not in driver._momentum_burst_pending


def test_momentum_burst_continuation_base_allows_extended_runner() -> None:
    ten_sec = [
        _ten_bar(i * 10, close, width_pct=0.025, volume=120_000)
        for i, close in enumerate([8.8, 8.7, 8.9, 9.0, 9.15, 9.3, 9.55, 9.8])
    ]
    driver = _momentum_burst_backtest_driver(ten_sec)

    ok, reason, meta = driver._momentum_burst_continuation_base_ok("MBUR", ten_sec[-1].ts)

    assert ok is True
    assert reason == "fresh continuation base"
    assert meta["recent_10s_volume"] >= 150_000


def test_momentum_burst_replay_rejects_gappy_10s_tape() -> None:
    ten_sec = [_ten_bar(i * 10, 2.00 + i * 0.01, width_pct=0.045, volume=5_000) for i in range(12)]
    driver = _momentum_burst_backtest_driver(ten_sec)

    smooth, median_range = driver._momentum_burst_10s_tape_is_smooth("MBUR", ten_sec[-1].ts)

    assert smooth is False
    assert median_range > 2.0


def test_backtest_csv_loader_groups_bars(tmp_path) -> None:
    path = tmp_path / "bars.csv"
    path.write_text(
        "symbol,ts,open,high,low,close,volume\n"
        "AAA,2026-06-12T13:30:00+00:00,1,1.1,0.9,1.05,1000\n"
        "BBB,2026-06-12T13:30:00+00:00,2,2.1,1.9,2.05,2000\n",
        encoding="utf-8",
    )

    bars = load_bars_csv(str(path))

    assert sorted(bars) == ["AAA", "BBB"]
    assert bars["AAA"][0].close == pytest.approx(1.05)


def test_run_backtest_service_uses_in_memory_bars_and_flags() -> None:
    bars = {
        "TEST": [
            _bar("TEST", 0, 8.20),
            _bar("TEST", 1, 8.45),
            _bar("TEST", 2, 8.80),
            _bar("TEST", 3, 9.20),
            _bar("TEST", 4, 9.55),
            _bar("TEST", 5, 10.10, volume=500_000),
        ]
    }

    result = run_backtest(
        "test",
        "2026-06-01",
        flags={"fresh_vwap_reclaim_scout": True, "level_breakout_scout": False},
        bars_by_symbol=bars,
    )

    assert result["ok"] is True
    assert result["symbol"] == "TEST"
    assert result["bars"] == 6
    assert len(result["bars_data"]) == 6
    assert result["bars_data"][0]["close"] == pytest.approx(8.20)
    assert "scan_events" in result
    assert "entry_decisions" in result
    assert result["flags"]["fresh_vwap_reclaim_scout"] is True
    assert result["flags"]["level_breakout_scout"] is False
    assert result["flags"]["execution_timer_10s"] is True
    assert result["execution_timer_source"] == "synthetic_1m_to_10s"
    assert "scorecard" in result
    manifest = result["manifest"]
    assert manifest["symbol"] == "TEST"
    assert manifest["date"] == "2026-06-01"
    assert manifest["data"]["source"] == "in_memory"
    assert manifest["data"]["bars_1m"] == 6
    assert manifest["flags"]["fresh_vwap_reclaim_scout"] is True
    assert "code_version" in manifest
    assert "momentum_burst_hit_run_end_et" in manifest["settings"]["strategy"]


def test_run_backtest_service_passes_runner_trail_settings(monkeypatch) -> None:
    bars = {
        "TEST": [
            _bar("TEST", 0, 8.20),
            _bar("TEST", 1, 8.45),
            _bar("TEST", 2, 8.80),
            _bar("TEST", 3, 9.20),
            _bar("TEST", 4, 9.55),
            _bar("TEST", 5, 10.10, volume=500_000),
        ]
    }
    seen = {}
    real_factory = __import__(
        "daytrading.pipeline.factory",
        fromlist=["create_scalping_pipeline"],
    ).create_scalping_pipeline

    def wrapped_factory(*args, **kwargs):
        seen["runner_trail_pct"] = kwargs.get("runner_trail_pct")
        seen["runner_min_confirm_pct"] = kwargs.get("runner_min_confirm_pct")
        seen["runner_trail_adaptive"] = kwargs.get("runner_trail_adaptive")
        seen["runner_trail_atr_mult"] = kwargs.get("runner_trail_atr_mult")
        seen["runner_trail_cap"] = kwargs.get("runner_trail_cap")
        seen["runner_give_room_after_partial"] = kwargs.get("runner_give_room_after_partial")
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(
        "daytrading.backtest.service.create_scalping_pipeline",
        wrapped_factory,
    )
    settings = Settings()
    settings.strategy = StrategyConfig(
        **{
            **settings.strategy.__dict__,
            "runner_trail_pct": 0.08,
            "runner_min_confirm_pct": 0.025,
            "runner_trail_adaptive": True,
            "runner_trail_atr_mult": 3.0,
            "runner_trail_cap": 0.12,
            "runner_give_room_after_partial": True,
        }
    )

    run_backtest(
        "test",
        "2026-06-01",
        bars_by_symbol=bars,
        settings=settings,
    )

    assert seen == {
        "runner_trail_pct": 0.08,
        "runner_min_confirm_pct": 0.025,
        "runner_trail_adaptive": True,
        "runner_trail_atr_mult": 3.0,
        "runner_trail_cap": 0.12,
        "runner_give_room_after_partial": True,
    }


def test_backtest_service_passes_chase_guard_settings(monkeypatch) -> None:
    """The chase guards are configured via methods, not factory kwargs, so the
    backtest must apply them from settings or it silently ignores chase config."""
    from daytrading.pipeline.engine import TradingPipeline

    bars = {"TEST": [_bar("TEST", i, 8.0 + i * 0.2) for i in range(6)]}
    seen = {}

    def cap_entry(self, *, pct_low, pct_high, price_tier):
        seen["entry"] = (pct_low, pct_high, price_tier)

    def cap_missed(self, *, window_sec, pct_sub5, pct_5plus, fresh_base_reset=False, fresh_base_pct=0.08):
        seen["missed"] = (window_sec, pct_sub5, pct_5plus)
        seen["fresh"] = (fresh_base_reset, fresh_base_pct)

    monkeypatch.setattr(TradingPipeline, "configure_entry_chase_guard", cap_entry)
    monkeypatch.setattr(TradingPipeline, "configure_missed_a_plus_chase_guard", cap_missed)

    settings = Settings()
    settings.strategy = StrategyConfig(
        **{
            **settings.strategy.__dict__,
            "entry_chase_pct_low": 0.07,
            "entry_chase_pct_high": 0.04,
            "entry_chase_price_tier": 8.0,
            "missed_a_plus_chase_window_sec": 900.0,
            "missed_a_plus_chase_pct_sub5": 0.06,
            "missed_a_plus_chase_pct_5plus": 0.04,
            "missed_a_plus_fresh_base_reset": True,
            "missed_a_plus_fresh_base_pct": 0.09,
        }
    )

    run_backtest("test", "2026-06-01", bars_by_symbol=bars, settings=settings)

    assert seen["entry"] == (0.07, 0.04, 8.0)
    assert seen["missed"] == (900.0, 0.06, 0.04)
    assert seen["fresh"] == (True, 0.09)


def test_backtest_service_accepts_european_date_shorthand() -> None:
    assert normalize_session_date("12/06/2026").isoformat() == "2026-06-12"
    assert normalize_session_date("2026-06-12").isoformat() == "2026-06-12"


def test_backtest_service_treats_plain_start_time_as_eastern() -> None:
    parsed = normalize_start_time("10:10", normalize_session_date("2026-06-15"))

    assert parsed is not None
    assert parsed.isoformat() == "2026-06-15T14:10:00+00:00"


def test_backtest_flags_default_experiments_off() -> None:
    flags = normalize_flags(None)

    assert flags["fresh_vwap_reclaim_scout"] is False
    assert flags["level_breakout_scout"] is False
    assert flags["elite_wide_spread"] is False
    assert flags["momentum_burst_live"] is False
    assert flags["momentum_burst_hit_run"] is False
    assert flags["warrior_squeeze_playbook"] is False
    assert flags["level_capped_entry"] is False
    assert flags["execution_timer_10s"] is True


def test_backtest_flags_accept_momentum_burst_hit_run() -> None:
    flags = normalize_flags({"momentum_burst_hit_run": True, "live_like_10s": True})

    assert flags["momentum_burst_hit_run"] is True
    assert flags["live_like_10s"] is True


def test_backtest_flags_accept_warrior_squeeze_playbook() -> None:
    flags = normalize_flags({
        "warrior_squeeze_playbook": True,
        "live_like_10s": True,
    })

    assert flags["warrior_squeeze_playbook"] is True
    assert flags["live_like_10s"] is True


def test_run_backtest_sweep_compares_experiments_against_baseline() -> None:
    bars = {
        "TEST": [
            _bar("TEST", 0, 8.20),
            _bar("TEST", 1, 8.45),
            _bar("TEST", 2, 8.80),
            _bar("TEST", 3, 9.20),
            _bar("TEST", 4, 9.55),
            _bar("TEST", 5, 10.10, volume=500_000),
        ]
    }

    result = run_backtest_sweep(
        ["test"],
        ["2026-06-01"],
        experiments={
            "baseline": {},
            "level_only": {"level_breakout_scout": True},
            "momentum_live": {"momentum_burst_live": True},
        },
        bars_by_symbol_date={("TEST", "2026-06-01"): bars},
    )

    assert result["ok"] is True
    assert result["symbols"] == ["TEST"]
    assert result["dates"] == ["2026-06-01"]
    assert result["experiments"]["baseline"]["flags"]["level_breakout_scout"] is False
    assert result["experiments"]["level_only"]["flags"]["level_breakout_scout"] is True
    assert result["experiments"]["momentum_live"]["flags"]["momentum_burst_live"] is True
    assert "level_only" in result["deltas_vs_baseline"]
    assert result["unsupported_flags"] == ["momentum_breakout"]
    assert "supplied symbol/date basket" in result["universe_note"]


def test_ledger_blends_scale_up_cost_and_charges_both_commissions() -> None:
    """A scale-up must blend into a volume-weighted cost and a round trip must
    pay commission on both legs — otherwise multi-entry PnL is overstated."""
    ts = datetime(2026, 6, 12, 14, 0, tzinfo=timezone.utc)
    ledger = BacktestLedger()
    ledger.record_entry(Fill("X", Side.BUY, 100, 5.00, ts, commission=1.0))
    ledger.record_entry(Fill("X", Side.BUY, 100, 6.00, ts, commission=1.0))  # scale up
    ledger.record_exit(Fill("X", Side.SELL, 200, 7.00, ts, commission=2.0))

    exit_row = [t for t in ledger.trades if t["trade_type"] == "exit"][0]
    assert exit_row["entry_price"] == pytest.approx(5.50)  # blended, not last add (6.00)
    # (7.00 - 5.50) * 200 = 300 gross, minus 2.0 exit + 2.0 entry commission
    assert exit_row["pnl"] == pytest.approx(296.0)


def test_ledger_partial_exit_keeps_proportional_entry_commission() -> None:
    ts = datetime(2026, 6, 12, 14, 0, tzinfo=timezone.utc)
    ledger = BacktestLedger()
    ledger.record_entry(Fill("X", Side.BUY, 100, 5.00, ts, commission=4.0))
    ledger.record_exit(Fill("X", Side.SELL, 50, 6.00, ts, commission=0.0))  # half out
    ledger.record_exit(Fill("X", Side.SELL, 50, 7.00, ts, commission=0.0))  # rest out

    exits = [t for t in ledger.trades if t["trade_type"] == "exit"]
    # 4.0 entry commission split 2.0 / 2.0 across the two 50-share exits
    assert exits[0]["pnl"] == pytest.approx((6.00 - 5.00) * 50 - 2.0)
    assert exits[1]["pnl"] == pytest.approx((7.00 - 5.00) * 50 - 2.0)


def test_normalize_symbol_rejects_path_traversal() -> None:
    assert normalize_symbol(" cupr ") == "CUPR"
    assert normalize_symbol("brk.b") == "BRK.B"
    for bad in ("../../etc/passwd", "A/B", "", "foo bar"):
        with pytest.raises(ValueError):
            normalize_symbol(bad)


def _watch_hit(symbol="VSME", *, level=2.00, breakout_pct=0.0, distance=0.0, vsurge=1.5):
    return ScanResult(
        symbol=symbol,
        scanner_name="level_breakout_watch",
        ts=_BASE_TS,
        score=40.0,
        criteria={
            "pattern": "level_breakout_watch",
            "setup_tier": "watch only",
            "breakout_level": level,
            "base_low": level * 0.97,
            "breakout_pct": breakout_pct,
            "distance_to_level_pct": distance,
            "volume_surge": vsurge,
            "close": level * (1 + breakout_pct / 100.0),
        },
        bars=[],
    )


def test_level_reclaim_contexts_promotes_near_level_watch_hit() -> None:
    cycle = PipelineResult()
    cycle.scan_hits = [_watch_hit(level=2.00, breakout_pct=0.1, vsurge=1.5)]
    ctx = PipelineBacktestDriver._level_reclaim_contexts(cycle)
    assert "VSME" in ctx
    row = ctx["VSME"][0]
    assert row["source"] == "level_reclaim"
    assert row["pattern"] == "level_breakout_reclaim"  # promoted from watch
    assert row["level"] == pytest.approx(2.00)


def test_level_reclaim_contexts_filters_far_and_thin_hits() -> None:
    cycle = PipelineResult()
    cycle.scan_hits = [
        _watch_hit(symbol="FAR", breakout_pct=-0.5, distance=2.0, vsurge=1.5),  # far below level
        _watch_hit(symbol="THIN", breakout_pct=0.1, vsurge=0.5),                # weak volume
    ]
    ctx = PipelineBacktestDriver._level_reclaim_contexts(cycle)
    assert ctx == {}


def test_daily_loser_blacklist_uses_consecutive_losses_not_single_scalp() -> None:
    """A single normal scalp loss must NOT ban a name (it can set up a clean
    re-entry later); ban only after consecutive losses, or one real blowout."""
    p = TradingPipeline(
        scanners=[], verifiers={}, broker=BacktestBroker(),
        portfolio=PortfolioState(cash=25_000), enable_daily_loser_blacklist=True,
    )

    def realize(sym, dollars):  # loss/win of $dollars via 1000 shares
        p.record_realized_exit(sym, 1.00, 1.00 + dollars / 1000.0, 1000)

    realize("AAA", -20)               # one normal scalp loss
    assert "AAA" not in p._daily_losers          # not banned
    realize("AAA", -20)               # second consecutive loss
    assert "AAA" in p._daily_losers              # now banned (2 consecutive)

    realize("BBB", -20)
    realize("BBB", +15)               # a win resets the consecutive counter
    realize("BBB", -20)
    assert "BBB" not in p._daily_losers          # loss-win-loss != 2 consecutive

    realize("CCC", -60)               # single blowout >= $50
    assert "CCC" in p._daily_losers              # banned immediately


def _ten_sec(symbol: str, minute: int, slot: int, close: float, *, high: float = None, low: float = None, volume: float = 40_000) -> Bar:
    """A real 10s bar with a distinct 10s timestamp (slot 0..5 within the minute)."""
    ts = _BASE_TS + timedelta(minutes=minute, seconds=slot * 10)
    return Bar(
        symbol=symbol,
        ts=ts,
        open=close - 0.01,
        high=high if high is not None else close + 0.01,
        low=low if low is not None else close - 0.01,
        close=close,
        volume=volume,
        timeframe=Timeframe.SEC_10,
    )


def _live_like_driver():
    one_min = {"TEST": [_bar("TEST", m, 4.00 + 0.05 * m) for m in range(4)]}
    # Minute 3 has a sharp intra-minute high wick at slot 2 that a 1m close hides.
    ten_sec = {
        "TEST": [
            _ten_sec("TEST", 3, 0, 4.16),
            _ten_sec("TEST", 3, 1, 4.18),
            _ten_sec("TEST", 3, 2, 4.40, high=4.55),  # spike wick
            _ten_sec("TEST", 3, 3, 4.30),
            _ten_sec("TEST", 3, 4, 4.22),
            _ten_sec("TEST", 3, 5, 4.20),
        ]
    }
    return PipelineBacktestDriver(
        one_min,
        timer_bars_by_symbol=ten_sec,
        use_execution_timer=True,
        live_like_10s=True,
    )


def test_live_like_mode_engages_only_with_real_10s_bars():
    driver = _live_like_driver()
    assert driver._live_like_10s is True
    # Without 10s bars, live-like must fall back to the 1m loop.
    fallback = PipelineBacktestDriver(
        {"TEST": [_bar("TEST", m, 4.0 + 0.05 * m) for m in range(4)]},
        use_execution_timer=True,
        live_like_10s=True,
    )
    assert fallback._live_like_10s is False


def test_partial_minute_bar_tracks_latest_10s_close_and_wick():
    driver = _live_like_driver()
    minute_start = _BASE_TS + timedelta(minutes=3)
    # Early in the minute: close == first slot close, no spike yet.
    early = driver._partial_minute_bar("TEST", minute_start, minute_start)
    assert early is not None
    assert early.close == pytest.approx(4.16)
    assert early.high == pytest.approx(4.17)
    assert early.timeframe is Timeframe.MIN_1
    # After the spike slot: close is the current 10s close, high captures the wick
    # that a 1m-close-only backtest would never see.
    after_spike = driver._partial_minute_bar(
        "TEST", minute_start, minute_start + timedelta(seconds=20)
    )
    assert after_spike.close == pytest.approx(4.40)
    assert after_spike.high == pytest.approx(4.55)
    assert after_spike.open == pytest.approx(4.15)  # first slot open


def test_ten_sec_bar_at_returns_exact_timestamp_bar():
    driver = _live_like_driver()
    t = _BASE_TS + timedelta(minutes=3, seconds=20)
    bar = driver._ten_sec_bar_at("TEST", t)
    assert bar is not None and bar.ts == t and bar.close == pytest.approx(4.40)
    assert driver._ten_sec_bar_at("TEST", t + timedelta(seconds=3)) is None


def test_live_like_run_uses_10s_clock_and_reports_source():
    driver = _live_like_driver()
    result = driver.run()
    assert result.execution_timer_source == "real_trades_10s_live_like"
    # 10s cadence over minute 3's 6 slots => more cycles than the 4 one-minute bars.
    assert result.cycles >= 6


def test_backtest_1m_cache_refreshes_partial_session(tmp_path):
    cache = tmp_path / "CAST_2026-06-15_1m.json"
    cache.write_text(json.dumps({
        "symbol": "CAST",
        "date": "2026-06-15",
        "bars": [{
            "symbol": "CAST",
            "ts": "2026-06-15T14:18:00+00:00",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1000,
            "timeframe": "1m",
        }],
    }))

    complete = [
        Bar(
            symbol="CAST",
            ts=datetime(2026, 6, 15, 8, 0, tzinfo=timezone.utc),
            open=1.0,
            high=1.1,
            low=0.9,
            close=1.0,
            volume=1000,
            timeframe=Timeframe.MIN_1,
        ),
        Bar(
            symbol="CAST",
            ts=datetime(2026, 6, 15, 23, 56, tzinfo=timezone.utc),
            open=2.0,
            high=2.1,
            low=1.9,
            close=2.0,
            volume=2000,
            timeframe=Timeframe.MIN_1,
        ),
    ]

    class _Feed:
        calls = 0

        def get_bars(self, symbols, timeframe, start, end, limit):
            self.calls += 1
            return {"CAST": complete}

    feed = _Feed()
    rows = fetch_alpaca_bars_for_day("CAST", date(2026, 6, 15), cache_dir=str(tmp_path), feed=feed)

    assert feed.calls == 1
    assert rows["CAST"][-1].close == pytest.approx(2.0)


def test_live_like_breakout_scalp_replay_can_enter_from_10s_hod_expansion():
    broker = BacktestBroker(FillModel(min_spread_cents=0.01, spread_pct_of_range=0.0))
    portfolio = PortfolioState(cash=10_000)
    pipeline = TradingPipeline(
        scanners=[],
        verifiers={},
        broker=broker,
        portfolio=portfolio,
        exit_manager=ExitManager(max_unrealized_loss=500.0),
    )
    pipeline._final_entry_quality_reject = lambda *args, **kwargs: None  # type: ignore[method-assign]
    one_min = {
        "CAST": [
            Bar("CAST", _BASE_TS + timedelta(minutes=idx), 1.8 + idx * 0.1, 1.9 + idx * 0.1, 1.75 + idx * 0.1, 1.85 + idx * 0.1, 200_000, Timeframe.MIN_1)
            for idx in range(4)
        ]
    }
    ten_sec = {
        "CAST": [
            *[
                _ten_sec(
                    "CAST",
                    1 + idx // 6,
                    idx % 6,
                    2.00 + idx * 0.01,
                    high=2.02 + idx * 0.01,
                    low=1.98 + idx * 0.01,
                    volume=35_000,
                )
                for idx in range(12)
            ],
            _ten_sec("CAST", 3, 0, 2.20, high=2.22, low=2.18, volume=80_000),
            _ten_sec("CAST", 3, 1, 2.22, high=2.23, low=2.20, volume=80_000),
            _ten_sec("CAST", 3, 2, 2.24, high=2.25, low=2.21, volume=80_000),
            _ten_sec("CAST", 3, 3, 2.50, high=2.52, low=2.42, volume=250_000),
        ]
    }

    result = PipelineBacktestDriver(
        one_min,
        pipeline=pipeline,
        portfolio=portfolio,
        timer_bars_by_symbol=ten_sec,
        use_execution_timer=True,
        live_like_10s=True,
        use_breakout_scalp_replay=True,
    ).run()

    assert any(t.get("strategy") == "breakout_scalp_replay" for t in result.trades)
    assert any(d.get("stage") == "breakout_scalp_replay" for d in result.entry_decisions)


def test_breakout_scalp_replay_rejects_violent_10s_without_high_close():
    driver = PipelineBacktestDriver(
        {"CAST": [_bar("CAST", 0, 2.00)]},
        timer_bars_by_symbol={
            "CAST": [
                _ten_sec("CAST", 0, 0, 9.55, high=9.70, low=9.30, volume=130_000),
                _ten_sec("CAST", 0, 1, 10.69, high=11.34, low=9.33, volume=291_000),
            ]
        },
    )

    reject = driver._breakout_scalp_10s_quality_reject(
        "CAST",
        driver._timer_bars_by_symbol["CAST"][-1],
    )

    assert reject == "10s breakout candle too volatile without strong close (68% location, 18.8% range)"


def test_breakout_scalp_replay_rejects_recent_10s_dump_before_breakout():
    driver = PipelineBacktestDriver(
        {"CAST": [_bar("CAST", 0, 2.00)]},
        timer_bars_by_symbol={
            "CAST": [
                _ten_sec("CAST", 0, 0, 2.35, high=2.38, low=2.28, volume=90_000),
                Bar(
                    "CAST",
                    _BASE_TS + timedelta(seconds=10),
                    2.29,
                    2.32,
                    2.21,
                    2.22,
                    175_000,
                    Timeframe.SEC_10,
                ),
                _ten_sec("CAST", 0, 2, 2.47, high=2.48, low=2.20, volume=130_000),
                _ten_sec("CAST", 0, 3, 2.60, high=2.62, low=2.50, volume=180_000),
            ]
        },
    )

    reject = driver._breakout_scalp_10s_quality_reject(
        "CAST",
        driver._timer_bars_by_symbol["CAST"][-1],
    )

    assert reject == "recent 10s dump candle before breakout (3.1% body, 9% close location)"


def test_breakout_scalp_replay_allows_high_close_wide_10s_without_prior_dump():
    driver = PipelineBacktestDriver(
        {"CAST": [_bar("CAST", 0, 2.00)]},
        timer_bars_by_symbol={
            "CAST": [
                _ten_sec("CAST", 0, 0, 4.37, high=4.42, low=4.15, volume=180_000),
                _ten_sec("CAST", 0, 1, 4.73, high=4.80, low=4.30, volume=450_000),
                _ten_sec("CAST", 0, 2, 5.17, high=5.17, low=4.61, volume=725_000),
            ]
        },
    )

    reject = driver._breakout_scalp_10s_quality_reject(
        "CAST",
        driver._timer_bars_by_symbol["CAST"][-1],
    )

    assert reject is None


def test_backtest_realized_symbol_pnl_reads_closed_trade_rows():
    ledger = BacktestLedger()
    ledger.trades.extend([
        {
            "symbol": "NCT",
            "trade_type": "exit",
            "strategy": "shallow_stair_continuation",
            "pnl": 101.76,
        },
        {
            "symbol": "NCT",
            "trade_type": "entry",
            "strategy": "warrior_squeeze_playbook",
            "pnl": None,
        },
        {
            "symbol": "OTHER",
            "trade_type": "exit",
            "strategy": "warrior_squeeze_playbook",
            "pnl": -50.0,
        },
    ])

    assert PipelineBacktestDriver._realized_symbol_pnl_from_ledger(ledger, "NCT") == pytest.approx(101.76)
