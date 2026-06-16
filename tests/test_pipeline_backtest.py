from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Sequence

import pytest

from daytrading.backtest.broker import BacktestBroker, FillModel
from daytrading.backtest.data_loader import load_bars_csv
from daytrading.backtest.data_loader import fetch_alpaca_bars_for_day
from daytrading.backtest.driver import PipelineBacktestDriver
from daytrading.backtest.report import BacktestLedger
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
    assert flags["level_capped_entry"] is False
    assert flags["execution_timer_10s"] is True


def test_backtest_flags_accept_momentum_burst_hit_run() -> None:
    flags = normalize_flags({"momentum_burst_hit_run": True, "live_like_10s": True})

    assert flags["momentum_burst_hit_run"] is True
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
