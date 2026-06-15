from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Sequence

import pytest

from daytrading.backtest.broker import BacktestBroker, FillModel
from daytrading.backtest.data_loader import load_bars_csv
from daytrading.backtest.driver import PipelineBacktestDriver
from daytrading.backtest.report import BacktestLedger
from daytrading.backtest.service import normalize_flags, normalize_session_date, run_backtest, run_backtest_sweep
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


def test_backtest_flags_default_experiments_off() -> None:
    flags = normalize_flags(None)

    assert flags["fresh_vwap_reclaim_scout"] is False
    assert flags["level_breakout_scout"] is False
    assert flags["elite_wide_spread"] is False
    assert flags["momentum_burst_live"] is False
    assert flags["level_capped_entry"] is False
    assert flags["execution_timer_10s"] is True


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
