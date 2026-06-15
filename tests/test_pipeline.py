"""Integration test for the full Scanner → Verify → Execute pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

from daytrading.execution.broker import PaperBroker
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.pipeline.engine import TradingPipeline, _entry_strategy_label
from daytrading.scanner.premarket_gap import PremarketGapScanner
from daytrading.strategy.gap_reversal import GapReversalVerifier
from daytrading.models import (
    Bar,
    PortfolioState,
    Quote,
    ScanResult,
    SignalAction,
    TradeSignal,
    TradingStyle,
)

TS = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)


class _OneHitScanner:
    def __init__(self, scanner_name: str, pattern: str) -> None:
        self.name = scanner_name
        self._pattern = pattern

    def scan(self, universe):
        bars = list(universe["HOT"])
        return [
            ScanResult(
                symbol="HOT",
                scanner_name=self.name,
                ts=TS,
                score=1.0,
                criteria={"pattern": self._pattern},
                bars=bars,
            )
        ]


class _OneHitScannerWithCriteria:
    def __init__(self, scanner_name: str, criteria: dict) -> None:
        self.name = scanner_name
        self._criteria = dict(criteria)

    def scan(self, universe):
        bars = list(universe["HOT"])
        criteria = dict(self._criteria)
        criteria.setdefault("close", bars[-1].close)
        return [
            ScanResult(
                symbol="HOT",
                scanner_name=self.name,
                ts=TS,
                score=50.0,
                criteria=criteria,
                bars=bars,
            )
        ]


class _SignalVerifier:
    def verify(self, hit, portfolio):
        latest = hit.bars[-1]
        return TradeSignal(
            symbol=hit.symbol,
            action=SignalAction.ENTER_LONG,
            quantity=10,
            entry_price=latest.close,
            stop_loss=latest.close - 0.20,
            take_profit=latest.close + 0.50,
            reason=hit.scanner_name,
            scan_result=hit,
        )


class _RejectVerifier:
    def __init__(self, reason: str) -> None:
        self._last_reject = reason
        self.calls = 0

    def verify(self, hit, portfolio):
        self.calls += 1
        return None


class _PatternSwitchVerifier:
    def __init__(self) -> None:
        self._last_reject = "unknown pattern: level_breakout_watch"
        self.calls = 0

    def verify(self, hit, portfolio):
        self.calls += 1
        if hit.criteria.get("pattern") == "level_breakout_watch":
            self._last_reject = "unknown pattern: level_breakout_watch"
            return None
        latest = hit.bars[-1]
        return TradeSignal(
            symbol=hit.symbol,
            action=SignalAction.ENTER_LONG,
            quantity=10,
            entry_price=latest.close,
            stop_loss=latest.close - 0.20,
            take_profit=latest.close + 0.50,
            reason=str(hit.criteria.get("pattern")),
            scan_result=hit,
        )


def test_entry_strategy_label_prefers_explicit_entry_mode() -> None:
    hit = ScanResult(
        symbol="HOT",
        scanner_name="hod_reclaim",
        ts=TS,
        score=197.0,
        criteria={
            "pattern": "hod_reclaim",
            "entry_mode": "elite_wide_spread",
            "spread_exception": "elite_wide_spread",
        },
    )
    signal = TradeSignal(
        symbol="HOT",
        action=SignalAction.ENTER_LONG,
        quantity=10,
        entry_price=7.75,
        stop_loss=7.55,
        take_profit=8.15,
        reason="HOD reclaim",
        scan_result=hit,
    )

    assert _entry_strategy_label(signal) == "elite_wide_spread"


def _bar(
    symbol: str, close: float, volume: float = 200_000,
    open_: float | None = None, high: float | None = None, low: float | None = None,
) -> Bar:
    o = open_ if open_ is not None else close
    h = high if high is not None else close + 1.0
    lo = low if low is not None else close - 1.0
    return Bar(symbol=symbol, ts=TS, open=o, high=h, low=lo, close=close, volume=volume)


def _vwap_scout_signal(
    *,
    entry_tier: str = "vwap_reclaim_scout",
    pattern: str = "vwap_pullback",
) -> TradeSignal:
    hit = ScanResult(
        symbol="HOT",
        scanner_name=pattern,
        ts=TS,
        score=77.0,
        criteria={
            "pattern": pattern,
            "entry_tier": entry_tier,
            "setup_tier": "A+ setup",
            "vwap": 2.60,
        },
    )
    return TradeSignal(
        symbol="HOT",
        action=SignalAction.ENTER_LONG,
        quantity=30,
        entry_price=2.67,
        stop_loss=2.55,
        take_profit=3.05,
        reason="VWAP reclaim scout",
        scan_result=hit,
    )


def _vwap_scout_bars() -> list[Bar]:
    closes = [2.45, 2.50, 2.54, 2.58, 2.62, 2.67]
    volumes = [100_000, 110_000, 105_000, 80_000, 75_000, 85_000]
    return [
        _bar(
            "HOT",
            close=close,
            open_=close - 0.03,
            high=close + 0.02,
            low=close - 0.04,
            volume=volume,
        )
        for close, volume in zip(closes, volumes)
    ]


def test_vwap_reclaim_scout_can_clear_mild_trade_guard_spread_trap() -> None:
    signal = _vwap_scout_signal()
    bars = _vwap_scout_bars()
    quotes = [Quote("HOT", TS, bid=2.659, ask=2.681, bid_size=1400, ask_size=1300)]

    assert TradingPipeline._allow_vwap_reclaim_scout_trade_guard_exception(
        signal,
        bars=bars,
        quotes=quotes,
        reason="liquidity trap: spread 0.82% with weak volume 58177 vs avg 71657",
    )


def test_vwap_reclaim_scout_trade_guard_exception_requires_tag() -> None:
    signal = _vwap_scout_signal(entry_tier="")
    bars = _vwap_scout_bars()
    quotes = [Quote("HOT", TS, bid=2.659, ask=2.681, bid_size=1400, ask_size=1300)]

    assert not TradingPipeline._allow_vwap_reclaim_scout_trade_guard_exception(
        signal,
        bars=bars,
        quotes=quotes,
        reason="liquidity trap: spread 0.82% with weak volume 58177 vs avg 71657",
    )


def test_vwap_reclaim_scout_trade_guard_exception_keeps_hard_traps() -> None:
    signal = _vwap_scout_signal()
    bars = _vwap_scout_bars()
    quotes = [Quote("HOT", TS, bid=2.63, ask=2.71, bid_size=1400, ask_size=1300)]

    assert not TradingPipeline._allow_vwap_reclaim_scout_trade_guard_exception(
        signal,
        bars=bars,
        quotes=quotes,
        reason="liquidity trap: spread 2.99% with weak volume 58177 vs avg 71657",
    )
    assert not TradingPipeline._allow_vwap_reclaim_scout_trade_guard_exception(
        signal,
        bars=bars,
        quotes=[Quote("HOT", TS, bid=2.659, ask=2.681, bid_size=1400, ask_size=1300)],
        reason="liquidity trap: spike-and-fade (wick 0.20 > 2x body 0.05)",
    )


class _CountingBroker(PaperBroker):
    def __init__(self) -> None:
        super().__init__()
        self.submits = 0

    def submit(self, order, bar, portfolio):
        self.submits += 1
        return super().submit(order, bar, portfolio)


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
    # Verifier should produce a signal (gap up + fade -> long)
    if len(result.signals) > 0:
        assert result.signals[0].action == SignalAction.ENTER_LONG
        assert len(result.fills) >= 1
        assert portfolio.cash < 50_000.0  # spent some cash


def test_pipeline_records_entry_decision_for_final_guard_reject(monkeypatch) -> None:
    monkeypatch.setattr(
        "daytrading.pipeline.engine.check_entry_quality",
        lambda *args, **kwargs: "entry score too low (75/100, need 80+)",
    )
    bars = [
        _bar("HOT", 5.00, volume=500_000, open_=4.90, high=5.05, low=4.85),
        _bar("HOT", 5.10, volume=600_000, open_=5.00, high=5.15, low=4.95),
        _bar("HOT", 5.20, volume=700_000, open_=5.10, high=5.25, low=5.05),
    ]
    broker = _CountingBroker()
    pipeline = TradingPipeline(
        scanners=[_OneHitScanner("vwap_pullback", "vwap_pullback")],
        verifiers={"vwap_pullback": _SignalVerifier()},
        broker=broker,
        portfolio=PortfolioState(cash=50_000.0, positions={}),
        max_positions=5,
        max_position_shares=500,
        max_order_shares=200,
    )

    result = pipeline.run_cycle({"HOT": bars})

    assert broker.submits == 0
    assert result.rejected_orders == 1
    final_decisions = [
        d for d in result.entry_decisions
        if d["stage"] == "final_entry_guard" and d["symbol"] == "HOT"
    ]
    assert final_decisions
    assert final_decisions[-1]["passed"] is False
    assert final_decisions[-1]["blocked_layer"] == "entry_guard"
    assert "entry score too low" in final_decisions[-1]["reason"]


def test_pipeline_records_entry_decision_for_verifier_reject() -> None:
    bars = [
        _bar("HOT", 5.00, volume=500_000),
        _bar("HOT", 5.10, volume=600_000),
        _bar("HOT", 5.20, volume=700_000),
    ]
    pipeline = TradingPipeline(
        scanners=[_OneHitScanner("vwap_pullback", "vwap_pullback")],
        verifiers={"vwap_pullback": _RejectVerifier("setup tape too slow")},
        broker=_CountingBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )

    result = pipeline.run_cycle({"HOT": bars})

    rejects = [
        d for d in result.entry_decisions
        if d["stage"] == "verifier" and d["symbol"] == "HOT"
    ]
    assert rejects
    assert rejects[-1]["passed"] is False
    assert rejects[-1]["reason"] == "setup tape too slow"


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


def test_hod_gate_blocks_raw_non_hod_signal() -> None:
    bars = [_bar("HOT", 10.0), _bar("HOT", 10.2), _bar("HOT", 10.4)]
    scanner = _OneHitScanner("momentum_burst", "momentum_burst")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"momentum_burst": _SignalVerifier()},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )
    pipeline.set_hod_entry_gate(lambda symbol: False, require=True)

    result = pipeline.run_cycle({"HOT": bars})

    assert not result.fills
    assert result.skipped
    assert pipeline.scan_rejections["HOT"] == "not on HOD momentum alert board"


def test_watch_only_scanner_hit_cannot_trade_without_live_verifier() -> None:
    bars = [_bar("HOT", 10.0), _bar("HOT", 10.2), _bar("HOT", 10.4)]
    scanner = _OneHitScanner("momentum_burst", "momentum_burst")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )

    result = pipeline.run_cycle({"HOT": bars})

    assert result.scan_hits
    assert not result.signals
    assert not result.fills
    assert result.scan_hits[0].criteria["setup_tier"] == "watch only"
    assert pipeline.scan_rejections["HOT"].startswith("watch only:")


def test_a_plus_scanner_hit_still_reaches_verifier(monkeypatch) -> None:
    bars = [_bar("HOT", 10.0), _bar("HOT", 10.2), _bar("HOT", 10.4)]
    scanner = _OneHitScanner("first_pullback_reclaim", "first_pullback_reclaim")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"first_pullback_reclaim": _SignalVerifier()},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )
    monkeypatch.setattr("daytrading.pipeline.engine.check_entry_quality", lambda *a, **k: None)

    result = pipeline.run_cycle({"HOT": bars})

    assert result.scan_hits[0].criteria["setup_tier"] == "A+ setup"
    assert len(result.fills) == 1


def test_pullback_base_is_labeled_live_a_plus_setup() -> None:
    bars = [_bar("HOT", 10.0), _bar("HOT", 10.2), _bar("HOT", 10.4)]
    scanner = _OneHitScanner("pullback_base", "pullback_base")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"pullback_base": _RejectVerifier("not ready")},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )

    result = pipeline.run_cycle({"HOT": bars})

    assert result.scan_hits[0].criteria["setup_tier"] == "A+ setup"


def test_pipeline_final_entry_guard_blocks_direct_verifier_signal(monkeypatch) -> None:
    """Even a verifier-produced signal cannot reach broker without final guard/ML."""
    bars = [_bar("HOT", 10.0), _bar("HOT", 10.2), _bar("HOT", 10.4)]
    scanner = _OneHitScanner("first_pullback_reclaim", "first_pullback_reclaim")
    broker = _CountingBroker()
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"first_pullback_reclaim": _SignalVerifier()},
        broker=broker,
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )

    monkeypatch.setattr(
        "daytrading.pipeline.engine.check_entry_quality",
        lambda *args, **kwargs: "entry score too low (65/100, need 80+)",
    )

    result = pipeline.run_cycle({"HOT": bars})

    assert result.signals
    assert not result.fills
    assert broker.submits == 0
    assert result.rejection_details[-1]["reason"].startswith("final entry guard:")


def test_pipeline_final_entry_guard_runs_before_deferred_timer(monkeypatch) -> None:
    """Execution timer cannot queue a signal that failed final guard/ML."""
    bars = [_bar("HOT", 10.0), _bar("HOT", 10.2), _bar("HOT", 10.4)]
    scanner = _OneHitScanner("first_pullback_reclaim", "first_pullback_reclaim")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"first_pullback_reclaim": _SignalVerifier()},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )
    pipeline._execution_timer = object()
    monkeypatch.setattr(
        "daytrading.pipeline.engine.check_entry_quality",
        lambda *args, **kwargs: "ML model low confidence (22%, need 30%)",
    )

    result = pipeline.run_cycle({"HOT": bars})

    assert result.signals
    assert not result.deferred_signals
    assert not result.fills
    assert result.rejection_details[-1]["reason"].startswith("final entry guard:")


def test_pipeline_blocks_direct_entry_chasing_current_setup_base(monkeypatch) -> None:
    bars = [
        _bar("HOT", 5.60, volume=300_000, open_=5.50, high=5.70, low=5.45),
        _bar("HOT", 5.75, volume=350_000, open_=5.60, high=5.80, low=5.55),
        _bar("HOT", 6.15, volume=450_000, open_=6.00, high=6.20, low=5.95),
    ]
    scanner = _OneHitScannerWithCriteria(
        "level_breakout_reclaim",
        {
            "pattern": "level_breakout_reclaim",
            "setup_tier": "A+ setup",
            "base_high": 5.75,
            "breakout_level": 5.75,
        },
    )
    broker = _CountingBroker()
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"level_breakout_reclaim": _SignalVerifier()},
        broker=broker,
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )
    monkeypatch.setattr("daytrading.pipeline.engine.check_entry_quality", lambda *a, **k: None)

    result = pipeline.run_cycle({"HOT": bars}, now=TS)

    assert broker.submits == 0
    assert not result.fills
    assert result.rejection_details[-1]["reason"].startswith("late chase")
    assert result.entry_decisions[-1]["stage"] == "entry_chase_guard"


def test_pipeline_blocks_late_entry_after_earlier_missed_a_plus(monkeypatch) -> None:
    early = [
        _bar("HOT", 3.10, volume=300_000, open_=3.00, high=3.20, low=2.95),
        _bar("HOT", 3.30, volume=450_000, open_=3.10, high=3.35, low=3.05),
        _bar("HOT", 3.53, volume=700_000, open_=3.30, high=3.60, low=3.25),
    ]
    late = [
        _bar("HOT", 5.90, volume=500_000, open_=5.75, high=6.00, low=5.70),
        _bar("HOT", 6.05, volume=600_000, open_=5.90, high=6.10, low=5.85),
        _bar("HOT", 6.15, volume=800_000, open_=6.05, high=6.20, low=6.00),
    ]
    scanner = _OneHitScannerWithCriteria(
        "abc_continuation",
        {"pattern": "abc_continuation", "setup_tier": "A+ setup"},
    )
    broker = _CountingBroker()
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"abc_continuation": _SignalVerifier()},
        broker=broker,
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )
    monkeypatch.setattr("daytrading.pipeline.engine.check_entry_quality", lambda *a, **k: None)
    hit = ScanResult(
        symbol="HOT",
        scanner_name="hod_reclaim",
        ts=TS,
        score=80.0,
        criteria={
            "pattern": "hod_reclaim",
            "setup_tier": "A+ setup",
            "close": 3.53,
            "volume": 700_000,
            "breakout_level": 3.43,
        },
        bars=early,
    )
    pipeline.missed_a_plus.record_blocked(
        layer="scanner",
        reason="not on HOD momentum alert board",
        universe={"HOT": early},
        hit=hit,
        now=TS,
    )

    result = pipeline.run_cycle({"HOT": late}, now=TS)

    assert broker.submits == 0
    assert not result.fills
    assert "earlier blocked A+" in result.rejection_details[-1]["reason"]


def test_final_entry_guard_applies_to_long_entry_reentry_and_scale(monkeypatch) -> None:
    pipeline = TradingPipeline(
        scanners=[],
        verifiers={},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )
    seen = []
    monkeypatch.setattr(
        "daytrading.pipeline.engine.check_entry_quality",
        lambda *args, **kwargs: seen.append(kwargs["symbol"]) or "blocked by shared guard",
    )

    for action in (
        SignalAction.ENTER_LONG,
        SignalAction.REENTER_LONG,
        SignalAction.SCALE_UP_LONG,
    ):
        sig = TradeSignal(
            symbol=action.value,
            action=action,
            quantity=10,
            entry_price=5.0,
        )

        reason = pipeline._final_entry_quality_reject(
            sig,
            universe={sig.symbol: [_bar(sig.symbol, 5.0), _bar(sig.symbol, 5.1), _bar(sig.symbol, 5.2)]},
        )

        assert reason == "blocked by shared guard"

    assert seen == ["enter_long", "reenter_long", "scale_up_long"]


def test_final_entry_guard_does_not_block_exit_or_short_flattening(monkeypatch) -> None:
    pipeline = TradingPipeline(
        scanners=[],
        verifiers={},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )
    monkeypatch.setattr(
        "daytrading.pipeline.engine.check_entry_quality",
        lambda *args, **kwargs: "should not be called",
    )
    sig = TradeSignal(
        symbol="HOT",
        action=SignalAction.EXIT_LONG,
        quantity=10,
        entry_price=5.0,
    )

    assert pipeline._final_entry_quality_reject(sig, universe={"HOT": []}) is None


def test_scalping_factory_live_verifiers_only_allow_clean_setups() -> None:
    pipeline = create_scalping_pipeline()
    cfg = pipeline._router.get_config(TradingStyle.SCALPING)
    assert cfg is not None

    assert set(cfg.verifiers) == {
        "vwap_pullback",
        "hod_reclaim",
        "pullback_base",
        "abc_continuation",
        "first_pullback_reclaim",
        "level_breakout_reclaim",
        "level_breakout_watch",
        "runner_reclaim_continuation",
        "shallow_stair_continuation",
        "early_vwap_reclaim_scout",
    }
    scanner_names = {scanner.name for scanner in cfg.scanners}
    assert {
        "momentum_burst",
        "bull_flag",
        "flat_top_breakout",
        "opening_range_breakout",
        "level_breakout_watch",
        "runner_reclaim_continuation",
        "shallow_stair_continuation",
        "early_vwap_reclaim_scout",
    }.issubset(scanner_names)


def test_scalping_factory_can_attach_momentum_burst_verifier_for_backtest_experiment() -> None:
    pipeline = create_scalping_pipeline(momentum_burst_live_enabled=True)
    cfg = pipeline._router.get_config(TradingStyle.SCALPING)
    assert cfg is not None

    assert "momentum_burst" in cfg.verifiers


def test_level_breakout_watch_is_never_live_without_verifier() -> None:
    bars = [_bar("HOT", 4.0), _bar("HOT", 4.1), _bar("HOT", 4.2)]
    scanner = _OneHitScanner("level_breakout_watch", "level_breakout_watch")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )

    result = pipeline.run_cycle({"HOT": bars})

    assert result.scan_hits[0].criteria["setup_tier"] == "watch only"
    assert result.fills == []
    assert pipeline.scan_rejections["HOT"].startswith("watch only:")

def test_hod_gate_allows_structured_hot_watch_bypass(monkeypatch) -> None:
    bars = [_bar("HOT", 10.0), _bar("HOT", 10.2), _bar("HOT", 10.4)]
    scanner = _OneHitScanner("first_pullback_reclaim", "first_pullback_reclaim")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"first_pullback_reclaim": _SignalVerifier()},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )
    pipeline.set_hod_entry_gate(
        lambda symbol: False,
        require=True,
        bypass_checker=lambda signal: (
            signal.scan_result is not None
            and signal.scan_result.scanner_name == "first_pullback_reclaim"
        ),
    )
    monkeypatch.setattr("daytrading.pipeline.engine.check_entry_quality", lambda *a, **k: None)

    result = pipeline.run_cycle({"HOT": bars})

    assert len(result.fills) == 1
    assert result.fills[0].symbol == "HOT"


def test_post_guard_reject_is_labeled_for_dashboard(monkeypatch) -> None:
    bars = [
        _bar("HOT", 5.00, volume=20_000, open_=4.99, high=5.01, low=4.98),
        _bar("HOT", 5.00, volume=20_000, open_=4.99, high=5.01, low=4.98),
        _bar("HOT", 5.00, volume=20_000, open_=4.99, high=5.01, low=4.98),
        _bar("HOT", 5.00, volume=20_000, open_=4.99, high=5.01, low=4.98),
        _bar("HOT", 5.00, volume=20_000, open_=4.99, high=5.01, low=4.98),
        _bar("HOT", 5.05, volume=10_000, open_=5.01, high=5.06, low=5.00),
    ]
    scanner = _OneHitScanner("flat_top_breakout", "flat_top_breakout")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"flat_top_breakout": _SignalVerifier()},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )
    monkeypatch.setattr("daytrading.pipeline.engine.check_entry_quality", lambda *a, **k: None)

    result = pipeline.run_cycle({"HOT": bars})

    assert result.rejected_orders == 1
    assert result.rejection_details[0]["reason"].startswith("post-guard:")
    assert pipeline.scan_rejections["HOT"].startswith("post-guard:")


def test_repeated_rule_reject_uses_short_cooldown() -> None:
    bars = [_bar("HOT", 10.0), _bar("HOT", 10.1), _bar("HOT", 10.2)]
    scanner = _OneHitScanner("vwap_pullback", "vwap_pullback")
    verifier = _RejectVerifier("late pullback tape too slow 5000 recent volume")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"vwap_pullback": verifier},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )

    first = pipeline.run_cycle({"HOT": bars}, now=TS)
    second = pipeline.run_cycle({"HOT": bars}, now=TS)

    assert verifier.calls == 1
    assert first.scan_hits
    assert second.scan_hits
    assert pipeline.scan_rejections["HOT"].startswith("cached reject:")


def test_reject_cooldown_rechecks_after_material_price_change() -> None:
    scanner = _OneHitScanner("vwap_pullback", "vwap_pullback")
    verifier = _RejectVerifier("late pullback tape too slow 5000 recent volume")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"vwap_pullback": verifier},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )

    pipeline.run_cycle({"HOT": [_bar("HOT", 10.0), _bar("HOT", 10.1)]}, now=TS)
    pipeline.run_cycle({"HOT": [_bar("HOT", 10.0), _bar("HOT", 10.25)]}, now=TS)

    assert verifier.calls == 2


def test_watch_pattern_reject_does_not_cooldown_live_reclaim_pattern(monkeypatch) -> None:
    bars = [_bar("HOT", 18.0), _bar("HOT", 18.3), _bar("HOT", 18.6)]
    scanner = _OneHitScanner("level_breakout_watch", "level_breakout_watch")
    verifier = _PatternSwitchVerifier()
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"level_breakout_watch": verifier},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )
    monkeypatch.setattr("daytrading.pipeline.engine.check_entry_quality", lambda *a, **k: None)

    first = pipeline.run_cycle({"HOT": bars}, now=TS)
    scanner._pattern = "level_breakout_reclaim"
    second = pipeline.run_cycle({"HOT": bars}, now=TS)

    assert verifier.calls == 2
    assert first.fills == []
    assert second.signals
    assert second.signals[0].scan_result.criteria["pattern"] == "level_breakout_reclaim"
    assert not pipeline.scan_rejections.get("HOT", "").startswith("cached reject:")


def test_watch_only_reject_is_rechecked_without_cache() -> None:
    bars = [_bar("HOT", 18.0), _bar("HOT", 18.3), _bar("HOT", 18.6)]
    scanner = _OneHitScanner("level_breakout_watch", "level_breakout_watch")
    verifier = _RejectVerifier("watch only: level breakout has not reclaimed")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"level_breakout_watch": verifier},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )

    pipeline.run_cycle({"HOT": bars}, now=TS)
    pipeline.run_cycle({"HOT": bars}, now=TS)

    assert verifier.calls == 2
    assert not pipeline.scan_rejections["HOT"].startswith("cached reject:")


def test_stale_data_reject_is_not_cached() -> None:
    bars = [_bar("HOT", 10.0), _bar("HOT", 10.1)]
    scanner = _OneHitScanner("pullback_base", "pullback_base")
    verifier = _RejectVerifier("stale data (600s old, max=300s)")
    pipeline = TradingPipeline(
        scanners=[scanner],
        verifiers={"pullback_base": verifier},
        broker=PaperBroker(),
        portfolio=PortfolioState(cash=50_000.0, positions={}),
    )

    pipeline.run_cycle({"HOT": bars}, now=TS)
    pipeline.run_cycle({"HOT": bars}, now=TS)

    assert verifier.calls == 2
    assert not pipeline.scan_rejections["HOT"].startswith("cached reject:")
