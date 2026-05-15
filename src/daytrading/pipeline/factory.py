"""Pipeline factories — one-call setup for common configurations.

Start with scalping on $1–$20 US equities. Day trading and swing
factories can be added later once scalping is proven.
"""

from __future__ import annotations

from typing import Optional

from daytrading.classifier.regime import MarketRegimeClassifier
from daytrading.classifier.router import AdaptiveRouter, StyleConfig
from daytrading.execution.broker import PaperBroker
from daytrading.exits.manager import ExitManager
from daytrading.exits.scaler import ReentryDetector, ReentryConfig
from daytrading.pipeline.engine import TradingPipeline
from daytrading.scanner.scalping.momentum_burst import MomentumBurstScanner
from daytrading.scanner.scalping.bull_flag import BullFlagScanner
from daytrading.scanner.scalping.flat_top_breakout import FlatTopBreakoutScanner
from daytrading.scanner.scalping.vwap_pullback import VWAPPullbackScanner
from daytrading.scanner.scalping.opening_range_breakout import OpeningRangeBreakoutScanner
from daytrading.scanner.scalping.hod_reclaim import HODReclaimScanner
from daytrading.strategy.scalping.momentum_pattern import MomentumPatternVerifier
from daytrading.models import PortfolioState, TradingStyle


def create_scalping_pipeline(
    *,
    initial_cash: float = 25_000.0,
    commission_per_share: float = 0.0,

    # Price range
    min_price: float = 1.0,
    max_price: float = 20.0,

    # Position limits
    max_positions: int = 3,
    max_position_shares: float = 1000,
    max_order_shares: float = 500,

    # Momentum burst scanner (used as pre-filter for momentum stocks)
    min_burst_pct: float = 0.5,
    burst_period: int = 3,
    min_burst_volume: float = 10_000,

    # Bull flag scanner
    bull_flag_min_pole_pct: float = 1.5,
    bull_flag_max_pullback_retrace: float = 0.50,

    # Flat top breakout scanner
    flat_top_min_drive_pct: float = 1.0,
    flat_top_tolerance_pct: float = 0.3,

    # Momentum pattern verifier (bull flag + flat top)
    pattern_max_risk_per_share: float = 0.50,
    pattern_reward_risk_ratio: float = 2.0,
    pattern_trail_ticks: int = 5,
    pattern_max_hold_sec: int = 600,
    pattern_max_dollar_risk: float = 100.0,

    # Classifier
    min_avg_volume: float = 5_000,
    high_liquidity_volume: float = 500_000,
    scalp_max_spread_pct: float = 0.15,

    portfolio: Optional[PortfolioState] = None,
    float_checker: object = None,
) -> TradingPipeline:
    """Create a fully wired scalping pipeline for $1–$20 stocks.

    Returns a TradingPipeline ready to call run_cycle().

    Example::

        pipeline = create_scalping_pipeline(initial_cash=10_000)
        result = pipeline.run_cycle(universe)
        for fill in result.fills:
            print(f"FILLED {fill.side.value} {fill.symbol} {fill.quantity} @ {fill.price}")
    """

    if portfolio is None:
        portfolio = PortfolioState(cash=initial_cash)

    # --- Scanners (Warrior Trading: Bull Flag + Flat Top Breakout) ---
    momentum_scanner = MomentumBurstScanner(
        min_burst_pct=min_burst_pct,
        burst_period=burst_period,
        min_volume=min_burst_volume,
        min_price=min_price,
        max_price=max_price,
    )
    bull_flag_scanner = BullFlagScanner(
        min_pole_pct=bull_flag_min_pole_pct,
        max_pullback_retrace=bull_flag_max_pullback_retrace,
        min_price=min_price,
        max_price=max_price,
    )
    flat_top_scanner = FlatTopBreakoutScanner(
        min_drive_pct=flat_top_min_drive_pct,
        flat_tolerance_pct=flat_top_tolerance_pct,
        min_price=min_price,
        max_price=max_price,
    )
    vwap_pullback_scanner = VWAPPullbackScanner(
        min_price=min_price,
        max_price=max_price,
    )
    orb_scanner = OpeningRangeBreakoutScanner(
        min_price=min_price,
        max_price=max_price,
    )
    hod_scanner = HODReclaimScanner(
        min_price=min_price,
        max_price=max_price,
    )

    # --- Verifier (Warrior Trading momentum pattern: 2:1 R/R, pattern-based stops) ---
    pattern_verifier = MomentumPatternVerifier(
        max_risk_per_share=pattern_max_risk_per_share,
        reward_risk_ratio=pattern_reward_risk_ratio,
        trail_ticks=pattern_trail_ticks,
        max_hold_seconds=pattern_max_hold_sec,
        max_dollar_risk=pattern_max_dollar_risk,
        min_price=min_price,
        max_price=max_price,
        float_checker=float_checker,
    )

    # --- Classifier + Router ---
    classifier = MarketRegimeClassifier(
        min_price=min_price,
        max_price=max_price,
        enable_scalping=True,
        enable_day_trading=False,
        enable_swing=False,
        min_avg_volume=min_avg_volume,
        high_liquidity_volume=high_liquidity_volume,
        scalp_max_spread_pct=scalp_max_spread_pct,
    )

    all_scanners = [
        momentum_scanner, bull_flag_scanner, flat_top_scanner,
        vwap_pullback_scanner, orb_scanner, hod_scanner,
    ]

    scalp_config = StyleConfig(
        scanners=all_scanners,
        verifiers={
            "momentum_burst": pattern_verifier,
            "bull_flag": pattern_verifier,
            "flat_top_breakout": pattern_verifier,
            "vwap_pullback": pattern_verifier,
            "opening_range_breakout": pattern_verifier,
            "hod_reclaim": pattern_verifier,
        },
    )

    router = AdaptiveRouter(
        classifier=classifier,
        style_configs={TradingStyle.SCALPING: scalp_config},
    )

    # --- Broker ---
    broker = PaperBroker(commission_per_share=commission_per_share)

    # --- Re-entry detector (watch recently exited stocks for continuation) ---
    reentry_detector = ReentryDetector(ReentryConfig(
        enabled=True,
        cooldown_seconds=30.0,
        max_reentries=2,
        reentry_size_pct=0.5,
    ))

    # --- Pipeline ---
    return TradingPipeline(
        scanners=all_scanners,
        verifiers={
            "momentum_burst": pattern_verifier,
            "bull_flag": pattern_verifier,
            "flat_top_breakout": pattern_verifier,
            "vwap_pullback": pattern_verifier,
            "opening_range_breakout": pattern_verifier,
            "hod_reclaim": pattern_verifier,
        },
        broker=broker,
        portfolio=portfolio,
        exit_manager=ExitManager(),
        router=router,
        reentry_detector=reentry_detector,
        max_positions=max_positions,
        max_position_shares=max_position_shares,
        max_order_shares=max_order_shares,
    )
