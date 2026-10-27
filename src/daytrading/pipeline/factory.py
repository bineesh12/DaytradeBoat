"""Pipeline factories — one-call setup for common configurations.

Start with scalping on $1–$20 US equities. Day trading and swing
factories can be added later once scalping is proven.
"""

from __future__ import annotations

from typing import Optional, Sequence

from daytrading.classifier.regime import MarketRegimeClassifier
from daytrading.classifier.router import AdaptiveRouter, StyleConfig
from daytrading.execution.broker import Broker, PaperBroker
from daytrading.exits.manager import ExitManager
from daytrading.exits.scaler import (
    PositionScaler,
    ReentryDetector,
    ReentryConfig,
    ScaleUpConfig,
)
from daytrading.pipeline.engine import TradingPipeline
from daytrading.scanner.scalping.momentum_burst import MomentumBurstScanner
from daytrading.scanner.scalping.bull_flag import BullFlagScanner
from daytrading.scanner.scalping.flat_top_breakout import FlatTopBreakoutScanner
from daytrading.scanner.scalping.vwap_pullback import VWAPPullbackScanner
from daytrading.scanner.scalping.opening_range_breakout import OpeningRangeBreakoutScanner
from daytrading.scanner.scalping.hod_reclaim import HODReclaimScanner
from daytrading.scanner.scalping.pullback_base import PullbackBaseScanner
from daytrading.scanner.scalping.abc_continuation import ABCContinuationScanner
from daytrading.scanner.scalping.first_pullback_reclaim import FirstPullbackReclaimScanner
from daytrading.scanner.scalping.level_breakout_reclaim import LevelBreakoutReclaimScanner
from daytrading.scanner.scalping.level_breakout_watch import LevelBreakoutWatchScanner
from daytrading.scanner.scalping.runner_reclaim_continuation import RunnerReclaimContinuationScanner
from daytrading.scanner.scalping.shallow_stair_continuation import ShallowStairContinuationScanner
from daytrading.scanner.scalping.early_vwap_reclaim_scout import EarlyVWAPReclaimScoutScanner
from daytrading.strategy.scalping.momentum_pattern import MomentumPatternVerifier
from daytrading.models import PortfolioState, TradingStyle


def create_scalping_pipeline(
    *,
    initial_cash: float = 25_000.0,
    commission_per_share: float = 0.0,

    # Price range
    min_price: float = 1.0,
    max_price: float = 20.0,

    # Deep-pullback tolerance (verifier "too far from HOD" gate)
    late_pullback_max_hod_pct: float = 12.0,
    late_pullback_max_hod_other_pct: float = 10.0,

    # Experimental conservative fresh-VWAP-reclaim scout (default off)
    fresh_vwap_reclaim_scout_enabled: bool = False,
    fresh_vwap_reclaim_scout_max_float: float = 20_000_000.0,
    vwap_reclaim_scout_enabled: bool = False,
    level_breakout_scout_enabled: bool = False,
    level_breakout_scout_min_session_move_pct: float = 3.0,
    momentum_burst_live_enabled: bool = False,
    entry_patterns: Optional[Sequence[str]] = None,
    prepared_reclaim_enabled: bool = False,
    unified_micro_entry_enabled: bool = False,
    micro_pullback_only_enabled: Optional[bool] = None,
    micro_minimum_room_r: float = 1.0,
    coordinated_micro_flow_enabled: bool = False,
    max_entry_risk_pct: float = 0.06,
    level_capped_entry_enabled: bool = False,

    # Runner back-half trail (post-partial). Wider rides continuation runners
    # further but gives back more on top-and-fade names. See StrategyConfig.
    runner_trail_pct: float = 0.03,
    runner_min_confirm_pct: float = 0.018,
    runner_trail_adaptive: bool = False,
    runner_trail_atr_mult: float = 2.5,
    runner_trail_cap: float = 0.10,
    runner_give_room_after_partial: bool = False,
    step_trail_exit_enabled: bool = False,
    step_trail_pct: float = 0.025,

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
    pattern_max_dollar_risk: float = 50.0,

    # Classifier
    min_avg_volume: float = 5_000,
    high_liquidity_volume: float = 500_000,
    scalp_max_spread_pct: float = 0.15,

    portfolio: Optional[PortfolioState] = None,
    broker: Optional[Broker] = None,
    float_checker: object = None,
    enable_daily_loser_blacklist: bool = False,
    daily_loser_blacklist_min_loss: float = 50.0,
    daily_loser_blacklist_max_losses: int = 2,
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

    # --- Scanners ---
    # All scanners stay active for watch/shadow learning. Only the live
    # verifier map below can turn a hit into an order candidate.
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
    pullback_base_scanner = PullbackBaseScanner(
        min_price=min_price,
        max_price=max_price,
        max_base_range_pct=5.0,
    )
    abc_scanner = ABCContinuationScanner(
        min_price=min_price,
        max_price=max_price,
    )
    first_pullback_scanner = FirstPullbackReclaimScanner(
        min_price=min_price,
        max_price=max_price,
    )
    level_breakout_scanner = LevelBreakoutReclaimScanner(
        min_price=min_price,
        max_price=max_price,
    )
    level_breakout_watch_scanner = LevelBreakoutWatchScanner(
        min_price=min_price,
        max_price=max_price,
        min_live_scout_session_move_pct=level_breakout_scout_min_session_move_pct,
        live_scout_enabled=level_breakout_scout_enabled,
    )
    runner_reclaim_scanner = RunnerReclaimContinuationScanner(
        min_price=min_price,
        max_price=max_price,
    )
    shallow_stair_scanner = ShallowStairContinuationScanner(
        min_price=min_price,
        max_price=max_price,
    )
    early_vwap_reclaim_scanner = EarlyVWAPReclaimScoutScanner(
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
        late_pullback_max_hod_pct=late_pullback_max_hod_pct,
        late_pullback_max_hod_other_pct=late_pullback_max_hod_other_pct,
        fresh_vwap_reclaim_scout_enabled=fresh_vwap_reclaim_scout_enabled,
        fresh_vwap_reclaim_scout_max_float=fresh_vwap_reclaim_scout_max_float,
        vwap_reclaim_scout_enabled=vwap_reclaim_scout_enabled,
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
        pullback_base_scanner, abc_scanner, first_pullback_scanner,
        level_breakout_scanner, level_breakout_watch_scanner,
        runner_reclaim_scanner,
        shallow_stair_scanner,
        early_vwap_reclaim_scanner,
    ]

    live_verifiers = {
        "vwap_pullback": pattern_verifier,
        "hod_reclaim": pattern_verifier,
        "pullback_base": pattern_verifier,
        "abc_continuation": pattern_verifier,
        "first_pullback_reclaim": pattern_verifier,
        "level_breakout_reclaim": pattern_verifier,
        "level_breakout_watch": pattern_verifier,
        "runner_reclaim_continuation": pattern_verifier,
        "shallow_stair_continuation": pattern_verifier,
        "early_vwap_reclaim_scout": pattern_verifier,
    }
    if entry_patterns is not None and 'micro_pullback_reclaim' in entry_patterns:
        if getattr(broker, '_paper', False) is not True:
            raise ValueError('micro_pullback_reclaim automatic execution requires PAPER broker')
        from daytrading.strategy.scalping.micro_pullback_runtime import MicroPullbackScanner, MicroPullbackVerifier
        from daytrading.strategy.scalping.micro_pullback import MicroRiskLimits
        micro_flow = None
        if coordinated_micro_flow_enabled:
            from daytrading.strategy.scalping.micro_flow import MicroFlowCollector
            micro_flow = MicroFlowCollector()
        all_scanners.append(MicroPullbackScanner(flow=micro_flow))
        live_verifiers['micro_pullback_reclaim'] = MicroPullbackVerifier(MicroRiskLimits(
            min_price=min_price, max_price=max_price,
            max_dollar_risk=pattern_max_dollar_risk, max_order_shares=max_order_shares,
            max_stop_pct=max_entry_risk_pct), float_checker, flow=micro_flow)
    if momentum_burst_live_enabled:
        live_verifiers["momentum_burst"] = pattern_verifier
        # The engine classifies A+ vs watch-only from module-level sets, so adding
        # the verifier is not enough — promote momentum_burst there too. The
        # backtest does this via a temp context manager; for a single live
        # pipeline we set it for the process. Idempotent.
        import daytrading.pipeline.engine as _engine
        _engine.LIVE_A_PLUS_SCANNERS = frozenset({*_engine.LIVE_A_PLUS_SCANNERS, "momentum_burst"})
        _engine.WATCH_ONLY_SCANNERS = frozenset(
            s for s in _engine.WATCH_ONLY_SCANNERS if s != "momentum_burst"
        )

    if unified_micro_entry_enabled and (not prepared_reclaim_enabled or not coordinated_micro_flow_enabled):
        raise ValueError('unified micro entry requires both PAPER micro paths and coordinated evidence')
    if prepared_reclaim_enabled:
        if getattr(broker, "_paper", False) is not True:
            raise ValueError("prepared reclaim automatic execution requires PAPER broker")
        if entry_patterns is None or "micro_pullback_reclaim" not in entry_patterns:
            raise ValueError("prepared reclaim requires explicit existing micro entry policy")
        from daytrading.strategy.scalping.event_reclaim import NAME, EventReclaimScanner, EventReclaimVerifier
        limits = live_verifiers["micro_pullback_reclaim"].limits
        all_scanners.append(EventReclaimScanner(flow=micro_flow if unified_micro_entry_enabled else None))
        live_verifiers[NAME] = EventReclaimVerifier(limits, float_checker, flow=micro_flow if unified_micro_entry_enabled else None)
        if unified_micro_entry_enabled:
            for name in (NAME,"micro_pullback_reclaim"):
                live_verifiers[name].entry_profile=True
            for scanner in all_scanners:
                if scanner.name=="micro_pullback_reclaim":scanner.entry_profile=True
        entry_patterns = tuple(entry_patterns) + (NAME,)
    from daytrading.strategy.entry_admission import MICRO_ROUTES
    micro_only = unified_micro_entry_enabled if micro_pullback_only_enabled is None else micro_pullback_only_enabled
    if micro_only:
        if not coordinated_micro_flow_enabled or not unified_micro_entry_enabled:
            raise ValueError('micro-only entries require unified coordinated micro evidence')
        if not 0 < micro_minimum_room_r < float('inf'):
            raise ValueError('micro minimum room R must be finite and positive')
        entry_patterns = tuple(name for name in (entry_patterns or ()) if name in MICRO_ROUTES)
    if entry_patterns is not None:
        allowed = set(entry_patterns)
        unknown = allowed - set(live_verifiers)
        if unknown:
            raise ValueError("Unknown entry patterns: " + ", ".join(sorted(unknown)))
        live_verifiers = {name: verifier for name, verifier in live_verifiers.items() if name in allowed}

    scalp_config = StyleConfig(
        scanners=all_scanners,
        verifiers=live_verifiers,
    )

    router = AdaptiveRouter(
        classifier=classifier,
        style_configs={TradingStyle.SCALPING: scalp_config},
    )

    # --- Broker ---
    broker = broker or PaperBroker(commission_per_share=commission_per_share)

    # --- Re-entry detector ---
    # After a full profitable exit, keep watching the same runner for a
    # pullback + reclaim. This helps avoid missing VERU-style second legs.
    reentry_detector = ReentryDetector(ReentryConfig(
        enabled=True,
        cooldown_seconds=60.0,
        max_reentries=2,
        reentry_size_pct=0.35,
        min_continuation_cents=5.0,
        pullback_max_cents=25.0,
        stop_cents=5.0,
        trail_cents=3.0,
        max_hold_seconds=120,
        require_clean_continuation_profile=True,
        min_pullback_depth_pct=1.0,
        max_pullback_depth_pct=10.0,
        max_base_range_pct=7.0,
        max_reentry_risk_pct=2.5,
    ))
    runner_readd_scaler = PositionScaler(ScaleUpConfig(
        max_scale_ups=2,
        size_decay=0.6,
        min_profit_cents=8.0,
        pullback_pct=0.8,
        bounce_pct=0.45,
        stop_advance_cents=3.0,
        require_protected_runner=True,
        require_clean_pullback_profile=True,
        min_pullback_depth_pct=1.0,
        max_pullback_depth_pct=10.0,
        max_base_range_pct=7.0,
        max_add_risk_pct=2.5,
    ))

    # --- Pipeline ---
    pipeline = TradingPipeline(
        scanners=all_scanners,
        verifiers=live_verifiers,
        broker=broker,
        portfolio=portfolio,
        exit_manager=ExitManager(
            max_unrealized_loss=pattern_max_dollar_risk,
            runner_trail_pct=runner_trail_pct,
            runner_min_confirm_pct=runner_min_confirm_pct,
            runner_trail_adaptive=runner_trail_adaptive,
            runner_trail_atr_mult=runner_trail_atr_mult,
            runner_trail_cap=runner_trail_cap,
            runner_give_room_after_partial=runner_give_room_after_partial,
            step_trail_exit_enabled=step_trail_exit_enabled,
            step_trail_pct=step_trail_pct,
        ),
        router=router,
        scaler=runner_readd_scaler if entry_patterns is None else None,
        reentry_detector=reentry_detector if entry_patterns is None else None,
        max_positions=max_positions,
        max_position_shares=max_position_shares,
        max_order_shares=max_order_shares,
        max_dollar_risk_per_trade=pattern_max_dollar_risk,
        enable_daily_loser_blacklist=enable_daily_loser_blacklist,
        daily_loser_blacklist_min_loss=daily_loser_blacklist_min_loss,
        daily_loser_blacklist_max_losses=daily_loser_blacklist_max_losses,
        level_capped_entry_enabled=level_capped_entry_enabled,
    )

    pipeline._unified_micro_entry = bool(unified_micro_entry_enabled)
    pipeline._micro_pullback_only = bool(micro_only)
    pipeline._micro_minimum_room_r = float(micro_minimum_room_r)
    pipeline._entry_policy.micro_pullback_only = bool(micro_only)
    pipeline._entry_policy.minimum_room_r = float(micro_minimum_room_r)
    def final_admission(signal, order, price):
        hit = getattr(signal, 'scan_result', None)
        if not micro_only and (hit is None or hit.scanner_name not in MICRO_ROUTES):
            return None
        from daytrading.strategy.submission import final_check, persist
        verifier = pipeline._verifiers.get(hit.scanner_name) if hit else None
        quotes = getattr(verifier, '_quote_buffer', {}).get(order.symbol, ())
        clock = getattr(broker, '_entry_deadline_clock', None) or (lambda:getattr(pipeline, '_market_now', None))
        reason,evidence = final_check(signal,order,policy=pipeline._entry_policy,
            bars=hit.bars if hit else (),quote=quotes[-1] if quotes else None,
            ticks=getattr(verifier,'_tick_buffer',{}).get(order.symbol,()),
            now=clock(),limits=getattr(verifier,'limits',None),boundary='pipeline_broker_submission')
        return persist(evidence,getattr(pipeline,'_entry_evidence_recorder',None),clock) or reason
    broker._strategy_admission = final_admission
    pipeline._entry_policy.entry_patterns = frozenset(entry_patterns) if entry_patterns is not None else None
    pipeline._entry_policy.min_price = min_price
    pipeline._entry_policy.max_price = max_price
    if coordinated_micro_flow_enabled:
        if 'micro_pullback_reclaim' not in live_verifiers:
            raise ValueError('coordinated flow requires the existing PAPER micro pattern')
        pipeline._coordinated_micro_flow = micro_flow
        pipeline._entry_policy.micro_flow_provider = micro_flow.quote_samples
    return pipeline
