"""Configuration — loads from .env file and environment variables.

No pydantic dependency. Uses python-dotenv + os.environ.
All settings use the DAYTRADING_ prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get("DAYTRADING_" + name, default)


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v else default


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v else default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name).lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return default


@dataclass(frozen=True)
class StrategyConfig:
    """Tunable live-strategy knobs loaded once from env.

    Keep fast-moving trading thresholds here instead of scattering magic
    numbers across runner threads.
    """

    hot_watch_enabled: bool = True
    hot_watch_ttl_minutes: float = 8.0
    hot_watch_strong_ttl_minutes: float = 15.0
    hot_watch_runner_ttl_minutes: float = 25.0
    hot_watch_max_symbols: int = 40
    hot_watch_min_change_pct: float = 5.0
    hot_watch_min_day_volume: float = 200_000
    hot_watch_sub5_min_day_volume: float = 500_000
    hot_watch_min_score: float = 0.30
    hot_watch_setup_refresh_enabled: bool = True
    hot_watch_setup_refresh_max_pullback_pct: float = 4.0
    hot_watch_setup_refresh_min_recent_volume: float = 100_000
    max_watchlist_symbols: int = 50
    candidate_hydrate_queue_max: int = 500
    candidate_hydrate_batch_max: int = 10
    fast_scan_process_max: int = 80
    timed_entry_anchor_ttl_sec: float = 300.0
    missed_a_plus_chase_window_sec: float = 1800.0
    missed_a_plus_chase_pct_sub5: float = 0.035
    missed_a_plus_chase_pct_5plus: float = 0.025
    # Normal anti-chase cap: max % a fill may sit above the setup base before the
    # entry is rejected as a late chase. Cheap fast movers (< price tier) get more
    # room; pricier names stay tight. Now config-driven (was hardcoded 0.025/0.035,
    # $5 tier). Loosened to 0.05 / $10: on the live 10s-execution-timer path a
    # 6/12 mover basket netted +$76 vs +$28 tight — the extra room added only the
    # CUPR HOD reclaim (+$49) with no added losers (the 10s timer stays selective;
    # the looser-but-less-selective 1m path did add losers, so this only holds with
    # execution_timer_10s on, which is the live default).
    entry_chase_pct_low: float = 0.05
    entry_chase_pct_high: float = 0.025
    entry_chase_price_tier: float = 10.0
    # Runner back-half trail (post-partial), for confirmed runner candidates.
    # A 6/12 sweep: 3% locks top-and-fade winners (EDHL +$32) but stops CUPR
    # early (+$49); 8% rides CUPR's continuation (+$179) but gives EDHL back
    # (+$2). No single flat value wins both, so default stays tight (3%) and is
    # tunable. runner_min_confirm_pct = how far past the partial it must run to
    # earn the wider trail.
    runner_trail_pct: float = 0.03
    runner_min_confirm_pct: float = 0.018
    # Adaptive runner trail: scale trail width to the name's own recent 1m
    # volatility so wide-swinging runners (CUPR, ~2.7% bars) breathe while smooth
    # names (EDHL, ~1.4% bars) trail tight. trail = clamp(atr_mult * median bar
    # range, runner_trail_pct floor, runner_trail_cap). Off by default = flat.
    # Kept OFF: backtests on CUPR/EDHL/SUNE showed adaptive == flat 8% and it
    # gives back the back-half on fade/chop names (EDHL, SUNE) to help only the
    # rare continuation runner (CUPR). Volatility is the wrong signal (it widens
    # near tops). Needs a structure-aware formula + unbiased basket before on.
    runner_trail_adaptive: bool = False
    runner_trail_atr_mult: float = 2.5
    runner_trail_cap: float = 0.10
    tick_entry_enabled: bool = False
    tick_entry_confirm_count: int = 2
    tick_entry_max_above_anchor: float = 0.02
    # Deep-pullback tolerance: how far below HOD a late pullback may sit before
    # the verifier rejects it (was 8/10 — too tight, missed reclaim runners).
    late_pullback_max_hod_pct: float = 12.0
    late_pullback_max_hod_other_pct: float = 10.0
    # EXPERIMENTAL momentum-breakout mode (default OFF). Lets the fast breakout
    # scalp fire on a high-ABSOLUTE-volume breakout whose relative volume has
    # faded from an earlier peak (the VSME case). Paper-test via the scorecard.
    momentum_breakout_enabled: bool = False
    momentum_breakout_min_rvol: float = 0.4
    momentum_breakout_min_day_volume: float = 5_000_000
    # Only fire the breakout when recent tape is smooth enough that a stop holds
    # (per-bar range). This is the key guard — it skips the violent gappy tape
    # (VSME-style 6-11% bars) where stops slip, and the slippage eats the edge.
    momentum_breakout_max_bar_range_pct: float = 3.0
    # Caveat-1 fix: catch the early breakout/VWAP reclaim that scores just under
    # the 80 gate. On the breakout-scalp path only, when the mode is on and the
    # tape is smooth + high-volume, allow a score down to this floor.
    momentum_breakout_score_floor: float = 72.0
    # EXPERIMENTAL conservative fresh-VWAP-reclaim scout (default OFF). After a
    # dump candle, allow a reduced-size scout IF a fresh green base rebuilt above
    # VWAP on strong volume + low float (the DSY case). First dump still blocked;
    # gappy/failed reclaims (the GMM case) stay rejected. Paper-test the scorecard.
    fresh_vwap_reclaim_scout_enabled: bool = False
    fresh_vwap_reclaim_scout_max_float: float = 20_000_000
    # Earlier reduced-size level-breakout scout for smooth/liquid names that
    # would otherwise enter later via first_pullback_reclaim (CONL timing case).
    level_breakout_scout_enabled: bool = False
    level_breakout_scout_min_session_move_pct: float = 3.0

    @classmethod
    def from_env(cls) -> "StrategyConfig":
        return cls(
            hot_watch_enabled=_env_bool("HOT_WATCH_ENABLED", cls.hot_watch_enabled),
            hot_watch_ttl_minutes=_env_float("HOT_WATCH_TTL_MINUTES", cls.hot_watch_ttl_minutes),
            hot_watch_strong_ttl_minutes=_env_float(
                "HOT_WATCH_STRONG_TTL_MINUTES",
                cls.hot_watch_strong_ttl_minutes,
            ),
            hot_watch_runner_ttl_minutes=_env_float(
                "HOT_WATCH_RUNNER_TTL_MINUTES",
                cls.hot_watch_runner_ttl_minutes,
            ),
            hot_watch_max_symbols=_env_int("HOT_WATCH_MAX_SYMBOLS", cls.hot_watch_max_symbols),
            hot_watch_min_change_pct=_env_float(
                "HOT_WATCH_MIN_CHANGE_PCT",
                cls.hot_watch_min_change_pct,
            ),
            hot_watch_min_day_volume=_env_float(
                "HOT_WATCH_MIN_DAY_VOLUME",
                cls.hot_watch_min_day_volume,
            ),
            hot_watch_sub5_min_day_volume=_env_float(
                "HOT_WATCH_SUB5_MIN_DAY_VOLUME",
                cls.hot_watch_sub5_min_day_volume,
            ),
            hot_watch_min_score=_env_float("HOT_WATCH_MIN_SCORE", cls.hot_watch_min_score),
            hot_watch_setup_refresh_enabled=_env_bool(
                "HOT_WATCH_SETUP_REFRESH_ENABLED",
                cls.hot_watch_setup_refresh_enabled,
            ),
            hot_watch_setup_refresh_max_pullback_pct=_env_float(
                "HOT_WATCH_SETUP_REFRESH_MAX_PULLBACK_PCT",
                cls.hot_watch_setup_refresh_max_pullback_pct,
            ),
            hot_watch_setup_refresh_min_recent_volume=_env_float(
                "HOT_WATCH_SETUP_REFRESH_MIN_RECENT_VOLUME",
                cls.hot_watch_setup_refresh_min_recent_volume,
            ),
            max_watchlist_symbols=_env_int("MAX_WATCHLIST_SYMBOLS", cls.max_watchlist_symbols),
            candidate_hydrate_queue_max=_env_int(
                "CANDIDATE_HYDRATE_QUEUE_MAX",
                cls.candidate_hydrate_queue_max,
            ),
            candidate_hydrate_batch_max=_env_int(
                "CANDIDATE_HYDRATE_BATCH_MAX",
                cls.candidate_hydrate_batch_max,
            ),
            fast_scan_process_max=_env_int("FAST_SCAN_PROCESS_MAX", cls.fast_scan_process_max),
            timed_entry_anchor_ttl_sec=_env_float(
                "TIMED_ENTRY_ANCHOR_TTL_SEC",
                cls.timed_entry_anchor_ttl_sec,
            ),
            missed_a_plus_chase_window_sec=_env_float(
                "MISSED_A_PLUS_CHASE_WINDOW_SEC",
                cls.missed_a_plus_chase_window_sec,
            ),
            missed_a_plus_chase_pct_sub5=_env_float(
                "MISSED_A_PLUS_CHASE_PCT_SUB5",
                cls.missed_a_plus_chase_pct_sub5,
            ),
            missed_a_plus_chase_pct_5plus=_env_float(
                "MISSED_A_PLUS_CHASE_PCT_5PLUS",
                cls.missed_a_plus_chase_pct_5plus,
            ),
            entry_chase_pct_low=_env_float("ENTRY_CHASE_PCT_LOW", cls.entry_chase_pct_low),
            entry_chase_pct_high=_env_float("ENTRY_CHASE_PCT_HIGH", cls.entry_chase_pct_high),
            entry_chase_price_tier=_env_float(
                "ENTRY_CHASE_PRICE_TIER",
                cls.entry_chase_price_tier,
            ),
            runner_trail_pct=_env_float("RUNNER_TRAIL_PCT", cls.runner_trail_pct),
            runner_min_confirm_pct=_env_float(
                "RUNNER_MIN_CONFIRM_PCT",
                cls.runner_min_confirm_pct,
            ),
            runner_trail_adaptive=_env_bool(
                "RUNNER_TRAIL_ADAPTIVE",
                cls.runner_trail_adaptive,
            ),
            runner_trail_atr_mult=_env_float(
                "RUNNER_TRAIL_ATR_MULT",
                cls.runner_trail_atr_mult,
            ),
            runner_trail_cap=_env_float("RUNNER_TRAIL_CAP", cls.runner_trail_cap),
            tick_entry_enabled=_env_bool("TICK_ENTRY_ENABLED", cls.tick_entry_enabled),
            tick_entry_confirm_count=_env_int(
                "TICK_ENTRY_CONFIRM_COUNT",
                cls.tick_entry_confirm_count,
            ),
            tick_entry_max_above_anchor=_env_float(
                "TICK_ENTRY_MAX_ABOVE_ANCHOR",
                cls.tick_entry_max_above_anchor,
            ),
            late_pullback_max_hod_pct=_env_float(
                "LATE_PULLBACK_MAX_HOD_PCT",
                cls.late_pullback_max_hod_pct,
            ),
            late_pullback_max_hod_other_pct=_env_float(
                "LATE_PULLBACK_MAX_HOD_OTHER_PCT",
                cls.late_pullback_max_hod_other_pct,
            ),
            momentum_breakout_enabled=_env_bool(
                "MOMENTUM_BREAKOUT_ENABLED",
                cls.momentum_breakout_enabled,
            ),
            momentum_breakout_min_rvol=_env_float(
                "MOMENTUM_BREAKOUT_MIN_RVOL",
                cls.momentum_breakout_min_rvol,
            ),
            momentum_breakout_min_day_volume=_env_float(
                "MOMENTUM_BREAKOUT_MIN_DAY_VOLUME",
                cls.momentum_breakout_min_day_volume,
            ),
            momentum_breakout_max_bar_range_pct=_env_float(
                "MOMENTUM_BREAKOUT_MAX_BAR_RANGE_PCT",
                cls.momentum_breakout_max_bar_range_pct,
            ),
            momentum_breakout_score_floor=_env_float(
                "MOMENTUM_BREAKOUT_SCORE_FLOOR",
                cls.momentum_breakout_score_floor,
            ),
            fresh_vwap_reclaim_scout_enabled=_env_bool(
                "FRESH_VWAP_RECLAIM_SCOUT_ENABLED",
                cls.fresh_vwap_reclaim_scout_enabled,
            ),
            fresh_vwap_reclaim_scout_max_float=_env_float(
                "FRESH_VWAP_RECLAIM_SCOUT_MAX_FLOAT",
                cls.fresh_vwap_reclaim_scout_max_float,
            ),
            level_breakout_scout_enabled=_env_bool(
                "LEVEL_BREAKOUT_SCOUT_ENABLED",
                cls.level_breakout_scout_enabled,
            ),
            level_breakout_scout_min_session_move_pct=_env_float(
                "LEVEL_BREAKOUT_SCOUT_MIN_SESSION_MOVE_PCT",
                cls.level_breakout_scout_min_session_move_pct,
            ),
        )


class Settings:
    """Load from environment / .env for API keys and runtime knobs."""

    def __init__(self) -> None:
        self.strategy: StrategyConfig = StrategyConfig.from_env()

        # Alpaca API
        self.alpaca_api_key: str = _env("ALPACA_API_KEY")
        self.alpaca_secret_key: str = _env("ALPACA_SECRET_KEY")
        self.alpaca_paper: bool = _env_bool("ALPACA_PAPER", True)
        self.alpaca_feed: str = _env("ALPACA_FEED", "iex")

        # Capital
        self.initial_cash: float = _env_float("INITIAL_CASH", 25_000.0)
        self.commission_per_share: float = _env_float("COMMISSION_PER_SHARE", 0.0)

        # Price filter
        self.min_price: float = _env_float("MIN_PRICE", 1.0)
        self.max_price: float = _env_float("MAX_PRICE", 500.0)

        # Float cache (SQLite, days before Yahoo refresh)
        self.float_cache_ttl_days: int = _env_int("FLOAT_CACHE_TTL_DAYS", 7)

        # Trading modes
        self.enable_scalping: bool = _env_bool("ENABLE_SCALPING", True)
        self.enable_day_trading: bool = _env_bool("ENABLE_DAY_TRADING", False)
        self.enable_swing: bool = _env_bool("ENABLE_SWING", False)

        # Pipeline limits
        self.max_positions: int = _env_int("MAX_POSITIONS", 3)
        self.max_position_shares: float = _env_float("MAX_POSITION_SHARES", 1000)
        self.max_order_shares: float = _env_float("MAX_ORDER_SHARES", 500)

        # Scalping: scanner defaults
        self.scalp_min_burst_pct: float = _env_float("SCALP_MIN_BURST_PCT", 0.15)
        self.scalp_burst_period: int = _env_int("SCALP_BURST_PERIOD", 3)
        self.scalp_min_volume: float = _env_float("SCALP_MIN_VOLUME", 5_000)
        self.scalp_min_imbalance: float = _env_float("SCALP_MIN_IMBALANCE", 0.4)
        self.scalp_min_tape_speed: float = _env_float("SCALP_MIN_TAPE_SPEED", 5.0)
        self.scalp_max_spread_cents: float = _env_float("SCALP_MAX_SPREAD_CENTS", 2.0)

        # Scalping: entry/exit in cents
        self.scalp_momentum_target_cents: float = _env_float("SCALP_MOMENTUM_TARGET_CENTS", 5.0)
        self.scalp_momentum_stop_cents: float = _env_float("SCALP_MOMENTUM_STOP_CENTS", 3.0)
        self.scalp_momentum_trail_cents: float = _env_float("SCALP_MOMENTUM_TRAIL_CENTS", 2.0)
        self.scalp_momentum_max_hold_sec: int = _env_int("SCALP_MOMENTUM_MAX_HOLD_SEC", 120)
        self.scalp_momentum_size: float = _env_float("SCALP_MOMENTUM_SIZE", 500)

        self.scalp_tape_target_cents: float = _env_float("SCALP_TAPE_TARGET_CENTS", 4.0)
        self.scalp_tape_stop_cents: float = _env_float("SCALP_TAPE_STOP_CENTS", 2.0)
        self.scalp_tape_trail_cents: float = _env_float("SCALP_TAPE_TRAIL_CENTS", 1.5)
        self.scalp_tape_max_hold_sec: int = _env_int("SCALP_TAPE_MAX_HOLD_SEC", 60)
        self.scalp_tape_size: float = _env_float("SCALP_TAPE_SIZE", 500)

        # Classifier
        self.min_avg_volume: float = _env_float("MIN_AVG_VOLUME", 50_000)
        self.scalp_max_spread_pct: float = _env_float("SCALP_MAX_SPREAD_PCT", 0.15)

        # Risk — block re-entry on symbols that lost money today
        self.enable_daily_loser_blacklist: bool = _env_bool(
            "ENABLE_DAILY_LOSER_BLACKLIST", True,
        )
        self.max_dollar_risk_per_trade: float = _env_float(
            "MAX_DOLLAR_RISK_PER_TRADE", 50.0,
        )

        # Extended hours — allow scanning/entries 4:00 PM–8:00 PM ET (paper testing)
        self.after_hours_enabled: bool = _env_bool("AFTER_HOURS_ENABLED", False)

        # HOD Momentum alert scanner
        self.hod_momentum_max_float: float = _env_float(
            "HOD_MOMENTUM_MAX_FLOAT", 20_000_000,
        )
        self.hod_momentum_min_price: float = _env_float(
            "HOD_MOMENTUM_MIN_PRICE", 2.0,
        )
        self.hod_momentum_max_price: float = _env_float(
            "HOD_MOMENTUM_MAX_PRICE", 20.0,
        )
        self.hod_sub2_momentum_enabled: bool = _env_bool(
            "HOD_SUB2_MOMENTUM_ENABLED", True,
        )
        self.hod_sub2_momentum_min_price: float = _env_float(
            "HOD_SUB2_MOMENTUM_MIN_PRICE", 1.0,
        )
        self.hod_sub2_momentum_max_price: float = _env_float(
            "HOD_SUB2_MOMENTUM_MAX_PRICE", 2.0,
        )
        self.hod_sub2_momentum_min_change_pct: float = _env_float(
            "HOD_SUB2_MOMENTUM_MIN_CHANGE_PCT", 10.0,
        )
        self.hod_sub2_momentum_min_day_volume: float = _env_float(
            "HOD_SUB2_MOMENTUM_MIN_DAY_VOLUME", 1_000_000,
        )
        self.hod_sub2_momentum_max_float: float = _env_float(
            "HOD_SUB2_MOMENTUM_MAX_FLOAT", 10_000_000,
        )
        # Requires SIP feed (alpaca_feed=sip). Not the RT mover scanner — HOD tick only.
        self.hod_momentum_tick_enabled: bool = _env_bool(
            "HOD_MOMENTUM_TICK_ENABLED", True,
        )
        self.hod_momentum_volume_surge_ratio: float = _env_float(
            "HOD_MOMENTUM_VOLUME_SURGE_RATIO", 3.0,
        )
        self.hod_momentum_tick_cooldown_seconds: float = _env_float(
            "HOD_MOMENTUM_TICK_COOLDOWN_SECONDS", 30.0,
        )
        self.hod_momentum_alert_ttl_minutes: float = _env_float(
            "HOD_MOMENTUM_ALERT_TTL_MINUTES", 15.0,
        )
        self.hod_momentum_watchlist_ttl_minutes: float = _env_float(
            "HOD_MOMENTUM_WATCHLIST_TTL_MINUTES", 20.0,
        )
        self.hod_momentum_bar_pool_max: int = _env_int(
            "HOD_MOMENTUM_BAR_POOL_MAX", 250,
        )
        self.hod_pool_refresh_minutes: int = _env_int(
            "HOD_POOL_REFRESH_MINUTES", 10,
        )

        # Gentle network — batched Alpaca bar fetches
        self.bar_fetch_batch_size: int = _env_int("BAR_FETCH_BATCH_SIZE", 10)
        self.bar_fetch_batch_delay_sec: float = _env_float(
            "BAR_FETCH_BATCH_DELAY_SEC", 0.5,
        )
        self.hod_hydrate_batch_max: int = _env_int("HOD_HYDRATE_BATCH_MAX", 25)
        self.hod_seed_batch_size: int = _env_int("HOD_SEED_BATCH_SIZE", 10)
        self.hod_seed_max_per_minute: int = _env_int("HOD_SEED_MAX_PER_MINUTE", 100)
        self.hod_momentum_require_alert_for_entry: bool = _env_bool(
            "HOD_MOMENTUM_REQUIRE_ALERT_FOR_ENTRY", True,
        )
        self.hod_momentum_max_alert_rows: int = _env_int(
            "HOD_MOMENTUM_MAX_ALERT_ROWS", 200,
        )
        self.hod_momentum_min_session_change_pct: float = _env_float(
            "HOD_MOMENTUM_MIN_SESSION_CHANGE_PCT", 5.0,
        )
        self.hod_momentum_min_day_volume: float = _env_float(
            "HOD_MOMENTUM_MIN_DAY_VOLUME", 200_000,
        )
        self.hod_momentum_require_break_prior_day_high: bool = _env_bool(
            "HOD_MOMENTUM_REQUIRE_BREAK_PRIOR_DAY_HIGH", True,
        )
        self.hod_momentum_rth_only: bool = _env_bool(
            "HOD_MOMENTUM_RTH_ONLY", False,
        )
        self.hod_momentum_former_momo_enabled: bool = _env_bool(
            "HOD_MOMENTUM_FORMER_MOMO_ENABLED", True,
        )
        self.hod_momentum_former_momo_min_price: float = _env_float(
            "HOD_MOMENTUM_FORMER_MOMO_MIN_PRICE", 20.0,
        )
        self.hod_momentum_former_momo_min_change_pct: float = _env_float(
            "HOD_MOMENTUM_FORMER_MOMO_MIN_CHANGE_PCT", 3.0,
        )

        # Hot watch — early mover watch before HOD alert. This does not buy
        # directly; it only lets structured pullback patterns reach rules + ML.
        self.hot_watch_enabled: bool = self.strategy.hot_watch_enabled
        self.hot_watch_ttl_minutes: float = self.strategy.hot_watch_ttl_minutes
        self.hot_watch_strong_ttl_minutes: float = self.strategy.hot_watch_strong_ttl_minutes
        self.hot_watch_runner_ttl_minutes: float = self.strategy.hot_watch_runner_ttl_minutes
        self.hot_watch_max_symbols: int = self.strategy.hot_watch_max_symbols
        self.hot_watch_min_change_pct: float = self.strategy.hot_watch_min_change_pct
        self.hot_watch_min_day_volume: float = self.strategy.hot_watch_min_day_volume
        self.hot_watch_sub5_min_day_volume: float = self.strategy.hot_watch_sub5_min_day_volume
        self.hot_watch_min_score: float = self.strategy.hot_watch_min_score

        # Live memory caps. Keep these bounded so candidate hydration and bar
        # history do not overwhelm small GCP instances during busy mover days.
        self.max_watchlist_symbols: int = self.strategy.max_watchlist_symbols
        self.candidate_hydrate_queue_max: int = self.strategy.candidate_hydrate_queue_max
        self.candidate_hydrate_batch_max: int = self.strategy.candidate_hydrate_batch_max
        self.fast_scan_process_max: int = self.strategy.fast_scan_process_max

        # Tape-hot detection — volume spike on SIP tape triggers bar load + scan
        self.hod_tape_hot_volume_threshold: int = _env_int(
            "HOD_TAPE_HOT_VOLUME_THRESHOLD", 50_000,
        )
        self.hod_tape_hot_ttl_minutes: float = _env_float(
            "HOD_TAPE_HOT_TTL_MINUTES", 5.0,
        )
        self.hod_bar_load_workers: int = _env_int(
            "HOD_BAR_LOAD_WORKERS", 4,
        )
        self.hod_hydrate_top_n: int = _env_int(
            "HOD_HYDRATE_TOP_N", 100,
        )
        self.hod_scanner_debug: bool = _env_bool(
            "DAYTRADING_HOD_SCANNER_DEBUG", False,
        )
