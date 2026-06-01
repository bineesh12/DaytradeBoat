"""Configuration — loads from .env file and environment variables.

No pydantic dependency. Uses python-dotenv + os.environ.
All settings use the DAYTRADING_ prefix.
"""

from __future__ import annotations

import os


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


class Settings:
    """Load from environment / .env for API keys and runtime knobs."""

    def __init__(self) -> None:
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
            "HOD_MOMENTUM_BAR_POOL_MAX", 1000,
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
