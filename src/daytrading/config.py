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
        self.max_price: float = _env_float("MAX_PRICE", 20.0)

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
