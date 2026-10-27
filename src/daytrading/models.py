from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    NEW = "new"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class SignalAction(str, Enum):
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    SCALE_UP_LONG = "scale_up_long"
    SCALE_UP_SHORT = "scale_up_short"
    REENTER_LONG = "reenter_long"
    REENTER_SHORT = "reenter_short"
    SKIP = "skip"


class Timeframe(str, Enum):
    TICK = "tick"
    SEC_1 = "1s"
    SEC_5 = "5s"
    SEC_10 = "10s"
    SEC_15 = "15s"
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    HOUR_1 = "1h"
    DAILY = "1d"


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    SIGNAL_EXIT = "signal_exit"
    MANUAL = "manual"


class TradingStyle(str, Enum):
    SCALPING = "scalping"          # seconds to minutes, tiny edge, high frequency
    DAY_TRADING = "day_trading"    # minutes to hours, intraday only
    SWING = "swing"                # days to weeks
    NOT_TRADEABLE = "not_tradeable"  # too wide spread, too thin, too choppy


# ---------------------------------------------------------------------------
# Market data types
# ---------------------------------------------------------------------------

class BarSource(str, Enum):
    UNKNOWN = "unknown"
    CANONICAL_INTERVAL = "canonical_interval"
    ROLLING_SNAPSHOT = "rolling_snapshot"
    PRICE_MARK = "price_mark"


@dataclass(frozen=True)
class Bar:
    """OHLCV value. Provenance and completion must be checked before history use."""

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: Timeframe = Timeframe.MIN_1
    provenance: BarSource = BarSource.UNKNOWN


@dataclass(frozen=True)
class CompletedBar:
    """A bar and the exchange event time at which it actually completed."""

    bar: Bar
    closed_at: datetime


@dataclass(frozen=True)
class ConfirmationWindow:
    """Immutable execution context, separate from minute/session quality bars."""

    bars: tuple[CompletedBar, ...]
    observed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "bars", tuple(self.bars))


@dataclass(frozen=True)
class Quote:
    """Best bid/ask snapshot — essential for scalping spread analysis."""

    symbol: str
    ts: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        mid = (self.bid + self.ask) / 2.0
        return (self.spread / mid * 100.0) if mid > 0 else 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class Tick:
    """Single trade print from the tape."""

    symbol: str
    ts: datetime
    price: float
    size: float
    side: Side  # aggressor side


# ---------------------------------------------------------------------------
# Scanner / signal types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketRegime:
    """Classifier output — describes a symbol's current behavior."""

    symbol: str
    ts: datetime
    style: TradingStyle
    confidence: float                     # 0.0–1.0

    # underlying metrics the classifier used
    volatility_pct: float                 # ATR as % of price
    spread_pct: float                     # bid-ask spread as % of mid
    relative_volume: float                # current vol / average vol
    trend_strength: float                 # 0 = choppy, 1 = strong trend
    avg_bar_range_pct: float              # average bar H-L as % of price
    liquidity_score: float                # 0 = illiquid, 1 = very liquid

    reasons: List[str] = field(default_factory=list)  # human-readable explanation


@dataclass(frozen=True)
class ScanResult:
    """Output of a scanner — a symbol that passed screening criteria."""

    symbol: str
    scanner_name: str
    ts: datetime
    score: float
    criteria: Dict[str, Any] = field(default_factory=dict)
    bars: List[Bar] = field(default_factory=list)
    confirmation_window: Optional[ConfirmationWindow] = None


@dataclass(frozen=True)
class TradeSignal:
    """Strategy-verified entry/exit signal ready for execution."""

    symbol: str
    action: SignalAction
    quantity: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_offset: Optional[float] = None
    max_hold_seconds: Optional[int] = None
    reason: str = ""
    scan_result: Optional[ScanResult] = None
    trend_strength: float = 0.5


# ---------------------------------------------------------------------------
# Order / fill types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    # Optional absolute execution ceiling/floor, preserved through repricing.
    hard_limit_price: Optional[float] = None
    # Optional original setup deadline: client cancels outstanding BUY shares.
    expires_at: Optional[datetime] = None
    # Live liquidation evidence; repricing/cancellation cannot refresh its age.
    liquidation_quote_ts: Optional[datetime] = None


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: Side
    quantity: float
    price: float
    ts: datetime
    commission: float = 0.0
    order_id: str = ""
    cumulative_quantity: Optional[float] = None
    cumulative_notional: Optional[float] = None
    account_id: str = ""
    execution_reason: str = ""


# ---------------------------------------------------------------------------
# Portfolio / position types
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    entry_ts: Optional[datetime] = None

    def market_value(self, last_price: float) -> float:
        return self.quantity * last_price

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0

    def unrealized_pnl(self, last_price: float) -> float:
        return self.quantity * (last_price - self.avg_price)


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
