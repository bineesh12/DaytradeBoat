"""Multi-timeframe bar aggregation.

Builds higher-timeframe bars from lower-timeframe data:
  - 5-minute bars from 1-minute bars (context layer)
  - 10-second bars from streaming ticks (execution layer)

Usage in the trading hierarchy:
  5-min  → Is the stock trending? (context)
  1-min  → Is there a pattern?   (setup)
  10-sec → Best moment to enter   (execution)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from daytrading.models import Bar, Tick, Timeframe

logger = logging.getLogger(__name__)


class BarAggregator:
    """Aggregates 1-min bars into 5-min bars and ticks into 10-sec bars."""

    def __init__(self, max_5m_bars: int = 80, max_10s_bars: int = 120) -> None:
        self._max_5m = max_5m_bars
        self._max_10s = max_10s_bars

        self._5m_bars: Dict[str, List[Bar]] = defaultdict(list)
        self._10s_bars: Dict[str, List[Bar]] = defaultdict(list)

        self._10s_accum: Dict[str, _TickAccumulator] = {}

    # ------------------------------------------------------------------
    # 5-minute bars from 1-minute bars
    # ------------------------------------------------------------------

    def build_5m_bars(self, symbol: str, bars_1m: Sequence[Bar]) -> List[Bar]:
        """Build 5-minute bars from a sequence of 1-minute bars.

        Groups by 5-minute boundary (e.g., 9:30-9:34, 9:35-9:39, etc.)
        and produces one 5-min OHLCV bar per group.
        """
        if len(bars_1m) < 5:
            return list(self._5m_bars.get(symbol, []))

        groups: Dict[str, List[Bar]] = {}
        for b in bars_1m:
            if b.ts is None:
                continue
            key = _5m_bucket_key(b.ts)
            groups.setdefault(key, []).append(b)

        result = []
        for key in sorted(groups.keys()):
            group = groups[key]
            if not group:
                continue
            result.append(Bar(
                symbol=symbol,
                ts=group[0].ts,
                open=group[0].open,
                high=max(b.high for b in group),
                low=min(b.low for b in group),
                close=group[-1].close,
                volume=sum(b.volume for b in group),
                timeframe=Timeframe.MIN_5,
            ))

        self._5m_bars[symbol] = result[-self._max_5m:]
        return self._5m_bars[symbol]

    def get_5m_bars(self, symbol: str) -> List[Bar]:
        return list(self._5m_bars.get(symbol, []))

    # ------------------------------------------------------------------
    # 10-second bars from streaming ticks
    # ------------------------------------------------------------------

    def on_tick(self, tick: Tick) -> Optional[Bar]:
        """Feed a tick and return a completed 10-sec bar if the window closed.

        Returns None if the current 10-sec window is still accumulating.
        """
        sym = tick.symbol
        if sym not in self._10s_accum:
            self._10s_accum[sym] = _TickAccumulator(sym)

        accum = self._10s_accum[sym]
        completed = accum.add(tick)

        if completed is not None:
            buf = self._10s_bars[sym]
            buf.append(completed)
            if len(buf) > self._max_10s:
                self._10s_bars[sym] = buf[-self._max_10s:]
            return completed
        return None

    def get_10s_bars(self, symbol: str) -> List[Bar]:
        return list(self._10s_bars.get(symbol, []))

    def get_latest_10s(self, symbol: str, count: int = 6) -> List[Bar]:
        """Get the last N 10-sec bars (default 6 = last 60 seconds)."""
        bars = self._10s_bars.get(symbol, [])
        return list(bars[-count:]) if bars else []

    # ------------------------------------------------------------------
    # Bulk update for all symbols
    # ------------------------------------------------------------------

    def update_all_5m(self, universe: Dict[str, List[Bar]]) -> None:
        """Rebuild 5-min bars for every symbol in the universe."""
        for sym, bars_1m in universe.items():
            self.build_5m_bars(sym, bars_1m)

    def clear_symbol(self, symbol: str) -> None:
        self._5m_bars.pop(symbol, None)
        self._10s_bars.pop(symbol, None)
        self._10s_accum.pop(symbol, None)


# ======================================================================
# Internal helpers
# ======================================================================

def _5m_bucket_key(ts: datetime) -> str:
    """Return a string key for the 5-minute bucket this timestamp falls in."""
    minute_bucket = (ts.minute // 5) * 5
    return f"{ts.date()}_{ts.hour:02d}:{minute_bucket:02d}"


class _TickAccumulator:
    """Accumulates ticks into a single 10-second bar."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._window_start: Optional[datetime] = None
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._close = 0.0
        self._volume = 0.0

    def add(self, tick: Tick) -> Optional[Bar]:
        """Add a tick. Returns a completed Bar if the 10-sec window closed."""
        if self._window_start is None:
            self._start_new(tick)
            return None

        elapsed = (tick.ts - self._window_start).total_seconds()

        if elapsed >= 10.0:
            completed = self._emit()
            self._start_new(tick)
            return completed

        self._high = max(self._high, tick.price)
        self._low = min(self._low, tick.price)
        self._close = tick.price
        self._volume += tick.size
        return None

    def _start_new(self, tick: Tick) -> None:
        self._window_start = tick.ts
        self._open = tick.price
        self._high = tick.price
        self._low = tick.price
        self._close = tick.price
        self._volume = tick.size

    def _emit(self) -> Bar:
        return Bar(
            symbol=self.symbol,
            ts=self._window_start,  # type: ignore[arg-type]
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            timeframe=Timeframe.SEC_10,
        )
