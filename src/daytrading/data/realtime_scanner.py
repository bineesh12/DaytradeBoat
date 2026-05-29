"""Real-time market scanner — detects movers instantly from the trade stream.

Instead of polling snapshots every 30 seconds, this listens to every trade
across the entire market via the wildcard '*' subscription and detects:
  - Volume bursts: sudden spikes in trading activity (rolling 60s windows)
  - Price momentum: sustained upward movement
  - Trade frequency: high trade count = institutional/algo interest

Discovery criteria (any stock meeting ALL of these gets promoted):
  - Price $1–$20
  - Accumulated volume ≥ threshold (absolute floor)
  - Volume velocity: current 60s window is ≥ 3x the prior window
  - OR: total accumulated volume is very high (top-tier liquidity)
  - Price is going UP (positive change)

This replaces the old "rank top N" approach which missed good stocks.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set

from daytrading.models import Tick

logger = logging.getLogger(__name__)

# Rolling window size in seconds for volume velocity detection
_WINDOW_SECS = 60


class RealtimeScanner:
    """Processes a firehose of trades and detects movers in real time.

    Uses a dual-window volume velocity approach: if a stock's volume in the
    current 60-second window is ≥ 3x the previous window, it's surging.
    Also promotes any stock exceeding a high absolute volume threshold.
    """

    def __init__(
        self,
        *,
        min_price: float = 1.0,
        max_price: float = 20.0,
        min_volume: int = 500_000,
        volume_surge_ratio: float = 3.0,
        high_volume_floor: int = 2_000_000,
        check_interval: float = 10.0,
    ) -> None:
        self._min_price = min_price
        self._max_price = max_price
        self._min_volume = min_volume
        self._volume_surge_ratio = volume_surge_ratio
        self._high_volume_floor = high_volume_floor
        self._check_interval = check_interval

        # Per-symbol rolling state
        self._lock = threading.Lock()
        self._total_volume: Dict[str, int] = defaultdict(int)
        self._current_window_vol: Dict[str, int] = defaultdict(int)
        self._prev_window_vol: Dict[str, int] = defaultdict(int)
        self._last_price: Dict[str, float] = {}
        self._low_price: Dict[str, float] = {}
        self._high_price: Dict[str, float] = {}
        self._trade_count: Dict[str, int] = defaultdict(int)

        self._known_watchlist: Set[str] = set()
        self._on_new_movers: Optional[Callable] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._total_trades: int = 0
        self._last_window_rotate: float = time.time()

    def on_trade(self, tick: Tick) -> None:
        """Called for every trade from the wildcard stream. Must be fast."""
        price = tick.price
        sym = tick.symbol
        size = int(tick.size)

        if price > self._max_price * 1.5 or price < self._min_price * 0.5:
            return

        with self._lock:
            self._total_volume[sym] += size
            self._current_window_vol[sym] += size
            self._trade_count[sym] += 1
            self._last_price[sym] = price
            if sym not in self._low_price or price < self._low_price[sym]:
                self._low_price[sym] = price
            if sym not in self._high_price or price > self._high_price[sym]:
                self._high_price[sym] = price
            self._total_trades += 1

    def start(
        self,
        on_new_movers: Callable[[List[str], List[Dict]], None],
        initial_watchlist: List[str],
    ) -> None:
        self._on_new_movers = on_new_movers
        self._known_watchlist = set(initial_watchlist)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._check_loop, daemon=True, name="rt-scanner",
        )
        self._thread.start()
        logger.info(
            "Real-time scanner started — volume-velocity detection every %.0fs",
            self._check_interval,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def add_to_known(self, symbols: List[str]) -> None:
        with self._lock:
            self._known_watchlist.update(symbols)

    def _check_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._check_interval)
            if self._stop_event.is_set():
                break
            try:
                self._rotate_windows()
                self._evaluate()
            except Exception as exc:
                logger.error("RT scanner check error: %s", exc)

    def _rotate_windows(self) -> None:
        """Shift current window → previous, reset current."""
        now = time.time()
        if now - self._last_window_rotate < _WINDOW_SECS:
            return
        with self._lock:
            self._prev_window_vol = dict(self._current_window_vol)
            self._current_window_vol = defaultdict(int)
            self._last_window_rotate = now

    def _evaluate(self) -> None:
        """Find ALL stocks that qualify — no top-N cap at discovery."""
        with self._lock:
            total_vols = dict(self._total_volume)
            curr_vols = dict(self._current_window_vol)
            prev_vols = dict(self._prev_window_vol)
            prices = dict(self._last_price)
            lows = dict(self._low_price)
            highs = dict(self._high_price)
            trade_counts = dict(self._trade_count)
            total = self._total_trades
            known = set(self._known_watchlist)

        if not total_vols:
            return

        new_discoveries: List[Dict] = []

        for sym, tot_vol in total_vols.items():
            if sym in known:
                continue

            price = prices.get(sym, 0)
            if price < self._min_price or price > self._max_price:
                continue

            if tot_vol < self._min_volume:
                continue

            low = lows.get(sym, price)
            if low <= 0:
                continue
            change_pct = (price - low) / low * 100

            # Must be going up
            if change_pct < 1.0:
                continue

            curr_vol = curr_vols.get(sym, 0)
            prev_vol = prev_vols.get(sym, 0)

            qualified = False
            reason = ""

            # Criterion 1: Volume velocity surge
            if prev_vol > 0 and curr_vol >= prev_vol * self._volume_surge_ratio:
                qualified = True
                reason = "volume surge {:.0f}x ({:,} vs {:,})".format(
                    curr_vol / prev_vol, curr_vol, prev_vol)

            # Criterion 2: High absolute volume (clearly active stock)
            if tot_vol >= self._high_volume_floor:
                qualified = True
                reason = "high volume {:,} shares".format(tot_vol)

            # Criterion 3: High trade frequency (many small trades = retail interest)
            trades = trade_counts.get(sym, 0)
            if trades >= 500 and tot_vol >= self._min_volume:
                qualified = True
                reason = "active tape {:,} trades, {:,} vol".format(trades, tot_vol)

            if not qualified:
                continue

            new_discoveries.append({
                "symbol": sym,
                "price": round(price, 4),
                "volume": tot_vol,
                "change_pct": round(change_pct, 2),
                "abs_change_pct": round(change_pct, 2),
                "prev_close": round(low, 4),
                "trades": trades,
                "reason": reason,
            })

        if new_discoveries:
            new_symbols = [d["symbol"] for d in new_discoveries]
            logger.info(
                "RT SCANNER: %d new movers from %d trades: %s",
                len(new_discoveries), total,
                ", ".join("{} ({})".format(d["symbol"], d["reason"])
                          for d in new_discoveries[:10]),
            )
            with self._lock:
                self._known_watchlist.update(new_symbols)
            if self._on_new_movers:
                self._on_new_movers(new_symbols, new_discoveries)

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_trades": self._total_trades,
                "symbols_seen": len(self._total_volume),
            }
