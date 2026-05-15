"""Real-time market scanner — detects movers instantly from the trade stream.

Instead of polling snapshots every 30 seconds, this listens to every trade
across the entire market via the wildcard '*' subscription and tracks:
  - Volume per symbol (rolling window)
  - Price change from open/prev close
  - Sudden volume spikes

When a stock in the $1-$20 range starts surging, the callback fires
immediately — no 30-second delay.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set

from daytrading.models import Tick

logger = logging.getLogger(__name__)


class RealtimeScanner:
    """Processes a firehose of trades and detects movers in real time.

    Accumulates volume and tracks price changes per symbol. Every
    `check_interval` seconds, ranks the top movers and calls back
    with any new symbols not yet on the watchlist.
    """

    def __init__(
        self,
        *,
        min_price: float = 1.0,
        max_price: float = 20.0,
        min_volume: int = 1_000_000,
        min_change_pct: float = 2.0,
        max_symbols: int = 15,
        check_interval: float = 5.0,
    ) -> None:
        self._min_price = min_price
        self._max_price = max_price
        self._min_volume = min_volume
        self._min_change_pct = min_change_pct
        self._max_symbols = max_symbols
        self._check_interval = check_interval

        # Per-symbol accumulators (reset periodically)
        self._volume: Dict[str, int] = defaultdict(int)
        self._last_price: Dict[str, float] = {}
        self._first_price: Dict[str, float] = {}
        self._trade_count: Dict[str, int] = defaultdict(int)

        self._lock = threading.Lock()
        self._known_watchlist: Set[str] = set()
        self._on_new_movers: Optional[Callable] = None

        # Background checker
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_check: float = 0.0
        self._total_trades: int = 0

    def on_trade(self, tick: Tick) -> None:
        """Called for every trade from the wildcard stream. Must be fast."""
        price = tick.price
        sym = tick.symbol
        size = int(tick.size)

        # Quick price filter — skip expensive stocks immediately
        if price > self._max_price * 1.5 or price < self._min_price * 0.5:
            return

        with self._lock:
            self._volume[sym] += size
            self._trade_count[sym] += 1
            self._last_price[sym] = price
            if sym not in self._first_price:
                self._first_price[sym] = price
            self._total_trades += 1

    def start(
        self,
        on_new_movers: Callable[[List[str], List[Dict]], None],
        initial_watchlist: List[str],
    ) -> None:
        """Start the background checker thread."""
        self._on_new_movers = on_new_movers
        self._known_watchlist = set(initial_watchlist)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._check_loop, daemon=True, name="rt-scanner",
        )
        self._thread.start()
        logger.info(
            "Real-time scanner started — checking every %.0fs",
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
                self._evaluate()
            except Exception as exc:
                logger.error("RT scanner check error: %s", exc)

    def _evaluate(self) -> None:
        """Rank accumulated data and detect new movers."""
        with self._lock:
            # Snapshot current state
            volumes = dict(self._volume)
            prices = dict(self._last_price)
            first_prices = dict(self._first_price)
            trade_counts = dict(self._trade_count)
            total = self._total_trades

        if not volumes:
            return

        # Build candidates
        candidates = []
        for sym, vol in volumes.items():
            price = prices.get(sym, 0)
            if price < self._min_price or price > self._max_price:
                continue
            if vol < self._min_volume:
                continue

            first = first_prices.get(sym, price)
            if first <= 0:
                continue
            change_pct = (price - first) / first * 100

            # Only pick stocks going UP — no shorts
            if change_pct < self._min_change_pct:
                continue

            candidates.append({
                "symbol": sym,
                "price": round(price, 4),
                "volume": vol,
                "change_pct": round(change_pct, 2),
                "abs_change_pct": round(change_pct, 2),
                "prev_close": round(first, 4),
                "trades": trade_counts.get(sym, 0),
            })

        if not candidates:
            return

        # Rank
        max_vol = max(c["volume"] for c in candidates)
        max_chg = max(c["abs_change_pct"] for c in candidates) or 1.0
        for c in candidates:
            c["score"] = (c["volume"] / max_vol) * 0.4 + (c["abs_change_pct"] / max_chg) * 0.6
        candidates.sort(key=lambda c: -c["score"])
        top = candidates[:self._max_symbols]

        # Detect new movers
        with self._lock:
            new_symbols = [
                s["symbol"] for s in top
                if s["symbol"] not in self._known_watchlist
            ]

        if new_symbols:
            logger.info(
                "RT SCANNER: %d new movers detected (from %d trades): %s",
                len(new_symbols), total, new_symbols,
            )
            with self._lock:
                self._known_watchlist.update(new_symbols)
            if self._on_new_movers:
                self._on_new_movers(new_symbols, top)
        elif top:
            # Still broadcast the rankings update even if no new symbols
            if self._on_new_movers:
                self._on_new_movers([], top)

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_trades": self._total_trades,
                "symbols_seen": len(self._volume),
            }
