"""Dynamic watchlist scanner — finds the best stocks to trade in real time.

Two modes:
  1. Full scan (startup): scans all 10,000+ US equities to build initial universe
  2. Live scan (background thread): monitors ~300-500 candidates every 30s
     to catch new movers the instant they spike

The live scanner runs in a background thread and calls a callback when
new movers are discovered, so they can be added to the trading pipeline
immediately.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetStatus
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.data.enums import DataFeed
    _HAS_ALPACA = True
except ImportError:
    _HAS_ALPACA = False


class WatchlistScanner:
    """Scans the market and returns the top stocks for scalping."""

    _ETF_BLACKLIST = frozenset({
        "TZA", "SOXS", "SOXL", "TQQQ", "SQQQ", "LABU", "LABD",
        "SPXS", "SPXL", "UVXY", "SVXY", "NUGT", "DUST", "JNUG",
        "JDST", "FAZ", "FAS", "ERX", "ERY", "BOIL", "KOLD",
        "DRIP", "GUSH", "UCO", "SCO", "SDOW", "UDOW", "TNA",
        "YANG", "YINN", "FNGU", "FNGD", "BULZ", "BERZ",
        "VIXY", "VXX", "ETHA", "IBIT", "BITO",
    })

    @staticmethod
    def _is_probable_warrant_symbol(symbol: str) -> bool:
        # Common warrant symbols are usually longer root tickers ending in W.
        # Short real tickers like WNW should stay in the scan universe.
        return len(symbol) >= 4 and symbol.endswith("W")

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        min_price: float = 1.0,
        max_price: float = 20.0,
        min_volume: int = 1_000_000,
        min_change_pct: float = 0.0,
        max_symbols: int = 25,
        feed: str = "iex",
        scan_interval: float = 30.0,
        premarket_min_volume: int = 10_000,
    ) -> None:
        if not _HAS_ALPACA:
            raise ImportError("alpaca-py is required")

        self._trading = TradingClient(api_key, secret_key, paper=True)
        self._data = StockHistoricalDataClient(api_key, secret_key)
        self._min_price = min_price
        self._max_price = max_price
        self._min_volume = min_volume
        self._premarket_min_volume = premarket_min_volume
        self._min_change = min_change_pct
        self._max_symbols = max_symbols
        self._feed = DataFeed(feed.lower())
        self._scan_interval = scan_interval
        self._is_premarket = False

        # Cached asset list (loaded once, refreshed hourly)
        self._all_symbols: List[str] = []
        self._symbols_loaded_at: float = 0.0

        # Candidate universe: symbols in the right price range
        # Built on first full scan, then only these are rescanned
        self._candidate_symbols: List[str] = []

        # Background scanner state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_new_movers: Optional[Callable] = None
        self._known_watchlist: set = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # One-shot full scan (used at startup)
    # ------------------------------------------------------------------

    def scan(self) -> List[Dict]:
        """Full scan: check all US equities and return ranked stocks.

        Returns ALL candidates that pass the minimum filters (price, volume,
        change). The downstream float filter will narrow this to tradeable
        low-float stocks. This prevents good low-float momentum plays from
        being hidden behind high-float stocks in the ranking.
        """
        t0 = time.time()

        symbols = self._load_all_symbols()
        candidates = self._get_snapshots(symbols)

        self._candidate_symbols = [c["symbol"] for c in candidates]

        ranked = self._rank(candidates)

        elapsed = time.time() - t0
        top_preview = ranked[:25]
        logger.info(
            "Full scan in %.1fs — %d candidates (returning all), top movers: %s",
            elapsed, len(candidates),
            ", ".join("{} ${:.2f} chg={:+.1f}%".format(
                s["symbol"], s["price"], s["change_pct"]
            ) for s in top_preview),
        )
        return ranked

    def scan_candidates(self, *, full_universe: bool = False, readonly: bool = False) -> List[Dict]:
        """Light rescan of cached movers (or full universe on first run)."""
        t0 = time.time()
        if full_universe or not self._candidate_symbols:
            symbols = self._load_all_symbols()
        else:
            symbols = list(self._candidate_symbols)

        candidates = self._get_snapshots(symbols)
        if candidates and not readonly:
            self._candidate_symbols = [c["symbol"] for c in candidates]

        ranked = self._rank(candidates)
        elapsed = time.time() - t0
        logger.info(
            "Candidate scan in %.1fs — %d pass filters",
            elapsed,
            len(candidates),
        )
        return ranked

    # ------------------------------------------------------------------
    # Background live scanner (runs every scan_interval seconds)
    # ------------------------------------------------------------------

    def start_live(
        self,
        on_new_movers: Callable[[List[str], List[Dict]], None],
        initial_watchlist: List[str],
    ) -> None:
        """Start the background scanner thread.

        Args:
            on_new_movers: callback(new_symbols, all_ranked) called when
                           new stocks are detected that aren't on the watchlist
            initial_watchlist: symbols already being tracked
        """
        self._on_new_movers = on_new_movers
        with self._lock:
            self._known_watchlist = set(initial_watchlist)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scan_loop, daemon=True, name="live-scanner",
        )
        self._thread.start()
        logger.info(
            "Live scanner started — checking for new movers every %.0fs",
            self._scan_interval,
        )

    def stop_live(self) -> None:
        """Stop the background scanner."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def add_to_known(self, symbols: List[str]) -> None:
        """Mark symbols as already tracked (won't trigger callback again)."""
        with self._lock:
            self._known_watchlist.update(symbols)

    def _scan_loop(self) -> None:
        """Background loop: rescan candidates and detect new movers."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self._scan_interval)
            if self._stop_event.is_set():
                break

            try:
                self._live_scan_cycle()
            except Exception as exc:
                logger.error("Live scan error: %s", exc)

    def _live_scan_cycle(self) -> None:
        """One cycle of the live scanner."""
        t0 = time.time()

        # Rescan only the candidate universe (much faster than all 10K)
        # Plus periodically add new candidates from full symbol list
        scan_symbols = list(self._candidate_symbols)

        # Every 10 minutes, do a broader sweep to find newly active stocks
        if not scan_symbols or (time.time() - self._symbols_loaded_at) > 600:
            scan_symbols = self._load_all_symbols()

        candidates = self._get_snapshots(scan_symbols)

        # Update candidate universe with any new finds
        new_candidates = [c["symbol"] for c in candidates]
        if len(new_candidates) > len(self._candidate_symbols):
            self._candidate_symbols = new_candidates

        ranked = self._rank(candidates)
        top = ranked[:self._max_symbols]

        # Detect new movers not yet on the watchlist
        with self._lock:
            new_symbols = [
                s["symbol"] for s in top
                if s["symbol"] not in self._known_watchlist
            ]

        elapsed = time.time() - t0

        if new_symbols:
            logger.info(
                "Live scan (%.1fs): found %d NEW movers: %s",
                elapsed, len(new_symbols), new_symbols,
            )
            with self._lock:
                self._known_watchlist.update(new_symbols)
            if self._on_new_movers:
                self._on_new_movers(new_symbols, top)
        else:
            logger.debug("Live scan (%.1fs): no new movers", elapsed)

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

    def _load_all_symbols(self) -> List[str]:
        now = time.time()
        if self._all_symbols and (now - self._symbols_loaded_at) < 3600:
            return self._all_symbols

        logger.info("Loading US equity list from Alpaca...")
        req = GetAssetsRequest(status=AssetStatus.ACTIVE)
        assets = self._trading.get_all_assets(req)
        self._all_symbols = [
            a.symbol for a in assets
            if a.exchange in ("NYSE", "NASDAQ", "ARCA", "AMEX")
            and a.tradable
            and len(a.symbol) <= 5
            and not self._is_probable_warrant_symbol(a.symbol)
            and a.symbol not in self._ETF_BLACKLIST
        ]
        self._symbols_loaded_at = now
        logger.info("Cached %d US symbols", len(self._all_symbols))
        return self._all_symbols

    def _get_snapshots(self, symbols: List[str]) -> List[Dict]:
        batch_size = 200
        candidates: List[Dict] = []
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                req = StockSnapshotRequest(
                    symbol_or_symbols=batch, feed=self._feed,
                )
                snaps = self._data.get_stock_snapshot(req)
                for sym, snap in snaps.items():
                    info = self._extract(sym, snap)
                    if info is not None:
                        candidates.append(info)
            except Exception as exc:
                logger.debug("Snapshot batch %d error: %s", i, exc)
        return candidates

    def _extract(self, symbol: str, snap) -> Optional[Dict]:
        bar = getattr(snap, "daily_bar", None)
        prev = getattr(snap, "previous_daily_bar", None)
        if bar is None or prev is None:
            return None

        # During pre-market, daily_bar is stale — use latest_trade for real price
        if self._is_premarket:
            latest = getattr(snap, "latest_trade", None)
            price = float(latest.price) if latest else float(bar.close)
        else:
            price = float(bar.close)

        if price < self._min_price or price > self._max_price:
            return None

        prev_close = float(prev.close)
        if prev_close <= 0:
            return None

        volume = float(bar.volume)
        # During pre-market, daily_bar.volume is stale (yesterday's).
        # Use minute_bar.volume which reflects actual real-time activity.
        if self._is_premarket:
            mbar = getattr(snap, "minute_bar", None)
            if mbar is not None and mbar.volume > volume:
                volume = float(mbar.volume)
        effective_min_vol = (
            self._premarket_min_volume if self._is_premarket else self._min_volume
        )
        if volume < effective_min_vol:
            change_pct_check = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0
            if abs(change_pct_check) >= 15.0 and volume >= 100_000:
                pass  # strong mover — accept with lower volume
            else:
                return None

        change_pct = (price - prev_close) / prev_close * 100

        # During pre-market, snapshot data is stale — accept any direction
        # The downstream HOD bar scanner will filter on actual session change
        if not self._is_premarket and change_pct < self._min_change:
            return None

        return {
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "change_pct": round(change_pct, 2),
            "abs_change_pct": round(abs(change_pct), 2),
            "prev_close": prev_close,
        }

    def _rank(self, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []

        import math

        max_vol = max(c["volume"] for c in candidates)
        max_chg = max(c["abs_change_pct"] for c in candidates) or 1.0

        for c in candidates:
            log_vol_score = math.log1p(c["volume"]) / math.log1p(max_vol)
            move_score = c["abs_change_pct"] / max_chg
            c["score"] = log_vol_score * 0.5 + move_score * 0.5

        candidates.sort(key=lambda c: -c["score"])
        return candidates
