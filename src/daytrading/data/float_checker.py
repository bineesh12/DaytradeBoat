"""Float checker — fetches and caches shares outstanding / float data.

Uses yfinance to get:
  - floatShares: publicly tradeable shares
  - sharesOutstanding: total outstanding shares

Lookup order: in-memory session cache → SQLite (7-day TTL) → Yahoo.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Dict, Optional, Tuple, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from daytrading.data.float_store import FloatStore

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

_FETCH_TIMEOUT = 15  # seconds — prevent slow Yahoo responses from blocking trades


class FloatChecker:
    """Thread-safe float data cache: memory → SQLite → yfinance."""

    def __init__(
        self,
        min_float: float = 1_000_000,
        *,
        store: Optional["FloatStore"] = None,
        cache_ttl_days: int = 7,
    ) -> None:
        self._min_float = min_float
        self._store = store
        self._cache_ttl_days = cache_ttl_days
        self._cache: Dict[str, Optional[Tuple[float, float]]] = {}
        self._avg_vol_cache: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="float")

    @property
    def store(self) -> Optional["FloatStore"]:
        return self._store

    @property
    def cache_ttl_days(self) -> int:
        return self._cache_ttl_days

    def warm_from_store(self, symbols: list) -> Tuple[int, int]:
        """Load fresh DB rows into memory. Returns (from_db, need_yahoo)."""
        if not self._store or not symbols:
            return 0, len(symbols)
        records = self._store.bulk_get(symbols)
        from_db = 0
        need_yahoo = 0
        sym_set = {s.upper().strip() for s in symbols if s}
        for sym in sym_set:
            rec = records.get(sym)
            if rec is None or not rec.is_fresh(self._cache_ttl_days):
                need_yahoo += 1
                continue
            self._apply_to_memory(
                sym,
                rec.float_shares,
                rec.outstanding_shares,
                rec.avg_volume,
            )
            from_db += 1
        return from_db, need_yahoo

    def _apply_to_memory(
        self,
        symbol: str,
        float_shares: Optional[float],
        outstanding: Optional[float],
        avg_vol: Optional[float],
    ) -> None:
        sym = symbol.upper().strip()
        with self._lock:
            if float_shares is not None:
                self._cache[sym] = (float_shares, outstanding or 0.0)
            else:
                self._cache[sym] = None
            if avg_vol is not None and avg_vol > 0:
                self._avg_vol_cache[sym] = avg_vol

    def get_float(self, symbol: str) -> Optional[float]:
        """Return the float (publicly tradeable shares) for *symbol*."""
        sym = symbol.upper().strip()
        with self._lock:
            if sym in self._cache:
                cached = self._cache[sym]
                if cached is None:
                    return None
                return cached[0]

        if self._store is not None:
            rec = self._store.get(sym)
            if rec is not None and rec.is_fresh(self._cache_ttl_days):
                self._apply_to_memory(
                    sym, rec.float_shares, rec.outstanding_shares, rec.avg_volume,
                )
                return rec.float_shares

        float_shares, outstanding, avg_vol = self._fetch_with_timeout(sym)
        if self._store is not None:
            self._store.upsert(
                sym,
                float_shares,
                outstanding,
                avg_volume=avg_vol,
                source="yfinance",
            )
        self._apply_to_memory(sym, float_shares, outstanding, avg_vol)
        return float_shares

    def get_float_cached(self, symbol: str) -> Optional[float]:
        """Return cached float only — never makes a network call.

        Returns None if the symbol isn't in memory cache or persistent store.
        Use this in hot paths (scanner loops) to avoid blocking.
        """
        sym = symbol.upper().strip()
        with self._lock:
            if sym in self._cache:
                cached = self._cache[sym]
                if cached is None:
                    return None
                return cached[0]

        if self._store is not None:
            rec = self._store.get(sym)
            if rec is not None and rec.is_fresh(self._cache_ttl_days):
                self._apply_to_memory(
                    sym, rec.float_shares, rec.outstanding_shares, rec.avg_volume,
                )
                return rec.float_shares
        return None

    def get_float_info(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Return (float_shares, outstanding_shares) for *symbol*."""
        sym = symbol.upper().strip()
        self.get_float(sym)
        with self._lock:
            cached = self._cache.get(sym)
            if cached is None:
                return None, None
            return cached

    def check(self, symbol: str) -> Optional[str]:
        """Return a rejection reason if float is below minimum, or ``None`` if OK."""
        float_shares = self.get_float(symbol)

        if float_shares is None:
            logger.debug("FLOAT %s: data unavailable, allowing trade", symbol)
            return None

        if float_shares < self._min_float:
            return "float {:.1f}M < {:.1f}M".format(
                float_shares / 1_000_000, self._min_float / 1_000_000,
            )

        logger.debug(
            "FLOAT %s: %.1fM shares (min %.1fM) ✓",
            symbol, float_shares / 1_000_000, self._min_float / 1_000_000,
        )
        return None

    def get_avg_volume(self, symbol: str) -> Optional[float]:
        """Return the 10-day average daily volume for *symbol*, or None."""
        sym = symbol.upper().strip()
        with self._lock:
            if sym in self._avg_vol_cache:
                return self._avg_vol_cache[sym]
        self.get_float(sym)
        with self._lock:
            return self._avg_vol_cache.get(sym)

    def needs_yahoo_refresh(self, symbol: str) -> bool:
        """True if symbol is not in fresh memory or DB cache."""
        sym = symbol.upper().strip()
        with self._lock:
            if sym in self._cache:
                return False
        if self._store is not None:
            rec = self._store.get(sym)
            if rec is not None and rec.is_fresh(self._cache_ttl_days):
                return False
        return True

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._avg_vol_cache.clear()

    def _fetch_with_timeout(self, symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Wrap _fetch in a timeout so slow Yahoo responses don't block trades."""
        try:
            future = self._executor.submit(self._fetch, symbol)
            return future.result(timeout=_FETCH_TIMEOUT)
        except FuturesTimeout:
            logger.warning("FLOAT %s: fetch timed out after %ds", symbol, _FETCH_TIMEOUT)
            return None, None, None
        except Exception as exc:
            logger.debug("FLOAT %s: fetch executor error: %s", symbol, exc)
            return None, None, None

    @staticmethod
    def _fetch(symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Fetch float, outstanding shares, and average volume from yfinance."""
        if not _HAS_YF:
            logger.warning("yfinance not installed — float check disabled")
            return None, None, None

        try:
            ticker = yf.Ticker(symbol)

            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            float_shares = info.get("floatShares")
            outstanding = info.get("sharesOutstanding")
            avg_volume = info.get("averageDailyVolume10Day") or info.get("averageVolume")

            if float_shares is not None:
                try:
                    float_shares = float(float_shares)
                except (TypeError, ValueError):
                    float_shares = None

            if outstanding is not None:
                try:
                    outstanding = float(outstanding)
                except (TypeError, ValueError):
                    outstanding = None

            if avg_volume is not None:
                try:
                    avg_volume = float(avg_volume)
                except (TypeError, ValueError):
                    avg_volume = None

            if float_shares is not None and float_shares > 0:
                logger.debug(
                    "FLOAT %s: float=%.1fM outstanding=%.1fM avgVol=%.0f",
                    symbol,
                    float_shares / 1_000_000,
                    (outstanding or 0) / 1_000_000,
                    avg_volume or 0,
                )
                return float_shares, outstanding, avg_volume

            if outstanding is not None and outstanding > 0:
                logger.debug(
                    "FLOAT %s: no float data, using outstanding=%.1fM as proxy",
                    symbol, outstanding / 1_000_000,
                )
                return outstanding, outstanding, avg_volume

            try:
                shares = float(ticker.fast_info.shares)
                if shares > 0:
                    logger.debug(
                        "FLOAT %s: using fast_info shares=%.1fM as proxy",
                        symbol, shares / 1_000_000,
                    )
                    return shares, shares, avg_volume
            except Exception:
                pass

            logger.debug("FLOAT %s: no share data available", symbol)
            return None, None, avg_volume

        except Exception as exc:
            logger.debug("FLOAT %s: fetch error: %s", symbol, exc)
            return None, None, None
