"""Float checker — fetches and caches shares outstanding / float data.

Uses yfinance to get:
  - floatShares: publicly tradeable shares
  - sharesOutstanding: total outstanding shares

Float = Outstanding shares − Insider/locked shares
(yfinance provides floatShares directly via Morningstar data)

Data is cached per symbol for the entire session to avoid
repeated API calls.

Requires Python version per ``pyproject.toml`` (>=3.10) and yfinance >= 1.2.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

_FETCH_TIMEOUT = 15  # seconds — prevent slow Yahoo responses from blocking trades


class FloatChecker:
    """Thread-safe float data cache backed by yfinance."""

    def __init__(self, min_float: float = 1_000_000) -> None:
        self._min_float = min_float
        self._cache: Dict[str, Optional[Tuple[float, float]]] = {}
        self._avg_vol_cache: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="float")

    def get_float(self, symbol: str) -> Optional[float]:
        """Return the float (publicly tradeable shares) for *symbol*.

        Returns ``None`` if the data is unavailable (API error, no data).
        Result is cached for the session.
        """
        with self._lock:
            if symbol in self._cache:
                cached = self._cache[symbol]
                if cached is None:
                    return None
                return cached[0]

        float_shares, outstanding, avg_vol = self._fetch_with_timeout(symbol)
        with self._lock:
            if float_shares is not None:
                self._cache[symbol] = (float_shares, outstanding or 0.0)
            else:
                self._cache[symbol] = None
            if avg_vol is not None and avg_vol > 0:
                self._avg_vol_cache[symbol] = avg_vol

        return float_shares

    def get_float_info(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Return (float_shares, outstanding_shares) for *symbol*."""
        with self._lock:
            if symbol in self._cache:
                cached = self._cache[symbol]
                if cached is None:
                    return None, None
                return cached

        float_shares, outstanding, avg_vol = self._fetch_with_timeout(symbol)
        with self._lock:
            if float_shares is not None:
                self._cache[symbol] = (float_shares, outstanding or 0.0)
            else:
                self._cache[symbol] = None
            if avg_vol is not None and avg_vol > 0:
                self._avg_vol_cache[symbol] = avg_vol

        return float_shares, outstanding

    def check(self, symbol: str) -> Optional[str]:
        """Return a rejection reason if float is below minimum, or ``None`` if OK.

        Returns ``None`` (pass) if data is unavailable — we don't block
        trades just because we can't fetch float data.
        """
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
        with self._lock:
            if symbol in self._avg_vol_cache:
                return self._avg_vol_cache[symbol]
        # Trigger a fetch if not cached yet (side-effect: populates avg_vol_cache)
        self.get_float(symbol)
        with self._lock:
            return self._avg_vol_cache.get(symbol)

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

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
        """Fetch float, outstanding shares, and average volume from yfinance.

        Returns (float_shares, outstanding_shares, avg_daily_volume).
        """
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
                logger.info(
                    "FLOAT %s: float=%.1fM outstanding=%.1fM avgVol=%.0f",
                    symbol,
                    float_shares / 1_000_000,
                    (outstanding or 0) / 1_000_000,
                    avg_volume or 0,
                )
                return float_shares, outstanding, avg_volume

            if outstanding is not None and outstanding > 0:
                logger.info(
                    "FLOAT %s: no float data, using outstanding=%.1fM as proxy",
                    symbol, outstanding / 1_000_000,
                )
                return outstanding, outstanding, avg_volume

            # Fallback: fast_info (lighter call, only has total shares)
            try:
                shares = float(ticker.fast_info.shares)
                if shares > 0:
                    logger.info(
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
