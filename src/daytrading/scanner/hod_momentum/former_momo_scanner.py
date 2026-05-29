"""Former Momo Stock alerts — price $20+, float 10M–300M, reheating vs prior close."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Set

from daytrading.models import Bar
from daytrading.scanner.hod_momentum.alert_store import HODAlertStore
from daytrading.scanner.hod_momentum.models import HODAlertRow
from daytrading.scanner.hod_momentum.prior_day import (
    PriorDayStats,
    gap_and_change_from_close,
)
from daytrading.scanner.hod_momentum.session_context import (
    session_context_from_bars,
    today_bars,
)
from daytrading.scanner.scanner_alert_labels import bar_volume_surge

logger = logging.getLogger(__name__)

ALERT_NAME = "Former Momo Stock"

# Symbols on board for display only — not used for hod_active entry gate
FORMER_MOMO_ALERT = ALERT_NAME


class FormerMomoScanner:
    """Bar scanner for former momentum names (monitoring, separate from low-float HOD).

    Tier 1: $20+, float 10M–300M, +3% from close (institutional momo)
    Tier 2: $2–$20, float ≤ 20M, +10% from close, day vol ≥ 500k,
            prior-day volume > 2M (low-float with history)
    """

    def __init__(
        self,
        store: HODAlertStore,
        *,
        float_checker: object = None,
        min_price: float = 20.0,
        max_price: float = 500.0,
        min_float: float = 10_000_000,
        max_float: float = 300_000_000,
        min_change_from_close_pct: float = 3.0,
        min_day_volume: float = 1_000_000,
        tier2_min_price: float = 2.0,
        tier2_max_price: float = 20.0,
        tier2_max_float: float = 20_000_000,
        tier2_min_change_pct: float = 10.0,
        tier2_min_day_volume: float = 500_000,
        tier2_min_prior_day_volume: float = 2_000_000,
        bar_cooldown_keys: Optional[Set[str]] = None,
    ) -> None:
        self._store = store
        self._float_checker = float_checker
        self._min_price = min_price
        self._max_price = max_price
        self._min_float = min_float
        self._max_float = max_float
        self._min_change_from_close = min_change_from_close_pct
        self._min_day_volume = min_day_volume
        self._tier2_min_price = tier2_min_price
        self._tier2_max_price = tier2_max_price
        self._tier2_max_float = tier2_max_float
        self._tier2_min_change = tier2_min_change_pct
        self._tier2_min_day_volume = tier2_min_day_volume
        self._tier2_min_prior_day_volume = tier2_min_prior_day_volume
        self._bar_cooldown_keys: Set[str] = (
            bar_cooldown_keys if bar_cooldown_keys is not None else set()
        )

    def scan(
        self,
        universe: Dict[str, Sequence[Bar]],
        *,
        prior_day_stats: Optional[Dict[str, PriorDayStats]] = None,
        rel_vols: Optional[Dict[str, float]] = None,
        verified_symbols: Optional[Set[str]] = None,
        rejections: Optional[Dict[str, str]] = None,
    ) -> None:
        prior_day_stats = prior_day_stats or {}
        rel_vols = rel_vols or {}
        verified_symbols = verified_symbols or set()
        rejections = rejections or {}

        for symbol, bars in universe.items():
            if len(bars) < 10:
                continue
            latest = bars[-1]
            price = latest.close

            today = today_bars(bars)
            if len(today) < 5:
                continue

            ctx = session_context_from_bars(bars)
            if ctx is None:
                continue

            float_shares = (
                self._float_checker.get_float_cached(symbol)
                if self._float_checker else None
            )
            if float_shares is None:
                continue

            prior = prior_day_stats.get(symbol)
            prior_close = prior.prior_close if prior else None
            gap_pct, change_from_close = gap_and_change_from_close(
                ctx.session_open, price, prior_close,
            )
            # Skip reverse-split / corporate action artifacts
            if change_from_close is not None and abs(change_from_close) > 500:
                continue
            change_session = (
                (price - ctx.session_open) / ctx.session_open * 100
                if ctx.session_open > 0 else 0.0
            )

            # Must be up on the day — no alerts for stocks falling from open
            if change_session < 0:
                continue

            # Determine which tier (if any) this symbol qualifies for
            qualifies = False

            # Tier 1: $20+, float 10M–300M, +3% change
            if (self._min_price <= price <= self._max_price
                    and self._min_float <= float_shares <= self._max_float
                    and ctx.day_volume >= self._min_day_volume):
                meets_change = (
                    (change_from_close is not None
                     and change_from_close >= self._min_change_from_close)
                    or change_session >= self._min_change_from_close
                )
                if meets_change:
                    qualifies = True

            # Tier 2: $2–$20, float ≤ 20M, +10% change, high prior-day volume
            if (not qualifies
                    and self._tier2_min_price <= price <= self._tier2_max_price
                    and float_shares <= self._tier2_max_float
                    and ctx.day_volume >= self._tier2_min_day_volume):
                prior_day_vol = prior.prior_volume if prior else 0
                if prior_day_vol >= self._tier2_min_prior_day_volume:
                    meets_t2 = (
                        (change_from_close is not None
                         and change_from_close >= self._tier2_min_change)
                        or change_session >= self._tier2_min_change
                    )
                    if meets_t2:
                        qualifies = True

            if not qualifies:
                continue

            key = symbol + "|" + ALERT_NAME
            if key in self._bar_cooldown_keys:
                continue
            self._bar_cooldown_keys.add(key)

            if latest.ts is not None:
                time_str = latest.ts.isoformat()
            else:
                time_str = datetime.now(timezone.utc).isoformat()

            low = ctx.session_low
            change_from_low = (price - low) / low * 100 if low > 0 else 0.0
            rel_vol = rel_vols.get(symbol, 0.0)

            row = HODAlertRow(
                symbol=symbol,
                time=time_str,
                price=price,
                alert_name=ALERT_NAME,
                source="bar",
                day_volume=float(ctx.day_volume),
                float_shares=float_shares,
                rel_vol=rel_vol,
                bar_rvol=bar_volume_surge(today),
                change_session_pct=change_session,
                change_from_low_pct=change_from_low,
                gap_pct=gap_pct,
                change_from_close_pct=change_from_close,
                hot=(
                    (change_from_close or 0) >= 5.0
                    or change_session >= 5.0
                    or rel_vol >= 3.0
                ),
                verified=symbol in verified_symbols,
                reject_reason=rejections.get(symbol),
            )
            self._store.add(row)
