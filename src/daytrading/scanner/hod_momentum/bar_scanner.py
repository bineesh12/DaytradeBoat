"""Bar-based HOD Momentum alerts — reclaim, squeeze labels, status merge."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Set

from daytrading.models import Bar
from daytrading.scanner.hod_momentum.alert_store import HODAlertStore
from daytrading.scanner.hod_momentum.hod_alert_logic import (
    classify_hod_breakout_alerts,
    classify_hod_reclaim,
)
from daytrading.scanner.hod_momentum.models import HODAlertRow
from daytrading.scanner.hod_momentum.prior_day import (
    PriorDayStats,
    gap_and_change_from_close,
)
from daytrading.scanner.hod_momentum.session_context import (
    session_change_pct,
    session_context_from_bars,
    today_bars,
)
from daytrading.scanner.scanner_alert_labels import (
    bar_volume_surge,
    change_pct_10m,
    change_pct_5m,
    classify_hod_momentum_alerts,
)
from daytrading.scanner.scalping.hod_reclaim import HODReclaimScanner

logger = logging.getLogger(__name__)

MIN_TODAY_BARS = 5


class HODMomentumScanner:
    """Enrich alert feed from 1m bars (not registered as a trade scanner)."""

    def __init__(
        self,
        store: HODAlertStore,
        *,
        float_checker: object = None,
        min_price: float = 2.0,
        max_price: float = 20.0,
        max_float: float = 20_000_000,
        min_session_change_pct: float = 5.0,
        min_day_volume: float = 200_000,
        require_break_prior_day_high: bool = True,
        rth_only: bool = False,
        intraday_reclaim_min_from_low_pct: float = 10.0,
        intraday_reclaim_min_day_volume: float = 1_000_000,
        intraday_reclaim_max_from_hod_pct: float = 12.0,
        sub2_enabled: bool = True,
        sub2_min_price: float = 1.0,
        sub2_max_price: float = 2.0,
        sub2_min_session_change_pct: float = 10.0,
        sub2_min_day_volume: float = 1_000_000,
        sub2_max_float: float = 10_000_000,
        bar_cooldown_keys: Optional[Set[str]] = None,
        debug: bool = False,
    ) -> None:
        self._store = store
        self._float_checker = float_checker
        self._min_price = min_price
        self._max_price = max_price
        self._max_float = max_float
        self._min_session_change_pct = min_session_change_pct
        self._min_day_volume = min_day_volume
        self._require_break_prior_day = require_break_prior_day_high
        self._rth_only = rth_only
        self._intraday_reclaim_min_from_low_pct = intraday_reclaim_min_from_low_pct
        self._intraday_reclaim_min_day_volume = intraday_reclaim_min_day_volume
        self._intraday_reclaim_max_from_hod_pct = intraday_reclaim_max_from_hod_pct
        self._sub2_enabled = sub2_enabled
        self._sub2_min_price = sub2_min_price
        self._sub2_max_price = sub2_max_price
        self._sub2_min_session_change_pct = sub2_min_session_change_pct
        self._sub2_min_day_volume = sub2_min_day_volume
        self._sub2_max_float = sub2_max_float
        self._debug = debug
        self._reclaim = HODReclaimScanner(
            min_price=sub2_min_price if sub2_enabled else min_price,
            max_price=max_price,
        )
        self._bar_cooldown_keys: Set[str] = bar_cooldown_keys if bar_cooldown_keys is not None else set()
        self._gapper_fired: Set[str] = set()

    def scan(
        self,
        universe: Dict[str, Sequence[Bar]],
        *,
        bars_5m: Optional[Dict[str, Sequence[Bar]]] = None,
        rel_vols: Optional[Dict[str, float]] = None,
        prior_day_stats: Optional[Dict[str, PriorDayStats]] = None,
        verified_symbols: Optional[Set[str]] = None,
        rejections: Optional[Dict[str, str]] = None,
        is_premarket: bool = False,
    ) -> Dict[str, int]:
        """Add bar-sourced rows to the alert store. Returns reject reason counts."""
        verified_symbols = verified_symbols or set()
        rejections = rejections or {}
        rel_vols = rel_vols or {}
        bars_5m = bars_5m or {}
        prior_day_stats = prior_day_stats or {}

        reject_stats: Dict[str, int] = {}
        new_alerts = 0

        for symbol, bars in universe.items():
            if len(bars) < 10:
                reject_stats["too_few_bars"] = reject_stats.get("too_few_bars", 0) + 1
                continue
            latest = bars[-1]
            price = latest.close
            if latest.ts is not None:
                time_str = latest.ts.isoformat()
            else:
                time_str = datetime.now(timezone.utc).isoformat()
            standard_price_band = self._min_price <= price <= self._max_price
            sub2_price_band = (
                self._sub2_enabled
                and self._sub2_min_price <= price < self._sub2_max_price
            )
            if not (standard_price_band or sub2_price_band):
                reject_stats["price_band"] = reject_stats.get("price_band", 0) + 1
                continue

            today = today_bars(bars, rth_only=self._rth_only)
            if len(today) < MIN_TODAY_BARS:
                reject_stats["too_few_today"] = reject_stats.get("too_few_today", 0) + 1
                continue

            prior = prior_day_stats.get(symbol)
            ctx = session_context_from_bars(
                bars, prior_day=prior, rth_only=self._rth_only,
            )
            if ctx is None:
                reject_stats["no_context"] = reject_stats.get("no_context", 0) + 1
                continue

            day_vol = ctx.day_volume
            min_day_volume = (
                self._sub2_min_day_volume if sub2_price_band else self._min_day_volume
            )
            if day_vol < min_day_volume:
                reject_stats["low_volume"] = reject_stats.get("low_volume", 0) + 1
                continue

            session_open = ctx.session_open
            change_session = session_change_pct(price, ctx)
            if sub2_price_band and change_session < self._sub2_min_session_change_pct:
                reject_stats["sub2_weak_change"] = reject_stats.get("sub2_weak_change", 0) + 1
                continue

            float_shares = (
                self._float_checker.get_float_cached(symbol)
                if self._float_checker else None
            )
            max_float = self._sub2_max_float if sub2_price_band else self._max_float
            if float_shares is None or float_shares > max_float:
                reject_stats["float"] = reject_stats.get("float", 0) + 1
                continue

            rel_vol = rel_vols.get(symbol, 0.0)
            bar_rvol = bar_volume_surge(today)
            b5 = bars_5m.get(symbol)
            ch5 = change_pct_5m(b5)
            ch10 = change_pct_10m(today)
            low = ctx.session_low
            change_from_low = (price - low) / low * 100 if low > 0 else 0.0
            session_high = ctx.session_high
            distance_from_hod_pct = (
                (session_high - price) / session_high * 100
                if session_high > 0 else 100.0
            )

            prior_hod = max(b.high for b in today[:-1]) if len(today) > 1 else 0.0
            prior_day_high = ctx.prior_day_high
            include_intraday_reclaim = (
                change_from_low >= self._intraday_reclaim_min_from_low_pct
                and day_vol >= self._intraday_reclaim_min_day_volume
                and distance_from_hod_pct <= self._intraday_reclaim_max_from_hod_pct
                and (
                    (ch5 is not None and ch5 >= 4.0)
                    or (ch10 is not None and ch10 >= 8.0)
                    or bar_rvol >= 4.0
                )
                and latest.close >= latest.open
            )

            if (self._require_break_prior_day
                    and prior_day_high is not None
                    and prior_day_high > 0
                    and price <= prior_day_high
                    and not include_intraday_reclaim):
                # Exception: massive gappers that pulled back below prior day high
                # but are still within 20% of session HOD
                prior_close = ctx.prior_day_close
                early_chg = (
                    (price - prior_close) / prior_close * 100
                    if prior_close and prior_close > 0 else 0.0
                )
                if (symbol not in self._gapper_fired
                        and early_chg >= 30.0):
                    session_high = ctx.session_high
                    if session_high > 0:
                        distance_from_hod_pct = (session_high - price) / session_high * 100
                        if distance_from_hod_pct <= 20.0:
                            pass  # Fall through to gapper check below
                        else:
                            reject_stats["below_prior_day_high"] = reject_stats.get("below_prior_day_high", 0) + 1
                            continue
                    else:
                        reject_stats["below_prior_day_high"] = reject_stats.get("below_prior_day_high", 0) + 1
                        continue
                else:
                    reject_stats["below_prior_day_high"] = reject_stats.get("below_prior_day_high", 0) + 1
                    if self._debug:
                        logger.debug(
                            "HOD near-miss: %s chg=%.1f%% price=$%.2f prior_high=$%.2f (below_prior_day_high)",
                            symbol, change_session, price, prior_day_high,
                        )
                    continue

            include_new_hod = False
            include_today_hod = False
            include_reclaim = False
            if change_session >= self._min_session_change_pct:
                include_new_hod, include_today_hod = classify_hod_breakout_alerts(
                    latest.high,
                    prior_hod,
                    latest.close,
                    latest.open,
                    prior_day_high,
                    require_break_prior_day=self._require_break_prior_day,
                )

            # Pre-market: if already above prior day high, treat as breakout
            # but only if the stock is actively trading AND trending up (not fading)
            if (is_premarket
                    and change_session >= self._min_session_change_pct
                    and not include_new_hod
                    and not include_today_hod):
                recent_vol = latest.volume if latest.volume else 0
                if recent_vol >= 5000:
                    # Require upward momentum: last bar must be green OR price near bar high
                    bar_is_green = latest.close >= latest.open
                    near_high = (latest.high - latest.close) <= 0.3 * (latest.high - latest.low + 0.001)
                    trending_up = bar_is_green or near_high
                    if trending_up:
                        if prior_day_high is not None and prior_day_high > 0:
                            if price > prior_day_high:
                                include_new_hod = True
                        else:
                            include_new_hod = True
                reclaim_hit = self._reclaim._detect(symbol, list(today))
                reclaim_hod = (
                    float(reclaim_hit.criteria.get("hod", 0))
                    if reclaim_hit is not None else 0.0
                )
                include_reclaim = classify_hod_reclaim(
                    reclaim_hit is not None,
                    latest.high,
                    prior_hod,
                    reclaim_hod,
                    prior_day_high,
                    require_break_prior_day=self._require_break_prior_day,
                )

            gap_pct, change_from_close = gap_and_change_from_close(
                session_open, price, ctx.prior_day_close,
            )
            if change_from_close is not None and abs(change_from_close) > 500:
                reject_stats["reverse_split"] = reject_stats.get("reverse_split", 0) + 1
                continue

            alert_names = classify_hod_momentum_alerts(
                price=price,
                float_shares=float_shares,
                rel_vol=rel_vol,
                bar_rvol=bar_rvol,
                change_session_pct=change_session,
                change_5m_pct=ch5,
                change_10m_pct=ch10,
                include_hod_breakout=include_new_hod,
                include_today_hod_breakout=include_today_hod,
                include_hod_reclaim=include_reclaim,
                include_intraday_low_reclaim=include_intraday_reclaim,
                max_float=max_float,
            )

            # Gapper Continuation: one-time alert for massive gappers still near HOD
            if not alert_names and symbol not in self._gapper_fired:
                if change_from_close is not None and change_from_close >= 30.0:
                    session_high = ctx.session_high
                    if session_high > 0:
                        if distance_from_hod_pct <= 20.0:
                            alert_names = ["Gapper Continuation"]
                            self._gapper_fired.add(symbol)

            if not alert_names:
                reject_stats["no_label_match"] = reject_stats.get("no_label_match", 0) + 1
                continue

            verified = symbol in verified_symbols
            reject_reason = rejections.get(symbol)

            for name in alert_names:
                key = symbol + "|" + name
                if key in self._bar_cooldown_keys and name != "HOD Reclaim":
                    continue
                self._bar_cooldown_keys.add(key)

                # Use current time if bar is stale (avoids immediate TTL pruning)
                alert_time = time_str
                if latest.ts is not None:
                    age_secs = (datetime.now(timezone.utc) - latest.ts).total_seconds()
                    if age_secs > 120:
                        alert_time = datetime.now(timezone.utc).isoformat()

                row = HODAlertRow(
                    symbol=symbol,
                    time=alert_time,
                    price=price,
                    alert_name=name,
                    source="bar",
                    day_volume=float(day_vol),
                    float_shares=float_shares,
                    rel_vol=rel_vol,
                    rel_vol_5m_pct=ch5,
                    bar_rvol=bar_rvol,
                    change_session_pct=change_session,
                    change_from_low_pct=change_from_low,
                    gap_pct=gap_pct,
                    change_from_close_pct=change_from_close,
                    hot=rel_vol >= 5.0 or change_session >= 10.0,
                    verified=verified,
                    reject_reason=reject_reason,
                )
                self._store.add(row)
                new_alerts += 1
                logger.info(
                    "HOD BAR ALERT %s %s @ %.2f chg=%.1f%% day_vol=%.0f",
                    symbol, name, price, change_session, day_vol,
                )

            self._store.merge_status(
                symbol,
                verified=verified,
                reject_reason=reject_reason,
            )

        reject_stats["_new_alerts"] = new_alerts
        reject_stats["_scanned"] = len(universe)
        return reject_stats
