"""Fast HOD detection on live trade ticks (SIP stream)."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional, Sequence, Set

from daytrading.models import Bar, Tick
from daytrading.scanner.hod_momentum.alert_store import HODAlertStore
from daytrading.scanner.hod_momentum.models import HODAlertRow
from daytrading.scanner.hod_momentum.prior_day import PriorDayStats, gap_and_change_from_close
from daytrading.scanner.hod_momentum.session_context import (
    SessionContext,
    session_change_pct,
    session_context_from_bars,
)

logger = logging.getLogger(__name__)

_WINDOW_SECS = 60


@dataclass
class _SymbolState:
    session_high: float = 0.0
    session_low: float = float("inf")
    session_open: float = 0.0
    bar_day_volume: float = 0.0
    tape_volume_since_bar: int = 0
    current_window_vol: int = 0
    prev_window_vol: int = 0
    last_price: float = 0.0
    session_date: Optional[str] = None
    session_seeded: bool = False
    alert_times: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    float_shares: Optional[float] = None
    float_checked: bool = False
    prior_day_high: Optional[float] = None
    prior_day_close: Optional[float] = None
    incomplete_history: bool = False
    is_gapper: bool = False
    gapper_reclaim_fired: bool = False
    gapper_pullback_low: float = float("inf")


class HODTickTracker:
    """Detect new session highs on the trade tape and push alert rows."""

    def __init__(
        self,
        store: HODAlertStore,
        *,
        float_checker: object = None,
        min_price: float = 2.0,
        max_price: float = 20.0,
        max_float: float = 20_000_000,
        min_day_volume: float = 200_000,
        volume_surge_ratio: float = 3.0,
        tick_cooldown_seconds: float = 30.0,
        require_break_prior_day_high: bool = True,
        on_new_symbol: Optional[Callable[[str], None]] = None,
        on_needs_seed: Optional[Callable[[str], None]] = None,
        on_alert: Optional[Callable[[str, float], None]] = None,
        known_symbols: Optional[Set[str]] = None,
    ) -> None:
        self._store = store
        self._float_checker = float_checker
        self._min_price = min_price
        self._max_price = max_price
        self._max_float = max_float
        self._min_day_volume = min_day_volume
        self._volume_surge_ratio = volume_surge_ratio
        self._tick_cooldown = tick_cooldown_seconds
        self._require_break_prior_day = require_break_prior_day_high
        self._on_new_symbol = on_new_symbol
        self._on_needs_seed = on_needs_seed
        self._on_alert = on_alert
        self._known_symbols: Set[str] = set(known_symbols or [])
        self._pending_seed: Set[str] = set()

        self._states: Dict[str, _SymbolState] = {}
        self._last_window_rotate = time.monotonic()
        self._total_trades = 0

    def add_known_symbols(self, symbols: List[str]) -> None:
        self._known_symbols.update(symbols)

    def update_session_from_bars(
        self,
        symbol: str,
        bars: Sequence[Bar],
        *,
        prior_day: Optional[PriorDayStats] = None,
    ) -> None:
        """Seed session HOD and today's volume from 1m bars (matches chart)."""
        ctx = session_context_from_bars(bars, prior_day=prior_day)
        if ctx is None:
            return
        st = self._states.get(symbol)
        if st is None:
            st = _SymbolState()
            self._states[symbol] = st
        if ctx.session_date and st.session_date and ctx.session_date != st.session_date:
            st = _SymbolState()
            self._states[symbol] = st
        st.session_high = ctx.session_high
        st.session_low = ctx.session_low
        st.session_open = ctx.session_open
        st.bar_day_volume = ctx.day_volume
        st.tape_volume_since_bar = 0
        st.session_date = ctx.session_date
        st.prior_day_high = ctx.prior_day_high
        st.prior_day_close = ctx.prior_day_close
        st.incomplete_history = ctx.incomplete_history
        st.session_seeded = True
        self._pending_seed.discard(symbol)

        # Auto-detect gappers at seed time: if change from close >= 30%
        # and we have enough bars, mark for near-HOD reclaim tracking
        if (ctx.prior_day_close and ctx.prior_day_close > 0
                and ctx.session_high > 0):
            chg_from_close = (ctx.session_high - ctx.prior_day_close) / ctx.prior_day_close * 100
            if chg_from_close >= 30.0:
                st.is_gapper = True
                st.gapper_pullback_low = st.session_low

    def is_seeded(self, symbol: str) -> bool:
        """True if the symbol has been seeded with bar data."""
        st = self._states.get(symbol)
        return st is not None and st.session_seeded

    def set_tracked_symbols(self, symbols: Set[str]) -> None:
        """Replace the set of symbols we process ticks for (pool + watchlist)."""
        self._known_symbols = set(symbols)

    def mark_gapper(self, symbol: str) -> None:
        """Mark a symbol as a gapper so near-HOD reclaim logic applies."""
        st = self._states.get(symbol)
        if st is None:
            st = _SymbolState()
            self._states[symbol] = st
        st.is_gapper = True

    def on_trade(self, tick: Tick) -> None:
        """Called for each SIP trade — must be fast."""
        sym = tick.symbol

        if sym not in self._known_symbols:
            return

        price = tick.price
        size = int(tick.size or 0)
        if price <= 0 or size <= 0:
            return

        if price < self._min_price or price > self._max_price:
            return

        self._rotate_windows_unlocked()
        st = self._states.get(sym)
        if st is None:
            st = _SymbolState()
            self._states[sym] = st

        needs_seed = (
            not st.session_seeded
            and self._on_needs_seed
            and sym not in self._pending_seed
        )
        if needs_seed:
            self._pending_seed.add(sym)
            try:
                self._on_needs_seed(sym)
            except Exception as exc:
                logger.warning("on_needs_seed %s failed: %s", sym, exc)
            return

        if not st.session_seeded:
            return

        trade_date = None
        if tick.ts is not None:
            try:
                trade_date = tick.ts.date().isoformat()
            except Exception:
                pass
        if trade_date and st.session_date and trade_date != st.session_date:
            self._states[sym] = _SymbolState()
            return

        st.tape_volume_since_bar += size
        st.current_window_vol += size
        st.last_price = price
        st.session_low = min(st.session_low, price)

        day_volume = st.bar_day_volume + st.tape_volume_since_bar
        if day_volume < self._min_day_volume:
            return

        prior_high = st.session_high
        if price > st.session_high:
            st.session_high = price

        new_hod = prior_high > 0 and price > prior_high
        if not new_hod:
            # Gapper near-HOD reclaim: alert when price recovers to 95%+ of
            # session high after pulling back at least 5%
            if st.is_gapper and not st.gapper_reclaim_fired and st.session_high > 0:
                st.gapper_pullback_low = min(st.gapper_pullback_low, price)
                pullback_pct = (st.session_high - st.gapper_pullback_low) / st.session_high * 100
                reclaim_pct = price / st.session_high * 100
                if pullback_pct >= 5.0 and reclaim_pct >= 95.0:
                    st.gapper_reclaim_fired = True
                    self._maybe_gapper_reclaim_alert(sym, price, st, day_volume)
            return

        self._maybe_alert_unlocked(sym, price, st, day_volume)

    def _maybe_alert_unlocked(
        self,
        sym: str,
        price: float,
        st: _SymbolState,
        day_volume: float,
    ) -> None:
        if price < self._min_price or price > self._max_price:
            return

        now = time.monotonic()
        if st.alert_times and (now - st.alert_times[-1]) < self._tick_cooldown:
            return

        curr_vol = st.current_window_vol
        prev_vol = st.prev_window_vol
        vol_ok = (
            (prev_vol > 0 and curr_vol >= prev_vol * self._volume_surge_ratio)
            or curr_vol >= 50_000
        )
        if not vol_ok:
            return

        float_shares = self._resolve_float(sym, st)
        if float_shares is None or float_shares > self._max_float:
            return

        if st.session_open > 0:
            change_from_low = (
                (price - st.session_low) / st.session_low * 100
                if st.session_low < float("inf") else 0.0
            )
        else:
            change_from_low = 0.0

        burst_text = ""
        recent = [t for t in st.alert_times if now - t <= 12.0]
        if len(recent) >= 3:
            burst_text = "({} in {:.0f}sec)".format(len(recent) + 1, 12.0)

        alert_name = "New HOD Breakout"
        if (
            st.prior_day_high is not None
            and st.prior_day_high > 0
            and price <= st.prior_day_high
            and self._require_break_prior_day
        ):
            alert_name = "Today HOD Breakout"
        elif (
            st.prior_day_high is not None
            and st.prior_day_high > 0
            and price <= st.prior_day_high
        ):
            return

        gap_pct, change_from_close = gap_and_change_from_close(
            st.session_open, price, st.prior_day_close,
        )
        chg_ctx = SessionContext(
            session_high=st.session_high,
            session_low=st.session_low,
            session_open=st.session_open,
            day_volume=day_volume,
            session_date=st.session_date,
            prior_day_close=st.prior_day_close,
            prior_day_high=st.prior_day_high,
            incomplete_history=st.incomplete_history,
        )
        change_session = session_change_pct(price, chg_ctx)

        time_str = datetime.now(timezone.utc).isoformat()
        row = HODAlertRow(
            symbol=sym,
            time=time_str,
            price=price,
            alert_name=alert_name,
            source="tick",
            day_volume=day_volume,
            float_shares=float_shares,
            rel_vol=round(curr_vol / prev_vol, 2) if prev_vol > 0 else 0.0,
            bar_rvol=round(curr_vol / prev_vol, 2) if prev_vol > 0 else 0.0,
            change_session_pct=round(change_session, 2),
            change_from_low_pct=round(change_from_low, 2),
            gap_pct=round(gap_pct, 2) if gap_pct is not None else None,
            change_from_close_pct=(
                round(change_from_close, 2) if change_from_close is not None else None
            ),
            hot=True,
            burst_text=burst_text,
        )
        st.alert_times.append(now)
        self._store.add(row)

        logger.info(
            "HOD TICK ALERT %s @ %.4f day_vol=%.0f tape_surge=%.1fx float=%.0f",
            sym,
            price,
            day_volume,
            curr_vol / prev_vol if prev_vol > 0 else 0,
            float_shares,
        )

        if sym not in self._known_symbols and self._on_new_symbol:
            self._known_symbols.add(sym)
            try:
                self._on_new_symbol(sym)
            except Exception as exc:
                logger.warning("on_new_symbol %s failed: %s", sym, exc)

        if self._on_alert:
            try:
                self._on_alert(sym, price)
            except Exception as exc:
                logger.warning("on_alert %s failed: %s", sym, exc)

    def _maybe_gapper_reclaim_alert(
        self,
        sym: str,
        price: float,
        st: _SymbolState,
        day_volume: float,
    ) -> None:
        """Fire a Gapper Continuation alert when a gapper reclaims near its HOD."""
        if price < self._min_price or price > self._max_price:
            return

        now = time.monotonic()
        if st.alert_times and (now - st.alert_times[-1]) < self._tick_cooldown:
            return

        curr_vol = st.current_window_vol
        prev_vol = st.prev_window_vol
        vol_ok = (
            (prev_vol > 0 and curr_vol >= prev_vol * self._volume_surge_ratio)
            or curr_vol >= 30_000
        )
        if not vol_ok:
            return

        float_shares = self._resolve_float(sym, st)
        if float_shares is None or float_shares > self._max_float:
            return

        change_from_low = (
            (price - st.session_low) / st.session_low * 100
            if st.session_low < float("inf") and st.session_low > 0 else 0.0
        )

        gap_pct, change_from_close = gap_and_change_from_close(
            st.session_open, price, st.prior_day_close,
        )
        chg_ctx = SessionContext(
            session_high=st.session_high,
            session_low=st.session_low,
            session_open=st.session_open,
            day_volume=day_volume,
            session_date=st.session_date,
            prior_day_close=st.prior_day_close,
            prior_day_high=st.prior_day_high,
            incomplete_history=st.incomplete_history,
        )
        change_session = session_change_pct(price, chg_ctx)

        time_str = datetime.now(timezone.utc).isoformat()
        row = HODAlertRow(
            symbol=sym,
            time=time_str,
            price=price,
            alert_name="Gapper Continuation",
            source="tick",
            day_volume=day_volume,
            float_shares=float_shares,
            rel_vol=round(curr_vol / prev_vol, 2) if prev_vol > 0 else 0.0,
            bar_rvol=round(curr_vol / prev_vol, 2) if prev_vol > 0 else 0.0,
            change_session_pct=round(change_session, 2),
            change_from_low_pct=round(change_from_low, 2),
            gap_pct=round(gap_pct, 2) if gap_pct is not None else None,
            change_from_close_pct=(
                round(change_from_close, 2) if change_from_close is not None else None
            ),
            hot=True,
        )
        st.alert_times.append(now)
        self._store.add(row)

        logger.info(
            "GAPPER RECLAIM ALERT %s @ %.4f day_vol=%.0f float=%.0f (pullback_low=%.2f, hod=%.2f)",
            sym,
            price,
            day_volume,
            float_shares,
            st.gapper_pullback_low,
            st.session_high,
        )

        if self._on_alert:
            try:
                self._on_alert(sym, price)
            except Exception as exc:
                logger.warning("on_alert %s failed: %s", sym, exc)

    def _resolve_float(self, sym: str, st: _SymbolState) -> Optional[float]:
        if st.float_checked and st.float_shares is not None:
            return st.float_shares
        if st.float_checked:
            return None
        if self._float_checker is None:
            return None
        try:
            shares = self._float_checker.get_float_cached(sym)
        except Exception:
            shares = None
        st.float_checked = True
        st.float_shares = shares
        return shares

    def _rotate_windows_unlocked(self) -> None:
        now = time.monotonic()
        if now - self._last_window_rotate < _WINDOW_SECS:
            return
        for st in self._states.values():
            st.prev_window_vol = st.current_window_vol
            st.current_window_vol = 0
        self._last_window_rotate = now

    def cleanup_stale(self, max_age: float = 3600.0) -> None:
        """Purge symbol states not in tracked set to limit memory."""
        stale = [s for s in self._states if s not in self._known_symbols]
        for s in stale:
            del self._states[s]
