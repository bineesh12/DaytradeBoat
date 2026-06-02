"""Live trading runner — connects Alpaca to the pipeline and runs it.

This is the main entry point for live paper trading.

Usage:
    from daytrading.runner import AlpacaRunner

    runner = AlpacaRunner.from_env()  # reads .env / env vars
    runner.run()                      # blocks until market close

Or from command line:
    python -m daytrading.runner
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import sys
import time
from collections import defaultdict, deque, namedtuple
from datetime import datetime, timezone, timedelta
from threading import Event, Lock, Thread
from typing import Dict, List, Optional, Sequence, Set

from daytrading.config import Settings
from daytrading.market_calendar import ET, is_us_trading_day, now_et
from daytrading.dashboard.hub import DashboardHub
from daytrading.dashboard.server import start_dashboard
from daytrading.data.alpaca_feed import AlpacaHistoricalFeed, AlpacaStreamFeed
from daytrading.data.float_checker import FloatChecker
from daytrading.data.float_store import FloatStore
from daytrading.data.news_checker import NewsChecker
from daytrading.data.watchlist_scanner import WatchlistScanner
from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.execution.live_prices import resolve_live_prices
from daytrading.execution.position_reconciler import PositionReconciler
from daytrading.exits.manager import ExitManager
from daytrading.exits.scaler import PositionScaler, ReentryDetector
from daytrading.exits.tape_pressure import TapePressureExit
from daytrading.journal.store import TradingJournal
from daytrading.pipeline.engine import PipelineResult, TradingPipeline
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.models import Bar, Fill, Order, PortfolioState, Position, Quote, ScanResult, Side, SignalAction, Tick, TradeSignal

logger = logging.getLogger(__name__)

# --- Event types for the lock-free producer→consumer queue ---
BarEvent = namedtuple("BarEvent", ["symbol", "bar"])
QuoteEvent = namedtuple("QuoteEvent", ["symbol", "quote"])
TradeEvent = namedtuple("TradeEvent", ["symbol", "tick"])
BarsLoadedEvent = namedtuple("BarsLoadedEvent", ["bars_by_symbol", "prior_day_stats"])
PoolRefreshEvent = namedtuple("PoolRefreshEvent", ["new_pool", "bars", "prior_day_stats"])
FastScanEvent = namedtuple("FastScanEvent", ["new_movers"])


class AlpacaRunner:
    """Runs the trading pipeline against Alpaca paper trading.

    Flow:
      1. Fetch account info + sync portfolio state
      2. Load historical bars for the watchlist
      3. Start real-time streaming (bars + quotes)
      4. On each new bar: run pipeline cycle
      5. Log fills, exits, scale-ups, re-entries
      6. Auto-close all positions at end-of-day
    """

    def __init__(
        self,
        broker: AlpacaBroker,
        hist_feed: AlpacaHistoricalFeed,
        stream_feed: AlpacaStreamFeed,
        pipeline: TradingPipeline,
        watchlist: Sequence[str],
        *,
        cycle_interval: float = 1.0,
        close_positions_at_eod: bool = True,
        dashboard_port: int = 8080,
    ) -> None:
        self._broker = broker
        self._hist = hist_feed
        self._stream = stream_feed
        self._pipeline = pipeline
        self._watchlist = list(watchlist)
        self._watchlist_set = set(watchlist)
        self._cycle_interval = cycle_interval
        self._close_at_eod = close_positions_at_eod
        self._dashboard_port = dashboard_port

        self._bar_buffer: Dict[str, deque] = {}
        self._quote_buffer: Dict[str, deque] = {}
        self._tick_buffer: Dict[str, deque] = {}
        self._event_queue: queue.Queue = queue.Queue(maxsize=100_000)
        self._shutdown = False
        self._max_bars_per_symbol = 1000
        self._session_bar_limit = 1000  # full extended session (~4 AM–8 PM ET)
        self._max_ticks_per_symbol = 200

        self._tape_pressure = TapePressureExit(threshold=60, min_hold_secs=30.0)

        from daytrading.data.bar_aggregator import BarAggregator
        from daytrading.strategy.execution_timer import ExecutionTimer
        self._bar_aggregator = BarAggregator()
        self._exec_timer = ExecutionTimer(max_wait_bars=1, enabled=True)
        self._timed_signal_queue: deque = deque()

        self._skip_counts: Dict[str, int] = defaultdict(int)
        self._SKIP_THRESHOLD = 10  # remove after 10 consecutive skips (~10 min)
        self._CLEANUP_EVERY = 10   # run cleanup every 10 cycles
        self._new_data = Event()
        self._hub = DashboardHub()
        self._journal = TradingJournal()
        self._hub.journal = self._journal
        self._hub._broker = self._broker
        self._hub._exit_manager = self._pipeline.exit_manager
        self._watchlist_data: List[dict] = []
        self._hod_bar_pool: List[str] = []
        self._scanner: Optional[WatchlistScanner] = None
        self._sip_feed: bool = False
        self._pos_sync_thread: Optional[Thread] = None
        self._news_checker: Optional[NewsChecker] = None
        self._float_checker: Optional[FloatChecker] = None
        self._last_synced_order_ids: set = set()
        self._recorded_exit_fill_keys: set = set()
        self._trade_analyzer = None
        self._analysis_interval = 10  # run analysis every N cycles
        self._reconciler = PositionReconciler()
        self._hod_active: Dict[str, datetime] = {}
        self._hod_last_alert_at: Dict[str, datetime] = {}
        self._hod_watchlist_ttl_minutes = 5.0
        self._news_pinned: Set[str] = set()
        self._hod_alert_store = None
        self._hod_tick_tracker = None
        self._hod_bar_scanner = None
        self._hod_former_momo_scanner = None
        self._prior_day_stats: Dict[str, object] = {}
        self._hod_alert_ttl_minutes = 5.0
        self._watchlist_pinned: Set[str] = {"SPY"}
        self._max_watchlist = 50
        self._hod_seed_queue: deque = deque()
        self._hod_seed_pending: Set[str] = set()
        self._hod_seed_retries: Dict[str, int] = {}  # symbol -> retry count
        self._hod_seed_blacklist: Set[str] = set()  # symbols that always fail
        self._hod_seed_lock = Lock()
        self._hod_seed_event = Event()
        self._hod_seed_thread: Optional[Thread] = None
        self._hod_seed_max = 80
        self._hod_min_price = 2.0
        self._hod_max_price = 20.0
        self._hod_sub2_enabled = True
        self._hod_sub2_min_price = 1.0
        self._hod_sub2_max_price = 2.0
        self._hod_sub2_min_change_pct = 10.0
        self._hod_sub2_min_day_volume = 1_000_000
        self._hod_sub2_max_float = 10_000_000
        self._hod_pool_max = 50
        self._hod_max_float = 20_000_000
        self._hod_pool_refresh_sec = 10 * 60
        self._pool_refresh_event = Event()
        self._pool_refresh_thread: Optional[Thread] = None
        self._pool_candidates_ready = False
        self._after_hours_enabled = False
        self._hod_hydrate_batch_max = 15
        self._bar_fetch_batch_delay_sec = 0.5
        self._hydrate_paused_until: float = 0.0
        self._network_failure_times: List[float] = []
        self._pool_refresh_in_progress = False
        self._hod_seed_batch_size = 10

        # Fast scan thread (Thread 2) - 30s interval
        self._fast_scan_thread: Optional[Thread] = None
        self._fast_scan_interval_sec = 30
        self._fast_scan_known: Set[str] = set()

        # Watchlist bar refresh — keep 1m bars fresh for pattern scanners
        self._last_watchlist_bar_refresh: float = 0.0
        self._watchlist_bar_refresh_sec: float = 30.0

        # Breakout scalp — instant entry on HOD tick alerts
        self._pending_breakout_scalps: deque = deque(maxlen=10)
        self._breakout_scalp_cooldown: Dict[str, float] = {}
        self._breakout_scalp_active: bool = False  # True if we have an open breakout scalp
        self._hod_seed_max_per_minute = 30
        self._hod_seed_minute_start = time.time()
        self._hod_seed_processed_this_minute = 0
        self._last_session_reset_day: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        watchlist: Optional[Sequence[str]] = None,
        settings: Optional[Settings] = None,
    ) -> AlpacaRunner:
        """Create a runner from environment variables / .env file.

        Set these env vars (or put them in .env):
            DAYTRADING_ALPACA_API_KEY=your_key
            DAYTRADING_ALPACA_SECRET_KEY=your_secret
            DAYTRADING_ALPACA_PAPER=true
        """
        cfg = settings or Settings()

        if not cfg.alpaca_api_key or not cfg.alpaca_secret_key:
            raise ValueError(
                "Missing Alpaca credentials. Set DAYTRADING_ALPACA_API_KEY "
                "and DAYTRADING_ALPACA_SECRET_KEY in .env or environment."
            )

        broker = AlpacaBroker(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            paper=cfg.alpaca_paper,
        )

        hist_feed = AlpacaHistoricalFeed(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            feed=cfg.alpaca_feed,
            bar_fetch_batch_size=cfg.bar_fetch_batch_size,
            bar_fetch_batch_delay_sec=cfg.bar_fetch_batch_delay_sec,
        )

        stream_feed = AlpacaStreamFeed(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            feed=cfg.alpaca_feed,
        )

        # sync cash from Alpaca account
        acct = broker.get_account()
        portfolio = PortfolioState(cash=acct["cash"])
        logger.info(
            "Alpaca account: cash=$%.2f, buying_power=$%.2f, equity=$%.2f",
            acct["cash"], acct["buying_power"], acct["equity"],
        )

        # IEX free feed reports ~10-15% of real market volume;
        # lower the liquidity thresholds so stocks aren't incorrectly filtered.
        vol_threshold = 10_000.0  # min avg bar volume to consider tradeable
        high_liq = 50_000.0      # avg bar vol for "fully liquid" (liq=1.0)
        if cfg.alpaca_feed.lower() == "iex":
            vol_threshold = 1_000
            high_liq = 10_000.0

        float_store = FloatStore()
        float_checker = FloatChecker(
            min_float=100_000,
            store=float_store,
            cache_ttl_days=cfg.float_cache_ttl_days,
        )
        logger.info(
            "Float cache ON — SQLite %s (TTL %d days)",
            float_store.db_path,
            cfg.float_cache_ttl_days,
        )

        pipeline = create_scalping_pipeline(
            initial_cash=acct["cash"],
            commission_per_share=cfg.commission_per_share,
            min_price=cfg.min_price,
            max_price=cfg.max_price,
            max_positions=cfg.max_positions,
            max_position_shares=cfg.max_position_shares,
            max_order_shares=cfg.max_order_shares,
            pattern_max_dollar_risk=cfg.max_dollar_risk_per_trade,
            min_avg_volume=vol_threshold,
            high_liquidity_volume=high_liq,
            portfolio=portfolio,
            float_checker=float_checker,
            enable_daily_loser_blacklist=cfg.enable_daily_loser_blacklist,
        )
        if cfg.enable_daily_loser_blacklist:
            logger.info("Daily loser blacklist: ON")
        else:
            logger.info("Daily loser blacklist: OFF (testing mode — re-entry allowed after losses)")

        # replace the PaperBroker in the pipeline with the real AlpacaBroker
        pipeline._broker = broker  # type: ignore[assignment]
        # Wire slippage guard into broker for smart limit pricing
        broker._slippage_guard = pipeline.trade_guard.slippage

        scanner_min_price = (
            cfg.hod_sub2_momentum_min_price
            if cfg.hod_sub2_momentum_enabled
            else cfg.hod_momentum_min_price
        )
        scanner = WatchlistScanner(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            min_price=scanner_min_price,
            max_price=cfg.hod_momentum_max_price,
            min_volume=10_000,
            min_change_pct=2.0,
            max_symbols=30,
            feed=cfg.alpaca_feed,
        )

        pinned: Set[str] = {"SPY"}
        scan_data: List[dict] = []
        hod_bar_pool: List[str] = []
        if watchlist is None:
            watchlist = ["SPY"]
            logger.info(
                "HOD-driven watchlist: starting with %s — symbols added when HOD board alerts",
                watchlist,
            )
            logger.info(
                "HOD bar pool: building in background ($%.0f–$%.0f, refresh every %d min)",
                cfg.hod_momentum_min_price,
                cfg.hod_momentum_max_price,
                cfg.hod_pool_refresh_minutes,
            )
        else:
            pinned = set(watchlist) | {"SPY"}
            watchlist = list(pinned)
            logger.info("Pinned watchlist symbols (always kept): %s", sorted(pinned))

        # Attach news sentiment checker (uses Alpaca News API — free)
        news_checker = NewsChecker(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            max_age_hours=24,
        )
        pipeline.set_news_checker(news_checker)
        logger.info("News sentiment checker active — will screen trades for bad news")

        logger.info("Initial trading watchlist (%d symbols): %s", len(watchlist), watchlist)

        runner = cls(
            broker=broker,
            hist_feed=hist_feed,
            stream_feed=stream_feed,
            pipeline=pipeline,
            watchlist=watchlist,
            dashboard_port=int(os.environ.get("DAYTRADING_DASHBOARD_PORT", "8080")),
        )
        runner._watchlist_data = scan_data
        runner._hod_bar_pool = hod_bar_pool
        runner._hod_watchlist_ttl_minutes = cfg.hod_momentum_watchlist_ttl_minutes
        runner._watchlist_pinned = pinned
        runner._scanner = scanner
        runner._sip_feed = cfg.alpaca_feed.lower() == "sip"
        runner._news_checker = news_checker
        runner._float_checker = float_checker
        runner._hod_min_price = cfg.hod_momentum_min_price
        runner._hod_max_price = cfg.hod_momentum_max_price
        runner._hod_sub2_enabled = cfg.hod_sub2_momentum_enabled
        runner._hod_sub2_min_price = cfg.hod_sub2_momentum_min_price
        runner._hod_sub2_max_price = cfg.hod_sub2_momentum_max_price
        runner._hod_sub2_min_change_pct = cfg.hod_sub2_momentum_min_change_pct
        runner._hod_sub2_min_day_volume = cfg.hod_sub2_momentum_min_day_volume
        runner._hod_sub2_max_float = cfg.hod_sub2_momentum_max_float
        runner._hod_pool_max = cfg.hod_momentum_bar_pool_max
        runner._hod_max_float = cfg.hod_momentum_max_float
        runner._hod_pool_refresh_sec = max(60, cfg.hod_pool_refresh_minutes * 60)
        runner._after_hours_enabled = cfg.after_hours_enabled
        runner._hod_hydrate_batch_max = cfg.hod_hydrate_batch_max
        runner._bar_fetch_batch_delay_sec = cfg.bar_fetch_batch_delay_sec
        runner._hod_seed_batch_size = cfg.hod_seed_batch_size
        runner._hod_seed_max_per_minute = cfg.hod_seed_max_per_minute
        runner._setup_hod_momentum(cfg, pipeline)
        logger.info(
            "Warrior-mode: pool=%d, refresh=%dmin, seed=%d/min, "
            "hydrate=%d (pool-only tick filtering)",
            cfg.hod_momentum_bar_pool_max,
            cfg.hod_pool_refresh_minutes,
            cfg.hod_seed_max_per_minute,
            cfg.hod_hydrate_batch_max,
        )
        if cfg.after_hours_enabled:
            logger.info(
                "After-hours trading ON — entries until 8:00 PM ET; "
                "no 3:30 PM flatten (testing)",
            )
        else:
            logger.info("After-hours trading OFF — entries stop at 3:30 PM ET")
        logger.info(
            "HOD-only mode: trading watchlist follows alert board; RT mover scanner off",
        )

        return runner

    @staticmethod
    def build_float_filtered_hod_pool(
        candidates: List[dict],
        float_checker: FloatChecker,
        *,
        min_price: float = 2.0,
        max_price: float = 20.0,
        max_float: float = 20_000_000,
        pool_max: int = 50,
        float_workers: int = 6,
        max_network_checks: int = 150,
        sub2_enabled: bool = True,
        sub2_min_price: float = 1.0,
        sub2_max_price: float = 2.0,
        sub2_min_change_pct: float = 10.0,
        sub2_min_day_volume: float = 1_000_000,
        sub2_max_float: float = 10_000_000,
    ) -> List[str]:
        """Keep ranked snapshot passers in price band with float <= max_float.

        Two-pass approach for speed:
        1. Check all eligible symbols against DB/memory cache (instant).
        2. For uncached symbols, do network lookups only for the top-ranked
           ones (capped by max_network_checks) to avoid multi-minute stalls.
        """
        eligible: List[dict] = []
        for row in candidates:
            sym = row.get("symbol")
            if not sym:
                continue
            price = row.get("price")
            if price is None:
                continue
            standard_price_band = min_price <= price <= max_price
            abs_change = abs(float(row.get("abs_change_pct", row.get("change_pct", 0.0)) or 0.0))
            volume = float(row.get("volume", 0.0) or 0.0)
            sub2_price_band = (
                sub2_enabled
                and sub2_min_price <= price < sub2_max_price
                and abs_change >= sub2_min_change_pct
                and volume >= sub2_min_day_volume
            )
            if not (standard_price_band or sub2_price_band):
                continue
            eligible.append(row)

        if not eligible:
            return []

        priority: List[dict] = []
        regular: List[dict] = []
        for row in eligible:
            abs_change = abs(float(row.get("abs_change_pct", row.get("change_pct", 0.0)) or 0.0))
            volume = float(row.get("volume", 0.0) or 0.0)
            if abs_change >= 30.0 and volume >= 100_000:
                priority.append(row)
            else:
                regular.append(row)
        if priority:
            eligible = priority + regular

        logger.info(
            "Float-filtering %d snapshot candidates ($%.0f–$%.0f, float ≤%.0fM, pool max %d)…",
            len(eligible),
            min_price,
            max_price,
            max_float / 1_000_000,
            pool_max,
        )
        t0 = time.time()
        symbols = [r["symbol"] for r in eligible]
        warm = getattr(float_checker, "warm_from_store", None)
        if warm is not None:
            from_db, _ = warm(symbols)
            logger.info(
                "Float cache: %d/%d loaded from DB",
                from_db,
                len(symbols),
            )

        # Pass 1: instant cache-only check (DB + memory)
        passed: List[str] = []
        need_network: List[dict] = []
        SPLIT_SANITY_CAP = 10_000_000_000  # $10B implied market cap = stale post-split data
        for row in eligible:
            if len(passed) >= pool_max:
                break
            sym = row["symbol"]
            price = row.get("price", 0)
            row_abs_change = abs(float(row.get("abs_change_pct", row.get("change_pct", 0.0)) or 0.0))
            row_volume = float(row.get("volume", 0.0) or 0.0)
            row_is_sub2 = (
                sub2_enabled
                and sub2_min_price <= price < sub2_max_price
                and row_abs_change >= sub2_min_change_pct
                and row_volume >= sub2_min_day_volume
            )
            row_max_float = sub2_max_float if row_is_sub2 else max_float
            shares = float_checker.get_float_cached(sym)
            if shares is not None:
                if price > 0 and shares * price > SPLIT_SANITY_CAP:
                    logger.warning(
                        "SPLIT DETECTED %s — cached float %.0f × $%.2f = $%.0fB implied mcap, re-fetching",
                        sym, shares, price, shares * price / 1e9,
                    )
                    need_network.append(row)
                elif shares <= row_max_float:
                    passed.append(sym)
            else:
                need_network.append(row)

        logger.info(
            "Float pass-1 (cache): %d in pool, %d uncached (%.1fs)",
            len(passed), len(need_network), time.time() - t0,
        )

        # Pass 2: network lookups for top-ranked uncached (capped)
        network_checked = 0
        for row in need_network:
            if len(passed) >= pool_max:
                break
            if network_checked >= max_network_checks:
                break
            sym = row["symbol"]
            network_checked += 1
            shares = float_checker.get_float(sym)
            if network_checked % 25 == 0:
                logger.info(
                    "Float network progress: %d/%d checked, %d in pool (%.0fs)",
                    network_checked,
                    min(max_network_checks, len(need_network)),
                    len(passed),
                    time.time() - t0,
                )
            price = row.get("price", 0)
            row_abs_change = abs(float(row.get("abs_change_pct", row.get("change_pct", 0.0)) or 0.0))
            row_volume = float(row.get("volume", 0.0) or 0.0)
            row_is_sub2 = (
                sub2_enabled
                and sub2_min_price <= price < sub2_max_price
                and row_abs_change >= sub2_min_change_pct
                and row_volume >= sub2_min_day_volume
            )
            row_max_float = sub2_max_float if row_is_sub2 else max_float
            if shares is None or shares > row_max_float:
                continue
            passed.append(sym)

        logger.info(
            "Float filter done in %.1fs — %d symbols in HOD bar pool",
            time.time() - t0,
            len(passed),
        )
        return passed

    def _latest_price(self, symbol: str) -> Optional[float]:
        bars = self._bar_buffer.get(symbol.upper())
        if bars:
            return float(bars[-1].close)
        quotes = self._quote_buffer.get(symbol.upper())
        if quotes:
            q = quotes[-1]
            if q.bid > 0 and q.ask > 0:
                return (q.bid + q.ask) / 2.0
        return None

    def _symbol_in_hod_price_band(self, symbol: str) -> bool:
        price = self._latest_price(symbol)
        if price is None:
            return True
        return (
            self._hod_min_price <= price <= self._hod_max_price
            or (
                self._hod_sub2_enabled
                and self._hod_sub2_min_price <= price < self._hod_sub2_max_price
            )
        )

    def _prune_hod_bar_pool_by_price(self) -> List[str]:
        removed: List[str] = []
        kept: List[str] = []
        for sym in self._hod_bar_pool:
            price = self._latest_price(sym)
            if price is not None and not self._symbol_in_hod_price_band(sym):
                removed.append(sym)
            else:
                kept.append(sym)
        if removed:
            logger.info(
                "HOD bar pool: removed %d outside $%.0f–$%.0f: %s",
                len(removed),
                self._hod_min_price,
                self._hod_max_price,
                removed[:12],
            )
        self._hod_bar_pool = kept
        return removed

    def _merge_hod_bar_pool(self, ranked_symbols: List[str]) -> tuple:
        old = set(self._hod_bar_pool)
        merged: List[str] = []
        seen: Set[str] = set()
        for sym in ranked_symbols:
            s = sym.upper()
            if s and s not in seen:
                merged.append(s)
                seen.add(s)
        for sym in self._hod_bar_pool:
            if sym in seen or len(merged) >= self._hod_pool_max:
                continue
            if self._symbol_in_hod_price_band(sym):
                merged.append(sym)
                seen.add(sym)
        merged = merged[: self._hod_pool_max]
        added = [s for s in merged if s not in old]
        removed = [s for s in old if s not in set(merged)]
        self._hod_bar_pool = merged
        return added, removed

    def _hydrate_new_pool_symbols(self, symbols: Sequence[str]) -> None:
        """Fetch bars for new pool symbols and push via event queue."""
        need = []
        for sym in symbols:
            if sym in self._watchlist_pinned:
                continue
            if len(self._bar_buffer.get(sym, deque())) < 10:
                need.append(sym)
        if not need:
            return
        batch = need[: self._hod_hydrate_batch_max]
        logger.info("HOD bar pool: hydrating %d new symbols", len(batch))
        self._hub.add_log("INFO", "HOD bar pool hydrating {} symbols".format(len(batch)))
        try:
            bars_by_symbol = self._fetch_session_bars(batch)
            prior_stats = self._fetch_prior_day_stats(batch)
            if bars_by_symbol or prior_stats:
                try:
                    self._event_queue.put_nowait(
                        BarsLoadedEvent(bars_by_symbol, prior_stats)
                    )
                except queue.Full:
                    pass
        except Exception as exc:
            logger.warning("HOD pool hydrate failed: %s", exc)

    def _run_pool_refresh(self, *, full_scan: bool = False) -> None:
        if self._scanner is None or self._float_checker is None:
            return
        phase = self._market_phase()
        active_phases = ("PRE-MARKET", "OPEN")
        if getattr(self, "_after_hours_enabled", False):
            active_phases = ("PRE-MARKET", "OPEN", "AFTER-HOURS")
        if phase not in active_phases:
            return

        self._pool_refresh_in_progress = True
        try:
            self._run_pool_refresh_inner(full_scan=full_scan)
        finally:
            self._pool_refresh_in_progress = False

    def _run_pool_refresh_inner(self, *, full_scan: bool = False) -> None:
        self._scanner._is_premarket = self._market_phase() == "PRE-MARKET"
        if full_scan or not self._pool_candidates_ready:
            scan_data = self._scanner.scan()
            self._pool_candidates_ready = True
        else:
            scan_data = self._scanner.scan_candidates()

        if scan_data:
            self._watchlist_data = scan_data
            self._hub.on_watchlist_scan(scan_data)

        ranked = self.build_float_filtered_hod_pool(
            scan_data,
            self._float_checker,
            min_price=self._hod_min_price,
            max_price=self._hod_max_price,
            max_float=self._hod_max_float,
            pool_max=self._hod_pool_max,
            sub2_enabled=self._hod_sub2_enabled,
            sub2_min_price=self._hod_sub2_min_price,
            sub2_max_price=self._hod_sub2_max_price,
            sub2_min_change_pct=self._hod_sub2_min_change_pct,
            sub2_min_day_volume=self._hod_sub2_min_day_volume,
            sub2_max_float=self._hod_sub2_max_float,
        )
        added, removed = self._merge_hod_bar_pool(ranked)
        self._prune_hod_bar_pool_by_price()

        # Hydrate new symbols and push pool refresh event
        bars_by_symbol: Dict = {}
        prior_stats: Dict = {}
        if added:
            need = [
                sym for sym in added
                if sym not in self._watchlist_pinned
            ]
            if need:
                batch = need[: self._hod_hydrate_batch_max]
                logger.info("HOD bar pool: hydrating %d new symbols", len(batch))
                self._hub.add_log("INFO", "HOD bar pool hydrating {} symbols".format(len(batch)))
                bars_by_symbol = self._fetch_session_bars(batch)
                prior_stats = self._fetch_prior_day_stats(batch)

        try:
            self._event_queue.put_nowait(
                PoolRefreshEvent(list(self._hod_bar_pool), bars_by_symbol, prior_stats)
            )
        except queue.Full:
            pass

        logger.info(
            "HOD pool refresh [%s]: %d symbols (%d added, %d removed)",
            self._market_phase(),
            len(self._hod_bar_pool),
            len(added),
            len(removed),
        )
        self._hub.add_log(
            "INFO",
            "HOD pool refresh [{}]: {} symbols ({} added, {} removed)".format(
                self._market_phase(), len(self._hod_bar_pool), len(added), len(removed),
            ),
        )

    def _request_pool_refresh_now(self) -> None:
        self._pool_refresh_event.set()

    def _sync_tick_tracker_pool(self) -> None:
        """Update tick tracker with current pool + watchlist symbols."""
        if not self._hod_tick_tracker:
            return
        tracked = set(self._hod_bar_pool) | set(self._watchlist)
        self._hod_tick_tracker.set_tracked_symbols(tracked)
        self._hod_tick_tracker.cleanup_stale()
        self._stream.set_trade_filter(tracked)

    def _start_pool_refresh_worker(self) -> None:
        if self._pool_refresh_thread and self._pool_refresh_thread.is_alive():
            return
        self._pool_refresh_thread = Thread(
            target=self._pool_refresh_worker_loop,
            daemon=True,
            name="hod-pool-refresh",
        )
        self._pool_refresh_thread.start()
        self._request_pool_refresh_now()

    def _pool_refresh_worker_loop(self) -> None:
        while not self._shutdown:
            self._pool_refresh_event.wait(timeout=float(self._hod_pool_refresh_sec))
            self._pool_refresh_event.clear()
            if self._shutdown:
                break
            try:
                self._run_pool_refresh(full_scan=True)
            except Exception as exc:
                logger.error("HOD pool refresh error: %s", exc)

    # ------------------------------------------------------------------
    # Fast scan thread (Thread 2) — rescans candidates every 30s
    # ------------------------------------------------------------------

    def _start_fast_scan_worker(self) -> None:
        if self._fast_scan_thread and self._fast_scan_thread.is_alive():
            return
        self._fast_scan_thread = Thread(
            target=self._fast_scan_worker_loop,
            daemon=True,
            name="hod-fast-scan",
        )
        self._fast_scan_thread.start()

    def _fast_scan_worker_loop(self) -> None:
        time.sleep(5)
        while not self._shutdown:
            try:
                self._run_fast_scan()
            except Exception as exc:
                logger.error("Fast scan error: %s", exc)
            for _ in range(self._fast_scan_interval_sec):
                if self._shutdown:
                    return
                time.sleep(1)

    def _run_fast_scan(self) -> None:
        """Quick rescan of cached candidates to find new movers."""
        if self._scanner is None:
            return
        phase = self._market_phase()
        active_phases = ("PRE-MARKET", "OPEN")
        if getattr(self, "_after_hours_enabled", False):
            active_phases = ("PRE-MARKET", "OPEN", "AFTER-HOURS")
        if phase not in active_phases:
            return
        if not self._pool_candidates_ready:
            return

        t0 = time.time()
        self._scanner._is_premarket = phase == "PRE-MARKET"
        candidates = self._scanner.scan_candidates(readonly=True)

        current_pool = set(self._hod_bar_pool)
        new_movers = []
        for c in candidates:
            sym = c["symbol"]
            # Force-add strong movers even if seen before
            is_strong = c.get("abs_change_pct", 0) >= 10.0 and c.get("volume", 0) >= 200_000
            if sym in current_pool:
                bars = self._bar_buffer.get(sym)
                bars_stale = True
                if bars:
                    last_bar = bars[-1]
                    if last_bar.ts is not None:
                        try:
                            bars_stale = (
                                datetime.now(timezone.utc) - last_bar.ts
                            ).total_seconds() > 120
                        except Exception:
                            bars_stale = False
                    else:
                        bars_stale = False
                if is_strong and (not bars or bars_stale):
                    new_movers.append(c)
                continue
            if sym in self._fast_scan_known and not is_strong:
                continue
            new_movers.append(c)

        if not new_movers:
            logger.debug("Fast scan: no new movers (%.1fs)", time.time() - t0)
            return

        for c in new_movers:
            self._fast_scan_known.add(c["symbol"])

        try:
            self._event_queue.put_nowait(FastScanEvent(new_movers))
        except queue.Full:
            pass

        logger.info(
            "Fast scan: %d new movers in %.1fs — %s",
            len(new_movers),
            time.time() - t0,
            ", ".join(c["symbol"] for c in new_movers[:10]),
        )
        self._hub.add_log(
            "INFO",
            "Fast scan: {} new movers — {}".format(
                len(new_movers), ", ".join(c["symbol"] for c in new_movers[:10]),
            ),
        )

    def _setup_hod_momentum(self, cfg: Settings, pipeline: TradingPipeline) -> None:
        """Initialize HOD Momentum alert system and entry gate."""
        from daytrading.scanner.hod_momentum import (
            HODAlertStore,
            HODTickTracker,
            HODMomentumScanner,
            FormerMomoScanner,
        )
        from daytrading.scanner.scanner_alert_labels import HOD_ENTRY_GATE_ALERTS

        self._hod_entry_gate_alerts = HOD_ENTRY_GATE_ALERTS

        self._hod_alert_ttl_minutes = cfg.hod_momentum_alert_ttl_minutes
        store = HODAlertStore(
            max_rows=cfg.hod_momentum_max_alert_rows,
            ttl_minutes=cfg.hod_momentum_alert_ttl_minutes,
        )
        store.set_on_change(self._on_hod_alerts_changed)
        self._hod_alert_store = store

        self._hod_bar_scanner = HODMomentumScanner(
            store,
            float_checker=self._float_checker,
            min_price=cfg.hod_momentum_min_price,
            max_price=cfg.hod_momentum_max_price,
            max_float=cfg.hod_momentum_max_float,
            min_session_change_pct=cfg.hod_momentum_min_session_change_pct,
            min_day_volume=cfg.hod_momentum_min_day_volume,
            require_break_prior_day_high=cfg.hod_momentum_require_break_prior_day_high,
            rth_only=cfg.hod_momentum_rth_only,
            sub2_enabled=cfg.hod_sub2_momentum_enabled,
            sub2_min_price=cfg.hod_sub2_momentum_min_price,
            sub2_max_price=cfg.hod_sub2_momentum_max_price,
            sub2_min_session_change_pct=cfg.hod_sub2_momentum_min_change_pct,
            sub2_min_day_volume=cfg.hod_sub2_momentum_min_day_volume,
            sub2_max_float=cfg.hod_sub2_momentum_max_float,
            debug=cfg.hod_scanner_debug,
        )

        if cfg.hod_momentum_former_momo_enabled:
            self._hod_former_momo_scanner = FormerMomoScanner(
                store,
                float_checker=self._float_checker,
                min_price=cfg.hod_momentum_former_momo_min_price,
                min_change_from_close_pct=cfg.hod_momentum_former_momo_min_change_pct,
            )

        if cfg.hod_momentum_tick_enabled and self._sip_feed:
            self._hod_tick_tracker = HODTickTracker(
                store,
                float_checker=self._float_checker,
                min_price=(
                    cfg.hod_sub2_momentum_min_price
                    if cfg.hod_sub2_momentum_enabled
                    else cfg.hod_momentum_min_price
                ),
                max_price=cfg.hod_momentum_max_price,
                max_float=cfg.hod_momentum_max_float,
                min_day_volume=cfg.hod_momentum_min_day_volume,
                volume_surge_ratio=cfg.hod_momentum_volume_surge_ratio,
                tick_cooldown_seconds=cfg.hod_momentum_tick_cooldown_seconds,
                require_break_prior_day_high=cfg.hod_momentum_require_break_prior_day_high,
                on_new_symbol=self._enqueue_hod_seed,
                on_needs_seed=self._enqueue_hod_seed,
                on_alert=self._on_breakout_scalp_alert,
                known_symbols=set(self._watchlist),
            )
            logger.info(
                "HOD Momentum tick tracker ON (float <= %.0fM, $%.0f-$%.0f, pool-only mode)",
                cfg.hod_momentum_max_float / 1_000_000,
                cfg.hod_momentum_min_price,
                cfg.hod_momentum_max_price,
            )
        else:
            logger.info(
                "HOD Momentum tick tracker OFF (need SIP feed + HOD_MOMENTUM_TICK_ENABLED=true)",
            )

        pipeline.set_hod_entry_gate(
            self.is_hod_active,
            require=cfg.hod_momentum_require_alert_for_entry,
        )
        if cfg.hod_momentum_require_alert_for_entry:
            logger.info(
                "HOD entry gate ON — entries only for symbols on momentum board (%.0f min)",
                self._hod_alert_ttl_minutes,
            )
        else:
            logger.info("HOD entry gate OFF")

    @staticmethod
    def _parse_hod_alert_time(time_str: Optional[str]) -> Optional[datetime]:
        if not time_str:
            return None
        try:
            return datetime.fromisoformat(str(time_str).replace("Z", "+00:00"))
        except Exception:
            return None

    def _on_hod_alerts_changed(self, alerts: List[dict]) -> None:
        gate_alerts = getattr(self, "_hod_entry_gate_alerts", frozenset())
        for row in alerts:
            sym = row.get("symbol")
            if not sym:
                continue
            alert_ts = self._parse_hod_alert_time(row.get("time")) or datetime.now(
                timezone.utc,
            )
            prev = self._hod_last_alert_at.get(sym)
            if prev is None or alert_ts > prev:
                self._hod_last_alert_at[sym] = alert_ts
            alert_name = row.get("alert_name", "")
            if alert_name in gate_alerts:
                prev_active = self._hod_active.get(sym)
                if prev_active is None or alert_ts > prev_active:
                    self._hod_active[sym] = alert_ts
        self._hub.on_hod_momentum_alerts(alerts)
        self._sync_watchlist_to_hod_alerts()

    def is_hod_active(self, symbol: str) -> bool:
        ts = self._hod_active.get(symbol)
        if ts is None:
            return False
        elapsed = (datetime.now(timezone.utc) - ts).total_seconds()
        return elapsed < self._hod_alert_ttl_minutes * 60

    def _on_breakout_scalp_alert(self, symbol: str, price: float) -> None:
        """Called from tick_tracker when a HOD alert fires — queue for instant scalp."""
        self._pending_breakout_scalps.append((symbol, price, time.time()))

    def _start_hod_seed_worker(self) -> None:
        if self._hod_seed_thread and self._hod_seed_thread.is_alive():
            return
        self._hod_seed_thread = Thread(
            target=self._hod_seed_worker_loop,
            daemon=True,
            name="hod-seed-worker",
        )
        self._hod_seed_thread.start()

    def _eligible_for_hod_seed(self, sym: str) -> bool:
        """Seed symbols in pool or watchlist."""
        if sym in self._watchlist_pinned:
            return False
        if sym in self._watchlist_set:
            return True
        if sym in set(self._hod_bar_pool):
            return True
        return False

    def _enqueue_hod_seed(self, symbol: str) -> None:
        """Queue SIP-discovered symbol for background bar load (non-blocking)."""
        sym = symbol.upper().strip()
        if not sym or not self._eligible_for_hod_seed(sym):
            return
        with self._hod_seed_lock:
            if sym in self._hod_seed_pending:
                return
            if len(self._hod_seed_pending) >= self._hod_seed_max:
                return
            self._hod_seed_pending.add(sym)
            self._hod_seed_queue.append(sym)
        self._hod_seed_event.set()

    def _enqueue_hod_seed_symbols(self, symbols: Sequence[str]) -> None:
        """Re-queue symbols for retry (skips eligibility check)."""
        with self._hod_seed_lock:
            for sym in symbols:
                if sym not in self._hod_seed_pending:
                    self._hod_seed_pending.add(sym)
                    self._hod_seed_queue.append(sym)
        self._hod_seed_event.set()

    def _pull_hod_seed_batch(self, max_count: int) -> List[str]:
        batch: List[str] = []
        with self._hod_seed_lock:
            while self._hod_seed_queue and len(batch) < max_count:
                sym = self._hod_seed_queue.popleft()
                self._hod_seed_pending.discard(sym)
                batch.append(sym)
        return batch

    def _hod_seed_rate_limit(self) -> int:
        """Symbols still allowed this minute (0 = wait)."""
        now = time.time()
        if now - self._hod_seed_minute_start >= 60.0:
            self._hod_seed_minute_start = now
            self._hod_seed_processed_this_minute = 0
        return max(0, self._hod_seed_max_per_minute - self._hod_seed_processed_this_minute)

    def _hod_seed_worker_loop(self) -> None:
        while not self._shutdown:
            self._hod_seed_event.wait(timeout=2.0)
            self._hod_seed_event.clear()
            if self._shutdown:
                break
            if self._pool_refresh_in_progress or self._bar_hydrate_paused():
                continue
            quota = self._hod_seed_rate_limit()
            if quota <= 0:
                time.sleep(1.0)
                continue
            batch_size = min(self._hod_seed_batch_size, quota)
            batch = self._pull_hod_seed_batch(batch_size)
            if not batch:
                continue
            try:
                self._process_hod_seed_batch(batch)
                self._hod_seed_processed_this_minute += len(batch)
            except Exception as exc:
                logger.warning("HOD seed batch failed: %s", exc)
            if self._bar_fetch_batch_delay_sec > 0:
                time.sleep(self._bar_fetch_batch_delay_sec)
            with self._hod_seed_lock:
                if self._hod_seed_queue:
                    self._hod_seed_event.set()

    def _bar_hydrate_paused(self) -> bool:
        return time.time() < self._hydrate_paused_until

    def _record_bar_fetch_failures(self, failure_count: int) -> None:
        if failure_count <= 0:
            return
        now = time.time()
        self._network_failure_times.extend([now] * failure_count)
        cutoff = now - 60.0
        self._network_failure_times = [
            t for t in self._network_failure_times if t >= cutoff
        ]
        if len(self._network_failure_times) >= 10:
            self._hydrate_paused_until = now + 30.0
            self._network_failure_times.clear()
            logger.warning(
                "Network unstable — pausing bar hydration for 30s "
                "(WebSocket stays connected)",
            )

    def _process_hod_seed_batch(self, symbols: Sequence[str]) -> None:
        """Load bars for a batch of pool/watchlist symbols (batched REST).

        Pushes a BarsLoadedEvent to the event queue instead of writing buffers directly.
        """
        if self._bar_hydrate_paused():
            return
        need: List[str] = []
        for raw in symbols:
            sym = raw.upper().strip()
            if not sym or sym in self._watchlist_pinned:
                continue
            if sym in self._hod_seed_blacklist:
                continue
            if len(self._bar_buffer.get(sym, deque())) >= 10:
                continue
            need.append(sym)
        if not need:
            return
        preview = ", ".join(need[:5])
        if len(need) > 5:
            preview += ", ..."
        logger.info("HOD seed batch: loading %d symbols — %s", len(need), preview)
        try:
            bars_by_symbol = self._fetch_session_bars(need)
            prior_stats = self._fetch_prior_day_stats(need)
            if bars_by_symbol or prior_stats:
                try:
                    self._event_queue.put_nowait(
                        BarsLoadedEvent(bars_by_symbol, prior_stats)
                    )
                except queue.Full:
                    pass
            stream_syms = [s for s in need if s in self._watchlist_set]
            if stream_syms:
                self._ensure_streaming_symbols(stream_syms)
            failed = [s for s in need if s not in bars_by_symbol]
            if failed:
                # Track retries — stop re-queuing symbols that always fail
                retryable = []
                for s in failed:
                    self._hod_seed_retries[s] = self._hod_seed_retries.get(s, 0) + 1
                    if self._hod_seed_retries[s] >= 3:
                        self._hod_seed_blacklist.add(s)
                        logger.debug("HOD seed: blacklisting %s (3 consecutive failures)", s)
                    else:
                        retryable.append(s)
                if retryable:
                    self._enqueue_hod_seed_symbols(retryable)
        except Exception as exc:
            logger.warning("HOD seed batch failed: %s", exc)

    def _on_hod_symbol_discovered(self, symbol: str) -> None:
        """Queue the symbol for background hydration via seed worker."""
        self._enqueue_hod_seed_symbols([symbol])

    def _protected_watchlist_symbols(self) -> Set[str]:
        open_syms = {
            sym for sym, pos in self._pipeline.portfolio.positions.items()
            if not pos.is_flat
        }
        tracked = set(self._pipeline.exit_manager.tracked.keys())
        return self._watchlist_pinned | open_syms | tracked

    def _trade_symbol_set(self) -> Set[str]:
        """Symbols eligible for the trading pipeline this cycle."""
        return set(self._watchlist) | self._protected_watchlist_symbols()

    def _trade_universe(
        self, bar_universe: Dict[str, List[Bar]],
    ) -> Dict[str, List[Bar]]:
        """Filter bar buffer to watchlist + open positions + exit-tracked only."""
        allowed = self._trade_symbol_set()
        return {s: bars for s, bars in bar_universe.items() if s in allowed}

    def _hod_watchlist_symbols(self) -> Set[str]:
        """Symbols with a HOD alert within the watchlist TTL window."""
        now = datetime.now(timezone.utc)
        ttl_secs = self._hod_watchlist_ttl_minutes * 60
        active: Set[str] = set(self._news_pinned)
        for sym, ts in list(self._hod_last_alert_at.items()):
            if (now - ts).total_seconds() < ttl_secs:
                active.add(sym)
            else:
                del self._hod_last_alert_at[sym]
        return active

    def _sync_watchlist_to_hod_alerts(self, alerts: Optional[List[dict]] = None) -> None:
        """Keep the trading watchlist aligned with recent HOD alerts (TTL)."""
        del alerts  # TTL tracked in _hod_last_alert_at via _on_hod_alerts_changed
        alert_syms = self._hod_watchlist_symbols()

        protected = self._protected_watchlist_symbols()
        target = (alert_syms | protected)
        if len(target) > self._max_watchlist:
            ordered = []
            seen: Set[str] = set()
            for sym in protected:
                if sym not in seen:
                    ordered.append(sym)
                    seen.add(sym)
            recent_alerts = sorted(
                alert_syms,
                key=lambda s: self._hod_last_alert_at.get(
                    s, datetime.min.replace(tzinfo=timezone.utc),
                ),
                reverse=True,
            )
            for sym in recent_alerts:
                if sym not in seen:
                    ordered.append(sym)
                    seen.add(sym)
                if len(ordered) >= self._max_watchlist:
                    break
            target = set(ordered)

        to_add = sorted(target - self._watchlist_set)
        to_remove = sorted(self._watchlist_set - target)

        if to_add:
            self._add_symbols_to_watchlist(to_add)
        if to_remove:
            self._remove_symbols_from_watchlist(to_remove)
        self._publish_trading_watchlist()

    def _publish_trading_watchlist(self) -> None:
        self._hub.on_trading_watchlist(
            list(self._watchlist),
            pinned=sorted(self._watchlist_pinned),
        )

    def _publish_hod_alert_board(self) -> None:
        if self._hod_alert_store is not None:
            self._hub.on_hod_momentum_alerts(self._hod_alert_store.snapshot())

    def _ensure_streaming_symbols(self, symbols: Sequence[str]) -> None:
        """Queue WS bar/quote subscribe (safe from background threads)."""
        if not symbols:
            return
        self._stream.subscribe(list(symbols), bars=True, quotes=True)
        self._stream.flush_pending_subscriptions()

    def _add_symbols_to_watchlist(self, symbols: Sequence[str]) -> None:
        new_symbols = [s for s in symbols if s not in self._watchlist_set]
        if not new_symbols:
            return
        logger.info(
            "WATCHLIST (HOD alert): adding %d — %s",
            len(new_symbols), new_symbols,
        )
        self._watchlist.extend(new_symbols)
        self._watchlist_set.update(new_symbols)
        self._stream.subscribe(new_symbols, bars=True, quotes=True)
        if self._hod_tick_tracker:
            self._hod_tick_tracker.add_known_symbols(new_symbols)
        self._enqueue_hod_seed_symbols(new_symbols)
        self._hub.add_log("INFO", "HOD watchlist +{}".format(", ".join(new_symbols)))

        if self._news_checker:
            for sym in new_symbols:
                if sym in self._news_pinned:
                    continue
                try:
                    score, headlines = self._news_checker.get_sentiment(sym)
                    if score > 0:
                        self._news_pinned.add(sym)
                        logger.info(
                            "NEWS PIN %s: positive sentiment %.2f — keeping all day — %s",
                            sym, score, headlines[:2],
                        )
                except Exception:
                    pass

    def _remove_symbols_from_watchlist(self, symbols: Sequence[str]) -> None:
        removed = []
        for sym in symbols:
            if sym not in self._watchlist_set:
                continue
            self._watchlist.remove(sym)
            self._watchlist_set.discard(sym)
            self._skip_counts.pop(sym, None)
            removed.append(sym)
            self._bar_buffer.pop(sym, None)
            self._quote_buffer.pop(sym, None)
            self._tick_buffer.pop(sym, None)
        if removed:
            logger.info(
                "WATCHLIST (no HOD alert): removed %d — %s — now %d: %s",
                len(removed), removed, len(self._watchlist), self._watchlist,
            )
            self._hub.add_log("INFO", "HOD watchlist -{}".format(", ".join(removed)))

    def _load_prior_day_stats(self, symbols: Sequence[str]) -> None:
        if not symbols:
            return
        try:
            from daytrading.scanner.hod_momentum.prior_day import fetch_prior_day_stats

            stats = fetch_prior_day_stats(
                self._hist._client,
                symbols,
                feed=self._hist._feed,
            )
            self._prior_day_stats.update(stats)
            logger.info("Loaded prior-day stats for %d symbols", len(stats))
        except Exception as exc:
            logger.warning("Prior-day stats load failed: %s", exc)

    def _fetch_session_bars(self, symbols: Sequence[str]) -> Dict[str, List]:
        """Fetch today's 1m bars and return them (does NOT write to buffers)."""
        if not symbols:
            return {}
        if self._bar_hydrate_paused():
            return {}
        try:
            now_utc = datetime.now(timezone.utc)
            now_et_time = now_utc.astimezone(ET)
            session_start_et = now_et_time.replace(hour=4, minute=0, second=0, microsecond=0)
            session_start_utc = session_start_et.astimezone(timezone.utc)
            bars = self._hist.get_bars(
                list(symbols),
                timeframe="1Min",
                limit=self._session_bar_limit,
                start=session_start_utc,
                end=now_utc,
            )
            self._record_bar_fetch_failures(self._hist.last_fetch_failures)
            return bars
        except Exception as exc:
            logger.warning("Session bar fetch failed: %s", exc)
            return {}

    def _fetch_prior_day_stats(self, symbols: Sequence[str]) -> Dict:
        """Fetch prior-day stats and return them (does NOT write to state)."""
        if not symbols:
            return {}
        try:
            from daytrading.scanner.hod_momentum.prior_day import fetch_prior_day_stats
            stats = fetch_prior_day_stats(
                self._hist._client,
                symbols,
                feed=self._hist._feed,
            )
            return stats
        except Exception as exc:
            logger.warning("Prior-day stats fetch failed: %s", exc)
            return {}

    def _load_session_bars_for_symbols(self, symbols: Sequence[str]) -> None:
        """Fetch today's 1m bars into bar_buffer (movers / HOD universe hydration)."""
        if not symbols:
            return
        if self._bar_hydrate_paused():
            logger.debug("Bar hydrate skipped — network pause active")
            return
        try:
            now_utc = datetime.now(timezone.utc)
            now_et = now_utc.astimezone(ET)
            session_start_et = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
            session_start_utc = session_start_et.astimezone(timezone.utc)
            bars = self._hist.get_bars(
                list(symbols),
                timeframe="1Min",
                limit=self._session_bar_limit,
                start=session_start_utc,
                end=now_utc,
            )
            self._record_bar_fetch_failures(self._hist.last_fetch_failures)
            today_et = now_et.date()
            for symbol, symbol_bars in bars.items():
                today_bars: List[Bar] = []
                for b in symbol_bars:
                    if b.ts is not None:
                        try:
                            if b.ts.astimezone(ET).date() == today_et:
                                today_bars.append(b)
                        except Exception:
                            today_bars.append(b)
                    else:
                        today_bars.append(b)
                if today_bars:
                    buf = deque(today_bars[-self._max_bars_per_symbol:], maxlen=self._max_bars_per_symbol)
                    self._bar_buffer[symbol] = buf
        except Exception as exc:
            logger.warning("Session bar load failed: %s", exc)

    def _hydrate_hod_bar_pool(self) -> None:
        """Load bars for HOD bar pool symbols missing session data."""
        syms = [s for s in self._hod_bar_pool if s and s not in self._watchlist_pinned]
        if syms:
            self._hydrate_new_pool_symbols(syms)

    def _expand_hod_universe(
        self, universe: Dict[str, List[Bar]],
    ) -> Dict[str, List[Bar]]:
        """Watchlist + HOD bar pool + tape-hot symbols — for alert evaluation."""
        expanded = {s: list(b) for s, b in universe.items()}
        pool_syms = {
            s for s in self._hod_bar_pool
            if self._symbol_in_hod_price_band(s)
        }
        extra: Set[str] = set(self._watchlist) | pool_syms

        need_bars = [s for s in extra if len(expanded.get(s, [])) < 10]
        if need_bars:
            self._enqueue_hod_seed_symbols(need_bars[:25])
            for s in need_bars:
                buf = self._bar_buffer.get(s)
                if buf:
                    expanded[s] = list(buf)

        missing_prior = [s for s in expanded if s not in self._prior_day_stats]
        if missing_prior:
            self._enqueue_hod_seed_symbols(missing_prior[:80])

        return expanded

    def _seed_hod_session(self, symbol: str) -> None:
        """Sync tick HOD tracker with today's bars (volume + session high)."""
        if self._hod_tick_tracker is None:
            return
        bars = self._bar_buffer.get(symbol)
        if bars:
            prior = self._prior_day_stats.get(symbol)
            self._hod_tick_tracker.update_session_from_bars(
                symbol, bars, prior_day=prior,
            )

    def _seed_all_hod_sessions(self) -> None:
        if self._hod_tick_tracker is None:
            return
        for sym, bars in self._bar_buffer.items():
            if bars:
                prior = self._prior_day_stats.get(sym)
                self._hod_tick_tracker.update_session_from_bars(
                    sym, bars, prior_day=prior,
                )

    @staticmethod
    def _now_et() -> datetime:
        return now_et()

    @classmethod
    def _is_trading_day(cls, when: Optional[datetime] = None) -> bool:
        """Weekday that is not a US market holiday."""
        try:
            t = when or cls._now_et()
            return is_us_trading_day(t.date())
        except Exception:
            return True

    @classmethod
    def _is_market_open(cls) -> bool:
        """Check if US regular market is open (9:30 AM - 4:00 PM ET, trading days)."""
        try:
            t = cls._now_et()
            if not cls._is_trading_day(t):
                return False
            market_open = t.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = t.replace(hour=16, minute=0, second=0, microsecond=0)
            return market_open <= t <= market_close
        except Exception:
            return True

    @classmethod
    def _is_premarket(cls) -> bool:
        """Check if we're in pre-market hours (4:00 AM - 9:30 AM ET, trading days)."""
        try:
            t = cls._now_et()
            if not cls._is_trading_day(t):
                return False
            premarket_start = t.replace(hour=4, minute=0, second=0, microsecond=0)
            market_open = t.replace(hour=9, minute=30, second=0, microsecond=0)
            return premarket_start <= t < market_open
        except Exception:
            return False

    @classmethod
    def _is_afterhours(cls) -> bool:
        """Check if we're in after-hours (4:00 PM - 8:00 PM ET, trading days)."""
        try:
            t = cls._now_et()
            if not cls._is_trading_day(t):
                return False
            ah_start = t.replace(hour=16, minute=0, second=0, microsecond=0)
            ah_end = t.replace(hour=20, minute=0, second=0, microsecond=0)
            return ah_start < t <= ah_end
        except Exception:
            return False

    @classmethod
    def _market_phase(cls) -> str:
        """Return current market phase as a string."""
        if cls._is_market_open():
            return "OPEN"
        if cls._is_premarket():
            return "PRE-MARKET"
        if cls._is_afterhours():
            return "AFTER-HOURS"
        return "CLOSED"

    def _is_in_trading_window(self, now_et: Optional[datetime] = None) -> bool:
        """True when new entries/scans are allowed (premarket through RTH or after-hours)."""
        now_et = now_et or self._now_et()
        if not self._is_trading_day(now_et):
            return False
        day_start = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
        if getattr(self, "_after_hours_enabled", False):
            day_end = now_et.replace(hour=20, minute=0, second=0, microsecond=0)
        else:
            day_end = now_et.replace(hour=15, minute=30, second=0, microsecond=0)
        return day_start <= now_et <= day_end

    def _connect_stream(self) -> None:
        """Connect the WebSocket stream."""
        self._stream.on_bar(self._on_bar)
        self._stream.on_quote(self._on_quote)
        self._stream.on_trade(self._on_stream_trade)
        self._stream.subscribe(self._watchlist, bars=True, quotes=True)
        # Subscribe to SPY bars for market panic detection
        self._stream.subscribe(["SPY"], bars=True, quotes=False)

        # SIP: market-wide trade tape for HOD tick alerts only (not RT mover scanner)
        if self._hod_tick_tracker is not None:
            self._stream.subscribe_all_trades()
            self._sync_tick_tracker_pool()
            logger.info("HOD tick tape active — SIP trades for HOD breakout detection")

        import os, time as _time
        _lock = os.path.join(os.path.dirname(__file__), ".stream_lock")
        if os.path.exists(_lock):
            try:
                age = _time.time() - os.path.getmtime(_lock)
                if age < 90:
                    wait = int(max(30, 60 - age))
                    logger.warning(
                        "Recent stream shutdown detected (%ds ago) — "
                        "waiting %ds for Alpaca to release the old connection …",
                        int(age), wait,
                    )
                    _time.sleep(wait)
            except Exception:
                pass
        try:
            with open(_lock, "w") as f:
                f.write(str(_time.time()))
        except Exception:
            pass

        self._stream.start(background=True)

    def run(self) -> None:
        """Main loop: load history -> stream -> run cycles until shutdown."""
        self._setup_signals()

        # Start dashboard web server (before heavy hydration)
        start_dashboard(self._hub, port=self._dashboard_port)
        self._start_hod_seed_worker()
        self._start_pool_refresh_worker()
        self._start_fast_scan_worker()

        # Push initial account info and trade history to dashboard
        try:
            acct = self._broker.get_account()
            self._hub.on_startup(acct["cash"], acct["equity"], acct["buying_power"])
            self._hub.starting_cash = acct["equity"]
            self._hub.total_pnl = 0.0
            logger.info(
                "Account equity=$%.2f — today's P&L starts at $0.00",
                acct["equity"],
            )
        except Exception:
            pass

        # Load today's trade history into Recent Activity (P&L already set from equity)
        self._sync_trade_history()

        # Initialize AI Trade Analyzer
        from daytrading.analytics.trade_analyzer import TradeAnalyzer
        self._trade_analyzer = TradeAnalyzer(min_trades=3, max_block_trades=3)

        if self._watchlist_data:
            self._hub.on_watchlist_scan(self._watchlist_data)

        self._publish_trading_watchlist()
        self._publish_hod_alert_board()

        # Push market phase immediately so dashboard shows correct status
        phase = self._market_phase()
        self._hub.on_market_status(phase != "CLOSED", False, phase)

        # Sync existing Alpaca positions into pipeline (survives restarts)
        self._sync_alpaca_positions()

        # Push synced positions to dashboard immediately
        self._push_positions_from_alpaca()

        # Start background thread for real-time position updates (every 3s)
        self._pos_sync_thread = Thread(target=self._position_sync_loop, daemon=True)
        self._pos_sync_thread.start()

        logger.info("=" * 60)
        logger.info("ALPACA PAPER TRADING — STARTING")
        logger.info("Watchlist: %s", self._watchlist)
        logger.info("Dashboard: http://localhost:%d", self._dashboard_port)
        logger.info("=" * 60)

        # 1. Load SPY history; HOD bar pool hydrates via background refresh + SIP seed
        self._load_history()
        if self._hod_bar_pool:
            self._hydrate_hod_bar_pool()

        # Wire multi-timeframe support into the pipeline
        self._pipeline._execution_timer = self._exec_timer
        self._pipeline._bar_aggregator = self._bar_aggregator
        # Give verifiers access to the 5-min bar aggregator
        for v in self._pipeline._verifiers.values():
            if hasattr(v, '_bar_aggregator'):
                v._bar_aggregator = self._bar_aggregator
            if hasattr(v, '_tick_buffer'):
                v._tick_buffer = self._tick_buffer
            if hasattr(v, '_quote_buffer'):
                v._quote_buffer = self._quote_buffer

        # 2. Initial classification with historical data
        if self._bar_buffer:
            logger.info("Initial classification with historical data...")
            self._run_one_cycle(0)

        # 3. Setup streaming — connect during pre-market, market hours, or after-hours
        phase = self._market_phase()
        stream_connected = False
        if phase in ("OPEN", "PRE-MARKET", "AFTER-HOURS"):
            logger.info("Market phase: %s — connecting stream...", phase)
            self._connect_stream()
            stream_connected = True
        elif not self._is_trading_day():
            logger.info(
                "Market is CLOSED (weekend or US holiday). "
                "Will auto-connect on the next trading day."
            )
        else:
            logger.info(
                "Market is currently CLOSED. "
                "Pre-market starts at 4:00 AM ET. "
                "Will auto-connect when trading hours begin."
            )

        self._hub.on_market_status(phase != "CLOSED", stream_connected, phase)

        # 4. Main cycle loop
        # Runs on new bar data OR every poll_interval seconds (for pre-market/low-activity)
        poll_interval = 30.0  # run scanner every 30s even without new bars
        logger.info("Pipeline running — scanning every %.0fs (Ctrl+C to stop)", poll_interval)
        cycle_count = 0
        last_market_check = 0.0
        last_cycle_time = time.time()
        last_market_phase = phase
        if phase == "OPEN":
            self._last_session_reset_day = self._now_et().date().isoformat()

        try:
            while not self._shutdown:
                now_ts = time.time()

                # Trading window: 4:00 AM ET through 3:30 PM (or 8:00 PM if after-hours on).
                now_et = self._now_et()
                in_trading_window = self._is_in_trading_window(now_et)
                self._maybe_daily_session_reset(now_et, self._market_phase())
                if getattr(self, "_after_hours_enabled", False):
                    flatten_after = now_et.replace(
                        hour=20, minute=0, second=0, microsecond=0,
                    )
                else:
                    flatten_after = now_et.replace(
                        hour=15, minute=30, second=0, microsecond=0,
                    )

                if not in_trading_window and not getattr(self, '_eod_flattened', False):
                    if now_et > flatten_after and self._is_trading_day(now_et):
                        tracked = self._pipeline.exit_manager.tracked
                        if tracked:
                            flatten_label = (
                                "8:00 PM ET" if getattr(self, "_after_hours_enabled", False)
                                else "3:30 PM ET"
                            )
                            logger.info(
                                "%s — FLATTENING %d positions (no overnight holds)",
                                flatten_label, len(tracked),
                            )
                            self._hub.add_log(
                                "WARNING", "{} — closing all positions".format(flatten_label),
                            )
                            try:
                                flatten_ts = datetime.now(timezone.utc)
                                positions_to_log = {
                                    sym: pos for sym, pos in tracked.items()
                                }
                                for sym in list(tracked.keys()):
                                    self._clear_broker_stop(sym)
                                    self._pipeline.exit_manager.untrack(sym)
                                self._broker.close_all_positions()
                                for sym, pos in positions_to_log.items():
                                    alpaca_pos = self._broker.get_positions().get(sym)
                                    exit_px = float(alpaca_pos["current_price"]) if alpaca_pos else pos.entry_price
                                    qty = pos.remaining_qty or pos.original_qty
                                    flatten_fill = Fill(
                                        symbol=sym,
                                        side=Side.SELL,
                                        quantity=qty,
                                        price=exit_px,
                                        ts=flatten_ts,
                                        commission=0.0,
                                    )
                                    self._record_trade_exit(
                                        flatten_fill, pos.entry_price, "eod_flatten",
                                    )
                                    logger.info(
                                        "EOD FLATTEN %s: entry=%.2f exit=%.2f qty=%.0f",
                                        sym, pos.entry_price, exit_px, qty,
                                    )
                            except Exception as exc:
                                logger.error("Error flattening positions: %s", exc)
                        self._eod_flattened = True
                        self._news_pinned.clear()

                        # Run nightly analysis after flattening (weekdays with trades only)
                        if self._pipeline._daily_pnl != 0 or len(self._pipeline._daily_losers) > 0:
                            self._run_nightly_analysis()

                if now_ts - last_market_check > 60:
                    last_market_check = now_ts
                    phase = self._market_phase()
                    reload_phases = ("PRE-MARKET", "OPEN")
                    if getattr(self, "_after_hours_enabled", False):
                        reload_phases = ("PRE-MARKET", "OPEN", "AFTER-HOURS")
                    if phase != last_market_phase and phase in reload_phases:
                        logger.info("Market phase %s → %s — reloading history", last_market_phase, phase)
                        self._load_history()
                    if phase == "PRE-MARKET" and last_market_phase != "PRE-MARKET":
                        logger.info("PRE-MARKET — daily session reset + refreshing HOD bar pool")
                        self._maybe_daily_session_reset(now_et, phase, force=True)
                        self._request_pool_refresh_now()
                    if phase == "OPEN" and last_market_phase != "OPEN":
                        logger.info("Market OPEN — refreshing HOD bar pool")
                        self._request_pool_refresh_now()
                    if (
                        getattr(self, "_after_hours_enabled", False)
                        and phase == "AFTER-HOURS"
                        and last_market_phase != "AFTER-HOURS"
                    ):
                        logger.info("AFTER-HOURS — refreshing HOD bar pool")
                        self._request_pool_refresh_now()
                    if phase != last_market_phase:
                        logger.info("Market phase: %s", phase)
                    last_market_phase = phase
                    self._hub.on_market_status(
                        phase != "CLOSED", stream_connected, phase,
                    )
                    if not stream_connected and phase != "CLOSED":
                        logger.info(
                            "Market phase: %s — connecting WebSocket stream...",
                            phase,
                        )
                        self._connect_stream()
                        stream_connected = True
                        self._hub.on_market_status(True, True, phase)

                got_data = self._new_data.wait(timeout=self._cycle_interval)
                if self._shutdown:
                    break

                # Drain all events from the queue into main-thread-owned buffers
                got_events = self._drain_events()
                if got_events:
                    got_data = True

                loop_elapsed = time.time() - now_ts
                if loop_elapsed > 5:
                    logger.warning(
                        "Main loop slow: %.1fs (got_data=%s)", loop_elapsed, got_data,
                    )

                # Always check exits every second for open positions
                # This prevents stop-loss slippage from 30s cycle gaps
                if self._pipeline.exit_manager.tracked:
                    self._check_exits_only()

                # Execute any timed entries that the 10-sec timer released
                while self._timed_signal_queue:
                    try:
                        sig = self._timed_signal_queue.popleft()
                        self._execute_timed_signal(sig)
                    except IndexError:
                        break

                # Also check for execution timer timeouts
                for timed_sig in self._exec_timer.check_timeouts():
                    self._execute_timed_signal(timed_sig)

                # Process instant breakout scalps from HOD tick alerts
                if in_trading_window:
                    self._process_breakout_scalps()

                should_run = False
                if got_data:
                    self._new_data.clear()
                    should_run = True
                elif now_ts - last_cycle_time >= poll_interval and self._bar_buffer:
                    should_run = True

                if should_run:
                    cycle_count += 1
                    last_cycle_time = now_ts
                    if in_trading_window:
                        self._run_one_cycle(cycle_count)
                    else:
                        # Outside trading window: still check exits but no new entries
                        if self._pipeline.exit_manager.tracked:
                            self._check_exits_only()
                        self._hub.on_cycle_heartbeat(
                            cycle_count,
                            "entry window closed",
                        )
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self._shutdown_gracefully()

    def _sync_trade_history(self) -> None:
        """Load today's closed orders into the dashboard and set daily P&L.

        Total P&L uses Alpaca's authoritative daily figure (equity − last_equity).
        Trade history includes overnight positions so entry prices are correct.
        """
        try:
            from collections import defaultdict as _defaultdict
            from datetime import timedelta
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            from daytrading.models import Fill, Side

            acct = self._broker.client.get_account()
            alpaca_daily_pnl = float(acct.equity) - float(acct.last_equity)

            today_start = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0,
            )

            lookback_start = (
                datetime.utcnow() - timedelta(days=3)
            ).replace(hour=0, minute=0, second=0, microsecond=0)
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=lookback_start,
                limit=500,
            )
            orders = self._broker.client.get_orders(filter=req)

            buy_count = 0
            sell_count = 0
            # Build holdings from ALL orders (including prev days)
            # so sells today match correctly against older buys
            all_holdings = _defaultdict(list)

            for o in reversed(orders):  # oldest first
                self._last_synced_order_ids.add(str(o.id))

                qty = float(o.filled_qty or 0)
                price = float(o.filled_avg_price or 0)
                if qty <= 0 or price <= 0:
                    continue

                status_val = o.status.value if hasattr(o.status, 'value') else str(o.status)
                side_val = o.side.value if hasattr(o.side, 'value') else str(o.side)
                side_val = side_val.lower()
                fill_time = o.filled_at or o.submitted_at
                is_today = fill_time and fill_time.replace(tzinfo=None) >= today_start

                if "buy" in side_val:
                    all_holdings[o.symbol].append((qty, price))
                    if is_today:
                        side = Side.BUY
                        fill = Fill(symbol=o.symbol, side=side, quantity=qty,
                                    price=price, ts=fill_time, commission=0.0)
                        self._hub.on_fill(fill, "entry")
                        buy_count += 1
                else:
                    # LIFO match against all holdings (including overnight)
                    remaining = qty
                    matched_cost = 0.0
                    matched_qty = 0.0
                    while remaining > 0 and all_holdings[o.symbol]:
                        lot_qty, lot_price = all_holdings[o.symbol][-1]
                        take = min(remaining, lot_qty)
                        matched_cost += take * lot_price
                        matched_qty += take
                        remaining -= take
                        if take >= lot_qty:
                            all_holdings[o.symbol].pop()
                        else:
                            all_holdings[o.symbol][-1] = (lot_qty - take, lot_price)

                    if not is_today:
                        continue

                    entry_price = matched_cost / matched_qty if matched_qty > 0 else 0.0
                    side = Side.SELL
                    fill = Fill(symbol=o.symbol, side=side, quantity=qty,
                                price=price, ts=fill_time, commission=0.0)
                    self._hub.on_exit_fill(fill, entry_price=entry_price,
                                           skip_pnl_accum=True)
                    sell_count += 1

            logger.info(
                "Synced trade history: %d buys, %d sells, seeded %d order IDs, "
                "Alpaca daily P&L=$%.2f",
                buy_count, sell_count, len(self._last_synced_order_ids),
                alpaca_daily_pnl,
            )
            # Use the sum of today's actual trade P&Ls (not Alpaca equity change
            # which includes overnight position value changes we didn't trade)
            trade_pnl = sum(
                t.pnl for t in self._hub.trades if t.pnl is not None
            )
            logger.info("Trade P&L=$%.2f (Alpaca daily=$%.2f)", trade_pnl, alpaca_daily_pnl)
            self._hub.total_pnl = trade_pnl
            self._hub.pnl_history.append({
                "ts": datetime.utcnow().isoformat() + "Z",
                "pnl": trade_pnl,
            })
        except Exception as exc:
            logger.warning("Could not sync trade history: %s", exc)

    def _sync_alpaca_positions(self) -> None:
        """Load existing Alpaca positions into the pipeline portfolio and exit manager.

        This ensures the bot won't re-enter stocks it already holds after a restart,
        and that stop losses are active for all existing positions.
        """
        try:
            alpaca_positions = self._broker.get_positions()
            rec = self._reconciler.reconcile(
                alpaca_positions,
                self._pipeline.portfolio,
                self._pipeline.exit_manager,
            )
            for sym in rec.adopted:
                tracked = self._pipeline.exit_manager.tracked.get(sym)
                stop = tracked.stop_loss if tracked else None
                if stop and tracked:
                    self._arm_broker_protection(sym, tracked.remaining_qty, stop)
            if alpaca_positions:
                logger.info(
                    "Synced %d positions from Alpaca (%d adopted)",
                    len(alpaca_positions), len(rec.adopted),
                )
            else:
                logger.info("No existing Alpaca positions — starting fresh")
        except Exception as exc:
            logger.warning("Could not sync Alpaca positions: %s", exc)

    def _push_positions_from_alpaca(self) -> None:
        """Reconcile portfolio + exit manager with Alpaca (broker is source of truth)."""
        try:
            alpaca_positions = self._broker.get_positions()
            self._reconciler.reconcile(
                alpaca_positions,
                self._pipeline.portfolio,
                self._pipeline.exit_manager,
            )
            prices = {
                sym: float(data.get("current_price", data["avg_entry"]))
                for sym, data in alpaca_positions.items()
            }
            self._hub.on_position_update(self._pipeline.portfolio.positions, prices)
        except Exception as exc:
            logger.debug("Position sync error: %s", exc)

    def _price_buffers_snapshot(self) -> tuple:
        quotes = {s: list(q) for s, q in self._quote_buffer.items() if q}
        bars = {s: list(b) for s, b in self._bar_buffer.items() if b}
        return quotes, bars

    def _live_prices(self, symbols: Sequence[str]) -> Dict[str, float]:
        quotes, bars = self._price_buffers_snapshot()
        broker_pos = self._broker.get_positions()
        return resolve_live_prices(
            symbols,
            broker_positions=broker_pos,
            quotes=quotes,
            bars=bars,
        )

    def _arm_broker_protection(
        self,
        symbol: str,
        qty: float,
        stop_loss: Optional[float],
    ) -> None:
        """Mark pending broker sync and place a broker-held stop order."""
        self._reconciler.mark_entry_pending(symbol)
        if not stop_loss or qty <= 0 or not hasattr(self._broker, "place_protective_stop"):
            return
        old_id = self._reconciler.pop_broker_stop_id(symbol)
        if old_id:
            self._broker.cancel_order_by_id(old_id)
        oid = self._broker.place_protective_stop(symbol, qty, stop_loss)
        if oid:
            self._reconciler.set_broker_stop_id(symbol, oid)

    def _on_position_opened(
        self,
        signal: TradeSignal,
        fill: Fill,
        *,
        strategy: str,
        execution_method: str,
        cycle_num: Optional[int] = None,
    ) -> None:
        """Register exits, mark broker pending, place broker-held stop."""
        now = datetime.now(timezone.utc)
        if signal.symbol not in self._pipeline.exit_manager.tracked:
            self._pipeline.exit_manager.register_from_signal(
                signal, now, fill_price=fill.price,
            )
        self._pipeline._original_sizes[signal.symbol] = signal.quantity
        tracked = self._pipeline.exit_manager.tracked.get(signal.symbol)
        stop = (tracked.stop_loss if tracked else None) or signal.stop_loss
        self._arm_broker_protection(signal.symbol, fill.quantity, stop)
        self._push_positions_from_alpaca()

    def _refresh_broker_stop(self, symbol: str) -> None:
        """Update broker stop after partial exit or stop step-up."""
        tracked = self._pipeline.exit_manager.tracked.get(symbol)
        if not tracked or not tracked.stop_loss or tracked.remaining_qty <= 0:
            return
        if not hasattr(self._broker, "replace_protective_stop"):
            return
        old_id = self._reconciler.get_broker_stop_id(symbol)
        new_id = self._broker.replace_protective_stop(
            symbol, old_id, tracked.remaining_qty, tracked.stop_loss,
        )
        if new_id:
            self._reconciler.set_broker_stop_id(symbol, new_id)

    def _clear_broker_stop(self, symbol: str) -> None:
        oid = self._reconciler.pop_broker_stop_id(symbol)
        if oid and hasattr(self._broker, "cancel_order_by_id"):
            self._broker.cancel_order_by_id(oid)

    def _record_trade_exit(
        self,
        fill: Fill,
        entry_price: float,
        reason: str,
        *,
        strategy: str = "",
        cycle_num: Optional[int] = None,
    ) -> float:
        """Single path for P&L, journal, dashboard, and broker stop cleanup."""
        fill_ts = fill.ts.isoformat() if hasattr(fill.ts, "isoformat") else str(fill.ts)
        fill_key = (
            fill.symbol,
            fill.side.value,
            round(float(fill.quantity), 4),
            round(float(fill.price), 4),
            fill_ts,
        )
        recorded_exit_fill_keys = getattr(self, "_recorded_exit_fill_keys", None)
        if recorded_exit_fill_keys is None:
            recorded_exit_fill_keys = set()
            self._recorded_exit_fill_keys = recorded_exit_fill_keys
        if fill_key in recorded_exit_fill_keys:
            logger.info(
                "Duplicate exit fill ignored %s %.0f @ %.4f",
                fill.symbol, fill.quantity, fill.price,
            )
            return 0.0
        recorded_exit_fill_keys.add(fill_key)
        if len(recorded_exit_fill_keys) > 1000:
            self._recorded_exit_fill_keys = set(list(recorded_exit_fill_keys)[-500:])

        pnl = 0.0
        if entry_price > 0:
            pnl = self._pipeline.record_realized_exit(
                fill.symbol, entry_price, fill.price, fill.quantity,
            )
        try:
            from daytrading.ml.shadow_collector import label_exit_snapshots
            label_exit_snapshots(fill.symbol, fill.price)
        except Exception:
            pass
        self._hub.on_exit_fill(fill, entry_price=entry_price)
        self._hub.add_log(
            "INFO",
            "EXIT {} {} {:.0f} @ ${:.2f}".format(
                fill.side.value.upper(), fill.symbol, fill.quantity, fill.price,
            ),
        )
        payload = {
            "symbol": fill.symbol,
            "side": fill.side.value,
            "quantity": fill.quantity,
            "entry_price": entry_price,
            "exit_price": fill.price,
            "pnl": pnl,
            "ts": fill.ts,
            "trade_type": "exit",
            "reason": reason,
        }
        if strategy:
            payload["strategy"] = strategy
        if cycle_num is not None:
            payload["market_context"] = {
                "phase": self._market_phase(),
                "cycle": cycle_num,
            }
        self._journal.record("trade_exit", payload, ts=fill.ts)

        if strategy == "breakout_scalp":
            self._breakout_scalp_active = False

        # Cancel any pending entry signals for this symbol (prevent re-entry race)
        self._exec_timer.cancel(fill.symbol)
        self._timed_signal_queue = deque(
            s for s in self._timed_signal_queue if s.symbol != fill.symbol
        )

        tracked = self._pipeline.exit_manager.tracked.get(fill.symbol)
        if tracked is None or tracked.remaining_qty <= 0:
            self._clear_broker_stop(fill.symbol)
        else:
            self._refresh_broker_stop(fill.symbol)
        return pnl

    def _seed_recent_order_ids(self) -> None:
        """Add recent closed order IDs to prevent duplicate dashboard pushes."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=20)
            orders = self._broker.client.get_orders(filter=req)
            for o in orders:
                self._last_synced_order_ids.add(str(o.id))
        except Exception:
            pass

    def _lookup_entry_price_from_orders(self, symbol: str) -> float:
        """Find entry price for a symbol from today's Alpaca buy orders."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            today_start = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED, after=today_start,
                limit=200, symbols=[symbol],
            )
            orders = self._broker.client.get_orders(filter=req)
            total_qty = 0.0
            total_cost = 0.0
            for o in orders:
                status_val = o.status.value if hasattr(o.status, 'value') else str(o.status)
                side_val = o.side.value if hasattr(o.side, 'value') else str(o.side)
                if status_val != "filled" or "buy" not in side_val:
                    continue
                qty = float(o.filled_qty or 0)
                price = float(o.filled_avg_price or 0)
                if qty > 0 and price > 0:
                    total_qty += qty
                    total_cost += qty * price
            if total_qty > 0:
                return total_cost / total_qty
        except Exception as exc:
            logger.debug("Entry price lookup for %s failed: %s", symbol, exc)
        return 0.0

    def _check_new_fills(self) -> None:
        """Check Alpaca for recently filled orders not yet in the dashboard."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            from daytrading.models import Fill, Side

            req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=10)
            orders = self._broker.client.get_orders(filter=req)

            for o in orders:
                oid = str(o.id)
                if oid in self._last_synced_order_ids:
                    continue

                status_val = o.status.value if hasattr(o.status, 'value') else str(o.status)
                if status_val != "filled":
                    self._last_synced_order_ids.add(oid)
                    continue

                side_val = o.side.value if hasattr(o.side, 'value') else str(o.side)
                qty = float(o.filled_qty or 0)
                price = float(o.filled_avg_price or 0)
                if qty <= 0 or price <= 0:
                    self._last_synced_order_ids.add(oid)
                    continue

                side = Side.BUY if "buy" in side_val else Side.SELL
                fill = Fill(
                    symbol=o.symbol, side=side, quantity=qty,
                    price=price, ts=o.filled_at or o.submitted_at,
                    commission=0.0,
                )

                if "buy" in side_val:
                    self._hub.on_fill(fill, "entry")
                    self._hub.add_log("INFO", "ENTRY {} {} {:.0f} @ ${:.2f}".format(
                        "BUY", o.symbol, qty, price))
                else:
                    entry_price = 0.0
                    tracked = self._pipeline.exit_manager.tracked
                    if o.symbol in tracked:
                        entry_price = tracked[o.symbol].entry_price
                    if entry_price == 0.0:
                        pos = self._pipeline.portfolio.positions.get(o.symbol)
                        if pos:
                            entry_price = pos.avg_price
                    if entry_price == 0.0:
                        entry_price = self._lookup_entry_price_from_orders(o.symbol)
                    self._record_trade_exit(
                        fill, entry_price, "broker_fill_sync",
                    )
                    self._push_positions_from_alpaca()

                self._last_synced_order_ids.add(oid)
        except Exception as exc:
            logger.debug("Fill sync error: %s", exc)

    def _position_sync_loop(self) -> None:
        """Background thread: push position + account + trade updates every 3 seconds."""
        tick = 0
        while not self._shutdown:
            time.sleep(3)
            if self._shutdown:
                break
            self._push_positions_from_alpaca()
            self._check_new_fills()
            tick += 1
            if tick % 5 == 0:
                try:
                    acct = self._broker.get_account()
                    self._hub.on_account_update(
                        acct["cash"], acct["equity"], acct["buying_power"],
                    )
                except Exception:
                    pass

    def _maybe_daily_session_reset(
        self,
        current_et: datetime,
        phase: str,
        *,
        force: bool = False,
    ) -> bool:
        """Run the once-per-trading-day reset at the premarket boundary."""
        if not self._is_trading_day(current_et):
            return False
        day_key = current_et.date().isoformat()
        if self._last_session_reset_day == day_key:
            return False
        premarket_start = current_et.replace(hour=4, minute=0, second=0, microsecond=0)
        reset_phase = phase in ("PRE-MARKET", "OPEN")
        if not force and (current_et < premarket_start or not reset_phase):
            return False
        if phase == "OPEN" and not force:
            logger.warning(
                "Daily session reset for %s was missed before premarket; running now",
                day_key,
            )
        self._daily_session_reset(day_key)
        return True

    def _daily_session_reset(self, day_key: Optional[str] = None) -> None:
        """Reset all session state for a new trading day.

        Called when entering PRE-MARKET phase. Clears stale bar buffers,
        prior-day stats, watchlist, and HOD pool so the scanner starts fresh.
        """
        logger.info("=== DAILY SESSION RESET — clearing stale data for new day ===")

        # Clear bar buffers (will be refilled from fresh data)
        self._bar_buffer.clear()
        self._quote_buffer.clear()
        self._tick_buffer.clear()

        # Reset watchlist to just SPY (will rebuild via HOD alerts)
        self._watchlist = list(self._watchlist_pinned)
        self._watchlist_set = set(self._watchlist)

        # Clear HOD scanner state
        if hasattr(self, '_hod_bar_scanner') and self._hod_bar_scanner:
            try:
                self._hod_bar_scanner._gapper_fired.clear()
            except Exception:
                pass
        if hasattr(self, '_hod_tick_tracker') and self._hod_tick_tracker:
            try:
                self._hod_tick_tracker._states.clear()
            except Exception:
                pass

        # Clear HOD seed retry state
        self._hod_seed_retries.clear()
        self._hod_seed_blacklist.clear()
        with self._hod_seed_lock:
            self._hod_seed_queue.clear()
            self._hod_seed_pending.clear()

        # Reset daily P&L tracking
        self._pipeline._daily_pnl = 0.0
        self._pipeline._daily_losers.clear()
        self._last_synced_order_ids.clear()
        self._recorded_exit_fill_keys.clear()

        # Reset EOD flatten flag
        self._eod_flattened = False

        # Reset network failure state
        self._network_failure_times.clear()
        self._hydrate_paused_until = 0.0

        # Reset ML monitor for new day
        try:
            from daytrading.strategy.entry_guard import get_ml_monitor
            monitor = get_ml_monitor()
            if monitor:
                monitor.reset_daily()
        except Exception:
            pass

        if self._trade_analyzer is not None:
            try:
                self._trade_analyzer.reset_blocks()
            except Exception:
                pass

        try:
            self._hub.reset_daily_overview()
            self._hub.add_log("INFO", "Daily overview reset for {}".format(
                day_key or self._now_et().date().isoformat()))
        except Exception:
            pass

        self._last_session_reset_day = day_key or self._now_et().date().isoformat()
        logger.info("Session reset complete — watchlist: %s", self._watchlist)

    def _load_history(self) -> None:
        """Fetch recent bars to seed the pipeline.

        Only keeps bars from **today's session** (premarket from 4:00 AM ET).
        This prevents yesterday's bars from inflating volume/momentum
        calculations and causing the bot to enter dead stocks.
        """
        logger.info("Loading historical bars for %d symbols...", len(self._watchlist))
        try:
            now_utc = datetime.now(timezone.utc)
            now_et = now_utc.astimezone(ET)

            if not is_us_trading_day(now_et.date()):
                logger.info("Not a US trading day — skipping historical bar fetch")
                return

            # Extended session starts 4:00 AM ET (premarket)
            session_start_et = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
            session_start_utc = session_start_et.astimezone(timezone.utc)

            if now_utc <= session_start_utc:
                logger.info(
                    "Before pre-market (04:00 ET) — skipping historical bar fetch"
                )
                return

            bars = self._hist.get_bars(
                self._watchlist,
                timeframe="1Min",
                limit=self._session_bar_limit,
                start=session_start_utc,
                end=now_utc,
            )
            today_et = now_et.date()
            for symbol, symbol_bars in bars.items():
                today_bars = []
                older_bars = []
                for b in symbol_bars:
                    if b.ts is not None:
                        try:
                            bar_et = b.ts.astimezone(ET).date()
                            if bar_et == today_et:
                                today_bars.append(b)
                            else:
                                older_bars.append(b)
                        except Exception:
                            today_bars.append(b)
                    else:
                        today_bars.append(b)
                logger.info(
                    "History %s: %d fetched → %d today, %d older (today_et=%s, last_bar_ts=%s)",
                    symbol, len(symbol_bars), len(today_bars), len(older_bars),
                    today_et, symbol_bars[-1].ts if symbol_bars else "none",
                )
                if len(today_bars) >= 5:
                    self._bar_buffer[symbol] = deque(today_bars[-self._max_bars_per_symbol:], maxlen=self._max_bars_per_symbol)
                else:
                    seed = older_bars[-20:] + today_bars
                    self._bar_buffer[symbol] = deque(seed[-self._max_bars_per_symbol:], maxlen=self._max_bars_per_symbol)
            loaded = {s: len(b) for s, b in self._bar_buffer.items() if b}
            logger.info("Loaded history: %s", loaded)
            self._load_prior_day_stats(list(self._watchlist))
            self._seed_all_hod_sessions()
        except Exception as exc:
            logger.error("Failed to load history: %s", exc)

    def _on_bar(self, bar: Bar) -> None:
        """Callback from stream — push bar event to queue."""
        if bar.symbol == "SPY":
            self._pipeline.trade_guard.market_panic.update_spy_bar(bar)
            return
        if bar.ts is not None:
            try:
                age = (datetime.now(timezone.utc) - bar.ts).total_seconds()
                if age > 14400:
                    return
            except Exception:
                pass
        try:
            self._event_queue.put_nowait(BarEvent(bar.symbol, bar))
            self._new_data.set()
        except queue.Full:
            pass

    def _on_quote(self, quote: Quote) -> None:
        """Callback from stream — push quote event to queue."""
        try:
            self._event_queue.put_nowait(QuoteEvent(quote.symbol, quote))
        except queue.Full:
            pass

    def _on_stream_trade(self, tick: Tick) -> None:
        """Callback from stream — push trade event to queue."""
        try:
            self._event_queue.put_nowait(TradeEvent(tick.symbol, tick))
        except queue.Full:
            pass

    def _drain_events(self) -> bool:
        """Drain all pending events from the queue into local buffers.

        Returns True if any bar events were processed (signals new data).
        Only the main thread calls this — all buffer/tick-tracker writes
        happen here, eliminating cross-thread contention.
        """
        got_bars = False
        drained = 0
        while True:
            try:
                evt = self._event_queue.get_nowait()
            except queue.Empty:
                break
            drained += 1

            if isinstance(evt, BarEvent):
                buf = self._bar_buffer.get(evt.symbol)
                if buf is None:
                    buf = deque(maxlen=self._max_bars_per_symbol)
                    self._bar_buffer[evt.symbol] = buf
                buf.append(evt.bar)
                self._seed_hod_session(evt.symbol)
                got_bars = True

            elif isinstance(evt, QuoteEvent):
                buf = self._quote_buffer.get(evt.symbol)
                if buf is None:
                    buf = deque(maxlen=200)
                    self._quote_buffer[evt.symbol] = buf
                buf.append(evt.quote)
                self._pipeline.trade_guard.slippage.update_quote(evt.quote)

            elif isinstance(evt, TradeEvent):
                tick = evt.tick
                if self._hod_tick_tracker is not None:
                    self._hod_tick_tracker.on_trade(tick)
                if tick.symbol in self._watchlist_set:
                    self._pipeline.trade_guard.halt_tracker.update_price(
                        tick.symbol, tick.price, tick.ts,
                    )
                    buf = self._tick_buffer.get(tick.symbol)
                    if buf is None:
                        buf = deque(maxlen=self._max_ticks_per_symbol)
                        self._tick_buffer[tick.symbol] = buf
                    buf.append(tick)

                    # Tick-level trailing stop for open positions (post-half only)
                    tracked_pos = self._pipeline.exit_manager._positions.get(tick.symbol)
                    if tracked_pos and tracked_pos.remaining_qty > 0 and tracked_pos.breakeven_locked:
                        if tick.price > tracked_pos.highest_price:
                            tracked_pos.highest_price = tick.price
                        # Only trail on ticks AFTER half-sell (let pre-half winners run to target)
                        if tracked_pos.sold_half:
                            trail_stop = round(tracked_pos.highest_price * 0.99, 4)
                            if trail_stop > tracked_pos.stop_loss:
                                tracked_pos.stop_loss = trail_stop
                            if tick.price <= tracked_pos.stop_loss:
                                self._instant_trail_exit(tick.symbol, tick.price)

                    # Tape pressure profit-protection exit (pre- and post-half)
                    if tracked_pos and tracked_pos.remaining_qty > 0:
                        if tick.price > tracked_pos.entry_price:
                            hold_secs = (datetime.now(timezone.utc) - tracked_pos.entry_ts).total_seconds()
                            tick_list = list(self._tick_buffer.get(tick.symbol, []))
                            quote_list = list(self._quote_buffer.get(tick.symbol, []))
                            if self._tape_pressure.check(
                                tick_list, quote_list,
                                tracked_pos.entry_price, tick.price, hold_secs,
                            ):
                                self._tape_pressure_exit(tick.symbol, tick.price)

                    completed_10s = self._bar_aggregator.on_tick(tick)
                    if completed_10s is not None:
                        if tick.symbol in self._pipeline.exit_manager._positions:
                            self._check_10s_candle_exit(tick.symbol, completed_10s)
                        ready_sig = self._exec_timer.on_10s_bar(completed_10s)
                        if ready_sig is not None:
                            self._timed_signal_queue.append(ready_sig)

            elif isinstance(evt, BarsLoadedEvent):
                today_et = self._now_et().date()
                for symbol, symbol_bars in evt.bars_by_symbol.items():
                    today_bars = [
                        b for b in symbol_bars
                        if b.ts is None or self._bar_is_today(b, today_et)
                    ]
                    if today_bars:
                        self._bar_buffer[symbol] = deque(
                            today_bars[-self._max_bars_per_symbol:],
                            maxlen=self._max_bars_per_symbol,
                        )
                if evt.prior_day_stats:
                    self._prior_day_stats.update(evt.prior_day_stats)
                for sym in evt.bars_by_symbol:
                    self._seed_hod_session(sym)

            elif isinstance(evt, PoolRefreshEvent):
                self._hod_bar_pool = list(evt.new_pool)
                if evt.bars:
                    today_et = self._now_et().date()
                    for symbol, symbol_bars in evt.bars.items():
                        today_bars = [
                            b for b in symbol_bars
                            if b.ts is None or self._bar_is_today(b, today_et)
                        ]
                        if today_bars:
                            self._bar_buffer[symbol] = deque(
                                today_bars[-self._max_bars_per_symbol:],
                                maxlen=self._max_bars_per_symbol,
                            )
                if evt.prior_day_stats:
                    self._prior_day_stats.update(evt.prior_day_stats)
                self._fast_scan_known = set(self._hod_bar_pool)
                self._sync_tick_tracker_pool()
                logger.info(
                    "HOD pool refresh [%s]: %d symbols",
                    self._market_phase(), len(self._hod_bar_pool),
                )

            elif isinstance(evt, FastScanEvent):
                self._handle_fast_scan_movers(evt.new_movers)
                got_bars = True

        if drained > 5000:
            logger.info("Drained %d events from queue", drained)
        return got_bars

    @staticmethod
    def _bar_is_today(bar: Bar, today_et) -> bool:
        try:
            return bar.ts.astimezone(ET).date() == today_et
        except Exception:
            return True

    def _handle_fast_scan_movers(self, new_movers: List[Dict]) -> None:
        """Process new movers from fast scan: float check → add to pool → hydrate."""
        if not self._float_checker or not new_movers:
            return

        hydrate_symbols = []
        pool_set = set(self._hod_bar_pool)
        now_ts = datetime.now(timezone.utc)
        for mover in new_movers:
            sym = mover["symbol"]
            is_strong = mover.get("abs_change_pct", 0) >= 10.0 and mover.get("volume", 0) >= 200_000
            in_pool = sym in pool_set
            bars = self._bar_buffer.get(sym)
            bars_stale = True
            if bars:
                last_bar = bars[-1]
                if last_bar.ts is not None:
                    try:
                        bars_stale = (now_ts - last_bar.ts).total_seconds() > 120
                    except Exception:
                        bars_stale = False
                else:
                    bars_stale = False
            if in_pool:
                if is_strong and (not bars or bars_stale):
                    hydrate_symbols.append(sym)
                continue
            if len(self._hod_bar_pool) >= self._hod_pool_max:
                break
            flt = self._float_checker.get_float_cached(sym)
            if flt is None and is_strong:
                flt = self._float_checker.get_float(sym)
            elif flt is not None and flt > self._hod_max_float and is_strong:
                flt = self._float_checker.get_float(sym)
            if flt is not None and flt > self._hod_max_float:
                continue
            if flt is None and not is_strong:
                continue
            self._hod_bar_pool.append(sym)
            pool_set.add(sym)
            hydrate_symbols.append(sym)

        if not hydrate_symbols:
            return

        logger.info(
            "Fast scan → hydrating %d HOD movers: %s",
            len(hydrate_symbols),
            ", ".join(hydrate_symbols[:10]),
        )
        self._hub.add_log(
            "INFO",
            "Fast scan hydrating {} HOD movers — {}".format(
                len(hydrate_symbols), ", ".join(hydrate_symbols[:10]),
            ),
        )

        batch = hydrate_symbols[:self._hod_hydrate_batch_max]
        try:
            bars_by_symbol = self._fetch_session_bars(batch)
            prior_stats = self._fetch_prior_day_stats(batch)
            today_et = self._now_et().date()
            for symbol, symbol_bars in bars_by_symbol.items():
                today_bars = [
                    b for b in symbol_bars
                    if b.ts is None or self._bar_is_today(b, today_et)
                ]
                if today_bars:
                    self._bar_buffer[symbol] = deque(
                        today_bars[-self._max_bars_per_symbol:],
                        maxlen=self._max_bars_per_symbol,
                    )
            if prior_stats:
                self._prior_day_stats.update(prior_stats)
            for sym in bars_by_symbol:
                self._seed_hod_session(sym)
        except Exception as exc:
            logger.warning("Fast scan hydrate failed: %s", exc)

        self._sync_tick_tracker_pool()

    def _check_exits_only(self) -> None:
        """Fast exit check every second using live Alpaca prices.

        This prevents stop-loss slippage that occurs when the full
        30-second scan cycle hasn't run yet but the price has moved.
        """
        try:
            tracked = self._pipeline.exit_manager.tracked
            if not tracked:
                return

            prices = self._live_prices(list(tracked.keys()))
            if not prices:
                return

            now = datetime.now(timezone.utc)

            # Snapshot entry prices from the INTERNAL positions dict
            # (tracked returns a copy, but we need prices before check_exits
            # untracks positions during step-up → stop-hit sequences)
            entry_prices = {
                sym: pos.entry_price
                for sym, pos in self._pipeline.exit_manager._positions.items()
            }

            exit_signals = self._pipeline.exit_manager.check_exits(prices, now)

            for sig in exit_signals:
                order = Order(
                    symbol=sig.symbol,
                    side=Side.SELL if sig.action is SignalAction.EXIT_LONG else Side.BUY,
                    quantity=sig.quantity,
                    limit_price=prices.get(sig.symbol, sig.entry_price),
                )
                bar = Bar(
                    symbol=sig.symbol,
                    open=prices.get(sig.symbol, 0),
                    high=prices.get(sig.symbol, 0),
                    low=prices.get(sig.symbol, 0),
                    close=prices.get(sig.symbol, 0),
                    volume=0, ts=now,
                )
                fill, status = self._broker.submit(order, bar, self._pipeline.portfolio)
                try:
                    from daytrading.ml.shadow_collector import log_execution_quality
                    log_execution_quality(
                        order=order, bar=bar, status=status, fill=fill,
                        source="fast_exit_limit",
                    )
                except Exception:
                    pass
                if fill:
                    from daytrading.execution.broker import apply_fill as _apply_fill
                    _apply_fill(self._pipeline.portfolio, fill)
                    ep = entry_prices.get(sig.symbol, 0.0)
                    if ep == 0.0:
                        pos = self._pipeline.portfolio.positions.get(sig.symbol)
                        if pos:
                            ep = pos.avg_price
                    pnl = self._record_trade_exit(
                        fill, ep, sig.reason or "fast_exit",
                    )
                    logger.info(
                        "FAST EXIT %s %s %.0f @ %.4f (entry=%.4f, P&L=$%.2f, day P&L=$%.2f)",
                        fill.side.value, fill.symbol, fill.quantity, fill.price,
                        ep, pnl, self._pipeline._daily_pnl,
                    )
                    # Seed recent order IDs so _check_new_fills won't duplicate
                    self._seed_recent_order_ids()
                    self._push_positions_from_alpaca()
                    self._pipeline.set_cooldown(fill.symbol)
                else:
                    # STOP LOSS MUST EXECUTE — retry with market order
                    logger.warning(
                        "FAST EXIT limit not filled for %s — retrying with guarded marketable limit",
                        sig.symbol,
                    )
                    market_order = Order(
                        symbol=sig.symbol,
                        side=Side.SELL if sig.action is SignalAction.EXIT_LONG else Side.BUY,
                        quantity=sig.quantity,
                        limit_price=None,  # broker converts to guarded marketable limit
                    )
                    fill2, status2 = self._broker.submit(market_order, bar, self._pipeline.portfolio)
                    try:
                        from daytrading.ml.shadow_collector import log_execution_quality
                        log_execution_quality(
                            order=market_order, bar=bar, status=status2, fill=fill2,
                            source="fast_exit_guarded_marketable",
                        )
                    except Exception:
                        pass
                    if fill2:
                        from daytrading.execution.broker import apply_fill as _apply_fill2
                        _apply_fill2(self._pipeline.portfolio, fill2)
                        ep = entry_prices.get(sig.symbol, 0.0)
                        if ep == 0.0:
                            pos = self._pipeline.portfolio.positions.get(sig.symbol)
                            if pos:
                                ep = pos.avg_price
                        pnl = self._record_trade_exit(
                            fill2, ep, sig.reason or "market_stop",
                        )
                        logger.info(
                            "MARKET EXIT %s %s %.0f @ %.4f (entry=%.4f, P&L=$%.2f)",
                            fill2.side.value, fill2.symbol, fill2.quantity, fill2.price, ep, pnl,
                        )
                        self._seed_recent_order_ids()
                        self._push_positions_from_alpaca()
                        self._pipeline.set_cooldown(fill2.symbol)
                    else:
                        logger.error(
                            "CRITICAL: MARKET EXIT also failed for %s — position unprotected!",
                            sig.symbol,
                        )
                    # Rollback half-sell state if both orders failed
                    if not fill2:
                        tracked_pos = self._pipeline.exit_manager._positions.get(sig.symbol)
                        if tracked_pos and tracked_pos.sold_half and tracked_pos.remaining_qty > 0:
                            tracked_pos.sold_half = False
                            tracked_pos.remaining_qty += sig.quantity
                            tracked_pos.stop_loss = tracked_pos.entry_price - tracked_pos.risk_per_share
                            tracked_pos.breakeven_locked = False
        except Exception as exc:
            logger.warning("Fast exit check error: %s", exc)

    def _instant_trail_exit(self, symbol: str, price: float) -> None:
        """Immediately exit a position when the tick-level trailing stop is hit."""
        try:
            from daytrading.execution.broker import apply_fill

            pos = self._pipeline.exit_manager._positions.get(symbol)
            if pos is None or pos.remaining_qty <= 0:
                return

            qty = pos.remaining_qty
            entry_price = pos.entry_price

            order = Order(
                symbol=symbol,
                side=Side.SELL,
                quantity=qty,
                limit_price=price,
            )
            now = datetime.now(timezone.utc)
            bar = Bar(
                symbol=symbol, open=price, high=price,
                low=price, close=price, volume=0, ts=now,
            )
            fill, status = self._broker.submit(order, bar, self._pipeline.portfolio)
            if fill:
                apply_fill(self._pipeline.portfolio, fill)
                pos.remaining_qty = 0
                strategy = pos.reason if hasattr(pos, "reason") else ""
                pnl = self._record_trade_exit(
                    fill, entry_price, "tick_trailing_stop",
                    strategy=strategy,
                )
                self._pipeline.exit_manager.untrack(symbol)
                logger.info(
                    "TICK TRAIL EXIT %s %.0f @ $%.4f (entry=$%.4f, high=$%.4f, P&L=$%.2f)",
                    symbol, fill.quantity, fill.price, entry_price,
                    pos.highest_price, pnl,
                )
                self._hub.add_log(
                    "INFO",
                    "TICK TRAIL EXIT {} {:.0f} @ ${:.2f} (P&L ${:.2f})".format(
                        symbol, fill.quantity, fill.price, pnl),
                )
                self._seed_recent_order_ids()
                self._push_positions_from_alpaca()
            else:
                logger.warning("TICK TRAIL EXIT order not filled %s (status=%s)", symbol, status)
        except Exception as exc:
            logger.error("Tick trail exit error %s: %s", symbol, exc)

    def _tape_pressure_exit(self, symbol: str, price: float) -> None:
        """Exit a position when tape pressure detects selling — protect profit."""
        try:
            from daytrading.execution.broker import apply_fill

            pos = self._pipeline.exit_manager._positions.get(symbol)
            if pos is None or pos.remaining_qty <= 0:
                return

            qty = pos.remaining_qty
            entry_price = pos.entry_price

            order = Order(
                symbol=symbol,
                side=Side.SELL,
                quantity=qty,
                limit_price=price,
            )
            now = datetime.now(timezone.utc)
            bar = Bar(
                symbol=symbol, open=price, high=price,
                low=price, close=price, volume=0, ts=now,
            )
            fill, status = self._broker.submit(order, bar, self._pipeline.portfolio)
            if fill:
                apply_fill(self._pipeline.portfolio, fill)
                pos.remaining_qty = 0
                strategy = pos.reason if hasattr(pos, "reason") else ""
                pnl = self._record_trade_exit(
                    fill, entry_price, "tape_pressure_sell",
                    strategy=strategy,
                )
                self._pipeline.exit_manager.untrack(symbol)
                logger.info(
                    "TAPE PRESSURE EXIT %s %.0f @ $%.4f (entry=$%.4f, high=$%.4f, P&L=$%.2f)",
                    symbol, fill.quantity, fill.price, entry_price,
                    pos.highest_price, pnl,
                )
                self._hub.add_log(
                    "INFO",
                    "TAPE PRESSURE EXIT {} {:.0f} @ ${:.2f} (P&L ${:.2f})".format(
                        symbol, fill.quantity, fill.price, pnl),
                )
                self._seed_recent_order_ids()
                self._push_positions_from_alpaca()
            else:
                logger.warning("TAPE PRESSURE EXIT order not filled %s (status=%s)", symbol, status)
        except Exception as exc:
            logger.error("Tape pressure exit error %s: %s", symbol, exc)

    def _check_10s_candle_exit(self, symbol: str, bar_10s: Bar) -> None:
        """Exit if a 10s candle closes red and below previous 10s bar's low.

        Only activates after 60s hold AND price has moved +2% from entry,
        to avoid killing winners on normal micro-pullbacks.
        """
        try:
            from daytrading.execution.broker import apply_fill

            pos = self._pipeline.exit_manager._positions.get(symbol)
            if pos is None or pos.remaining_qty <= 0:
                return

            if pos.entry_ts is None:
                return
            hold_secs = (datetime.now(timezone.utc) - pos.entry_ts).total_seconds()
            if hold_secs < 60:
                return

            if not pos.breakeven_locked:
                return

            # Only activate after price has run at least +2% from entry
            if pos.entry_price > 0:
                max_run_pct = (pos.highest_price - pos.entry_price) / pos.entry_price
                if max_run_pct < 0.02:
                    return

            is_red = bar_10s.close < bar_10s.open
            if not is_red:
                return

            prev_bars = self._bar_aggregator.get_latest_10s(symbol, count=2)
            if len(prev_bars) < 2:
                return
            prev_bar = prev_bars[-2]

            if bar_10s.close >= prev_bar.low:
                return

            price = bar_10s.close
            qty = pos.remaining_qty
            entry_price = pos.entry_price

            order = Order(
                symbol=symbol,
                side=Side.SELL,
                quantity=qty,
                limit_price=price,
            )
            now = datetime.now(timezone.utc)
            bar = Bar(
                symbol=symbol, open=price, high=price,
                low=price, close=price, volume=0, ts=now,
            )
            fill, status = self._broker.submit(order, bar, self._pipeline.portfolio)
            if fill:
                apply_fill(self._pipeline.portfolio, fill)
                pos.remaining_qty = 0
                strategy = pos.reason if hasattr(pos, "reason") else ""
                pnl = self._record_trade_exit(
                    fill, entry_price, "red_10s_candle",
                    strategy=strategy,
                )
                self._pipeline.exit_manager.untrack(symbol)
                logger.info(
                    "RED 10s EXIT %s %.0f @ $%.4f (entry=$%.4f, 10s close=$%.4f < prev low=$%.4f, P&L=$%.2f)",
                    symbol, fill.quantity, fill.price, entry_price,
                    bar_10s.close, prev_bar.low, pnl,
                )
                self._hub.add_log(
                    "INFO",
                    "RED 10s EXIT {} {:.0f} @ ${:.2f} (P&L ${:.2f})".format(
                        symbol, fill.quantity, fill.price, pnl),
                )
                self._seed_recent_order_ids()
                self._push_positions_from_alpaca()
            else:
                logger.warning("RED 10s EXIT order not filled %s (status=%s)", symbol, status)
        except Exception as exc:
            logger.error("10s candle exit error %s: %s", symbol, exc)

    def _execute_timed_signal(self, signal: TradeSignal) -> None:
        """Execute a deferred signal that the execution timer released."""
        try:
            from daytrading.execution.broker import apply_fill
            from daytrading.models import Side, SignalAction

            sym = signal.symbol
            if self._new_entries_blocked(sym, "TIMED ENTRY"):
                return

            # Re-check cooldown (may have been set after signal was queued)
            last_exit_ts = self._pipeline._exit_cooldowns.get(sym)
            if last_exit_ts is not None:
                elapsed = (datetime.now(timezone.utc) - last_exit_ts).total_seconds()
                if elapsed < self._pipeline._cooldown_seconds:
                    logger.info(
                        "TIMED ENTRY skip %s — on cooldown (%.0fs ago)", sym, elapsed,
                    )
                    return

            # Don't enter if already holding
            pos = self._pipeline.portfolio.positions.get(sym)
            if pos and not pos.is_flat:
                logger.info("TIMED ENTRY skip %s — already in position", sym)
                return

            # Daily loser blacklist
            if sym in self._pipeline._daily_losers:
                logger.info("TIMED ENTRY skip %s — daily loser blacklist", sym)
                return

            # Per-symbol max entries
            if self._pipeline._symbol_entry_counts.get(sym, 0) >= self._pipeline._max_entries_per_symbol:
                logger.info(
                    "TIMED ENTRY skip %s — max entries reached (%d)",
                    sym, self._pipeline._max_entries_per_symbol,
                )
                return

            order = Order(
                symbol=sym,
                side=Side.BUY if signal.action is SignalAction.ENTER_LONG else Side.SELL,
                quantity=signal.quantity,
                limit_price=signal.entry_price,
            )
            # Use latest known price for the bar
            bars = self._bar_buffer.get(signal.symbol, deque())
            if bars:
                bar = bars[-1]
            else:
                bar = Bar(
                    symbol=signal.symbol,
                    open=signal.entry_price, high=signal.entry_price,
                    low=signal.entry_price, close=signal.entry_price,
                    volume=0, ts=datetime.now(timezone.utc),
                )

            chase_reject = self._timed_entry_chase_reject(signal, bar)
            if chase_reject:
                logger.info("TIMED ENTRY skip %s — %s", sym, chase_reject)
                self._hub.add_log("WARNING", "ENTRY SKIP {}: {}".format(sym, chase_reject))
                return

            fill, status = self._broker.submit(order, bar, self._pipeline.portfolio)
            try:
                from daytrading.ml.shadow_collector import log_execution_quality
                log_execution_quality(
                    order=order, bar=bar, status=status, fill=fill, source="timed_entry",
                )
            except Exception:
                pass
            if (
                fill is None
                and self._is_hot_hod_timed_signal(signal)
                and signal.action is SignalAction.ENTER_LONG
            ):
                fill, status = self._retry_hot_hod_timed_entry(signal, status, bar)
            if fill:
                apply_fill(self._pipeline.portfolio, fill)
                self._pipeline._symbol_entry_counts[sym] = self._pipeline._symbol_entry_counts.get(sym, 0) + 1
                strategy = (
                    signal.scan_result.scanner_name if signal.scan_result
                    else signal.reason or "unknown"
                )
                self._on_position_opened(
                    signal, fill, strategy=strategy, execution_method="10s_timed",
                )
                logger.info(
                    "TIMED ENTRY %s %s %.0f @ $%.4f (strategy=%s)",
                    fill.side.value, fill.symbol, fill.quantity, fill.price, strategy,
                )
                self._hub.on_fill(fill, "entry")
                self._hub.add_log("INFO", "ENTRY {} {} {:.0f} @ ${:.2f} (10s timed)".format(
                    fill.side.value.upper(), fill.symbol, fill.quantity, fill.price))
                self._journal.record("trade_fill", {
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "ts": fill.ts,
                    "trade_type": "entry",
                    "strategy": strategy,
                    "execution_method": "10s_timed",
                    "market_context": {
                        "phase": self._market_phase(),
                    },
                }, ts=fill.ts)
                self._seed_recent_order_ids()
            else:
                logger.warning(
                    "TIMED ENTRY order not filled for %s (status=%s)",
                    signal.symbol, status,
                )
        except Exception as exc:
            logger.error("Timed signal execution error for %s: %s", signal.symbol, exc)

    def _timed_entry_chase_reject(
        self,
        signal: TradeSignal,
        fallback_bar: Bar,
    ) -> Optional[str]:
        """Cancel delayed timed entries that have become chase entries."""
        if signal.action is not SignalAction.ENTER_LONG:
            return None
        original = float(signal.entry_price or 0.0)
        if original <= 0:
            return None

        live = 0.0
        try:
            live = float(self._live_prices([signal.symbol]).get(signal.symbol) or 0.0)
        except Exception:
            live = 0.0
        if live <= 0:
            try:
                live = float(self._latest_price(signal.symbol) or 0.0)
            except Exception:
                live = 0.0
        if live <= 0:
            live = float(fallback_bar.close or original)

        hit = signal.scan_result
        pattern = ""
        scanner = ""
        if hit is not None:
            pattern = str(hit.criteria.get("pattern", ""))
            scanner = str(hit.scanner_name or "")

        hot_patterns = {
            "momentum_burst",
            "abc_continuation",
            "vwap_pullback",
            "hod_reclaim",
            "pullback_base",
            "breakout_scalp",
        }
        is_hot = pattern in hot_patterns or scanner in hot_patterns
        max_chase_pct = 0.025 if original >= 5.0 else 0.035
        if pattern in ("abc_continuation", "pullback_base", "vwap_pullback"):
            max_chase_pct = min(max_chase_pct, 0.025)
        if is_hot and live > original * (1.0 + max_chase_pct):
            return (
                "live price {:.4f} ran {:.1f}% above signal {:.4f} "
                "(max {:.1f}%)"
            ).format(live, (live - original) / original * 100.0, original, max_chase_pct * 100.0)

        quotes = list(self._quote_buffer.get(signal.symbol, []))
        recent_quotes = [q for q in quotes[-3:] if q.ask > q.bid > 0]
        if recent_quotes:
            avg_spread_pct = sum(
                (q.ask - q.bid) / ((q.ask + q.bid) / 2.0) * 100.0
                for q in recent_quotes
            ) / len(recent_quotes)
            max_spread = 0.9 if live < 5.0 else 0.6
            if avg_spread_pct > max_spread:
                return "spread widened to {:.2f}% (max {:.1f}%)".format(
                    avg_spread_pct, max_spread)

        if is_hot and self._bar_aggregator is not None:
            latest_10s = self._bar_aggregator.get_latest_10s(signal.symbol, count=1)
            if latest_10s:
                b = latest_10s[-1]
                if b.close < b.open and live >= original:
                    return "latest 10s candle turned red during entry wait"

        return None

    @staticmethod
    def _is_hot_hod_timed_signal(signal: TradeSignal) -> bool:
        hit = signal.scan_result
        if hit is None:
            return False
        pattern = str(hit.criteria.get("pattern", ""))
        hot_patterns = {
            "hod_reclaim",
            "vwap_pullback",
            "breakout_scalp",
            "momentum_burst",
            "abc_continuation",
        }
        if pattern not in hot_patterns:
            if hit.scanner_name not in hot_patterns:
                return False
        price = float(hit.criteria.get("close") or signal.entry_price or 0.0)
        if not (2.0 <= price <= 20.0):
            return False
        if pattern == "breakout_scalp" or hit.scanner_name == "breakout_scalp":
            return True
        rally_pct = float(
            hit.criteria.get("rally_pct")
            or hit.criteria.get("change_session_pct")
            or 0.0
        )
        latest_volume = float(hit.criteria.get("volume") or 0.0)
        return rally_pct >= 25.0 and latest_volume >= 100_000

    def _retry_hot_hod_timed_entry(
        self,
        signal: TradeSignal,
        first_status: OrderStatus,
        fallback_bar: Bar,
    ) -> tuple[Optional[Fill], OrderStatus]:
        """One controlled retry for fast HOD squeezes that outrun the first limit."""
        try:
            live = self._live_prices([signal.symbol]).get(signal.symbol)
            if live is None or live <= 0:
                live = self._latest_price(signal.symbol)
            if live is None or live <= 0:
                return None, first_status

            original = signal.entry_price
            max_chase_pct = 0.035 if original < 5.0 else 0.025
            max_price = original * (1.0 + max_chase_pct)
            if live > max_price:
                logger.warning(
                    "TIMED ENTRY hot retry skipped %s — live %.4f too far from signal %.4f (max %.1f%%)",
                    signal.symbol, live, original, max_chase_pct * 100,
                )
                return None, first_status

            retry_bar = Bar(
                symbol=signal.symbol,
                open=fallback_bar.open or live,
                high=max(fallback_bar.high, live),
                low=min(fallback_bar.low, live),
                close=live,
                volume=fallback_bar.volume,
                ts=datetime.now(timezone.utc),
            )
            retry_order = Order(
                symbol=signal.symbol,
                side=Side.BUY,
                quantity=signal.quantity,
                limit_price=live,
            )
            logger.info(
                "TIMED ENTRY hot retry %s %.0f @ %.4f (signal %.4f, cap %.1f%%)",
                signal.symbol, signal.quantity, live, original, max_chase_pct * 100,
            )
            fill, status = self._broker.submit(retry_order, retry_bar, self._pipeline.portfolio)
            try:
                from daytrading.ml.shadow_collector import log_execution_quality
                log_execution_quality(
                    order=retry_order, bar=retry_bar, status=status, fill=fill,
                    source="timed_entry_hot_retry",
                )
            except Exception:
                pass
            return fill, status
        except Exception as exc:
            logger.warning("TIMED ENTRY hot retry failed %s: %s", signal.symbol, exc)
            return None, first_status

    def _process_breakout_scalps(self) -> None:
        """Process pending breakout scalps queued by HOD tick alerts.

        Instant entry for fast HOD/tape movers. This deliberately uses a
        separate quick-scalp guard instead of the normal ML/pullback guard.
        The goal is quick profit on explosive tape, with smaller size and a
        short hold window.
        Max 1 breakout scalp open, 5-min cooldown per symbol.
        """
        if not self._pending_breakout_scalps:
            return

        if self._new_entries_blocked(None, "BREAKOUT SCALP"):
            self._pending_breakout_scalps.clear()
            return

        from daytrading.execution.broker import apply_fill

        now_mono = time.time()

        while self._pending_breakout_scalps:
            try:
                sym, alert_price, alert_ts = self._pending_breakout_scalps.popleft()
            except IndexError:
                break

            if now_mono - alert_ts > 10.0:
                continue

            if self._breakout_scalp_active:
                logger.debug("BREAKOUT SCALP skip %s — already have one open", sym)
                continue

            cooldown_until = self._breakout_scalp_cooldown.get(sym, 0)
            if now_mono < cooldown_until:
                logger.debug("BREAKOUT SCALP skip %s — cooldown", sym)
                continue

            # Also check pipeline exit cooldowns (unified)
            last_exit_ts = self._pipeline._exit_cooldowns.get(sym)
            if last_exit_ts is not None:
                elapsed = (datetime.now(timezone.utc) - last_exit_ts).total_seconds()
                if elapsed < self._pipeline._cooldown_seconds:
                    logger.debug("BREAKOUT SCALP skip %s — pipeline cooldown (%.0fs)", sym, elapsed)
                    continue

            # Per-symbol max entries check
            if self._pipeline._symbol_entry_counts.get(sym, 0) >= self._pipeline._max_entries_per_symbol:
                logger.info("BREAKOUT SCALP skip %s — max entries (%d)", sym, self._pipeline._max_entries_per_symbol)
                continue

            pos = self._pipeline.portfolio.positions.get(sym)
            if pos and not pos.is_flat:
                continue

            bars = list(self._bar_buffer.get(sym, deque()))
            if len(bars) < 3:
                logger.debug("BREAKOUT SCALP skip %s — only %d bars", sym, len(bars))
                continue

            reject = self._check_quick_scalp_entry(sym, bars)
            if reject is not None:
                logger.info("BREAKOUT SCALP reject %s: %s", sym, reject)
                continue

            rr = self._quick_scalp_tick_rr(sym, bars, alert_price)
            if rr is None:
                logger.info("BREAKOUT SCALP reject %s: no usable tick R:R", sym)
                continue
            price, stop_price, target_price, rr_note = rr
            risk_per_share = price - stop_price

            max_dollar_risk = 50.0
            quantity = int(max_dollar_risk / risk_per_share) if risk_per_share > 0 else 0
            quantity = max(1, min(quantity, 750))

            signal = TradeSignal(
                symbol=sym,
                action=SignalAction.ENTER_LONG,
                quantity=quantity,
                entry_price=price,
                stop_loss=stop_price,
                take_profit=target_price,
                max_hold_seconds=90,
                reason="Quick Momentum Scalp {} ${:.2f}, stop=${:.2f}, target=${:.2f} ({})".format(
                    sym, price, stop_price, target_price, rr_note),
                scan_result=ScanResult(
                    symbol=sym, scanner_name="breakout_scalp",
                    ts=datetime.now(timezone.utc), score=0.0,
                    criteria={"pattern": "breakout_scalp", "direction": "up"},
                ),
                trend_strength=0.8,
            )

            order = Order(
                symbol=sym,
                side=Side.BUY,
                quantity=quantity,
                limit_price=price,
            )
            bar = bars[-1]
            fill, status = self._broker.submit(order, bar, self._pipeline.portfolio)
            try:
                from daytrading.ml.shadow_collector import log_execution_quality
                log_execution_quality(
                    order=order, bar=bar, status=status, fill=fill,
                    source="breakout_scalp",
                )
            except Exception:
                pass
            if fill:
                apply_fill(self._pipeline.portfolio, fill)
                self._on_position_opened(
                    signal, fill, strategy="breakout_scalp",
                    execution_method="instant_breakout",
                )
                self._breakout_scalp_active = True
                self._breakout_scalp_cooldown[sym] = now_mono + 300.0
                self._pipeline._symbol_entry_counts[sym] = self._pipeline._symbol_entry_counts.get(sym, 0) + 1
                logger.info(
                    "QUICK SCALP ENTRY %s %.0f @ $%.4f stop=$%.2f target=$%.2f %s",
                    sym, fill.quantity, fill.price, stop_price, target_price, rr_note,
                )
                self._hub.on_fill(fill, "entry")
                self._hub.add_log(
                    "INFO",
                    "QUICK SCALP {} {:.0f} @ ${:.2f}".format(sym, fill.quantity, fill.price),
                )
                self._journal.record("trade_fill", {
                    "symbol": sym,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "ts": fill.ts,
                    "trade_type": "entry",
                    "strategy": "breakout_scalp",
                    "execution_method": "instant_breakout",
                    "market_context": {"phase": self._market_phase()},
                }, ts=fill.ts)
                self._seed_recent_order_ids()
                break
            else:
                logger.warning("BREAKOUT SCALP order not filled %s (status=%s)", sym, status)

    def _new_entries_blocked(self, symbol: Optional[str], source: str) -> bool:
        """Return True when global controls should block any new entry path."""
        if getattr(self._hub, "trading_paused", False):
            if symbol:
                logger.info("%s skip %s — trading paused", source, symbol)
            else:
                logger.info("%s skip — trading paused", source)
            return True
        if getattr(self._pipeline, "_circuit_breaker_tripped", False):
            pnl = getattr(self._pipeline, "_daily_pnl", 0.0)
            if symbol:
                logger.warning("%s block %s — circuit breaker daily P&L $%.2f", source, symbol, pnl)
            else:
                logger.warning("%s block — circuit breaker daily P&L $%.2f", source, pnl)
            return True
        return False

    def _quick_scalp_tick_rr(
        self,
        symbol: str,
        bars: Sequence[Bar],
        alert_price: float,
    ) -> Optional[tuple[float, float, float, str]]:
        """Calculate quick-scalp entry/stop/target from the recent tick tape."""
        fallback_price = bars[-1].close if bars and bars[-1].close > 0 else alert_price
        if fallback_price <= 0:
            return None

        now = datetime.now(timezone.utc)
        ticks = [
            t for t in self._tick_buffer.get(symbol, [])
            if t.price > 0 and t.ts is not None
        ]
        recent_ticks = []
        for tick in ticks:
            try:
                age = (now - tick.ts).total_seconds()
            except Exception:
                continue
            if 0 <= age <= 20:
                recent_ticks.append(tick)

        if len(recent_ticks) < 5:
            recent_ticks = ticks[-10:]

        prices = [t.price for t in recent_ticks if t.price > 0]
        if not prices:
            price = fallback_price
            tick_low = min(b.low for b in bars[-3:] if b.low > 0)
            source = "bar-risk"
        else:
            price = prices[-1]
            tick_low = min(prices)
            source = "tick-risk"

        buffer = max(0.01, price * 0.003)
        raw_stop = tick_low - buffer
        min_risk = price * 0.012
        max_risk = price * 0.04
        risk = price - raw_stop
        if risk <= 0:
            risk = min_risk
        if risk < min_risk:
            risk = min_risk
        if risk > max_risk:
            return None

        stop_price = round(price - risk, 2)
        target_risk = max(risk * 1.25, price * 0.02)
        target_risk = min(target_risk, price * 0.06)
        target_price = round(price + target_risk, 2)
        rr_note = "{} risk={:.1f}% target={:.1f}%".format(
            source, risk / price * 100, target_risk / price * 100)
        return price, stop_price, target_price, rr_note

    def _check_quick_scalp_entry(self, symbol: str, bars: Sequence[Bar]) -> Optional[str]:
        """Fast-mover guard for HOD tick scalps.

        This is intentionally different from the normal entry guard. It lets
        explosive low-float momentum through earlier, but keeps hard execution
        safety around volume, spread, tape pressure, and distance from HOD.
        """
        if len(bars) < 3:
            return "insufficient bars for quick scalp"

        latest = bars[-1]
        price = latest.close
        if price <= 0:
            return "invalid price"

        today = list(bars)
        if latest.ts is not None:
            try:
                today_date = latest.ts.date()
                today = [b for b in bars if b.ts is None or b.ts.date() == today_date]
            except Exception:
                today = list(bars)

        if len(today) < 3:
            return "too few today bars for quick scalp"

        day_volume = sum(b.volume for b in today)
        if day_volume < 500_000:
            return "quick scalp volume too low {:.0f} (need 500K+)".format(day_volume)

        session_open = today[0].open
        day_change = ((price - session_open) / session_open * 100) if session_open > 0 else 0.0

        recent_window = today[-5:] if len(today) >= 5 else today
        hod = max(b.high for b in today)
        recent_hod = max(b.high for b in recent_window)
        distance_from_hod = 0.0
        distance_from_recent_hod = 0.0
        if hod > 0:
            distance_from_hod = (hod - price) / hod * 100
        if recent_hod > 0:
            distance_from_recent_hod = (recent_hod - price) / recent_hod * 100
        tradeable_hod_distance = min(distance_from_hod, distance_from_recent_hod)
        if hod > 0 and recent_hod > 0:
            if distance_from_hod > 12.0 and distance_from_recent_hod > 8.0:
                return "quick scalp too far from HOD {:.1f}%".format(distance_from_hod)

        recent_lows = [b.low for b in recent_window if b.low > 0]
        recent_low = min(recent_lows) if recent_lows else 0.0
        recent_move = ((price - recent_low) / recent_low * 100) if recent_low > 0 else 0.0
        recent_hod_push = recent_move >= 8.0 and tradeable_hod_distance <= 8.0
        if day_change < 20.0 and not recent_hod_push:
            return (
                "quick scalp movement too small day={:.1f}% recent={:.1f}% "
                "(need day 20%+ or recent 8% near HOD)"
            ).format(day_change, recent_move)

        recent = today[-3:]
        recent_volume = sum(b.volume for b in recent)
        min_recent_volume = 50_000
        if day_volume >= 3_000_000 and day_change >= 30.0:
            min_recent_volume = 40_000
        if recent_volume < min_recent_volume:
            return "quick scalp tape too slow {:.0f} recent volume".format(recent_volume)

        quotes = list(self._quote_buffer.get(symbol, []))
        if quotes:
            recent_quotes = [
                q for q in quotes[-5:]
                if q.ask > q.bid > 0
            ]
            if recent_quotes:
                avg_spread_pct = sum(
                    (q.ask - q.bid) / ((q.ask + q.bid) / 2.0) * 100
                    for q in recent_quotes
                ) / len(recent_quotes)
                momentum_pct = max(day_change, recent_move)
                max_spread = 1.5 if day_volume >= 1_000_000 and momentum_pct >= 50.0 else 0.8
                if avg_spread_pct > max_spread:
                    return "quick scalp spread too wide {:.2f}% (max {:.1f}%)".format(
                        avg_spread_pct, max_spread)

        ticks = list(self._tick_buffer.get(symbol, []))
        if len(ticks) >= 10:
            recent_ticks = ticks[-30:]
            buy_vol = sum(t.size for t in recent_ticks if t.side is Side.BUY)
            sell_vol = sum(t.size for t in recent_ticks if t.side is Side.SELL)
            if sell_vol > 0 and buy_vol / sell_vol < 0.8:
                return "quick scalp tape weak buy/sell {:.2f}".format(buy_vol / sell_vol)

        return None

    def _run_trade_analysis(self) -> None:
        """Run AI trade analysis on recent trades and apply adjustments."""
        if self._trade_analyzer is None:
            return
        try:
            from daytrading.analytics.trade_analyzer import TradeRecord as ATR

            hub_trades = list(self._hub.trades)
            records = []
            for t in hub_trades:
                if t.trade_type != "exit":
                    continue
                records.append(ATR(
                    symbol=t.symbol, side=t.side, quantity=t.quantity,
                    entry_price=t.entry_price, exit_price=t.exit_price,
                    pnl=t.pnl, exit_reason=t.exit_reason,
                    entry_time=t.entry_time, exit_time=t.exit_time,
                ))

            result = self._trade_analyzer.analyze(records)

            for sym in result.blocked_symbols:
                self._pipeline.set_cooldown(sym)
                self._hub.add_log("AI", "BLOCKED {}: {}".format(
                    sym, self._trade_analyzer.blocked_symbols.get(sym, "")))

            if "position_size" in result.adjusted_params:
                new_size = result.adjusted_params["position_size"]
                self._hub.add_log("AI", "Position size adjusted to {:.0f} shares".format(new_size))

            if "cooldown_seconds" in result.adjusted_params:
                new_cd = int(result.adjusted_params["cooldown_seconds"])
                self._pipeline._cooldown_seconds = new_cd
                self._hub.add_log("AI", "Cooldown increased to {}s".format(new_cd))

            if "min_momentum_quality" in result.adjusted_params:
                self._hub.add_log("AI", "Momentum quality threshold raised to {:.0f}".format(
                    result.adjusted_params["min_momentum_quality"]))

            self._hub.ai_analysis = self._trade_analyzer.snapshot()
            self._hub.ai_analysis["score"] = result.score
            self._hub._broadcast("ai_update", self._hub.ai_analysis)

            for ins in result.insights:
                level = "WARNING" if ins.severity in ("critical", "warning") else "INFO"
                self._hub.add_log(level, "AI: {}".format(ins.message))

        except Exception as exc:
            logger.warning("Trade analysis error: %s", exc)

    def _refresh_watchlist_bars(self) -> None:
        """Refresh 1-minute bars for watchlist symbols every 30 seconds.

        This prevents pattern scanners from rejecting with 'stale data'
        when a stock hasn't had new ticks in a few minutes.
        """
        now_mono = time.time()
        if now_mono - self._last_watchlist_bar_refresh < self._watchlist_bar_refresh_sec:
            return

        self._last_watchlist_bar_refresh = now_mono

        stale_syms = []
        now_utc = datetime.now(timezone.utc)
        for sym in self._watchlist_set:
            if sym in self._watchlist_pinned:
                continue
            bars = self._bar_buffer.get(sym)
            if not bars:
                stale_syms.append(sym)
                continue
            last_bar = bars[-1]
            if last_bar.ts is not None:
                age = (now_utc - last_bar.ts).total_seconds()
                if age > 120:
                    stale_syms.append(sym)

        if not stale_syms:
            return

        batch = stale_syms[:10]
        try:
            fresh = self._hist.get_bars(batch, limit=30)
            today_et = self._now_et().date()
            refreshed = 0
            for sym, sym_bars in fresh.items():
                today_bars = [
                    b for b in sym_bars
                    if b.ts is None or self._bar_is_today(b, today_et)
                ]
                if today_bars:
                    self._bar_buffer[sym] = deque(
                        today_bars[-self._max_bars_per_symbol:],
                        maxlen=self._max_bars_per_symbol,
                    )
                    refreshed += 1
            if refreshed > 0:
                logger.info(
                    "BAR REFRESH: updated %d/%d watchlist symbols",
                    refreshed, len(batch),
                )
        except Exception as exc:
            logger.debug("Watchlist bar refresh error: %s", exc)

    def _update_ml_shadow(self, bar_universe: dict) -> None:
        """Feed latest prices to ML monitor and check shadow outcomes."""
        try:
            from daytrading.strategy.entry_guard import get_ml_monitor
            monitor = get_ml_monitor()
            if monitor is None:
                return
            for sym, bars in bar_universe.items():
                if bars:
                    monitor.update_price(sym, bars[-1].close)
            monitor.check_shadow_outcomes()
        except Exception:
            pass
        try:
            from daytrading.ml.shadow_collector import (
                log_exit_snapshot,
                update_deferred_outcomes,
            )
            update_deferred_outcomes(bar_universe)
            tracked = self._pipeline.exit_manager.tracked
            for sym, pos in tracked.items():
                bars = bar_universe.get(sym) or list(self._bar_buffer.get(sym, []))
                if not bars:
                    continue
                price = bars[-1].close
                log_exit_snapshot(
                    symbol=sym,
                    price=price,
                    entry_price=pos.entry_price,
                    remaining_qty=pos.remaining_qty,
                    sold_half=pos.sold_half,
                    breakeven_locked=pos.breakeven_locked,
                    reason=pos.reason,
                    bars=bars,
                )
        except Exception:
            pass

    def _inject_recent_trade_bars(self, bar_universe: dict) -> None:
        """Append a live synthetic 1m bar when REST bars are stale but tape is fresh.

        Alpaca's historical bars can lag or return the last completed bar for thin
        names. If we have fresh trade prints from the WebSocket, use them to keep
        entry checks from rejecting a still-active HOD/watchlist symbol as stale.
        """
        now_ts = datetime.now(timezone.utc)
        injected = 0
        for sym in list(self._watchlist_set):
            if sym in self._watchlist_pinned:
                continue

            bars = self._bar_buffer.get(sym)
            if not bars:
                continue
            last_bar = bars[-1]
            if last_bar.ts is None:
                continue

            try:
                bar_age = (now_ts - last_bar.ts).total_seconds()
            except Exception:
                continue
            if bar_age <= 120:
                continue

            ticks = list(self._tick_buffer.get(sym, []))
            recent_ticks = [
                t for t in ticks
                if t.ts is not None and (now_ts - t.ts).total_seconds() <= 60
            ]
            if not recent_ticks:
                continue

            prices = [t.price for t in recent_ticks if t.price > 0]
            if not prices:
                continue

            live_bar = Bar(
                symbol=sym,
                ts=now_ts,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=sum(t.size for t in recent_ticks if t.size > 0),
            )
            bars.append(live_bar)
            bar_universe[sym] = list(bars)
            injected += 1

        if injected:
            logger.info("LIVE BAR INJECT: refreshed %d stale watchlist symbols from tape", injected)

    def _run_one_cycle(self, cycle_num: int) -> None:
        """Execute one pipeline cycle with current buffered data.

        Also injects live Alpaca prices for open positions that have
        no bars in the buffer, so the exit manager can always check
        stops/targets/trails — even for dead stocks with no new bars.
        """
        if self._stream.is_running:
            try:
                self._stream.flush_pending_subscriptions()
            except Exception:
                pass

        self._refresh_watchlist_bars()

        bar_universe = {s: list(b) for s, b in self._bar_buffer.items() if b}
        quotes = {s: list(q) for s, q in self._quote_buffer.items() if q}
        ticks = {s: list(t) for s, t in self._tick_buffer.items() if t}
        self._inject_recent_trade_bars(bar_universe)

        trade_universe = self._trade_universe(bar_universe)
        trade_syms = set(trade_universe.keys())
        quotes = {s: q for s, q in quotes.items() if s in trade_syms}
        ticks = {s: t for s, t in ticks.items() if s in trade_syms}

        # Build 5-minute bars from 1-minute data for higher-timeframe context
        self._bar_aggregator.update_all_5m(trade_universe)

        # Inject live prices for open positions.
        # Without this, the exit manager and max-loss checks use stale bar data.
        open_syms = set(self._pipeline.exit_manager.tracked.keys())
        open_syms.update(
            sym for sym, pos in self._pipeline.portfolio.positions.items()
            if not pos.is_flat
        )
        if open_syms:
            try:
                now_ts = datetime.now(timezone.utc)
                live = self._live_prices(list(open_syms))
                for sym, price in live.items():
                    if price <= 0:
                        continue
                    live_bar = Bar(
                        symbol=sym, open=price, high=price,
                        low=price, close=price, volume=0,
                        ts=now_ts,
                    )
                    if sym in trade_universe:
                        trade_universe[sym].append(live_bar)
                    else:
                        trade_universe[sym] = [live_bar]
                    if sym in bar_universe:
                        bar_universe[sym].append(live_bar)
                    else:
                        bar_universe[sym] = [live_bar]
            except Exception:
                pass

        if not trade_universe and not bar_universe:
            self._hub.on_cycle_heartbeat(cycle_num, "no bars yet")
            return

        now = datetime.now(timezone.utc)

        # Snapshot entry prices before run_cycle, because the exit manager
        # untracks positions after generating exit signals.
        entry_prices = {
            sym: pos.entry_price
            for sym, pos in self._pipeline.exit_manager.tracked.items()
        }

        try:
            if self._hub.trading_paused:
                logger.debug("Trading paused — skipping cycle %d", cycle_num)
                self._hub.on_cycle_heartbeat(cycle_num, "trading paused")
                return
            if trade_universe:
                if self._pipeline._router is not None:
                    self._pipeline._router.classifier.is_premarket = (
                        self._market_phase() == "PRE-MARKET"
                    )
                result = self._pipeline.run_cycle(
                    trade_universe, now=now, quotes=quotes, ticks=ticks,
                )
                result._entry_prices = entry_prices
                self._process_result(
                    result, cycle_num, trade_universe,
                    hod_bar_count=len(bar_universe),
                )
            else:
                self._refresh_hod_momentum_alerts(bar_universe, None)

            if cycle_num > 0 and cycle_num % self._analysis_interval == 0:
                self._run_trade_analysis()
        except Exception as exc:
            logger.error("Cycle %d error: %s", cycle_num, exc, exc_info=True)
            self._hub.add_log("ERROR", "Cycle {} error: {}".format(cycle_num, exc))

        # ML shadow mode: update prices and check outcomes
        self._update_ml_shadow(bar_universe)

    def _process_result(
        self,
        result: PipelineResult,
        cycle_num: int,
        universe: Dict[str, List[Bar]],
        *,
        hod_bar_count: int = 0,
    ) -> None:
        """Log events and push them to the dashboard."""
        # Push classifications to dashboard
        for sym, regime in result.regimes.items():
            self._hub.on_classification(sym, regime)
            self._journal.record("classification", {
                "symbol": sym,
                "style": regime.style.value,
                "confidence": regime.confidence,
                "reasons": list(regime.reasons),
                "metrics": {
                    "volatility_pct": regime.volatility_pct,
                    "spread_pct": regime.spread_pct,
                    "relative_volume": regime.relative_volume,
                    "trend_strength": regime.trend_strength,
                    "liquidity_score": regime.liquidity_score,
                },
            })

        # Push scanner hits with rejection reasons
        rejections = self._pipeline.scan_rejections
        for hit in result.scan_hits:
            is_verified = hit.symbol not in rejections
            reject_reason = rejections.get(hit.symbol)
            self._hub.on_scan_hit(hit, verified=is_verified, reject_reason=reject_reason)
            self._journal.record("scan_hit", {
                "symbol": hit.symbol,
                "scanner": hit.scanner_name,
                "score": hit.score,
                "criteria": dict(hit.criteria),
                "verified": is_verified,
                "reject_reason": reject_reason,
                "candle_snapshot": self._journal.candle_snapshot(hit.bars, limit=30),
            })
            if reject_reason:
                self._journal.record("mistake", {
                    "symbol": hit.symbol,
                    "kind": "scan_rejection",
                    "reason": reject_reason,
                    "scanner": hit.scanner_name,
                })

        # Push signals + news sentiment
        for sig in result.signals:
            self._hub.on_signal(sig)
            sig_bars = universe.get(sig.symbol, [])
            self._journal.record("signal", {
                "symbol": sig.symbol,
                "action": sig.action.value,
                "quantity": sig.quantity,
                "entry_price": sig.entry_price,
                "stop_loss": sig.stop_loss,
                "take_profit": sig.take_profit,
                "reason": sig.reason,
                "strategy": sig.scan_result.scanner_name if sig.scan_result else "",
                "criteria": dict(sig.scan_result.criteria) if sig.scan_result else {},
                "candle_snapshot": self._journal.candle_snapshot(sig_bars, limit=40),
            })
            if self._news_checker:
                try:
                    score, headlines = self._news_checker.get_sentiment(sig.symbol)
                    self._hub.on_news(sig.symbol, score, headlines)
                except Exception:
                    pass

        # Push entry fills
        for f in result.fills:
            logger.info(
                "[Cycle %d] ENTRY %s %s %.0f @ $%.2f",
                cycle_num, f.side.value.upper(), f.symbol, f.quantity, f.price,
            )
            self._hub.on_fill(f, "entry")
            self._hub.add_log("INFO", "ENTRY {} {} {:.0f} @ ${:.2f}".format(
                f.side.value.upper(), f.symbol, f.quantity, f.price))
            strategy = result.entry_strategies.get(f.symbol, "")
            self._journal.record("trade_fill", {
                "symbol": f.symbol,
                "side": f.side.value,
                "quantity": f.quantity,
                "price": f.price,
                "ts": f.ts,
                "trade_type": "entry",
                "strategy": strategy,
                "market_context": {
                    "phase": self._market_phase(),
                    "cycle": cycle_num,
                },
            }, ts=f.ts)
            tracked = self._pipeline.exit_manager.tracked.get(f.symbol)
            if tracked and tracked.stop_loss:
                self._arm_broker_protection(
                    f.symbol, f.quantity, tracked.stop_loss,
                )

        # Push exit fills
        entry_prices = getattr(result, '_entry_prices', {})
        for f in result.exit_fills:
            logger.info(
                "[Cycle %d] EXIT %s %s %.0f @ $%.2f",
                cycle_num, f.side.value.upper(), f.symbol, f.quantity, f.price,
            )
            entry_price = entry_prices.get(f.symbol, 0.0)
            if entry_price == 0.0:
                tracked = self._pipeline.exit_manager.tracked
                if f.symbol in tracked:
                    entry_price = tracked[f.symbol].entry_price
            if entry_price == 0.0:
                pos = self._pipeline.portfolio.positions.get(f.symbol)
                if pos:
                    entry_price = pos.avg_price
            exit_reason = result.exit_reasons.get(f.symbol, "")
            self._record_trade_exit(
                f,
                entry_price,
                exit_reason,
                strategy=result.entry_strategies.get(f.symbol, ""),
                cycle_num=cycle_num,
            )

        # Push scale-up fills
        if hasattr(result, 'scale_up_fills') and result.scale_up_fills:
            for f in result.scale_up_fills:
                logger.info(
                    "[Cycle %d] SCALE UP %s %s +%.0f @ $%.2f",
                    cycle_num, f.side.value.upper(), f.symbol, f.quantity, f.price,
                )
                self._hub.on_fill(f, "scale_up")
                self._hub.add_log("INFO", "SCALE UP {} +{:.0f} @ ${:.2f}".format(
                    f.symbol, f.quantity, f.price))
                self._journal.record("trade_fill", {
                    "symbol": f.symbol,
                    "side": f.side.value,
                    "quantity": f.quantity,
                    "price": f.price,
                    "ts": f.ts,
                    "trade_type": "scale_up",
                }, ts=f.ts)

        # Push re-entry fills
        if hasattr(result, 'reentry_fills') and result.reentry_fills:
            for f in result.reentry_fills:
                logger.info(
                    "[Cycle %d] RE-ENTRY %s %s %.0f @ $%.2f",
                    cycle_num, f.side.value.upper(), f.symbol, f.quantity, f.price,
                )
                self._hub.on_fill(f, "reentry")
                self._hub.add_log("INFO", "RE-ENTRY {} {:.0f} @ ${:.2f}".format(
                    f.symbol, f.quantity, f.price))
                self._journal.record("trade_fill", {
                    "symbol": f.symbol,
                    "side": f.side.value,
                    "quantity": f.quantity,
                    "price": f.price,
                    "ts": f.ts,
                    "trade_type": "reentry",
                }, ts=f.ts)

        # Queue deferred signals into execution timer (10-sec micro-entry)
        for sig in getattr(result, 'deferred_signals', []):
            self._exec_timer.queue(sig)

        # Seed order IDs from pipeline fills to prevent duplicate pushes
        if result.fills or result.exit_fills:
            try:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=10)
                recent = self._broker.client.get_orders(filter=req)
                for o in recent:
                    self._last_synced_order_ids.add(str(o.id))
            except Exception:
                pass

        # Push rejected count with details
        for detail in result.rejection_details:
            self._hub.on_rejected()
            self._journal.record("mistake", {
                "kind": "risk_rejection",
                "symbol": detail.get("symbol", ""),
                "reason": detail.get("reason", ""),
                "cycle": cycle_num,
                "phase": self._market_phase(),
            })

        # Safety net: force-close any position losing more than $100
        # Use live Alpaca position data for accurate P&L (not stale bar data)
        max_loss_per_position = 100.0
        prices = {sym: bars[-1].close for sym, bars in universe.items() if bars}
        try:
            alpaca_positions = self._broker.get_positions()
            for sym, pdata in alpaca_positions.items():
                live_price = float(pdata.get("current_price", 0))
                if live_price > 0:
                    prices[sym] = live_price
        except Exception:
            pass
        for sym, pos in list(self._pipeline.portfolio.positions.items()):
            if pos.is_flat:
                continue
            price = prices.get(sym)
            if price is None:
                continue
            unrealized = pos.unrealized_pnl(price)
            if unrealized < -max_loss_per_position:
                logger.warning(
                    "MAX LOSS TRIGGERED %s: P&L=$%.2f (limit=$%.0f) — force closing",
                    sym, unrealized, max_loss_per_position,
                )
                self._hub.add_log("WARNING", "MAX LOSS {} P&L=${:.2f} — force closing".format(sym, unrealized))
                try:
                    actual_qty = int(alpaca_positions.get(sym, {}).get("qty", 0))
                    if actual_qty <= 0:
                        logger.info("FORCE CLOSE %s — already closed on Alpaca", sym)
                        pos.quantity = 0.0
                        self._pipeline.exit_manager.untrack(sym)
                        continue

                    from alpaca.trading.requests import LimitOrderRequest
                    from alpaca.trading.enums import OrderSide, TimeInForce
                    self._broker._cancel_open_orders_for(sym)
                    limit_price = round(price * 0.995, 2)
                    req = LimitOrderRequest(
                        symbol=sym, qty=actual_qty, side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                        limit_price=limit_price,
                        extended_hours=True,
                    )
                    self._broker.client.submit_order(order_data=req)
                    logger.info("FORCE CLOSE submitted: limit sell %s %d @ $%.2f", sym, actual_qty, limit_price)
                except Exception as exc:
                    logger.error("Failed to force close %s: %s", sym, exc)

        # After any fills or exits, immediately sync from Alpaca for accuracy
        if result.fills or result.exit_fills:
            self._push_positions_from_alpaca()
        else:
            self._hub.on_position_update(self._pipeline.portfolio.positions, prices)

        # Cycle complete
        self._hub.on_cycle_complete(cycle_num, result)
        self._journal.record("market_regime", {
            "cycle": cycle_num,
            "phase": self._market_phase(),
            "symbols": {
                sym: {
                    "style": reg.style.value,
                    "confidence": reg.confidence,
                    "trend_strength": reg.trend_strength,
                    "relative_volume": reg.relative_volume,
                    "spread_pct": reg.spread_pct,
                    "reasons": list(reg.reasons),
                }
                for sym, reg in result.regimes.items()
            },
        })
        self._journal.record("cycle", {
            "cycle": cycle_num,
            "phase": self._market_phase(),
            "symbols_scanned": len(result.regimes),
            "scan_hits": len(result.scan_hits),
            "signals": len(result.signals),
            "fills": len(result.fills),
            "exits": len(result.exit_fills),
            "rejected": result.rejected_orders,
        })

        # Summary log — always push to dashboard
        routed = len([s for s, r in result.regimes.items() if r.style.value != "not_tradeable"])
        phase = self._market_phase()
        trade_n = len(universe)
        hod_n = hod_bar_count or trade_n
        summary = (
            "[Cycle {}] [{}] {} trade symbols scanned ({} with bars for HOD), "
            "{} routed, {} scan hits, {} signals, {} fills"
        ).format(
            cycle_num,
            phase,
            trade_n,
            hod_n,
            routed,
            len(result.scan_hits),
            len(result.signals),
            len(result.fills),
        )
        self._hub.add_log("INFO", summary)
        logger.info(summary)

        # Track consecutive skips and prune stale watchlist symbols
        for sym, reg in result.regimes.items():
            if reg.style.value == "not_tradeable":
                self._skip_counts[sym] += 1
            else:
                self._skip_counts[sym] = 0

        if cycle_num > 0 and cycle_num % self._CLEANUP_EVERY == 0:
            self._prune_watchlist()

        bar_for_hod = {s: list(b) for s, b in self._bar_buffer.items() if b}
        self._refresh_hod_momentum_alerts(bar_for_hod, result)

    def _refresh_hod_momentum_alerts(
        self,
        universe: Dict[str, List[Bar]],
        result: Optional[PipelineResult] = None,
    ) -> None:
        """Bar-path enrichment for HOD Momentum alert table."""
        if self._hod_bar_scanner is None or self._hod_alert_store is None:
            return

        self._prune_hod_bar_pool_by_price()

        now = datetime.now(timezone.utc)
        ttl_secs = self._hod_alert_ttl_minutes * 60
        for sym, ts in list(self._hod_active.items()):
            if (now - ts).total_seconds() >= ttl_secs:
                del self._hod_active[sym]

        rejections = self._pipeline.scan_rejections if result else {}
        verified_syms = set()
        if result:
            for hit in result.scan_hits:
                if hit.symbol not in rejections:
                    verified_syms.add(hit.symbol)

        rel_vols = {}
        if result:
            for sym, reg in result.regimes.items():
                rel_vols[sym] = reg.relative_volume

        scan_universe = self._expand_hod_universe(universe)

        bars_5m = {}
        if self._bar_aggregator:
            self._bar_aggregator.update_all_5m(scan_universe)
            for sym in scan_universe:
                b5 = self._bar_aggregator.get_5m_bars(sym)
                if b5:
                    bars_5m[sym] = b5

        if self._hod_tick_tracker is not None:
            for sym, bars in scan_universe.items():
                if bars and not self._hod_tick_tracker.is_seeded(sym):
                    prior = self._prior_day_stats.get(sym)
                    self._hod_tick_tracker.update_session_from_bars(
                        sym, bars, prior_day=prior,
                    )

        reject_stats = self._hod_bar_scanner.scan(
            scan_universe,
            bars_5m=bars_5m,
            rel_vols=rel_vols,
            prior_day_stats=self._prior_day_stats,
            verified_symbols=verified_syms,
            rejections=rejections,
            is_premarket=(self._market_phase() == "PRE-MARKET"),
        )

        # Mark gapper symbols in tick tracker for near-HOD reclaim detection
        if self._hod_tick_tracker is not None:
            for sym in self._hod_bar_scanner._gapper_fired:
                self._hod_tick_tracker.mark_gapper(sym)

        scanned = reject_stats.pop("_scanned", 0)
        new_alerts = reject_stats.pop("_new_alerts", 0)
        on_board = len(self._hod_alert_store.snapshot()) if self._hod_alert_store else 0
        reject_summary = " ".join(f"{k}={v}" for k, v in sorted(reject_stats.items()) if v)
        logger.info(
            "HOD scanner [%s]: %d scanned, %d alerts, %d on board | rejects: %s",
            self._market_phase(), scanned, new_alerts, on_board,
            reject_summary or "none",
        )

        if self._hod_former_momo_scanner is not None:
            self._hod_former_momo_scanner.scan(
                scan_universe,
                prior_day_stats=self._prior_day_stats,
                rel_vols=rel_vols,
                verified_symbols=verified_syms,
                rejections=rejections,
            )

        if self._hod_alert_store is not None:
            self._sync_watchlist_to_hod_alerts()
            self._publish_hod_alert_board()

    def _prune_watchlist(self) -> None:
        """Remove symbols that have been consecutively skipped (illiquid/dead)."""
        open_syms = {
            sym for sym, pos in self._pipeline.portfolio.positions.items()
            if not pos.is_flat
        }
        tracked_syms = set(self._pipeline.exit_manager.tracked.keys())
        protected = open_syms | tracked_syms | self._watchlist_pinned

        to_remove = []
        for sym, count in self._skip_counts.items():
            if count >= self._SKIP_THRESHOLD and sym not in protected:
                to_remove.append(sym)

        if not to_remove:
            return

        for sym in to_remove:
            if sym in self._watchlist_set:
                self._watchlist.remove(sym)
                self._watchlist_set.discard(sym)
                self._skip_counts.pop(sym, None)
                self._bar_buffer.pop(sym, None)
                self._quote_buffer.pop(sym, None)
                self._tick_buffer.pop(sym, None)

        logger.info("WATCHLIST CLEANUP: removed %d stale symbols: %s — watchlist now %d: %s",
                    len(to_remove), to_remove, len(self._watchlist), self._watchlist)

    def _run_nightly_analysis(self) -> None:
        """Run the nightly trade analyst after market close."""
        try:
            from daytrading.analyst.collector import NightlyAnalyst
            report_dir = os.path.join(os.path.dirname(self._journal.base_dir), "reports")
            analyst = NightlyAnalyst(db_path=self._journal.db_path, report_dir=report_dir)
            report = analyst.run()
            status = report.get("status", "ok")
            if status in ("no_trades", "holiday"):
                logger.info("NIGHTLY ANALYST: %s — no report generated", status)
            else:
                day = report.get("day", "unknown")
                problems = report.get("problems", [])
                summary = report.get("summary", {})
                md_path = os.path.join(report_dir, f"{day}.md")
                logger.info(
                    "NIGHTLY ANALYST: report for %s — %dW/%dL, P&L $%.2f, %d problems detected",
                    day, summary.get("win_count", 0), summary.get("loss_count", 0),
                    summary.get("total_pnl", 0), len(problems),
                )
                logger.info("NIGHTLY REPORT saved: %s", md_path)
                for p in problems:
                    logger.warning(
                        "NIGHTLY PROBLEM [%s]: %s",
                        p["severity"], p["problem"],
                    )
                self._hub.add_log("INFO", f"Nightly report ready: {md_path}")
        except Exception as exc:
            logger.error("NIGHTLY ANALYST failed: %s", exc)

        # Log ML monitor summary for the day
        self._log_ml_daily_summary()

        # Daily ML model retrain after market close
        self._retrain_ml_model()

    def _log_ml_daily_summary(self) -> None:
        """Log ML model performance stats at end of day."""
        try:
            from daytrading.strategy.entry_guard import get_ml_monitor
            monitor = get_ml_monitor()
            if monitor is None:
                return
            summary = monitor.get_summary_line()
            logger.info("ML DAILY: %s", summary)
            self._hub.add_log("INFO", f"ML Daily: {summary}")
            stats = monitor.stats
            if stats.model_disabled:
                logger.warning("ML MODEL was auto-disabled today: %s", stats.disable_reason)
                self._hub.add_log("WARN", f"ML model disabled: {stats.disable_reason}")
            monitor.reset_daily()
        except Exception as exc:
            logger.debug("ML daily summary error: %s", exc)

    def _retrain_ml_model(self) -> None:
        """Retrain XGBoost model with collected data after market close."""
        try:
            from daytrading.ml.data_collector import count_labeled
            labeled = count_labeled()
            if labeled < 5:
                logger.info("ML RETRAIN: skipped — only %d labeled samples (need 5+)", labeled)
            else:
                logger.info("ML RETRAIN: starting with %d labeled samples...", labeled)
                from daytrading.ml.train import train
                trained = train()
                if trained:
                    logger.info("ML RETRAIN: complete — model updated for tomorrow")
                    self._hub.add_log("INFO", f"ML model retrained ({labeled} samples)")
                else:
                    logger.info("ML RETRAIN: skipped — trainer did not write a model")
                    self._hub.add_log("INFO", "ML retrain skipped — not enough usable data")
        except Exception as exc:
            logger.warning("ML RETRAIN failed: %s", exc)

        try:
            from daytrading.ml.shadow_collector import dataset_counts
            from daytrading.ml.shadow_train import train_all_shadow_models
            counts = dataset_counts()
            logger.info("ML SHADOW DATASETS: %s", counts)
            results = train_all_shadow_models(min_samples=20)
            trained = [name for name, ok in results.items() if ok]
            if trained:
                logger.info("ML SHADOW RETRAIN: trained %s", ", ".join(trained))
                self._hub.add_log("INFO", "ML shadow trained: {}".format(", ".join(trained)))
            else:
                logger.info("ML SHADOW RETRAIN: no advisory models trained")
        except Exception as exc:
            logger.warning("ML SHADOW RETRAIN failed: %s", exc)

    def _shutdown_gracefully(self) -> None:
        """Clean shutdown: stop scanner, stream, optionally close positions."""
        if getattr(self, '_did_shutdown', False):
            return
        self._did_shutdown = True

        logger.info("Shutting down...")

        self._pool_refresh_event.set()

        if self._scanner:
            self._scanner.stop_live()

        self._stream.stop()

        import os, time as _time
        _lock = os.path.join(os.path.dirname(__file__), ".stream_lock")
        try:
            with open(_lock, "w") as f:
                f.write(str(_time.time()))
        except Exception:
            pass

        if self._close_at_eod:
            logger.info("Closing all positions (end-of-day safety)...")
            try:
                self._broker.close_all_positions()
            except Exception as exc:
                logger.error("Error closing positions: %s", exc)

        # print final state
        try:
            acct = self._broker.get_account()
            logger.info(
                "Final account: cash=$%.2f, equity=$%.2f",
                acct["cash"], acct["equity"],
            )
        except Exception:
            pass

        portfolio = self._pipeline.portfolio
        logger.info("Final portfolio cash: $%.2f", portfolio.cash)
        if portfolio.positions:
            for sym, pos in portfolio.positions.items():
                logger.info("  %s: %.0f shares @ $%.2f avg", sym, pos.quantity, pos.avg_price)

        logger.info("=" * 60)
        logger.info("SESSION COMPLETE")
        logger.info("=" * 60)

    def _setup_signals(self) -> None:
        """Handle Ctrl+C and SIGTERM gracefully, with atexit fallback."""
        import atexit

        def handler(sig: int, frame: object) -> None:
            self._shutdown = True
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        atexit.register(self._shutdown_gracefully)


def _fallback_watchlist() -> List[str]:
    """Fallback list if the dynamic scan fails."""
    return [
        "SOUN", "SOFI", "NIO", "RIVN", "MARA",
        "LCID", "SNAP", "AAL", "PLUG", "OPEN",
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Alpaca paper trading bot from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("""
    ╔══════════════════════════════════════════════╗
    ║   DAY TRADING BOT — ALPACA PAPER TRADING     ║
    ║                                              ║
    ║   Scalping $1–$20 stocks                     ║
    ║   Smart exits · Scale up · Re-entry          ║
    ║   Breakeven lock · Adaptive trailing          ║
    ║                                              ║
    ║   Press Ctrl+C to stop                       ║
    ╚══════════════════════════════════════════════╝
    """)

    try:
        runner = AlpacaRunner.from_env()
        runner.run()
    except ValueError as exc:
        logger.error(str(exc))
        print(f"\nERROR: {exc}")
        print("\nCreate a .env file with:")
        print("  DAYTRADING_ALPACA_API_KEY=your_key_here")
        print("  DAYTRADING_ALPACA_SECRET_KEY=your_secret_here")
        print("  DAYTRADING_ALPACA_PAPER=true")
        sys.exit(1)


if __name__ == "__main__":
    main()
