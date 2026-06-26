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
from typing import Any, Dict, List, Optional, Sequence, Set

from daytrading.config import Settings
from daytrading.market_calendar import ET, is_us_trading_day, now_et
from daytrading.dashboard.hub import DashboardHub
from daytrading.dashboard.server import start_dashboard
from daytrading.data.alpaca_feed import AlpacaHistoricalFeed, AlpacaStreamFeed, configure_alpaca_stream_logging
from daytrading.data.float_checker import FloatChecker
from daytrading.data.float_store import FloatStore
from daytrading.data.market_data_service import MarketDataService
from daytrading.data.news_checker import NewsChecker
from daytrading.data.watchlist_scanner import WatchlistScanner
from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.execution.entry_executor import EntryExecutionContext, EntryExecutor
from daytrading.execution.live_prices import resolve_live_prices
from daytrading.execution.position_reconciler import PositionReconciler
from daytrading.exits.manager import ExitManager, is_hit_run_strategy
from daytrading.exits.scaler import PositionScaler, ReentryDetector
from daytrading.exits.tape_pressure import TapePressureExit
from daytrading.indicators.core import vwap
from daytrading.journal.store import TradingJournal
from daytrading.pipeline.engine import PipelineResult, TradingPipeline
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.models import Bar, Fill, Order, OrderStatus, PortfolioState, Position, Quote, ScanResult, Side, SignalAction, Tick, TradeSignal
from daytrading.strategy.entry_guard import (
    assess_opportunity_scaled_spread,
    check_entry_quality,
    tick_aware_spread_ok,
)
from daytrading.strategy.entry_policy import EntryDecision, EntryPolicy
from daytrading.strategy import warrior_lanes
from daytrading.strategy.warrior_engine import WarriorEngine

logger = logging.getLogger(__name__)

# --- Event types for the lock-free producer→consumer queue ---
BarEvent = namedtuple("BarEvent", ["symbol", "bar"])
QuoteEvent = namedtuple("QuoteEvent", ["symbol", "quote"])
TradeEvent = namedtuple("TradeEvent", ["symbol", "tick"])
BarsLoadedEvent = namedtuple("BarsLoadedEvent", ["bars_by_symbol", "prior_day_stats"])
PoolRefreshEvent = namedtuple("PoolRefreshEvent", ["new_pool", "bars", "prior_day_stats"])
FastScanEvent = namedtuple("FastScanEvent", ["new_movers"])

HOT_WATCH_PATTERNS = frozenset({
    "abc_continuation",
    "first_pullback_reclaim",
    "pullback_base",
    "vwap_pullback",
    "hod_reclaim",
    "level_breakout_reclaim",
    "shallow_stair_continuation",
    "early_vwap_reclaim_scout",
    "flat_top_breakout",
})

_FLOAT_SPLIT_SANITY_LAST_CHECK: Dict[str, float] = {}


def _ensure_market_data_service(runner) -> MarketDataService:
    service = getattr(runner, "_market_data", None)
    if service is None:
        queue_max = 500
        existing_queue = getattr(runner, "_candidate_hydrate_queue", None)
        if hasattr(existing_queue, "maxsize") and int(existing_queue.maxsize or 0) > 0:
            queue_max = int(existing_queue.maxsize)
        service = MarketDataService(
            candidate_queue_max=queue_max,
            candidate_batch_max=getattr(runner, "_candidate_hydrate_batch_max", 10),
            hot_watch_max_symbols=getattr(runner, "_hot_watch_max_symbols", 40),
        )
        if isinstance(existing_queue, queue.PriorityQueue):
            service._candidate_queue = existing_queue
        existing_pending = getattr(runner, "_candidate_hydrate_pending", None)
        if isinstance(existing_pending, set):
            service._candidate_pending = existing_pending
        existing_hot = getattr(runner, "_hot_watch", None)
        if isinstance(existing_hot, dict):
            service.hot_watch.update(existing_hot)
        runner._market_data = service
        runner._candidate_hydrate_queue = service.candidate_queue
        runner._candidate_hydrate_pending = service._candidate_pending
        if isinstance(existing_hot, dict):
            runner._hot_watch = service.hot_watch
        return service
    existing_hot = getattr(runner, "_hot_watch", None)
    if isinstance(existing_hot, dict) and existing_hot is not service.hot_watch:
        service.hot_watch.clear()
        service.hot_watch.update(existing_hot)
        runner._hot_watch = service.hot_watch
    return service


def _ensure_entry_executor(runner) -> EntryExecutor:
    policy = getattr(runner, "_entry_policy", None)
    if policy is None:
        policy = EntryPolicy(
            guard=lambda *args, **kwargs: check_entry_quality(*args, **kwargs),
        )
        runner._entry_policy = policy
    executor = getattr(runner, "_entry_executor", None)
    if executor is None:
        executor = EntryExecutor(policy, runner._record_entry_decision)
        runner._entry_executor = executor
    else:
        executor.set_policy(policy)
        executor.set_recorder(runner._record_entry_decision)
    return executor


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
        self._exec_timer = ExecutionTimer(max_wait_bars=3, enabled=True)
        self._timed_signal_queue: deque = deque()

        self._skip_counts: Dict[str, int] = defaultdict(int)
        self._SKIP_THRESHOLD = 10  # remove after 10 consecutive skips (~10 min)
        self._CLEANUP_EVERY = 10   # run cleanup every 10 cycles
        self._new_data = Event()
        self._hub = DashboardHub()
        self._journal = TradingJournal()
        self._entry_policy = EntryPolicy(
            guard=lambda *args, **kwargs: check_entry_quality(*args, **kwargs),
        )
        self._entry_executor = EntryExecutor(self._entry_policy, self._record_entry_decision)
        try:
            self._pipeline._entry_policy = self._entry_policy
        except Exception:
            pass
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
        self._accidental_short_cleanup_enabled = True
        self._accidental_short_max_qty = float(
            os.getenv("DAYTRADING_ACCIDENTAL_SHORT_MAX_QTY", "0") or 0.0
        )
        self._accidental_short_cooldown_sec = 30.0
        self._accidental_short_cleanup_at: Dict[str, float] = {}
        # Persistent per-symbol anti-chase anchor. Pins the price where a
        # timed-entry setup FIRST deferred so re-queues of a grinding name
        # cannot crawl the chase ceiling up with the price.
        # symbol -> (anchor_price, last_seen_monotonic)
        self._timed_entry_anchor: Dict[str, tuple] = {}
        self._timed_entry_anchor_ttl_sec = 300.0
        self._momentum_breakout_enabled = False
        self._momentum_breakout_min_rvol = 0.4
        self._momentum_breakout_min_day_volume = 5_000_000.0
        self._momentum_breakout_max_bar_range_pct = 3.0
        self._momentum_breakout_score_floor = 72.0
        # symbol -> monotonic time when the experimental breakout bypass fired,
        # so the resulting entry can be tagged for isolated scorecard measurement
        self._momentum_breakout_armed: Dict[str, float] = {}
        self._momentum_burst_cycle_enabled = False
        self._momentum_burst_window_sec = 300.0
        self._momentum_burst_scalp_cooldown_sec = 300.0
        # symbol -> {ts, breakout_close} of a 10s high awaiting next-bar confirm
        self._momentum_burst_hit_run_enabled = False
        self._momentum_burst_hit_run_max_entries = 1
        self._momentum_burst_hit_run_win_cooldown_sec = 15.0
        self._momentum_burst_hit_run_loss_cooldown_sec = 90.0
        self._momentum_burst_hit_run_max_hold_sec = 45.0
        self._momentum_burst_hit_run_reward_risk = 1.0
        self._momentum_burst_hit_run_end_et = "11:30"
        self._momentum_burst_hit_run_stop_after_giveback = True
        self._momentum_burst_hit_run_max_giveback = 50.0
        self._momentum_burst_hit_run_daily_loss_stop = 50.0
        self._warrior_max_concurrent_trades = 1
        self._warrior_watch_capacity = 10
        self._warrior_watch_until_premarket_end = True
        self._warrior_engine = WarriorEngine.with_defaults(max_concurrent_warrior_trades=1)
        self._warrior_watch = self._warrior_engine.watch
        self._momentum_burst_armed = self._warrior_watch.armed
        self._momentum_burst_window_high = self._warrior_watch.window_high
        self._momentum_burst_session_anchor_high = self._warrior_watch.session_anchor_high
        self._momentum_burst_pending = self._warrior_watch.pending
        self._momentum_burst_hit_run_counts = self._warrior_watch.hit_run_counts
        self._momentum_burst_hit_run_block_until = self._warrior_watch.hit_run_block_until
        self._momentum_burst_hit_run_symbol_pnl = self._warrior_watch.symbol_pnl
        self._momentum_burst_hit_run_symbol_peak_pnl = self._warrior_watch.symbol_peak_pnl
        self._momentum_burst_hit_run_day_blocked = self._warrior_watch.day_blocked
        self._warrior_squeeze_enabled = False
        self._warrior_squeeze_min_reclaim_price = 3.5
        self._warrior_squeeze_starter_size_factor = 0.35
        self._warrior_squeeze_position_value = 2000.0
        self._warrior_squeeze_max_dollar_risk = 150.0
        self._warrior_squeeze_max_entries = 3
        self._warrior_squeeze_win_cooldown_sec = 10.0
        self._warrior_squeeze_reward_risk = 3.0
        self._warrior_squeeze_add_reward_risk = 1.0
        self._warrior_squeeze_rejection_high = self._warrior_watch.rejection_high
        self._warrior_squeeze_rejection_reason = self._warrior_watch.rejection_reason
        self._warrior_squeeze_target_wins = self._warrior_watch.target_wins
        self._warrior_squeeze_last_target_at = self._warrior_watch.last_target_at
        self._warrior_squeeze_failed_burst = self._warrior_watch.failed_burst
        self._warrior_squeeze_failed_burst_high = self._warrior_watch.failed_burst_high
        self._warrior_squeeze_post_target_reclaim_allowed = (
            self._warrior_watch.post_target_reclaim_allowed
        )
        self._warrior_squeeze_last_entry_trigger = self._warrior_watch.last_entry_trigger
        self._warrior_normal_fallback_rejects = self._warrior_watch.normal_fallback_rejects
        self._warrior_normal_fallback_last_reason = (
            self._warrior_watch.normal_fallback_last_reason
        )
        self._warrior_failed_momentum = self._warrior_watch.failed_momentum
        self._warrior_ignition_entries: Dict[str, int] = {}
        self._warrior_ignition_failed_entries: Dict[str, int] = {}
        self._warrior_ignition_peak_price: Dict[str, float] = {}
        self._warrior_ignition_peak_day_move: Dict[str, float] = {}
        self._warrior_ignition_trade_pnl: Dict[str, float] = {}
        self._trade_analyzer = None
        self._analysis_interval = 10  # run analysis every N cycles
        self._reconciler = PositionReconciler()
        self._hod_active: Dict[str, datetime] = {}
        self._hod_last_alert_at: Dict[str, datetime] = {}
        self._hod_tradeable_alert_at: Dict[str, datetime] = {}
        self._hod_watchlist_ttl_minutes = 5.0
        self._hod_watchlist_min_day_volume = 500_000
        self._hod_watchlist_min_rel_vol = 1.0
        self._hod_watchlist_min_bar_rvol = 1.2
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
        self._fast_scan_process_max = 80
        self._candidate_hydrate_thread: Optional[Thread] = None
        self._market_data = MarketDataService(
            candidate_queue_max=2_000,
            candidate_batch_max=10,
            hot_watch_max_symbols=40,
        )
        self._candidate_hydrate_queue: queue.PriorityQueue = self._market_data.candidate_queue
        self._candidate_hydrate_pending: Set[str] = set()
        self._candidate_hydrate_lock = Lock()
        self._candidate_hydrate_seq = 0
        self._candidate_hydrate_batch_max = 10
        self._deferred_fast_scan_movers: List[Dict] = []
        self._hot_watch_enabled = True
        self._hot_watch_ttl_minutes = 8.0
        self._hot_watch_strong_ttl_minutes = 15.0
        self._priority_bar_refresh: Set[str] = set()
        self._hot_watch_runner_ttl_minutes = 25.0
        self._hot_watch_max_symbols = 40
        self._hot_watch_min_change_pct = 5.0
        self._hot_watch_min_day_volume = 200_000
        self._hot_watch_sub5_min_day_volume = 500_000
        self._hot_watch_min_score = 0.30
        self._hot_watch_setup_refresh_enabled = True
        self._hot_watch_setup_refresh_max_pullback_pct = 4.0
        self._hot_watch_setup_refresh_min_recent_volume = 100_000.0

        # Watchlist bar refresh — keep 1m bars fresh for pattern scanners
        self._last_watchlist_bar_refresh: float = 0.0
        self._watchlist_bar_refresh_sec: float = 30.0

        # Capital-aware sizing: risk a % of live equity per trade, scaled to
        # account size and capped at buying power. Equity is cached (~60s) so we
        # don't hit the broker API on every entry.
        self._risk_pct_of_equity: float = 0.015
        self._max_dollar_risk_per_trade: float = 50.0
        self._max_position_pct_of_equity: float = 1.0
        self._min_risk_dollars: float = 5.0
        self._fallback_equity: float = 2000.0
        self._account_equity: float = 0.0
        self._account_equity_at: float = 0.0

        # Breakout scalp — instant entry on HOD tick alerts
        self._pending_breakout_scalps: deque = deque(maxlen=10)
        self._breakout_scalp_cooldown: Dict[str, float] = {}
        self._breakout_scalp_active: bool = False  # True if we have an open breakout scalp
        self._quick_scalp_spread_size_factors: Dict[str, float] = {}
        self._recent_quick_scalp_rejects: Dict[str, tuple[float, str]] = {}
        self._hod_seed_max_per_minute = 30
        self._hod_seed_minute_start = time.time()
        self._hod_seed_processed_this_minute = 0
        self._last_session_reset_day: Optional[str] = None
        self._nightly_analysis_day: Optional[str] = None

    def _market_data_service(self) -> MarketDataService:
        return _ensure_market_data_service(self)

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
            late_pullback_max_hod_pct=cfg.strategy.late_pullback_max_hod_pct,
            late_pullback_max_hod_other_pct=cfg.strategy.late_pullback_max_hod_other_pct,
            fresh_vwap_reclaim_scout_enabled=cfg.strategy.fresh_vwap_reclaim_scout_enabled,
            fresh_vwap_reclaim_scout_max_float=cfg.strategy.fresh_vwap_reclaim_scout_max_float,
            vwap_reclaim_scout_enabled=cfg.strategy.vwap_reclaim_scout_enabled,
            level_breakout_scout_enabled=cfg.strategy.level_breakout_scout_enabled,
            level_breakout_scout_min_session_move_pct=cfg.strategy.level_breakout_scout_min_session_move_pct,
            momentum_burst_live_enabled=cfg.strategy.momentum_burst_live_enabled,
            runner_trail_pct=cfg.strategy.runner_trail_pct,
            runner_min_confirm_pct=cfg.strategy.runner_min_confirm_pct,
            runner_trail_adaptive=cfg.strategy.runner_trail_adaptive,
            runner_trail_atr_mult=cfg.strategy.runner_trail_atr_mult,
            runner_trail_cap=cfg.strategy.runner_trail_cap,
            runner_give_room_after_partial=cfg.strategy.runner_give_room_after_partial,
            max_positions=cfg.max_positions,
            max_position_shares=cfg.max_position_shares,
            max_order_shares=cfg.max_order_shares,
            pattern_max_dollar_risk=cfg.max_dollar_risk_per_trade,
            min_avg_volume=vol_threshold,
            high_liquidity_volume=high_liq,
            portfolio=portfolio,
            float_checker=float_checker,
            enable_daily_loser_blacklist=cfg.enable_daily_loser_blacklist,
            daily_loser_blacklist_min_loss=cfg.daily_loser_blacklist_min_loss,
            daily_loser_blacklist_max_losses=cfg.daily_loser_blacklist_max_losses,
        )
        if cfg.enable_daily_loser_blacklist:
            logger.info("Daily loser blacklist: ON")
        else:
            logger.info("Daily loser blacklist: OFF (testing mode — re-entry allowed after losses)")
        pipeline.configure_missed_a_plus_chase_guard(
            window_sec=cfg.strategy.missed_a_plus_chase_window_sec,
            pct_sub5=cfg.strategy.missed_a_plus_chase_pct_sub5,
            pct_5plus=cfg.strategy.missed_a_plus_chase_pct_5plus,
            fresh_base_reset=cfg.strategy.missed_a_plus_fresh_base_reset,
            fresh_base_pct=cfg.strategy.missed_a_plus_fresh_base_pct,
        )
        pipeline.configure_entry_chase_guard(
            pct_low=cfg.strategy.entry_chase_pct_low,
            pct_high=cfg.strategy.entry_chase_pct_high,
            price_tier=cfg.strategy.entry_chase_price_tier,
        )
        pipeline._max_entry_risk_pct = float(cfg.strategy.max_entry_risk_pct)

        # replace the PaperBroker in the pipeline with the real AlpacaBroker
        pipeline._broker = broker  # type: ignore[assignment]
        # Wire slippage guard into broker for smart limit pricing
        broker._slippage_guard = pipeline.trade_guard.slippage

        scanner_min_price = (
            cfg.hod_sub2_momentum_min_price
            if cfg.hod_sub2_momentum_enabled
            else cfg.hod_momentum_min_price
        )
        # Part B of volume-surge discovery: when enabled, broaden the PRE-MARKET
        # candidate pool by lowering the minute-volume gate, so sleepy small-caps
        # (SDOT-type) are pre-loaded and the surge detector can watch them BEFORE
        # they explode. Off = the normal 10k gate (production unchanged).
        _surge_disc = os.environ.get("DAYTRADING_VOLUME_SURGE_DISCOVERY", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        scanner = WatchlistScanner(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            min_price=scanner_min_price,
            max_price=cfg.hod_momentum_max_price,
            min_volume=10_000,
            premarket_min_volume=1_000 if _surge_disc else 10_000,
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
        runner._max_watchlist = max(1, cfg.max_watchlist_symbols)
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
        runner._hot_watch_enabled = cfg.hot_watch_enabled
        runner._hot_watch_ttl_minutes = cfg.hot_watch_ttl_minutes
        runner._hot_watch_strong_ttl_minutes = cfg.hot_watch_strong_ttl_minutes
        runner._hot_watch_runner_ttl_minutes = cfg.hot_watch_runner_ttl_minutes
        runner._hot_watch_max_symbols = cfg.hot_watch_max_symbols
        runner._hot_watch_min_change_pct = cfg.hot_watch_min_change_pct
        runner._hot_watch_min_day_volume = cfg.hot_watch_min_day_volume
        runner._hot_watch_sub5_min_day_volume = cfg.hot_watch_sub5_min_day_volume
        runner._hot_watch_min_score = cfg.hot_watch_min_score
        runner._hot_watch_setup_refresh_enabled = cfg.strategy.hot_watch_setup_refresh_enabled
        runner._hot_watch_setup_refresh_max_pullback_pct = cfg.strategy.hot_watch_setup_refresh_max_pullback_pct
        runner._hot_watch_setup_refresh_min_recent_volume = cfg.strategy.hot_watch_setup_refresh_min_recent_volume
        runner._timed_entry_anchor_ttl_sec = cfg.strategy.timed_entry_anchor_ttl_sec
        runner._momentum_breakout_enabled = bool(cfg.strategy.momentum_breakout_enabled)
        runner._momentum_breakout_min_rvol = float(cfg.strategy.momentum_breakout_min_rvol)
        runner._momentum_breakout_min_day_volume = float(cfg.strategy.momentum_breakout_min_day_volume)
        runner._momentum_breakout_max_bar_range_pct = float(cfg.strategy.momentum_breakout_max_bar_range_pct)
        runner._momentum_breakout_score_floor = float(cfg.strategy.momentum_breakout_score_floor)
        runner._momentum_burst_cycle_enabled = bool(cfg.strategy.momentum_burst_cycle_enabled)
        runner._momentum_burst_window_sec = float(cfg.strategy.momentum_burst_window_sec)
        runner._momentum_burst_scalp_cooldown_sec = float(cfg.strategy.momentum_burst_scalp_cooldown_sec)
        runner._momentum_burst_hit_run_enabled = bool(cfg.strategy.momentum_burst_hit_run_enabled)
        runner._risk_pct_of_equity = float(cfg.strategy.risk_pct_of_equity)
        runner._max_dollar_risk_per_trade = float(cfg.max_dollar_risk_per_trade)
        runner._max_position_pct_of_equity = float(cfg.strategy.max_position_pct_of_equity)
        runner._min_risk_dollars = float(cfg.strategy.min_risk_dollars)
        runner._fallback_equity = float(cfg.strategy.fallback_equity)
        runner._momentum_burst_hit_run_max_entries = int(cfg.strategy.momentum_burst_hit_run_max_entries)
        runner._momentum_burst_hit_run_win_cooldown_sec = float(cfg.strategy.momentum_burst_hit_run_win_cooldown_sec)
        runner._momentum_burst_hit_run_loss_cooldown_sec = float(cfg.strategy.momentum_burst_hit_run_loss_cooldown_sec)
        runner._momentum_burst_hit_run_max_hold_sec = float(cfg.strategy.momentum_burst_hit_run_max_hold_sec)
        runner._momentum_burst_hit_run_reward_risk = float(cfg.strategy.momentum_burst_hit_run_reward_risk)
        runner._momentum_burst_hit_run_end_et = str(cfg.strategy.momentum_burst_hit_run_end_et or "")
        runner._momentum_burst_hit_run_stop_after_giveback = bool(
            cfg.strategy.momentum_burst_hit_run_stop_after_giveback
        )
        runner._momentum_burst_hit_run_max_giveback = float(
            cfg.strategy.momentum_burst_hit_run_max_giveback
        )
        runner._momentum_burst_hit_run_daily_loss_stop = float(
            cfg.strategy.momentum_burst_hit_run_daily_loss_stop
        )
        runner._warrior_squeeze_enabled = bool(cfg.strategy.warrior_squeeze_enabled)
        runner._warrior_squeeze_min_reclaim_price = float(
            cfg.strategy.warrior_squeeze_min_reclaim_price
        )
        runner._warrior_squeeze_starter_size_factor = float(
            cfg.strategy.warrior_squeeze_starter_size_factor
        )
        runner._warrior_squeeze_position_value = float(
            cfg.strategy.warrior_squeeze_position_value
        )
        runner._warrior_squeeze_max_dollar_risk = float(
            cfg.strategy.warrior_squeeze_max_dollar_risk
        )
        runner._warrior_squeeze_max_entries = int(cfg.strategy.warrior_squeeze_max_entries)
        runner._warrior_max_concurrent_trades = int(
            cfg.strategy.warrior_max_concurrent_trades
        )
        runner._warrior_watch_capacity = int(cfg.strategy.warrior_watch_capacity)
        runner._warrior_watch_until_premarket_end = bool(
            cfg.strategy.warrior_watch_until_premarket_end
        )
        runner._warrior_engine.risk.max_concurrent_warrior_trades = max(
            1,
            runner._warrior_max_concurrent_trades,
        )
        runner._warrior_squeeze_win_cooldown_sec = float(
            cfg.strategy.warrior_squeeze_win_cooldown_sec
        )
        runner._warrior_squeeze_reward_risk = float(cfg.strategy.warrior_squeeze_reward_risk)
        runner._warrior_squeeze_add_reward_risk = float(
            cfg.strategy.warrior_squeeze_add_reward_risk
        )
        runner._exec_timer._tick_entry_enabled = bool(cfg.strategy.tick_entry_enabled)
        runner._exec_timer._tick_entry_confirm_count = max(1, int(cfg.strategy.tick_entry_confirm_count))
        runner._exec_timer._tick_entry_max_above_anchor = float(cfg.strategy.tick_entry_max_above_anchor)
        runner._fast_scan_process_max = max(1, cfg.fast_scan_process_max)
        _ensure_market_data_service(runner).configure(
            candidate_queue_max=max(1, cfg.candidate_hydrate_queue_max),
            candidate_batch_max=max(1, cfg.candidate_hydrate_batch_max),
            hot_watch_max_symbols=max(1, cfg.hot_watch_max_symbols),
        )
        runner._candidate_hydrate_queue = _ensure_market_data_service(runner).candidate_queue
        runner._candidate_hydrate_batch_max = max(1, cfg.candidate_hydrate_batch_max)
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
        if cfg.hot_watch_enabled:
            logger.info(
                "HOD + hot-watch mode: HOD alert board plus structured early pullback watch",
            )
        else:
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
        split_sanity_cooldown_sec: float = 3600.0,
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
        now_mono = time.time()
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
                    if shares >= 1_000_000_000.0:
                        continue
                    last_check = _FLOAT_SPLIT_SANITY_LAST_CHECK.get(sym)
                    if (
                        last_check is not None
                        and now_mono - last_check < split_sanity_cooldown_sec
                    ):
                        continue
                    _FLOAT_SPLIT_SANITY_LAST_CHECK[sym] = now_mono
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

    def _handle_market_phase_transition(
        self,
        previous_phase: str,
        current_phase: str,
        current_et: datetime,
    ) -> None:
        """Handle session transitions without blocking the main cycle loop."""
        if current_phase == "PRE-MARKET" and previous_phase != "PRE-MARKET":
            logger.info("PRE-MARKET — daily session reset + refreshing HOD bar pool")
            self._maybe_daily_session_reset(current_et, current_phase, force=True)
            self._request_pool_refresh_now()
            return

        if current_phase == "OPEN" and previous_phase != "OPEN":
            logger.info("Market OPEN — refreshing HOD bar pool")
            self._request_pool_refresh_now()
            return

        if (
            getattr(self, "_after_hours_enabled", False)
            and current_phase == "AFTER-HOURS"
            and previous_phase != "AFTER-HOURS"
        ):
            logger.info("AFTER-HOURS — refreshing HOD bar pool")
            self._request_pool_refresh_now()

    def _sync_tick_tracker_pool(self) -> None:
        """Update tick tracker with current pool + watchlist symbols."""
        if not self._hod_tick_tracker:
            return
        self._prune_hot_watch()
        tracked = set(self._hod_bar_pool) | set(self._watchlist) | _ensure_market_data_service(self).hot_watch_keys()
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

    def _start_candidate_hydration_worker(self) -> None:
        if (
            self._candidate_hydrate_thread
            and self._candidate_hydrate_thread.is_alive()
        ):
            return
        service = _ensure_market_data_service(self)
        self._candidate_hydrate_thread = Thread(
            target=service.run_candidate_hydration_worker,
            kwargs={
                "stop_requested": lambda: self._shutdown,
                "pause_state": self._candidate_hydration_pause_state,
                "process_batch": self._process_candidate_hydration_batch,
                "publish_status": self._publish_candidate_hydration_status,
            },
            daemon=True,
            name="candidate-hydration",
        )
        self._candidate_hydrate_thread.start()

    def _candidate_hydration_priority(self, mover: Dict) -> int:
        sym = str(mover.get("symbol", "")).upper()
        if sym in self._hod_active or sym in self._watchlist_set:
            return 0
        volume = float(mover.get("volume", 0.0) or 0.0)
        abs_change = float(
            mover.get("abs_change_pct", mover.get("change_pct", 0.0)) or 0.0
        )
        if abs_change >= 20.0 and volume >= 500_000:
            return 1
        if abs_change >= 10.0 and volume >= 200_000:
            return 2
        return 3

    def _enqueue_candidate_hydration(
        self,
        movers: Sequence[Dict],
        *,
        source: str = "fast scan",
    ) -> int:
        queued = 0
        for mover in movers:
            sym = str(mover.get("symbol", "")).upper().strip()
            if not sym:
                continue
            priority = self._candidate_hydration_priority(mover)
            if _ensure_market_data_service(self).enqueue_candidate(mover, priority=priority, source=source):
                queued += 1
            else:
                try:
                    self._hub.on_candidate_hydration(dropped=1)
                except Exception:
                    pass
                logger.debug("Candidate hydration skipped/dropped %s", sym)
        if queued:
            logger.debug(
                "Candidate hydration queued %d movers from %s", queued, source,
            )
            try:
                self._hub.on_candidate_hydration(
                    queued=queued,
                    pending=_ensure_market_data_service(self).candidate_pending_size(),
                    last_source=source,
                )
            except Exception:
                pass
        return queued

    def _pull_candidate_hydration_batch(self) -> List[Dict]:
        return _ensure_market_data_service(self).pull_candidate_batch(self._candidate_hydrate_batch_max)

    def _candidate_hydration_pause_state(self) -> tuple[bool, bool]:
        pending_entry = self._has_pending_timed_entry()
        return (
            pending_entry
            or self._pool_refresh_in_progress
            or self._bar_hydrate_paused(),
            pending_entry,
        )

    def _process_candidate_hydration_batch(self, batch: List[Dict]) -> int:
        try:
            loaded_count = self._handle_fast_scan_movers(batch, push_event=True)
            logger.debug(
                "Candidate hydration processed %d movers (loaded=%d)",
                len(batch), loaded_count,
            )
            return loaded_count
        except Exception as exc:
            logger.warning("Candidate hydration batch failed: %s", exc)
            return 0

    def _publish_candidate_hydration_status(self, **payload) -> None:
        try:
            self._hub.on_candidate_hydration(**payload)
        except Exception:
            pass

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

    def _volume_surge_detected(self, c: Dict[str, Any]) -> bool:
        """True if this symbol's volume just spiked vs its OWN trailing baseline.

        Uses the snapshot volume (current-minute volume in pre-market) tracked
        across fast-scan cycles. Fires on a >=5x jump over the symbol's recent
        median, with a meaningful absolute floor and price ticking up — so a
        sleepy stock that suddenly explodes (SDOT-type) is caught immediately
        instead of waiting for slow cumulative thresholds."""
        if not hasattr(self, "_surge_vol_hist"):
            self._surge_vol_hist: Dict[str, List[float]] = {}
        sym = str(c.get("symbol") or "")
        vol = float(c.get("volume") or 0.0)
        price = float(c.get("price") or 0.0)
        chg = float(c.get("change_pct") or 0.0)
        if not sym or vol <= 0 or not (1.0 <= price <= 20.0):
            return False
        hist = self._surge_vol_hist.setdefault(sym, [])
        prior = list(hist)
        hist.append(vol)
        if len(hist) > 6:
            hist.pop(0)
        if len(prior) < 3 or vol < 20_000 or chg <= 0:
            return False
        sprior = sorted(prior)
        baseline = sprior[len(sprior) // 2]   # median of prior samples
        return vol >= 5.0 * max(baseline, 1.0)

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
        surge_on = os.environ.get("DAYTRADING_VOLUME_SURGE_DISCOVERY", "").strip().lower() in (
            "1", "true", "yes", "on",
        ) and phase == "PRE-MARKET"
        new_movers = []
        for c in candidates:
            sym = c["symbol"]
            # Force-add strong movers even if seen before
            is_strong = c.get("abs_change_pct", 0) >= 10.0 and c.get("volume", 0) >= 200_000
            # Volume-surge fast-discovery: catch a sleepy stock the instant its
            # volume spikes vs its OWN baseline (SDOT-type early-premarket launch),
            # not after slow cumulative thresholds. Tag it so the promote path can
            # bypass the +5% change gate.
            surged = surge_on and self._volume_surge_detected(c)
            if surged:
                c["surge"] = True
                if sym not in current_pool:
                    new_movers.append(c)
                    continue
            hot_watch_candidate = (
                not _ensure_market_data_service(self).hot_watch_contains(sym)
                and self._hot_watch_reject_reason(c, None) is None
            )
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
                if hot_watch_candidate or (is_strong and (not bars or bars_stale)):
                    new_movers.append(c)
                continue
            if phase == "PRE-MARKET" and not is_strong and not hot_watch_candidate:
                # Premarket scanner snapshots can include many baseline-only
                # candidates with stale/flat change. Track them for surge history,
                # but do not enqueue them for hydration unless they actually surge
                # or independently qualify as a real mover/hot-watch setup.
                continue
            if sym in self._fast_scan_known and not is_strong and not hot_watch_candidate:
                continue
            new_movers.append(c)

        if not new_movers:
            logger.debug("Fast scan: no new movers (%.1fs)", time.time() - t0)
            return

        total_new = len(new_movers)
        process_movers = new_movers[: self._fast_scan_process_max]
        for c in new_movers:
            self._fast_scan_known.add(c["symbol"])

        try:
            self._event_queue.put_nowait(FastScanEvent(process_movers))
        except queue.Full:
            pass

        logger.info(
            "Fast scan: %d new movers in %.1fs, processing top %d — %s",
            total_new,
            time.time() - t0,
            len(process_movers),
            ", ".join(c["symbol"] for c in process_movers[:10]),
        )
        self._hub.add_log(
            "INFO",
            "Fast scan: {} new movers, processing top {} — {}".format(
                total_new,
                len(process_movers),
                ", ".join(c["symbol"] for c in process_movers[:10]),
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
            bypass_checker=self.is_hot_watch_entry_allowed,
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
            if self._is_tradeable_hod_watchlist_alert(row):
                prev_tradeable = self._hod_tradeable_alert_at.get(sym)
                if prev_tradeable is None or alert_ts > prev_tradeable:
                    self._hod_tradeable_alert_at[sym] = alert_ts
            else:
                logger.debug(
                    "HOD watchlist watch-only %s: day_vol=%.0f rel_vol=%.1fx bar_rvol=%.1fx",
                    sym,
                    float(row.get("day_volume") or 0.0),
                    float(row.get("rel_vol") or 0.0),
                    float(row.get("bar_rvol") or 0.0),
                )
            alert_name = row.get("alert_name", "")
            if alert_name in gate_alerts:
                prev_active = self._hod_active.get(sym)
                if prev_active is None or alert_ts > prev_active:
                    self._hod_active[sym] = alert_ts
            self._maybe_arm_warrior_from_hod_alert(row)
        self._hub.on_hod_momentum_alerts(alerts)
        self._sync_watchlist_to_hod_alerts()

    def _maybe_arm_warrior_from_hod_alert(self, row: dict) -> None:
        """Put extreme HOD/mover alerts into Warrior watch-only monitoring.

        This is not an entry promotion. It only keeps the symbol in the Warrior
        watch loop so existing 10s pullback/reclaim lanes can evaluate it. The
        PLSM class of miss was detected by HOD/Hot Watch but never reached
        Warrior state because no scanner arming event survived after hydration.
        """
        if not getattr(self, "_warrior_squeeze_enabled", False):
            return
        sym = str(row.get("symbol") or "").upper().strip()
        if not sym:
            return
        if sym in getattr(self, "_momentum_burst_hit_run_day_blocked", {}):
            return
        try:
            price = float(row.get("price") or 0.0)
            day_volume = float(row.get("day_volume") or 0.0)
            rel_vol = float(row.get("rel_vol") or 0.0)
            bar_rvol = float(row.get("bar_rvol") or 0.0)
            change_session_pct = float(row.get("change_session_pct") or 0.0)
            change_from_close_pct = float(row.get("change_from_close_pct") or 0.0)
            change_from_low_pct = float(row.get("change_from_low_pct") or 0.0)
            float_shares = float(row.get("float_shares") or 0.0)
        except (TypeError, ValueError):
            return

        if price <= 0:
            return
        effective_min_price = max(
            1.5,
            float(getattr(self, "_hod_sub2_min_price", 1.0) or 1.0),
        )
        max_price = float(getattr(self, "_hod_max_price", 20.0) or 20.0)
        if price < effective_min_price or price > max_price:
            return
        max_float = float(getattr(self, "_hod_max_float", 20_000_000) or 20_000_000)
        if float_shares > 0 and float_shares > max_float:
            return

        active_rvol = max(rel_vol, bar_rvol)
        momentum_pct = max(change_session_pct, change_from_close_pct, change_from_low_pct)
        alert_name = str(row.get("alert_name") or "").lower()
        is_squeeze_alert = "squeeze" in alert_name or "hod" in alert_name
        exceptional_tape_watch = (
            day_volume >= 200_000
            and active_rvol >= 8.0
            and momentum_pct >= 40.0
            and is_squeeze_alert
        )
        if day_volume < 1_000_000 and not exceptional_tape_watch:
            return
        if momentum_pct < 80.0 and active_rvol < 1.0 and not is_squeeze_alert:
            return

        candidate_high = price
        for key in ("session_high", "high"):
            try:
                candidate_high = max(candidate_high, float(row.get(key) or 0.0))
            except (TypeError, ValueError):
                pass
        if not self._ensure_warrior_watch_capacity(sym, candidate_high):
            return

        now_mono = time.monotonic()
        self._momentum_burst_armed.setdefault(sym, now_mono)
        self._momentum_burst_window_high[sym] = max(
            candidate_high,
            float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0),
        )
        self._momentum_burst_session_anchor_high.setdefault(sym, candidate_high)
        logger.info(
            "WARRIOR SQUEEZE watch-only armed %s from HOD alert price=$%.4f "
            "day_vol=%.0f move=%.1f%%",
            sym,
            price,
            day_volume,
            momentum_pct,
        )

    def _is_tradeable_hod_watchlist_alert(self, row: dict) -> bool:
        """Return True when a HOD alert is liquid enough for active routing.

        The HOD board can show early movers with modest volume, but the active
        trading watchlist should avoid names that will almost certainly fail the
        initial entry guard for weak tape/liquidity.
        """
        try:
            price = float(row.get("price") or 0.0)
            day_volume = float(row.get("day_volume") or 0.0)
            rel_vol = float(row.get("rel_vol") or 0.0)
            bar_rvol = float(row.get("bar_rvol") or 0.0)
        except (TypeError, ValueError):
            return False

        effective_min_price = self._hod_min_price
        if self._hod_sub2_enabled:
            effective_min_price = max(1.5, self._hod_sub2_min_price)
        if price < effective_min_price or price > self._hod_max_price:
            return False
        if day_volume >= 1_000_000:
            return True
        if day_volume < self._hod_watchlist_min_day_volume:
            return False
        return (
            rel_vol >= self._hod_watchlist_min_rel_vol
            or bar_rvol >= self._hod_watchlist_min_bar_rvol
        )

    def is_hod_active(self, symbol: str) -> bool:
        ts = self._hod_active.get(symbol)
        if ts is None:
            return False
        elapsed = (datetime.now(timezone.utc) - ts).total_seconds()
        return elapsed < self._hod_alert_ttl_minutes * 60

    def _prune_hot_watch(self) -> None:
        service = _ensure_market_data_service(self)
        if service.hot_watch_len() <= 0:
            return
        now = datetime.now(timezone.utc)
        changed = False
        for sym, meta in service.hot_watch_items():
            added_at = meta.get("added_at")
            if not isinstance(added_at, datetime):
                service.hot_watch_delete(sym)
                changed = True
                continue
            ttl_minutes = float(meta.get("ttl_minutes", self._hot_watch_ttl_minutes))
            if (now - added_at).total_seconds() >= ttl_minutes * 60:
                refresh_reason = self._hot_watch_setup_refresh_reason(sym)
                if refresh_reason is not None:
                    meta["added_at"] = now
                    meta["last_seen"] = now
                    meta["reason"] = refresh_reason
                    service.hot_watch_set(sym, meta)
                    changed = True
                    self._journal.record("hot_watch", {
                        "symbol": sym,
                        "stage": "setup_refresh",
                        "reason": refresh_reason,
                        "age_seconds": round((now - added_at).total_seconds(), 1),
                        "mode": meta.get("mode", "watch"),
                        "ttl_minutes": ttl_minutes,
                    })
                    continue
                service.hot_watch_delete(sym)
                changed = True
                self._journal.record("hot_watch", {
                    "symbol": sym,
                    "stage": "expired",
                    "reason": "ttl expired",
                    "age_seconds": round((now - added_at).total_seconds(), 1),
                    "mode": meta.get("mode", "watch"),
                    "ttl_minutes": ttl_minutes,
                })
        if changed:
            self._publish_hot_watch()
            self._publish_trading_watchlist()

    def _hot_watch_setup_refresh_reason(self, symbol: str) -> Optional[str]:
        """Keep watched symbols alive while they are basing near HOD.

        A mover often becomes tradeable only after it stops extending and builds
        a tight base. Do not expire it during that setup if volume is still
        active, price is near session high, and VWAP support is intact.
        """
        if not getattr(self, "_hot_watch_setup_refresh_enabled", True):
            return None
        bars = list(self._bar_buffer.get(symbol, []))
        if len(bars) < 6:
            return None
        latest = bars[-1]
        price = float(getattr(latest, "close", 0.0) or 0.0)
        if price <= 0:
            return None
        session_high = max(float(getattr(b, "high", 0.0) or 0.0) for b in bars)
        if session_high <= 0:
            return None
        pullback_pct = (session_high - price) / session_high * 100.0
        if pullback_pct > float(self._hot_watch_setup_refresh_max_pullback_pct):
            return None

        recent = bars[-3:]
        recent_volume = sum(float(getattr(b, "volume", 0.0) or 0.0) for b in recent)
        if recent_volume < float(self._hot_watch_setup_refresh_min_recent_volume):
            return None

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap > 0 and price < current_vwap * 0.995:
            return None

        return (
            "setup refresh: holding {:.1f}% below HOD with {:.0f} recent volume"
        ).format(pullback_pct, recent_volume)

    def _hot_watch_snapshot(self) -> List[dict]:
        now = datetime.now(timezone.utc)
        rows: List[dict] = []
        for sym, meta in _ensure_market_data_service(self).hot_watch_items():
            mover = meta.get("mover", {}) if isinstance(meta.get("mover"), dict) else {}
            live = self._hot_watch_live_metrics(sym)
            added_at = meta.get("added_at")
            ttl_minutes = float(meta.get("ttl_minutes", self._hot_watch_ttl_minutes))
            remaining = 0.0
            if isinstance(added_at, datetime):
                remaining = max(0.0, ttl_minutes * 60 - (now - added_at).total_seconds())
            rows.append({
                "symbol": sym,
                "mode": meta.get("mode", "watch"),
                "ttl_minutes": ttl_minutes,
                "remaining_seconds": round(remaining, 1),
                "price": mover.get("price"),
                "change_pct": mover.get("change_pct"),
                "abs_change_pct": mover.get("abs_change_pct"),
                "volume": mover.get("volume"),
                "score": mover.get("score"),
                "float": meta.get("float"),
                "short_change_pct": live.get("short_change_pct"),
                "pullback_from_high_pct": live.get("pullback_from_high_pct"),
                "session_high": live.get("session_high"),
                "reason": meta.get("reason", ""),
                "added_at": added_at.isoformat() if isinstance(added_at, datetime) else "",
                "last_seen": (
                    meta.get("last_seen").isoformat()
                    if isinstance(meta.get("last_seen"), datetime)
                    else ""
                ),
            })
        mode_rank = {"runner_watch": 0, "strong_watch": 1, "watch": 2}
        rows.sort(key=lambda r: (
            mode_rank.get(str(r.get("mode")), 9),
            -float(r.get("score") or 0.0),
            str(r.get("symbol") or ""),
        ))
        return rows

    def _hot_watch_live_metrics(self, symbol: str) -> Dict[str, Optional[float]]:
        bars = list(self._bar_buffer.get(symbol, []))
        if not bars:
            return {
                "short_change_pct": None,
                "pullback_from_high_pct": None,
                "session_high": None,
            }

        latest = bars[-1]
        price = float(getattr(latest, "close", 0.0) or 0.0)
        lookback = bars[-6] if len(bars) >= 6 else bars[0]
        lookback_close = float(getattr(lookback, "close", 0.0) or 0.0)
        short_change = None
        if price > 0 and lookback_close > 0:
            short_change = (price - lookback_close) / lookback_close * 100.0

        session_high = max(float(getattr(b, "high", 0.0) or 0.0) for b in bars)
        pullback = None
        if price > 0 and session_high > 0:
            pullback = (price - session_high) / session_high * 100.0

        return {
            "short_change_pct": round(short_change, 2) if short_change is not None else None,
            "pullback_from_high_pct": round(pullback, 2) if pullback is not None else None,
            "session_high": round(session_high, 4) if session_high else None,
        }

    def _warrior_watch_snapshot(self) -> List[dict]:
        """Dashboard view of current Warrior playbook state.

        This is intentionally read-only telemetry.  The strategy keeps using the
        WarriorWatchBook dictionaries as before; the dashboard only receives a
        compact explanation of what each symbol is waiting on.
        """
        watch = getattr(self, "_warrior_watch", None)
        if watch is None:
            return []
        symbols = set()
        for mapping_name in (
            "armed",
            "pending",
            "window_high",
            "session_anchor_high",
            "rejection_high",
            "rejection_reason",
            "target_wins",
            "failed_burst",
            "failed_burst_high",
            "hit_run_counts",
            "hit_run_block_until",
            "symbol_pnl",
            "symbol_peak_pnl",
            "day_blocked",
            "normal_fallback_rejects",
            "normal_fallback_last_reason",
            "last_entry_trigger",
        ):
            mapping = getattr(watch, mapping_name, {})
            if isinstance(mapping, dict):
                symbols.update(str(s).upper() for s in mapping.keys() if s)
        symbols.update(_ensure_market_data_service(self).hot_watch_keys())
        symbols.update(str(s).upper() for s in self._trade_symbol_set())

        now_mono = time.monotonic()
        rows: List[dict] = []
        for sym in sorted(symbols):
            pending = watch.pending.get(sym, {}) if isinstance(watch.pending.get(sym), dict) else {}
            armed_at = watch.armed.get(sym)
            block_until = watch.hit_run_block_until.get(sym)
            block_seconds = 0.0
            if block_until is not None:
                try:
                    block_seconds = max(0.0, float(block_until) - now_mono)
                except (TypeError, ValueError):
                    block_seconds = 0.0

            if sym in watch.day_blocked:
                state = "day_blocked"
            elif block_seconds > 0:
                state = "cooldown"
            elif pending:
                state = "pending"
            elif sym in watch.armed:
                state = "watching"
            elif sym in watch.failed_burst:
                state = "failed_burst"
            elif sym in watch.rejection_high:
                state = "proof_wait"
            else:
                state = "candidate"

            reason = (
                watch.day_blocked.get(sym)
                or watch.failed_burst.get(sym)
                or watch.rejection_reason.get(sym)
                or watch.normal_fallback_last_reason.get(sym)
                or ""
            )
            armed_for = 0.0
            if armed_at is not None:
                try:
                    armed_for = max(0.0, now_mono - float(armed_at))
                except (TypeError, ValueError):
                    armed_for = 0.0
            rows.append({
                "symbol": sym,
                "state": state,
                "pending_trigger": (
                    pending.get("entry_trigger")
                    or pending.get("scanner_name")
                    or watch.last_entry_trigger.get(sym, "")
                    or ""
                ),
                "window_high": round(float(watch.window_high.get(sym, 0.0) or 0.0), 4),
                "session_anchor_high": round(
                    float(watch.session_anchor_high.get(sym, 0.0) or 0.0),
                    4,
                ),
                "rejection_high": round(float(watch.rejection_high.get(sym, 0.0) or 0.0), 4),
                "failed_burst_high": round(
                    float(watch.failed_burst_high.get(sym, 0.0) or 0.0),
                    4,
                ),
                "target_wins": int(watch.target_wins.get(sym, 0) or 0),
                "entries": int(watch.hit_run_counts.get(sym, 0) or 0),
                "pnl": round(float(watch.symbol_pnl.get(sym, 0.0) or 0.0), 2),
                "peak_pnl": round(float(watch.symbol_peak_pnl.get(sym, 0.0) or 0.0), 2),
                "fallback_rejects": int(watch.normal_fallback_rejects.get(sym, 0) or 0),
                "block_seconds": round(block_seconds, 1),
                "armed_for_seconds": round(armed_for, 1),
                "reason": str(reason or ""),
            })

        state_rank = {
            "pending": 0,
            "watching": 1,
            "cooldown": 2,
            "proof_wait": 3,
            "failed_burst": 4,
            "day_blocked": 5,
            "candidate": 6,
        }
        rows.sort(key=lambda r: (
            state_rank.get(str(r.get("state")), 9),
            -float(r.get("target_wins") or 0),
            -float(r.get("window_high") or 0.0),
            str(r.get("symbol") or ""),
        ))
        capacity = int(getattr(self, "_warrior_watch_capacity", 10) or 0)
        if capacity <= 0:
            return rows

        active_rows = [r for r in rows if str(r.get("state") or "") != "candidate"]
        candidate_rows = [r for r in rows if str(r.get("state") or "") == "candidate"]
        remaining_slots = max(0, capacity - len(active_rows))
        return active_rows + candidate_rows[:remaining_slots]

    def _publish_hot_watch(self) -> None:
        self._hub.on_hot_watch(self._hot_watch_snapshot())
        self._publish_warrior_watch()

    def _publish_warrior_watch(self) -> None:
        handler = getattr(self._hub, "on_warrior_watch", None)
        if handler is not None:
            handler(self._warrior_watch_snapshot())

    def is_hot_watch_active(self, symbol: str) -> bool:
        self._prune_hot_watch()
        return _ensure_market_data_service(self).hot_watch_contains(symbol)

    def is_hot_watch_entry_allowed(self, signal: TradeSignal) -> bool:
        """Allow HOD-gate bypass only for structured hot-watch setups."""
        sym = signal.symbol
        if not self.is_hot_watch_active(sym):
            return False
        scanner = ""
        pattern = ""
        if signal.scan_result is not None:
            scanner = signal.scan_result.scanner_name or ""
            pattern = str(signal.scan_result.criteria.get("pattern", "") or "")
        allowed = scanner in HOT_WATCH_PATTERNS or pattern in HOT_WATCH_PATTERNS
        self._journal.record("hot_watch", {
            "symbol": sym,
            "stage": "entry_gate",
            "allowed": allowed,
            "scanner": scanner,
            "pattern": pattern,
            "reason": "structured pattern" if allowed else "pattern not hot-watch eligible",
        })
        return allowed

    def _hot_watch_reject_reason(self, mover: Dict, flt: Optional[float]) -> Optional[str]:
        if not self._hot_watch_enabled:
            return "hot watch disabled"

        sym = mover.get("symbol")
        if not sym:
            return "missing symbol"
        if "." in sym:
            return "warrant/unit symbol"
        if sym in self._watchlist_pinned:
            return "pinned symbol"

        price = float(mover.get("price", 0.0) or 0.0)
        if price < self._hod_sub2_min_price or price > self._hod_max_price:
            return "price outside hot-watch band"

        change = abs(float(mover.get("abs_change_pct", mover.get("change_pct", 0.0)) or 0.0))
        early_level_scout = self._hot_watch_level_breakout_scout_candidate(mover)
        # A detected volume surge promotes on the spike alone — don't wait for the
        # full +5% change gate (catches the launch a leg earlier).
        volume_surge = bool(mover.get("surge"))
        if change < self._hot_watch_min_change_pct and not early_level_scout and not volume_surge:
            return "change {:.1f}% < {:.1f}%".format(
                change, self._hot_watch_min_change_pct,
            )

        volume = float(mover.get("volume", 0.0) or 0.0)
        min_volume = self._hot_watch_min_day_volume
        if price < 5.0:
            min_volume = self._hot_watch_sub5_min_day_volume
        if self._market_phase() == "PRE-MARKET":
            # The fast scanner uses current/recent minute volume before the
            # open so stale prior-day volume cannot create fake hot movers.
            # Treat this as tape volume, not full day volume.
            min_volume = 50_000 if price < 5.0 else 40_000
        if volume < min_volume and not early_level_scout:
            return "volume {:.0f} < {:.0f}".format(volume, min_volume)

        score = float(mover.get("score", 0.0) or 0.0)
        if score < self._hot_watch_min_score and not early_level_scout:
            return "score {:.2f} < {:.2f}".format(score, self._hot_watch_min_score)

        if flt is not None and flt > self._hod_max_float:
            return "float {:.1f}M > {:.1f}M".format(
                flt / 1_000_000, self._hod_max_float / 1_000_000,
            )

        return None

    def _hot_watch_level_breakout_scout_candidate(self, mover: Dict) -> bool:
        """Hydrate smooth/liquid early level-break candidates before +5%.

        Snapshot data cannot prove the chart pattern yet, but high day volume,
        decent score, and a 3%+ move are enough to start bar/tick tracking. The
        actual order still needs the level_breakout_scout scanner, verifier,
        final guard, timer, spread, and anti-chase checks.
        """
        try:
            price = float(mover.get("price", 0.0) or 0.0)
            change = abs(float(mover.get("abs_change_pct", mover.get("change_pct", 0.0)) or 0.0))
            volume = float(mover.get("volume", 0.0) or 0.0)
            score = float(mover.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        if not (self._hod_sub2_min_price <= price <= self._hod_max_price):
            return False
        min_change = 3.0
        min_volume = 5_000_000.0 if price >= 5.0 else 2_000_000.0
        min_score = max(self._hot_watch_min_score, 0.40)
        return change >= min_change and volume >= min_volume and score >= min_score

    def _hot_watch_mode(self, mover: Dict) -> tuple[str, float]:
        """Return watch mode and TTL from current mover strength."""
        price = float(mover.get("price", 0.0) or 0.0)
        change = abs(float(mover.get("abs_change_pct", mover.get("change_pct", 0.0)) or 0.0))
        volume = float(mover.get("volume", 0.0) or 0.0)
        score = float(mover.get("score", 0.0) or 0.0)

        if (
            change >= 25.0
            and volume >= max(self._hot_watch_sub5_min_day_volume, 1_000_000)
            and score >= max(self._hot_watch_min_score, 0.35)
        ):
            return "runner_watch", self._hot_watch_runner_ttl_minutes

        strong_volume = 1_000_000 if price < 5.0 else 500_000
        if (
            change >= 10.0
            and volume >= strong_volume
            and score >= self._hot_watch_min_score
        ):
            return "strong_watch", self._hot_watch_strong_ttl_minutes

        return "watch", self._hot_watch_ttl_minutes

    def _promote_hot_watch(
        self,
        mover: Dict,
        *,
        flt: Optional[float],
        reason: str,
    ) -> None:
        sym = mover.get("symbol")
        if not sym:
            return
        self._prune_hot_watch()
        service = _ensure_market_data_service(self)
        already_active = service.hot_watch_contains(sym)
        if not already_active and service.hot_watch_len() >= self._hot_watch_max_symbols:
            oldest = service.hot_watch_oldest_symbol()
            if oldest:
                service.hot_watch_delete(oldest)
            if oldest:
                self._journal.record("hot_watch", {
                    "symbol": oldest,
                    "stage": "removed",
                    "reason": "max hot-watch symbols",
                })

        now = datetime.now(timezone.utc)
        already_active = service.hot_watch_contains(sym)
        mode, ttl_minutes = self._hot_watch_mode(mover)
        prior = service.hot_watch_get(sym, {})
        prior_added = prior.get("added_at", now)
        prior_ttl = float(prior.get("ttl_minutes", ttl_minutes))
        prior_mode = str(prior.get("mode", mode))
        if already_active and ttl_minutes > prior_ttl:
            prior_added = now
        elif already_active:
            if mode in {"strong_watch", "runner_watch"} and ttl_minutes >= prior_ttl:
                prior_added = now
            else:
                mode = prior_mode
            ttl_minutes = max(ttl_minutes, prior_ttl)

        service.hot_watch_set(sym, {
            "added_at": prior_added,
            "last_seen": now,
            "mover": dict(mover),
            "float": flt,
            "reason": reason,
            "mode": mode,
            "ttl_minutes": ttl_minutes,
        })
        self._maybe_arm_warrior_from_hot_watch_mover(
            mover,
            flt=flt,
            mode=mode,
        )
        self._ensure_streaming_symbols([sym])
        self._enqueue_hod_seed_symbols([sym])
        self._journal.record("hot_watch", {
            "symbol": sym,
            "stage": "promoted" if not already_active else "refreshed",
            "reason": reason,
            "price": mover.get("price"),
            "change_pct": mover.get("change_pct"),
            "abs_change_pct": mover.get("abs_change_pct"),
            "volume": mover.get("volume"),
            "score": mover.get("score"),
            "float": flt,
            "mode": mode,
            "ttl_minutes": ttl_minutes,
        })
        if not already_active:
            logger.info(
                "HOT WATCH +%s [%s %.0fm]: %s chg=%.1f%% vol=%.0f score=%.2f",
                sym,
                mode,
                ttl_minutes,
                reason,
                float(mover.get("abs_change_pct", mover.get("change_pct", 0.0)) or 0.0),
                float(mover.get("volume", 0.0) or 0.0),
                float(mover.get("score", 0.0) or 0.0),
            )
            self._hub.add_log(
                "INFO",
                "Hot watch +{} [{} {:.0f}m]: {}".format(
                    sym, mode, ttl_minutes, reason,
                ),
            )
        self._publish_hot_watch()
        self._publish_trading_watchlist()
        return

    def _maybe_arm_warrior_from_hot_watch_mover(
        self,
        mover: Dict,
        *,
        flt: Optional[float],
        mode: str,
    ) -> None:
        """Put strong fast-scan movers into Warrior watch-only monitoring.

        Hot Watch premarket volume is recent tape volume, not full day volume,
        so it cannot reuse the HOD-alert day-volume gate. This closes the PLSM
        path where fast scan finds the runner but no current HOD alert row
        exists to arm Warrior.
        """
        if not getattr(self, "_warrior_squeeze_enabled", False):
            return
        sym = str(mover.get("symbol") or "").upper().strip()
        if not sym:
            return
        if sym in getattr(self, "_momentum_burst_hit_run_day_blocked", {}):
            return
        try:
            price = float(mover.get("price") or 0.0)
            volume = float(mover.get("volume") or 0.0)
            score = float(mover.get("score") or 0.0)
            change_pct = abs(
                float(mover.get("abs_change_pct", mover.get("change_pct", 0.0)) or 0.0)
            )
            short_change_pct = abs(float(mover.get("short_change_pct") or 0.0))
            session_high = float(mover.get("session_high") or 0.0)
        except (TypeError, ValueError):
            return

        effective_min_price = max(
            1.5,
            float(getattr(self, "_hod_sub2_min_price", 1.0) or 1.0),
        )
        max_price = float(getattr(self, "_hod_max_price", 20.0) or 20.0)
        if price < effective_min_price or price > max_price:
            return
        max_float = float(getattr(self, "_hod_max_float", 20_000_000) or 20_000_000)
        if flt is not None and flt > max_float:
            return

        min_recent_volume = 50_000.0 if price < 5.0 else 40_000.0
        strong_mover = (
            mode in {"strong_watch", "runner_watch"}
            or change_pct >= 80.0
            or short_change_pct >= 10.0
        )
        if not strong_mover:
            return
        if volume < min_recent_volume:
            return
        if score < max(0.30, float(getattr(self, "_hot_watch_min_score", 0.30) or 0.30)):
            return

        candidate_high = max(price, session_high)
        if candidate_high <= 0:
            return
        if not self._ensure_warrior_watch_capacity(sym, candidate_high):
            return
        now_mono = time.monotonic()
        self._momentum_burst_armed.setdefault(sym, now_mono)
        self._momentum_burst_window_high[sym] = max(
            candidate_high,
            float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0),
        )
        self._momentum_burst_session_anchor_high.setdefault(sym, candidate_high)
        logger.info(
            "WARRIOR SQUEEZE watch-only armed %s from hot watch price=$%.4f "
            "vol=%.0f change=%.1f%% mode=%s",
            sym,
            price,
            volume,
            change_pct,
            mode,
        )

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

    def _request_priority_bar_refresh(self, symbol: str) -> None:
        """Prioritize a hot/watchlist symbol that failed because bars were stale."""
        sym = symbol.upper().strip()
        if not sym or sym in self._watchlist_pinned:
            return
        if sym not in self._watchlist_set and not _ensure_market_data_service(self).hot_watch_contains(sym):
            return
        self._priority_bar_refresh.add(sym)

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

    def _record_bar_fetch_failures(
        self,
        failure_count: int,
        attempted_count: int = 0,
    ) -> None:
        if failure_count <= 0:
            return
        now = time.time()
        failure_ratio = (
            failure_count / attempted_count if attempted_count > 0 else 1.0
        )
        if attempted_count > 0 and attempted_count < 5:
            logger.debug(
                "Bar hydration small-batch miss: %d/%d symbols failed; no pause",
                failure_count, attempted_count,
            )
            return
        if attempted_count > 0 and failure_ratio < 0.65 and failure_count < 15:
            logger.debug(
                "Bar hydration partial miss: %d/%d symbols failed; no pause",
                failure_count, attempted_count,
            )
            return
        self._network_failure_times.append(now)
        cutoff = now - 60.0
        self._network_failure_times = [
            t for t in self._network_failure_times if t >= cutoff
        ]
        if len(self._network_failure_times) >= 3:
            self._hydrate_paused_until = now + 20.0
            self._network_failure_times.clear()
            logger.warning(
                "Network unstable - pausing bar hydration for 20s after "
                "repeated weak fetch batches (last failures=%d/%d; "
                "WebSocket stays connected)",
                failure_count,
                attempted_count,
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
        warrior_watch = {
            str(sym).upper()
            for sym in getattr(self, "_momentum_burst_armed", {}).keys()
            if sym
        }
        warrior_watch.update(
            str(sym).upper()
            for sym in getattr(self, "_momentum_burst_pending", {}).keys()
            if sym
        )
        return self._watchlist_pinned | open_syms | tracked | warrior_watch

    def _trade_symbol_set(self) -> Set[str]:
        """Symbols eligible for the trading pipeline this cycle."""
        if hasattr(self, "_prune_hot_watch"):
            self._prune_hot_watch()
        hot = _ensure_market_data_service(self).hot_watch_keys()
        return (
            set(self._watchlist)
            | hot
            | self._protected_watchlist_symbols()
        )

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
        tradeable_alerts = getattr(
            self,
            "_hod_tradeable_alert_at",
            self._hod_last_alert_at,
        )
        for sym, ts in list(tradeable_alerts.items()):
            if (now - ts).total_seconds() < ttl_secs:
                active.add(sym)
            else:
                del tradeable_alerts[sym]
                self._hod_last_alert_at.pop(sym, None)
        return active

    def _sync_watchlist_to_hod_alerts(self, alerts: Optional[List[dict]] = None) -> None:
        """Keep the trading watchlist aligned with recent HOD alerts (TTL)."""
        del alerts  # TTL tracked in _hod_last_alert_at via _on_hod_alerts_changed
        alert_syms = self._hod_watchlist_symbols()
        hot_syms = _ensure_market_data_service(self).hot_watch_keys()

        protected = self._protected_watchlist_symbols()
        target = alert_syms | protected | hot_syms
        if len(target) > self._max_watchlist:
            ordered = []
            seen: Set[str] = set()
            for sym in protected:
                if sym not in seen:
                    ordered.append(sym)
                    seen.add(sym)
            for sym in hot_syms:
                if sym not in seen:
                    ordered.append(sym)
                    seen.add(sym)
                if len(ordered) >= self._max_watchlist:
                    break
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
        active_symbols = sorted(self._trade_symbol_set())
        self._hub.on_trading_watchlist(
            active_symbols,
            pinned=sorted(self._watchlist_pinned),
        )
        self._publish_warrior_watch()

    def _publish_hod_alert_board(self) -> None:
        if self._hod_alert_store is not None:
            self._hub.on_hod_momentum_alerts(self._hod_alert_store.snapshot())

    def _ensure_streaming_symbols(self, symbols: Sequence[str]) -> None:
        """Queue live bar/quote/trade subscriptions for active symbols."""
        syms = [s.upper() for s in symbols if s]
        if not syms:
            return
        self._stream.subscribe(syms, bars=True, quotes=True)
        if self._hod_tick_tracker:
            self._hod_tick_tracker.add_known_symbols(syms)
            self._stream.add_trade_filter_symbols(syms)
            logger.info("10s tick feed requested for %d symbols — %s", len(syms), syms[:8])
        else:
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
        self._ensure_streaming_symbols(new_symbols)
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
            self._record_bar_fetch_failures(
                self._hist.last_fetch_failures, len(symbols),
            )
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
            self._record_bar_fetch_failures(
                self._hist.last_fetch_failures, len(symbols),
            )
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
        self._start_candidate_hydration_worker()
        self._start_fast_scan_worker()

        # Push initial account info and trade history to dashboard
        try:
            acct = self._broker.get_account()
            self._account_equity = float(acct.get("equity") or 0.0)
            self._account_equity_at = time.monotonic()
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

        self._publish_hot_watch()
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

                self._maybe_run_after_market_training(now_et, self._market_phase())

                if now_ts - last_market_check > 60:
                    last_market_check = now_ts
                    phase = self._market_phase()
                    if phase != last_market_phase:
                        self._handle_market_phase_transition(
                            last_market_phase, phase, now_et,
                        )
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

                wait_timeout = self._cycle_interval
                next_entry_timeout = self._exec_timer.seconds_until_next_timeout()
                if next_entry_timeout is not None:
                    wait_timeout = min(wait_timeout, max(0.25, next_entry_timeout))
                got_data = self._new_data.wait(timeout=wait_timeout)
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

                timed_entry_pending = self._has_pending_timed_entry()
                if (
                    in_trading_window
                    and not timed_entry_pending
                    and self._deferred_fast_scan_movers
                ):
                    movers = self._take_deferred_fast_scan_movers([])
                    self._enqueue_candidate_hydration(
                        movers,
                        source="deferred fast scan",
                    )

                # Process instant breakout scalps from HOD tick alerts
                if in_trading_window and not timed_entry_pending:
                    self._process_breakout_scalps()
                    self._process_momentum_burst_scalps()

                should_run = False
                if timed_entry_pending:
                    if got_data:
                        self._new_data.clear()
                elif got_data:
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
                            "new-entry pipeline paused by time window",
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
            self._cleanup_accidental_shorts(alpaca_positions)
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
            self._cleanup_accidental_shorts(alpaca_positions)
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

    def _cleanup_accidental_shorts(self, broker_positions: Dict[str, dict]) -> None:
        """Cover unexpected broker shorts in this long-only strategy."""
        if not getattr(self, "_accidental_short_cleanup_enabled", True):
            return
        if not broker_positions:
            return

        now_ts = time.time()
        max_qty = float(getattr(self, "_accidental_short_max_qty", 0.0) or 0.0)
        cooldown_sec = float(getattr(self, "_accidental_short_cooldown_sec", 30.0))
        cleanup_at = getattr(self, "_accidental_short_cleanup_at", None)
        if cleanup_at is None:
            cleanup_at = {}
            self._accidental_short_cleanup_at = cleanup_at

        for symbol, data in list(broker_positions.items()):
            try:
                qty = float(data.get("qty", 0) or 0)
            except (TypeError, ValueError):
                continue
            if qty >= 0:
                continue

            cover_qty = abs(qty)
            last_attempt = float(cleanup_at.get(symbol, 0.0))
            if now_ts - last_attempt < cooldown_sec:
                continue
            cleanup_at[symbol] = now_ts

            if max_qty > 0 and cover_qty > max_qty:
                msg = (
                    f"ACCIDENTAL SHORT {symbol}: broker shows {qty:.0f} shares; "
                    f"auto-cover skipped because max cleanup size is {max_qty:.0f}"
                )
                logger.error(msg)
                if hasattr(self, "_hub"):
                    self._hub.add_log("ERROR", msg)
                continue

            price = float(
                data.get("current_price")
                or data.get("avg_entry")
                or data.get("avg_entry_price")
                or 0.0
            )
            if price <= 0:
                msg = f"ACCIDENTAL SHORT {symbol}: no valid cover price for {cover_qty:.0f} shares"
                logger.error(msg)
                if hasattr(self, "_hub"):
                    self._hub.add_log("ERROR", msg)
                continue

            order = Order(symbol=symbol, side=Side.BUY, quantity=cover_qty)
            bar = Bar(
                symbol=symbol,
                ts=datetime.now(timezone.utc),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0,
            )
            logger.warning(
                "ACCIDENTAL SHORT COVER %s: buying %.0f shares to flatten broker short",
                symbol, cover_qty,
            )
            fill, status = self._submit_fast_exit_order(order, bar)
            if fill is None:
                msg = (
                    f"ACCIDENTAL SHORT COVER FAILED {symbol}: "
                    f"{cover_qty:.0f} shares status={status.value}"
                )
                logger.error(msg)
                if hasattr(self, "_hub"):
                    self._hub.add_log("ERROR", msg)
                continue

            try:
                from daytrading.execution.broker import apply_fill

                pos = self._pipeline.portfolio.positions.get(symbol)
                if pos is not None and pos.quantity < 0:
                    apply_fill(self._pipeline.portfolio, fill)
            except Exception as exc:
                logger.debug("Could not apply accidental short cover locally: %s", exc)

            self._pipeline.exit_manager.untrack(symbol)
            self._clear_broker_stop(symbol)
            if hasattr(self, "_exec_timer"):
                self._exec_timer.cancel(symbol)
            if hasattr(self, "_timed_signal_queue"):
                self._timed_signal_queue = deque(
                    s for s in self._timed_signal_queue if s.symbol != symbol
                )

            msg = (
                f"ACCIDENTAL SHORT COVERED {symbol}: bought {fill.quantity:.0f} "
                f"@ ${fill.price:.2f}"
            )
            logger.warning(msg)
            if hasattr(self, "_hub"):
                self._hub.add_log("WARNING", msg)
            if hasattr(self, "_journal"):
                self._journal.record(
                    "short_cleanup",
                    {
                        "symbol": symbol,
                        "quantity": fill.quantity,
                        "price": fill.price,
                        "side": fill.side.value,
                        "reason": "accidental_short_cover",
                        "broker_qty": qty,
                    },
                    ts=fill.ts,
                )
            self._seed_recent_order_ids()
            broker_positions.pop(symbol, None)

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

    def _has_pending_timed_entry(self) -> bool:
        return bool(self._exec_timer.pending_symbols or self._timed_signal_queue)

    def _defer_fast_scan_movers(self, movers: List[Dict]) -> None:
        if not movers:
            return
        existing = {
            str(m.get("symbol", "")).upper()
            for m in self._deferred_fast_scan_movers
        }
        for mover in movers:
            sym = str(mover.get("symbol", "")).upper()
            if sym and sym not in existing:
                self._deferred_fast_scan_movers.append(mover)
                existing.add(sym)
        max_pending = max(self._fast_scan_process_max * 2, 40)
        if len(self._deferred_fast_scan_movers) > max_pending:
            self._deferred_fast_scan_movers = self._deferred_fast_scan_movers[-max_pending:]
        logger.debug(
            "Fast scan deferred while timed entry pending: %d movers queued",
            len(self._deferred_fast_scan_movers),
        )

    def _take_deferred_fast_scan_movers(self, movers: List[Dict]) -> List[Dict]:
        if not self._deferred_fast_scan_movers:
            return movers
        combined: List[Dict] = []
        seen: Set[str] = set()
        for mover in self._deferred_fast_scan_movers + list(movers):
            sym = str(mover.get("symbol", "")).upper()
            if sym and sym not in seen:
                combined.append(mover)
                seen.add(sym)
        self._deferred_fast_scan_movers = []
        return combined

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
        # Setup consumed — drop its chase anchor so a future setup re-anchors.
        if getattr(self, "_timed_entry_anchor", None) is not None:
            self._timed_entry_anchor.pop(signal.symbol, None)
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
        tracked_for_exit = self._pipeline.exit_manager.tracked.get(fill.symbol)
        if str(getattr(tracked_for_exit, "entry_trigger", "") or "") == "warrior_ignition":
            try:
                completed = tracked_for_exit is None or float(getattr(tracked_for_exit, "remaining_qty", 0.0) or 0.0) <= 0.0
                self._record_warrior_ignition_exit(fill.symbol, pnl, reason, completed=completed)
            except Exception:
                logger.debug("failed to record warrior ignition exit", exc_info=True)
        elif (
            pnl < 0.0
            and tracked_for_exit is not None
            and (
                str(getattr(tracked_for_exit, "entry_strategy", "") or "") == "warrior_squeeze_playbook"
                or str(getattr(tracked_for_exit, "entry_trigger", "") or "").startswith("warrior_")
            )
        ):
            failed = getattr(self, "_warrior_failed_momentum", None)
            if failed is None:
                failed = {}
                self._warrior_failed_momentum = failed
            failed[str(fill.symbol or "").upper()] = reason or "warrior momentum loss"
        try:
            from daytrading.ml.shadow_collector import label_exit_snapshots
            label_exit_snapshots(fill.symbol, fill.price)
        except Exception:
            pass
        try:
            bars = list(self._bar_buffer.get(fill.symbol, deque()))
            if bars:
                universe = {fill.symbol: bars}
                quotes = {fill.symbol: list(self._quote_buffer.get(fill.symbol, deque()))}
                self._pipeline.missed_a_plus.record_early_exit(
                    symbol=fill.symbol,
                    entry_price=entry_price,
                    exit_price=fill.price,
                    reason=reason or "exit",
                    universe=universe,
                    quotes=quotes,
                    now=fill.ts if isinstance(fill.ts, datetime) else datetime.now(timezone.utc),
                )
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

        if strategy in (
            "breakout_scalp",
            "breakout_scalp_momentum",
            "momentum_burst_scalp",
        ) or is_hit_run_strategy(strategy):
            self._breakout_scalp_active = False
        if is_hit_run_strategy(strategy):
            try:
                block_reason = self._record_momentum_burst_hit_run_pnl(fill.symbol, pnl)
                if block_reason:
                    self._hub.add_log(
                        "WARN",
                        "MOMENTUM BURST HIT-RUN {} stopped for day: {}".format(
                            fill.symbol, block_reason,
                        ),
                    )
                lower_reason = (reason or "").lower()
                is_win = pnl > 0.0 and "stop" not in lower_reason and "loss" not in lower_reason
                if is_win and strategy == "warrior_squeeze_playbook":
                    if "target" in lower_reason or "take_profit" in lower_reason:
                        symbol_pnl = float(
                            self._momentum_burst_hit_run_symbol_pnl.get(fill.symbol, 0.0) or 0.0
                        )
                        entry_trigger = str(
                            getattr(self, "_warrior_squeeze_last_entry_trigger", {}).get(
                                fill.symbol, ""
                            )
                        )
                        if symbol_pnl > 0:
                            self._warrior_squeeze_target_wins[fill.symbol] = (
                                self._warrior_squeeze_target_wins.get(fill.symbol, 0) + 1
                            )
                            self._warrior_squeeze_last_target_at[fill.symbol] = time.monotonic()
                            if entry_trigger != "warrior_low_price_proof_reclaim":
                                self._pipeline._symbol_entry_counts[fill.symbol] = (
                                    self._pipeline._max_entries_per_symbol
                                )
                            getattr(self, "_warrior_squeeze_failed_burst", {}).pop(fill.symbol, None)
                            getattr(self, "_warrior_squeeze_failed_burst_high", {}).pop(fill.symbol, None)
                        else:
                            self._momentum_burst_hit_run_day_blocked[fill.symbol] = (
                                "Warrior recovery target did not restore positive symbol P&L "
                                "(${:.2f}); stop trading symbol for day".format(symbol_pnl)
                            )
                            self._pipeline._symbol_entry_counts[fill.symbol] = (
                                self._pipeline._max_entries_per_symbol
                            )
                        self._momentum_burst_pending.pop(fill.symbol, None)
                        recent_10s = self._momentum_burst_recent_10s(fill.symbol, count=1)
                        if recent_10s:
                            self._momentum_burst_window_high[fill.symbol] = max(
                                float(self._momentum_burst_window_high.get(fill.symbol, 0.0) or 0.0),
                                float(recent_10s[-1].high or 0.0),
                            )
                    cooldown = float(
                        getattr(self, "_warrior_squeeze_win_cooldown_sec", 0.0) or 0.0
                    )
                elif is_win:
                    cooldown = self._momentum_burst_hit_run_win_cooldown_sec
                else:
                    cooldown = self._momentum_burst_hit_run_loss_cooldown_sec
                    if (
                        strategy == "warrior_squeeze_playbook"
                        and self._warrior_squeeze_target_wins.get(fill.symbol, 0) <= 0
                    ):
                        failed_bursts = self._warrior_squeeze_failed_burst
                        failed_bursts[fill.symbol] = (
                            "first Warrior burst failed via {}; require a fresh new window".format(reason)
                        )
                        failed_highs = self._warrior_squeeze_failed_burst_high
                        recent_10s = self._momentum_burst_recent_10s(fill.symbol, count=4)
                        failed_highs[fill.symbol] = max(
                            [float(fill.price or 0.0)]
                            + [float(bar.high or 0.0) for bar in recent_10s]
                        )
                    elif (
                        strategy == "warrior_squeeze_playbook"
                        and self._warrior_squeeze_target_wins.get(fill.symbol, 0) > 0
                        and pnl < 0.0
                    ):
                        post_target_loss = abs(float(pnl or 0.0))
                        symbol_pnl = float(
                            self._momentum_burst_hit_run_symbol_pnl.get(fill.symbol, 0.0)
                            or 0.0
                        )
                        if symbol_pnl > 0.0 and post_target_loss <= max(5.0, symbol_pnl * 0.35):
                            self._warrior_squeeze_post_target_reclaim_allowed[fill.symbol] = max(
                                1,
                                int(
                                    self._warrior_squeeze_post_target_reclaim_allowed.get(
                                        fill.symbol,
                                        0,
                                    )
                                    or 0
                                ),
                            )
                        else:
                            self._momentum_burst_hit_run_day_blocked[fill.symbol] = (
                                "Warrior post-target loss ${:.2f}; stop trading symbol for day".format(
                                    post_target_loss
                                )
                            )
                            self._pipeline._symbol_entry_counts[fill.symbol] = (
                                self._pipeline._max_entries_per_symbol
                            )
                self._momentum_burst_hit_run_block_until[fill.symbol] = time.monotonic() + cooldown
            except Exception:
                self._momentum_burst_hit_run_block_until[fill.symbol] = (
                    time.monotonic() + self._momentum_burst_hit_run_loss_cooldown_sec
                )

        # Cancel any pending entry signals for this symbol (prevent re-entry race)
        self._exec_timer.cancel(fill.symbol)
        self._timed_signal_queue = deque(
            s for s in self._timed_signal_queue if s.symbol != fill.symbol
        )

        tracked = self._pipeline.exit_manager.tracked.get(fill.symbol)
        portfolio_pos = getattr(self._pipeline, "portfolio", None)
        portfolio_pos = (
            portfolio_pos.positions.get(fill.symbol)
            if portfolio_pos is not None and hasattr(portfolio_pos, "positions")
            else None
        )
        broker_should_be_flat = (
            fill.side is Side.SELL
            and portfolio_pos is not None
            and (
                portfolio_pos.is_flat
                or portfolio_pos.quantity <= 0
            )
        )
        if tracked is None or tracked.remaining_qty <= 0 or broker_should_be_flat:
            self._pipeline.exit_manager.untrack(fill.symbol)
            if hasattr(self, "_reconciler"):
                self._reconciler.clear_pending(fill.symbol)
            self._clear_broker_stop(fill.symbol)
        else:
            self._refresh_broker_stop(fill.symbol)
        return pnl

    def _record_momentum_burst_hit_run_pnl(self, symbol: str, pnl: float) -> Optional[str]:
        """Track per-symbol hit-run realized P&L and block overtrading after give-back."""
        sym = symbol.upper()
        current = float(self._momentum_burst_hit_run_symbol_pnl.get(sym, 0.0) or 0.0) + float(pnl or 0.0)
        self._momentum_burst_hit_run_symbol_pnl[sym] = current
        peak = max(float(self._momentum_burst_hit_run_symbol_peak_pnl.get(sym, 0.0) or 0.0), current)
        self._momentum_burst_hit_run_symbol_peak_pnl[sym] = peak

        loss_stop = max(0.0, float(getattr(self, "_momentum_burst_hit_run_daily_loss_stop", 0.0) or 0.0))
        if loss_stop > 0 and current <= -loss_stop:
            reason = "daily hit-run loss ${:.2f} reached stop ${:.2f}".format(abs(current), loss_stop)
            self._momentum_burst_hit_run_day_blocked[sym] = reason
            return reason

        giveback_stop = max(0.0, float(getattr(self, "_momentum_burst_hit_run_max_giveback", 0.0) or 0.0))
        giveback = peak - current
        if (
            getattr(self, "_momentum_burst_hit_run_stop_after_giveback", True)
            and peak > 0
            and giveback_stop > 0
            and giveback >= giveback_stop
        ):
            reason = "gave back ${:.2f} from hit-run peak ${:.2f}".format(giveback, peak)
            self._momentum_burst_hit_run_day_blocked[sym] = reason
            return reason
        return None

    def _active_warrior_trade_count(self) -> int:
        """Count active Warrior/burst scalps for the shared risk allocator."""
        count = 0
        try:
            tracked = getattr(self._pipeline.exit_manager, "tracked", {})
            for pos in tracked.values():
                reason = str(getattr(pos, "reason", "") or "").lower()
                if (
                    "warrior squeeze" in reason
                    or "momentum burst hit-run" in reason
                    or "momentum burst scalp" in reason
                    or "breakout scalp" in reason
                ):
                    count += 1
        except Exception:
            return 1 if getattr(self, "_breakout_scalp_active", False) else 0
        if count <= 0 and getattr(self, "_breakout_scalp_active", False):
            return 1
        return count

    def _active_warrior_symbols(self) -> set[str]:
        symbols: set[str] = set()
        try:
            tracked = getattr(self._pipeline.exit_manager, "tracked", {})
            for sym, pos in tracked.items():
                reason = str(getattr(pos, "reason", "") or "").lower()
                if (
                    "warrior squeeze" in reason
                    or "momentum burst hit-run" in reason
                    or "momentum burst scalp" in reason
                    or "breakout scalp" in reason
                ):
                    symbols.add(str(sym).upper())
        except Exception:
            return symbols
        return symbols

    def _ensure_warrior_watch_capacity(self, symbol: str, high: float) -> bool:
        watch = getattr(self, "_warrior_watch", None)
        if watch is None:
            return True
        capacity = int(getattr(self, "_warrior_watch_capacity", 10) or 0)
        ok = watch.ensure_capacity(
            symbol,
            capacity=capacity,
            candidate_high=high,
            active_symbols=self._active_warrior_symbols(),
        )
        if not ok:
            logger.info(
                "WARRIOR SQUEEZE watch full (%s); not arming %s",
                capacity,
                symbol.upper(),
            )
        return ok

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
            today_start = self._session_start_utc()
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

    def _session_start_utc(self) -> datetime:
        now_et = self._now_et()
        session_start_et = now_et.replace(
            hour=4, minute=0, second=0, microsecond=0,
        )
        return session_start_et.astimezone(timezone.utc).replace(tzinfo=None)

    def _order_is_from_current_session(self, order: object) -> bool:
        fill_time = getattr(order, "filled_at", None) or getattr(order, "submitted_at", None)
        if fill_time is None:
            return False
        if getattr(fill_time, "tzinfo", None) is not None:
            fill_time = fill_time.astimezone(timezone.utc).replace(tzinfo=None)
        return fill_time >= self._session_start_utc()

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
                if not self._order_is_from_current_session(o):
                    self._last_synced_order_ids.add(oid)
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
    ) -> int:
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

        # Clear per-symbol timed-entry chase anchors for the new day
        if getattr(self, "_timed_entry_anchor", None):
            self._timed_entry_anchor.clear()
        if getattr(self, "_warrior_watch", None):
            engine = getattr(self, "_warrior_engine", None)
            if engine is not None:
                engine.reset_session()
            else:
                self._warrior_watch.reset_session()
        else:
            for attr in (
                "_momentum_burst_armed",
                "_momentum_burst_window_high",
                "_momentum_burst_session_anchor_high",
                "_momentum_burst_pending",
                "_momentum_burst_hit_run_counts",
                "_momentum_burst_hit_run_block_until",
                "_momentum_burst_hit_run_symbol_pnl",
                "_momentum_burst_hit_run_symbol_peak_pnl",
                "_momentum_burst_hit_run_day_blocked",
                "_warrior_squeeze_rejection_high",
                "_warrior_squeeze_rejection_reason",
                "_warrior_squeeze_target_wins",
                "_warrior_squeeze_last_target_at",
                "_warrior_squeeze_failed_burst",
                "_warrior_squeeze_failed_burst_high",
                "_warrior_squeeze_post_target_reclaim_allowed",
                "_warrior_squeeze_last_entry_trigger",
                "_warrior_normal_fallback_rejects",
                "_warrior_normal_fallback_last_reason",
                "_warrior_failed_momentum",
            ):
                value = getattr(self, attr, None)
                if value:
                    value.clear()
        # Defensive: never carry a latched quick-scalp-open flag across a session
        self._breakout_scalp_active = False
        if getattr(self, "_recent_quick_scalp_rejects", None):
            self._recent_quick_scalp_rejects.clear()
        if getattr(self, "_surge_vol_hist", None):
            self._surge_vol_hist.clear()
        for attr in (
            "_warrior_ignition_entries",
            "_warrior_ignition_failed_entries",
            "_warrior_ignition_peak_price",
            "_warrior_ignition_peak_day_move",
            "_warrior_ignition_trade_pnl",
            "_warrior_ignition_watch",
        ):
            value = getattr(self, attr, None)
            if value:
                value.clear()

        # Reset daily P&L tracking
        self._pipeline._daily_pnl = 0.0
        self._pipeline._daily_losers.clear()
        if hasattr(self._pipeline, "_daily_loss_counts"):
            self._pipeline._daily_loss_counts.clear()
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
        if getattr(self, "_broker", None) is not None:
            try:
                self._sync_trade_history()
            except Exception as exc:
                logger.debug("Trade history restore after daily reset failed: %s", exc)
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
        max_drain = 5000
        while drained < max_drain:
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
                if tick.symbol in self._watchlist_set or self.is_hot_watch_active(tick.symbol):
                    self._pipeline.trade_guard.halt_tracker.update_price(
                        tick.symbol, tick.price, tick.ts,
                    )
                    buf = self._tick_buffer.get(tick.symbol)
                    if buf is None:
                        buf = deque(maxlen=self._max_ticks_per_symbol)
                        self._tick_buffer[tick.symbol] = buf
                    buf.append(tick)
                    # Tick-based early entry: catch fast movers near the base before a
                    # 10s bar can confirm. Gated on symbols with a pending timed entry
                    # (small set). The chase/spread guards still run at the execute path.
                    if tick.symbol in self._exec_timer.pending_symbols:
                        ready_sig = self._exec_timer.on_tick(tick)
                        if ready_sig is not None:
                            self._execute_timed_signal(ready_sig)

                    # Tick-level trailing stop for open positions (post-half only)
                    tracked_pos = self._pipeline.exit_manager._positions.get(tick.symbol)
                    if tracked_pos and tracked_pos.remaining_qty > 0 and tracked_pos.breakeven_locked:
                        if tick.price > tracked_pos.highest_price:
                            tracked_pos.highest_price = tick.price
                        # Only trail on ticks AFTER half-sell (let pre-half winners run to target)
                        if tracked_pos.sold_half:
                            trail_stop = self._tick_trail_stop_for(tracked_pos)
                            if trail_stop > tracked_pos.stop_loss:
                                tracked_pos.stop_loss = trail_stop
                            if tick.price <= tracked_pos.stop_loss:
                                self._instant_trail_exit(tick.symbol, tick.price)

                    # Tape pressure profit-protection exit (pre- and post-half)
                    if tracked_pos and tracked_pos.remaining_qty > 0:
                        if (
                            not self._uses_runner_core_management(tracked_pos)
                            and tick.price > tracked_pos.entry_price
                        ):
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
                loaded_any = False
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
                        loaded_any = True
                if evt.prior_day_stats:
                    self._prior_day_stats.update(evt.prior_day_stats)
                for sym in evt.bars_by_symbol:
                    self._seed_hod_session(sym)
                got_bars = got_bars or loaded_any

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
                if self._has_pending_timed_entry():
                    self._defer_fast_scan_movers(evt.new_movers)
                    continue
                movers = self._take_deferred_fast_scan_movers(evt.new_movers)
                self._enqueue_candidate_hydration(
                    movers,
                    source="fast scan event",
                )

        if drained >= max_drain:
            logger.warning(
                "Event drain capped at %d events; leaving backlog for next loop",
                max_drain,
            )
        elif drained > 2500:
            logger.info("Drained %d events from queue", drained)
        return got_bars

    @staticmethod
    def _bar_is_today(bar: Bar, today_et) -> bool:
        try:
            return bar.ts.astimezone(ET).date() == today_et
        except Exception:
            return True

    def _handle_fast_scan_movers(
        self,
        new_movers: List[Dict],
        *,
        push_event: bool = False,
    ) -> bool:
        """Process new movers from fast scan: float check → add to pool → hydrate.

        When called from the candidate worker, fetched bars are returned through
        BarsLoadedEvent so the main loop remains the only writer to bar buffers.
        """
        if not self._float_checker or not new_movers:
            return 0

        hydrate_symbols = []
        skipped_fresh = 0
        pool_set = set(self._hod_bar_pool)
        now_ts = datetime.now(timezone.utc)
        for mover in new_movers:
            sym = mover["symbol"]
            is_strong = mover.get("abs_change_pct", 0) >= 10.0 and mover.get("volume", 0) >= 200_000
            flt = self._float_checker.get_float_cached(sym)

            hot_reject = self._hot_watch_reject_reason(mover, flt)
            if hot_reject is None:
                self._promote_hot_watch(
                    mover,
                    flt=flt,
                    reason="fast scan mover",
                )
            else:
                self._journal.record("hot_watch", {
                    "symbol": sym,
                    "stage": "rejected",
                    "reason": hot_reject,
                    "price": mover.get("price"),
                    "change_pct": mover.get("change_pct"),
                    "abs_change_pct": mover.get("abs_change_pct"),
                    "volume": mover.get("volume"),
                    "score": mover.get("score"),
                    "float": flt,
                })

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
                else:
                    skipped_fresh += 1
                continue
            if len(self._hod_bar_pool) >= self._hod_pool_max:
                break
            if flt is not None and flt > self._hod_max_float:
                continue
            if flt is None and not is_strong:
                continue
            self._hod_bar_pool.append(sym)
            pool_set.add(sym)
            hydrate_symbols.append(sym)

        if not hydrate_symbols:
            self._sync_tick_tracker_pool()
            if skipped_fresh:
                try:
                    self._hub.on_candidate_hydration(skipped_fresh=skipped_fresh)
                except Exception:
                    pass
            return 0

        logger.info(
            "Candidate hydrate → loading %d HOD movers: %s",
            len(hydrate_symbols),
            ", ".join(hydrate_symbols[:10]),
        )
        self._hub.add_log(
            "INFO",
            "Candidate hydrate {} HOD movers — {}".format(
                len(hydrate_symbols), ", ".join(hydrate_symbols[:10]),
            ),
        )

        candidate_batch_max = getattr(self, "_candidate_hydrate_batch_max", 10)
        max_batch = min(self._hod_hydrate_batch_max, candidate_batch_max)
        batch = hydrate_symbols[:max_batch]
        loaded_count = 0
        try:
            bars_by_symbol = self._fetch_session_bars(batch)
            prior_stats = self._fetch_prior_day_stats(batch)
            if push_event:
                loaded_count = len(bars_by_symbol)
                if bars_by_symbol or prior_stats:
                    try:
                        self._event_queue.put_nowait(
                            BarsLoadedEvent(bars_by_symbol, prior_stats)
                        )
                    except queue.Full:
                        logger.warning("Event queue full; dropped candidate bars")
            else:
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
                        loaded_count += 1
                if prior_stats:
                    self._prior_day_stats.update(prior_stats)
                for sym in bars_by_symbol:
                    self._seed_hod_session(sym)
        except Exception as exc:
            logger.warning("Candidate hydrate failed: %s", exc)

        if skipped_fresh:
            try:
                self._hub.on_candidate_hydration(skipped_fresh=skipped_fresh)
            except Exception:
                pass
        self._sync_tick_tracker_pool()
        return loaded_count

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
            entry_strategies = {
                sym: getattr(pos, "entry_strategy", "") or ""
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
                fill, status = self._submit_fast_exit_order(order, bar)
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
                        strategy=entry_strategies.get(sig.symbol, ""),
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
                    if not self._broker_has_compatible_exit_position(
                        sig.symbol,
                        Side.SELL if sig.action is SignalAction.EXIT_LONG else Side.BUY,
                    ):
                        logger.info(
                            "FAST EXIT %s: broker already flat/incompatible after clamp; syncing state",
                            sig.symbol,
                        )
                        self._push_positions_from_alpaca()
                        self._pipeline.set_cooldown(sig.symbol)
                        continue
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
                    fill2, status2 = self._submit_fast_exit_order(market_order, bar)
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
                            strategy=entry_strategies.get(sig.symbol, ""),
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

    def _submit_fast_exit_order(self, order: Order, bar: Bar):
        """Submit an urgent exit with a short fill wait before retry logic.

        Entry orders can wait for a better fill. Stop exits should not: if the
        first limit does not fill quickly, the caller retries with a guarded
        marketable limit.
        """
        order = self._clamp_fast_exit_order_to_broker_position(order)
        if order is None:
            return None, OrderStatus.CANCELLED

        old_wait = getattr(self._broker, "_max_wait", None)
        try:
            if old_wait is not None:
                self._broker._max_wait = min(float(old_wait), 1.0)
            return self._broker.submit(order, bar, self._pipeline.portfolio)
        finally:
            if old_wait is not None:
                self._broker._max_wait = old_wait

    def _broker_has_compatible_exit_position(self, symbol: str, side: Side) -> bool:
        """Return whether Alpaca still shows quantity compatible with an exit side."""
        get_positions = getattr(self._broker, "get_positions", None)
        if not callable(get_positions):
            return True
        try:
            invalidate = getattr(self._broker, "_invalidate_position_cache", None)
            if callable(invalidate):
                invalidate()
            broker_positions = get_positions() or {}
        except Exception as exc:
            logger.warning("FAST EXIT %s: broker position recheck failed: %s", symbol, exc)
            return True
        pos = broker_positions.get(symbol)
        if not pos:
            return False
        try:
            broker_qty = float(pos.get("qty", 0.0) if isinstance(pos, dict) else getattr(pos, "qty", 0.0))
        except (TypeError, ValueError):
            return True
        if side is Side.SELL:
            return broker_qty > 0
        return broker_qty < 0

    def _clamp_fast_exit_order_to_broker_position(self, order: Order) -> Optional[Order]:
        """Cap urgent exit quantity to the fresh broker position.

        Fast exits can run immediately after a partial fill, cancelled order, or
        broker-held protective stop. In that window local tracking may still say
        more shares exist than Alpaca will accept. Clamp to broker reality before
        submitting the urgent exit/retry.
        """
        get_positions = getattr(self._broker, "get_positions", None)
        if not callable(get_positions):
            return order

        try:
            invalidate = getattr(self._broker, "_invalidate_position_cache", None)
            if callable(invalidate):
                invalidate()
            broker_positions = get_positions() or {}
        except Exception as exc:
            logger.warning("FAST EXIT %s: broker position refresh failed: %s", order.symbol, exc)
            return order

        pos = broker_positions.get(order.symbol)
        if not pos:
            logger.warning(
                "FAST EXIT %s: broker shows no position; skipping %.0f-share exit",
                order.symbol, order.quantity,
            )
            return None

        try:
            broker_qty = float(pos.get("qty", 0.0) if isinstance(pos, dict) else getattr(pos, "qty", 0.0))
        except (TypeError, ValueError):
            return order

        if order.side is Side.SELL:
            available_qty = max(0, int(broker_qty))
        else:
            available_qty = max(0, int(abs(broker_qty))) if broker_qty < 0 else 0

        if available_qty <= 0:
            logger.warning(
                "FAST EXIT %s: broker qty %.0f incompatible with %s exit; skipping",
                order.symbol, broker_qty, order.side.value,
            )
            return None

        requested_qty = int(order.quantity)
        if requested_qty <= available_qty:
            return order

        logger.warning(
            "FAST EXIT %s: clamping exit quantity %.0f → %d from fresh broker qty %.0f",
            order.symbol, order.quantity, available_qty, broker_qty,
        )
        return Order(
            symbol=order.symbol,
            side=order.side,
            quantity=float(available_qty),
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            client_order_id=order.client_order_id,
        )

    def _submit_protective_exit_order(self, order: Order, bar: Bar, label: str):
        """Submit a protective exit and escalate when the first limit misses."""
        fill, status = self._submit_fast_exit_order(order, bar)
        if fill:
            return fill, status

        logger.warning(
            "%s limit not filled for %s — escalating to urgent marketable exit",
            label, order.symbol,
        )
        if not self._broker_has_compatible_exit_position(order.symbol, order.side):
            logger.info(
                "%s %s: broker already flat/incompatible after failed exit; syncing state",
                label, order.symbol,
            )
            self._push_positions_from_alpaca()
            return None, status

        if hasattr(self._broker, "submit_urgent_exit"):
            return self._broker.submit_urgent_exit(
                order, bar, self._pipeline.portfolio,
            )

        fallback_order = Order(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            limit_price=None,
        )
        return self._submit_fast_exit_order(fallback_order, bar)

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
            fill, status = self._submit_protective_exit_order(order, bar, "TICK TRAIL EXIT")
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

    @staticmethod
    def _tick_trail_stop_for(tracked_pos) -> float:
        """Return the tick trailing stop for a tracked position.

        Ordinary scalps keep the tight 1% tick trail. Confirmed runners have
        already paid the first partial, so the remaining shares get wider room
        to catch CHAI/VERU-style continuation.
        """
        trail_pct = 0.01
        if getattr(tracked_pos, "runner_confirmed", False):
            trail_pct = float(getattr(tracked_pos, "runner_trail_pct", 0.03) or 0.03)
            trail_pct = max(0.01, min(trail_pct, 0.06))
        return round(float(tracked_pos.highest_price) * (1.0 - trail_pct), 4)

    @staticmethod
    def _uses_runner_core_management(tracked_pos) -> bool:
        """Let confirmed runner cores use runner trail/re-add instead of fast scalp exits."""
        return bool(
            getattr(tracked_pos, "runner_confirmed", False)
            and getattr(tracked_pos, "sold_half", False)
            and getattr(tracked_pos, "remaining_qty", 0) > 0
        )

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
            fill, status = self._submit_protective_exit_order(order, bar, "TAPE PRESSURE EXIT")
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

            # Confirmed runners already paid the first partial. Let the wider
            # runner trail/re-add playbook manage normal 10s pullbacks.
            if self._uses_runner_core_management(pos):
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
            fill, status = self._submit_protective_exit_order(order, bar, "RED 10s EXIT")
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

    def _record_missed_a_plus_signal(
        self,
        signal: TradeSignal,
        *,
        layer: str,
        reason: str,
        fallback_price: float = 0.0,
    ) -> None:
        try:
            bars = list(self._bar_buffer.get(signal.symbol, deque()))
            if signal.scan_result is not None and signal.scan_result.bars:
                bars = list(signal.scan_result.bars)
            if not bars:
                return
            universe = {signal.symbol: bars}
            quotes = {signal.symbol: list(self._quote_buffer.get(signal.symbol, deque()))}
            self._pipeline.missed_a_plus.record_blocked(
                layer=layer,
                reason=reason,
                universe=universe,
                quotes=quotes,
                signal=signal,
                fallback_price=fallback_price or signal.entry_price,
            )
            self._pipeline.missed_a_plus.update_prices(
                universe, now=datetime.now(timezone.utc),
            )
            self._hub.on_missed_a_plus(self._pipeline.missed_a_plus_report())
            self._hub.on_scanner_near_miss(self._pipeline.scanner_near_miss_summary())
        except Exception:
            pass

    @staticmethod
    def _is_watch_only_decision(decision: Any) -> bool:
        """True for shadow-scanner 'collecting data, not live A+' monitoring rows.

        These are not real entry attempts and are excluded from the funnel.
        """
        if isinstance(decision, dict):
            text = "{} {} {}".format(
                decision.get("reason") or "",
                decision.get("blocked_layer") or "",
                decision.get("setup_tier") or "",
            ).lower()
        else:
            text = "{} {}".format(
                getattr(decision, "reason", "") or "",
                getattr(decision, "blocked_layer", "") or "",
            ).lower()
        return "watch only" in text or "collecting data" in text

    def _record_entry_decision(
        self,
        decision: EntryDecision,
        *,
        source: str = "runner",
    ) -> None:
        """Persist one compact structured funnel record for entry debugging."""
        try:
            payload = decision.to_payload()
            payload["source"] = source
            payload["market_phase"] = self._market_phase()
            self._journal.record("entry_decision", payload, ts=decision.ts)
        except Exception:
            pass

    def _record_entry_reject(
        self,
        signal: TradeSignal,
        *,
        stage: str,
        reason: str,
        source: str = "runner",
        price: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> EntryDecision:
        executor = _ensure_entry_executor(self)
        return executor.reject(
            symbol=signal.symbol,
            stage=stage,
            reason=reason,
            blocked_layer=stage,
            signal=signal,
            source=source,
            price=price or signal.entry_price,
            metadata=metadata,
        )

    def _entry_execution_context(self) -> EntryExecutionContext:
        class _NoopJournal:
            def record(self, *args, **kwargs) -> None:
                return None

        return EntryExecutionContext(
            new_entries_blocked=self._new_entries_blocked,
            pipeline=self._pipeline,
            bar_buffer=getattr(self, "_bar_buffer", {}),
            hub=self._hub,
            broker=self._broker,
            journal=getattr(self, "_journal", _NoopJournal()),
            chase_reject=self._timed_entry_chase_reject,
            record_entry_reject=self._record_entry_reject,
            record_missed_a_plus=self._record_missed_a_plus_signal,
            shared_entry_quality_reject=self._shared_entry_quality_reject,
            execution_learning_context=self._execution_learning_context,
            is_hot_hod_timed_signal=self._is_hot_hod_timed_signal,
            retry_hot_hod_timed_entry=self._retry_hot_hod_timed_entry,
            on_position_opened=self._on_position_opened,
            market_phase=self._market_phase,
            seed_recent_order_ids=self._seed_recent_order_ids,
        )

    def _execute_timed_signal(self, signal: TradeSignal) -> None:
        """Execute a deferred signal that the execution timer released."""
        _ensure_entry_executor(self).execute_timed_signal(
            self._entry_execution_context(),
            signal,
        )

    def _timed_entry_chase_reject(
        self,
        signal: TradeSignal,
        fallback_bar: Bar,
    ) -> Optional[str]:
        """Cancel delayed timed entries that have become chase entries."""
        if signal.action not in (SignalAction.ENTER_LONG, SignalAction.SCALE_UP_LONG):
            return None
        original = self._timed_entry_chase_anchor(signal)
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
            "level_breakout_reclaim",
            "pullback_base",
            "shallow_stair_continuation",
            "early_vwap_reclaim_scout",
            "abc_reentry",
            "breakout_scalp",
            "opening_range_breakout",
            "runner_readd",
        }
        is_hot = pattern in hot_patterns or scanner in hot_patterns

        if pattern == "opening_range_breakout" or scanner == "opening_range_breakout":
            if live < original * 0.97:
                return "live price {:.4f} pulled back too far from breakout signal {:.4f}".format(
                    live, original,
                )
        if pattern == "level_breakout_reclaim" or scanner == "level_breakout_reclaim":
            criteria = hit.criteria if hit is not None else {}
            try:
                breakout_level = float(
                    criteria.get("breakout_level")
                    or criteria.get("base_high")
                    or 0.0
                )
            except (TypeError, ValueError):
                breakout_level = 0.0
            if breakout_level > 0 and live < breakout_level:
                return "live price {:.4f} lost breakout level {:.4f}".format(
                    live, breakout_level,
                )
            if breakout_level > 0 and live > breakout_level * 1.025:
                return "live price {:.4f} too extended from breakout level {:.4f} (max 2.5%)".format(
                    live, breakout_level,
                )

        if pattern == "vwap_pullback" or scanner == "vwap_pullback":
            criteria = hit.criteria if hit is not None else {}
            vwap_anchor = self._vwap_pullback_extension_anchor(criteria)
            if vwap_anchor > 0 and live > vwap_anchor * 1.025:
                return "live price {:.4f} too extended from VWAP pullback base {:.4f} (max 2.5%)".format(
                    live, vwap_anchor,
                )

        if pattern == "hod_reclaim" or scanner == "hod_reclaim":
            hod_10s_reject = self._timed_hod_reclaim_10s_reject(signal, live, original)
            if hod_10s_reject is not None:
                return hod_10s_reject

        max_chase_pct = 0.025 if original >= 5.0 else 0.035
        if pattern in (
            "abc_continuation",
            "hod_reclaim",
            "level_breakout_reclaim",
            "pullback_base",
            "vwap_pullback",
            "early_vwap_reclaim_scout",
            "runner_readd",
            "abc_reentry",
        ):
            max_chase_pct = min(max_chase_pct, 0.025)
        if is_hot and live > original * (1.0 + max_chase_pct):
            return (
                "live price {:.4f} ran {:.1f}% above signal {:.4f} "
                "(max {:.1f}%)"
            ).format(live, (live - original) / original * 100.0, original, max_chase_pct * 100.0)

        quotes = list(self._quote_buffer.get(signal.symbol, []))
        recent_quotes = [q for q in quotes[-3:] if q.ask > q.bid > 0]
        if recent_quotes:
            avg_spread = sum(q.ask - q.bid for q in recent_quotes) / len(recent_quotes)
            avg_mid = sum((q.ask + q.bid) / 2.0 for q in recent_quotes) / len(recent_quotes)
            avg_spread_pct = avg_spread / avg_mid * 100.0 if avg_mid > 0 else 0.0
            max_spread = 0.9 if live < 5.0 else 0.6
            criteria = hit.criteria if hit is not None else {}
            avg_depth = sum(min(q.bid_size, q.ask_size) for q in recent_quotes) / len(recent_quotes)
            day_volume = 0.0
            recent_avg_volume = 0.0
            latest_volume = float(getattr(fallback_bar, "volume", 0.0) or 0.0)
            try:
                bars = list(self._bar_buffer.get(signal.symbol, []))
            except Exception:
                bars = []
            if bars:
                day_volume = float(sum(getattr(b, "volume", 0.0) or 0.0 for b in bars))
                recent = bars[-5:]
                recent_avg_volume = (
                    float(sum(getattr(b, "volume", 0.0) or 0.0 for b in recent)) / len(recent)
                    if recent else 0.0
                )
                latest_volume = float(getattr(bars[-1], "volume", 0.0) or latest_volume)
            session_high = max((float(getattr(b, "high", 0.0) or 0.0) for b in bars), default=live)
            distance_from_hod = (session_high - live) / session_high if session_high > 0 else 1.0
            spread_decision = assess_opportunity_scaled_spread(
                price=avg_mid or live,
                spread=avg_spread,
                pattern=pattern,
                setup_tier=str(criteria.get("setup_tier") or ""),
                entry_tier=str(criteria.get("entry_tier") or ""),
                day_volume=day_volume,
                recent_avg_volume=recent_avg_volume,
                latest_volume=latest_volume,
                distance_from_hod=distance_from_hod,
                quote_depth=avg_depth,
                normal_pct_limit=0.005,
                setup_score=float(getattr(hit, "score", 0.0) or 0.0) if hit is not None else 0.0,
            )
            if not spread_decision.ok:
                return "spread {:.2f}% ({:.2f}c) rejected: {}".format(
                    avg_spread_pct, avg_spread * 100.0,
                    spread_decision.reason or "too wide")
            if spread_decision.exception and hit is not None:
                current_factor = float(criteria.get("size_factor") or 1.0)
                criteria["size_factor"] = round(min(current_factor, spread_decision.size_factor), 2)
                spread_mode = spread_decision.mode or "opportunity_scaled"
                criteria["spread_exception"] = spread_mode
                criteria["spread_size_factor"] = round(spread_decision.size_factor, 2)
                if spread_mode == "elite_wide_spread":
                    criteria["entry_mode"] = "elite_wide_spread"

        if (
            is_hot
            and self._bar_aggregator is not None
            and not self._is_timed_scout_signal(signal)
        ):
            latest_10s = self._bar_aggregator.get_latest_10s(signal.symbol, count=1)
            if latest_10s:
                b = latest_10s[-1]
                if b.close < b.open and live >= original:
                    return "latest 10s candle turned red during entry wait"

        return None

    @staticmethod
    def _vwap_pullback_extension_anchor(criteria: dict) -> float:
        """Anchor VWAP pullback chase checks to the actual reclaim/base level.

        The signal price can be re-queued while a setup waits for 10s
        confirmation. For VWAP pullbacks, the trade thesis is the reclaim of
        VWAP/base, not paying far above it later. Use the highest meaningful
        setup level so a delayed release must still be near the setup.
        """
        anchors = []
        for key in (
            "setup_anchor",
            "base_high",
            "reclaim_level",
            "breakout_level",
            "vwap",
            "reclaim_vwap",
            "fresh_base_high",
        ):
            try:
                value = float(criteria.get(key) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                anchors.append(value)
        return max(anchors) if anchors else 0.0

    def _timed_hod_reclaim_10s_reject(
        self,
        signal: TradeSignal,
        live: float,
        original: float,
    ) -> Optional[str]:
        """Require HOD timed releases to still show 10s follow-through.

        HOD reclaims are especially sensitive to buying the last push after a
        deferred timer wait. The normal timer can release on a green 10s bar; this
        final check makes sure that bar is not just a weak hold before a fade.
        """
        aggregator = getattr(self, "_bar_aggregator", None)
        if aggregator is None:
            return None
        try:
            bars_10s = list(aggregator.get_latest_10s(signal.symbol, count=2) or [])
        except Exception:
            return None
        if not bars_10s:
            return None
        latest = bars_10s[-1]
        latest_open = float(getattr(latest, "open", 0.0) or 0.0)
        latest_high = float(getattr(latest, "high", 0.0) or 0.0)
        latest_low = float(getattr(latest, "low", 0.0) or 0.0)
        latest_close = float(getattr(latest, "close", 0.0) or 0.0)
        latest_volume = float(getattr(latest, "volume", 0.0) or 0.0)
        if latest_close <= 0:
            return None

        hit = signal.scan_result
        criteria = hit.criteria if hit is not None else {}
        try:
            reclaim_level = float(
                criteria.get("hod")
                or criteria.get("close")
                or signal.entry_price
                or original
                or 0.0
            )
        except (TypeError, ValueError):
            reclaim_level = float(signal.entry_price or original or 0.0)
        if reclaim_level > 0 and latest_close < reclaim_level * 0.998:
            return "HOD reclaim 10s close {:.4f} lost reclaim level {:.4f}".format(
                latest_close,
                reclaim_level,
            )
        if latest_close <= latest_open:
            return "HOD reclaim 10s confirmation red/flat"

        latest_range = max(latest_high - latest_low, 0.0)
        if latest_range > 0:
            close_location = (latest_close - latest_low) / latest_range
            if close_location < 0.65:
                return "HOD reclaim 10s confirmation weak close ({:.0%} location)".format(
                    close_location,
                )
            if latest_close < 3.0 and close_location < 0.75:
                return (
                    "low-price HOD reclaim 10s close too weak "
                    "({:.0%} location, need 75%+)"
                ).format(close_location)

        if len(bars_10s) >= 2:
            prev = bars_10s[-2]
            prev_high = float(getattr(prev, "high", 0.0) or 0.0)
            prev_volume = float(getattr(prev, "volume", 0.0) or 0.0)
            if prev_high > 0 and latest_high <= prev_high * 1.001:
                return "HOD reclaim 10s confirmation no expansion"
            if (
                latest_close < 3.0
                and prev_volume > 0
                and latest_volume < prev_volume * 0.65
            ):
                return (
                    "low-price HOD reclaim 10s volume faded "
                    "{:.0f} vs prior {:.0f}"
                ).format(latest_volume, prev_volume)

        try:
            setup_volume = float(criteria.get("volume") or 0.0)
        except (TypeError, ValueError):
            setup_volume = 0.0
        min_volume = 1_000.0
        if setup_volume >= 50_000:
            min_volume = min(50_000.0, max(10_000.0, setup_volume * 0.25))
        elif setup_volume >= 20_000:
            min_volume = max(5_000.0, setup_volume * 0.20)
        if latest_volume < min_volume:
            return "HOD reclaim 10s volume {:.0f} below follow-through floor {:.0f}".format(
                latest_volume,
                min_volume,
            )
        return None

    def _timed_entry_chase_anchor(self, signal: TradeSignal) -> float:
        """Persistent per-symbol anti-chase anchor.

        Pins the price where a timed-entry setup FIRST deferred, keyed by
        symbol. When a grinding name is cancelled and re-queued higher on each
        scan, the anchor keeps the original level so the chase ceiling cannot
        crawl up with the price. The anchor refreshes its liveness while the
        setup keeps re-deferring and is dropped once stale (TTL) so a genuinely
        new setup on the same symbol re-anchors. The pinned value is stamped
        into the signal criteria as ``setup_anchor`` so the execution timer's
        early-strength release measures from the same level.
        """
        sym = signal.symbol
        now = time.monotonic()
        ttl = float(getattr(self, "_timed_entry_anchor_ttl_sec", 300.0) or 0.0)
        store = getattr(self, "_timed_entry_anchor", None)
        if store is None:
            store = {}
            self._timed_entry_anchor = store
        anchor = 0.0
        existing = store.get(sym)
        if existing is not None:
            anchor_price, last_seen = existing
            if anchor_price > 0 and (now - last_seen) <= ttl:
                anchor = anchor_price
                store[sym] = (anchor_price, now)  # keep original, refresh liveness
        if anchor <= 0:
            anchor = self._timed_entry_anchor_candidate(signal)
            if anchor > 0:
                store[sym] = (anchor, now)
        if anchor > 0 and signal.scan_result is not None:
            try:
                signal.scan_result.criteria["setup_anchor"] = anchor
            except Exception:
                pass
        return anchor

    @staticmethod
    def _timed_entry_anchor_candidate(signal: TradeSignal) -> float:
        """First-defer anchor price: prefer stable structural levels, then the
        queued price, then the live close / signal price."""
        hit = signal.scan_result
        criteria = hit.criteria if hit is not None else {}
        for key in (
            "breakout_level",
            "base_high",
            "queued_entry_price",
            "trigger_price",
            "close",
        ):
            try:
                value = float(criteria.get(key) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value
        return float(signal.entry_price or 0.0)

    def _execution_learning_context(
        self,
        signal: TradeSignal,
        bar: Bar,
    ) -> dict:
        """Small stable feature pack for fill-quality learning."""
        hit = signal.scan_result
        criteria = hit.criteria if hit is not None else {}
        pattern = str(criteria.get("pattern") or (hit.scanner_name if hit else "") or "")
        setup_quality = str(criteria.get("setup_quality") or "")
        try:
            size_factor = float(criteria.get("size_factor") or 1.0)
        except (TypeError, ValueError):
            size_factor = 1.0
        signal_price = float(signal.entry_price or 0.0)
        live_price = float(bar.close or signal_price or 0.0)
        chase_pct = (
            (live_price - signal_price) / signal_price * 100.0
            if signal_price > 0 and live_price > 0 else 0.0
        )
        breakout_level = 0.0
        try:
            breakout_level = float(
                criteria.get("breakout_level")
                or criteria.get("base_high")
                or 0.0
            )
        except (TypeError, ValueError):
            breakout_level = 0.0
        from_level_pct = (
            (live_price - breakout_level) / breakout_level * 100.0
            if breakout_level > 0 and live_price > 0 else 0.0
        )
        return {
            "pattern": pattern,
            "setup_quality": setup_quality,
            "size_factor": round(size_factor, 2),
            "signal_price": round(signal_price, 4),
            "chase_pct": round(chase_pct, 4),
            "breakout_level": round(breakout_level, 4),
            "from_breakout_level_pct": round(from_level_pct, 4),
        }

    def _shared_entry_quality_reject(
        self,
        symbol: str,
        bars: Sequence[Bar],
        *,
        signal: Optional[TradeSignal] = None,
        stage: str = "runner_final_entry_guard",
        source: str = "runner",
    ) -> Optional[str]:
        """Run live buy/re-entry/add paths through shared entry policy."""
        decision = self._shared_entry_quality_decision(
            symbol, bars, signal=signal, stage=stage, source=source,
        )
        return decision.reject_reason

    def _shared_entry_quality_decision(
        self,
        symbol: str,
        bars: Sequence[Bar],
        *,
        signal: Optional[TradeSignal] = None,
        stage: str = "runner_final_entry_guard",
        source: str = "runner",
    ) -> EntryDecision:
        """Run live buy/re-entry/add paths through shared entry policy."""
        executor = _ensure_entry_executor(self)
        if signal is not None and signal.action not in (
            SignalAction.ENTER_LONG,
            SignalAction.REENTER_LONG,
            SignalAction.SCALE_UP_LONG,
        ):
            return executor.record_decision(
                self._entry_policy.decision(
                    symbol=symbol,
                    stage=stage,
                    passed=True,
                    signal=signal,
                    metadata={"skipped": "non-entry action"},
                ),
                source=source,
            )
        if not bars:
            if signal is None:
                signal = TradeSignal(
                    symbol=symbol,
                    action=SignalAction.ENTER_LONG,
                    quantity=0,
                    entry_price=0.0,
                    reason="no bars for shared entry quality",
                )
            return executor.reject(
                symbol=symbol,
                stage=stage,
                reason="no bars for shared entry quality",
                blocked_layer="entry_guard",
                signal=signal,
                source=source,
            )

        bars_5m = None
        aggregator = getattr(self, "_bar_aggregator", None)
        if aggregator is not None:
            try:
                bars_5m = aggregator.get_5m_bars(symbol)
            except Exception:
                bars_5m = None

        float_shares = None
        avg_daily_volume = None
        float_checker = getattr(self, "_float_checker", None)
        if float_checker is not None:
            float_shares = self._resolve_entry_float_shares(float_checker, symbol)
            try:
                avg_cache = getattr(float_checker, "_avg_vol_cache", None)
                if isinstance(avg_cache, dict):
                    avg_daily_volume = avg_cache.get(symbol.upper()) or avg_cache.get(symbol)
            except Exception:
                avg_daily_volume = None

        criteria = signal.scan_result.criteria if signal and signal.scan_result else {}
        scanner_name = signal.scan_result.scanner_name if signal and signal.scan_result else ""
        if signal is None:
            inferred_pattern = str(criteria.get("pattern") or scanner_name or source or "")
            inferred_criteria = {"pattern": inferred_pattern}
            if source == "post_blowoff_micro_base_scout":
                inferred_criteria.update({
                    "entry_mode": "post_blowoff_micro_base_scout",
                    "setup_tier": "A+ setup",
                })
            signal = TradeSignal(
                symbol=symbol,
                action=SignalAction.ENTER_LONG,
                quantity=0,
                entry_price=float(bars[-1].close or 0.0),
                reason="shared entry quality",
                scan_result=ScanResult(
                    symbol=symbol,
                    scanner_name=scanner_name or source or "shared_entry_quality",
                    ts=datetime.now(timezone.utc),
                    score=0.0,
                    criteria=inferred_criteria,
                ),
            )
        return executor.evaluate_quality(
            signal,
            bars=bars,
            stage=stage,
            source=source,
            min_day_change_pct=0.0,
            avg_daily_volume=avg_daily_volume,
            bars_5m=bars_5m,
            float_shares=float_shares,
            ticks=list(self._tick_buffer.get(symbol, [])),
            quotes=list(self._quote_buffer.get(symbol, [])),
        )

    @staticmethod
    def _resolve_entry_float_shares(float_checker: object, symbol: str) -> Optional[float]:
        """Resolve float for final entry guard without network latency.

        FloatChecker.get_float_cached() checks memory and the SQLite FloatStore,
        but deliberately avoids Yahoo/network fetches on the hot submit path.
        """
        cached = getattr(float_checker, "get_float_cached", None)
        if callable(cached):
            try:
                return cached(symbol)
            except Exception:
                logger.debug("Cached float lookup failed for %s before entry guard", symbol)
        return None

    def _execute_timed_scale_up(self, signal: TradeSignal) -> None:
        """Execute a protected-runner re-add released by the 10s timer."""
        _ensure_entry_executor(self).execute_timed_scale_up(
            self._entry_execution_context(),
            signal,
        )

    @staticmethod
    def _is_timed_scout_signal(signal: TradeSignal) -> bool:
        reason = str(signal.reason or "")
        return "continuation_scout" in reason or "pullback_scout" in reason

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
            "pullback_base",
            "level_breakout_reclaim",
            "shallow_stair_continuation",
            "early_vwap_reclaim_scout",
        }
        if pattern not in hot_patterns:
            if hit.scanner_name not in hot_patterns:
                return False
        price = float(hit.criteria.get("close") or signal.entry_price or 0.0)
        if not (1.5 <= price <= 20.0):
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
            try:
                criteria = signal.scan_result.criteria if signal.scan_result is not None else {}
                spread_size_factor = float(criteria.get("spread_size_factor") or 1.0)
            except (AttributeError, TypeError, ValueError):
                spread_size_factor = 1.0
            quantity = signal.quantity
            if 0 < spread_size_factor < 1.0 and quantity > 1:
                quantity = float(max(1, int(float(quantity) * spread_size_factor)))
            retry_order = Order(
                symbol=signal.symbol,
                side=Side.BUY,
                quantity=quantity,
                limit_price=live,
            )
            quality_reject = self._shared_entry_quality_reject(
                signal.symbol, list(self._bar_buffer.get(signal.symbol, deque())), signal=signal,
                stage="timed_hot_retry_final_guard",
                source="timed_entry_hot_retry",
            )
            if quality_reject:
                logger.info(
                    "TIMED ENTRY hot retry skipped %s — shared entry quality %s",
                    signal.symbol, quality_reject,
                )
                return None, first_status
            logger.info(
                "TIMED ENTRY hot retry %s %.0f @ %.4f (signal %.4f, cap %.1f%%)",
                signal.symbol, quantity, live, original, max_chase_pct * 100,
            )
            fill, status = self._broker.submit(retry_order, retry_bar, self._pipeline.portfolio)
            try:
                from daytrading.ml.shadow_collector import log_execution_quality
                log_execution_quality(
                    order=retry_order, bar=retry_bar, status=status, fill=fill,
                    source="timed_entry_hot_retry",
                    context=self._execution_learning_context(signal, retry_bar),
                )
            except Exception:
                pass
            return fill, status
        except Exception as exc:
            logger.warning("TIMED ENTRY hot retry failed %s: %s", signal.symbol, exc)
            return None, first_status

    def _current_equity(self) -> float:
        """Live account equity, cached ~60s; falls back to configured equity.

        Reading the broker account on every entry would be slow, so we cache it
        and refresh at most once a minute.
        """
        now = time.monotonic()
        cached = float(getattr(self, "_account_equity", 0.0) or 0.0)
        last = float(getattr(self, "_account_equity_at", 0.0) or 0.0)
        if cached > 0 and (now - last) < 60.0:
            return cached
        try:
            eq = float((self._broker.get_account() or {}).get("equity") or 0.0)
            if eq > 0:
                self._account_equity = eq
                self._account_equity_at = now
                return eq
        except Exception:
            pass
        return cached if cached > 0 else float(getattr(self, "_fallback_equity", 2000.0) or 2000.0)

    def _capital_aware_quantity(
        self,
        price: float,
        stop_price: float,
        *,
        max_dollar_risk: Optional[float] = None,
    ) -> int:
        """Shares so the trade risks ~risk_pct of equity and the position fits
        buying power. Returns 0 when even a safe minimum can't be afforded so
        the caller skips the trade rather than over-leveraging.
        """
        price = float(price or 0.0)
        if price <= 0:
            return 0
        risk_pct = float(getattr(self, "_risk_pct_of_equity", 0.0) or 0.0)
        equity = self._current_equity()
        risk_per_share = price - float(stop_price or 0.0)
        if risk_pct <= 0:
            # Capital-aware sizing disabled — keep the legacy fixed-$ behavior.
            risk_dollars = 50.0
        else:
            risk_dollars = max(float(getattr(self, "_min_risk_dollars", 5.0) or 0.0), equity * risk_pct)
        if max_dollar_risk is not None:
            cap = float(max_dollar_risk or 0.0)
            if cap > 0:
                risk_dollars = min(risk_dollars, cap)
        qty_by_risk = int(risk_dollars / risk_per_share) if risk_per_share > 0 else 0
        max_pos_value = equity * float(getattr(self, "_max_position_pct_of_equity", 1.0) or 1.0)
        qty_by_capital = int(max_pos_value / price) if price > 0 else 0
        qty = min(qty_by_risk, qty_by_capital) if qty_by_capital > 0 else qty_by_risk
        return max(0, int(qty))

    def _process_breakout_scalps(self) -> None:
        """Process pending breakout scalps queued by HOD tick alerts.

        Fast entry for HOD/tape movers, but still gated by the normal entry
        quality/ML layer and fresh 10s confirmation before any order is sent.
        The goal is quick profit on explosive tape with smaller size and a
        short hold window, without bypassing the shared safety checks.
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

            alert_reject = self._quick_scalp_hod_alert_reject(sym)
            if alert_reject is not None:
                logger.info("BREAKOUT SCALP reject %s: %s", sym, alert_reject)
                self._remember_quick_scalp_reject(sym, alert_reject)
                probe_signal = self._quick_scalp_probe_signal(sym, alert_price, "breakout_scalp")
                self._record_entry_reject(
                    probe_signal,
                    stage="breakout_scalp_alert",
                    reason=alert_reject,
                    source="breakout_scalp",
                    price=alert_price,
                )
                continue

            recent_reject = self._quick_scalp_recent_normal_reject(
                sym,
                allow_fresh_hod_breakout=True,
                require_clean_reclaim=True,
            )
            if recent_reject is not None:
                logger.info("BREAKOUT SCALP reject %s: %s", sym, recent_reject)
                self._remember_quick_scalp_reject(sym, recent_reject)
                probe_signal = self._quick_scalp_probe_signal(sym, alert_price, "breakout_scalp")
                self._record_entry_reject(
                    probe_signal,
                    stage="breakout_scalp_recent_reject",
                    reason=recent_reject,
                    source="breakout_scalp",
                    price=alert_price,
                )
                continue

            reject = self._check_quick_scalp_entry(sym, bars)
            if reject is not None:
                logger.info("BREAKOUT SCALP reject %s: %s", sym, reject)
                self._remember_quick_scalp_reject(sym, reject)
                probe_signal = self._quick_scalp_probe_signal(sym, bars[-1].close, "breakout_scalp")
                self._record_entry_reject(
                    probe_signal,
                    stage="breakout_scalp_shape",
                    reason=reject,
                    source="breakout_scalp",
                    price=bars[-1].close,
                )
                continue

            quality_reject = self._quick_scalp_shared_quality_reject(sym, bars)
            if quality_reject is not None:
                logger.info("BREAKOUT SCALP reject %s: shared entry quality %s", sym, quality_reject)
                continue

            ten_second_reject = self._breakout_scalp_10s_reject(sym)
            if ten_second_reject is not None:
                logger.info("BREAKOUT SCALP reject %s: %s", sym, ten_second_reject)
                self._remember_quick_scalp_reject(sym, ten_second_reject)
                probe_signal = self._quick_scalp_probe_signal(sym, bars[-1].close, "breakout_scalp")
                self._record_entry_reject(
                    probe_signal,
                    stage="breakout_scalp_10s",
                    reason=ten_second_reject,
                    source="breakout_scalp",
                    price=bars[-1].close,
                )
                continue

            rr = self._quick_scalp_tick_rr(sym, bars, alert_price)
            if rr is None:
                logger.info("BREAKOUT SCALP reject %s: no usable tick R:R", sym)
                self._remember_quick_scalp_reject(sym, "no usable tick R:R")
                probe_signal = self._quick_scalp_probe_signal(sym, bars[-1].close, "breakout_scalp")
                self._record_entry_reject(
                    probe_signal,
                    stage="breakout_scalp_rr",
                    reason="no usable tick R:R",
                    source="breakout_scalp",
                    price=bars[-1].close,
                )
                continue
            price, stop_price, target_price, rr_note = rr
            risk_per_share = price - stop_price

            quantity = self._capital_aware_quantity(price, stop_price)
            if quantity < 1:
                logger.info("BREAKOUT SCALP skip %s — position too large for buying power", sym)
                continue
            spread_size_factor = float(
                getattr(self, "_quick_scalp_spread_size_factors", {}).pop(sym, 1.0)
            )
            if 0 < spread_size_factor < 1.0 and quantity > 1:
                original_quantity = quantity
                quantity = max(1, int(quantity * spread_size_factor))
                logger.info(
                    "BREAKOUT SCALP %s size down %d → %d for opportunity-scaled spread",
                    sym,
                    original_quantity,
                    quantity,
                )

            # Tag entries that the experimental momentum-breakout bypass allowed,
            # so the scorecard can measure that mode's standalone expectancy.
            mb_armed = self._momentum_breakout_consume(sym)
            scanner_name = "breakout_scalp_momentum" if mb_armed else "breakout_scalp"
            fill_strategy = scanner_name

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
                    symbol=sym, scanner_name=scanner_name,
                    ts=datetime.now(timezone.utc), score=0.0,
                    criteria={
                        "pattern": "breakout_scalp",
                        "direction": "up",
                        **({"entry_mode": "momentum_breakout"} if mb_armed else {}),
                        **(
                            {
                                "spread_exception": "opportunity_scaled",
                                "spread_size_factor": round(spread_size_factor, 2),
                            }
                            if 0 < spread_size_factor < 1.0 else {}
                        ),
                    },
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
                    signal, fill, strategy=fill_strategy,
                    execution_method="instant_breakout",
                )
                self._breakout_scalp_active = True
                self._breakout_scalp_cooldown[sym] = now_mono + 300.0
                self._pipeline._symbol_entry_counts[sym] = self._pipeline._symbol_entry_counts.get(sym, 0) + 1
                logger.info(
                    "QUICK SCALP ENTRY %s %.0f @ $%.4f stop=$%.2f target=$%.2f %s",
                    sym, fill.quantity, fill.price, stop_price, target_price, rr_note,
                )
                self._hub.on_fill(fill, "entry", strategy=fill_strategy)
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
                    "strategy": fill_strategy,
                    "execution_method": "instant_breakout",
                    "market_context": {"phase": self._market_phase()},
                }, ts=fill.ts)
                self._seed_recent_order_ids()
                break
            logger.warning("BREAKOUT SCALP order not filled %s (status=%s)", sym, status)
            self._record_entry_reject(
                signal,
                stage="breakout_scalp_order",
                reason="order_{}".format(status.value if status else "not_filled"),
                source="breakout_scalp",
                price=price,
                metadata={"status": status.value if status else "not_filled"},
            )

    def _quick_scalp_hod_alert_reject(self, symbol: str) -> Optional[str]:
        """Block instant HOD scalps when the latest alert is watch-only/rejected."""
        store = getattr(self, "_hod_alert_store", None)
        if store is None:
            return None
        try:
            rows = store.snapshot()
        except Exception:
            return None
        if not isinstance(rows, list):
            return None

        sym = symbol.upper()
        latest = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol", "")).upper() != sym:
                continue
            latest = row
            break
        if latest is None:
            return None

        reason = latest.get("reject_reason")
        if reason:
            text = str(reason)
            lower = text.lower()
            if (
                "watch only" in lower
                and "momentum_burst" in lower
                and self._quick_scalp_allows_extreme_hod_runner_alert(latest)
            ):
                logger.info(
                    "BREAKOUT SCALP %s: extreme HOD runner alert promoted past "
                    "watch-only prefilter; shared ML/entry guard and 10s "
                    "confirmation still required",
                    sym,
                )
            else:
                hard_terms = (
                    "watch only",
                    "cached reject",
                    "selling pressure",
                    "tape too slow",
                    "weak",
                    "too far from hod",
                    "thin liquidity",
                    "below vwap",
                    "spread too wide",
                )
                if any(term in lower for term in hard_terms):
                    return "HOD alert not tradeable: {}".format(text)

        try:
            rel_vol = float(latest.get("rel_vol") or 0.0)
            bar_rvol = float(latest.get("bar_rvol") or 0.0)
        except (TypeError, ValueError):
            return None
        active_rvol = max(rel_vol, bar_rvol)
        if active_rvol > 0 and active_rvol < 1.0:
            # EXPERIMENTAL momentum-breakout mode: a real breakout can run on
            # high ABSOLUTE volume even when relative volume has faded from an
            # earlier peak (the VSME case). Allow it through — tagged for the
            # scorecard — instead of rejecting on relative volume alone. Still
            # subject to quick-scalp structure, shared quality, 10s, and R:R.
            if getattr(self, "_momentum_breakout_enabled", False):
                day_vol = 0.0
                try:
                    day_vol = float(latest.get("day_volume") or 0.0)
                except (TypeError, ValueError):
                    day_vol = 0.0
                # Caveat-2 fix: only allow if recent tape is smooth enough that a
                # tight stop actually holds. Violent gappy breakouts (VSME-style
                # 6-11% bars) are skipped — that's where the stop slips and the
                # edge dies. No bar data → can't assess → fall through to reject.
                smooth = self._momentum_breakout_tape_is_smooth(sym)
                if (
                    active_rvol >= self._momentum_breakout_min_rvol
                    and day_vol >= self._momentum_breakout_min_day_volume
                    and smooth
                ):
                    self._momentum_breakout_armed[sym] = time.monotonic()
                    logger.info(
                        "MOMENTUM BREAKOUT %s: rvol %.2fx faded but day_vol %.0f high "
                        "and tape smooth — allowing (experimental mode)",
                        sym, active_rvol, day_vol,
                    )
                    return None
            return "HOD alert active RVOL too weak {:.2f}x (need 1.0x+)".format(active_rvol)

        return None

    @staticmethod
    def _quick_scalp_allows_extreme_hod_runner_alert(row: dict) -> bool:
        """Let elite HOD runner alerts reach the real entry gates.

        This does not make momentum_burst a live scanner. It only prevents the
        fast HOD scalp path from stopping at the alert label when the alert has
        extreme runner evidence; the path still runs quick-scalp structure,
        shared ML/rule entry quality, 10s confirmation, and R:R before submit.
        """
        try:
            price = float(row.get("price") or 0.0)
            change_session_pct = float(row.get("change_session_pct") or 0.0)
            change_from_close_pct = float(row.get("change_from_close_pct") or 0.0)
            day_volume = float(row.get("day_volume") or 0.0)
            rel_vol = float(row.get("rel_vol") or 0.0)
            bar_rvol = float(row.get("bar_rvol") or 0.0)
            float_shares = float(row.get("float_shares") or 0.0)
        except (TypeError, ValueError):
            return False

        if not (1.5 <= price <= 20.0):
            return False
        if day_volume < 2_000_000:
            return False
        if float_shares > 0 and float_shares > 20_000_000:
            return False
        if max(rel_vol, bar_rvol) < 1.0:
            return False
        return change_session_pct >= 80.0 or change_from_close_pct >= 120.0

    def _momentum_breakout_tape_is_smooth(self, symbol: str) -> bool:
        """True when recent bars are tight enough that a tight stop holds.

        Median per-bar range must be within the configured cap. Violent gappy
        tape (where stops slip) returns False; missing bar data also returns
        False (cannot confirm safety -> do not fire the experimental entry).
        """
        try:
            bars = list(getattr(self, "_bar_buffer", {}).get(symbol.upper(), []))
        except Exception:
            return False
        recent = [b for b in bars[-6:] if float(getattr(b, "close", 0) or 0) > 0]
        if len(recent) < 3:
            return False
        ranges = sorted(
            (float(b.high) - float(b.low)) / float(b.close) * 100.0 for b in recent
        )
        median = ranges[len(ranges) // 2]
        return median <= self._momentum_breakout_max_bar_range_pct

    def _momentum_breakout_consume(self, symbol: str, ttl_sec: float = 5.0) -> bool:
        """Return True (and clear the marker) when the experimental breakout
        bypass fired for this symbol this cycle — so the resulting entry can be
        tagged for isolated scorecard measurement.
        """
        armed = getattr(self, "_momentum_breakout_armed", None)
        if not armed:
            return False
        fired_at = armed.pop(symbol, None)
        if fired_at is None:
            return False
        return (time.monotonic() - fired_at) <= ttl_sec

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

    def _quick_scalp_has_tradeable_hod_alert(self, symbol: str) -> bool:
        store = getattr(self, "_hod_alert_store", None)
        if store is None:
            return False
        try:
            rows = store.snapshot()
        except Exception:
            return False
        sym = symbol.upper()
        for row in rows or []:
            if str(row.get("symbol") or "").upper() != sym:
                continue
            reject_reason = str(row.get("reject_reason") or "").strip()
            if reject_reason and not reject_reason.lower().startswith("watch only"):
                return False
            try:
                price = float(row.get("price") or 0.0)
                day_volume = float(row.get("day_volume") or 0.0)
                rel_vol = float(row.get("rel_vol") or 0.0)
                bar_rvol = float(row.get("bar_rvol") or 0.0)
            except Exception:
                price = day_volume = rel_vol = bar_rvol = 0.0
            if price > 0 and day_volume >= 1_000_000 and max(rel_vol, bar_rvol) >= 1.0:
                return True
            # Some tick HOD rows can arrive without normalized RVOL, but a
            # cleared reject_reason means the alert gate already accepted them.
            if not reject_reason and price > 0 and day_volume >= 5_000_000:
                return True
        return False

    @staticmethod
    def _quick_scalp_can_ignore_recent_shape_reject(reason: str) -> bool:
        lower = str(reason or "").lower()
        stale_shape_terms = (
            "too far from hod",
            "pullback too small",
            "base range too wide",
            "watching for fresh reclaim",
        )
        return any(term in lower for term in stale_shape_terms)

    def _quick_scalp_recent_normal_reject(
        self,
        symbol: str,
        *,
        allow_fresh_hod_breakout: bool = False,
        require_clean_reclaim: bool = False,
    ) -> Optional[str]:
        """Block quick scalps when the regular entry path just saw a hard reject."""
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is None:
            return None
        rejections = None
        try:
            rejections = getattr(pipeline, "scan_rejections", None)
        except Exception:
            rejections = None
        if callable(rejections):
            try:
                rejections = rejections()
            except Exception:
                rejections = None
        if rejections is None:
            rejections = getattr(pipeline, "_scan_rejections", None)
        if not isinstance(rejections, dict):
            return None

        reason = rejections.get(symbol)
        if not reason:
            reason = rejections.get(symbol.upper())
        if not reason:
            return None

        lower = str(reason).lower()
        if (
            allow_fresh_hod_breakout
            and self._quick_scalp_can_ignore_recent_shape_reject(reason)
            and self._quick_scalp_has_tradeable_hod_alert(symbol)
        ):
            if require_clean_reclaim:
                continuation_ok, continuation_reason, _continuation_meta = (
                    self._momentum_burst_continuation_base_ok(symbol)
                )
                if not continuation_ok:
                    return (
                        "fresh HOD breakout needs clean 10s reclaim after recent reject: "
                        "{}".format(continuation_reason)
                    )
            logger.info(
                "BREAKOUT SCALP ignore stale normal reject %s: %s",
                symbol,
                reason,
            )
            return None

        hard_terms = (
            "spread too wide",
            "late continuation too weak",
            "pullback too small",
            "below vwap",
            "tape too slow",
            "selling pressure",
            "red volume too heavy",
            "weak reclaim volume",
            "too far from hod",
            "not strong above vwap",
            "dead price action",
            "dead cat bounce",
            "thin liquidity",
            "watch-only liquidity",
            "low day volume",
            "outside range",
            "stale data",
        )
        if any(term in lower for term in hard_terms):
            return "recent normal entry reject: {}".format(reason)
        return None

    def _quick_scalp_shared_quality_reject(
        self,
        symbol: str,
        bars: Sequence[Bar],
    ) -> Optional[str]:
        """Run quick scalps through the shared entry guard and ML monitor."""
        reject = self._shared_entry_quality_reject(
            symbol,
            bars,
            stage="breakout_scalp_final_guard",
            source="breakout_scalp",
        )
        # Caveat-1 fix (experimental momentum-breakout mode): catch the early
        # breakout/reclaim that scores just under the 80 gate — but ONLY when the
        # tape is smooth (stop holds). Scoped to the breakout-scalp path.
        if (
            reject
            and getattr(self, "_momentum_breakout_enabled", False)
            and "entry score too low" in reject
        ):
            score_val = None
            try:
                score_val = int(reject.split("(", 1)[1].split("/100", 1)[0])
            except (IndexError, ValueError):
                score_val = None
            if (
                score_val is not None
                and score_val >= self._momentum_breakout_score_floor
                and self._momentum_breakout_tape_is_smooth(symbol)
            ):
                self._momentum_breakout_armed[symbol] = time.monotonic()
                logger.info(
                    "MOMENTUM BREAKOUT %s: score %d below 80 but >= floor and tape "
                    "smooth — allowing (experimental mode)",
                    symbol, score_val,
                )
                return None
        return reject

    def _warrior_recent_bad_tape_reject(self, symbol: str) -> Optional[str]:
        """Return a recent same-symbol normal-path bad-tape reject.

        Warrior is allowed to be a faster path, but it should not buy the next
        green tick immediately after the standard scanners saw distribution.
        When this fires, the Warrior loop keeps watching and requires a fresh
        10s micro-base/reclaim instead of taking the current bar.
        """
        recent_reject = self._quick_scalp_recent_normal_reject(symbol)
        bad_tape_terms = (
            "selling pressure",
            "dump candle",
            "weak reclaim volume",
            "red volume too heavy",
            "dead price action",
            "dead cat bounce",
        )
        if recent_reject and any(term in recent_reject.lower() for term in bad_tape_terms):
            return recent_reject
        quick_reject = self._recent_quick_scalp_rejects.get(str(symbol or "").upper())
        if quick_reject:
            ts, reason = quick_reject
            age = time.monotonic() - float(ts or 0.0)
            quick_tape_terms = (
                "active rvol too weak",
                "selling pressure",
                "weak reclaim volume",
                "volume faded",
                "red 10s",
                "dump candle",
                "no fresh 10s expansion",
                "confirmation faded",
                "confirmation weak close",
                "too volatile without strong close",
            )
            if 0.0 <= age <= 120.0 and any(term in str(reason).lower() for term in quick_tape_terms):
                return "recent breakout scalp reject: {}".format(reason)
        return None

    def _remember_quick_scalp_reject(self, symbol: str, reason: str) -> None:
        symbol_key = str(symbol or "").upper()
        reason_text = str(reason or "")
        if not symbol_key or not reason_text:
            return
        self._recent_quick_scalp_rejects[symbol_key] = (time.monotonic(), reason_text)

    @staticmethod
    def _quick_scalp_probe_signal(
        symbol: str,
        price: float,
        pattern: str,
    ) -> TradeSignal:
        return TradeSignal(
            symbol=symbol,
            action=SignalAction.ENTER_LONG,
            quantity=0,
            entry_price=float(price or 0.0),
            reason=pattern,
            scan_result=ScanResult(
                symbol=symbol,
                scanner_name=pattern,
                ts=datetime.now(timezone.utc),
                score=0.0,
                criteria={"pattern": pattern, "setup_tier": "A+ setup"},
            ),
        )

    def _maybe_arm_momentum_burst_scalp(self, hit: ScanResult) -> None:
        """Arm a fixed monitor window from a momentum_burst scanner hit."""
        if not (
            getattr(self, "_momentum_burst_cycle_enabled", False)
            or getattr(self, "_momentum_burst_hit_run_enabled", False)
            or getattr(self, "_warrior_squeeze_enabled", False)
        ):
            return
        pattern = str((hit.criteria or {}).get("pattern") or hit.scanner_name or "")
        warrior_enabled = bool(getattr(self, "_warrior_squeeze_enabled", False))
        if hit.scanner_name != "momentum_burst" and pattern != "momentum_burst":
            return
        sym = hit.symbol.upper()
        high = 0.0
        try:
            if hit.bars:
                high = max(float(bar.high or 0.0) for bar in hit.bars[-3:])
        except Exception:
            high = 0.0
        if high <= 0:
            try:
                high = float((hit.criteria or {}).get("close") or 0.0)
            except (TypeError, ValueError):
                high = 0.0
        if high <= 0:
            return
        if warrior_enabled:
            should_arm, reason = self._warrior_squeeze_should_arm(hit, high)
            if not should_arm:
                logger.info("WARRIOR SQUEEZE watch %s: %s", sym, reason)
                reject_high = float(self._warrior_squeeze_rejection_high.get(sym, 0.0) or 0.0)
                if reject_high > 0 and self._ensure_warrior_watch_capacity(sym, max(high, reject_high)):
                    now_mono = time.monotonic()
                    self._momentum_burst_armed.setdefault(sym, now_mono)
                    self._momentum_burst_window_high[sym] = max(
                        high,
                        reject_high,
                        float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0),
                    )
                    self._momentum_burst_session_anchor_high.setdefault(sym, max(high, reject_high))
                    logger.info(
                        "WARRIOR SQUEEZE watch-only armed %s after %s",
                        sym,
                        reason,
                    )
                return
        if (
            (
                getattr(self, "_momentum_burst_hit_run_enabled", False)
                or getattr(self, "_warrior_squeeze_enabled", False)
            )
            and sym in getattr(self, "_momentum_burst_hit_run_day_blocked", {})
        ):
            logger.info(
                "MOMENTUM BURST HIT-RUN not arming %s — %s",
                sym,
                self._momentum_burst_hit_run_day_blocked.get(sym),
            )
            return
        now_mono = time.monotonic()
        if warrior_enabled and not self._ensure_warrior_watch_capacity(sym, high):
            return
        self._momentum_burst_armed[sym] = now_mono
        self._momentum_burst_window_high[sym] = max(
            high,
            float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0),
        )
        self._momentum_burst_session_anchor_high.setdefault(sym, high)
        logger.info(
            "MOMENTUM BURST SCALP armed %s for %.0fs above $%.4f",
            sym,
            self._momentum_burst_window_sec,
            self._momentum_burst_window_high[sym],
        )

    def _warrior_squeeze_should_arm(self, hit: ScanResult, high: float) -> tuple[bool, str]:
        """Classify a momentum popup for the separate Warrior squeeze playbook.

        The scanner is attention only. This playbook ignores the first cheap or
        ugly spike, records its rejection high, and only arms after a later
        reclaim proves the stock is still squeezing.
        """
        sym = hit.symbol.upper()
        bars = list(hit.bars or [])
        latest = bars[-1] if bars else None
        min_price = max(0.0, float(getattr(self, "_warrior_squeeze_min_reclaim_price", 3.5) or 0.0))
        reject_high = float(self._warrior_squeeze_rejection_high.get(sym, 0.0) or 0.0)
        if latest is not None:
            close = float(latest.close or 0.0)
            open_ = float(latest.open or 0.0)
            low = float(latest.low or 0.0)
            bar_high = float(latest.high or high or 0.0)
            rng = max(bar_high - low, 0.0)
            upper_wick = (bar_high - max(open_, close)) / rng if rng > 0 else 0.0
            is_red = close < open_
            prior_vol = [float(b.volume or 0.0) for b in bars[-6:-1]]
            avg_prior = sum(prior_vol) / len(prior_vol) if prior_vol else 0.0
            high_volume_reject = (
                is_red
                and upper_wick >= 0.45
                and float(latest.volume or 0.0) >= max(75_000.0, avg_prior * 1.2)
            )
            if high < min_price and reject_high <= 0:
                self._warrior_squeeze_rejection_high[sym] = max(reject_high, bar_high)
                self._warrior_squeeze_rejection_reason[sym] = "first cheap spike under ${:.2f}".format(min_price)
                return False, self._warrior_squeeze_rejection_reason[sym]
            if high_volume_reject:
                self._warrior_squeeze_rejection_high[sym] = max(reject_high, bar_high)
                self._warrior_squeeze_rejection_reason[sym] = "high-volume shooting-star rejection"
                return False, self._warrior_squeeze_rejection_reason[sym]
        if reject_high > 0:
            reclaim_level = max(reject_high * 1.03, min_price)
            if high < reclaim_level:
                return False, "waiting for reclaim above rejected high ${:.2f}".format(reject_high)
            return True, "reclaimed rejected high ${:.2f}".format(reject_high)
        if high < min_price:
            return False, "waiting for squeeze above ${:.2f}".format(min_price)
        return True, "proved squeeze above ${:.2f}".format(min_price)

    def _maybe_arm_warrior_squeeze_from_10s(
        self,
        symbol: str,
        latest_10s: Bar,
        now_mono: float,
    ) -> None:
        """Arm Warrior mode from 10s tape before 1m scanners catch up."""
        sym = symbol.upper()
        if sym in self._momentum_burst_armed:
            return
        if sym in self._momentum_burst_hit_run_day_blocked:
            return
        open_ = float(latest_10s.open or 0.0)
        high = float(latest_10s.high or 0.0)
        low = float(latest_10s.low or 0.0)
        close = float(latest_10s.close or 0.0)
        volume = float(latest_10s.volume or 0.0)
        if open_ <= 0 or high <= 0 or close <= 0:
            return
        min_price = max(0.0, float(getattr(self, "_warrior_squeeze_min_reclaim_price", 3.5) or 0.0))
        reject_high = float(self._warrior_squeeze_rejection_high.get(sym, 0.0) or 0.0)
        if high < min_price and reject_high <= 0:
            self._warrior_squeeze_rejection_high[sym] = high
            self._warrior_squeeze_rejection_reason[sym] = (
                "first cheap 10s spike under ${:.2f}".format(min_price)
            )
            if self._ensure_warrior_watch_capacity(sym, high):
                self._momentum_burst_armed.setdefault(sym, now_mono)
                self._momentum_burst_window_high[sym] = max(
                    high,
                    float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0),
                )
                self._momentum_burst_session_anchor_high.setdefault(sym, high)
            return
        first_pullback_context = self._warrior_trend_pullback_reclaim_context(
            sym,
            latest_10s,
            window_high=max(
                high,
                float(self._warrior_squeeze_rejection_high.get(sym, 0.0) or 0.0),
            ),
        )
        if (
            first_pullback_context is not None
            and warrior_lanes.is_warrior_initial_starter_trigger(
                first_pullback_context.get("entry_trigger")
            )
        ):
            if not self._ensure_warrior_watch_capacity(sym, max(high, close)):
                return
            self._momentum_burst_armed[sym] = now_mono
            self._momentum_burst_window_high[sym] = max(
                high,
                float(self._warrior_squeeze_rejection_high.get(sym, 0.0) or 0.0),
            )
            self._momentum_burst_session_anchor_high.setdefault(sym, high)
            bar_ts = getattr(latest_10s, "ts", None)
            pending_ts = (
                bar_ts - timedelta(seconds=10)
                if isinstance(bar_ts, datetime)
                else datetime.now(timezone.utc) - timedelta(seconds=10)
            )
            pending_breakout_high = high
            if first_pullback_context.get("entry_trigger") == "warrior_low_price_proof_reclaim":
                pending_breakout_high = float(
                    first_pullback_context.get("pullaway_level")
                    or first_pullback_context.get("psych_level")
                    or high
                )
            self._momentum_burst_pending[sym] = {
                "ts": pending_ts,
                "breakout_close": close,
                "breakout_high": pending_breakout_high,
                "breakout_volume": volume,
                **first_pullback_context,
            }
            logger.info(
                "WARRIOR SQUEEZE %s initial 10s starter armed below blue-sky trigger",
                sym,
            )
            return
        if reject_high > 0:
            reclaim_level = max(reject_high * 1.03, min_price)
            if high < reclaim_level:
                return
            if not self._ensure_warrior_watch_capacity(sym, max(high, reject_high)):
                return
            self._momentum_burst_armed[sym] = now_mono
            self._momentum_burst_window_high[sym] = max(high, reject_high)
            self._momentum_burst_session_anchor_high.setdefault(sym, high)
            bar_ts = getattr(latest_10s, "ts", None)
            pending_ts = (
                bar_ts - timedelta(seconds=10)
                if isinstance(bar_ts, datetime)
                else datetime.now(timezone.utc) - timedelta(seconds=10)
            )
            self._momentum_burst_pending[sym] = {
                "ts": pending_ts,
                "breakout_close": close,
                "breakout_high": high,
                "breakout_volume": volume,
                "entry_trigger": "warrior_a_plus_reclaim",
            }
            return

        if volume < 25_000:
            return
        impulse_pct = (close - open_) / open_
        range_pct = (high - low) / close if close > 0 else 0.0
        if impulse_pct < 0.08 and range_pct < 0.10:
            return
        self._warrior_squeeze_rejection_high[sym] = high
        self._warrior_squeeze_rejection_reason[sym] = "first explosive 10s spike"

    @staticmethod
    def _warrior_ignition_active() -> bool:
        return os.environ.get("DAYTRADING_WARRIOR_IGNITION", "").strip().lower() in (
            "1", "true", "yes", "on",
        )

    def _maybe_enter_warrior_ignition(self) -> None:
        """Learned premarket-ignition path (paper). Score base-breakout ignitions,
        log every candidate for the adaptive dataset, and place orders sized by the
        model's runner-conviction through the shared Warrior execution (which keeps
        all the live risk caps, entry-guard, ML and position bookkeeping)."""
        if self._market_phase() != "PRE-MARKET":
            return
        if self._new_entries_blocked(None, "WARRIOR IGNITION"):
            return
        try:
            from daytrading.strategy.warrior_ignition import (
                detect_ignition,
                get_model,
                ignition_suppression_reason,
                prior_day_high,
            )
            from daytrading.strategy.warrior_ignition_log import get_logger
        except Exception:
            return
        model = get_model()
        ig_logger = get_logger()
        counts = getattr(self, "_warrior_ignition_entries", None)
        if counts is None:
            counts = {}
            self._warrior_ignition_entries = counts
        failed_counts = getattr(self, "_warrior_ignition_failed_entries", None)
        if failed_counts is None:
            failed_counts = {}
            self._warrior_ignition_failed_entries = failed_counts
        peak_prices = getattr(self, "_warrior_ignition_peak_price", None)
        if peak_prices is None:
            peak_prices = {}
            self._warrior_ignition_peak_price = peak_prices
        peak_moves = getattr(self, "_warrior_ignition_peak_day_move", None)
        if peak_moves is None:
            peak_moves = {}
            self._warrior_ignition_peak_day_move = peak_moves
        engine = getattr(self, "_warrior_engine", None)
        # LIVE bar source: the aggregator's rolling 10s buffer, keyed by the
        # symbols we are actively tracking this cycle (watchlist + hot-watch +
        # protected). NOTE: ``_timer_bars_by_symbol`` is a BACKTEST-ONLY
        # structure (set in backtest/driver.py); it is always empty in the live
        # runner, so iterating it here scored nothing and never traded.
        aggregator = getattr(self, "_bar_aggregator", None)
        if aggregator is None:
            return
        watch = getattr(self, "_warrior_ignition_watch", None)
        if watch is None:
            watch = {}
            self._warrior_ignition_watch = watch
        now_mono = time.monotonic()
        entered = 0
        max_per_cycle = 3      # was 1 (hard break) — let several real ignitions fire
        floor = 0.20

        def _mark(symu, st, conv, px, stp, sz, why):
            # visible, real-time watch record on the ignition's OWN surface — it
            # deliberately does NOT touch the WarriorWatchBook the squeeze lanes read.
            watch[symu] = {
                "symbol": symu, "state": st, "conviction": round(float(conv), 3),
                "entry_ref": round(float(px), 4), "stop": round(float(stp), 4),
                "size_factor": round(float(sz), 2), "reason": why, "ts": now_mono,
            }

        for sym in sorted(self._trade_symbol_set()):
            symu = sym.upper()
            try:
                raw_bars = aggregator.get_10s_bars(sym)
            except Exception:
                continue
            bars = [b for b in (raw_bars or []) if float(getattr(b, "close", 0.0) or 0.0) > 0]
            if len(bars) < 24:
                continue
            latest_10s = bars[-1]
            try:
                ph = prior_day_high(sym, latest_10s.ts.date())
                sig = detect_ignition(bars, model, prior_high=ph)
            except Exception:
                continue
            price = float(latest_10s.close or 0.0)
            if not sig.detected:
                # surface near-misses (a real base, in band) so you can SEE what is
                # forming but not yet igniting; skip pure noise (warming/out-of-band).
                if sig.reject in ("no ignition (gate)", "bad base"):
                    _mark(symu, "no_ignition", 0.0, price, 0.0, 0.0, sig.reject)
                continue
            conv = float(sig.conviction)
            size_factor = sig.size_factor(model.cutoff)
            stop = float(sig.stop)
            suppress_reason = ""
            if conv >= floor:
                peak_prices[symu] = max(
                    float(peak_prices.get(symu, 0.0) or 0.0),
                    float(sig.entry_ref or price or 0.0),
                )
                peak_moves[symu] = max(
                    float(peak_moves.get(symu, 0.0) or 0.0),
                    float(sig.features.get("day_move", 0.0) or 0.0),
                )
                suppress_reason = ignition_suppression_reason(
                    sig,
                    failed_entries=int(failed_counts.get(symu, 0) or 0),
                    peak_price=float(peak_prices.get(symu, 0.0) or 0.0),
                    peak_day_move=float(peak_moves.get(symu, 0.0) or 0.0),
                )
            try:
                ig_logger.log(
                    ts_iso=latest_10s.ts.isoformat(), symbol=sym, signal=sig,
                    would_enter=conv >= floor and not suppress_reason, size_factor=size_factor,
                )
            except Exception:
                pass
            # record a visible state + reason for EVERY detected ignition
            if conv < floor:
                _mark(symu, "low_conviction", conv, price, stop, size_factor,
                      "conv {:.2f} < {:.2f} floor".format(conv, floor))
            else:
                if suppress_reason:
                    _mark(symu, "suppressed", conv, price, stop, size_factor, suppress_reason)
            if conv < floor:
                pass
            elif suppress_reason:
                pass
            elif counts.get(symu, 0) >= 4:
                _mark(symu, "capped", conv, price, stop, size_factor, "per-symbol cap (4 entries)")
            elif not (price > stop > 0):
                _mark(symu, "bad_stop", conv, price, stop, size_factor, "stop not below price")
            elif entered >= max_per_cycle:
                _mark(symu, "queued", conv, price, stop, size_factor, "entry slot taken this cycle")
            else:
                allowed, why = True, ""
                if engine is not None:
                    try:
                        decision = engine.allow_entry(
                            symu, open_positions=self._active_warrior_trade_count() + entered)
                        allowed = bool(decision.allowed)
                        why = getattr(decision, "reason", "") or "concurrency full"
                    except Exception:
                        allowed = True
                if not allowed:
                    _mark(symu, "concurrency", conv, price, stop, size_factor, why)
                else:
                    _mark(symu, "ready", conv, price, stop, size_factor,
                          "conv {:.2f} → ENTER".format(conv))
            rec = watch.get(symu, {})
            logger.info(
                "⚡ IGNITION %s [%s] conv=%.2f size=%.2f entry=%.3f stop=%.3f %s",
                symu, rec.get("state", "?"), conv, size_factor, price, stop, rec.get("reason", ""),
            )
            if rec.get("state") != "ready":
                continue
            risk = price - stop
            ctx = {
                "entry_trigger": "warrior_ignition",
                "entry_price_override": round(price, 4),
                "stop_price_override": round(stop, 4),
                "target_price_override": round(price + risk * 3.0, 2),
                "skip_unstable_confirm_stop_check": True,
                "rr_note_override": "warrior ignition conv={:.2f}".format(conv),
                "variant_override": "warrior_ignition",
                "max_hold_seconds_override": 600.0,
                "window_high": float(self._momentum_burst_window_high.get(symu, 0.0) or 0.0),
            }
            fill = self._execute_momentum_burst_scalp(
                sym, latest_10s, entry_context=ctx,
                strategy_override="warrior_squeeze_playbook",
                size_factor_override=size_factor,
            )
            if fill is not None:
                counts[symu] = counts.get(symu, 0) + 1
                entered += 1
        self._publish_warrior_ignition_watch()

    def _record_warrior_ignition_exit(
        self,
        symbol: str,
        pnl: float,
        reason: str,
        *,
        completed: bool,
    ) -> None:
        sym = symbol.upper()
        failures = getattr(self, "_warrior_ignition_failed_entries", None)
        if failures is None:
            failures = {}
            self._warrior_ignition_failed_entries = failures
        trade_pnls = getattr(self, "_warrior_ignition_trade_pnl", None)
        if trade_pnls is None:
            trade_pnls = {}
            self._warrior_ignition_trade_pnl = trade_pnls
        trade_pnls[sym] = float(trade_pnls.get(sym, 0.0) or 0.0) + float(pnl or 0.0)
        if not completed:
            return
        net_pnl = float(trade_pnls.pop(sym, 0.0) or 0.0)
        if net_pnl < 0.0:
            failures[sym] = int(failures.get(sym, 0) or 0) + 1
            failed = getattr(self, "_warrior_failed_momentum", None)
            if failed is None:
                failed = {}
                self._warrior_failed_momentum = failed
            failed[sym] = reason or "warrior ignition loss"
            logger.info(
                "IGNITION failed %s count=%d net_pnl=$%.2f reason=%s",
                sym,
                failures[sym],
                net_pnl,
                reason,
            )
        elif net_pnl > 0.0:
            failures.pop(sym, None)
            failed = getattr(self, "_warrior_failed_momentum", None)
            if failed is not None:
                failed.pop(sym, None)

    def _warrior_ignition_watch_snapshot(self) -> List[dict]:
        """Read-only telemetry of what the learned ignition path is watching this
        cycle and WHY each symbol did/didn't fire. This is its OWN surface, kept
        separate from the WarriorWatchBook that the squeeze lanes mutate."""
        watch = getattr(self, "_warrior_ignition_watch", None)
        if not watch:
            return []
        now_mono = time.monotonic()
        rank = {"ready": 0, "queued": 1, "concurrency": 2, "capped": 3,
                "suppressed": 4, "low_conviction": 5, "bad_stop": 6, "no_ignition": 7}
        rows: List[dict] = []
        for symu, rec in list(watch.items()):
            try:
                age = now_mono - float(rec.get("ts", 0.0))
            except (TypeError, ValueError):
                age = 0.0
            if age > 120.0:          # drop stale scores so the table reflects NOW
                watch.pop(symu, None)
                continue
            row = {k: v for k, v in rec.items() if k != "ts"}
            row["age_seconds"] = round(age, 1)
            rows.append(row)
        rows.sort(key=lambda r: (rank.get(str(r.get("state")), 9), -float(r.get("conviction") or 0.0)))
        return rows[:15]

    def _publish_warrior_ignition_watch(self) -> None:
        handler = getattr(self._hub, "on_warrior_ignition_watch", None)
        if handler is not None:
            try:
                handler(self._warrior_ignition_watch_snapshot())
            except Exception:
                pass

    def _process_momentum_burst_scalps(self) -> None:
        """Monitor armed momentum_burst symbols and scalp fresh 10s highs.

        This is intentionally a runner-side experiment: the momentum_burst
        scanner arms a short window, but orders still go through the same quick
        scalp shape, shared entry guard/ML, 10s confirmation, R:R, risk, broker,
        and position-open bookkeeping as HOD breakout scalps.
        """
        # Learned premarket-ignition path. When enabled (env flag) AND it is
        # pre-market, it takes over the Warrior path: trade ONLY base-breakout
        # ignitions (momentum bursts) sized by model conviction, and log them for
        # the adaptive dataset. Outside pre-market it falls through to the normal
        # Warrior lanes below. Wrapped so a scoring bug can't disrupt the loop.
        if (
            os.environ.get("DAYTRADING_WARRIOR_IGNITION", "").strip().lower()
            in ("1", "true", "yes", "on")
            and hasattr(self, "_maybe_enter_warrior_ignition")
            and self._market_phase() == "PRE-MARKET"
        ):
            try:
                self._maybe_enter_warrior_ignition()
            except Exception:
                logger.debug("warrior ignition path skipped", exc_info=True)
            return
        cycle_enabled = bool(getattr(self, "_momentum_burst_cycle_enabled", False))
        hit_run_enabled = bool(getattr(self, "_momentum_burst_hit_run_enabled", False))
        warrior_enabled = bool(getattr(self, "_warrior_squeeze_enabled", False))
        fast_squeeze_enabled = hit_run_enabled or warrior_enabled
        if not (cycle_enabled or fast_squeeze_enabled):
            return
        armed = getattr(self, "_momentum_burst_armed", None)
        if warrior_enabled:
            for sym, bars in list(getattr(self, "_timer_bars_by_symbol", {}).items()):
                if bars:
                    self._maybe_arm_warrior_squeeze_from_10s(sym, bars[-1], time.monotonic())
        if not armed:
            return
        if self._new_entries_blocked(None, "MOMENTUM BURST SCALP"):
            armed.clear()
            self._momentum_burst_window_high.clear()
            self._momentum_burst_pending.clear()
            return

        pending = self._momentum_burst_pending
        now_mono = time.monotonic()
        warrior_last_target_at = getattr(self, "_warrior_squeeze_last_target_at", {})
        warrior_target_wins = getattr(self, "_warrior_squeeze_target_wins", {})
        warrior_post_target_allowed = getattr(
            self,
            "_warrior_squeeze_post_target_reclaim_allowed",
            {},
        )
        for sym, armed_at in list(armed.items()):
            effective_window_sec = self._momentum_burst_window_sec
            last_target_at = warrior_last_target_at.get(sym)
            raw_warrior_target_wins = int(warrior_target_wins.get(sym, 0) or 0)
            warrior_target_fresh = (
                bool(warrior_enabled and raw_warrior_target_wins > 0)
                and (
                    last_target_at is None
                    or now_mono - float(last_target_at or 0.0) <= 1200.0
                )
            )
            if warrior_enabled and (
                warrior_target_fresh
                or warrior_post_target_allowed.get(sym, 0) > 0
            ):
                effective_window_sec = max(effective_window_sec, 900.0)
            hold_until_premarket_end = (
                warrior_enabled
                and bool(getattr(self, "_warrior_watch_until_premarket_end", True))
                and self._market_phase() == "PRE-MARKET"
            )
            if (
                not hold_until_premarket_end
                and now_mono - float(armed_at or 0.0) > effective_window_sec
            ):
                armed.pop(sym, None)
                self._momentum_burst_window_high.pop(sym, None)
                pending.pop(sym, None)
                self._momentum_burst_hit_run_counts.pop(sym, None)
                self._momentum_burst_hit_run_block_until.pop(sym, None)
                getattr(self, "_warrior_squeeze_failed_burst", {}).pop(sym, None)
                getattr(self, "_warrior_squeeze_failed_burst_high", {}).pop(sym, None)
                continue

            warrior_engine = getattr(self, "_warrior_engine", None)
            if warrior_engine is not None:
                risk_decision = warrior_engine.allow_entry(
                    sym,
                    open_positions=self._active_warrior_trade_count(),
                )
                if not risk_decision.allowed:
                    if sym in self._momentum_burst_hit_run_day_blocked:
                        pending.pop(sym, None)
                        logger.info(
                            "WARRIOR SQUEEZE %s blocked for day: %s",
                            sym,
                            risk_decision.reason,
                        )
                        continue
                    return
            elif self._breakout_scalp_active:
                return

            if fast_squeeze_enabled:
                day_block = self._momentum_burst_hit_run_day_blocked.get(sym)
                if day_block:
                    pending.pop(sym, None)
                    logger.info("MOMENTUM BURST HIT-RUN %s blocked for day: %s", sym, day_block)
                    continue
                failed_burst = (
                    getattr(self, "_warrior_squeeze_failed_burst", {}).get(sym)
                    if warrior_enabled else None
                )
                if failed_burst and self._warrior_squeeze_target_wins.get(sym, 0) <= 0:
                    latest_10s = self._latest_momentum_burst_10s_bar(sym)
                    second_leg_context = (
                        self._warrior_trend_pullback_reclaim_context(
                            sym,
                            latest_10s,
                            window_high=float(
                                self._momentum_burst_window_high.get(sym, 0.0) or 0.0
                            ),
                        )
                        if latest_10s is not None
                        else None
                    )
                    if (
                        second_leg_context is not None
                        and second_leg_context.get("entry_trigger")
                        == "warrior_second_leg_vwap_reclaim"
                    ):
                        getattr(self, "_warrior_squeeze_failed_burst", {}).pop(sym, None)
                        getattr(self, "_warrior_squeeze_failed_burst_high", {}).pop(sym, None)
                        self._momentum_burst_hit_run_block_until.pop(sym, None)
                        current_high = float(latest_10s.high or 0.0)
                        trigger_high = float(
                            second_leg_context.get("base_high")
                            or second_leg_context.get("entry_price_override")
                            or current_high
                        )
                        trigger_close = float(
                            second_leg_context.get("entry_price_override")
                            or latest_10s.close
                            or 0.0
                        )
                        bar_ts = getattr(latest_10s, "ts", None)
                        pending[sym] = {
                            "ts": (
                                bar_ts - timedelta(seconds=10)
                                if isinstance(bar_ts, datetime)
                                else datetime.now(timezone.utc) - timedelta(seconds=10)
                            ),
                            "breakout_close": trigger_close,
                            "breakout_high": trigger_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                            **second_leg_context,
                        }
                        logger.info(
                            "WARRIOR SQUEEZE %s second-leg VWAP reclaim reset failed burst",
                            sym,
                        )
                        continue
                    halt_resume_context = (
                        self._warrior_halt_resume_continuation_context(
                            sym,
                            latest_10s,
                            failed_high=float(
                                getattr(self, "_warrior_squeeze_failed_burst_high", {}).get(sym, 0.0) or 0.0
                            ),
                            window_high=float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0),
                        )
                        if latest_10s is not None
                        else None
                    )
                    if halt_resume_context is not None:
                        getattr(self, "_warrior_squeeze_failed_burst", {}).pop(sym, None)
                        getattr(self, "_warrior_squeeze_failed_burst_high", {}).pop(sym, None)
                        self._momentum_burst_hit_run_block_until.pop(sym, None)
                        current_high = float(latest_10s.high or 0.0)
                        self._momentum_burst_window_high[sym] = max(
                            float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0),
                            current_high,
                        )
                        trigger_high = float(
                            halt_resume_context.get("pullaway_level")
                            or halt_resume_context.get("entry_price_override")
                            or current_high
                        )
                        trigger_close = float(
                            halt_resume_context.get("entry_price_override")
                            or latest_10s.close
                            or 0.0
                        )
                        bar_ts = getattr(latest_10s, "ts", None)
                        pending[sym] = {
                            "ts": (
                                bar_ts - timedelta(seconds=10)
                                if isinstance(bar_ts, datetime)
                                else datetime.now(timezone.utc) - timedelta(seconds=10)
                            ),
                            "breakout_close": trigger_close,
                            "breakout_high": trigger_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                            **halt_resume_context,
                        }
                        logger.info(
                            "WARRIOR SQUEEZE %s halt-resume continuation reset failed burst",
                            sym,
                        )
                        continue
                    recovery_context = (
                        self._warrior_failed_burst_recovery_context(
                            sym,
                            latest_10s,
                            failed_high=float(
                                getattr(self, "_warrior_squeeze_failed_burst_high", {}).get(sym, 0.0) or 0.0
                            ),
                            window_high=float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0),
                        )
                        if latest_10s is not None
                        else None
                    )
                    if recovery_context is None:
                        pending.pop(sym, None)
                        logger.info("WARRIOR SQUEEZE %s failed burst block: %s", sym, failed_burst)
                        continue
                    getattr(self, "_warrior_squeeze_failed_burst", {}).pop(sym, None)
                    getattr(self, "_warrior_squeeze_failed_burst_high", {}).pop(sym, None)
                    self._momentum_burst_hit_run_block_until.pop(sym, None)
                    current_high = float(latest_10s.high or 0.0)
                    self._momentum_burst_window_high[sym] = max(
                        float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0),
                        current_high,
                    )
                    bar_ts = getattr(latest_10s, "ts", None)
                    pending[sym] = {
                        "ts": (
                            bar_ts - timedelta(seconds=10)
                            if isinstance(bar_ts, datetime)
                            else datetime.now(timezone.utc) - timedelta(seconds=10)
                        ),
                        "breakout_close": float(latest_10s.close or 0.0),
                        "breakout_high": current_high,
                        "breakout_volume": float(latest_10s.volume or 0.0),
                        **recovery_context,
                    }
                    logger.info(
                        "WARRIOR SQUEEZE %s failed burst recovered on fresh high",
                        sym,
                    )
                    continue
                effective_max_entries = (
                    int(getattr(self, "_warrior_squeeze_max_entries", 3) or 3)
                    if warrior_enabled
                    else self._momentum_burst_hit_run_max_entries
                )
                if self._momentum_burst_hit_run_counts.get(sym, 0) >= effective_max_entries:
                    continue
                if now_mono < self._momentum_burst_hit_run_block_until.get(sym, 0.0):
                    continue
            else:
                cooldown_until = self._breakout_scalp_cooldown.get(sym, 0.0)
                if now_mono < cooldown_until:
                    continue
            last_exit_ts = self._pipeline._exit_cooldowns.get(sym)
            if last_exit_ts is not None and not fast_squeeze_enabled:
                try:
                    elapsed = (datetime.now(timezone.utc) - last_exit_ts).total_seconds()
                    if elapsed < self._pipeline._cooldown_seconds:
                        continue
                except Exception:
                    continue
            if (
                not fast_squeeze_enabled
                and self._pipeline._symbol_entry_counts.get(sym, 0) >= self._pipeline._max_entries_per_symbol
            ):
                continue
            pos = self._pipeline.portfolio.positions.get(sym)
            if pos and not pos.is_flat:
                continue

            latest_10s = self._latest_momentum_burst_10s_bar(sym)
            if latest_10s is None:
                continue
            if fast_squeeze_enabled:
                if (
                    not warrior_enabled
                    and not self._momentum_burst_hit_run_time_allowed(
                        getattr(latest_10s, "ts", None)
                    )
                ):
                    armed.pop(sym, None)
                    self._momentum_burst_window_high.pop(sym, None)
                    pending.pop(sym, None)
                    logger.info(
                        "MOMENTUM BURST HIT-RUN %s outside time window ending %s ET",
                        sym,
                        self._momentum_burst_hit_run_end_et,
                    )
                    continue
                reentry = self._momentum_burst_hit_run_counts.get(sym, 0) > 0
                anchor_high = float(self._momentum_burst_session_anchor_high.get(sym, 0.0) or 0.0)
                current_close = float(latest_10s.close or 0.0)
                stop_reason = self._momentum_burst_stop_trading_reason(sym)
                if stop_reason:
                    continuation_ok, _continuation_reason, _continuation_meta = (
                        self._momentum_burst_continuation_base_ok(sym)
                    )
                    if continuation_ok:
                        stop_reason = ""
                if stop_reason and not warrior_enabled:
                    pending.pop(sym, None)
                    logger.info("MOMENTUM BURST HIT-RUN %s stop trading: %s", sym, stop_reason)
                    continue
                if not warrior_enabled and anchor_high > 0 and current_close > anchor_high * 1.5:
                    continuation_ok, continuation_reason, _continuation_meta = (
                        self._momentum_burst_continuation_base_ok(sym)
                    )
                    if not continuation_ok:
                        logger.info(
                            "MOMENTUM BURST HIT-RUN %s extended without continuation base: %s",
                            sym,
                            continuation_reason,
                        )
                        continue
                if reentry and not warrior_enabled:
                    continuation_ok, continuation_reason, _continuation_meta = (
                        self._momentum_burst_continuation_base_ok(sym)
                    )
                    if not continuation_ok:
                        logger.info(
                            "MOMENTUM BURST HIT-RUN %s re-entry needs fresh micro-base: %s",
                            sym,
                            continuation_reason,
                        )
                        continue
            bar_ts = getattr(latest_10s, "ts", None)
            current_high = float(latest_10s.high or 0.0)
            window_high = float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0)
            warrior_target_win_count = (
                int(self._warrior_squeeze_target_wins.get(sym, 0) or 0)
                if warrior_enabled and warrior_target_fresh else 0
            )
            if warrior_enabled and raw_warrior_target_wins > 0 and not warrior_target_fresh:
                pending.pop(sym, None)
                logger.info(
                    "WARRIOR SQUEEZE %s target win is stale; no late re-entry",
                    sym,
                )
                continue
            if (
                warrior_enabled
                and warrior_target_win_count >= 1
                and window_high > 0
                and pending.get(sym) is None
            ):
                armed_prior_runner_pullback = False
                prior_runner_context = self._warrior_prior_runner_continuation_pullback_context(
                    sym,
                    latest_10s,
                    window_high=window_high,
                )
                if prior_runner_context is not None:
                    pending[sym] = {
                        "ts": (
                            bar_ts - timedelta(seconds=10)
                            if bar_ts is not None else bar_ts
                        ),
                        "breakout_close": float(latest_10s.close or 0.0),
                        "breakout_high": current_high,
                        "breakout_volume": float(latest_10s.volume or 0.0),
                        **prior_runner_context,
                    }
                    logger.info(
                        "WARRIOR SQUEEZE %s prior-runner continuation pullback executing on reclaim bar",
                        sym,
                    )
                    armed_prior_runner_pullback = True
                if not armed_prior_runner_pullback:
                    if int(self._warrior_squeeze_post_target_reclaim_allowed.get(sym, 0) or 0) > 0:
                        post_target_context = self._warrior_post_target_pullback_reclaim_context(
                            sym,
                            latest_10s,
                            window_high=window_high,
                        )
                        if post_target_context is not None:
                            pending[sym] = {
                                "ts": (
                                    bar_ts - timedelta(seconds=10)
                                    if bar_ts is not None else bar_ts
                                ),
                                "breakout_close": float(latest_10s.close or 0.0),
                                "breakout_high": current_high,
                                "breakout_volume": float(latest_10s.volume or 0.0),
                                **post_target_context,
                            }
                            logger.info(
                                "WARRIOR SQUEEZE %s post-target pullback reclaim armed",
                                sym,
                            )
                            continue
                    parabolic_reclaim_context = self._warrior_parabolic_micro_pullback_reclaim_context(
                        sym,
                        latest_10s,
                        window_high=window_high,
                    )
                    if (
                        parabolic_reclaim_context is not None
                        and parabolic_reclaim_context.get("entry_trigger")
                        == "warrior_parabolic_micro_pullback_reclaim"
                    ):
                        pending[sym] = {
                            "ts": (
                                bar_ts - timedelta(seconds=10)
                                if bar_ts is not None else bar_ts
                            ),
                            "breakout_close": float(latest_10s.close or 0.0),
                            "breakout_high": current_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                            **parabolic_reclaim_context,
                        }
                        logger.info(
                            "WARRIOR SQUEEZE %s post-target parabolic micro-pullback reclaim armed",
                            sym,
                        )
                        continue
                    fresh_continuation_context = (
                        self._warrior_post_target_fresh_continuation_context(
                            sym,
                            latest_10s,
                            window_high=window_high,
                        )
                        if current_high > window_high * 1.001
                        else None
                    )
                    if fresh_continuation_context is not None:
                        self._momentum_burst_window_high[sym] = current_high
                        pending[sym] = {
                            "ts": bar_ts,
                            "breakout_close": float(latest_10s.close or 0.0),
                            "breakout_high": current_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                            **fresh_continuation_context,
                        }
                        logger.info(
                            "WARRIOR SQUEEZE %s post-target fresh continuation armed",
                            sym,
                        )
                        continue
                    second_leg_context = self._warrior_squeeze_second_leg_reclaim_context(
                        sym,
                        latest_10s,
                        {
                            "ts": bar_ts,
                            "breakout_close": float(latest_10s.close or 0.0),
                            "breakout_high": current_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                        },
                        window_high=window_high,
                    )
                    if second_leg_context is not None:
                        self._momentum_burst_window_high[sym] = current_high
                        pending[sym] = {
                            "ts": bar_ts,
                            "breakout_close": float(latest_10s.close or 0.0),
                            "breakout_high": current_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                            **second_leg_context,
                        }
                        logger.info(
                            "WARRIOR SQUEEZE %s second-leg reclaim armed after deep washout",
                            sym,
                        )
                        continue
                    pending.pop(sym, None)
                    lock_reason = (
                        "target win banked; needs fresh high above {:.2f}".format(window_high)
                    )
                    if current_high > window_high * 1.001:
                        lock_reason = "target win banked; fresh high alone is not enough after first win"
                    logger.info(
                        "WARRIOR SQUEEZE %s profit lock: %s",
                        sym,
                        lock_reason,
                    )
                    continue

            # A fresh 10s high arms a pending breakout; we do NOT buy that spike
            # bar. Entry waits for the NEXT 10s bar to prove continuation: green,
            # holding the breakout close, and trading through the breakout high.
            # A sideways hold under the spike high is not enough for hit-run.
            if warrior_enabled and self._momentum_burst_hit_run_counts.get(sym, 0) == 0:
                trend_pullback_fn = getattr(
                    self,
                    "_warrior_trend_pullback_reclaim_context",
                    None,
                )
                level_break_fn = getattr(
                    self,
                    "_warrior_level_break_starter_context",
                    None,
                )
                level_break_context = (
                    level_break_fn(
                        sym,
                        latest_10s,
                        window_high=window_high,
                    )
                    if callable(level_break_fn)
                    else None
                )
                high_base_context = (
                    trend_pullback_fn(
                        sym,
                        latest_10s,
                        window_high=window_high,
                    )
                    if callable(trend_pullback_fn)
                    else None
                )
                starter_context = level_break_context or high_base_context
                if (
                    starter_context is not None
                    and warrior_lanes.is_warrior_initial_starter_trigger(
                        starter_context.get("entry_trigger")
                    )
                ):
                    pending[sym] = {
                        "ts": (
                            bar_ts - timedelta(seconds=10)
                            if bar_ts is not None else bar_ts
                        ),
                        "breakout_close": float(latest_10s.close or 0.0),
                        "breakout_high": current_high,
                        "breakout_volume": float(latest_10s.volume or 0.0),
                        **starter_context,
                    }
            pend = pending.get(sym)
            if pend is not None and bar_ts is not None and pend.get("ts") is not None:
                try:
                    pending_ts = pend.get("original_ts") or pend["ts"]
                    gap = (bar_ts - pending_ts).total_seconds()
                except Exception:
                    gap = 999.0
                if gap > 30.0:  # confirmation never arrived — drop stale breakout
                    pending.pop(sym, None)
                    pend = None

            if pend is not None:
                # Still inside the breakout bar itself — wait for the next bar.
                if bar_ts is not None and pend.get("ts") is not None and bar_ts <= pend["ts"]:
                    continue
                confirmed = (
                    float(latest_10s.close or 0.0) >= float(latest_10s.open or 0.0)
                    and float(latest_10s.close or 0.0) >= float(pend.get("breakout_close") or 0.0)
                )
                breakout_high = float(
                    pend.get("breakout_high")
                    or pend.get("breakout_close")
                    or 0.0
                )
                confirm_high = float(latest_10s.high or 0.0)
                confirm_low = float(latest_10s.low or 0.0)
                continuation_buffer = max(0.005, breakout_high * 0.001)
                confirm_range = max(confirm_high - confirm_low, 0.0)
                close_location = (
                    (float(latest_10s.close or 0.0) - confirm_low) / confirm_range
                    if confirm_range > 0 else 0.0
                )
                violent_ok, _violent_meta = self._momentum_burst_violent_liquid_ok(sym)
                reentry = fast_squeeze_enabled and self._momentum_burst_hit_run_counts.get(sym, 0) > 0
                if (
                    warrior_enabled
                    and warrior_target_win_count >= 1
                    and not warrior_lanes.is_warrior_entry_trigger(
                        pend.get("entry_trigger")
                    )
                ):
                    pending.pop(sym, None)
                    logger.info(
                        "WARRIOR SQUEEZE %s profit lock: discard pending generic add after target win",
                        sym,
                    )
                    continue
                volume_ratio = 0.5 if reentry else (0.25 if hit_run_enabled and violent_ok else 0.5)
                chase_cap = 0.03 if reentry else (0.08 if hit_run_enabled and violent_ok else 0.03)
                first_pullback_context = None
                current_pullaway_context = (
                    self._warrior_squeeze_pullaway_context(sym, latest_10s, pend)
                    if warrior_enabled else None
                )
                if warrior_enabled:
                    trend_pullback_fn = getattr(
                        self,
                        "_warrior_trend_pullback_reclaim_context",
                        None,
                    )
                    first_pullback_candidate = (
                        trend_pullback_fn(
                            sym,
                            latest_10s,
                            window_high=window_high,
                        )
                        if callable(trend_pullback_fn)
                        else None
                    )
                    if (
                        first_pullback_candidate is not None
                        and first_pullback_candidate.get("entry_trigger")
                        in {
                            "warrior_first_pullback_reclaim",
                            "warrior_low_price_proof_reclaim",
                        }
                    ):
                        first_pullback_context = first_pullback_candidate
                if (
                    current_pullaway_context is not None
                    and current_pullaway_context.get("variant_override")
                    == "warrior_clwt_fast_pullaway"
                ):
                    pullaway_context = current_pullaway_context
                elif first_pullback_context is not None:
                    pullaway_context = first_pullback_context
                elif warrior_enabled and (
                    warrior_lanes.is_warrior_entry_trigger(pend.get("entry_trigger"))
                    or pend.get("variant_override") == "warrior_clwt_fast_pullaway"
                ):
                    pullaway_context = dict(pend)
                else:
                    pullaway_context = current_pullaway_context
                curl_context = (
                    self._warrior_squeeze_curl_reclaim_context(
                        sym,
                        latest_10s,
                        pend,
                        window_high=window_high,
                    )
                    if warrior_enabled and pullaway_context is None else None
                )
                equal_high_context = (
                    self._warrior_squeeze_equal_high_pullaway_context(
                        sym,
                        latest_10s,
                        pend,
                        window_high=max(window_high, breakout_high),
                    )
                    if warrior_enabled
                    and pullaway_context is None
                    and curl_context is None
                    else None
                )
                high_base_context = (
                    self._warrior_trend_pullback_reclaim_context(
                        sym,
                        latest_10s,
                        window_high=window_high,
                    )
                    if warrior_enabled
                    and pullaway_context is None
                    and curl_context is None
                    and equal_high_context is None
                    else None
                )
                if (
                    high_base_context is not None
                    and not warrior_lanes.is_warrior_high_base_confirm_trigger(
                        high_base_context.get("entry_trigger")
                    )
                ):
                    high_base_context = None
                reject_reason = None
                if pullaway_context is not None:
                    pend = {**pend, **pullaway_context}
                elif curl_context is not None:
                    pend = {**pend, **curl_context}
                elif equal_high_context is not None:
                    pend = {**pend, **equal_high_context}
                elif high_base_context is not None:
                    pend = {**pend, **high_base_context}
                elif warrior_enabled:
                    reject_reason = "warrior setup not confirmed by playbook pattern"
                elif not confirmed:
                    reject_reason = "breakout not confirmed by next 10s bar"
                elif (
                    fast_squeeze_enabled
                    and breakout_high > 0
                    and confirm_high < breakout_high + continuation_buffer
                ):
                    reject_reason = (
                        "confirm bar did not break continuation high "
                        "({:.2f} <= {:.2f})"
                    ).format(confirm_high, breakout_high)
                elif fast_squeeze_enabled and close_location < 0.65:
                    reject_reason = "confirm bar did not close with strength"
                elif (
                    fast_squeeze_enabled
                    and float(pend.get("psych_level") or 0.0) > 0
                    and float(latest_10s.close or 0.0) < float(pend.get("psych_level") or 0.0)
                ):
                    reject_reason = "confirm bar failed to hold ${:.2f} level".format(
                        float(pend.get("psych_level") or 0.0)
                    )
                elif float(latest_10s.volume or 0.0) < volume_ratio * float(pend.get("breakout_volume") or 0.0):
                    reject_reason = "confirm-bar volume too light (no follow-through)"
                elif (
                    float(pend.get("original_breakout_close") or pend.get("breakout_close") or 0.0) > 0
                    and float(latest_10s.close or 0.0)
                    > float(pend.get("original_breakout_close") or pend.get("breakout_close") or 0.0) * (1.0 + chase_cap)
                ):
                    reject_reason = "chasing: confirm {:.2f} >{:.0%} above breakout {:.2f}".format(
                        float(latest_10s.close or 0.0),
                        chase_cap,
                        float(pend.get("original_breakout_close") or pend.get("breakout_close") or 0.0),
                    )
                if reject_reason is not None:
                    if (
                        pend.get("entry_trigger") != "warrior_post_target_pullback_reclaim"
                        and self._momentum_burst_rebase_pending_after_reject(
                        sym,
                        latest_10s,
                        pend,
                        reject_reason,
                        hit_run=fast_squeeze_enabled,
                        )
                    ):
                        continue
                    failed_watch_reason = (
                        warrior_lanes.warrior_failed_burst_watch_reason(latest_10s)
                        if warrior_enabled
                        else None
                    )
                    if failed_watch_reason:
                        self._warrior_squeeze_failed_burst[sym] = failed_watch_reason
                        self._warrior_squeeze_failed_burst_high[sym] = max(
                            float(self._warrior_squeeze_failed_burst_high.get(sym, 0.0) or 0.0),
                            float(latest_10s.high or 0.0),
                        )
                    pending.pop(sym, None)
                    tracker = getattr(self, "_track_warrior_normal_fallback_state", None)
                    if callable(tracker):
                        tracker(
                            sym,
                            "warrior_squeeze_playbook_unconfirmed",
                            reject_reason,
                        )
                    logger.info("MOMENTUM BURST SCALP %s: %s - skip", sym, reject_reason)
                    continue
                if warrior_enabled:
                    recent_bad_tape_fn = getattr(
                        self,
                        "_warrior_recent_bad_tape_reject",
                        None,
                    )
                    recent_bad_tape = (
                        recent_bad_tape_fn(sym)
                        if callable(recent_bad_tape_fn)
                        else None
                    )
                    if (
                        recent_bad_tape
                        and pend.get("variant_override")
                        in {
                            "warrior_clwt_fast_pullaway",
                            "warrior_low_price_proof_reclaim",
                            "warrior_second_leg_vwap_reclaim",
                        }
                    ):
                        recent_bad_tape = None
                    if recent_bad_tape:
                        rebase_reason = "volume too light after {}".format(recent_bad_tape)
                        if (
                            pend.get("entry_trigger") != "warrior_post_target_pullback_reclaim"
                            and self._momentum_burst_rebase_pending_after_reject(
                                sym,
                                latest_10s,
                                pend,
                                rebase_reason,
                                hit_run=fast_squeeze_enabled,
                            )
                        ):
                            logger.info(
                                "WARRIOR SQUEEZE %s waiting for fresh 10s reclaim after bad tape: %s",
                                sym,
                                recent_bad_tape,
                            )
                            continue
                        pending.pop(sym, None)
                        tracker = getattr(self, "_track_warrior_normal_fallback_state", None)
                        if callable(tracker):
                            tracker(
                                sym,
                                "warrior_squeeze_playbook_micro_base_wait",
                                recent_bad_tape,
                            )
                        logger.info(
                            "WARRIOR SQUEEZE %s waiting after recent bad tape: %s",
                            sym,
                            recent_bad_tape,
                        )
                        continue
                post_blowoff_micro_base = bool(hit_run_enabled and pend.get("reset_from_stale_high"))
                pending.pop(sym, None)
                fill = self._execute_momentum_burst_scalp(
                    sym,
                    latest_10s,
                    hit_run=fast_squeeze_enabled,
                    violent_liquid=bool(fast_squeeze_enabled and violent_ok),
                    post_blowoff_micro_base=post_blowoff_micro_base,
                    entry_context=pend,
                    strategy_override=(
                        "warrior_squeeze_playbook" if warrior_enabled else None
                    ),
                    size_factor_override=(
                        self._warrior_squeeze_starter_size_factor
                        if warrior_enabled else None
                    ),
                    window_high=window_high,
                )
                if fill is not None:
                    if not fast_squeeze_enabled:
                        armed.pop(sym, None)
                        self._momentum_burst_window_high.pop(sym, None)
                    break
                continue

            if current_high > 0 and window_high > 0 and current_high <= window_high:
                if (
                    warrior_enabled
                    and self._momentum_burst_hit_run_counts.get(sym, 0) == 0
                ):
                    equal_high_context = self._warrior_squeeze_equal_high_pullaway_context(
                        sym,
                        latest_10s,
                        {
                            "ts": bar_ts,
                            "breakout_close": float(latest_10s.close or 0.0),
                            "breakout_high": current_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                            **self._momentum_burst_level_context(window_high, max(current_high, window_high)),
                        },
                        window_high=window_high,
                    )
                    if equal_high_context is not None:
                        fill = self._execute_momentum_burst_scalp(
                            sym,
                            latest_10s,
                            hit_run=True,
                            violent_liquid=False,
                            entry_context=equal_high_context,
                            strategy_override="warrior_squeeze_playbook",
                            size_factor_override=self._warrior_squeeze_starter_size_factor,
                            window_high=window_high,
                        )
                        if fill is not None:
                            break
                if warrior_enabled:
                    prior_runner_context = self._warrior_prior_runner_continuation_pullback_context(
                        sym,
                        latest_10s,
                        window_high=window_high,
                    )
                    if prior_runner_context is not None:
                        pending[sym] = {
                            "ts": bar_ts,
                            "breakout_close": float(latest_10s.close or 0.0),
                            "breakout_high": current_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                            **prior_runner_context,
                        }
                        logger.info(
                            "WARRIOR SQUEEZE %s prior-runner continuation pullback armed",
                            sym,
                        )
                        continue
                if warrior_enabled:
                    trend_pullback_context = self._warrior_trend_pullback_reclaim_context(
                        sym,
                        latest_10s,
                        window_high=window_high,
                    )
                    if trend_pullback_context is not None:
                        trigger_high = float(
                            trend_pullback_context.get("base_high")
                            or trend_pullback_context.get("entry_price_override")
                            or current_high
                        )
                        trigger_close = float(
                            trend_pullback_context.get("entry_price_override")
                            or latest_10s.close
                            or 0.0
                        )
                        pending[sym] = {
                            "ts": (
                                bar_ts - timedelta(seconds=10)
                                if bar_ts is not None else bar_ts
                            ),
                            "breakout_close": trigger_close,
                            "breakout_high": trigger_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                            **trend_pullback_context,
                        }
                        logger.info(
                            "WARRIOR SQUEEZE %s trend pullback reclaim armed",
                            sym,
                        )
                        continue
                if warrior_enabled and warrior_target_win_count >= 1:
                    second_leg_context = self._warrior_squeeze_second_leg_reclaim_context(
                        sym,
                        latest_10s,
                        {
                            "ts": bar_ts,
                            "breakout_close": float(latest_10s.close or 0.0),
                            "breakout_high": current_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                        },
                        window_high=window_high,
                    )
                    if second_leg_context is not None:
                        self._momentum_burst_window_high[sym] = current_high
                        pending[sym] = {
                            "ts": bar_ts,
                            "breakout_close": float(latest_10s.close or 0.0),
                            "breakout_high": current_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                            **second_leg_context,
                        }
                        logger.info(
                            "WARRIOR SQUEEZE %s second-leg reclaim armed after deep washout",
                            sym,
                        )
                        continue
                    logger.info(
                        "WARRIOR SQUEEZE %s profit lock: target win banked; "
                        "needs controlled continuation pullback",
                        sym,
                    )
                    pending.pop(sym, None)
                    continue
                if warrior_enabled and self._momentum_burst_hit_run_counts.get(sym, 0) > 0:
                    second_leg_context = self._warrior_squeeze_second_leg_reclaim_context(
                        sym,
                        latest_10s,
                        {
                            "ts": bar_ts,
                            "breakout_close": float(latest_10s.close or 0.0),
                            "breakout_high": current_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                        },
                        window_high=window_high,
                    )
                    if second_leg_context is not None:
                        self._momentum_burst_window_high[sym] = current_high
                        pending[sym] = {
                            "ts": bar_ts,
                            "breakout_close": float(latest_10s.close or 0.0),
                            "breakout_high": current_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                            **second_leg_context,
                        }
                        logger.info(
                            "WARRIOR SQUEEZE %s second-leg reclaim armed after deep washout",
                            sym,
                        )
                        continue
                if (
                    warrior_enabled
                    and self._momentum_burst_hit_run_counts.get(sym, 0) > 0
                    and current_high >= window_high * 0.995
                ):
                    add_pending = {
                        "ts": bar_ts,
                        "breakout_close": float(latest_10s.close or 0.0),
                        "breakout_high": current_high,
                        "breakout_volume": float(latest_10s.volume or 0.0),
                        **self._momentum_burst_level_context(window_high * 0.99, current_high),
                    }
                    pullaway_context = self._warrior_squeeze_pullaway_context(
                        sym,
                        latest_10s,
                        add_pending,
                    )
                    if pullaway_context is not None:
                        add_context = {**add_pending, **pullaway_context}
                        fill = self._execute_momentum_burst_scalp(
                            sym,
                            latest_10s,
                            hit_run=True,
                            violent_liquid=False,
                            entry_context=add_context,
                            strategy_override="warrior_squeeze_playbook",
                            size_factor_override=self._warrior_squeeze_starter_size_factor,
                            window_high=window_high,
                        )
                        if fill is not None:
                            break
                if fast_squeeze_enabled and window_high > current_high * 1.08:
                    continuation_ok, _continuation_reason, continuation_meta = (
                        self._momentum_burst_continuation_base_ok(sym)
                    )
                    if continuation_ok:
                        self._momentum_burst_window_high[sym] = current_high
                        pending[sym] = {
                            "ts": bar_ts,
                            "breakout_close": float(latest_10s.close or 0.0),
                            "breakout_high": current_high,
                            "breakout_volume": float(latest_10s.volume or 0.0),
                            "reset_from_stale_high": round(window_high, 4),
                            "base_high": continuation_meta.get("base_high"),
                            "base_low": continuation_meta.get("base_low"),
                            **self._momentum_burst_level_context(window_high, current_high),
                        }
                continue

            if current_high > 0 and (window_high <= 0 or current_high > window_high):
                if warrior_enabled and warrior_target_win_count >= 1:
                    add_pending = {
                        "ts": bar_ts,
                        "breakout_close": float(latest_10s.close or 0.0),
                        "breakout_high": current_high,
                        "breakout_volume": float(latest_10s.volume or 0.0),
                        **self._momentum_burst_level_context(window_high, current_high),
                    }
                    prior_runner_context = self._warrior_prior_runner_continuation_pullback_context(
                        sym,
                        latest_10s,
                        window_high=window_high,
                    )
                    if prior_runner_context is not None:
                        pending[sym] = {
                            **add_pending,
                            **prior_runner_context,
                        }
                        logger.info(
                            "WARRIOR SQUEEZE %s prior-runner continuation pullback armed on fresh reclaim",
                            sym,
                        )
                        continue
                    logger.info(
                        "WARRIOR SQUEEZE %s profit lock: target win banked; "
                        "blocks generic fresh-high add",
                        sym,
                    )
                    pending.pop(sym, None)
                    self._momentum_burst_window_high[sym] = current_high
                    continue
                if warrior_enabled and self._momentum_burst_hit_run_counts.get(sym, 0) > 0:
                    target_wins = warrior_target_win_count
                    add_pending = {
                        "ts": bar_ts,
                        "breakout_close": float(latest_10s.close or 0.0),
                        "breakout_high": current_high,
                        "breakout_volume": float(latest_10s.volume or 0.0),
                        **self._momentum_burst_level_context(window_high, current_high),
                    }
                    if target_wins >= 1:
                        prior_runner_context = self._warrior_prior_runner_continuation_pullback_context(
                            sym,
                            latest_10s,
                            window_high=window_high,
                        )
                        if prior_runner_context is not None:
                            pending[sym] = {
                                **add_pending,
                                **prior_runner_context,
                            }
                            logger.info(
                                "WARRIOR SQUEEZE %s prior-runner continuation pullback armed on fresh reclaim",
                                sym,
                            )
                            continue
                        logger.info(
                            "WARRIOR SQUEEZE %s profit lock: target win banked; "
                            "needs controlled continuation pullback",
                            sym,
                        )
                        pending.pop(sym, None)
                        self._momentum_burst_window_high[sym] = current_high
                        continue
                    pullaway_context = self._warrior_squeeze_pullaway_context(
                        sym,
                        latest_10s,
                        add_pending,
                    )
                    if pullaway_context is not None:
                        add_context = {**add_pending, **pullaway_context}
                        fill = self._execute_momentum_burst_scalp(
                            sym,
                            latest_10s,
                            hit_run=True,
                            violent_liquid=False,
                            entry_context=add_context,
                            strategy_override="warrior_squeeze_playbook",
                            size_factor_override=self._warrior_squeeze_starter_size_factor,
                            window_high=window_high,
                        )
                        if fill is not None:
                            self._momentum_burst_window_high[sym] = current_high
                            break
                first_clwt_context = None
                first_reclaim_context = None
                if warrior_enabled and self._momentum_burst_hit_run_counts.get(sym, 0) == 0:
                    first_pending = {
                        "ts": bar_ts,
                        "breakout_close": float(latest_10s.close or 0.0),
                        "breakout_high": current_high,
                        "breakout_volume": float(latest_10s.volume or 0.0),
                        **self._momentum_burst_level_context(window_high, current_high),
                    }
                    candidate = self._warrior_squeeze_pullaway_context(
                        sym,
                        latest_10s,
                        first_pending,
                    )
                    if (
                        candidate is not None
                        and candidate.get("variant_override") == "warrior_clwt_fast_pullaway"
                    ):
                        first_clwt_context = {**first_pending, **candidate}
                    reclaim_candidate = self._warrior_trend_pullback_reclaim_context(
                        sym,
                        latest_10s,
                        window_high=window_high,
                    )
                    if (
                        reclaim_candidate is not None
                        and warrior_lanes.is_warrior_fresh_reclaim_trigger(
                            reclaim_candidate.get("entry_trigger")
                        )
                    ):
                        first_reclaim_context = {**first_pending, **reclaim_candidate}
                self._momentum_burst_window_high[sym] = current_high
                pending[sym] = {
                    "ts": bar_ts,
                    "breakout_close": float(latest_10s.close or 0.0),
                    "breakout_high": current_high,
                    "breakout_volume": float(latest_10s.volume or 0.0),
                    **self._momentum_burst_level_context(window_high, current_high),
                }
                initial_warrior_context = warrior_lanes.choose_initial_warrior_context(
                    first_clwt_context,
                    first_reclaim_context,
                )
                if initial_warrior_context is not None:
                    pending[sym].update(initial_warrior_context)
        if warrior_enabled:
            publisher = getattr(self, "_publish_warrior_watch", None)
            if callable(publisher):
                publisher()

    @staticmethod
    def _momentum_burst_level_context(previous_high: float, current_high: float) -> Dict[str, Any]:
        return warrior_lanes.momentum_burst_level_context(previous_high, current_high)

    def _warrior_history_until(self, symbol: str, latest_10s: Bar, *, count: int) -> List[Bar]:
        history = self._momentum_burst_recent_10s(symbol, count=count)
        current_ts = getattr(latest_10s, "ts", None)
        if current_ts is not None:
            history = [
                bar for bar in history
                if getattr(bar, "ts", None) is None or bar.ts <= current_ts
            ]
        return [bar for bar in history if float(bar.close or 0.0) > 0]

    def _warrior_squeeze_pullaway_context(
        self,
        symbol: str,
        latest_10s: Bar,
        pending_breakout: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        sym = symbol.upper()
        return warrior_lanes.warrior_squeeze_pullaway_context(
            latest_10s,
            pending_breakout,
            history=AlpacaRunner._warrior_history_until(self, sym, latest_10s, count=6),
            reject_high=float(self._warrior_squeeze_rejection_high.get(sym, 0.0) or 0.0),
            rejection_reason=self._warrior_squeeze_rejection_reason.get(sym),
            reentry_count=int(self._momentum_burst_hit_run_counts.get(sym, 0) or 0),
            min_reclaim_price=float(getattr(self, "_warrior_squeeze_min_reclaim_price", 3.5) or 0.0),
            reward_risk_value=float(getattr(self, "_warrior_squeeze_reward_risk", 3.0) or 3.0),
            add_reward_risk_value=float(getattr(self, "_warrior_squeeze_add_reward_risk", 1.0) or 1.0),
        )

    def _warrior_squeeze_first_starter_has_proof_hold(
        self,
        symbol: str,
        latest_10s: Bar,
        proof_level: float,
    ) -> bool:
        return warrior_lanes.first_starter_has_proof_hold(
            AlpacaRunner._warrior_history_until(self, symbol, latest_10s, count=6),
            proof_level,
        )

    def _warrior_squeeze_equal_high_pullaway_context(
        self,
        symbol: str,
        latest_10s: Bar,
        pending_breakout: Dict[str, Any],
        *,
        window_high: float,
    ) -> Optional[Dict[str, Any]]:
        sym = symbol.upper()
        return warrior_lanes.warrior_squeeze_equal_high_pullaway_context(
            latest_10s,
            pending_breakout,
            history=AlpacaRunner._warrior_history_until(self, sym, latest_10s, count=8),
            window_high=window_high,
            reject_high=float(self._warrior_squeeze_rejection_high.get(sym, 0.0) or 0.0),
            rejection_reason=self._warrior_squeeze_rejection_reason.get(sym),
            reentry_count=int(self._momentum_burst_hit_run_counts.get(sym, 0) or 0),
            min_reclaim_price=float(getattr(self, "_warrior_squeeze_min_reclaim_price", 3.5) or 0.0),
        )

    def _warrior_squeeze_second_leg_reclaim_context(
        self,
        symbol: str,
        latest_10s: Bar,
        pending_breakout: Dict[str, Any],
        *,
        window_high: float,
    ) -> Optional[Dict[str, Any]]:
        return warrior_lanes.warrior_squeeze_second_leg_reclaim_context(
            latest_10s,
            history=AlpacaRunner._warrior_history_until(self, symbol, latest_10s, count=60),
            window_high=window_high,
        )

    def _warrior_prior_runner_continuation_pullback_context(
        self,
        symbol: str,
        latest_10s: Bar,
        *,
        window_high: float,
    ) -> Optional[Dict[str, Any]]:
        return warrior_lanes.warrior_prior_runner_continuation_pullback_context(
            latest_10s,
            history=AlpacaRunner._warrior_history_until(self, symbol, latest_10s, count=36),
            window_high=window_high,
        )

    def _warrior_post_target_pullback_reclaim_context(
        self,
        symbol: str,
        latest_10s: Bar,
        *,
        window_high: float,
    ) -> Optional[Dict[str, Any]]:
        return warrior_lanes.warrior_post_target_pullback_reclaim_context(
            latest_10s,
            history=AlpacaRunner._warrior_history_until(self, symbol, latest_10s, count=36),
            window_high=window_high,
        )

    def _warrior_post_target_fresh_continuation_context(
        self,
        symbol: str,
        latest_10s: Bar,
        *,
        window_high: float,
    ) -> Optional[Dict[str, Any]]:
        return warrior_lanes.warrior_post_target_fresh_continuation_context(
            latest_10s,
            history=AlpacaRunner._warrior_history_until(self, symbol, latest_10s, count=30),
            window_high=window_high,
        )

    def _warrior_failed_burst_recovery_context(
        self,
        symbol: str,
        latest_10s: Bar,
        *,
        failed_high: float,
        window_high: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        return warrior_lanes.warrior_failed_burst_recovery_context(
            latest_10s,
            history=AlpacaRunner._warrior_history_until(self, symbol, latest_10s, count=10),
            failed_high=failed_high,
            window_high=window_high,
        )

    def _warrior_halt_resume_continuation_context(
        self,
        symbol: str,
        latest_10s: Bar,
        *,
        failed_high: float,
        window_high: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        return warrior_lanes.warrior_halt_resume_continuation_context(
            latest_10s,
            history=AlpacaRunner._warrior_history_until(self, symbol, latest_10s, count=20),
            failed_high=failed_high,
            window_high=window_high,
        )

    def _warrior_level_break_starter_context(
        self,
        symbol: str,
        latest_10s: Bar,
        *,
        window_high: float,
    ) -> Optional[Dict[str, Any]]:
        return warrior_lanes.warrior_level_break_starter_context(
            latest_10s,
            history=AlpacaRunner._warrior_history_until(self, symbol, latest_10s, count=1000),
            window_high=window_high,
            min_reclaim_price=float(getattr(self, "_warrior_squeeze_min_reclaim_price", 3.5) or 0.0),
        )

    def _warrior_trend_pullback_reclaim_context(
        self,
        symbol: str,
        latest_10s: Bar,
        *,
        window_high: float,
    ) -> Optional[Dict[str, Any]]:
        history = AlpacaRunner._warrior_history_until(self, symbol, latest_10s, count=90)
        return warrior_lanes.warrior_trend_playbook_context(
            latest_10s,
            history=history,
            window_high=window_high,
        )

    def _warrior_parabolic_micro_pullback_reclaim_context(
        self,
        symbol: str,
        latest_10s: Bar,
        *,
        window_high: float,
    ) -> Optional[Dict[str, Any]]:
        return warrior_lanes.warrior_parabolic_micro_pullback_reclaim_context(
            latest_10s,
            history=AlpacaRunner._warrior_history_until(self, symbol, latest_10s, count=90),
            window_high=window_high,
        )

    def _warrior_squeeze_curl_reclaim_context(
        self,
        symbol: str,
        latest_10s: Bar,
        pending_breakout: Dict[str, Any],
        *,
        window_high: float,
    ) -> Optional[Dict[str, Any]]:
        sym = symbol.upper()
        return warrior_lanes.warrior_squeeze_curl_reclaim_context(
            latest_10s,
            pending_breakout,
            history=AlpacaRunner._warrior_history_until(self, sym, latest_10s, count=6),
            window_high=window_high,
            reentry_count=int(self._momentum_burst_hit_run_counts.get(sym, 0) or 0),
            min_reclaim_price=float(getattr(self, "_warrior_squeeze_min_reclaim_price", 3.5) or 0.0),
        )

    def _momentum_burst_rebase_pending_after_reject(
        self,
        symbol: str,
        latest_10s: Bar,
        pending_breakout: Dict[str, Any],
        reject_reason: str,
        *,
        hit_run: bool,
    ) -> bool:
        """Keep watching a failed burst if it formed a clean 10s micro-base.

        The first follow-through bar often pauses instead of immediately
        clearing the spike high. For hit-run we can keep the armed window alive
        by rebasing the pending trigger to the fresh micro-base, but only when
        the same continuation-base guard says the tape is still healthy.
        """
        if not hit_run:
            return False
        if int(pending_breakout.get("rebase_count") or 0) >= 1:
            return False
        reason = (reject_reason or "").lower()
        if "chasing:" in reason:
            return False
        if not any(
            token in reason
            for token in (
                "breakout not confirmed",
                "did not break continuation high",
                "did not close with strength",
                "volume too light",
            )
        ):
            return False
        continuation_ok, continuation_reason, continuation_meta = (
            self._momentum_burst_continuation_base_ok(symbol)
        )
        if not continuation_ok:
            return False
        current_high = float(latest_10s.high or 0.0)
        current_close = float(latest_10s.close or 0.0)
        current_volume = float(latest_10s.volume or 0.0)
        if current_high <= 0 or current_close <= 0:
            return False
        self._momentum_burst_pending[symbol] = {
            "ts": getattr(latest_10s, "ts", None),
            "original_ts": pending_breakout.get("original_ts") or pending_breakout.get("ts"),
            "breakout_close": current_close,
            "breakout_high": current_high,
            "original_breakout_close": pending_breakout.get("original_breakout_close")
            or pending_breakout.get("breakout_close"),
            "original_breakout_high": pending_breakout.get("original_breakout_high")
            or pending_breakout.get("breakout_high"),
            "breakout_volume": current_volume,
            "micro_base_reclaim": True,
            "base_high": continuation_meta.get("base_high"),
            "base_low": continuation_meta.get("base_low"),
            "prior_reject": reject_reason,
            "rebase_count": int(pending_breakout.get("rebase_count") or 0) + 1,
            **self._momentum_burst_level_context(
                float(pending_breakout.get("breakout_high") or 0.0),
                current_high,
            ),
        }
        for key in (
            "entry_trigger",
            "entry_tier_reason",
            "variant_override",
            "size_factor",
            "reward_risk",
            "max_pay_above_trigger_pct",
            "max_chase_pct",
            "psych_level",
        ):
            if key in pending_breakout and key not in self._momentum_burst_pending[symbol]:
                self._momentum_burst_pending[symbol][key] = pending_breakout.get(key)
        if "psych_level" in pending_breakout and "psych_level" not in self._momentum_burst_pending[symbol]:
            self._momentum_burst_pending[symbol]["psych_level"] = pending_breakout.get("psych_level")
            self._momentum_burst_pending[symbol]["entry_trigger"] = pending_breakout.get(
                "entry_trigger", "psych_level_break"
            )
        if "reset_from_stale_high" in pending_breakout:
            self._momentum_burst_pending[symbol]["reset_from_stale_high"] = pending_breakout.get(
                "reset_from_stale_high"
            )
        self._momentum_burst_window_high[symbol] = current_high
        logger.info(
            "MOMENTUM BURST HIT-RUN %s rebased pending trigger after %s: %s",
            symbol,
            reject_reason,
            continuation_reason,
        )
        return True

    def _momentum_burst_hit_run_time_allowed(self, ts: Optional[datetime] = None) -> bool:
        end_text = str(getattr(self, "_momentum_burst_hit_run_end_et", "") or "").strip()
        if not end_text:
            return True
        try:
            hour_text, minute_text = end_text.split(":", 1)
            end_hour = int(hour_text)
            end_minute = int(minute_text)
        except Exception:
            return True
        try:
            current = ts or datetime.now(timezone.utc)
            if current.tzinfo is None:
                current = current.replace(tzinfo=timezone.utc)
            current_et = current.astimezone(ET).time()
        except Exception:
            return True
        return current_et.hour < end_hour or (
            current_et.hour == end_hour and current_et.minute <= end_minute
        )

    def _latest_momentum_burst_10s_bar(self, symbol: str) -> Optional[Bar]:
        aggregator = getattr(self, "_bar_aggregator", None)
        if aggregator is None:
            return None
        try:
            bars_10s = aggregator.get_latest_10s(symbol, count=4)
        except Exception:
            return None
        if not bars_10s:
            return None
        return bars_10s[-1]

    def _momentum_burst_violent_liquid_ok(self, symbol: str) -> tuple[bool, Dict[str, float]]:
        bars_10s = self._momentum_burst_recent_10s(symbol, count=12)
        if not bars_10s or len(bars_10s) < 6:
            return False, {}
        recent = [b for b in bars_10s[-6:] if float(b.close or 0.0) > 0]
        if len(recent) < 3:
            return False, {}
        ranges = sorted(
            (float(b.high or 0.0) - float(b.low or 0.0)) / float(b.close or 1.0) * 100.0
            for b in recent
        )
        median_range = float(ranges[len(ranges) // 2])
        latest_volume = float(bars_10s[-1].volume or 0.0)
        recent_volume = sum(float(b.volume or 0.0) for b in bars_10s[-3:])
        day_proxy_volume = sum(float(b.volume or 0.0) for b in bars_10s)
        ok = (
            median_range <= 9.0
            and latest_volume >= 50_000
            and recent_volume >= 150_000
            and day_proxy_volume >= 500_000
        )
        return ok, {
            "median_10s_range_pct": round(median_range, 3),
            "latest_10s_volume": round(latest_volume, 0),
            "recent_10s_volume": round(recent_volume, 0),
            "day_proxy_10s_volume": round(day_proxy_volume, 0),
        }

    def _momentum_burst_recent_10s(self, symbol: str, *, count: int = 12) -> List[Bar]:
        aggregator = getattr(self, "_bar_aggregator", None)
        if aggregator is None:
            return []
        try:
            return list(aggregator.get_latest_10s(symbol, count=count) or [])
        except Exception:
            return []

    def _momentum_burst_stop_trading_reason(self, symbol: str) -> str:
        history = self._momentum_burst_recent_10s(symbol, count=12)
        if len(history) < 6:
            return ""
        latest = history[-1]
        rng = float(latest.high or 0.0) - float(latest.low or 0.0)
        close = float(latest.close or 0.0)
        if rng <= 0 or close <= 0:
            return ""
        range_pct = rng / close * 100.0
        upper_wick = (float(latest.high or 0.0) - max(float(latest.open or 0.0), close)) / rng
        prior_vol = [float(b.volume or 0.0) for b in history[-6:-1]]
        avg_prior_vol = sum(prior_vol) / len(prior_vol) if prior_vol else 0.0
        is_red = close < float(latest.open or 0.0)
        if upper_wick >= 0.78 and range_pct >= 6.0 and float(latest.volume or 0.0) >= avg_prior_vol * 1.1:
            return "big topping wick in 10s burst tape"
        if is_red and range_pct >= 6.0 and float(latest.volume or 0.0) >= avg_prior_vol * 1.2:
            return "heavy red dump candle in 10s burst tape"
        if len(history) >= 10:
            closes = [float(b.close or 0.0) for b in history[-9:]]
            ema = closes[0]
            alpha = 2.0 / 10.0
            for value in closes[1:]:
                ema = value * alpha + ema * (1.0 - alpha)
            if close < ema * 0.985 and close < min(float(b.low or 0.0) for b in history[-5:-1]):
                return "lost 10s trend support"
        return ""

    def _momentum_burst_continuation_base_ok(self, symbol: str) -> tuple[bool, str, Dict[str, float]]:
        history = self._momentum_burst_recent_10s(symbol, count=12)
        if len(history) < 8:
            return False, "not enough 10s history", {}
        latest = history[-1]
        prior = history[-6:-1]
        close = float(latest.close or 0.0)
        if close <= float(latest.open or 0.0):
            return False, "confirm bar is not green", {}
        latest_volume = float(latest.volume or 0.0)
        recent_volume = sum(float(b.volume or 0.0) for b in history[-3:])
        if latest_volume < 50_000 or recent_volume < 150_000:
            return False, "volume faded", {
                "latest_10s_volume": round(latest_volume, 0),
                "recent_10s_volume": round(recent_volume, 0),
            }
        base_high = max(float(b.high or 0.0) for b in prior)
        base_low = min(float(b.low or 0.0) for b in prior)
        base_range_pct = (base_high - base_low) / close * 100.0 if close > 0 else 999.0
        pullback_pct = (base_high - base_low) / base_high * 100.0 if base_high > 0 else 999.0
        fresh_high = float(latest.high or 0.0) >= base_high or close >= max(float(b.close or 0.0) for b in prior)
        if not fresh_high:
            return False, "no fresh 10s high/reclaim", {
                "base_range_pct": round(base_range_pct, 2),
                "pullback_pct": round(pullback_pct, 2),
            }
        if base_range_pct > 18.0 or pullback_pct > 18.0:
            return False, "pullback/base too wide", {
                "base_range_pct": round(base_range_pct, 2),
                "pullback_pct": round(pullback_pct, 2),
            }
        red_dump = any(
            float(b.close or 0.0) < float(b.open or 0.0)
            and (float(b.open or 0.0) - float(b.close or 0.0)) / float(b.close or 1.0) * 100.0 > 5.0
            for b in history[-4:-1]
        )
        if red_dump:
            return False, "recent pullback had a dump candle", {}
        return True, "fresh continuation base", {
            "base_high": round(base_high, 4),
            "base_low": round(base_low, 4),
            "base_range_pct": round(base_range_pct, 2),
            "pullback_pct": round(pullback_pct, 2),
            "latest_10s_volume": round(latest_volume, 0),
            "recent_10s_volume": round(recent_volume, 0),
        }

    def _execute_momentum_burst_scalp(
        self,
        sym: str,
        latest_10s: Bar,
        *,
        hit_run: bool = False,
        violent_liquid: bool = False,
        post_blowoff_micro_base: bool = False,
        entry_context: Optional[Dict[str, Any]] = None,
        strategy_override: Optional[str] = None,
        size_factor_override: Optional[float] = None,
        window_high: Optional[float] = None,
    ) -> Optional[Fill]:
        strategy_label = (
            strategy_override
            or (
                "post_blowoff_micro_base_scout"
                if post_blowoff_micro_base
                else ("momentum_burst_hit_run" if hit_run else "momentum_burst_scalp")
            )
        )
        log_label = (
            "WARRIOR SQUEEZE"
            if strategy_label == "warrior_squeeze_playbook"
            else ("MOMENTUM BURST HIT-RUN" if hit_run else "MOMENTUM BURST SCALP")
        )
        stage_prefix = strategy_label
        warrior_override = bool(
            entry_context
            and warrior_lanes.is_warrior_entry_trigger(
                entry_context.get("entry_trigger")
            )
        )
        bars = list(self._bar_buffer.get(sym, deque()))
        if len(bars) < 3:
            return None

        if strategy_label == "warrior_squeeze_playbook" and not warrior_override:
            reason = (
                "warrior setup required: generic momentum/violent-liquid "
                "fallback is not a playbook lane"
            )
            logger.info("WARRIOR SQUEEZE reject %s: %s", sym, reason)
            self._record_entry_reject(
                self._quick_scalp_probe_signal(sym, bars[-1].close, strategy_label),
                stage="{}_setup".format(stage_prefix),
                reason=reason,
                source=strategy_label,
                price=bars[-1].close,
            )
            return None

        if warrior_override and strategy_label == "warrior_squeeze_playbook":
            if violent_liquid:
                violent_reject = warrior_lanes.warrior_violent_liquid_reject(
                    latest_10s,
                    history=AlpacaRunner._warrior_history_until(
                        self, sym, latest_10s, count=12
                    ),
                    target_wins=int(self._warrior_squeeze_target_wins.get(sym, 0) or 0),
                    entry_trigger=str((entry_context or {}).get("entry_trigger") or ""),
                )
                if violent_reject:
                    logger.info("WARRIOR SQUEEZE reject %s: %s", sym, violent_reject)
                    self._record_entry_reject(
                        self._quick_scalp_probe_signal(sym, bars[-1].close, strategy_label),
                        stage="{}_violent_liquid".format(stage_prefix),
                        reason=violent_reject,
                        source=strategy_label,
                        price=bars[-1].close,
                    )
                    return None
            late_reentry_reject = warrior_lanes.warrior_late_reentry_reject(
                latest_10s,
                history=AlpacaRunner._warrior_history_until(
                    self, sym, latest_10s, count=12
                ),
                window_high=float(
                    window_high
                    if window_high is not None
                    else (entry_context or {}).get("window_high") or 0.0
                ),
                reentry_count=int(self._momentum_burst_hit_run_counts.get(sym, 0) or 0),
                target_wins=int(self._warrior_squeeze_target_wins.get(sym, 0) or 0),
                entry_trigger=str((entry_context or {}).get("entry_trigger") or ""),
            )
            if late_reentry_reject:
                logger.info(
                    "WARRIOR SQUEEZE reject %s: %s",
                    sym,
                    late_reentry_reject,
                )
                self._record_entry_reject(
                    self._quick_scalp_probe_signal(sym, bars[-1].close, strategy_label),
                    stage="{}_late_reentry".format(stage_prefix),
                    reason=late_reentry_reject,
                    source=strategy_label,
                    price=bars[-1].close,
                )
                return None
            if (
                str((entry_context or {}).get("entry_trigger") or "")
                == "warrior_failed_burst_recovery"
            ):
                realized_pnl = self._realized_symbol_pnl(sym)
                if realized_pnl > 0.0:
                    reason = (
                        "skip failed-burst Warrior recovery after ${:.2f} "
                        "already banked on symbol"
                    ).format(realized_pnl)
                    logger.info("WARRIOR SQUEEZE reject %s: %s", sym, reason)
                    self._record_entry_reject(
                        self._quick_scalp_probe_signal(sym, bars[-1].close, strategy_label),
                        stage="{}_profit_lock".format(stage_prefix),
                        reason=reason,
                        source=strategy_label,
                        price=bars[-1].close,
                    )
                    return None

        if not warrior_override:
            reject = self._check_quick_scalp_entry(sym, bars)
            if reject is not None:
                logger.info("%s reject %s: %s", log_label, sym, reject)
                self._record_entry_reject(
                    self._quick_scalp_probe_signal(sym, bars[-1].close, strategy_label),
                    stage="{}_shape".format(stage_prefix),
                    reason=reject,
                    source=strategy_label,
                    price=bars[-1].close,
                )
                return None

        quality_bars = (bars + [latest_10s]) if warrior_override else bars
        quality_reject = self._shared_entry_quality_reject(
            sym,
            quality_bars,
            stage="{}_final_guard".format(stage_prefix),
            source=strategy_label,
        )
        if quality_reject is not None:
            logger.info("%s reject %s: shared entry quality %s", log_label, sym, quality_reject)
            return None

        if not warrior_override:
            ten_second_reject = self._quick_scalp_10s_reject(sym)
            if ten_second_reject is not None:
                logger.info("%s reject %s: %s", log_label, sym, ten_second_reject)
                self._record_entry_reject(
                    self._quick_scalp_probe_signal(sym, bars[-1].close, strategy_label),
                    stage="{}_10s".format(stage_prefix),
                    reason=ten_second_reject,
                    source=strategy_label,
                    price=bars[-1].close,
                )
                return None

        if warrior_override:
            price = float(entry_context.get("entry_price_override") or 0.0)
            stop_price = float(entry_context.get("stop_price_override") or 0.0)
            target_price = float(entry_context.get("target_price_override") or 0.0)
            rr_note = str(entry_context.get("rr_note_override") or "warrior level pull-away starter")
            if price <= 0 or stop_price <= 0 or target_price <= price or stop_price >= price:
                return None
        else:
            rr = self._quick_scalp_tick_rr(sym, bars, float(latest_10s.close or bars[-1].close))
            if rr is None:
                logger.info("%s reject %s: no usable tick R:R", log_label, sym)
                self._record_entry_reject(
                    self._quick_scalp_probe_signal(sym, bars[-1].close, strategy_label),
                    stage="{}_rr".format(stage_prefix),
                    reason="no usable tick R:R",
                    source=strategy_label,
                    price=bars[-1].close,
                )
                return None
            price, stop_price, target_price, rr_note = rr
        risk_per_share = price - stop_price
        if warrior_override:
            pass
        elif post_blowoff_micro_base:
            risk_per_share = max(price * 0.015, 0.06)
            stop_price = round(price - risk_per_share, 2)
            target_price = round(price + risk_per_share, 2)
            rr_note = "post-blowoff micro-base scout 1R risk={:.1f}% target={:.1f}%".format(
                risk_per_share / price * 100.0 if price else 0.0,
                (target_price - price) / price * 100.0 if price else 0.0,
            )
        elif hit_run and violent_liquid:
            risk_per_share = max(price * 0.02, 0.06)
            stop_price = round(price - risk_per_share, 2)
            target_price = round(price + risk_per_share, 2)
            rr_note = "violent-liquid hit-run 1R risk={:.1f}% target={:.1f}%".format(
                risk_per_share / price * 100.0 if price else 0.0,
                (target_price - price) / price * 100.0 if price else 0.0,
            )
        if hit_run and not warrior_override and risk_per_share > 0:
            target_price = round(
                price + risk_per_share * max(0.1, float(self._momentum_burst_hit_run_reward_risk)),
                2,
            )
            if not violent_liquid:
                rr_note = "hit-run 1R risk={:.1f}% target={:.1f}%".format(
                    risk_per_share / price * 100.0 if price else 0.0,
                    (target_price - price) / price * 100.0 if price else 0.0,
                )
        if (
            hit_run
            and not bool(entry_context and entry_context.get("skip_unstable_confirm_stop_check"))
            and float(latest_10s.low or 0.0) <= float(stop_price or 0.0)
        ):
            reason = "confirm 10s bar already traded through planned stop"
            logger.info(
                "%s reject %s: %s (low %.2f <= stop %.2f)",
                log_label,
                sym,
                reason,
                float(latest_10s.low or 0.0),
                float(stop_price or 0.0),
            )
            self._record_entry_reject(
                self._quick_scalp_probe_signal(sym, bars[-1].close, strategy_label),
                stage="{}_unstable_confirm".format(stage_prefix),
                reason=reason,
                source=strategy_label,
                price=bars[-1].close,
            )
            return None
        quantity = self._capital_aware_quantity(
            price,
            stop_price,
            max_dollar_risk=getattr(self, "_max_dollar_risk_per_trade", 50.0),
        )
        if quantity < 1:
            logger.info("%s skip %s — position too large for buying power", log_label, sym)
            return None
        spread_size_factor = float(
            getattr(self, "_quick_scalp_spread_size_factors", {}).pop(sym, 1.0)
        )
        if 0 < spread_size_factor < 1.0 and quantity > 1:
            original_quantity = quantity
            quantity = max(1, int(quantity * spread_size_factor))
            logger.info(
                "MOMENTUM BURST SCALP %s size down %d → %d for opportunity-scaled spread",
                sym,
                original_quantity,
                quantity,
            )
        if post_blowoff_micro_base and quantity > 1:
            original_quantity = quantity
            quantity = max(1, int(quantity * 0.35))
            logger.info(
                "POST-BLOWOFF MICRO-BASE %s size down %d -> %d",
                sym,
                original_quantity,
                quantity,
            )
        elif (
            hit_run
            and violent_liquid
            and size_factor_override is None
            and quantity > 1
        ):
            original_quantity = quantity
            quantity = max(1, int(quantity * 0.35))
            logger.info(
                "MOMENTUM BURST HIT-RUN %s violent-liquid size down %d -> %d",
                sym,
                original_quantity,
                quantity,
            )
        if size_factor_override is not None and quantity > 1:
            factor = max(0.01, min(1.0, float(size_factor_override or 1.0)))
            if factor < 1.0:
                original_quantity = quantity
                quantity = max(1, int(quantity * factor))
                logger.info(
                    "%s %s starter size down %d -> %d (factor %.2f)",
                    log_label,
                    sym,
                    original_quantity,
                    quantity,
                    factor,
                )
        if warrior_override and quantity > 0 and risk_per_share > 0:
            position_value = float(
                getattr(self, "_warrior_squeeze_position_value", 0.0) or 0.0
            )
            risk_cap = float(
                getattr(self, "_warrior_squeeze_max_dollar_risk", 0.0) or 0.0
            )
            if position_value > 0:
                qty_by_value = int(position_value / price)
                qty_by_risk = int(risk_cap / risk_per_share) if risk_cap > 0 else qty_by_value
                warrior_quantity = min(qty_by_value, qty_by_risk)
                if 0 < spread_size_factor < 1.0:
                    warrior_quantity = int(warrior_quantity * spread_size_factor)
                if warrior_quantity > 0:
                    original_quantity = quantity
                    quantity = max(1, min(quantity, warrior_quantity))
                    logger.info(
                        "WARRIOR SQUEEZE %s position-value size %d -> %d "
                        "(value cap $%.0f, risk cap $%.0f)",
                        sym,
                        original_quantity,
                        quantity,
                        position_value,
                        risk_cap,
                    )

        signal = TradeSignal(
            symbol=sym,
            action=SignalAction.ENTER_LONG,
            quantity=quantity,
            entry_price=price,
            stop_loss=stop_price,
            take_profit=target_price,
            max_hold_seconds=(
                float(entry_context.get("max_hold_seconds_override"))
                if entry_context and entry_context.get("max_hold_seconds_override")
                else (self._momentum_burst_hit_run_max_hold_sec if hit_run else 90)
            ),
            reason="{} {} ${:.2f}, stop=${:.2f}, target=${:.2f} ({})".format(
                "Warrior Squeeze" if strategy_label == "warrior_squeeze_playbook"
                else ("Momentum Burst Hit-Run" if hit_run else "Momentum Burst Scalp"),
                sym, price, stop_price, target_price, rr_note),
            scan_result=ScanResult(
                symbol=sym,
                scanner_name=strategy_label,
                ts=datetime.now(timezone.utc),
                score=0.0,
                criteria={
                    "pattern": strategy_label,
                    "direction": "up",
                    "entry_mode": strategy_label,
                    "setup_tier": "A+ setup",
                    "source_scanner": "momentum_burst",
                    "max_hit_run_entries": self._momentum_burst_hit_run_max_entries if hit_run else None,
                    "warrior_max_entries": (
                        int(getattr(self, "_warrior_squeeze_max_entries", 3) or 3)
                        if strategy_label == "warrior_squeeze_playbook" else None
                    ),
                    "variant": (
                        entry_context.get("variant_override")
                        if entry_context and entry_context.get("variant_override")
                        else (
                            warrior_lanes.warrior_variant_for_entry_trigger(
                                entry_context.get("entry_trigger") if entry_context else ""
                            )
                            if warrior_override
                            else "warrior_reclaim_starter"
                            if strategy_label == "warrior_squeeze_playbook"
                            else (
                                "post_blowoff_micro_base"
                                if post_blowoff_micro_base
                                else ("violent_liquid" if violent_liquid else "smooth_confirmed")
                            )
                        )
                    ),
                    "size_factor": (
                        round(max(0.01, min(1.0, float(size_factor_override))), 2)
                        if size_factor_override is not None
                        else (0.35 if (violent_liquid or post_blowoff_micro_base) else 1.0)
                    ),
                    **(
                        {
                            key: entry_context.get(key)
                            for key in (
                                "psych_level",
                                "entry_trigger",
                                "variant_override",
                                "pullaway_level",
                                "max_pay",
                            )
                            if entry_context.get(key) is not None
                        }
                        if entry_context else {}
                    ),
                    **(
                        {
                            "spread_exception": "opportunity_scaled",
                            "spread_size_factor": round(spread_size_factor, 2),
                        }
                        if 0 < spread_size_factor < 1.0 else {}
                    ),
                },
            ),
            trend_strength=0.8,
        )
        order = Order(symbol=sym, side=Side.BUY, quantity=quantity, limit_price=price)
        bar = bars[-1]
        fill, status = self._broker.submit(order, bar, self._pipeline.portfolio)
        try:
            from daytrading.ml.shadow_collector import log_execution_quality
            log_execution_quality(
                order=order, bar=bar, status=status, fill=fill,
                source=strategy_label,
            )
        except Exception:
            pass
        if fill:
            from daytrading.execution.broker import apply_fill
            if float(target_price or 0.0) <= float(fill.price or 0.0):
                planned_reward = max(
                    float(target_price or 0.0) - float(price or 0.0),
                    0.0,
                )
                reward = max(
                    planned_reward,
                    float(fill.price or 0.0) - float(stop_price or 0.0),
                    float(fill.price or 0.0) * 0.01,
                    0.04,
                )
                target_price = round(float(fill.price or 0.0) + reward, 4)
                signal = TradeSignal(
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=signal.quantity,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=target_price,
                    trailing_stop_offset=signal.trailing_stop_offset,
                    max_hold_seconds=signal.max_hold_seconds,
                    reason=signal.reason,
                    scan_result=signal.scan_result,
                    trend_strength=signal.trend_strength,
                )
            apply_fill(self._pipeline.portfolio, fill)
            self._on_position_opened(
                signal,
                fill,
                strategy=strategy_label,
                execution_method="momentum_burst_hit_run" if hit_run else "momentum_burst_new_high",
            )
            self._breakout_scalp_active = True
            if hit_run:
                self._momentum_burst_hit_run_counts[sym] = (
                    self._momentum_burst_hit_run_counts.get(sym, 0) + 1
                )
                if strategy_label == "warrior_squeeze_playbook":
                    self._warrior_squeeze_last_entry_trigger[sym] = str(
                        (signal.scan_result.criteria or {}).get("entry_trigger") or ""
                    )
                if (
                    signal.scan_result
                    and (signal.scan_result.criteria or {}).get("entry_trigger")
                    == "warrior_post_target_pullback_reclaim"
                ):
                    self._warrior_squeeze_post_target_reclaim_allowed[sym] = max(
                        0,
                        int(
                            self._warrior_squeeze_post_target_reclaim_allowed.get(sym, 0)
                            or 0
                        ) - 1,
                    )
            else:
                self._breakout_scalp_cooldown[sym] = (
                    time.monotonic() + self._momentum_burst_scalp_cooldown_sec
                )
                self._pipeline._symbol_entry_counts[sym] = self._pipeline._symbol_entry_counts.get(sym, 0) + 1
            logger.info(
                "%s ENTRY %s %.0f @ $%.4f stop=$%.2f target=$%.2f %s",
                log_label,
                sym, fill.quantity, fill.price, stop_price, target_price, rr_note,
            )
            # Tag ignition entries distinctly for the dashboard (the trade record's
            # strategy is what the live "Warrior Ignition" panel reads). The actual
            # strategy logic still uses strategy_label, so behavior is unchanged.
            hub_strategy = (
                "warrior_ignition"
                if (entry_context and entry_context.get("entry_trigger") == "warrior_ignition")
                else strategy_label
            )
            self._hub.on_fill(fill, "entry", strategy=hub_strategy)
            self._hub.add_log(
                "INFO",
                "{} {} {:.0f} @ ${:.2f}".format(log_label, sym, fill.quantity, fill.price),
            )
            self._journal.record("trade_fill", {
                "symbol": sym,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.price,
                "ts": fill.ts,
                "trade_type": "entry",
                "strategy": strategy_label,
                "execution_method": "momentum_burst_hit_run" if hit_run else "momentum_burst_new_high",
                "market_context": {"phase": self._market_phase()},
            }, ts=fill.ts)
            self._seed_recent_order_ids()
            return fill

        logger.warning("%s order not filled %s (status=%s)", log_label, sym, status)
        self._record_entry_reject(
            signal,
            stage="{}_order".format(stage_prefix),
            reason="order_{}".format(status.value if status else "not_filled"),
            source=strategy_label,
            price=price,
            metadata={"status": status.value if status else "not_filled"},
        )
        return None

    def _quick_scalp_10s_reject(self, symbol: str) -> Optional[str]:
        """Require fresh 10-second confirmation before instant breakout entry."""
        aggregator = getattr(self, "_bar_aggregator", None)
        if aggregator is None:
            return "no 10s confirmation feed"
        try:
            bars_10s = aggregator.get_latest_10s(symbol, count=2)
        except Exception:
            return "no 10s confirmation feed"
        if not bars_10s:
            return "waiting for 10s confirmation"

        latest = bars_10s[-1]
        if latest.ts is not None:
            try:
                bar_time = latest.ts if latest.ts.tzinfo else latest.ts.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - bar_time).total_seconds()
                if age > 30:
                    return "stale 10s confirmation ({:.0f}s old)".format(age)
            except Exception:
                pass

        if latest.close <= latest.open:
            return "10s confirmation red/flat"
        if len(bars_10s) >= 2:
            prev = bars_10s[-2]
            if prev.close > 0 and latest.high <= prev.high and latest.close < prev.close * 1.003:
                return "10s confirmation no expansion"
        return None

    def _breakout_scalp_10s_reject(self, symbol: str) -> Optional[str]:
        """Stricter 10s confirmation for instant HOD/breakout scalps."""
        base_reject = self._quick_scalp_10s_reject(symbol)
        if base_reject is not None:
            return base_reject

        aggregator = getattr(self, "_bar_aggregator", None)
        if aggregator is None:
            return "no 10s confirmation feed"
        try:
            bars_10s = aggregator.get_latest_10s(symbol, count=2)
        except Exception:
            return "no 10s confirmation feed"
        if not bars_10s:
            return "waiting for 10s confirmation"

        latest = bars_10s[-1]
        bar_range = float(latest.high or 0.0) - float(latest.low or 0.0)
        if bar_range > 0:
            close_location = (float(latest.close) - float(latest.low)) / bar_range
            if close_location < 0.65:
                return "10s confirmation weak close ({:.0%} location)".format(close_location)
            price = float(latest.close or 0.0)
            range_pct = (bar_range / price) if price > 0 else 0.0
            if range_pct >= 0.06 and close_location < 0.75:
                return "10s breakout candle too volatile without strong close ({:.0%} location, {:.1%} range)".format(
                    close_location,
                    range_pct,
                )
        if len(bars_10s) >= 2:
            prev = bars_10s[-2]
            if float(prev.close or 0.0) > 0 and latest.close < float(prev.close) * 0.998:
                return "10s confirmation faded below prior close"
            if float(prev.high or 0.0) > 0 and latest.high <= float(prev.high) * 1.001:
                return "10s confirmation no expansion"
            prev_volume = float(prev.volume or 0.0)
            latest_volume = float(latest.volume or 0.0)
            if prev_volume > 0 and latest_volume < prev_volume * 0.5:
                return "10s confirmation volume faded {:.0f} < 50% prior {:.0f}".format(
                    latest_volume, prev_volume)
        for recent in bars_10s[-4:-1]:
            recent_open = float(recent.open or 0.0)
            recent_close = float(recent.close or 0.0)
            recent_high = float(recent.high or 0.0)
            recent_low = float(recent.low or 0.0)
            if recent_open <= 0 or recent_close >= recent_open:
                continue
            recent_range = recent_high - recent_low
            recent_range_pct = (recent_range / recent_open) if recent_open > 0 else 0.0
            body_pct = ((recent_open - recent_close) / recent_open) if recent_open > 0 else 0.0
            close_location = ((recent_close - recent_low) / recent_range) if recent_range > 0 else 1.0
            if (
                body_pct >= 0.025
                and recent_range_pct >= 0.035
                and close_location <= 0.25
                and float(recent.volume or 0.0) >= 75_000
            ):
                return "recent 10s dump candle before breakout ({:.1%} body, {:.0%} close location)".format(
                    body_pct,
                    close_location,
                )
        return None

    def _check_quick_scalp_entry(self, symbol: str, bars: Sequence[Bar]) -> Optional[str]:
        """Fast-mover guard for HOD tick scalps.

        This local guard handles the HOD/tape-specific shape. The caller still
        runs shared entry quality, ML, and 10s confirmation before ordering.
        """
        if len(bars) < 3:
            return "insufficient bars for quick scalp"

        latest = bars[-1]
        price = latest.close
        if price <= 0:
            return "invalid price"
        if price < 1.5 or price > 20.0:
            return "quick scalp price ${:.2f} outside range $1.50-$20.00".format(price)

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
                avg_spread = sum(q.ask - q.bid for q in recent_quotes) / len(recent_quotes)
                avg_mid = sum((q.ask + q.bid) / 2.0 for q in recent_quotes) / len(recent_quotes)
                avg_spread_pct = avg_spread / avg_mid * 100 if avg_mid > 0 else 0.0
                momentum_pct = max(day_change, recent_move)
                max_spread = 1.5 if day_volume >= 1_000_000 and momentum_pct >= 50.0 else 0.8
                avg_depth = sum(min(q.bid_size, q.ask_size) for q in recent_quotes) / len(recent_quotes)
                spread_decision = assess_opportunity_scaled_spread(
                    price=avg_mid or price,
                    spread=avg_spread,
                    pattern="breakout_scalp",
                    setup_tier="A+ setup" if momentum_pct >= 50.0 else "",
                    entry_tier="",
                    day_volume=day_volume,
                    recent_avg_volume=recent_volume / 3.0,
                    latest_volume=float(getattr(latest, "volume", 0.0) or 0.0),
                    distance_from_hod=tradeable_hod_distance / 100.0,
                    quote_depth=avg_depth,
                    normal_pct_limit=max_spread / 100.0,
                    setup_score=0.0,
                )
                if not spread_decision.ok:
                    return "quick scalp spread too wide {:.2f}% ({:.2f}c, max {:.1f}% or 1 tick)".format(
                        avg_spread_pct, avg_spread * 100.0, max_spread)
                factors = getattr(self, "_quick_scalp_spread_size_factors", None)
                if factors is None:
                    factors = {}
                    self._quick_scalp_spread_size_factors = factors
                if spread_decision.exception:
                    factors[symbol] = spread_decision.size_factor
                else:
                    factors.pop(symbol, None)

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
        refresh_symbols = set(self._watchlist_set) | _ensure_market_data_service(self).hot_watch_keys()
        priority_syms = [
            sym for sym in list(self._priority_bar_refresh)
            if sym in refresh_symbols and sym not in self._watchlist_pinned
        ]
        for sym in refresh_symbols:
            if sym in self._watchlist_pinned:
                continue
            if sym in self._priority_bar_refresh:
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
            if priority_syms:
                stale_syms = priority_syms
            else:
                return

        ordered = []
        seen_refresh = set()
        for sym in priority_syms + stale_syms:
            if sym in seen_refresh:
                continue
            seen_refresh.add(sym)
            ordered.append(sym)
        batch = ordered[:10]
        for sym in batch:
            self._priority_bar_refresh.discard(sym)
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
            from daytrading.ml.data_collector import update_deferred_entry_outcomes
            entry_labeled = update_deferred_entry_outcomes(bar_universe)
            if entry_labeled:
                logger.info("ML ENTRY LABELS: labeled %d deferred entry candidates", entry_labeled)
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
                    entry_strategy=getattr(pos, "entry_strategy", ""),
                    entry_pattern=getattr(pos, "entry_pattern", ""),
                    entry_score=getattr(pos, "entry_score", None),
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
        for decision in getattr(result, "entry_decisions", []) or []:
            try:
                # Skip watch-only rows — shadow scanners (momentum_burst,
                # bull_flag, level_breakout_watch) log "collecting data, not live
                # A+" every cycle. They are monitoring chatter, not real entry
                # attempts, and would dominate the funnel and bloat the journal.
                if self._is_watch_only_decision(decision):
                    continue
                payload = dict(decision)
                payload["source"] = payload.get("source") or "pipeline"
                payload["market_phase"] = self._market_phase()
                payload["cycle"] = cycle_num
                self._journal.record("entry_decision", payload)
            except Exception:
                pass

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
            self._maybe_arm_momentum_burst_scalp(hit)
            self._hub.on_scan_hit(hit, verified=is_verified, reject_reason=reject_reason)
            if self.is_hot_watch_active(hit.symbol):
                self._journal.record("hot_watch", {
                    "symbol": hit.symbol,
                    "stage": "pattern_found",
                    "scanner": hit.scanner_name,
                    "score": hit.score,
                    "verified": is_verified,
                    "reject_reason": reject_reason,
                    "criteria": dict(hit.criteria),
                })
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
                if "stale data" in str(reject_reason).lower():
                    self._request_priority_bar_refresh(hit.symbol)
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
            strategy = result.entry_strategies.get(f.symbol, "")
            logger.info(
                "[Cycle %d] ENTRY %s %s %.0f @ $%.2f",
                cycle_num, f.side.value.upper(), f.symbol, f.quantity, f.price,
            )
            self._hub.on_fill(f, "entry", strategy=strategy)
            self._hub.add_log("INFO", "ENTRY {} {} {:.0f} @ ${:.2f}".format(
                f.side.value.upper(), f.symbol, f.quantity, f.price))
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
                strategy = result.entry_strategies.get(f.symbol, "")
                logger.info(
                    "[Cycle %d] SCALE UP %s %s +%.0f @ $%.2f",
                    cycle_num, f.side.value.upper(), f.symbol, f.quantity, f.price,
                )
                self._hub.on_fill(f, "scale_up", strategy=strategy)
                self._hub.add_log("INFO", "SCALE UP {} +{:.0f} @ ${:.2f}".format(
                    f.symbol, f.quantity, f.price))
                self._journal.record("trade_fill", {
                    "symbol": f.symbol,
                    "side": f.side.value,
                    "quantity": f.quantity,
                    "price": f.price,
                    "ts": f.ts,
                    "trade_type": "scale_up",
                    "strategy": strategy,
                }, ts=f.ts)

        # Push re-entry fills
        if hasattr(result, 'reentry_fills') and result.reentry_fills:
            for f in result.reentry_fills:
                strategy = result.entry_strategies.get(f.symbol, "")
                logger.info(
                    "[Cycle %d] RE-ENTRY %s %s %.0f @ $%.2f",
                    cycle_num, f.side.value.upper(), f.symbol, f.quantity, f.price,
                )
                self._hub.on_fill(f, "reentry", strategy=strategy)
                self._hub.add_log("INFO", "RE-ENTRY {} {:.0f} @ ${:.2f}".format(
                    f.symbol, f.quantity, f.price))
                self._journal.record("trade_fill", {
                    "symbol": f.symbol,
                    "side": f.side.value,
                    "quantity": f.quantity,
                    "price": f.price,
                    "ts": f.ts,
                    "trade_type": "reentry",
                    "strategy": strategy,
                }, ts=f.ts)

        # Queue deferred signals into execution timer (10-sec micro-entry).
        # Pin the per-symbol chase anchor on first defer so re-queues of a
        # grinding name keep the original level instead of crawling up.
        for sig in getattr(result, 'deferred_signals', []):
            fallback_reject = self._warrior_normal_fallback_reject(sig)
            if fallback_reject:
                self._hub.on_rejected()
                self._journal.record("mistake", {
                    "kind": "risk_rejection",
                    "symbol": sig.symbol,
                    "reason": fallback_reject,
                    "cycle": cycle_num,
                    "phase": self._market_phase(),
                })
                self._hub.add_log("INFO", "SKIP {}: {}".format(sig.symbol, fallback_reject))
                logger.info("WARRIOR FALLBACK REJECT %s: %s", sig.symbol, fallback_reject)
                continue
            overtrade_reject = self._normal_fallback_overtrade_reject(sig)
            if overtrade_reject:
                self._hub.on_rejected()
                self._journal.record("mistake", {
                    "kind": "risk_rejection",
                    "symbol": sig.symbol,
                    "reason": overtrade_reject,
                    "cycle": cycle_num,
                    "phase": self._market_phase(),
                })
                self._hub.add_log("INFO", "SKIP {}: {}".format(sig.symbol, overtrade_reject))
                logger.info("NORMAL FALLBACK OVERTRADE REJECT %s: %s", sig.symbol, overtrade_reject)
                continue
            self._timed_entry_chase_anchor(sig)
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

        try:
            self._pipeline.missed_a_plus.update_prices(
                universe, now=datetime.now(timezone.utc),
            )
            self._hub.on_missed_a_plus(self._pipeline.missed_a_plus_report())
            self._hub.on_scanner_near_miss(self._pipeline.scanner_near_miss_summary())
        except Exception:
            pass

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

    def _track_warrior_normal_fallback_state(
        self,
        symbol: str,
        layer: str,
        reason: str,
    ) -> None:
        if not getattr(self, "_warrior_squeeze_enabled", False):
            return
        sym = str(symbol or "").upper()
        if not sym:
            return
        text = "{} {}".format(layer, reason).lower()
        watch_terms = (
            "unconfirmed",
            "trend_pullback_wait",
            "final_guard",
            "time_window",
            "micro_base_wait",
            "failed burst",
            "not confirmed",
        )
        if not any(term in text for term in watch_terms):
            return
        self._warrior_normal_fallback_rejects[sym] = (
            self._warrior_normal_fallback_rejects.get(sym, 0) + 1
        )
        self._warrior_normal_fallback_last_reason[sym] = reason or layer

    def _warrior_normal_fallback_reject(self, signal: TradeSignal) -> Optional[str]:
        if not getattr(self, "_warrior_squeeze_enabled", False):
            return None
        if signal.action not in (SignalAction.ENTER_LONG, SignalAction.REENTER_LONG):
            return None
        hit = signal.scan_result
        if hit is None:
            return None
        criteria = hit.criteria or {}
        pattern = str(criteria.get("pattern") or hit.scanner_name or "")
        if pattern not in (
            "abc_continuation",
            "pullback_base",
            "level_breakout_reclaim",
            "vwap_pullback",
            "hod_reclaim",
        ):
            return None
        sym = str(signal.symbol or "").upper()
        setup_tier = str(criteria.get("setup_tier") or "")
        try:
            score = float(criteria.get("entry_score") or hit.score or 0.0)
        except (TypeError, ValueError):
            score = float(hit.score or 0.0)
        if "A+" in setup_tier and score >= 90.0:
            return None
        failed_reason = getattr(self, "_warrior_failed_momentum", {}).get(sym)
        if failed_reason:
            return (
                "Warrior/Ignition already failed on {} today ({}); "
                "blocking weak normal {} fallback"
            ).format(sym, failed_reason, pattern)
        watch_count = int(self._warrior_normal_fallback_rejects.get(sym, 0) or 0)
        if watch_count < 3:
            return None
        last_reason = self._warrior_normal_fallback_last_reason.get(sym, "not confirmed")
        return (
            "Warrior watched {} but did not confirm a named setup ({} rejects; last: {}); "
            "blocking weak normal {} fallback"
        ).format(sym, watch_count, last_reason, pattern)

    def _normal_fallback_overtrade_reject(self, signal: TradeSignal) -> Optional[str]:
        if signal.action not in (SignalAction.ENTER_LONG, SignalAction.REENTER_LONG):
            return None
        hit = signal.scan_result
        if hit is None:
            return None
        criteria = hit.criteria or {}
        pattern = str(criteria.get("pattern") or hit.scanner_name or "")
        if pattern not in ("abc_continuation", "pullback_base", "level_breakout_reclaim"):
            return None
        entry_count = int(self._pipeline._symbol_entry_counts.get(signal.symbol, 0) or 0)
        if entry_count < 2:
            return None
        setup_tier = str(criteria.get("setup_tier") or "")
        try:
            score = float(criteria.get("entry_score") or hit.score or 0.0)
        except (TypeError, ValueError):
            score = float(hit.score or 0.0)
        if "A+" in setup_tier and score >= 90.0:
            return None
        return (
            "normal {} overtrade: {} prior entries on {}; needs A+ score >=90 for another attempt"
        ).format(pattern, entry_count, signal.symbol)

    def _realized_symbol_pnl(self, symbol: str) -> float:
        sym = str(symbol or "").upper()
        total = 0.0
        try:
            trades = list(getattr(self._hub, "trades", []))
        except Exception:
            return 0.0
        for trade in trades:
            try:
                trade_symbol = str(getattr(trade, "symbol", "") or "").upper()
                pnl = getattr(trade, "pnl", None)
            except Exception:
                continue
            if trade_symbol != sym or pnl is None:
                continue
            try:
                total += float(pnl)
            except (TypeError, ValueError):
                continue
        return total

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

    def _nightly_marker_path(self, day_key: str) -> str:
        report_dir = os.path.join(os.path.dirname(self._journal.base_dir), "reports")
        return os.path.join(report_dir, f".nightly-{day_key}.done")

    def _nightly_already_ran(self, day_key: str) -> bool:
        if self._nightly_analysis_day == day_key:
            return True
        try:
            return os.path.exists(self._nightly_marker_path(day_key))
        except Exception:
            return False

    def _mark_nightly_ran(self, day_key: str) -> None:
        self._nightly_analysis_day = day_key
        try:
            marker = self._nightly_marker_path(day_key)
            os.makedirs(os.path.dirname(marker), exist_ok=True)
            with open(marker, "w") as f:
                f.write(datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            logger.debug("Nightly marker write failed: %s", exc)

    def _maybe_run_after_market_training(self, current_et: datetime, phase: str) -> bool:
        """Run nightly report and ML retrain once after the market session ends."""
        if not self._is_trading_day(current_et):
            return False
        day_key = current_et.date().isoformat()
        if self._nightly_already_ran(day_key):
            return False

        close_hour = 20 if getattr(self, "_after_hours_enabled", False) else 16
        cutoff = current_et.replace(hour=close_hour, minute=5, second=0, microsecond=0)
        if current_et < cutoff:
            return False

        exit_manager = getattr(getattr(self, "_pipeline", None), "exit_manager", None)
        open_positions = getattr(exit_manager, "tracked", {}) if exit_manager is not None else {}
        if open_positions:
            logger.info(
                "AFTER-MARKET TRAINING: waiting for %d tracked positions to close",
                len(open_positions),
            )
            return False

        logger.info(
            "AFTER-MARKET TRAINING: running nightly report + ML retrain for %s (%s)",
            day_key,
            phase,
        )
        try:
            self._hub.add_log("INFO", f"After-market ML training started for {day_key}")
        except Exception:
            pass
        self._run_nightly_analysis()
        self._mark_nightly_ran(day_key)
        return True

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

        # Nightly Warrior-ignition retrain (adaptive loop): retrain on cached +
        # paper-logged candidates, validate out-of-sample, and DEPLOY ONLY IF it
        # beats the current model. Wrapped so a failure never breaks the nightly.
        try:
            from daytrading.strategy.warrior_ignition_retrain import retrain
            res = retrain(deploy=True)
            if res.get("status") == "ok":
                msg = (
                    "IGNITION RETRAIN: {} candidates, OOS {:.0%}->{:.0%}, {}".format(
                        res["rows"], res["current_oos"], res["retrained_oos"],
                        "DEPLOYED new model" if res["deployed"] else "kept current (no improvement)",
                    )
                )
                logger.info(msg)
                self._hub.add_log("INFO", msg)
            else:
                logger.info("IGNITION RETRAIN: %s (rows=%s)", res.get("status"), res.get("rows"))
        except Exception as exc:
            logger.error("IGNITION RETRAIN failed: %s", exc)

        # Daily ML model retrain after market close
        self._retrain_ml_model()
        try:
            from daytrading.analyst.collector import NightlyAnalyst
            report_dir = os.path.join(os.path.dirname(self._journal.base_dir), "reports")
            analyst = NightlyAnalyst(db_path=self._journal.db_path, report_dir=report_dir)
            refreshed = analyst.run()
            if refreshed.get("status") not in ("no_trades", "holiday"):
                logger.info(
                    "NIGHTLY ANALYST: refreshed report after ML training for %s",
                    refreshed.get("day", "unknown"),
                )
        except Exception as exc:
            logger.debug("Nightly report refresh after ML training failed: %s", exc)

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
    configure_alpaca_stream_logging()

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
