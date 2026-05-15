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
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from threading import Event, Lock, Thread
from typing import Dict, List, Optional, Sequence

from daytrading.config import Settings
from daytrading.dashboard.hub import DashboardHub
from daytrading.dashboard.server import start_dashboard
from daytrading.data.alpaca_feed import AlpacaHistoricalFeed, AlpacaStreamFeed
from daytrading.data.float_checker import FloatChecker
from daytrading.data.news_checker import NewsChecker
from daytrading.data.realtime_scanner import RealtimeScanner
from daytrading.data.watchlist_scanner import WatchlistScanner
from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.exits.manager import ExitManager
from daytrading.exits.scaler import PositionScaler, ReentryDetector
from daytrading.journal.store import TradingJournal
from daytrading.pipeline.engine import PipelineResult, TradingPipeline
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.models import Bar, Order, PortfolioState, Position, Quote, Side, SignalAction, Tick

logger = logging.getLogger(__name__)


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

        self._bar_buffer: Dict[str, List[Bar]] = defaultdict(list)
        self._quote_buffer: Dict[str, List[Quote]] = defaultdict(list)
        self._tick_buffer: Dict[str, List[Tick]] = defaultdict(list)
        self._lock = Lock()
        self._shutdown = False
        self._max_bars_per_symbol = 200
        self._max_ticks_per_symbol = 200
        self._new_data = Event()
        self._hub = DashboardHub()
        self._journal = TradingJournal()
        self._hub.journal = self._journal
        self._hub._broker = self._broker
        self._hub._exit_manager = self._pipeline.exit_manager
        self._watchlist_data: List[dict] = []
        self._scanner: Optional[WatchlistScanner] = None
        self._rt_scanner: Optional[RealtimeScanner] = None
        self._use_realtime_scanner: bool = False
        self._pos_sync_thread: Optional[Thread] = None
        self._news_checker: Optional[NewsChecker] = None
        self._float_checker: Optional[FloatChecker] = None
        self._last_synced_order_ids: set = set()
        self._trade_analyzer = None
        self._analysis_interval = 10  # run analysis every N cycles

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
        vol_threshold = 5_000.0  # low-float momentum stocks: even 5K/bar is active
        high_liq = 500_000.0
        if cfg.alpaca_feed.lower() == "iex":
            vol_threshold = 500  # IEX 1-min bars average 1k-8k volume
            high_liq = 10_000.0  # scale down proportionally

        float_checker = FloatChecker(min_float=1_000_000)

        pipeline = create_scalping_pipeline(
            initial_cash=acct["cash"],
            commission_per_share=cfg.commission_per_share,
            min_price=cfg.min_price,
            max_price=cfg.max_price,
            max_positions=cfg.max_positions,
            max_position_shares=cfg.max_position_shares,
            max_order_shares=cfg.max_order_shares,
            min_avg_volume=vol_threshold,
            high_liquidity_volume=high_liq,
            portfolio=portfolio,
            float_checker=float_checker,
        )

        # replace the PaperBroker in the pipeline with the real AlpacaBroker
        pipeline._broker = broker  # type: ignore[assignment]
        # Wire slippage guard into broker for smart limit pricing
        broker._slippage_guard = pipeline.trade_guard.slippage

        scanner = WatchlistScanner(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            min_price=cfg.min_price,
            max_price=cfg.max_price,
            min_volume=50_000,
            min_change_pct=5.0,
            max_symbols=25,
            feed=cfg.alpaca_feed,
        )

        if watchlist is None:
            results = scanner.scan()
            watchlist = [r["symbol"] for r in results] if results else _fallback_watchlist()
            scan_data = results or []
        else:
            scan_data = []

        # Attach news sentiment checker (uses Alpaca News API — free)
        news_checker = NewsChecker(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            max_age_hours=24,
        )
        pipeline.set_news_checker(news_checker)
        logger.info("News sentiment checker active — will screen trades for bad news")

        # Pre-filter watchlist by float — only keep low-float stocks
        filtered_watchlist = []
        for sym in watchlist:
            float_shares = float_checker.get_float(sym)
            if float_shares is None:
                filtered_watchlist.append(sym)  # unknown float = keep it
            elif 1_000_000 <= float_shares <= 20_000_000:
                filtered_watchlist.append(sym)
            else:
                logger.info("WATCHLIST FILTER %s: float %.1fM — removed", sym, float_shares / 1_000_000)
        if not filtered_watchlist:
            logger.warning("No low-float stocks found — keeping full watchlist")
            filtered_watchlist = list(watchlist)
        watchlist = filtered_watchlist
        logger.info("Float-filtered watchlist (%d symbols): %s", len(watchlist), watchlist)

        logger.info("Final watchlist (%d symbols): %s", len(watchlist), watchlist)

        runner = cls(
            broker=broker,
            hist_feed=hist_feed,
            stream_feed=stream_feed,
            pipeline=pipeline,
            watchlist=watchlist,
            dashboard_port=int(os.environ.get("DAYTRADING_DASHBOARD_PORT", "8080")),
        )
        runner._watchlist_data = scan_data
        runner._scanner = scanner
        runner._use_realtime_scanner = cfg.alpaca_feed.lower() == "sip"
        runner._news_checker = news_checker
        runner._float_checker = float_checker

        return runner

    @staticmethod
    def _now_et() -> datetime:
        et_offset = timedelta(hours=-4)  # EDT (summer)
        return datetime.now(timezone.utc) + et_offset

    @classmethod
    def _is_market_open(cls) -> bool:
        """Check if US regular market is open (9:30 AM - 4:00 PM ET, Mon-Fri)."""
        try:
            now_et = cls._now_et()
            if now_et.weekday() >= 5:
                return False
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            return market_open <= now_et <= market_close
        except Exception:
            return True

    @classmethod
    def _is_premarket(cls) -> bool:
        """Check if we're in pre-market hours (4:00 AM - 9:30 AM ET, Mon-Fri)."""
        try:
            now_et = cls._now_et()
            if now_et.weekday() >= 5:
                return False
            premarket_start = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            return premarket_start <= now_et < market_open
        except Exception:
            return False

    @classmethod
    def _is_afterhours(cls) -> bool:
        """Check if we're in after-hours (4:00 PM - 8:00 PM ET, Mon-Fri)."""
        try:
            now_et = cls._now_et()
            if now_et.weekday() >= 5:
                return False
            ah_start = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            ah_end = now_et.replace(hour=20, minute=0, second=0, microsecond=0)
            return ah_start < now_et <= ah_end
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

    def _connect_stream(self) -> None:
        """Connect the WebSocket stream."""
        self._stream.on_bar(self._on_bar)
        self._stream.on_quote(self._on_quote)
        self._stream.on_trade(self._on_tick)
        self._stream.subscribe(self._watchlist, bars=True, quotes=True)
        # Subscribe to SPY bars for market panic detection
        self._stream.subscribe(["SPY"], bars=True, quotes=False)

        # With SIP: subscribe to ALL trades for real-time scanning
        if self._use_realtime_scanner:
            self._rt_scanner = RealtimeScanner(
                min_price=1.0,
                max_price=20.0,
                min_volume=10_000,
                min_change_pct=5.0,
                max_symbols=25,
                check_interval=5.0,
            )
            self._stream.on_trade(self._rt_scanner.on_trade)
            self._stream.subscribe_all_trades()
            self._rt_scanner.start(
                on_new_movers=self._on_new_movers_detected,
                initial_watchlist=self._watchlist,
            )
            logger.info("Real-time scanner active — watching EVERY trade on the market")

        self._stream.start(background=True)

    def run(self) -> None:
        """Main loop: load history -> stream -> run cycles until shutdown."""
        self._setup_signals()

        # Start dashboard web server
        start_dashboard(self._hub, port=self._dashboard_port)

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
        self._trade_analyzer = TradeAnalyzer(min_trades=5)

        if self._watchlist_data:
            self._hub.on_watchlist_scan(self._watchlist_data)

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

        # 1. Load historical bars
        self._load_history()

        # 2. Initial classification with historical data
        if self._bar_buffer:
            logger.info("Running initial analysis with historical data...")
            self._run_one_cycle(0)

        # 3. Setup streaming — connect during pre-market, market hours, or after-hours
        phase = self._market_phase()
        stream_connected = False
        if phase in ("OPEN", "PRE-MARKET", "AFTER-HOURS"):
            logger.info("Market phase: %s — connecting stream...", phase)
            self._connect_stream()
            stream_connected = True
        else:
            logger.info(
                "Market is currently CLOSED. "
                "Pre-market starts at 4:00 AM ET (10:00 AM your time). "
                "Will auto-connect when trading hours begin."
            )

        self._hub.on_market_status(phase != "CLOSED", stream_connected, phase)

        # 4. Start live scanner
        # SIP: real-time scanner already started in _connect_stream (uses trade stream)
        # IEX: fall back to snapshot polling every 30s
        if not self._use_realtime_scanner and self._scanner:
            self._scanner.start_live(
                on_new_movers=self._on_new_movers_detected,
                initial_watchlist=self._watchlist,
            )

        # 5. Main cycle loop
        # Runs on new bar data OR every poll_interval seconds (for pre-market/low-activity)
        poll_interval = 30.0  # run scanner every 30s even without new bars
        logger.info("Pipeline running — scanning every %.0fs (Ctrl+C to stop)", poll_interval)
        cycle_count = 0
        last_market_check = 0.0
        last_cycle_time = time.time()

        try:
            while not self._shutdown:
                now_ts = time.time()

                # --- Trading window: 4:30 AM - 3:30 PM ET ---
                # No new entries outside this window.
                # At 3:30 PM ET, flatten all positions (no overnight holds).
                now_et = self._now_et()
                trading_start = now_et.replace(hour=4, minute=30, second=0, microsecond=0)
                trading_end = now_et.replace(hour=15, minute=30, second=0, microsecond=0)
                in_trading_window = trading_start <= now_et <= trading_end and now_et.weekday() < 5

                if not in_trading_window and not getattr(self, '_eod_flattened', False):
                    if now_et > trading_end and now_et.weekday() < 5:
                        tracked = self._pipeline.exit_manager.tracked
                        if tracked:
                            logger.info("3:30 PM ET — FLATTENING %d positions (no overnight holds)", len(tracked))
                            self._hub.add_log("WARNING", "3:30 PM ET — closing all positions")
                            try:
                                self._broker.close_all_positions()
                                for sym in list(tracked.keys()):
                                    self._pipeline.exit_manager.untrack(sym)
                            except Exception as exc:
                                logger.error("Error flattening positions: %s", exc)
                        self._eod_flattened = True

                if not stream_connected and now_ts - last_market_check > 60:
                    last_market_check = now_ts
                    phase = self._market_phase()
                    if phase != "CLOSED":
                        logger.info("Market phase: %s — connecting WebSocket stream...", phase)
                        self._connect_stream()
                        stream_connected = True
                        self._hub.on_market_status(True, True, phase)

                got_data = self._new_data.wait(timeout=self._cycle_interval)
                if self._shutdown:
                    break

                # Always check exits every second for open positions
                # This prevents stop-loss slippage from 30s cycle gaps
                if self._pipeline.exit_manager.tracked:
                    self._check_exits_only()

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
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self._shutdown_gracefully()

    def _on_new_movers_detected(self, new_symbols: List[str], all_ranked: List[dict]) -> None:
        """Callback from the live scanner when new movers are found."""
        # Update dashboard
        self._watchlist_data = all_ranked
        self._hub.on_watchlist_scan(all_ranked)
        self._hub.on_rt_movers(new_symbols, all_ranked)

        if not new_symbols:
            return

        # Float-filter new movers before adding
        if self._float_checker:
            filtered = []
            for sym in new_symbols:
                float_shares = self._float_checker.get_float(sym)
                if float_shares is None:
                    filtered.append(sym)
                elif 1_000_000 <= float_shares <= 20_000_000:
                    filtered.append(sym)
                else:
                    logger.info("RT FLOAT FILTER %s: float %.1fM — skipping", sym, float_shares / 1_000_000)
            new_symbols = filtered
            if not new_symbols:
                return

        logger.info("LIVE SCANNER: %d new movers detected: %s", len(new_symbols), new_symbols)
        self._watchlist.extend(new_symbols)
        self._watchlist_set.update(new_symbols)

        # Load history for new symbols (today only)
        try:
            bars = self._hist.get_bars(new_symbols, timeframe="1Min", limit=100)
            today_et = self._now_et().date()
            with self._lock:
                for symbol, symbol_bars in bars.items():
                    today_bars = []
                    for b in symbol_bars:
                        if b.ts is not None:
                            try:
                                bar_et = (b.ts - timedelta(hours=4)).date()
                                if bar_et == today_et:
                                    today_bars.append(b)
                            except Exception:
                                today_bars.append(b)
                        else:
                            today_bars.append(b)
                    self._bar_buffer[symbol] = today_bars[-self._max_bars_per_symbol:]
        except Exception as exc:
            logger.error("Failed to load history for new symbols: %s", exc)

        # Subscribe to stream for new symbols
        try:
            self._stream.subscribe(new_symbols, bars=True, quotes=True)
        except Exception as exc:
            logger.error("Failed to subscribe new symbols: %s", exc)

        self._hub.add_log("INFO", "New movers added: {}".format(", ".join(new_symbols)))

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

                status_val = o.status.value if hasattr(o.status, 'value') else str(o.status)
                if status_val != "filled":
                    continue
                side_val = o.side.value if hasattr(o.side, 'value') else str(o.side)
                qty = float(o.filled_qty or 0)
                price = float(o.filled_avg_price or 0)
                if qty <= 0 or price <= 0:
                    continue

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

                    entry_price = matched_cost / matched_qty if matched_qty > 0 else price
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
        from daytrading.exits.manager import TrackedPosition, TICK
        from daytrading.models import Side
        try:
            alpaca_positions = self._broker.get_positions()
            portfolio = self._pipeline.portfolio
            exit_mgr = self._pipeline.exit_manager
            now = datetime.now(timezone.utc)
            for sym, data in alpaca_positions.items():
                qty = data["qty"]
                avg = data["avg_entry"]
                if qty > 0:
                    portfolio.positions[sym] = Position(
                        symbol=sym, quantity=qty, avg_price=avg,
                    )
                    if sym not in exit_mgr.tracked:
                        side = Side.BUY
                        stop = avg * 0.98  # 2% below entry
                        exit_mgr.track(TrackedPosition(
                            symbol=sym, side=side, quantity=qty,
                            remaining_qty=qty, entry_price=avg,
                            entry_ts=now, stop_loss=stop,
                            max_hold_seconds=600,
                            reason="synced from Alpaca",
                        ))
                    logger.info("Synced position from Alpaca: %s %.0f @ $%.2f (stop=$%.4f, -2%%)", sym, qty, avg, avg * 0.98)
            if alpaca_positions:
                logger.info("Synced %d existing positions from Alpaca", len(alpaca_positions))
            else:
                logger.info("No existing Alpaca positions — starting fresh")
        except Exception as exc:
            logger.warning("Could not sync Alpaca positions: %s", exc)

    def _push_positions_from_alpaca(self) -> None:
        """Fetch positions from Alpaca and push to dashboard with live prices.

        Always syncs quantity and avg_price from Alpaca — this is the
        source of truth, especially after partial exits (tiered selling).
        """
        try:
            alpaca_positions = self._broker.get_positions()
            prices = {}
            for sym, data in alpaca_positions.items():
                qty = data["qty"]
                avg = data["avg_entry"]
                cur = data.get("current_price", avg)
                prices[sym] = cur
                pos = self._pipeline.portfolio.positions.get(sym)
                if pos is None or pos.is_flat:
                    self._pipeline.portfolio.positions[sym] = Position(
                        symbol=sym, quantity=qty, avg_price=avg,
                    )
                else:
                    pos.quantity = qty
                    pos.avg_price = avg
            # Clear positions closed on Alpaca
            for sym in list(self._pipeline.portfolio.positions.keys()):
                if sym not in alpaca_positions:
                    pos = self._pipeline.portfolio.positions[sym]
                    if not pos.is_flat:
                        pos.quantity = 0.0
                        # Also untrack from exit manager
                        self._pipeline.exit_manager.untrack(sym)
            self._hub.on_position_update(self._pipeline.portfolio.positions, prices)
        except Exception as exc:
            logger.debug("Position sync error: %s", exc)

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
                    self._hub.on_exit_fill(fill, entry_price=entry_price)
                    self._hub.add_log("INFO", "EXIT {} {} {:.0f} @ ${:.2f}".format(
                        "SELL", o.symbol, qty, price))

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
                    self._hub.account_cash = acct["cash"]
                    self._hub.account_equity = acct["equity"]
                    self._hub.account_buying_power = acct["buying_power"]
                except Exception:
                    pass

    def _load_history(self) -> None:
        """Fetch recent bars to seed the pipeline.

        Only keeps bars from **today's trading session** (based on ET date).
        This prevents yesterday's bars from inflating volume/momentum
        calculations and causing the bot to enter dead stocks.
        """
        logger.info("Loading historical bars for %d symbols...", len(self._watchlist))
        try:
            # Fetch today's bars specifically to avoid getting only old data
            today_open_et = self._now_et().replace(hour=9, minute=30, second=0, microsecond=0)
            today_start_utc = today_open_et + timedelta(hours=4)  # ET→UTC
            bars = self._hist.get_bars(
                self._watchlist, timeframe="1Min", limit=100,
                start=today_start_utc,
            )
            today_et = self._now_et().date()
            with self._lock:
                for symbol, symbol_bars in bars.items():
                    today_bars = []
                    older_bars = []
                    for b in symbol_bars:
                        if b.ts is not None:
                            try:
                                bar_et = (b.ts - timedelta(hours=4)).date()
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
                    # Use today's bars, but if too few (<5), keep the
                    # most recent older bars as seed for classification.
                    # The session boundary flush in _on_bar() will clear
                    # these once fresh stream bars arrive.
                    if len(today_bars) >= 5:
                        self._bar_buffer[symbol] = today_bars[-self._max_bars_per_symbol:]
                    else:
                        seed = older_bars[-20:] + today_bars
                        self._bar_buffer[symbol] = seed[-self._max_bars_per_symbol:]
            loaded = {s: len(b) for s, b in self._bar_buffer.items() if b}
            logger.info("Loaded history: %s", loaded)
        except Exception as exc:
            logger.error("Failed to load history: %s", exc)

    def _on_bar(self, bar: Bar) -> None:
        """Callback from stream — buffer the bar and signal new data."""
        # Feed SPY bars to market panic detector
        if bar.symbol == "SPY":
            self._pipeline.trade_guard.market_panic.update_spy_bar(bar)
            return

        with self._lock:
            buf = self._bar_buffer[bar.symbol]

            # Also reject bars that are too far in the past (> 4 hours from now)
            if bar.ts is not None:
                try:
                    age = (datetime.now(timezone.utc) - bar.ts).total_seconds()
                    if age > 14400:  # 4 hours
                        return
                except Exception:
                    pass

            buf.append(bar)
            if len(buf) > self._max_bars_per_symbol:
                self._bar_buffer[bar.symbol] = buf[-self._max_bars_per_symbol:]
        self._new_data.set()

    def _on_quote(self, quote: Quote) -> None:
        """Callback from stream — buffer the quote and update slippage guard."""
        with self._lock:
            buf = self._quote_buffer[quote.symbol]
            buf.append(quote)
            if len(buf) > 50:
                self._quote_buffer[quote.symbol] = buf[-50:]
        # Feed live quotes to slippage guard for smart limit pricing
        self._pipeline.trade_guard.slippage.update_quote(quote)

    def _on_tick(self, tick: Tick) -> None:
        """Callback from stream — buffer ticks for watchlist symbols only."""
        if tick.symbol not in self._watchlist_set:
            return
        # Feed watchlist ticks to halt tracker for freeze/gap detection
        self._pipeline.trade_guard.halt_tracker.update_price(
            tick.symbol, tick.price, tick.ts,
        )
        with self._lock:
            buf = self._tick_buffer[tick.symbol]
            buf.append(tick)
            if len(buf) > self._max_ticks_per_symbol:
                self._tick_buffer[tick.symbol] = buf[-self._max_ticks_per_symbol:]

    def _check_exits_only(self) -> None:
        """Fast exit check every second using live Alpaca prices.

        This prevents stop-loss slippage that occurs when the full
        30-second scan cycle hasn't run yet but the price has moved.
        """
        try:
            tracked = self._pipeline.exit_manager.tracked
            if not tracked:
                return

            alpaca_pos = self._broker.get_positions()
            prices = {}
            for sym in tracked:
                data = alpaca_pos.get(sym)
                if data:
                    prices[sym] = float(data.get("current_price", data["avg_entry"]))

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
                if fill:
                    from daytrading.execution.broker import apply_fill as _apply_fill
                    _apply_fill(self._pipeline.portfolio, fill)
                    ep = entry_prices.get(sig.symbol, 0.0)
                    if ep == 0.0:
                        pos = self._pipeline.portfolio.positions.get(sig.symbol)
                        if pos:
                            ep = pos.avg_price
                    logger.info(
                        "FAST EXIT %s %s %.0f @ %.4f (entry=%.4f, P&L=$%.2f)",
                        fill.side.value, fill.symbol, fill.quantity, fill.price,
                        ep, (fill.price - ep) * fill.quantity if ep > 0 else 0,
                    )
                    self._hub.on_exit_fill(fill, entry_price=ep)
                    self._hub.add_log("INFO", "EXIT {} {} {:.0f} @ ${:.2f}".format(
                        fill.side.value, fill.symbol, fill.quantity, fill.price))
                    # Seed recent order IDs so _check_new_fills won't duplicate
                    self._seed_recent_order_ids()
                    self._push_positions_from_alpaca()
                    self._pipeline.set_cooldown(fill.symbol)
                else:
                    logger.warning("FAST EXIT order not filled for %s (status=%s)", sig.symbol, status)
                    # Rollback half-sell state if the order failed
                    tracked_pos = self._pipeline.exit_manager._positions.get(sig.symbol)
                    if tracked_pos and tracked_pos.sold_half and tracked_pos.remaining_qty > 0:
                        tracked_pos.sold_half = False
                        tracked_pos.remaining_qty += sig.quantity
                        tracked_pos.stop_loss = tracked_pos.entry_price - tracked_pos.risk_per_share
                        tracked_pos.breakeven_locked = False
                        logger.info(
                            "ROLLBACK half-sell %s: restored qty=%d, stop=%.4f",
                            sig.symbol, int(tracked_pos.remaining_qty), tracked_pos.stop_loss,
                        )
        except Exception as exc:
            logger.warning("Fast exit check error: %s", exc)

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

    def _run_one_cycle(self, cycle_num: int) -> None:
        """Execute one pipeline cycle with current buffered data.

        Also injects live Alpaca prices for open positions that have
        no bars in the buffer, so the exit manager can always check
        stops/targets/trails — even for dead stocks with no new bars.
        """
        with self._lock:
            universe = {s: list(b) for s, b in self._bar_buffer.items() if b}
            quotes = {s: list(q) for s, q in self._quote_buffer.items() if q}
            ticks = {s: list(t) for s, t in self._tick_buffer.items() if t}

        # Inject live prices for open positions.
        # Without this, the exit manager and max-loss checks use stale bar data.
        open_syms = {
            sym for sym, pos in self._pipeline.portfolio.positions.items()
            if not pos.is_flat
        }
        if open_syms:
            try:
                alpaca_pos = self._broker.get_positions()
                now_ts = datetime.now(timezone.utc)
                for sym in open_syms:
                    data = alpaca_pos.get(sym)
                    if not data:
                        continue
                    price = float(data.get("current_price", data["avg_entry"]))
                    live_bar = Bar(
                        symbol=sym, open=price, high=price,
                        low=price, close=price, volume=0,
                        ts=now_ts,
                    )
                    if sym in universe:
                        universe[sym].append(live_bar)
                    else:
                        universe[sym] = [live_bar]
            except Exception:
                pass

        if not universe:
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
                return
            result = self._pipeline.run_cycle(universe, now=now, quotes=quotes, ticks=ticks)
            result._entry_prices = entry_prices
            self._process_result(result, cycle_num, universe)

            if cycle_num > 0 and cycle_num % self._analysis_interval == 0:
                self._run_trade_analysis()
        except Exception as exc:
            logger.error("Cycle %d error: %s", cycle_num, exc, exc_info=True)
            self._hub.add_log("ERROR", "Cycle {} error: {}".format(cycle_num, exc))

    def _process_result(self, result: PipelineResult, cycle_num: int, universe: Dict[str, List[Bar]]) -> None:
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
            self._journal.record("trade_fill", {
                "symbol": f.symbol,
                "side": f.side.value,
                "quantity": f.quantity,
                "price": f.price,
                "ts": f.ts,
                "trade_type": "entry",
                "market_context": {
                    "phase": self._market_phase(),
                    "cycle": cycle_num,
                },
            }, ts=f.ts)

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
            self._hub.on_exit_fill(f, entry_price=entry_price)
            self._hub.add_log("INFO", "EXIT {} {} {:.0f} @ ${:.2f}".format(
                f.side.value.upper(), f.symbol, f.quantity, f.price))
            pnl = (f.price - entry_price) * f.quantity if entry_price > 0 else 0.0
            if f.side.value == "buy":
                pnl = -pnl
            self._journal.record("trade_exit", {
                "symbol": f.symbol,
                "side": f.side.value,
                "quantity": f.quantity,
                "entry_price": entry_price,
                "exit_price": f.price,
                "pnl": pnl,
                "ts": f.ts,
                "trade_type": "exit",
                "market_context": {
                    "phase": self._market_phase(),
                    "cycle": cycle_num,
                },
            }, ts=f.ts)

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

        # Push rejected count
        for _ in range(result.rejected_orders):
            self._hub.on_rejected()
            self._journal.record("mistake", {
                "kind": "risk_rejection",
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
        skipped = len(result.regimes) - routed
        phase = self._market_phase()
        summary = "[Cycle {}] [{}] {} symbols scanned, {} scan hits, {} signals, {} fills".format(
            cycle_num, phase, routed, len(result.scan_hits), len(result.signals), len(result.fills),
        )
        self._hub.add_log("INFO", summary)
        logger.info(summary)

    def _shutdown_gracefully(self) -> None:
        """Clean shutdown: stop scanner, stream, optionally close positions."""
        logger.info("Shutting down...")

        if self._rt_scanner:
            self._rt_scanner.stop()
        if self._scanner:
            self._scanner.stop_live()

        self._stream.stop()

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
        """Handle Ctrl+C gracefully."""
        def handler(sig: int, frame: object) -> None:
            self._shutdown = True
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


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
