"""Dashboard data hub — thread-safe store for pipeline events.

The pipeline pushes data here; the web server reads from here.
Uses a simple pub/sub pattern with SSE (Server-Sent Events).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Deque, Dict, List, Optional


@dataclass
class TradeRecord:
    """A completed or active trade."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: str
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    trade_type: str = "entry"  # entry, exit, scale_up, reentry
    strategy: str = ""


@dataclass
class ScannerHit:
    """A scanner detection event."""
    symbol: str
    scanner_name: str
    score: float
    time: str
    price: float = 0.0
    criteria: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    action_taken: Optional[str] = None


@dataclass
class SymbolStatus:
    """Current classification status of a symbol."""
    symbol: str
    style: str
    confidence: float
    price: float
    volatility_pct: float
    spread_pct: float
    relative_volume: float
    trend_strength: float
    liquidity_score: float
    reasons: List[str] = field(default_factory=list)


class DashboardHub:
    """Central data hub for the dashboard — collects all pipeline events."""

    def __init__(self, max_history: int = 500) -> None:
        self._lock = Lock()
        self._max = max_history

        # Account
        self.account_cash: float = 0.0
        self.account_equity: float = 0.0
        self.account_buying_power: float = 0.0
        self.starting_cash: float = 0.0

        # Counters
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_pnl: float = 0.0
        self.total_scan_hits: int = 0
        self._seen_scan_keys: set = set()
        self.total_signals: int = 0
        self.total_rejected: int = 0
        self.cycle_count: int = 0

        # Current state
        self.positions: Dict[str, dict] = {}
        self.symbol_status: Dict[str, SymbolStatus] = {}
        self.market_open: bool = False
        self.market_phase: str = "CLOSED"
        self.stream_connected: bool = False
        self.bot_start_time: Optional[str] = None

        # Watchlist scan results
        self.watchlist_scan: List[dict] = []
        # Real-time movers that met criteria
        self.rt_movers: List[dict] = []
        # HOD Momentum alert feed (one row per alert type)
        self.hod_momentum_alerts: List[dict] = []
        # Early fast-scan movers being watched for structured pullback entries
        self.hot_watch: List[dict] = []
        self.missed_a_plus: List[dict] = []
        self.scanner_near_miss: Dict[str, Any] = {}
        # Active trading watchlist (HOD TTL + pinned + open positions)
        self.trading_watchlist: List[str] = []
        self.watchlist_pinned: List[str] = []
        self.candidate_hydration: Dict[str, Any] = {
            "queued": 0,
            "hydrated": 0,
            "skipped_fresh": 0,
            "dropped": 0,
            "batches": 0,
            "pending": 0,
            "paused_for_entry": False,
            "last_batch_size": 0,
            "last_loaded": 0,
            "last_source": "",
            "last_update": "",
        }

        # History (ring buffers)
        self.trades: Deque[TradeRecord] = deque(maxlen=max_history)
        self.scanner_hits: Deque[ScannerHit] = deque(maxlen=max_history)
        self.pnl_history: Deque[dict] = deque(maxlen=max_history)
        self.log_messages: Deque[dict] = deque(maxlen=200)

        # News sentiment cache per symbol
        self.news_data: Dict[str, dict] = {}

        # AI Trade Analyzer results
        self.ai_analysis: dict = {}
        # Optional persistent journal (injected by runner)
        self.journal: Optional[Any] = None

        # Trading control (pause/resume from dashboard)
        self.trading_paused: bool = False

        # SSE subscribers
        self._sse_queues: List[deque] = []

    def _broadcast(self, event_type: str, data: dict) -> None:
        """Push an SSE event to all subscribers."""
        msg = {"type": event_type, "data": data, "ts": _now_str()}
        dead = []
        for q in self._sse_queues:
            try:
                q.append(msg)
                if len(q) > 100:
                    q.popleft()
            except Exception:
                dead.append(q)
        for q in dead:
            self._sse_queues.remove(q)

    def subscribe(self) -> deque:
        """Create a new SSE subscriber queue."""
        q: deque = deque(maxlen=200)
        with self._lock:
            self._sse_queues.append(q)
        return q

    def unsubscribe(self, q: deque) -> None:
        with self._lock:
            if q in self._sse_queues:
                self._sse_queues.remove(q)

    # ------------------------------------------------------------------
    # Pipeline event handlers
    # ------------------------------------------------------------------

    def on_watchlist_scan(self, results: list) -> None:
        with self._lock:
            self.watchlist_scan = results
        self._broadcast("watchlist_scan", {
            "stocks": results,
            "scan_time": _now_str(),
        })

    def on_hod_momentum_alerts(self, alerts: List[dict]) -> None:
        """Replace HOD Momentum scanner feed."""
        with self._lock:
            self.hod_momentum_alerts = alerts[:200]
        self._broadcast("hod_momentum_alerts", {"alerts": self.hod_momentum_alerts})

    def on_hot_watch(self, symbols: List[dict]) -> None:
        """Replace active hot-watch candidate list."""
        with self._lock:
            self.hot_watch = symbols[:100]
            merged = list(self.trading_watchlist)
            seen = set(merged)
            for item in self.hot_watch:
                sym = str(item.get("symbol", "")).strip().upper()
                if sym and sym not in seen:
                    merged.append(sym)
                    seen.add(sym)
        self._broadcast("hot_watch", {"symbols": self.hot_watch})
        self._broadcast("trading_watchlist", {
            "symbols": merged,
            "pinned": self.watchlist_pinned,
        })

    def on_trading_watchlist(
        self,
        symbols: List[str],
        *,
        pinned: Optional[List[str]] = None,
    ) -> None:
        """Symbols the bot is actively scanning for trades."""
        with self._lock:
            self.trading_watchlist = list(symbols)
            if pinned is not None:
                self.watchlist_pinned = list(pinned)
        self._broadcast("trading_watchlist", {
            "symbols": self.trading_watchlist,
            "pinned": self.watchlist_pinned,
        })

    def on_candidate_hydration(
        self,
        *,
        queued: int = 0,
        hydrated: int = 0,
        skipped_fresh: int = 0,
        dropped: int = 0,
        batches: int = 0,
        pending: Optional[int] = None,
        paused_for_entry: Optional[bool] = None,
        last_batch_size: Optional[int] = None,
        last_loaded: Optional[int] = None,
        last_source: Optional[str] = None,
    ) -> None:
        with self._lock:
            stats = self.candidate_hydration
            stats["queued"] = int(stats.get("queued", 0)) + queued
            stats["hydrated"] = int(stats.get("hydrated", 0)) + hydrated
            stats["skipped_fresh"] = int(stats.get("skipped_fresh", 0)) + skipped_fresh
            stats["dropped"] = int(stats.get("dropped", 0)) + dropped
            stats["batches"] = int(stats.get("batches", 0)) + batches
            if pending is not None:
                stats["pending"] = pending
            if paused_for_entry is not None:
                stats["paused_for_entry"] = paused_for_entry
            if last_batch_size is not None:
                stats["last_batch_size"] = last_batch_size
            if last_loaded is not None:
                stats["last_loaded"] = last_loaded
            if last_source is not None:
                stats["last_source"] = last_source
            stats["last_update"] = _now_str()
            data = dict(stats)
        self._broadcast("candidate_hydration", data)

    def on_missed_a_plus(self, rows: List[dict]) -> None:
        """Replace missed A+ setup report rows."""
        with self._lock:
            self.missed_a_plus = rows[:100]
        self._broadcast("missed_a_plus", {"rows": self.missed_a_plus})

    def on_scanner_near_miss(self, summary: Dict[str, Any]) -> None:
        """Replace the scanner-near-miss report summary (report-only)."""
        with self._lock:
            self.scanner_near_miss = dict(summary or {})
        self._broadcast("scanner_near_miss", {"summary": self.scanner_near_miss})

    def on_rt_movers(self, new_symbols: list, all_ranked: list) -> None:
        """Deprecated — RT mover scanner removed (HOD-only mode). No-op."""
        return

    def on_startup(self, cash: float, equity: float, buying_power: float) -> None:
        with self._lock:
            self.account_cash = cash
            self.account_equity = equity
            self.account_buying_power = buying_power
            self.starting_cash = cash
            self.bot_start_time = _now_str()
        self.on_account_update(cash, equity, buying_power)

    def on_account_update(
        self, cash: float, equity: float, buying_power: float,
    ) -> None:
        """Push account balances to dashboard (no HTTP poll)."""
        with self._lock:
            self.account_cash = cash
            self.account_equity = equity
            self.account_buying_power = buying_power
        self._broadcast("account", {
            "cash": round(cash, 2),
            "equity": round(equity, 2),
            "buying_power": round(buying_power, 2),
        })

    def on_market_status(self, is_open: bool, stream_connected: bool, phase: str = "") -> None:
        with self._lock:
            self.market_open = is_open
            self.stream_connected = stream_connected
            if phase:
                self.market_phase = phase
        self._broadcast("market_status", {
            "market_open": is_open,
            "stream_connected": stream_connected,
            "market_phase": phase or self.market_phase,
        })

    def on_classification(self, symbol: str, regime: Any) -> None:
        status = SymbolStatus(
            symbol=symbol,
            style=regime.style.value,
            confidence=regime.confidence,
            price=0.0,
            volatility_pct=regime.volatility_pct,
            spread_pct=regime.spread_pct,
            relative_volume=regime.relative_volume,
            trend_strength=regime.trend_strength,
            liquidity_score=regime.liquidity_score,
            reasons=list(regime.reasons),
        )
        with self._lock:
            self.symbol_status[symbol] = status
            # Remove from live movers if classified as not tradeable
            if regime.style.value == "not_tradeable":
                self.rt_movers = [m for m in self.rt_movers if m["symbol"] != symbol]
        self._broadcast("classification", _status_dict(status))

    def on_scan_hit(self, hit: Any, verified: bool = False, reject_reason: Optional[str] = None) -> None:
        price = 0.0
        bars = getattr(hit, "bars", None)
        if bars and len(bars) > 0:
            price = float(bars[-1].close)

        key = (hit.symbol, hit.scanner_name)
        rec = ScannerHit(
            symbol=hit.symbol,
            scanner_name=hit.scanner_name,
            score=hit.score,
            time=_now_str(),
            price=price,
            criteria={k: _safe_val(v) for k, v in hit.criteria.items()},
            verified=verified,
            action_taken=reject_reason if reject_reason else ("SIGNAL" if verified else None),
        )
        with self._lock:
            # Replace existing entry for the same symbol+scanner
            self.scanner_hits = deque(
                (s for s in self.scanner_hits
                 if not (s.symbol == hit.symbol and s.scanner_name == hit.scanner_name)),
                maxlen=self.scanner_hits.maxlen,
            )
            self.scanner_hits.append(rec)
            if key not in self._seen_scan_keys:
                self._seen_scan_keys.add(key)
                self.total_scan_hits += 1
        self._broadcast("scan_hit", _scanner_dict(rec))

    def on_signal(self, signal: Any) -> None:
        with self._lock:
            self.total_signals += 1

    def on_fill(self, fill: Any, trade_type: str = "entry", strategy: str = "") -> None:
        rec = TradeRecord(
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=fill.quantity,
            entry_price=fill.price,
            entry_time=str(fill.ts),
            trade_type=trade_type,
            strategy=strategy,
        )
        with self._lock:
            self.trades.append(rec)
            if trade_type == "entry":
                self.total_trades += 1
        self._broadcast("trade", _trade_dict(rec))

    def on_exit_fill(self, fill: Any, entry_price: float = 0.0, reason: str = "",
                     skip_pnl_accum: bool = False) -> None:
        pnl: Optional[float] = None
        if entry_price > 0:
            pnl = (fill.price - entry_price) * fill.quantity
            if fill.side.value == "buy":
                pnl = -pnl

        rec = TradeRecord(
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=fill.quantity,
            entry_price=entry_price,
            entry_time="",
            exit_price=fill.price,
            exit_time=str(fill.ts),
            pnl=pnl,
            exit_reason=reason,
            trade_type="exit",
        )
        with self._lock:
            self.trades.append(rec)
            if pnl is not None and not skip_pnl_accum:
                self.total_pnl += pnl
            if pnl is not None:
                if pnl >= 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            self.pnl_history.append({"ts": _now_str(), "pnl": self.total_pnl})
        self._broadcast("exit", _trade_dict(rec))

    def on_position_update(self, positions: Dict[str, Any], prices: Dict[str, float]) -> None:
        pos_data = {}
        for sym, pos in positions.items():
            if pos.is_flat:
                continue
            price = prices.get(sym, pos.avg_price)
            upnl = pos.unrealized_pnl(price)
            pos_data[sym] = {
                "symbol": sym,
                "quantity": pos.quantity,
                "avg_price": round(pos.avg_price, 4),
                "current_price": round(price, 4),
                "unrealized_pnl": round(upnl, 2),
                "market_value": round(pos.market_value(price), 2),
                "side": "LONG" if pos.quantity > 0 else "SHORT",
            }
        with self._lock:
            self.positions = pos_data
        self._broadcast("positions", pos_data)

    def on_cycle_complete(self, cycle_num: int, result: Any) -> None:
        with self._lock:
            self.cycle_count = max(self.cycle_count, cycle_num)
            visible_cycle = self.cycle_count
        summary = {
            "cycle": visible_cycle,
            "scan_hits": len(result.scan_hits),
            "signals": len(result.signals),
            "fills": len(result.fills),
            "exits": len(result.exit_fills),
            "rejected": result.rejected_orders,
        }
        self._broadcast("cycle", summary)

    def on_cycle_heartbeat(self, cycle_num: int, reason: str = "") -> None:
        """Mark loop progress when a full pipeline result is not available."""
        log_entry = None
        with self._lock:
            self.cycle_count = max(self.cycle_count, cycle_num)
            visible_cycle = self.cycle_count
            if reason:
                log_entry = {
                    "level": "INFO",
                    "message": "Cycle {} heartbeat: {}".format(visible_cycle, reason),
                    "ts": _now_str(),
                }
                self.log_messages.append(log_entry)
        self._broadcast("cycle", {
            "cycle": visible_cycle,
            "scan_hits": 0,
            "signals": 0,
            "fills": 0,
            "exits": 0,
            "rejected": 0,
            "reason": reason,
        })
        if log_entry is not None:
            self._broadcast("log", log_entry)

    def on_news(self, symbol: str, score: float, headlines: list) -> None:
        """Store and broadcast news sentiment for a symbol."""
        data = {
            "symbol": symbol,
            "score": round(score, 2),
            "headlines": headlines[:5],
            "sentiment": "positive" if score >= 0.3 else ("negative" if score <= -0.3 else "neutral"),
            "ts": _now_str(),
        }
        with self._lock:
            self.news_data[symbol] = data
        self._broadcast("news", data)

    def on_rejected(self) -> None:
        with self._lock:
            self.total_rejected += 1

    def reset_daily_overview(self) -> None:
        """Clear dashboard session counters for a fresh trading day."""
        with self._lock:
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            self.total_pnl = 0.0
            self.total_scan_hits = 0
            self._seen_scan_keys.clear()
            self.total_signals = 0
            self.total_rejected = 0
            self.cycle_count = 0
            self.trades.clear()
            self.scanner_hits.clear()
            self.pnl_history.clear()
            self.symbol_status.clear()
            self.missed_a_plus.clear()
            self.ai_analysis = {}
            self.candidate_hydration = {
                "queued": 0,
                "hydrated": 0,
                "skipped_fresh": 0,
                "dropped": 0,
                "batches": 0,
                "pending": 0,
                "paused_for_entry": False,
                "last_batch_size": 0,
                "last_loaded": 0,
                "last_source": "",
                "last_update": "",
            }
        self._broadcast("daily_reset", {"ts": _now_str()})

    def add_log(self, level: str, message: str) -> None:
        entry = {"level": level, "message": message, "ts": _now_str()}
        with self._lock:
            self.log_messages.append(entry)
        self._broadcast("log", entry)

    # ------------------------------------------------------------------
    # Snapshot for initial page load
    # ------------------------------------------------------------------

    _ROLLING_CACHE_TTL_SEC = 45.0

    def _cached_rolling_scorecard(self) -> dict:
        """Rolling journal scorecard, cached ~45s.

        It scans the (large) trades / market_context tables by ts, which is far
        too heavy to recompute on every dashboard snapshot. Computed OUTSIDE the
        hub lock (the journal DB has its own synchronization) so a cold/expired
        scan can't stall every other hub mutation and SSE broadcast; a rare
        concurrent double-scan is harmless.
        """
        now = time.monotonic()
        cached = getattr(self, "_rolling_cache", None)
        cached_at = getattr(self, "_rolling_cache_at", 0.0)
        if cached is not None and (now - cached_at) < self._ROLLING_CACHE_TTL_SEC:
            return cached
        result = _rolling_journal_scorecard(self.journal)
        self._rolling_cache = result
        self._rolling_cache_at = now
        return result

    def snapshot(self) -> dict:
        """Full current state for initial page load."""
        # Heavy journal scan first, before taking the hub lock (see method docs).
        rolling_scorecard = self._cached_rolling_scorecard()
        with self._lock:
            win_rate = 0.0
            total_closed = self.winning_trades + self.losing_trades
            if total_closed > 0:
                win_rate = self.winning_trades / total_closed * 100
            trade_rows = [_trade_dict(t) for t in list(self.trades)]
            daily_scorecard = _daily_scorecard(
                trades=trade_rows,
                total_trades=self.total_trades,
                total_scan_hits=self.total_scan_hits,
                total_signals=self.total_signals,
                total_rejected=self.total_rejected,
                cycle_count=self.cycle_count,
                missed_a_plus=list(self.missed_a_plus),
            )

            return {
                "account": {
                    "cash": round(self.account_cash, 2),
                    "equity": round(self.account_equity, 2),
                    "buying_power": round(self.account_buying_power, 2),
                    "starting_cash": round(self.starting_cash, 2),
                },
                "stats": {
                    "total_trades": self.total_trades,
                    "winning_trades": self.winning_trades,
                    "losing_trades": self.losing_trades,
                    "win_rate": round(win_rate, 1),
                    "total_pnl": round(self.total_pnl, 2),
                    "total_scan_hits": self.total_scan_hits,
                    "total_signals": self.total_signals,
                    "total_rejected": self.total_rejected,
                    "cycle_count": self.cycle_count,
                },
                "positions": dict(self.positions),
                "symbols": {
                    sym: _status_dict(s)
                    for sym, s in self.symbol_status.items()
                },
                "recent_trades": trade_rows[-50:],
                "recent_scans": [_scanner_dict(s) for s in list(self.scanner_hits)[-50:]],
                "logs": list(self.log_messages),
                "pnl_history": list(self.pnl_history),
                "daily_scorecard": daily_scorecard,
                "rolling_scorecard": rolling_scorecard,
                "market_open": self.market_open,
                "market_phase": self.market_phase,
                "stream_connected": self.stream_connected,
                "bot_start_time": self.bot_start_time,
                "watchlist_scan": self.watchlist_scan,
                "rt_movers": self.rt_movers,
                "hod_momentum_alerts": list(self.hod_momentum_alerts),
                "hot_watch": list(self.hot_watch),
                "missed_a_plus": list(self.missed_a_plus),
                "scanner_near_miss": dict(self.scanner_near_miss),
                "trading_watchlist": list(self.trading_watchlist),
                "watchlist_pinned": list(self.watchlist_pinned),
                "candidate_hydration": dict(self.candidate_hydration),
                "news": dict(self.news_data),
                "ai_analysis": dict(self.ai_analysis),
                "trading_paused": self.trading_paused,
            }


def _now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_val(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, datetime):
        return str(v)
    return v


def _trade_dict(t: TradeRecord) -> dict:
    return {
        "symbol": t.symbol,
        "side": t.side,
        "quantity": t.quantity,
        "entry_price": round(t.entry_price, 4),
        "entry_time": t.entry_time,
        "exit_price": round(t.exit_price, 4) if t.exit_price else None,
        "exit_time": t.exit_time,
        "pnl": round(t.pnl, 2) if t.pnl is not None else None,
        "exit_reason": t.exit_reason,
        "trade_type": t.trade_type,
        "strategy": t.strategy,
    }


def _round_trip_pnls(trades: List[dict]) -> List[float]:
    """Group partial exits into round-trips.

    A position that exits in halves/partials produces multiple exit rows; those
    must NOT each count as a separate closed trade (it inflates the count and can
    push closed-rate over 100% and trip the go-live gate too early). Walk the
    trades in time order: an entry opens/rolls a round-trip per symbol, exits
    accumulate their P&L into it, and a completed round-trip contributes one
    realized-P&L value.
    """
    def _ts(t: dict) -> str:
        return str(t.get("ts") or t.get("exit_time") or t.get("entry_time") or "")

    open_trip: dict = {}
    trips: List[tuple] = []
    for t in sorted(trades, key=_ts):
        sym = t.get("symbol")
        ttype = t.get("trade_type")
        if ttype in ("entry", "reentry"):
            prev = open_trip.get(sym)
            if prev is not None and prev["has_exit"]:
                trips.append((prev["pnl"], prev["strategy"]))
            open_trip[sym] = {
                "pnl": 0.0, "has_exit": False, "strategy": str(t.get("strategy") or ""),
            }
        elif ttype == "exit" and t.get("pnl") is not None:
            cur = open_trip.get(sym)
            if cur is None:
                cur = {"pnl": 0.0, "has_exit": False, "strategy": str(t.get("strategy") or "")}
                open_trip[sym] = cur
            cur["pnl"] += float(t.get("pnl") or 0.0)
            cur["has_exit"] = True
    for cur in open_trip.values():
        if cur["has_exit"]:
            trips.append((cur["pnl"], cur["strategy"]))
    return trips


def _daily_scorecard(
    *,
    trades: List[dict],
    total_trades: int,
    total_scan_hits: int,
    total_signals: int,
    total_rejected: int,
    cycle_count: int,
    missed_a_plus: List[dict],
) -> dict:
    # Count round-trips, not raw exit rows — a position that scales out in
    # partials emits several exit rows but is ONE closed trade.
    trips = _round_trip_pnls(trades)
    trip_pnls = [p for p, _ in trips]
    # A scratch (pnl == 0) is neither a win nor a loss; counting breakevens as
    # wins inflated win_rate while contributing nothing to total_win.
    wins = [p for p in trip_pnls if p > 0.0]
    losses = [p for p in trip_pnls if p < 0.0]

    # Isolate the experimental momentum-breakout mode so its standalone
    # expectancy can be judged separately from normal entries.
    by_entry_mode: Dict[str, dict] = {}
    by_strategy: Dict[str, dict] = {}
    for pnl, strategy in trips:
        strategy_name = strategy or "unknown"
        _s = strategy.lower()
        if "momentum" in _s:
            mode = "momentum_breakout"
        elif "fresh_vwap_reclaim" in _s:
            mode = "fresh_vwap_reclaim_scout"
        elif "vwap_reclaim_scout" in _s:
            mode = "vwap_reclaim_scout"
        elif "level_breakout_scout" in _s:
            mode = "level_breakout_scout"
        elif "level_capped_scout" in _s:
            mode = "level_capped_scout"
        elif "ten_second_breakout_scout" in _s:
            mode = "ten_second_breakout_scout"
        elif "elite_wide_spread" in _s:
            mode = "elite_wide_spread"
        else:
            mode = "standard"
        bucket = by_entry_mode.setdefault(
            mode, {"closed_trades": 0, "wins": 0, "total_pnl": 0.0}
        )
        bucket["closed_trades"] += 1
        bucket["wins"] += 1 if pnl > 0.0 else 0
        bucket["total_pnl"] = round(bucket["total_pnl"] + pnl, 2)
        strategy_bucket = by_strategy.setdefault(
            strategy_name, {"closed_trades": 0, "wins": 0, "total_pnl": 0.0}
        )
        strategy_bucket["closed_trades"] += 1
        strategy_bucket["wins"] += 1 if pnl > 0.0 else 0
        strategy_bucket["total_pnl"] = round(strategy_bucket["total_pnl"] + pnl, 2)

    total_win = sum(wins)
    total_loss = abs(sum(losses))
    total_pnl = total_win - total_loss
    closed_trades = len(trip_pnls)
    win_rate = (len(wins) / closed_trades * 100.0) if closed_trades else 0.0
    avg_win = total_win / len(wins) if wins else 0.0
    avg_loss = total_loss / len(losses) if losses else 0.0
    profit_factor = (
        total_win / total_loss
        if total_loss > 0
        else (999.0 if total_win > 0 else 0.0)
    )
    expectancy = ((win_rate / 100.0) * avg_win) - ((1.0 - (win_rate / 100.0)) * avg_loss)

    decision_attempts = total_signals + total_rejected
    signal_to_entry = (total_trades / total_signals * 100.0) if total_signals else 0.0
    hit_to_signal = (total_signals / total_scan_hits * 100.0) if total_scan_hits else 0.0
    reject_rate = (total_rejected / decision_attempts * 100.0) if decision_attempts else 0.0
    closed_rate = (closed_trades / total_trades * 100.0) if total_trades else 0.0

    missed_rows = list(missed_a_plus or [])
    missed_opportunities = sum(1 for r in missed_rows if r.get("outcome") == "missed_opportunity")
    correct_rejects = sum(1 for r in missed_rows if r.get("outcome") == "correct_reject")
    neutral = sum(1 for r in missed_rows if r.get("outcome") == "neutral")
    pending = max(0, len(missed_rows) - missed_opportunities - correct_rejects - neutral)
    best_missed = None
    if missed_rows:
        best_missed = max(missed_rows, key=lambda r: float(r.get("move_after_pct") or 0.0))

    if closed_trades < 5:
        verdict = "collecting"
    elif expectancy > 0 and profit_factor >= 1.2:
        verdict = "positive_expectancy"
    elif expectancy < 0 or profit_factor < 1.0:
        verdict = "negative_expectancy"
    else:
        verdict = "mixed"

    return {
        "trades_taken": total_trades,
        "closed_trades": closed_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy_per_trade": round(expectancy, 2),
        "by_entry_mode": by_entry_mode,
        "by_strategy": by_strategy,
        "cycles": cycle_count,
        "funnel": {
            "scan_hits": total_scan_hits,
            "signals": total_signals,
            "entries": total_trades,
            "rejected": total_rejected,
            "hit_to_signal_pct": round(hit_to_signal, 1),
            "signal_to_entry_pct": round(signal_to_entry, 1),
            "reject_rate_pct": round(reject_rate, 1),
            "closed_rate_pct": round(closed_rate, 1),
        },
        "missed_a_plus": {
            "rows": len(missed_rows),
            "missed_opportunities": missed_opportunities,
            "correct_rejects": correct_rejects,
            "pending": pending,
            "best_symbol": best_missed.get("symbol") if best_missed else "",
            "best_move_pct": round(float(best_missed.get("move_after_pct") or 0.0), 1) if best_missed else 0.0,
            "best_pattern": best_missed.get("pattern") if best_missed else "",
            "best_reason": best_missed.get("reason") if best_missed else "",
        },
        "verdict": verdict,
    }


def _rolling_journal_scorecard(journal: Optional[Any], window_days: int = 20) -> dict:
    min_closed_trades = 25
    min_sessions = 10
    if journal is None or not getattr(journal, "db_path", None):
        return {
            "available": False,
            "window_days": window_days,
            "min_closed_trades": min_closed_trades,
            "min_sessions": min_sessions,
            "reason": "journal not configured",
        }

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    cutoff_iso = cutoff.isoformat()
    db_path = str(journal.db_path)
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT symbol, side, trade_type, strategy, quantity, entry_price, exit_price, pnl, reason, ts
                FROM trades
                WHERE ts >= ?
                ORDER BY ts ASC
                """,
                (cutoff_iso,),
            )
            trades = [
                {
                    "symbol": r["symbol"],
                    "side": r["side"],
                    "trade_type": r["trade_type"],
                    "strategy": r["strategy"],
                    "quantity": r["quantity"],
                    "entry_price": r["entry_price"] or 0.0,
                    "entry_time": r["ts"],
                    "exit_price": r["exit_price"],
                    "exit_time": r["ts"] if r["trade_type"] == "exit" else None,
                    "pnl": r["pnl"],
                    "exit_reason": r["reason"],
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(scan_hits), 0) AS scan_hits,
                    COALESCE(SUM(signals), 0) AS signals,
                    COALESCE(SUM(rejected), 0) AS rejected,
                    COUNT(DISTINCT substr(ts, 1, 10)) AS sessions
                FROM market_context
                WHERE ts >= ?
                """,
                (cutoff_iso,),
            )
            ctx = cur.fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return {
            "available": False,
            "window_days": window_days,
            "min_closed_trades": min_closed_trades,
            "min_sessions": min_sessions,
            "reason": str(exc),
        }

    total_trades = sum(1 for t in trades if t.get("trade_type") == "entry")
    scorecard = _daily_scorecard(
        trades=trades,
        total_trades=total_trades,
        total_scan_hits=int(ctx["scan_hits"] or 0) if ctx else 0,
        total_signals=int(ctx["signals"] or 0) if ctx else 0,
        total_rejected=int(ctx["rejected"] or 0) if ctx else 0,
        cycle_count=0,
        missed_a_plus=[],
    )
    scorecard["available"] = True
    scorecard["window_days"] = window_days
    scorecard["sessions"] = int(ctx["sessions"] or 0) if ctx else 0
    scorecard["min_closed_trades"] = min_closed_trades
    scorecard["min_sessions"] = min_sessions
    scorecard["cutoff"] = cutoff_iso
    if (
        int(scorecard.get("closed_trades") or 0) < min_closed_trades
        or int(scorecard.get("sessions") or 0) < min_sessions
    ):
        scorecard["verdict"] = "collecting"
        scorecard["verdict_reason"] = (
            "needs at least {} closed trades and {} sessions".format(
                min_closed_trades,
                min_sessions,
            )
        )
    return scorecard


def _scanner_dict(s: ScannerHit) -> dict:
    return {
        "symbol": s.symbol,
        "scanner_name": s.scanner_name,
        "score": round(s.score, 4),
        "time": s.time,
        "price": round(s.price, 4),
        "criteria": s.criteria,
        "verified": s.verified,
        "action_taken": s.action_taken,
    }


def _status_dict(s: SymbolStatus) -> dict:
    return {
        "symbol": s.symbol,
        "style": s.style,
        "confidence": round(s.confidence * 100, 1),
        "price": round(s.price, 4),
        "volatility_pct": round(s.volatility_pct, 3),
        "spread_pct": round(s.spread_pct, 4),
        "relative_volume": round(s.relative_volume, 2),
        "trend_strength": round(s.trend_strength, 3),
        "liquidity_score": round(s.liquidity_score, 3),
        "reasons": s.reasons,
    }
