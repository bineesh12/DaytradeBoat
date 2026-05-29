"""Dashboard data hub — thread-safe store for pipeline events.

The pipeline pushes data here; the web server reads from here.
Uses a simple pub/sub pattern with SSE (Server-Sent Events).
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
        # Active trading watchlist (HOD TTL + pinned + open positions)
        self.trading_watchlist: List[str] = []
        self.watchlist_pinned: List[str] = []

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

    def on_fill(self, fill: Any, trade_type: str = "entry") -> None:
        rec = TradeRecord(
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=fill.quantity,
            entry_price=fill.price,
            entry_time=str(fill.ts),
            trade_type=trade_type,
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
            self.cycle_count = cycle_num
        summary = {
            "cycle": cycle_num,
            "scan_hits": len(result.scan_hits),
            "signals": len(result.signals),
            "fills": len(result.fills),
            "exits": len(result.exit_fills),
            "rejected": result.rejected_orders,
        }
        self._broadcast("cycle", summary)

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

    def add_log(self, level: str, message: str) -> None:
        entry = {"level": level, "message": message, "ts": _now_str()}
        with self._lock:
            self.log_messages.append(entry)
        self._broadcast("log", entry)

    # ------------------------------------------------------------------
    # Snapshot for initial page load
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Full current state for initial page load."""
        with self._lock:
            win_rate = 0.0
            total_closed = self.winning_trades + self.losing_trades
            if total_closed > 0:
                win_rate = self.winning_trades / total_closed * 100

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
                "recent_trades": [_trade_dict(t) for t in list(self.trades)[-50:]],
                "recent_scans": [_scanner_dict(s) for s in list(self.scanner_hits)[-50:]],
                "pnl_history": list(self.pnl_history),
                "market_open": self.market_open,
                "market_phase": self.market_phase,
                "stream_connected": self.stream_connected,
                "bot_start_time": self.bot_start_time,
                "watchlist_scan": self.watchlist_scan,
                "rt_movers": self.rt_movers,
                "hod_momentum_alerts": list(self.hod_momentum_alerts),
                "trading_watchlist": list(self.trading_watchlist),
                "watchlist_pinned": list(self.watchlist_pinned),
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
    }


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
