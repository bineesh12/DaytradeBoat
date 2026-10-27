"""Trading pipeline: Classify → Scanner → Strategy Verify → Risk → Execute → Exit.

This is the main orchestrator. Each cycle:
  0. Classify each symbol → SCALPING / DAY_TRADING / SWING / NOT_TRADEABLE
  1. Check exits on open positions (stops, targets, trailing, time)
  2. Route symbols to the correct scanners for their style
  3. Strategy verifiers check each hit against entry rules → TradeSignals
  4. Risk manager does pre-trade checks
  5. Broker executes approved orders
  6. New fills are registered with the exit manager

Supports swing trading, day trading, and scalping through a single pipeline.
"""

from __future__ import annotations
from daytrading.models import BarSource

from daytrading.strategy.scalping.micro_vwap import TEMPORARY_ENTRY_BLOCKS
from daytrading.execution.fill_identity import identity as fill_identity

import inspect
import logging
import re
from threading import RLock
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from daytrading.classifier.regime import MarketRegimeClassifier
from daytrading.classifier.router import AdaptiveRouter, StyleConfig
from daytrading.analytics.missed_a_plus import MissedAPlusTracker
from daytrading.data.news_checker import NewsChecker
from daytrading.execution.broker import Broker, apply_fill
from daytrading.exits.manager import ExitManager
from daytrading.exits.scaler import PositionScaler, ReentryDetector
from daytrading.risk.manager import allow_order
from daytrading.risk.guards import TradeGuard
from daytrading.scanner.base import Scanner
from daytrading.strategy.entry_guard import check_entry_quality
from daytrading.strategy.entry_policy import EntryDecision, EntryPolicy
from daytrading.strategy.pending_entry import PendingEntryBook
from daytrading.strategy.verifier import StrategyVerifier
from daytrading.strategy import warrior_lanes
from daytrading.models import (
    Bar,
    Fill,
    MarketRegime,
    Order,
    OrderStatus,
    PortfolioState,
    Quote,
    ScanResult,
    Side,
    SignalAction,
    Tick,
    TradeSignal,
    TradingStyle,
)
from daytrading.opportunities import hit_identity

logger = logging.getLogger(__name__)


LIVE_A_PLUS_SCANNERS = frozenset({
    "prepared_reclaim_tick",
    "micro_pullback_reclaim",
    "vwap_pullback",
    "abc_continuation",
    "first_pullback_reclaim",
    "hod_reclaim",
    "pullback_base",
    "level_breakout_reclaim",
    "runner_reclaim_continuation",
    "shallow_stair_continuation",
    "early_vwap_reclaim_scout",
})

WATCH_ONLY_SCANNERS = frozenset({
    "momentum_burst",
    "bull_flag",
    "flat_top_breakout",
    "opening_range_breakout",
    "level_breakout_watch",
})


@dataclass
class _RejectCooldown:
    reason: str
    ts: datetime
    price: float = 0.0
    volume: float = 0.0
    ttl_seconds: float = 60.0


@dataclass
class PipelineResult:
    """Result of one pipeline cycle."""
    regimes: Dict[str, MarketRegime] = field(default_factory=dict)
    scan_hits: List[ScanResult] = field(default_factory=list)
    signals: List[TradeSignal] = field(default_factory=list)
    skipped: List[TradeSignal] = field(default_factory=list)
    fills: List[Fill] = field(default_factory=list)
    exit_fills: List[Fill] = field(default_factory=list)
    scale_up_fills: List[Fill] = field(default_factory=list)
    reentry_fills: List[Fill] = field(default_factory=list)
    rejected_orders: int = 0
    rejection_details: List[Dict[str, str]] = field(default_factory=list)
    symbols_by_style: Dict[str, List[str]] = field(default_factory=dict)
    entry_strategies: Dict[str, str] = field(default_factory=dict)
    exit_reasons: Dict[str, str] = field(default_factory=dict)
    deferred_signals: List[TradeSignal] = field(default_factory=list)
    entry_decisions: List[Dict[str, Any]] = field(default_factory=list)


def _entry_strategy_label(signal: TradeSignal) -> str:
    """Strategy label for a fill — distinct for experimental scout tiers so the
    scorecard can measure their standalone expectancy."""
    sr = getattr(signal, "scan_result", None)
    if sr is not None:
        criteria = sr.criteria or {}
        entry_mode = str(criteria.get("entry_mode") or "")
        if entry_mode:
            return entry_mode
        tier = str(criteria.get("entry_tier") or "")
        if tier == "fresh_vwap_reclaim_scout":
            return "fresh_vwap_reclaim_scout"
        if tier == "vwap_reclaim_scout":
            return "vwap_reclaim_scout"
        if sr.scanner_name:
            return sr.scanner_name
    return signal.reason or "unknown"


class TradingPipeline:
    """Orchestrates: classify → exits → scan → verify → risk → execute."""

    def __init__(
        self,
        scanners: Sequence[Scanner],
        verifiers: Dict[str, StrategyVerifier],
        broker: Broker,
        portfolio: PortfolioState,
        *,
        exit_manager: Optional[ExitManager] = None,
        router: Optional[AdaptiveRouter] = None,
        scaler: Optional[PositionScaler] = None,
        reentry_detector: Optional[ReentryDetector] = None,
        max_positions: int = 5,
        max_position_shares: float = 500,
        max_order_shares: float = 200,
        max_dollar_risk_per_trade: float = 50.0,
        enable_daily_loser_blacklist: bool = False,
        daily_loser_blacklist_min_loss: float = 50.0,
        daily_loser_blacklist_max_losses: int = 2,
        level_capped_entry_enabled: bool = False,
    ) -> None:
        self._scanners = list(scanners)
        self._verifiers = verifiers
        self._broker = broker
        self._portfolio = portfolio
        self._exit_manager = exit_manager or ExitManager()
        self._router = router
        self._scaler = scaler
        self._reentry = reentry_detector
        self._max_positions = max_positions
        self._max_position_shares = max_position_shares
        self._max_order_shares = max_order_shares
        self._max_dollar_risk_per_trade = max_dollar_risk_per_trade
        self._original_sizes: Dict[str, float] = {}  # for re-entry size calc
        self._cached_regimes: Dict[str, MarketRegime] = {}  # persist trend data across cycles
        self._scan_only = False
        self._news_checker: Optional[NewsChecker] = None
        self._exit_cooldowns: Dict[str, datetime] = {}
        self._cooldown_seconds: int = 300  # 5 min cooldown after exiting a stock
        self._scan_rejections: Dict[str, str] = {}  # symbol -> last rejection reason
        self._reject_cooldowns: Dict[str, _RejectCooldown] = {}
        self.pending_entries = PendingEntryBook()
        self._trade_guard = TradeGuard()
        self._micro_pullback_only = False
        self._micro_minimum_room_r = 1.0
        self._entry_policy = EntryPolicy(
            guard=lambda *args, **kwargs: check_entry_quality(*args, **kwargs),
        )
        self._missed_a_plus = MissedAPlusTracker()
        self._enable_daily_loser_blacklist = enable_daily_loser_blacklist
        self._daily_losers: set = set()  # symbols blocked from re-entry today
        self._daily_loss_counts: Dict[str, int] = {}
        # A single normal scalp loss shouldn't ban a name for the day — a name
        # that loses a small morning scalp can set up a clean afternoon re-entry
        # (the GLXG case: −$20 morning loss blocked a +$100 afternoon reclaim).
        # Ban only on a real blowout (>= min_loss) OR after max_losses losses.
        self._daily_loser_blacklist_min_loss: float = max(0.0, float(daily_loser_blacklist_min_loss))
        self._daily_loser_blacklist_max_losses: int = max(1, int(daily_loser_blacklist_max_losses))
        self._symbol_entry_counts: Dict[str, int] = {}  # per-symbol entries this session
        self._max_entries_per_symbol: int = 3
        self._daily_pnl: float = 0.0  # running realized P&L for the day
        # Filled exit legs, grouped until the actual position is flat. Independent
        # of exit-manager intent (which may decrement quantity before a fill).
        self._realized_trade_legs: Dict[str, tuple] = {}
        self._realized_exit_fills: Dict[tuple, float] = {}
        self._exit_accounting_lock = RLock()
        self._max_daily_loss: float = -200.0  # Warrior Trading: stop after -$200
        self._circuit_breaker_tripped: bool = False
        self._execution_timer = None  # set by runner if 10s timing is enabled
        self._bar_aggregator = None   # set by runner for 5m context
        self._require_hod_alert_for_entry: bool = False
        self._hod_active_checker = None  # Callable[[str], bool] set by runner
        self._hod_entry_bypass_checker = None
        self._missed_a_plus_chase_window_sec: float = 1800.0
        self._missed_a_plus_chase_pct_sub5: float = 0.035
        self._missed_a_plus_chase_pct_5plus: float = 0.025
        self._missed_a_plus_fresh_base_reset: bool = False
        self._missed_a_plus_fresh_base_pct: float = 0.08
        # Normal anti-chase cap (see configure_entry_chase_guard / StrategyConfig).
        self._entry_chase_pct_low: float = 0.05
        self._entry_chase_pct_high: float = 0.025
        self._entry_chase_price_tier: float = 10.0
        # Max stop distance (fraction below entry) allowed before rejecting a
        # long entry as too-wide-risk. 0 disables. Set from StrategyConfig.
        self._max_entry_risk_pct: float = 0.0
        self._level_capped_entry_enabled = bool(level_capped_entry_enabled)

    def _append_entry_decision(
        self,
        result: PipelineResult,
        decision: EntryDecision,
    ) -> None:
        result.entry_decisions.append(decision.to_payload())

    def _merge_pending_hits(self, hits, pending_hits, symbols):
        hits = [hit for hit in hits if
            (hit.symbol not in self.pending_entries.rows or (getattr(self,'_unified_micro_entry',False) and hit.scanner_name in ('micro_pullback_reclaim','prepared_reclaim_tick') and hit.criteria.get('_unified_micro_entry'))) and hit.symbol not in self.pending_entries.inflight and
            (hit.scanner_name not in ("vwap_pullback", "hod_reclaim") or
             hit_identity(hit)[0] not in self.pending_entries.retired)]
        return hits + [hit for symbol, hit in pending_hits.items() if symbol in symbols]

    def _flush_pending_events(self, result):
        for hit, status, reason in self.pending_entries.drain_events():
            signal = TradeSignal(symbol=hit.symbol, action=SignalAction.ENTER_LONG, quantity=0,
                                 entry_price=hit.bars[-1].close, scan_result=hit)
            result.entry_decisions.append(self._entry_policy.decision(
                symbol=hit.symbol, stage="pending_entry", passed=status in ("confirming", "submitted"), signal=signal,
                reason=reason, metadata={"status": status, "ttl_seconds": self.pending_entries.ttl_seconds},
            ).to_payload())

    def _append_entry_reject(
        self,
        result: PipelineResult,
        *,
        symbol: str,
        stage: str,
        reason: str,
        hit: Optional[ScanResult] = None,
        signal: Optional[TradeSignal] = None,
        price: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if signal is None and hit is not None:
            signal = TradeSignal(
                symbol=symbol,
                action=SignalAction.ENTER_LONG,
                quantity=0,
                entry_price=price or self._hit_price_volume(hit)[0],
                reason=reason,
                scan_result=hit,
            )
        pending_hit = signal.scan_result if signal is not None else hit
        if getattr(self, "_pending_entries_enabled", True):
            self.pending_entries.reject(pending_hit, reason, getattr(self, "_cycle_now", None) or datetime.now(timezone.utc))
        self._append_entry_decision(
            result,
            self._entry_policy.decision(
                symbol=symbol,
                stage=stage,
                passed=False,
                reason=reason,
                blocked_layer=stage,
                signal=signal,
                price=price or None,
                metadata=metadata,
            ),
        )

    def set_hod_entry_gate(
        self,
        checker,
        *,
        require: bool = True,
        bypass_checker=None,
    ) -> None:
        """Only allow new entries when checker(symbol) is True."""
        self._hod_active_checker = checker
        self._require_hod_alert_for_entry = require
        self._hod_entry_bypass_checker = bypass_checker

    def configure_missed_a_plus_chase_guard(
        self,
        *,
        window_sec: float,
        pct_sub5: float,
        pct_5plus: float,
        fresh_base_reset: bool = False,
        fresh_base_pct: float = 0.08,
    ) -> None:
        self._missed_a_plus_chase_window_sec = max(0.0, float(window_sec))
        self._missed_a_plus_chase_pct_sub5 = max(0.0, float(pct_sub5))
        self._missed_a_plus_chase_pct_5plus = max(0.0, float(pct_5plus))
        self._missed_a_plus_fresh_base_reset = bool(fresh_base_reset)
        self._missed_a_plus_fresh_base_pct = max(0.0, float(fresh_base_pct))

    def configure_entry_chase_guard(
        self,
        *,
        pct_low: float,
        pct_high: float,
        price_tier: float,
    ) -> None:
        self._entry_chase_pct_low = max(0.0, float(pct_low))
        self._entry_chase_pct_high = max(0.0, float(pct_high))
        self._entry_chase_price_tier = max(0.0, float(price_tier))

    @staticmethod
    def _setup_tier(hit: ScanResult) -> str:
        scanner = hit.scanner_name
        pattern = str(hit.criteria.get("pattern") or "")
        if scanner in LIVE_A_PLUS_SCANNERS or pattern in LIVE_A_PLUS_SCANNERS:
            return "A+ setup"
        if scanner in WATCH_ONLY_SCANNERS or pattern in WATCH_ONLY_SCANNERS:
            return "watch only"
        return "watch only"

    @staticmethod
    def _watch_only_reason(hit: ScanResult) -> str:
        return "watch only: {} collecting data, not live A+ setup".format(
            hit.scanner_name,
        )

    @property
    def portfolio(self) -> PortfolioState:
        return self._portfolio

    @property
    def exit_manager(self) -> ExitManager:
        return self._exit_manager

    @property
    def trade_guard(self) -> TradeGuard:
        return self._trade_guard

    def set_scan_only(self, value: bool) -> None:
        """When True, classify/scan/verify without new entries; exits remain active."""
        self._scan_only = value

    @property
    def scan_rejections(self) -> Dict[str, str]:
        return dict(self._scan_rejections)

    @property
    def missed_a_plus(self) -> MissedAPlusTracker:
        return self._missed_a_plus

    def missed_a_plus_report(self, limit: int = 30) -> List[Dict[str, Any]]:
        return self._missed_a_plus.report(limit=limit)

    def missed_a_plus_spread_summary(self) -> Dict[str, Any]:
        return self._missed_a_plus.spread_summary()

    def missed_a_plus_risk_summary(self) -> Dict[str, Any]:
        return self._missed_a_plus.risk_summary()

    def scanner_near_miss_summary(self) -> Dict[str, Any]:
        return self._missed_a_plus.scanner_near_miss_summary()

    def _scanner_float_shares(self, symbol: str) -> Optional[float]:
        for verifier in self._verifiers.values():
            getter = getattr(verifier, "_get_float_shares", None)
            if callable(getter):
                try:
                    return getter(symbol)
                except Exception:
                    return None
        return None

    def _record_scanner_near_misses(
        self,
        universe: Dict[str, Sequence[Bar]],
        now: Optional[datetime],
    ) -> None:
        """Report-only: log low-float, high-volume names that died at the scanner
        stage (no clean A+ pattern) so the scanner-gap report can later separate
        a real missed setup from a gappy washout. Changes nothing the bot trades.
        """
        for symbol, reason in list(self._scan_rejections.items()):
            try:
                self._missed_a_plus.record_scanner_near_miss(
                    symbol=symbol,
                    reason=reason,
                    universe=universe,
                    float_shares=self._scanner_float_shares(symbol),
                    now=now,
                )
            except Exception:
                continue

    @staticmethod
    def _get_last_reject(symbol: str, verifier: Any) -> Optional[str]:
        """Try to extract the last rejection reason from verifier logs."""
        last_reject = getattr(verifier, '_last_reject', None)
        return last_reject

    def set_news_checker(self, checker: NewsChecker) -> None:
        """Attach a news checker for pre-trade sentiment screening."""
        self._news_checker = checker

    @staticmethod
    def _hit_price_volume(hit: ScanResult) -> tuple[float, float]:
        from daytrading.strategy.scalping.micro_cache import ROUTES, context
        if hit.scanner_name in ROUTES:
            micro = context(hit)
            return (micro['price'],micro['volume']) if micro else (0.,0.)
        price = float(
            hit.criteria.get("close")
            or hit.criteria.get("price")
            or hit.criteria.get("entry_price")
            or 0.0
        )
        volume = float(hit.criteria.get("volume") or 0.0)
        if hit.bars:
            latest = hit.bars[-1]
            price = price or float(latest.close or 0.0)
            volume = volume or float(latest.volume or 0.0)
        return price, volume

    @staticmethod
    def _reject_ttl_seconds(reason: str) -> float:
        text = reason.lower()
        if "stale data" in text:
            return 0.0
        if "watch only" in text:
            return 0.0
        if "unknown pattern: level_breakout_watch" in text:
            return 0.0
        # Quote spread can tighten within seconds on active momentum names.
        # Treat it as a fast-changing market condition, not a structural setup
        # failure, so Warrior/runner setups keep monitoring for a tradeable book.
        if "spread too wide" in text:
            return 5.0
        if "thin sub-$5 liquidity" in text or "too illiquid" in text:
            return 180.0
        if "tape too slow" in text:
            return 120.0
        if "below vwap" in text or "not strong above vwap" in text:
            return 45.0
        if "price $" in text and "outside range" in text:
            return 300.0
        if "not enough movement" in text:
            return 120.0
        return 60.0

    @staticmethod
    def _materially_changed(
        old_price: float,
        old_volume: float,
        new_price: float,
        new_volume: float,
    ) -> bool:
        if old_price > 0 and new_price > 0:
            if abs(new_price - old_price) / old_price >= 0.01:
                return True
        if old_volume > 0 and new_volume > 0:
            if new_volume >= old_volume * 1.25:
                return True
        return False

    @staticmethod
    def _cooldown_pattern(hit: ScanResult) -> str:
        pattern = str(hit.criteria.get("pattern") or "").strip()
        return pattern or hit.scanner_name

    def _cooldown_key(self, hit: ScanResult) -> str:
        from daytrading.strategy.scalping.micro_cache import key
        return key(hit, "{}:{}".format(hit.symbol, self._cooldown_pattern(hit)))

    def _reject_cooldown_reason(self, hit: ScanResult, now: datetime) -> Optional[str]:
        from daytrading.strategy.scalping.micro_cache import ROUTES, context
        from daytrading.strategy.entry_checks import record, diagnostics
        micro = context(hit) if hit.scanner_name in ROUTES else None
        if hit.scanner_name in ROUTES and micro is None:
            return None  # Missing identity cannot reuse a legacy cross-trigger record.
        cd = self._reject_cooldowns.get(self._cooldown_key(hit))
        if micro:
            changed = cd is not None and (cd.price!=micro['price'] or cd.volume!=micro['volume'])
            record(diagnostics(hit),'micro_reject_cache',now=now,measured=micro['volume'],
                threshold=cd.volume if cd else None,units='shares in original 10s trigger',
                passed=not (cd is not None and not changed and 0<=(now-cd.ts).total_seconds()<cd.ttl_seconds and not self._transient_micro_rejection(hit,cd.reason)),comparison='unchanged original setup inputs',
                window={'source':'original trigger','setup_id':micro['identity'],'route':hit.scanner_name},
                cached_reason=cd.reason if cd else None,cache_found=cd is not None,inputs_changed=changed)
            if changed:return None
        if cd is not None and self._transient_micro_rejection(hit, cd.reason):
            return None  # Preserve the historical record, but re-evaluate current data.
        if cd is None or cd.ttl_seconds <= 0:
            return None
        elapsed = (now - cd.ts).total_seconds()
        if elapsed >= cd.ttl_seconds:
            return None
        price, volume = self._hit_price_volume(hit)
        if self._materially_changed(cd.price, cd.volume, price, volume):
            return None
        return "cached reject: {} ({:.0f}s left)".format(
            cd.reason,
            max(0.0, cd.ttl_seconds - elapsed),
        )

    def _record_reject_cooldown(
        self,
        hit: ScanResult,
        reason: str,
        now: datetime,
    ) -> None:
        if self._transient_micro_rejection(hit, reason):
            return  # Never replace a structural rejection with a temporary clock/quote error.
        from daytrading.strategy.scalping.micro_cache import ROUTES, context, prune
        micro = context(hit) if hit.scanner_name in ROUTES else None
        if hit.scanner_name in ROUTES and micro is None:return
        ttl = self._reject_ttl_seconds(reason)
        if micro:
            ttl = min(ttl,max(0.,(micro['deadline']-now).total_seconds()))
            prune(self._reject_cooldowns,now,maximum=2047)
        key = self._cooldown_key(hit)
        if ttl <= 0:
            self._reject_cooldowns.pop(key, None)
            return
        price, volume = self._hit_price_volume(hit)
        self._reject_cooldowns[key] = _RejectCooldown(
            reason=reason,
            ts=now,
            price=price,
            volume=volume,
            ttl_seconds=ttl,
        )

    def _clear_reject_cooldown(self, hit: ScanResult) -> None:
        self._reject_cooldowns.pop(self._cooldown_key(hit), None)

    @staticmethod
    def _transient_micro_rejection(hit: ScanResult, reason: str) -> bool:
        from daytrading.strategy.scalping.micro_failure import temporary
        return hit.scanner_name in ('micro_pullback_reclaim', 'prepared_reclaim_tick') and temporary(reason)

    def set_cooldown(self, symbol: str, now: Optional[datetime] = None) -> None:
        """Record an exit time for cooldown enforcement."""
        self._exit_cooldowns[symbol] = now or datetime.now(timezone.utc)

    @staticmethod
    def exit_fill_key(fill: Fill) -> tuple:
        """Shared identity for a fill seen by pipeline, runner and reconciliation."""
        return fill_identity(fill)

    def record_realized_exit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        *,
        now: Optional[datetime] = None,
        remaining_quantity: Optional[float] = None,
        fill_key: Optional[tuple] = None,
    ) -> float:
        """Book each fill immediately; classify a loss only at position close.

        Execution callers supply actual remaining quantity, never intended exit
        size. Legacy standalone callers without a tracked position represent a
        complete trade. Fill identity prevents pipeline/runner double booking.
        """
        with self._exit_accounting_lock:
            if fill_key and fill_key[0] == 'broker':
                from daytrading.execution.fill_identity import delta
                from daytrading.models import Fill, Side
                _, sym, side, oid, cumulative_qty, cumulative_cost = fill_key
                states = self.__dict__.setdefault('_fill_projections', {})
                watermark = states.setdefault('realized', {})
                observed = Fill(sym, Side(side), cumulative_qty, cumulative_cost/cumulative_qty,
                                now or datetime.now(timezone.utc), order_id=oid,
                                cumulative_quantity=cumulative_qty, cumulative_notional=cumulative_cost)
                try:
                    incremental, projection_key, projected_state = delta(observed, watermark)
                except ValueError:
                    self._fill_reconciliation_required = True
                    raise
                if incremental is None:
                    return 0.0
                quantity, exit_price = incremental.quantity, incremental.price
            else:
                projected_state = None
            if fill_key is not None and fill_key in self._realized_exit_fills:
                return 0.0
            if entry_price <= 0 or quantity <= 0:
                return 0.0
            tracked = self._exit_manager.tracked.get(symbol)
            if remaining_quantity is None:
                remaining_quantity = tracked.remaining_qty if tracked is not None else 0.0
            if remaining_quantity < 0:
                raise ValueError("Exit accounting requires nonnegative remaining quantity")
            complete = remaining_quantity <= 1e-8
            trade_pnl = (exit_price - entry_price) * quantity
            total_pnl, total_cost, total_qty = self._realized_trade_legs.get(symbol, (0.0, 0.0, 0.0))
            total_pnl += trade_pnl
            total_cost += entry_price * quantity
            total_qty += quantity
            self._daily_pnl += trade_pnl
            if fill_key is not None:
                self._realized_exit_fills[fill_key] = trade_pnl
            if complete:
                self._realized_trade_legs.pop(symbol, None)
            else:
                self._realized_trade_legs[symbol] = (total_pnl, total_cost, total_qty)

            if self._enable_daily_loser_blacklist:
                # The dollar guard remains immediate, including a large negative
                # leg after profitable partials. Accumulated losses also count.
                dollar_loss = max(0.0, -trade_pnl, -total_pnl)
                if dollar_loss >= self._daily_loser_blacklist_min_loss and dollar_loss > 0:
                    self._daily_losers.add(symbol)
                    logger.info("DAILY BLACKLIST %s: realized dollar loss $%.2f", symbol, dollar_loss)
                if complete and total_pnl < -1e-8:
                    loss_count = self._daily_loss_counts.get(symbol, 0) + 1
                    self._daily_loss_counts[symbol] = loss_count
                    if loss_count >= self._daily_loser_blacklist_max_losses:
                        self._daily_losers.add(symbol)
                    logger.info("COMPLETED TRADE LOSS %s: net $%.2f, %d consecutive loss(es)",
                                symbol, total_pnl, loss_count)
            if complete and total_pnl > 1e-8:
                self._daily_loss_counts[symbol] = 0
                self._exit_cooldowns[symbol] = (now or datetime.now(timezone.utc)) - timedelta(
                    seconds=self._cooldown_seconds - 120
                )
            if self._daily_pnl <= self._max_daily_loss and not self._circuit_breaker_tripped:
                self._circuit_breaker_tripped = True
                logger.warning("CIRCUIT BREAKER: daily P&L $%.2f hit max loss $%.2f — NO MORE TRADES TODAY",
                               self._daily_pnl, self._max_daily_loss)
            if projected_state is not None:
                watermark[projection_key] = projected_state
            logger.info("EXIT ACCOUNTING %s: fill $%.2f, position realized $%.2f, remaining %.4f, complete=%s",
                        symbol, trade_pnl, total_pnl, remaining_quantity, complete)
            if complete:
                try:
                    from daytrading.ml.data_collector import log_trade_outcome
                    log_trade_outcome(
                        symbol=symbol, entry_price=entry_price, exit_price=exit_price,
                        entry_time=getattr(tracked, "entry_ts", None),
                        exit_time=now or datetime.now(timezone.utc),
                        outcome_pct=total_pnl / total_cost * 100 if total_cost > 0 else 0.0,
                    )
                except Exception:
                    pass
            return trade_pnl

    @staticmethod
    def _latest_quote(quotes: Optional[Dict[str, Sequence[Quote]]], symbol: str) -> Optional[Quote]:
        if not quotes:
            return None
        seq = quotes.get(symbol)
        if seq:
            return seq[-1]
        return None

    @staticmethod
    def _latest_price_for_symbol(
        universe: Dict[str, Sequence[Bar]], symbol: str, fallback: float = 0.0,
    ) -> float:
        bars = universe.get(symbol)
        if bars:
            return float(bars[-1].close)
        return float(fallback or 0.0)

    def _log_shadow_missed(
        self,
        *,
        symbol: str,
        reason: str,
        universe: Dict[str, Sequence[Bar]],
        quotes: Optional[Dict[str, Sequence[Quote]]],
        scanner: str = "",
        fallback_price: float = 0.0,
        hit: Optional[ScanResult] = None,
    ) -> None:
        try:
            tracker_hit = hit
            if tracker_hit is None:
                bars = list(universe.get(symbol, []))
                if bars:
                    tracker_hit = ScanResult(
                        symbol=symbol,
                        scanner_name=scanner or "unknown",
                        ts=bars[-1].ts,
                        score=0.0,
                        criteria={
                            "pattern": scanner or "unknown",
                            "close": fallback_price or bars[-1].close,
                        },
                        bars=bars,
                    )
            self._missed_a_plus.record_blocked(
                layer=self._blocked_layer(reason),
                reason=reason,
                universe=universe,
                quotes=quotes,
                hit=tracker_hit,
                fallback_price=fallback_price,
            )
        except Exception:
            pass
        try:
            from daytrading.ml.shadow_collector import log_missed_opportunity
            bars = list(hit.bars) if hit is not None and hit.bars else universe.get(symbol, [])
            quote = self._latest_quote(quotes, symbol)
            hit_price = 0.0
            if hit is not None:
                hit_price = float(hit.criteria.get("close") or hit.criteria.get("price") or 0.0)
            price = hit_price or self._latest_price_for_symbol(universe, symbol, fallback_price)
            if price > 0:
                log_missed_opportunity(
                    symbol=symbol,
                    price=price,
                    reason=reason,
                    scanner=scanner,
                    bars=bars,
                    quotes=[quote] if quote else None,
                    scanner_score=float(hit.score) if hit is not None else None,
                    criteria=hit.criteria if hit is not None else None,
                )
        except Exception:
            pass

    @staticmethod
    def _blocked_layer(reason: str) -> str:
        text = str(reason or "").lower()
        if "ml model" in text or "ml low confidence" in text:
            return "ml"
        if "final entry guard" in text or "entry guard" in text or "entry score" in text:
            return "entry_guard"
        if "r:r" in text or "risk" in text or "position" in text or "risk cap" in text:
            return "risk"
        if "order_" in text or "no bar data" in text:
            return "order"
        if "watch only" in text or "did not pass verifier" in text:
            return "verifier"
        if "hod momentum" in text or "cooldown" in text or "blacklist" in text:
            return "risk"
        return "scanner"

    def _log_shadow_pullback(
        self,
        hit: ScanResult,
        universe: Dict[str, Sequence[Bar]],
        quotes: Optional[Dict[str, Sequence[Quote]]],
    ) -> None:
        try:
            pattern = str(hit.criteria.get("pattern", ""))
            if hit.scanner_name not in (
                "pullback_base", "vwap_pullback", "hod_reclaim",
                "abc_continuation", "early_vwap_reclaim_scout",
            ) and (
                pattern not in (
                    "pullback_base", "vwap_pullback", "hod_reclaim",
                    "abc_continuation", "early_vwap_reclaim_scout",
                )
            ):
                return
            from daytrading.ml.shadow_collector import log_pullback_candidate
            bars = hit.bars or list(universe.get(hit.symbol, []))
            quote = self._latest_quote(quotes, hit.symbol)
            price = float(hit.criteria.get("close") or self._latest_price_for_symbol(
                universe, hit.symbol,
            ))
            if price > 0:
                log_pullback_candidate(
                    symbol=hit.symbol,
                    price=price,
                    scanner=hit.scanner_name,
                    criteria=hit.criteria,
                    bars=bars,
                    quotes=[quote] if quote else None,
                )
        except Exception:
            pass

    @staticmethod
    def _allow_vwap_reclaim_scout_trade_guard_exception(
        signal: TradeSignal,
        *,
        bars: Optional[Sequence[Bar]],
        quotes: Optional[Sequence[Quote]],
        reason: Optional[str],
    ) -> bool:
        """Let a reduced VWAP reclaim scout pass the duplicate liquidity-trap check.

        The main entry guard already scored this as a near-miss A+ VWAP reclaim.
        This exception is intentionally narrow: it only clears the trade guard's
        spread/weak-volume trap when the scout is tagged and the recent tape is
        still active enough. Spike-and-fade and gap-up trap rejects remain hard.
        """
        text = str(reason or "").lower()
        if "liquidity trap: spread" not in text:
            return False
        sr = signal.scan_result
        criteria = sr.criteria if sr is not None else {}
        if str(criteria.get("entry_tier") or "") != "vwap_reclaim_scout":
            return False
        if not bars or len(bars) < 6 or not quotes:
            return False

        latest = bars[-1]
        quote = quotes[-1]
        if latest.close <= 0 or latest.close <= latest.open:
            return False
        if quote.spread_pct > 1.0:
            return False

        day_volume = sum(max(0.0, float(b.volume or 0.0)) for b in bars)
        recent_volume = sum(max(0.0, float(b.volume or 0.0)) for b in bars[-3:])
        prior_5 = list(bars[-6:-1])
        avg_prior = (
            sum(max(0.0, float(b.volume or 0.0)) for b in prior_5) / len(prior_5)
            if prior_5 else 0.0
        )
        latest_volume = max(0.0, float(latest.volume or 0.0))
        if day_volume < 500_000 or recent_volume < 150_000 or latest_volume < 40_000:
            return False
        if avg_prior > 0 and latest_volume < avg_prior * 0.75:
            return False

        body = abs(latest.close - latest.open)
        upper_wick = latest.high - max(latest.open, latest.close)
        if body > 0 and upper_wick > body * 2.0:
            return False

        vwap = float(criteria.get("vwap") or 0.0)
        if vwap > 0 and latest.close < vwap * 1.003:
            return False
        return True

    @staticmethod
    def _allow_warrior_starter_guard_exception(
        signal: TradeSignal,
        *,
        bars: Optional[Sequence[Bar]],
        reason: Optional[str],
    ) -> bool:
        """Clear only narrow final-guard false blocks for Warrior starters."""
        text = str(reason or "").lower()
        score_match = re.search(r"entry score too low \((\d+)/100", text)
        liquidity_watch_only = "watch-only liquidity score" in text
        low_day_volume = "low day volume" in text
        sr = signal.scan_result
        criteria = sr.criteria if sr is not None else {}
        entry_trigger = str(criteria.get("entry_trigger") or "")
        if not warrior_lanes.is_warrior_entry_trigger(entry_trigger):
            return False
        score_floor = (
            60
            if entry_trigger == "warrior_smooth_10s_pullback_continuation"
            else 70
            if entry_trigger == "warrior_high_base_reclaim"
            else 75
        )
        score_near_miss = bool(score_match and int(score_match.group(1)) >= score_floor)
        if (
            "dead cat bounce" not in text
            and not score_near_miss
            and not liquidity_watch_only
            and not (
                low_day_volume
                and entry_trigger == "warrior_early_10s_pullback_reclaim"
            )
        ):
            return False
        if str(criteria.get("entry_mode") or "") != "warrior_squeeze_playbook":
            return False
        if str(criteria.get("setup_tier") or "").lower() != "a+ setup":
            return False

        try:
            size_factor = float(criteria.get("size_factor") or 1.0)
        except (TypeError, ValueError):
            size_factor = 1.0
        if size_factor > 0.35:
            return False

        price = float(signal.entry_price or 0.0)
        stop = float(signal.stop_loss or 0.0)
        psych_level = float(criteria.get("psych_level") or 0.0)
        if price <= 0 or stop <= 0 or stop >= price:
            return False
        psych_tolerance = (
            0.985
            if entry_trigger == "warrior_smooth_10s_pullback_continuation"
            else 1.0
        )
        if psych_level > 0 and price < psych_level * psych_tolerance:
            return False
        if (price - stop) / price > 0.09:
            return False

        if not bars:
            return False
        latest = bars[-1]
        if float(latest.close or 0.0) <= float(latest.open or 0.0):
            return False
        rng = float(latest.high or 0.0) - float(latest.low or 0.0)
        if rng <= 0:
            return False
        close_location = (float(latest.close or 0.0) - float(latest.low or 0.0)) / rng
        min_close_location = 0.30 if liquidity_watch_only else 0.55
        if close_location < min_close_location:
            return False

        try:
            criteria_latest_volume = float(criteria.get("latest_volume") or 0.0)
        except (TypeError, ValueError):
            criteria_latest_volume = 0.0
        try:
            criteria_recent_volume = float(criteria.get("recent_volume") or 0.0)
        except (TypeError, ValueError):
            criteria_recent_volume = 0.0
        latest_volume = max(float(latest.volume or 0.0), criteria_latest_volume)
        recent_volume = max(
            sum(float(b.volume or 0.0) for b in bars[-3:]),
            criteria_recent_volume,
        )
        if entry_trigger == "warrior_smooth_10s_pullback_continuation":
            min_latest_volume = 30_000
            min_recent_volume = 80_000
        elif entry_trigger == "warrior_early_10s_pullback_reclaim":
            min_latest_volume = 25_000
            min_recent_volume = 65_000
        elif entry_trigger == "warrior_second_leg_vwap_reclaim":
            min_latest_volume = 60_000
            min_recent_volume = 180_000
        else:
            min_latest_volume = (
                75_000
                if liquidity_watch_only or entry_trigger == "warrior_prior_runner_continuation_pullback"
                else 100_000
            )
            min_recent_volume = 150_000 if liquidity_watch_only else 250_000
        if latest_volume < min_latest_volume or recent_volume < min_recent_volume:
            return False

        return True

    @staticmethod
    def _log_shadow_execution(
        *,
        order: Order,
        bar: Bar,
        status: OrderStatus,
        fill: Optional[Fill],
        source: str,
    ) -> None:
        try:
            from daytrading.ml.shadow_collector import log_execution_quality
            log_execution_quality(
                order=order,
                bar=bar,
                status=status,
                fill=fill,
                source=source,
            )
        except Exception:
            pass

    def run_cycle(
        self,
        universe: Dict[str, Sequence[Bar]],
        now: Optional[datetime] = None,
        quotes: Optional[Dict[str, Sequence[Quote]]] = None,
        ticks: Optional[Dict[str, Sequence[Tick]]] = None,
        *, allow_entries: bool = True, only_scanners: Optional[frozenset[str]] = None,
        market_now: Optional[datetime] = None,
        session_universe: Optional[Dict[str, Sequence[Bar]]] = None,
        entry_blocker: Optional[str] = None,
    ) -> PipelineResult:
        """Run one full cycle: classify → exits → scan → verify → execute."""
        result = PipelineResult()
        entries_enabled = allow_entries and not self._scan_only
        self._pending_entries_enabled = entries_enabled
        self._entry_global_blocker = entry_blocker if not entries_enabled else None
        self._missed_a_plus.update_prices(universe, now=now)
        if only_scanners is None:
            self._record_scanner_near_misses(universe, now)

        if now is None:
            for bars in universe.values():
                if bars:
                    now = bars[-1].ts
                    break

        self._cycle_now = now or datetime.now(timezone.utc)
        self._market_now = market_now or self._cycle_now
        self._cycle_ticks = ticks or {}
        pending_hits = {} if only_scanners is not None else self.pending_entries.prepare(
            universe, quotes, ticks, self._portfolio, self._cycle_now,
            entries_enabled=entries_enabled, enabled_patterns=self._entry_policy.entry_patterns,
        )

        # --- Step 0: check exits on open positions ---
        prices: Dict[str, float] = {}
        if now is not None:
            prices = {
                sym: brs[-1].close
                for sym, brs in universe.items()
                if brs
            }
            fresh_exits = getattr(self._broker, 'requires_fresh_exit_quotes', False) is True
            if fresh_exits:
                from daytrading.execution.live_prices import resolve_exit_prices
                exit_quote_details = {}
                prices = resolve_exit_prices(list(self._exit_manager.tracked),
                    sides={s:p.side for s,p in self._exit_manager.tracked.items()},
                    now=self._market_now, quotes=quotes, diagnostics=exit_quote_details)
                self._broker._exit_quote_evidence = exit_quote_details
            # Feed bar close data for red candle exit detection
            for sym, brs in universe.items():
                if len(brs) >= 2:
                    self._exit_manager.update_bar_close(
                        sym, brs[-2].close, brs[-2].open, brs[-2].volume,
                        high_price=brs[-2].high, low_price=brs[-2].low,
                    )
            pre_exit_positions = {
                sym: {
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "remaining_qty": pos.remaining_qty,
                    "entry_price": pos.entry_price,
                    "highest_price": pos.highest_price,
                    "lowest_price": pos.lowest_price,
                    "tracked_object": pos,
                    "sold_half": pos.sold_half,
                    "current_step": getattr(pos, "current_step", 0),
                    "breakeven_locked": getattr(pos, "breakeven_locked", False),
                }
                for sym, pos in self._exit_manager.tracked.items()
            }
            exit_signals = self._exit_manager.check_exits(prices, now)
            for exit_sig in exit_signals:
                exit_order = self._signal_to_order(exit_sig)
                if exit_order is None:
                    continue
                bar = self._latest_bar(universe, exit_sig.symbol)
                if fresh_exits and exit_sig.symbol in prices:
                    mark = prices[exit_sig.symbol]
                    exit_order = replace(exit_order, limit_price=mark,
                        liquidation_quote_ts=datetime.fromisoformat(exit_quote_details[exit_sig.symbol]['quote_ts']))
                    bar = Bar(symbol=exit_sig.symbol, open=mark, high=mark, low=mark,
                              close=mark, volume=0, ts=self._market_now, provenance=BarSource.PRICE_MARK)
                if bar is None:
                    continue
                fill, status = self._broker.submit(exit_order, bar, self._portfolio)
                if status is OrderStatus.FILLED and fill is not None:
                    apply_fill(self._portfolio, fill)
                    self._exit_manager.commit_coordinated_exit_fill(fill)
                    result.exit_fills.append(fill)
                    result.exit_reasons[exit_sig.symbol] = exit_sig.reason or "unknown"
                    self._exit_cooldowns[exit_sig.symbol] = now or datetime.now(timezone.utc)

                    tracked_pos = self._exit_manager._positions.get(exit_sig.symbol)
                    entry_px = float(pre_exit_positions.get(exit_sig.symbol, {}).get("entry_price") or (tracked_pos.entry_price if tracked_pos else 0))
                    if entry_px > 0:
                        self.record_realized_exit(
                            exit_sig.symbol, entry_px, fill.price, fill.quantity,
                            now=now,
                            remaining_quantity=abs(self._portfolio.positions[fill.symbol].quantity) if fill.symbol in self._portfolio.positions else 0.0,
                            fill_key=self.exit_fill_key(fill),
                        )

                    logger.info(
                        "EXIT FILLED %s %.0f @ %.4f | %s (cooldown %ds, day P&L $%.2f)",
                        exit_sig.symbol, fill.quantity, fill.price, exit_sig.reason,
                        self._cooldown_seconds, self._daily_pnl,
                    )
                    if self._reentry is not None:
                        remaining_after = 0.0
                        live_tracked = self._exit_manager.tracked.get(exit_sig.symbol)
                        if live_tracked is not None:
                            remaining_after = live_tracked.remaining_qty
                        if remaining_after <= 0:
                            snapshot = pre_exit_positions.get(exit_sig.symbol, {})
                            entry_price = float(snapshot.get("entry_price") or entry_px or fill.price)
                            original_qty = float(
                                self._original_sizes.get(exit_sig.symbol)
                                or snapshot.get("quantity")
                                or fill.quantity
                            )
                            self._original_sizes[exit_sig.symbol] = original_qty
                            self._reentry.record_full_exit(
                                symbol=exit_sig.symbol,
                                side=snapshot.get(
                                    "side",
                                    Side.BUY if exit_sig.action is SignalAction.EXIT_LONG else Side.SELL,
                                ),
                                exit_price=fill.price,
                                exit_ts=now,
                                highest_price=max(
                                    float(snapshot.get("highest_price") or fill.price),
                                    fill.price,
                                ),
                                entry_price=entry_price,
                            )
                            if self._scaler:
                                self._scaler.clear(exit_sig.symbol)
                else:
                    previous = pre_exit_positions.get(exit_sig.symbol, {})
                    tracked_pos = previous.get("tracked_object")
                    position = self._portfolio.positions.get(exit_sig.symbol)
                    if tracked_pos is not None and position is not None:
                        tracked_pos.remaining_qty = min(
                            float(previous["remaining_qty"]), abs(float(position.quantity)),
                        )
                        tracked_pos.sold_half = previous["sold_half"]
                        tracked_pos.current_step = previous["current_step"]
                        tracked_pos.breakeven_locked = previous["breakeven_locked"]
                        if tracked_pos.remaining_qty > 0:
                            self._exit_manager._positions[exit_sig.symbol] = tracked_pos
                            logger.warning(
                                "EXIT NOT FILLED %s: retained qty=%g and tightened stop=%s",
                                exit_sig.symbol, tracked_pos.remaining_qty, tracked_pos.stop_loss,
                            )
        # --- Step 0b: scale-up check on winning positions ---
        if only_scanners is None and entries_enabled and self._scaler is not None and now is not None:
            scale_signals = self._scaler.check_scale_ups(
                self._exit_manager, prices, universe,
            )
            for scale_sig in scale_signals:
                final_quality_reject = self._final_entry_quality_reject(
                    scale_sig, universe=universe, quotes=quotes,
                    result=result, stage="scale_up_final", now=now,
                )
                if final_quality_reject:
                    result.rejected_orders += 1
                    result.rejection_details.append({
                        "symbol": scale_sig.symbol,
                        "reason": "final entry guard: {}".format(final_quality_reject),
                    })
                    self._scan_rejections[scale_sig.symbol] = "final entry guard: {}".format(
                        final_quality_reject,
                    )
                    logger.info(
                        "FINAL ENTRY GUARD REJECT scale-up %s: %s",
                        scale_sig.symbol, final_quality_reject,
                    )
                    continue
                scale_order = self._signal_to_order(scale_sig)
                if scale_order is None:
                    continue
                bar = self._latest_bar(universe, scale_sig.symbol)
                if bar is None:
                    continue
                ok = allow_order(
                    scale_order, bar, self._portfolio,
                    max_position_shares=self._max_position_shares,
                    max_order_shares=self._max_order_shares,
                )
                if not ok:
                    logger.info("Risk check rejected scale-up for %s", scale_sig.symbol)
                    result.rejected_orders += 1
                    result.rejection_details.append({
                        "symbol": scale_sig.symbol,
                        "reason": "scale_up_position_risk_limit",
                    })
                    continue
                if self._execution_timer is not None:
                    result.deferred_signals.append(scale_sig)
                    result.entry_strategies[scale_sig.symbol] = "runner_readd"
                    logger.info(
                        "DEFERRED SCALE-UP %s %.0f @ %.4f → waiting for 10s re-add confirmation | %s",
                        scale_sig.symbol, scale_sig.quantity,
                        scale_sig.entry_price, scale_sig.reason,
                    )
                    continue
                fill, status = self._broker.submit(scale_order, bar, self._portfolio)
                if status is OrderStatus.FILLED and fill is not None:
                    apply_fill(self._portfolio, fill)
                    result.scale_up_fills.append(fill)
                    self._exit_manager.scale_up(
                        scale_sig.symbol, fill.quantity, fill.price, scale_sig.stop_loss,
                    )
                    logger.info(
                        "SCALE UP FILLED %s +%.0f @ %.4f | %s",
                        scale_sig.symbol, fill.quantity, fill.price, scale_sig.reason,
                    )

        # --- Step 0c: re-entry check on recently exited symbols ---
        if only_scanners is None and entries_enabled and self._reentry is not None and now is not None:
            reentry_signals = self._reentry.check_reentries(
                prices, universe, now, self._original_sizes,
            )
            for re_sig in reentry_signals:
                if self._at_position_limit():
                    break
                final_quality_reject = self._final_entry_quality_reject(
                    re_sig, universe=universe, quotes=quotes,
                    result=result, stage="reentry_final", now=now,
                )
                if final_quality_reject:
                    result.rejected_orders += 1
                    result.rejection_details.append({
                        "symbol": re_sig.symbol,
                        "reason": "final entry guard: {}".format(final_quality_reject),
                    })
                    self._scan_rejections[re_sig.symbol] = "final entry guard: {}".format(
                        final_quality_reject,
                    )
                    logger.info(
                        "FINAL ENTRY GUARD REJECT re-entry %s: %s",
                        re_sig.symbol, final_quality_reject,
                    )
                    continue
                re_order = self._signal_to_order(re_sig)
                if re_order is None:
                    continue
                bar = self._latest_bar(universe, re_sig.symbol)
                if bar is None:
                    continue
                if not allow_order(
                    re_order, bar, self._portfolio,
                    max_position_shares=self._max_position_shares,
                    max_order_shares=self._max_order_shares,
                ):
                    result.rejected_orders += 1
                    result.rejection_details.append({
                        "symbol": re_sig.symbol, "reason": "reentry_position_risk_limit",
                    })
                    continue
                if self._execution_timer is not None:
                    result.deferred_signals.append(re_sig)
                    result.entry_strategies[re_sig.symbol] = "abc_reentry"
                    logger.info(
                        "DEFERRED RE-ENTRY %s %.0f @ %.4f → waiting for 10s confirmation | %s",
                        re_sig.symbol, re_sig.quantity,
                        re_sig.entry_price, re_sig.reason,
                    )
                    continue
                fill, status = self._broker.submit(re_order, bar, self._portfolio)
                if status is OrderStatus.FILLED and fill is not None:
                    apply_fill(self._portfolio, fill)
                    result.reentry_fills.append(fill)
                    self._exit_manager.register_from_signal(
                        re_sig, now, fill_price=fill.price, fill_quantity=fill.quantity,
                    )
                    logger.info(
                        "RE-ENTRY FILLED %s %.0f @ %.4f | %s",
                        re_sig.symbol, fill.quantity, fill.price, re_sig.reason,
                    )

        # --- Step 1: classify & route (if router is configured) ---
        if self._router is not None:
            route_result = self._router.route(universe, quotes)
            result.regimes = route_result.regimes
            self._cached_regimes.update(route_result.regimes)
            for style, group in route_result.groups.items():
                result.symbols_by_style[style.value] = list(group.keys())
                cfg = self._router.get_config(style)
                if cfg is None:
                    continue

                group_syms = set(group.keys())
                style_hits: List[ScanResult] = []

                for scanner in cfg.scanners:
                    if only_scanners is not None and scanner.name not in only_scanners:
                        continue
                    # Bar-based scanning (momentum burst)
                    scanner_hits = list(self._scan_with_session_context(scanner, group, session_universe))

                    # Tick-based scanning (tape reader)
                    if ticks and hasattr(scanner, "scan_ticks"):
                        tick_group = {s: t for s, t in ticks.items() if s in group_syms}
                        if tick_group:
                            tick_hits = scanner.scan_ticks(tick_group)
                            for th in tick_hits:
                                if not th.bars and th.symbol in group:
                                    th = ScanResult(
                                        symbol=th.symbol, scanner_name=th.scanner_name,
                                        ts=th.ts, score=th.score, criteria=th.criteria,
                                        bars=list(group[th.symbol][-20:]),
                                    )
                                scanner_hits.append(th)

                    # Quote-based scanning (spread filter)
                    if quotes and hasattr(scanner, "scan_quotes"):
                        quote_group = {s: q for s, q in quotes.items() if s in group_syms}
                        if quote_group:
                            quote_hits = scanner.scan_quotes(quote_group)
                            for qh in quote_hits:
                                if not qh.bars and qh.symbol in group:
                                    qh = ScanResult(
                                        symbol=qh.symbol, scanner_name=qh.scanner_name,
                                        ts=qh.ts, score=qh.score, criteria=qh.criteria,
                                        bars=list(group[qh.symbol][-20:]),
                                    )
                                scanner_hits.append(qh)

                    style_hits.extend(scanner_hits)
                    (logger.info if scanner_hits else logger.debug)("Scanner %s (%s) found %d hits", scanner.name, style.value, len(scanner_hits))

                # Retry retained, verified structures independently of a new
                # detector hit. All verification, risk and timer paths below
                # remain shared with ordinary scanner entries.
                style_hits = self._merge_pending_hits(style_hits, pending_hits, group_syms)
                result.scan_hits.extend(style_hits)
                for hit in style_hits:
                    hit.criteria["setup_tier"] = self._setup_tier(hit) if hit.scanner_name in cfg.verifiers else "watch only"
                    cooldown_reason = (None if hit.criteria.get("_pending_recheck") else
                                       self._reject_cooldown_reason(hit, now or datetime.now(timezone.utc)))
                    if cooldown_reason is not None:
                        self._scan_rejections[hit.symbol] = cooldown_reason
                        self._append_entry_reject(result, symbol=hit.symbol, stage="cooldown", reason=cooldown_reason, hit=hit)
                        continue
                    self._log_shadow_pullback(hit, universe, quotes)
                    verifier = cfg.verifiers.get(hit.scanner_name)
                    if verifier is None:
                        reason = self._watch_only_reason(hit)
                        self._scan_rejections[hit.symbol] = reason
                        self._append_entry_reject(
                            result, symbol=hit.symbol, stage="scanner",
                            reason=reason, hit=hit,
                        )
                        self._log_shadow_missed(
                            symbol=hit.symbol,
                            reason=reason,
                            universe=universe,
                            quotes=quotes,
                            scanner=hit.scanner_name,
                            hit=hit,
                        )
                        continue
                    signal = self._verify_hit(verifier, hit, now=now,
                                              session_bars=(session_universe if session_universe is not None else universe).get(hit.symbol))
                    if signal is None:
                        reject = getattr(hit, '_reject_reason', None)
                        if reject is None:
                            reject = self._get_last_reject(hit.symbol, verifier)
                        self._scan_rejections[hit.symbol] = reject or "did not pass verifier"
                        self._record_reject_cooldown(
                            hit,
                            self._scan_rejections[hit.symbol],
                            now or datetime.now(timezone.utc),
                        )
                        self._append_entry_reject(
                            result, symbol=hit.symbol, stage="verifier",
                            reason=self._scan_rejections[hit.symbol], hit=hit,
                        )
                        self._log_shadow_missed(
                            symbol=hit.symbol,
                            reason=self._scan_rejections[hit.symbol],
                            universe=universe,
                            quotes=quotes,
                            scanner=hit.scanner_name,
                            hit=hit,
                        )
                        continue
                    if signal.action is SignalAction.SKIP:
                        result.skipped.append(signal)
                        self._append_entry_reject(
                            result, symbol=signal.symbol, stage="strategy_skip",
                            reason=signal.reason or "strategy skip",
                            signal=signal,
                            price=signal.entry_price,
                        )
                        self._log_shadow_missed(
                            symbol=signal.symbol,
                            reason=signal.reason or "strategy skip",
                            universe=universe,
                            quotes=quotes,
                            scanner=hit.scanner_name,
                            fallback_price=signal.entry_price,
                        )
                        continue
                    # Enrich signal with trend_strength from classifier (check cache too)
                    regime = result.regimes.get(hit.symbol) or self._cached_regimes.get(hit.symbol)
                    if regime is not None and regime.trend_strength != signal.trend_strength:
                        signal = TradeSignal(
                            symbol=signal.symbol, action=signal.action,
                            quantity=signal.quantity, entry_price=signal.entry_price,
                            stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                            trailing_stop_offset=signal.trailing_stop_offset,
                            max_hold_seconds=signal.max_hold_seconds,
                            reason=signal.reason, scan_result=signal.scan_result,
                            trend_strength=regime.trend_strength,
                        )
                    self._scan_rejections.pop(hit.symbol, None)
                    self._clear_reject_cooldown(hit)
                    result.signals.append(signal)
                    self._append_entry_decision(
                        result,
                        self._entry_policy.decision(
                            symbol=signal.symbol,
                            stage="verifier",
                            passed=True,
                            signal=signal,
                            price=signal.entry_price,
                        ),
                    )
        else:
            # no router → run all scanners against full universe (legacy mode)
            for scanner in self._scanners:
                if only_scanners is not None and scanner.name not in only_scanners:
                    continue
                hits = self._scan_with_session_context(scanner, universe, session_universe)
                result.scan_hits.extend(hits)
                (logger.info if hits else logger.debug)("Scanner %s found %d hits", scanner.name, len(hits))
            result.scan_hits = self._merge_pending_hits(result.scan_hits, pending_hits, set(universe))
            for hit in result.scan_hits:
                hit.criteria["setup_tier"] = self._setup_tier(hit) if hit.scanner_name in self._verifiers else "watch only"
                cooldown_reason = (None if hit.criteria.get("_pending_recheck") else
                                   self._reject_cooldown_reason(hit, now or datetime.now(timezone.utc)))
                if cooldown_reason is not None:
                    self._scan_rejections[hit.symbol] = cooldown_reason
                    self._append_entry_reject(result, symbol=hit.symbol, stage="cooldown", reason=cooldown_reason, hit=hit)
                    continue
                self._log_shadow_pullback(hit, universe, quotes)
                verifier = self._verifiers.get(hit.scanner_name)
                if verifier is None:
                    reason = self._watch_only_reason(hit)
                    self._scan_rejections[hit.symbol] = reason
                    self._append_entry_reject(
                        result, symbol=hit.symbol, stage="scanner",
                        reason=reason, hit=hit,
                    )
                    self._log_shadow_missed(
                        symbol=hit.symbol,
                        reason=reason,
                        universe=universe,
                        quotes=quotes,
                        scanner=hit.scanner_name,
                        hit=hit,
                    )
                    continue
                signal = self._verify_hit(verifier, hit, now=now,
                                          session_bars=(session_universe if session_universe is not None else universe).get(hit.symbol))
                if signal is None:
                    reject = self._get_last_reject(hit.symbol, verifier)
                    self._scan_rejections[hit.symbol] = reject or "did not pass verifier"
                    self._record_reject_cooldown(
                        hit,
                        self._scan_rejections[hit.symbol],
                        now or datetime.now(timezone.utc),
                    )
                    self._append_entry_reject(
                        result, symbol=hit.symbol, stage="verifier",
                        reason=self._scan_rejections[hit.symbol], hit=hit,
                    )
                    self._log_shadow_missed(
                        symbol=hit.symbol,
                        reason=self._scan_rejections[hit.symbol],
                        universe=universe,
                        quotes=quotes,
                        scanner=hit.scanner_name,
                        hit=hit,
                    )
                    continue
                if signal.action is SignalAction.SKIP:
                    result.skipped.append(signal)
                    self._append_entry_reject(
                        result, symbol=signal.symbol, stage="strategy_skip",
                        reason=signal.reason or "strategy skip",
                        signal=signal,
                        price=signal.entry_price,
                    )
                    self._log_shadow_missed(
                        symbol=signal.symbol,
                        reason=signal.reason or "strategy skip",
                        universe=universe,
                        quotes=quotes,
                        scanner=hit.scanner_name,
                        fallback_price=signal.entry_price,
                    )
                    continue
                self._scan_rejections.pop(hit.symbol, None)
                self._clear_reject_cooldown(hit)
                result.signals.append(signal)
                self._append_entry_decision(
                    result,
                    self._entry_policy.decision(
                        symbol=signal.symbol,
                        stage="verifier",
                        passed=True,
                        signal=signal,
                        price=signal.entry_price,
                    ),
                )

        # Observation cycles still classify/scan/verify and manage exits, but
        # never submit fresh entries, re-adds, re-entries or deferred signals.
        if not entries_enabled:
            for signal in result.signals:
                self._append_entry_reject(
                    result, symbol=signal.symbol, stage="observation_only",
                    reason="global entry block: {}".format(entry_blocker or ("scan_only" if self._scan_only else "entries_disabled")),
                    signal=signal,
                )
            self._flush_pending_events(result)
            return result

        # --- Step 3 & 4: risk check + execute entries ---
        # Pre-score all entry signals by news sentiment so we prioritize
        # stocks with positive catalysts when the position limit is tight.
        signal_news: dict = {}  # symbol -> (score, headlines)
        if self._news_checker is not None:
            for signal in result.signals:
                if signal.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT):
                    if signal.symbol not in signal_news:
                        try:
                            score, headlines = self._news_checker.get_sentiment(signal.symbol)
                            signal_news[signal.symbol] = (score, headlines)
                        except Exception:
                            signal_news[signal.symbol] = (0.0, [])

        # Sort signals: positive news first, then neutral, then no news
        def _signal_priority(sig: TradeSignal) -> float:
            ns = signal_news.get(sig.symbol, (0.0, []))
            return -ns[0]  # negative so higher scores sort first

        sorted_signals = sorted(result.signals, key=_signal_priority)

        entered_this_cycle: set = set()
        for signal in sorted_signals:
            # Only one entry per symbol per cycle
            if signal.symbol in entered_this_cycle:
                continue

            # Skip if already holding a position in this symbol
            pos = self._portfolio.positions.get(signal.symbol)
            if pos and not pos.is_flat and signal.action in (
                SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT,
            ):
                continue

            # Cooldown: don't re-enter a stock too soon after exiting
            if signal.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT):
                if (
                    self._require_hod_alert_for_entry
                    and self._hod_active_checker is not None
                    and not self._hod_active_checker(signal.symbol)
                ):
                    if (
                        self._hod_entry_bypass_checker is not None
                        and self._hod_entry_bypass_checker(signal)
                    ):
                        logger.info(
                            "HOD GATE BYPASS %s: structured hot-watch setup",
                            signal.symbol,
                        )
                    else:
                        logger.info(
                            "HOD GATE %s: not on HOD momentum board — skip entry",
                            signal.symbol,
                        )
                        result.skipped.append(signal)
                        self._scan_rejections[signal.symbol] = (
                            "not on HOD momentum alert board"
                        )
                        self._append_entry_reject(result, symbol=signal.symbol, stage="eligibility",
                                                  reason="not on HOD momentum alert board", signal=signal)
                        self._log_shadow_missed(
                            symbol=signal.symbol,
                            reason="not on HOD momentum alert board",
                            universe=universe,
                            quotes=quotes,
                            scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                            fallback_price=signal.entry_price,
                        )
                        continue

                # Circuit breaker: stop all new entries after max daily loss
                if self._circuit_breaker_tripped:
                    logger.warning(
                        "CIRCUIT BREAKER BLOCK %s: daily P&L $%.2f — no new trades",
                        signal.symbol, self._daily_pnl,
                    )
                    result.skipped.append(signal)
                    self._append_entry_reject(result, symbol=signal.symbol, stage="eligibility",
                                              reason="circuit breaker", signal=signal)
                    self._log_shadow_missed(
                        symbol=signal.symbol,
                        reason="circuit breaker",
                        universe=universe,
                        quotes=quotes,
                        scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                        fallback_price=signal.entry_price,
                    )
                    continue

                # Daily loser blacklist: don't re-enter a stock that lost today
                if self._enable_daily_loser_blacklist and signal.symbol in self._daily_losers:
                    logger.info(
                        "BLACKLISTED %s: already lost money today — no re-entry",
                        signal.symbol,
                    )
                    result.skipped.append(signal)
                    self._append_entry_reject(result, symbol=signal.symbol, stage="eligibility",
                                              reason="daily loser blacklist", signal=signal)
                    self._log_shadow_missed(
                        symbol=signal.symbol,
                        reason="daily loser blacklist",
                        universe=universe,
                        quotes=quotes,
                        scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                        fallback_price=signal.entry_price,
                    )
                    continue

                last_exit_ts = self._exit_cooldowns.get(signal.symbol)
                if last_exit_ts is not None:
                    elapsed = ((now or datetime.now(timezone.utc)) - last_exit_ts).total_seconds()
                    if elapsed < self._cooldown_seconds:
                        remaining = self._cooldown_seconds - elapsed
                        logger.info(
                            "COOLDOWN %s: exited %.0fs ago, wait %.0fs more",
                            signal.symbol, elapsed, remaining,
                        )
                        result.skipped.append(signal)
                        self._append_entry_reject(result, symbol=signal.symbol, stage="eligibility",
                                                  reason="cooldown", signal=signal)
                        self._log_shadow_missed(
                            symbol=signal.symbol,
                            reason="cooldown",
                            universe=universe,
                            quotes=quotes,
                            scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                            fallback_price=signal.entry_price,
                        )
                        continue

                # Per-symbol max entries: prevent over-trading one stock
                if self._symbol_entry_counts.get(signal.symbol, 0) >= self._max_entries_per_symbol:
                    logger.info(
                        "MAX ENTRIES %s: already entered %d times today — skip",
                        signal.symbol, self._max_entries_per_symbol,
                    )
                    result.skipped.append(signal)
                    self._append_entry_reject(result, symbol=signal.symbol, stage="eligibility",
                                              reason="max entries", signal=signal)
                    self._log_shadow_missed(
                        symbol=signal.symbol,
                        reason="max entries",
                        universe=universe,
                        quotes=quotes,
                        scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                        fallback_price=signal.entry_price,
                    )
                    continue

            if self._at_position_limit():
                logger.warning("Position limit %d reached, skipping %s", self._max_positions, signal.symbol)
                result.skipped.append(signal)
                self._log_shadow_missed(
                    symbol=signal.symbol,
                    reason="position limit",
                    universe=universe,
                    quotes=quotes,
                    scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                    fallback_price=signal.entry_price,
                )
                continue

            # News sentiment gate: block negative, boost positive
            news_score = 0.0
            if signal.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT):
                ns = signal_news.get(signal.symbol, (0.0, []))
                news_score, headlines = ns[0], ns[1]

                if news_score <= -0.3:
                    logger.info(
                        "NEWS BLOCK %s: sentiment=%.2f — %s",
                        signal.symbol, news_score,
                        headlines[0] if headlines else "negative news",
                    )
                    result.skipped.append(signal)
                    self._log_shadow_missed(
                        symbol=signal.symbol,
                        reason="negative news",
                        universe=universe,
                        quotes=quotes,
                        scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                        fallback_price=signal.entry_price,
                    )
                    continue

                if news_score >= 0.3:
                    logger.info(
                        "NEWS PRIORITY %s: sentiment=+%.2f — %s",
                        signal.symbol, news_score,
                        headlines[0] if headlines else "positive news",
                    )
                elif headlines:
                    logger.debug(
                        "NEWS NEUTRAL %s: sentiment=%.2f — %s",
                        signal.symbol, news_score, headlines[0],
	                    )

            if signal.action in (SignalAction.ENTER_LONG, SignalAction.REENTER_LONG):
                signal = self._maybe_apply_level_capped_entry(signal, universe)

            will_defer_to_timer = (
                self._execution_timer is not None
                and signal.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT)
            )

            if (
                signal.action in (
                    SignalAction.ENTER_LONG,
                    SignalAction.REENTER_LONG,
                    SignalAction.SCALE_UP_LONG,
                )
                and not will_defer_to_timer
            ):
                final_quality_reject = self._final_entry_quality_reject(
                    signal, universe=universe, quotes=quotes,
                    result=result, stage="final_entry_guard", now=now,
                )
                if final_quality_reject:
                    logger.info(
                        "FINAL ENTRY GUARD REJECT %s: %s",
                        signal.symbol, final_quality_reject,
                    )
                    result.rejected_orders += 1
                    result.rejection_details.append({
                        "symbol": signal.symbol,
                        "reason": "final entry guard: {}".format(final_quality_reject),
                    })
                    self._scan_rejections[signal.symbol] = "final entry guard: {}".format(
                        final_quality_reject,
                    )
                    self._log_shadow_missed(
                        symbol=signal.symbol,
                        reason="final entry guard: {}".format(final_quality_reject),
                        universe=universe,
                        quotes=quotes,
                        scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                        fallback_price=signal.entry_price,
                    )
                    continue

                chase_reject = self._normal_entry_chase_reject(
                    signal, universe=universe, now=now,
                )
                if chase_reject:
                    logger.info("ENTRY CHASE REJECT %s: %s", signal.symbol, chase_reject)
                    result.rejected_orders += 1
                    result.rejection_details.append({
                        "symbol": signal.symbol,
                        "reason": chase_reject,
                    })
                    self._scan_rejections[signal.symbol] = chase_reject
                    self._append_entry_reject(
                        result,
                        symbol=signal.symbol,
                        stage="entry_chase_guard",
                        reason=chase_reject,
                        signal=signal,
                        price=self._latest_price_for_signal(signal, universe),
                    )
                    self._log_shadow_missed(
                        symbol=signal.symbol,
                        reason=chase_reject,
                        universe=universe,
                        quotes=quotes,
                        scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                        fallback_price=self._latest_price_for_signal(signal, universe),
                    )
                    continue

            if will_defer_to_timer:
                final_quality_reject = self._final_entry_quality_reject(
                    signal, universe=universe, quotes=quotes,
                    result=result, stage="final_entry_guard", now=now,
                )
                if final_quality_reject:
                    logger.info(
                        "FINAL ENTRY GUARD REJECT %s: %s",
                        signal.symbol, final_quality_reject,
                    )
                    result.rejected_orders += 1
                    result.rejection_details.append({
                        "symbol": signal.symbol,
                        "reason": "final entry guard: {}".format(final_quality_reject),
                    })
                    self._scan_rejections[signal.symbol] = "final entry guard: {}".format(
                        final_quality_reject,
                    )
                    self._log_shadow_missed(
                        symbol=signal.symbol,
                        reason="final entry guard: {}".format(final_quality_reject),
                        universe=universe,
                        quotes=quotes,
                        scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                        fallback_price=signal.entry_price,
                    )
                    continue

            # Risk-reward gate: reject signals worse than 1:2
            # Relax to 1:1.5 for stocks with strong positive news catalysts
            if signal.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT):
                if signal.stop_loss is not None and signal.take_profit is not None:
                    risk = abs(signal.entry_price - signal.stop_loss)
                    reward = abs(signal.take_profit - signal.entry_price)
                    min_rr = 1.4 if news_score >= 0.5 else 1.8
                    if risk > 0 and reward / risk < min_rr:
                        logger.info(
                            "R:R REJECT %s: risk=$%.4f reward=$%.4f ratio=1:%.1f (min 1:%.1f)",
                            signal.symbol, risk, reward, reward / risk, min_rr,
                        )
                        result.rejected_orders += 1
                        result.rejection_details.append({"symbol": signal.symbol, "reason": "R:R {:.1f} < {:.1f}".format(reward / risk, min_rr)})
                        self._append_entry_reject(
                            result,
                            symbol=signal.symbol,
                            stage="risk_reward",
                            reason="R:R {:.1f} < {:.1f}".format(reward / risk, min_rr),
                            signal=signal,
                            price=signal.entry_price,
                            metadata={"risk": risk, "reward": reward, "min_rr": min_rr},
                        )
                        self._log_shadow_missed(
                            symbol=signal.symbol,
                            reason="R:R {:.1f} < {:.1f}".format(reward / risk, min_rr),
                            universe=universe,
                            quotes=quotes,
                            scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                            fallback_price=signal.entry_price,
                        )
                        continue

                    max_dollar_risk = self._max_dollar_risk_per_trade
                    if risk > 0:
                        dollar_risk = risk * signal.quantity
                        if dollar_risk > max_dollar_risk:
                            safe_qty = int(max_dollar_risk / risk)
                            if safe_qty < 10:
                                logger.info(
                                    "RISK CAP REJECT %s: $%.2f risk too high even at min size",
                                    signal.symbol, dollar_risk,
                                )
                                result.rejected_orders += 1
                                result.rejection_details.append({"symbol": signal.symbol, "reason": "risk_cap ${:.2f}".format(dollar_risk)})
                                self._append_entry_reject(
                                    result,
                                    symbol=signal.symbol,
                                    stage="risk_cap",
                                    reason="risk cap",
                                    signal=signal,
                                    price=signal.entry_price,
                                    metadata={"dollar_risk": dollar_risk, "max_dollar_risk": max_dollar_risk},
                                )
                                self._log_shadow_missed(
                                    symbol=signal.symbol,
                                    reason="risk cap",
                                    universe=universe,
                                    quotes=quotes,
                                    scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                                    fallback_price=signal.entry_price,
                                )
                                continue
                            logger.info(
                                "RISK CAP %s: %.0f → %d shares ($%.2f → $%.2f risk)",
                                signal.symbol, signal.quantity, safe_qty,
                                dollar_risk, risk * safe_qty,
                            )
                            signal = replace(signal, quantity=float(safe_qty))

                    # Also cap to max_order_shares
                    if signal.quantity > self._max_order_shares:
                        logger.info(
                            "SIZE CAP %s: %.0f → %d shares (max %d)",
                            signal.symbol, signal.quantity, int(self._max_order_shares), int(self._max_order_shares),
                        )
                        signal = replace(signal, quantity=float(int(self._max_order_shares)))

            order = self._signal_to_order(signal)
            if order is None:
                continue

            bar = self._latest_bar(universe, signal.symbol)
            if bar is None:
                logger.warning("No bar data for %s, cannot execute", signal.symbol)
                result.rejected_orders += 1
                result.rejection_details.append({"symbol": signal.symbol, "reason": "no_bar_data"})
                self._append_entry_reject(
                    result,
                    symbol=signal.symbol,
                    stage="order_context",
                    reason="no bar data",
                    signal=signal,
                    price=signal.entry_price,
                )
                self._log_shadow_missed(
                    symbol=signal.symbol,
                    reason="no bar data",
                    universe=universe,
                    quotes=quotes,
                    scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                    fallback_price=signal.entry_price,
                )
                continue

            ok = allow_order(
                order, bar, self._portfolio,
                max_position_shares=self._max_position_shares,
                max_order_shares=self._max_order_shares,
            )
            if not ok:
                logger.info("Risk check rejected %s for %s", signal.action.value, signal.symbol)
                result.rejected_orders += 1
                result.rejection_details.append({"symbol": signal.symbol, "reason": "position_risk_limit"})
                self._append_entry_reject(
                    result,
                    symbol=signal.symbol,
                    stage="risk_manager",
                    reason="position_risk_limit",
                    signal=signal,
                    price=signal.entry_price,
                )
                self._log_shadow_missed(
                    symbol=signal.symbol,
                    reason="position_risk_limit",
                    universe=universe,
                    quotes=quotes,
                    scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                    fallback_price=signal.entry_price,
                )
                continue

            # Advanced guards: false breakout, liquidity trap, halt, market panic, spread
            if signal.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT):
                sym_bars = universe.get(signal.symbol)
                sym_quotes = quotes.get(signal.symbol) if quotes else None
                guard_context = {}
                guard_metadata = {}
                if signal.scan_result and signal.scan_result.scanner_name == "micro_pullback_reclaim":
                    window = signal.scan_result.confirmation_window
                    guard_context = {"breakout_context": window, "now": self._market_now}
                    guard_metadata = {
                        "breakout_timeframe": "10s",
                        "quality_timeframe": "1m/session",
                        "confirmation_context": "original_completed_trigger" if window else "missing",
                    }
                    if window and len(window.bars) >= 6:
                        last_six = window.bars[-6:]
                        prior_volume = sum(x.bar.volume for x in last_six[:-1]) / 5
                        guard_metadata.update({
                            "confirmation_bar_start": last_six[-1].bar.ts.isoformat(),
                            "confirmation_closed_at": last_six[-1].closed_at.isoformat(),
                            "confirmation_volume_ratio": last_six[-1].bar.volume / prior_volume if prior_volume > 0 else None,
                        })
                if signal.scan_result and signal.scan_result.scanner_name == "prepared_reclaim_tick":
                    guard_context = {"event_context": signal.scan_result.criteria.get("_event_evidence"), "now": self._market_now}
                    guard_metadata = {"breakout_timeframe": "10s event-aligned", "quality_timeframe": "1m/session", "confirmation_context": "trade_quote_event"}
                guard_ok, guard_reason = self._trade_guard.check_entry(
                    signal, bars=sym_bars, quotes=sym_quotes, **guard_context,
                )
                if (
                    not guard_ok
                    and self._allow_vwap_reclaim_scout_trade_guard_exception(
                        signal, bars=sym_bars, quotes=sym_quotes, reason=guard_reason,
                    )
                ):
                    guard_ok = True
                    logger.info(
                        "GUARD PASS %s: VWAP reclaim scout liquidity exception: %s",
                        signal.symbol, guard_reason,
                    )
                if not guard_ok:
                    logger.info("GUARD REJECT %s: %s", signal.symbol, guard_reason)
                    result.rejected_orders += 1
                    post_guard_reason = "post-guard: {}".format(guard_reason)
                    result.rejection_details.append({"symbol": signal.symbol, "reason": post_guard_reason})
                    self._scan_rejections[signal.symbol] = post_guard_reason
                    self._append_entry_reject(
                        result,
                        symbol=signal.symbol,
                        stage="trade_guard",
                        reason=post_guard_reason,
                        signal=signal,
                        price=signal.entry_price,
                        metadata=guard_metadata,
                    )
                    self._log_shadow_missed(
                        symbol=signal.symbol,
                        reason=post_guard_reason,
                        universe=universe,
                        quotes=quotes,
                        scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                        fallback_price=signal.entry_price,
                    )
                    continue

                if guard_metadata:
                    self._append_entry_decision(result, self._entry_policy.decision(
                        symbol=signal.symbol, stage="trade_guard", passed=True,
                        signal=signal, price=signal.entry_price, metadata=guard_metadata,
                    ))

            # Defer to execution timer if enabled, otherwise submit immediately
            if (self._execution_timer is not None
                    and signal.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT)):
                result.deferred_signals.append(signal)
                self.pending_entries.finish(signal.symbol, "confirming", "entry gates passed; awaiting existing 10s confirmation")
                result.entry_strategies[signal.symbol] = _entry_strategy_label(signal)
                entered_this_cycle.add(signal.symbol)
                logger.info(
                    "DEFERRED %s %s %.0f @ %.4f → waiting for 10s micro-entry | %s",
                    signal.action.value, signal.symbol,
                    signal.quantity, signal.entry_price, signal.reason,
                )
                self._append_entry_decision(
                    result,
                    self._entry_policy.decision(
                        symbol=signal.symbol,
                        stage="deferred_to_timer",
                        passed=True,
                        signal=signal,
                        price=signal.entry_price,
                    ),
                )
                continue

            from daytrading.execution.origin import submit_signal
            fill, status = submit_signal(self._broker, order, bar, self._portfolio, signal)
            self.pending_entries.finish(signal.symbol, "submitted", "handed to existing order path")
            self._log_shadow_execution(
                order=order, bar=bar, status=status, fill=fill, source="pipeline_entry",
            )
            if status is OrderStatus.FILLED and fill is not None:
                apply_fill(self._portfolio, fill)
                result.fills.append(fill)
                self._symbol_entry_counts[signal.symbol] = self._symbol_entry_counts.get(signal.symbol, 0) + 1
                result.entry_strategies[signal.symbol] = _entry_strategy_label(signal)
                entered_this_cycle.add(signal.symbol)
                logger.info(
                    "FILLED %s %s %.0f @ %.4f | %s",
                    signal.action.value, signal.symbol,
                    fill.quantity, fill.price, signal.reason,
                )
                if now is not None:
                    self._exit_manager.register_from_signal(
                        signal, now, fill_price=fill.price, fill_quantity=fill.quantity,
                    )
                    self._original_sizes[signal.symbol] = fill.quantity
                self._append_entry_decision(
                    result,
                    self._entry_policy.decision(
                        symbol=signal.symbol,
                        stage="order",
                        passed=True,
                        signal=signal,
                        price=fill.price,
                        metadata={"status": status.value},
                    ),
                )
            else:
                logger.info("Order %s for %s: %s", signal.action.value, signal.symbol, status.value)
                result.rejected_orders += 1
                reason = "order_{}".format(status.value)
                result.rejection_details.append({"symbol": signal.symbol, "reason": reason})
                self._append_entry_reject(
                    result,
                    symbol=signal.symbol,
                    stage="order",
                    reason=reason,
                    signal=signal,
                    price=signal.entry_price,
                    metadata={"status": status.value},
                )
                self._log_shadow_missed(
                    symbol=signal.symbol,
                    reason=reason,
                    universe=universe,
                    quotes=quotes,
                    scanner=signal.scan_result.scanner_name if signal.scan_result else "",
                    fallback_price=signal.entry_price,
                )

        self._flush_pending_events(result)
        return result

    def _final_entry_quality_reject(
        self,
        signal: TradeSignal,
        *,
        universe: Dict[str, Sequence[Bar]],
        quotes: Optional[Dict[str, Sequence[Quote]]] = None,
        result: Optional[PipelineResult] = None,
        stage: str = "final_entry_guard",
        now: Optional[datetime] = None,
    ) -> Optional[str]:
        """Last shared rule/ML gate before any long entry/add/re-entry can order."""
        if signal.action not in (
            SignalAction.ENTER_LONG,
            SignalAction.REENTER_LONG,
            SignalAction.SCALE_UP_LONG,
        ):
            return None
        # Risk-profile gate: a stop more than max_entry_risk_pct below entry is
        # too wide for a scalp/momentum entry — one fail erases many wins.
        cap = float(getattr(self, "_max_entry_risk_pct", 0.0) or 0.0)
        entry_px = float(signal.entry_price or 0.0)
        stop_px = float(signal.stop_loss or 0.0)
        if cap > 0 and entry_px > 0 and 0 < stop_px < entry_px:
            risk_pct = (entry_px - stop_px) / entry_px
            # A stop calculated as entry * (1 - cap) can round just above
            # the cap (GENK: 0.06000000000000008). Allow only floating-point
            # noise; a genuinely wider stop must still fail.
            if risk_pct > cap + 1e-12:
                reason = "stop too wide: {:.1f}% risk > {:.1f}% cap".format(
                    risk_pct * 100.0, cap * 100.0,
                )
                if result is not None:
                    self._append_entry_reject(
                        result,
                        symbol=signal.symbol,
                        stage=stage,
                        reason=reason,
                        signal=signal,
                        price=entry_px,
                    )
                return reason
        sym_bars = universe.get(signal.symbol) or []
        sym_quotes = quotes.get(signal.symbol) if quotes else None
        # The verifier uses this same available trend context. Dropping it
        # here can turn a valid setup into a spurious lower-score rejection
        # (SPRC: 84 with context, 79 with the no-data fallback).
        bars_5m = (
            self._bar_aggregator.get_5m_bars(signal.symbol)
            if self._bar_aggregator is not None else None
        )
        decision = self._entry_policy.evaluate(
            signal,
            bars=sym_bars,
            quotes=sym_quotes,
            ticks=getattr(self, "_cycle_ticks", {}).get(signal.symbol),
            bars_5m=bars_5m,
            stage=stage,
            min_day_change_pct=0.0,
            now=(getattr(self,'_market_now',None) or now) if signal.scan_result and signal.scan_result.scanner_name in ('micro_pullback_reclaim', 'prepared_reclaim_tick') else now,
        )
        reject_reason = decision.reject_reason
        self._micro_vwap_decision(signal.scan_result, reject_reason, getattr(self, "_market_now", None) or now)
        if reject_reason:
            self.pending_entries.reject(signal.scan_result, reject_reason, now or datetime.now(timezone.utc))
        if reject_reason and self._allow_warrior_starter_guard_exception(
            signal,
            bars=sym_bars,
            reason=reject_reason,
        ):
            if result is not None:
                result.entry_decisions.append(EntryDecision(
                    symbol=decision.symbol,
                    stage=decision.stage,
                    passed=True,
                    action=decision.action,
                    pattern=decision.pattern,
                    scanner=decision.scanner,
                    setup_tier=decision.setup_tier,
                    entry_tier=decision.entry_tier,
                    price=decision.price,
                    metadata={
                        **dict(decision.metadata),
                        "cleared_reason": reject_reason,
                        "guard_exception": "warrior_squeeze_starter",
                    },
                    ts=decision.ts,
                ).to_payload())
            return None
        if result is not None:
            self._append_entry_decision(result, decision)
        return reject_reason

    def _normal_entry_chase_reject(
        self,
        signal: TradeSignal,
        *,
        universe: Dict[str, Sequence[Bar]],
        now: Optional[datetime] = None,
    ) -> Optional[str]:
        """Block normal entries that arrive far above the setup base.

        Timed entries already have a base-pinned chase guard in the runner. This
        covers the direct pipeline path before a signal can be ordered or queued
        to the timer.
        """
        if signal.action not in (
            SignalAction.ENTER_LONG,
            SignalAction.REENTER_LONG,
            SignalAction.SCALE_UP_LONG,
        ):
            return None
        live = (
            float(signal.entry_price or 0.0)
            if self._is_level_capped_entry_signal(signal)
            else self._latest_price_for_signal(signal, universe)
        )
        if live <= 0:
            return None

        # Own-base anchor first: it both gates the primary chase check below and
        # tells the memory whether the setup base has moved up off a stale level.
        anchor = self._signal_chase_anchor(signal)
        missed_reject = self._missed_a_plus.chase_reject(
            symbol=signal.symbol,
            price=live,
            now=now,
            signal=signal,
            max_age_seconds=self._missed_a_plus_chase_window_sec,
            max_chase_pct_sub5=self._missed_a_plus_chase_pct_sub5,
            max_chase_pct_5plus=self._missed_a_plus_chase_pct_5plus,
            fresh_base_anchor=(anchor if self._missed_a_plus_fresh_base_reset else 0.0),
            fresh_base_reset_pct=self._missed_a_plus_fresh_base_pct,
        )
        if missed_reject:
            return missed_reject

        if anchor <= 0:
            return None
        # Price-tiered: cheap fast movers cover ground quickly between signal and
        # fill, so they get more room; pricier names stay tight. Config-driven via
        # StrategyConfig.entry_chase_* (was hardcoded 0.025/0.035 with a $5 tier,
        # which rejected every entry on sub-$10 runners like CUPR).
        max_chase_pct = (
            self._entry_chase_pct_high
            if anchor >= self._entry_chase_price_tier
            else self._entry_chase_pct_low
        )
        if live <= anchor * (1.0 + max_chase_pct):
            return None
        return (
            "late chase: ${:.4f} is {:.1f}% above setup base ${:.4f} "
            "(max {:.1f}%)"
        ).format(
            live,
            (live - anchor) / anchor * 100.0,
            anchor,
            max_chase_pct * 100.0,
        )

    @staticmethod
    def _is_level_capped_entry_signal(signal: TradeSignal) -> bool:
        hit = signal.scan_result
        criteria = hit.criteria if hit is not None else {}
        return str(criteria.get("entry_mode") or "") == "level_capped_scout"

    def _maybe_apply_level_capped_entry(
        self,
        signal: TradeSignal,
        universe: Dict[str, Sequence[Bar]],
    ) -> TradeSignal:
        """Backtest experiment: model a stop-limit scout near the breakout level.

        The normal 1m pipeline confirms at bar close, which is too late on
        vertical candles. This opt-in path only rewrites the signal when the
        current 1m bar actually traded through a tight cap above the setup base.
        Live/timed entries stay unchanged unless this explicit flag is enabled.
        """
        if not self._level_capped_entry_enabled:
            return signal
        if signal.action not in (SignalAction.ENTER_LONG, SignalAction.REENTER_LONG):
            return signal
        hit = signal.scan_result
        if hit is None:
            return signal
        criteria = hit.criteria or {}
        pattern = str(criteria.get("pattern") or hit.scanner_name or "")
        entry_tier = str(criteria.get("entry_tier") or "")
        entry_mode = str(criteria.get("entry_mode") or "")
        if pattern not in (
            "shallow_stair_continuation",
            "level_breakout_reclaim",
            "level_breakout_watch",
        ) and entry_tier != "stair_scout" and entry_mode != "level_breakout_scout":
            return signal
        try:
            base = float(
                criteria.get("breakout_level")
                or criteria.get("base_high")
                or criteria.get("trigger_price")
                or 0.0
            )
        except (TypeError, ValueError):
            base = 0.0
        if base <= 0:
            return signal
        bar = self._latest_bar(universe, signal.symbol)
        if bar is None:
            return signal
        cap = round(base * 1.01, 4)
        if signal.entry_price <= cap:
            return signal
        if not (float(bar.low) <= cap <= float(bar.high)):
            return signal
        try:
            stop = float(signal.stop_loss or criteria.get("stop_price") or 0.0)
        except (TypeError, ValueError):
            stop = 0.0
        if stop <= 0 or stop >= cap:
            return signal
        target = signal.take_profit
        if target is None or target <= cap:
            risk = cap - stop
            target = cap + max(risk * 1.8, cap * 0.015)
        criteria["entry_mode"] = "level_capped_scout"
        criteria["level_capped_entry_price"] = cap
        criteria["uncapped_entry_price"] = round(float(signal.entry_price or 0.0), 4)
        criteria["level_capped_note"] = "backtest stop-limit cap at breakout level"
        return TradeSignal(
            symbol=signal.symbol,
            action=signal.action,
            quantity=signal.quantity,
            entry_price=cap,
            stop_loss=signal.stop_loss,
            take_profit=target,
            trailing_stop_offset=signal.trailing_stop_offset,
            max_hold_seconds=signal.max_hold_seconds,
            reason="{} (level capped scout {:.4f})".format(signal.reason, cap),
            scan_result=signal.scan_result,
            trend_strength=signal.trend_strength,
        )

    @staticmethod
    def _signal_pattern(signal: TradeSignal) -> str:
        hit = signal.scan_result
        if hit is None:
            return ""
        return str(hit.criteria.get("pattern") or hit.scanner_name or "")

    @staticmethod
    def _signal_chase_anchor(signal: TradeSignal) -> float:
        hit = signal.scan_result
        criteria = hit.criteria if hit is not None else {}
        for key in (
            "setup_anchor",
            "breakout_level",
            "base_high",
            "resistance",
            "trigger_price",
            "queued_entry_price",
            "orb_high",
        ):
            try:
                value = float(criteria.get(key) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value
        return 0.0

    @staticmethod
    def _latest_price_for_signal(
        signal: TradeSignal,
        universe: Dict[str, Sequence[Bar]],
    ) -> float:
        bars = universe.get(signal.symbol) or []
        if bars:
            try:
                return float(bars[-1].close or signal.entry_price or 0.0)
            except (TypeError, ValueError):
                pass
        try:
            return float(signal.entry_price or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _at_position_limit(self) -> bool:
        open_positions = sum(1 for p in self._portfolio.positions.values() if not p.is_flat)
        return open_positions >= self._max_positions

    def _scan_with_session_context(self, scanner, universe, session_universe=None):
        kwargs={}
        if getattr(scanner,'accepts_cycle_time',False) is True:kwargs['now']=self._market_now
        if getattr(scanner,'accepts_session_context',False) is True:
            kwargs['session_universe']=session_universe if session_universe is not None else universe
        return scanner.scan(universe,**kwargs)

    def _verify_hit(
        self,
        verifier: StrategyVerifier,
        hit: ScanResult,
        *,
        now: Optional[datetime] = None,
        session_bars: Optional[Sequence[Bar]] = None,
    ) -> Optional[TradeSignal]:
        """Pass replay time and complete session context when supported.

        Legacy tests and simple verifiers may still implement the old
        two-argument signature. Preserve the scanner's local pattern bars;
        session-aware verifiers receive the complete universe separately.
        """
        if hit.scanner_name == "prepared_reclaim_tick" and self._execution_timer is None:
            verifier._last_reject = "prepared reclaim requires protected runner dispatch"
            return None
        try:
            signature = inspect.signature(verifier.verify)
            params = signature.parameters
            supports_now = (
                "now" in params
                or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            )
        except (TypeError, ValueError):
            params = {}
            supports_now = True
        kwargs = {}
        if supports_now:
            kwargs["now"] = ((getattr(self,'_market_now',None) or now)
                              if hit.scanner_name in ('micro_pullback_reclaim', 'prepared_reclaim_tick') else now)
        if "session_bars" in params:
            kwargs["session_bars"] = session_bars
        signal = verifier.verify(hit, self._portfolio, **kwargs)
        blocker = getattr(self, '_entry_global_blocker', None)
        if signal is not None and blocker in TEMPORARY_ENTRY_BLOCKS:
            scanner = next((s for s in self._scanners if s.name == hit.scanner_name), None)
            book = getattr(scanner, 'vwap_retries', None)
            if book is not None:
                book.hold_confirmed(hit, blocker, kwargs.get('now', now))
        else:
            self._micro_vwap_decision(hit, getattr(verifier, "_last_reject", None) if signal is None else None, kwargs.get("now", now))
        return signal

    def _micro_vwap_decision(self, hit, reason, now):
        if hit is None or hit.scanner_name not in ("micro_pullback_reclaim", "prepared_reclaim_tick"):
            return
        scanner = next((s for s in self._scanners if s.name == hit.scanner_name), None)
        book = getattr(scanner, "vwap_retries", None)
        if book is not None:
            book.decide(hit, reason, now)

    @staticmethod
    def _signal_to_order(signal: TradeSignal) -> Optional[Order]:
        buy_actions = (
            SignalAction.ENTER_LONG, SignalAction.EXIT_SHORT,
            SignalAction.SCALE_UP_LONG, SignalAction.REENTER_LONG,
        )
        sell_actions = (
            SignalAction.ENTER_SHORT, SignalAction.EXIT_LONG,
            SignalAction.SCALE_UP_SHORT, SignalAction.REENTER_SHORT,
        )
        if signal.action in buy_actions:
            return Order(
                symbol=signal.symbol,
                side=Side.BUY,
                quantity=signal.quantity,
                limit_price=signal.entry_price,
            )
        if signal.action in sell_actions:
            return Order(
                symbol=signal.symbol,
                side=Side.SELL,
                quantity=signal.quantity,
                limit_price=signal.entry_price,
            )
        return None

    @staticmethod
    def _latest_bar(
        universe: Dict[str, Sequence[Bar]], symbol: str,
    ) -> Optional[Bar]:
        bars = universe.get(symbol)
        if bars and len(bars) > 0:
            return bars[-1]
        return None
