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

import inspect
import logging
import re
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)


LIVE_A_PLUS_SCANNERS = frozenset({
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
        self._trade_guard = TradeGuard()
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
        self._append_entry_decision(
            result,
            self._entry_policy.decision(
                symbol=symbol,
                stage=stage,
                passed=False,
                reason=reason,
                blocked_layer=stage,
                signal=signal,
                price=price,
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
        """When True, classify and scan but skip trade execution."""
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
        return "{}:{}".format(hit.symbol, self._cooldown_pattern(hit))

    def _reject_cooldown_reason(self, hit: ScanResult, now: datetime) -> Optional[str]:
        cd = self._reject_cooldowns.get(self._cooldown_key(hit))
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
        ttl = self._reject_ttl_seconds(reason)
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

    def set_cooldown(self, symbol: str, now: Optional[datetime] = None) -> None:
        """Record an exit time for cooldown enforcement."""
        self._exit_cooldowns[symbol] = now or datetime.now(timezone.utc)

    def record_realized_exit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        *,
        now: Optional[datetime] = None,
    ) -> float:
        """Update daily P&L, loser blacklist, and circuit breaker (any exit path)."""
        if entry_price <= 0 or quantity <= 0:
            return 0.0
        trade_pnl = (exit_price - entry_price) * quantity
        self._daily_pnl += trade_pnl
        if trade_pnl < 0 and self._enable_daily_loser_blacklist:
            loss_amount = abs(trade_pnl)
            # CONSECUTIVE losses (reset on a win below) — a name that loses a
            # normal scalp then wins shouldn't carry that loss toward a ban.
            loss_count = self._daily_loss_counts.get(symbol, 0) + 1
            self._daily_loss_counts[symbol] = loss_count
            if (
                loss_amount >= self._daily_loser_blacklist_min_loss
                or loss_count >= self._daily_loser_blacklist_max_losses
            ):
                self._daily_losers.add(symbol)
                logger.info(
                    "DAILY BLACKLIST %s: %d consecutive loss(es), lost $%.2f - no re-entry today",
                    symbol, loss_count, loss_amount,
                )
            else:
                logger.info(
                    "DAILY LOSS %s: %d consecutive loss(es), lost $%.2f - not blacklisted yet",
                    symbol, loss_count, loss_amount,
                )
        # Shorter cooldown for profitable exits to allow pullback re-entry
        if trade_pnl > 0:
            # A win resets the consecutive-loss counter for this symbol.
            self._daily_loss_counts[symbol] = 0
            self._exit_cooldowns[symbol] = (now or datetime.now(timezone.utc)) - timedelta(
                seconds=self._cooldown_seconds - 120
            )
        if self._daily_pnl <= self._max_daily_loss and not self._circuit_breaker_tripped:
            self._circuit_breaker_tripped = True
            logger.warning(
                "CIRCUIT BREAKER: daily P&L $%.2f hit max loss $%.2f — "
                "NO MORE TRADES TODAY",
                self._daily_pnl, self._max_daily_loss,
            )
        # Log trade outcome for ML data collection
        try:
            from daytrading.ml.data_collector import log_trade_outcome
            log_trade_outcome(
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
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
        if "dead cat bounce" not in text and not score_near_miss and not liquidity_watch_only:
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

        if not bars or len(bars) < 3:
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

        latest_volume = float(latest.volume or 0.0)
        recent_volume = sum(float(b.volume or 0.0) for b in bars[-3:])
        if entry_trigger == "warrior_smooth_10s_pullback_continuation":
            min_latest_volume = 30_000
            min_recent_volume = 80_000
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
    ) -> PipelineResult:
        """Run one full cycle: classify → exits → scan → verify → execute."""
        result = PipelineResult()
        self._missed_a_plus.update_prices(universe, now=now)
        self._record_scanner_near_misses(universe, now)

        if now is None:
            for bars in universe.values():
                if bars:
                    now = bars[-1].ts
                    break

        # --- Step 0: check exits on open positions ---
        prices: Dict[str, float] = {}
        if now is not None:
            prices = {
                sym: brs[-1].close
                for sym, brs in universe.items()
                if brs
            }
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
                }
                for sym, pos in self._exit_manager.tracked.items()
            }
            exit_signals = self._exit_manager.check_exits(prices, now)
            for exit_sig in exit_signals:
                exit_order = self._signal_to_order(exit_sig)
                if exit_order is None:
                    continue
                bar = self._latest_bar(universe, exit_sig.symbol)
                if bar is None:
                    continue
                fill, status = self._broker.submit(exit_order, bar, self._portfolio)
                if status is OrderStatus.FILLED and fill is not None:
                    apply_fill(self._portfolio, fill)
                    result.exit_fills.append(fill)
                    result.exit_reasons[exit_sig.symbol] = exit_sig.reason or "unknown"
                    self._exit_cooldowns[exit_sig.symbol] = now or datetime.now(timezone.utc)

                    tracked_pos = self._exit_manager._positions.get(exit_sig.symbol)
                    entry_px = tracked_pos.entry_price if tracked_pos else 0
                    if entry_px > 0:
                        self.record_realized_exit(
                            exit_sig.symbol, entry_px, fill.price, fill.quantity,
                            now=now,
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
                    tracked_pos = self._exit_manager._positions.get(exit_sig.symbol)
                    if tracked_pos and tracked_pos.sold_half and tracked_pos.remaining_qty > 0:
                        tracked_pos.sold_half = False
                        tracked_pos.remaining_qty += exit_sig.quantity
                        tracked_pos.stop_loss = tracked_pos.entry_price - tracked_pos.risk_per_share
                        tracked_pos.breakeven_locked = False
                        logger.info(
                            "ROLLBACK half-sell %s: restored qty=%d, stop=%.4f",
                            exit_sig.symbol, int(tracked_pos.remaining_qty), tracked_pos.stop_loss,
                        )
        # --- Step 0b: scale-up check on winning positions ---
        if self._scaler is not None and now is not None:
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
        if self._reentry is not None and now is not None:
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
                    self._exit_manager.register_from_signal(re_sig, now, fill_price=fill.price)
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
                    # Bar-based scanning (momentum burst)
                    hits = scanner.scan(group)
                    style_hits.extend(hits)

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
                                style_hits.append(th)

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
                                style_hits.append(qh)

                    logger.info("Scanner %s (%s) found %d hits", scanner.name, style.value, len(hits))

                result.scan_hits.extend(style_hits)
                for hit in style_hits:
                    hit.criteria["setup_tier"] = self._setup_tier(hit)
                    cooldown_reason = self._reject_cooldown_reason(hit, now or datetime.now(timezone.utc))
                    if cooldown_reason is not None:
                        self._scan_rejections[hit.symbol] = cooldown_reason
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
                    signal = self._verify_hit(verifier, hit, now=now)
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
                hits = scanner.scan(universe)
                result.scan_hits.extend(hits)
                logger.info("Scanner %s found %d hits", scanner.name, len(hits))
            for hit in result.scan_hits:
                hit.criteria["setup_tier"] = self._setup_tier(hit)
                cooldown_reason = self._reject_cooldown_reason(hit, now or datetime.now(timezone.utc))
                if cooldown_reason is not None:
                    self._scan_rejections[hit.symbol] = cooldown_reason
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
                signal = self._verify_hit(verifier, hit, now=now)
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
                            signal = TradeSignal(
                                symbol=signal.symbol,
                                action=signal.action,
                                quantity=float(safe_qty),
                                entry_price=signal.entry_price,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                                trailing_stop_offset=signal.trailing_stop_offset,
                                max_hold_seconds=signal.max_hold_seconds,
                                reason=signal.reason,
                                scan_result=signal.scan_result,
                            )

                    # Also cap to max_order_shares
                    if signal.quantity > self._max_order_shares:
                        logger.info(
                            "SIZE CAP %s: %.0f → %d shares (max %d)",
                            signal.symbol, signal.quantity, int(self._max_order_shares), int(self._max_order_shares),
                        )
                        signal = TradeSignal(
                            symbol=signal.symbol,
                            action=signal.action,
                            quantity=float(int(self._max_order_shares)),
                            entry_price=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            trailing_stop_offset=signal.trailing_stop_offset,
                            max_hold_seconds=signal.max_hold_seconds,
                            reason=signal.reason,
                            scan_result=signal.scan_result,
                        )

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
                guard_ok, guard_reason = self._trade_guard.check_entry(
                    signal, bars=sym_bars, quotes=sym_quotes,
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

            # Defer to execution timer if enabled, otherwise submit immediately
            if (self._execution_timer is not None
                    and signal.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT)):
                result.deferred_signals.append(signal)
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

            fill, status = self._broker.submit(order, bar, self._portfolio)
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
                    self._exit_manager.register_from_signal(signal, now, fill_price=fill.price)
                    self._original_sizes[signal.symbol] = signal.quantity
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
            if risk_pct > cap:
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
        decision = self._entry_policy.evaluate(
            signal,
            bars=sym_bars,
            quotes=sym_quotes,
            stage=stage,
            min_day_change_pct=0.0,
            now=now,
        )
        reject_reason = decision.reject_reason
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

    def _verify_hit(
        self,
        verifier: StrategyVerifier,
        hit: ScanResult,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[TradeSignal]:
        """Call verifiers with replay clock when supported.

        Legacy tests and simple verifiers may still implement the old
        two-argument signature, so keep that path compatible.
        """
        try:
            signature = inspect.signature(verifier.verify)
            params = signature.parameters
            supports_now = (
                "now" in params
                or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            )
        except (TypeError, ValueError):
            supports_now = True
        if supports_now:
            return verifier.verify(hit, self._portfolio, now=now)
        return verifier.verify(hit, self._portfolio)

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
