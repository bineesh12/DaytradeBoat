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

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from daytrading.classifier.regime import MarketRegimeClassifier
from daytrading.classifier.router import AdaptiveRouter, StyleConfig
from daytrading.data.news_checker import NewsChecker
from daytrading.execution.broker import Broker, apply_fill
from daytrading.exits.manager import ExitManager
from daytrading.exits.scaler import PositionScaler, ReentryDetector
from daytrading.risk.manager import allow_order
from daytrading.risk.guards import TradeGuard
from daytrading.scanner.base import Scanner
from daytrading.strategy.verifier import StrategyVerifier
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
    symbols_by_style: Dict[str, List[str]] = field(default_factory=dict)


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
        self._original_sizes: Dict[str, float] = {}  # for re-entry size calc
        self._cached_regimes: Dict[str, MarketRegime] = {}  # persist trend data across cycles
        self._scan_only = False
        self._news_checker: Optional[NewsChecker] = None
        self._exit_cooldowns: Dict[str, datetime] = {}
        self._cooldown_seconds: int = 180  # 3 min cooldown after exiting a stock
        self._scan_rejections: Dict[str, str] = {}  # symbol -> last rejection reason
        self._trade_guard = TradeGuard()

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

    @staticmethod
    def _get_last_reject(symbol: str, verifier: Any) -> Optional[str]:
        """Try to extract the last rejection reason from verifier logs."""
        last_reject = getattr(verifier, '_last_reject', None)
        return last_reject

    def set_news_checker(self, checker: NewsChecker) -> None:
        """Attach a news checker for pre-trade sentiment screening."""
        self._news_checker = checker

    def set_cooldown(self, symbol: str) -> None:
        """Record an exit time for cooldown enforcement."""
        self._exit_cooldowns[symbol] = datetime.now(timezone.utc)

    def run_cycle(
        self,
        universe: Dict[str, Sequence[Bar]],
        now: Optional[datetime] = None,
        quotes: Optional[Dict[str, Sequence[Quote]]] = None,
        ticks: Optional[Dict[str, Sequence[Tick]]] = None,
    ) -> PipelineResult:
        """Run one full cycle: classify → exits → scan → verify → execute."""
        result = PipelineResult()

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
                    self._exit_manager.update_bar_close(sym, brs[-2].close, brs[-2].open, brs[-2].volume)
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
                    self._exit_cooldowns[exit_sig.symbol] = datetime.now(timezone.utc)
                    logger.info(
                        "EXIT FILLED %s %.0f @ %.4f | %s (cooldown %ds)",
                        exit_sig.symbol, fill.quantity, fill.price, exit_sig.reason,
                        self._cooldown_seconds,
                    )
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

            # track fully closed positions for re-entry
            if self._reentry is not None:
                for exit_sig in exit_signals:
                    sym = exit_sig.symbol
                    tracked = self._exit_manager.tracked
                    if sym not in tracked:
                        pos_data = self._exit_manager.tracked  # already removed
                        self._reentry.record_full_exit(
                            symbol=sym,
                            side=Side.BUY if exit_sig.action is SignalAction.EXIT_LONG else Side.SELL,
                            exit_price=exit_sig.entry_price,
                            exit_ts=now,
                            highest_price=exit_sig.entry_price,
                            entry_price=self._original_sizes.get(sym, exit_sig.entry_price),
                        )
                        if self._scaler:
                            self._scaler.clear(sym)

        # --- Step 0b: scale-up check on winning positions ---
        if self._scaler is not None and now is not None:
            scale_signals = self._scaler.check_scale_ups(
                self._exit_manager, prices, universe,
            )
            for scale_sig in scale_signals:
                scale_order = self._signal_to_order(scale_sig)
                if scale_order is None:
                    continue
                bar = self._latest_bar(universe, scale_sig.symbol)
                if bar is None:
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
                re_order = self._signal_to_order(re_sig)
                if re_order is None:
                    continue
                bar = self._latest_bar(universe, re_sig.symbol)
                if bar is None:
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
                    verifier = cfg.verifiers.get(hit.scanner_name)
                    if verifier is None:
                        continue
                    signal = verifier.verify(hit, self._portfolio)
                    if signal is None:
                        reject = getattr(hit, '_reject_reason', None)
                        if reject is None:
                            reject = self._get_last_reject(hit.symbol, verifier)
                        self._scan_rejections[hit.symbol] = reject or "did not pass verifier"
                        continue
                    if signal.action is SignalAction.SKIP:
                        result.skipped.append(signal)
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
                    result.signals.append(signal)
        else:
            # no router → run all scanners against full universe (legacy mode)
            for scanner in self._scanners:
                hits = scanner.scan(universe)
                result.scan_hits.extend(hits)
                logger.info("Scanner %s found %d hits", scanner.name, len(hits))
            for hit in result.scan_hits:
                verifier = self._verifiers.get(hit.scanner_name)
                if verifier is None:
                    continue
                signal = verifier.verify(hit, self._portfolio)
                if signal is None:
                    reject = self._get_last_reject(hit.symbol, verifier)
                    self._scan_rejections[hit.symbol] = reject or "did not pass verifier"
                    continue
                if signal.action is SignalAction.SKIP:
                    result.skipped.append(signal)
                    continue
                self._scan_rejections.pop(hit.symbol, None)
                result.signals.append(signal)

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
                last_exit_ts = self._exit_cooldowns.get(signal.symbol)
                if last_exit_ts is not None:
                    elapsed = (datetime.now(timezone.utc) - last_exit_ts).total_seconds()
                    if elapsed < self._cooldown_seconds:
                        remaining = self._cooldown_seconds - elapsed
                        logger.info(
                            "COOLDOWN %s: exited %.0fs ago, wait %.0fs more",
                            signal.symbol, elapsed, remaining,
                        )
                        result.skipped.append(signal)
                        continue

            if self._at_position_limit():
                logger.warning("Position limit %d reached, skipping %s", self._max_positions, signal.symbol)
                result.skipped.append(signal)
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
                        continue

                    # Cap max dollar risk per trade to $100
                    max_dollar_risk = 100.0
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

            order = self._signal_to_order(signal)
            if order is None:
                continue

            bar = self._latest_bar(universe, signal.symbol)
            if bar is None:
                logger.warning("No bar data for %s, cannot execute", signal.symbol)
                result.rejected_orders += 1
                continue

            ok = allow_order(
                order, bar, self._portfolio,
                max_position_shares=self._max_position_shares,
                max_order_shares=self._max_order_shares,
            )
            if not ok:
                logger.info("Risk check rejected %s for %s", signal.action.value, signal.symbol)
                result.rejected_orders += 1
                continue

            # Advanced guards: false breakout, liquidity trap, halt, market panic, spread
            if signal.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT):
                sym_bars = universe.get(signal.symbol)
                sym_quotes = quotes.get(signal.symbol) if quotes else None
                guard_ok, guard_reason = self._trade_guard.check_entry(
                    signal, bars=sym_bars, quotes=sym_quotes,
                )
                if not guard_ok:
                    logger.info("GUARD REJECT %s: %s", signal.symbol, guard_reason)
                    result.rejected_orders += 1
                    continue

            fill, status = self._broker.submit(order, bar, self._portfolio)
            if status is OrderStatus.FILLED and fill is not None:
                apply_fill(self._portfolio, fill)
                result.fills.append(fill)
                entered_this_cycle.add(signal.symbol)
                logger.info(
                    "FILLED %s %s %.0f @ %.4f | %s",
                    signal.action.value, signal.symbol,
                    fill.quantity, fill.price, signal.reason,
                )
                if now is not None:
                    self._exit_manager.register_from_signal(signal, now, fill_price=fill.price)
                    self._original_sizes[signal.symbol] = signal.quantity
            else:
                logger.info("Order %s for %s: %s", signal.action.value, signal.symbol, status.value)
                result.rejected_orders += 1

        return result

    def _at_position_limit(self) -> bool:
        open_positions = sum(1 for p in self._portfolio.positions.values() if not p.is_flat)
        return open_positions >= self._max_positions

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
