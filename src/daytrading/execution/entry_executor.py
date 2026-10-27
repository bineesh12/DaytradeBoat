"""Thin entry-execution boundary around the shared EntryPolicy."""

from __future__ import annotations
from daytrading.models import BarSource

from daytrading.execution.fill_identity import report_execution
from daytrading.execution.origin import signal_execution

from daytrading.reporting import report_safely, reporting_failures

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Sequence

from daytrading.execution.broker import apply_fill
from daytrading.models import Bar, Order, OrderStatus, Quote, Side, SignalAction, Tick, TradeSignal
from daytrading.risk.manager import allow_order
from daytrading.strategy.entry_policy import EntryDecision, EntryPolicy

logger = logging.getLogger(__name__)


@dataclass
class EntryExecutionContext:
    """Narrow runtime surface needed to execute timed entries."""

    new_entries_blocked: Callable[[Optional[str], str], bool]
    pipeline: object
    bar_buffer: Dict
    hub: object
    broker: object
    journal: object
    chase_reject: Callable[[TradeSignal, Bar], Optional[str]]
    record_entry_reject: Callable[..., EntryDecision]
    record_missed_a_plus: Callable[..., None]
    shared_entry_quality_reject: Callable[..., Optional[str]]
    execution_learning_context: Callable[[TradeSignal, Bar], Dict]
    is_hot_hod_timed_signal: Callable[[TradeSignal], bool]
    retry_hot_hod_timed_entry: Callable
    on_position_opened: Callable[..., None]
    market_phase: Callable[[], str]
    seed_recent_order_ids: Callable[[], None]
    quote_buffer: Optional[Dict] = None
    market_now: Optional[Callable] = None
    market_entry_window: Optional[Callable] = None
    submit_entry: Optional[Callable] = None
    finish_entry: Optional[Callable] = None
    entry_pending: Optional[Callable] = None
    observe_entry: Optional[Callable] = None


class EntryExecutor:
    """Evaluate entry eligibility and emit a single structured decision."""

    def __init__(
        self,
        policy: EntryPolicy,
        recorder: Optional[Callable[[EntryDecision, str], None]] = None,
    ) -> None:
        self._policy = policy
        self._recorder = recorder

    def set_policy(self, policy: EntryPolicy) -> None:
        self._policy = policy

    def set_recorder(self, recorder: Optional[Callable[[EntryDecision, str], None]]) -> None:
        self._recorder = recorder

    def _record(self, decision: EntryDecision, source: str) -> EntryDecision:
        if self._recorder is not None:
            try:
                self._recorder(decision, source=source)
            except TypeError:
                self._recorder(decision, source)
        return decision

    def record_decision(self, decision: EntryDecision, *, source: str) -> EntryDecision:
        return self._record(decision, source)

    def evaluate_quality(
        self,
        signal: TradeSignal,
        *,
        bars: Sequence[Bar],
        stage: str,
        source: str,
        quotes: Optional[Sequence[Quote]] = None,
        ticks: Optional[Sequence[Tick]] = None,
        bars_5m: Optional[Sequence[Bar]] = None,
        avg_daily_volume: Optional[float] = None,
        float_shares: Optional[float] = None,
        min_day_change_pct: float = 0.0,
        metadata: Optional[Dict] = None,
        now: Optional[datetime] = None,
    ) -> EntryDecision:
        decision = self._policy.evaluate(
            signal,
            bars=bars,
            stage=stage,
            quotes=quotes,
            ticks=ticks,
            bars_5m=bars_5m,
            avg_daily_volume=avg_daily_volume,
            float_shares=float_shares,
            min_day_change_pct=min_day_change_pct,
            metadata=metadata,
            now=now,
        )
        return self._record(decision, source)

    def reject(
        self,
        *,
        symbol: str,
        stage: str,
        reason: str,
        source: str,
        signal: Optional[TradeSignal] = None,
        blocked_layer: str = "",
        price: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> EntryDecision:
        from daytrading.execution.origin import entry_refused
        entry_refused(reason)
        decision = self._policy.decision(
            symbol=symbol,
            stage=stage,
            passed=False,
            reason=reason,
            blocked_layer=blocked_layer or stage,
            signal=signal,
            price=price or None,
            metadata=metadata,
        )
        return self._record(decision, source)

    def _reject_order_risk(
        self, ctx: EntryExecutionContext, signal: TradeSignal, order: Order,
        bar: Bar, *, source: str,
    ) -> bool:
        """Recheck risk against the portfolio when a delayed order is released."""
        portfolio = ctx.pipeline.portfolio
        position = portfolio.positions.get(order.symbol)
        opening_position = position is None or position.is_flat
        max_positions = getattr(ctx.pipeline, "_max_positions", None)
        reason = None
        if opening_position and max_positions is not None:
            open_positions = sum(not pos.is_flat for pos in portfolio.positions.values())
            if open_positions >= max_positions:
                reason = "position limit"
        if reason is None and not allow_order(
            order, bar, portfolio,
            max_position_shares=getattr(ctx.pipeline, "_max_position_shares", None),
            max_order_shares=getattr(ctx.pipeline, "_max_order_shares", None),
        ):
            reason = "position_risk_limit"
        if reason is None:
            return False
        ctx.record_entry_reject(
            signal, stage="{}_risk".format(source), reason=reason,
            source=source, price=bar.close,
        )
        logger.info("%s skip %s — %s", source, order.symbol, reason)
        return True

    @signal_execution
    def execute_timed_signal(self, ctx: EntryExecutionContext, signal: TradeSignal) -> None:
        """Execute a deferred signal released by the 10s execution timer."""
        try:
            sym = signal.symbol
            if ctx.new_entries_blocked(sym, "TIMED ENTRY"):
                return

            if signal.action in (SignalAction.SCALE_UP_LONG, SignalAction.SCALE_UP_SHORT):
                self.execute_timed_scale_up(ctx, signal)
                return

            last_exit_ts = ctx.pipeline._exit_cooldowns.get(sym)
            if last_exit_ts is not None:
                elapsed = (datetime.now(timezone.utc) - last_exit_ts).total_seconds()
                if elapsed < ctx.pipeline._cooldown_seconds:
                    logger.info(
                        "TIMED ENTRY skip %s — on cooldown (%.0fs ago)", sym, elapsed,
                    )
                    return

            pos = ctx.pipeline.portfolio.positions.get(sym)
            if pos and not pos.is_flat:
                logger.info("TIMED ENTRY skip %s — already in position", sym)
                return

            if sym in ctx.pipeline._daily_losers:
                logger.info("TIMED ENTRY skip %s — daily loser blacklist", sym)
                return

            if (
                ctx.pipeline._symbol_entry_counts.get(sym, 0)
                >= ctx.pipeline._max_entries_per_symbol
            ):
                logger.info(
                    "TIMED ENTRY skip %s — max entries reached (%d)",
                    sym,
                    ctx.pipeline._max_entries_per_symbol,
                )
                return

            order = Order(
                symbol=sym,
                side=Side.BUY if signal.action in (
                    SignalAction.ENTER_LONG,
                    SignalAction.REENTER_LONG,
                ) else Side.SELL,
                quantity=signal.quantity,
                limit_price=signal.entry_price,
            )
            bars = ctx.bar_buffer.get(signal.symbol, deque())
            if bars:
                bar = bars[-1]
            else:
                bar = Bar(
                    symbol=signal.symbol,
                    open=signal.entry_price,
                    high=signal.entry_price,
                    low=signal.entry_price,
                    close=signal.entry_price,
                    volume=0,
                    ts=datetime.now(timezone.utc),
                 provenance=BarSource.PRICE_MARK)

            chase_reject = ctx.chase_reject(signal, bar)
            if chase_reject:
                logger.info("TIMED ENTRY skip %s — %s", sym, chase_reject)
                report_safely(ctx.hub.add_log, "WARNING", "ENTRY SKIP {}: {}".format(sym, chase_reject))
                ctx.record_entry_reject(
                    signal,
                    stage="timed_entry_chase",
                    reason=chase_reject,
                    source="timed_entry",
                    price=bar.close,
                )
                ctx.record_missed_a_plus(
                    signal,
                    layer="timed_entry",
                    reason=chase_reject,
                    fallback_price=bar.close,
                )
                return

            hit = signal.scan_result
            criteria = hit.criteria if hit is not None else {}
            try:
                spread_size_factor = float(criteria.get("spread_size_factor") or 1.0)
            except (TypeError, ValueError):
                spread_size_factor = 1.0
            if 0 < spread_size_factor < 1.0 and order.quantity > 1:
                reduced_qty = max(1, int(float(order.quantity) * spread_size_factor))
                if reduced_qty < order.quantity:
                    logger.info(
                        "TIMED ENTRY %s size down %.0f → %d for opportunity-scaled spread",
                        sym,
                        order.quantity,
                        reduced_qty,
                    )
                    order = Order(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=float(reduced_qty),
                        limit_price=order.limit_price,
                    )

            quality_reject = ctx.shared_entry_quality_reject(
                sym,
                list(bars),
                signal=signal,
                stage="timed_entry_final_guard",
                source="timed_entry",
            )
            if quality_reject:
                logger.info("TIMED ENTRY skip %s — shared entry quality %s", sym, quality_reject)
                report_safely(ctx.hub.add_log, "WARNING", "ENTRY SKIP {}: {}".format(sym, quality_reject))
                ctx.record_missed_a_plus(
                    signal,
                    layer="entry_guard",
                    reason=quality_reject,
                    fallback_price=bar.close,
                )
                return

            if hit is not None and hit.scanner_name in ('micro_pullback_reclaim', 'prepared_reclaim_tick'):
                from daytrading.strategy.scalping.micro_pullback import MicroRiskLimits, Rejection
                from daytrading.strategy.scalping.micro_pullback_runtime import bounded_micro_order
                now = ctx.market_now() if ctx.market_now is not None else datetime.now(timezone.utc)
                quotes = list((ctx.quote_buffer or {}).get(sym, []))
                reason = None
                if now is None:
                    reason = 'clock_alignment: fresh broker clock calibration required'
                elif ctx.market_entry_window is not None and not ctx.market_entry_window(now):
                    reason = 'new entries closed by configured trading window (broker UTC)'
                elif getattr(ctx.broker, '_paper', False) is not True:
                    reason = 'micro pullback orders require PAPER mode'
                else:
                    guard_context = ({"event_context": hit.criteria.get("_event_evidence")}
                                     if hit.scanner_name == "prepared_reclaim_tick" else {"breakout_context": hit.confirmation_window})
                    ok, why = ctx.pipeline.trade_guard.check_entry(signal, bars=list(bars),
                        quotes=quotes, now=now, **guard_context)
                    if not ok:
                        reason = why
                if reason is None:
                    limits = MicroRiskLimits(min_price=self._policy.min_price or 1,
                        max_price=self._policy.max_price or 20,
                        max_dollar_risk=ctx.pipeline._max_dollar_risk_per_trade,
                        max_order_shares=ctx.pipeline._max_order_shares,
                        max_stop_pct=ctx.pipeline._max_entry_risk_pct)
                    if hit.scanner_name == "prepared_reclaim_tick":
                        from daytrading.strategy.scalping.event_reclaim import bounded_event_order
                        planned = bounded_event_order(signal, quotes[-1] if quotes else None, bar,
                            ticks=list(getattr(ctx.pipeline._verifiers.get(hit.scanner_name), "_tick_buffer", {}).get(sym, ())), now=now, limits=limits)
                    else:
                        planned = bounded_micro_order(signal, quotes[-1] if quotes else None,
                                                      bar, now=now, limits=limits)
                    if isinstance(planned, Rejection):
                        from daytrading.execution.entry_outcome import RecoverableEntryRefusal
                        if isinstance(planned, RecoverableEntryRefusal):
                            from daytrading.execution.origin import entry_refused
                            entry_refused(planned)
                        reason = planned.code+': '+planned.reason
                    else:
                        order = planned
                if reason:
                    ctx.record_entry_reject(signal, stage='micro_execution', reason=reason,
                                            source='micro_pullback', price=bar.close)
                    return

            if self._reject_order_risk(ctx, signal, order, bar, source="timed_entry"):
                return

            if hit is not None and hit.scanner_name in ('micro_pullback_reclaim','prepared_reclaim_tick'):
                from daytrading.strategy.submission import final_check, persist
                now=ctx.market_now() if ctx.market_now is not None else datetime.now(timezone.utc)
                qs=list((ctx.quote_buffer or {}).get(sym,()))
                quote=qs[-1] if qs else None
                verifier=getattr(ctx.pipeline,'_verifiers',{}).get(hit.scanner_name)
                ticks=list(getattr(verifier,'_tick_buffer',{}).get(sym,()))
                why,evidence=final_check(signal,order,policy=self._policy,bars=list(bars),
                    quote=quote,ticks=ticks,now=now,limits=limits,boundary='order_submission')
                why=persist(evidence,ctx.journal.record,ctx.market_now) or why
                if why:
                    ctx.record_entry_reject(signal,stage='submission_recheck',reason=why,source='micro_pullback')
                    return
            else:
                reason = self._policy.admission_reject(signal,
                    cap=order.hard_limit_price or order.limit_price, bars=list(bars),
                    now=ctx.market_now() if ctx.market_now else datetime.now(timezone.utc))
                if reason:
                    ctx.record_entry_reject(signal,stage='final_room_risk',reason=reason,source='timed_entry')
                    return

            fill, status = (ctx.submit_entry(signal, order, bar) if ctx.submit_entry is not None
                            else ctx.broker.submit(order, bar, ctx.pipeline.portfolio))
            try:
                from daytrading.ml.shadow_collector import log_execution_quality
                log_execution_quality(
                    order=order,
                    bar=bar,
                    status=status,
                    fill=fill,
                    source="timed_entry",
                    context=ctx.execution_learning_context(signal, bar),
                )
            except Exception:
                pass
            if (
                fill is None
                and status is not OrderStatus.NEW
                and getattr(ctx.broker, "_conflict_fill_during_cancel", False) is not True
                and ctx.is_hot_hod_timed_signal(signal)
                and signal.action in (SignalAction.ENTER_LONG, SignalAction.REENTER_LONG)
            ):
                fill, status = ctx.retry_hot_hod_timed_entry(signal, status, bar)
            if fill:
                if ctx.submit_entry is None:
                    self.record_timed_fill(ctx, signal, fill)
                elif ctx.observe_entry is not None:
                    ctx.observe_entry(signal, fill)
            elif ctx.entry_pending is not None and ctx.entry_pending(sym):
                logger.warning("ENTRY SETTLEMENT PENDING %s: partial protection remains active", sym)
            else:
                logger.warning(
                    "TIMED ENTRY order not filled for %s (status=%s)",
                    signal.symbol,
                    status,
                )
                ctx.record_entry_reject(
                    signal,
                    stage="timed_entry_order",
                    reason="timed entry order {}".format(status.value if status else "not_filled"),
                    source="timed_entry",
                    price=bar.close,
                    metadata={"status": status.value if status else "not_filled"},
                )
                ctx.record_missed_a_plus(
                    signal,
                    layer="order",
                    reason="timed entry order {}".format(status.value if status else "not_filled"),
                    fallback_price=bar.close,
                )
        except Exception as exc:
            logger.error("Timed signal execution error for %s: %s", signal.symbol, exc)
        finally:
            if ctx.finish_entry is not None:
                ctx.finish_entry(signal)

    def record_timed_fill(self, ctx, signal, fill, *, already_applied=False):
        """One aggregate entry report; partial progress is accounted separately."""
        sym = signal.symbol
        if not already_applied:
            apply_fill(ctx.pipeline.portfolio, fill)
            ctx.pipeline._symbol_entry_counts[sym] = (
                ctx.pipeline._symbol_entry_counts.get(sym, 0) + 1
            )
        strategy = (
            signal.scan_result.scanner_name if signal.scan_result
            else signal.reason or "unknown"
        )
        ctx.on_position_opened(
            signal,
            fill,
            strategy=strategy,
            execution_method="10s_timed",
        )
        logger.info(
            "TIMED ENTRY %s %s %.0f @ $%.4f (strategy=%s)",
            fill.side.value,
            fill.symbol,
            fill.quantity,
            fill.price,
            strategy,
        )
        report_safely(ctx.hub.on_fill, fill, "entry", strategy=strategy)
        report_safely(ctx.hub.add_log,
            "INFO",
            "ENTRY {} {} {:.0f} @ ${:.2f} (10s timed)".format(
                fill.side.value.upper(),
                fill.symbol,
                fill.quantity,
                fill.price,
            ),
        )
        report_execution(ctx.journal, "trade_fill", {
            "symbol": fill.symbol,
            "side": fill.side.value,
            "quantity": fill.quantity,
            "price": fill.price,
            "ts": fill.ts,
            "trade_type": "entry",
            "strategy": strategy,
            "execution_method": "10s_timed",
            "market_context": {
                "phase": ctx.market_phase(),
            },
        }, fill)
        ctx.seed_recent_order_ids()

    @signal_execution
    def execute_timed_scale_up(self, ctx: EntryExecutionContext, signal: TradeSignal) -> None:
        """Execute a protected-runner re-add released by the 10s timer."""
        try:
            sym = signal.symbol
            tracked = ctx.pipeline.exit_manager._positions.get(sym)
            pos = ctx.pipeline.portfolio.positions.get(sym)
            if tracked is None or pos is None or pos.is_flat:
                logger.info("TIMED SCALE-UP skip %s — no active protected runner", sym)
                return
            if signal.action is not SignalAction.SCALE_UP_LONG or tracked.side is not Side.BUY:
                logger.info("TIMED SCALE-UP skip %s — unsupported side/action", sym)
                return
            if not tracked.sold_half or not tracked.breakeven_locked:
                logger.info("TIMED SCALE-UP skip %s — runner not protected yet", sym)
                return

            bars = ctx.bar_buffer.get(sym, deque())
            if bars:
                bar = bars[-1]
            else:
                bar = Bar(
                    symbol=sym,
                    open=signal.entry_price,
                    high=signal.entry_price,
                    low=signal.entry_price,
                    close=signal.entry_price,
                    volume=0,
                    ts=datetime.now(timezone.utc),
                 provenance=BarSource.PRICE_MARK)

            chase_reject = ctx.chase_reject(signal, bar)
            if chase_reject:
                logger.info("TIMED SCALE-UP skip %s — %s", sym, chase_reject)
                report_safely(ctx.hub.add_log, "WARNING", "RE-ADD SKIP {}: {}".format(sym, chase_reject))
                ctx.record_entry_reject(
                    signal,
                    stage="timed_scale_up_chase",
                    reason=chase_reject,
                    source="timed_scale_up",
                    price=bar.close,
                )
                return

            quality_reject = ctx.shared_entry_quality_reject(
                sym,
                list(bars),
                signal=signal,
                stage="timed_scale_up_final_guard",
                source="timed_scale_up",
            )
            if quality_reject:
                logger.info("TIMED SCALE-UP skip %s — shared entry quality %s", sym, quality_reject)
                report_safely(ctx.hub.add_log, "WARNING", "RE-ADD SKIP {}: {}".format(sym, quality_reject))
                return

            try:
                criteria = signal.scan_result.criteria if signal.scan_result is not None else {}
                spread_size_factor = float(criteria.get("spread_size_factor") or 1.0)
            except (AttributeError, TypeError, ValueError):
                spread_size_factor = 1.0
            quantity = signal.quantity
            if 0 < spread_size_factor < 1.0 and quantity > 1:
                reduced_qty = max(1, int(float(quantity) * spread_size_factor))
                if reduced_qty < quantity:
                    logger.info(
                        "TIMED SCALE-UP %s size down %.0f → %d for opportunity-scaled spread",
                        sym,
                        quantity,
                        reduced_qty,
                    )
                    quantity = float(reduced_qty)

            order = Order(
                symbol=sym,
                side=Side.BUY,
                quantity=quantity,
                limit_price=signal.entry_price,
            )
            if self._reject_order_risk(ctx, signal, order, bar, source="timed_scale_up"):
                return

            fill, status = ctx.broker.submit(order, bar, ctx.pipeline.portfolio)
            try:
                from daytrading.ml.shadow_collector import log_execution_quality
                log_execution_quality(
                    order=order,
                    bar=bar,
                    status=status,
                    fill=fill,
                    source="timed_runner_readd",
                    context=ctx.execution_learning_context(signal, bar),
                )
            except Exception:
                pass
            if fill:
                apply_fill(ctx.pipeline.portfolio, fill)
                ctx.pipeline.exit_manager.scale_up(
                    sym,
                    fill.quantity,
                    fill.price,
                    signal.stop_loss,
                )
                logger.info(
                    "TIMED RUNNER RE-ADD %s +%.0f @ $%.4f stop=$%.4f | %s",
                    sym,
                    fill.quantity,
                    fill.price,
                    signal.stop_loss or 0.0,
                    signal.reason,
                )
                strategy = (
                    signal.scan_result.scanner_name if signal.scan_result
                    else signal.reason or "unknown"
                )
                report_safely(ctx.hub.on_fill, fill, "scale_up", strategy=strategy)
                report_safely(ctx.hub.add_log,
                    "INFO",
                    "RE-ADD {} +{:.0f} @ ${:.2f}".format(
                        sym,
                        fill.quantity,
                        fill.price,
                    ),
                )
                report_execution(ctx.journal, "trade_fill", {
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "ts": fill.ts,
                    "trade_type": "scale_up",
                    "strategy": strategy,
                    "execution_method": "10s_timed",
                    "market_context": {"phase": ctx.market_phase()},
                }, fill)
                ctx.seed_recent_order_ids()
            else:
                logger.warning(
                    "TIMED RUNNER RE-ADD order not filled for %s (status=%s)",
                    sym,
                    status,
                )
        except Exception as exc:
            logger.error("Timed scale-up execution error for %s: %s", signal.symbol, exc)
