"""Thin entry-execution boundary around the shared EntryPolicy."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Sequence

from daytrading.execution.broker import apply_fill
from daytrading.models import Bar, Order, Quote, Side, SignalAction, Tick, TradeSignal
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
                )

            chase_reject = ctx.chase_reject(signal, bar)
            if chase_reject:
                logger.info("TIMED ENTRY skip %s — %s", sym, chase_reject)
                ctx.hub.add_log("WARNING", "ENTRY SKIP {}: {}".format(sym, chase_reject))
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

            quality_reject = ctx.shared_entry_quality_reject(
                sym,
                list(bars),
                signal=signal,
                stage="timed_entry_final_guard",
                source="timed_entry",
            )
            if quality_reject:
                logger.info("TIMED ENTRY skip %s — shared entry quality %s", sym, quality_reject)
                ctx.hub.add_log("WARNING", "ENTRY SKIP {}: {}".format(sym, quality_reject))
                ctx.record_missed_a_plus(
                    signal,
                    layer="entry_guard",
                    reason=quality_reject,
                    fallback_price=bar.close,
                )
                return

            fill, status = ctx.broker.submit(order, bar, ctx.pipeline.portfolio)
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
                and ctx.is_hot_hod_timed_signal(signal)
                and signal.action in (SignalAction.ENTER_LONG, SignalAction.REENTER_LONG)
            ):
                fill, status = ctx.retry_hot_hod_timed_entry(signal, status, bar)
            if fill:
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
                ctx.hub.on_fill(fill, "entry")
                ctx.hub.add_log(
                    "INFO",
                    "ENTRY {} {} {:.0f} @ ${:.2f} (10s timed)".format(
                        fill.side.value.upper(),
                        fill.symbol,
                        fill.quantity,
                        fill.price,
                    ),
                )
                ctx.journal.record("trade_fill", {
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
                }, ts=fill.ts)
                ctx.seed_recent_order_ids()
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
                )

            chase_reject = ctx.chase_reject(signal, bar)
            if chase_reject:
                logger.info("TIMED SCALE-UP skip %s — %s", sym, chase_reject)
                ctx.hub.add_log("WARNING", "RE-ADD SKIP {}: {}".format(sym, chase_reject))
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
                ctx.hub.add_log("WARNING", "RE-ADD SKIP {}: {}".format(sym, quality_reject))
                return

            order = Order(
                symbol=sym,
                side=Side.BUY,
                quantity=signal.quantity,
                limit_price=signal.entry_price,
            )
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
                ctx.hub.on_fill(fill, "scale_up")
                ctx.hub.add_log(
                    "INFO",
                    "RE-ADD {} +{:.0f} @ ${:.2f}".format(
                        sym,
                        fill.quantity,
                        fill.price,
                    ),
                )
                ctx.journal.record("trade_fill", {
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "ts": fill.ts,
                    "trade_type": "scale_up",
                    "strategy": "runner_readd",
                    "execution_method": "10s_timed",
                    "market_context": {"phase": ctx.market_phase()},
                }, ts=fill.ts)
                ctx.seed_recent_order_ids()
            else:
                logger.warning(
                    "TIMED RUNNER RE-ADD order not filled for %s (status=%s)",
                    sym,
                    status,
                )
        except Exception as exc:
            logger.error("Timed scale-up execution error for %s: %s", signal.symbol, exc)
