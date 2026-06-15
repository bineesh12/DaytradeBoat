from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from daytrading.backtest.broker import BacktestBroker, FillModel
from daytrading.backtest.data_loader import merge_bar_times, trim_universe_to_time
from daytrading.backtest.report import BacktestLedger, build_backtest_scorecard
from daytrading.execution.broker import apply_fill
from daytrading.models import Bar, Fill, OrderStatus, PortfolioState, Quote, ScanResult, SignalAction, Timeframe, TradeSignal
from daytrading.pipeline.engine import PipelineResult, TradingPipeline, _entry_strategy_label
from daytrading.pipeline.factory import create_scalping_pipeline
from daytrading.risk.manager import allow_order
from daytrading.strategy.execution_timer import ExecutionTimer


@dataclass
class PipelineBacktestResult:
    fills: List[Fill] = field(default_factory=list)
    trades: List[dict] = field(default_factory=list)
    scorecard: dict = field(default_factory=dict)
    cycles: int = 0
    scan_hits: int = 0
    signals: int = 0
    rejected: int = 0
    deferred: int = 0
    final_portfolio: Optional[PortfolioState] = None
    missed_a_plus: List[dict] = field(default_factory=list)
    entry_decisions: List[dict] = field(default_factory=list)
    scan_events: List[dict] = field(default_factory=list)
    rejection_details: List[dict] = field(default_factory=list)
    deferred_signals: List[dict] = field(default_factory=list)
    micro_opportunities: List[dict] = field(default_factory=list)
    rejected_by_layer: Dict[str, int] = field(default_factory=dict)
    rejected_reasons_by_layer: Dict[str, List[dict]] = field(default_factory=dict)
    execution_timer_source: str = "off"
    _rejection_keys: set = field(default_factory=set, repr=False)


def estimate_quote_from_bar(bar: Bar, broker: BacktestBroker) -> Quote:
    spread = broker.estimated_spread(bar)
    mid = float(bar.close)
    return Quote(
        symbol=bar.symbol,
        ts=bar.ts,
        bid=round(max(mid - spread / 2.0, 0.01), 4),
        ask=round(mid + spread / 2.0, 4),
        bid_size=max(float(bar.volume) * 0.10, 100.0),
        ask_size=max(float(bar.volume) * 0.10, 100.0),
    )


class PipelineBacktestDriver:
    """Replay historical bars through the live scalping pipeline."""

    def __init__(
        self,
        bars_by_symbol: Dict[str, Sequence[Bar]],
        *,
        pipeline: Optional[TradingPipeline] = None,
        portfolio: Optional[PortfolioState] = None,
        initial_cash: float = 25_000.0,
        fill_model: Optional[FillModel] = None,
        max_bars_per_symbol: int = 120,
        use_execution_timer: bool = False,
        timer_bars_by_symbol: Optional[Dict[str, Sequence[Bar]]] = None,
        use_micro_breakout_scout: bool = False,
        use_level_reclaim_10s_scout: bool = False,
        live_like_10s: bool = False,
    ) -> None:
        self._bars_by_symbol = {
            sym.upper(): sorted(list(bars), key=lambda b: b.ts)
            for sym, bars in bars_by_symbol.items()
            if bars
        }
        self._broker = BacktestBroker(fill_model)
        self._portfolio = portfolio or PortfolioState(cash=initial_cash)
        self._pipeline = pipeline or create_scalping_pipeline(
            initial_cash=initial_cash,
            portfolio=self._portfolio,
            broker=self._broker,
        )
        self._max_bars_per_symbol = max(5, int(max_bars_per_symbol))
        self._use_execution_timer = bool(use_execution_timer)
        self._use_micro_breakout_scout = bool(use_micro_breakout_scout)
        self._use_level_reclaim_10s_scout = bool(use_level_reclaim_10s_scout)
        self._timer = ExecutionTimer(max_wait_bars=3, enabled=True) if self._use_execution_timer else None
        self._timer_bars_by_symbol = {
            sym.upper(): sorted(list(bars), key=lambda b: b.ts)
            for sym, bars in (timer_bars_by_symbol or {}).items()
            if bars
        }
        # Live-like mode only engages when we actually have real 10s bars to
        # build partial 1m bars from; with synthetic 10s (all stamped at the 1m
        # ts) there is no real 10s clock to iterate, so we fall back to 1m.
        self._live_like_10s = bool(live_like_10s) and bool(self._timer_bars_by_symbol)
        if self._live_like_10s and self._timer is None:
            self._timer = ExecutionTimer(max_wait_bars=3, enabled=True)
            self._use_execution_timer = True
            self._pipeline._execution_timer = self._timer
        self._execution_timer_source = (
            ("real_trades_10s_live_like" if self._live_like_10s else "real_trades_10s")
            if self._timer is not None and self._timer_bars_by_symbol
            else ("synthetic_1m_to_10s" if self._timer is not None else "off")
        )
        if self._timer is not None:
            self._pipeline._execution_timer = self._timer

    def run(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> PipelineBacktestResult:
        ledger = BacktestLedger()
        result = PipelineBacktestResult(
            final_portfolio=self._portfolio,
            execution_timer_source=self._execution_timer_source,
        )
        if self._live_like_10s:
            self._run_live_like(result, ledger, start=start, end=end)
        else:
            self._run_one_minute(result, ledger, start=start, end=end)

        result.trades = ledger.trades
        result.rejected = len(result.rejection_details)
        result.missed_a_plus = self._pipeline.missed_a_plus_report(limit=100)
        result.scorecard = build_backtest_scorecard(
            trades=result.trades,
            total_scan_hits=result.scan_hits,
            total_signals=result.signals,
            total_rejected=result.rejected,
            total_deferred=result.deferred,
            cycle_count=result.cycles,
            missed_a_plus=result.missed_a_plus,
            rejected_by_layer=result.rejected_by_layer,
            rejected_reasons_by_layer=result.rejected_reasons_by_layer,
        )
        result.final_portfolio = self._portfolio
        return result

    def _run_one_minute(
        self,
        result: PipelineBacktestResult,
        ledger: BacktestLedger,
        *,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> None:
        """Original replay: one cycle per closed 1m bar.

        Scans and exits both run on closed 1-minute bars. Faithful for
        gate/funnel analysis; intra-minute price action (the wick that hits a
        stop, the spike a trail would ride) is invisible. The 10s execution
        timer still gives entries partial intra-minute timing.
        """
        times = merge_bar_times(self._bars_by_symbol)
        for now in times:
            if start is not None and now < start:
                continue
            if end is not None and now > end:
                continue
            universe = trim_universe_to_time(
                self._bars_by_symbol,
                now,
                max_bars=self._max_bars_per_symbol,
            )
            if not universe:
                continue
            quotes = {
                sym: [estimate_quote_from_bar(bars[-1], self._broker)]
                for sym, bars in universe.items()
                if bars
            }
            self._feed_execution_timer(
                result,
                ledger,
                universe=universe,
                quotes=quotes,
                bar_by_symbol={sym: bars[-1] for sym, bars in universe.items() if bars},
                now=now,
            )
            cycle = self._pipeline.run_cycle(universe, now=now, quotes=quotes)
            self._record_cycle(result, ledger, cycle, now=now)
            self._queue_deferred(cycle)
            self._record_10s_opportunities(result, ledger, universe=universe, now=now, cycle=cycle)

    def _run_live_like(
        self,
        result: PipelineBacktestResult,
        ledger: BacktestLedger,
        *,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> None:
        """Live-like replay: run the pipeline on a 10s clock.

        Mirrors the live bot, which polls a *partial* (in-progress) 1m bar every
        ~10s and runs exits off the current price. At each 10s step we rebuild
        each symbol's universe as ``closed 1m history + partial 1m bar`` (the
        partial aggregated from the real 10s bars seen so far this minute) and
        run one full cycle — so scans, stops and trails all react intra-minute
        instead of waiting for the 1m close. Requires real 10s bars.
        """
        times10 = merge_bar_times(self._timer_bars_by_symbol)
        for t10 in times10:
            if start is not None and t10 < start:
                continue
            if end is not None and t10 > end:
                continue
            minute_start = t10.replace(second=0, microsecond=0)
            universe = self._live_like_universe(minute_start, t10)
            if not universe:
                continue
            quotes = {
                sym: [estimate_quote_from_bar(bars[-1], self._broker)]
                for sym, bars in universe.items()
                if bars
            }
            # Feed the current 10s bar to the timer first (release entries that
            # were deferred on an earlier cycle) — matching the 1m loop's order
            # of timer-feed before run_cycle queues anything new.
            if self._timer is not None:
                for symbol in list(universe.keys()):
                    ten_sec = self._ten_sec_bar_at(symbol, t10)
                    if ten_sec is None:
                        continue
                    released = self._timer.on_10s_bar(ten_sec)
                    if released is not None:
                        self._execute_timed_signal(
                            released,
                            ten_sec,
                            result,
                            ledger,
                            universe=universe,
                            quotes=quotes,
                        )
            cycle = self._pipeline.run_cycle(universe, now=t10, quotes=quotes)
            self._record_cycle(result, ledger, cycle, now=t10)
            self._queue_deferred(cycle)

    def _live_like_universe(
        self,
        minute_start: datetime,
        t10: datetime,
    ) -> Dict[str, List[Bar]]:
        """Universe at a 10s step: closed 1m history + partial in-progress 1m bar."""
        universe: Dict[str, List[Bar]] = {}
        symbols = set(self._bars_by_symbol) | set(self._timer_bars_by_symbol)
        for sym in symbols:
            one_min = self._bars_by_symbol.get(sym, [])
            closed = [bar for bar in one_min if bar.ts < minute_start]
            partial = self._partial_minute_bar(sym, minute_start, t10)
            if partial is not None:
                bars = closed[-(self._max_bars_per_symbol - 1):] + [partial]
            else:
                bars = closed[-self._max_bars_per_symbol:]
            if bars:
                universe[sym] = bars
        return universe

    def _partial_minute_bar(
        self,
        symbol: str,
        minute_start: datetime,
        t10: datetime,
    ) -> Optional[Bar]:
        """Aggregate the real 10s bars of the current minute up to ``t10``.

        The close advances to the latest 10s close (== current price), so exits
        and chase guards see the live intra-minute price, while high/low capture
        the wick that a 1m-close-only backtest would miss.
        """
        tens = self._timer_bars_by_symbol.get(symbol.upper(), [])
        slice_ = [bar for bar in tens if minute_start <= bar.ts <= t10]
        if not slice_:
            return None
        one_min = self._bars_by_symbol.get(symbol, [])
        timeframe = one_min[0].timeframe if one_min else Timeframe.MIN_1
        return Bar(
            symbol=symbol,
            ts=minute_start,
            open=float(slice_[0].open),
            high=max(float(bar.high) for bar in slice_),
            low=min(float(bar.low) for bar in slice_),
            close=float(slice_[-1].close),
            volume=sum(float(bar.volume or 0.0) for bar in slice_),
            timeframe=timeframe,
        )

    def _ten_sec_bar_at(self, symbol: str, t10: datetime) -> Optional[Bar]:
        """The single real 10s bar for ``symbol`` stamped exactly at ``t10``."""
        for bar in self._timer_bars_by_symbol.get(symbol.upper(), []):
            if bar.ts == t10:
                return bar
        return None

    def _record_10s_opportunities(
        self,
        result: PipelineBacktestResult,
        ledger: BacktestLedger,
        *,
        universe: Dict[str, Sequence[Bar]],
        now: datetime,
        cycle: Optional[PipelineResult] = None,
    ) -> None:
        if not self._timer_bars_by_symbol:
            return
        contexts = self._a_plus_micro_contexts(cycle)
        if self._use_level_reclaim_10s_scout:
            for symbol, rows in self._level_reclaim_contexts(cycle).items():
                contexts.setdefault(symbol, []).extend(rows)
        for symbol, bars in universe.items():
            one_minute = list(bars)
            if len(one_minute) < 6:
                continue
            prior = [bar for bar in one_minute if bar.ts < now]
            if len(prior) < 5:
                continue
            level = max(float(bar.high or 0.0) for bar in prior[-5:])
            if level <= 0:
                continue
            for bar in self._timer_bars_for_minute(symbol, now):
                close = float(bar.close or 0.0)
                if close <= level * 1.003:
                    continue
                if float(bar.volume or 0.0) < 50_000:
                    continue
                if any(
                    row.get("symbol") == symbol
                    and row.get("ts") == bar.ts.isoformat()
                    for row in result.micro_opportunities
                ):
                    continue
                context = self._matching_micro_context(symbol, level, contexts)
                max_after = self._max_price_after(symbol, bar.ts)
                move_after = ((max_after - close) / close * 100.0) if close > 0 and max_after > 0 else 0.0
                row = {
                    "ts": bar.ts.isoformat(),
                    "symbol": symbol,
                    "pattern": (
                        "10s_{}".format(context["pattern"])
                        if context is not None else "10s_level_breakout"
                    ),
                    "price": round(close, 4),
                    "breakout_level": round(level, 4),
                    "volume": round(float(bar.volume or 0.0), 0),
                    "move_after_pct": round(move_after, 2),
                    "max_after": round(max_after, 4),
                    "context_scanner": context.get("scanner", "") if context else "",
                    "context_score": round(float(context.get("score", 0.0)), 3) if context else 0.0,
                    "context_level": round(float(context.get("level", 0.0)), 4) if context else 0.0,
                    "tradeable_context": context is not None,
                    "reason": (
                        "10s intrabar trigger for A+ {} context".format(context["pattern"])
                        if context is not None
                        else "10s breakout only; no current A+ setup context"
                    ),
                }
                result.micro_opportunities.append(row)
                ctx_source = str(context.get("source") or "") if context else ""
                should_execute = context is not None and (
                    (self._use_micro_breakout_scout and ctx_source == "a_plus")
                    or (self._use_level_reclaim_10s_scout and ctx_source == "level_reclaim")
                )
                if should_execute:
                    self._execute_10s_breakout_scout(
                        symbol,
                        bar,
                        level,
                        result,
                        ledger=ledger,
                        prior_bars=prior,
                        context=context,
                    )

    @staticmethod
    def _a_plus_micro_contexts(cycle: Optional[PipelineResult]) -> Dict[str, List[dict]]:
        if cycle is None:
            return {}
        accepted_keys = {
            (
                signal.symbol.upper(),
                getattr(signal.scan_result, "scanner_name", ""),
            )
            for signal in list(cycle.deferred_signals)
            if getattr(signal, "scan_result", None) is not None
        }
        contexts: Dict[str, List[dict]] = {}
        for hit in cycle.scan_hits:
            if (hit.symbol.upper(), hit.scanner_name) not in accepted_keys:
                continue
            criteria = dict(hit.criteria or {})
            setup_tier = str(criteria.get("setup_tier") or "")
            if "A+" not in setup_tier:
                continue
            try:
                score = float(hit.score or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score < 80.0:
                continue
            level = PipelineBacktestDriver._micro_context_level(hit, criteria)
            if level <= 0:
                continue
            pattern = str(criteria.get("pattern") or hit.scanner_name or "a_plus_setup")
            contexts.setdefault(hit.symbol.upper(), []).append({
                "scanner": hit.scanner_name,
                "pattern": pattern,
                "score": score,
                "level": level,
                "criteria": criteria,
                "source": "a_plus",
            })
        return contexts

    @staticmethod
    def _level_reclaim_contexts(cycle: Optional[PipelineResult]) -> Dict[str, List[dict]]:
        """Contexts for the level_reclaim_10s_scout experiment.

        The live verifier holds ``level_breakout_watch`` hits as watch-only
        because the 1-minute bar did not close cleanly above the level. This
        surfaces those hits as contexts so a clean 10s close above the level can
        promote them to a reduced-size, final-guarded entry. Kept conservative:
        only near-the-level hits on real session moves with volume support.
        """
        if cycle is None:
            return {}
        contexts: Dict[str, List[dict]] = {}
        for hit in cycle.scan_hits:
            criteria = dict(hit.criteria or {})
            pattern = str(criteria.get("pattern") or "")
            if pattern != "level_breakout_watch":
                continue
            level = PipelineBacktestDriver._micro_context_level(hit, criteria)
            if level <= 0:
                continue
            try:
                distance_below = float(criteria.get("distance_to_level_pct") or 0.0)
                breakout_pct = float(criteria.get("breakout_pct") or 0.0)
                volume_surge = float(criteria.get("volume_surge") or 0.0)
            except (TypeError, ValueError):
                continue
            # Only promote a near-the-level base on real volume — not a name
            # sitting far under resistance (those stay watch-only).
            if breakout_pct < 0.0 and distance_below > 1.0:
                continue
            if volume_surge < 1.0:
                continue
            try:
                score = float(hit.score or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            contexts.setdefault(hit.symbol.upper(), []).append({
                "scanner": hit.scanner_name,
                "pattern": "level_breakout_reclaim",
                "score": score,
                "level": level,
                "criteria": criteria,
                "source": "level_reclaim",
            })
        return contexts

    @staticmethod
    def _micro_context_level(hit: ScanResult, criteria: dict) -> float:
        for key in (
            "breakout_level",
            "base_high",
            "trigger_price",
            "setup_anchor",
            "hod",
            "session_high",
            "entry_price",
            "close",
        ):
            try:
                value = float(criteria.get(key) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value
        if hit.bars:
            try:
                return float(hit.bars[-1].close or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _matching_micro_context(
        symbol: str,
        level: float,
        contexts: Dict[str, List[dict]],
    ) -> Optional[dict]:
        candidates = contexts.get(symbol.upper()) or []
        if not candidates or level <= 0:
            return None
        ranked = []
        for context in candidates:
            ctx_level = float(context.get("level") or 0.0)
            if ctx_level <= 0:
                continue
            distance = abs(level - ctx_level) / ctx_level
            ranked.append((distance, -float(context.get("score") or 0.0), context))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1]))
        distance, _, context = ranked[0]
        if distance > 0.03:
            return None
        return context

    def _max_price_after(self, symbol: str, ts: datetime) -> float:
        bars = self._bars_by_symbol.get(symbol.upper()) or []
        highs = [float(bar.high or 0.0) for bar in bars if bar.ts >= ts]
        return max(highs) if highs else 0.0

    def _execute_10s_breakout_scout(
        self,
        symbol: str,
        bar: Bar,
        level: float,
        result: PipelineBacktestResult,
        *,
        ledger: BacktestLedger,
        prior_bars: Sequence[Bar],
        context: Optional[dict] = None,
    ) -> None:
        pos = self._portfolio.positions.get(symbol)
        if pos and not pos.is_flat:
            return
        if self._pipeline.exit_manager.tracked.get(symbol) is not None:
            return
        if self._pipeline._symbol_entry_counts.get(symbol, 0) >= self._pipeline._max_entries_per_symbol:
            return
        price = float(bar.close or 0.0)
        if price <= 0 or level <= 0:
            return
        if price > level * 1.035:
            return
        recent_low = min(float(b.low or price) for b in list(prior_bars)[-3:]) if prior_bars else level * 0.98
        stop = round(max(min(recent_low, level) - 0.02, price * 0.94), 4)
        if stop <= 0 or stop >= price:
            return
        risk = price - stop
        if risk <= 0:
            return
        quantity = max(1, min(150, int(35.0 / risk)))
        target = round(price + max(risk * 1.8, price * 0.02), 4)
        bars_for_guard = list(prior_bars[-30:]) + [bar]
        context = context or {}
        context_criteria = dict(context.get("criteria") or {})
        pattern = str(context.get("pattern") or context_criteria.get("pattern") or "level_breakout_reclaim")
        scanner = str(context.get("scanner") or pattern)
        score = float(context.get("score") or 125.0)
        setup_tier = str(context_criteria.get("setup_tier") or "A+ setup")
        entry_tier = str(context_criteria.get("entry_tier") or "")
        hit = ScanResult(
            symbol=symbol,
            scanner_name=scanner,
            ts=bar.ts,
            score=score,
            criteria={
                **context_criteria,
                "pattern": pattern,
                "setup_tier": setup_tier,
                "entry_tier": entry_tier,
                "entry_mode": "ten_second_breakout_scout",
                "breakout_level": round(level, 4),
                "base_high": round(float(context.get("level") or level), 4),
                "close": round(price, 4),
                "volume": float(bar.volume or 0.0),
                "stop_price": stop,
                "size_factor": 0.35,
                "micro_context_scanner": scanner,
            },
            bars=bars_for_guard,
        )
        signal = TradeSignal(
            symbol=symbol,
            action=SignalAction.ENTER_LONG,
            quantity=float(quantity),
            entry_price=price,
            stop_loss=stop,
            take_profit=target,
            max_hold_seconds=600,
            reason="10s breakout scout {} ${:.2f}, stop=${:.2f}, target=${:.2f}".format(
                symbol, price, stop, target,
            ),
            scan_result=hit,
            trend_strength=0.8,
        )
        final_reject = self._pipeline._final_entry_quality_reject(
            signal,
            universe={symbol: bars_for_guard},
            quotes={symbol: [estimate_quote_from_bar(bar, self._broker)]},
            stage="ten_second_breakout_final_guard",
            now=bar.ts,
        )
        if final_reject:
            self._append_rejection(result, {
                "ts": bar.ts.isoformat(),
                "symbol": symbol,
                "blocked_layer": "ten_second_breakout_final_guard",
                "reason": "final entry guard: {}".format(final_reject),
            })
            return
        order = self._pipeline._signal_to_order(signal)
        if order is None:
            return
        if not allow_order(
            order,
            bar,
            self._pipeline.portfolio,
            max_position_shares=self._pipeline._max_position_shares,
            max_order_shares=self._pipeline._max_order_shares,
        ):
            self._append_rejection(result, {
                "ts": bar.ts.isoformat(),
                "symbol": symbol,
                "blocked_layer": "ten_second_breakout_risk",
                "reason": "position_risk_limit",
            })
            return
        fill, status = self._broker.submit(order, bar, self._pipeline.portfolio)
        if status is not OrderStatus.FILLED or fill is None:
            self._append_rejection(result, {
                "ts": bar.ts.isoformat(),
                "symbol": symbol,
                "blocked_layer": "ten_second_breakout_order",
                "reason": "order_{}".format(status.value if status else "not_filled"),
            })
            return
        apply_fill(self._pipeline.portfolio, fill)
        result.fills.append(fill)
        self._pipeline._symbol_entry_counts[symbol] = self._pipeline._symbol_entry_counts.get(symbol, 0) + 1
        self._pipeline.exit_manager.register_from_signal(signal, bar.ts, fill_price=fill.price)
        ledger.record_entry(fill, strategy="ten_second_breakout_scout")
        result.entry_decisions.append({
            "ts": bar.ts.isoformat(),
            "symbol": symbol,
            "stage": "ten_second_breakout_scout",
            "passed": True,
            "blocked_layer": "",
            "reason": "",
            "action": signal.action.value,
            "pattern": "ten_second_breakout_scout",
            "scanner": scanner,
            "setup_tier": setup_tier,
            "entry_tier": entry_tier,
            "price": fill.price,
            "metadata": {"source": "real_trades_10s", "context_pattern": pattern},
        })

    def _queue_deferred(self, cycle: PipelineResult) -> None:
        if self._timer is None:
            return
        for signal in cycle.deferred_signals:
            self._timer.queue(signal)

    def _feed_execution_timer(
        self,
        result: PipelineBacktestResult,
        ledger: BacktestLedger,
        *,
        universe: Dict[str, Sequence[Bar]],
        quotes: Dict[str, Sequence[Quote]],
        bar_by_symbol: Dict[str, Bar],
        now: datetime,
    ) -> None:
        if self._timer is None:
            return
        for symbol, bar in bar_by_symbol.items():
            timer_bars = self._timer_bars_for_minute(symbol, now)
            if not timer_bars:
                timer_bars = self._synthetic_10s_bars(bar)
            for ten_sec in timer_bars:
                released = self._timer.on_10s_bar(ten_sec)
                if released is not None:
                    self._execute_timed_signal(
                        released,
                        ten_sec,
                        result,
                        ledger,
                        universe=universe,
                        quotes=quotes,
                    )

    def _timer_bars_for_minute(self, symbol: str, now: datetime) -> List[Bar]:
        bars = self._timer_bars_by_symbol.get(symbol.upper()) or []
        if not bars:
            return []
        end = now + timedelta(minutes=1)
        return [bar for bar in bars if now <= bar.ts < end]

    @staticmethod
    def _synthetic_10s_bars(bar: Bar) -> List[Bar]:
        """Build deterministic 10s slices from a 1m OHLC bar.

        This exercises the live ExecutionTimer path when real historical 10s
        bars are unavailable. The path is intentionally conservative: it walks
        open → low → high → close so pullback/reclaim logic can cancel weak
        setups instead of seeing only the final close.
        """
        points = [
            float(bar.open),
            float(bar.low),
            (float(bar.low) + float(bar.close)) / 2.0,
            (float(bar.open) + float(bar.close)) / 2.0,
            float(bar.high),
            float(bar.close),
        ]
        bars: List[Bar] = []
        volume = float(bar.volume or 0.0) / 6.0
        prev = float(bar.open)
        for idx, close in enumerate(points):
            open_ = prev if idx > 0 else float(bar.open)
            high = max(open_, close)
            low = min(open_, close)
            if idx == 1:
                low = min(low, float(bar.low))
            if idx == 4:
                high = max(high, float(bar.high))
            bars.append(Bar(
                symbol=bar.symbol,
                ts=bar.ts,
                open=round(open_, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(close, 4),
                volume=volume,
                timeframe=Timeframe.SEC_10,
            ))
            prev = close
        return bars

    def _execute_timed_signal(
        self,
        signal: TradeSignal,
        bar: Bar,
        result: PipelineBacktestResult,
        ledger: BacktestLedger,
        *,
        universe: Dict[str, Sequence[Bar]],
        quotes: Dict[str, Sequence[Quote]],
    ) -> None:
        if signal.action not in (SignalAction.ENTER_LONG, SignalAction.REENTER_LONG):
            return
        final_reject = self._pipeline._final_entry_quality_reject(
            signal,
            universe=universe,
            quotes=quotes,
            stage="timed_entry_final_guard",
            now=bar.ts,
        )
        if final_reject:
            self._append_rejection(result, {
                "ts": bar.ts.isoformat(),
                "symbol": signal.symbol,
                "blocked_layer": "timed_entry_final_guard",
                "reason": "final entry guard: {}".format(final_reject),
            })
            return
        chase_reject = self._pipeline._normal_entry_chase_reject(
            signal,
            universe={signal.symbol: [bar]},
            now=bar.ts,
        )
        if chase_reject:
            self._append_rejection(result, {
                "ts": bar.ts.isoformat(),
                "symbol": signal.symbol,
                "blocked_layer": "timed_entry_chase",
                "reason": chase_reject,
            })
            return
        order = self._pipeline._signal_to_order(signal)
        if order is None:
            return
        criteria = signal.scan_result.criteria if signal.scan_result is not None else {}
        try:
            spread_size_factor = float(criteria.get("spread_size_factor") or 1.0)
        except (TypeError, ValueError):
            spread_size_factor = 1.0
        if 0 < spread_size_factor < 1.0 and order.quantity > 1:
            order = type(order)(
                symbol=order.symbol,
                side=order.side,
                quantity=float(max(1, int(float(order.quantity) * spread_size_factor))),
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                client_order_id=order.client_order_id,
            )
        if not allow_order(
            order,
            bar,
            self._pipeline.portfolio,
            max_position_shares=self._pipeline._max_position_shares,
            max_order_shares=self._pipeline._max_order_shares,
        ):
            self._append_rejection(result, {
                "ts": bar.ts.isoformat(),
                "symbol": signal.symbol,
                "blocked_layer": "timed_entry_risk",
                "reason": "position_risk_limit",
            })
            return
        fill, status = self._broker.submit(order, bar, self._pipeline.portfolio)
        if status is not OrderStatus.FILLED or fill is None:
            self._append_rejection(result, {
                "ts": bar.ts.isoformat(),
                "symbol": signal.symbol,
                "blocked_layer": "timed_entry_order",
                "reason": "timed entry order {}".format(status.value if status else "not_filled"),
            })
            return
        apply_fill(self._pipeline.portfolio, fill)
        result.fills.append(fill)
        self._pipeline._symbol_entry_counts[signal.symbol] = (
            self._pipeline._symbol_entry_counts.get(signal.symbol, 0) + 1
        )
        self._pipeline.exit_manager.register_from_signal(signal, bar.ts, fill_price=fill.price)
        ledger.record_entry(fill, strategy=_entry_strategy_label(signal))

    def _record_cycle(
        self,
        result: PipelineBacktestResult,
        ledger: BacktestLedger,
        cycle: PipelineResult,
        *,
        now: datetime,
    ) -> None:
        result.cycles += 1
        result.scan_hits += len(cycle.scan_hits)
        result.signals += len(cycle.signals)
        result.rejected += int(cycle.rejected_orders)
        result.deferred += len(cycle.deferred_signals)
        for detail in cycle.rejection_details:
            row = self._normalize_rejection_detail(detail, now=now)
            self._append_rejection(result, row)
        for signal in cycle.deferred_signals:
            result.deferred_signals.append(self._deferred_signal_payload(signal, now=now))
        signal_keys = {
            (signal.symbol, getattr(signal.scan_result, "scanner_name", ""))
            for signal in cycle.signals
        }
        for hit in cycle.scan_hits:
            criteria = dict(hit.criteria or {})
            setup_tier = str(criteria.get("setup_tier") or "")
            reject_reason = str(getattr(hit, "_reject_reason", "") or "")
            accepted = (hit.symbol, hit.scanner_name) in signal_keys
            result.scan_events.append({
                "ts": hit.ts.isoformat(),
                "symbol": hit.symbol,
                "scanner": hit.scanner_name,
                "pattern": criteria.get("pattern") or hit.scanner_name,
                "score": round(float(hit.score or 0.0), 3),
                "setup_tier": setup_tier,
                "entry_tier": criteria.get("entry_tier") or "",
                "entry_mode": criteria.get("entry_mode") or "",
                "price": float(
                    criteria.get("close")
                    or criteria.get("entry_price")
                    or (hit.bars[-1].close if hit.bars else 0.0)
                ),
                "status": "accepted" if accepted else ("rejected" if reject_reason else "scan_hit"),
                "blocked_layer": "verifier" if reject_reason else "",
                "reason": reject_reason,
                "criteria": criteria,
                "a_plus": "A+" in setup_tier,
            })
        for decision in cycle.entry_decisions:
            row = dict(decision or {})
            row.setdefault("ts", now.isoformat())
            result.entry_decisions.append(row)
            if not row.get("passed", False):
                self._append_rejection(result, self._normalize_rejection_detail(row, now=now))
        for fill in cycle.fills + cycle.scale_up_fills + cycle.reentry_fills:
            result.fills.append(fill)
            ledger.record_entry(
                fill,
                strategy=cycle.entry_strategies.get(fill.symbol, ""),
            )
        for fill in cycle.exit_fills:
            result.fills.append(fill)
            ledger.record_exit(
                fill,
                reason=cycle.exit_reasons.get(fill.symbol, ""),
            )

    @staticmethod
    def _append_rejection(result: PipelineBacktestResult, row: dict) -> None:
        key = (
            str(row.get("ts") or ""),
            str(row.get("symbol") or ""),
            str(row.get("blocked_layer") or ""),
            str(row.get("reason") or ""),
        )
        if key in result._rejection_keys:
            return
        result._rejection_keys.add(key)
        result.rejection_details.append(row)
        layer = row["blocked_layer"]
        result.rejected_by_layer[layer] = result.rejected_by_layer.get(layer, 0) + 1
        PipelineBacktestDriver._refresh_reason_histogram(result)

    @staticmethod
    def _normalize_rejection_detail(detail: dict, *, now: datetime) -> dict:
        row = dict(detail or {})
        reason = str(row.get("reason") or row.get("reject_reason") or "")
        layer = str(
            row.get("blocked_layer")
            or row.get("layer")
            or row.get("stage")
            or TradingPipeline._blocked_layer(reason)
            or "scanner"
        )
        row["ts"] = str(row.get("ts") or now.isoformat())
        row["symbol"] = str(row.get("symbol") or "")
        row["reason"] = reason
        row["blocked_layer"] = layer
        return row

    @staticmethod
    def _deferred_signal_payload(signal: TradeSignal, *, now: datetime) -> dict:
        scan_result = getattr(signal, "scan_result", None)
        criteria = dict(getattr(scan_result, "criteria", None) or {})
        return {
            "ts": now.isoformat(),
            "symbol": signal.symbol,
            "stage": "deferred",
            "reason": signal.reason,
            "entry_price": float(signal.entry_price or 0.0),
            "quantity": float(signal.quantity or 0.0),
            "scanner": getattr(scan_result, "scanner_name", "") if scan_result else "",
            "pattern": criteria.get("pattern") or (getattr(scan_result, "scanner_name", "") if scan_result else ""),
            "setup_tier": criteria.get("setup_tier") or "",
            "entry_tier": criteria.get("entry_tier") or "",
            "entry_mode": criteria.get("entry_mode") or "",
        }

    @staticmethod
    def _refresh_reason_histogram(result: PipelineBacktestResult, *, limit: int = 5) -> None:
        counts: Dict[Tuple[str, str], int] = {}
        for row in result.rejection_details:
            layer = str(row.get("blocked_layer") or "scanner")
            reason = str(row.get("reason") or "")
            counts[(layer, reason)] = counts.get((layer, reason), 0) + 1

        grouped: Dict[str, List[dict]] = {}
        for (layer, reason), count in counts.items():
            grouped.setdefault(layer, []).append({
                "reason": reason,
                "count": count,
            })
        result.rejected_reasons_by_layer = {
            layer: sorted(rows, key=lambda r: (-int(r["count"]), str(r["reason"])))[:limit]
            for layer, rows in grouped.items()
        }
