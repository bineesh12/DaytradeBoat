from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from daytrading.backtest.broker import BacktestBroker, FillModel
from daytrading.backtest.data_loader import merge_bar_times, trim_universe_to_time
from daytrading.backtest.report import BacktestLedger, build_backtest_scorecard
from daytrading.exits.manager import is_hit_run_strategy
from daytrading.execution.broker import apply_fill
from daytrading.market_calendar import ET
from daytrading.models import Bar, Fill, Order, OrderStatus, PortfolioState, Quote, ScanResult, Side, SignalAction, Timeframe, TradeSignal
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
    _rejection_reason_counts: Dict[Tuple[str, str], int] = field(default_factory=dict, repr=False)


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
        use_breakout_scalp_replay: bool = False,
        use_momentum_burst_replay: bool = False,
        use_momentum_burst_hit_run: bool = False,
        momentum_burst_window_sec: float = 300.0,
        momentum_burst_cooldown_sec: float = 300.0,
        momentum_burst_hit_run_max_entries: int = 1,
        momentum_burst_hit_run_win_cooldown_sec: float = 15.0,
        momentum_burst_hit_run_loss_cooldown_sec: float = 90.0,
        momentum_burst_hit_run_max_hold_sec: float = 45.0,
        momentum_burst_hit_run_reward_risk: float = 1.0,
        momentum_burst_hit_run_stop_after_giveback: bool = True,
        momentum_burst_hit_run_max_giveback: float = 50.0,
        momentum_burst_hit_run_daily_loss_stop: float = 50.0,
        momentum_burst_hit_run_end_et: str = "11:30",
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
        self._use_breakout_scalp_replay = bool(use_breakout_scalp_replay)
        # breakout_scalp_replay fires on every fresh 10s HOD expansion, so left
        # alone it re-chases the same move (a 30s re-entry right after a winner)
        # and stacks late top-chases. The live runner takes one entry per
        # breakout; mirror that with a per-symbol cooldown between replay entries.
        self._breakout_scalp_last_entry: Dict[str, datetime] = {}
        self._breakout_scalp_cooldown_sec = 600.0
        # momentum_burst_replay: the watch-only momentum_burst scanner arms a
        # fixed window; fresh 10s highs inside that window take a quick scalp.
        # Mirrors the live runner's _process_momentum_burst_scalps path.
        self._use_momentum_burst_replay = bool(use_momentum_burst_replay)
        self._use_momentum_burst_hit_run = bool(use_momentum_burst_hit_run)
        self._momentum_burst_window_sec = float(momentum_burst_window_sec)
        self._momentum_burst_cooldown_sec = float(momentum_burst_cooldown_sec)
        self._momentum_burst_armed: Dict[str, datetime] = {}
        self._momentum_burst_window_high: Dict[str, float] = {}
        self._momentum_burst_session_anchor_high: Dict[str, float] = {}
        self._momentum_burst_last_entry: Dict[str, datetime] = {}
        self._mb_hit_run_counts: Dict[str, int] = {}
        self._mb_hit_run_block_until: Dict[str, datetime] = {}
        self._mb_hit_run_max_entries = int(momentum_burst_hit_run_max_entries)
        self._mb_hit_run_win_cooldown_sec = float(momentum_burst_hit_run_win_cooldown_sec)
        self._mb_hit_run_loss_cooldown_sec = float(momentum_burst_hit_run_loss_cooldown_sec)
        self._mb_hit_run_max_hold_sec = float(momentum_burst_hit_run_max_hold_sec)
        self._mb_hit_run_end_et = str(momentum_burst_hit_run_end_et or "")
        self._mb_hit_run_stop_after_giveback = bool(momentum_burst_hit_run_stop_after_giveback)
        self._mb_hit_run_max_giveback = float(momentum_burst_hit_run_max_giveback)
        self._mb_hit_run_daily_loss_stop = float(momentum_burst_hit_run_daily_loss_stop)
        self._mb_hit_run_symbol_pnl: Dict[str, float] = {}
        self._mb_hit_run_symbol_peak_pnl: Dict[str, float] = {}
        self._mb_hit_run_day_blocked: Dict[str, str] = {}
        # symbol -> {ts, breakout_close} of a fresh 10s high awaiting next-bar
        # confirmation; we never buy the spike bar itself.
        self._momentum_burst_pending: Dict[str, Dict[str, Any]] = {}
        # Entry-quality knobs (tunable):
        #  (A) confirm bar volume must be >= ratio * breakout bar volume
        #  (C) confirm close must be within chase_cap above the breakout close
        #  (B) stop sits below the recent 10s swing low (structure), not a flat %
        self._mb_confirm_min_vol_ratio = 0.5
        self._mb_chase_cap_pct = 0.03
        self._mb_structure_stop_lookback = 4
        self._mb_structure_max_risk_pct = 0.05
        # Violent 10s tape is where 1:1 stops slip and winners get clipped.
        # Keep momentum-burst entries to smooth recent tape until the sweep says
        # otherwise.
        self._mb_smooth_max_median_range_pct = 2.0
        # Warrior-style momentum bursts are often not smooth; allow a separate
        # reduced-size lane only when the violent tape is also very liquid.
        self._mb_violent_max_median_range_pct = 9.0
        self._mb_violent_chase_cap_pct = 0.08
        self._mb_violent_min_latest_volume = 50_000.0
        self._mb_violent_min_recent_volume = 150_000.0
        self._mb_violent_min_day_volume = 500_000.0
        self._mb_violent_size_factor = 0.35
        # Simple symmetric 1:1 bracket (full exit at +1R or -1R, NO partials/
        # trailing) — replaces the shared exit manager for momentum_burst only.
        # symbol -> {stop, target, qty, entry, ts, max_hold}
        self._mb_bracket: Dict[str, Dict[str, Any]] = {}
        self._mb_reward_risk = 1.0
        self._mb_hit_run_reward_risk = float(momentum_burst_hit_run_reward_risk)
        self._timer = ExecutionTimer(max_wait_bars=3, enabled=True) if self._use_execution_timer else None
        self._timer_bars_by_symbol = {
            sym.upper(): sorted(list(bars), key=lambda b: b.ts)
            for sym, bars in (timer_bars_by_symbol or {}).items()
            if bars
        }
        self._timer_bar_by_ts: Dict[str, Dict[datetime, Bar]] = {
            sym: {bar.ts: bar for bar in bars}
            for sym, bars in self._timer_bars_by_symbol.items()
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
        self._wire_bar_aggregator()

    def _wire_bar_aggregator(self) -> None:
        """Give the backtest a 5-minute bar context, like the live runner.

        Without this the entry scorer always sees ``no5m=5`` (no 5-minute
        bars) and scores ~5 points lower than live, which silently rejects
        otherwise-passing setups. We build 5m bars from the 1m universe each
        cycle and attach the aggregator to the pipeline + verifiers, mirroring
        ``runner.py`` so the backtest scores entries on the same inputs.
        """
        from daytrading.data.bar_aggregator import BarAggregator

        self._bar_aggregator = BarAggregator()
        self._pipeline._bar_aggregator = self._bar_aggregator
        for verifier in getattr(self._pipeline, "_verifiers", {}).values():
            if hasattr(verifier, "_bar_aggregator"):
                verifier._bar_aggregator = self._bar_aggregator

    def _refresh_5m_context(self, universe: Dict[str, Sequence[Bar]]) -> None:
        """Rebuild 5m bars from the current 1m universe before a scan cycle."""
        if self._bar_aggregator is None:
            return
        self._bar_aggregator.update_all_5m({
            sym: list(bars) for sym, bars in universe.items() if bars
        })

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
            self._refresh_5m_context(universe)
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
            self._refresh_5m_context(universe)
            cycle = self._pipeline.run_cycle(universe, now=t10, quotes=quotes)
            self._record_cycle(result, ledger, cycle, now=t10)
            self._queue_deferred(cycle)
            if self._use_breakout_scalp_replay:
                self._maybe_execute_breakout_scalp_replay(
                    result,
                    ledger,
                    universe=universe,
                    quotes=quotes,
                    now=t10,
                )
            if self._use_momentum_burst_replay or self._use_momentum_burst_hit_run:
                self._process_mb_brackets(result, ledger, t10)
                self._arm_momentum_burst_from_cycle(cycle, t10)
                self._maybe_execute_momentum_burst_replay(
                    result,
                    ledger,
                    universe=universe,
                    quotes=quotes,
                    now=t10,
                )

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
        return self._timer_bar_by_ts.get(symbol.upper(), {}).get(t10)

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

    def _maybe_execute_breakout_scalp_replay(
        self,
        result: PipelineBacktestResult,
        ledger: BacktestLedger,
        *,
        universe: Dict[str, Sequence[Bar]],
        quotes: Dict[str, Sequence[Quote]],
        now: datetime,
    ) -> None:
        """Replay the runner's HOD-alert breakout scalp path on 10s bars.

        The live runner fires this path from HOD tick alerts and true tick
        prints. Historical backtests do not have that runner loop, so without a
        replay hook a profitable paper quick-scalp can look like "0 trades".
        Keep this intentionally narrow: fresh 10s HOD expansion, real volume,
        final entry guard, risk/order checks, then normal exit manager handling.
        """
        for symbol, bars in universe.items():
            if self._portfolio.positions.get(symbol) and not self._portfolio.positions[symbol].is_flat:
                continue
            if self._pipeline.exit_manager.tracked.get(symbol) is not None:
                continue
            if self._pipeline._symbol_entry_counts.get(symbol, 0) >= self._pipeline._max_entries_per_symbol:
                continue
            last_entry = self._breakout_scalp_last_entry.get(symbol)
            if last_entry is not None and (now - last_entry).total_seconds() < self._breakout_scalp_cooldown_sec:
                continue
            ten_sec = self._ten_sec_bar_at(symbol, now)
            if ten_sec is None:
                continue
            ten_second_reject = self._breakout_scalp_10s_quality_reject(symbol, ten_sec)
            if ten_second_reject is not None:
                self._append_rejection(result, {
                    "ts": now.isoformat(),
                    "symbol": symbol,
                    "blocked_layer": "breakout_scalp_replay_10s",
                    "reason": ten_second_reject,
                })
                continue
            signal = self._breakout_scalp_replay_signal(symbol, ten_sec, list(bars))
            if signal is None:
                continue
            minute_start = now.replace(second=0, microsecond=0)
            guard_bars = [b for b in list(bars) if b.ts < minute_start] or list(bars)
            final_reject = self._pipeline._final_entry_quality_reject(
                signal,
                universe={symbol: guard_bars},
                quotes=quotes,
                stage="breakout_scalp_replay_final_guard",
                now=now,
            )
            if final_reject:
                self._append_rejection(result, {
                    "ts": now.isoformat(),
                    "symbol": symbol,
                    "blocked_layer": "breakout_scalp_replay_final_guard",
                    "reason": "final entry guard: {}".format(final_reject),
                })
                continue
            order = self._pipeline._signal_to_order(signal)
            if order is None:
                continue
            if not allow_order(
                order,
                ten_sec,
                self._pipeline.portfolio,
                max_position_shares=self._pipeline._max_position_shares,
                max_order_shares=self._pipeline._max_order_shares,
            ):
                self._append_rejection(result, {
                    "ts": now.isoformat(),
                    "symbol": symbol,
                    "blocked_layer": "breakout_scalp_replay_risk",
                    "reason": "position_risk_limit",
                })
                continue
            fill, status = self._broker.submit(order, ten_sec, self._pipeline.portfolio)
            if status is not OrderStatus.FILLED or fill is None:
                self._append_rejection(result, {
                    "ts": now.isoformat(),
                    "symbol": symbol,
                    "blocked_layer": "breakout_scalp_replay_order",
                    "reason": "order_{}".format(status.value if status else "not_filled"),
                })
                continue
            apply_fill(self._pipeline.portfolio, fill)
            result.fills.append(fill)
            self._pipeline._symbol_entry_counts[symbol] = self._pipeline._symbol_entry_counts.get(symbol, 0) + 1
            self._pipeline.exit_manager.register_from_signal(signal, now, fill_price=fill.price)
            self._breakout_scalp_last_entry[symbol] = now
            ledger.record_entry(fill, strategy="breakout_scalp_replay")
            result.entry_decisions.append({
                "ts": now.isoformat(),
                "symbol": symbol,
                "stage": "breakout_scalp_replay",
                "passed": True,
                "blocked_layer": "",
                "reason": "",
                "action": signal.action.value,
                "pattern": "breakout_scalp",
                "scanner": "breakout_scalp_replay",
                "setup_tier": "A+ setup",
                "entry_tier": "quick_scalp",
                "price": fill.price,
                "metadata": {"source": "real_trades_10s_live_like"},
            })
            break

    def _breakout_scalp_10s_quality_reject(self, symbol: str, bar: Bar) -> Optional[str]:
        """Extra replay/live parity checks for unstable quick-scalp 10s bars."""
        history = [
            b for b in self._timer_bars_by_symbol.get(symbol.upper(), [])
            if b.ts <= bar.ts
        ]
        if not history:
            return "waiting for 10s confirmation"
        latest = history[-1]
        bar_range = float(latest.high or 0.0) - float(latest.low or 0.0)
        if bar_range > 0:
            close_location = (float(latest.close or 0.0) - float(latest.low or 0.0)) / bar_range
            if close_location < 0.65:
                return "10s confirmation weak close ({:.0%} location)".format(close_location)
            price = float(latest.close or 0.0)
            range_pct = (bar_range / price) if price > 0 else 0.0
            if range_pct >= 0.06 and close_location < 0.75:
                return "10s breakout candle too volatile without strong close ({:.0%} location, {:.1%} range)".format(
                    close_location,
                    range_pct,
                )
        if len(history) >= 2:
            prev = history[-2]
            if float(prev.close or 0.0) > 0 and float(latest.close or 0.0) < float(prev.close) * 0.998:
                return "10s confirmation faded below prior close"
            if float(prev.high or 0.0) > 0 and float(latest.high or 0.0) <= float(prev.high) * 1.001:
                return "10s confirmation no expansion"
            prev_volume = float(prev.volume or 0.0)
            latest_volume = float(latest.volume or 0.0)
            if prev_volume > 0 and latest_volume < prev_volume * 0.5:
                return "10s confirmation volume faded {:.0f} < 50% prior {:.0f}".format(
                    latest_volume,
                    prev_volume,
                )
        for recent in history[-4:-1]:
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

    def _breakout_scalp_replay_signal(
        self,
        symbol: str,
        bar: Bar,
        bars: Sequence[Bar],
    ) -> Optional[TradeSignal]:
        price = float(bar.close or 0.0)
        if price < 1.5 or price > 20.0:
            return None
        ten_history = [
            b for b in self._timer_bars_by_symbol.get(symbol.upper(), [])
            if b.ts <= bar.ts
        ]
        if len(ten_history) < 12:
            return None
        prior = ten_history[:-1]
        prior_hod = max(float(b.high or 0.0) for b in prior[-36:])
        if prior_hod <= 0 or float(bar.high or 0.0) < prior_hod * 1.05:
            return None
        if price < prior_hod * 1.03:
            return None
        if float(bar.close or 0.0) <= float(bar.open or 0.0):
            return None
        latest_volume = float(bar.volume or 0.0)
        recent_volume = sum(float(b.volume or 0.0) for b in ten_history[-3:])
        day_volume = sum(float(b.volume or 0.0) for b in ten_history)
        if day_volume < 500_000 or latest_volume < 100_000 or recent_volume < 350_000:
            return None
        closed_bars = [b for b in bars if b.ts <= bar.ts.replace(second=0, microsecond=0)]
        if len(closed_bars) < 3:
            return None
        session_open = float(closed_bars[0].open or 0.0)
        day_change = ((price - session_open) / session_open * 100.0) if session_open > 0 else 0.0
        if day_change < 30.0:
            return None
        risk = max(price * 0.016, 0.08)
        risk = min(risk, price * 0.04)
        stop_price = round(price - risk, 2)
        target_price = round(price + max(risk * 1.25, price * 0.02), 2)
        if stop_price <= 0 or stop_price >= price:
            return None
        max_order = int(getattr(self._pipeline, "_max_order_shares", 750) or 750)
        quantity = max(1, min(750, max_order, int(50.0 / (price - stop_price))))
        hit = ScanResult(
            symbol=symbol,
            scanner_name="breakout_scalp_replay",
            ts=bar.ts,
            score=125.0,
            criteria={
                "pattern": "breakout_scalp",
                "setup_tier": "A+ setup",
                "entry_tier": "quick_scalp",
                "entry_mode": "breakout_scalp_replay",
                "breakout_level": round(prior_hod, 4),
                "day_volume": round(day_volume, 0),
                "recent_volume": round(recent_volume, 0),
                "latest_volume": round(latest_volume, 0),
                "stop_price": stop_price,
                "size_factor": 1.0,
            },
            bars=list(closed_bars[-30:]) + [bar],
        )
        return TradeSignal(
            symbol=symbol,
            action=SignalAction.ENTER_LONG,
            quantity=float(quantity),
            entry_price=price,
            stop_loss=stop_price,
            take_profit=target_price,
            max_hold_seconds=90,
            reason="Quick Momentum Scalp {} ${:.2f}, stop=${:.2f}, target=${:.2f} (10s replay)".format(
                symbol, price, stop_price, target_price,
            ),
            scan_result=hit,
            trend_strength=0.8,
        )

    def _arm_momentum_burst_from_cycle(self, cycle: PipelineResult, now: datetime) -> None:
        """Arm a fixed window when the momentum_burst scanner prints a hit.

        Mirrors ``AlpacaRunner._maybe_arm_momentum_burst_scalp``: the watch-only
        momentum_burst scanner is the trigger; arming seeds a window-high so the
        first fresh 10s high inside the window can take a scalp.
        """
        for hit in getattr(cycle, "scan_hits", []) or []:
            pattern = str((hit.criteria or {}).get("pattern") or hit.scanner_name or "")
            if hit.scanner_name != "momentum_burst" and pattern != "momentum_burst":
                continue
            sym = hit.symbol.upper()
            high = 0.0
            try:
                if hit.bars:
                    high = max(float(b.high or 0.0) for b in hit.bars[-3:])
            except Exception:
                high = 0.0
            if high <= 0:
                try:
                    high = float((hit.criteria or {}).get("close") or 0.0)
                except (TypeError, ValueError):
                    high = 0.0
            if high <= 0:
                continue
            self._momentum_burst_armed[sym] = now
            self._momentum_burst_window_high[sym] = max(
                high, float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0)
            )
            self._momentum_burst_session_anchor_high.setdefault(sym, high)

    def _process_mb_brackets(
        self,
        result: PipelineBacktestResult,
        ledger: BacktestLedger,
        now: datetime,
    ) -> None:
        """Exit momentum_burst positions on a simple full-size 1:1 bracket.

        On each 10s bar: if the bar's low hits the stop we exit the WHOLE
        position at the stop; if its high hits the target we exit the whole
        position at the target; otherwise after max_hold we time-exit at the
        close. No partials, no trailing — a clean symmetric scalp. Stop is
        checked first (conservative) when a single bar spans both levels.
        """
        for sym, br in list(self._mb_bracket.items()):
            bar = self._ten_sec_bar_at(sym, now)
            exit_price = None
            reason = None
            if bar is not None and float(bar.low or 0.0) <= br["stop"]:
                exit_price, reason = br["stop"], "mb_bracket_stop"
            elif bar is not None and float(bar.high or 0.0) >= br["target"]:
                exit_price, reason = br["target"], "mb_bracket_target"
            elif (now - br["ts"]).total_seconds() >= br["max_hold"]:
                exit_price = float(bar.close) if bar is not None else br["entry"]
                reason = "mb_bracket_time"
            if exit_price is None:
                continue
            price_bar = Bar(
                symbol=sym,
                ts=now,
                open=exit_price,
                high=exit_price,
                low=exit_price,
                close=exit_price,
                volume=float(getattr(bar, "volume", 0.0) or 0.0),
                timeframe=Timeframe.SEC_10,
            )
            order = Order(symbol=sym, side=Side.SELL, quantity=br["qty"], limit_price=exit_price)
            fill, status = self._broker.submit(order, price_bar, self._pipeline.portfolio)
            if status is not OrderStatus.FILLED or fill is None:
                continue
            apply_fill(self._pipeline.portfolio, fill)
            result.fills.append(fill)
            self._mb_bracket.pop(sym, None)
            self._pipeline._exit_cooldowns[sym] = now
            strategy = str(br.get("strategy") or "momentum_burst_replay")
            label = "Momentum Burst Hit-Run" if is_hit_run_strategy(strategy) else "Momentum Burst Scalp"
            ledger.record_exit(fill, reason="{}: {}".format(reason, label))
            if is_hit_run_strategy(strategy):
                last_trade = ledger.trades[-1] if ledger.trades else {}
                self._record_mb_hit_run_pnl(sym, float(last_trade.get("pnl") or 0.0))
                cooldown = (
                    self._mb_hit_run_win_cooldown_sec
                    if reason == "mb_bracket_target"
                    else self._mb_hit_run_loss_cooldown_sec
                )
                self._mb_hit_run_block_until[sym] = now + timedelta(seconds=cooldown)

    def _record_mb_hit_run_pnl(self, symbol: str, pnl: float) -> str:
        sym = symbol.upper()
        current = float(self._mb_hit_run_symbol_pnl.get(sym, 0.0) or 0.0) + float(pnl or 0.0)
        self._mb_hit_run_symbol_pnl[sym] = current
        peak = max(float(self._mb_hit_run_symbol_peak_pnl.get(sym, 0.0) or 0.0), current)
        self._mb_hit_run_symbol_peak_pnl[sym] = peak

        loss_stop = max(0.0, float(self._mb_hit_run_daily_loss_stop or 0.0))
        if loss_stop > 0 and current <= -loss_stop:
            reason = "daily hit-run loss ${:.2f} reached stop ${:.2f}".format(abs(current), loss_stop)
            self._mb_hit_run_day_blocked[sym] = reason
            return reason

        giveback_stop = max(0.0, float(self._mb_hit_run_max_giveback or 0.0))
        giveback = peak - current
        if (
            self._mb_hit_run_stop_after_giveback
            and peak > 0
            and giveback_stop > 0
            and giveback >= giveback_stop
        ):
            reason = "gave back ${:.2f} from hit-run peak ${:.2f}".format(giveback, peak)
            self._mb_hit_run_day_blocked[sym] = reason
            return reason
        return ""

    def _maybe_execute_momentum_burst_replay(
        self,
        result: PipelineBacktestResult,
        ledger: BacktestLedger,
        *,
        universe: Dict[str, Sequence[Bar]],
        quotes: Dict[str, Sequence[Quote]],
        now: datetime,
    ) -> None:
        """Scalp fresh 10s highs while a momentum_burst window is armed.

        The same shape/guard/risk/order/exit path as breakout_scalp_replay; only
        the trigger differs — a new high inside the scanner-armed window rather
        than a +30% runner HOD expansion.
        """
        hit_run = bool(self._use_momentum_burst_hit_run)
        strategy_label = "momentum_burst_hit_run" if hit_run else "momentum_burst_replay"
        pattern_label = "momentum_burst_hit_run" if hit_run else "momentum_burst_scalp"
        for sym in list(self._momentum_burst_armed.keys()):
            armed_at = self._momentum_burst_armed.get(sym)
            if armed_at is None:
                continue
            if (now - armed_at).total_seconds() > self._momentum_burst_window_sec:
                self._momentum_burst_armed.pop(sym, None)
                self._momentum_burst_window_high.pop(sym, None)
                self._momentum_burst_pending.pop(sym, None)
                self._mb_hit_run_counts.pop(sym, None)
                self._mb_hit_run_block_until.pop(sym, None)
                continue
            if self._portfolio.positions.get(sym) and not self._portfolio.positions[sym].is_flat:
                continue
            if self._pipeline.exit_manager.tracked.get(sym) is not None:
                continue
            if hit_run:
                day_block = self._mb_hit_run_day_blocked.get(sym)
                if day_block:
                    self._momentum_burst_pending.pop(sym, None)
                    self._append_rejection(result, {
                        "ts": now.isoformat(),
                        "symbol": sym,
                        "blocked_layer": "{}_daily_stop".format(strategy_label),
                        "reason": day_block,
                    })
                    continue
                if self._mb_hit_run_counts.get(sym, 0) >= self._mb_hit_run_max_entries:
                    self._append_rejection(result, {
                        "ts": now.isoformat(),
                        "symbol": sym,
                        "blocked_layer": "{}_max_entries".format(strategy_label),
                        "reason": "max hit-run entries reached ({})".format(self._mb_hit_run_max_entries),
                    })
                    continue
                block_until = self._mb_hit_run_block_until.get(sym)
                if block_until is not None and now < block_until:
                    continue
            elif self._pipeline._symbol_entry_counts.get(sym, 0) >= self._pipeline._max_entries_per_symbol:
                continue
            last_entry = self._momentum_burst_last_entry.get(sym)
            if (
                not hit_run
                and last_entry is not None
                and (now - last_entry).total_seconds() < self._momentum_burst_cooldown_sec
            ):
                continue
            ten_sec = self._ten_sec_bar_at(sym, now)
            if ten_sec is None:
                continue
            if hit_run:
                if not self._momentum_burst_hit_run_time_allowed(now):
                    self._momentum_burst_armed.pop(sym, None)
                    self._momentum_burst_window_high.pop(sym, None)
                    self._momentum_burst_pending.pop(sym, None)
                    self._append_rejection(result, {
                        "ts": now.isoformat(),
                        "symbol": sym,
                        "blocked_layer": "{}_time_window".format(strategy_label),
                        "reason": "outside hit-run time window ending {} ET".format(
                            self._mb_hit_run_end_et,
                        ),
                    })
                    continue
                reentry = self._mb_hit_run_counts.get(sym, 0) > 0
                anchor_high = float(self._momentum_burst_session_anchor_high.get(sym, 0.0) or 0.0)
                current_close = float(ten_sec.close or 0.0)
                stop_reason = self._momentum_burst_stop_trading_reason(sym, now)
                if stop_reason:
                    continuation_ok, _continuation_reason, _continuation_meta = (
                        self._momentum_burst_continuation_base_ok(sym, now)
                    )
                    if continuation_ok:
                        stop_reason = ""
                if stop_reason:
                    self._momentum_burst_pending.pop(sym, None)
                    self._append_rejection(result, {
                        "ts": now.isoformat(),
                        "symbol": sym,
                        "blocked_layer": "{}_stop_trading".format(strategy_label),
                        "reason": stop_reason,
                    })
                    continue
                if anchor_high > 0 and current_close > anchor_high * 1.5:
                    continuation_ok, continuation_reason, continuation_meta = (
                        self._momentum_burst_continuation_base_ok(sym, now)
                    )
                    if not continuation_ok:
                        self._append_rejection(result, {
                            "ts": now.isoformat(),
                            "symbol": sym,
                            "blocked_layer": "{}_extension".format(strategy_label),
                            "reason": "extended without fresh continuation base: {}".format(
                                continuation_reason,
                            ),
                            "metadata": continuation_meta,
                        })
                        continue
                if reentry:
                    continuation_ok, continuation_reason, continuation_meta = (
                        self._momentum_burst_continuation_base_ok(sym, now)
                    )
                    if not continuation_ok:
                        self._append_rejection(result, {
                            "ts": now.isoformat(),
                            "symbol": sym,
                            "blocked_layer": "{}_reentry_base".format(strategy_label),
                            "reason": "re-entry needs fresh micro-base: {}".format(
                                continuation_reason,
                            ),
                            "metadata": continuation_meta,
                        })
                        continue
            current_high = float(ten_sec.high or 0.0)
            window_high = float(self._momentum_burst_window_high.get(sym, 0.0) or 0.0)

            # Confirmation-bar rule: a fresh 10s high arms a pending breakout but
            # we never buy that spike bar. Entry waits for the NEXT 10s bar to
            # prove continuation: green, holding the breakout close, and trading
            # through the breakout high. A sideways hold under the spike high is
            # not enough for a hit-and-run entry.
            pend = self._momentum_burst_pending.get(sym)
            post_blowoff_micro_base = False
            if pend is not None and (now - pend["ts"]).total_seconds() > 30.0:
                self._momentum_burst_pending.pop(sym, None)
                pend = None
            if pend is not None:
                if now <= pend["ts"]:  # still the breakout bar — wait for next
                    continue
                breakout_close = float(pend["breakout_close"])
                breakout_high = float(pend.get("breakout_high") or breakout_close)
                confirm_close = float(ten_sec.close or 0.0)
                confirm_high = float(ten_sec.high or 0.0)
                confirm_low = float(ten_sec.low or 0.0)
                continuation_buffer = max(0.005, breakout_high * 0.001)
                confirm_range = max(confirm_high - confirm_low, 0.0)
                close_location = (
                    (confirm_close - confirm_low) / confirm_range if confirm_range > 0 else 0.0
                )
                confirmed = confirm_close >= float(ten_sec.open or 0.0) and confirm_close >= breakout_close
                reject_reason = None
                if not confirmed:
                    reject_reason = "breakout not confirmed by next 10s bar"
                elif hit_run and confirm_high < breakout_high + continuation_buffer:
                    reject_reason = (
                        "confirm bar did not break continuation high "
                        "({:.2f} <= {:.2f})"
                    ).format(confirm_high, breakout_high)
                elif hit_run and close_location < 0.65:
                    reject_reason = "confirm bar did not close with strength"
                violent_ok, _violent_meta = self._momentum_burst_violent_liquid_ok(
                    sym, now, median_range=None,
                )
                reentry = hit_run and self._mb_hit_run_counts.get(sym, 0) > 0
                volume_ratio = (
                    self._mb_confirm_min_vol_ratio
                    if reentry
                    else (0.25 if hit_run and violent_ok else self._mb_confirm_min_vol_ratio)
                )
                chase_cap = (
                    self._mb_chase_cap_pct
                    if reentry
                    else (self._mb_violent_chase_cap_pct if hit_run and violent_ok else self._mb_chase_cap_pct)
                )
                # (A) Confirmation volume — the next bar must show real demand,
                # not a quiet drift up, or the move has no follow-through.
                if (
                    reject_reason is None
                    and float(ten_sec.volume or 0.0) < volume_ratio * float(pend.get("breakout_volume") or 0.0)
                ):
                    reject_reason = "confirm-bar volume too light (no follow-through)"
                # (C) Chase cap — don't buy a confirm bar that already ran far
                # above the breakout; that is buying the extension.
                if (
                    reject_reason is None
                    and breakout_close > 0
                    and confirm_close > breakout_close * (1.0 + chase_cap)
                ):
                    reject_reason = "chasing: confirm {:.2f} >{:.0%} above breakout {:.2f}".format(
                        confirm_close, chase_cap, breakout_close)
                post_blowoff_micro_base = bool(hit_run and pend.get("reset_from_stale_high"))
                self._momentum_burst_pending.pop(sym, None)
                if reject_reason is not None:
                    self._append_rejection(result, {
                        "ts": now.isoformat(),
                        "symbol": sym,
                        "blocked_layer": "{}_unconfirmed".format(strategy_label),
                        "reason": reject_reason,
                    })
                    continue
                # confirmed — fall through to build/execute the scalp on this bar
            else:
                if current_high <= 0:
                    continue
                if current_high <= window_high:
                    if hit_run and window_high > current_high * 1.08:
                        continuation_ok, _continuation_reason, continuation_meta = (
                            self._momentum_burst_continuation_base_ok(sym, now)
                        )
                        if continuation_ok:
                            self._momentum_burst_window_high[sym] = current_high
                            self._momentum_burst_pending[sym] = {
                                "ts": now,
                                "breakout_close": float(ten_sec.close or 0.0),
                                "breakout_high": current_high,
                                "breakout_volume": float(ten_sec.volume or 0.0),
                                "reset_from_stale_high": round(window_high, 4),
                                "base_high": continuation_meta.get("base_high"),
                                "base_low": continuation_meta.get("base_low"),
                            }
                    continue
                # Fresh high — arm pending, update window high, do not buy yet.
                self._momentum_burst_window_high[sym] = current_high
                self._momentum_burst_pending[sym] = {
                    "ts": now,
                    "breakout_close": float(ten_sec.close or 0.0),
                    "breakout_high": current_high,
                    "breakout_volume": float(ten_sec.volume or 0.0),
                }
                continue
            bars = universe.get(sym)
            if not bars:
                continue
            smooth, median_range = self._momentum_burst_10s_tape_is_smooth(sym, now)
            violent_ok, violent_meta = self._momentum_burst_violent_liquid_ok(
                sym, now, median_range=median_range,
            )
            if not smooth and not (hit_run and violent_ok):
                self._append_rejection(result, {
                    "ts": now.isoformat(),
                    "symbol": sym,
                    "blocked_layer": "{}_smoothness".format(strategy_label),
                    "reason": "10s tape too gappy (median range {:.2f}% > {:.2f}%)".format(
                        median_range,
                        self._mb_smooth_max_median_range_pct,
                    ),
                })
                continue
            signal = self._momentum_burst_replay_signal(
                sym,
                ten_sec,
                list(bars),
                hit_run=hit_run,
                violent_liquid=bool(hit_run and violent_ok),
                post_blowoff_micro_base=post_blowoff_micro_base,
            )
            if signal is None:
                continue
            minute_start = now.replace(second=0, microsecond=0)
            guard_bars = [b for b in list(bars) if b.ts < minute_start] or list(bars)
            final_reject = self._pipeline._final_entry_quality_reject(
                signal,
                universe={sym: guard_bars},
                quotes=quotes,
                stage="{}_final_guard".format(strategy_label),
                now=now,
            )
            if final_reject:
                self._append_rejection(result, {
                    "ts": now.isoformat(),
                    "symbol": sym,
                    "blocked_layer": "{}_final_guard".format(strategy_label),
                    "reason": "final entry guard: {}".format(final_reject),
                })
                continue
            order = self._pipeline._signal_to_order(signal)
            if order is None:
                continue
            if not allow_order(
                order,
                ten_sec,
                self._pipeline.portfolio,
                max_position_shares=self._pipeline._max_position_shares,
                max_order_shares=self._pipeline._max_order_shares,
            ):
                self._append_rejection(result, {
                    "ts": now.isoformat(),
                    "symbol": sym,
                    "blocked_layer": "{}_risk".format(strategy_label),
                    "reason": "position_risk_limit",
                })
                continue
            fill, status = self._broker.submit(order, ten_sec, self._pipeline.portfolio)
            if status is not OrderStatus.FILLED or fill is None:
                self._append_rejection(result, {
                    "ts": now.isoformat(),
                    "symbol": sym,
                    "blocked_layer": "{}_order".format(strategy_label),
                    "reason": "order_{}".format(status.value if status else "not_filled"),
                })
                continue
            apply_fill(self._pipeline.portfolio, fill)
            result.fills.append(fill)
            self._pipeline._symbol_entry_counts[sym] = self._pipeline._symbol_entry_counts.get(sym, 0) + 1
            # Own the exit with a simple full-position 1:1 bracket instead of the
            # shared exit manager (which scales out partials + trails). Not
            # registering with exit_manager means run_cycle won't touch it.
            stop_price = float(signal.stop_loss)
            target_price = float(signal.take_profit)
            if hit_run:
                fill_price = float(fill.price)
                risk = max(fill_price - stop_price, fill_price * 0.02, 0.06)
                stop_price = round(fill_price - risk, 4)
                target_price = round(fill_price + risk, 4)
            self._mb_bracket[sym] = {
                "stop": stop_price,
                "target": target_price,
                "qty": float(fill.quantity),
                "entry": float(fill.price),
                "ts": now,
                "max_hold": float(signal.max_hold_seconds or 90),
                "strategy": strategy_label,
            }
            if hit_run:
                self._mb_hit_run_counts[sym] = self._mb_hit_run_counts.get(sym, 0) + 1
            else:
                self._momentum_burst_last_entry[sym] = now
            ledger.record_entry(fill, strategy=strategy_label)
            result.entry_decisions.append({
                "ts": now.isoformat(),
                "symbol": sym,
                "stage": strategy_label,
                "passed": True,
                "blocked_layer": "",
                "reason": "",
                "action": signal.action.value,
                "pattern": pattern_label,
                "scanner": strategy_label,
                "setup_tier": "A+ setup",
                "entry_tier": "quick_scalp",
                "price": fill.price,
                "metadata": {
                    "source": "real_trades_10s_live_like",
                    "violent_liquid": bool(hit_run and violent_ok),
                    **(violent_meta if hit_run and violent_ok else {}),
                },
            })
            break

    def _momentum_burst_hit_run_time_allowed(self, ts: datetime) -> bool:
        end_text = str(getattr(self, "_mb_hit_run_end_et", "") or "").strip()
        if not end_text:
            return True
        try:
            hour_text, minute_text = end_text.split(":", 1)
            end_hour = int(hour_text)
            end_minute = int(minute_text)
        except Exception:
            return True
        try:
            current = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            current_et = current.astimezone(ET).time()
        except Exception:
            return True
        return current_et.hour < end_hour or (
            current_et.hour == end_hour and current_et.minute <= end_minute
        )

    def _momentum_burst_10s_tape_is_smooth(
        self,
        symbol: str,
        now: datetime,
    ) -> tuple[bool, float]:
        """Return whether recent 10s bars are tight enough for stops to hold."""
        try:
            history = [
                b for b in self._timer_bars_by_symbol.get(symbol.upper(), [])
                if b.ts <= now and float(b.close or 0.0) > 0
            ]
        except Exception:
            return False, 999.0
        recent = history[-6:]
        if len(recent) < 3:
            return False, 999.0
        ranges = sorted(
            (float(b.high or 0.0) - float(b.low or 0.0)) / float(b.close or 1.0) * 100.0
            for b in recent
        )
        median = float(ranges[len(ranges) // 2])
        return median <= self._mb_smooth_max_median_range_pct, median

    def _momentum_burst_violent_liquid_ok(
        self,
        symbol: str,
        now: datetime,
        *,
        median_range: Optional[float],
    ) -> tuple[bool, Dict[str, float]]:
        history = [
            b for b in self._timer_bars_by_symbol.get(symbol.upper(), [])
            if b.ts <= now and float(b.close or 0.0) > 0
        ]
        if len(history) < 6:
            return False, {}
        if median_range is None:
            _smooth, median_range = self._momentum_burst_10s_tape_is_smooth(symbol, now)
        latest_volume = float(history[-1].volume or 0.0)
        recent_volume = sum(float(b.volume or 0.0) for b in history[-3:])
        day_volume = sum(float(b.volume or 0.0) for b in history)
        ok = (
            median_range <= self._mb_violent_max_median_range_pct
            and latest_volume >= self._mb_violent_min_latest_volume
            and recent_volume >= self._mb_violent_min_recent_volume
            and day_volume >= self._mb_violent_min_day_volume
        )
        return ok, {
            "median_10s_range_pct": round(float(median_range), 3),
            "latest_10s_volume": round(latest_volume, 0),
            "recent_10s_volume": round(recent_volume, 0),
            "day_10s_volume": round(day_volume, 0),
        }

    def _momentum_burst_history(self, symbol: str, now: datetime) -> List[Bar]:
        return [
            b for b in self._timer_bars_by_symbol.get(symbol.upper(), [])
            if b.ts <= now and float(b.close or 0.0) > 0
        ]

    def _momentum_burst_stop_trading_reason(self, symbol: str, now: datetime) -> str:
        history = self._momentum_burst_history(symbol, now)
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
            alpha = 2.0 / (9.0 + 1.0)
            for value in closes[1:]:
                ema = value * alpha + ema * (1.0 - alpha)
            if close < ema * 0.985 and close < min(float(b.low or 0.0) for b in history[-5:-1]):
                return "lost 10s trend support"
        return ""

    def _momentum_burst_continuation_base_ok(
        self,
        symbol: str,
        now: datetime,
    ) -> tuple[bool, str, Dict[str, float]]:
        history = self._momentum_burst_history(symbol, now)
        if len(history) < 8:
            return False, "not enough 10s history", {}
        latest = history[-1]
        prior = history[-6:-1]
        close = float(latest.close or 0.0)
        open_ = float(latest.open or 0.0)
        if close <= open_:
            return False, "confirm bar is not green", {}
        latest_volume = float(latest.volume or 0.0)
        recent_volume = sum(float(b.volume or 0.0) for b in history[-3:])
        if latest_volume < self._mb_violent_min_latest_volume or recent_volume < self._mb_violent_min_recent_volume:
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

    def _momentum_burst_replay_signal(
        self,
        symbol: str,
        bar: Bar,
        bars: Sequence[Bar],
        *,
        hit_run: bool = False,
        violent_liquid: bool = False,
        post_blowoff_micro_base: bool = False,
    ) -> Optional[TradeSignal]:
        strategy_label = (
            "post_blowoff_micro_base_scout"
            if post_blowoff_micro_base
            else ("momentum_burst_hit_run" if hit_run else "momentum_burst_replay")
        )
        pattern_label = strategy_label if hit_run else "momentum_burst_scalp"
        price = float(bar.close or 0.0)
        if price < 1.5 or price > 20.0:
            return None
        ten_history = [
            b for b in self._timer_bars_by_symbol.get(symbol.upper(), [])
            if b.ts <= bar.ts
        ]
        if len(ten_history) < 12:
            return None
        if float(bar.close or 0.0) <= float(bar.open or 0.0):
            return None
        # Premarket-light volume floors (~10x lighter than breakout_scalp_replay)
        # so a real premarket burst can still be evaluated; the shared final
        # entry guard remains the arbiter of liquidity/spread quality.
        latest_volume = float(bar.volume or 0.0)
        recent_volume = sum(float(b.volume or 0.0) for b in ten_history[-3:])
        day_volume = sum(float(b.volume or 0.0) for b in ten_history)
        if day_volume < 50_000 or latest_volume < 500 or recent_volume < 1_500:
            return None
        # (B) Structure stop: sit just below the recent 10s swing low instead of
        # a flat %. A flat 2% stop on a stock printing 10%/10s bars is pure noise
        # and gets stopped instantly; a structure stop gives confirmed entries
        # room. If the structure is so wide the risk blows past the cap, skip —
        # the R:R is bad rather than buy with a too-tight noise stop.
        lookback = max(2, int(self._mb_structure_stop_lookback))
        swing_low = min((float(b.low or 0.0) for b in ten_history[-lookback:]), default=0.0)
        if post_blowoff_micro_base:
            risk = max(price * 0.015, 0.06)
        elif hit_run and violent_liquid:
            # This lane is deliberately a Warrior-style hit-and-run: take reduced
            # size in violent liquid tape and look for the immediate 1R push.
            # A deep structure stop makes 1R unreachable and turns a scalp into a
            # slow swing. Keep the risk tactical and let the small size absorb the
            # higher stop-slip risk.
            risk = max(price * 0.02, 0.06)
        else:
            buffer = max(0.02, price * 0.002)
            struct_stop = swing_low - buffer if swing_low > 0 else 0.0
            min_risk = max(price * 0.012, 0.08)
            struct_risk = price - struct_stop if struct_stop > 0 else 0.0
            if struct_risk >= min_risk and struct_risk <= price * self._mb_structure_max_risk_pct:
                risk = struct_risk
            elif struct_risk > price * self._mb_structure_max_risk_pct:
                return None  # swing low too far — bad R:R, don't chase with a tight stop
            else:
                risk = min_risk  # structure tighter than noise floor — use the floor
        stop_price = round(price - risk, 2)
        # Symmetric 1:1 (or configured R): reward = reward_risk × risk. Full
        # position exits at this target or the stop — no partial scale-outs.
        reward_risk = self._mb_hit_run_reward_risk if hit_run else self._mb_reward_risk
        target_price = round(price + reward_risk * risk, 2)
        if stop_price <= 0 or stop_price >= price:
            return None
        if hit_run and float(bar.low or 0.0) <= stop_price:
            return None
        max_order = int(getattr(self._pipeline, "_max_order_shares", 750) or 750)
        quantity = max(1, min(750, max_order, int(50.0 / (price - stop_price))))
        if post_blowoff_micro_base:
            quantity = max(1, int(quantity * 0.35))
        elif hit_run and violent_liquid:
            quantity = max(1, int(quantity * self._mb_violent_size_factor))
        closed_bars = [b for b in bars if b.ts <= bar.ts.replace(second=0, microsecond=0)]
        hit = ScanResult(
            symbol=symbol,
            scanner_name=strategy_label,
            ts=bar.ts,
            score=100.0,
            criteria={
                "pattern": pattern_label,
                "setup_tier": "A+ setup",
                "entry_tier": "quick_scalp",
                "entry_mode": strategy_label,
                "source_scanner": "momentum_burst",
                "day_volume": round(day_volume, 0),
                "recent_volume": round(recent_volume, 0),
                "latest_volume": round(latest_volume, 0),
                "median_10s_range_pct": round(
                    self._momentum_burst_10s_tape_is_smooth(symbol, bar.ts)[1],
                    3,
                ),
                "stop_price": stop_price,
                "size_factor": (
                    0.35
                    if post_blowoff_micro_base
                    else (self._mb_violent_size_factor if violent_liquid else 1.0)
                ),
                "variant": (
                    "post_blowoff_micro_base"
                    if post_blowoff_micro_base
                    else ("violent_liquid" if violent_liquid else "smooth_confirmed")
                ),
            },
            bars=list(closed_bars[-30:]) + [bar],
        )
        return TradeSignal(
            symbol=symbol,
            action=SignalAction.ENTER_LONG,
            quantity=float(quantity),
            entry_price=price,
            stop_loss=stop_price,
            take_profit=target_price,
            max_hold_seconds=self._mb_hit_run_max_hold_sec if hit_run else 90,
            reason="{} {} ${:.2f}, stop=${:.2f}, target=${:.2f} (10s replay)".format(
                "Momentum Burst Hit-Run" if hit_run else "Momentum Burst Scalp",
                symbol,
                price,
                stop_price,
                target_price,
            ),
            scan_result=hit,
            trend_strength=0.8,
        )

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
        reason = str(row.get("reason") or "")
        result._rejection_reason_counts[(str(layer), reason)] = (
            result._rejection_reason_counts.get((str(layer), reason), 0) + 1
        )
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
        grouped: Dict[str, List[dict]] = {}
        counts = getattr(result, "_rejection_reason_counts", {}) or {}
        for (layer, reason), count in counts.items():
            grouped.setdefault(layer, []).append({
                "reason": reason,
                "count": count,
            })
        result.rejected_reasons_by_layer = {
            layer: sorted(rows, key=lambda r: (-int(r["count"]), str(r["reason"])))[:limit]
            for layer, rows in grouped.items()
        }
