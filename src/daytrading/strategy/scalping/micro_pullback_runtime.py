"""Opt-in PAPER adapter for the frozen generic earlier-impulse hypothesis.

Discovery, quality, entry policy, post-guards and execution remain shared.
No timer is needed after this detector's completed 10-second confirmation.
"""
from collections import Counter
from datetime import datetime, timezone
from math import floor

from daytrading.execution.pricing import ExecutionPricing, marketable_limit
from daytrading.models import ConfirmationWindow, Order, ScanResult, Side, SignalAction, TradeSignal
from daytrading.strategy.entry_guard import check_entry_quality, record_rule_rejection
from daytrading.strategy.scalping.micro_pullback import (
    NAME, MicroCandidate, MicroPullbackRules, MicroRiskLimits, Rejection,
    assess_micro_entry, earlier_impulse, evaluate_micro_pullback,
)


def candidate_from_hit(hit):
    w = hit.confirmation_window
    if w is None or len(w.bars) < 6:
        return Rejection('confirmation', 'six original completed micro bars required')
    last = w.bars[-1]
    b = last.bar
    loc = (b.close-b.low)/(b.high-b.low) if b.high>b.low else 0
    return MicroCandidate(hit.symbol, b.ts, last.closed_at, b.close,
        round(b.low-.01, 4), w.bars[-2].bar.high, loc,
        sum(x.bar.volume for x in w.bars[-3:]), hit.score)


def assess_with_checks(hit,candidate,quote,*,now,limits,stage):
    """Observe native structural checks without changing their thresholds."""
    from daytrading.strategy.entry_checks import record,diagnostics
    trace=diagnostics(hit)
    age=(now-quote.ts).total_seconds() if quote is not None and now is not None and now.tzinfo is not None and quote.ts.tzinfo is not None else None
    record(trace,'micro_quote_age',now=now,measured=age,threshold=limits.max_quote_age_seconds,
        units='seconds',comparison='0 <= age <=',passed=age is not None and 0<=age<=limits.max_quote_age_seconds,
        window={'source':'current quote','stage':stage},quote=quote)
    if quote is not None and quote.ask>0 and candidate.anchor>0:
        window={'source':'original 10s structure + current ask','confirmed_at':candidate.confirmed_at.isoformat(),'stage':stage}
        extension=quote.ask/candidate.anchor-1
        width=(quote.ask-candidate.technical_stop)/quote.ask
        record(trace,'micro_extension',now=now,measured=extension,threshold=limits.max_chase_pct,
            units='fraction',passed=extension<=limits.max_chase_pct,window=window,quote=quote)
        record(trace,'micro_structural_stop_width',now=now,measured=width,
            threshold={'minimum':limits.min_technical_stop_pct,'maximum':limits.max_stop_pct},units='fraction',
            comparison='minimum <= width <= maximum',passed=limits.min_technical_stop_pct<=width<=limits.max_stop_pct+1e-12,
            window=window,quote=quote,original_stop=candidate.technical_stop)
        effective=max(quote.ask-candidate.technical_stop,limits.min_exit_risk_dollars,quote.ask*limits.min_exit_risk_pct)/quote.ask
        record(trace,'micro_effective_stop_width',now=now,measured=effective,threshold=limits.max_stop_pct,
            units='fraction',passed=effective<=limits.max_stop_pct+1e-12,window=window,quote=quote)
    return assess_micro_entry(candidate,quote,now=now,limits=limits)


def execution_recheck(signal,quote,now,limits,stage):
    hit=signal.scan_result
    if hit is None or hit.scanner_name not in ('micro_pullback_reclaim','prepared_reclaim_tick'):return None
    if now is None:return 'clock_alignment: fresh broker clock calibration required'
    if hit.scanner_name=='prepared_reclaim_tick':
        from daytrading.strategy.scalping.event_reclaim import candidate_from_event
        candidate=candidate_from_event(hit.criteria.get('_event_evidence'),now)
    else:candidate=candidate_from_hit(hit)
    if isinstance(candidate,Rejection):return candidate.code+': '+candidate.reason
    result=assess_with_checks(hit,candidate,quote,now=now,limits=limits,stage=stage)
    if isinstance(result,Rejection):return result.code+': '+result.reason
    from daytrading.strategy.entry_spread import current
    from daytrading.strategy.entry_checks import diagnostics
    if not current(diagnostics(hit),stage+'.current_spread',quote,now):
        return stage+': current quote invalid, stale or wider than configured spread limit'
    return None


class MicroPullbackScanner:
    name = NAME
    accepts_cycle_time = True

    def __init__(self, *, flow=None):
        self.flow = flow
        self._bar_aggregator = None
        self.rules = MicroPullbackRules()
        self.seen = {}
        self.rejections = Counter()
        self.last_rejection = {}
        self.candidates = 0
        self._now = lambda: datetime.now(timezone.utc)
        self._clock_ready = lambda: True
        self._future_windows = {}
        self._native_states = {}
        from daytrading.strategy.scalping.micro_vwap import MicroVWAPRetries
        if flow is None:
            self.vwap_retries = MicroVWAPRetries()
        else:
            from daytrading.strategy.scalping.micro_flow import CoordinatedMicroRetries
            self.vwap_retries = CoordinatedMicroRetries(flow)

    @property
    def needs_recheck(self):
        if self.flow is not None:
            # ready() has no effects and returns False for an empty book.
            # Do not consult the broker clock for that provably empty state.
            return bool(self._future_windows) or bool(self.vwap_retries.rows) and self.vwap_retries.ready(self._now())
        return bool(self._future_windows) or bool(self.vwap_retries.dirty)

    def track_native(self, hit, now):
        from daytrading.strategy.scalping.micro_invalidation import original_state
        self._native_states = {k: state for k, state in self._native_states.items()
                               if now < state['deadline']}
        state = original_state(hit)
        key = state['setup_id']
        if key not in self._native_states and len(self._native_states) >= 256:
            state.update(reason='native_path_capacity', breached_at=now)
        else:
            state = self._native_states.setdefault(key, state)
        hit.criteria['_native_setup_state'] = state

    def observe_native(self, tick, now):
        from daytrading.strategy.scalping.micro_invalidation import observe
        for key, state in tuple(self._native_states.items()):
            if now is not None and now.tzinfo is not None and now >= state['deadline']:
                self._native_states.pop(key, None)
            elif tick.symbol == state['symbol']:
                observe(state, tick, now)

    def scan(self, universe, *, now=None):
        if self._bar_aggregator is None:
            return []
        now = now or self._now()
        if not self._clock_ready():
            self.rejections['clock_alignment'] += 1
            return []
        if self.flow is not None:
            for symbol, context in universe.items():self.flow.prepare(symbol, context, now)
        hits = self.vwap_retries.take(universe, now)
        self._future_windows = {s:v for s,v in self._future_windows.items() if s in universe}
        for symbol, context in universe.items():
            if symbol in self.vwap_retries.rows:
                continue
            window = self._bar_aggregator.get_completed_10s(symbol, count=6)
            if not window or self.seen.get(symbol) == window[-1].bar.ts:
                continue
            result = evaluate_micro_pullback(symbol, window, now=now,
                rules=self.rules, minute_bars=context)
            if isinstance(result, Rejection) and result.code == 'future':
                old = self._future_windows.get(symbol)
                first = old[1] if old and old[0] == window[-1].bar.ts else now
                if (now-first).total_seconds() <= self.rules.max_confirmation_age_seconds:
                    self._future_windows[symbol] = (window[-1].bar.ts,first)
                else:
                    self._future_windows.pop(symbol,None)
                    self.seen[symbol] = window[-1].bar.ts
            else:
                self._future_windows.pop(symbol,None)
                self.seen[symbol] = window[-1].bar.ts
            if isinstance(result, Rejection):
                self.rejections[result.code] += 1
                self.last_rejection[symbol] = dict(reason=result.reason, rule=result.code,
                    time=now.isoformat(), trigger=window[-1].bar.ts.isoformat())
                continue
            impulse = earlier_impulse(symbol, context, now=now, before=window[-3].bar.ts)
            self.candidates += 1
            self.last_rejection.pop(symbol, None)
            criteria = dict(pattern=NAME, direction='up', stop_price=result.technical_stop,
                close=result.anchor, base_high=result.base_high, base_low=window[-1].bar.low,
                setup_anchor=result.anchor, volume=result.volume,
                micro_bar_start=result.bar_start.isoformat(),
                micro_confirmed_at=result.confirmed_at.isoformat(),
                micro_close_location=result.close_location, micro_setup_id=result.setup_id,
                earlier_impulse=impulse, execution_confirmation='completed_10s',
                entry_trigger=NAME, paper_trial=True)
            if self.flow is not None:criteria['_coordinated_micro_flow']=True
            if getattr(self,'entry_profile',False):criteria['_unified_micro_entry']=True
            hit = ScanResult(symbol, NAME, now, result.score, criteria=criteria,
                bars=list(context), confirmation_window=ConfirmationWindow(tuple(window),now))
            self.track_native(hit, now)
            hits.append(hit)
        return hits


class MicroPullbackVerifier:
    name = NAME

    def __init__(self, limits, float_checker=None, *, flow=None):
        self.flow = flow
        self.limits = limits
        self._float_checker = float_checker
        self._bar_aggregator = None
        self._quote_buffer = {}
        self._tick_buffer = {}
        self._last_reject = None
        self._clock_ready = lambda: True

    def _reject(self, symbol, reason):
        self._last_reject = reason
        record_rule_rejection(symbol=symbol, reason=reason)

    def _get_avg_volume(self, symbol):
        return self._float_checker.get_avg_volume(symbol) if self._float_checker else None

    def _get_float_shares(self, symbol):
        return self._float_checker.get_float(symbol) if self._float_checker else None

    def _candidate(self, hit):
        return candidate_from_hit(hit)

    def verify(self, hit, portfolio, *, now=None, session_bars=None):
        self._last_reject = None
        now = now or datetime.now(timezone.utc)
        from daytrading.strategy.scalping.micro_cache import context
        if context(hit) is None:
            self._reject(hit.symbol,'micro_identity: original typed trigger identity required')
            return None
        if not self._clock_ready():
            self._reject(hit.symbol, 'clock_alignment: fresh broker clock calibration required')
            return None
        candidate = self._candidate(hit)
        if hit.scanner_name == NAME:
            from daytrading.strategy.scalping.micro_invalidation import check
            invalid = check(hit, list(self._tick_buffer.get(hit.symbol, ())), now)
            if invalid:
                self._reject(hit.symbol, invalid)
                return None
        if isinstance(candidate, Rejection):
            self._reject(hit.symbol, candidate.reason)
            return None
        qs = list(self._quote_buffer.get(hit.symbol, []))
        if qs and not 0 < qs[-1].bid < qs[-1].ask:
            self._reject(hit.symbol, 'quote: unlocked two-sided quote required')
            return None
        plan = assess_with_checks(hit,candidate, qs[-1] if qs else None,
                                  now=now, limits=self.limits,stage='micro_verification')
        if isinstance(plan, Rejection):
            self._reject(hit.symbol, plan.code+': '+plan.reason)
            return None
        evidence = None
        if self.flow is not None:
            from daytrading.strategy.scalping.micro_flow import build_hit_evidence
            evidence=build_hit_evidence(hit,session_bars or hit.bars,
                list(self._tick_buffer.get(hit.symbol,[])),self.flow.quote_samples(hit.symbol,now),now)
            if isinstance(evidence,Rejection):
                self._reject(hit.symbol,evidence.code+': '+evidence.reason);return None
            hit.criteria['flow_inputs']=evidence.metadata()
            hit.criteria['_micro_quote_samples']=evidence.quote_cost.samples
            self._bar_aggregator.build_5m_bars(hit.symbol,evidence.session_minutes)
        profile=None
        if getattr(self,'entry_profile',False):
            from daytrading.strategy.scalping.micro_profile import qualify
            profile=qualify(hit,evidence,now)
            if isinstance(profile,Rejection):
                self._reject(hit.symbol,profile.code+': '+profile.reason);return None
            hit.criteria['_unified_micro_entry']=True
            hit.criteria['micro_profile']=profile.metadata()
        reason = check_entry_quality(evidence.session_minutes if evidence is not None else session_bars or hit.bars, symbol=hit.symbol,
            min_price=self.limits.min_price, max_price=self.limits.max_price,
            avg_daily_volume=self._get_avg_volume(hit.symbol),
            bars_5m=self._bar_aggregator.get_5m_bars(hit.symbol) if self._bar_aggregator else None,
            float_shares=self._get_float_shares(hit.symbol),
            ticks=list(self._tick_buffer.get(hit.symbol, [])), quotes=qs,
            entry_pattern=NAME, now=now, micro_flow_evidence=evidence, **({'micro_qualification':profile} if profile is not None else {}), vwap_diagnostics=hit.criteria.setdefault("quality_inputs", {}))
        if reason:
            self._reject(hit.symbol, reason)
            return None
        return TradeSignal(hit.symbol, SignalAction.ENTER_LONG, plan.quantity,
            plan.price, plan.technical_stop, plan.technical_2r_target,
            max_hold_seconds=60, reason='PAPER micro pullback: earlier impulse + completed 10s reclaim',
            scan_result=hit)


def bounded_micro_order(signal, quote, bar, *, now, limits):
    """Reserve worst fill risk including the existing exit manager's 3% room.

    The broker receives an absolute price ceiling; later smart repricing can
    reject it but cannot raise this limit or quantity. No retry/headroom added.
    """
    if quote is None or not 0 < quote.bid < quote.ask:
        return Rejection('quote', 'unlocked two-sided quote required')
    candidate = candidate_from_hit(signal.scan_result)
    if isinstance(candidate, Rejection):
        return candidate
    return bounded_candidate_order(signal, candidate, quote, bar, now=now, limits=limits)


def bounded_candidate_order(signal, candidate, quote, bar, *, now, limits):
    """Shared existing price/risk reservation; no special event-trigger exemption."""
    if quote is None or not 0 < quote.bid < quote.ask:
        return Rejection('quote', 'unlocked two-sided quote required')
    from daytrading.strategy.entry_spread import current,get_policy
    from daytrading.strategy.entry_checks import diagnostics,record
    spread_policy=get_policy()
    trace=diagnostics(signal.scan_result)
    criteria=signal.scan_result.criteria
    if criteria.get('micro_original_technical_stop',signal.stop_loss) != candidate.technical_stop:
        return Rejection('stop', 'original micro technical stop changed')
    plan = assess_with_checks(signal.scan_result,candidate, quote, now=now, limits=limits,stage='micro_order_plan')
    if isinstance(plan, Rejection):
        return plan
    # A movement allowance never grants permission to cross a wider book.
    # Use the same normal policy as historical/current execution checks;
    # liquidity-trap and broker-safety rules keep their independent roles.
    if not current(trace,'micro_order_plan.current_spread',quote,now):
        from daytrading.execution.entry_outcome import EntryRecovery, RecoverableEntryRefusal
        return RecoverableEntryRefusal('spread',
            'current quote invalid, stale or wider than configured spread limit; buffer does not override spread policy',
            EntryRecovery.ORDER_PLAN_SPREAD)
    if criteria.get('micro_risk_plan'):
        from daytrading.exits.coordinated_micro import from_signal
        try:
            frozen=from_signal(signal)
            deadline=datetime.fromisoformat(criteria['execution_inputs']['original_deadline'])
            if now>=deadline:return Rejection('micro_risk_plan','original approved deadline expired')
            if not frozen.stop<quote.ask<=frozen.cap:
                return Rejection('micro_risk_plan','fresh ask outside immutable approved plan')
            return Order(signal.symbol,Side.BUY,frozen.quantity,limit_price=frozen.cap,
                         hard_limit_price=frozen.cap,expires_at=deadline)
        except (ValueError,KeyError) as exc:
            return Rejection('micro_risk_plan',str(exc))
    anchor = float(signal.scan_result.criteria.get('setup_anchor') or candidate.anchor)
    old = Order(signal.symbol, Side.BUY, signal.quantity, signal.entry_price)
    try:
        limit = marketable_limit(old, bar, quote_price=quote.ask, policy=ExecutionPricing())
    except ValueError as exc:
        return Rejection('pricing', str(exc))
    if limit > anchor*(1+limits.max_chase_pct)+1e-12:
        return Rejection('chase', 'micro order exceeds original 2.5% base anchor ceiling')
    effective = max(limit-signal.stop_loss, .05, limit*.03)
    if effective/limit > limits.max_stop_pct+1e-12:
        return Rejection('stop', 'priced micro stop exceeds maximum width')
    original_risk = max(signal.entry_price-signal.stop_loss, .05, signal.entry_price*.03)
    budget = min(limits.max_dollar_risk, signal.quantity*original_risk)
    quantity = min(int(signal.quantity), plan.quantity, int(limits.max_order_shares), floor(budget/effective))
    if quantity < 1:
        return Rejection('quantity', 'micro order has no quantity within original risk budget')
    if signal.scan_result.criteria.get('_coordinated_micro_flow'):
        from daytrading.execution.micro_buffer import Intent,Rejected,plan as buffer_plan,refresh
        from daytrading.backtest.matching_profile import default_matching_profile
        from statistics import NormalDist
        from math import exp
        profile=default_matching_profile()
        horizon=exp(profile.initial_delay.mu+NormalDist().inv_cdf(.9)*profile.initial_delay.sigma)
        original_budget=min(budget,quantity*effective)
        ceiling=round(signal.entry_price*(1+ExecutionPricing().buy_window(bar)),2)
        intent=Intent(signal.symbol,now,candidate.confirmed_at,limit,quantity,signal.stop_loss,
            signal.take_profit,anchor,ceiling,original_budget,
            max_chase_pct=limits.max_chase_pct,max_stop_pct=limits.max_stop_pct,
            min_price=limits.min_price,max_price=limits.max_price,
            max_quote_age_seconds=min(limits.max_quote_age_seconds,spread_policy.quote_max_age_seconds))
        planned=buffer_plan(intent,quote,signal.scan_result.criteria.get('_micro_quote_samples',()),
            'adaptive',now=now,horizon=horizon)
        if isinstance(planned,Rejected):return Rejection('micro_buffer',planned.reason)
        # Final quality/post guards have just run on this same fresh evidence.
        refreshed=refresh(planned,intent,quote,now=now)
        if isinstance(refreshed,Rejected):return Rejection('micro_buffer',refreshed.reason)
        ceiling_values=dict(signal_window=ceiling,
            extension=anchor*(1+min(.025,limits.max_chase_pct)),
            structural_stop=signal.stop_loss/(1-min(.06,limits.max_stop_pct)),
            universe=limits.max_price,target=signal.take_profit-.01,
            requested_buffer=limit+planned.requested_allowance)
        record(trace,'micro_buffer_approved_cap',now=now,measured=planned.cap,
            threshold=min(ceiling_values.values()),units='USD/share',
            passed=planned.cap<=min(ceiling_values.values())+1e-12,
            window={'source':'original signal and completed 10s structure + current quote'},
            quote=quote,ceilings_usd=ceiling_values,
            requested_allowance_usd=planned.requested_allowance,effective_allowance_usd=planned.extra_allowance)
        record(trace,'micro_buffer_reserved_risk',now=now,measured=planned.reserved_risk,
            threshold=original_budget,units='USD',passed=planned.reserved_risk<=original_budget+1e-8,
            window={'source':'maximum permitted fill and fixed stop; rounding reserve included'},
            quote=quote,quantity_shares=planned.quantity)
        signal.scan_result.criteria['execution_inputs']=dict(policy='bounded-adaptive-v2',
            original_cap=limit,approved_cap=planned.cap,original_budget=original_budget,
            reserved_risk=planned.reserved_risk,quantity=planned.quantity,
            original_deadline=planned.deadline.isoformat(),limiting_ceiling=planned.limiting_ceiling,
            measured_matching_horizon=horizon,network_delay_measured=False,
            matching_horizon_source='historical_profile_p90',matching_horizon_units='seconds',
            matching_profile_version=profile.version,
            configured_limits=dict(max_chase_fraction=limits.max_chase_pct,max_stop_fraction=limits.max_stop_pct,
                effective_chase_fraction=min(.025,limits.max_chase_pct),effective_stop_fraction=min(.06,limits.max_stop_pct),
                min_price_usd=limits.min_price,max_price_usd=limits.max_price),
            buffer_mode='adaptive',reference_price_usd=signal.entry_price,
            requested_buffer_usd=planned.requested_allowance,effective_buffer_usd=planned.extra_allowance,
            current_spread_usd=quote.spread,normal_spread_limit_usd=spread_policy.limit(quote.ask),
            quote_max_age_seconds=intent.max_quote_age_seconds,
            buffer_raises_spread_limit=False)
        if signal.scan_result.criteria.get('_unified_micro_entry'):
            from daytrading.exits.coordinated_micro import approve
            try:
                approve(signal,planned.cap,planned.quantity,original_budget)
            except ValueError as exc:
                return Rejection('micro_risk_plan',str(exc))
        return Order(signal.symbol,Side.BUY,planned.quantity,limit_price=planned.cap,hard_limit_price=planned.cap,expires_at=planned.deadline)
    return Order(signal.symbol, Side.BUY, quantity, limit_price=limit, hard_limit_price=limit)


def submission_reclaim_recheck(signal, quote, tick, now, *, ticks=None):
    """Same completed-micro reclaim rule, measured again at submission."""
    from math import isfinite
    from daytrading.strategy.entry_checks import record,diagnostics
    hit=signal.scan_result
    if hit is None or hit.scanner_name!='micro_pullback_reclaim':return None
    if now is None or now.tzinfo is None:return 'clock_alignment: aware submission clock required'
    candidate=candidate_from_hit(hit)
    if isinstance(candidate,Rejection):return candidate.code+': '+candidate.reason
    from daytrading.strategy.scalping.micro_invalidation import check
    invalid = check(hit, ticks if ticks is not None else ([tick] if tick is not None else []), now)
    if invalid:return invalid
    age=(now-tick.ts).total_seconds() if tick is not None and tick.ts is not None and tick.ts.tzinfo is not None else None
    fresh=tick is not None and tick.symbol==signal.symbol and age is not None and 0<=age<=1
    fresh=fresh and all(isfinite(v) and v>0 for v in (tick.price,tick.size))
    trace=diagnostics(hit)
    window={'source':'latest consumed trade at submission','trade_symbol':tick.symbol if tick is not None else None,'trade_at':tick.ts.isoformat() if tick is not None and tick.ts is not None else None,'original_trigger_start':candidate.bar_start.isoformat()}
    record(trace,'submission_trade_age',now=now,measured=age,threshold=1.,units='seconds',comparison='0 <= age <=',passed=bool(fresh),window=window,quote=quote)
    valid=bool(fresh and tick.price>=candidate.base_high)
    record(trace,'submission_reclaim',now=now,measured=tick.price if tick is not None and isfinite(tick.price) else None,threshold=candidate.base_high,units='USD/share',comparison='>=',passed=valid,window=window,quote=quote)
    if not fresh:return 'submission_reclaim: fresh same-symbol trade <=1s required'
    if not valid:return 'submission_reclaim: fresh trade lost original reclaim level'
    return None
