"""Single owner for final entry-policy decisions.

Scanners and pattern verifiers still own pattern discovery. This module owns the
shared final question: can this setup become an order right now?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Sequence

from daytrading.opportunities import hit_identity
from daytrading.models import Bar, Quote, SignalAction, Tick, TradeSignal
from daytrading.strategy.entry_guard import check_entry_quality


@dataclass(frozen=True)
class EntryDecision:
    symbol: str
    stage: str
    passed: bool
    blocked_layer: str = ""
    reason: str = ""
    action: str = ""
    pattern: str = ""
    scanner: str = ""
    setup_tier: str = ""
    entry_tier: str = ""
    price: float = 0.0
    opportunity_id: str = ""
    bar_ts: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def reject_reason(self) -> Optional[str]:
        return None if self.passed else self.reason

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "opportunity_id": self.opportunity_id,
            "bar_ts": self.bar_ts,
            "symbol": self.symbol,
            "stage": self.stage,
            "passed": self.passed,
            "blocked_layer": self.blocked_layer,
            "reason": self.reason,
            "action": self.action,
            "pattern": self.pattern,
            "scanner": self.scanner,
            "setup_tier": self.setup_tier,
            "entry_tier": self.entry_tier,
            "price": self.price,
            "metadata": dict(self.metadata),
        }
        return payload


class EntryPolicy:
    """Centralized final entry policy used by live order paths."""

    ENTRY_ACTIONS = {
        SignalAction.ENTER_LONG,
        SignalAction.REENTER_LONG,
        SignalAction.SCALE_UP_LONG,
        SignalAction.ENTER_SHORT,
        SignalAction.REENTER_SHORT,
        SignalAction.SCALE_UP_SHORT,
    }

    def __init__(
        self,
        guard: Optional[Callable[..., Optional[str]]] = None,
        *, min_price: Optional[float] = None, max_price: Optional[float] = None,
    ) -> None:
        self._guard = guard or check_entry_quality
        self.min_price = min_price
        self.max_price = max_price
        self.entry_patterns = None
        self.micro_pullback_only = False
        self.minimum_room_r = 1.0

    def admission_reject(self, signal, *, cap, bars, now, market_price=None):
        """Also called at native broker submission to cover alternate paths."""
        from daytrading.strategy.entry_admission import route_rejection, room_check
        reason = route_rejection(signal, self.micro_pullback_only)
        if reason or not self.micro_pullback_only:
            return reason
        reason, evidence = room_check(signal, cap, bars, now, self.minimum_room_r, market_price)
        signal.scan_result.criteria['room_risk'] = evidence
        from daytrading.strategy.entry_checks import record, diagnostics
        record(diagnostics(signal.scan_result), 'nearby_high_room_risk', now=now,
            measured=evidence['room_r'], threshold=self.minimum_room_r, units='R',
            comparison='>=', passed=reason is None,
            window={'source':evidence['source'], 'known_high_count':evidence['known_high_count']},
            approved_cap=cap, original_stop=evidence['structural_stop'],
            nearby_high=evidence['nearby_high'], market_reference=market_price)
        return reason

    @staticmethod
    def _signal_context(signal: Optional[TradeSignal]) -> Dict[str, Any]:
        if signal is None:
            return {
                "symbol": "",
                "action": "",
                "pattern": "",
                "scanner": "",
                "setup_tier": "",
                "entry_tier": "",
                "price": 0.0,
            }
        hit = signal.scan_result
        criteria = hit.criteria if hit is not None else {}
        scanner = str(hit.scanner_name or "") if hit is not None else ""
        pattern = str(criteria.get("pattern") or scanner or "")
        setup_score = 0.0
        for key in ("entry_score_at_signal", "setup_score"):
            try:
                setup_score = float(criteria.get(key) or 0.0)
            except (TypeError, ValueError):
                setup_score = 0.0
            if setup_score > 0:
                break
        if setup_score <= 0 and hit is not None:
            setup_score = float(getattr(hit, "score", 0.0) or 0.0)
        return {
            "symbol": signal.symbol,
            "action": signal.action.value,
            "pattern": pattern,
            "scanner": scanner,
            "setup_tier": str(criteria.get("setup_tier") or ""),
            "entry_tier": str(criteria.get("entry_tier") or ""),
            "price": float(signal.entry_price or 0.0),
            "setup_score": setup_score,
        }

    def decision(
        self,
        *,
        symbol: str,
        stage: str,
        passed: bool,
        reason: str = "",
        blocked_layer: str = "",
        signal: Optional[TradeSignal] = None,
        price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EntryDecision:
        ctx = self._signal_context(signal)
        symbol = symbol or str(ctx["symbol"])
        if signal and signal.scan_result and signal.scan_result.scanner_name in ('micro_pullback_reclaim','prepared_reclaim_tick'):
            from types import SimpleNamespace
            from daytrading.strategy.entry_checks import record,diagnostics
            hit=signal.scan_result
            qstamp=hit.criteria.get('flow_inputs',{}).get('last_quote_ts')
            q=SimpleNamespace(ts=datetime.fromisoformat(qstamp)) if qstamp else None
            record(diagnostics(hit),'entry_stage:'+stage,now=datetime.now(timezone.utc),measured=passed,threshold=True,
                units='boolean',comparison='must pass',passed=passed,window={'source':stage,'scanner':hit.scanner_name},quote=q,
                reason=reason or '',symbol=symbol)
        meta = dict(signal.scan_result.criteria.get("quality_inputs", {}) if signal and signal.scan_result else {})
        meta.update(metadata or {})
        if reason and "entry_score" not in meta:
            match = re.search(r"(\d+)/100", str(reason))
            if match:
                meta["entry_score"] = int(match.group(1))
        oid, bar_ts = hit_identity(signal.scan_result if signal else None)
        return EntryDecision(
            opportunity_id=oid, bar_ts=bar_ts,
            symbol=symbol,
            stage=stage,
            passed=passed,
            blocked_layer="" if passed else (blocked_layer or stage),
            reason="" if passed else str(reason or "rejected"),
            action=str(ctx["action"]),
            pattern=str(ctx["pattern"]),
            scanner=str(ctx["scanner"]),
            setup_tier=str(ctx["setup_tier"]),
            entry_tier=str(ctx["entry_tier"]),
            price=float(price if price is not None else ctx["price"] or 0.0),
            metadata=meta,
        )

    def evaluate(
        self,
        signal: TradeSignal,
        *,
        bars: Sequence[Bar],
        stage: str,
        quotes: Optional[Sequence[Quote]] = None,
        ticks: Optional[Sequence[Tick]] = None,
        bars_5m: Optional[Sequence[Bar]] = None,
        avg_daily_volume: Optional[float] = None,
        float_shares: Optional[float] = None,
        min_day_change_pct: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> EntryDecision:
        """Evaluate the shared rule/ML entry gate and return a decision."""
        if signal.action not in self.ENTRY_ACTIONS:
            return self.decision(
                symbol=signal.symbol,
                stage=stage,
                passed=True,
                signal=signal,
                metadata={"skipped": "non-entry action", **dict(metadata or {})},
            )
        ctx = self._signal_context(signal)
        pattern = str(ctx["pattern"])
        reason = self.admission_reject(signal, cap=signal.entry_price, bars=bars, now=now,
            market_price=quotes[-1].ask if quotes else None)
        if reason:
            return self.decision(symbol=signal.symbol, stage=stage, passed=False,
                reason=reason, blocked_layer='entry_admission', signal=signal,
                metadata=dict(room_risk=signal.scan_result.criteria.get('room_risk') if signal.scan_result else None))
        if self.entry_patterns is not None and pattern not in self.entry_patterns:
            return self.decision(symbol=signal.symbol, stage=stage, passed=False,
                                 reason="pattern {} disabled in diagnostic baseline".format(pattern or "unknown"),
                                 blocked_layer="entry_patterns", signal=signal)
        price_bounds = {k: v for k, v in (("min_price", self.min_price), ("max_price", self.max_price)) if v is not None}
        from daytrading.strategy.entry_checks import diagnostics
        vwap_inputs = diagnostics(signal.scan_result) if signal.scan_result is not None else {}
        session_vwap_required = (pattern == "vwap_pullback" and signal.scan_result is not None
            and bool(signal.scan_result.criteria.get("vwap_reference_source")))
        flow_kwargs = {}
        if signal.scan_result is not None and signal.scan_result.criteria.get('_coordinated_micro_flow'):
            from daytrading.strategy.scalping.micro_flow import build_hit_evidence
            from daytrading.data.completed_bars import Rejection
            provider=getattr(self,'micro_flow_provider',None)
            evidence=build_hit_evidence(signal.scan_result,bars,
                ticks or (),provider(signal.symbol,now) if provider is not None and now is not None else (),now)
            if isinstance(evidence,Rejection):
                return self.decision(symbol=signal.symbol,stage=stage,passed=False,
                    reason=evidence.code+': '+evidence.reason,blocked_layer='micro_flow_evidence',signal=signal)
            signal.scan_result.criteria['flow_inputs']=evidence.metadata()
            signal.scan_result.criteria['_micro_quote_samples']=evidence.quote_cost.samples
            flow_kwargs['micro_flow_evidence']=evidence
            if signal.scan_result.criteria.get('_unified_micro_entry'):
                from daytrading.strategy.scalping.micro_profile import qualify
                profile=qualify(signal.scan_result,evidence,now)
                if isinstance(profile,Rejection):return self.decision(symbol=signal.symbol,stage=stage,passed=False,reason=profile.code+': '+profile.reason,blocked_layer='micro_profile',signal=signal)
                flow_kwargs['micro_qualification']=profile
                signal.scan_result.criteria['micro_profile']=profile.metadata()
            bars=evidence.session_minutes
            from daytrading.data.bar_aggregator import BarAggregator
            bars_5m=BarAggregator().build_5m_bars(signal.symbol,bars)
        reason = self._guard(
            bars,
            symbol=signal.symbol,
            **price_bounds,
            min_day_change_pct=min_day_change_pct,
            avg_daily_volume=avg_daily_volume,
            bars_5m=bars_5m,
            float_shares=float_shares,
            ticks=ticks,
            quotes=quotes,
            entry_pattern=pattern,
            setup_tier=str(ctx["setup_tier"]),
            entry_tier=str(ctx["entry_tier"]),
            setup_score=float(ctx["setup_score"] or 0.0),
            **({"vwap_diagnostics": vwap_inputs} if pattern == "micro_pullback_reclaim" else
               {"vwap_diagnostics":vwap_inputs,"session_vwap_required":True} if session_vwap_required else {}),
            now=now,
            **flow_kwargs,
        )
        session_bars = [b for b in bars if b.ts and bars[-1].ts and b.ts.date() == bars[-1].ts.date()] if bars else []
        if len(session_bars) < 3:
            session_bars = list(bars)
        return self.decision(
            symbol=signal.symbol,
            stage=stage,
            passed=reason is None,
            reason=reason or "",
            blocked_layer="entry_guard",
            signal=signal,
            metadata={
                "price_min": self.min_price, "price_max": self.max_price,
                "movement_reference": "session_open", "min_session_change_pct": min_day_change_pct,
                "session_change_pct": ((session_bars[-1].close / session_bars[0].open - 1) * 100 if session_bars and session_bars[0].open > 0 else None),
                "float_shares": float_shares,
                **dict(metadata or {}), **vwap_inputs,
            },
        )
