"""Track A+ setups that were blocked, then label later outcomes.

This module is deliberately observational. It does not approve trades or alter
guards; it builds a daily truth table for tuning from evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from daytrading.indicators.core import vwap
from daytrading.models import Bar, Quote, ScanResult, TradeSignal


A_PLUS_PATTERNS = frozenset({
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


@dataclass
class MissedAPlusRecord:
    id: str
    symbol: str
    pattern: str
    scanner: str
    first_seen: datetime
    blocked_at: datetime
    blocked_layer: str
    reason: str
    price_at_reject: float
    score: float = 0.0
    criteria: Dict[str, Any] = field(default_factory=dict)
    a_plus_score: int = 0
    max_price_after: float = 0.0
    min_price_after: float = 0.0
    last_price: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    outcome: str = "pending"
    move_after_pct: float = 0.0
    dump_after_pct: float = 0.0
    correct: Optional[bool] = None
    suggested_fix: str = ""
    observations: List[str] = field(default_factory=list)
    spread_cents: float = 0.0
    spread_pct: float = 0.0
    risk_per_share: float = 0.0
    risk_pct: float = 0.0
    tactical_stop_price: float = 0.0
    tactical_stop_survived: Optional[bool] = None
    median_bar_range_pct: float = 0.0
    smooth_for_tactical_stop: bool = False
    tactical_stop_clean_survival: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "pattern": self.pattern,
            "scanner": self.scanner,
            "first_seen": self.first_seen.isoformat(),
            "blocked_at": self.blocked_at.isoformat(),
            "blocked_layer": self.blocked_layer,
            "reason": self.reason,
            "price_at_reject": round(self.price_at_reject, 4),
            "max_price_after": round(self.max_price_after, 4),
            "min_price_after": round(self.min_price_after, 4),
            "last_price": round(self.last_price, 4),
            "move_after_pct": round(self.move_after_pct, 2),
            "dump_after_pct": round(self.dump_after_pct, 2),
            "outcome": self.outcome,
            "correct": self.correct,
            "suggested_fix": self.suggested_fix,
            "a_plus_score": self.a_plus_score,
            "score": round(self.score, 3),
            "observations": list(self.observations),
            "criteria": dict(self.criteria),
            "spread_cents": round(self.spread_cents, 4),
            "spread_pct": round(self.spread_pct, 4),
            "is_spread_reject": "spread" in self.reason.lower(),
            "risk_per_share": round(self.risk_per_share, 4),
            "risk_pct": round(self.risk_pct, 2),
            "tactical_stop_price": round(self.tactical_stop_price, 4),
            "tactical_stop_survived": self.tactical_stop_survived,
            "median_bar_range_pct": round(self.median_bar_range_pct, 2),
            "smooth_for_tactical_stop": self.smooth_for_tactical_stop,
            "tactical_stop_clean_survival": self.tactical_stop_clean_survival,
            "is_risk_reject": _is_risk_reject(self.reason, self.blocked_layer),
        }


class MissedAPlusTracker:
    """Keeps recent blocked A+ setups and labels later price action."""

    def __init__(
        self,
        *,
        max_records: int = 300,
        dedupe_seconds: float = 120.0,
        label_after_seconds: float = 180.0,
    ) -> None:
        self._records: Dict[str, MissedAPlusRecord] = {}
        self._order: List[str] = []
        self._max_records = max_records
        self._dedupe_seconds = dedupe_seconds
        self._label_after_seconds = label_after_seconds

    def record_blocked(
        self,
        *,
        layer: str,
        reason: str,
        universe: Dict[str, Sequence[Bar]],
        quotes: Optional[Dict[str, Sequence[Quote]]] = None,
        hit: Optional[ScanResult] = None,
        signal: Optional[TradeSignal] = None,
        fallback_price: float = 0.0,
        now: Optional[datetime] = None,
    ) -> Optional[MissedAPlusRecord]:
        now = _utc(now)
        symbol = _symbol(hit=hit, signal=signal)
        if not symbol:
            return None
        bars = _bars_for(symbol, universe, hit=hit)
        if not bars:
            return None

        pattern = _pattern(hit=hit, signal=signal)
        scanner = _scanner(hit=hit, signal=signal)
        price = _price(hit=hit, signal=signal, bars=bars, fallback=fallback_price)
        if price <= 0:
            return None

        score, observations = self._a_plus_score(
            pattern=pattern,
            scanner=scanner,
            bars=bars,
            quotes=list((quotes or {}).get(symbol, [])),
            hit=hit,
        )
        if score < 75:
            return None

        rec_id = self._dedupe_id(
            symbol=symbol, pattern=pattern, layer=layer, reason=reason, now=now,
        )
        existing = self._records.get(rec_id)
        if existing is not None:
            existing.reason = reason
            existing.blocked_at = now
            existing.last_update = now
            existing.observations = observations
            existing.a_plus_score = score
            return existing

        criteria = {}
        raw_criteria = hit.criteria if hit is not None else (
            signal.scan_result.criteria if signal and signal.scan_result else {}
        )
        for key, value in dict(raw_criteria).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                criteria[key] = value

        spread_cents, spread_pct = _spread_metrics(
            reason=reason,
            price=price,
            quotes=list((quotes or {}).get(symbol, [])),
        )
        risk_per_share, risk_pct, tactical_stop_price = _risk_metrics(
            reason=reason,
            price=price,
            bars=bars,
            criteria=criteria,
        )
        median_bar_range_pct = _median_bar_range_pct(bars)
        smooth_for_tactical_stop = (
            median_bar_range_pct > 0 and median_bar_range_pct <= 2.0
        )
        record = MissedAPlusRecord(
            id=rec_id,
            symbol=symbol,
            pattern=pattern or scanner or "unknown",
            scanner=scanner,
            first_seen=now,
            blocked_at=now,
            blocked_layer=layer,
            reason=reason,
            price_at_reject=price,
            score=float(hit.score if hit is not None else 0.0),
            criteria=criteria,
            a_plus_score=score,
            max_price_after=price,
            min_price_after=price,
            last_price=price,
            last_update=now,
            observations=observations,
            suggested_fix=self._suggested_fix(layer, reason),
            spread_cents=spread_cents,
            spread_pct=spread_pct,
            risk_per_share=risk_per_share,
            risk_pct=risk_pct,
            tactical_stop_price=tactical_stop_price,
            median_bar_range_pct=median_bar_range_pct,
            smooth_for_tactical_stop=smooth_for_tactical_stop,
        )
        self._records[rec_id] = record
        self._order.append(rec_id)
        self._trim()
        return record

    def record_early_exit(
        self,
        *,
        symbol: str,
        entry_price: float,
        exit_price: float,
        reason: str,
        universe: Dict[str, Sequence[Bar]],
        quotes: Optional[Dict[str, Sequence[Quote]]] = None,
        now: Optional[datetime] = None,
    ) -> Optional[MissedAPlusRecord]:
        bars = list(universe.get(symbol, []))
        if not bars:
            return None
        hit = ScanResult(
            symbol=symbol,
            scanner_name="early_exit",
            ts=_utc(now),
            score=0.0,
            criteria={
                "pattern": "early_exit",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "close": exit_price,
            },
            bars=bars,
        )
        return self.record_blocked(
            layer="early_exit",
            reason=reason,
            universe=universe,
            quotes=quotes,
            hit=hit,
            fallback_price=exit_price,
            now=now,
        )

    def update_prices(
        self,
        universe: Dict[str, Sequence[Bar]],
        *,
        now: Optional[datetime] = None,
    ) -> None:
        now = _utc(now)
        for record in self._records.values():
            bars = universe.get(record.symbol)
            if not bars:
                continue
            latest = bars[-1]
            price = float(latest.close or 0.0)
            if price <= 0:
                continue
            high = float(latest.high or price)
            low = float(latest.low or price)
            record.last_price = price
            record.max_price_after = max(record.max_price_after, high, price)
            record.min_price_after = min(record.min_price_after, low, price)
            record.last_update = now
            self._label(record, now)

    def report(self, *, limit: int = 30) -> List[Dict[str, Any]]:
        # scanner near-misses are a separate report — keep them out of the
        # A+ card so it stays "we blocked a real A+ setup" only.
        rows = [
            self._records[rid] for rid in self._order
            if rid in self._records and self._records[rid].blocked_layer != "scanner_near_miss"
        ]
        rows.sort(
            key=lambda r: (
                r.outcome != "missed_opportunity",
                -r.move_after_pct,
                r.blocked_at,
            )
        )
        return [r.as_dict() for r in rows[:limit]]

    def spread_summary(self) -> Dict[str, Any]:
        spread_rows = [r for r in self._records.values() if "spread" in r.reason.lower()]
        false_blocks = [r for r in spread_rows if r.outcome == "missed_opportunity"]
        correct_rejects = [r for r in spread_rows if r.outcome == "correct_reject"]
        pending = [r for r in spread_rows if r.outcome == "pending"]
        by_symbol: Dict[str, int] = {}
        for rec in spread_rows:
            by_symbol[rec.symbol] = by_symbol.get(rec.symbol, 0) + 1
        return {
            "spread_blocked_runners": len(spread_rows),
            "spread_false_blocks": len(false_blocks),
            "spread_correct_rejects": len(correct_rejects),
            "spread_pending": len(pending),
            "symbols": by_symbol,
        }

    def risk_summary(self) -> Dict[str, Any]:
        risk_rows = [
            r for r in self._records.values()
            if _is_risk_reject(r.reason, r.blocked_layer)
        ]
        false_blocks = [r for r in risk_rows if r.outcome == "missed_opportunity"]
        correct_rejects = [r for r in risk_rows if r.outcome == "correct_reject"]
        pending = [r for r in risk_rows if r.outcome == "pending"]
        survived = [r for r in risk_rows if r.tactical_stop_survived is True]
        failed = [r for r in risk_rows if r.tactical_stop_survived is False]
        clean_survived = [r for r in risk_rows if r.tactical_stop_clean_survival is True]
        clean_failed = [r for r in risk_rows if r.tactical_stop_clean_survival is False]
        choppy_survived = [
            r for r in risk_rows
            if r.tactical_stop_survived is True and not r.smooth_for_tactical_stop
        ]
        symbols: Dict[str, int] = {}
        for rec in risk_rows:
            symbols[rec.symbol] = symbols.get(rec.symbol, 0) + 1
        return {
            "risk_blocked_runners": len(risk_rows),
            "risk_false_blocks": len(false_blocks),
            "risk_correct_rejects": len(correct_rejects),
            "risk_pending": len(pending),
            "tactical_stop_survived": len(survived),
            "tactical_stop_failed": len(failed),
            "clean_tactical_stop_survived": len(clean_survived),
            "clean_tactical_stop_failed": len(clean_failed),
            "choppy_tactical_stop_survived": len(choppy_survived),
            "symbols": symbols,
        }

    def record_scanner_near_miss(
        self,
        *,
        symbol: str,
        reason: str,
        universe: Dict[str, Sequence[Bar]],
        float_shares: Optional[float] = None,
        max_float: float = 20_000_000.0,
        min_day_volume: float = 1_000_000.0,
        now: Optional[datetime] = None,
    ) -> Optional[MissedAPlusRecord]:
        """Record a hot-watch name that died at the SCANNER stage.

        Missed-A+ only sees rows after an A+ setup exists. This catches the
        upstream blind spot: a low-float, high-volume hot-watch name the scanner
        never turned into a clean pattern (the ASBP case). Reuses the smoothness
        + tactical-stop machinery so the report can separate a real scanner_gap
        (smooth, ran, stop would have held) from a washout (gappy runner).
        Report-only — changes nothing the bot trades.
        """
        now = _utc(now)
        symbol = symbol.upper()
        bars = list(universe.get(symbol, []))
        if not bars:
            return None
        price = float(bars[-1].close or 0.0)
        if price <= 0:
            return None
        day_volume = sum(float(b.volume or 0.0) for b in bars)
        if float_shares is not None and max_float > 0 and float_shares > max_float:
            return None
        if day_volume < min_day_volume:
            return None

        rec_id = self._dedupe_id(
            symbol=symbol, pattern="scanner_near_miss", layer="scanner_near_miss",
            reason=reason, now=now,
        )
        existing = self._records.get(rec_id)
        if existing is not None:
            existing.last_update = now
            return existing

        median_bar_range_pct = _median_bar_range_pct(bars)
        smooth = median_bar_range_pct > 0 and median_bar_range_pct <= 2.0
        recent = list(bars[-3:])
        lows = [float(b.low or 0.0) for b in recent if float(b.low or 0.0) > 0]
        tactical_stop = (min(lows) - 0.02) if lows else 0.0
        if tactical_stop <= 0 or tactical_stop >= price:
            tactical_stop = 0.0

        record = MissedAPlusRecord(
            id=rec_id,
            symbol=symbol,
            pattern="scanner_near_miss",
            scanner="scanner",
            first_seen=now,
            blocked_at=now,
            blocked_layer="scanner_near_miss",
            reason=reason,
            price_at_reject=price,
            max_price_after=price,
            min_price_after=price,
            last_price=price,
            last_update=now,
            tactical_stop_price=tactical_stop,
            median_bar_range_pct=median_bar_range_pct,
            smooth_for_tactical_stop=smooth,
        )
        self._records[rec_id] = record
        self._order.append(rec_id)
        self._trim()
        return record

    def scanner_near_miss_summary(self, *, min_move_pct: float = 8.0) -> Dict[str, Any]:
        """Separate real scanner gaps (smooth, ran, stop held) from washouts."""
        rows = [r for r in self._records.values() if r.blocked_layer == "scanner_near_miss"]
        moved = [r for r in rows if r.move_after_pct >= min_move_pct]
        scanner_gaps = [
            r for r in moved
            if r.tactical_stop_clean_survival is True  # smooth AND tactical stop held
        ]
        washouts = [r for r in moved if not r.smooth_for_tactical_stop]
        quiet = [r for r in rows if r.move_after_pct < min_move_pct]
        return {
            "scanner_near_misses": len(rows),
            "moved": len(moved),
            "scanner_gaps": len(scanner_gaps),   # smooth runners the scanner had no pattern for
            "washouts": len(washouts),           # gappy runners (ASBP) — correctly ignored
            "quiet": len(quiet),
            "gap_symbols": [r.symbol for r in scanner_gaps],
        }

    def chase_reject(
        self,
        *,
        symbol: str,
        price: float,
        now: Optional[datetime] = None,
        signal: Optional[TradeSignal] = None,
        max_age_seconds: float = 1800.0,
        max_chase_pct_sub5: float = 0.035,
        max_chase_pct_5plus: float = 0.025,
        fresh_base_anchor: float = 0.0,
        fresh_base_reset_pct: float = 0.0,
    ) -> Optional[str]:
        """Reject late buys far above a recent blocked A+ setup.

        This keeps the system from missing the early A+ decision point and then
        buying the same move much higher a few minutes later.

        ``fresh_base_anchor`` is the CURRENT signal's own setup base (the price
        the primary chase guard measures from). When it has migrated materially
        above a stale blocked level (by >= ``fresh_base_reset_pct``), that level
        is treated as stale — a new, higher base has formed, so this is a fresh
        setup, not a re-chase of the old move. The primary own-base chase guard
        still applies after this returns.
        """
        now = _utc(now)
        if price <= 0:
            return None
        candidates: List[MissedAPlusRecord] = []
        for rec in self._records.values():
            if rec.symbol != symbol:
                continue
            if rec.correct is True:
                continue
            if rec.a_plus_score < 80 or rec.price_at_reject <= 0:
                continue
            if (now - rec.first_seen).total_seconds() > max_age_seconds:
                continue
            if self._is_non_actionable_chase_anchor(rec):
                continue
            if (
                fresh_base_anchor > 0
                and fresh_base_reset_pct > 0
                and fresh_base_anchor >= rec.price_at_reject * (1.0 + fresh_base_reset_pct)
            ):
                # Setup base moved up off this stale level -> new base, not a chase.
                continue
            candidates.append(rec)
        if not candidates:
            return None
        anchor = min(candidates, key=lambda rec: rec.price_at_reject)
        max_chase_pct = (
            float(max_chase_pct_sub5)
            if anchor.price_at_reject < 5.0
            else float(max_chase_pct_5plus)
        )
        max_price = anchor.price_at_reject * (1.0 + max_chase_pct)
        if price <= max_price:
            return None
        return (
            "late chase: ${:.4f} is {:.1f}% above earlier blocked A+ {} "
            "at ${:.4f} (max {:.1f}%)"
        ).format(
            price,
            (price - anchor.price_at_reject) / anchor.price_at_reject * 100.0,
            anchor.pattern or anchor.scanner or "setup",
            anchor.price_at_reject,
            max_chase_pct * 100.0,
        )

    @staticmethod
    def _is_fresh_reclaim_retry(signal: Optional[TradeSignal]) -> bool:
        if signal is None or signal.scan_result is None:
            return False
        criteria = signal.scan_result.criteria or {}
        entry_tier = str(criteria.get("entry_tier") or "").lower()
        entry_reason = str(criteria.get("entry_tier_reason") or "").lower()
        entry_mode = str(criteria.get("entry_mode") or "").lower()
        return (
            entry_tier in {"a_plus_retry_watch", "fresh_vwap_reclaim_scout"}
            or "fresh base reclaim" in entry_reason
            or "fresh reclaim" in entry_reason
            or "fresh_vwap_reclaim" in entry_mode
        )

    @staticmethod
    def _is_non_actionable_chase_anchor(record: MissedAPlusRecord) -> bool:
        """Skip watch-state rejects as anti-chase anchors.

        These are not valid missed entry prices; they explicitly mean "wait for
        a new base/reclaim." Anchoring to their dip lows creates a Catch-22
        where the later reclaim is rejected for being above the watch price.
        """
        reason = str(record.reason or "").lower()
        pattern = str(record.pattern or record.scanner or "").lower()
        return (
            ("too far from hod" in reason and (
                "watching for fresh reclaim" in reason
                or "fresh reclaim" in reason
                or "late pullback" in reason
            ))
            or "wait for new base" in reason
            or "pullback has dump candle" in reason
            or "risk too wide" in reason
            or (
                pattern == "momentum_burst"
                and (
                    "entry score too low" in reason
                    or "risk too wide" in reason
                    or "skip loose setup" in reason
                )
            )
        )

    def _a_plus_score(
        self,
        *,
        pattern: str,
        scanner: str,
        bars: Sequence[Bar],
        quotes: Sequence[Quote],
        hit: Optional[ScanResult],
    ) -> tuple[int, List[str]]:
        latest = bars[-1]
        price = float(latest.close or 0.0)
        if price <= 0:
            return 0, []

        observations: List[str] = []
        score = 0
        is_known_a_plus = (
            pattern in A_PLUS_PATTERNS
            or scanner in A_PLUS_PATTERNS
            or scanner == "early_exit"
        )
        setup_tier = str((hit.criteria if hit else {}).get("setup_tier", "")).lower()
        if is_known_a_plus or "a+" in setup_tier:
            score += 25
            observations.append("known A+ setup")

        day_volume = sum(float(b.volume or 0.0) for b in bars)
        recent = list(bars[-5:])
        recent_avg = sum(float(b.volume or 0.0) for b in recent) / len(recent)
        earlier = list(bars[:-5])
        earlier_avg = (
            sum(float(b.volume or 0.0) for b in earlier) / len(earlier)
            if earlier else 0.0
        )
        bar_rvol = recent_avg / earlier_avg if earlier_avg > 0 else 0.0
        if day_volume >= 2_000_000 or recent_avg >= 150_000:
            score += 25
            observations.append("strong volume")
        elif day_volume >= 500_000 and recent_avg >= 50_000:
            score += 18
            observations.append("good volume")
        if bar_rvol >= 1.5:
            score += 10
            observations.append("RVOL expanding")

        vwap_vals = vwap(list(bars))
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        criteria = hit.criteria if hit is not None else {}
        level = float(
            criteria.get("breakout_level")
            or criteria.get("base_high")
            or criteria.get("vwap")
            or 0.0
        )
        if current_vwap > 0 and price >= current_vwap * 0.995:
            score += 15
            observations.append("above VWAP")
        if level > 0 and price >= level * 0.995:
            score += 15
            observations.append("level reclaim")

        recent_high = max(float(b.high or 0.0) for b in bars[-20:])
        if recent_high > 0:
            pullback = (recent_high - price) / recent_high * 100.0
            if pullback <= 8.0:
                score += 10
                observations.append("near local HOD")

        if quotes:
            valid = [q for q in quotes[-5:] if q.ask > q.bid > 0]
            if valid:
                spread = sum(q.spread_pct for q in valid) / len(valid)
                max_spread = 0.9 if price < 5.0 else 0.6
                if spread <= max_spread:
                    score += 10
                    observations.append("spread acceptable")
                else:
                    score -= 25
                    observations.append("spread too wide")

        if latest.high > latest.low:
            close_position = (latest.close - latest.low) / (latest.high - latest.low)
            if latest.close < latest.open and close_position < 0.35:
                score -= 30
                observations.append("dump candle")
            elif close_position >= 0.45:
                score += 5
                observations.append("no dump candle")

        return max(0, min(100, score)), observations

    def _label(self, record: MissedAPlusRecord, now: datetime) -> None:
        if record.price_at_reject <= 0:
            return
        record.move_after_pct = (
            (record.max_price_after - record.price_at_reject)
            / record.price_at_reject * 100.0
        )
        record.dump_after_pct = (
            (record.min_price_after - record.price_at_reject)
            / record.price_at_reject * 100.0
        )
        if record.tactical_stop_price > 0:
            record.tactical_stop_survived = record.min_price_after > record.tactical_stop_price
            if record.smooth_for_tactical_stop:
                record.tactical_stop_clean_survival = record.tactical_stop_survived
            else:
                record.tactical_stop_clean_survival = None
        age = (now - record.first_seen).total_seconds()
        if age < self._label_after_seconds:
            record.outcome = "pending"
            return
        hard_dump_after = (
            record.dump_after_pct <= -6.0
            or (
                record.dump_after_pct <= -3.0
                and abs(record.dump_after_pct) >= max(3.0, record.move_after_pct * 1.25)
            )
        )
        small_pop = record.move_after_pct < 8.0
        if hard_dump_after and small_pop:
            record.outcome = "correct_reject"
            record.correct = True
            record.suggested_fix = (
                "Leave strict: small post-reject pop was followed by hard dump"
            )
        elif record.move_after_pct >= 3.0:
            record.outcome = "missed_opportunity"
            record.correct = False
            record.suggested_fix = self._missed_opportunity_fix(
                record.blocked_layer, record.reason,
            )
        elif record.dump_after_pct <= -3.0 or record.last_price <= record.price_at_reject * 0.985:
            record.outcome = "correct_reject"
            record.correct = True
        else:
            record.outcome = "neutral"
            record.correct = None

    def _dedupe_id(
        self,
        *,
        symbol: str,
        pattern: str,
        layer: str,
        reason: str,
        now: datetime,
    ) -> str:
        reason_key = " ".join(str(reason).lower().split())[:80]
        for rid in reversed(self._order):
            rec = self._records.get(rid)
            if rec is None:
                continue
            if rec.symbol != symbol or rec.pattern != (pattern or rec.pattern):
                continue
            if rec.blocked_layer != layer:
                continue
            if rec.reason.lower()[:80] != reason_key:
                continue
            if (now - rec.blocked_at).total_seconds() <= self._dedupe_seconds:
                return rid
        return "{}:{}:{}:{:.0f}".format(
            symbol, pattern or "unknown", layer, now.timestamp(),
        )

    def _trim(self) -> None:
        while len(self._order) > self._max_records:
            rid = self._order.pop(0)
            self._records.pop(rid, None)

    @staticmethod
    def _suggested_fix(layer: str, reason: str) -> str:
        text = reason.lower()
        if "10s" in text or "timed" in layer:
            return "Review 10s confirmation timing for elite runners"
        if "ml" in text:
            return "Review ML threshold/features for this A+ pattern"
        if "entry score" in text or "final entry guard" in layer:
            return "Review final guard scoring only for elite A+ cases"
        if "r:r" in text or "risk" in layer:
            return "Review stop/target sizing or reduced scout size"
        if "spread" in text:
            return "Leave strict unless spread improved before move"
        if "risk" in text or "r:r" in text:
            return "Track wide-risk outcome; do not loosen until tactical-stop survival is proven"
        if "early_exit" in layer:
            return "Review runner/partial exit logic for early shakeouts"
        if "watch only" in text:
            return "Consider promoting only clean A+ watch pattern variants"
        return "Review blocked layer with chart and later outcome"

    @staticmethod
    def _missed_opportunity_fix(layer: str, reason: str) -> str:
        text = reason.lower()
        if "spread" in text:
            return "Spread rule was too strict for this runner; review tick-aware/elite spread handling"
        if "risk" in text or "r:r" in text:
            return "Wide-risk block missed a runner; review tactical-stop survival before allowing reduced scouts"
        if "hod momentum" in text:
            return "HOD board lagged the setup; allow clean A+ scanner signals to reach guard"
        if "price $" in text and "outside range" in text:
            return "Review elite sub-$1.50 runner exception with strict liquidity/spread controls"
        if "watch only" in text:
            return "Promote only clean A+ watch variants when level reclaim and volume confirm"
        if "vwap" in text:
            return "Review VWAP/reclaim tolerance for elite runners that quickly resume"
        if "tape" in text or "volume" in text:
            return "Review tape/volume threshold for elite runners; require retry instead of final reject"
        if "10s" in text or "timed" in layer:
            return "Release earlier on clean 10s hold, but keep base-anchored chase guard"
        if "entry guard" in layer or "entry score" in text:
            return "Review final guard scoring only for elite A+ cases"
        if "risk" in layer or "r:r" in text:
            return "Review reduced scout size or stop model for elite A+ runners"
        return "Tune the blocked layer; later price action proved this was a missed A+ runner"


def _utc(value: Optional[datetime]) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _symbol(
    *,
    hit: Optional[ScanResult] = None,
    signal: Optional[TradeSignal] = None,
) -> str:
    if hit is not None:
        return hit.symbol.upper()
    if signal is not None:
        return signal.symbol.upper()
    return ""


def _bars_for(
    symbol: str,
    universe: Dict[str, Sequence[Bar]],
    *,
    hit: Optional[ScanResult] = None,
) -> List[Bar]:
    if hit is not None and hit.bars:
        return list(hit.bars)
    return list(universe.get(symbol, []))


def _pattern(
    *,
    hit: Optional[ScanResult] = None,
    signal: Optional[TradeSignal] = None,
) -> str:
    source = hit or (signal.scan_result if signal and signal.scan_result else None)
    if source is None:
        return ""
    return str(source.criteria.get("pattern") or source.scanner_name or "")


def _scanner(
    *,
    hit: Optional[ScanResult] = None,
    signal: Optional[TradeSignal] = None,
) -> str:
    source = hit or (signal.scan_result if signal and signal.scan_result else None)
    return str(source.scanner_name if source is not None else "")


def _price(
    *,
    hit: Optional[ScanResult],
    signal: Optional[TradeSignal],
    bars: Sequence[Bar],
    fallback: float,
) -> float:
    if hit is not None:
        try:
            return float(
                hit.criteria.get("close")
                or hit.criteria.get("price")
                or hit.criteria.get("entry_price")
                or 0.0
            ) or float(bars[-1].close or 0.0)
        except Exception:
            return float(bars[-1].close or fallback or 0.0)
    if signal is not None:
        return float(signal.entry_price or fallback or bars[-1].close or 0.0)
    return float(fallback or bars[-1].close or 0.0)


def _spread_metrics(
    *,
    reason: str,
    price: float,
    quotes: Sequence[Quote],
) -> tuple[float, float]:
    valid = [q for q in list(quotes)[-5:] if q.ask > q.bid > 0]
    if valid:
        spread = sum(q.ask - q.bid for q in valid) / len(valid)
        mid = sum((q.ask + q.bid) / 2.0 for q in valid) / len(valid)
        spread_pct = spread / mid * 100.0 if mid > 0 else 0.0
        return spread * 100.0, spread_pct

    text = str(reason or "")
    cents_match = re.search(r"\(([\d.]+)c\s*=", text)
    pct_match = re.search(r"=\s*([\d.]+)%", text)
    spread_cents = float(cents_match.group(1)) if cents_match else 0.0
    spread_pct = float(pct_match.group(1)) if pct_match else 0.0
    if spread_pct <= 0 and spread_cents > 0 and price > 0:
        spread_pct = (spread_cents / 100.0) / price * 100.0
    return spread_cents, spread_pct


def _is_risk_reject(reason: str, layer: str = "") -> bool:
    text = "{} {}".format(reason or "", layer or "").lower()
    return "risk too wide" in text or "r:r" in text or "risk/reward" in text


def _risk_metrics(
    *,
    reason: str,
    price: float,
    bars: Sequence[Bar],
    criteria: Dict[str, Any],
) -> tuple[float, float, float]:
    if not _is_risk_reject(reason):
        return 0.0, 0.0, 0.0

    risk_per_share = 0.0
    risk_pct = 0.0
    text = str(reason or "")
    match = re.search(
        r"risk too wide:\s*\$?([\d.]+)\s*\(([\d.]+)%\s+of\s+\$?([\d.]+)\)",
        text,
        re.IGNORECASE,
    )
    if match:
        risk_per_share = float(match.group(1))
        risk_pct = float(match.group(2))
        if price <= 0:
            price = float(match.group(3))

    if risk_per_share <= 0 and price > 0:
        stop = _criteria_float(criteria, "stop_price")
        if stop <= 0:
            stop = _criteria_float(criteria, "base_low")
        if stop <= 0:
            stop = _criteria_float(criteria, "pullback_low")
        if stop > 0 and stop < price:
            risk_per_share = price - stop
            risk_pct = risk_per_share / price * 100.0

    tactical_stop = 0.0
    if bars and price > 0:
        recent = list(bars[-3:])
        lows = [float(b.low or 0.0) for b in recent if float(b.low or 0.0) > 0]
        if lows:
            tactical_stop = min(lows) - 0.02
            if tactical_stop <= 0 or tactical_stop >= price:
                tactical_stop = 0.0

    return risk_per_share, risk_pct, tactical_stop


def _median_bar_range_pct(bars: Sequence[Bar], *, lookback: int = 3) -> float:
    ranges: List[float] = []
    for bar in list(bars)[-lookback:]:
        high = float(bar.high or 0.0)
        low = float(bar.low or 0.0)
        close = float(bar.close or 0.0)
        ref = close if close > 0 else (high + low) / 2.0
        if high > low > 0 and ref > 0:
            ranges.append((high - low) / ref * 100.0)
    if not ranges:
        return 0.0
    ranges.sort()
    mid = len(ranges) // 2
    if len(ranges) % 2:
        return ranges[mid]
    return (ranges[mid - 1] + ranges[mid]) / 2.0


def _criteria_float(criteria: Dict[str, Any], key: str) -> float:
    try:
        return float(criteria.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0
