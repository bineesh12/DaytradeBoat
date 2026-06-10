"""Track A+ setups that were blocked, then label later outcomes.

This module is deliberately observational. It does not approve trades or alter
guards; it builds a daily truth table for tuning from evidence.
"""

from __future__ import annotations

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
        rows = [self._records[rid] for rid in self._order if rid in self._records]
        rows.sort(
            key=lambda r: (
                r.outcome != "missed_opportunity",
                -r.move_after_pct,
                r.blocked_at,
            )
        )
        return [r.as_dict() for r in rows[:limit]]

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
        if "early_exit" in layer:
            return "Review runner/partial exit logic for early shakeouts"
        if "watch only" in text:
            return "Consider promoting only clean A+ watch pattern variants"
        return "Review blocked layer with chart and later outcome"


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
