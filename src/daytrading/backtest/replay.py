"""Offline replay of journaled setup decisions through EntryPolicy."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from daytrading.journal.store import TradingJournal
from daytrading.models import Bar, ScanResult, SignalAction, Timeframe, TradeSignal
from daytrading.strategy.entry_policy import EntryDecision, EntryPolicy


@dataclass
class ReplayResult:
    decisions: List[EntryDecision] = field(default_factory=list)
    observed_decisions: List[Dict[str, Any]] = field(default_factory=list)
    skipped: int = 0


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        text = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _bars_from_payload(payload: Dict[str, Any], symbol: str) -> List[Bar]:
    raw = (
        payload.get("candle_snapshot")
        or payload.get("bars")
        or payload.get("bar_snapshot")
        or []
    )
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    bars: List[Bar] = []
    if isinstance(raw, dict):
        raw = raw.get(symbol) or raw.get(symbol.upper()) or raw.get("bars") or []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            timeframe = item.get("timeframe") or item.get("tf") or "1m"
            bars.append(
                Bar(
                    symbol=str(item.get("symbol") or symbol),
                    ts=_parse_ts(item.get("ts") or item.get("timestamp")),
                    open=float(item.get("open", item.get("o", 0.0)) or 0.0),
                    high=float(item.get("high", item.get("h", 0.0)) or 0.0),
                    low=float(item.get("low", item.get("l", 0.0)) or 0.0),
                    close=float(item.get("close", item.get("c", 0.0)) or 0.0),
                    volume=float(item.get("volume", item.get("v", 0.0)) or 0.0),
                    timeframe=Timeframe(timeframe),
                )
            )
        except Exception:
            continue
    return bars


def _signal_from_payload(payload: Dict[str, Any], ts: datetime) -> Optional[TradeSignal]:
    symbol = str(payload.get("symbol") or "").upper().strip()
    if not symbol:
        return None
    action_raw = str(payload.get("action") or SignalAction.ENTER_LONG.value)
    try:
        action = SignalAction(action_raw)
    except ValueError:
        action = SignalAction.ENTER_LONG
    criteria = dict(payload.get("criteria") or {})
    pattern = payload.get("pattern") or criteria.get("pattern")
    if pattern:
        criteria["pattern"] = pattern
    scanner = str(payload.get("scanner") or payload.get("scanner_name") or pattern or "journal_replay")
    score = float(payload.get("score", criteria.get("score", 0.0)) or 0.0)
    price = float(
        payload.get("entry_price")
        or payload.get("price")
        or payload.get("trigger_price")
        or 0.0
    )
    hit = ScanResult(
        symbol=symbol,
        scanner_name=scanner,
        ts=ts,
        score=score,
        criteria=criteria,
        bars=[],
    )
    return TradeSignal(
        symbol=symbol,
        action=action,
        quantity=float(payload.get("quantity", 0.0) or 0.0),
        entry_price=price,
        stop_loss=payload.get("stop_loss"),
        take_profit=payload.get("take_profit"),
        reason=str(payload.get("reason") or "journal replay"),
        scan_result=hit,
    )


class JournalReplayRunner:
    """Replay journaled signals through the same EntryPolicy used live."""

    def __init__(
        self,
        *,
        journal: Optional[TradingJournal] = None,
        db_path: Optional[str] = None,
        policy: Optional[EntryPolicy] = None,
    ) -> None:
        self._journal = journal or TradingJournal(db_path=db_path)
        self._policy = policy or EntryPolicy()

    def replay(
        self,
        *,
        day: Optional[str] = None,
        limit: Optional[int] = None,
        event_types: Sequence[str] = ("signal", "scan_hit"),
    ) -> ReplayResult:
        frames = self._journal.replay_frames(day=day, limit=limit)
        result = ReplayResult()
        for frame in frames:
            payload = frame.get("payload") or {}
            frame_type = str(frame.get("type") or "")
            if frame_type == "entry_decision":
                result.observed_decisions.append(payload)
                continue
            if frame_type not in event_types:
                continue
            ts = _parse_ts(frame.get("ts"))
            signal = _signal_from_payload(payload, ts)
            if signal is None:
                result.skipped += 1
                continue
            bars = _bars_from_payload(payload, signal.symbol)
            if not bars:
                result.skipped += 1
                continue
            decision = self._policy.evaluate(
                signal,
                bars=bars,
                stage="journal_replay",
                metadata={"source_event": frame_type, "replay_ts": frame.get("ts")},
            )
            result.decisions.append(decision)
        return result
