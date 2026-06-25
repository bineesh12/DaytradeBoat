"""Append-only logger for Warrior ignition candidates — the adaptive dataset.

Every ignition the model scores live is written as one JSONL row (timestamp,
symbol, conviction, entry/stop, and the 11 features). Outcomes are labelled
OFFLINE later by replaying the saved bars, so the retrain job can join these
rows with what actually happened and produce the next, better model.

This is the data stream that makes the model adaptive: log -> retrain offline ->
validate -> deploy. It writes only; it never touches trading.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Optional

from daytrading.strategy.warrior_ignition import IgnitionSignal

_DEFAULT_PATH = os.path.join("data", "ml", "warrior_ignition_candidates.jsonl")


class IgnitionLogger:
    """Thread-safe JSONL appender for ignition candidates."""

    def __init__(self, path: str = _DEFAULT_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # de-dup: only log a symbol's candidate once per distinct ignition bar
        self._last_logged: dict[str, str] = {}

    def log(
        self,
        *,
        ts_iso: str,
        symbol: str,
        signal: IgnitionSignal,
        would_enter: bool,
        size_factor: float,
        session: str = "premarket",
    ) -> bool:
        """Append one candidate row. Returns True if written (False if de-duped)."""
        sym = symbol.upper()
        key = f"{sym}@{ts_iso}"
        with self._lock:
            if self._last_logged.get(sym) == ts_iso:
                return False
            self._last_logged[sym] = ts_iso
            row = {
                "ts": ts_iso,
                "symbol": sym,
                "session": session,
                "conviction": round(float(signal.conviction), 4),
                "would_enter": bool(would_enter),
                "size_factor": round(float(size_factor), 3),
                "entry_ref": round(float(signal.entry_ref), 4),
                "stop": round(float(signal.stop), 4),
                "base_high": round(float(signal.base_high), 4),
                "features": {k: round(float(v), 6) for k, v in signal.features.items()},
            }
            row["kind"] = "candidate"
            try:
                with open(self._path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                return True
            except OSError:
                return False

    def log_outcome(
        self,
        *,
        ts_iso: str,
        symbol: str,
        entry: float,
        stop: float,
        exit_price: float,
        realized_pnl: float,
        bars_held: int,
        exit_reason: str,
    ) -> bool:
        """Append the trade OUTCOME for a logged candidate. Joined with the
        candidate row (symbol + entry ts) this gives the full entry->exit pair;
        the bar-by-bar trajectory in between is reconstructed offline from the
        saved bars for the future RL exit model."""
        risk = max(1e-9, float(entry) - float(stop))
        row = {
            "kind": "outcome",
            "ts": ts_iso,
            "symbol": symbol.upper(),
            "entry": round(float(entry), 4),
            "stop": round(float(stop), 4),
            "exit": round(float(exit_price), 4),
            "realized_pnl": round(float(realized_pnl), 4),
            "realized_R": round((float(exit_price) - float(entry)) / risk, 3),
            "bars_held": int(bars_held),
            "exit_reason": str(exit_reason),
        }
        with self._lock:
            try:
                with open(self._path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                return True
            except OSError:
                return False


_LOGGER: Optional[IgnitionLogger] = None


def get_logger(path: str = _DEFAULT_PATH) -> IgnitionLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = IgnitionLogger(path)
    return _LOGGER
