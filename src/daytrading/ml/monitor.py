"""ML model monitoring — shadow mode, auto-disable, and stats.

Shadow mode: when ML rejects an entry, tracks the stock price for 5 minutes
to determine if the rejection was correct (price went down) or wrong (price went up).

Auto-disable: if ML rejects too many entries or shadow accuracy drops below 50%,
the model is automatically disabled to prevent harm.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "ml"
_SHADOW_FILE = _DATA_DIR / "shadow_results.jsonl"


@dataclass
class ShadowEntry:
    """A rejected entry being tracked to see if ML was right."""
    symbol: str
    price: float
    ml_prob: float
    reject_time: float  # time.time()
    score: int


@dataclass
class MLMonitorStats:
    """Daily ML monitoring statistics."""
    entries_passed: int = 0
    entries_rejected_by_ml: int = 0
    entries_rejected_by_rules: int = 0
    shadow_correct: int = 0  # ML rejected and price went down (good)
    shadow_wrong: int = 0    # ML rejected but price went up (bad)
    model_disabled: bool = False
    disable_reason: str = ""

    @property
    def total_scored(self) -> int:
        return self.entries_passed + self.entries_rejected_by_ml

    @property
    def rejection_rate(self) -> float:
        if self.total_scored == 0:
            return 0.0
        return self.entries_rejected_by_ml / self.total_scored

    @property
    def shadow_accuracy(self) -> float:
        total = self.shadow_correct + self.shadow_wrong
        if total == 0:
            return 1.0  # no data yet, assume good
        return self.shadow_correct / total

    def to_dict(self) -> dict:
        return {
            "entries_passed": self.entries_passed,
            "entries_rejected_by_ml": self.entries_rejected_by_ml,
            "entries_rejected_by_rules": self.entries_rejected_by_rules,
            "shadow_correct": self.shadow_correct,
            "shadow_wrong": self.shadow_wrong,
            "shadow_accuracy_pct": round(self.shadow_accuracy * 100, 1),
            "rejection_rate_pct": round(self.rejection_rate * 100, 1),
            "model_disabled": self.model_disabled,
            "disable_reason": self.disable_reason,
        }


class MLMonitor:
    """Monitors ML model performance and auto-disables if degraded."""

    SHADOW_TIMEOUT_SEC = 300  # Track for 5 minutes
    MAX_REJECTION_RATE = 0.92  # Disable if rejecting > 92%
    MIN_SHADOW_ACCURACY = 0.35  # Disable if shadow accuracy < 35%
    MIN_SAMPLES_FOR_DISABLE = 25  # Need at least N scored entries before auto-disable

    def __init__(self) -> None:
        self._stats = MLMonitorStats()
        self._shadow_entries: Deque[ShadowEntry] = deque(maxlen=100)
        self._lock = threading.Lock()
        self._model_enabled = True
        self._prices: Dict[str, float] = {}  # latest known prices

    @property
    def is_model_enabled(self) -> bool:
        return self._model_enabled

    @property
    def stats(self) -> MLMonitorStats:
        return self._stats

    def record_entry_passed(self) -> None:
        """Called when an entry passes both rules and ML."""
        with self._lock:
            self._stats.entries_passed += 1

    def record_ml_rejection(self, symbol: str, price: float, ml_prob: float, score: int) -> None:
        """Called when ML rejects an entry that passed rules."""
        with self._lock:
            self._stats.entries_rejected_by_ml += 1
            self._shadow_entries.append(ShadowEntry(
                symbol=symbol,
                price=price,
                ml_prob=ml_prob,
                reject_time=time.time(),
                score=score,
            ))

    def record_rule_rejection(self) -> None:
        """Called when rules reject (before ML even scores)."""
        with self._lock:
            self._stats.entries_rejected_by_rules += 1

    def update_price(self, symbol: str, price: float) -> None:
        """Update latest price for shadow tracking."""
        self._prices[symbol] = price

    def check_shadow_outcomes(self) -> None:
        """Check if any shadow entries have expired (5 min) and score them."""
        now = time.time()
        with self._lock:
            resolved = []
            remaining = deque(maxlen=100)
            for entry in self._shadow_entries:
                if now - entry.reject_time >= self.SHADOW_TIMEOUT_SEC:
                    resolved.append(entry)
                else:
                    remaining.append(entry)
            self._shadow_entries = remaining

        for entry in resolved:
            current_price = self._prices.get(entry.symbol)
            if current_price is None:
                continue

            price_change_pct = (current_price - entry.price) / entry.price * 100

            # ML was correct if price went DOWN (rejecting was right)
            ml_was_correct = price_change_pct <= 0

            with self._lock:
                if ml_was_correct:
                    self._stats.shadow_correct += 1
                else:
                    self._stats.shadow_wrong += 1

            self._log_shadow_result(entry, current_price, price_change_pct, ml_was_correct)

        # Auto-disable check
        self._check_auto_disable()

    def _check_auto_disable(self) -> None:
        """Disable model if performance is degraded."""
        with self._lock:
            total_shadow = self._stats.shadow_correct + self._stats.shadow_wrong
            total_scored = self._stats.total_scored

            # Need minimum samples before making decisions
            if total_scored < self.MIN_SAMPLES_FOR_DISABLE:
                return

            # Check 1: Rejecting too many entries
            if self._stats.rejection_rate > self.MAX_REJECTION_RATE:
                self._model_enabled = False
                self._stats.model_disabled = True
                self._stats.disable_reason = (
                    "rejection rate {:.0f}% > {:.0f}% max".format(
                        self._stats.rejection_rate * 100,
                        self.MAX_REJECTION_RATE * 100,
                    )
                )
                logger.warning(
                    "ML MODEL AUTO-DISABLED: %s", self._stats.disable_reason,
                )
                return

            # Check 2: Shadow accuracy too low.
            # Shadow outcomes are noisy during the first few rejects, so wait
            # for the same minimum sample size before disabling the live model.
            if (
                total_shadow >= self.MIN_SAMPLES_FOR_DISABLE
                and self._stats.shadow_accuracy < self.MIN_SHADOW_ACCURACY
            ):
                self._model_enabled = False
                self._stats.model_disabled = True
                self._stats.disable_reason = (
                    "shadow accuracy {:.0f}% < {:.0f}% min ({} correct, {} wrong)".format(
                        self._stats.shadow_accuracy * 100,
                        self.MIN_SHADOW_ACCURACY * 100,
                        self._stats.shadow_correct,
                        self._stats.shadow_wrong,
                    )
                )
                logger.warning(
                    "ML MODEL AUTO-DISABLED: %s", self._stats.disable_reason,
                )

    def _log_shadow_result(
        self, entry: ShadowEntry, final_price: float,
        change_pct: float, ml_correct: bool,
    ) -> None:
        """Persist shadow result to file."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": entry.symbol,
                "entry_price": entry.price,
                "final_price": round(final_price, 4),
                "change_pct": round(change_pct, 4),
                "ml_prob": entry.ml_prob,
                "score": entry.score,
                "ml_correct": ml_correct,
            }
            with open(_SHADOW_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def reset_daily(self) -> None:
        """Reset stats for a new trading day (keep model enabled/disabled state)."""
        with self._lock:
            was_disabled = self._stats.model_disabled
            self._stats = MLMonitorStats()
            if was_disabled:
                # Re-enable for new day — give model fresh chance
                self._model_enabled = True
                logger.info("ML MODEL re-enabled for new trading day")

    def get_summary_line(self) -> str:
        """One-line summary for logs."""
        s = self._stats
        shadow_total = s.shadow_correct + s.shadow_wrong
        if shadow_total > 0:
            return (
                "ML: {}/{} passed, {} ML-rejected, shadow {}/{} correct ({:.0f}%){}".format(
                    s.entries_passed, s.total_scored,
                    s.entries_rejected_by_ml,
                    s.shadow_correct, shadow_total,
                    s.shadow_accuracy * 100,
                    " [DISABLED]" if s.model_disabled else "",
                )
            )
        return "ML: {}/{} passed, {} ML-rejected, shadow: pending{}".format(
            s.entries_passed, s.total_scored,
            s.entries_rejected_by_ml,
            " [DISABLED]" if s.model_disabled else "",
        )
