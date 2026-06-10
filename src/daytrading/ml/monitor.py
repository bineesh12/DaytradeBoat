"""ML model monitoring — shadow mode, auto-disable, and stats.

Shadow mode: when ML rejects an entry, tracks the stock price for 5 minutes
to determine if the rejection was correct (price went down) or wrong (price went up).

Auto-disable: if ML rejects too many entries or shadow accuracy drops below 50%,
the model is automatically disabled to prevent harm.
"""

from __future__ import annotations

import json
import logging
import re
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
    max_price: float = 0.0
    min_price: float = 0.0


@dataclass
class MLMonitorStats:
    """Daily ML monitoring statistics."""
    entries_passed: int = 0
    entries_rejected_by_ml: int = 0
    entries_rejected_by_rules: int = 0
    shadow_correct: int = 0  # ML rejected and price went down (good)
    shadow_wrong: int = 0    # ML rejected but price went up (bad)
    elite_false_rejects: int = 0
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
            "elite_false_rejects": self.elite_false_rejects,
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
    ELITE_FALSE_REJECT_MIN_SCORE = 95
    ELITE_FALSE_REJECT_RUN_PCT = 8.0
    ELITE_FALSE_REJECT_DISABLE_COUNT = 2

    def __init__(self) -> None:
        self._stats = MLMonitorStats()
        self._shadow_entries: Deque[ShadowEntry] = deque(maxlen=100)
        self._lock = threading.Lock()
        self._model_enabled = True
        self._prices: Dict[str, float] = {}  # latest known prices
        self._rule_rejection_keys: set = set()
        self._ml_rejection_keys: set = set()

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

    def _ml_rejection_key(self, symbol: str, price: float, score: int) -> tuple:
        price_bucket = round(price, 2) if price > 0 else 0.0
        score_bucket = int(score // 5) * 5
        return (symbol, price_bucket, score_bucket)

    def record_ml_rejection(
        self,
        symbol: str,
        price: float,
        ml_prob: float,
        score: int,
        *,
        counted: bool = True,
    ) -> None:
        """Called when ML rejects or disagrees with an entry that passed rules.

        ``counted=False`` is for strong rule-score soft passes: we still track
        shadow outcome, but do not treat it as a live ML block for auto-disable.
        """
        with self._lock:
            key = self._ml_rejection_key(symbol, price, score)
            if key in self._ml_rejection_keys:
                return
            self._ml_rejection_keys.add(key)
            if counted:
                self._stats.entries_rejected_by_ml += 1
            self._shadow_entries.append(ShadowEntry(
                symbol=symbol,
                price=price,
                ml_prob=ml_prob,
                reject_time=time.time(),
                score=score,
                max_price=price,
                min_price=price,
            ))

    @staticmethod
    def _normalize_rule_reason(reason: Optional[str]) -> str:
        """Collapse changing numbers so repeated rejects group together."""
        if not reason:
            return ""
        return re.sub(r"\d+(?:\.\d+)?", "#", reason)

    def record_rule_rejection(
        self,
        symbol: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Called when rules reject before ML scores.

        Count each symbol/reason once per day so a stale symbol rejected every
        scan cycle does not make the dashboard look stricter than it is.
        """
        with self._lock:
            if symbol or reason:
                key = (
                    symbol or "",
                    self._normalize_rule_reason(reason),
                )
                if key in self._rule_rejection_keys:
                    return
                self._rule_rejection_keys.add(key)
            self._stats.entries_rejected_by_rules += 1

    def update_price(self, symbol: str, price: float) -> None:
        """Update latest price for shadow tracking."""
        with self._lock:
            self._prices[symbol] = price
            for entry in self._shadow_entries:
                if entry.symbol == symbol and price > 0:
                    entry.max_price = max(entry.max_price or entry.price, price)
                    entry.min_price = min(entry.min_price or entry.price, price)

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
            max_run_pct = (entry.max_price - entry.price) / entry.price * 100

            # ML was wrong if the setup gave a usable scalp after the reject,
            # even if it later faded before the 5-minute shadow label expires.
            ml_was_correct = price_change_pct <= 0 and max_run_pct < 3.0
            elite_false_reject = (
                not ml_was_correct
                and entry.score >= self.ELITE_FALSE_REJECT_MIN_SCORE
                and max_run_pct >= self.ELITE_FALSE_REJECT_RUN_PCT
            )

            with self._lock:
                if ml_was_correct:
                    self._stats.shadow_correct += 1
                else:
                    self._stats.shadow_wrong += 1
                    if elite_false_reject:
                        self._stats.elite_false_rejects += 1

            self._log_shadow_result(
                entry, current_price, price_change_pct, max_run_pct, ml_was_correct,
            )

        # Auto-disable check
        self._check_auto_disable()

    def _check_auto_disable(self) -> None:
        """Disable model if performance is degraded."""
        with self._lock:
            total_shadow = self._stats.shadow_correct + self._stats.shadow_wrong
            total_scored = self._stats.total_scored

            # Need minimum samples before making decisions
            if self._stats.elite_false_rejects >= self.ELITE_FALSE_REJECT_DISABLE_COUNT:
                self._model_enabled = False
                self._stats.model_disabled = True
                self._stats.disable_reason = (
                    "{} elite false ML rejects moved >= {:.0f}% after reject".format(
                        self._stats.elite_false_rejects,
                        self.ELITE_FALSE_REJECT_RUN_PCT,
                    )
                )
                logger.warning(
                    "ML MODEL AUTO-DISABLED: %s", self._stats.disable_reason,
                )
                return

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
        change_pct: float, max_run_pct: float, ml_correct: bool,
    ) -> None:
        """Persist shadow result to file."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": entry.symbol,
                "entry_price": entry.price,
                "final_price": round(final_price, 4),
                "max_price": round(entry.max_price, 4),
                "max_run_pct": round(max_run_pct, 4),
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
            self._rule_rejection_keys.clear()
            self._ml_rejection_keys.clear()
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
