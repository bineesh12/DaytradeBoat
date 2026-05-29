"""Thread-safe store for HOD Momentum alert rows."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Callable, List, Optional

from daytrading.scanner.hod_momentum.models import HODAlertRow


def _row_time_key(row: HODAlertRow) -> datetime:
    try:
        return datetime.fromisoformat(row.time.replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


class HODAlertStore:
    """In-memory alert feed for the dashboard."""

    def __init__(self, max_rows: int = 200, ttl_minutes: float = 5.0) -> None:
        self._max_rows = max_rows
        self._ttl_minutes = ttl_minutes
        self._rows: List[HODAlertRow] = []
        self._lock = Lock()
        self._on_change: Optional[Callable[[List[dict]], None]] = None

    def set_on_change(self, callback: Callable[[List[dict]], None]) -> None:
        self._on_change = callback

    def add(self, row: HODAlertRow, *, replace_same_key: bool = True) -> None:
        key = row.symbol + "|" + row.alert_name
        with self._lock:
            if replace_same_key:
                self._rows = [
                    r for r in self._rows
                    if (r.symbol + "|" + r.alert_name) != key
                ]
            self._rows.append(row)
            self._sort_and_trim()
            payload = self.snapshot_unlocked()
        self._notify(payload)

    def merge_status(
        self,
        symbol: str,
        *,
        verified: bool,
        reject_reason: Optional[str],
    ) -> None:
        with self._lock:
            for row in self._rows:
                if row.symbol == symbol:
                    row.verified = verified
                    row.reject_reason = reject_reason
            self._sort_and_trim()
            payload = self.snapshot_unlocked()
        self._notify(payload)

    def snapshot(self) -> List[dict]:
        with self._lock:
            self._sort_and_trim()
            return self.snapshot_unlocked()

    def snapshot_unlocked(self) -> List[dict]:
        return [r.to_dict() for r in self._rows]

    def _sort_and_trim(self) -> None:
        now = datetime.now(timezone.utc)
        cutoff_secs = self._ttl_minutes * 60
        self._rows = [
            r for r in self._rows
            if (now - _row_time_key(r)).total_seconds() < cutoff_secs
        ]
        self._rows.sort(key=_row_time_key, reverse=True)
        if len(self._rows) > self._max_rows:
            self._rows = self._rows[: self._max_rows]

    def _notify(self, payload: List[dict]) -> None:
        if self._on_change:
            self._on_change(payload)
