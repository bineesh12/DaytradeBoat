"""Thread-safe market-data coordination for runner background work."""

from __future__ import annotations

import queue
import time
from threading import Lock, RLock
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple


class MarketDataService:
    """Own hot-watch state and candidate hydration queues behind locks."""

    def __init__(
        self,
        *,
        candidate_queue_max: int = 500,
        candidate_batch_max: int = 10,
        hot_watch_max_symbols: int = 40,
    ) -> None:
        self._candidate_queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=max(1, candidate_queue_max),
        )
        self._candidate_pending: Set[str] = set()
        self._candidate_lock = Lock()
        self._candidate_seq = 0
        self._candidate_batch_max = max(1, candidate_batch_max)
        self._hot_watch: Dict[str, Dict] = {}
        self._hot_watch_lock = RLock()
        self._hot_watch_max_symbols = max(1, hot_watch_max_symbols)

    @property
    def hot_watch(self) -> Dict[str, Dict]:
        """Compatibility view. Prefer methods for new code."""
        return self._hot_watch

    @property
    def candidate_queue(self) -> queue.PriorityQueue:
        """Compatibility view. Prefer queue methods for new code."""
        return self._candidate_queue

    def configure(
        self,
        *,
        candidate_queue_max: Optional[int] = None,
        candidate_batch_max: Optional[int] = None,
        hot_watch_max_symbols: Optional[int] = None,
    ) -> None:
        if candidate_queue_max is not None:
            with self._candidate_lock:
                if self._candidate_queue.empty():
                    self._candidate_queue = queue.PriorityQueue(maxsize=max(1, candidate_queue_max))
        if candidate_batch_max is not None:
            self._candidate_batch_max = max(1, candidate_batch_max)
        if hot_watch_max_symbols is not None:
            self._hot_watch_max_symbols = max(1, hot_watch_max_symbols)

    def enqueue_candidate(self, mover: Dict, *, priority: int, source: str = "") -> bool:
        sym = str(mover.get("symbol", "")).upper().strip()
        if not sym:
            return False
        queued_mover = dict(mover)
        if source:
            queued_mover["_source"] = source
        with self._candidate_lock:
            if sym in self._candidate_pending:
                return False
            self._candidate_pending.add(sym)
            self._candidate_seq += 1
            item = (priority, self._candidate_seq, queued_mover)
        try:
            self._candidate_queue.put_nowait(item)
            return True
        except queue.Full:
            with self._candidate_lock:
                self._candidate_pending.discard(sym)
            return False

    def pull_candidate_batch(self, max_count: Optional[int] = None) -> List[Dict]:
        batch: List[Dict] = []
        limit = max(1, max_count or self._candidate_batch_max)
        while len(batch) < limit:
            try:
                _priority, _seq, mover = self._candidate_queue.get_nowait()
            except queue.Empty:
                break
            sym = str(mover.get("symbol", "")).upper().strip()
            with self._candidate_lock:
                self._candidate_pending.discard(sym)
            batch.append(mover)
        return batch

    def candidate_pending_size(self) -> int:
        return self._candidate_queue.qsize()

    def run_candidate_hydration_worker(
        self,
        *,
        stop_requested: Callable[[], bool],
        pause_state: Callable[[], Tuple[bool, bool]],
        process_batch: Callable[[List[Dict]], int],
        publish_status: Optional[Callable[..., None]] = None,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval: float = 0.25,
        max_iterations: Optional[int] = None,
    ) -> None:
        """Run the candidate hydration worker loop.

        The runner still supplies market-specific callbacks, but queue
        ownership, pending accounting, and pause bookkeeping live here.
        """
        was_paused_for_entry = False
        iterations = 0
        while not stop_requested():
            if max_iterations is not None and iterations >= max_iterations:
                return
            iterations += 1

            paused, pending_entry = pause_state()
            if paused:
                if pending_entry and not was_paused_for_entry and publish_status is not None:
                    publish_status(
                        paused_for_entry=True,
                        pending=self.candidate_pending_size(),
                    )
                was_paused_for_entry = pending_entry
                sleep(poll_interval)
                continue

            if was_paused_for_entry and publish_status is not None:
                publish_status(
                    paused_for_entry=False,
                    pending=self.candidate_pending_size(),
                )
            was_paused_for_entry = False

            batch = self.pull_candidate_batch()
            if not batch:
                sleep(poll_interval)
                continue

            loaded_count = process_batch(batch)
            source = str(batch[0].get("_source", "candidate worker"))
            if publish_status is not None:
                publish_status(
                    batches=1,
                    hydrated=loaded_count,
                    pending=self.candidate_pending_size(),
                    last_batch_size=len(batch),
                    last_loaded=loaded_count,
                    last_source=source,
                )

    def hot_watch_contains(self, symbol: str) -> bool:
        sym = symbol.upper().strip()
        with self._hot_watch_lock:
            return sym in self._hot_watch

    def hot_watch_keys(self) -> Set[str]:
        with self._hot_watch_lock:
            return set(self._hot_watch.keys())

    def hot_watch_items(self) -> List[Tuple[str, Dict]]:
        with self._hot_watch_lock:
            return [(sym, dict(meta)) for sym, meta in self._hot_watch.items()]

    def hot_watch_get(self, symbol: str, default: Optional[Dict] = None) -> Dict:
        sym = symbol.upper().strip()
        with self._hot_watch_lock:
            meta = self._hot_watch.get(sym, default or {})
            return dict(meta) if isinstance(meta, dict) else {}

    def hot_watch_set(self, symbol: str, meta: Dict) -> None:
        sym = symbol.upper().strip()
        with self._hot_watch_lock:
            self._hot_watch[sym] = dict(meta)

    def hot_watch_delete(self, symbol: str) -> bool:
        sym = symbol.upper().strip()
        with self._hot_watch_lock:
            return self._hot_watch.pop(sym, None) is not None

    def hot_watch_len(self) -> int:
        with self._hot_watch_lock:
            return len(self._hot_watch)

    def hot_watch_oldest_symbol(self) -> Optional[str]:
        with self._hot_watch_lock:
            if not self._hot_watch:
                return None
            return sorted(
                self._hot_watch.items(),
                key=lambda item: str(item[1].get("added_at") or ""),
            )[0][0]

    def trim_hot_watch(self, keep_symbols: Iterable[str] = ()) -> List[str]:
        keep = {sym.upper().strip() for sym in keep_symbols}
        removed: List[str] = []
        with self._hot_watch_lock:
            while len(self._hot_watch) > self._hot_watch_max_symbols:
                candidate = self.hot_watch_oldest_symbol()
                if candidate is None or candidate in keep:
                    break
                self._hot_watch.pop(candidate, None)
                removed.append(candidate)
        return removed
