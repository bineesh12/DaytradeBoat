from __future__ import annotations

import time

from daytrading.runner import AlpacaRunner


def _runner() -> AlpacaRunner:
    runner = object.__new__(AlpacaRunner)
    runner._network_failure_times = []
    runner._hydrate_paused_until = 0.0
    return runner


def test_partial_bar_fetch_misses_do_not_pause_hydration() -> None:
    runner = _runner()

    runner._record_bar_fetch_failures(9, attempted_count=25)
    runner._record_bar_fetch_failures(9, attempted_count=25)
    runner._record_bar_fetch_failures(9, attempted_count=18)
    runner._record_bar_fetch_failures(1, attempted_count=1)
    runner._record_bar_fetch_failures(4, attempted_count=4)

    assert runner._network_failure_times == []
    assert runner._hydrate_paused_until == 0.0


def test_repeated_weak_bar_fetch_batches_pause_hydration() -> None:
    runner = _runner()

    runner._record_bar_fetch_failures(20, attempted_count=25)
    runner._record_bar_fetch_failures(18, attempted_count=25)
    runner._record_bar_fetch_failures(17, attempted_count=25)

    assert runner._network_failure_times == []
    assert runner._hydrate_paused_until > time.time()
