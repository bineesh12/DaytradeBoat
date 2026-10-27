"""Regressions for the repeated sparse IEX fetch observed in paper supervision."""
from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

from daytrading.runner import AlpacaRunner


def runner():
    return SimpleNamespace(
        _bar_hydrate_paused=lambda: False,
        _watchlist_pinned=set(), _watchlist_set=set(),
        _bar_buffer={'AIRI': deque([object()] * 6)},
        _prior_day_stats={'AIRI': object()},
        _hod_seed_blacklist=set(), _hod_seed_retries={},
        _hod_seed_last_attempt={}, _event_queue=Queue(),
        _fetch_session_bars=Mock(return_value={'AIRI': [object()] * 6}),
        _fetch_prior_day_stats=Mock(return_value={}),
        _ensure_streaming_symbols=Mock(), _enqueue_hod_seed_symbols=Mock(),
    )


def test_sparse_session_waits_one_minute_before_retry():
    r = runner()
    with patch('daytrading.runner.time.monotonic', return_value=100):
        AlpacaRunner._process_hod_seed_batch(r, ['AIRI'])
    for instant in (101, 120, 159.9):
        with patch('daytrading.runner.time.monotonic', return_value=instant):
            AlpacaRunner._process_hod_seed_batch(r, ['AIRI'])
    assert r._fetch_session_bars.call_count == 1
    assert r._event_queue.qsize() == 1
    assert not r._hod_seed_blacklist
    with patch('daytrading.runner.time.monotonic', return_value=160):
        AlpacaRunner._process_hod_seed_batch(r, ['AIRI'])
    assert r._fetch_session_bars.call_count == 2
    r._fetch_prior_day_stats.assert_not_called()


def test_complete_bars_missing_prior_fetches_only_prior():
    r = runner()
    r._bar_buffer['AIRI'] = deque([object()] * 10)
    r._prior_day_stats.clear()
    r._fetch_prior_day_stats.return_value = {'AIRI': object()}
    AlpacaRunner._process_hod_seed_batch(r, ['AIRI'])
    r._fetch_session_bars.assert_not_called()
    r._fetch_prior_day_stats.assert_called_once_with(['AIRI'])
    event = r._event_queue.get_nowait()
    assert event.prior_day_stats['AIRI'] is not None


def test_sparse_cooldown_does_not_block_another_symbol():
    r = runner()
    r._hod_seed_last_attempt['AIRI'] = 100
    with patch('daytrading.runner.time.monotonic', return_value=101):
        AlpacaRunner._process_hod_seed_batch(r, ['AIRI', 'NEW'])
    r._fetch_session_bars.assert_called_once_with(['NEW'])
    assert r._hod_seed_last_attempt['NEW'] == 101
