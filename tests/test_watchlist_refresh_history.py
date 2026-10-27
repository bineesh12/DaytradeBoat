from daytrading.models import BarSource
"""A short REST refresh must not discard previously observed session context."""
from collections import deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from daytrading.models import Bar
from daytrading.runner import AlpacaRunner


START = datetime(2026, 10, 15, 8, tzinfo=timezone.utc)
NOW = START + timedelta(minutes=196)


def bar(minute, *, high=1.22, volume=10):
    return Bar('PFSA', START + timedelta(minutes=minute), 1.2, high, 1.1, 1.2, volume, provenance=BarSource.CANONICAL_INTERVAL)


@pytest.fixture
def runner(monkeypatch):
    r = AlpacaRunner.__new__(AlpacaRunner)
    r._watchlist_pinned = {'SPY'}
    r._watchlist_set = {'SPY', 'PFSA'}
    r._hot_watch = {}
    r._priority_bar_refresh = {'PFSA'}
    r._last_watchlist_bar_refresh = 0.0
    r._watchlist_bar_refresh_sec = 30
    r._max_bars_per_symbol = 1000
    r._bar_buffer = {'PFSA': deque([bar(i, high=1.56 if i == 154 else 1.22) for i in range(180)], maxlen=1000)}
    r.fresh = [bar(i, volume=20) for i in range(165, 195)]
    def fetch(symbols, limit):
        assert symbols == ['PFSA'] and limit == 30
        return {'PFSA': r.fresh}
    r._hist = SimpleNamespace(get_bars=fetch)
    monkeypatch.setattr(AlpacaRunner, '_now_et', classmethod(lambda cls: NOW))
    monkeypatch.setattr('daytrading.runner.datetime', SimpleNamespace(now=lambda tz=None: NOW))
    return r


def refresh(r):
    r._priority_bar_refresh.add('PFSA')
    r._last_watchlist_bar_refresh = 0
    r._refresh_watchlist_bars()
    return list(r._bar_buffer['PFSA'])


def test_preserves_early_high_and_volume_context(runner):
    result = refresh(runner)
    assert len(result) == 195
    assert result[0].ts == START
    assert max(b.high for b in result) == 1.56
    assert sum(b.volume for b in result) == 2250
    # The previous replacement behavior retained only 600 shares / $1.22 high.
    assert sum(b.volume for b in runner.fresh) == 600
    assert max(b.high for b in runner.fresh) == 1.22


def test_overlapping_corrections_replace_without_double_counting(runner):
    result = refresh(runner)
    overlap = next(b for b in result if b.ts == bar(170).ts)
    assert overlap.volume == 20
    assert len({b.ts for b in result}) == len(result)
    assert refresh(runner) == result


def test_preserves_newer_streamed_minutes(runner):
    latest = bar(195, high=1.35, volume=75)
    runner._bar_buffer['PFSA'].append(latest)
    result = refresh(runner)
    assert result[-1] == latest
    assert len(result) == 196


def test_filters_prior_new_york_date_from_both_sources(runner):
    # 02:00 UTC is still the prior New York date.
    prior = bar(-360, high=99, volume=99999)
    runner._bar_buffer['PFSA'].appendleft(prior)
    runner.fresh.insert(0, prior)
    result = refresh(runner)
    assert prior not in result
    assert max(b.high for b in result) == 1.56


def test_respects_history_capacity(runner):
    runner._max_bars_per_symbol = 100
    result = refresh(runner)
    assert len(result) == 100
    assert result[0].ts == bar(95).ts
    assert result[-1].ts == bar(194).ts
    assert runner._bar_buffer['PFSA'].maxlen == 100


def test_empty_refresh_does_not_erase_history(runner):
    runner.fresh = []
    old = runner._bar_buffer['PFSA']
    refresh(runner)
    assert runner._bar_buffer['PFSA'] is old


def test_initial_refresh_sorts_and_deduplicates(runner):
    runner._bar_buffer.clear()
    runner.fresh = [bar(2), bar(1), bar(2, volume=40)]
    result = refresh(runner)
    assert [b.ts for b in result] == [bar(1).ts, bar(2).ts]
    assert result[-1].volume == 40
