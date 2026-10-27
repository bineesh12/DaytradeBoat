from datetime import timedelta
from types import SimpleNamespace as NS

import pytest
from test_micro_paper_runtime import AT, hit, minutes
from daytrading.pipeline.engine import TradingPipeline
from daytrading.strategy.scalping.micro_failure import classify, Disposition
from daytrading.strategy.scalping.micro_flow import CoordinatedMicroRetries
from daytrading.strategy.scalping.micro_vwap import deadline, MicroVWAPRetries


@pytest.mark.parametrize('reason', [
    'micro_spread: current spread too wide', 'below VWAP (fresh bid)',
    'micro_flow_evidence: fresh trade required', 'quote_age: quote age 10.1s',
    'clock_alignment: unavailable', 'event_live_quote: no longer held',
    'confirmation_age: confirmation age -0.1s',
    'event_live_trade: current trade no longer confirms reclaim',
    'micro_profile: fresh trade lost original reclaim level',
    'tape shows selling pressure',
])
@pytest.mark.parametrize('coordinated', [False, True])
def test_temporary_classification_matches_cache_and_retry_without_extending_deadline(reason, coordinated):
    h = hit()
    b = CoordinatedMicroRetries(NS(counts={}, observe=lambda *a, **k: None)) if coordinated else MicroVWAPRetries()
    end = deadline(h)
    assert classify(reason) is Disposition.TEMPORARY
    assert TradingPipeline._transient_micro_rejection(h, reason)
    b.decide(h, reason, AT)
    assert h.symbol in b.rows
    b.dirty.add(h.symbol)
    again = b.take({'TEST': minutes()}, AT + timedelta(seconds=.25))
    assert len(again) == 1 and deadline(again[0]) == end
    assert b.take({'TEST': minutes()}, end) == [] and not b.rows


@pytest.mark.parametrize('reason', [
    'native_invalidation: original_support_or_chase_broken',
    'room_risk: 0.1R below 1R', 'stop: too wide', 'extension: chase',
    'micro_profile: trigger volume 0.5x < 0.8x',
    'confirmation_age: original expired', 'micro confirmation: stale: expired',
    'unrecognized failure',
])
def test_terminal_or_unknown_reason_never_becomes_a_retry(reason):
    h = hit()
    b = MicroVWAPRetries()
    assert classify(reason) is Disposition.TERMINAL
    assert not TradingPipeline._transient_micro_rejection(h, reason)
    b.decide(h, reason, AT)
    assert not b.rows and not b.hold_confirmed(h, 'broker_position_sync_stale', AT)
