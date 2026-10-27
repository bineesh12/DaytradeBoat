"""Shared retry/cache disposition; unknown failures terminate the proposal.

Temporary permits re-evaluation only, never a waiver of expiry, terminal path
invalidation, fresh reclaim, room, costs, sizing or admission controls.
"""
from enum import Enum


class Disposition(str, Enum):
    PASSED = 'passed'
    TEMPORARY = 'temporary'
    TERMINAL = 'terminal'


def classify(reason):
    if not reason:
        return Disposition.PASSED
    from daytrading.execution.entry_outcome import recovery_code
    if recovery_code(reason) is not None:
        return Disposition.TEMPORARY
    text = str(reason).strip().lower()
    # These exact final-boundary messages are retryable only in combination
    # with a structured, explicit not-submitted execution outcome.
    if text in tuple(stage+': current quote invalid, stale or wider than configured spread limit'
                     for stage in ('broker_submission','order_submission','micro_execution')):
        return Disposition.TEMPORARY
    if text.startswith((
        'below vwap (', 'vwap_evidence:', 'micro_spread:',
        'micro_flow_evidence:', 'spread too wide (', 'quote_age:',
        'clock_alignment:', 'micro confirmation: future:',
        'confirmation_age: confirmation age -',
        'event_live_quote:', 'event_live_trade:',
        'micro_profile: fresh trade lost original reclaim level',
        'tape shows selling pressure',
    )) or text in (
        'quote: unlocked two-sided quote required',
        'quote: valid two-sided executable quote required',
    ):
        return Disposition.TEMPORARY
    return Disposition.TERMINAL


def temporary(reason):
    return classify(reason) is Disposition.TEMPORARY
