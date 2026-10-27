"""Additional release probes, intentionally outside the frozen candidate tree."""
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

from test_micro_clock_alignment import execution_context
from test_micro_paper_runtime import AT, signal
from daytrading.runner import AlpacaRunner
from daytrading.strategy.entry_policy import EntryPolicy
from daytrading.strategy.scalping.micro_pullback_runtime import MicroPullbackScanner


def case(wide=True):
    sig = signal()
    ctx = execution_context(AT, quote_offset=0)
    ctx.quote_buffer['TEST'][-1] = replace(ctx.quote_buffer['TEST'][-1], bid=2.37 if wide else 2.41, ask=2.42)
    ctx.pipeline._verifiers['micro_pullback_reclaim']._tick_buffer['TEST'][-1] = replace(
        ctx.pipeline._verifiers['micro_pullback_reclaim']._tick_buffer['TEST'][-1], price=2.42)
    scanner = MicroPullbackScanner()
    scanner.track_native(sig.scan_result, AT)
    ctx.pipeline._scanners = [scanner]
    r = AlpacaRunner.__new__(AlpacaRunner)
    r._pipeline = ctx.pipeline
    r._broker = ctx.broker
    r._hub = NS(trading_paused=False)
    r._entry_policy = EntryPolicy(min_price=1, max_price=20)
    r._entry_policy.micro_pullback_only = True
    r._exec_timer = NS(pending_symbols=set())
    r._record_entry_decision = Mock()
    r._entry_block_reason = lambda *args: None
    r._micro_market_now = lambda: AT
    r._flush_micro_control_evidence = Mock()
    r._entry_execution_context = lambda: ctx
    ctx.record_entry_reject = r._record_entry_reject
    return r, ctx, sig, scanner


def test_valid_narrow_quote_control_reaches_submission():
    r, ctx, sig, scanner = case(wide=False)
    r._dispatch_confirmed_entry(sig)
    ctx.broker.submit.assert_called_once()


def test_order_plan_spread_refusal_can_retry_same_original_setup():
    r, ctx, sig, scanner = case()
    r._dispatch_confirmed_entry(sig)
    ctx.broker.submit.assert_not_called()
    rejects = [c.args[0] for c in r._record_entry_decision.call_args_list
               if not c.args[0].passed]
    assert any(x.stage == 'micro_execution' and x.reason.startswith('spread:') for x in rejects)
    book = scanner.vwap_retries
    setup = sig.scan_result.criteria['micro_setup_id']
    assert 'TEST' in book.rows and setup not in book.terminal, (
        'Recoverable order-plan spread refusal consumed the original setup before any SDK call')
    # No expiry/threshold relaxation: only the quote becomes acceptable.
    ctx.quote_buffer['TEST'][-1] = replace(ctx.quote_buffer['TEST'][-1], bid=2.41, ask=2.42)
    r._dispatch_confirmed_entry(sig)
    ctx.broker.submit.assert_called_once()
