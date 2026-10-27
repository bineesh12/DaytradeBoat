from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from test_bkyi_refresh_repair import broker_case
from daytrading.models import PortfolioState
from daytrading.runner import AlpacaRunner
from test_micro_paper_runtime import AT, signal
from daytrading.strategy.entry_policy import EntryPolicy
from daytrading.strategy.scalping.micro_flow import CoordinatedMicroRetries
from daytrading.strategy.scalping.micro_vwap import deadline, identity


@pytest.mark.parametrize('uncertainty', ['partial', 'unknown_response', 'cancel', 'cancel_ids', 'protection'])
def test_completed_empty_snapshot_cannot_clear_inventory_uncertainty(uncertainty):
    b, future, wake = broker_case()
    r = AlpacaRunner.__new__(AlpacaRunner)
    r._broker = b
    r._market_wakeup = wake
    r._pipeline = NS(portfolio=PortfolioState(cash=1000), exit_manager=NS(tracked={}), _scanners=[])
    r._push_positions_from_alpaca = Mock()
    r._flush_micro_control_evidence = Mock()
    with patch('daytrading.execution.alpaca_broker.time.monotonic', return_value=100):
        b.bind_position_refresh_owner(r._position_refresh_context(), wake)
        b.get_positions()
        future.set_result([])
        if uncertainty == 'partial':
            r._partial_entries = {'TEST': object()}
        elif uncertainty == 'unknown_response':
            b._unsettled_orders = {'TEST': (None, 'original-client-id')}
        elif uncertainty == 'cancel':
            b._cancel_unconfirmed = True
        elif uncertainty == 'cancel_ids':
            b._unconfirmed_cancels = {'old-order'}
        else:
            b._safety_blocked = True
        r._service_position_refresh()
        assert b._position_sync_stale
        assert b._position_future is future
        r._push_positions_from_alpaca.assert_not_called()
        # Clearing the control cannot make a pre-operation snapshot authoritative.
        r._partial_entries = {}
        b._unsettled_orders = {}
        b._cancel_unconfirmed = False
        b._unconfirmed_cancels = set()
        b._safety_blocked = False
        r._service_position_refresh()
        assert b._position_sync_stale
        r._push_positions_from_alpaca.assert_not_called()
        assert b.pop_position_refresh_events()[-1]['reason'] == 'position_version_changed'


@pytest.mark.parametrize('outcome', ['unknown_response', 'partial_fill'])
def test_consumed_id_cannot_resubmit_after_uncertain_or_partial_outcome(outcome):
    sig = signal()
    book = CoordinatedMicroRetries(NS(counts={}, observe=lambda *a, **k: None))
    r = AlpacaRunner.__new__(AlpacaRunner)
    r._broker = NS(_unsettled_orders={})
    r._hub = NS(trading_paused=False)
    r._pipeline = NS(_scanners=[NS(name=sig.scan_result.scanner_name, vwap_retries=book)],
                     _symbol_entry_counts={})
    r._exec_timer = NS(pending_symbols=set())
    r._entry_policy = EntryPolicy()
    r._record_entry_decision = Mock()
    r._record_entry_reject = Mock()
    r._flush_micro_control_evidence = Mock()
    r._micro_market_now = lambda: AT
    r._entry_block_reason = lambda *a: 'broker_safety_pending' if r._broker._unsettled_orders else None
    def execute(_):
        if outcome == 'unknown_response':
            r._broker._unsettled_orders['TEST'] = (None, 'same-client-id')
        else:
            r._partial_entries = {'TEST': NS(quantity=10, active_submit=False)}
            r._entry_reconcile_epoch = 1
    r._execute_timed_signal = Mock(side_effect=execute)
    r._dispatch_confirmed_entry(sig)
    assert book.terminal[identity(sig.scan_result)] == deadline(sig.scan_result)
    # A duplicate wake, then completed reconciliation, cannot issue this ID again.
    r._dispatch_confirmed_entry(sig)
    r._broker._unsettled_orders.clear()
    r._partial_entries = {}
    r._dispatch_confirmed_entry(sig)
    r._execute_timed_signal.assert_called_once()
