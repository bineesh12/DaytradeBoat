"""Keep observed no-progress timeouts distinct from triggered stop losses."""
from collections import deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from daytrading.exits.manager import ExitManager, TrackedPosition
from daytrading.models import Fill, Side
from daytrading.runner import AlpacaRunner


def test_ncpl_no_progress_exit_is_a_timeout_not_a_stop():
    entered = datetime(2026, 10, 15, 12, 9, 2, tzinfo=timezone.utc)
    em = ExitManager()
    pos = TrackedPosition(symbol='NCPL', side=Side.BUY, quantity=500,
        remaining_qty=500, original_qty=500, entry_price=2.05,
        entry_ts=entered, stop_loss=1.97, risk_per_share=.08,
        first_target_price=2.13, trend_strength=.5, reason='VWAP pullback')
    pos._halt_timeout_secs = 10000
    pos._range_timeout_secs = 10000
    em.track(pos)
    assert not em.check_exits({'NCPL':2.02}, entered+timedelta(seconds=179))
    exits = em.check_exits({'NCPL':2.01}, entered+timedelta(seconds=181))
    assert len(exits) == 1
    assert exits[0].quantity == 500
    assert exits[0].entry_price == 2.01
    assert exits[0].reason.startswith('time_exit: stale_timeout 180s:')
    assert pos.stop_loss == 1.97


def test_exit_reason_reaches_both_dashboard_and_journal():
    r = AlpacaRunner.__new__(AlpacaRunner)
    r._pipeline = SimpleNamespace(exit_manager=ExitManager(), record_realized_exit=lambda *a,**k:-25.0)
    calls = []
    r._hub = SimpleNamespace(on_exit_fill=lambda *a,**k:calls.append(k), add_log=lambda *a,**k:None)
    journal = []
    r._journal = SimpleNamespace(record=lambda event,payload,**kw:journal.append((event,payload)))
    r._exec_timer = SimpleNamespace(cancel=lambda *a,**k:None)
    r._timed_signal_queue = deque()
    r._clear_broker_stop = lambda *a:None
    r._refresh_broker_stop = lambda *a:None
    r._market_phase = lambda:'PRE-MARKET'
    fill = Fill('NCPL', Side.SELL, 500, 2.0, datetime(2026,10,15,12,12,4,tzinfo=timezone.utc))
    reason = 'time_exit: stale_timeout 180s: VWAP pullback'
    assert r._record_trade_exit(fill, 2.05, reason, strategy='vwap_pullback') == -25
    assert calls[0]['reason'] == reason
    assert journal[0][1]['reason'] == reason
