"""Native heartbeat driver for recorded and conditional offline replay.

No strategy decisions or alternate exit cadence live here. I/O and market
arrival actors use the supplied virtual clock; the runner owns all decisions.
"""
from .engine import ACTIVE


def run_native_heartbeat(runner, clock, invoke, context, *, stop_requested=lambda: False):
    previous_phase = None
    last_phase_check = 0.
    while not stop_requested():
        now_ts = clock.now.timestamp()
        now_et, entries, patterns = invoke('_session_iteration_context')
        if now_ts-last_phase_check > 60:
            token = ACTIVE.set(context)
            try:
                phase = runner._market_phase()
            finally:
                ACTIVE.reset(token)
            if previous_phase is not None and phase != previous_phase:
                invoke('_handle_market_phase_transition', previous_phase, phase, now_et)
            previous_phase = phase
            last_phase_check = now_ts
        timeout = float(runner._cycle_interval)
        token = ACTIVE.set(context)
        try:
            wait_policy = getattr(runner, '_event_wait_timeout', None)
            if callable(wait_policy):
                timeout = wait_policy()
            else:
                # Lightweight compatibility fixtures have no native FIFO.
                due = runner._exec_timer.seconds_until_next_timeout()
                if due is not None:
                    timeout = min(timeout, max(.25, due))
            market_wait = getattr(runner, '_wait_for_market_events', None)
            got_data = (market_wait(timeout) if callable(market_wait)
                        else runner._new_data.wait(timeout=timeout))
            native_turn = callable(getattr(runner, '_process_market_turn', None))
            if not native_turn:
                got_events = runner._drain_events()
        finally:
            ACTIVE.reset(token)
        if native_turn:
            invoke('_process_market_turn', now_ts, got_data, entries, patterns)
            continue
        args = (now_ts, got_data, got_events, entries, patterns)
        fifo = getattr(runner, '_event_queue', None)
        if fifo is not None:
            args += (not fifo.empty(),)
        invoke('_session_step', *args)
