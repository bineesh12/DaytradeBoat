"""Synthetic timing contracts; no actual broker fills or target profits."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from daytrading.replay.predict import PredictClock
from daytrading.replay.scheduler import VirtualScheduler, VirtualEvent
from daytrading.replay.heartbeat import run_native_heartbeat

T = datetime(2026, 10, 16, 8, tzinfo=timezone.utc)


def drive(arrivals=(), due=None, io_delay=0., end=3.):
    clock = PredictClock(T.isoformat(), 0)
    scheduler = VirtualScheduler(clock, T+timedelta(seconds=end))
    queue, steps, phases = [], [], []
    runner = SimpleNamespace(_cycle_interval=1., _new_data=VirtualEvent(scheduler),
                             _exec_timer=SimpleNamespace(seconds_until_next_timeout=lambda: due),
                             _market_phase=lambda: 'PRE-MARKET')

    def drain():
        result = bool(queue)
        queue.clear()
        return result
    runner._drain_events = drain

    def invoke(name, *args):
        if name == '_session_iteration_context':
            return clock.now, True, True
        if name == '_session_step':
            steps.append((round((clock.now-T).total_seconds(), 6), args))
            runner._new_data.clear()
            if io_delay:
                clock.advance(clock.now+timedelta(seconds=io_delay))
        elif name == '_handle_market_phase_transition':
            phases.append(args)
        else:
            raise AssertionError(name)

    def market():
        for elapsed, kind in arrivals:
            clock.advance(T+timedelta(seconds=elapsed))
            queue.append(kind)
            if kind == 'bar':
                runner._new_data.set()

    scheduler.spawn('a-market', T, market)
    scheduler.spawn('main', T, lambda: run_native_heartbeat(runner, clock, invoke, None))
    scheduler.run()
    return steps


def test_completed_bar_wakes_native_exit_heartbeat_before_one_second():
    steps = drive([(0.25, 'bar')])
    assert steps[0][0] == .25
    assert steps[0][1][1:3] == (True, True)


def test_quote_does_not_invent_a_bar_wakeup():
    steps = drive([(.25, 'quote')])
    assert steps[0][0] == 1.
    assert steps[0][1][1:3] == (False, True)


def test_http_wait_advances_native_heartbeat_without_a_second_custom_timer():
    steps = drive([(.25, 'bar')], io_delay=.4)
    assert [t for t, _ in steps[:2]] == [.25, 1.65]


def test_timer_deadline_uses_native_quarter_second_floor():
    steps = drive(due=.05)
    assert steps[0][0] == .25


def test_future_suffix_cannot_change_earlier_heartbeat():
    a = drive([(.25, 'bar')], end=.9)
    b = drive([(.25, 'bar'), (1.2, 'bar')], end=.9)
    assert a == b


def test_all_queued_events_are_drained_at_native_step():
    steps = drive([(.1, 'quote'), (.2, 'trade'), (.3, 'bar')])
    assert steps[0][0] == .3
    assert steps[0][1][1:3] == (True, True)
