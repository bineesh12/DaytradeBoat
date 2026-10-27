"""Bounded broker-UTC calibration for market timestamps, independent of tape.

The operating-system clock and strategy durations are not changed. A fresh,
validated broker anchor advances using monotonic time. Quotes/trades can never
set the anchor or make themselves fresh. No network call occurs on an entry path.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from threading import Lock
import math
import time


@dataclass(frozen=True)
class _Anchor:
    utc: datetime
    monotonic: float
    rtt: float
    offset: float


class MarketClock:
    def __init__(self, *, wall=None, monotonic=None, max_age=180.0,
                 max_rtt=1.0, max_offset=5.0):
        self._wall = wall or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic or time.monotonic
        self._max_age, self._max_rtt, self._max_offset = max_age, max_rtt, max_offset
        self._anchor = None
        self._error = 'broker clock not calibrated'
        self._lock = Lock()
        self._last_calibration = None

    def synchronize(self, read_clock):
        samples, observations = [], []
        for attempt in range(3):
            observation = dict(attempt=attempt+1, accepted=False)
            try:
                wall0, mono0 = self._wall(), self._monotonic()
                observation['request_wall_utc'] = wall0.astimezone(timezone.utc).isoformat()
                response = read_clock()
                mono1, wall1 = self._monotonic(), self._wall()
                server = response.timestamp
                rtt = mono1-mono0
                observation.update(round_trip_seconds=rtt,receipt_wall_utc=wall1.astimezone(timezone.utc).isoformat(),
                                   wall_elapsed_seconds=(wall1-wall0).total_seconds())
                if server.tzinfo is None or server.utcoffset() is None:
                    raise ValueError('broker clock is not timezone aware')
                if not math.isfinite(rtt) or not 0 <= rtt <= self._max_rtt:
                    raise ValueError('broker clock request too slow')
                if abs((wall1-wall0).total_seconds()-rtt) > .05:
                    raise ValueError('local wall clock changed during calibration')
                at_receipt = server.astimezone(timezone.utc)+timedelta(seconds=rtt/2)
                offset = (at_receipt-wall1).total_seconds()
                observation.update(server_utc=server.astimezone(timezone.utc).isoformat(),offset_seconds=offset)
                if abs(offset) > self._max_offset:
                    raise ValueError('broker clock offset exceeds bound')
                samples.append(_Anchor(at_receipt, mono1, rtt, offset))
                observation['accepted'] = True
            except Exception as exc:
                with self._lock:
                    self._error = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                    observation.update(reason=self._error,error_type=type(exc).__name__)
            observations.append(observation)
        diagnostic = dict(samples=observations,accepted_samples=len(samples),success=False,
                          limits=dict(max_age_seconds=self._max_age,max_rtt_seconds=self._max_rtt,
                                      max_offset_seconds=self._max_offset))
        if len(samples) < 2:
            with self._lock:self._last_calibration = diagnostic
            return False  # Retain a prior good anchor only until its normal expiry.
        best = min(samples, key=lambda a:a.rtt)
        if abs(best.offset-median(a.offset for a in samples)) > .25:
            with self._lock:
                self._error = 'broker clock samples disagree'
                diagnostic['reason'] = self._error
                self._last_calibration = diagnostic
            return False
        with self._lock:
            self._anchor, self._error = best, None
            diagnostic['success'] = True
            self._last_calibration = diagnostic
        return True

    def now(self):
        with self._lock:anchor = self._anchor
        if anchor is None:
            return None
        elapsed = self._monotonic()-anchor.monotonic
        if not 0 <= elapsed <= self._max_age:
            return None
        return anchor.utc+timedelta(seconds=elapsed)

    def status(self):
        with self._lock:
            anchor, error = self._anchor, self._error
            calibration = self._last_calibration
            diagnostic = (dict(calibration,samples=[dict(row) for row in calibration['samples']],
                               limits=dict(calibration['limits'])) if calibration is not None else None)
        elapsed = self._monotonic()-anchor.monotonic if anchor else None
        valid = elapsed is not None and 0 <= elapsed <= self._max_age
        state = 'valid' if valid else ('missing' if anchor is None else 'monotonic_rollback' if elapsed<0 else 'expired')
        return dict(synchronized=valid, error=error,anchor_state=state,last_calibration=diagnostic,
            offset_seconds=round(anchor.offset,6) if anchor else None,
            round_trip_seconds=round(anchor.rtt,6) if anchor else None,
            sample_age_seconds=round(elapsed,3) if anchor else None,
            reference='broker UTC + monotonic elapsed; market timestamps unchanged')
