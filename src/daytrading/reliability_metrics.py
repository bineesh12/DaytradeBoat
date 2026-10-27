"""Bounded observational timings; excluded from causal trading state."""
from functools import wraps
from contextlib import contextmanager
from threading import Lock
from time import perf_counter_ns


class StageMetrics:
    def __init__(self):
        self.lock = Lock()
        self.rows = {}

    def add(self, stage, duration_ns=0):
        with self.lock:
            row = self.rows.get(stage)
            if row is None:
                row = dict(count=0, total_ns=0, max_ns=0, histogram={})
                self.rows[stage] = row
            row['count'] += 1
            row['total_ns'] += max(0, duration_ns)
            row['max_ns'] = max(row['max_ns'], duration_ns)
            bucket=max(0,int(duration_ns)).bit_length()
            row['histogram'][bucket]=row['histogram'].get(bucket,0)+1

    def status(self):
        with self.lock:
            result={}
            for key,row in self.rows.items():
                value=dict(row, histogram=dict(row['histogram']), mean_ns=row['total_ns']/row['count'])
                for percentile in (50,95,99):
                    seen=0
                    for bucket,count in sorted(row['histogram'].items()):
                        seen+=count
                        if seen*100 >= row['count']*percentile:
                            value['p'+str(percentile)+'_ns_upper_bound']=(1<<bucket)-1
                            break
                result[key]=value
            return result


def measured(stage):
    def decorate(method):
        @wraps(method)
        def call(self, *args, **kwargs):
            start = perf_counter_ns()
            try:
                return method(self, *args, **kwargs)
            finally:
                metrics = getattr(self, '_stage_metrics', None)
                if metrics is not None:
                    kind = type(args[0]).__name__ if args else ''
                    metrics.add(stage+':'+kind, perf_counter_ns()-start)
        return call
    return decorate


@contextmanager
def timed_lock(lock, metrics, stage):
    waiting=perf_counter_ns()
    lock.acquire()
    acquired=perf_counter_ns()
    try:
        yield
    finally:
        lock.release()
        metrics.add(stage+'_wait',acquired-waiting)
        metrics.add(stage+'_held',perf_counter_ns()-acquired)
