"""Best-effort telemetry; broker/risk operations never use this boundary."""
import logging
import threading
import time

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_failures = 0
_last_warning = {}

def report_safely(callback, *args, **kwargs):
    global _failures
    try:
        return callback(*args, **kwargs)
    except Exception:
        name = getattr(callback, "__qualname__", type(callback).__name__)
        now = time.monotonic()
        with _lock:
            _failures += 1
            warn = now - _last_warning.get(name, -60) >= 60
            if warn:
                _last_warning[name] = now
        if warn:
            logger.warning("Reporting failed in %s; order management continues", name, exc_info=True)
        return None

def reporting_failures():
    with _lock:
        return _failures


def report_event(target, name, *args, **kwargs):
    """Optional reporting surfaces also work with headless runners."""
    if target is None:
        return None
    return report_safely(lambda: getattr(target, name, lambda *a, **k: None)(*args, **kwargs))
