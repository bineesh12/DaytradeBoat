"""Original signal context for admission and durable paper/replay order lineage."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from daytrading.opportunities import hit_identity
from daytrading.execution.entry_outcome import RecoverableEntryRefusal, recovery_code

EXECUTION_ORIGIN=ContextVar('execution_opportunity',default=None)
EXECUTION_SIGNAL=ContextVar('execution_signal',default=None)
EXECUTION_REQUEST_ID=ContextVar('execution_request_id',default=None)
ENTRY_ATTEMPT=ContextVar('entry_attempt',default=None)

@contextmanager
def entry_attempt(signal):
    """Attempt-local outcome; unknown or submitted paths never grant a retry."""
    row=dict(symbol=signal.symbol,setup_id=lineage(signal)['setup_id'],
             state='unobserved',reason=None,client_order_id=None,recovery=None)
    token=ENTRY_ATTEMPT.set(row)
    try:yield row
    finally:ENTRY_ATTEMPT.reset(token)

def entry_refused(reason):
    row=ENTRY_ATTEMPT.get()
    if row is not None and row['state']=='unobserved':
        # Preserve the specific admission reason across generic order-status
        # reporting ("timed entry order rejected") later in the same attempt.
        recovery = recovery_code(reason)
        message = reason.code+': '+reason.reason if isinstance(reason, RecoverableEntryRefusal) else str(reason)
        row.update(state='not_submitted',reason=message,recovery=recovery)

def entry_sdk_started(symbol,client_order_id):
    row=ENTRY_ATTEMPT.get()
    if row is not None:
        # Any actual entry SDK call consumes the attempt, including HTTP
        # rejection, unknown response, later cancellation or partial fill.
        row.update(state='sdk_started',reason=None,client_order_id=client_order_id,recovery=None)
@contextmanager
def signal_origin(signal):
    hit=getattr(signal,'scan_result',None)
    token=EXECUTION_ORIGIN.set(hit_identity(hit)[0] if hit is not None else None)
    signal_token=EXECUTION_SIGNAL.set(signal)
    try:yield
    finally:
        EXECUTION_SIGNAL.reset(signal_token)
        EXECUTION_ORIGIN.reset(token)

def lineage(signal):
    hit=getattr(signal,'scan_result',None)
    return dict(strategy=hit.scanner_name if hit else 'unknown_recovery',
                entry_reason=getattr(signal,'reason','') or 'original entry reason unavailable',
                opportunity_id=hit_identity(hit)[0] if hit else None,
                final_check_id=hit.criteria.get('_final_entry_check_id') if hit else None,
                setup_id=(hit.criteria.get('event_setup_id') or hit.criteria.get('micro_setup_id')) if hit else None)

def signal_execution(function):
    @wraps(function)
    def wrapped(self,ctx,signal,*args,**kwargs):
        with signal_origin(signal):return function(self,ctx,signal,*args,**kwargs)
    return wrapped

def submit_signal(broker,order,bar,portfolio,signal):
    with signal_origin(signal):return broker.submit(order,bar,portfolio)
