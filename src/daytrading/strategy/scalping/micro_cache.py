"""Rejection cache identity never crosses a micro route or original trigger."""
from datetime import datetime, timedelta
from hashlib import sha256
from math import isfinite
from daytrading.models import Timeframe

ROUTES=('micro_pullback_reclaim','prepared_reclaim_tick')
MARKER='|micro:v2|'


def context(hit):
    if hit.scanner_name not in ROUTES:return None
    c=hit.criteria
    try:
        if hit.scanner_name=='prepared_reclaim_tick':
            e=c['_event_evidence'];identity=c['event_setup_id']
            if identity!=e.setup_id or e.symbol!=hit.symbol:return None
            rows=e.windows;deadline=e.deadline
        else:
            identity=c['micro_setup_id'];rows=hit.confirmation_window.bars
            if rows[-1].bar.ts!=datetime.fromisoformat(c.get('micro_bar_start',rows[-1].bar.ts.isoformat())) or rows[-1].closed_at!=datetime.fromisoformat(c.get('micro_confirmed_at',rows[-1].closed_at.isoformat())):return None
            deadline=rows[-1].closed_at+timedelta(seconds=15)
        if not isinstance(identity,str) or not identity.strip() or len(rows)!=6 or deadline.tzinfo is None:return None
        if any(w.bar.symbol!=hit.symbol or w.bar.timeframe!=Timeframe.SEC_10 for w in rows):return None
        price=float(rows[-1].bar.close);volume=float(rows[-1].bar.volume)
        if not all(isfinite(x) and x>0 for x in (price,volume)):return None
        return dict(identity=identity,deadline=deadline,price=price,volume=volume)
    except (AttributeError,KeyError,IndexError,TypeError,ValueError):return None


def key(hit, legacy):
    c=context(hit)
    return legacy+MARKER+hit.scanner_name+'|'+sha256(c['identity'].encode()).hexdigest() if c else legacy


def prune(rows, now, maximum=2048):
    # Only new derived cache entries are bounded/pruned. Historical legacy
    # records survive native handover byte-for-byte and are not reused for micro.
    scoped=[k for k in rows if MARKER in k]
    for k in scoped:
        row=rows[k]
        if (now-row.ts).total_seconds()>=row.ttl_seconds:rows.pop(k)
    scoped=sorted((k for k in rows if MARKER in k),key=lambda k:rows[k].ts)
    for k in scoped[:max(0,len(scoped)-maximum)]:rows.pop(k)
