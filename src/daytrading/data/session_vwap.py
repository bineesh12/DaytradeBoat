"""Shared completed-minute session VWAP; no price/strategy/risk exemptions."""
from daytrading.data.bar_provenance import canonical_minute
from datetime import datetime,timedelta
from math import isfinite
from daytrading.models import Timeframe
from daytrading.indicators.core import vwap
from daytrading.data.completed_bars import NY,Rejection

SOURCE = "HLC3 VWAP of available completed session minutes from 04:00 America/New_York"

def session_vwap_context(bars, *, symbol, now):
    if now is None or now.tzinfo is None or now.utcoffset() is None:
        return Rejection("vwap_evidence", "timezone-aware VWAP observation time required")
    start=now.astimezone(NY).replace(hour=4,minute=0,second=0,microsecond=0)
    if now<start:return Rejection("vwap_evidence", "04:00 New York session has not started")
    minutes={}
    for b in bars or ():
        if (not canonical_minute(b, now) or b.symbol!=symbol or b.timeframe!=Timeframe.MIN_1 or b.ts is None
                or b.ts.tzinfo is None or b.ts.utcoffset() is None
                or not start<=b.ts or b.ts+timedelta(minutes=1)>now):continue
        # Last observed revision wins; do not count repeated minute updates twice.
        minutes[b.ts]=b
    rows=[minutes[t] for t in sorted(minutes)]
    for b in rows:
        if (not all(isfinite(x) for x in (b.open,b.high,b.low,b.close,b.volume)) or b.volume<0
                or not 0<b.low<=min(b.open,b.close)<=max(b.open,b.close)<=b.high):
            return Rejection("vwap_evidence", "invalid completed session minute")
    if len(rows)<3 or sum(b.volume for b in rows)<=0:
        return Rejection("vwap_evidence", "three completed same-session minutes with volume required")
    values=vwap(rows)
    if not isfinite(values[-1]) or values[-1]<=0:return Rejection("vwap_evidence", "valid session VWAP required")
    return dict(bars=rows,values=values,metadata=dict(
        vwap_reference=values[-1],vwap_reference_source=SOURCE,
        vwap_session_anchor=start.isoformat(),vwap_session_first_bar=rows[0].ts.isoformat(),
        vwap_session_last_bar=rows[-1].ts.isoformat(),vwap_session_bar_count=len(rows),
        vwap_observed_at=now.isoformat(),vwap_timeframe="completed 1m",vwap_formula="HLC3 * volume / volume"))
