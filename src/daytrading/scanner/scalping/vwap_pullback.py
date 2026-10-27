"""VWAP pullback scanner.

Detects the first pullback to VWAP after an initial run-up:
1. Stock must have rallied (made a strong move up from open)
2. Price pulls back to touch or slightly pierce VWAP
3. Bounce candle: a green candle off VWAP = entry signal

Classic Warrior Trading entry — buying the first dip to support.
"""

from __future__ import annotations

from typing import Dict, List, Sequence
from datetime import datetime, timedelta
from daytrading.data.session_vwap import session_vwap_context, SOURCE
from daytrading.data.completed_bars import Rejection

from daytrading.indicators.core import vwap
from daytrading.models import Bar, ScanResult


class VWAPPullbackScanner:
    """Detects first pullback to VWAP after a strong move up."""

    def __init__(
        self,
        *,
        min_rally_pct: float = 3.0,
        vwap_touch_tolerance: float = 0.005,
        min_price: float = 1.0,
        max_price: float = 20.0,
    ) -> None:
        self.session_references = {}
        self._min_rally_pct = min_rally_pct
        self._vwap_touch_tolerance = vwap_touch_tolerance
        self._min_price = min_price
        self._max_price = max_price

    @property
    def name(self) -> str:
        return "vwap_pullback"

    accepts_cycle_time = True
    accepts_session_context = True

    def session_vwap_status(self):
        return dict(reference=SOURCE, symbols=dict(self.session_references))

    def scan(self, universe: Dict[str, Sequence[Bar]], *, now=None, session_universe=None) -> List[ScanResult]:
        results=[]
        for symbol,bars in universe.items():
            context_bars=(session_universe if session_universe is not None else universe).get(symbol,())
            # Legacy standalone callers may infer a historical completed-bar boundary;
            # paper and replay always inject the actual observation time.
            stamps=[b.ts for b in bars if b.ts is not None and b.ts.tzinfo is not None and b.symbol==symbol]
            observed=now if now is not None else (max(stamps)+timedelta(minutes=1) if stamps else None)
            context=session_vwap_context(context_bars,symbol=symbol,now=observed)
            if isinstance(context,Rejection):
                self.session_references[symbol]=dict(reason=context.reason,status="unavailable",vwap_reference_source=SOURCE)
                self.session_references[symbol]=self.session_references.pop(symbol)
                while len(self.session_references)>128:self.session_references.pop(next(iter(self.session_references)))
                continue
            rows=context['bars'];meta=dict(context['metadata'],price=rows[-1].close,
                above_vwap=rows[-1].close>=context['metadata']['vwap_reference'],status="measured")
            self.session_references[symbol]=meta;self.session_references[symbol]=self.session_references.pop(symbol)
            while len(self.session_references)>128:self.session_references.pop(next(iter(self.session_references)))
            if len(rows)<10 or not self._min_price<=rows[-1].close<=self._max_price:continue
            hit=self._detect(symbol,rows[-120:],session_context=context)
            if hit is not None:results.append(hit)
        results.sort(key=lambda r:r.score,reverse=True)
        return results

    def _detect(self, symbol: str, bars: List[Bar], *, session_context=None) -> ScanResult | None:
        if len(bars)<10:return None
        if session_context is None:
            stamp=bars[-1].ts
            if stamp is None or stamp.tzinfo is None:return None
            session_context=session_vwap_context(bars,symbol=symbol,now=stamp+timedelta(minutes=1))
            if isinstance(session_context,Rejection):return None
            bars=session_context['bars'][-120:]
            if len(bars)<10:return None
        latest,prev=bars[-1],bars[-2]
        vwap_vals=session_context['values']
        current_vwap=vwap_vals[-1]

        # Latest bar must be green (bounce)
        if latest.close <= latest.open:
            return None

        # Latest bar must close above VWAP (bouncing off it)
        if latest.close < current_vwap:
            return None

        # Previous bar or the low of latest bar must have touched VWAP
        prev_vwap = vwap_vals[-2]
        tolerance = current_vwap * self._vwap_touch_tolerance
        touched = (
            prev.low <= prev_vwap + tolerance
            or latest.low <= current_vwap + tolerance
        )
        if not touched:
            return None

        # Must have rallied first — find the high of day before the pullback
        session_high = max(b.high for b in bars)
        session_open = bars[0].open
        if session_open <= 0:
            return None

        rally_pct = (session_high - session_open) / session_open * 100
        if rally_pct < self._min_rally_pct:
            return None

        # Price must have pulled back from the high (not still at the top)
        pullback_depth = (session_high - prev.low) / session_high * 100
        if pullback_depth < 0.5:
            return None

        # Stop below the VWAP touch low
        stop_price = min(prev.low, latest.low) - 0.02
        score = rally_pct * (1 + pullback_depth / 10)

        return ScanResult(
            symbol=symbol,
            scanner_name=self.name,
            ts=latest.ts,
            score=round(score, 3),
            criteria={
                "pattern": "vwap_pullback",
                "direction": "up",
                "session_high": round(session_high, 4),
                "rally_pct": round(rally_pct, 2),
                "vwap": round(current_vwap, 4),
                **session_context["metadata"],
                "vwap_price_source": "completed 1m trigger close",
                "vwap_price_timestamp": latest.ts.isoformat(),
                "pullback_low": round(min(prev.low, latest.low), 4),
                "stop_price": round(stop_price, 4),
                "close": latest.close,
                "volume": latest.volume,
            },
            bars=list(bars),
        )
