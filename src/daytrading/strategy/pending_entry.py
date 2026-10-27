"""Bounded setup memory. Never creates signals or submits orders."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from daytrading.opportunities import hit_identity
from daytrading.models import SignalAction


def temporary_gate(reason):
    text = str(reason or "").lower()
    return any(term in text for term in (
        "spread too wide", "spread exception", "entry score too low",
        "tape shows selling pressure", "thin sub-$5 liquidity", "too illiquid", "fresh valid quote",
    ))


@dataclass
class PendingSetup:
    hit: object
    created: datetime
    expires: datetime
    entry: float
    support: float
    reason: str
    dirty: bool = False
    last_check: object = None
    invalid: str = ""
    needs_reclaim: bool = False
    saw_pullback: bool = False


class PendingEntryBook:
    def __init__(self, ttl_seconds=60, interval_seconds=1, max_setups=20):
        self.ttl_seconds = ttl_seconds
        self.interval_seconds = interval_seconds
        self.max_setups = max_setups
        self.rows = {}
        self.inflight = {}
        self.retired = OrderedDict()
        self.events = []

    def clear(self):
        self.rows.clear()
        self.inflight.clear()
        self.retired.clear()
        self.events.clear()

    def reject(self, hit, reason, now):
        if hit is None or (hit.scanner_name != "vwap_pullback" and
                           not hit.criteria.get("_pending_hod_retry")):
            return
        if not temporary_gate(reason):
            if hit.symbol in self.rows or hit.symbol in self.inflight:
                self.finish(hit.symbol, "cancelled", reason)
            return
        if not hit.criteria.get("_pending_structure_valid"):
            return
        oid, _ = hit_identity(hit)
        if oid in self.retired:
            return
        if hit.symbol in self.inflight:
            row = self.inflight.pop(hit.symbol)
            self.rows[hit.symbol] = row
            if now >= row.expires:
                self.finish(hit.symbol, "cancelled", "pending setup expired during confirmation")
                return
            row.reason, row.dirty = reason, False
            self.events.append((row.hit, "waiting", reason))
            return
        if hit.symbol in self.rows:
            self.rows[hit.symbol].reason = reason
            return  # Do not move the original anchor or extend the deadline.
        bars = hit.bars
        entry = float(bars[-1].close or 0) if bars else 0
        support = float(hit.criteria.get("pullback_low") or hit.criteria.get("stop_price") or 0)
        stop = float(hit.criteria.get("stop_price") or 0)
        if not (0 < stop <= support < entry) or len(self.rows)+len(self.inflight) >= self.max_setups:
            return
        if bars[-1].ts is None or not (0 <= (now-bars[-1].ts).total_seconds() <= 120):
            return
        self.rows[hit.symbol] = PendingSetup(deepcopy(hit), now, now+timedelta(seconds=self.ttl_seconds),
                                             entry, support, reason)
        self.events.append((hit, "waiting", reason))

    def timer_cancel(self, signal, reason, now):
        """One HOD retry, only after a fully approved signal's timing expiry.

        Hard failures never arm a watch. A retry that times out is terminal;
        neither repeated signals nor transient gates extend its first deadline.
        """
        hit = signal.scan_result
        timing_only = (str(reason).startswith("no favorable confirmation after ") or
                       str(reason).startswith("no favorable 10s confirmation before "))
        if hit is None or hit.scanner_name != "hod_reclaim" or not timing_only:
            self.reject(hit, reason, now)
            return
        if hit.criteria.get("_pending_hod_retry"):
            self.finish(hit.symbol, "cancelled", "HOD retry confirmation expired; no further retry")
            return
        oid, _ = hit_identity(hit)
        if oid in self.retired or hit.symbol in self.rows or hit.symbol in self.inflight:
            return
        entry = float(hit.criteria.get("setup_anchor") or hit.criteria.get("queued_entry_price") or signal.entry_price or 0)
        stop = float(signal.stop_loss or 0)
        if (signal.action is not SignalAction.ENTER_LONG or signal.quantity <= 0 or
                not (0 < stop < entry) or not hit.bars or hit.bars[-1].ts is None or
                not (0 <= (now-hit.bars[-1].ts).total_seconds() <= 120) or
                len(self.rows)+len(self.inflight) >= self.max_setups):
            return
        retained = deepcopy(hit)
        retained.criteria.update(_pending_hod_retry=True, _pending_structure_valid=True,
                                 setup_anchor=entry, queued_entry_price=entry,
                                 stop_price=stop)
        self.rows[hit.symbol] = PendingSetup(retained, now, now+timedelta(seconds=self.ttl_seconds),
                                             entry, stop, reason, needs_reclaim=True)
        self.events.append((retained, "waiting", "HOD timing expired; watching for pullback and recovery"))

    def mark(self, symbol, price=None, stamp=None, *, dirty=True, quote=False):
        row = self.rows.get(symbol) or self.inflight.get(symbol)
        if row is None:
            return
        # An old event must not cancel or wake a new setup.
        if stamp is not None and stamp < row.created:
            return
        row.dirty = row.dirty or dirty
        # HOD retry structure follows executable NBBO bids and closed 1m bars.
        # Raw trade prints still feed all existing tape/10s execution guards.
        if row.needs_reclaim and not quote:
            return
        if price and price <= row.support:
            row.invalid = "setup support broken"
        elif price and price > row.entry * 1.025:
            row.invalid = "price extended more than 2.5% from original setup"
        elif row.needs_reclaim and price and price <= row.entry * 0.995:
            row.saw_pullback = True

    def due(self, now):
        return any(row.invalid or now >= row.expires or
                   (row.dirty and (row.last_check is None or
                    (now-row.last_check).total_seconds() >= self.interval_seconds))
                   for row in self.rows.values())

    def finish(self, symbol, status, reason):
        row = self.rows.pop(symbol, None) or self.inflight.pop(symbol, None)
        if row is None:
            return
        if status == "confirming":
            self.inflight[symbol] = row
            self.events.append((row.hit, status, reason))
            return
        oid, _ = hit_identity(row.hit)
        self.retired[oid] = True
        while len(self.retired) > 2048:
            self.retired.popitem(last=False)
        self.events.append((row.hit, status, reason))

    def execution_reject(self, signal, quote, now, *, max_risk_pct, max_dollar_risk):
        """Keep a delayed HOD retry inside its original controls at the ask."""
        if not signal.scan_result or not signal.scan_result.criteria.get("_pending_hod_retry"):
            return None
        row = self.inflight.get(signal.symbol)
        if row is None:
            return "HOD retry no longer active"
        if row.invalid or now >= row.expires:
            return row.invalid or "HOD retry setup expired"
        if quote is None or not (0 < quote.bid < quote.ask) or not (0 <= (now-quote.ts).total_seconds() <= 5):
            return "waiting for fresh valid quote (max age 5s)"
        if quote.bid <= row.support:
            return "HOD retry setup support broken"
        if quote.ask > row.entry * 1.025:
            return "HOD retry price extended more than 2.5% from original setup"
        stop = float(signal.stop_loss or 0)
        if stop < row.support:
            return "HOD retry protective stop widened from original setup"
        if not (0 < stop < quote.ask):
            return "HOD retry invalid protective stop"
        risk = quote.ask-stop
        if max_risk_pct > 0 and risk/quote.ask > max_risk_pct+1e-12:
            return "HOD retry live stop risk {:.2%} exceeds {:.2%} cap".format(risk/quote.ask,max_risk_pct)
        if max_dollar_risk > 0 and risk*signal.quantity > max_dollar_risk+1e-9:
            return "HOD retry live dollar risk exceeds configured trade limit"
        return None

    def prepare(self, universe, quotes, ticks, portfolio, now, *, entries_enabled, enabled_patterns):
        ready = {}
        for symbol, row in list(self.rows.items()):
            bars = universe.get(symbol) or []
            pos = portfolio.positions.get(symbol)
            reason = row.invalid
            if not entries_enabled:
                reason = "new entries paused or trading window closed"
            elif enabled_patterns is not None and row.hit.scanner_name not in enabled_patterns:
                reason = "{} entry pattern disabled".format(row.hit.scanner_name)
            elif pos and not pos.is_flat:
                reason = "position already open"
            elif now >= row.expires:
                reason = "pending setup expired after {}s".format(self.ttl_seconds)
            elif not bars:
                reason = "symbol left active watchlist"
            elif bars[-1].ts is None or not (0 <= (now-bars[-1].ts).total_seconds() <= 120):
                reason = "setup candle data stale"
            elif bars[-1].close <= row.support:
                reason = "setup support broken"
            elif any(b.ts > row.hit.bars[-1].ts and b.low <= row.support for b in bars):
                reason = "setup support broken in a new candle"
            if reason:
                self.finish(symbol, "cancelled", reason)
                continue
            if not row.dirty or (row.last_check and
                    (now-row.last_check).total_seconds() < self.interval_seconds):
                continue
            row.last_check, row.dirty = now, False
            quote_seq = (quotes or {}).get(symbol) or []
            quote = quote_seq[-1] if quote_seq else None
            if quote is None or quote.ts is None or not (0 <= (now-quote.ts).total_seconds() <= 5) or not (0 < quote.bid < quote.ask):
                self.events.append((row.hit, "waiting", "waiting for fresh valid quote (max age 5s)"))
                continue
            if row.needs_reclaim:
                self.mark(symbol, quote.bid, quote.ts, dirty=False, quote=True)
                if row.invalid:
                    self.finish(symbol, "cancelled", row.invalid)
                    continue
                if not row.saw_pullback or quote.bid < row.entry * 0.995:
                    reason = ("HOD pullback observed; waiting for recovery near original entry" if row.saw_pullback
                              else "waiting for HOD pullback before retry")
                    self.events.append((row.hit, "waiting", reason))
                    continue
            tick_seq = (ticks or {}).get(symbol) or []
            tick = tick_seq[-1] if tick_seq else None
            price = (tick.price if not row.needs_reclaim and tick and tick.ts and
                     0 <= (now-tick.ts).total_seconds() <= 5 else quote.mid)
            if price <= row.support or quote.bid <= row.support:
                self.finish(symbol, "cancelled", "setup support broken")
                continue
            if max(price, quote.ask) > row.entry * 1.025:
                self.finish(symbol, "cancelled", "price extended more than 2.5% from original setup")
                continue
            criteria = deepcopy(row.hit.criteria)
            oid, stamp = hit_identity(row.hit)
            criteria.update(_pending_recheck=True, _pending_origin_id=oid, _pending_origin_bar_ts=stamp)
            ready[symbol] = replace(row.hit, criteria=criteria, bars=list(bars))
            self.events.append((row.hit, "rechecking", "fresh market data; rechecking all entry gates"))
        return ready

    def drain_events(self):
        events, self.events = self.events, []
        return events
