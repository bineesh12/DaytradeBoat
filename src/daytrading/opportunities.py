"""Stable setup IDs and an observed execution funnel (no inferred fills)."""

import hashlib
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from threading import Lock

STAGES = ("pattern", "qualified", "submitted", "filled", "closed")


def hit_identity(hit):
    if hit is None:
        return "", ""
    criteria = getattr(hit, "criteria", {}) or {}
    if criteria.get("_pending_recheck") and criteria.get("_pending_origin_id"):
        return criteria["_pending_origin_id"], criteria["_pending_origin_bar_ts"]
    bars = getattr(hit, "bars", None) or []
    ts = bars[-1].ts if bars else hit.ts
    # Micro triggers share their one-minute quality context. Identify them by
    # the actual completed trigger, rather than merging a minute's setups.
    if hit.scanner_name == "micro_pullback_reclaim":
        try:
            trigger = datetime.fromisoformat(criteria["micro_bar_start"])
            if trigger.tzinfo is not None and trigger <= hit.ts:
                ts = trigger
        except (KeyError, TypeError, ValueError):
            pass
    if hit.scanner_name == "prepared_reclaim_tick":
        evidence = criteria.get("_event_evidence")
        if evidence is not None:
            ts = evidence.reclaimed_at
    stamp = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    raw = "|".join((hit.symbol, hit.scanner_name, stamp))
    return hashlib.sha256(raw.encode()).hexdigest()[:24], stamp


def decision_stages(payload):
    reached = []
    stage = str(payload.get("stage", ""))
    passed = payload.get("passed") is True
    if passed and stage == "deferred_to_timer":
        reached.append("qualified")
    if passed and stage == "fill":
        reached.extend(("qualified", "submitted", "filled"))
    if passed and stage == "closed":
        reached.append("closed")
    status = str(payload.get("metadata", {}).get("status", "")).lower()
    if stage == "order" or stage.endswith("_order"):
        if status in (
            "new",
            "accepted",
            "filled",
            "partially_filled",
            "canceled",
            "expired",
        ):
            reached.append("submitted")
        if passed and status in ("filled", "partially_filled"):
            reached.extend(("qualified", "filled"))
    return reached


def decision_status(payload):
    stage = str(payload.get("stage", ""))
    reason = str(payload.get("reason", "")).lower()
    status = str(payload.get("metadata", {}).get("status", "")).lower()
    if "watch only" in reason or "collecting data" in reason:
        return "watch_only"
    if stage == "closed":
        return "closed"
    if stage == "fill" or status in ("filled", "partially_filled"):
        return "filled"
    if status in ("new", "accepted"):
        return "submitted"
    if stage == "pending_entry" and status not in ("confirming", "submitted"):
        return "waiting" if status in ("watching", "waiting", "armed") else "rejected"
    if stage == "cooldown" or (stage == "eligibility" and reason == "cooldown"):
        return "waiting"
    if payload.get("passed") is False:
        return "rejected"
    if stage == "deferred_to_timer":
        return "confirming"
    if stage == "execution_confirmation":
        return "released"
    if stage == "pattern":
        return "detected"
    return "checking"


class OpportunityTracker:
    def __init__(self):
        self.lock = Lock()
        self.counts = dict.fromkeys(STAGES, 0)
        self.rows = OrderedDict()
        self.seen = {s: set() for s in STAGES}
        self.active = {}

    def observe(self, payload, reached=()):
        oid = payload.get("opportunity_id")
        if not oid:
            return
        payload = deepcopy(payload)
        with self.lock:
            row = self.rows.setdefault(oid, {"opportunity_id": oid, "history": []})
            # Re-detection is not a new gate decision. Do not erase the actual
            # rejection/confirmation/order result with "pattern detected".
            redetected = payload.get("stage") == "pattern" and row.get("stage") not in (
                None,
                "pattern",
            )
            if redetected:
                row["last_seen"] = payload.get("ts", "")
            else:
                transition = {
                    k: payload.get(k)
                    for k in ("ts", "stage", "passed", "reason", "blocked_layer")
                }
                transition["status"] = decision_status(payload)
                history = row["history"]
                if not history or any(
                    transition.get(k) != history[-1].get(k)
                    for k in ("stage", "passed", "reason", "status")
                ):
                    history.append(transition)
                    del history[:-16]
                else:
                    history[-1]["ts"] = transition.get("ts")
                row.update(payload)
                row["history"] = history
                row["status"] = transition["status"]
                if row["status"] in ("rejected", "watch_only"):
                    row["last_rejection"] = dict(transition)
            done = set(row.get("reached", []))
            for stage in reached:
                if oid not in self.seen[stage]:
                    self.seen[stage].add(oid)
                    self.counts[stage] += 1
                done.add(stage)
            row["reached"] = [s for s in STAGES if s in done]
            self.rows.move_to_end(oid)
            while len(self.rows) > 10000:
                self.rows.popitem(last=False)
            if "filled" in reached:
                self.active[payload["symbol"]] = oid

    def exit(self, symbol, flat, ts=""):
        with self.lock:
            oid = self.active.get(symbol)
            row = dict(self.rows.get(oid, {}))
            if flat:
                self.active.pop(symbol, None)
        if oid and flat:
            row.update(
                stage="closed",
                status="closed",
                passed=True,
                reason="Position fully exited",
                ts=ts,
            )
            self.observe(row, ("closed",))
            return row

    def snapshot(self, query="", status="", limit=100):
        query = query.strip().lower()
        with self.lock:
            rows = list(self.rows.values())[::-1]
            if query:
                rows = [
                    r
                    for r in rows
                    if query
                    in " ".join(
                        str(r.get(k, ""))
                        for k in (
                            "symbol",
                            "scanner",
                            "pattern",
                            "reason",
                            "opportunity_id",
                        )
                    ).lower()
                ]
            if status:
                rows = [r for r in rows if r.get("status") == status]
            return {
                "counts": dict(self.counts),
                "decisions": deepcopy(rows[:limit]),
                "matching_setups": len(rows),
                "retained_setups": len(self.rows),
                "limit": limit,
            }
