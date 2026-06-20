# Tick-Based Early Entry Trigger — Implementation Plan

> **STATUS: IMPLEMENTED (behind flag, default off).** All 5 changes + 6 tests landed;
> full suite green (672 passed). Enable with `DAYTRADING_TICK_ENTRY_ENABLED=true`,
> then validate against the funnel before committing. The sections below are the
> as-built reference.

## Goal
Let the execution timer release an entry off **ticks** (not just a closed 10s bar)
when a fast mover is holding the base, so we catch it before it extends —
**without weakening the anti-chase** (the execute-path chase guard still runs on
every release).

### Why
On fast momentum names, price runs ~3–4% in the first ~15 seconds — it blows past
the early-strength near-base window *before* a 10s bar can confirm. So the bot can
neither enter early nor (correctly) chase, and ends with no fill. Observed live on
PPCB (anchor $5.75 → price ran to $5.81/$5.85 in seconds; both releases correctly
chase-rejected at +3.9% / +4.7%, then cancelled with no entry).

---

## Design principle
`on_tick` only decides *"is it time to enter near the base?"* It does **not**
re-check spread/chase — those already run in `_timed_entry_chase_reject` /
`_shared_entry_quality_reject` at the execute path. Keep `on_tick` lean and let the
existing guards be the single source of truth. A tick-triggered release that has
already run too far will still be rejected downstream.

---

## Change 1 — `PendingEntry` state (`strategy/execution_timer.py`, ~line 28)
Add a consecutive-confirmation counter:
```python
@dataclass
class PendingEntry:
    ...
    tick_confirm: int = 0   # consecutive qualifying ticks (for tick trigger)
```

## Change 2 — `ExecutionTimer.on_tick()` (new method, `strategy/execution_timer.py`)
```python
def on_tick(self, tick) -> Optional[TradeSignal]:
    """Release a pending entry off a live tick when price holds the base.

    Mirrors _allows_early_strength_release but on tick granularity, so fast
    movers are caught near the anchor before a 10s bar can confirm. The
    execute-path chase/spread guards still run on release.
    """
    if not self._tick_entry_enabled:
        return None
    sym = tick.symbol
    pending = self._pending.get(sym)
    if pending is None or not pending.require_micro_signal:
        return None

    price = float(getattr(tick, "price", 0) or 0.0)
    if price <= 0:
        return None

    crit = pending.signal.scan_result.criteria if pending.signal.scan_result else {}
    anchor = (self._criteria_float(crit, "setup_anchor")
              or self._criteria_float(crit, "queued_entry_price")
              or float(pending.signal.entry_price or 0.0))
    if anchor <= 0:
        return None
    vwap = self._criteria_float(crit, "vwap")

    qualifies = (
        price <= anchor * (1.0 + self._tick_entry_max_above_anchor)   # still near base
        and price >= anchor * 0.985                                   # not broken down
        and (vwap is None or vwap <= 0 or price >= vwap * 1.003)       # holding VWAP
    )
    pending.tick_confirm = pending.tick_confirm + 1 if qualifies else 0
    if pending.tick_confirm >= self._tick_entry_confirm_count:
        return self._release(sym)
    return None
```
- Reuse the existing `_criteria_float` and `_release` helpers.
- `tick_confirm` resets on any non-qualifying tick → requires **consecutive** ticks,
  which kills single-print wicks.

## Change 3 — Timer config (`strategy/execution_timer.py` `__init__`, ~line 55)
```python
def __init__(self, max_wait_bars=1, enabled=True, *,
             tick_entry_enabled=False,
             tick_entry_confirm_count=2,
             tick_entry_max_above_anchor=0.02):
    ...
    self._tick_entry_enabled = tick_entry_enabled
    self._tick_entry_confirm_count = int(tick_entry_confirm_count)
    self._tick_entry_max_above_anchor = float(tick_entry_max_above_anchor)
```

## Change 4 — Runner wiring (`runner.py`, the `TradeEvent` branch ~line 3348)
After the existing tick buffering:
```python
elif isinstance(evt, TradeEvent):
    tick = evt.tick
    # ... existing _hod_tick_tracker / halt / _tick_buffer append ...
    # Tick-based early entry: only for symbols with a pending timed entry
    if tick.symbol in self._exec_timer.pending_symbols:
        ready_sig = self._exec_timer.on_tick(tick)
        if ready_sig is not None:
            self._execute_timed_signal(ready_sig)   # runs chase/spread guards
```
- Gated on `pending_symbols` (a small set), so it's cheap at high tick rates.
- `_execute_timed_signal` already routes through `EntryExecutor` →
  `_timed_entry_chase_reject` + `_shared_entry_quality_reject`, so **spread + chase
  + anchor are all enforced on the tick release**. No duplicate guard logic.

## Change 5 — `StrategyConfig` (`config.py`)
```python
tick_entry_enabled: bool = False
tick_entry_confirm_count: int = 2
tick_entry_max_above_anchor: float = 0.02
# in from_env():
tick_entry_enabled=_env_bool("TICK_ENTRY_ENABLED", cls.tick_entry_enabled),
tick_entry_confirm_count=_env_int("TICK_ENTRY_CONFIRM_COUNT", cls.tick_entry_confirm_count),
tick_entry_max_above_anchor=_env_float("TICK_ENTRY_MAX_ABOVE_ANCHOR", cls.tick_entry_max_above_anchor),
```
Wire into the `ExecutionTimer(...)` construction in the runner (`from_env`, where the
timer is created ~line 156) so the flag flows through.

---

## Tests (`tests/test_execution_timer.py`)
1. **Releases after N consecutive qualifying ticks** — feed 2 ticks near anchor +
   above VWAP → returns the signal.
2. **No release when extended** — tick at `anchor * 1.05` → `None`.
3. **No release below VWAP** — tick below `vwap*1.003` → `None`.
4. **Confirmation resets** — qualifying, then non-qualifying, then qualifying → only
   1 confirm, no release.
5. **Flag off** — `tick_entry_enabled=False` → always `None`.
6. **Integration** — a tick release that's already run gets rejected by
   `_timed_entry_chase_reject` at the execute path (proves the guard still bites).

---

## Rollout & validation
1. Ship with `TICK_ENTRY_ENABLED=false` (default) — no behavior change, safe deploy.
2. Flip it on for a session; pull the `entry_decision` funnel and compare vs bar-only:
   - **fills count** up?
   - **entry-vs-anchor** still tight (<= ~2%)? (proves it enters *near base*, not chasing)
   - any fills where `tick_confirm` fired but the trade immediately reversed →
     tighten `tick_entry_confirm_count` to 3.
3. Keep `_allows_early_strength_release` (the 10s-bar path) as the fallback — the tick
   path only *adds* faster entries.

---

## The one thing to get right
**The confirmation count + the near-base ceiling.** Too loose (`confirm_count=1`,
`max_above_anchor` high) → enter wicks and chase. Too tight → never fires, back to
bar-only. Start at **2 ticks / +2% ceiling**, watch the funnel, adjust from data.
Everything else is mechanical; this is the judgment call.

---

## Context / related code
- `_allows_early_strength_release` (`execution_timer.py`) — the 10s-bar early-entry
  path this mirrors; same anchor/VWAP/near-base logic.
- `setup_anchor` is stamped into `criteria` by `runner._timed_entry_chase_anchor`
  (persistent per-symbol anchor). The tick path reads the same pinned value.
- `_timed_entry_chase_reject` (`runner.py`) — the execute-path anti-chase guard that
  still runs on every release (tick or bar).
- `pending_symbols` property on `ExecutionTimer` — gate the runner hook on this.
