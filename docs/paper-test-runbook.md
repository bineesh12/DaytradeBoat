# Paper-test runbook — runner-capture / chase tuning

Goal: validate in **paper** (no financial risk) the strategy work that the backtest
either confirmed or can't prove. Everything below is config-driven via `DAYTRADING_*`
env vars — no code changes to flip behavior. Paper deployment must have
`DAYTRADING_ALPACA_PAPER=true`.

## What ships ON by default (deploy this branch — nothing to set)

These are live-wired and either validated or pure correctness fixes:

1. **Tick-exit fix** (`runner.py`): tick trailing-stops, tape-pressure exit, and 10s
   bar aggregation now run for every watched tick (were wrongly gated behind
   `pending_symbols` for open positions). **Watch this first** — it's a correctness
   fix to live exit management.
2. **Runner-trail fix** (`exits/manager.py`): `runner_trail_pct`/`runner_confirmed`
   were dead code; confirmed runners now actually trail (flat 3% default).
3. **Chase cap 5% / $10 tier** (`DAYTRADING_ENTRY_CHASE_PCT_LOW=0.05`, `_PRICE_TIER=10`):
   validated net-positive on the 10s path (the CUPR HOD-reclaim unlock).
4. **10s execution timer**: always on in live/paper (not flag-gated).

## The experiment: does the SUNE-style re-entry pay in paper?

SUNE 6/08 ran $1.17→$9.45. The bot scalped the first pop (+$9.74) but the stale
anti-chase memory then blocked the profitable re-entry. In backtest, opening the
chase guard turned +$9.74 into +$81.84 (a later `hod_reclaim` re-entry). That needed
chase *fully* open (not safe), so paper is where we find a **moderate** setting.

**Change ONE knob at a time, watch several sessions, attribute via the scorecard.**

Baseline (current default):
```
DAYTRADING_MISSED_A_PLUS_CHASE_WINDOW_SEC=1800
DAYTRADING_ENTRY_CHASE_PCT_LOW=0.05
```

Experiment A — shorten the anti-chase memory (most direct test of the re-entry thesis):
```
DAYTRADING_MISSED_A_PLUS_CHASE_WINDOW_SEC=600
```

Experiment B — modest chase widening (only after A is understood):
```
DAYTRADING_ENTRY_CHASE_PCT_LOW=0.08
```

Do **not** go wide-open — that chases everything; it was only a mechanism demo.

## What to watch on the dashboard scorecard

- `funnel`: `signals` → `trades_taken` (did re-entries actually fire?), and
  `rejected_by_layer` / `top_reject_reasons_by_layer` — confirm fewer
  `entry_chase_guard` / "earlier blocked A+" rejects.
- `by_entry_mode` / per-strategy rows: did the *extra* entries (the re-entries) net
  positive, or are they `hod_reclaim`/`pullback_base` losers? This is the verdict.
- `win_rate`, `expectancy_per_trade`, `total_pnl` vs the baseline sessions.

Decision rule: keep the change only if the **incremental** entries it enabled are net
positive — not just if total entries went up. (Repeated lesson: more entries ≠ profit.)

## Do NOT enable in paper

- `RUNNER_TRAIL_ADAPTIVE=true` — behaves like flat 8%, gives back fade/chop names
  (EDHL, SUNE). Keep `false`.
- `ten_second_breakout_scout`, `level_reclaim_10s_scout`, `level_capped_entry`,
  `momentum_burst_live` — **backtest-only flags, not wired to the live runner**, and
  net-negative in backtest. They can't be flipped in paper without new code.

## Why paper, not more backtest

Paper exercises real fills, spreads, and live tick timing into the 10s timer (the
backtest uses historical trades). It's the right validator for the 10s-entry-timing
thesis, and the only way to test the chase change on an unbiased forward stream
instead of the hand-picked CUPR/EDHL/SUNE basket.
