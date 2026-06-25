# Warrior Ignition — premarket momentum (paper)

A learned model that catches the premarket **ignition off the base** (the $4→$19
launch), sized by conviction, with an adaptive runner exit. Entry is a model;
exit is a validated rule (a learned exit lost out-of-sample — see below).

## The system (locked in)

| Piece | What does it | Why |
|---|---|---|
| **Discovery** | live HOD scanner (unchanged) | only trade symbols the bot has surfaced |
| **Entry** | **learned model** `warrior_ignition_model.json` | base-breakout + volume-surge ignition, scored; OOS top-quartile +0.86R |
| **Sizing** | model conviction | bigger on high-conviction launches |
| **Exit** | **adaptive rule**: scale ⅓ at target, then trail `max(18%, 6×ATR)` below the high | beat a learned exit +200R vs +54R out-of-sample |

The entry is a model because the model won; the exit is a rule because the rule
won. Evidence decides each piece.

## Run it on paper

Set the env flag and run the bot in paper mode (Warrior squeeze enabled):

```
DAYTRADING_WARRIOR_IGNITION=1   # premarket ignition path takes over the Warrior lane path
```

When on, in **premarket** it:
- scores every base-breakout ignition with the model,
- logs every candidate to `data/ml/warrior_ignition_candidates.jsonl` (features +
  conviction), and logs the **outcome** when the trade closes,
- places paper orders (conviction-sized) through the shared, fully-guarded
  Warrior execution (real risk caps, entry-guard, position checks).

Flag off (default) = production untouched, the 20 lanes run as before.

### What to watch in the first sessions
- **How many ignitions actually fill.** They pass through the live entry-guard /
  ML / confirm gates, which may reject some — that's the true backtest-vs-live
  test. If paper fills far fewer than backtest, those gates are why.
- Paper fills will be **lower than backtest** (real slippage, real confirm timing).
- Expect **losing days** — it loses small on fakeouts and wins big on the
  monsters (NXTS-type). Judge it over many days, not one.

## The adaptive loop

```
paper trade (flag on) -> logs candidates + outcomes daily
        |
        v  weekly, offline:
scripts/retrain_ignition_model.py            # dry run: report OOS comparison
scripts/retrain_ignition_model.py --deploy   # overwrite model ONLY if it beats current OOS
```

The retrain **never ships a worse model** — it validates out-of-sample and keeps
the current weights unless the new ones improve. There is no live/online learning
(that would chase noise and blow up); adaptation happens in controlled, validated
batches. The live model loads instantly with no warm-up.

## Deferred (honestly)
- **Learned exit (RL).** A per-bar learned exit was built and **lost** to the rule
  out-of-sample (it exits early and cuts runners). A real learned exit needs
  reinforcement learning (optimize total R, not per-bar) and thousands of trades.
  The trade-trajectory data is now logged so this can be attempted later — but
  only adopted if it beats the rule.
