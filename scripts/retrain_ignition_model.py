#!/usr/bin/env python
"""Retrain the Warrior ignition model (CLI) — offline half of the adaptive loop.

Trains on the cached bootstrap + live-logged paper candidates, validates
out-of-sample, and DEPLOYS ONLY IF it beats the current model.

  .venv/bin/python scripts/retrain_ignition_model.py            # dry run (report only)
  .venv/bin/python scripts/retrain_ignition_model.py --deploy   # ship if it improves
"""
import argparse
import sys

sys.path.insert(0, "src")
from daytrading.strategy.warrior_ignition_retrain import retrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true", help="overwrite the model if it improves OOS")
    args = ap.parse_args()
    r = retrain(deploy=args.deploy)
    if r.get("status") != "ok":
        print(f"retrain: {r.get('status')} (rows={r.get('rows', 0)}) — nothing deployed")
        return
    print(f"dataset: {r['rows']} candidates, runner rate {r['runner_rate']:.0%}")
    print(f"current model OOS top-quartile:   {r['current_oos']:.0%}")
    print(f"retrained model OOS top-quartile: {r['retrained_oos']:.0%}")
    if r["deployed"]:
        print("DEPLOYED new model (beat current out-of-sample).")
    elif r["improved"]:
        print("Retrained model improves — pass --deploy to ship it.")
    else:
        print("Retrained model did NOT improve — keeping the current model.")


if __name__ == "__main__":
    main()
