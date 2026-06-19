"""A/B the warrior_news_continuation lane across the full cached universe.

Runs every cached symbol/date pair through the FULL warrior_squeeze_playbook
preset twice -- news OFF (baseline) and ON -- and reports per-symbol and total
realized P&L. Uses the correct preset (live_like_10s on) and the correct P&L
key ('pnl').
"""
import copy
import glob
import os
import re
from collections import defaultdict

from daytrading.backtest.service import run_backtest, DEFAULT_EXPERIMENTS
from daytrading.config import Settings

PRESET = DEFAULT_EXPERIMENTS["warrior_squeeze_playbook"]
CACHE = os.path.join("data", "backtest_cache")

pairs = []
for path in sorted(glob.glob(os.path.join(CACHE, "*_10s_trades.json"))):
    m = re.match(r"(.+)_(\d{4}-\d{2}-\d{2})_10s_trades\.json$", os.path.basename(path))
    if m:
        pairs.append((m.group(1), m.group(2)))


def pnl_of(result):
    total = 0.0
    warr = 0.0
    n = 0
    for rt in result.get("round_trips", []):
        p = float(rt.get("pnl") or 0.0)
        total += p
        n += 1
        if "warrior" in str(rt.get("pattern") or rt.get("mode") or "").lower():
            warr += p
    return total, warr, n


def run(news_on):
    flags = copy.deepcopy(PRESET)
    flags["warrior_news_continuation"] = news_on
    settings = Settings()
    by_sym = defaultdict(lambda: [0.0, 0.0, 0])
    for sym, day in pairs:
        res = run_backtest(sym, day, flags=flags, settings=settings)
        t, w, n = pnl_of(res)
        by_sym[sym][0] += t
        by_sym[sym][1] += w
        by_sym[sym][2] += n
    return by_sym


base = run(False)
news = run(True)

print("=" * 72)
print(f"{'SYMBOL':<8} {'base$':>10} {'baseWarr$':>10} {'news$':>10} {'newsWarr$':>10} {'delta$':>9}")
print("-" * 72)
syms = sorted(set(base) | set(news))
gb = gw_b = gn = gw_n = 0.0
for s in syms:
    b = base[s]
    nw = news[s]
    delta = nw[0] - b[0]
    gb += b[0]; gw_b += b[1]; gn += nw[0]; gw_n += nw[1]
    if abs(delta) < 0.005 and b[2] == 0:
        continue
    flag = "  <-- WORSE" if delta < -0.5 else ("  ++better" if delta > 0.5 else "")
    print(f"{s:<8} {b[0]:>10.2f} {b[1]:>10.2f} {nw[0]:>10.2f} {nw[1]:>10.2f} {delta:>9.2f}{flag}")
print("-" * 72)
print(f"{'TOTAL':<8} {gb:>10.2f} {gw_b:>10.2f} {gn:>10.2f} {gw_n:>10.2f} {gn-gb:>9.2f}")
print("=" * 72)
