#!/bin/bash
# Dump what the LIVE/paper bot actually saw for a given symbol+day, straight
# from the GCP VM — the real trade(s) from the journal DB, plus every entry
# candidate it scored (score + breakdown + rvol). Use this to calibrate the
# backtest's entry score against live (e.g. the CAST rvol gap).
#
# Usage:
#   ./scripts/dump_live_entry.sh CAST 2026-06-15
#
# Data lives on the host at /var/lib/daytrading-data (mounted into the
# container at /app/data); the container has python, so we docker-exec there.

set -euo pipefail

SYMBOL="${1:-CAST}"
DAY="${2:-2026-06-15}"
INSTANCE="daytrading-bot-c"
ZONE="us-east1-c"

echo "=== Pulling live entry data for $SYMBOL on $DAY from $INSTANCE ==="

# Remote python runs inside the running bot container (it has python + the
# mounted data). Kept read-only; prints small, copy-pasteable output.
read -r -d '' REMOTE_PY <<'PYEOF' || true
import sqlite3, json, sys
SYM = "__SYMBOL__"
DAY = "__DAY__"
db = "/app/data/journal/journal.db"
print("=== TRADES (journal.db) ===")
try:
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    cur = con.execute(
        "SELECT ts, side, trade_type, strategy, quantity, entry_price, exit_price, pnl, reason "
        "FROM trades WHERE symbol=? AND ts LIKE ? ORDER BY ts",
        (SYM, DAY + "%"),
    )
    rows = cur.fetchall()
    if not rows:
        print("  (no trades for %s on %s)" % (SYM, DAY))
    for r in rows:
        print("  ts=%s side=%s type=%s strat=%s qty=%s entry=%s exit=%s pnl=%s :: %s"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], str(r[8])[:60]))
except Exception as exc:
    print("  journal read error:", exc)

print("=== ENTRY CANDIDATES — PASSED (entry_candidates.jsonl) ===")
path = "/app/data/ml/entry_candidates.jsonl"
tok = '"symbol": "%s"' % SYM
n = 0
try:
    with open(path) as fh:
        for line in fh:
            if tok not in line or DAY not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if str(rec.get("symbol")) != SYM or not str(rec.get("ts", "")).startswith(DAY):
                continue
            if not rec.get("passed"):
                continue
            n += 1
            print("  %s score=%s rvol=%s px=%s :: %s"
                  % (rec.get("ts"), rec.get("score"), rec.get("rel_vol"),
                     rec.get("price"), rec.get("breakdown")))
    if n == 0:
        print("  (no PASSED candidates — re-run without the passed filter if needed)")
    print("  passed_count=%d" % n)
except Exception as exc:
    print("  candidates read error:", exc)
PYEOF

REMOTE_PY="${REMOTE_PY//__SYMBOL__/$SYMBOL}"
REMOTE_PY="${REMOTE_PY//__DAY__/$DAY}"

# Base64 the snippet so quoting survives the SSH hop, then run it in the container.
B64=$(printf '%s' "$REMOTE_PY" | base64 | tr -d '\n')
REMOTE_CMD="cid=\$(docker ps -q | head -1); docker exec \$cid python -c \"import base64;exec(base64.b64decode('$B64').decode())\""

gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="$REMOTE_CMD"

echo ""
echo "=== Done. Paste the output back so the backtest rvol can be calibrated to live. ==="
