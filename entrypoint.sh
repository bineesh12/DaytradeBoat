#!/bin/bash
# Entrypoint: train model only if sufficient live data exists, then start the bot.
set -e

MODEL_PATH="/app/data/models/entry_model.json"
DATA_FILE="/app/data/ml/entry_candidates.jsonl"
MIN_SAMPLES=200

# Only train if we have enough real live data (at least 200 labeled candidates)
if [ ! -f "$MODEL_PATH" ] && [ -f "$DATA_FILE" ]; then
    LABELED=$(grep -c '"outcome_pnl"' "$DATA_FILE" 2>/dev/null || echo "0")
    if [ "$LABELED" -ge "$MIN_SAMPLES" ]; then
        echo "$(date) — Found $LABELED labeled samples, training XGBoost model..."
        PYTHONPATH=/app/src python -m daytrading.ml.train || echo "$(date) — Training failed (non-fatal), continuing without ML model"
    else
        echo "$(date) — Only $LABELED labeled samples (need $MIN_SAMPLES) — skipping ML training, collecting data..."
    fi
fi

# Start the bot
exec python -m daytrading.runner
