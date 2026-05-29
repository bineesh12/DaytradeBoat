#!/bin/bash
# Weekly XGBoost model retraining — runs inside the GCP container.
# Scheduled via Cloud Scheduler or cron at 6:00 UTC every Sunday.
#
# Usage (manual):
#   docker exec <container_id> /app/retrain-model.sh
#
# The bot will pick up the new model on the next restart (Monday premarket).

set -e

cd /app

echo "$(date) — Starting XGBoost model training..."
PYTHONPATH=/app/src python -m daytrading.ml.train

if [ -f /app/data/models/entry_model.json ]; then
    echo "$(date) — Model trained successfully: $(ls -la /app/data/models/entry_model.json)"
else
    echo "$(date) — ERROR: Model file not found after training"
    exit 1
fi
