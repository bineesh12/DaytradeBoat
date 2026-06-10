#!/bin/bash
# Deploy DayTrading bot to Google Cloud Compute Engine
#
# Prerequisites:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. Run: gcloud auth login
#   3. Run: gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   ./deploy-gcp.sh

set -e

# --- Configuration ---
PROJECT_ID=$(gcloud config get-value project)
ZONE="us-east1-c"          # Close to Alpaca's servers (east coast)
INSTANCE_NAME="daytrading-bot-c"
MACHINE_TYPE="e2-medium"   # 2 vCPU, 4GB RAM — more headroom for live scans/journal
IMAGE_NAME="daytrading-bot"
IMAGE_URI="gcr.io/$PROJECT_ID/$IMAGE_NAME"

echo "=== DayTrading Bot — Google Cloud Deploy ==="
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Machine: $MACHINE_TYPE"
echo ""

# --- Step 1: Build and push Docker image ---
echo "[1/3] Building Docker image..."
gcloud builds submit --tag "$IMAGE_URI" .
IMAGE_DIGEST=$(gcloud container images describe "$IMAGE_URI:latest" \
    --format='get(image_summary.digest)' 2>/dev/null || true)
if [ -n "$IMAGE_DIGEST" ]; then
    DEPLOY_IMAGE="$IMAGE_URI@$IMAGE_DIGEST"
    echo "Deploy image digest: $IMAGE_DIGEST"
else
    DEPLOY_IMAGE="$IMAGE_URI:latest"
    echo "Deploy image digest lookup failed; falling back to latest tag"
fi

# --- Step 2: Create VM instance (if not exists) ---
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
    echo "[2/3] Instance exists — updating container..."
    CURRENT_MACHINE_TYPE=$(gcloud compute instances describe $INSTANCE_NAME \
        --zone=$ZONE \
        --format='get(machineType)' | awk -F/ '{print $NF}')
    if [ "$CURRENT_MACHINE_TYPE" != "$MACHINE_TYPE" ]; then
        echo "Resizing instance: $CURRENT_MACHINE_TYPE -> $MACHINE_TYPE"
        gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --quiet
        gcloud compute instances set-machine-type $INSTANCE_NAME \
            --zone=$ZONE \
            --machine-type=$MACHINE_TYPE
        gcloud compute instances start $INSTANCE_NAME --zone=$ZONE --quiet
    fi
    gcloud compute instances update-container $INSTANCE_NAME \
        --zone=$ZONE \
        --container-image="$DEPLOY_IMAGE" \
        --container-env-file=.env \
        --container-mount-host-path=host-path=/var/lib/daytrading-data,mount-path=/app/data
else
    echo "[2/3] Creating new VM instance..."
    gcloud compute instances create-with-container $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --container-image="$DEPLOY_IMAGE" \
        --container-env-file=.env \
        --container-restart-policy=always \
        --container-mount-host-path=host-path=/var/lib/daytrading-data,mount-path=/app/data \
        --tags=daytrading \
        --boot-disk-size=20GB \
        --maintenance-policy=MIGRATE
fi

# --- Step 3: Open firewall for dashboard ---
if ! gcloud compute firewall-rules describe allow-daytrading-dashboard &>/dev/null; then
    echo "[3/3] Creating firewall rule for dashboard (port 8080)..."
    gcloud compute firewall-rules create allow-daytrading-dashboard \
        --allow=tcp:8080 \
        --target-tags=daytrading \
        --description="Allow access to DayTrading dashboard"
else
    echo "[3/3] Firewall rule already exists"
fi

# --- Done ---
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "=== DEPLOYED ==="
echo "Dashboard: http://$EXTERNAL_IP:8080"
echo ""
echo "=== Auto-start schedule ==="
# Ensure VM starts before premarket (3:55 AM ET = 7:55 UTC) on weekdays
SCHEDULER_JOB="start-daytrading-bot"
if gcloud scheduler jobs describe $SCHEDULER_JOB --location=us-east1 &>/dev/null 2>&1; then
    echo "Scheduler job '$SCHEDULER_JOB' already exists"
else
    echo "Creating Cloud Scheduler job to auto-start VM at 7:55 UTC weekdays..."
    gcloud scheduler jobs create http $SCHEDULER_JOB \
        --location=us-east1 \
        --schedule="55 7 * * 1-5" \
        --uri="https://compute.googleapis.com/compute/v1/projects/$PROJECT_ID/zones/$ZONE/instances/$INSTANCE_NAME/start" \
        --http-method=POST \
        --oauth-service-account-email="$(gcloud iam service-accounts list --format='value(email)' --filter='displayName:Compute Engine default')" \
        --oauth-token-scope="https://www.googleapis.com/auth/compute" \
        --description="Start daytrading bot before premarket" \
        2>/dev/null || echo "  (Cloud Scheduler setup skipped — configure manually if needed)"
fi

# --- Step 4: Weekly model retrain (Sundays 6:00 UTC) ---
echo ""
echo "=== Weekly ML retrain ==="
RETRAIN_JOB="retrain-daytrading-model"
if gcloud scheduler jobs describe $RETRAIN_JOB --location=us-east1 &>/dev/null 2>&1; then
    echo "Retrain job '$RETRAIN_JOB' already exists"
else
    echo "Creating Cloud Scheduler job to retrain ML model every Sunday 6:00 UTC..."
    gcloud scheduler jobs create http $RETRAIN_JOB \
        --location=us-east1 \
        --schedule="0 6 * * 0" \
        --uri="https://compute.googleapis.com/compute/v1/projects/$PROJECT_ID/zones/$ZONE/instances/$INSTANCE_NAME/start" \
        --http-method=POST \
        --oauth-service-account-email="$(gcloud iam service-accounts list --format='value(email)' --filter='displayName:Compute Engine default')" \
        --oauth-token-scope="https://www.googleapis.com/auth/compute" \
        --description="Start VM for weekly model retrain" \
        2>/dev/null || echo "  (Retrain scheduler skipped — configure manually if needed)"
fi
echo ""
echo "Useful commands:"
echo "  Logs:    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- 'docker logs \$(docker ps -q) -f'"
echo "  Stop:    gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo "  Start:   gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
echo "  SSH:     gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "  Delete:  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
