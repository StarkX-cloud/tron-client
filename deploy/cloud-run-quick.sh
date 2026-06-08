#!/bin/bash
set -e

echo "=== TRON Cloud Run Deploy ==="
echo "This script deploys TRON to Google Cloud Run in minutes."
echo

# Check for gcloud
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI not found. Install it from https://cloud.google.com/sdk"
    exit 1
fi

# Get project ID
read -p "Enter your GCP Project ID: " PROJECT_ID
read -p "Enter desired Cloud Run service name (default: tron-core): " SERVICE_NAME
SERVICE_NAME=${SERVICE_NAME:-tron-core}
read -p "Enter Cloud Run region (default: us-central1): " REGION
REGION=${REGION:-us-central1}

echo
echo "Configuring gcloud..."
gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"

echo "Building container image..."
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"
gcloud builds submit --tag "$IMAGE"

echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 9000 \
  --set-env-vars TRON_HOST=0.0.0.0,TRON_PORT=9000,TRON_RELOAD=false \
  --memory 512Mi \
  --cpu 1

echo
echo "=== Deployment complete! ==="
echo
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --platform managed --region "$REGION" --format 'value(status.url)')
echo "TRON server URL: $SERVICE_URL"
echo
echo "Configure your developers to use:"
echo "  import tron"
echo "  tron.config(\"$SERVICE_URL\")"
echo
