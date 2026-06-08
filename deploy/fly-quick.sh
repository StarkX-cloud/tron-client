#!/bin/bash
set -e

echo "=== TRON Fly.io Deploy ==="
echo "This script deploys TRON to Fly.io in minutes."
echo

# Check for flyctl
if ! command -v flyctl &> /dev/null; then
    echo "ERROR: flyctl CLI not found. Install it from https://fly.io/docs/getting-started/installing-flyctl/"
    exit 1
fi

# Get app name
read -p "Enter desired Fly.io app name (e.g., my-tron-server): " APP_NAME
if [ -z "$APP_NAME" ]; then
    echo "ERROR: App name required"
    exit 1
fi

read -p "Enter desired region (default: sjc - San Jose): " REGION
REGION=${REGION:-sjc}

echo
echo "Creating Fly.io app..."
flyctl launch --name "$APP_NAME" --no-deploy --region "$REGION" << EOF
TRON
Y
8080
EOF

echo "Configuring for TRON..."
cat > fly.toml << EOF
app = "$APP_NAME"
primary_region = "$REGION"

[env]
TRON_HOST = "0.0.0.0"
TRON_PORT = "8080"
TRON_RELOAD = "false"

[[services]]
protocol = "tcp"
internal_port = 8080
processes = ["app"]

  [[services.ports]]
  port = 80
  handlers = ["http"]

  [[services.ports]]
  port = 443
  handlers = ["tls", "http"]
EOF

echo "Deploying to Fly.io..."
flyctl deploy

echo
echo "=== Deployment complete! ==="
APP_URL=$(flyctl info --json | grep -o '"appUrl":"[^"]*' | cut -d'"' -f4)
echo "TRON server URL: $APP_URL"
echo
echo "Configure your developers to use:"
echo "  import tron"
echo "  tron.config(\"$APP_URL\")"
echo
