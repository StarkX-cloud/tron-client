#!/bin/bash

echo "=== TRON Render Deploy ==="
echo "This script deploys TRON to Render (one-click via GitHub)."
echo

# Render uses GitHub integration, so this is mostly instructional
echo
echo "Render deployment uses a one-click button:"
echo
echo "1. Commit this repo to GitHub"
echo "2. Visit: https://render.com/new"
echo "3. Select 'Web Service'"
echo "4. Connect your GitHub account"
echo "5. Select this repository"
echo "6. Use these settings:"
echo
echo "   Name: tron-core"
echo "   Environment: Docker"
echo "   Plan: Free (or Starter if you want always-on)"
echo
echo "7. Under 'Environment Variables', add:"
echo "   TRON_HOST=0.0.0.0"
echo "   TRON_PORT=10000"
echo "   TRON_RELOAD=false"
echo
echo "8. Click 'Create Web Service'"
echo
echo "Render will auto-deploy from your GitHub repo."
echo
echo "Once deployed, your TRON URL will be: https://<service-name>.onrender.com"
echo
