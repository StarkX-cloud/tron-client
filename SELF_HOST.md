# TRON Self-Hosting Guide

This guide walks you through deploying your own TRON distributed computing server. Once deployed, your developers only install the SDK and point to your server.

## Overview

**You run TRON.** No vendor lock-in. Your data, your infrastructure, your rules.

Deployment options:
- **Cloud Run** (Google) — free tier, fastest to deploy
- **Fly.io** — global, generous free tier
- **Render** — one-click from GitHub
- **Docker Compose** — local or self-managed VPS
- **Kubernetes** — scale to thousands of tasks

## Prerequisites

- This TRON repo cloned or downloaded
- Docker installed (for all cloud options)
- CLI tool for your chosen platform (gcloud, flyctl, etc.)

## Option 1: Google Cloud Run (Recommended for free tier)

**Why:** Free tier includes 180,000 vCPU-seconds/month. Perfect for small teams.

### Quick deploy

```bash
bash deploy/cloud-run-quick.sh
```

### Manual deploy

```bash
# 1. Install gcloud CLI
# https://cloud.google.com/sdk

# 2. Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region us-central1

# 3. Build container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/tron-core:latest

# 4. Deploy to Cloud Run
gcloud run deploy tron-core \
  --image gcr.io/YOUR_PROJECT_ID/tron-core:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 9000 \
  --set-env-vars TRON_HOST=0.0.0.0,TRON_PORT=9000,TRON_RELOAD=false \
  --memory 512Mi \
  --cpu 1
```

Your TRON server URL:

```bash
gcloud run services describe tron-core --region us-central1 --format 'value(status.url)'
```

### Developer setup

```python
import tron
tron.config("https://<your-cloud-run-url>")

@tron.remote
def my_task(x):
    return x * 2

result = my_task(5).get()
```

### Scaling on Cloud Run

To handle more tasks, increase memory or CPU:

```bash
gcloud run deploy tron-core \
  --image gcr.io/YOUR_PROJECT_ID/tron-core:latest \
  --memory 2Gi \
  --cpu 4
```

---

## Option 2: Fly.io (Global, generous free tier)

**Why:** Deploys globally with built-in HTTP/HTTPS. Great for distributed teams.

### Quick deploy

```bash
bash deploy/fly-quick.sh
```

### Manual deploy

```bash
# 1. Install flyctl
# https://fly.io/docs/getting-started/installing-flyctl/

# 2. Create app
flyctl launch --name my-tron-server

# 3. Set environment variables
flyctl secrets set TRON_HOST=0.0.0.0 TRON_PORT=8080 TRON_RELOAD=false

# 4. Deploy
flyctl deploy
```

Your TRON server URL:

```bash
flyctl info | grep hostname
```

### Scaling on Fly.io

Scale to multiple regions:

```bash
# Add a region
flyctl regions add lax  # Los Angeles

# Scale instances
flyctl scale count=3
```

---

## Option 3: Render (One-click from GitHub)

**Why:** Simplest setup. GitHub integration, no CLI needed.

### Deploy

1. Push this repo to GitHub
2. Visit https://render.com
3. Click **New** → **Web Service**
4. Connect GitHub and select your repository
5. Use settings:
   - **Name:** `tron-core`
   - **Environment:** Docker
   - **Plan:** Free (or Starter for always-on)
6. Add environment variables:
   - `TRON_HOST=0.0.0.0`
   - `TRON_PORT=10000`
   - `TRON_RELOAD=false`
7. Click **Create Web Service**

Your TRON server URL: `https://<service-name>.onrender.com`

### Enable auto-deploy

Render auto-deploys on every GitHub push. No extra config needed.

---

## Option 4: Docker Compose (Local or VPS)

**Why:** Full control. Run on a laptop, VPS, or private server.

### Local development

```bash
docker compose up
```

TRON accessible at `http://localhost:9000`.

### Production on a VPS

```bash
# 1. Copy repo to your VPS
scp -r ./ user@your-vps:/opt/tron

# 2. SSH in
ssh user@your-vps

# 3. Deploy
cd /opt/tron
docker compose up -d

# 4. Check logs
docker compose logs -f tron-core
```

Your TRON server URL: `http://<your-vps-ip>:9000` or `https://<your-domain>:9000` (with reverse proxy).

### Add HTTPS (nginx reverse proxy on VPS)

```bash
# 1. Install certbot
sudo apt install certbot python3-certbot-nginx

# 2. Get certificate
sudo certbot certonly --standalone -d your-domain.com

# 3. Update nginx config (deploy/nginx/tron.conf)
# Point to localhost:9000

# 4. Restart nginx
sudo systemctl restart nginx
```

---

## Option 5: Kubernetes (Scale to thousands)

**Why:** For large teams or enterprises.

```bash
# 1. Create Deployment
kubectl create deployment tron-core --image=gcr.io/YOUR_PROJECT/tron-core:latest

# 2. Expose as Service
kubectl expose deployment tron-core --type=LoadBalancer --port=9000

# 3. Get external IP
kubectl get services
```

Your TRON server URL: `http://<external-ip>:9000`

Scale workers:

```bash
kubectl scale deployment tron-core --replicas=5
```

---

## Customization

### Environment Variables

All deployments support these settings:

- `TRON_HOST` — Listen address (default: `0.0.0.0`)
- `TRON_PORT` — Listen port (default: `9000`)
- `TRON_RELOAD` — Auto-reload on code change (default: `false`)

Example:

```bash
export TRON_HOST=0.0.0.0
export TRON_PORT=9000
export TRON_RELOAD=false
python queue_server.py
```

### Storage & Database

By default, TRON uses in-memory storage (`job_store` dict). For production:

1. Add persistent storage (PostgreSQL, Redis, etc.)
2. Update `queue_server.py` to use it
3. Redeploy

Contact the maintainers for production-grade storage support.

### Authentication & Security

By default, TRON allows unauthenticated access. For private servers:

1. Add API key middleware to `queue_server.py`
2. Configure in developer code:

```python
import tron
tron.config("https://your-tron-server", api_key="YOUR_SECRET_KEY")
```

---

## Troubleshooting

### Server not starting

Check logs:

```bash
# Cloud Run
gcloud run services describe tron-core

# Fly.io
flyctl logs

# Docker Compose
docker compose logs tron-core
```

### Developers can't connect

1. Verify server URL is correct
2. Check firewall rules (port 9000 or 8080 open)
3. Test with curl:
   ```bash
   curl https://your-tron-server/status/test
   ```

### Out of memory

Increase resources:
- **Cloud Run:** Use `--memory 2Gi --cpu 4`
- **Fly.io:** Update `fly.toml` memory settings
- **Docker:** Increase Docker daemon memory

---

## Feedback & Support

See [README.md](../README.md) for links and support channels.
