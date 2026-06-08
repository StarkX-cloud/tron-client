# TRON Deployment Guide

This guide shows the simplest always-on deployment for your hosted TRON runtime.

## What this repo includes

- `Dockerfile` — containerizes the TRON runtime.
- `docker-compose.yml` — local development stack.
- `docker-compose.prod.yml` — production-ready stack with Nginx reverse proxy.
- `deploy/nginx/tron.conf` — proxy configuration for a domain.
- `deploy/tron-core.service` — example `systemd` service for a Linux host.

## Local development

From the repo root:

```bash
docker compose up --build
```

The TRON API will be available at:

- `http://localhost:9000`

If you want to run it in the background:

```bash
docker compose up --build -d
```

## Cloud Run deployment (recommended free/low-cost host)

Google Cloud Run is a strong option for operator-hosted TRON because it runs your container as an always-on service and supports a free tier for small workloads.

Prerequisites:

- GCP project with Cloud Run enabled
- Dockerfile in the repo
- GitHub secrets `GCP_PROJECT` and `GCP_SA_KEY`

Manual deploy:

```bash
gcloud auth login

gcloud config set project YOUR_PROJECT_ID

gcloud config set run/region us-central1

gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/tron-core:latest

gcloud run deploy tron-core \
  --image gcr.io/YOUR_PROJECT_ID/tron-core:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 9000 \
  --set-env-vars TRON_HOST=0.0.0.0,TRON_PORT=9000,TRON_RELOAD=false
```

Once deployed, point app clients at the Cloud Run URL:

```python
import tron
tron.config("https://<your-cloud-run-url>")
```

## Production deployment

For a real host or VPS, use `docker-compose.prod.yml`.

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

This starts:

- `tron-core` on `9000`
- `tron-dashboard` on `8501`
- `nginx` on `80` and `443`

### Domain and TLS

1. Copy the example env file and fill in your values:

```bash
cp .env.example .env
# then edit .env
```

2. Point your domain to the server IP.
3. Start the production stack:

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

4. Generate TLS certificates using the built-in certbot service:

```bash
docker compose -f docker-compose.prod.yml run --rm certbot
```

5. After certificates are issued, copy the SSL template into place and reload nginx:

```bash
cp deploy/nginx/ssl/tron-ssl.conf.template deploy/nginx/ssl/tron-ssl.conf

docker compose -f docker-compose.prod.yml restart nginx
```

6. Confirm HTTPS works at `https://<your-domain>`.

If you want to use a self-managed certificate or a cloud-managed TLS service, keep the same `docker-compose.prod.yml` and place the certs under `deploy/nginx/letsencrypt`.

## Systemd service example

Use `deploy/tron-core.service` to keep the compose stack alive on reboot.

```bash
sudo cp deploy/tron-core.service /etc/systemd/system/tron-core.service
sudo systemctl daemon-reload
sudo systemctl enable --now tron-core.service
```

## Runtime environment variables

The TRON runtime supports these environment variables:

- `TRON_HOST` — listen host, default `0.0.0.0`
- `TRON_PORT` — listen port, default `9000`
- `TRON_RELOAD` — live reload during development, default `false`
- `TRON_WORKERS` — number of Uvicorn worker processes, default `1`

## Connect clients

From Python:

```python
import tron
tron.config("https://tron.yourdomain.com")
```

If you need a private host name or IP, use `tron.config("http://<host>:9000")`.
