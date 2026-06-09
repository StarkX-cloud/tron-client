# TRON Deployment Guide

Self-host TRON's distributed computing infrastructure on your own servers, cloud platforms, or local development environment.

## Quick Start (Local Development)

Run everything locally in 2 terminals:

### Terminal 1: Start Queue Server
```bash
python queue_server.py
```
Output: `Uvicorn running on http://0.0.0.0:9000`

### Terminal 2: Start Worker(s)
```bash
python worker.py
```
Output: `Worker registered. Polling for jobs...`

### Terminal 3: Run Your Code
```python
import tron

@tron.remote
def expensive_compute(x, y):
    return x + y

# Executes on remote worker
future = expensive_compute(5, 7)
result = future.get()  # Blocks until result available
print(result)  # 12
```

## Architecture

```
┌─────────────────────┐
│  Client (SDK)       │
│  @tron.remote       │
└──────────┬──────────┘
           │ HTTP POST /submit
           ▼
┌─────────────────────────────────┐
│  Queue Server (queue_server.py) │
│  FastAPI on port 9000           │
│  - Job queue management         │
│  - Worker coordination          │
│  - Result caching               │
└──────────┬─────────────────────┘
           │ HTTP GET /next_job
           │ HTTP POST /complete
           ▼
┌──────────────────────────────────┐
│  Worker Pool (worker.py)         │
│  - Long-polling for jobs        │
│  - Function deserialization     │
│  - Job execution                │
│  - Result submission            │
└──────────────────────────────────┘
```

## Deployment Options

### Option 1: Local Development (Recommended for Testing)

```bash
# Terminal 1
python queue_server.py

# Terminal 2
python worker.py

# Terminal 3
python your_script.py
```

**Pros:**
- Instant setup, no external dependencies
- Perfect for development and debugging
- All logs in console

**Cons:**
- Single machine only
- No persistence across restarts

---

### Option 2: Docker Compose (Multi-Worker)

```bash
docker-compose up -d
```

**What it runs:**
- 1 Queue Server (port 9000)
- 3 Worker instances
- All networked together

**Scale workers:**
```bash
docker-compose up -d --scale worker=5
```

**Check status:**
```bash
docker-compose logs -f
docker-compose ps
```

**Stop:**
```bash
docker-compose down
```

---

### Option 3: Render.com (Free Tier + Web Service)

1. **Create Render.com account** → Connect GitHub

2. **Fork this repo** to your GitHub account

3. **Create Web Service:**
   - New → Web Service
   - Connect to your TRON fork
   - Build command: `pip install -r requirements.txt`
   - Start command: `python queue_server.py`
   - Environment: Python 3.11
   - Instance type: Free tier (~$7/mo with autoscale)
   - **Add Env Var**: `PYTHONUNBUFFERED=true`

4. **Create Background Worker (repeat 2-3 times):**
   - New → Background Job
   - Same repo + fork settings
   - Start command: `python worker.py`
   - Select appropriate instance size

5. **Connect Your Client:**
   ```python
   import tron
   
   tron.config.set_config_url("https://your-service.onrender.com:9000")
   
   @tron.remote
   def my_job(x):
       return x * 2
   
   result = my_job(21).get()  # Executes on Render
   ```

**Pricing:**
- Queue Server: Free tier ($0)
- Workers: $7/mo each (free tier starts at 0.5 CPU)

---

### Option 4: Fly.io (Production Ready)

1. **Install flyctl:** `curl -L https://fly.io/install.sh | sh`

2. **Login:** `flyctl auth login`

3. **Create app:**
   ```bash
   flyctl launch --image python:3.11
   ```

4. **Deploy Queue Server:**
   ```toml
   # fly.toml
   [env]
   PORT = "9000"
   
   [[services]]
   internal_port = 9000
   protocol = "tcp"
   ```

5. **Deploy Worker (separate app):**
   ```bash
   flyctl launch --name tron-worker-1
   ```

6. **Scale horizontally:**
   ```bash
   flyctl machines create --app tron-worker-1 --count 5
   ```

7. **Connect client:**
   ```python
   tron.config.set_config_url("https://your-app.fly.dev:9000")
   ```

**Pricing:**
- $5/mo shared CPU per service
- $15/mo dedicated CPU

---

### Option 5: AWS EC2 (Full Control)

1. **Launch EC2 instances:**
   - AMI: Ubuntu 22.04 LTS
   - Instance: t3.micro (free tier eligible)

2. **Install dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3.11 python3-pip
   pip install -r requirements.txt
   ```

3. **Start Queue Server on primary instance:**
   ```bash
   python queue_server.py
   ```

4. **Start Workers on secondary instances:**
   ```bash
   export TRON_SERVER=http://<primary-ip>:9000
   python worker.py
   ```

5. **Security Group setup:**
   - Inbound: Port 9000 from your IP (queue server)
   - Inbound: Port 22 for SSH (your IP only)

6. **Use systemd for persistence:**
   ```bash
   sudo systemctl enable tron-server
   sudo systemctl start tron-server
   ```

---

### Option 6: Kubernetes (Enterprise Scale)

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tron-queue-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tron-server
  template:
    metadata:
      labels:
        app: tron-server
    spec:
      containers:
      - name: server
        image: your-registry/tron:latest
        command: ["python", "queue_server.py"]
        ports:
        - containerPort: 9000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tron-workers
spec:
  replicas: 5  # Scale this up/down as needed
  selector:
    matchLabels:
      app: tron-worker
  template:
    metadata:
      labels:
        app: tron-worker
    spec:
      containers:
      - name: worker
        image: your-registry/tron:latest
        command: ["python", "worker.py"]
        env:
        - name: TRON_SERVER
          value: http://tron-queue-server:9000
```

Deploy:
```bash
kubectl apply -f deployment.yaml
kubectl scale deployment tron-workers --replicas=10
kubectl logs -f deployment/tron-queue-server
```

---

## Configuration

### Client-Side (Python SDK)

```python
import tron

# Option 1: Auto-discovery (localhost:9000)
# (automatic, no config needed)

# Option 2: Explicit server URL
tron.config.set_config_url("https://my-server.com:9000")

# Option 3: Environment variable
# export TRON_URL=https://my-server.com:9000
# tron automatically reads it

# Option 4: Mixed local+remote execution
@tron.remote(local_first=True)  # Try local first, fallback to remote
def sometimes_remote(x):
    return x * 2

# Option 5: Remote-only (no local fallback)
@tron.remote(remote_only=True)
def always_remote(x):
    return x * 2

# Option 6: Resource hints
@tron.remote(gpu=True, memory_gb=8)
def needs_gpu(data):
    import torch
    return torch.cuda.is_available()
```

### Server-Side (Environment Variables)

```bash
# Optional - customize port (default: 9000)
TRON_PORT=8000

# Optional - job timeout in seconds (default: 3600)
TRON_JOB_TIMEOUT=7200

# Optional - max workers (default: unlimited)
TRON_MAX_WORKERS=100

# Optional - enable metrics (default: true)
TRON_METRICS_ENABLED=true
```

---

## Monitoring & Debugging

### Check Queue Server Health
```bash
curl http://localhost:9000/health
```

### View Active Jobs
```bash
curl http://localhost:9000/jobs
```

### Check Worker Status
```bash
curl http://localhost:9000/workers
```

### Stream Job Output
```python
import tron

@tron.remote
def my_job(x):
    print("Processing...")
    return x * 2

future = my_job(5)
# Watch output in real-time
for line in future.stream():
    print(line)
```

### Debug Failed Jobs
```python
try:
    result = future.get()
except Exception as e:
    print(f"Job failed: {e}")
    # Re-submit with local_first=True to debug locally
    debug_result = my_job.local(5)  # Force local execution
```

---

## Best Practices

### 1. Resource Limits
```python
@tron.remote(
    timeout_seconds=300,
    memory_gb=4,
    cpu_cores=2
)
def bounded_task(data):
    return process(data)
```

### 2. Retry Logic
```python
@tron.remote(max_retries=3, retry_delay_seconds=5)
def resilient_task(x):
    # Automatically retried up to 3 times on failure
    return expensive_compute(x)
```

### 3. Batch Processing
```python
results = []
for item in items:
    future = process_item.remote(item)
    results.append(future)

# Wait for all to complete
final_results = [f.get() for f in results]
```

### 4. Pipeline Execution
```python
@tron.remote
def step1(x):
    return x * 2

@tron.remote
def step2(x):
    return x + 10

@tron.remote
def step3(x):
    return x ** 2

# Chain them
future1 = step1(5)
future2 = step2(future1.get())
future3 = step3(future2.get())
result = future3.get()
```

---

## Troubleshooting

### "Connection refused" on client
```
Problem: Client can't reach queue server
Solution:
1. Check server is running: python queue_server.py
2. Check port: curl http://localhost:9000/health
3. If remote, check firewall allows port 9000
4. Verify TRON_URL env var is correct
```

### Worker polling but no jobs execute
```
Problem: Jobs submitted but not executing
Solution:
1. Check worker is registered: curl http://localhost:9000/workers
2. Check worker logs for errors: python worker.py (view console)
3. Verify job serialization works: test with simple function
4. Check job timeout hasn't elapsed
```

### Out of memory on worker
```
Problem: Worker crashes on large jobs
Solution:
1. Add memory_gb hint: @tron.remote(memory_gb=16)
2. Chunk data before sending: send smaller batches
3. Use external storage for large results: save to S3, return path
```

### Slow job execution
```
Problem: Remote execution slower than local
Solution:
1. Check network latency: ping server
2. Profile job: use cProfile locally first
3. Consider batch processing: combine multiple small jobs
4. Use local_first=True for hybrid execution
```

---

## Security Considerations

### 1. Firewall Rules
```bash
# Only allow connections from trusted networks
# Queue Server port 9000 should not be public
sudo ufw allow from 10.0.0.0/8 to any port 9000
```

### 2. Authentication (Future)
```python
# Planned: API key authentication
tron.config.set_auth_token("your-api-key")
```

### 3. Encryption (Future)
```python
# Planned: TLS for job transport
tron.config.enable_tls(cert_path="/path/to/cert")
```

### 4. Job Sandboxing
- Jobs run in isolated worker processes
- No direct filesystem access (use explicit APIs)
- Timeouts prevent infinite loops

---

## Production Checklist

- [ ] Server runs with systemd/Docker for auto-restart
- [ ] Worker pool sized for expected load (start with 2x expected concurrent jobs)
- [ ] Monitoring alerts set up for failed jobs
- [ ] Logs persisted (Docker volumes or cloud logging)
- [ ] Firewall configured to restrict port 9000 access
- [ ] Regular backups of job results if persisting to database
- [ ] Performance tested with actual job workloads
- [ ] Rollback plan documented

---

## Support & Examples

See `/examples` directory for complete working demos:
- `ai_developer_simulation.py` - ML pipeline with TRON
- `magic_demo.py` - End-to-end walkthrough
- `developer_demo.py` - Developer workflow example

Questions? Check the README.md or review `queue_server.py` source.
