import json
import threading
import time
from collections import defaultdict, deque

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ====================================
# CONFIG
# ====================================

DEFAULT_QUEUE_URL = "http://127.0.0.1:9000"

st.set_page_config(
    page_title="TRON Developer Dashboard",
    layout="wide"
)

if "queue_url" not in st.session_state:
    st.session_state.queue_url = DEFAULT_QUEUE_URL

queue_url = st.sidebar.text_input(
    "TRON Queue Server URL",
    value=st.session_state.queue_url,
    help="Start the TRON runtime with python queue_server.py and point the dashboard here."
).strip() or DEFAULT_QUEUE_URL
st.session_state.queue_url = queue_url

if "event_buffers" not in st.session_state:
    st.session_state.event_buffers = defaultdict(lambda: deque(maxlen=100))

if "stream_threads" not in st.session_state:
    st.session_state.stream_threads = {}

st.markdown(
    """
    <div class="hero">
        <div class="hero__content">
            <div class="hero__eyebrow">TRON v3</div>
            <h1>Developer dashboard view</h1>
            <p>Monitor workers, jobs, queue state, live event streams, and runtime metadata in one polished developer console.</p>
            <div class="hero__buttons">
                <span class="hero__pill">Local-first</span>
                <span class="hero__pill">Live streaming</span>
                <span class="hero__pill">Job explorer</span>
                <span class="hero__pill">Developer-first</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

STYLE = """
    <style>
    .reportview-container, .main, .block-container {
        background: linear-gradient(135deg, #050813 0%, #131e3a 45%, #08121f 100%);
        color: #e5e9f0;
    }
    .hero {
        padding: 1.6rem;
        border-radius: 22px;
        background: radial-gradient(circle at top left, rgba(80, 135, 255, 0.18), transparent 30%),
                    linear-gradient(180deg, rgba(12, 49, 121, 0.85), rgba(9, 14, 35, 0.98));
        margin-bottom: 1.5rem;
    }
    .hero__eyebrow {
        color: #86b7ff;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        margin-bottom: 0.6rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        margin: 0;
        color: #f8fafc;
    }
    .hero p {
        font-size: 1rem;
        max-width: 760px;
        line-height: 1.65rem;
        color: #cbd5e1;
    }
    .hero__buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1rem;
    }
    .hero__pill {
        background: rgba(255, 255, 255, 0.08);
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        font-size: 0.9rem;
        color: #dbeafe;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .card {
        border-radius: 18px;
        padding: 1.2rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 50px rgba(0,0,0,0.12);
    }
    .metric-card {
        border-radius: 16px;
        padding: 1rem 1.2rem;
        background: linear-gradient(180deg, rgba(16, 86, 250, 0.18), rgba(8, 18, 62, 0.98));
        border: 1px solid rgba(255,255,255,0.1);
        color: #f8fafc;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1rem;
        color: #a5b4fc;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .metric-card .label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.35rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #5b8cff, #7d7dff);
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #8db7ff, #a59dff);
        color: white;
    }
    .streamline {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

st.sidebar.header("TRON Quick Start")
st.sidebar.markdown(
    """
    - `python queue_server.py`
    - `.\\.venv\\Scripts\\python.exe -m streamlit run dashboard_v3.py`
    - Submit jobs and inspect live events
    """
)
st.sidebar.header("Current runtime URL")
st.sidebar.text_input("TRON Queue Server URL", value=queue_url, key="queue_url", help="Point the dashboard to your running queue server.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    #### Developer-first focus
    - No YAML manifests
    - Same code locally/remotely
    - Instant job introspection
    """
)

st_autorefresh(interval=4000, key="tron_dashboard_refresh")

# ====================================
# API HELPERS
# ====================================

def safe_get(path):
    try:
        response = requests.get(f"{queue_url}{path}", timeout=4)
        if response.status_code != 200:
            return {"_error": f"HTTP {response.status_code} for {path}"}
        return response.json()
    except Exception as exc:
        return {"_error": str(exc)}


def format_elapsed(seconds):
    try:
        if seconds < 60:
            return f"{seconds:.1f}s"
        return f"{seconds/60:.1f}m"
    except Exception:
        return "-"


def start_stream_thread(job_id):
    if job_id in st.session_state.stream_threads:
        return

    def stream_job():
        try:
            with requests.get(
                f"{queue_url}/stream/{job_id}",
                stream=True,
                timeout=60,
            ) as stream_response:
                for raw_line in stream_response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    try:
                        payload = json.loads(line)
                    except Exception:
                        payload = {"raw": line}

                    st.session_state.event_buffers[job_id].appendleft(
                        {
                            "timestamp": time.strftime("%H:%M:%S"),
                            "payload": payload,
                        }
                    )
        except Exception as exc:
            st.session_state.event_buffers[job_id].appendleft(
                {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "payload": {"error": str(exc)},
                }
            )

    thread = threading.Thread(target=stream_job, daemon=True)
    st.session_state.stream_threads[job_id] = thread
    thread.start()


# ====================================
# LOAD DATA
# ====================================

metrics = safe_get("/metrics")
workers = safe_get("/workers")
running = safe_get("/running")
queue_data = safe_get("/queue")
history = safe_get("/history")

server_online = isinstance(metrics, dict) and "_error" not in metrics

with st.container():
    if server_online:
        st.success(f"TRON server online at {queue_url}")
    else:
        st.error(f"Cannot reach TRON server at {queue_url}")
        if isinstance(metrics, dict) and metrics.get("_error"):
            st.write(metrics.get("_error"))

    summary_cols = st.columns([1, 1, 1, 1, 1])
    summary_cols[0].metric("Workers", metrics.get("workers", 0) if server_online else 0)
    summary_cols[1].metric("Queue", metrics.get("queue_size", 0) if server_online else 0)
    summary_cols[2].metric("Running", metrics.get("running_jobs", 0) if server_online else 0)
    summary_cols[3].metric("Completed", metrics.get("completed_jobs", 0) if server_online else 0)

    load = (metrics.get("running_jobs", 0) + metrics.get("queue_size", 0)) if server_online else 0
    health = "GREEN"
    if load > 10:
        health = "YELLOW"
    if load > 25:
        health = "RED"
    summary_cols[4].metric("Health", health)

    st.markdown("---")
    with st.expander("Runtime summary", expanded=True):
        if server_online:
            locality_status = (
                "Remote scaling ready" if metrics.get("workers", 0) > 0 else "Server ready, waiting for workers"
            )
        else:
            locality_status = "Server offline. Start queue_server.py to scale."

        st.markdown(
            f"""
            **Locality layer:** {locality_status}
            - Your Python code stays the same locally and remotely.
            - No YAML or cluster config required to start.
            - Use this dashboard to inspect jobs, workers, and runtime health.
            """
        )
        st.write(f"**Current load:** {load} active or queued jobs")
    st.divider()

# ====================================
# WORKER + JOB DATA FRAMES
# ====================================

workers_df = None
if isinstance(workers, dict) and "_error" not in workers and workers:
    worker_rows = []
    for name, data in workers.items():
        worker_rows.append(
            {
                "Worker": name,
                "GPU": data.get("gpu", False),
                "Memory (GB)": data.get("memory_gb", 0),
                "Load": data.get("load", 0),
                "Status": data.get("status", "unknown"),
                "Last seen sec": round(time.time() - data.get("last_heartbeat", time.time()), 1),
            }
        )
    workers_df = pd.DataFrame(worker_rows)

queue_df = None
running_df = None
active_job_ids = []
job_map = {}
all_job_ids = []

if isinstance(queue_data, dict) and "_error" not in queue_data:
    queue_rows = []
    for job in queue_data.get("queue", []):
        queue_rows.append(
            {
                "Job ID": job.get("id"),
                "Task": job.get("task_type"),
                "Priority": job.get("priority", 1),
                "GPU": job.get("gpu", False),
                "Memory": job.get("memory_gb", 0),
                "Submitted": time.strftime(
                    "%H:%M:%S", time.localtime(job.get("submitted_at", time.time()))
                ),
            }
        )
        job_id = job.get("id")
        if job_id:
            job_map[job_id] = job
            all_job_ids.append(job_id)
    queue_df = pd.DataFrame(queue_rows)

if isinstance(running, dict) and "_error" not in running and running:
    running_rows = []
    for job_id, data in running.items():
        job = data.get("job", {})
        runtime = time.time() - data.get("start_time", time.time())
        running_rows.append(
            {
                "Job ID": job_id,
                "Worker": data.get("worker"),
                "Task": job.get("task_type", "unknown"),
                "Priority": job.get("priority", 1),
                "GPU": data.get("gpu", False),
                "Runtime": format_elapsed(runtime),
                "Status": data.get("status", "running"),
                "Prompt": str(job.get("prompt", ""))[:80],
            }
        )
        active_job_ids.append(job_id)
        if job_id:
            job_map[job_id] = job
            all_job_ids.append(job_id)
    running_df = pd.DataFrame(running_rows)

history_df = None
if isinstance(history, dict) and "_error" not in history and history.get("jobs"):
    history_rows = []
    for job in history.get("jobs", []):
        job_id = job.get("id")
        history_rows.append(
            {
                "Job ID": job_id,
                "Status": job.get("status"),
                "Runtime": format_elapsed(job.get("runtime", 0)),
                "Cost": f"${job.get('cost', 0):.3f}",
            }
        )
        if job_id and job_id not in job_map:
            job_map[job_id] = job
            all_job_ids.append(job_id)
    history_df = pd.DataFrame(history_rows[-50:])

all_job_ids = [job_id for job_id in all_job_ids if job_id]
all_job_ids = sorted(set(all_job_ids), key=lambda x: x)

# ====================================
# DEVELOPER VIEWS
# ====================================

with st.expander("📍 Developer Control Panel", expanded=True):
    st.markdown(
        """
        **TRON is designed for developers, not operators.**
        - Build and test locally first
        - Scale with one command
        - See the runtime state here with no YAML
        - Understand what every job is doing in real time
        """
    )
    st.info(
        "This dashboard includes live runtime events, job detail inspection, and a developer-friendly status summary."
    )
    if not server_online:
        st.warning("Queue server is not reachable. Start `python queue_server.py` and refresh.")

with st.expander("📌 Locality & Simplicity", expanded=False):
    st.markdown(
        """
        TRON keeps distributed execution feeling local:
        - Your Python function is the same everywhere
        - No separate cluster API is required
        - No infrastructure config is needed to start
        - The dashboard surfaces the runtime, not the infrastructure
        """
    )

left, right = st.columns([2, 1])

with left:
    st.subheader("🧠 Worker Map")
    if workers_df is not None and not workers_df.empty:
        st.dataframe(workers_df, use_container_width=True)
    else:
        st.info("No workers are registered yet.")

    st.subheader("🔥 Active Jobs")
    if running_df is not None and not running_df.empty:
        st.dataframe(running_df, use_container_width=True)
    else:
        st.info("No active jobs at the moment.")

    st.subheader("📦 Queue Snapshot")
    if queue_df is not None and not queue_df.empty:
        st.dataframe(queue_df, use_container_width=True)
    else:
        st.success("Queue is empty.")

with right:
    st.subheader("📊 Recent History")
    if history_df is not None and not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No historical jobs yet.")

    st.subheader("🛠 Developer Commands")
    st.markdown(
        """
        - `python queue_server.py` — start TRON runtime
        - `streamlit run dashboard_v3.py` — open this dashboard
        - `python developer_getting_started.py` — learn by example
        - `python developer_experience_test.py` — see all developer patterns
        """
    )

    st.subheader("📎 Locality Signal")
    if server_online and metrics.get("workers", 0) > 0:
        st.success("Remote scaling is active. Your code path remains the same.")
    elif server_online:
        st.info("Server is online, but no workers are registered yet.")
    else:
        st.warning("No TRON runtime available. Start queue_server.py to scale.")

st.divider()
st.subheader("🔍 Job Detail Explorer")
if all_job_ids:
    job_labels = [
        f"{job_id[:8]}... [{job_map.get(job_id, {}).get('task_type', 'unknown')}]"
        for job_id in all_job_ids
    ]
    selected_label = st.selectbox("Select a known job", job_labels, key="job_detail_select")
    selected_job_id = all_job_ids[job_labels.index(selected_label)]
    status_data = safe_get(f"/status/{selected_job_id}")
    result_data = safe_get(f"/result/{selected_job_id}")

    st.markdown(f"**Job ID:** `{selected_job_id}`")
    with st.expander("Job status & metadata", expanded=True):
        if isinstance(status_data, dict) and status_data.get("status") == "not_found":
            st.warning("Job was not found in the runtime store.")
        elif isinstance(status_data, dict) and "_error" in status_data:
            st.error(status_data["_error"])
        else:
            st.json(status_data)

    with st.expander("Job payload & routing details", expanded=False):
        payload = job_map.get(selected_job_id, status_data)
        st.json(payload)

    with st.expander("Result / completion data", expanded=False):
        if isinstance(result_data, dict) and result_data.get("status") == "not_found":
            st.info("No result yet; job is still queued or running.")
        elif isinstance(result_data, dict) and "_error" in result_data:
            st.error(result_data["_error"])
        else:
            if result_data.get("status") == "completed":
                st.json(result_data.get("result", {}))
            else:
                st.info(f"Job status is {result_data.get('status', 'unknown')}.")

    if selected_job_id in active_job_ids:
        st.info("This job is currently active. Use the live stream below to inspect runtime events.")
    else:
        st.info("This job is not active right now.")
else:
    st.info("No job IDs available yet. Submit work and refresh this page.")

# ====================================
# LIVE STREAM
# ====================================

st.divider()
st.subheader("📡 Live Job Stream")

if active_job_ids:
    selected_job = st.selectbox("Select active job to stream", active_job_ids, key="selected_job")
    if selected_job:
        start_stream_thread(selected_job)
        stream_data = list(st.session_state.event_buffers.get(selected_job, []))
        if stream_data:
            st.dataframe(pd.DataFrame(stream_data), use_container_width=True)
        else:
            st.info("Waiting for live job stream events...")
else:
    st.info("No active jobs to stream right now.")

st.divider()
st.subheader("✨ Developer Value")
st.markdown(
    """
    - Same Python code locally and remotely
    - No Kubernetes manifests or cluster wiring
    - Dashboard reflects the actual job lifecycle
    - Local-first mode for rapid developer feedback
    - Remote scaling when you need it
    """
)
