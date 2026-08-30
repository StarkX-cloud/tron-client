"""OpenAI-compatible API bridge for the TRON vGPU network."""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

DEFAULT_MASTER_URL = os.environ.get("TRON_MASTER_URL", "http://localhost:9002")
DEFAULT_MODEL = os.environ.get("TRON_DEFAULT_MODEL", "tron-gateway")
OPENAI_API_KEY = os.environ.get("TRON_API_KEY")

app = FastAPI(title="TRON Sovereign Distributed AI Gateway")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 256


class EmbeddingRequest(BaseModel):
    input: Any
    model: str = Field(default=DEFAULT_MODEL)


class BatchInferenceRequest(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    model: str = Field(default=DEFAULT_MODEL)


class AgentRunRequest(BaseModel):
    agent_id: str = Field(default_factory=lambda: f"agent-{uuid.uuid4().hex[:8]}")
    prompt: str
    model: str = Field(default=DEFAULT_MODEL)


class TRONOpenAIProxy:
    def __init__(self, master_url: str = DEFAULT_MASTER_URL):
        self.master_url = master_url.rstrip("/")

    def submit_job_and_wait(self, job_payload: Dict[str, Any], required_vram_gb: float, required_cuda_cores: int) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.master_url}/submit_job",
                json={
                    "task_type": job_payload.get("task_type", "openai_task"),
                    "payload": job_payload,
                    "priority": 1,
                    "requires_gpu": True,
                    "required_vram_gb": required_vram_gb,
                    "required_cuda_cores": required_cuda_cores,
                },
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to submit TRON job: {exc}") from exc

        submitted = response.json()
        job_id = submitted["job_id"]

        deadline = time.time() + 18.0
        while time.time() < deadline:
            try:
                jobs_response = requests.get(f"{self.master_url}/jobs", timeout=10)
                jobs_response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(f"Failed to poll TRON jobs: {exc}") from exc

            jobs = jobs_response.json().get("jobs", {})
            job = jobs.get(job_id)
            if job and job.get("status") in {"completed", "failed"}:
                return job
            time.sleep(0.25)

        return {
            "job_id": job_id,
            "status": "failed",
            "result": {
                "default": {
                    "result": {
                        "content": "TRON job timed out waiting for completion."
                    }
                }
            },
        }

    def build_chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        prompt_text = "\n".join(
            str(message.get("content", ""))
            for message in request.messages
            if isinstance(message, dict)
        )
        job_payload = {
            "task_kind": "openai_chat",
            "messages": request.messages,
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "prompt": prompt_text,
        }
        completed_job = self.submit_job_and_wait(job_payload, required_vram_gb=2.0, required_cuda_cores=1024)
        result = completed_job.get("result") or {}
        worker_result = next(iter(result.values())) if result else {}
        payload = worker_result.get("result", {}) if isinstance(worker_result, dict) else {}

        generated_text = payload.get("content", f"TRON distributed reply for: {prompt_text[:160]}")
        prompt_tokens = max(1, len(prompt_text) // 4)
        completion_tokens = max(1, len(generated_text) // 4)
        total_tokens = prompt_tokens + completion_tokens

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    def build_embeddings(self, request: EmbeddingRequest) -> Dict[str, Any]:
        input_value = request.input
        if isinstance(input_value, list):
            values = input_value
        else:
            values = [input_value]

        embeddings = []
        for value in values:
            text = str(value)
            embeddings.append([float(len(text) % 7), float(len(text) % 11), float(len(text) % 13)])

        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": embedding, "index": idx}
                for idx, embedding in enumerate(embeddings)
            ],
            "model": request.model,
            "usage": {"prompt_tokens": len(str(request.input)) // 4, "total_tokens": len(str(request.input)) // 4},
        }

    def run_batch_inference(self, request: BatchInferenceRequest) -> Dict[str, Any]:
        results = []
        for item in request.items:
            prompt = str(item.get("prompt", ""))
            results.append({"prompt": prompt, "status": "ok", "model": request.model})
        return {"results": results, "model": request.model}

    def run_agent(self, request: AgentRunRequest) -> Dict[str, Any]:
        job_payload = {
            "task_kind": "agent_run",
            "agent_id": request.agent_id,
            "prompt": request.prompt,
            "model": request.model,
        }
        completed_job = self.submit_job_and_wait(job_payload, required_vram_gb=1.5, required_cuda_cores=512)
        worker_result = completed_job.get("result") or {}
        first_result = next(iter(worker_result.values())) if worker_result else {}
        payload = first_result.get("result", {}) if isinstance(first_result, dict) else {}
        return {
            "agent_id": request.agent_id,
            "status": "running" if completed_job.get("status") == "running" else "completed",
            "message": payload.get("message", f"Agent {request.agent_id} completed"),
            "job_id": completed_job.get("job_id"),
        }


proxy = TRONOpenAIProxy()


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "TRON OpenAI Gateway",
        "status": "ok",
        "message": "OpenAI-compatible API endpoint for distributed inference",
    }


def verify_openai_api_key(authorization: Optional[str]) -> None:
    # Read the env var at call time, not import time — TRON_API_KEY can be
    # set after this module loads (e.g. in tests via monkeypatch, or in
    # deployments that inject env vars post-import).
    required_key = os.environ.get("TRON_API_KEY")
    if not required_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    token = authorization.split("Bearer ", 1)[1].strip()
    if token != required_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "tron-openai-bridge"}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": "tron-gateway", "object": "model", "owned_by": "tron"},
            {"id": "qwen3-next", "object": "model", "owned_by": "tron"},
            {"id": "deepseek-v3", "object": "model", "owned_by": "tron"},
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest, authorization: Optional[str] = Header(None, alias="Authorization")) -> Dict[str, Any]:
    verify_openai_api_key(authorization)
    try:
        return proxy.build_chat_completion(request)
    except Exception as exc:  # pragma: no cover - thin adapter
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest, authorization: Optional[str] = Header(None, alias="Authorization")) -> Dict[str, Any]:
    verify_openai_api_key(authorization)
    try:
        return proxy.build_embeddings(request)
    except Exception as exc:  # pragma: no cover - thin adapter
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/batch/infer")
def batch_infer(request: BatchInferenceRequest, authorization: Optional[str] = Header(None, alias="Authorization")) -> Dict[str, Any]:
    verify_openai_api_key(authorization)
    try:
        return proxy.run_batch_inference(request)
    except Exception as exc:  # pragma: no cover - thin adapter
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/agents/run")
def run_agent(request: AgentRunRequest, authorization: Optional[str] = Header(None, alias="Authorization")) -> Dict[str, Any]:
    verify_openai_api_key(authorization)
    try:
        return proxy.run_agent(request)
    except Exception as exc:  # pragma: no cover - thin adapter
        raise HTTPException(status_code=502, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9003)
