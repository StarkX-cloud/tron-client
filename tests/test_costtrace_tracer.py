from types import SimpleNamespace

import pytest

from tron_costtrace.store import JsonlStore
from tron_costtrace.tracer import trace


def _openai_shaped_response(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _anthropic_shaped_response(model="claude-haiku-3-5", input_tokens=10, output_tokens=5):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_records_openai_shaped_response(tmp_path):
    store = JsonlStore(tmp_path / "log.jsonl")
    with trace("classify", store=store) as t:
        t.record(_openai_shaped_response())
        t.mark_success(True)

    records = store.read_all()
    assert len(records) == 1
    r = records[0]
    assert r.task == "classify"
    assert r.model == "gpt-4o-mini"
    assert r.input_tokens == 10
    assert r.output_tokens == 5
    assert r.success is True
    assert r.cost_usd is not None and r.cost_usd > 0


def test_records_anthropic_shaped_response(tmp_path):
    store = JsonlStore(tmp_path / "log.jsonl")
    with trace("classify", store=store) as t:
        t.record(_anthropic_shaped_response())
        t.mark_success(True)

    r = store.read_all()[0]
    assert r.model == "claude-haiku-3-5"
    assert r.input_tokens == 10
    assert r.output_tokens == 5


def test_unmarked_call_defaults_success_true_when_no_exception(tmp_path):
    store = JsonlStore(tmp_path / "log.jsonl")
    with trace("classify", store=store) as t:
        t.record(_openai_shaped_response())

    r = store.read_all()[0]
    assert r.success is True


def test_exception_inside_trace_records_failure_and_reraises(tmp_path):
    store = JsonlStore(tmp_path / "log.jsonl")

    with pytest.raises(ValueError):
        with trace("classify", store=store) as t:
            t.record(_openai_shaped_response())
            raise ValueError("boom")

    r = store.read_all()[0]
    assert r.success is False
    assert "boom" in r.error


def test_explicit_mark_success_false_overrides_no_exception(tmp_path):
    store = JsonlStore(tmp_path / "log.jsonl")
    with trace("classify", store=store) as t:
        t.record(_openai_shaped_response())
        t.mark_success(False)

    r = store.read_all()[0]
    assert r.success is False


def test_retry_count_is_recorded(tmp_path):
    store = JsonlStore(tmp_path / "log.jsonl")
    with trace("classify", store=store) as t:
        t.retry()
        t.retry()
        t.record(_openai_shaped_response())
        t.mark_success(True)

    r = store.read_all()[0]
    assert r.retry_count == 2


def test_no_record_call_leaves_model_unknown_and_cost_none(tmp_path):
    store = JsonlStore(tmp_path / "log.jsonl")
    with trace("classify", store=store):
        pass

    r = store.read_all()[0]
    assert r.model == "unknown"
    assert r.cost_usd is None


def test_latency_is_measured(tmp_path):
    import time

    store = JsonlStore(tmp_path / "log.jsonl")
    with trace("classify", store=store) as t:
        time.sleep(0.01)
        t.record(_openai_shaped_response())

    r = store.read_all()[0]
    assert r.latency_seconds >= 0.01
