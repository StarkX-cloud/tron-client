from tron_costtrace.schema import CallRecord
from tron_costtrace.store import JsonlStore


def _record(task="t", model="gpt-4o-mini"):
    return CallRecord(
        task=task,
        model=model,
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.001,
        latency_seconds=0.5,
        retry_count=0,
        success=True,
        error=None,
        timestamp=1234.5,
    )


def test_append_and_read_roundtrip(tmp_path):
    store = JsonlStore(tmp_path / "log.jsonl")
    store.append(_record())
    store.append(_record(task="other"))

    records = store.read_all()
    assert len(records) == 2
    assert records[0].task == "t"
    assert records[1].task == "other"


def test_read_all_on_missing_file_returns_empty(tmp_path):
    store = JsonlStore(tmp_path / "does_not_exist.jsonl")
    assert store.read_all() == []


def test_creates_parent_directory(tmp_path):
    nested = tmp_path / "a" / "b" / "log.jsonl"
    store = JsonlStore(nested)
    store.append(_record())
    assert nested.exists()
