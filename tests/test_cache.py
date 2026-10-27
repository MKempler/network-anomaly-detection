import gzip
import json

import pytest

from src.cache import CacheMiss, ModelCache, function_calls, output_text, request_key, strip


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, mode="json"):
        return self.payload


class FakeClient:
    """Counts calls; each response echoes the request's input and carries a large blob."""

    def __init__(self):
        self.calls = 0
        self.responses = self

    def create(self, **request):
        self.calls += 1
        return FakeResponse({
            "output": [
                {"type": "reasoning", "id": "r1", "encrypted_content": "X" * 5000},
                {"type": "message", "content": [{"type": "output_text", "text": f"echo {request['input']}"}]},
                {"type": "function_call", "name": "top_deviations", "arguments": "{}", "call_id": "c1"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 15}},
        })


def test_second_identical_request_is_served_from_cache(tmp_path):
    client = FakeClient()
    cache = ModelCache(tmp_path / "calls.jsonl.gz", client)
    first = cache.create(model="m", input="hello")
    second = cache.create(model="m", input="hello")
    assert client.calls == 1 and (cache.misses, cache.hits) == (1, 1)
    assert output_text(first) == output_text(second) == "echo hello"


def test_cache_persists_across_instances_and_replays_offline(tmp_path):
    path = tmp_path / "calls.jsonl.gz"
    ModelCache(path, FakeClient()).create(model="m", input="hello")
    replay = ModelCache(path, client=None, offline=True)
    assert output_text(replay.create(model="m", input="hello")) == "echo hello"
    with pytest.raises(CacheMiss):
        replay.create(model="m", input="something new")


def test_encrypted_blobs_are_not_stored_or_keyed(tmp_path):
    path = tmp_path / "calls.jsonl.gz"
    live = ModelCache(path, FakeClient()).create(model="m", input="hi")
    assert live["output"][0]["encrypted_content"]                      # live caller still gets the blob
    with gzip.open(path, "rt") as f:
        stored = f.read()
    assert "encrypted_content" not in stored and "XXXX" not in stored
    with_blob = {"model": "m", "input": [{"type": "reasoning", "encrypted_content": "abc"}]}
    without = {"model": "m", "input": [{"type": "reasoning"}]}
    assert request_key(with_blob) == request_key(without)


def test_key_is_order_independent_and_sensitive_to_content():
    assert request_key({"a": 1, "b": 2}) == request_key({"b": 2, "a": 1})
    assert request_key({"a": 1}) != request_key({"a": 2})


def test_usage_totals_and_helpers(tmp_path):
    cache = ModelCache(tmp_path / "calls.jsonl.gz", FakeClient())
    response = cache.create(model="m", input="a")
    cache.create(model="m", input="b")
    assert cache.usage() == {"calls": 2, "input_tokens": 20, "output_tokens": 40, "reasoning_tokens": 30}
    assert [c["name"] for c in function_calls(response)] == ["top_deviations"]


def test_strip_leaves_other_fields_alone():
    value = {"keep": [1, {"encrypted_content": "z", "id": "r"}], "encrypted_content": "y"}
    assert strip(value) == {"keep": [1, {"id": "r"}]}
    assert json.dumps(strip(value))


def test_live_latency_is_recorded_and_returned_on_replay(tmp_path):
    path = tmp_path / "calls.jsonl.gz"
    live = ModelCache(path, FakeClient()).create(model="m", input="x")
    assert live["_cache"]["hit"] is False and live["_cache"]["latency_s"] >= 0
    replayed = ModelCache(path, offline=True).create(model="m", input="x")
    assert replayed["_cache"] == {"hit": True, "latency_s": live["_cache"]["latency_s"]}
    with gzip.open(path, "rt") as f:
        assert "_cache" not in f.read()     # bookkeeping is not stored inside the response
