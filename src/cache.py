"""Record-and-replay cache for Meta Model API calls.

Every request and its response are appended to a gzip JSONL file under runs/,
keyed by a hash of the request. A repeated request is answered from the file,
so re-running an evaluation costs nothing and returns exactly the recorded
answers. With `offline=True` a cache miss is an error, which is how
`run_eval.py --from-cache` reproduces every reported number without network
access or credits.

Encrypted reasoning blobs (`encrypted_content`, which stateless replay resends
every turn) are dropped from both the key and the stored copy; they would
otherwise dominate the file size. Replay rebuilds the same stripped requests,
so every turn of an agent conversation still hits.
"""

import gzip
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

DROPPED_FIELDS = {"encrypted_content"}


class CacheMiss(KeyError):
    pass


def strip(value):
    """Copy of a JSON value without encrypted reasoning blobs."""
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in DROPPED_FIELDS}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def request_key(request: dict) -> str:
    blob = json.dumps(strip(request), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class ModelCache:
    def __init__(self, path: Path, client=None, offline: bool = False):
        self.path, self.client, self.offline = Path(path), client, offline
        self.entries: dict[str, dict] = {}
        self.hits = self.misses = 0
        self._lock = threading.Lock()
        if self.path.exists():
            with gzip.open(self.path, "rt") as f:
                for line in f:
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry

    def create(self, **request) -> dict:
        """responses.create, answered from the cache when this exact request was seen."""
        key = request_key(request)
        with self._lock:
            if key in self.entries:
                self.hits += 1
                return self.entries[key]["response"]
        if self.offline or self.client is None:
            raise CacheMiss(f"no recorded response for request {key[:12]} (offline replay)")
        response = self.client.responses.create(**request).model_dump(mode="json")
        entry = {"key": key, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "request": strip(request), "response": strip(response)}
        with self._lock:
            self.misses += 1
            self.entries[key] = entry
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.path, "at") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        return response  # live responses keep their blobs for the next turn

    def usage(self) -> dict:
        """Token totals over every recorded call."""
        totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        for entry in self.entries.values():
            u = entry["response"].get("usage") or {}
            totals["calls"] += 1
            totals["input_tokens"] += u.get("input_tokens", 0)
            totals["output_tokens"] += u.get("output_tokens", 0)
            totals["reasoning_tokens"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        return totals


def output_text(response: dict) -> str:
    """Concatenated assistant text from a Responses API response dict."""
    return "".join(part.get("text", "") for item in response.get("output", []) if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")


def function_calls(response: dict) -> list[dict]:
    return [item for item in response.get("output", []) if item.get("type") == "function_call"]
