"""The pre-registered train/test split (data_splits.json, built by scripts/make_splits.py)."""

import json
from functools import lru_cache
from pathlib import Path

SPLITS_FILE = Path(__file__).resolve().parents[1] / "data_splits.json"


@lru_cache(maxsize=1)
def _splits() -> dict:
    return json.loads(SPLITS_FILE.read_text())["suites"]


def case_ids(suite: str, part: str) -> list[str]:
    """Case ids in `part` ("train" or "test") of `suite`."""
    if part not in ("train", "test"):
        raise ValueError(f"unknown split part {part!r}")
    return _splits()[suite][part]
