"""Meta Model API client.

The API is OpenAI-compatible at https://api.meta.ai/v1. The key is read from
MODEL_API_KEY in the environment or in a git-ignored .env file at the repo
root, and is never logged.
"""

import os
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_MODEL = "muse-spark-1.3"


def api_key() -> str:
    if key := os.environ.get("MODEL_API_KEY"):
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "MODEL_API_KEY" and value.strip():
                return value.strip()
    raise RuntimeError("MODEL_API_KEY is not set (environment or .env)")


def client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=5)
