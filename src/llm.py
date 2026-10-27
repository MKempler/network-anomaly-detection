"""Clients for the two ways we run Meta's models.

- meta: Meta Model API, hosted Muse Spark 1.3 (https://api.meta.ai/v1)
- baseten: Baseten, Muse Spark Open, Meta's open-weight model
  (https://inference.baseten.co/v1)
- glimmer, glimmer-tuned: Muse Glimmer, Meta's 30B open-weight model, and our
  fine-tuned adapter on it, served by vLLM on Modal (scripts/finetune/modal_app.py)

The first two serve the Responses API. vLLM serves Glimmer over Chat
Completions, so those clients are wrapped by src/chat_adapter.py; either way
the same rankers and agent loop run unchanged. Keys and the Glimmer server's
URL are read from the environment or a git-ignored .env file at the repo root,
and are never logged.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from src.chat_adapter import ChatAsResponses

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

# A stuck connection otherwise waits the SDK default of 10 minutes per attempt;
# one agent case hung for almost two hours that way. Typical calls take ~25 s.
REQUEST_TIMEOUT_S = 120


@dataclass(frozen=True)
class Provider:
    base_url: str
    key_name: str
    model: str
    url_name: str = ""          # read the base URL from this setting instead (a server we deploy)
    chat: bool = False          # serves Chat Completions only
    timeout_s: int = REQUEST_TIMEOUT_S


PROVIDERS = {
    # Contributor mode: the same Muse Spark 1.3 model at about a fifteenth of the
    # Standard price, in exchange for Meta being allowed to train on requests.
    # Everything sent is public benchmark data (SMD, RCAEval), so that trade is acceptable.
    "meta": Provider("https://api.meta.ai/v1", "MODEL_API_KEY", "muse-spark-1.3-contributor"),
    # The hackathon key only works with this exact model name.
    "baseten": Provider("https://inference.baseten.co/v1", "BASETEN_API_KEY", "meta-models/muse-spark-open-1026"),
    # One vLLM server holds the base model and the LoRA adapter; the model name picks one.
    # Many investigations share one GPU, so a turn can wait in the queue for a while.
    "glimmer": Provider("", "GLIMMER_API_KEY", "muse-glimmer", url_name="GLIMMER_BASE_URL", chat=True, timeout_s=600),
    "glimmer-tuned": Provider("", "GLIMMER_API_KEY", "glimmer-tuned", url_name="GLIMMER_BASE_URL", chat=True,
                              timeout_s=600),
}
DEFAULT_MODEL = PROVIDERS["meta"].model


def api_key(name: str = "MODEL_API_KEY") -> str:
    if key := os.environ.get(name):
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            key_name, sep, value = line.partition("=")
            if sep and key_name.strip() == name and value.strip():
                return value.strip()
    raise RuntimeError(f"{name} is not set (environment or .env)")


def client(provider: str = "meta"):
    p = PROVIDERS[provider]
    base_url = api_key(p.url_name) if p.url_name else p.base_url
    c = OpenAI(base_url=base_url, api_key=api_key(p.key_name), max_retries=5, timeout=p.timeout_s)
    return ChatAsResponses(c) if p.chat else c
