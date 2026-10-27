"""Turn the hosted agent's correct train-split investigations into fine-tuning data.

    python scripts/finetune/build_sft.py

Keeps a train case only if hosted Spark (all sources) ranked the true root
cause first and the verifier accepted that cause. Only train cases are used;
test cases are never touched. Each kept investigation is rebuilt from the
committed model-call cache as an OpenAI-style chat:
- the agent's instructions
- the first message
- every assistant turn, with its tool calls
- every tool result
- the final submit_ranking call

The conversion is the one the evaluation uses to run Glimmer
(src/chat_adapter.py), so training and serving render conversations alike.

Writes runs/finetune/{train,validation}.jsonl and a manifest.
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.chat_adapter import chat_tools, to_messages  # noqa: E402
from src.config import DEFAULT  # noqa: E402
from src.evaluate_explain import hits  # noqa: E402
from src.explain import one_shot_input  # noqa: E402
from src.load_rcaeval import case_index, load_case, load_labels  # noqa: E402
from src.tools import Investigation  # noqa: E402

OUT = ROOT / "runs" / "finetune"
RANKER = "spark-agent-all"
VALIDATION_SHARE = 0.1


def conversations(suite: str) -> dict[str, dict]:
    """first user message -> the final request/response pair of that conversation, from the cache."""
    by_first: dict[str, dict] = {}
    for line in gzip.open(ROOT / "runs" / "llm" / f"{RANKER}_{suite}.jsonl.gz", "rt"):
        entry = json.loads(line)
        items = entry["request"]["input"]
        if not isinstance(items, list) or not items or items[0].get("role") != "user":
            continue
        first = items[0]["content"]
        if first not in by_first or len(items) > len(by_first[first]["request"]["input"]):
            by_first[first] = entry
    return by_first


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kept, skipped = [], {"wrong_or_unverified": 0, "failed": 0, "not_in_cache": 0, "unfinished": 0}
    for suite in ("RE1", "RE2", "RE3"):
        records = ROOT / "runs" / "explain" / f"{RANKER}_{suite}_train.jsonl"
        if not records.exists():
            continue
        index, labels, convs = case_index(suite), load_labels(suite), conversations(suite)
        for rec in map(json.loads, open(records)):
            if rec["services"] is None:
                skipped["failed"] += 1
                continue
            # scored with the benchmark's own rule; labels only choose examples, never enter one
            top = rec["causes"][0] if rec["causes"] else {}
            if not (top.get("verified") and hits(rec["services"], labels[rec["case_id"]].root_service)[0]):
                skipped["wrong_or_unverified"] += 1
                continue
            case = load_case(suite, rec["case_id"], index[rec["case_id"]])
            first, _ = one_shot_input(Investigation(case, labels[rec["case_id"]].inject_time, DEFAULT.investigation))
            entry = convs.get(first)
            if entry is None:
                skipped["not_in_cache"] += 1
                continue
            final = entry["response"].get("output", [])
            if not any(i.get("type") == "function_call" and i.get("name") == "submit_ranking" for i in final):
                skipped["unfinished"] += 1
                continue
            messages = to_messages(entry["request"]["instructions"], entry["request"]["input"] + final)
            kept.append({"id": rec["case_id"], "suite": suite, "messages": messages,
                         "tools": chat_tools(entry["request"]["tools"])})
    train, validation = [], []
    for ex in kept:
        share = int(hashlib.sha256(ex["id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (validation if share < VALIDATION_SHARE else train).append(ex)
    for name, rows in (("train", train), ("validation", validation)):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
    chars = [sum(len(json.dumps(m)) for m in ex["messages"]) + len(json.dumps(ex["tools"])) for ex in kept]
    manifest = {"source_ranker": RANKER, "kept": len(kept), "train": len(train), "validation": len(validation),
                "skipped": skipped, "approx_tokens": {"median": sorted(chars)[len(chars) // 2] // 4 if chars else 0,
                                                       "max": max(chars) // 4 if chars else 0}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
