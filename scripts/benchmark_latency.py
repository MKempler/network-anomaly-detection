"""Measure how long one investigation takes, and what it costs, per model.

    python scripts/benchmark_latency.py                 # both models, 20 RE1 train cases
    python scripts/benchmark_latency.py --cases 5 --models open

Evaluation runs investigate many cases in parallel and replay from the cache,
so their timings say little about one incident. This runs the metrics-only
agent on a fixed set of train cases, one case at a time, with fresh calls in
a separate cache (runs/latency/), and records each case's wall-clock time,
model time, turns and tokens. Train cases only: the test split is untouched.
Results go to runs/latency_summary.jsonl.
"""

import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import investigate  # noqa: E402
from src.cache import ModelCache  # noqa: E402
from src.config import DEFAULT  # noqa: E402
from src.llm import PROVIDERS, client  # noqa: E402
from src.load_rcaeval import case_index, load_case, load_labels  # noqa: E402
from src.provenance import git_state  # noqa: E402
from src.splits import case_ids  # noqa: E402
from src.tools import Investigation  # noqa: E402

MODELS = {"hosted": "meta", "open": "baseten"}
META_PRICES = {"standard": (1.25, 4.25), "contributor": (0.10, 0.20)}  # USD per 1M tokens (input, output)


def pick_cases(n: int) -> list[str]:
    """Every k-th RE1 train case, so all three systems are represented."""
    ids = case_ids("RE1", "train")
    step = max(1, len(ids) // n)
    return ids[::step][:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--models", nargs="+", default=sorted(MODELS), choices=sorted(MODELS))
    args = parser.parse_args()

    index, labels = case_index("RE1"), load_labels("RE1")
    ids = pick_cases(args.cases)
    commit, dirty = git_state()
    print(f"{len(ids)} RE1 train cases, sequential | commit={commit}{' (dirty)' if dirty else ''}")
    for name in args.models:
        provider = MODELS[name]
        model = PROVIDERS[provider].model
        cache = ModelCache(ROOT / "runs" / "latency" / f"{name}_RE1_train.jsonl.gz", client(provider))
        rows = []
        for cid in ids:
            inv = Investigation(load_case("RE1", cid, index[cid]), labels[cid].inject_time, DEFAULT.investigation)
            started = time.monotonic()
            run = investigate(inv, cache, model, frozenset({"metrics"}), DEFAULT.investigation.max_steps, f"{name}-agent")
            rows.append({"case_id": cid, "wall_s": round(time.monotonic() - started, 1), "turns": run.turns,
                         "error": run.ranking.error, **run.ranking.usage})
            print(f"  {name} {cid}: {rows[-1]['wall_s']}s, {run.turns} turns", flush=True)
        ok = [r for r in rows if not r["error"]]
        wall = [r["wall_s"] for r in ok]
        summary = {"cases": len(rows), "failed": len(rows) - len(ok),
                   "wall_s_median": statistics.median(wall), "wall_s_p90": sorted(wall)[int(0.9 * (len(wall) - 1))],
                   "model_s_median": statistics.median(r["model_seconds"] for r in ok),
                   "turns_median": statistics.median(r["turns"] for r in ok),
                   "input_tokens_mean": round(statistics.mean(r["input_tokens"] for r in ok)),
                   "output_tokens_mean": round(statistics.mean(r["output_tokens"] for r in ok))}
        if provider == "meta":
            for mode, (p_in, p_out) in META_PRICES.items():
                summary[f"usd_per_case_{mode}"] = round(
                    (summary["input_tokens_mean"] * p_in + summary["output_tokens_mean"] * p_out) / 1e6, 4)
        print(f"{name}: {summary}")
        entry = {"run_id": uuid.uuid4().hex[:12], "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "git_commit": commit, "git_dirty": dirty, "model_name": name, "provider": provider, "model": model,
                 "suite": "RE1", "part": "train", "ranker": "agent (metrics)", "summary": summary, "cases": rows}
        with open(ROOT / "runs" / "latency_summary.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
