"""How far each agent can be trusted: the verifier, the misses, escalation and cost.

    python scripts/report_trust.py --part train      # where the definitions were developed
    python scripts/report_trust.py --part test       # after eval-spec-v1.2 froze them

Reads the recorded runs in runs/explain/; no model is called. For every agent rung
it reports:
- the verifier: the agent's own first pick against the verified one, how many first
  picks it rejected, and how many of those were wrong answers caught versus right
  answers whose cited evidence didn't hold up
- why each top-1 miss missed (src/evaluate_explain.py, FAILURES)
- escalation: a risk-coverage curve over the first cause's confidence, with cases that
  have no verified first answer always escalated. For hosted Spark, the threshold is
  fitted on its train-split runs (lowest confidence reaching TRUST_TARGET AC@1) and
  applied unchanged to test; other models get the curve and its area only.
- cost: tokens, model seconds, turns and tool calls per investigation, and dollars at
  the Meta Model API's published prices for the hosted model.
A summary line goes to runs/trust_summary.jsonl.
"""

import argparse
import json
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_latency import META_PRICES  # noqa: E402
from src.evaluate_explain import (TRUST_TARGET, at_threshold, fit_threshold, trust_case,  # noqa: E402
                                  trust_summary)
from src.load_rcaeval import load_labels  # noqa: E402
from src.provenance import git_state  # noqa: E402

RUNS = ROOT / "runs" / "explain"
SUITES = ("RE1", "RE2", "RE3")
# the agent rungs, each with the verifier; RE1 has metrics only, so its hosted agent is spark-agent
AGENTS = {"spark": {"RE1": "spark-agent", "RE2": "spark-agent-all", "RE3": "spark-agent-all"},
          "spark-open": {"RE1": "open-agent", "RE2": "open-agent-all", "RE3": "open-agent-all"},
          "glimmer": {s: "glimmer-agent-all" for s in SUITES},
          "glimmer-tuned": {s: "glimmer-tuned-agent-all" for s in SUITES}}
LABELS = {"spark": "Muse Spark agent (hosted)", "spark-open": "Muse Spark Open agent",
          "glimmer": "Glimmer 30B agent", "glimmer-tuned": "Glimmer 30B agent, fine-tuned"}
TRAIN_RANKER = "spark-agent-all"   # the only rung run on the train split


def records(ranker: str, suite: str, part: str) -> list[dict]:
    path = RUNS / f"{ranker}_{suite}_{part}.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else []


def cost(rows: list[dict], hosted: bool) -> dict:
    def med(values):
        return statistics.median(values) if values else None

    def p90(values):
        return sorted(values)[int(0.9 * (len(values) - 1))] if values else None
    used = [r for r in rows if r.get("usage")]
    tokens_in = [r["usage"].get("input_tokens", 0) for r in used]
    tokens_out = [r["usage"].get("output_tokens", 0) for r in used]
    seconds = [r["usage"]["model_seconds"] for r in used if r["usage"].get("model_seconds")]
    out = {"input_tokens_median": med(tokens_in), "output_tokens_median": med(tokens_out),
           "model_seconds_median": med(seconds), "model_seconds_p90": p90(seconds),
           "turns_median": med([r["turns"] for r in rows if r.get("turns")]),
           "tool_calls_median": med([sum(s.get("type") == "tool" for s in r.get("trail") or []) for r in rows])}
    if hosted and used:
        for tier, (p_in, p_out) in META_PRICES.items():
            out[f"usd_per_investigation_{tier}"] = round(
                statistics.mean(i * p_in + o * p_out for i, o in zip(tokens_in, tokens_out)) / 1e6, 4)
    return out


def trust_cases(ranker: str, suite: str, part: str, labels: dict) -> list:
    return [trust_case(r, labels[r["case_id"]].root_service) for r in records(ranker, suite, part)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["train", "test"], required=True)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    labels = {s: load_labels(s) for s in SUITES}
    commit, dirty = git_state()

    train = [c for s in SUITES for c in trust_cases(TRAIN_RANKER, s, "train", labels[s])]
    threshold = fit_threshold(train)
    print(f"escalation threshold for hosted Spark, fitted on {len(train)} train runs to reach AC@1 "
          f"{TRUST_TARGET}: {threshold}  (commit {commit}{' dirty' if dirty else ''})\n")

    agents = {"spark": {s: TRAIN_RANKER for s in SUITES}} if args.part == "train" else AGENTS
    result = {}
    for key, by_suite in agents.items():
        for suite, ranker in by_suite.items():
            rows = records(ranker, suite, args.part)
            if not rows:
                continue
            cases = [trust_case(r, labels[suite][r["case_id"]].root_service) for r in rows]
            summary = trust_summary(cases)
            if key == "spark" and threshold is not None:
                summary["escalation"] = at_threshold(cases, threshold)
            summary["cost"] = cost(rows, hosted=key == "spark")
            result[f"{key}/{suite}"] = {"ranker": ranker, **summary}
            f, rc = summary["failures"], summary["risk_coverage"]
            print(f"{LABELS[key]} · {suite} {args.part} · {summary['cases']} cases · ranker {ranker}")
            print(f"  AC@1 {summary['ac1']:.3f}  (agent's own first pick, before verification: "
                  f"{summary['ac1_before_verification']:.3f})")
            print(f"  verifier: rejected {summary['causes_rejected']} of {summary['causes']} causes; first pick "
                  f"rejected on {summary['first_pick_rejected']} cases: {summary['first_pick_rejected_and_wrong']} "
                  f"wrong answers caught, {summary['first_pick_right_but_unsupported']} right but unsupported")
            print("  top-1 misses: " + (", ".join(f"{k} {v}" for k, v in f.items() if v) or "none"))
            print(f"  risk-coverage area {rc['aurc']:.4f}; always escalated (no verified first answer): "
                  f"{rc['never_answered']}")
            if "escalation" in summary:
                e = summary["escalation"]
                print(f"  at threshold {e['threshold']}: answers {e['answered']}, escalates {e['escalated']} "
                      f"(coverage {e['coverage']:.3f}), AC@1 on answered {e['ac1_answered']:.3f}, "
                      f"wrong answers given {e['wrong_answered']}")
            c = summary["cost"]
            usd = f", ${c['usd_per_investigation_contributor']} Contributor / ${c['usd_per_investigation_standard']} " \
                  f"Standard per investigation" if "usd_per_investigation_contributor" in c else ""
            secs = (f"{c['model_seconds_median']:.1f} s model time (p90 {c['model_seconds_p90']:.1f})"
                    if c["model_seconds_median"] is not None else "model time not recorded for these runs")
            print(f"  cost: median {c['input_tokens_median']:.0f} in / {c['output_tokens_median']:.0f} out tokens, "
                  f"{secs}, {c['turns_median']} turns, {c['tool_calls_median']} tool calls{usd}\n")

    if not args.no_record:
        line = {"run_id": uuid.uuid4().hex[:12], "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_commit": commit, "git_dirty": dirty, "part": args.part, "target": TRUST_TARGET,
                "threshold": threshold, "train_runs": len(train), "results": result}
        with open(ROOT / "runs" / "trust_summary.jsonl", "a") as f:
            f.write(json.dumps(line, default=float) + "\n")
        print(f"recorded run {line['run_id']}")


if __name__ == "__main__":
    main()
