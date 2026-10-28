"""Comparison table for every ranker scored on one suite's split.

    python scripts/report_explain.py --suite RE1 --part test

Reads the per-case records in runs/explain/ (written by run_explain.py),
prints AC@1, AC@3 and Avg@5 with group-bootstrap intervals for each ranker,
then exact McNemar tests on AC@1 and AC@3 for every pair, on the same cases.
With --markdown it prints the same tables as Markdown for the README.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT  # noqa: E402
from src.evaluate_explain import bootstrap_ci, mcnemar, score  # noqa: E402
from src.load_rcaeval import case_index, load_labels  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_explain import to_results  # noqa: E402

ORDER = ["baro", "nsigma", "mmbaro", "no-ai", "spark-one-shot", "spark-agent", "spark-agent-all",
         "open-one-shot", "open-agent", "open-agent-all", "glimmer-agent-all", "glimmer-tuned-agent-all",
         "spark-agent-nologs", "spark-agent-notraces", "spark-agent-noisy", "spark-agent-late"]
LABELS = {"baro": "BARO (published)", "nsigma": "N-Sigma (published)", "mmbaro": "Multi-source BARO (published)",
          "no-ai": "No-AI control", "spark-one-shot": "Spark one-shot", "spark-agent": "Spark agent (metrics)",
          "spark-agent-all": "Spark agent (metrics + logs + traces)", "open-one-shot": "Spark Open one-shot",
          "open-agent": "Spark Open agent (metrics)", "open-agent-all": "Spark Open agent (metrics + logs + traces)",
          "glimmer-agent-all": "Glimmer 30B agent (all sources)",
          "glimmer-tuned-agent-all": "Glimmer 30B fine-tuned agent (all sources)",
          "spark-agent-nologs": "Spark agent, logs removed", "spark-agent-notraces": "Spark agent, traces removed",
          "spark-agent-noisy": "Spark agent, noisy metrics", "spark-agent-late": "Spark agent, alarm 60 s late"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, choices=["RE1", "RE2", "RE3"])
    parser.add_argument("--part", default="test", choices=["train", "test"])
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--anchor", default="inject", choices=["inject", "detector"])
    args = parser.parse_args()

    index, labels = case_index(args.suite), load_labels(args.suite)
    results = {}
    for ranker in ORDER:
        suffix = "" if args.anchor == "inject" else "_detector"
        path = ROOT / "runs" / "explain" / f"{ranker}_{args.suite}_{args.part}{suffix}.jsonl"
        if path.exists():
            results[ranker] = to_results([json.loads(line) for line in open(path)], index, labels)
    if not results:
        sys.exit(f"no records for {args.suite} {args.part} in runs/explain/")

    unflagged = set()
    if args.anchor == "detector":
        no_ai = ROOT / "runs" / "explain" / f"no-ai_{args.suite}_{args.part}_detector.jsonl"
        unflagged = {r["case_id"] for r in map(json.loads, open(no_ai)) if r.get("error") == "detector never flagged"}
    samples, seed = DEFAULT.stats.bootstrap_samples, DEFAULT.stats.seed
    rows = []
    for ranker, rs in results.items():
        s, ci = score(rs), bootstrap_ci(rs, samples, seed)
        rows.append((LABELS[ranker], s, ci))
    pairs = []
    for a, b in itertools.combinations(results, 2):
        if {r.case_id for r in results[a]} == {r.case_id for r in results[b]}:
            pairs.append((LABELS[b], LABELS[a], mcnemar(results[b], results[a], 1), mcnemar(results[b], results[a], 3)))

    cases = len(next(iter(results.values())))
    if args.markdown:
        print(f"| Ranker ({args.suite} {args.part}, {cases} cases) | AC@1 | AC@3 | Avg@5 | Failed |")
        print("|---|---|---|---|---|")
        for name, s, ci in rows:
            print(f"| {name} | {s['ac1']:.3f} [{ci['ac1'][0]:.3f}, {ci['ac1'][1]:.3f}] | {s['ac3']:.3f} "
                  f"[{ci['ac3'][0]:.3f}, {ci['ac3'][1]:.3f}] | {s['avg5']:.3f} | {s['failed']} |")
        print("\n| Comparison | AC@1: wins / losses, p | AC@3: wins / losses, p |\n|---|---|---|")
        for b, a, m1, m3 in pairs:
            print(f"| {b} vs {a} | {m1['a_only']} / {m1['b_only']}, {m1['p_value']:.2g} | "
                  f"{m3['a_only']} / {m3['b_only']}, {m3['p_value']:.2g} |")
        return
    print(f"{args.suite} {args.part}, anchored at {args.anchor}: {cases} cases (95% intervals: bootstrap over fault groups)")
    print(f"{'ranker':<40}{'AC@1':>22}{'AC@3':>22}{'Avg@5':>22}{'failed':>8}")
    for name, s, ci in rows:
        print(f"{name:<40}" + "".join(f"{s[k]:>8.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" for k in ("ac1", "ac3", "avg5"))
              + f"{s['failed']:>8}")
    if args.anchor == "detector":
        print("\nWhen the detector flagged (cases it never flagged are misses for every ranker):")
        for ranker, rs in results.items():
            flagged = [r for r in rs if r.case_id not in unflagged]
            s = score(flagged)
            print(f"  {LABELS[ranker]:<40} AC@1 {s['ac1']:.3f}  AC@3 {s['ac3']:.3f}  on {s['cases']} flagged cases")
    print("\nPaired exact McNemar (cases only the first ranker gets / only the second gets, p):")
    for b, a, m1, m3 in pairs:
        print(f"  {b} vs {a}: AC@1 {m1['a_only']}/{m1['b_only']} p={m1['p_value']:.2g} | "
              f"AC@3 {m3['a_only']}/{m3['b_only']} p={m3['p_value']:.2g}")


if __name__ == "__main__":
    main()
