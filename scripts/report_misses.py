"""Where hosted Spark's wrong answers come from: detection or diagnosis, and victims.

    python scripts/report_misses.py

Reads recorded runs only; no model is called.
- Diagnosis misses (investigation started at the fault time): for each wrong first
  answer, the agent's own first pick is placed relative to the true cause in the call
  graph recorded from that outage's traces (src/evaluate_explain.py, graph_relation):
  an upstream victim, downstream of the cause, unconnected, or unknown without traces.
  A first pick that was right but whose evidence the verifier rejected is counted apart.
- End to end (our detector chose where to look): misses split into outages the
  detector never flagged and outages it flagged but the agent got wrong, for the
  metric-only detector (eval-spec-v1) and the one that also watches error logs
  (eval-spec-v1.4), with an exact McNemar test of the two on AC@1.
A summary line goes to runs/miss_summary.jsonl.
"""

import json
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate_explain import graph_relation, hits, mcnemar, score, service_of_item  # noqa: E402
from src.load_rcaeval import case_index, load_labels  # noqa: E402
from src.provenance import git_state  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_explain import to_results  # noqa: E402

SUITES = ("RE1", "RE2", "RE3")
HOSTED = {"RE1": "spark-agent", "RE2": "spark-agent-all", "RE3": "spark-agent-all"}   # RE1 has metrics only
RELATIONS = ("upstream victim", "downstream of the cause", "unconnected", "no traces")
DETECTORS = {"v1": "_detector", "v1.4": "_detector_logs"}   # metrics only; metrics and error logs


def rows(name: str) -> list[dict]:
    path = ROOT / "runs" / "explain" / f"{name}.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else []


def main():
    commit, dirty = git_state()
    diagnosis, end_to_end = {}, {}
    for suite in SUITES:
        labels = load_labels(suite)
        tally = Counter()
        for r in rows(f"{HOSTED[suite]}_{suite}_test"):
            truth = labels[r["case_id"]].root_service
            if r["services"] is not None and hits(r["services"], truth)[0]:
                continue
            first = (r["causes"] or [{}])[0].get("service")
            if first is None:
                tally["no ranking"] += 1
            elif service_of_item(first) == service_of_item(truth):
                tally["right service, evidence rejected"] += 1
            else:
                bundle = json.loads((ROOT / "runs" / "replay" / suite / f"{r['case_id']}.json").read_text())
                tally[graph_relation(bundle.get("graph") or [], first, truth)] += 1
        diagnosis[suite] = dict(tally)

        end_to_end[suite], scored = {}, {}
        for version, suffix in DETECTORS.items():
            flagged = rows(f"spark-agent-all_{suite}_test{suffix}")
            if not flagged:
                continue
            never = sum(r.get("error") == "detector never flagged" for r in flagged)
            wrong = sum(r.get("error") != "detector never flagged" and not (
                r["services"] is not None and hits(r["services"], labels[r["case_id"]].root_service)[0]) for r in flagged)
            scored[version] = to_results(flagged, case_index(suite), labels)
            end_to_end[suite][version] = {"outages": len(flagged), "detector_never_flagged": never,
                                          "flagged_but_wrong": wrong, "ac1": score(scored[version])["ac1"]}
        if len(scored) == 2:
            end_to_end[suite]["v1.4_vs_v1"] = mcnemar(scored["v1.4"], scored["v1"], 1)

    total = Counter()
    for t in diagnosis.values():
        total.update(t)
    print(f"Hosted Spark's wrong first answers, investigation started at the fault time (commit {commit}"
          f"{' dirty' if dirty else ''}):")
    for suite, t in diagnosis.items():
        print(f"  {suite}: {sum(t.values())} wrong: " + ", ".join(f"{k} {v}" for k, v in sorted(t.items())))
    print(f"  all: {sum(total.values())} wrong: " + ", ".join(f"{k} {v}" for k, v in sorted(total.items())))
    print("\nEnd to end, our detector choosing where to look (hosted Spark, all sources):")
    for suite, e in end_to_end.items():
        for version in DETECTORS:
            if version in e:
                v = e[version]
                print(f"  {suite} {version}: {v['outages']} outages; detector never flagged {v['detector_never_flagged']}, "
                      f"flagged but diagnosed wrong {v['flagged_but_wrong']}; AC@1 {v['ac1']:.3f}")
        if "v1.4_vs_v1" in e:
            m = e["v1.4_vs_v1"]
            print(f"  {suite} v1.4 vs v1 (exact McNemar, AC@1): {m['a_only']} right only with v1.4, "
                  f"{m['b_only']} only with v1, p = {m['p_value']:.2g}")

    line = {"run_id": uuid.uuid4().hex[:12], "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit, "git_dirty": dirty, "diagnosis_misses": diagnosis, "end_to_end": end_to_end}
    with open(ROOT / "runs" / "miss_summary.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")
    print(f"\nrecorded run {line['run_id']}")


if __name__ == "__main__":
    main()
