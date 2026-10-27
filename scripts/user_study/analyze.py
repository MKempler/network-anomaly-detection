"""Summarise the user study.

    python scripts/user_study/analyze.py

Answers are scored against the benchmark's labels read locally (no answer key
is committed). With three to five participants this is reported as raw
per-person numbers and medians: direction, not statistical significance.
Each participant did three outages with Epicenter and three without, so the
per-person difference is the most informative figure.
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluate_explain import service_of_item  # noqa: E402
from src.load_rcaeval import load_labels  # noqa: E402

STUDY = ROOT / "runs" / "study"


def load(name):
    path = STUDY / name
    return [json.loads(line) for line in open(path)] if path.exists() else []


def main():
    plan = json.loads((STUDY / "plan.json").read_text())
    truth = {c: service_of_item(l.root_service) for c, l in load_labels(plan["suite"]).items() if c in plan["cases"]}
    answers = {}
    for a in load("answers.jsonl"):                 # the last answer per participant and case counts
        answers[(a["participant"], a["case_id"])] = a
    rows = list(answers.values())
    for r in rows:
        r["correct"] = service_of_item(r["answer"]) == truth[r["case_id"]]
    if not rows:
        sys.exit("no answers recorded yet")

    def summary(subset):
        times = [r["seconds"] for r in subset]
        return (f"{sum(r['correct'] for r in subset)}/{len(subset)} correct, "
                f"median {statistics.median(times):.0f} s, mean confidence "
                f"{statistics.mean(r['confidence'] for r in subset):.1f}") if subset else "–"

    with_tool, without = [r for r in rows if r["tool"]], [r for r in rows if not r["tool"]]
    print(f"{len({r['participant'] for r in rows})} participants, {len(rows)} answers")
    print(f"  with Epicenter: {summary(with_tool)}")
    print(f"  charts only:    {summary(without)}")
    print("\nper participant (median seconds, correct):")
    diffs = []
    for p in sorted({r["participant"] for r in rows}):
        mine_t = [r for r in with_tool if r["participant"] == p]
        mine_n = [r for r in without if r["participant"] == p]
        if mine_t and mine_n:
            mt, mn = statistics.median(r["seconds"] for r in mine_t), statistics.median(r["seconds"] for r in mine_n)
            diffs.append(mn - mt)
            print(f"  {p}: with {mt:.0f} s ({sum(r['correct'] for r in mine_t)}/{len(mine_t)}) · "
                  f"without {mn:.0f} s ({sum(r['correct'] for r in mine_n)}/{len(mine_n)}) · saved {mn - mt:+.0f} s")
    if diffs:
        print(f"  median time saved per person: {statistics.median(diffs):+.0f} s; faster with the tool: "
              f"{sum(d > 0 for d in diffs)}/{len(diffs)}")
    exits = {e["participant"]: e for e in load("exit.jsonl")}
    if exits:
        print(f"\nwould use (1-5): {sorted(e['would_use'] for e in exits.values())}")
        for e in exits.values():
            if e["quote_ok"] and e["comment"].strip():
                print(f"  “{e['comment'].strip()}” — {e['display_name'] or 'participant'}")


if __name__ == "__main__":
    main()
