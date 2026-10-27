"""Build the user study: case selection, counterbalanced assignment, study bundles.

    python scripts/user_study/build_study.py

Six held-out RE2 test outages (two per system) are drawn with a fixed seed, so
the selection is reproducible and not hand-picked; it includes whatever the
agent got wrong. Participants see the same eight strongest-deviating metrics in
both conditions. Agent-named indicators are deliberately left out, so the
charts never hint at the agent's answer. "with tool" adds the agent's recorded
trail and verdict.

Assignment: slot s sees all six cases in a slot-specific order, and case k
gets the tool when k + s is even. Each participant does three with the tool and
three without, and every case is seen both ways across slots (both are
asserted).

Writes runs/study/plan.json and runs/study/cases/<case_id>.json, and no answer key.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT  # noqa: E402
from src.load_rcaeval import SYSTEM_NAMES, case_index, load_case, load_labels  # noqa: E402
from src.splits import case_ids  # noqa: E402
from src.tools import Investigation, sig  # noqa: E402

SEED, SUITE, PER_SYSTEM, SLOTS, METRICS, POINTS = 20261101, "RE2", 2, 6, 8, 240
OUT = ROOT / "runs" / "study"


def main():
    index, labels = case_index(SUITE), load_labels(SUITE)
    recorded = {r["case_id"]: r for r in map(json.loads, open(ROOT / "runs" / "explain" / f"spark-agent-all_{SUITE}_test.jsonl"))}
    rng = random.Random(SEED)
    chosen = []
    for system in ("OB", "SS", "TT"):
        pool = sorted(c for c in case_ids(SUITE, "test") if index[c].parent.parent.name.endswith(system) and c in recorded)
        chosen += rng.sample(pool, PER_SYSTEM)
    rng.shuffle(chosen)

    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    for cid in chosen:
        case = load_case(SUITE, cid, index[cid])
        inv = Investigation(case, labels[cid].inject_time, DEFAULT.investigation)
        series = []
        for metric in inv.metric_deviations().head(METRICS)["metric"]:
            s = case.metrics[metric]
            rel = s.index.to_numpy(dtype=float) - inv.anchor
            s = s[(rel >= -inv.cfg.baseline_s) & (rel < inv.cfg.window_s)]
            step = max(1, len(s) // POINTS)
            series.append({"metric": metric, "service": case.services[metric],
                           "points": [[int(round(t - inv.anchor)), sig(float(v))] for t, v in zip(s.index[::step], s.to_numpy()[::step])]})
        rec = recorded[cid]
        (OUT / "cases" / f"{cid}.json").write_text(json.dumps({
            "case_id": cid, "system_name": SYSTEM_NAMES[case.system], "candidates": case.candidate_services,
            "window": {"baseline_s": inv.cfg.baseline_s, "window_s": inv.cfg.window_s}, "series": series,
            "trail": rec["trail"], "causes": rec["causes"]}, separators=(",", ":")))

    # Condition follows the case, not its position: case k gets the tool in slot s when k + s is even.
    # (Deriving it from position after rotating the order cancels out and pins the same cases to the tool.)
    slots = []
    for s in range(SLOTS):
        order = chosen[s % len(chosen):] + chosen[:s % len(chosen)]
        slots.append([{"case_id": cid, "tool": (chosen.index(cid) + s) % 2 == 0} for cid in order])
    for k, cid in enumerate(chosen):
        conditions = {next(x["tool"] for x in slot if x["case_id"] == cid) for slot in slots}
        assert conditions == {True, False}, f"{cid} is not seen in both conditions"
    assert all(sum(x["tool"] for x in slot) == len(chosen) // 2 for slot in slots)
    # No answer key is written: participants may browse this public repo. The analysis reads the
    # benchmark labels locally instead.
    (OUT / "plan.json").write_text(json.dumps({"seed": SEED, "suite": SUITE, "cases": chosen, "slots": slots}, indent=1))
    for cid in chosen:
        print(cid, labels[cid].root_service, labels[cid].fault, "| agent right:", recorded[cid]["services"][0] == labels[cid].root_service if recorded[cid]["services"] else None)
    for s, slot in enumerate(slots):
        print(f"slot {s}: " + " ".join(f"{chosen.index(x['case_id'])}{'T' if x['tool'] else '-'}" for x in slot))


if __name__ == "__main__":
    main()
