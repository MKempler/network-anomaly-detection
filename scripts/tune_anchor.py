"""Choose the log signal for end-to-end mode on train cases only (eval-spec-v1.4).

    python scripts/tune_anchor.py

Our detector anchors an end-to-end investigation (src/anchor.py). Version 1
watches metrics only and never flags most of RE3's code-level faults. This adds
a second signal, each service's error-log lines. The combined anchor is
whichever signal flags first; RE1 has no logs, so only its metric anchor counts.

The settings are chosen by two rules written before any test run:
1. The objective version 1 was chosen by: the most combined anchors within
   0-120 s after the fault starts, ties to fewer early alarms.
2. If no setting beats the metric-only detector on that objective (on train,
   none does: RE3's nine train cases have no Train Ticket outages and only two
   fault types, unlike the RE3 outages it misses), take the setting that does
   no harm (scores exactly as metric-only) and whose log signal alone anchors
   the most train cases within 0-120 s, then the fewest early. Ties go to the
   most conservative setting: larger window, higher threshold, longer persistence,
   larger minimum count.
So the frozen setting is the one shown to be harmless and most able to fire on
its own, not one shown to gain; that is how it is reported.

Only train cases are read (data_splits.json); test cases are never loaded.
Per-second error counts are cached under data/cache/anchor_logs/. The grid,
the metric-only baseline and the winner go to runs/anchor_tuning.jsonl.
"""

import itertools
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anchor import detector_anchor, error_counts, log_flag, log_split  # noqa: E402
from src.config import DEFAULT  # noqa: E402
from src.load_rcaeval import case_index, load_case, load_labels, load_logs  # noqa: E402
from src.provenance import git_state  # noqa: E402
from src.splits import case_ids  # noqa: E402

SUITES = ("RE1", "RE2", "RE3")
CACHE = ROOT / "data" / "cache" / "anchor_logs"
GRID = {"window_s": (10, 30, 60), "threshold_sigma": (3.0, 5.0, 10.0, 20.0, 30.0),
        "persist_s": (10, 30, 60, 120), "min_count": (3, 10, 30)}


def train_case(suite: str, case_id: str, case_dir: Path, inject: int) -> dict:
    """The metric anchor and, where the case has logs, its cached per-second error counts."""
    case = load_case(suite, case_id, case_dir)
    times = case.metrics.index.to_numpy()
    out = {"case_id": case_id, "suite": suite, "inject": inject,
           "metric_anchor": detector_anchor(case, DEFAULT.anchor), "counts": None}
    if "logs" not in case.sources:
        return out
    path = CACHE / f"{case_id}.npz"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        grid, counts, _ = error_counts(load_logs(case_id), times[0], times[-1])
        np.savez_compressed(path, times=grid, counts=counts)
    data = np.load(path)
    out.update(counts=data["counts"], times=data["times"], split=log_split(times, data["times"], DEFAULT.anchor))
    return out


def classify(anchor: float | None, inject: int) -> str:
    if anchor is None:
        return "never"
    offset = anchor - inject
    return "early" if offset < 0 else "within_120s" if offset <= 120 else "late"


def score(cases: list[dict], params: dict | None) -> dict:
    tally = {suite: {"never": 0, "early": 0, "within_120s": 0, "late": 0} for suite in SUITES}
    for c in cases:
        anchor = c["metric_anchor"]
        if params is not None and c["counts"] is not None:
            flag = log_flag(c["counts"], c["times"], c["split"], **params)
            anchor = flag if anchor is None else anchor if flag is None else min(anchor, flag)
        tally[c["suite"]][classify(anchor, c["inject"])] += 1
    tally["all"] = {k: sum(tally[s][k] for s in SUITES) for k in tally["RE1"]}
    return tally


def main():
    commit, dirty = git_state()
    cases = []
    for suite in SUITES:
        labels, index = load_labels(suite), case_index(suite)
        ids = case_ids(suite, "train")
        for i, cid in enumerate(ids):
            cases.append(train_case(suite, cid, index[cid], labels[cid].inject_time))
            print(f"\r{suite} {i + 1}/{len(ids)}", end="", flush=True)
        print()

    baseline = score(cases, None)
    results = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        results.append({"params": params, "tally": score(cases, params)})
    rank = lambda r: (-r["tally"]["all"]["within_120s"], r["tally"]["all"]["early"])  # noqa: E731
    results.sort(key=rank)
    if rank(results[0]) < rank({"tally": baseline}):
        best, rule = results[0], "1: beats metric-only"
    else:
        logged = [c for c in cases if c["counts"] is not None]
        for r in results:
            if rank(r) == rank({"tally": baseline}):
                own = [classify(log_flag(c["counts"], c["times"], c["split"], **r["params"]), c["inject"]) for c in logged]
                r["logs_alone"] = {k: own.count(k) for k in ("within_120s", "early", "late", "never")}
        harmless = [r for r in results if "logs_alone" in r]
        conservative = lambda p: (p["window_s"], p["threshold_sigma"], p["persist_s"], p["min_count"])  # noqa: E731
        harmless.sort(key=lambda r: (-r["logs_alone"]["within_120s"], r["logs_alone"]["early"],
                                     tuple(-v for v in conservative(r["params"]))))
        best, rule = harmless[0], f"2: no setting beats metric-only; harmless ({len(harmless)} settings), most on-time flags from logs alone"

    def line(t):
        return " | ".join(f"{s} within {t[s]['within_120s']} early {t[s]['early']} late {t[s]['late']} never {t[s]['never']}"
                          for s in (*SUITES, "all"))
    print(f"{len(cases)} train cases (commit {commit}{' dirty' if dirty else ''})")
    print(f"metric only (v1): {line(baseline)}")
    for r in results[:10]:
        print(f"{r['params']}: {line(r['tally'])}")
    print(f"chosen by rule {rule}: {best['params']}" + (f", logs alone {best['logs_alone']}" if "logs_alone" in best else ""))

    record = {"run_id": uuid.uuid4().hex[:12], "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "git_commit": commit, "git_dirty": dirty, "part": "train", "cases": len(cases),
              "objective": "most anchors within 0-120 s after the fault, ties to fewer early alarms",
              "grid": GRID, "baseline": baseline, "rule": rule, "best": best, "top": results[:10]}
    with open(ROOT / "runs" / "anchor_tuning.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"recorded run {record['run_id']}")


if __name__ == "__main__":
    main()
