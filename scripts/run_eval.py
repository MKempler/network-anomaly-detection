"""Run detectors over SMD and print event-level scores.

    python scripts/run_eval.py --detector zscore
    python scripts/run_eval.py --detector zscore iforest lstm  # comparison table
    python scripts/run_eval.py --detector iforest --machines machine-1-1

Every detector is fit per server on train and scored on that server's test
split with the same scoring. With several detectors, the summary compares each
one with the first (the baseline) using a paired bootstrap over servers.

Each detector's run is appended to runs/detection.jsonl with its config
sections, config id and git commit, so every reported number traces back to
the exact code and settings that produced it. Runs from a tree with
uncommitted changes are marked dirty.
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT, DETECTION_SECTIONS  # noqa: E402
from src.detectors import IsolationForestDetector, LSTMAutoencoderDetector, ZScoreDetector  # noqa: E402
from src.evaluate_detect import bootstrap_ci, bootstrap_f1_difference, combine, score_events  # noqa: E402
from src.load_smd import MinMaxScaler, list_machines, load_machine  # noqa: E402

RUNS = ROOT / "runs" / "detection.jsonl"
DETECTORS = {
    "zscore": lambda cfg: ZScoreDetector(cfg.zscore),
    "iforest": lambda cfg: IsolationForestDetector(cfg.iforest),
    "lstm": lambda cfg: LSTMAutoencoderDetector(cfg.lstm),
}


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip())
    return commit, dirty


def counts(s) -> dict:
    return {"true_events": s.true_events, "detected_events": s.detected_events,
            "flagged_events": s.flagged_events, "correct_flags": s.correct_flags,
            "precision": round(s.precision, 4), "recall": round(s.recall, 4), "f1": round(s.f1, 4),
            "mean_delay": None if s.mean_delay != s.mean_delay else round(s.mean_delay, 2)}


def row(label, s) -> str:
    return (f"{label:<14}{s.true_events:>7}{s.detected_events:>7}{s.flagged_events:>8}"
            f"{s.precision:>8.3f}{s.recall:>8.3f}{s.f1:>8.3f}{s.mean_delay:>8.1f}")


def run_detector(name, cfg, machines):
    print(f"\n{name}  config={cfg.section_id(*DETECTION_SECTIONS, name)}")
    print(f"{'machine':<14}{'events':>7}{'found':>7}{'flags':>8}{'prec':>8}{'recall':>8}{'f1':>8}{'delay':>8}")
    started, per_machine = time.time(), {}
    for machine in machines:
        m = load_machine(machine)
        scaler = MinMaxScaler().fit(m.train)
        detector = DETECTORS[name](cfg).fit(scaler.transform(m.train))
        flags = detector.score(scaler.transform(m.test)) > detector.threshold
        per_machine[machine] = score_events(m.labels, flags, cfg.scoring.merge_gap)
        print(row(machine, per_machine[machine]))
    print(row("ALL (pooled)", combine(list(per_machine.values()))))
    print(f"{time.time() - started:.1f}s")
    return per_machine


def record(name, cfg, per_machine, ci, commit, dirty) -> str:
    sections = (*DETECTION_SECTIONS, name)
    entry = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty,
        "dataset": "SMD", "detector": name,
        "config_id": cfg.section_id(*sections), "config": cfg.sections(*sections),
        "total": counts(combine(list(per_machine.values()))),
        "ci95": {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in ci.items()},
        "machines": {machine: counts(s) for machine, s in per_machine.items()},
    }
    RUNS.parent.mkdir(exist_ok=True)
    with open(RUNS, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["run_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", nargs="+", required=True, choices=sorted(DETECTORS),
                        help="one or more; the first is the baseline in the comparison")
    parser.add_argument("--machines", nargs="*", help="default: all 28")
    parser.add_argument("--no-record", action="store_true", help="print only, do not append to runs/")
    args = parser.parse_args()

    cfg, machines = DEFAULT, args.machines or list_machines()
    commit, dirty = git_state()
    print(f"commit={commit}{' (dirty)' if dirty else ''}  machines={len(machines)}")

    results = {name: run_detector(name, cfg, machines) for name in args.detector}
    samples, seed = cfg.stats.bootstrap_samples, cfg.stats.seed
    cis = {name: bootstrap_ci(list(pm.values()), samples, seed) for name, pm in results.items()}

    baseline = args.detector[0]
    print(f"\nSummary, pooled over {len(machines)} servers (95% bootstrap intervals over servers)")
    print(f"{'detector':<10}{'precision':>22}{'recall':>22}{'f1':>22}{'delay':>8}{'  F1 vs ' + baseline:>26}")
    for name, per_machine in results.items():
        total, ci = combine(list(per_machine.values())), cis[name]
        cells = "".join(f"{getattr(total, k):>8.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" for k in ("precision", "recall", "f1"))
        if name == baseline:
            versus = "baseline"
        else:
            lo, hi = bootstrap_f1_difference(list(per_machine.values()), list(results[baseline].values()), samples, seed)
            versus = f"{total.f1 - combine(list(results[baseline].values())).f1:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(f"{name:<10}{cells}{total.mean_delay:>8.1f}{versus:>26}")

    if not args.no_record:
        for name, per_machine in results.items():
            run_id = record(name, cfg, per_machine, cis[name], commit, dirty)
            print(f"recorded {name} run {run_id} -> {RUNS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
