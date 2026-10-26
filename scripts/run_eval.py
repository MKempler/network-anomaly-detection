"""Run a detector over SMD and print event-level scores.

    python scripts/run_eval.py --detector zscore
    python scripts/run_eval.py --detector zscore --machines machine-1-1

Each run is appended to runs/detection.jsonl with its config, config id and git
commit, so every reported number traces back to the exact code and settings
that produced it. Runs from a tree with uncommitted changes are marked dirty.
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

from src.config import DEFAULT  # noqa: E402
from src.detectors import ZScoreDetector  # noqa: E402
from src.evaluate_detect import combine, score_events  # noqa: E402
from src.load_smd import MinMaxScaler, list_machines, load_machine  # noqa: E402

RUNS = ROOT / "runs" / "detection.jsonl"
DETECTORS = {"zscore": lambda cfg: ZScoreDetector(cfg.zscore)}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", required=True, choices=sorted(DETECTORS))
    parser.add_argument("--machines", nargs="*", help="default: all 28")
    parser.add_argument("--no-record", action="store_true", help="print only, do not append to runs/")
    args = parser.parse_args()

    cfg = DEFAULT
    machines = args.machines or list_machines()
    commit, dirty = git_state()
    print(f"detector={args.detector} config={cfg.config_id} commit={commit}{' (dirty)' if dirty else ''}")
    print(f"{'machine':<14}{'events':>7}{'found':>7}{'flags':>8}{'prec':>8}{'recall':>8}{'f1':>8}{'delay':>8}")

    started = time.time()
    per_machine = {}
    for name in machines:
        m = load_machine(name)
        scaler = MinMaxScaler().fit(m.train)
        detector = DETECTORS[args.detector](cfg).fit(scaler.transform(m.train))
        flags = detector.score(scaler.transform(m.test)) > detector.threshold
        s = score_events(m.labels, flags, cfg.scoring.merge_gap)
        per_machine[name] = s
        print(f"{name:<14}{s.true_events:>7}{s.detected_events:>7}{s.flagged_events:>8}"
              f"{s.precision:>8.3f}{s.recall:>8.3f}{s.f1:>8.3f}{s.mean_delay:>8.1f}")

    total = combine(list(per_machine.values()))
    print(f"{'ALL (pooled)':<14}{total.true_events:>7}{total.detected_events:>7}{total.flagged_events:>8}"
          f"{total.precision:>8.3f}{total.recall:>8.3f}{total.f1:>8.3f}{total.mean_delay:>8.1f}")
    print(f"{time.time() - started:.1f}s")

    if args.no_record:
        return
    record = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty,
        "dataset": "SMD", "detector": args.detector,
        "config_id": cfg.config_id, "config": cfg.to_dict(),
        "total": counts(total),
        "machines": {name: counts(s) for name, s in per_machine.items()},
    }
    RUNS.parent.mkdir(exist_ok=True)
    with open(RUNS, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"recorded run {record['run_id']} -> {RUNS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
