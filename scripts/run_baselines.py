"""Run published RCA methods with RCAEval's own code on our exact cases.

Runs in the separate baselines environment (RCAEval pins older dependencies,
and needs Python 3.10, 3.12 or 3.14 to import its methods lazily):

    python3.14 -m venv venv-baselines
    venv-baselines/bin/pip install RCAEval==1.8.1 numpy pandas scikit-learn tqdm requests
    venv-baselines/bin/python scripts/run_baselines.py --suite RE1 --method baro nsigma
    venv-baselines/bin/python scripts/run_baselines.py --suite RE2 --method baro nsigma mmbaro

Each case is prepared exactly as RCAEval 1.8.1's main.py prepares it:
- read data.csv (RE1) or simple_metrics.csv (RE2/RE3)
- drop the duplicate time column and p50 latency, set infinities to
  missing, forward-fill, then fill what's left with 0
- keep the last 600 rows before the inject time and the first 600 after
  (`--length 20`, the default)
- rename p90 latency to "latency"
- for multi-source methods, window logts and tracets to the 40 rows (15 s
  bins) either side of the inject time

The inject time is given to these methods, as in RCAEval's protocol. Ranked
outputs go to runs/baselines/<method>_<suite>.jsonl under opaque case ids, and
src/evaluate_explain.py scores them with the same scorer as every other ranker.
"""

import argparse
import json
import subprocess
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.load_rcaeval import METRICS_FILE, case_index  # noqa: E402

LENGTH_MIN = 20                      # RCAEval main.py default
METRIC_ROWS = LENGTH_MIN * 60 // 2   # rows on each side of the inject time
SERIES_ROWS = LENGTH_MIN * 4 // 2    # logts/tracets rows (15 s bins) on each side
MULTI_SOURCE = {"mmbaro"}
KEEP_RANKS = 50


def window(df: pd.DataFrame, inject_time: int, rows: int) -> pd.DataFrame:
    before = df[df["time"] < inject_time].tail(rows)
    after = df[df["time"] >= inject_time].head(rows)
    return pd.concat([before, after], ignore_index=True)


def prepare(case_dir: Path, suite: str, method: str):
    data = pd.read_csv(case_dir / METRICS_FILE[suite])
    if "time.1" in data:
        data = data.drop(columns=["time.1"])
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    inject_time = int((case_dir / "inject_time.txt").read_text().splitlines()[0].strip())
    data = window(data, inject_time, METRIC_ROWS)
    n_iter = len(data.columns) - 1
    data = data.rename(columns={c: c.replace("_latency-90", "_latency") for c in data.columns
                                if c.endswith("_latency-90")})
    if method in MULTI_SOURCE:
        def series(name):
            path = case_dir / name
            return window(pd.read_csv(path), inject_time, SERIES_ROWS) if path.exists() else pd.DataFrame()
        data = {"metric": data.copy(deep=True), "logts": series("logts.csv"),
                "tracets_err": series("tracets_err.csv"), "tracets_lat": series("tracets_lat.csv")}
    return data, inject_time, n_iter


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, choices=sorted(METRICS_FILE))
    parser.add_argument("--method", nargs="+", required=True, choices=["baro", "nsigma", "mmbaro"])
    args = parser.parse_args()

    from RCAEval.e2e import baro, mmbaro, nsigma  # imported here: only the baselines venv has RCAEval
    methods = {"baro": baro, "nsigma": nsigma, "mmbaro": mmbaro}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    index = case_index(args.suite)
    out_dir = ROOT / "runs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.method:
        if name in MULTI_SOURCE and args.suite == "RE1":
            print(f"{name}: RE1 has no logs or traces, skipped")
            continue
        out = out_dir / f"{name}_{args.suite}.jsonl"
        started, failures = time.time(), 0
        with open(out, "w") as f:
            for case_id, case_dir in index.items():
                data, inject_time, n_iter = prepare(case_dir, args.suite, name)
                record = {"case_id": case_id, "method": name, "suite": args.suite,
                          "rcaeval_version": version("RCAEval"), "length_min": LENGTH_MIN, "git_commit": commit}
                try:
                    t = time.time()
                    out_ranks = methods[name](data, inject_time, dataset=args.suite.lower(), anomalies=None,
                                              dk_select_useful=False, sli=None, verbose=False, n_iter=n_iter)
                    record["ranks"] = list(out_ranks["ranks"][:KEEP_RANKS])
                    record["seconds"] = round(time.time() - t, 3)
                except Exception as error:  # recorded, never silently dropped
                    failures += 1
                    record["error"] = f"{type(error).__name__}: {error}"
                f.write(json.dumps(record) + "\n")
        print(f"{name} on {args.suite}: {len(index)} cases, {failures} failed, "
              f"{time.time() - started:.0f}s -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
