"""Build replay bundles: small JSON files the console plays back without raw data.

    python scripts/build_replays.py --suite RE2
    python scripts/build_replays.py --suite RE1 --ranker spark-agent

For every test case a ranker was scored on, a bundle holds the metric series
the investigation is about (the strongest deviations plus every indicator the
ranker named), the agent's recorded trail and ranked causes, what the no-AI
control and N-Sigma put first, and the benchmark's answer for the console's
reveal step. Times are relative to the anchor. Bundles go to
runs/replay/<suite>/<case_id>.json, with an index.json per suite.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT  # noqa: E402
from src.evaluate_explain import hits, ranked_services  # noqa: E402
from src.load_rcaeval import SYSTEM_NAMES, case_index, load_case, load_labels  # noqa: E402
from src.tools import Investigation, sig  # noqa: E402

DEFAULT_RANKER = {"RE1": "spark-agent", "RE2": "spark-agent-all", "RE3": "spark-agent-all"}
CHART_METRICS = 6     # strongest deviations shown, plus the named indicators
POINTS = 240          # points per series across baseline + window


def records(ranker, suite):
    path = ROOT / "runs" / "explain" / f"{ranker}_{suite}_test.jsonl"
    return {r["case_id"]: r for r in map(json.loads, open(path))} if path.exists() else {}


def series(inv: Investigation, metric: str) -> list:
    s = inv.case.metrics[metric]
    rel = s.index.to_numpy(dtype=float) - inv.anchor
    s = s[(rel >= -inv.cfg.baseline_s) & (rel < inv.cfg.window_s)]
    step = max(1, len(s) // POINTS)
    return [[int(round(t - inv.anchor)), sig(float(v))] for t, v in zip(s.index[::step], s.to_numpy()[::step])]


def build(suite: str, ranker: str) -> list[dict]:
    index, labels = case_index(suite), load_labels(suite)
    main, control, nsigma = records(ranker, suite), records("no-ai", suite), records("nsigma", suite)
    entries = []
    out_dir = ROOT / "runs" / "replay" / suite
    out_dir.mkdir(parents=True, exist_ok=True)
    for case_id, rec in main.items():
        lab = labels[case_id]
        case = load_case(suite, case_id, index[case_id])
        inv = Investigation(case, lab.inject_time, DEFAULT.investigation)
        top = inv.metric_deviations().head(CHART_METRICS)["metric"].tolist()
        named = [c["indicator"] for c in rec["causes"] if c.get("indicator") in case.services]
        metrics = list(dict.fromkeys(top + named))
        h = hits(rec["services"], lab.root_service)
        bundle = {
            "case_id": case_id, "suite": suite, "system": case.system, "system_name": SYSTEM_NAMES[case.system],
            "sources": sorted(case.sources), "ranker": ranker,
            "window": {"baseline_s": inv.cfg.baseline_s, "window_s": inv.cfg.window_s},
            "series": [{"metric": m, "service": case.services[m], "points": series(inv, m)} for m in metrics],
            "trail": rec["trail"], "causes": rec["causes"], "error": rec["error"],
            "turns": rec["turns"], "usage": rec["usage"],
            "others": {
                "no-ai": ranked_services(control[case_id]["services"] or [])[:5] if case_id in control else None,
                "nsigma": ranked_services(nsigma[case_id]["services"] or [])[:5] if case_id in nsigma else None,
            },
            "answer": {"service": lab.root_service, "fault": lab.fault},
            "outcome": {"top1": bool(h[0]), "top3": bool(h[2])},
        }
        (out_dir / f"{case_id}.json").write_text(json.dumps(bundle, separators=(",", ":")))
        entries.append({"case_id": case_id, "system": case.system, "fault": lab.fault,
                        "turns": rec["turns"], "top1": bundle["outcome"]["top1"], "tool_calls":
                        sum(1 for s in rec["trail"] if s["type"] == "tool")})
    (out_dir / "index.json").write_text(json.dumps({"suite": suite, "ranker": ranker, "cases": entries}, indent=1))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, choices=["RE1", "RE2", "RE3"])
    parser.add_argument("--ranker", help="default: the strongest recorded agent for the suite")
    args = parser.parse_args()
    ranker = args.ranker or DEFAULT_RANKER[args.suite]
    entries = build(args.suite, ranker)
    size = sum(p.stat().st_size for p in (ROOT / "runs" / "replay" / args.suite).glob("*.json"))
    print(f"{args.suite}: {len(entries)} bundles from {ranker}, {size / 1e6:.1f} MB, "
          f"{sum(e['top1'] for e in entries)} with the true cause ranked first")


if __name__ == "__main__":
    main()
