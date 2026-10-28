"""Run one root-cause ranker over RCAEval cases and score it.

    python scripts/run_explain.py --suite RE1 --part test --ranker no-ai
    python scripts/run_explain.py --suite RE1 --part test --ranker spark-agent --workers 24
    python scripts/run_explain.py --suite RE2 --part test --ranker spark-agent-all --workers 8
    python scripts/run_explain.py --suite RE1 --part test --ranker baro          # published baseline
    python scripts/run_explain.py --suite RE1 --part test --ranker spark-agent --offline  # replay only

Rankers:
- no-ai: services by strongest metric deviation (the control)
- spark-one-shot: one prompt with the same top deviations
- spark-agent: tool-using investigation, metrics only
- spark-agent-all: tool-using investigation with logs and traces where the case has them
- open-one-shot, open-agent, open-agent-all: the same three on Muse Spark Open, Meta's
  open-weight model, served by Baseten; only the model and provider change
- spark-agent-nologs, spark-agent-notraces, spark-agent-noisy, spark-agent-late: the all-sources
  hosted agent with one natural degradation each (src/robustness.py): logs removed, traces removed,
  metrics with noise, or the investigation starting a minute after the fault
- glimmer-agent-all, glimmer-tuned-agent-all: the all-sources agent on Muse Glimmer
  (30B, open weights) before and after fine-tuning on hosted Spark's verified
  train-split investigations (scripts/finetune/)
- baro, nsigma, mmbaro: published methods, from runs/baselines/ (scripts/run_baselines.py)

By default the investigation is anchored at the benchmark's inject time, which
is how RCAEval gives it to every published method. With --anchor detector, our
own detector (src/anchor.py) picks the start instead, and a case it never flags
is a recorded miss for every ranker; those runs write *_detector files. Model calls go through the
record-and-replay cache at runs/llm/<ranker>_<suite>.jsonl.gz; --offline
replays it with no network access. Per-case rankings, verification and agent
trails go to runs/explain/, and a summary line goes to runs/explain_summary.jsonl.
"""

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import investigate  # noqa: E402
from src.anchor import detector_anchor  # noqa: E402
from src.cache import ModelCache  # noqa: E402
from src.config import DEFAULT  # noqa: E402
from src.evaluate_explain import (CaseResult, bootstrap_ci, breakdown, calibration, hits, mcnemar,  # noqa: E402
                                  score)
from src.explain import no_ai_ranking, spark_one_shot  # noqa: E402
from src.llm import PROVIDERS, client  # noqa: E402
from src.load_rcaeval import case_index, load_case, load_labels  # noqa: E402
from src.provenance import git_state  # noqa: E402
from src.robustness import LATE_ALARM_S, add_noise  # noqa: E402
from src.splits import case_ids  # noqa: E402
from src.tools import Investigation  # noqa: E402

BASELINES = {"baro", "nsigma", "mmbaro"}
# ranker -> (method, provider, sources the agent may use)
LLM_RANKERS = {
    "spark-one-shot": ("one-shot", "meta", None),
    "spark-agent": ("agent", "meta", frozenset({"metrics"})),
    "spark-agent-all": ("agent", "meta", frozenset({"metrics", "logs", "traces"})),
    "open-one-shot": ("one-shot", "baseten", None),
    "open-agent": ("agent", "baseten", frozenset({"metrics"})),
    "open-agent-all": ("agent", "baseten", frozenset({"metrics", "logs", "traces"})),
    # robustness (eval-spec-v1.3): hosted Spark, all sources, with one natural degradation each
    "spark-agent-nologs": ("agent", "meta", frozenset({"metrics", "traces"})),
    "spark-agent-notraces": ("agent", "meta", frozenset({"metrics", "logs"})),
    "spark-agent-noisy": ("agent", "meta", frozenset({"metrics", "logs", "traces"})),
    "spark-agent-late": ("agent", "meta", frozenset({"metrics", "logs", "traces"})),
    "glimmer-agent-all": ("agent", "glimmer", frozenset({"metrics", "logs", "traces"})),
    "glimmer-tuned-agent-all": ("agent", "glimmer-tuned", frozenset({"metrics", "logs", "traces"})),
}
RANKERS = sorted(BASELINES | set(LLM_RANKERS) | {"no-ai"})


def run_case(ranker, suite, case_id, case_dir, anchor, cache, model) -> dict:
    """Rank one case. Sees the anchor time, never the labels."""
    if anchor is None:
        return {"case_id": case_id, "services": None, "causes": [], "error": "detector never flagged",
                "usage": {}, "turns": None, "seconds": 0.0, "trail": []}
    case = load_case(suite, case_id, case_dir)
    if ranker == "spark-agent-noisy":
        case = add_noise(case, anchor, DEFAULT.investigation.baseline_s)
    if ranker == "spark-agent-late":
        anchor = anchor + LATE_ALARM_S
    inv = Investigation(case, anchor, DEFAULT.investigation)
    started = time.time()
    trail, turns = [], None
    if ranker == "no-ai":
        ranking = no_ai_ranking(inv)
    elif LLM_RANKERS[ranker][0] == "one-shot":
        ranking = spark_one_shot(inv, cache, model)
        ranking.ranker = ranker
    else:
        sources = LLM_RANKERS[ranker][2]
        run = investigate(inv, cache, model, sources, DEFAULT.investigation.max_steps, ranker)
        ranking, trail, turns = run.ranking, run.trail, run.turns
    return {"case_id": case_id, "services": None if ranking.error else ranking.services,
            "causes": ranking.causes, "error": ranking.error, "usage": ranking.usage, "turns": turns,
            "seconds": round(time.time() - started, 1), "trail": trail}


def baseline_records(method, suite, suffix="") -> dict:
    path = ROOT / "runs" / "baselines" / f"{method}_{suite}{suffix}.jsonl"
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["case_id"]] = {"case_id": r["case_id"], "services": None if "error" in r else r["ranks"],
                             "causes": [], "error": r.get("error"), "usage": {}, "turns": None, "trail": []}
    return out


def to_results(records, index, labels) -> list[CaseResult]:
    results = []
    for r in records:
        lab, system = labels[r["case_id"]], index[r["case_id"]].parent.parent.name.split("-")[-1]
        top = r["causes"][0] if r["causes"] else {}
        conf = top.get("confidence") if top.get("verified") is not False else 0.0
        results.append(CaseResult(r["case_id"], (system, lab.root_service, lab.fault), system, lab.fault,
                                  hits(r["services"], lab.root_service), r["services"] is None,
                                  conf if isinstance(conf, (int, float)) else None))
    return results


def load_records(ranker, suite, part, suffix=""):
    path = ROOT / "runs" / "explain" / f"{ranker}_{suite}_{part}{suffix}.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, choices=["RE1", "RE2", "RE3"])
    parser.add_argument("--part", required=True, choices=["train", "test"])
    parser.add_argument("--ranker", required=True, choices=RANKERS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only the first N cases (development)")
    parser.add_argument("--offline", action="store_true", help="replay model calls from the cache only")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--anchor", default="inject", choices=["inject", "detector"],
                        help="start the investigation at the inject time (default) or where our detector flags")
    args = parser.parse_args()
    suffix = "" if args.anchor == "inject" else "_detector"

    cfg = DEFAULT
    provider = LLM_RANKERS[args.ranker][1] if args.ranker in LLM_RANKERS else None
    model = PROVIDERS[provider].model if provider else None
    index, labels = case_index(args.suite), load_labels(args.suite)
    ids = case_ids(args.suite, args.part)[: args.limit]
    commit, dirty = git_state()
    print(f"{args.ranker} on {args.suite} {args.part}: {len(ids)} cases | anchor={args.anchor} | "
          f"commit={commit}{' (dirty)' if dirty else ''}")
    if args.anchor == "inject":
        anchors = {c: labels[c].inject_time for c in ids}
    else:
        anchors = {c: detector_anchor(load_case(args.suite, c, index[c]), cfg.anchor) for c in ids}
    anchor_stats = None
    if args.anchor == "detector":
        offsets = [None if anchors[c] is None else anchors[c] - labels[c].inject_time for c in ids]
        anchor_stats = {"never": sum(o is None for o in offsets),
                        "early": sum(o is not None and o < 0 for o in offsets),
                        "within_120s": sum(o is not None and 0 <= o <= 120 for o in offsets),
                        "late": sum(o is not None and o > 120 for o in offsets)}
        print(f"  detector anchors vs the real fault time: {anchor_stats}")

    started = time.time()
    if args.ranker in BASELINES:
        recorded = baseline_records(args.ranker, args.suite, suffix)
        records = [recorded[c] for c in ids]
        cache = None
    else:
        cache = None
        if args.ranker in LLM_RANKERS:
            cache = ModelCache(ROOT / "runs" / "llm" / f"{args.ranker}_{args.suite}{suffix}.jsonl.gz",
                               client=None if args.offline else client(provider), offline=args.offline)
        records, done = [], 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_case, args.ranker, args.suite, c, index[c], anchors[c],
                                   cache, model): c for c in ids}
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as error:  # a crashed case is a recorded failure, never a dropped one
                    records.append({"case_id": futures[future], "services": None, "causes": [], "turns": None,
                                    "error": f"crashed: {type(error).__name__}: {error}", "usage": {}, "trail": []})
                done += 1
                if done % 25 == 0 or done == len(ids):
                    print(f"  {done}/{len(ids)} cases, {time.time() - started:.0f}s", flush=True)
        order = {c: i for i, c in enumerate(ids)}
        records.sort(key=lambda r: order[r["case_id"]])

    results = to_results(records, index, labels)
    summary = score(results)
    ci = bootstrap_ci(results, cfg.stats.bootstrap_samples, cfg.stats.seed)
    print(f"\nAC@1 {summary['ac1']:.3f} [{ci['ac1'][0]:.3f}, {ci['ac1'][1]:.3f}]  "
          f"AC@3 {summary['ac3']:.3f} [{ci['ac3'][0]:.3f}, {ci['ac3'][1]:.3f}]  "
          f"Avg@5 {summary['avg5']:.3f} [{ci['avg5'][0]:.3f}, {ci['avg5'][1]:.3f}]  "
          f"failed {summary['failed']}/{summary['cases']}")
    by_system, by_fault = breakdown(results, "system"), breakdown(results, "fault")
    for name, table in (("system", by_system), ("fault", by_fault)):
        print(f"  by {name}: " + "  ".join(f"{k} {v['ac1']:.2f}/{v['ac3']:.2f}/{v['avg5']:.2f}" for k, v in table.items()))

    versus = {}
    control = load_records("no-ai", args.suite, args.part, suffix)
    if control and args.ranker != "no-ai":
        control_results = {r.case_id: r for r in to_results(control, index, labels)}
        if all(r.case_id in control_results for r in results):
            paired = [control_results[r.case_id] for r in results]
            versus = {f"ac{k}": mcnemar(results, paired, k) for k in (1, 3)}
            for k, m in versus.items():
                print(f"  vs no-ai {k}: {m['a_only']} cases only this ranker gets, {m['b_only']} only no-ai gets, "
                      f"p = {m['p_value']:.3g}")
    cal = calibration(results)
    if cal:
        print(f"  calibration: ECE {cal['expected_calibration_error']:.3f}; "
              + "  ".join(f"conf>={s['threshold']}: {s['coverage']:.0%} of cases at AC@1 "
                          f"{s['ac1'] if s['ac1'] is None else round(s['ac1'], 2)}" for s in cal["selective"]))
    usage = {k: round(sum(r["usage"].get(k) or 0 for r in records), 1)
             for k in ("input_tokens", "output_tokens", "reasoning_tokens", "model_seconds")}
    if cache:
        print(f"  model calls: {cache.misses} new, {cache.hits} from cache | tokens {usage}")
    print(f"{time.time() - started:.0f}s")

    if args.no_record or args.limit:
        return
    sections = ("investigation", "stats") if args.anchor == "inject" else ("investigation", "stats", "anchor")
    out_dir = ROOT / "runs" / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.ranker}_{args.suite}_{args.part}{suffix}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    entry = {"run_id": uuid.uuid4().hex[:12], "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "git_commit": commit, "git_dirty": dirty, "suite": args.suite, "part": args.part,
             "ranker": args.ranker, "anchor": "inject_time" if args.anchor == "inject" else "detector",
             "anchor_stats": anchor_stats,
             "provider": provider, "model": model,
             "config_id": cfg.section_id(*sections), "config": cfg.sections(*sections),
             "summary": summary, "ci95": ci, "by_system": by_system, "by_fault": by_fault,
             "vs_no_ai": versus, "calibration": cal, "usage": usage}
    with open(ROOT / "runs" / "explain_summary.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"recorded run {entry['run_id']}")


if __name__ == "__main__":
    main()
