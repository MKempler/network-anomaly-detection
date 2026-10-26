"""Score ranked root causes against RCAEval ground truth.

Service-level scoring follows RCAEval's evaluation (main.py), so our rankers
and the published baselines are scored identically:
- a ranked item's service is its text before the first "_", with "-db"
  removed (carts-db counts as carts)
- each service counts at its first appearance; repeats waste a rank
- AC@k: the true service is within the first k; Avg@5: mean of AC@1..AC@5

Rules on top of that, fixed before any LLM results:
- a ranker that failed on a case (error, nothing submitted) scores a miss
  there, and failures are counted and reported, never dropped
- a cause the verifier rejected keeps its rank as a placeholder, so it misses
- 95% intervals come from a bootstrap that resamples (system, service, fault)
  groups, because cases in one group are near duplicates of each other
- two rankers are compared with an exact McNemar test on the same cases

This module is the only place ground truth meets rankings.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

KS = (1, 2, 3, 4, 5)


def service_of_item(item: str) -> str:
    """RCAEval's rule for a ranked item (an indicator like 'carts-db_cpu' or a service name)."""
    return item.split("_")[0].replace("-db", "")


def ranked_services(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        s = item if item.startswith("<rejected") else service_of_item(item)
        if s not in out:
            out.append(s)
    return out


def hits(items: list[str] | None, truth: str) -> np.ndarray:
    """AC@1..AC@5 for one case as 0/1; None (a failed run) misses everywhere."""
    if items is None:
        return np.zeros(len(KS))
    services = ranked_services(items)
    target = service_of_item(truth)
    return np.array([target in services[:k] for k in KS], dtype=float)


@dataclass
class CaseResult:
    case_id: str
    group: tuple            # (system, service, fault): bootstrap cluster
    system: str
    fault: str
    hits: np.ndarray        # AC@1..AC@5
    failed: bool
    confidence: float | None = None   # stated confidence of the top cause, if any


def score(results: list[CaseResult]) -> dict:
    h = np.array([r.hits for r in results]) if results else np.zeros((0, len(KS)))
    return {"cases": len(results), "failed": sum(r.failed for r in results),
            "ac1": float(h[:, 0].mean()) if len(h) else 0.0, "ac3": float(h[:, 2].mean()) if len(h) else 0.0,
            "ac5": float(h[:, 4].mean()) if len(h) else 0.0, "avg5": float(h.mean()) if len(h) else 0.0}


def breakdown(results: list[CaseResult], key: str) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[getattr(r, key)].append(r)
    return {k: score(v) for k, v in sorted(groups.items())}


def bootstrap_ci(results: list[CaseResult], samples: int, seed: int, level: float = 0.95) -> dict:
    """Percentile intervals for AC@1, AC@3 and Avg@5, resampling fault groups."""
    groups = defaultdict(list)
    for r in results:
        groups[r.group].append(r.hits)
    blocks = [np.array(v) for v in groups.values()]
    sums = np.array([b.sum(axis=0) for b in blocks])
    counts = np.array([len(b) for b in blocks])
    picks = np.random.default_rng(seed).integers(0, len(blocks), size=(samples, len(blocks)))
    rates = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)[:, None]
    tail = (1 - level) / 2 * 100
    cis = {}
    for name, column in (("ac1", rates[:, 0]), ("ac3", rates[:, 2]), ("avg5", rates.mean(axis=1))):
        cis[name] = (float(np.percentile(column, tail)), float(np.percentile(column, 100 - tail)))
    return cis


def mcnemar(a: list[CaseResult], b: list[CaseResult], k: int) -> dict:
    """Exact McNemar test on AC@k for two rankers scored on the same cases."""
    by_id = {r.case_id: r for r in b}
    if set(by_id) != {r.case_id for r in a}:
        raise ValueError("rankers must be scored on the same cases")
    i = KS.index(k)
    a_only = sum(1 for r in a if r.hits[i] and not by_id[r.case_id].hits[i])
    b_only = sum(1 for r in a if not r.hits[i] and by_id[r.case_id].hits[i])
    discordant = a_only + b_only
    p = binomtest(a_only, discordant, 0.5).pvalue if discordant else 1.0
    return {"k": k, "a_only": a_only, "b_only": b_only, "p_value": float(p)}


def calibration(results: list[CaseResult], bins: int = 5) -> dict:
    """Reliability of the top cause's stated confidence, and accuracy when confident."""
    rated = [(r.confidence, r.hits[0]) for r in results if r.confidence is not None]
    if not rated:
        return {}
    conf, hit = np.array(rated).T
    edges = np.linspace(0, 1, bins + 1)
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if mask.any():
            table.append({"bin": [round(lo, 2), round(hi, 2)], "cases": int(mask.sum()),
                          "mean_confidence": float(conf[mask].mean()), "ac1": float(hit[mask].mean())})
    selective = []
    for t in (0.5, 0.7, 0.8, 0.9):
        mask = conf >= t
        selective.append({"threshold": t, "coverage": float(mask.mean()),
                          "ac1": float(hit[mask].mean()) if mask.any() else None})
    ece = sum(row["cases"] / len(conf) * abs(row["mean_confidence"] - row["ac1"]) for row in table)
    return {"bins": table, "selective": selective, "expected_calibration_error": float(ece)}
