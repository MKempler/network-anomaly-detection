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


# ---- trust (eval-spec-v1.2) ------------------------------------------------------------
# Post-hoc analyses of recorded runs; they re-score nothing above. Defined on train-split
# runs and frozen before being computed on the test split.
#
# - the verifier: the agent's own first pick (before verification) against the scored one;
#   a rejected first pick is either a wrong answer caught or a right answer the evidence
#   didn't support ("right for the wrong reasons")
# - why each top-1 miss missed, checked in this order:
#     no ranking, named it but its evidence was rejected, named it at rank 2-5,
#     examined it with a tool but didn't name it, never examined it
# - escalation: a case with no verified first answer is escalated to a human; the rest are
#   ordered by the first cause's stated confidence, giving a risk-coverage curve. The
#   operating threshold is the lowest confidence whose answered cases reach TRUST_TARGET
#   AC@1 on the train split, then applied unchanged to test. The target is the on-call
#   bar of being wrong at most once in 50 answers; on the train split, targets of 0.97,
#   0.98 and 0.99 all give the same threshold, so the choice isn't knife-edge.
#   (Also tried on train only: the margin between the first two confidences, and the count
#   of verified causes. The first confidence ranked best and was kept.)

TRUST_TARGET = 0.98
FAILURES = ("no ranking", "named it, evidence rejected", "named it at rank 2-5",
            "examined it, not named", "never examined it")


@dataclass
class TrustCase:
    case_id: str
    answered: bool
    correct: bool            # scored AC@1: the verified first cause is the root cause
    raw_correct: bool        # the agent's own first pick, before verification
    top_rejected: bool       # the verifier rejected that first pick
    score: float             # escalation score: first cause's confidence; -inf with no verified first answer
    failure: str | None      # why a top-1 miss missed; None when correct
    causes: int
    rejected: int            # causes the verifier rejected
    rejected_wrong: int      # ... of which named a service that isn't the root cause


def trust_case(record: dict, truth: str) -> TrustCase:
    """Trust facts for one recorded run. `truth` is the case's root-cause item."""
    target = service_of_item(truth)
    causes = record.get("causes") or []
    raw = [c.get("service", "") for c in causes]
    answered = record.get("services") is not None
    correct = bool(hits(record.get("services"), truth)[0]) if answered else False
    raw_correct = answered and bool(raw) and service_of_item(raw[0]) == target
    top_rejected = bool(causes) and causes[0].get("verified") is False
    if not answered or top_rejected:
        score = float("-inf")
    else:
        score = float(causes[0].get("confidence") or 0.0)
    rejected = [c for c in causes if c.get("verified") is False]
    failure = None
    if not correct:
        named = [c for c in causes if service_of_item(c.get("service", "")) == target]
        examined = False
        for step in record.get("trail") or []:
            if step.get("type") != "tool":
                continue
            values = [v for v in (step.get("args") or {}).values() if isinstance(v, str)]
            values += [e.get("service", "") for e in step.get("evidence") or []]
            if any(service_of_item(v) == target for v in values):
                examined = True
                break
        if not answered:
            failure = FAILURES[0]
        elif any(c.get("verified") is False for c in named):
            failure = FAILURES[1]
        elif named:
            failure = FAILURES[2]
        elif examined:
            failure = FAILURES[3]
        else:
            failure = FAILURES[4]
    return TrustCase(record["case_id"], answered, correct, raw_correct, top_rejected, score, failure,
                     len(causes), len(rejected),
                     sum(service_of_item(c.get("service", "")) != target for c in rejected))


def risk_coverage(cases: list[TrustCase]) -> dict:
    """Answering only cases at or above each confidence: coverage and AC@1 there, plus AURC."""
    ranked = [c for c in cases if c.score != float("-inf")]
    points, aurc, prev = [], 0.0, 0.0
    for t in sorted({c.score for c in ranked}, reverse=True):
        answered = [c for c in ranked if c.score >= t]
        coverage = len(answered) / len(cases)
        ac1 = sum(c.correct for c in answered) / len(answered)
        aurc += (coverage - prev) * (1 - ac1)
        prev = coverage
        points.append({"threshold": t, "coverage": coverage, "ac1": ac1})
    return {"points": points, "aurc": aurc, "never_answered": len(cases) - len(ranked)}


def fit_threshold(cases: list[TrustCase], target: float = TRUST_TARGET) -> float | None:
    """Lowest confidence whose answered cases reach `target` AC@1 (most coverage that does)."""
    ok = [p["threshold"] for p in risk_coverage(cases)["points"] if p["ac1"] >= target]
    return min(ok) if ok else None


def at_threshold(cases: list[TrustCase], threshold: float) -> dict:
    answered = [c for c in cases if c.score >= threshold]
    return {"threshold": threshold, "answered": len(answered), "escalated": len(cases) - len(answered),
            "coverage": len(answered) / len(cases),
            "ac1_answered": sum(c.correct for c in answered) / len(answered) if answered else None,
            "wrong_answered": sum(not c.correct for c in answered)}


def trust_summary(cases: list[TrustCase]) -> dict:
    n = len(cases)
    raw_right = [c for c in cases if c.raw_correct]
    top_rejected = [c for c in cases if c.top_rejected]
    return {
        "cases": n,
        "ac1": sum(c.correct for c in cases) / n,
        "ac1_before_verification": len(raw_right) / n,
        "causes": sum(c.causes for c in cases),
        "causes_rejected": sum(c.rejected for c in cases),
        "rejected_naming_wrong_service": sum(c.rejected_wrong for c in cases),
        "first_pick_rejected": len(top_rejected),
        "first_pick_rejected_and_wrong": sum(not c.raw_correct for c in top_rejected),
        "first_pick_right_but_unsupported": sum(c.raw_correct for c in top_rejected),
        "failures": {k: sum(c.failure == k for c in cases) for k in FAILURES},
        "risk_coverage": risk_coverage(cases),
    }
