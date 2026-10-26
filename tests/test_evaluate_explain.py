import numpy as np
import pytest

from src.evaluate_explain import (CaseResult, bootstrap_ci, breakdown, calibration, hits, mcnemar,
                                  ranked_services, score, service_of_item)


@pytest.mark.parametrize("item, service", [
    ("carts-db_container-memory-rss", "carts"),     # RCAEval folds -db into its parent
    ("carts-db", "carts"),
    ("ts-order-mongo", "ts-order-mongo"),           # but not -mongo
    ("adservice_cpu", "adservice"),
    ("frontend-external_load", "frontend-external"),
])
def test_service_of_item_matches_rcaeval(item, service):
    assert service_of_item(item) == service


def test_ranked_services_dedupe_after_folding_and_keep_rejected_slots():
    items = ["carts-db_cpu", "carts_mem", "<rejected #3>", "orders_cpu", "user_cpu"]
    assert ranked_services(items) == ["carts", "<rejected #3>", "orders", "user"]


def test_hits_at_each_k():
    assert hits(["a", "b", "c", "d", "e"], "c").tolist() == [0, 0, 1, 1, 1]
    assert hits(["<rejected #1>", "c"], "c").tolist() == [0, 1, 1, 1, 1]
    assert hits(None, "c").tolist() == [0, 0, 0, 0, 0]


def result(case_id, group, h, system="OB", fault="cpu", failed=False, confidence=None):
    return CaseResult(case_id, group, system, fault, np.array(h, dtype=float), failed, confidence)


def test_score_and_avg5():
    rs = [result("a", 1, [1, 1, 1, 1, 1]), result("b", 2, [0, 0, 1, 1, 1], failed=False),
          result("c", 3, [0, 0, 0, 0, 0], failed=True)]
    s = score(rs)
    assert s["cases"] == 3 and s["failed"] == 1
    assert s["ac1"] == pytest.approx(1 / 3) and s["ac3"] == pytest.approx(2 / 3)
    assert s["avg5"] == pytest.approx((5 + 3 + 0) / 15)


def test_breakdown_by_fault():
    rs = [result("a", 1, [1] * 5, fault="cpu"), result("b", 2, [0] * 5, fault="mem")]
    assert breakdown(rs, "fault")["cpu"]["ac1"] == 1.0 and breakdown(rs, "fault")["mem"]["ac1"] == 0.0


def test_bootstrap_resamples_groups_and_contains_estimate():
    rs = [result(f"c{i}", i // 3, [i % 2] * 5) for i in range(30)]
    ci = bootstrap_ci(rs, samples=500, seed=0)
    assert ci["ac1"][0] <= score(rs)["ac1"] <= ci["ac1"][1]
    assert ci == bootstrap_ci(rs, samples=500, seed=0)


def test_mcnemar_counts_discordant_pairs():
    a = [result(str(i), i, [1] * 5) for i in range(10)]
    b = [result(str(i), i, [1] * 5 if i < 2 else [0] * 5) for i in range(10)]
    m = mcnemar(a, b, k=1)
    assert (m["a_only"], m["b_only"]) == (8, 0) and m["p_value"] < 0.01
    assert mcnemar(a, a, k=1)["p_value"] == 1.0
    with pytest.raises(ValueError):
        mcnemar(a, b[:5], k=1)


def test_calibration_bins_and_selective_accuracy():
    rs = [result("a", 1, [1] * 5, confidence=0.95), result("b", 2, [1] * 5, confidence=0.9),
          result("c", 3, [0] * 5, confidence=0.3), result("d", 4, [0] * 5, confidence=None)]
    cal = calibration(rs)
    assert sum(b["cases"] for b in cal["bins"]) == 3
    at_09 = next(s for s in cal["selective"] if s["threshold"] == 0.9)
    assert at_09["coverage"] == pytest.approx(2 / 3) and at_09["ac1"] == 1.0
    assert calibration([result("x", 1, [1] * 5)]) == {}
