"""The trust analyses: verifier accounting, why a miss missed, escalation."""

from src.evaluate_explain import FAILURES, at_threshold, fit_threshold, risk_coverage, trust_case

TRUTH = "cartservice_cpu"


def record(causes, services="from causes", trail=()):
    if services == "from causes":
        services = [c["service"] if c.get("verified", True) else f"<rejected #{i + 1}>" for i, c in enumerate(causes)]
    return {"case_id": "c", "causes": causes, "services": services, "trail": list(trail)}


def cause(service, verified=True, confidence=0.9):
    return {"service": service, "verified": verified, "confidence": confidence}


def test_a_verified_right_answer_is_correct_with_its_confidence_as_score():
    c = trust_case(record([cause("cartservice", confidence=0.95), cause("frontend")]), TRUTH)
    assert c.correct and c.raw_correct and not c.top_rejected and c.failure is None and c.score == 0.95


def test_a_right_first_pick_with_unsupported_evidence_is_a_miss_and_escalated():
    c = trust_case(record([cause("cartservice", verified=False), cause("frontend")]), TRUTH)
    assert not c.correct and c.raw_correct and c.top_rejected
    assert c.failure == "named it, evidence rejected" and c.score == float("-inf")
    assert c.rejected == 1 and c.rejected_wrong == 0


def test_each_kind_of_miss_is_told_apart():
    rank3 = trust_case(record([cause("frontend"), cause("checkoutservice"), cause("cartservice")]), TRUTH)
    looked = trust_case(record([cause("frontend")], trail=[{"type": "tool", "name": "get_logs",
                                                             "args": {"service": "cartservice"}}]), TRUTH)
    seen_in_evidence = trust_case(record([cause("frontend")], trail=[{"type": "tool", "args": {"metric": "x"},
                                                                       "evidence": [{"service": "cartservice"}]}]), TRUTH)
    never = trust_case(record([cause("frontend")], trail=[{"type": "tool", "args": {"service": "frontend"}}]), TRUTH)
    failed = trust_case(record([], services=None), TRUTH)
    assert rank3.failure == "named it at rank 2-5"
    assert looked.failure == seen_in_evidence.failure == "examined it, not named"
    assert never.failure == "never examined it"
    assert failed.failure == "no ranking" and not failed.answered and failed.score == float("-inf")
    assert set(FAILURES) >= {rank3.failure, looked.failure, never.failure, failed.failure}


def test_risk_coverage_and_the_fitted_threshold():
    cases = [trust_case(record([cause("cartservice", confidence=0.95)]), TRUTH),
             trust_case(record([cause("cartservice", confidence=0.95)]), TRUTH),
             trust_case(record([cause("frontend", confidence=0.90)]), TRUTH),
             trust_case(record([cause("cartservice", verified=False)]), TRUTH)]      # always escalated
    rc = risk_coverage(cases)
    assert [(p["threshold"], p["coverage"], round(p["ac1"], 3)) for p in rc["points"]] == [(0.95, 0.5, 1.0),
                                                                                            (0.9, 0.75, 0.667)]
    assert rc["never_answered"] == 1 and round(rc["aurc"], 4) == round(0.25 * (1 - 2 / 3), 4)
    assert fit_threshold(cases, 0.98) == 0.95 and fit_threshold(cases, 0.6) == 0.9
    op = at_threshold(cases, 0.95)
    assert op["answered"] == 2 and op["escalated"] == 2 and op["ac1_answered"] == 1.0 and op["wrong_answered"] == 0


def test_a_wrong_pick_is_placed_relative_to_the_true_cause_in_the_call_graph():
    from src.evaluate_explain import graph_relation
    graph = [{"caller": "frontend", "callee": "checkoutservice"}, {"caller": "checkoutservice", "callee": "paymentservice"},
             {"caller": "frontend", "callee": "adservice"}]
    assert graph_relation(graph, "checkoutservice", "paymentservice") == "upstream victim"
    assert graph_relation(graph, "frontend", "paymentservice") == "upstream victim"        # through checkout
    assert graph_relation(graph, "paymentservice", "checkoutservice") == "downstream of the cause"
    assert graph_relation(graph, "adservice", "paymentservice") == "unconnected"
    assert graph_relation([], "frontend", "paymentservice") == "no traces"
