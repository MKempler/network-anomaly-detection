import json

from src.agent import FINAL_NUDGE, investigate, tool_specs
from src.explain import Ranking
from src.tools import Investigation
from src.verify import verify
from tests.test_tools import ANCHOR, CFG, make_case, make_logs, make_traces

ALL = frozenset({"metrics", "logs", "traces"})


def inv(sources=("metrics", "logs", "traces")):
    return Investigation(make_case(sources), ANCHOR, CFG, logs=make_logs, traces=make_traces)


def call(name, args, call_id):
    return {"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": call_id}


def cause(service, evidence, indicator="cart_cpu"):
    return {"service": service, "indicator": indicator, "reasoning": "r", "confidence": 0.5, "evidence": evidence}


def submit(causes, call_id="fc_submit"):
    return call("submit_ranking", {"causes": causes}, call_id)


class ScriptedModel:
    """Plays back one list of output items per turn and records each request."""

    def __init__(self, turns):
        self.turns, self.requests = list(turns), []

    def create(self, **request):
        self.requests.append(request)
        output = self.turns.pop(0) if self.turns else []
        return {"output": output, "usage": {"input_tokens": 10, "output_tokens": 5,
                                            "output_tokens_details": {"reasoning_tokens": 3}}}


def five_causes(evidence):
    return [cause("cart", evidence), cause("redis", evidence), cause("frontend", evidence),
            cause("cart", evidence), cause("redis", evidence)]


def test_agent_investigates_then_submits_verified_ranking():
    model = ScriptedModel([
        [call("top_deviations", {"k": 3}, "fc_1")],
        [call("compare_to_baseline", {"metric": "cart_cpu"}, "fc_2")],
        [submit(five_causes(["c1", "c2"]))],
    ])
    run = investigate(inv(), model, "m", ALL)
    assert run.turns == 3 and run.ranking.error is None
    assert run.ranking.causes[0]["verified"] is True
    assert run.ranking.services[0] == "cart"
    # tool results go back under the model's own call ids, each carrying our citable id
    outputs = [i for i in model.requests[1]["input"] if i.get("type") == "function_call_output"]
    assert outputs[0]["call_id"] == "fc_1" and json.loads(outputs[0]["output"])["call_id"] == "c1"
    assert [step["type"] for step in run.trail] == ["tool", "tool", "submit"]
    assert run.ranking.usage == {"input_tokens": 30, "output_tokens": 15, "reasoning_tokens": 9}


def test_first_message_matches_one_shot_and_requests_replay_reasoning():
    model = ScriptedModel([[submit(five_causes([]))]])
    investigate(inv(), model, "m", ALL)
    [request] = model.requests
    assert "Top deviating metrics" in request["input"][0]["content"]
    assert request["store"] is False and request["include"] == ["reasoning.encrypted_content"]


def test_unsupported_causes_are_rejected_but_keep_their_rank():
    model = ScriptedModel([
        [call("compare_to_baseline", {"metric": "cart_cpu"}, "fc_1")],   # c1 shows only cart
        [submit([cause("redis", ["c1"]), cause("cart", ["c1"]), cause("frontend", []),
                 cause("cart", ["c9"]), cause("redis", ["c1"])])],
    ])
    run = investigate(inv(), model, "m", ALL)
    flags = [(c["verified"], c["verification"]) for c in run.ranking.causes]
    assert flags[0] == (False, "cited calls show no evidence for this service")
    assert flags[1][0] is True
    assert flags[2] == (False, "cites no tool calls")
    assert flags[3] == (False, "cites tool calls that were never made")
    assert run.ranking.services[:2] == ["<rejected #1>", "cart"]   # cart stays at rank 2, not promoted to 1


def test_metrics_only_rung_offers_no_log_or_trace_tools():
    names = {t["name"] for t in tool_specs(inv(), frozenset({"metrics"}))}
    assert names == {"top_deviations", "compare_to_baseline", "get_metric_window", "submit_ranking"}
    sock_shop = {t["name"] for t in tool_specs(inv(("metrics", "logs")), ALL)}
    assert "get_logs" in sock_shop and "get_trace_errors" not in sock_shop


def test_calling_a_tool_that_was_not_offered_returns_an_error():
    model = ScriptedModel([[call("get_logs", {"service": "cart"}, "fc_1")], [submit(five_causes(["c1"]))]])
    run = investigate(inv(), model, "m", frozenset({"metrics"}))
    assert "unknown tool" in run.trail[0]["error"]
    assert all(c["verified"] is False for c in run.ranking.causes)


def test_budget_exhaustion_forces_a_submit_only_turn_then_fails_cleanly():
    stalling = [[call("top_deviations", {}, f"fc_{i}")] for i in range(3)]
    model = ScriptedModel(stalling + [[]])
    run = investigate(inv(), model, "m", ALL, max_steps=3)
    final = model.requests[-1]
    assert [t["name"] for t in final["tools"]] == ["submit_ranking"]
    assert final["input"][-1] == {"role": "user", "content": FINAL_NUDGE}
    assert run.ranking.causes == [] and "step budget" in run.ranking.error


def test_unparseable_submission_is_an_error():
    bad = {"type": "function_call", "name": "submit_ranking", "arguments": "{not json", "call_id": "x"}
    run = investigate(inv(), ScriptedModel([[bad]]), "m", ALL)
    assert "unparseable" in run.ranking.error


def test_verify_threshold_uses_config():
    i = inv()
    record = i.call("top_deviations", {})
    weak = next(e for e in record.evidence if e["score"] < CFG.evidence_threshold)
    [checked] = verify([cause(weak["service"], [record.call_id])], i)
    assert checked["verified"] is (max(e["score"] for e in record.evidence if e["service"] == weak["service"])
                                   >= CFG.evidence_threshold)


def test_ranking_placeholder_for_rejected():
    r = Ranking("x", [{"service": "a", "verified": False}, {"service": "b", "verified": True}])
    assert r.services == ["<rejected #1>", "b"]
