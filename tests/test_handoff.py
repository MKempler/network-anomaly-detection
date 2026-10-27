import json

from src.handoff import SCHEMA, draft, evidence_digest, groundedness

TRAIL = [
    {"type": "note", "text": "looking at cart"},
    {"type": "tool", "call_id": "c1", "name": "compare_to_baseline", "args": {"metric": "cartservice_latency"},
     "evidence": [{"service": "cartservice", "signal": "cartservice_latency", "kind": "metric", "score": 99.7}]},
    {"type": "tool", "call_id": "c2", "name": "get_logs", "args": {"service": "frontend"}, "evidence": []},
]
CAUSES = [{"service": "cartservice", "indicator": "cartservice_latency", "confidence": 0.96,
           "reasoning": "cart latency jumped first", "evidence": ["c1"], "verified": True}]


class FakeCache:
    def __init__(self, drafts):
        self.drafts, self.requests = drafts, []

    def create(self, **request):
        self.requests.append(request)
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(self.drafts)}]}],
                "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_digest_keeps_only_calls_with_evidence():
    assert [d["call_id"] for d in evidence_digest(TRAIL)] == ["c1"]


def test_groundedness_requires_a_citation_that_returned_evidence():
    drafts = {"internal_note": [{"text": "Cart latency jumped.", "evidence": ["c1"]},
                                {"text": "Frontend logs were checked.", "evidence": ["c2"]},
                                {"text": "Probably a deploy.", "evidence": []}],
              "customer_update": [{"text": "Checkout may be slow.", "evidence": ["c1", "c9"]}]}
    scores = groundedness(drafts, TRAIL)
    assert [s["grounded"] for s in drafts["internal_note"]] == [True, False, False]
    assert scores == {"internal_note_grounded": 1 / 3, "customer_update_grounded": 1.0}


def test_draft_sends_findings_and_schema_and_scores_the_result():
    fake = FakeCache({"internal_note": [{"text": "cartservice latency rose 2.6 to 851 ms.", "evidence": ["c1"]}],
                      "customer_update": [{"text": "Some customers may see slow checkout.", "evidence": ["c1"]}]})
    out = draft(CAUSES, TRAIL, fake, model="m")
    [request] = fake.requests
    payload = json.loads(request["input"])
    assert payload["ranked_causes"][0]["service"] == "cartservice" and payload["tool_results"][0]["call_id"] == "c1"
    assert request["text"]["format"] is SCHEMA and request["store"] is False
    assert out["internal_note_grounded"] == 1.0 and out["customer_update_grounded"] == 1.0


def test_unparseable_draft_is_reported():
    class Broken(FakeCache):
        def create(self, **request):
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "{nope"}]}]}
    assert "unparseable" in draft(CAUSES, TRAIL, Broken({}), model="m")["error"]
