import json

from src.explain import ONE_SHOT_INSTRUCTIONS, Ranking, no_ai_ranking, one_shot_input, spark_one_shot
from src.tools import Investigation
from tests.test_tools import ANCHOR, CFG, make_case, make_logs, make_traces


class FakeCache:
    """Returns a canned Responses API payload and remembers the request."""

    def __init__(self, causes=None, text=None):
        self.text = text if text is not None else json.dumps({"causes": causes})
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": self.text}]}],
                "usage": {"input_tokens": 100, "output_tokens": 50}}


def inv():
    return Investigation(make_case(), ANCHOR, CFG, logs=make_logs, traces=make_traces)


def cause(service, indicator="cart_cpu"):
    return {"service": service, "indicator": indicator, "reasoning": "r", "confidence": 0.5}


def test_ranking_services_keep_first_appearance():
    r = Ranking("x", [cause("cart"), cause("redis"), cause("cart"), cause("frontend")])
    assert r.services == ["cart", "redis", "frontend"]


def test_no_ai_ranks_services_by_strongest_deviation():
    r = no_ai_ranking(inv())
    d = inv().metric_deviations().drop_duplicates("service")
    assert r.services == d["service"].tolist()[:CFG.causes]
    assert r.causes[0]["indicator"] == d.iloc[0]["metric"]


def test_one_shot_input_lists_services_and_top_metrics_without_absolute_times():
    text, metrics = one_shot_input(inv())
    assert "System services: cart, frontend, redis" in text
    assert metrics[0] in text and str(ANCHOR) not in text


def test_one_shot_request_constrains_output_to_case_services_and_listed_metrics():
    fake = FakeCache([cause("cart"), cause("redis"), cause("frontend"), cause("cart"), cause("redis")])
    r = spark_one_shot(inv(), fake, model="m")
    [request] = fake.requests
    schema = request["text"]["format"]["schema"]["properties"]["causes"]
    assert schema["items"]["properties"]["service"]["enum"] == ["cart", "frontend", "redis"]
    assert set(schema["items"]["properties"]["indicator"]["enum"]) == set(one_shot_input(inv())[1])
    assert schema["minItems"] == schema["maxItems"] == CFG.causes
    assert request["instructions"] == ONE_SHOT_INSTRUCTIONS and request["store"] is False
    assert r.error is None and r.services == ["cart", "redis", "frontend"]


def test_unparseable_output_is_an_error_not_a_crash():
    r = spark_one_shot(inv(), FakeCache(text="not json"), model="m")
    assert r.causes == [] and "unparseable" in r.error


def test_unknown_service_is_rejected():
    r = spark_one_shot(inv(), FakeCache([cause("billing-db")] + [cause("cart")] * 4), model="m")
    assert r.causes == [] and "not in the input" in r.error
