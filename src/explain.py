"""Rank likely root causes for one flagged event.

Rung 1, no-AI: services ordered by their strongest metric deviation. This is
the control group; any gap between it and Spark measures reasoning, not access
to information.

Rung 2, Spark one-shot: a fixed prompt holding the same top deviations, and an
output schema whose enums are this case's services and the listed metrics,
so constrained decoding can't name a service or metric that isn't there.

Both rankers see only an `Investigation` (observables relative to the flag).
Ground truth never reaches this module.
"""

import json
from dataclasses import dataclass, field

from src.cache import ModelCache, output_text
from src.tools import Investigation, sig

ONE_SHOT_INSTRUCTIONS = (
    "You are diagnosing a microservice failure. Below are the metrics that deviated most during the "
    "flagged window, with their deviation scores, and the services they belong to. Return exactly 5 "
    "ranked candidate root causes as JSON: each with service, indicator, and one sentence of reasoning "
    "tied to a metric in the input. Rank by likelihood. Do not invent services or metrics not in the "
    "input. Name 5 different services. For each cause also give your confidence, from 0 to 1, that it "
    "is the root cause."
)


@dataclass
class Ranking:
    ranker: str
    causes: list[dict]                       # service, indicator, reasoning, confidence (and citations for agents)
    usage: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def services(self) -> list[str]:
        """Ranked services, each at its first appearance.

        A cause the verifier rejected keeps its rank as a placeholder, so it
        scores as a miss instead of letting later guesses move up.
        """
        seen: list[str] = []
        for i, cause in enumerate(self.causes):
            if cause.get("verified") is False:
                seen.append(f"<rejected #{i + 1}>")
            elif cause.get("service") and cause["service"] not in seen:
                seen.append(cause["service"])
        return seen


def flat_usage(response: dict) -> dict:
    """Token counts from a Responses API response, with reasoning tokens at the top level."""
    u = response.get("usage") or {}
    return {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
            "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}


def no_ai_ranking(inv: Investigation) -> Ranking:
    order: list[str] = []
    deviations = inv.metric_deviations()
    for row in deviations.itertuples():
        if row.service not in order:
            order.append(row.service)
        if len(order) == inv.cfg.causes:
            break
    top = deviations.drop_duplicates("service").set_index("service")
    causes = [{"service": s, "indicator": top.loc[s, "metric"], "confidence": None,
               "reasoning": f"largest deviation is {top.loc[s, 'metric']} at {sig(top.loc[s, 'score'])}"}
              for s in order]
    return Ranking("no-ai", causes)


def one_shot_input(inv: Investigation) -> tuple[str, list[str]]:
    """The fixed-format event description and the metrics it lists."""
    listing = inv.top_deviations()["metrics"]
    lines = [f"- {m['metric']} (service: {m['service']}): deviation {m['score']}, {m['direction']} from "
             f"{m['baseline_mean']} to {m['window_mean']}" for m in listing]
    text = (f"System services: {', '.join(inv.case.candidate_services)}\n"
            f"Flagged window: the {inv.cfg.window_s} s after the flag, compared with the {inv.cfg.baseline_s} s before it.\n"
            f"Top deviating metrics:\n" + "\n".join(lines))
    return text, [m["metric"] for m in listing]


def ranking_schema(services: list[str], metrics: list[str], causes: int) -> dict:
    cause = {
        "type": "object",
        "properties": {
            "service": {"type": "string", "enum": services},
            "indicator": {"type": "string", "enum": metrics},
            "reasoning": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["service", "indicator", "reasoning", "confidence"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "name": "ranked_causes", "strict": True, "schema": {
        "type": "object",
        "properties": {"causes": {"type": "array", "items": cause, "minItems": causes, "maxItems": causes}},
        "required": ["causes"],
        "additionalProperties": False,
    }}


def spark_one_shot(inv: Investigation, cache: ModelCache, model: str) -> Ranking:
    text, metrics = one_shot_input(inv)
    request = {"model": model, "instructions": ONE_SHOT_INSTRUCTIONS, "input": text, "store": False,
               "text": {"format": ranking_schema(inv.case.candidate_services, metrics, inv.cfg.causes)}}
    response = cache.create(**request)
    usage = flat_usage(response)
    try:
        causes = json.loads(output_text(response))["causes"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return Ranking("spark-one-shot", [], usage, error=f"unparseable output: {error}")
    known_services, known_metrics = set(inv.case.candidate_services), set(metrics)
    if any(c.get("service") not in known_services or c.get("indicator") not in known_metrics for c in causes):
        return Ranking("spark-one-shot", [], usage, error="named a service or metric not in the input")
    return Ranking("spark-one-shot", causes, usage)
