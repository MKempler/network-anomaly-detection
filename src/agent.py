"""Spark as an investigator: a tool-calling loop over one flagged event.

The agent starts from exactly what the one-shot ranker sees (the same top
deviations), so the gap between the two rungs measures investigation. It can
then query metrics, logs, traces and the call graph through `tools.py`, and
must finish by calling `submit_ranking` with 5 ranked causes. Each cause
cites the ids of the tool calls that support it, and `verify.py` checks those
citations.

The loop runs on the Responses API with stateless encrypted reasoning replay:
each turn resends the conversation, including the model's earlier output items,
so its reasoning carries across tool calls without server-side state. Every
turn goes through the record-and-replay cache.
"""

import json
from dataclasses import dataclass, field

from src.cache import ModelCache, function_calls, output_text
from src.explain import Ranking, flat_usage, one_shot_input
from src.tools import Investigation
from src.verify import verify

AGENT_INSTRUCTIONS = """You are an on-call engineer diagnosing a microservice failure. An anomaly was \
flagged; the first message lists the services and the metrics that deviated most in the flagged window \
compared with the baseline before it.

Investigate before you conclude. Use the tools to confirm or rule out candidates: compare a metric with its \
baseline, look at its series, and, where available, check a service's logs, its trace latency and errors, \
and what it calls or is called by. A root cause often makes its callers look unhealthy too, so check \
whether a deviating service is a cause or a victim.

Every tool result has a call_id. When you are done, call submit_ranking with exactly 5 different services, \
most likely first. For each, give the indicator (the metric or signal that best shows the problem), one \
sentence of reasoning, your confidence from 0 to 1, and the call_ids of the tool results that support it. \
Only cite results that show that service deviating; unsupported causes are rejected.

You have a budget of {steps} turns. Use what you need, then submit."""

FINAL_NUDGE = "You have reached your investigation budget. Call submit_ranking now with your best 5 causes."


def _service(services: list[str], description: str) -> dict:
    return {"type": "string", "enum": services, "description": description}


def tool_specs(inv: Investigation, sources: frozenset[str]) -> list[dict]:
    """Function tools for this case; log and trace tools only where that data exists and is allowed."""
    services = inv.case.candidate_services
    specs = [
        {"name": "top_deviations", "description": "Metrics ranked by deviation score (window vs baseline). "
         "Optionally limited to one service.",
         "parameters": {"type": "object", "properties": {
             "k": {"type": "integer", "description": "how many metrics (default 15)"},
             "service": _service(services, "only this service's metrics")}, "required": []}},
        {"name": "compare_to_baseline", "description": "One metric: baseline mean/std, window mean/max/min, "
         "deviation score, and seconds after the flag when it first crossed the threshold.",
         "parameters": {"type": "object", "properties": {
             "metric": {"type": "string", "description": "metric name, e.g. from top_deviations"}},
             "required": ["metric"]}},
        {"name": "get_metric_window", "description": "One metric's series as [seconds relative to the flag, value] "
         "points, downsampled. Negative seconds are before the flag.",
         "parameters": {"type": "object", "properties": {
             "metric": {"type": "string"}, "start_s": {"type": "integer"}, "end_s": {"type": "integer"}},
             "required": ["metric"]}},
    ]
    available = sources & inv.case.sources
    if "logs" in available:
        specs.append({"name": "get_logs", "description": "A service's log volume, error/warning lines, and the "
                      "log patterns whose frequency changed most, window vs baseline.",
                      "parameters": {"type": "object", "properties": {
                          "service": _service(services, "service whose logs to read")}, "required": ["service"]}})
    if "traces" in available:
        specs.append({"name": "get_trace_errors", "description": "A service's trace operations with latency "
                      "(p50/p95) and, where recorded, error rates, window vs baseline.",
                      "parameters": {"type": "object", "properties": {
                          "service": _service(services, "service whose spans to read")}, "required": ["service"]}})
        specs.append({"name": "get_dependencies", "description": "Which services call this service and which it "
                      "calls, from traces.",
                      "parameters": {"type": "object", "properties": {
                          "service": _service(services, "service to look up")}, "required": ["service"]}})
    return [{"type": "function", **spec} for spec in specs] + [submit_spec(inv)]


def submit_spec(inv: Investigation) -> dict:
    cause = {"type": "object", "properties": {
        "service": _service(inv.case.candidate_services, "suspected root-cause service"),
        "indicator": {"type": "string", "description": "metric or signal that best shows the problem"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "description": "0 to 1"},
        "evidence": {"type": "array", "items": {"type": "string"}, "description": "supporting call_ids"},
    }, "required": ["service", "indicator", "reasoning", "confidence", "evidence"], "additionalProperties": False}
    return {"type": "function", "name": "submit_ranking", "description": "Submit the final ranked root causes.",
            "parameters": {"type": "object", "properties": {"causes": {
                "type": "array", "items": cause, "minItems": inv.cfg.causes, "maxItems": inv.cfg.causes}},
                "required": ["causes"], "additionalProperties": False}}


def _clean(value):
    """Drop None fields from response items before sending them back as input."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


@dataclass
class AgentRun:
    ranking: Ranking
    trail: list[dict] = field(default_factory=list)   # what the console replays
    turns: int = 0


def investigate(inv: Investigation, cache: ModelCache, model: str, sources: frozenset[str],
                max_steps: int = 10, ranker: str = "spark-agent", on_step=None) -> AgentRun:
    """Run the investigation. `on_step`, if given, sees each trail entry as it happens (live view)."""
    text, _ = one_shot_input(inv)
    tools = tool_specs(inv, sources)
    conversation: list[dict] = [{"role": "user", "content": text}]
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "model_seconds": 0.0}
    trail: list[dict] = []

    def log(entry: dict):
        trail.append(entry)
        if on_step is not None:
            on_step(entry)

    instructions = AGENT_INSTRUCTIONS.format(steps=max_steps)

    for turn in range(1, max_steps + 2):
        final = turn > max_steps
        if final:
            conversation.append({"role": "user", "content": FINAL_NUDGE})
        response = cache.create(model=model, instructions=instructions, input=conversation,
                                tools=[submit_spec(inv)] if final else tools, store=False,
                                include=["reasoning.encrypted_content"])
        for key, value in flat_usage(response).items():
            usage[key] += value
        conversation += [_clean(item) for item in response.get("output", [])]
        if note := output_text(response).strip():
            log({"turn": turn, "type": "note", "text": note})

        calls = function_calls(response)
        submitted = next((c for c in calls if c["name"] == "submit_ranking"), None)
        if submitted is not None:
            try:
                causes = json.loads(submitted["arguments"])["causes"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                return AgentRun(Ranking(ranker, [], usage, error=f"unparseable submission: {error}"), trail, turn)
            causes = verify(causes, inv)
            log({"turn": turn, "type": "submit", "causes": causes})
            return AgentRun(Ranking(ranker, causes, usage), trail, turn)

        if not calls and not final:
            conversation.append({"role": "user", "content": "Continue investigating with the tools, or call "
                                 "submit_ranking if you are ready."})
        for call in calls:
            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            allowed = {t["name"] for t in tools}
            record = inv.call(call["name"], args) if call["name"] in allowed else \
                inv.call("__not_offered__", {})
            log({"turn": turn, "type": "tool", "call_id": record.call_id, "name": call["name"],
                          "args": args, "evidence": record.evidence[:5], "error": record.result.get("error")})
            conversation.append({"type": "function_call_output", "call_id": call["call_id"],
                                 "output": inv.output(record)})
        if final:
            break
    return AgentRun(Ranking(ranker, [], usage, error="no ranking submitted within the step budget"), trail, max_steps)
