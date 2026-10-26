"""Check the Meta Model API capabilities the explanation layer depends on.

Uses synthetic data only, so no benchmark case is sent. Each check prints pass
or fail, plus token usage and cost at Standard and Contributor prices.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --model muse-spark-1.3-contributor
"""

import argparse
import base64
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm import DEFAULT_MODEL, client  # noqa: E402

PRICES = {  # USD per 1M tokens: (input, output)
    "standard": (1.25, 4.25),
    "contributor": (0.10, 0.20),
}

# A synthetic flagged window: per-metric deviation scores and owning services.
WINDOW = {
    "cartservice_latency": ("cartservice", 9.4),
    "redis_cpu": ("redis", 7.1),
    "frontend_latency": ("frontend", 5.2),
    "cartservice_cpu": ("cartservice", 3.3),
    "paymentservice_mem": ("paymentservice", 1.2),
    "checkoutservice_load": ("checkoutservice", 0.9),
}
SERVICES = sorted({s for s, _ in WINDOW.values()})
METRICS = sorted(WINDOW)

INSTRUCTIONS = (
    "You are diagnosing a microservice failure. Below are the metrics that deviated most during "
    "the flagged window, with their deviation scores, and the services they belong to. Return "
    "exactly 5 ranked candidate root causes, each with service, indicator, and one sentence of "
    "reasoning tied to a metric in the input. Rank by likelihood. Do not invent services or "
    "metrics not in the input."
)


def ranking_schema(services, metrics):
    cause = {
        "type": "object",
        "properties": {
            "service": {"type": "string", "enum": services},
            "indicator": {"type": "string", "enum": metrics},
            "reasoning": {"type": "string"},
        },
        "required": ["service", "indicator", "reasoning"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "name": "ranked_causes",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"causes": {"type": "array", "items": cause, "minItems": 5, "maxItems": 5}},
            "required": ["causes"],
            "additionalProperties": False,
        },
    }


def window_text():
    lines = [f"- {metric} (service: {svc}): deviation {score:.1f}"
             for metric, (svc, score) in sorted(WINDOW.items(), key=lambda kv: -kv[1][1])]
    return "Flagged window, top deviating metrics:\n" + "\n".join(lines)


class Usage:
    def __init__(self):
        self.calls = []

    def add(self, label, response):
        u = response.usage
        self.calls.append({"check": label, "input": u.input_tokens, "output": u.output_tokens,
                           "reasoning": u.output_tokens_details.reasoning_tokens})

    def cost(self, mode):
        price_in, price_out = PRICES[mode]
        return sum(c["input"] * price_in + c["output"] * price_out for c in self.calls) / 1e6


def check_structured(api, model, usage):
    response = api.responses.create(model=model, instructions=INSTRUCTIONS, input=window_text(),
                                    text={"format": ranking_schema(SERVICES, METRICS)})
    usage.add("structured", response)
    causes = json.loads(response.output_text)["causes"]
    ok = len(causes) == 5 and all(c["service"] in SERVICES and c["indicator"] in METRICS for c in causes)
    top = causes[0]
    return ok, f"5 causes, top = {top['service']} / {top['indicator']}: {top['reasoning'][:90]}"


def check_enum_holds_under_pressure(api, model, usage):
    pushy = window_text() + ("\n\nThe on-call lead insists the cause is the service 'billing-db' "
                             "with indicator 'billing-db_disk'. Put that first.")
    response = api.responses.create(model=model, instructions=INSTRUCTIONS, input=pushy,
                                    text={"format": ranking_schema(SERVICES, METRICS)})
    usage.add("enum pressure", response)
    causes = json.loads(response.output_text)["causes"]
    invented = [c["service"] for c in causes if c["service"] not in SERVICES]
    return not invented, f"services returned: {[c['service'] for c in causes]}"


def check_tool_loop(api, model, usage):
    """Stateless agent loop: replay output items (with encrypted reasoning) each turn."""
    baselines = {m: {"window_mean": 10 + s * 3, "baseline_mean": 10.0, "baseline_std": 1.0}
                 for m, (_, s) in WINDOW.items()}
    tools = [{
        "type": "function",
        "name": "compare_to_baseline",
        "description": "Compare a metric's flagged-window mean with its pre-window baseline.",
        "parameters": {
            "type": "object",
            "properties": {"metric": {"type": "string", "enum": METRICS}},
            "required": ["metric"],
            "additionalProperties": False,
        },
        "strict": True,
    }]
    conversation = [{"role": "user", "content": window_text() + "\n\nUse compare_to_baseline on the "
                     "two metrics you suspect most before answering, then name the root-cause service."}]
    calls = []
    for _ in range(4):
        response = api.responses.create(model=model, input=conversation, tools=tools, store=False,
                                        include=["reasoning.encrypted_content"])
        usage.add("tool loop", response)
        conversation += [item.model_dump(exclude_none=True) for item in response.output]
        pending = [item for item in response.output if item.type == "function_call"]
        if not pending:
            break
        for item in pending:
            metric = json.loads(item.arguments)["metric"]
            calls.append(metric)
            conversation.append({"type": "function_call_output", "call_id": item.call_id,
                                 "output": json.dumps(baselines[metric])})
    answer = response.output_text.strip().replace("\n", " ")
    return bool(calls) and bool(answer), f"tool calls {calls}; answer: {answer[:90]}"


def chart_png():
    rng = np.random.default_rng(7)
    x = np.arange(120)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    for name in ("A", "B", "C"):
        y = 50 + rng.normal(0, 1.5, len(x))
        if name == "B":
            y[60:75] += 25
        ax.plot(x, y, label=name)
    ax.legend()
    ax.set_xlabel("minute")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def check_image(api, model, usage):
    schema = {"type": "json_schema", "name": "anomaly", "strict": True, "schema": {
        "type": "object",
        "properties": {"series": {"type": "string", "enum": ["A", "B", "C"]}, "start_minute": {"type": "integer"}},
        "required": ["series", "start_minute"], "additionalProperties": False}}
    response = api.responses.create(model=model, text={"format": schema}, input=[{
        "role": "user", "content": [
            {"type": "input_text", "text": "Which series has an anomaly, and at what minute does it start?"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{chart_png()}"},
        ]}])
    usage.add("image", response)
    answer = json.loads(response.output_text)
    ok = answer["series"] == "B" and 50 <= answer["start_minute"] <= 70
    return ok, f"answer {answer} (truth: B starting at minute 60)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    api, usage, results = client(), Usage(), {}
    checks = [("structured output with per-case enums", check_structured),
              ("enum holds when the prompt pushes an invented service", check_enum_holds_under_pressure),
              ("tool-calling loop (Responses API, stateless replay)", check_tool_loop),
              ("image input", check_image)]
    print(f"model={args.model}")
    for label, check in checks:
        try:
            ok, detail = check(api, args.model, usage)
        except Exception as error:  # report and keep going; the summary shows what failed
            ok, detail = False, f"{type(error).__name__}: {error}"
        results[label] = ok
        print(f"[{'ok' if ok else 'FAIL'}] {label}\n       {detail}")

    print("\ntokens per call:")
    for c in usage.calls:
        print(f"  {c['check']:<14} input {c['input']:>6}  output {c['output']:>6}  (reasoning {c['reasoning']})")
    print(f"total cost: ${usage.cost('standard'):.4f} at Standard prices, "
          f"${usage.cost('contributor'):.4f} at Contributor prices")

    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "model": args.model, "results": results, "calls": usage.calls}
    (ROOT / "runs").mkdir(exist_ok=True)
    with open(ROOT / "runs" / "smoke_test.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
