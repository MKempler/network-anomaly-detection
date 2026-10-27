"""Draft what a support engineer writes during an incident, grounded in evidence.

After an investigation, one more model call turns the ranked causes and the
investigation's tool results into:
- an internal ticket note for the team (technical, specific), and
- a status update for the affected customer (plain language, no internal
  jargon, nothing promised that the evidence doesn't support).

Every sentence must cite the call ids it rests on. `groundedness` checks each
citation against the recorded trail: a sentence counts as grounded only if it
cites at least one call that actually ran and returned evidence. The draft is
built from the trail, not live tools, so the same code serves live incidents
and recorded replays.
"""

import json

from src.cache import ModelCache, output_text
from src.explain import flat_usage

INSTRUCTIONS = """You are the on-call support engineer. An automated investigation has finished. Write two short \
drafts from its findings, using only what the evidence below shows.

1. internal_note: 3-5 sentences for the incident ticket. Name the suspected root-cause service and the signal \
that shows it, the services affected as a consequence, and the confidence.
2. customer_update: 2-4 sentences for the customer. Plain language, no service names, metric names or call ids \
in the text. Say what customers may notice and that the team has identified the likely source and is working \
on it. Do not promise a fix time or claim the issue is resolved.

Every sentence must cite, in its evidence list, the call_ids of the tool results it relies on. A sentence \
without evidence that holds up is flagged."""


def _sentences(name: str) -> dict:
    return {"type": "array", "minItems": 1, "maxItems": 6, "items": {
        "type": "object", "additionalProperties": False, "required": ["text", "evidence"],
        "properties": {"text": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}}}}


SCHEMA = {"type": "json_schema", "name": "handoff", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["internal_note", "customer_update"],
    "properties": {"internal_note": _sentences("internal_note"), "customer_update": _sentences("customer_update")}}}


def evidence_digest(trail: list[dict]) -> list[dict]:
    """The tool results a draft may cite: id, what was asked, and the strongest evidence returned."""
    digest = []
    for step in trail:
        if step.get("type") == "tool" and step.get("evidence"):
            digest.append({"call_id": step["call_id"], "tool": step["name"], "args": step.get("args", {}),
                           "evidence": step["evidence"][:3]})
    return digest


def draft(causes: list[dict], trail: list[dict], cache: ModelCache, model: str) -> dict:
    findings = [{k: c.get(k) for k in ("service", "indicator", "confidence", "reasoning", "evidence", "verified")}
                for c in causes[:3]]
    payload = json.dumps({"ranked_causes": findings, "tool_results": evidence_digest(trail)}, separators=(",", ":"))
    response = cache.create(model=model, instructions=INSTRUCTIONS, input=payload, store=False,
                            text={"format": SCHEMA})
    try:
        drafts = json.loads(output_text(response))
    except json.JSONDecodeError as error:
        return {"error": f"unparseable draft: {error}", "usage": flat_usage(response)}
    return {**drafts, **groundedness(drafts, trail), "usage": flat_usage(response)}


def groundedness(drafts: dict, trail: list[dict]) -> dict:
    """Mark each sentence grounded or not, and report the grounded share per draft."""
    valid = {s["call_id"] for s in trail if s.get("type") == "tool" and s.get("evidence")}
    scores = {}
    for part in ("internal_note", "customer_update"):
        sentences = drafts.get(part) or []
        for sentence in sentences:
            sentence["grounded"] = any(cid in valid for cid in sentence.get("evidence", []))
        scores[f"{part}_grounded"] = (sum(s["grounded"] for s in sentences) / len(sentences)) if sentences else 0.0
    return scores
