"""Check an agent's ranked causes against the tool results it cites.

A cause is verified when at least one of the tool calls it cites actually
happened and returned evidence for the cause's service scoring at or above the
evidence threshold. Anything else (no citations, ids that were never issued,
or evidence for some other service, or too weak) rejects the cause.

A rejected cause keeps its rank and scores as a miss. Dropping it would let the
next guess move up and be credited for a slot the agent filled with an
unsupported claim.
"""

from src.tools import Investigation

REJECTED = "<rejected>"


def check_cause(cause: dict, inv: Investigation) -> tuple[bool, str]:
    calls = {c.call_id: c for c in inv.calls}
    cited = [cid for cid in cause.get("evidence", []) if cid in calls]
    if not cause.get("evidence"):
        return False, "cites no tool calls"
    if not cited:
        return False, "cites tool calls that were never made"
    best = max((e["score"] for cid in cited for e in calls[cid].evidence if e["service"] == cause["service"]),
               default=None)
    if best is None:
        return False, "cited calls show no evidence for this service"
    if best < inv.cfg.evidence_threshold:
        return False, f"strongest cited evidence {best} is below {inv.cfg.evidence_threshold}"
    return True, f"supported by evidence scoring {best}"


def verify(causes: list[dict], inv: Investigation) -> list[dict]:
    """Annotate each cause with `verified` and `verification`; ranks are preserved."""
    out = []
    for cause in causes:
        ok, why = check_cause(cause, inv)
        out.append({**cause, "verified": ok, "verification": why})
    return out
