"""Where Epicenter meets the tools teams already use: alerts in, Slack out.

- `parse_alertmanager` reads a Prometheus Alertmanager webhook payload and
  returns when the earliest firing alert started, which anchors the
  investigation exactly like the detector's flag does.
- `slack_message` renders a finished investigation as a Slack Block Kit
  message: ranked causes with confidence and the verifier's verdict, the
  handoff drafts, and a link back to the console.
- `post_slack` sends it to an incoming-webhook URL.
"""

from datetime import datetime

import requests


def parse_alertmanager(payload: dict) -> dict:
    """{'start': unix seconds of the earliest firing alert, 'alerts': [...]} or ValueError."""
    firing = [a for a in payload.get("alerts", []) if a.get("status", "firing") == "firing"]
    if not firing:
        raise ValueError("no firing alerts in payload")
    alerts = []
    for a in firing:
        started = datetime.fromisoformat(a["startsAt"].replace("Z", "+00:00")).timestamp()
        labels = a.get("labels", {})
        alerts.append({"name": labels.get("alertname", "alert"), "service": labels.get("service") or labels.get("job"),
                       "severity": labels.get("severity"), "start": started})
    return {"start": min(a["start"] for a in alerts), "alerts": alerts}


def _text(kind: str, text: str) -> dict:
    return {"type": kind, "text": text}


def slack_message(incident: dict, console_url: str = "") -> dict:
    """Block Kit message for a finished incident (the console's public incident record)."""
    causes = incident.get("causes") or []
    verdict = incident.get("verdict") or {}
    top = causes[0] if causes else None
    headline = (f"Likely root cause: {top['service']} ({round(100 * top['confidence'])}%)"
                if top and isinstance(top.get("confidence"), (int, float)) else "Investigation finished without a ranking")
    blocks = [{"type": "header", "text": _text("plain_text", headline[:150])}]
    context = [f"incident {incident['id']}"]
    if verdict.get("investigation_s") is not None:
        context.append(f"investigated in {verdict['investigation_s']} s")
    if incident.get("source"):
        context.append(f"triggered by {incident['source']}")
    blocks.append({"type": "context", "elements": [_text("mrkdwn", " · ".join(context))]})
    lines = []
    for i, c in enumerate(causes[:3], 1):
        check = "verified" if c.get("verified") else "unverified" if c.get("verified") is False else ""
        conf = f"{round(100 * c['confidence'])}%" if isinstance(c.get("confidence"), (int, float)) else "–"
        lines.append(f"*{i}. {c['service']}* · {conf} · `{c.get('indicator', '')}` {('· ' + check) if check else ''}\n"
                     f"{(c.get('reasoning') or '')[:240]}")
    if lines:
        blocks.append({"type": "section", "text": _text("mrkdwn", "\n\n".join(lines))})
    handoff = incident.get("handoff") or {}
    for part, title in (("internal_note", "Internal note (draft)"), ("customer_update", "Customer update (draft)")):
        sentences = handoff.get(part) or []
        if sentences:
            body = " ".join(s["text"] for s in sentences)
            grounded = handoff.get(f"{part}_grounded")
            suffix = f"\n_{round(100 * grounded)}% of sentences cite checked evidence_" if grounded is not None else ""
            blocks.append({"type": "section", "text": _text("mrkdwn", f"*{title}*\n{body}{suffix}")})
    if console_url:
        blocks.append({"type": "actions", "elements": [{"type": "button", "text": _text("plain_text", "Open in Epicenter"),
                                                        "url": console_url}]})
    return {"text": headline, "blocks": blocks}


def post_slack(url: str, message: dict) -> bool:
    try:
        return requests.post(url, json=message, timeout=10).ok
    except requests.RequestException:
        return False
