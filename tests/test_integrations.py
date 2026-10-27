import pytest

from src.integrations import parse_alertmanager, slack_message

PAYLOAD = {"version": "4", "status": "firing", "alerts": [
    {"status": "firing", "labels": {"alertname": "HighLatency", "service": "frontend", "severity": "page"},
     "startsAt": "2026-10-27T16:20:05Z"},
    {"status": "firing", "labels": {"alertname": "ErrorBudgetBurn", "job": "checkout"}, "startsAt": "2026-10-27T16:19:58.5Z"},
    {"status": "resolved", "labels": {"alertname": "Old"}, "startsAt": "2026-10-27T15:00:00Z"},
]}


def test_alertmanager_anchor_is_the_earliest_firing_alert():
    parsed = parse_alertmanager(PAYLOAD)
    assert parsed["start"] == pytest.approx(1793118000 - 1.5, abs=1e5) and len(parsed["alerts"]) == 2
    assert {a["service"] for a in parsed["alerts"]} == {"frontend", "checkout"}
    assert parsed["start"] == min(a["start"] for a in parsed["alerts"])


def test_payload_without_firing_alerts_is_rejected():
    with pytest.raises(ValueError):
        parse_alertmanager({"alerts": [{"status": "resolved", "startsAt": "2026-10-27T15:00:00Z"}]})


INCIDENT = {"id": "abc123", "source": "alertmanager", "verdict": {"investigation_s": 23.8},
            "causes": [{"service": "cartservice", "confidence": 0.96, "indicator": "cartservice_latency", "verified": True,
                        "reasoning": "Earliest and largest spike."},
                       {"service": "frontend", "confidence": 0.85, "indicator": "frontend_latency", "verified": False,
                        "reasoning": "Victim of cart."}],
            "handoff": {"internal_note": [{"text": "cartservice is the likely cause."}], "internal_note_grounded": 1.0,
                        "customer_update": [{"text": "Checkout may be slow."}], "customer_update_grounded": 0.5}}


def test_slack_message_carries_causes_verdicts_and_drafts():
    msg = slack_message(INCIDENT, "https://epicenter.example/#live")
    assert msg["text"] == "Likely root cause: cartservice (96%)"
    body = str(msg["blocks"])
    assert "verified" in body and "unverified" in body and "triggered by alertmanager" in body
    assert "Customer update (draft)" in body and "50% of sentences cite checked evidence" in body
    assert msg["blocks"][-1]["elements"][0]["url"] == "https://epicenter.example/#live"


def test_slack_message_without_a_ranking():
    assert slack_message({"id": "x", "causes": []})["text"] == "Investigation finished without a ranking"
