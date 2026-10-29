import json

import pytest
from fastapi.testclient import TestClient

from tests.test_lab import make_lab


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    db = tmp_path_factory.mktemp("db") / "app.db"
    monkeypatch_module.setenv("DATABASE_URL", f"sqlite:///{db}")
    from src.api import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def first_case(client, suite="RE2"):
    return client.get(f"/api/replays?suite={suite}").json()["cases"][0]["case_id"]


def test_console_and_health(client):
    assert client.get("/api/health").json() == {"ok": True}
    page = client.get("/")
    assert page.status_code == 200 and "Epicenter" in page.text


def test_replay_hides_the_answer_until_reveal(client):
    case_id = first_case(client)
    hidden = client.get(f"/api/replays/RE2/{case_id}").json()
    assert "answer" not in hidden and "outcome" not in hidden
    assert hidden["trail"] and hidden["series"]
    shown = client.get(f"/api/replays/RE2/{case_id}?reveal=true").json()
    assert shown["answer"]["service"]


def test_replay_index_hides_outcomes(client):
    for case in client.get("/api/replays?suite=RE1").json()["cases"]:
        assert "fault" not in case and "top1" not in case


def test_unknown_or_traversal_paths_are_rejected(client):
    assert client.get("/api/replays/RE9/x").status_code == 404
    assert client.get("/api/replays/RE1/..%2F..%2Fsrc%2Fapi").status_code == 404
    assert client.get("/api/replays/RE1/does-not-exist").status_code == 404


def test_results_use_latest_clean_run_per_ranker(client):
    rankers = {r["ranker"]: r for r in client.get("/api/results?suite=RE1").json()["rankers"]}
    assert rankers["open-agent"]["failed"] == 0          # the complete re-run, not the rate-limited one
    assert rankers["spark-agent"]["ac1"] > rankers["nsigma"]["ac1"]


def test_detection_lists_three_detectors(client):
    assert {d["detector"] for d in client.get("/api/detection").json()} == {"zscore", "iforest", "lstm"}


def test_detect_finds_a_sustained_shift(client):
    times = list(range(1000, 1600))
    rows = [[10.0 + (0.5 if i % 2 else -0.5) + (60.0 if i >= 400 else 0.0)] for i in range(600)]
    flag = client.post("/api/detect", json={"times": times, "columns": ["svc_cpu"], "rows": rows}).json()
    assert flag["flag_time"] == 1400
    bad = client.post("/api/detect", json={"times": [1, 2], "columns": ["a_b"], "rows": [[1.0]]})
    assert bad.status_code == 422


# -- live lab endpoints, against the simulated lab from test_lab ---------------


@pytest.fixture
def live(client, tmp_path):
    lab, injector, seen = make_lab(tmp_path)
    lab.telemetry.rows = [(1000, {"cartservice_latency": 10.0})]
    client.app.state.lab = lab
    yield lab
    client.app.state.lab = None


def test_live_endpoints_404_without_the_lab(client):
    client.app.state.lab = None
    assert client.get("/api/live/state").status_code == 404
    assert client.get("/api/live/menu").status_code == 200


def test_break_queues_and_rate_limits(client, live):
    ok = client.post("/api/live/break", json={"service": "cartservice", "fault": "delay"},
                     headers={"x-forwarded-for": "9.9.9.9"})
    assert ok.status_code == 200 and ok.json()["position"] == 1
    again = client.post("/api/live/break", json={"service": "frontend", "fault": "cpu"},
                        headers={"x-forwarded-for": "9.9.9.9"})
    assert again.status_code == 429
    bad = client.post("/api/live/break", json={"service": "investigator", "fault": "pause"},
                      headers={"x-forwarded-for": "8.8.8.8"})
    assert bad.status_code == 400
    state = client.get("/api/live/state").json()
    assert [q["service"] for q in state["queue"]] == ["cartservice"] and "visitor" not in json.dumps(state)


def test_feedback_is_validated_and_stored(client, live, tmp_path, monkeypatch):
    import src.api as api
    monkeypatch.setattr(api, "ROOT", tmp_path)
    assert client.post("/api/live/feedback", json={"incident_id": "abc", "vote": "up", "comment": "nice"}).json() == {"ok": True}
    assert client.post("/api/live/feedback", json={"incident_id": "abc", "vote": "maybe"}).status_code == 422
    assert client.post("/api/live/feedback", json={"incident_id": "abc", "vote": "up", "comment": "x" * 501}).status_code == 422
    stored = (tmp_path / "runs" / "live" / "feedback.jsonl").read_text().splitlines()
    assert len(stored) == 1 and json.loads(stored[0])["comment"] == "nice"


def test_event_stream_greets_then_ticks(client, live):
    with client.stream("GET", "/api/live/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                lines.append(json.loads(line[6:]))
            if len(lines) == 2:
                break
    assert lines[0]["type"] == "hello" and lines[1]["type"] == "tick"
    assert lines[1]["row"] == {"cartservice_latency": 10.0}


def test_spoofed_forwarded_for_does_not_bypass_the_cooldown(client, live):
    first = client.post("/api/live/break", json={"service": "cartservice", "fault": "delay"},
                        headers={"x-forwarded-for": "1.1.1.1, 7.7.7.7"})
    assert first.status_code == 200
    spoofed = client.post("/api/live/break", json={"service": "frontend", "fault": "cpu"},
                          headers={"x-forwarded-for": "2.2.2.2, 7.7.7.7"})
    assert spoofed.status_code == 429      # same proxy-observed address, new fake prefix


ALERT = {"alerts": [{"status": "firing", "labels": {"alertname": "HighLatency", "service": "frontend"},
                     "startsAt": "2026-10-27T16:20:05Z"}]}


def test_webhook_is_disabled_without_a_token(client, live, monkeypatch):
    monkeypatch.delenv("WEBHOOK_TOKEN", raising=False)
    assert client.post("/webhook/alertmanager", json=ALERT).status_code == 404


def test_webhook_needs_the_right_token_and_a_real_payload(client, live, monkeypatch):
    monkeypatch.setenv("WEBHOOK_TOKEN", "s3cret")
    assert client.post("/webhook/alertmanager", json=ALERT).status_code == 401
    assert client.post("/webhook/alertmanager", json=ALERT, headers={"authorization": "Bearer nope"}).status_code == 401
    assert client.post("/webhook/alertmanager", json={"alerts": []},
                       headers={"authorization": "Bearer s3cret"}).status_code == 400
    ok = client.post("/webhook/alertmanager", json=ALERT, headers={"authorization": "Bearer s3cret"})
    assert ok.status_code == 202 and ok.json()["alerts"] == 1
    assert live.waiting[0].source == "alertmanager"


def test_slack_preview_for_a_recorded_incident(client, live):
    live.submit("cartservice", "delay", "4.4.4.4")
    incident = live.waiting.popleft()
    live.run(incident)
    msg = client.get(f"/api/live/slack/{incident.id}").json()
    assert msg["text"].startswith("Likely root cause: cartservice")
    assert client.get("/api/live/slack/nope").status_code == 404


# -- user study ------------------------------------------------------------------

def test_study_slots_are_counterbalanced(client):
    plan = [client.get(f"/api/study/{s}").json()["cases"] for s in range(6)]
    for slot in plan:
        assert sum(c["tool"] for c in slot) == 3
    by_case = {}
    for slot in plan:
        for c in slot:
            by_case.setdefault(c["case_id"], set()).add(c["tool"])
    assert all(v == {True, False} for v in by_case.values())
    assert client.get("/api/study/99").status_code == 404


def test_study_case_never_includes_the_answer(client):
    cid = client.get("/api/study/0").json()["cases"][0]["case_id"]
    bare = client.get(f"/api/study/case/{cid}").json()
    tool = client.get(f"/api/study/case/{cid}?tool=1").json()
    assert "trail" not in bare and "causes" not in bare and bare["candidates"]
    assert tool["trail"] and tool["causes"]
    for b in (bare, tool):
        assert "answer" not in b and "answer_key" not in json.dumps(b)
    assert client.get("/api/study/case/re2-not-in-study").status_code == 404


def test_study_answers_validate_and_names_need_quote_permission(client, tmp_path, monkeypatch):
    import src.api as api
    real = api.STUDY
    monkeypatch.setattr(api, "STUDY", tmp_path)
    (tmp_path / "plan.json").write_text((real / "plan.json").read_text())
    cid = json.loads((real / "plan.json").read_text())["cases"][0]
    ok = {"participant": "p01", "slot": 0, "case_id": cid, "tool": True, "answer": "cart", "confidence": 4, "seconds": 93.5}
    assert client.post("/api/study/answer", json=ok).json() == {"ok": True}
    assert client.post("/api/study/answer", json={**ok, "participant": "<script>"}).status_code == 422
    assert client.post("/api/study/answer", json={**ok, "confidence": 9}).status_code == 422
    client.post("/api/study/exit", json={"participant": "p01", "would_use": 5, "display_name": "Sam", "quote_ok": False})
    client.post("/api/study/exit", json={"participant": "p02", "would_use": 4, "display_name": "Ana", "quote_ok": True})
    exits = [json.loads(l) for l in (tmp_path / "exit.jsonl").read_text().splitlines()]
    assert [e["display_name"] for e in exits] == ["", "Ana"]


def test_menu_offers_the_models_the_lab_runs_and_break_takes_one(client, tmp_path):
    lab, injector, seen = make_lab(tmp_path, models={"spark-open": None, "glimmer-tuned": 3}, warm=lambda m: None)
    client.app.state.lab = lab
    try:
        menu = client.get("/api/live/menu").json()
        assert [(m["key"], m["left_today"]) for m in menu["models"]] == [("spark-open", None), ("glimmer-tuned", 3)]
        ok = client.post("/api/live/break", json={"service": "cartservice", "fault": "delay", "model": "glimmer-tuned"},
                         headers={"x-forwarded-for": "7.7.7.7"})
        assert ok.status_code == 200 and lab.waiting[0].model == "glimmer-tuned"
        assert client.get("/api/live/menu").json()["models"][1]["left_today"] == 2
        bad = client.post("/api/live/break", json={"service": "frontend", "fault": "cpu", "model": "gpt"},
                          headers={"x-forwarded-for": "6.6.6.6"})
        assert bad.status_code == 400
    finally:
        client.app.state.lab = None


def test_glimmer_option_is_off_unless_enabled(monkeypatch):
    from src.api import glimmer_enabled
    monkeypatch.delenv("LIVE_GLIMMER", raising=False)
    assert glimmer_enabled() is False
    monkeypatch.setenv("LIVE_GLIMMER", "1")
    for name in ("GLIMMER_BASE_URL", "GLIMMER_API_KEY", "MODAL_PROXY_KEY", "MODAL_PROXY_SECRET"):
        monkeypatch.setenv(name, "x")
    assert glimmer_enabled() is True


def test_trust_serves_the_latest_clean_test_analysis(client):
    t = client.get("/api/trust").json()
    assert t["part"] == "test" and not t["git_dirty"] and t["threshold"] is not None
    spark = t["results"]["spark/RE2"]
    assert spark["escalation"]["answered"] + spark["escalation"]["escalated"] == spark["cases"]



def test_approving_when_nothing_awaits_is_refused(client, live):
    r = client.post("/api/live/approve", json={"incident_id": "abc"})
    assert r.status_code == 409


def test_history_serves_recent_rows_of_the_asked_metrics(client, live, tmp_path):
    from src.lab.telemetry import Telemetry
    live.telemetry = Telemetry(docker_client=None, spans_file=tmp_path / "none.jsonl")
    live.telemetry.rows.extend((t, {"cart_latency": float(t), "cart_cpu": 1.0}) for t in range(100))
    rows = client.get("/api/live/history?metrics=cart_latency&seconds=5").json()
    assert rows[0] == [94, {"cart_latency": 94.0}] and rows[-1] == [99, {"cart_latency": 99.0}]
    assert client.get("/api/live/history?metrics=cart_cpu&seconds=601").status_code == 422
