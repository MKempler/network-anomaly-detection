import pytest
from fastapi.testclient import TestClient


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
    assert page.status_code == 200 and "Incident Investigator" in page.text


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
