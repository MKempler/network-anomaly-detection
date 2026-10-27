"""HTTP API and console.

    uvicorn src.api:app --reload          # http://127.0.0.1:8000

Serves static/console.html and:
- GET  /api/health
- GET  /api/results?suite=RE1&anchor=inject   latest clean run per ranker
- GET  /api/runs, GET /api/runs/{run_id}      every recorded evaluation run
- GET  /api/detection                         recorded SMD detector runs
- GET  /api/replays?suite=RE1                 replay library index
- GET  /api/replays/{suite}/{case_id}         one replay bundle; the benchmark's
                                              answer only with ?reveal=true
- POST /api/detect                            where our detector would flag a
                                              posted metric stream

With LAB_ENABLED=1 (the live-outage lab, lab/docker-compose.yml) it also runs
the shared live system:
- GET  /api/live/menu, GET /api/live/state    the fault menu; current incident,
                                              queue and public tally
- POST /api/live/break                        join the queue with one fault
- GET  /api/live/events                       Server-Sent Events: phases, agent
                                              steps, and a metrics tick per second
- POST /api/live/feedback                     a vote and optional comment
- POST /webhook/alertmanager                  a Prometheus Alertmanager webhook; needs
                                              WEBHOOK_TOKEN, queues an investigation
                                              anchored at the alert's start
- GET  /api/live/slack/{incident_id}          the Slack message for a finished incident
                                              (posted automatically if SLACK_WEBHOOK_URL)

The user study (scripts/user_study/) is served on any host:
- GET  /api/study/{slot}                      a participant slot's case order and conditions
- GET  /api/study/case/{case_id}?tool=0|1     charts only, or charts plus Epicenter's
                                              investigation; never the answer
- POST /api/study/answer, /api/study/exit     one answer; the closing questions

Everything served comes from the committed records under runs/, imported into
the database at startup, so the console needs no raw benchmark data.
"""

import hmac
import json
import os
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.anchor import detector_anchor
from src.config import DEFAULT
from src.db import DetectionRun, EvalResult, setup
from src.integrations import parse_alertmanager, slack_message
from src.load_rcaeval import Case, service_of

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
REPLAYS = ROOT / "runs" / "replay"
SUITES = ("RE1", "RE2", "RE3")
HIDDEN_UNTIL_REVEAL = ("answer", "outcome")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = setup()
    app.state.lab = start_lab() if os.environ.get("LAB_ENABLED") == "1" else None
    yield


def start_lab():
    """Start the live telemetry recorder and the incident loop (lab mode only)."""
    import docker

    from src import handoff
    from src.agent import investigate
    from src.cache import ModelCache
    from src.integrations import post_slack, slack_message
    from src.lab.faults import FaultInjector
    from src.lab.orchestrator import Lab
    from src.lab.telemetry import Telemetry
    from src.llm import PROVIDERS, client

    engine_client = docker.from_env()
    telemetry = Telemetry(engine_client, Path(os.environ.get("LAB_SPANS_FILE", "/spans/spans.jsonl")))
    telemetry.start()
    cfg = DEFAULT.live
    cache = ModelCache(ROOT / "runs" / "live" / "llm.jsonl.gz", client(cfg.provider))
    model = PROVIDERS[cfg.provider].model
    sources = frozenset({"metrics", "logs", "traces"})

    def run_investigation(inv, on_step):
        return investigate(inv, cache, model, sources, DEFAULT.investigation.max_steps, "live-agent", on_step)

    def write_handoff(causes, trail):
        return handoff.draft(causes, trail, cache, model)

    slack_url, console = os.environ.get("SLACK_WEBHOOK_URL"), os.environ.get("CONSOLE_URL", "")

    def notify(incident):
        if slack_url:
            post_slack(slack_url, slack_message(incident, f"{console}/#live" if console else ""))

    lab = Lab(telemetry, FaultInjector(engine_client), run_investigation, handoff=write_handoff, notify=notify)
    threading.Thread(target=lab.run_forever, daemon=True).start()
    return lab


app = FastAPI(title="Epicenter", lifespan=lifespan)


def _row(obj) -> dict:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/results")
def results(suite: str = Query("RE1"), anchor: str = Query("inject"), part: str = Query("test")):
    """The latest clean run per ranker for one suite, as the console's results table."""
    anchor_name = "inject_time" if anchor == "inject" else anchor
    with Session(app.state.engine) as s:
        rows = s.scalars(select(EvalResult).where(EvalResult.suite == suite, EvalResult.part == part,
                                                  EvalResult.anchor == anchor_name,
                                                  EvalResult.git_dirty.is_(False))
                         .order_by(EvalResult.timestamp)).all()
    latest = {}
    for r in rows:
        latest[r.ranker] = _row(r)   # later runs replace earlier ones
    return {"suite": suite, "anchor": anchor, "part": part, "rankers": list(latest.values())}


@app.get("/api/runs")
def runs():
    with Session(app.state.engine) as s:
        return [_row(r) for r in s.scalars(select(EvalResult).order_by(EvalResult.timestamp))]


@app.get("/api/runs/{run_id}")
def run(run_id: str):
    with Session(app.state.engine) as s:
        r = s.get(EvalResult, run_id) or s.get(DetectionRun, run_id)
        if r is None:
            raise HTTPException(404, "no such run")
        return _row(r)


@app.get("/api/detection")
def detection():
    with Session(app.state.engine) as s:
        rows = s.scalars(select(DetectionRun).where(DetectionRun.git_dirty.is_(False))
                         .order_by(DetectionRun.timestamp)).all()
    latest = {}
    for r in rows:
        latest[r.detector] = _row(r)
    return list(latest.values())


@app.get("/api/replays")
def replay_index(suite: str = Query("RE1")):
    if suite not in SUITES:
        raise HTTPException(404, "unknown suite")
    index = json.loads((REPLAYS / suite / "index.json").read_text())
    cases = [{k: v for k, v in c.items() if k not in ("fault", "top1")} for c in index["cases"]]
    return {"suite": suite, "ranker": index["ranker"], "cases": cases}


@app.get("/api/replays/{suite}/{case_id}")
def replay(suite: str, case_id: str, reveal: bool = False):
    if suite not in SUITES or not case_id.replace("-", "").isalnum():
        raise HTTPException(404, "unknown case")
    path = REPLAYS / suite / f"{case_id}.json"
    if not path.exists():
        raise HTTPException(404, "unknown case")
    bundle = json.loads(path.read_text())
    if not reveal:
        for key in HIDDEN_UNTIL_REVEAL:
            bundle.pop(key, None)
    return bundle


class MetricStream(BaseModel):
    times: list[float] = Field(description="unix seconds, one per row, 1 s apart")
    columns: list[str] = Field(description="metric names, '<service>_<metric>'")
    rows: list[list[float]]


@app.post("/api/detect")
def detect(stream: MetricStream):
    """Where our detector would start an investigation on this stream (null if never)."""
    if len(stream.rows) != len(stream.times) or any(len(r) != len(stream.columns) for r in stream.rows):
        raise HTTPException(422, "rows must match times and columns")
    services = {c: service_of(c) or c for c in stream.columns}
    frame = pd.DataFrame(stream.rows, index=pd.Index(stream.times), columns=stream.columns)
    case = Case("posted", "-", "-", frame, services, frozenset({"metrics"}))
    anchor = detector_anchor(case, DEFAULT.anchor)
    return {"flag_time": anchor, "detector": "per-metric z-score alarm", "config": DEFAULT.sections("anchor")}


def _lab(request: Request):
    lab = request.app.state.lab
    if lab is None:
        raise HTTPException(404, "the live lab is not running on this server")
    return lab


def _visitor(request: Request) -> str:
    """The visitor's address for rate limiting.

    A client can put anything in X-Forwarded-For, but the proxy in front of us
    (the tunnel) appends the address it actually saw, so only the last entry is
    trusted. Without the header, the socket peer is used.
    """
    forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
    return forwarded[-1] if forwarded else (request.client.host if request.client else "unknown")


@app.get("/api/live/menu")
def live_menu():
    from src.lab.faults import FAULTS, TARGETS
    return {"faults": FAULTS, "targets": list(TARGETS)}


@app.get("/api/live/state")
def live_state(request: Request):
    lab = _lab(request)
    return {"current": lab.current.public() if lab.current else None, "queue": lab.queue_view(),
            "tally": lab.tally()}


class BreakRequest(BaseModel):
    service: str
    fault: str


@app.post("/api/live/break")
def live_break(body: BreakRequest, request: Request):
    lab = _lab(request)
    try:
        incident = lab.submit(body.service, body.fault, _visitor(request))
    except ValueError as error:
        raise HTTPException(400, str(error))
    except (OverflowError, PermissionError) as error:
        raise HTTPException(429, str(error))
    return {"id": incident.id, "position": len(lab.queue_view())}


@app.get("/api/live/events")
def live_events(request: Request):
    lab = _lab(request)
    watcher = lab.watch()

    def stream():
        try:
            yield f"data: {json.dumps({'type': 'hello', 'tally': lab.tally()})}\n\n"
            while True:
                try:
                    event = watcher.get(timeout=1.0)
                except queue.Empty:
                    rows = lab.telemetry.rows
                    event = {"type": "tick", "time": rows[-1][0], "row": rows[-1][1]} if rows else {"type": "tick"}
                yield f"data: {json.dumps(event, default=str)}\n\n"
        finally:
            lab.unwatch(watcher)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class Feedback(BaseModel):
    incident_id: str = Field(max_length=32)
    vote: str = Field(pattern="^(up|down)$")
    comment: str = Field(default="", max_length=500)


@app.post("/api/live/feedback")
def live_feedback(body: Feedback, request: Request):
    _lab(request)
    path = ROOT / "runs" / "live" / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **body.model_dump()}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"ok": True}


@app.post("/webhook/alertmanager", status_code=202)
async def alertmanager(request: Request):
    lab = _lab(request)
    token = os.environ.get("WEBHOOK_TOKEN")
    if not token:
        raise HTTPException(404, "webhook disabled: set WEBHOOK_TOKEN to enable it")
    given = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or request.query_params.get("token")
    if not hmac.compare_digest(given or "", token):
        raise HTTPException(401, "bad token")
    try:
        alert = parse_alertmanager(await request.json())
        incident = lab.submit_alert(alert["start"], alert["alerts"])
    except (ValueError, KeyError) as error:
        raise HTTPException(400, f"not an Alertmanager payload with firing alerts: {error}")
    except OverflowError as error:
        raise HTTPException(429, str(error))
    return {"id": incident.id, "anchor": alert["start"], "alerts": len(alert["alerts"])}


@app.get("/api/live/slack/{incident_id}")
def live_slack(incident_id: str, request: Request):
    lab = _lab(request)
    if not incident_id.isalnum() or not lab.records.exists():
        raise HTTPException(404, "unknown incident")
    for line in open(lab.records):
        record = json.loads(line)
        if record["id"] == incident_id:
            return slack_message(record)
    raise HTTPException(404, "unknown incident")


STUDY = ROOT / "runs" / "study"


def _study_plan() -> dict:
    path = STUDY / "plan.json"
    if not path.exists():
        raise HTTPException(404, "no study configured on this server")
    return json.loads(path.read_text())


@app.get("/api/study/{slot}")
def study_slot(slot: int):
    plan = _study_plan()
    if not 0 <= slot < len(plan["slots"]):
        raise HTTPException(404, "unknown slot")
    return {"slot": slot, "cases": plan["slots"][slot]}


@app.get("/api/study/case/{case_id}")
def study_case(case_id: str, tool: bool = False):
    if case_id not in _study_plan()["cases"]:
        raise HTTPException(404, "unknown case")
    bundle = json.loads((STUDY / "cases" / f"{case_id}.json").read_text())
    if not tool:
        bundle.pop("trail", None)
        bundle.pop("causes", None)
    return bundle


class StudyAnswer(BaseModel):
    participant: str = Field(pattern="^[A-Za-z0-9-]{2,16}$")
    slot: int = Field(ge=0, le=31)
    case_id: str = Field(max_length=32)
    tool: bool
    answer: str = Field(max_length=64)
    confidence: int = Field(ge=1, le=5)
    seconds: float = Field(ge=0, le=7200)


class StudyExit(BaseModel):
    participant: str = Field(pattern="^[A-Za-z0-9-]{2,16}$")
    would_use: int = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=1000)
    quote_ok: bool = False
    display_name: str = Field(default="", max_length=60)


def _append(name: str, entry: dict):
    STUDY.mkdir(parents=True, exist_ok=True)
    with open(STUDY / name, "a") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}) + "\n")


@app.post("/api/study/answer")
def study_answer(body: StudyAnswer):
    if body.case_id not in _study_plan()["cases"]:
        raise HTTPException(404, "unknown case")
    _append("answers.jsonl", body.model_dump())
    return {"ok": True}


@app.post("/api/study/exit")
def study_exit(body: StudyExit):
    entry = body.model_dump()
    if not entry["quote_ok"]:
        entry["display_name"] = ""          # a name is kept only with permission to quote
    _append("exit.jsonl", entry)
    return {"ok": True}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def console():
    return FileResponse(STATIC / "console.html")
