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

Everything served comes from the committed records under runs/, imported into
the database at startup, so the console needs no raw benchmark data.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.anchor import detector_anchor
from src.config import DEFAULT
from src.db import DetectionRun, EvalResult, setup
from src.load_rcaeval import Case, service_of

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
REPLAYS = ROOT / "runs" / "replay"
SUITES = ("RE1", "RE2", "RE3")
HIDDEN_UNTIL_REVEAL = ("answer", "outcome")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = setup()
    yield


app = FastAPI(title="Incident investigator", lifespan=lifespan)


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


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def console():
    return FileResponse(STATIC / "console.html")
