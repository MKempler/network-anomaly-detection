"""Storage for recorded results: Postgres in Docker, SQLite for local work.

Three tables, created on startup and filled from the committed records under
runs/, so the database is always a view of what the repo can reproduce:

- detection_runs: one row per detector run on SMD (event-level scores)
- eval_results: one row per explanation run (ranker, suite, AC@k, intervals)
- explanations: one row per replayable case (ranked causes, hit or miss)

Importing is idempotent: rows are keyed by run id or (suite, ranker, case), so
restarting the API never duplicates anything.
"""

import json
import os
from pathlib import Path

from sqlalchemy import JSON, Boolean, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DEFAULT_URL = f"sqlite:///{ROOT / 'data' / 'app.db'}"


class Base(DeclarativeBase):
    pass


class DetectionRun(Base):
    __tablename__ = "detection_runs"
    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    detector: Mapped[str] = mapped_column(String(32))
    config_id: Mapped[str] = mapped_column(String(16))
    git_commit: Mapped[str] = mapped_column(String(16))
    git_dirty: Mapped[bool] = mapped_column(Boolean)
    servers: Mapped[int] = mapped_column(Integer)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    f1: Mapped[float] = mapped_column(Float)
    f1_ci: Mapped[list | None] = mapped_column(JSON)
    mean_delay: Mapped[float | None] = mapped_column(Float)
    timestamp: Mapped[str] = mapped_column(String(32))


class EvalResult(Base):
    __tablename__ = "eval_results"
    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    suite: Mapped[str] = mapped_column(String(8))
    part: Mapped[str] = mapped_column(String(8))
    anchor: Mapped[str] = mapped_column(String(16))
    ranker: Mapped[str] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(64))
    config_id: Mapped[str] = mapped_column(String(16))
    git_commit: Mapped[str] = mapped_column(String(16))
    git_dirty: Mapped[bool] = mapped_column(Boolean)
    cases: Mapped[int] = mapped_column(Integer)
    failed: Mapped[int] = mapped_column(Integer)
    ac1: Mapped[float] = mapped_column(Float)
    ac3: Mapped[float] = mapped_column(Float)
    avg5: Mapped[float] = mapped_column(Float)
    ci95: Mapped[dict] = mapped_column(JSON)
    timestamp: Mapped[str] = mapped_column(String(32))


class Explanation(Base):
    __tablename__ = "explanations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suite: Mapped[str] = mapped_column(String(8), index=True)
    ranker: Mapped[str] = mapped_column(String(32))
    case_id: Mapped[str] = mapped_column(String(32), index=True)
    system: Mapped[str] = mapped_column(String(8))
    causes: Mapped[list] = mapped_column(JSON)
    top1: Mapped[bool] = mapped_column(Boolean)
    top3: Mapped[bool] = mapped_column(Boolean)


def engine(url: str | None = None):
    url = url or os.environ.get("DATABASE_URL", DEFAULT_URL)
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, future=True)


def _jsonl(path: Path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


def import_runs(session: Session, runs: Path = RUNS) -> dict:
    """Load every committed record into the database; returns counts of new rows."""
    added = {"detection_runs": 0, "eval_results": 0, "explanations": 0}
    for r in _jsonl(runs / "detection.jsonl"):
        if session.get(DetectionRun, r["run_id"]):
            continue
        t = r["total"]
        session.add(DetectionRun(run_id=r["run_id"], detector=r["detector"], config_id=r["config_id"],
                                 git_commit=r["git_commit"], git_dirty=r["git_dirty"], servers=len(r["machines"]),
                                 precision=t["precision"], recall=t["recall"], f1=t["f1"],
                                 f1_ci=(r.get("ci95") or {}).get("f1"), mean_delay=t["mean_delay"],
                                 timestamp=r["timestamp"]))
        added["detection_runs"] += 1
    for r in _jsonl(runs / "explain_summary.jsonl"):
        if session.get(EvalResult, r["run_id"]):
            continue
        s = r["summary"]
        session.add(EvalResult(run_id=r["run_id"], suite=r["suite"], part=r["part"], anchor=r["anchor"],
                               ranker=r["ranker"], model=r.get("model"), config_id=r["config_id"],
                               git_commit=r["git_commit"], git_dirty=r["git_dirty"], cases=s["cases"],
                               failed=s["failed"], ac1=s["ac1"], ac3=s["ac3"], avg5=s["avg5"], ci95=r["ci95"],
                               timestamp=r["timestamp"]))
        added["eval_results"] += 1
    for index_path in sorted((runs / "replay").glob("*/index.json")):
        index = json.loads(index_path.read_text())
        known = set(session.scalars(select(Explanation.case_id).where(Explanation.suite == index["suite"],
                                                                      Explanation.ranker == index["ranker"])))
        for entry in index["cases"]:
            if entry["case_id"] in known:
                continue
            bundle = json.loads((index_path.parent / f"{entry['case_id']}.json").read_text())
            session.add(Explanation(suite=index["suite"], ranker=index["ranker"], case_id=entry["case_id"],
                                    system=bundle["system"], causes=bundle["causes"],
                                    top1=bundle["outcome"]["top1"], top3=bundle["outcome"]["top3"]))
            added["explanations"] += 1
    session.commit()
    return added


def setup(url: str | None = None):
    """Create the tables and import the committed records. Returns the engine."""
    eng = engine(url)
    Base.metadata.create_all(eng)
    with Session(eng) as session:
        import_runs(session)
    return eng
