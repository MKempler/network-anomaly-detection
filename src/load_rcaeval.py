"""Load RCAEval failure cases.

On disk a case lives at `<suite>/<suite>-<system>/<service>_<fault>/<n>/`,
so the path itself names the root cause. Two rules follow from that:

* `Case` carries an opaque id and observable telemetry only. Nothing a model,
  tool, or prompt receives can reveal the path.
* Ground truth (root-cause service, fault, inject time) comes from
  `load_labels`, which only the evaluator and the window-anchoring code call.
"""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RCAEVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "rcaeval"
SUITES = ("RE1", "RE2", "RE3")
SYSTEM_NAMES = {"OB": "Online Boutique", "SS": "Sock Shop", "TT": "Train Ticket"}
METRICS_FILE = {"RE1": "data.csv"}


@dataclass(frozen=True)
class Case:
    case_id: str            # opaque, e.g. "re1-4f1c2a9b0d"
    suite: str
    system: str             # "OB", "SS" or "TT"; known to any operator
    metrics: pd.DataFrame   # index: unix seconds; columns: "<service>_<metric>"
    services: dict[str, str]  # metric column -> service

    @property
    def candidate_services(self) -> list[str]:
        return sorted(set(self.services.values()))


@dataclass(frozen=True)
class CaseLabels:
    root_service: str
    fault: str
    inject_time: int


def case_id_for(suite: str, rel_path: str) -> str:
    """Stable opaque id from the case's path relative to its suite folder."""
    return f"{suite.lower()}-{hashlib.sha256(rel_path.encode()).hexdigest()[:10]}"


def case_index(suite: str, root: Path = RCAEVAL_DIR) -> dict[str, Path]:
    """Opaque case id -> case directory, in a stable order."""
    if suite not in METRICS_FILE:
        raise NotImplementedError(f"{suite}: layout not wired up yet")
    suite_dir = root / suite
    dirs = sorted(p.parent for p in suite_dir.rglob("inject_time.txt"))
    return {case_id_for(suite, d.relative_to(suite_dir).as_posix()): d for d in dirs}


def service_of(column: str) -> str | None:
    """`ts-auth-service_container-cpu-usage-seconds-total` -> `ts-auth-service`."""
    service, sep, metric = column.rpartition("_")
    return service if sep and service and metric else None


def read_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Online Boutique files repeat the time column mid-file (pandas reads it as "time.1").
    df = df.drop(columns=[c for c in df.columns if c.startswith("time.")])
    # Six RE1 cases end with rows that have no timestamp (8 to 262 rows, all
    # after the inject point). They can't be placed in time, so drop them.
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = df.index.astype("int64")
    # Some Train Ticket columns are empty for the whole case; they carry no signal.
    return df.astype("float64").dropna(axis=1, how="all").ffill().bfill()


def load_case(suite: str, case_id: str, case_dir: Path) -> Case:
    system = case_dir.parent.parent.name.split("-")[-1]
    metrics = read_metrics(case_dir / METRICS_FILE[suite])
    services = {c: service_of(c) for c in metrics.columns}
    unmapped = [c for c, s in services.items() if s is None]
    if unmapped:
        raise ValueError(f"{case_id}: {len(unmapped)} columns without a service, e.g. {unmapped[0]!r}")
    return Case(case_id, suite, system, metrics, services)


def iter_cases(suite: str, root: Path = RCAEVAL_DIR) -> Iterator[Case]:
    """Yield cases one at a time; a full suite does not fit comfortably in memory."""
    for case_id, case_dir in case_index(suite, root).items():
        yield load_case(suite, case_id, case_dir)


def load_labels(suite: str, root: Path = RCAEVAL_DIR) -> dict[str, CaseLabels]:
    """Ground truth per case id. Evaluation and window anchoring only."""
    labels = {}
    for case_id, case_dir in case_index(suite, root).items():
        service, _, fault = case_dir.parent.name.rpartition("_")
        inject_time = int((case_dir / "inject_time.txt").read_text().strip())
        labels[case_id] = CaseLabels(service, fault, inject_time)
    return labels
