"""Load RCAEval failure cases.

On disk a case lives at `<suite>/<suite>-<system>/<service>_<fault>/<n>/`,
so the path itself names the root cause. Three rules follow from that:

* `Case` carries an opaque id and observable telemetry only. Nothing a model,
  tool, or prompt receives can reveal the path.
* Ground truth (root-cause service, fault, inject time) comes from
  `load_labels`, which only the evaluator and the window-anchoring code call.
* Only an allowlist of telemetry files is ever opened. Some RE3 cases also ship
  `root_cause.txt` (the root-cause log line) and `metrics.png`; neither is read.
"""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

RCAEVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "rcaeval"
SUITES = ("RE1", "RE2", "RE3")
SYSTEM_NAMES = {"OB": "Online Boutique", "SS": "Sock Shop", "TT": "Train Ticket"}
# RE2/RE3 also ship a fuller metrics.csv, but RCAEval's own evaluation reads
# simple_metrics.csv for them; every ranker here sees the same files it does.
METRICS_FILE = {"RE1": "data.csv", "RE2": "simple_metrics.csv", "RE3": "simple_metrics.csv"}
LOGS_FILE = "logs.csv"
TRACES_FILE = "traces.csv"

# Folder names are <service>_<fault>, and one RE3 folder adds a variant suffix
# (ts-route-service_f3_1). Faults: RE1/RE2 resource faults, RE3 code faults f1-f5.
FOLDER = re.compile(r"^(?P<service>.+?)_(?P<fault>cpu|mem|disk|delay|loss|socket|f[1-5])(?:_\d+)?$")

_case_dirs: dict[str, Path] = {}  # opaque id -> directory, filled by case_index


@dataclass(frozen=True)
class Case:
    case_id: str              # opaque, e.g. "re1-4f1c2a9b0d"
    suite: str
    system: str               # "OB", "SS" or "TT"; known to any operator
    metrics: pd.DataFrame     # index: unix seconds; columns: "<service>_<metric>"
    services: dict[str, str]  # metric column -> service
    sources: frozenset[str]   # observable sources on disk: metrics, logs, traces

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


def parse_folder(name: str) -> tuple[str, str]:
    match = FOLDER.match(name)
    if not match:
        raise ValueError(f"unrecognised case folder {name!r}")
    return match["service"], match["fault"]


def case_index(suite: str, root: Path = RCAEVAL_DIR) -> dict[str, Path]:
    """Opaque case id -> case directory, in a stable order.

    Cases sit at exactly <system>/<service>_<fault>/<instance number>/. The
    RE2-OB archive also contains a stray copy of RCAEval's multi-source sample
    at that depth, which the numeric-instance rule leaves out.
    """
    suite_dir = root / suite
    dirs = sorted(p.parent for p in suite_dir.glob("*/*/*/inject_time.txt") if p.parent.name.isdigit())
    index = {case_id_for(suite, d.relative_to(suite_dir).as_posix()): d for d in dirs}
    if any(_case_dirs.get(case_id, d) != d for case_id, d in index.items()):
        # Same relative layout under a different root (tests): drop cached telemetry.
        load_logs.cache_clear()
        load_traces.cache_clear()
    _case_dirs.update(index)
    return index


def _dir(case_id: str) -> Path:
    if case_id not in _case_dirs:
        case_index(case_id.split("-")[0].upper())
    return _case_dirs[case_id]


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
    # Same column handling as RCAEval's main.py: keep p90 latency as "latency", drop p50.
    df = df.loc[:, ~df.columns.str.endswith("_latency-50")]
    df = df.rename(columns={c: c.removesuffix("-90") for c in df.columns if c.endswith("_latency-90")})
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    # Some Train Ticket columns are empty for the whole case; they carry no signal.
    return df.astype("float64").dropna(axis=1, how="all").ffill().bfill()


def load_case(suite: str, case_id: str, case_dir: Path) -> Case:
    system = case_dir.parent.parent.name.split("-")[-1]
    metrics = read_metrics(case_dir / METRICS_FILE[suite])
    services = {c: service_of(c) for c in metrics.columns}
    unmapped = [c for c, s in services.items() if s is None]
    if unmapped:
        raise ValueError(f"{case_id}: {len(unmapped)} columns without a service, e.g. {unmapped[0]!r}")
    sources = {"metrics"} | {name for name, file in (("logs", LOGS_FILE), ("traces", TRACES_FILE))
                             if _has_rows(case_dir / file)}
    return Case(case_id, suite, system, metrics, services, frozenset(sources))


def _has_rows(path: Path) -> bool:
    """True if a CSV exists and has at least one row after the header.

    The 13 RE3 Sock Shop cases that ship traces.csv ship it header-only.
    """
    if not path.exists():
        return False
    with open(path) as f:
        f.readline()
        return bool(f.readline().strip())


def iter_cases(suite: str, root: Path = RCAEVAL_DIR) -> Iterator[Case]:
    """Yield cases one at a time; a full suite does not fit comfortably in memory."""
    for case_id, case_dir in case_index(suite, root).items():
        yield load_case(suite, case_id, case_dir)


@lru_cache(maxsize=2)
def load_logs(case_id: str) -> pd.DataFrame:
    """Log lines: time (unix seconds), service, message, and level where recorded."""
    raw = pd.read_csv(_dir(case_id) / LOGS_FILE, usecols=lambda c: c in {"timestamp", "container_name",
                                                                          "message", "level"})
    logs = pd.DataFrame({"time": raw["timestamp"] / 1e9, "service": raw["container_name"],
                         "message": raw["message"].fillna("").astype(str)})
    if "level" in raw:
        logs["level"] = raw["level"]
    return logs.sort_values("time", kind="stable").reset_index(drop=True)


@lru_cache(maxsize=2)
def load_traces(case_id: str) -> pd.DataFrame:
    """Spans: start (unix seconds), ids, service, operation, duration (us), status code."""
    raw = pd.read_csv(_dir(case_id) / TRACES_FILE, usecols=["traceID", "spanID", "parentSpanID", "serviceName",
                                                             "operationName", "startTime", "duration", "statusCode"])
    return pd.DataFrame({
        "start": raw["startTime"] / 1e6, "trace_id": raw["traceID"], "span_id": raw["spanID"],
        "parent_span_id": raw["parentSpanID"], "service": raw["serviceName"],
        "operation": raw["operationName"], "duration_us": raw["duration"], "status_code": raw["statusCode"],
    }).sort_values("start", kind="stable").reset_index(drop=True)


def load_labels(suite: str, root: Path = RCAEVAL_DIR) -> dict[str, CaseLabels]:
    """Ground truth per case id. Evaluation and window anchoring only."""
    labels = {}
    for case_id, case_dir in case_index(suite, root).items():
        service, fault = parse_folder(case_dir.parent.name)
        inject_time = int((case_dir / "inject_time.txt").read_text().strip())
        labels[case_id] = CaseLabels(service, fault, inject_time)
    return labels
