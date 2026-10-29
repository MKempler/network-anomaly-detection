"""Live telemetry from the lab's containers, in the benchmark's shape.

Every second the recorder turns the latest Docker stats into one row of
per-service metrics named like RCAEval's simple_metrics (`<service>_cpu`,
`_mem`, `_net`, `_diskio`), adds per-service span metrics from OpenTelemetry
(`_latency` p90 in ms, `_workload` spans/s, `_error` failed spans/s), and keeps
recent logs and spans. `case(start, end)` hands any time range to the
investigation tools as an ordinary `Case` with logs and traces, so the agent
runs on live data exactly as it does on the benchmark.

The arithmetic lives in small pure functions so it is testable without Docker.
"""

import json
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from src.load_rcaeval import Case

KEEP_S = 1800   # seconds of history kept in memory
SPAN_GRACE_S = 6  # spans reach the file 1-6 s after they end; a second's row waits this long


def cpu_percent(stats: dict) -> float:
    """Docker's CPU % for one stats sample (100 = one full core)."""
    cpu, pre = stats.get("cpu_stats", {}), stats.get("precpu_stats", {})
    delta = cpu.get("cpu_usage", {}).get("total_usage", 0) - pre.get("cpu_usage", {}).get("total_usage", 0)
    system = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
    cores = cpu.get("online_cpus") or len(cpu.get("cpu_usage", {}).get("percpu_usage") or []) or 1
    return 100.0 * delta / system * cores if delta > 0 and system > 0 else 0.0


def memory_bytes(stats: dict) -> float:
    mem = stats.get("memory_stats", {})
    cache = mem.get("stats", {}).get("inactive_file", 0)
    return float(max(mem.get("usage", 0) - cache, 0))


def net_bytes(stats: dict) -> float:
    return float(sum(v.get("rx_bytes", 0) + v.get("tx_bytes", 0) for v in (stats.get("networks") or {}).values()))


def disk_bytes(stats: dict) -> float:
    entries = (stats.get("blkio_stats") or {}).get("io_service_bytes_recursive") or []
    return float(sum(e.get("value", 0) for e in entries))


# RCAEval measured every service from outside, through a sidecar proxy, so a
# service's latency included what its callers experienced. The lab does the
# same: a caller's client span to "hipstershop.CartService/GetCart" is how long
# cartservice took to answer as seen from outside, and becomes a cartservice
# span too. That matters for three cases:
# - services Online Boutique doesn't instrument (cartservice, adservice)
# - faults on the network path, which a service's own spans never see
# - a frozen service, which emits nothing while its callers time out
# A service's own server spans are kept alongside.
CLIENT_CALL = re.compile(r"^/?hipstershop\.(\w+)/(\w+)$")


def peer_span(span: dict) -> dict | None:
    """The caller-observed view of a client span, attributed to the service it called, or None."""
    m = CLIENT_CALL.match(span["operation"])
    if not m:
        return None
    callee = m.group(1).lower()               # ShippingService -> shippingservice
    if span["service"] == callee:             # the callee's own server span, not a call to it
        return None
    return {**span, "service": callee, "operation": m.group(2),
            "span_id": f"{span['span_id']}:peer", "parent_span_id": span["span_id"]}


def parse_spans(line: str) -> list[dict]:
    """One OTLP JSON export line -> spans as dicts in the trace loader's columns."""
    out = []
    for rs in json.loads(line).get("resourceSpans", []):
        attrs = {a["key"]: a["value"].get("stringValue") for a in rs.get("resource", {}).get("attributes", [])}
        service = attrs.get("service.name", "unknown")
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                start, end = int(sp["startTimeUnixNano"]), int(sp["endTimeUnixNano"])
                out.append({"start": start / 1e9, "end": end / 1e9, "trace_id": sp.get("traceId"),
                            "span_id": sp.get("spanId"), "parent_span_id": sp.get("parentSpanId") or None,
                            "service": service, "operation": sp.get("name", ""),
                            "duration_us": (end - start) / 1e3,
                            # OTLP status code 2 is ERROR; map onto gRPC-style codes (0 OK, 2 UNKNOWN)
                            "status_code": 2.0 if (sp.get("status") or {}).get("code") == 2 else 0.0})
    return out + [p for p in map(peer_span, out) if p is not None]


def span_metrics(spans: list[dict], second: int) -> dict[str, float]:
    """Per-service latency p90 (ms), workload and errors for spans ending in `second`."""
    by_service = defaultdict(list)
    for s in spans:
        if int(s["end"]) == second:
            by_service[s["service"]].append(s)
    row = {}
    for service, group in by_service.items():
        durations = np.array([g["duration_us"] for g in group]) / 1000.0
        row[f"{service}_latency"] = float(np.percentile(durations, 90))
        row[f"{service}_workload"] = float(len(group))
        row[f"{service}_error"] = float(sum(g["status_code"] != 0 for g in group))
    return row


class Telemetry:
    """Background recorder over the Docker engine and the collector's span file."""

    def __init__(self, docker_client, spans_file: Path, label: str = "lab.service"):
        self.docker, self.spans_file, self.label = docker_client, Path(spans_file), label
        self.rows: deque[tuple[int, dict]] = deque(maxlen=KEEP_S)
        self.spans: deque[dict] = deque(maxlen=400_000)
        self.logs: deque[dict] = deque(maxlen=200_000)
        self.latest: dict[str, dict] = {}
        self._previous: dict[str, tuple[float, float]] = {}
        self._pending: dict[int, dict] = {}   # second -> container metrics, waiting for its spans
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # -- background threads ----------------------------------------------------

    def start(self):
        for container in self.docker.containers.list(filters={"label": self.label}):
            service = container.labels[self.label]
            threading.Thread(target=self._stats_loop, args=(container, service), daemon=True).start()
            threading.Thread(target=self._logs_loop, args=(container, service), daemon=True).start()
        threading.Thread(target=self._spans_loop, daemon=True).start()
        threading.Thread(target=self._tick_loop, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _stats_loop(self, container, service):
        while not self._stop.is_set():
            try:
                for stats in container.stats(stream=True, decode=True):
                    self.latest[service] = stats
                    if self._stop.is_set():
                        return
            except Exception:
                time.sleep(1)   # container paused or restarting: keep trying

    def _logs_loop(self, container, service):
        while not self._stop.is_set():
            try:
                for line in container.logs(stream=True, follow=True, since=int(time.time()), timestamps=False):
                    with self._lock:
                        self.logs.append({"time": time.time(), "service": service,
                                          "message": line.decode(errors="replace").strip()})
            except Exception:
                time.sleep(1)

    def _spans_loop(self):
        position = 0
        while not self._stop.is_set():
            try:
                position = self._read_spans(position)
            except FileNotFoundError:
                pass
            except Exception as error:   # never let one bad read stop span collection
                print(f"telemetry: span read failed: {type(error).__name__}: {error}", flush=True)
            time.sleep(0.5)

    def _read_spans(self, position: int) -> int:
        """Parse complete lines after `position`; a line still being written is left for the next pass."""
        with open(self.spans_file, "rb") as f:
            size = f.seek(0, 2)
            if size < position:              # the collector rotated the file
                position = 0
            f.seek(position)
            chunk = f.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return position
        for line in chunk[:end].split(b"\n"):
            if line.strip():
                try:
                    parsed = parse_spans(line.decode())
                except (ValueError, KeyError):
                    continue
                with self._lock:
                    self.spans.extend(parsed)
        return position + end + 1

    def _tick_loop(self):
        while not self._stop.is_set():
            now = int(time.time())
            self._pending[now] = self._container_row()
            self._finalize(now - SPAN_GRACE_S)
            time.sleep(max(0.0, 1 - (time.time() % 1)))

    def _finalize(self, second: int):
        """Join a second's container metrics with its spans, once those have had time to arrive."""
        row = self._pending.pop(second, None)
        for stale in [t for t in self._pending if t < second]:
            del self._pending[stale]
        if row is None:
            return
        with self._lock:
            row.update(span_metrics([s for s in self.spans if s["end"] >= second - 1], second))
            self.rows.append((second, row))

    def _container_row(self) -> dict:
        row = {}
        for service, stats in list(self.latest.items()):
            row[f"{service}_cpu"] = cpu_percent(stats)
            row[f"{service}_mem"] = memory_bytes(stats) / 1e6
            net, disk = net_bytes(stats), disk_bytes(stats)
            prev_net, prev_disk = self._previous.get(service, (net, disk))
            row[f"{service}_net"] = max(net - prev_net, 0.0) / 1e3
            row[f"{service}_diskio"] = max(disk - prev_disk, 0.0) / 1e3
            self._previous[service] = (net, disk)
        return row

    # -- reading ---------------------------------------------------------------

    def recent(self, seconds: float, metrics: set[str]) -> list[tuple[int, dict]]:
        """The last `seconds` of rows, cut down to `metrics`, for a page that just opened."""
        with self._lock:
            rows = list(self.rows)
        since = rows[-1][0] - seconds if rows else 0
        return [(t, {k: v for k, v in r.items() if k in metrics}) for t, r in rows if t >= since]

    def frame(self, start: float, end: float) -> pd.DataFrame:
        with self._lock:
            rows = [(t, r) for t, r in self.rows if start <= t < end]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([r for _, r in rows], index=pd.Index([t for t, _ in rows], name="time"))
        span_cols = [c for c in df.columns if c.rpartition("_")[2] in ("workload", "error")]
        df[span_cols] = df[span_cols].fillna(0.0)          # no spans that second means zero, not unknown
        return df.astype("float64").ffill().bfill().fillna(0.0)

    def case(self, start: float, end: float, case_id: str) -> tuple[Case, pd.DataFrame, pd.DataFrame]:
        """A Case for the tools, plus logs and traces in the loaders' formats, for [start, end)."""
        metrics = self.frame(start, end)
        services = {c: c.rpartition("_")[0] for c in metrics.columns}
        with self._lock:
            logs = pd.DataFrame([l for l in self.logs if start <= l["time"] < end],
                                columns=["time", "service", "message"])
            spans = pd.DataFrame([s for s in self.spans if start <= s["start"] < end],
                                 columns=["start", "trace_id", "span_id", "parent_span_id", "service",
                                          "operation", "duration_us", "status_code"])
        sources = {"metrics"} | ({"logs"} if len(logs) else set()) | ({"traces"} if len(spans) else set())
        return Case(case_id, "LIVE", "OB", metrics, services, frozenset(sources)), logs, spans
