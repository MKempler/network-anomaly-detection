"""Runs live incidents one at a time: queue, inject, alarm, investigate, recover, record.

A visitor picks a service and a fault. Incidents run strictly one after
another on the single shared system, so a diagnosis is never confused by
somebody else's fault. Each incident moves through these phases, each published
to every watcher as it happens:

  queued -> settling -> injected -> alarm (or missed) -> investigating -> diagnosed -> recovered

The alarm uses the same sustained-deviation rule as the end-to-end evaluation,
judged against the minutes just before the fault. The agent never learns what
was injected; the verdict compares its top-ranked service with the injected
one afterwards. Every incident is appended to runs/live/incidents.jsonl, the
public tally.
"""

import hashlib
import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from src.anchor import sustained_flag
from src.config import DEFAULT
from src.evaluate_explain import ranked_services
from src.lab.faults import FAULTS, TARGETS
from src.tools import Investigation

RECORDS = Path(__file__).resolve().parents[2] / "runs" / "live" / "incidents.jsonl"


@dataclass
class Incident:
    id: str
    service: str
    fault: str
    visitor: str
    phase: str = "queued"
    times: dict = field(default_factory=dict)      # phase -> unix time
    causes: list = field(default_factory=list)
    verdict: dict = field(default_factory=dict)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("visitor")
        return d


class Lab:
    def __init__(self, telemetry, injector, investigate, cfg=None, records: Path = RECORDS,
                 clock=time.time, sleep=time.sleep):
        self.telemetry, self.injector, self.investigate = telemetry, injector, investigate
        self.cfg = cfg or DEFAULT.live
        self.records, self.clock, self.sleep = Path(records), clock, sleep
        self.waiting: deque[Incident] = deque()
        self.current: Incident | None = None
        self.last_request: dict[str, float] = {}
        self.investigations_today: tuple[str, int] = ("", 0)
        self.watchers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.last_recovered = clock()

    # -- requests ---------------------------------------------------------------

    @staticmethod
    def visitor_id(address: str) -> str:
        return hashlib.sha256(address.encode()).hexdigest()[:12]

    def submit(self, service: str, fault: str, address: str) -> Incident:
        if service not in TARGETS or fault not in FAULTS:
            raise ValueError("not on the menu")
        visitor, now = self.visitor_id(address), self.clock()
        with self._lock:
            if len(self.waiting) >= self.cfg.queue_max:
                raise OverflowError("the queue is full, try again in a few minutes")
            if now - self.last_request.get(visitor, -1e18) < self.cfg.visitor_cooldown_s:
                raise PermissionError("one break every ten minutes per visitor")
            incident = Incident(uuid.uuid4().hex[:10], service, fault, visitor, times={"queued": now})
            self.waiting.append(incident)
            self.last_request[visitor] = now
        self.publish({"type": "queue", "queue": self.queue_view()})
        return incident

    def queue_view(self) -> list[dict]:
        with self._lock:
            return [{"id": i.id, "service": i.service, "fault": i.fault} for i in self.waiting]

    # -- watchers ---------------------------------------------------------------

    def watch(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        self.watchers.append(q)
        return q

    def unwatch(self, q: queue.Queue):
        if q in self.watchers:
            self.watchers.remove(q)

    def publish(self, event: dict):
        for q in list(self.watchers):
            try:
                q.put_nowait(event)
            except queue.Full:
                self.unwatch(q)

    def _phase(self, incident: Incident, phase: str, **detail):
        incident.phase = phase
        incident.times[phase] = self.clock()
        self.publish({"type": "phase", "incident": incident.public(), **detail})

    # -- the run loop -------------------------------------------------------------

    def run_forever(self):
        while True:
            with self._lock:
                incident = self.waiting.popleft() if self.waiting else None
            if incident is None:
                self.sleep(1)
                continue
            self.run(incident)

    def _budget_left(self) -> bool:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.investigations_today[0] != day:
            self.investigations_today = (day, 0)
        return self.investigations_today[1] < self.cfg.daily_investigations

    def run(self, incident: Incident):
        cfg = self.cfg
        self.current = incident
        self._phase(incident, "settling")
        ready = max(self.last_recovered + cfg.settle_s, self.clock())
        while self.clock() < ready:
            self.sleep(1)
        history_end = self.clock()
        history = self.telemetry.frame(history_end - cfg.history_s, history_end)

        active = self.injector.inject(incident.service, incident.fault)
        self._phase(incident, "injected")
        try:
            alarm = self._await_alarm(history, history_end)
            if alarm is None:
                self._phase(incident, "missed")
                incident.verdict = {"detected": False}
            else:
                # "anomaly_start" is where the abnormal stretch began (the investigation's anchor);
                # the "alarm" phase time is when the persistence rule was met and the alarm was raised.
                incident.times["anomaly_start"] = alarm
                self._phase(incident, "alarm", detect_s=round(self.clock() - incident.times["injected"], 1))
                while self.clock() < alarm + cfg.window_s:
                    self.sleep(1)
                self._diagnose(incident, history_end, alarm)
        finally:
            self.injector.revert(active)
            self.last_recovered = self.clock()
            self._phase(incident, "recovered")
            self.current = None
            self._record(incident)

    def _await_alarm(self, history, history_end) -> float | None:
        cfg = self.cfg
        deadline = history_end + cfg.alarm_timeout_s
        while self.clock() < deadline:
            self.sleep(1)
            stream = self.telemetry.frame(history_end, self.clock())
            columns = [c for c in history.columns if c in stream.columns]
            if len(stream) < cfg.persist_s or not columns:
                continue
            flag = sustained_flag(history[columns].to_numpy(), stream[columns].to_numpy(),
                                  stream.index.to_numpy(), cfg.threshold_sigma, cfg.persist_s, cfg.rel_std_floor)
            if flag is not None:
                return flag
        return None

    def _diagnose(self, incident: Incident, history_end: float, alarm: float):
        cfg = self.cfg
        if not self._budget_left():
            self._phase(incident, "diagnosed", note="daily investigation budget used up")
            incident.verdict = {"detected": True, "investigated": False}
            return
        self.investigations_today = (self.investigations_today[0], self.investigations_today[1] + 1)
        # Same framing as the evaluation: the baseline is the stretch just before the alarm, and the agent is
        # never told when (or what) the fault was injected.
        case, logs, spans = self.telemetry.case(alarm - cfg.history_s, alarm + cfg.window_s, f"live-{incident.id}")
        inv_cfg = replace(DEFAULT.investigation, baseline_s=cfg.history_s, window_s=cfg.window_s)
        inv = Investigation(case, alarm, inv_cfg, logs=lambda: logs, traces=lambda: spans)
        self._phase(incident, "investigating")
        started = self.clock()
        run = self.investigate(inv, on_step=lambda step: self.publish({"type": "step", "id": incident.id,
                                                                         "step": step}))
        incident.causes = run.ranking.causes
        services = ranked_services(run.ranking.services or [])
        incident.verdict = {"detected": True, "investigated": True, "error": run.ranking.error,
                            "top1": bool(services) and services[0] == incident.service,
                            "top3": incident.service in services[:3],
                            "investigation_s": round(self.clock() - started, 1), "turns": run.turns}
        self._phase(incident, "diagnosed", verdict=incident.verdict)

    def _record(self, incident: Incident):
        self.records.parent.mkdir(parents=True, exist_ok=True)
        with open(self.records, "a") as f:
            f.write(json.dumps(incident.public()) + "\n")
        self.publish({"type": "tally", "tally": self.tally()})

    def tally(self) -> dict:
        rows = [json.loads(l) for l in open(self.records)] if self.records.exists() else []
        investigated = [r for r in rows if r["verdict"].get("investigated")]
        return {"incidents": len(rows), "detected": sum(r["verdict"].get("detected", False) for r in rows),
                "investigated": len(investigated), "top1": sum(r["verdict"].get("top1", False) for r in investigated),
                "top3": sum(r["verdict"].get("top3", False) for r in investigated)}
