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

The visitor can also pick which model investigates, from those the lab was
started with (Muse Spark Open by default; Muse Glimmer fine-tuned, on a rented
GPU, when enabled). A model can carry its own daily cap, and an optional
`warm` hook is called when an incident for a non-default model is queued, so a
server that scales to zero starts while the fault settles and the alarm fires.
"""

import hashlib
import ipaddress
import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from src.anchor import abnormal_rows, sustained_flag
from src.config import DEFAULT
from src.evaluate_explain import ranked_services
from src.lab.faults import FAULTS, TARGETS
from src.tools import Investigation

RECORDS = Path(__file__).resolve().parents[2] / "runs" / "live" / "incidents.jsonl"
DEFAULT_MODEL = "spark-open"


def visitor_key(address: str) -> str:
    """Who a request comes from, for rate limits: an IPv4 address, or an IPv6 host's whole /64, since one
    IPv6 connection gets billions of addresses to rotate through."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return address
    if ip.version == 4:
        return str(ip)
    if ip.ipv4_mapped is not None:
        return str(ip.ipv4_mapped)
    return str(ipaddress.ip_network(f"{ip}/64", strict=False))


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
    source: str = "visitor"                        # "visitor" (a fault on the menu) or "alertmanager"
    trail: list = field(default_factory=list)      # the agent's steps, so live incidents can be replayed
    handoff: dict = field(default_factory=dict)    # ticket note and customer update drafts
    model: str = DEFAULT_MODEL                     # which model investigates
    graph: list = field(default_factory=list)      # calls seen in the traces, for the blast radius
    remediation: dict = field(default_factory=dict)  # the proposed rollback, whether a human approved, recovery

    def public(self) -> dict:
        d = asdict(self)
        d.pop("visitor")
        return d


class Lab:
    def __init__(self, telemetry, injector, investigate, cfg=None, records: Path = RECORDS,
                 clock=time.time, sleep=time.sleep, handoff=None, notify=None, models=None, warm=None):
        self.telemetry, self.injector, self.investigate = telemetry, injector, investigate
        self.handoff, self.notify = handoff, notify      # optional: draft writers and a notifier (Slack)
        self.models = models or {DEFAULT_MODEL: None}    # model -> its daily cap (None: the shared budget only)
        self.warm = warm
        self.cfg = cfg or DEFAULT.live
        self.records, self.clock, self.sleep = Path(records), clock, sleep
        self.model_uses = self._uses_today()      # read back from the records, so a restart keeps the caps
        self.approved: set[str] = set()
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
        return hashlib.sha256(visitor_key(address).encode()).hexdigest()[:12]

    def submit(self, service: str, fault: str, address: str, model: str = DEFAULT_MODEL) -> Incident:
        if service not in TARGETS or fault not in FAULTS or model not in self.models:
            raise ValueError("not on the menu")
        visitor, now = self.visitor_id(address), self.clock()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if len(self.waiting) >= self.cfg.queue_max:
                raise OverflowError("the queue is full, try again in a few minutes")
            if now - self.last_request.get(visitor, -1e18) < self.cfg.visitor_cooldown_s:
                raise PermissionError("one break every ten minutes per visitor")
            used = self.model_uses.get(model, (day, 0))
            used = used if used[0] == day else (day, 0)
            cap = self.models[model]
            if cap is not None and used[1] >= cap:
                raise OverflowError("today's outages for this model are used up; pick the other one")
            self.model_uses[model] = (day, used[1] + 1)
            incident = Incident(uuid.uuid4().hex[:10], service, fault, visitor, times={"queued": now}, model=model)
            self.waiting.append(incident)
            self.last_request[visitor] = now
        if self.warm is not None and model != DEFAULT_MODEL:
            self.warm(model)
        self.publish({"type": "queue", "queue": self.queue_view()})
        return incident

    def _uses_today(self) -> dict[str, tuple[str, int]]:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        counts: dict[str, int] = {}
        if self.records.exists():
            for line in open(self.records):
                r = json.loads(line)
                queued = r.get("times", {}).get("queued")
                if queued and datetime.fromtimestamp(queued, timezone.utc).strftime("%Y-%m-%d") == day:
                    model = r.get("model", DEFAULT_MODEL)
                    counts[model] = counts.get(model, 0) + 1
        return {m: (day, n) for m, n in counts.items()}

    def models_left(self) -> dict:
        """model -> outages left today under its cap (None: no cap of its own)."""
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            return {m: None if cap is None else
                    cap - (self.model_uses[m][1] if self.model_uses.get(m, ("", 0))[0] == day else 0)
                    for m, cap in self.models.items()}

    def submit_alert(self, start: float, alerts: list[dict]) -> Incident:
        """Queue an investigation anchored at an external alert's start; nothing is injected."""
        with self._lock:
            if len(self.waiting) >= self.cfg.queue_max:
                raise OverflowError("the queue is full")
            names = ", ".join(sorted({a["name"] for a in alerts}))[:120]
            incident = Incident(uuid.uuid4().hex[:10], "unknown", names, "alertmanager", source="alertmanager",
                                times={"queued": self.clock(), "anomaly_start": start})
            self.waiting.append(incident)
        self.publish({"type": "queue", "queue": self.queue_view()})
        return incident

    def queue_view(self) -> list[dict]:
        with self._lock:
            return [{"id": i.id, "service": i.service, "fault": i.fault, "model": i.model} for i in self.waiting]

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
            self.publish({"type": "queue", "queue": self.queue_view()})   # it left the queue: watchers update
            self.run(incident)

    def _budget_left(self) -> bool:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.investigations_today[0] != day:
            self.investigations_today = (day, 0)
        return self.investigations_today[1] < self.cfg.daily_investigations

    def run(self, incident: Incident):
        if incident.source == "alertmanager":
            return self._run_alert(incident)
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
        rolled_back = False
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
                rolled_back = self._remediate(incident, active, history)
        finally:
            if not rolled_back:            # nothing approved or proposed: the lab restores the service itself
                self.injector.revert(active)
            self.last_recovered = self.clock()
            self._phase(incident, "recovered")
            self.current = None
            self._record(incident)

    # -- remediation ----------------------------------------------------------------
    # The lab's change log is the fault it applied, the way a deploy history records changes. A rollback
    # is proposed only when the diagnosed service has a recorded change; it runs only on a human's approval
    # (or, after approval_timeout_s, as the lab's own restore), and recovery is verified by the alarm rule.

    def approve(self, incident_id: str) -> bool:
        """A human approves the proposed rollback for the incident awaiting it."""
        with self._lock:
            current = self.current
            if current is None or current.id != incident_id or current.remediation.get("status") != "proposed":
                return False
            self.approved.add(incident_id)
            return True

    def _remediate(self, incident: Incident, active, history) -> bool:
        """Propose, await approval, roll back and verify. Returns whether this rolled the fault back."""
        cause = incident.causes[0] if incident.causes else None
        changes = {active.service: active.fault}      # the change log: what the lab changed, and how
        if cause is None or cause.get("verified") is False:
            incident.remediation = {"status": "none", "reason": "no verified diagnosis to act on"}
            return False
        target = cause["service"]
        if target not in changes:
            incident.remediation = {"status": "none", "target": target,
                                    "reason": f"no recorded change on {target}, so nothing safe to roll back"}
            return False
        incident.remediation = {"status": "proposed", "action": "rollback", "target": target,
                                "change": changes[target], "change_description": FAULTS[changes[target]],
                                "approval_timeout_s": self.cfg.approval_timeout_s}
        self._phase(incident, "fixing")
        deadline = self.clock() + self.cfg.approval_timeout_s
        while self.clock() < deadline and incident.id not in self.approved:
            self.sleep(1)
        approved = incident.id in self.approved
        self.approved.discard(incident.id)
        self.injector.revert(active)
        fixed_at = self.clock()
        incident.remediation.update(status="approved" if approved else "timed out", approved=approved,
                                    rolled_back_at=fixed_at)
        self.publish({"type": "remediation", "id": incident.id, "remediation": incident.remediation})
        recovered = self._await_recovery(history, fixed_at)
        incident.remediation["recovered_s"] = None if recovered is None else round(recovered - fixed_at, 1)
        self.publish({"type": "remediation", "id": incident.id, "remediation": incident.remediation})
        return True

    def _await_recovery(self, history, since: float) -> float | None:
        """When the last persist_s seconds are all normal again, by the alarm's own rule; None on timeout."""
        cfg = self.cfg
        while self.clock() < since + cfg.recovery_timeout_s:
            self.sleep(1)
            stream = self.telemetry.frame(since, self.clock())
            columns = [c for c in history.columns if c in stream.columns]
            if len(stream) < cfg.persist_s or not columns:
                continue
            recent = stream[columns].to_numpy()[-cfg.persist_s:]
            if not abnormal_rows(history[columns].to_numpy(), recent, cfg.threshold_sigma, cfg.rel_std_floor).any():
                return self.clock()
        return None

    def _run_alert(self, incident: Incident):
        """An external alert: wait for the evidence window after its start, then investigate."""
        self.current = incident
        start = incident.times["anomaly_start"]
        try:
            self._phase(incident, "alarm", detect_s=None)
            while self.clock() < start + self.cfg.window_s:
                self.sleep(1)
            self._diagnose(incident, None, start)
        finally:
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

    def _diagnose(self, incident: Incident, history_end: float | None, alarm: float):
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

        def on_step(step):
            incident.trail.append(step)
            self.publish({"type": "step", "id": incident.id, "step": step})

        run = self.investigate(inv, on_step=on_step, model=incident.model)
        try:
            incident.graph = inv.call_graph()
        except Exception:   # the graph only feeds a display; never lose the diagnosis over it
            incident.graph = []
        incident.causes = run.ranking.causes
        services = ranked_services(run.ranking.services or [])
        incident.verdict = {"detected": True, "investigated": True, "error": run.ranking.error,
                            "investigation_s": round(self.clock() - started, 1), "turns": run.turns}
        if incident.source == "visitor":   # only a visitor's fault has a known answer to score against
            incident.verdict.update(top1=bool(services) and services[0] == incident.service,
                                    top3=incident.service in services[:3])
        self._phase(incident, "diagnosed", verdict=incident.verdict)
        if self.handoff is not None and incident.causes:
            try:
                incident.handoff = self.handoff(incident.causes, incident.trail, model=incident.model)
            except Exception as error:   # drafts are a bonus; never lose the diagnosis over them
                incident.handoff = {"error": f"{type(error).__name__}: {error}"}
            self.publish({"type": "handoff", "id": incident.id, "handoff": incident.handoff})
        if self.notify is not None:
            try:
                self.notify(incident.public())
            except Exception:
                pass

    def _record(self, incident: Incident):
        self.records.parent.mkdir(parents=True, exist_ok=True)
        with open(self.records, "a") as f:
            f.write(json.dumps(incident.public()) + "\n")
        self.publish({"type": "tally", "tally": self.tally()})

    def tally(self) -> dict:
        rows = [json.loads(l) for l in open(self.records)] if self.records.exists() else []
        rows = [r for r in rows if r.get("source", "visitor") == "visitor"]   # alerts have no known answer
        investigated = [r for r in rows if r["verdict"].get("investigated")]
        return {"incidents": len(rows), "detected": sum(r["verdict"].get("detected", False) for r in rows),
                "investigated": len(investigated), "top1": sum(r["verdict"].get("top1", False) for r in investigated),
                "top3": sum(r["verdict"].get("top3", False) for r in investigated)}
