import json

import pytest

from src.lab.faults import FAULTS, FaultInjector
from src.lab.telemetry import cpu_percent, disk_bytes, memory_bytes, net_bytes, parse_spans, span_metrics


def test_cpu_percent_matches_docker_formula():
    stats = {"cpu_stats": {"cpu_usage": {"total_usage": 3_000_000}, "system_cpu_usage": 20_000_000, "online_cpus": 4},
             "precpu_stats": {"cpu_usage": {"total_usage": 1_000_000}, "system_cpu_usage": 10_000_000}}
    assert cpu_percent(stats) == pytest.approx(100 * 2_000_000 / 10_000_000 * 4)
    assert cpu_percent({}) == 0.0


def test_memory_excludes_page_cache_and_counters_sum():
    assert memory_bytes({"memory_stats": {"usage": 500, "stats": {"inactive_file": 200}}}) == 300
    assert net_bytes({"networks": {"eth0": {"rx_bytes": 10, "tx_bytes": 5}, "eth1": {"rx_bytes": 1, "tx_bytes": 1}}}) == 17
    assert disk_bytes({"blkio_stats": {"io_service_bytes_recursive": [{"value": 7}, {"value": 3}]}}) == 10
    assert disk_bytes({"blkio_stats": {"io_service_bytes_recursive": None}}) == 0


def otlp_line(service, spans):
    return json.dumps({"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": service}}]},
        "scopeSpans": [{"spans": spans}]}]})


def span(start_s, dur_ms, error=False, parent=None, sid="s1"):
    s = {"traceId": "t1", "spanId": sid, "name": "GetCart", "startTimeUnixNano": str(int(start_s * 1e9)),
         "endTimeUnixNano": str(int((start_s + dur_ms / 1000) * 1e9)), "status": {"code": 2 if error else 0}}
    if parent:
        s["parentSpanId"] = parent
    return s


def test_parse_spans_reads_otlp_json():
    [s] = parse_spans(otlp_line("cartservice", [span(100.0, 250, error=True, parent="p")]))
    assert (s["service"], s["operation"], s["parent_span_id"]) == ("cartservice", "GetCart", "p")
    assert s["duration_us"] == pytest.approx(250_000) and s["status_code"] == 2.0


def test_span_metrics_per_service_for_one_second():
    spans = parse_spans(otlp_line("cart", [span(100.1, 100, sid="a"), span(100.2, 200, sid="b"),
                                           span(100.3, 300, error=True, sid="c"), span(101.5, 999, sid="d")]))
    row = span_metrics(spans, 100)
    assert row["cart_workload"] == 3 and row["cart_error"] == 1
    assert row["cart_latency"] == pytest.approx(280.0)   # p90 of 100, 200, 300 ms


class FakeContainer:
    def __init__(self):
        self.id, self.calls = "abc123", []
        self.attrs = {"HostConfig": {"CpuPeriod": 0, "CpuQuota": 0}}

    def update(self, **kw): self.calls.append(("update", kw))
    def pause(self): self.calls.append(("pause",))
    def unpause(self): self.calls.append(("unpause",))


class FakeDocker:
    def __init__(self):
        self.container, self.runs = FakeContainer(), []
        self.containers = self

    def list(self, filters):
        return [self.container] if filters["label"] in ("lab.service=cartservice",) else []

    def run(self, image, command, **kw):
        self.runs.append((image, command, kw))


def test_netem_faults_run_a_sidecar_in_the_target_namespace_and_undo():
    docker = FakeDocker()
    inj = FaultInjector(docker)
    active = inj.inject("cartservice", "delay")
    inj.revert(active)
    (img, add, kw), (_, delete, _) = docker.runs
    assert add == ["tc", "qdisc", "add", "dev", "eth0", "root", "netem", "delay", "400ms"]
    assert delete == ["tc", "qdisc", "del", "dev", "eth0", "root"]
    assert kw["network_mode"] == "container:abc123" and kw["cap_add"] == ["NET_ADMIN"]


def test_cpu_and_pause_faults_restore_what_they_changed():
    docker = FakeDocker()
    inj = FaultInjector(docker)
    inj.revert(inj.inject("cartservice", "cpu"))
    inj.revert(inj.inject("cartservice", "pause"))
    assert docker.container.calls == [("update", {"cpu_period": 100000, "cpu_quota": 1000}),
                                      ("update", {"cpu_period": 100000, "cpu_quota": -1}),
                                      ("pause",), ("unpause",)]


def test_only_menu_faults_against_lab_targets():
    inj = FaultInjector(FakeDocker())
    with pytest.raises(ValueError):
        inj.inject("cartservice", "rm -rf")
    with pytest.raises(ValueError):
        inj.inject("investigator", "pause")
    with pytest.raises(LookupError):
        inj.inject("adservice", "pause")      # on the menu but not running here
    assert set(FAULTS) == {"delay", "loss", "cpu", "pause"}


# -- orchestrator -------------------------------------------------------------

import numpy as np
import pandas as pd

from src.config import Live
from src.explain import Ranking
from src.lab.orchestrator import Lab


class Clock:
    def __init__(self, t=1_000.0): self.t = t
    def __call__(self): return self.t
    def sleep(self, s): self.t += s


class FakeTelemetry:
    """Normal noise, then cartservice latency jumps once a fault is active."""
    def __init__(self, clock): self.clock, self.fault_at = clock, None

    def frame(self, start, end):
        t = np.arange(int(start), int(end))
        rng = np.random.default_rng(int(start))
        lat = 10 + rng.normal(0, 0.5, len(t))
        if self.fault_at is not None:
            lat[t >= self.fault_at + 5] += 200
        return pd.DataFrame({"cartservice_latency": lat, "frontend_cpu": 5 + rng.normal(0, 0.2, len(t))},
                            index=pd.Index(t.astype(float)))

    def case(self, start, end, case_id):
        from src.load_rcaeval import Case
        m = self.frame(start, end)
        return (Case(case_id, "LIVE", "OB", m, {c: c.rpartition("_")[0] for c in m.columns}, frozenset({"metrics"})),
                pd.DataFrame(columns=["time", "service", "message"]), pd.DataFrame())


class FakeInjector:
    def __init__(self, telemetry, clock): self.telemetry, self.clock, self.log = telemetry, clock, []
    def inject(self, service, fault):
        from src.lab.faults import ActiveFault
        self.telemetry.fault_at = self.clock(); self.log.append(("inject", service, fault))
        return ActiveFault(service, fault)
    def revert(self, active):
        self.telemetry.fault_at = None; self.log.append(("revert", active.service, active.fault))


class Run:
    def __init__(self, services): self.ranking, self.turns = Ranking("x", [{"service": s} for s in services]), 3


def make_lab(tmp_path, top=("cartservice", "frontend"), **lab_kwargs):
    clock = Clock()
    tel = FakeTelemetry(clock)
    inj = FakeInjector(tel, clock)
    seen = []
    def investigate(inv, on_step, model=None):
        seen.append(inv)
        inv.model_used = model
        on_step({"type": "note", "text": "looking"})
        return Run(list(top))
    cfg = Live(history_s=120, settle_s=10, alarm_timeout_s=60, window_s=20, threshold_sigma=8, persist_s=5)
    lab = Lab(tel, inj, investigate, cfg, records=tmp_path / "incidents.jsonl", clock=clock, sleep=clock.sleep,
              **lab_kwargs)
    return lab, inj, seen


def test_incident_runs_end_to_end_and_is_recorded(tmp_path):
    lab, inj, seen = make_lab(tmp_path)
    events = lab.watch()
    incident = lab.submit("cartservice", "delay", "1.2.3.4")
    lab.run(lab.waiting.popleft())
    assert inj.log == [("inject", "cartservice", "delay"), ("revert", "cartservice", "delay")]
    assert incident.verdict["top1"] is True and incident.verdict["detected"] is True
    assert lab.tally() == {"incidents": 1, "detected": 1, "investigated": 1, "top1": 1, "top3": 1}
    raw = (tmp_path / "incidents.jsonl").read_text()
    record = json.loads(raw)
    assert "visitor" not in record and record["source"] == "visitor"        # the field, not the hash
    assert Lab.visitor_id("1.2.3.4") not in raw and "1.2.3.4" not in raw


def test_agent_is_never_told_the_injection_time(tmp_path):
    lab, inj, seen = make_lab(tmp_path)
    incident = lab.submit("cartservice", "delay", "1.2.3.4")
    lab.run(lab.waiting.popleft())
    [inv] = seen
    assert inv.anchor == incident.times["anomaly_start"]
    assert incident.times["injected"] < incident.times["anomaly_start"] < incident.times["alarm"]
    assert inv.cfg.baseline_s == 120 and inv.cfg.window_s == 20


def test_phases_are_published_in_order(tmp_path):
    lab, inj, seen = make_lab(tmp_path)
    q = lab.watch()
    lab.submit("cartservice", "delay", "1.2.3.4")
    lab.run(lab.waiting.popleft())
    phases = []
    while not q.empty():
        e = q.get_nowait()
        if e["type"] == "phase":
            phases.append(e["incident"]["phase"])
    assert phases == ["settling", "injected", "alarm", "investigating", "diagnosed", "fixing", "recovered"]


def test_wrong_top_pick_scores_a_miss(tmp_path):
    lab, inj, seen = make_lab(tmp_path, top=("frontend", "redis-cart", "cartservice"))
    incident = lab.submit("cartservice", "delay", "1.2.3.4")
    lab.run(lab.waiting.popleft())
    assert incident.verdict["top1"] is False and incident.verdict["top3"] is True


def test_visitor_cooldown_and_menu_are_enforced(tmp_path):
    lab, inj, seen = make_lab(tmp_path)
    lab.submit("cartservice", "delay", "1.2.3.4")
    with pytest.raises(PermissionError):
        lab.submit("frontend", "cpu", "1.2.3.4")
    with pytest.raises(ValueError):
        lab.submit("investigator", "pause", "5.6.7.8")


def test_fault_is_reverted_even_if_the_investigation_crashes(tmp_path):
    lab, inj, seen = make_lab(tmp_path)
    def boom(inv, on_step, model=None): raise RuntimeError("model down")
    lab.investigate = boom
    lab.submit("cartservice", "delay", "1.2.3.4")
    with pytest.raises(RuntimeError):
        lab.run(lab.waiting.popleft())
    assert inj.log[-1] == ("revert", "cartservice", "delay")


def test_span_reader_skips_a_half_written_line_and_resumes(tmp_path):
    from src.lab.telemetry import Telemetry
    path = tmp_path / "spans.jsonl"
    full = otlp_line("cart", [span(100.0, 10, sid="a")])
    partial = otlp_line("cart", [span(101.0, 10, sid="b")])
    path.write_text(full + "\n" + partial[:40])
    tel = Telemetry(docker_client=None, spans_file=path)
    pos = tel._read_spans(0)
    assert [s["span_id"] for s in tel.spans] == ["a"]
    path.write_text(full + "\n" + partial + "\n")
    tel._read_spans(pos)
    assert [s["span_id"] for s in tel.spans] == ["a", "b"]


def test_rows_wait_for_late_spans_before_being_finalized(tmp_path):
    from src.lab.telemetry import Telemetry
    tel = Telemetry(docker_client=None, spans_file=tmp_path / "none.jsonl")
    tel._pending[100] = {"cart_cpu": 1.0}
    tel.spans.extend(parse_spans(otlp_line("cart", [span(100.2, 50)])))   # arrives late, before finalizing
    tel._finalize(100)
    [(second, row)] = list(tel.rows)
    assert second == 100 and row["cart_cpu"] == 1.0 and row["cart_workload"] == 1
    assert 100 not in tel._pending


def test_untraced_services_are_measured_from_their_callers():
    client = span(100.0, 400, sid="c1")
    client["name"] = "hipstershop.CartService/GetCart"
    spans = parse_spans(otlp_line("frontend", [client]))
    peer = next(s for s in spans if s["service"] == "cartservice")
    assert (peer["operation"], peer["parent_span_id"], peer["duration_us"]) == ("GetCart", "c1", 400_000)
    row = span_metrics(spans, 100)
    assert row["cartservice_latency"] == pytest.approx(400.0) and row["frontend_latency"] == pytest.approx(400.0)


def test_every_callee_is_also_measured_from_its_callers():
    client = span(100.0, 450, sid="c2")
    client["name"] = "hipstershop.ShippingService/GetQuote"        # e.g. delay on shipping's network path
    spans = parse_spans(otlp_line("frontend", [client]))
    assert sorted(s["service"] for s in spans) == ["frontend", "shippingservice"]


def test_a_services_own_server_spans_are_not_duplicated():
    server = span(100.0, 50, sid="s9")
    server["name"] = "hipstershop.ProductCatalogService/GetProduct"
    spans = parse_spans(otlp_line("productcatalogservice", [server]))
    assert [s["service"] for s in spans] == ["productcatalogservice"]


def test_alert_incidents_investigate_without_injecting_and_skip_the_tally(tmp_path):
    lab, inj, seen = make_lab(tmp_path)
    drafted, notified = [], []
    lab.handoff = lambda causes, trail, model=None: drafted.append(len(trail)) or {"internal_note": []}
    lab.notify = notified.append
    lab.telemetry.fault_at = lab.clock()                       # something is already wrong
    incident = lab.submit_alert(lab.clock() + 5, [{"name": "HighLatency", "service": "frontend"}])
    lab.run(lab.waiting.popleft())
    assert inj.log == []                                        # nothing injected
    assert "top1" not in incident.verdict and incident.verdict["investigated"] is True
    assert drafted == [1] and notified[0]["source"] == "alertmanager"
    assert lab.tally()["incidents"] == 0                       # alerts have no known answer


def test_visitor_picks_the_model_and_it_is_recorded(tmp_path):
    warmed = []
    lab, inj, seen = make_lab(tmp_path, models={"spark-open": None, "glimmer-tuned": 2}, warm=warmed.append)
    incident = lab.submit("cartservice", "delay", "1.2.3.4", model="glimmer-tuned")
    assert warmed == ["glimmer-tuned"]                      # the server starts while the fault settles
    assert lab.queue_view()[0]["model"] == "glimmer-tuned"
    lab.run(lab.waiting.popleft())
    assert seen[0].model_used == "glimmer-tuned"
    assert json.loads((tmp_path / "incidents.jsonl").read_text())["model"] == "glimmer-tuned"
    lab.submit("frontend", "cpu", "5.6.7.8")                 # the default model needs no warming
    assert warmed == ["glimmer-tuned"] and lab.waiting[-1].model == "spark-open"


def test_a_model_off_the_menu_or_over_its_cap_is_refused(tmp_path):
    lab, inj, seen = make_lab(tmp_path, models={"spark-open": None, "glimmer-tuned": 1})
    with pytest.raises(ValueError):
        lab.submit("cartservice", "delay", "1.2.3.4", model="some-other-model")
    lab.submit("cartservice", "delay", "1.2.3.4", model="glimmer-tuned")
    assert lab.models_left() == {"spark-open": None, "glimmer-tuned": 0}
    with pytest.raises(OverflowError):
        lab.submit("frontend", "cpu", "5.6.7.8", model="glimmer-tuned")
    lab.submit("frontend", "cpu", "5.6.7.8")                 # the default model is still open
    assert lab.waiting[-1].model == "spark-open"


def test_only_the_default_model_without_a_lab_menu(tmp_path):
    lab, inj, seen = make_lab(tmp_path)
    with pytest.raises(ValueError):
        lab.submit("cartservice", "delay", "1.2.3.4", model="glimmer-tuned")


def test_model_caps_survive_a_restart(tmp_path):
    lab, inj, seen = make_lab(tmp_path, models={"spark-open": None, "glimmer-tuned": 2})
    lab.clock.t = __import__("time").time()                  # records are dated by their queue time
    lab.submit("cartservice", "delay", "1.2.3.4", model="glimmer-tuned")
    lab.run(lab.waiting.popleft())
    again, _, _ = make_lab(tmp_path, models={"spark-open": None, "glimmer-tuned": 2})
    assert again.models_left()["glimmer-tuned"] == 1



def approving_sleep(lab):
    """A sleep that approves the proposed rollback as soon as one is waiting, as a human would."""
    plain = lab.sleep
    def sleep(seconds):
        if lab.current is not None and lab.current.remediation.get("status") == "proposed":
            assert lab.approve(lab.current.id)
        plain(seconds)
    return sleep


def test_a_right_diagnosis_gets_a_rollback_that_a_human_approves_and_recovery_is_verified(tmp_path):
    lab, inj, seen = make_lab(tmp_path)
    lab.sleep = approving_sleep(lab)
    incident = lab.submit("cartservice", "delay", "1.2.3.4")
    lab.run(lab.waiting.popleft())
    r = incident.remediation
    assert r["target"] == "cartservice" and r["change"] == "delay" and r["status"] == "approved" and r["approved"]
    assert r["recovered_s"] is not None and r["recovered_s"] <= lab.cfg.recovery_timeout_s
    assert inj.log.count(("revert", "cartservice", "delay")) == 1          # rolled back once, by the approval


def test_without_approval_the_lab_rolls_back_itself_after_the_timeout(tmp_path):
    lab, inj, seen = make_lab(tmp_path)
    incident = lab.submit("cartservice", "delay", "1.2.3.4")
    lab.run(lab.waiting.popleft())
    r = incident.remediation
    assert r["status"] == "timed out" and not r["approved"] and r["recovered_s"] is not None
    assert inj.log.count(("revert", "cartservice", "delay")) == 1


def test_a_wrong_diagnosis_gets_no_rollback_and_the_lab_restores_the_service(tmp_path):
    lab, inj, seen = make_lab(tmp_path, top=("frontend", "cartservice"))
    q = lab.watch()
    incident = lab.submit("cartservice", "delay", "1.2.3.4")
    lab.run(lab.waiting.popleft())
    assert incident.remediation["status"] == "none" and "no recorded change on frontend" in incident.remediation["reason"]
    phases = []
    while not q.empty():
        e = q.get_nowait()
        if e["type"] == "phase":
            phases.append(e["incident"]["phase"])
    assert "fixing" not in phases and inj.log[-1] == ("revert", "cartservice", "delay")


def test_approval_only_counts_for_the_incident_awaiting_it(tmp_path):
    lab, inj, seen = make_lab(tmp_path)
    assert lab.approve("nothing-running") is False



def test_an_incident_leaving_the_queue_is_announced(tmp_path):
    lab, inj, seen = make_lab(tmp_path)
    lab.submit("cartservice", "delay", "1.2.3.4")
    q = lab.watch()
    lab.run = lambda incident: (_ for _ in ()).throw(KeyboardInterrupt)   # stop right after the pop
    try:
        lab.run_forever()
    except KeyboardInterrupt:
        pass
    events = [q.get_nowait() for _ in range(q.qsize())]
    assert {"type": "queue", "queue": []} in events
