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
    assert docker.container.calls == [("update", {"cpu_period": 100000, "cpu_quota": 5000}),
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
