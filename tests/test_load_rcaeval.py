import dataclasses

import numpy as np
import pytest

from src.load_rcaeval import (Case, case_index, iter_cases, load_labels, load_logs, load_traces, parse_folder,
                              service_of)

HEADER = "time,adservice_cpu,ts-auth-service_container-cpu-usage-seconds-total,time,empty_metric"


def write_case(root, system, folder, instance, rows, inject_time):
    case_dir = root / "RE1" / f"RE1-{system}" / folder / str(instance)
    case_dir.mkdir(parents=True)
    (case_dir / "data.csv").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    (case_dir / "inject_time.txt").write_text(f"{inject_time}\n")


@pytest.fixture
def rcaeval_root(tmp_path):
    write_case(tmp_path, "OB", "adservice_cpu", 1,
               ["100,1.0,5.0,100,", "101,,6.0,101,", "102,3.0,7.0,102,", ",9.0,9.0,,"], 101)
    write_case(tmp_path, "TT", "ts-auth-service_mem", 2,
               ["200,1.0,2.0,200,", "201,1.0,2.0,201,"], 201)
    return tmp_path


def test_case_ids_are_opaque_and_stable(rcaeval_root):
    ids = list(case_index("RE1", rcaeval_root))
    assert ids == list(case_index("RE1", rcaeval_root))
    for case_id in ids:
        assert case_id.startswith("re1-")
        assert not any(word in case_id for word in ("adservice", "auth", "cpu", "mem", "OB", "TT"))


def test_case_carries_no_ground_truth(rcaeval_root):
    fields = {f.name for f in dataclasses.fields(Case)}
    assert fields == {"case_id", "suite", "system", "metrics", "services", "sources"}
    for case in iter_cases("RE1", rcaeval_root):
        assert "inject" not in repr(case).lower()


def test_labels_come_from_the_path(rcaeval_root):
    labels = sorted(load_labels("RE1", rcaeval_root).values(), key=lambda l: l.inject_time)
    assert [(l.root_service, l.fault, l.inject_time) for l in labels] == [
        ("adservice", "cpu", 101), ("ts-auth-service", "mem", 201)]


def test_metrics_cleaning(rcaeval_root):
    case = next(c for c in iter_cases("RE1", rcaeval_root) if c.system == "OB")
    m = case.metrics
    assert list(m.index) == [100, 101, 102]              # row without a timestamp dropped
    assert "time.1" not in m.columns                      # duplicate time column dropped
    assert "empty_metric" not in m.columns                # all-empty column dropped
    assert m.loc[101, "adservice_cpu"] == 1.0             # gap forward-filled
    assert not np.isnan(m.to_numpy()).any()


def test_service_map(rcaeval_root):
    case = next(iter_cases("RE1", rcaeval_root))
    assert case.services["ts-auth-service_container-cpu-usage-seconds-total"] == "ts-auth-service"
    assert case.candidate_services == ["adservice", "ts-auth-service"]


@pytest.mark.parametrize("column, service", [
    ("adservice_cpu", "adservice"),
    ("frontend-external_load", "frontend-external"),
    ("carts-db_container-memory-rss", "carts-db"),
    ("nounderscore", None),
    ("_cpu", None),
])
def test_service_of(column, service):
    assert service_of(column) == service


SENTINEL = "ROOT_CAUSE_SENTINEL"


@pytest.fixture
def re3_root(tmp_path):
    """An RE3-style case with logs, traces, and the files that must never be read."""
    case_dir = tmp_path / "RE3" / "RE3-TT" / "ts-route-service_f3_1" / "1"
    case_dir.mkdir(parents=True)
    (case_dir / "metrics.csv").write_text("time,ts-route-service_cpu\n100,1.0\n101,2.0\n")
    (case_dir / "inject_time.txt").write_text("100\n")
    (case_dir / "logs.csv").write_text(
        "time,timestamp,container_name,message,pod_name,node_name\n"
        "00:01,101000000000,ts-route-service,route lookup failed,pod-a,node-1\n"
        "00:01,100000000000,ts-auth-service,login ok,pod-b,node-1\n")
    (case_dir / "traces.csv").write_text(
        "time,traceID,spanID,serviceName,methodName,operationName,startTimeMillis,startTime,duration,statusCode,parentSpanID\n"
        "00:01,t1,s2,ts-route-service,GET,getRoute,100500,100500000,900,500,s1\n"
        "00:01,t1,s1,ts-gateway,GET,route,100400,100400000,1200,0,\n")
    (case_dir / "root_cause.txt").write_text(f"{SENTINEL} ts-route-service NullPointerException\n")
    (case_dir / "metrics.png").write_bytes(SENTINEL.encode())
    stray = tmp_path / "RE3" / "RE3-TT" / "ts-route-service_f3_1" / "multi-source-data"
    stray.mkdir()
    (stray / "inject_time.txt").write_text("1\n")
    return tmp_path


def test_stray_non_numbered_folders_are_not_cases(re3_root):
    assert len(case_index("RE3", re3_root)) == 1


@pytest.mark.parametrize("folder, expected", [
    ("adservice_cpu", ("adservice", "cpu")),
    ("ts-auth-service_socket", ("ts-auth-service", "socket")),
    ("front-end_f2", ("front-end", "f2")),
    ("ts-route-service_f3_1", ("ts-route-service", "f3")),
])
def test_parse_folder(folder, expected):
    assert parse_folder(folder) == expected


def test_parse_folder_rejects_unknown_faults():
    with pytest.raises(ValueError):
        parse_folder("adservice_meltdown")


def test_logs_and_traces_load_by_opaque_id_and_sort_by_time(re3_root):
    case = next(iter_cases("RE3", re3_root))
    assert case.sources == {"metrics", "logs", "traces"}
    logs = load_logs(case.case_id)
    assert list(logs.columns) == ["time", "service", "message"]
    assert logs["service"].tolist() == ["ts-auth-service", "ts-route-service"]
    assert logs["time"].tolist() == [100.0, 101.0]
    traces = load_traces(case.case_id)
    assert traces["span_id"].tolist() == ["s1", "s2"]
    assert traces.loc[1, "parent_span_id"] == "s1" and traces.loc[1, "status_code"] == 500


def test_answer_files_are_never_read(re3_root):
    case = next(iter_cases("RE3", re3_root))
    seen = repr(case) + case.metrics.to_csv() + load_logs(case.case_id).to_csv() + load_traces(case.case_id).to_csv()
    assert SENTINEL not in seen
    assert load_labels("RE3", re3_root)[case.case_id].fault == "f3"
