import dataclasses

import numpy as np
import pytest

from src.load_rcaeval import Case, case_index, iter_cases, load_labels, service_of

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
    assert fields == {"case_id", "suite", "system", "metrics", "services"}
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
