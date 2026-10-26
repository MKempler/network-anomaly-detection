import inspect
import json
import re

import numpy as np
import pandas as pd
import pytest

from src.config import Investigation as InvestigationConfig
from src.load_rcaeval import Case
from src.tools import Investigation, template

ANCHOR = 1_700_000_000  # a realistic unix time, so absolute times would be easy to spot
CFG = InvestigationConfig(baseline_s=100, window_s=50, top_k=5, series_points=10, log_patterns=5)


def make_case(sources=("metrics", "logs", "traces")):
    t = np.arange(ANCHOR - 100, ANCHOR + 50)
    rng = np.random.default_rng(0)
    after = t >= ANCHOR
    metrics = pd.DataFrame({
        "cart_cpu": 10 + rng.normal(0, 1, len(t)) + 20 * after,   # clear spike at the anchor
        "cart_mem": 50 + rng.normal(0, 1, len(t)),                  # normal
        "redis_cpu": 5 + rng.normal(0, 1, len(t)) + 2 * after,       # mild shift
        "redis_flag": np.where(after, 1.0, 0.0),                     # flat, then moves
        "frontend_static": np.full(len(t), 7.0),                     # constant throughout
    }, index=pd.Index(t, name="time"))
    services = {c: c.rpartition("_")[0] for c in metrics.columns}
    return Case("re2-0123456789", "RE2", "OB", metrics, services, frozenset(sources))


def make_logs():
    rows = [(ANCHOR - 90 + i, "cart", f"GET /cart/{i} 200 from 10.0.0.{i % 9}", "info") for i in range(80)]
    rows += [(ANCHOR + i, "cart", f"ERROR timeout after {i} ms contacting redis", "error") for i in range(30)]
    rows += [(ANCHOR - 50, "redis", "ready to accept connections", "info")]
    return pd.DataFrame(rows, columns=["time", "service", "message", "level"])


def make_traces(with_status=True):
    rows = []
    for i, start in enumerate(range(ANCHOR - 100, ANCHOR + 50, 2)):
        late = start >= ANCHOR
        trace = f"t{i}"
        rows.append((start, trace, f"{trace}a", None, "frontendservice", "GET /cart", 900,
                     0.0 if with_status else np.nan))
        rows.append((start, trace, f"{trace}b", f"{trace}a", "cart", "GetCart", 3000 if late else 1000,
                     (14.0 if late and i % 2 else 0.0) if with_status else np.nan))
    return pd.DataFrame(rows, columns=["start", "trace_id", "span_id", "parent_span_id", "service",
                                       "operation", "duration_us", "status_code"])


@pytest.fixture
def inv():
    return Investigation(make_case(), ANCHOR, CFG, logs=make_logs, traces=make_traces)


def all_outputs(inv):
    for name, args in [("top_deviations", {}), ("compare_to_baseline", {"metric": "cart_cpu"}),
                       ("get_metric_window", {"metric": "cart_cpu"}), ("get_logs", {"service": "cart"}),
                       ("get_trace_errors", {"service": "cart"}), ("get_dependencies", {"service": "cart"})]:
        inv.call(name, args)
    return [inv.output(c) for c in inv.calls]


def test_investigation_takes_no_labels():
    params = set(inspect.signature(Investigation).parameters)
    assert params == {"case", "anchor", "cfg", "logs", "traces"}


def test_outputs_never_contain_absolute_times(inv):
    text = " ".join(all_outputs(inv))
    assert not re.search(r"\d{9,}", text), "an absolute unix time leaked into a tool result"


def test_outputs_contain_no_paths_or_label_words(inv):
    text = " ".join(all_outputs(inv)).lower()
    for word in ("inject", "root_cause", "data/rcaeval", "label"):
        assert word not in text


def test_spike_outranks_noisy_metrics_and_constant_metric_scores_zero(inv):
    d = inv.metric_deviations()
    noisy = d[d["baseline_std"] > 0]           # metrics that already varied before the flag
    assert noisy.iloc[0]["metric"] == "cart_cpu"
    d = d.set_index("metric")
    assert d.loc["frontend_static", "score"] == 0.0
    assert d.loc["cart_mem", "score"] < 1.0


def test_flat_metric_that_moves_is_capped_not_exploded(inv):
    score = inv.metric_deviations().set_index("metric").loc["redis_flag", "score"]
    assert 50 <= score <= 2 / CFG.rel_std_floor


def test_compare_to_baseline_reports_relative_onset(inv):
    r = inv.call("compare_to_baseline", {"metric": "cart_cpu"})
    assert r.result["first_exceeds_threshold_s"] == 0
    assert r.evidence == [{"service": "cart", "signal": "cart_cpu", "kind": "metric", "score": r.result["score"]}]


def test_metric_window_is_downsampled_and_relative(inv):
    points = inv.call("get_metric_window", {"metric": "cart_cpu"}).result["points_relative_s_value"]
    assert len(points) == CFG.series_points
    assert points[0][0] == -CFG.baseline_s and points[-1][0] < CFG.window_s


def test_new_error_log_pattern_is_flagged(inv):
    r = inv.call("get_logs", {"service": "cart"}).result
    top = r["patterns"][0]
    assert top["pattern"] == "ERROR timeout after <n> ms contacting redis"
    assert top["window_count"] == 30 and top["score"] > CFG.evidence_threshold
    assert r["error_or_warning_lines"] == {"baseline": 0, "window": 30}


def test_template_masks_numbers_ids_and_addresses():
    msg = "2024-01-22 09:57:56.184 user 3f2a9c1e-1b2c-4d5e-8f90-123456789abc from 10.1.2.3:8080 took 45ms"
    assert template(msg) == "<n> <n> user <id> from <ip> took <n>ms"


def test_trace_errors_score_error_rate_and_latency(inv):
    r = inv.call("get_trace_errors", {"service": "cart"}).result
    [op] = r["operations"]
    assert r["status_codes_recorded"] is True
    assert op["error_rate"]["baseline"] == 0.0 and op["error_rate"]["window"] > 0.4
    assert op["latency_score"] > CFG.evidence_threshold


def test_trace_errors_without_status_codes_use_latency_only():
    inv = Investigation(make_case(), ANCHOR, CFG, logs=make_logs, traces=lambda: make_traces(with_status=False))
    r = inv.call("get_trace_errors", {"service": "cart"}).result
    assert r["status_codes_recorded"] is False
    assert "error_rate" not in r["operations"][0]


def test_dependencies_map_trace_names_onto_metric_services(inv):
    r = inv.call("get_dependencies", {"service": "cart"}).result
    assert r["called_by"] == [{"service": "frontend", "calls": 75}]
    lone = inv.call("get_dependencies", {"service": "redis"}).result
    assert lone["called_by"] == lone["calls"] == [] and "untraced" in lone["note"]


def test_missing_sources_return_errors_not_exceptions():
    inv = Investigation(make_case(sources=("metrics",)), ANCHOR, CFG)
    assert "no logs" in inv.call("get_logs", {"service": "cart"}).result["error"]
    assert "no traces" in inv.call("get_trace_errors", {"service": "cart"}).result["error"]


def test_dispatch_records_calls_and_reports_bad_requests(inv):
    ok = inv.call("top_deviations", {"k": 2})
    unknown_tool = inv.call("delete_everything", {})
    unknown_service = inv.call("get_logs", {"service": "billing-db"})
    bad_args = inv.call("compare_to_baseline", {"metrc": "cart_cpu"})
    unknown_metric = inv.call("compare_to_baseline", {"metric": "nope"})
    assert [c.call_id for c in inv.calls] == ["c1", "c2", "c3", "c4", "c5"]
    assert len(ok.evidence) == 2
    for record in (unknown_tool, unknown_service, bad_args, unknown_metric):
        assert "error" in record.result and record.evidence == []
    assert json.loads(inv.output(ok))["call_id"] == "c1"
