import numpy as np
import pandas as pd

from src.anchor import detector_anchor, error_counts, log_flag
from src.config import Anchor
from src.load_rcaeval import Case

CFG = Anchor(history_fraction=0.25, threshold_sigma=10.0, persist_s=20)


def case_with(values):
    t = np.arange(1_700_000_000, 1_700_000_000 + len(values))
    df = pd.DataFrame({"svc_cpu": values, "svc_flat": np.full(len(values), 3.0)}, index=t)
    return Case("re1-x", "RE1", "OB", df, {"svc_cpu": "svc", "svc_flat": "svc"}, frozenset({"metrics"}))


def noisy(n, seed=0):
    return 10 + np.random.default_rng(seed).normal(0, 1, n)


def test_sustained_shift_anchors_at_its_start():
    v = noisy(400)
    v[250:] += 40
    assert detector_anchor(case_with(v), CFG) == 1_700_000_000 + 250


def test_short_spike_is_ignored():
    v = noisy(400)
    v[250:260] += 40             # 10 s, shorter than persist_s
    assert detector_anchor(case_with(v), CFG) is None


def test_quiet_recording_is_never_flagged():
    assert detector_anchor(case_with(noisy(400)), CFG) is None


def test_metric_constant_in_history_does_not_trigger():
    v = noisy(400)
    case = case_with(v)
    case.metrics.loc[case.metrics.index[200:], "svc_flat"] = 1000.0
    assert detector_anchor(case, CFG) is None


def test_history_window_itself_is_never_flagged():
    v = noisy(400)
    v[:50] += 40                 # abnormal only inside the first 25%
    assert detector_anchor(case_with(v), CFG) is None


def log_lines(rows):
    return pd.DataFrame(rows, columns=["time", "service", "message"])


def test_error_counts_bin_error_lines_per_second_and_service():
    logs = log_lines([(100.2, "cart", "java.lang.NullPointerException"), (100.9, "cart", "POST /cart 500 3 ms"),
                      (101.5, "cart", "POST /cart 200 3 ms"), (102.0, "user", "ERROR db down"), (99.0, "user", "error")])
    times, counts, services = error_counts(logs, 100, 103)
    assert list(times) == [100, 101, 102, 103] and services == ["cart", "user"]
    assert counts[:, 0].tolist() == [2, 0, 0, 0] and counts[:, 1].tolist() == [0, 0, 1, 0]   # 99.0 is before start


def test_errors_from_a_service_that_was_silent_in_history_flag():
    counts = np.zeros((600, 2))
    counts[400:, 1] = 2                                    # zero errors while healthy, then two a second
    times = np.arange(600)
    flag = log_flag(counts, times, split=150, window_s=10, threshold_sigma=5.0, persist_s=30, min_count=5)
    assert flag is not None and 400 <= flag <= 410


def test_a_service_that_always_logs_errors_needs_a_real_rise():
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, (600, 1)).astype(float)       # noisy but steady error chatter
    times = np.arange(600)
    assert log_flag(counts, times, split=150, window_s=10, threshold_sigma=5.0, persist_s=30, min_count=5) is None
    counts[400:] += 20
    assert log_flag(counts, times, split=150, window_s=10, threshold_sigma=5.0, persist_s=30, min_count=5) is not None


def test_a_brief_error_burst_or_a_handful_of_lines_does_not_flag():
    counts = np.zeros((600, 1)); times = np.arange(600)
    counts[400:405] = 50                                   # five seconds of errors, then quiet
    assert log_flag(counts, times, split=150, window_s=10, threshold_sigma=5.0, persist_s=30, min_count=5) is None
    counts = np.zeros((600, 1)); counts[400::20] = 1         # one line every 20 s stays under min_count
    assert log_flag(counts, times, split=150, window_s=10, threshold_sigma=5.0, persist_s=30, min_count=5) is None


def test_combined_anchor_takes_the_earlier_signal_and_falls_back_to_metrics():
    from src.anchor import detector_anchor_with_logs
    from src.config import LogAnchor
    log_cfg = LogAnchor(window_s=10, threshold_sigma=5.0, persist_s=10, min_count=5)
    values = np.concatenate([noisy(200), noisy(400, 1) + 100])   # metrics shift at row 200
    case = case_with(values)
    t0 = int(case.metrics.index[0])
    assert detector_anchor_with_logs(case, None, CFG, log_cfg) == detector_anchor(case, CFG)
    quiet = log_lines([(t0 + 10.0, "svc", "started")])
    assert detector_anchor_with_logs(case, quiet, CFG, log_cfg) == detector_anchor(case, CFG)
    early = log_lines([(t0 + 170 + i / 4, "svc", "java.lang.IllegalStateException") for i in range(400)])
    assert detector_anchor_with_logs(case, early, CFG, log_cfg) < detector_anchor(case, CFG)
