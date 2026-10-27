import numpy as np
import pandas as pd

from src.anchor import detector_anchor
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
