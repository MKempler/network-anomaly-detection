"""When to start an investigation, decided by our own detector.

In end-to-end mode nothing from the benchmark's answer key is used. The
detector learns each metric's normal level from the start of the recording,
then watches the rest: a row is abnormal when some metric is more than
`threshold_sigma` (floored) standard deviations from its history mean, and the
investigation is anchored at the start of the first run of abnormal rows that
lasts `persist_s` seconds. A recording the detector never flags gets no
investigation and scores as a miss.
"""

import numpy as np

from src.config import Anchor
from src.evaluate_detect import segments
from src.load_rcaeval import Case


def abnormal_rows(history: np.ndarray, stream: np.ndarray, threshold_sigma: float, rel_std_floor: float) -> np.ndarray:
    """Per row of `stream`: does any column deviate from `history` by more than `threshold_sigma`?"""
    mean, std = history.mean(axis=0), history.std(axis=0)
    floor = np.maximum.reduce([std, rel_std_floor * np.abs(mean), np.full_like(std, 1e-6)])
    deviation = np.abs(stream - mean) / floor
    deviation[:, std < 1e-8] = 0.0   # constant during history: no reference spread to measure against
    return deviation.max(axis=1) > threshold_sigma


def sustained_flag(history: np.ndarray, stream: np.ndarray, times: np.ndarray, threshold_sigma: float,
                   persist_s: int, rel_std_floor: float) -> float | None:
    """Start time of the first run of `persist_s` abnormal rows in `stream`, judged against `history`."""
    abnormal = abnormal_rows(history, stream, threshold_sigma, rel_std_floor)
    for run in segments(abnormal):
        if run.end - run.start + 1 >= persist_s:
            return float(times[run.start])
    return None


def detector_anchor(case: Case, cfg: Anchor) -> float | None:
    """Unix time of the detector's first sustained flag after the history window, or None."""
    values, times = case.metrics.to_numpy(), case.metrics.index.to_numpy()
    split = int(len(values) * cfg.history_fraction)
    return sustained_flag(values[:split], values[split:], times[split:], cfg.threshold_sigma, cfg.persist_s,
                          cfg.rel_std_floor)
