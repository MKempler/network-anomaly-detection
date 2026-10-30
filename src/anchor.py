"""When to start an investigation, decided by our own detector.

In end-to-end mode nothing from the benchmark's answer key is used. The
detector learns each metric's normal level from the start of the recording,
then watches the rest: a row is abnormal when some metric is more than
`threshold_sigma` (floored) standard deviations from its history mean, and the
investigation is anchored at the start of the first run of abnormal rows that
lasts `persist_s` seconds. A recording the detector never flags gets no
investigation and scores as a miss.

With logs (eval-spec-v1.4), a second signal watches each service's error lines:
code-level faults often leave CPU, memory and latency normal while the service
starts logging exceptions. Error counts are mostly zero when healthy, which the
metric rule would ignore as constant, so they get their own count-aware rule;
the investigation starts at whichever signal flags first.
"""

import re
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.config import Anchor, LogAnchor
from src.evaluate_detect import segments
from src.load_rcaeval import Case, load_case, load_logs


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


# a log line that reports a failure: an error word anywhere (NullPointerException, ERROR, failed), or an
# HTTP 5xx status in an access log; healthy lines that match anyway are absorbed by each service's history
ERROR_LINE = re.compile(r"(?i)(?:error|exception|fail|fatal|panic|traceback|refused|unavailable|timed out|timeout)"
                        r"|\s5\d\d\s")


def error_counts(logs: pd.DataFrame, start: float, end: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per second from `start` to `end`: each service's count of error lines. Returns (times, counts, services)."""
    times = np.arange(int(start), int(end) + 1)
    services = sorted(logs["service"].dropna().unique())
    counts = np.zeros((len(times), len(services)))
    errors = logs[logs["message"].str.contains(ERROR_LINE)]
    errors = errors[(errors["time"] >= times[0]) & (errors["time"] < times[-1] + 1)]
    if len(errors):
        rows = (errors["time"].to_numpy() - times[0]).astype(int)
        cols = pd.Index(services).get_indexer(errors["service"])
        np.add.at(counts, (rows, cols), 1)
    return times, counts, services


def log_flag(counts: np.ndarray, times: np.ndarray, split: int, window_s: int, threshold_sigma: float,
             persist_s: int, min_count: int) -> float | None:
    """Start of the first `persist_s`-second run where some service's error lines over the last `window_s`
    seconds stand out from its history (rows before `split`) and number at least `min_count`."""
    if split < window_s or split >= len(counts):
        return None
    total = np.vstack([np.zeros((1, counts.shape[1])), np.cumsum(counts, axis=0)])
    rolling = np.empty_like(counts)
    rolling[window_s - 1:] = total[window_s:] - total[:-window_s]
    rolling[:window_s - 1] = total[1:window_s]
    history, stream = rolling[window_s - 1:split], rolling[split:]
    mean, std = history.mean(axis=0), history.std(axis=0)
    floor = np.maximum.reduce([std, np.sqrt(mean), np.ones_like(std)])   # counts: Poisson spread, at least one line
    abnormal = (((stream - mean) / floor > threshold_sigma) & (stream >= min_count)).any(axis=1)
    for run in segments(abnormal):
        if run.end - run.start + 1 >= persist_s:
            return float(times[split + run.start])
    return None


def log_split(metric_times: np.ndarray, grid: np.ndarray, cfg: Anchor) -> int:
    """Row of the per-second log grid where the metric history window ends."""
    return int(np.searchsorted(grid, metric_times[int(len(metric_times) * cfg.history_fraction)]))


def detector_anchor_with_logs(case: Case, logs: pd.DataFrame | None, cfg: Anchor, log_cfg: LogAnchor) -> float | None:
    """The earlier of the metric flag and the error-log flag (eval-spec-v1.4); metric-only without logs."""
    metric = detector_anchor(case, cfg)
    if logs is None or not len(logs):
        return metric
    times = case.metrics.index.to_numpy()
    grid, counts, _ = error_counts(logs, times[0], times[-1])
    flag = log_flag(counts, grid, log_split(times, grid, cfg), **asdict(log_cfg))
    return flag if metric is None else metric if flag is None else min(metric, flag)


def end_to_end_anchor(mode: str, suite: str, case_id: str, case_dir, cfg) -> float | None:
    """Where our detector starts an end-to-end investigation: "detector" watches metrics (eval-spec-v1),
    "detector-logs" metrics and error logs (eval-spec-v1.4)."""
    case = load_case(suite, case_id, case_dir)
    if mode == "detector":
        return detector_anchor(case, cfg.anchor)
    logs = load_logs(case_id) if "logs" in case.sources else None
    return detector_anchor_with_logs(case, logs, cfg.anchor, cfg.log_anchor)
