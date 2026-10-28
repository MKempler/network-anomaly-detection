"""Natural degradations of an outage's telemetry, for robustness tests (eval-spec-v1.3).

Each is applied to the case before the investigation starts, so the agent, its tools and
the verifier are unchanged:
- noisy metrics: every metric gets Gaussian noise with a standard deviation of NOISE_SCALE
  times its own spread over the baseline window, seeded by the case id so a run is
  reproducible
- a late alarm: the investigation starts LATE_ALARM_S after the fault time, as if the
  alarm fired that much later
Removing logs or traces needs no change to the case: the agent is simply not offered them.
"""

import hashlib
from dataclasses import replace

import numpy as np

from src.load_rcaeval import Case

NOISE_SCALE = 0.5
LATE_ALARM_S = 60


def add_noise(case: Case, anchor: float, baseline_s: int, scale: float = NOISE_SCALE) -> Case:
    """The case with each metric jittered by `scale` times its baseline standard deviation."""
    metrics = case.metrics
    baseline = metrics[(metrics.index >= anchor - baseline_s) & (metrics.index < anchor)]
    spread = baseline.std().fillna(0.0).to_numpy()
    seed = int(hashlib.sha256(case.case_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    noisy = metrics + rng.normal(0.0, 1.0, metrics.shape) * (scale * spread)
    return replace(case, metrics=noisy)
