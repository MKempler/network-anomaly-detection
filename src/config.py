"""Every tunable parameter in one place.

Reported numbers name the config id they ran with: a short hash of exactly the
sections that produced them. A detection run hashes the scoring and statistics
sections plus its own detector's section, so adding or changing another
detector never changes an existing result's id.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class EventScoring:
    # SMD rows are one minute apart: flags within ten minutes are one incident.
    merge_gap: int = 10


@dataclass(frozen=True)
class Stats:
    bootstrap_samples: int = 2000   # resamples of servers for 95% intervals
    seed: int = 0


@dataclass(frozen=True)
class ZScore:
    window: int = 100               # points before the current one; it is never in its own baseline
    threshold: float = 3.0          # a point is flagged when any metric's score exceeds this
    std_floor: float = 1e-6         # rolling std used is max(std, floor)
    constant_train_std: float = 1e-8  # metrics this flat in train always score 0


@dataclass(frozen=True)
class IsolationForest:
    n_estimators: int = 100
    train_quantile: float = 0.99    # flag threshold: this quantile of train scores
    random_state: int = 0


@dataclass(frozen=True)
class LSTMAutoencoder:
    window: int = 100               # consecutive points per input window, all metrics
    hidden: tuple = (64, 32)        # encoder sizes; the decoder mirrors them
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    train_stride: int = 10          # train on every 10th window; test scores every row
    train_quantile: float = 0.99    # flag threshold: this quantile of train reconstruction errors
    seed: int = 0


@dataclass(frozen=True)
class Investigation:
    """What the explanation layer looks at around a flag (RCAEval rows are 1 s apart)."""
    baseline_s: int = 600           # normal reference: this many seconds before the flag
    window_s: int = 600             # flagged window: this many seconds from the flag (RCAEval: 10 min each side)
    rel_std_floor: float = 0.01     # baseline std used is max(std, floor * level, abs floor)
    abs_std_floor: float = 1e-6
    top_k: int = 15                 # metrics per top-deviations listing
    series_points: int = 60         # max points returned for one metric's series
    log_patterns: int = 10          # log templates per service listing
    evidence_threshold: float = 3.0  # cited evidence must score at least this
    causes: int = 5                 # ranked causes each ranker returns
    max_steps: int = 10             # agent turns before it must submit


@dataclass(frozen=True)
class Anchor:
    """End-to-end mode: our detector, not the benchmark, decides when to investigate.

    Chosen on train cases only (RE1 + RE2, 258 cases) by one objective set in
    advance: the most anchors within 0-120 s after the fault starts, ties to
    fewer early alarms. An isolation forest anchored only 83 of 258 there; this
    per-metric z-score alarm anchors 188.
    """
    history_fraction: float = 0.25  # "normal" is learned from the start of each recording
    threshold_sigma: float = 30.0   # a metric this many (floored) stds from its history mean is abnormal
    persist_s: int = 120            # ...and must stay abnormal this long before an investigation starts
    rel_std_floor: float = 0.01     # history std used is max(std, floor * |mean|, 1e-6)


@dataclass(frozen=True)
class Config:
    scoring: EventScoring = field(default_factory=EventScoring)
    stats: Stats = field(default_factory=Stats)
    zscore: ZScore = field(default_factory=ZScore)
    iforest: IsolationForest = field(default_factory=IsolationForest)
    lstm: LSTMAutoencoder = field(default_factory=LSTMAutoencoder)
    investigation: Investigation = field(default_factory=Investigation)
    anchor: Anchor = field(default_factory=Anchor)

    def sections(self, *names: str) -> dict:
        return {name: asdict(getattr(self, name)) for name in names}

    def section_id(self, *names: str) -> str:
        blob = json.dumps(self.sections(*names), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]


DETECTION_SECTIONS = ("scoring", "stats")

DEFAULT = Config()
