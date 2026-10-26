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
class Config:
    scoring: EventScoring = field(default_factory=EventScoring)
    stats: Stats = field(default_factory=Stats)
    zscore: ZScore = field(default_factory=ZScore)
    iforest: IsolationForest = field(default_factory=IsolationForest)

    def sections(self, *names: str) -> dict:
        return {name: asdict(getattr(self, name)) for name in names}

    def section_id(self, *names: str) -> str:
        blob = json.dumps(self.sections(*names), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]


DETECTION_SECTIONS = ("scoring", "stats")

DEFAULT = Config()
