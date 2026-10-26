"""Every tunable parameter in one place.

Reported numbers name the `config_id` they ran with, a short hash of this
configuration, so a change to any parameter produces a different id.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class EventScoring:
    # SMD rows are one minute apart: flags within ten minutes are one incident.
    merge_gap: int = 10


@dataclass(frozen=True)
class ZScore:
    window: int = 100               # points before the current one; it is never in its own baseline
    threshold: float = 3.0          # a point is flagged when any metric's score exceeds this
    std_floor: float = 1e-6         # rolling std used is max(std, floor)
    constant_train_std: float = 1e-8  # metrics this flat in train always score 0


@dataclass(frozen=True)
class Config:
    scoring: EventScoring = field(default_factory=EventScoring)
    zscore: ZScore = field(default_factory=ZScore)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def config_id(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]


DEFAULT = Config()
