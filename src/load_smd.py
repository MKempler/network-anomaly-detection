"""Load the Server Machine Dataset (SMD).

Each of the 28 machines has a train split (normal operation), a test split, and
a per-row test label (1 = anomalous). Metrics are 38 unnamed columns.

Labels are evaluation data: `Machine.train` and `Machine.test` are what a
detector may see, and `Machine.labels` is only for the scorer.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SMD_DIR = Path(__file__).resolve().parents[1] / "data" / "smd"


@dataclass(frozen=True)
class Machine:
    name: str
    train: np.ndarray   # (rows, 38) float
    test: np.ndarray    # (rows, 38) float
    labels: np.ndarray  # (test rows,) int, 1 = anomalous


@dataclass(frozen=True)
class Interpretation:
    """One labeled segment with the metric columns that contributed (0-indexed)."""
    start: int
    end: int  # inclusive
    columns: tuple[int, ...]


def list_machines(root: Path = SMD_DIR) -> list[str]:
    names = [p.stem for p in (root / "test").glob("machine-*.txt")]
    return sorted(names, key=lambda n: tuple(int(x) for x in n.split("-")[1:]))


def load_machine(name: str, root: Path = SMD_DIR) -> Machine:
    def read(split):
        return np.loadtxt(root / split / f"{name}.txt", delimiter=",", dtype=np.float64)

    train, test = read("train"), read("test")
    labels = read("test_label").astype(np.int8)
    if len(test) != len(labels):
        raise ValueError(f"{name}: {len(test)} test rows but {len(labels)} labels")
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"{name}: train has {train.shape[1]} columns, test has {test.shape[1]}")
    return Machine(name, train, test, labels)


def load_interpretation(name: str, root: Path = SMD_DIR) -> list[Interpretation]:
    """Parse lines like `15849-16368:1,9,10` (1-indexed columns in the file).

    These spans only approximately match `test_label` (machine-1-1's first
    segment is 15849-16368 here but 15849-16394 in the labels), so event
    boundaries for scoring always come from `test_label`.
    """
    segments = []
    for line in (root / "interpretation_label" / f"{name}.txt").read_text().split():
        span, cols = line.split(":")
        start, end = (int(x) for x in span.split("-"))
        segments.append(Interpretation(start, end, tuple(int(c) - 1 for c in cols.split(","))))
    return segments


class MinMaxScaler:
    """Scale each metric to [0, 1] using the train split only.

    Test values outside the train range are not clipped: an excursion beyond
    anything seen in training is exactly the signal detectors look for.
    """

    def fit(self, train: np.ndarray) -> "MinMaxScaler":
        self.low = train.min(axis=0)
        span = train.max(axis=0) - self.low
        self.span = np.where(span > 0, span, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.low) / self.span
