"""Event-level detection scoring.

A true event is one contiguous labeled anomaly segment, however long. A
flagged event is one contiguous run of flagged rows; runs separated by at most
`merge_gap` unflagged rows count as one. Flags are grouped without looking at
labels.

- recall    = true events overlapped by a flagged event / true events
- precision = flagged events that overlap a true event / flagged events
- delay     = rows from a true event's start to the start of the first flagged
              event overlapping it (0 if that flag began earlier)

Point-adjusted scoring, where one flagged row marks its whole segment as
detected at the point level, is deliberately not used: it inflates results.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: int
    end: int  # inclusive

    def overlaps(self, other: "Segment") -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class EventScores:
    true_events: int
    detected_events: int
    flagged_events: int
    correct_flags: int
    delays: tuple[int, ...]

    @property
    def recall(self) -> float:
        return self.detected_events / self.true_events if self.true_events else 0.0

    @property
    def precision(self) -> float:
        return self.correct_flags / self.flagged_events if self.flagged_events else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mean_delay(self) -> float:
        return float(np.mean(self.delays)) if self.delays else float("nan")

    def __add__(self, other: "EventScores") -> "EventScores":
        return EventScores(
            self.true_events + other.true_events,
            self.detected_events + other.detected_events,
            self.flagged_events + other.flagged_events,
            self.correct_flags + other.correct_flags,
            self.delays + other.delays,
        )


EMPTY = EventScores(0, 0, 0, 0, ())


def segments(mask: np.ndarray, merge_gap: int = 0) -> list[Segment]:
    """Contiguous runs of True, merging runs separated by <= merge_gap False rows."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    runs = [Segment(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2])]
    merged: list[Segment] = []
    for run in runs:
        if merged and run.start - merged[-1].end - 1 <= merge_gap:
            merged[-1] = Segment(merged[-1].start, run.end)
        else:
            merged.append(run)
    return merged


def score_events(labels: np.ndarray, flags: np.ndarray, merge_gap: int = 0) -> EventScores:
    if len(labels) != len(flags):
        raise ValueError(f"{len(labels)} labels but {len(flags)} flags")
    truth = segments(labels)
    flagged = segments(flags, merge_gap)

    delays = []
    for event in truth:
        hits = [f for f in flagged if f.overlaps(event)]
        if hits:
            delays.append(max(0, hits[0].start - event.start))
    correct = sum(any(f.overlaps(t) for t in truth) for f in flagged)
    return EventScores(len(truth), len(delays), len(flagged), correct, tuple(delays))


def combine(scores: list[EventScores]) -> EventScores:
    """Pool event counts across servers (micro-average)."""
    return sum(scores, EMPTY)
