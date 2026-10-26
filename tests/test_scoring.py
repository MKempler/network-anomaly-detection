import math

import numpy as np
import pytest

from src.evaluate_detect import (EventScores, Segment, bootstrap_ci, bootstrap_f1_difference, combine,
                                 score_events, segments)


def mask(n, *spans):
    m = np.zeros(n, dtype=bool)
    for start, end in spans:
        m[start:end + 1] = True
    return m


def test_segments_finds_runs_including_edges():
    assert segments(mask(10, (0, 1), (5, 5), (9, 9))) == [Segment(0, 1), Segment(5, 5), Segment(9, 9)]


def test_segments_merges_small_gaps_only():
    m = mask(20, (2, 3), (6, 7), (15, 15))
    assert segments(m, merge_gap=2) == [Segment(2, 7), Segment(15, 15)]
    assert segments(m, merge_gap=1) == [Segment(2, 3), Segment(6, 7), Segment(15, 15)]


def test_one_flagged_row_detects_a_long_event_once():
    s = score_events(mask(100, (10, 59)), mask(100, (30, 30)))
    assert (s.true_events, s.detected_events, s.flagged_events, s.correct_flags) == (1, 1, 1, 1)
    assert s.delays == (20,)


def test_several_flags_inside_one_event_are_all_correct_but_detect_it_once():
    s = score_events(mask(100, (10, 59)), mask(100, (12, 12), (20, 25), (40, 40)))
    assert s.detected_events == 1 and s.flagged_events == 3 and s.correct_flags == 3
    assert s.delays == (2,)   # first overlapping flag


def test_flag_spanning_two_events_keeps_precision_bounded():
    s = score_events(mask(100, (10, 19), (30, 39)), mask(100, (15, 35)))
    assert s.recall == 1.0 and s.precision == 1.0
    assert s.delays == (5, 0)  # flag started before the second event


def test_false_alarms_lower_precision_not_recall():
    s = score_events(mask(100, (10, 19)), mask(100, (12, 12), (50, 50), (70, 70), (90, 90)))
    assert s.recall == 1.0
    assert s.precision == pytest.approx(0.25)
    assert s.f1 == pytest.approx(0.4)


def test_flag_adjacent_to_event_is_not_an_overlap():
    s = score_events(mask(50, (10, 19)), mask(50, (20, 25)))
    assert s.detected_events == 0 and s.correct_flags == 0


def test_no_flags_scores_zero_without_dividing_by_zero():
    s = score_events(mask(50, (10, 19)), mask(50))
    assert (s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0)
    assert math.isnan(s.mean_delay)


def test_merge_gap_turns_flicker_into_one_flagged_event():
    flicker = mask(100, (40, 40), (42, 42), (44, 44))
    assert score_events(mask(100, (10, 19)), flicker, merge_gap=0).flagged_events == 3
    assert score_events(mask(100, (10, 19)), flicker, merge_gap=1).flagged_events == 1


def test_combine_pools_counts_across_servers():
    a = score_events(mask(50, (10, 19)), mask(50, (12, 12)))
    b = score_events(mask(50, (10, 19), (30, 39)), mask(50, (45, 45)))
    total = combine([a, b])
    assert (total.true_events, total.detected_events, total.flagged_events, total.correct_flags) == (3, 1, 2, 1)
    assert total.delays == (2,)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_events(np.zeros(5), np.zeros(6))


def _server(true, detected, flagged, correct):
    return EventScores(true, detected, flagged, correct, ())


def test_bootstrap_interval_contains_pooled_estimate_and_is_reproducible():
    servers = [_server(10, 8, 20, 9), _server(5, 5, 50, 5), _server(8, 2, 4, 3), _server(12, 10, 30, 12)]
    pooled = combine(servers)
    ci = bootstrap_ci(servers, samples=500, seed=0)
    for metric in ("precision", "recall", "f1"):
        lo, hi = ci[metric]
        assert lo <= getattr(pooled, metric) <= hi
    assert ci == bootstrap_ci(servers, samples=500, seed=0)


def test_bootstrap_interval_collapses_for_identical_servers():
    servers = [_server(10, 5, 10, 5)] * 6
    lo, hi = bootstrap_ci(servers, samples=200, seed=1)["recall"]
    assert lo == hi == 0.5


def test_paired_difference_is_zero_for_the_same_detector():
    servers = [_server(10, 8, 20, 9), _server(5, 5, 50, 5), _server(8, 2, 4, 3)]
    assert bootstrap_f1_difference(servers, servers, samples=200, seed=0) == (0.0, 0.0)


def test_paired_difference_is_positive_when_a_dominates():
    better = [_server(10, 9, 12, 11), _server(8, 8, 9, 8), _server(6, 5, 6, 5)]
    worse = [_server(10, 9, 120, 11), _server(8, 8, 90, 8), _server(6, 5, 60, 5)]
    lo, hi = bootstrap_f1_difference(better, worse, samples=500, seed=0)
    assert lo > 0
