import numpy as np
import pytest

from src.config import ZScore
from src.detectors import ZScoreDetector

CFG = ZScore(window=5, threshold=3.0)


def test_current_point_is_not_in_its_own_baseline():
    train = np.array([[1.0], [2.0], [1.0], [2.0], [1.0]])
    test = np.array([[100.0]])
    z = ZScoreDetector(CFG).fit(train).metric_scores(test)
    baseline = train[:, 0]
    expected = (100.0 - baseline.mean()) / baseline.std(ddof=1)
    assert z[0, 0] == pytest.approx(expected)


def test_test_stream_continues_from_train_tail():
    train = np.arange(10, dtype=float).reshape(-1, 1)
    test = np.array([[10.0], [11.0]])
    z = ZScoreDetector(CFG).fit(train).metric_scores(test)
    assert not np.isnan(z).any()
    assert z[1, 0] == pytest.approx(ZScoreDetector(CFG).fit(np.arange(11, dtype=float).reshape(-1, 1))
                                    .metric_scores(np.array([[11.0]]))[0, 0])


def test_constant_train_metric_scores_zero():
    train = np.column_stack([np.ones(10), np.arange(10.0)])
    test = np.array([[50.0, 10.0]])
    z = ZScoreDetector(CFG).fit(train).metric_scores(test)
    assert z[0, 0] == 0.0


def test_std_floor_keeps_scores_finite():
    train = np.column_stack([np.r_[np.zeros(9), 1.0], np.zeros(10)])  # metric 0 not constant in train
    test = np.zeros((10, 2))
    test[-1, 0] = 1.0  # flat window, then a jump
    z = ZScoreDetector(CFG).fit(train).metric_scores(test)
    assert np.isfinite(z).all()
    assert z[-1, 0] == pytest.approx(1.0 / CFG.std_floor)


def test_row_score_is_the_largest_metric_score():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(50, 3))
    test = rng.normal(size=(20, 3))
    det = ZScoreDetector(CFG).fit(train)
    assert np.array_equal(det.score(test), det.metric_scores(test).max(axis=1))


def test_short_train_is_rejected():
    with pytest.raises(ValueError):
        ZScoreDetector(CFG).fit(np.zeros((3, 2)))
