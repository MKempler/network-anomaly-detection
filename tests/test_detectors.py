import numpy as np
import pytest

from src.config import IsolationForest as IsolationForestConfig
from src.config import LSTMAutoencoder as LSTMConfig
from src.config import ZScore
from src.detectors import IsolationForestDetector, LSTMAutoencoderDetector, ZScoreDetector

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


def test_iforest_threshold_comes_from_train_quantile():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(500, 4))
    det = IsolationForestDetector(IsolationForestConfig(n_estimators=50, train_quantile=0.99)).fit(train)
    flagged_train = (det.score(train) > det.threshold).mean()
    assert flagged_train == pytest.approx(0.01, abs=0.005)


def test_iforest_scores_outliers_higher():
    rng = np.random.default_rng(2)
    train = rng.normal(size=(500, 4))
    det = IsolationForestDetector(IsolationForestConfig(n_estimators=50)).fit(train)
    normal, outlier = det.score(np.zeros((1, 4)))[0], det.score(np.full((1, 4), 6.0))[0]
    assert outlier > det.threshold > normal


def test_iforest_is_reproducible():
    rng = np.random.default_rng(3)
    train, test = rng.normal(size=(300, 3)), rng.normal(size=(50, 3))
    cfg = IsolationForestConfig(n_estimators=30)
    a = IsolationForestDetector(cfg).fit(train).score(test)
    b = IsolationForestDetector(cfg).fit(train).score(test)
    assert np.array_equal(a, b)


TINY_LSTM = LSTMConfig(window=10, hidden=(8, 4), epochs=15, batch_size=16, train_stride=1, seed=0)


def periodic(rows, metrics=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(rows)[:, None]
    return np.sin(t / 5 + np.arange(metrics)) + rng.normal(0, 0.05, (rows, metrics))


def test_lstm_scores_one_value_per_test_row_using_train_history():
    det = LSTMAutoencoderDetector(TINY_LSTM).fit(periodic(200))
    assert det.score(periodic(30, seed=1)).shape == (30,)


def test_lstm_reconstructs_normal_better_than_anomalous():
    det = LSTMAutoencoderDetector(TINY_LSTM).fit(periodic(300))
    test = periodic(60, seed=2)
    test[40:50] += 3.0
    scores = det.score(test)
    assert scores[45:50].mean() > 3 * scores[:30].mean()
    assert scores[45:50].min() > det.threshold


def test_lstm_is_reproducible_with_a_seed():
    a = LSTMAutoencoderDetector(TINY_LSTM).fit(periodic(150)).score(periodic(20, seed=3))
    b = LSTMAutoencoderDetector(TINY_LSTM).fit(periodic(150)).score(periodic(20, seed=3))
    assert np.allclose(a, b)
