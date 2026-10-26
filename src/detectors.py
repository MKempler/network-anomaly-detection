"""Anomaly detectors.

Every detector is fit on one server's train split (normal operation only) and
then scores that server's test split as a continuation of it. A detector
exposes per-row scores and a threshold; a row is flagged when its score is
above the threshold. Thresholds come from train data or fixed config, never
from test labels.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import IsolationForest as IsolationForestConfig
from src.config import ZScore


class ZScoreDetector:
    """Rolling z-score per metric, the transparent baseline.

    Each point is compared with the mean and standard deviation of the
    `window` points before it. The point itself is excluded, so an extreme
    value can't inflate the baseline it is judged against. The first test
    points use the tail of train as their history.
    """

    name = "zscore"

    def __init__(self, cfg: ZScore):
        self.cfg = cfg
        self.threshold = cfg.threshold

    def fit(self, train: np.ndarray) -> "ZScoreDetector":
        if len(train) < self.cfg.window:
            raise ValueError(f"train has {len(train)} rows, window needs {self.cfg.window}")
        self.constant = train.std(axis=0) < self.cfg.constant_train_std
        self.history = train[-self.cfg.window:]
        return self

    def metric_scores(self, test: np.ndarray) -> np.ndarray:
        """(rows, metrics) absolute z-scores."""
        stream = pd.DataFrame(np.vstack([self.history, test]))
        rolling = stream.rolling(self.cfg.window, min_periods=self.cfg.window)
        mean = rolling.mean().shift(1).to_numpy()[len(self.history):]
        std = rolling.std().shift(1).to_numpy()[len(self.history):]
        z = np.abs(test - mean) / np.maximum(std, self.cfg.std_floor)
        z[:, self.constant] = 0.0
        return z

    def score(self, test: np.ndarray) -> np.ndarray:
        """Per-row score: the largest metric z-score."""
        return self.metric_scores(test).max(axis=1)


class IsolationForestDetector:
    """Isolation forest over all metrics at once.

    Catches combinations of metrics that are unusual together even when no
    single metric's z-score stands out. No contamination rate is set: train is
    normal operation, so the flag threshold is simply a high quantile of the
    scores train itself receives.
    """

    name = "iforest"

    def __init__(self, cfg: IsolationForestConfig):
        self.cfg = cfg

    def fit(self, train: np.ndarray) -> "IsolationForestDetector":
        self.model = IsolationForest(n_estimators=self.cfg.n_estimators,
                                     random_state=self.cfg.random_state).fit(train)
        self.threshold = float(np.quantile(self.score(train), self.cfg.train_quantile))
        return self

    def score(self, test: np.ndarray) -> np.ndarray:
        """Per-row anomaly score; higher is more anomalous."""
        return -self.model.score_samples(test)
