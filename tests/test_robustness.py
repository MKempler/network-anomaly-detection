import numpy as np
import pandas as pd

from src.load_rcaeval import Case
from src.robustness import add_noise


def case(seed=0):
    t = np.arange(0, 1200, 1.0)
    rng = np.random.default_rng(seed)
    m = pd.DataFrame({"cart_latency": 10 + rng.normal(0, 2, len(t)), "cart_cpu": np.full(len(t), 5.0)}, index=t)
    return Case("re2-abc", "RE2", "OB", m, {"cart_latency": "cart", "cart_cpu": "cart"}, frozenset({"metrics"}))


def test_noise_scales_with_each_metrics_own_baseline_spread_and_is_reproducible():
    c = case()
    noisy = add_noise(c, anchor=600, baseline_s=600, scale=0.5)
    added = noisy.metrics - c.metrics
    assert abs(added["cart_latency"].std() - 0.5 * c.metrics["cart_latency"].iloc[:600].std()) < 0.15
    assert (added["cart_cpu"] == 0).all()                     # a flat metric has no spread, so no noise
    again = add_noise(c, anchor=600, baseline_s=600, scale=0.5)
    assert noisy.metrics.equals(again.metrics)                 # seeded by the case id
    assert c.metrics["cart_latency"].iloc[0] != noisy.metrics["cart_latency"].iloc[0]   # the original is untouched
