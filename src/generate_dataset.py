"""Synthetic dataset generator for offline model training.

Produces labelled feature windows from simulated episodes:
  - 30 healthy episodes x 400 steps
  - 15 episodes per fault type x 400 steps
Randomised progression rates per episode to avoid overfitting
to a single degradation speed.

Can also be run standalone to inspect the CSV:
    python src/generate_dataset.py
"""

import numpy as np
import pandas as pd
from src.config import SIGNALS, FAULTS, WINDOW
from src.asset_model import PumpTwin
from src.features import window_features, FEATURE_NAMES


def run_episode(fault, seed, rate_mult, steps=400):
    """Simulate one episode and return labelled feature rows."""
    twin = PumpTwin(seed=seed)
    if fault != "healthy":
        twin.set_fault(fault, rate_mult)

    buf, rows = [], []
    for _ in range(steps):
        tel, truth = twin.step()
        buf.append([tel[s] for s in SIGNALS])
        buf = buf[-WINDOW:]
        if len(buf) == WINDOW:
            r = dict(zip(FEATURE_NAMES, window_features(buf)))
            r.update(label=truth["fault"], severity=truth["severity"])
            rows.append(r)
    return rows


def build_dataset():
    """Build the full training DataFrame from simulated episodes."""
    plan = {"healthy": 30, **{f: 15 for f in FAULTS[1:]}}
    rows, ep = [], 0
    for fault, n_eps in plan.items():
        for _ in range(n_eps):
            ep += 1
            rate = np.random.default_rng(ep).uniform(0.5, 1.5)
            for r in run_episode(fault, ep, rate):
                r.update(episode=ep, episode_type=fault)
                rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print(df.shape)
    print(df["label"].value_counts())
