"""Extended feature-engineering module.

Provides two layers:

1. window_features()  (backward-compatible — replaces src/features.py interface)
   Computes the existing 14-feature vector used by analytics.py at inference time.
   Signature and output are IDENTICAL to the original features.py so that
   analytics.py and generate_dataset.py continue to work without any change.

2. rich_window_features()
   Computes a richer, 40+ feature vector for offline training experiments.
   Used by the real-data loaders and real_data_train.py.

3. DatasetCharacteriser
   Calculates distributions, correlations, rolling statistics, and vibration
   characteristics DIRECTLY from a real dataset DataFrame.
   Results are used to calibrate / inform the Digital Twin model — NOT hardcoded.

Window-size decisions
---------------------
- Digital Twin / SKAB : WINDOW = 30  (30 s at 1 Hz, matches analytics.py)
- NASA IMS            : 2048 samples (~0.1 s at 20 kHz, per signal)
- AI4I                : window = 1   (data is already per-operation-cycle)

All rolling operations use min_periods=1 so partial windows at the start of
a series do not introduce NaN rows.
"""

from __future__ import annotations

import logging
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.config import SIGNALS

logger = logging.getLogger(__name__)

# ── Feature name lists ─────────────────────────────────────────────────────────

# Original 14-feature set (MUST stay identical — analytics.py depends on this)
FEATURE_NAMES = (
    [f"{s}_mean" for s in SIGNALS]
    + [f"{s}_std"  for s in SIGNALS]
    + ["vib_slope", "temp_slope"]
)

# Rich feature names for offline experiments
_RICH_BASE = [
    "mean", "std", "min", "max", "range", "median",
    "q25", "q75", "rms", "peak", "crest_factor",
    "skewness", "kurtosis",
]
RICH_FEATURE_NAMES: list[str] = (
    [f"{s}_{stat}" for s in SIGNALS for stat in _RICH_BASE]
    + [
        "vib_slope",         # vibration linear trend
        "temp_slope",        # temperature linear trend
        "current_slope",     # current linear trend
        "vib_rolling_rms",   # last rolling-window RMS of vibration
        "vib_rolling_std",   # last rolling-window std of vibration
        "temp_rolling_slope",# slope of rolling-mean temperature
        "current_load_ratio",# current / (load proxy = flow normalised)
        "pressure_flow_ratio",  # pressure / flow (pump operating point)
        "vib_temp_corr",     # Pearson r: vibration vs temperature in window
        "current_rpm_corr",  # Pearson r: current vs rpm in window
    ]
)


# ── 1) Original window_features  (backward-compatible) ────────────────────────

def window_features(win) -> np.ndarray:
    """Compute the 14-feature vector for one sliding window.

    Parameters
    ----------
    win : array-like, shape (WINDOW, len(SIGNALS))
        Rows are time-ordered samples; columns match SIGNALS order.

    Returns
    -------
    np.ndarray, shape (len(FEATURE_NAMES),)  — 14 features

    Notes
    -----
    This function is called by analytics.py at 1 Hz and by generate_dataset.py
    during training.  Its signature and output dimensionality MUST NOT change.
    """
    a = np.asarray(win, dtype=float)
    x = np.arange(len(a))
    vib_idx  = SIGNALS.index("vibration_mms")
    temp_idx = SIGNALS.index("temp_c")
    vs = np.polyfit(x, a[:, vib_idx],  1)[0]
    ts = np.polyfit(x, a[:, temp_idx], 1)[0]
    return np.concatenate([a.mean(0), a.std(0), [vs, ts]])


# ── 2) Rich window features  (for offline experiments) ────────────────────────

def rich_window_features(win, signal_names: Optional[Sequence[str]] = None) -> np.ndarray:
    """Compute a 40+ rich feature vector for one sliding window.

    Parameters
    ----------
    win : array-like, shape (n_samples, n_signals)
    signal_names : signal column names in order (defaults to SIGNALS)

    Returns
    -------
    np.ndarray, shape (len(RICH_FEATURE_NAMES),)
    """
    a = np.asarray(win, dtype=float)
    if signal_names is None:
        signal_names = SIGNALS
    n, _ = a.shape
    x    = np.arange(n)

    feats: list[float] = []

    # Per-signal statistics
    for col_idx in range(len(signal_names)):
        col = a[:, col_idx]
        col_mean   = float(col.mean())
        col_std    = float(col.std())
        col_min    = float(col.min())
        col_max    = float(col.max())
        col_range  = col_max - col_min
        col_median = float(np.median(col))
        col_q25    = float(np.percentile(col, 25))
        col_q75    = float(np.percentile(col, 75))
        col_rms    = float(np.sqrt(np.mean(col ** 2)))
        col_peak   = float(np.max(np.abs(col)))
        col_crest  = col_peak / col_rms if col_rms > 1e-9 else 0.0
        col_skew   = float(scipy_stats.skew(col))
        col_kurt   = float(scipy_stats.kurtosis(col))

        feats.extend([
            col_mean, col_std, col_min, col_max, col_range, col_median,
            col_q25, col_q75, col_rms, col_peak, col_crest,
            col_skew, col_kurt,
        ])

    # Trend slopes
    snames = list(signal_names)

    def _slope(idx: int) -> float:
        return float(np.polyfit(x, a[:, idx], 1)[0]) if n > 1 else 0.0

    vib_idx     = snames.index("vibration_mms") if "vibration_mms" in snames else -1
    temp_idx    = snames.index("temp_c")        if "temp_c"        in snames else -1
    current_idx = snames.index("current_a")     if "current_a"     in snames else -1
    flow_idx    = snames.index("flow_m3h")      if "flow_m3h"      in snames else -1
    pressure_idx= snames.index("pressure_bar")  if "pressure_bar"  in snames else -1
    rpm_idx     = snames.index("rpm")           if "rpm"           in snames else -1

    vib_slope  = _slope(vib_idx)     if vib_idx     >= 0 else 0.0
    temp_slope = _slope(temp_idx)    if temp_idx    >= 0 else 0.0
    cur_slope  = _slope(current_idx) if current_idx >= 0 else 0.0

    # Rolling RMS and std of vibration (last 10 samples)
    sub = 10 if n >= 10 else n
    vib_rolling_rms = (
        float(np.sqrt(np.mean(a[-sub:, vib_idx] ** 2))) if vib_idx >= 0 else 0.0
    )
    vib_rolling_std = (
        float(a[-sub:, vib_idx].std()) if vib_idx >= 0 else 0.0
    )

    # Slope of rolling-mean temperature (5-sample rolling, then slope)
    if temp_idx >= 0 and n >= 5:
        roll_t = np.convolve(a[:, temp_idx], np.ones(5) / 5, mode="valid")
        temp_rolling_slope = float(np.polyfit(np.arange(len(roll_t)), roll_t, 1)[0])
    else:
        temp_rolling_slope = 0.0

    # Cross-sensor ratios
    if current_idx >= 0 and flow_idx >= 0:
        mean_flow = float(a[:, flow_idx].mean())
        flow_norm = mean_flow / 60.0 if mean_flow > 1e-9 else 1.0  # normalise to ~0-1
        current_load_ratio = float(a[:, current_idx].mean()) / (flow_norm + 1e-9)
    else:
        current_load_ratio = 0.0

    if pressure_idx >= 0 and flow_idx >= 0:
        mean_flow = float(a[:, flow_idx].mean()) + 1e-9
        pressure_flow_ratio = float(a[:, pressure_idx].mean()) / mean_flow
    else:
        pressure_flow_ratio = 0.0

    # Cross-sensor Pearson correlations
    if vib_idx >= 0 and temp_idx >= 0 and n > 2:
        vib_temp_corr = float(np.corrcoef(a[:, vib_idx], a[:, temp_idx])[0, 1])
        vib_temp_corr = 0.0 if np.isnan(vib_temp_corr) else vib_temp_corr
    else:
        vib_temp_corr = 0.0

    if current_idx >= 0 and rpm_idx >= 0 and n > 2:
        current_rpm_corr = float(np.corrcoef(a[:, current_idx], a[:, rpm_idx])[0, 1])
        current_rpm_corr = 0.0 if np.isnan(current_rpm_corr) else current_rpm_corr
    else:
        current_rpm_corr = 0.0

    feats.extend([
        vib_slope, temp_slope, cur_slope,
        vib_rolling_rms, vib_rolling_std, temp_rolling_slope,
        current_load_ratio, pressure_flow_ratio,
        vib_temp_corr, current_rpm_corr,
    ])

    return np.array(feats, dtype=float)


# ── 3) DatasetCharacteriser  (computes from actual data — nothing hardcoded) ───

class DatasetCharacteriser:
    """Characterise a real-world sensor dataset by actually computing its
    statistical properties.

    All values in self.stats are derived computationally from the provided
    DataFrame — no values are hardcoded.

    Parameters
    ----------
    df : pd.DataFrame
        Time-ordered sensor DataFrame (rows = timestamps, cols = signals).
    signal_cols : list[str]
        Column names of the sensor signals to characterise.
    label_col : str or None
        Column name of anomaly/fault labels (binary or multi-class).
    window : int
        Rolling window size in rows.
    dataset_name : str
        For logging purposes.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        signal_cols: list[str],
        label_col: Optional[str] = None,
        window: int = 30,
        dataset_name: str = "unknown",
    ):
        self.df           = df
        self.signal_cols  = signal_cols
        self.label_col    = label_col
        self.window       = window
        self.dataset_name = dataset_name
        self.stats: dict  = {}

    def compute(self) -> dict:
        """Run all characterisation routines. Returns self.stats dict."""
        logger.info("[Characteriser:%s] Computing dataset characteristics...", self.dataset_name)
        self._global_stats()
        self._correlation_matrix()
        self._rolling_stats()
        if self.label_col is not None:
            self._fault_signatures()
        self._distribution_params()
        logger.info("[Characteriser:%s] Done. Keys: %s", self.dataset_name, list(self.stats.keys()))
        return self.stats

    def _global_stats(self):
        """Per-signal mean, std, min, max, median, IQR, skew, kurtosis."""
        num = self.df[self.signal_cols].select_dtypes(include="number")
        self.stats["global"] = {
            col: {
                "mean":     float(num[col].mean()),
                "std":      float(num[col].std()),
                "min":      float(num[col].min()),
                "max":      float(num[col].max()),
                "median":   float(num[col].median()),
                "iqr":      float(num[col].quantile(0.75) - num[col].quantile(0.25)),
                "skewness": float(num[col].skew()),
                "kurtosis": float(num[col].kurtosis()),
                "q01":      float(num[col].quantile(0.01)),
                "q99":      float(num[col].quantile(0.99)),
            }
            for col in self.signal_cols if col in num.columns
        }

    def _correlation_matrix(self):
        """Pearson correlation matrix for all signal pairs."""
        num = self.df[self.signal_cols].select_dtypes(include="number")
        corr = num.corr(method="pearson")
        self.stats["correlation_matrix"] = corr.to_dict()

    def _rolling_stats(self):
        """Rolling mean and rolling std for each signal at the given window size."""
        num = self.df[self.signal_cols].select_dtypes(include="number")
        result = {}
        for col in self.signal_cols:
            if col not in num.columns:
                continue
            roll_mean = num[col].rolling(self.window, min_periods=1).mean()
            roll_std  = num[col].rolling(self.window, min_periods=1).std()
            result[col] = {
                "rolling_mean_mean": float(roll_mean.mean()),
                "rolling_mean_std":  float(roll_mean.std()),
                "rolling_std_mean":  float(roll_std.mean()),
                "rolling_std_max":   float(roll_std.max()),
            }
        self.stats["rolling"] = result

    def _fault_signatures(self):
        """Per-fault-class signal statistics (mean ± std) computed from actual labels."""
        df = self.df.copy()
        if self.label_col not in df.columns:
            return
        result = {}
        for cls, grp in df.groupby(self.label_col):
            num = grp[self.signal_cols].select_dtypes(include="number")
            result[str(cls)] = {
                col: {
                    "mean": float(num[col].mean()),
                    "std":  float(num[col].std()),
                    "median": float(num[col].median()),
                }
                for col in self.signal_cols if col in num.columns
            }
        self.stats["fault_signatures"] = result

    def _distribution_params(self):
        """
        Fit a normal distribution to each signal (healthy subset if available).

        These fitted mu/sigma values are later used to calibrate the
        simulator's sensor noise parameters.
        """
        df = self.df
        if self.label_col is not None and self.label_col in df.columns:
            healthy_mask = df[self.label_col].isin([0, "0", "normal", "healthy", False])
            healthy      = df[healthy_mask] if healthy_mask.sum() > 10 else df
        else:
            healthy = df

        num = healthy[self.signal_cols].select_dtypes(include="number")
        result = {}
        for col in self.signal_cols:
            if col not in num.columns:
                continue
            vals = num[col].dropna().values
            if len(vals) > 2:
                mu, sigma = float(np.mean(vals)), float(np.std(vals))
            else:
                mu, sigma = 0.0, 1.0
            result[col] = {"mu": mu, "sigma": sigma}
        self.stats["healthy_distribution"] = result

    def anomaly_separation(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """Compute separation metrics between normal and anomaly score distributions.

        Parameters
        ----------
        scores : 1-D array of anomaly scores (higher = more anomalous convention)
        labels : 1-D binary array (0=normal, 1=anomaly)

        Returns dict with: normal_mean, anomaly_mean, delta, cohen_d
        """
        normal_scores  = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        if len(normal_scores) == 0 or len(anomaly_scores) == 0:
            return {}
        delta   = float(np.mean(anomaly_scores) - np.mean(normal_scores))
        pooled  = float(np.sqrt((np.std(normal_scores)**2 + np.std(anomaly_scores)**2) / 2))
        cohen_d = delta / pooled if pooled > 1e-9 else 0.0
        return {
            "normal_mean":  float(np.mean(normal_scores)),
            "anomaly_mean": float(np.mean(anomaly_scores)),
            "delta":        delta,
            "cohen_d":      cohen_d,
        }
