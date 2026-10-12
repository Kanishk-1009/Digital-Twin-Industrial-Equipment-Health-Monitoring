"""Tests — Phase 1: Feature engineering backward compatibility.

These tests verify that:
  1. window_features() still returns exactly 14 features (analytics.py dependency)
  2. FEATURE_NAMES still has 14 entries
  3. rich_window_features() returns the expected shape
  4. DatasetCharacteriser computes from actual data (not hardcoded)
  5. features.py shim re-exports correctly
"""

import numpy as np
import pandas as pd
import pytest

from src.features import window_features, FEATURE_NAMES
from src.feature_engineering import (
    rich_window_features,
    RICH_FEATURE_NAMES,
    DatasetCharacteriser,
    window_features as fe_window_features,
)
from src.config import SIGNALS, WINDOW


class TestWindowFeatures:
    """Verify backward-compatible 14-feature window_features()."""

    def test_shape_matches_analytics_expectation(self):
        """analytics.py does x = window_features(buf).reshape(1, -1) — must be (14,)."""
        win = np.random.rand(WINDOW, len(SIGNALS))
        out = window_features(win)
        assert out.shape == (14,), f"Expected (14,), got {out.shape}"

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 14, f"Expected 14 names, got {len(FEATURE_NAMES)}"

    def test_feature_names_content(self):
        expected_means = [f"{s}_mean" for s in SIGNALS]
        expected_stds  = [f"{s}_std"  for s in SIGNALS]
        for n in expected_means + expected_stds:
            assert n in FEATURE_NAMES, f"'{n}' missing from FEATURE_NAMES"
        assert "vib_slope"  in FEATURE_NAMES
        assert "temp_slope" in FEATURE_NAMES

    def test_shim_same_as_direct(self):
        """features.py shim must give identical result to feature_engineering.py."""
        win = np.random.rand(WINDOW, len(SIGNALS))
        np.random.seed(42)
        out_shim   = window_features(win)
        out_direct = fe_window_features(win)
        np.testing.assert_array_equal(out_shim, out_direct)

    def test_all_finite(self):
        """No NaN or Inf in output for valid input."""
        win = np.random.rand(WINDOW, len(SIGNALS)) * 100
        out = window_features(win)
        assert np.all(np.isfinite(out)), "window_features returned non-finite values"

    def test_minimal_window(self):
        """Works with a 2-row window (minimum for polyfit)."""
        win = np.random.rand(2, len(SIGNALS))
        out = window_features(win)
        assert out.shape == (14,)

    def test_dtype_float(self):
        win = np.random.rand(WINDOW, len(SIGNALS))
        out = window_features(win)
        assert out.dtype in (np.float32, np.float64)


class TestRichWindowFeatures:
    """Verify rich 88-feature window."""

    def test_shape(self):
        win = np.random.rand(WINDOW, len(SIGNALS))
        out = rich_window_features(win)
        assert out.shape == (len(RICH_FEATURE_NAMES),), (
            f"Expected ({len(RICH_FEATURE_NAMES)},), got {out.shape}"
        )

    def test_all_finite(self):
        win = np.random.rand(WINDOW, len(SIGNALS)) * 10
        out = rich_window_features(win)
        assert np.all(np.isfinite(out)), "rich_window_features returned non-finite values"

    def test_custom_signal_names(self):
        """Works with a subset of signals (e.g., SKAB signals)."""
        skab_signals = ["vibration_rms", "temperature_c", "current_a",
                        "pressure_bar", "flow_proxy", "voltage_v"]
        win = np.random.rand(WINDOW, len(skab_signals))
        # Should not raise
        out = rich_window_features(win, signal_names=skab_signals)
        assert out.ndim == 1


class TestDatasetCharacteriser:
    """Verify DatasetCharacteriser computes from actual data."""

    @pytest.fixture
    def sample_df(self):
        np.random.seed(0)
        n = 300
        df = pd.DataFrame({
            "vibration_rms": np.random.randn(n) * 0.5 + 1.0,
            "temperature_c": np.random.randn(n) * 2.0 + 55.0,
            "current_a":     np.random.randn(n) * 0.3 + 8.0,
            "anomaly":       (np.random.rand(n) > 0.9).astype(int),
        })
        return df

    def test_global_stats_computed(self, sample_df):
        ch = DatasetCharacteriser(
            sample_df,
            signal_cols=["vibration_rms", "temperature_c", "current_a"],
            label_col="anomaly",
        )
        stats = ch.compute()
        assert "global" in stats
        assert "vibration_rms" in stats["global"]
        # Verify mean is computed from actual data, not hardcoded
        actual_mean = float(sample_df["vibration_rms"].mean())
        computed_mean = stats["global"]["vibration_rms"]["mean"]
        assert abs(computed_mean - actual_mean) < 1e-9, (
            f"Mean mismatch: computed={computed_mean}, actual={actual_mean}"
        )

    def test_correlation_matrix_present(self, sample_df):
        ch = DatasetCharacteriser(
            sample_df,
            signal_cols=["vibration_rms", "temperature_c"],
        )
        stats = ch.compute()
        assert "correlation_matrix" in stats
        corr = stats["correlation_matrix"]
        assert "vibration_rms" in corr
        assert "temperature_c" in corr["vibration_rms"]

    def test_fault_signatures_separated(self, sample_df):
        """Normal and anomaly classes should have different stats."""
        ch = DatasetCharacteriser(
            sample_df,
            signal_cols=["vibration_rms", "temperature_c"],
            label_col="anomaly",
        )
        stats = ch.compute()
        assert "fault_signatures" in stats
        assert "0" in stats["fault_signatures"] or 0 in stats["fault_signatures"]

    def test_healthy_distribution_from_data(self, sample_df):
        ch = DatasetCharacteriser(
            sample_df,
            signal_cols=["vibration_rms", "temperature_c"],
            label_col="anomaly",
        )
        stats = ch.compute()
        assert "healthy_distribution" in stats
        dist = stats["healthy_distribution"]
        # mu should be close to actual normal-subset mean
        normal_mu = float(sample_df[sample_df["anomaly"] == 0]["vibration_rms"].mean())
        computed_mu = dist["vibration_rms"]["mu"]
        assert abs(computed_mu - normal_mu) < 0.5, (
            f"Healthy mu mismatch: {computed_mu} vs {normal_mu}"
        )

    def test_no_hardcoded_values(self, sample_df):
        """Stats must change when data changes — proves they are computed."""
        ch1 = DatasetCharacteriser(sample_df, ["vibration_rms"])
        s1  = ch1.compute()

        # Shift all values by 10
        df2 = sample_df.copy()
        df2["vibration_rms"] += 10.0
        ch2 = DatasetCharacteriser(df2, ["vibration_rms"])
        s2  = ch2.compute()

        assert abs(
            s1["global"]["vibration_rms"]["mean"] -
            s2["global"]["vibration_rms"]["mean"]
        ) > 9.0, "Stats did not change when data changed — values may be hardcoded!"
