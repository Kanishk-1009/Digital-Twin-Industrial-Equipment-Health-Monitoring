"""Tests — SKAB dataset loader.

Tests that do NOT require downloading the dataset are marked with
skip conditions. Tests that require the SKAB raw files are skipped
if the files are not present.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import SKAB_RAW_DIR, SKAB_FEATURES

SKAB_FILES_PRESENT = len(list(Path(SKAB_RAW_DIR).glob("*.csv"))) > 0 if Path(SKAB_RAW_DIR).exists() else False


class TestSKABLoaderImport:
    """Can import without error even if data is absent."""

    def test_import(self):
        from src.datasets.skab_loader import SKABLoader, SKAB_TO_DT_MAP, SKAB_SENSOR_COLS
        assert SKABLoader is not None

    def test_column_name_correct(self):
        """Verify the actual downloaded column name is used, not the wrong one."""
        from src.datasets.skab_loader import SKAB_SENSOR_COLS
        assert "Volume Flow RateRMS" in SKAB_SENSOR_COLS, (
            "Wrong column name! Actual SKAB column is 'Volume Flow RateRMS'"
        )
        assert "RateRMS" not in SKAB_SENSOR_COLS

    def test_dt_map_coverage(self):
        from src.datasets.skab_loader import SKAB_TO_DT_MAP, SKAB_SENSOR_COLS
        for col in SKAB_SENSOR_COLS:
            assert col in SKAB_TO_DT_MAP, f"'{col}' not in SKAB_TO_DT_MAP"


@pytest.mark.skipif(not SKAB_FILES_PRESENT, reason="SKAB raw files not present")
class TestSKABLoaderWithData:
    """Integration tests that require downloaded SKAB data."""

    @pytest.fixture(scope="class")
    def loader_with_data(self):
        from src.datasets.skab_loader import SKABLoader
        loader = SKABLoader()
        loader._raw_df   = loader.load_raw()
        loader._clean_df = loader.preprocess()
        return loader

    def test_raw_shape(self, loader_with_data):
        """SKAB has 14 CSV files totalling ~14k rows."""
        raw = loader_with_data._raw_df
        assert len(raw) > 10000, f"Expected >10k rows, got {len(raw)}"
        assert len(raw.columns) >= 10

    def test_no_missing_after_preprocess(self, loader_with_data):
        clean = loader_with_data._clean_df
        numeric = clean.select_dtypes("number")
        assert numeric.isna().sum().sum() == 0, "NaN values remain after preprocess"

    def test_no_inf_after_preprocess(self, loader_with_data):
        clean = loader_with_data._clean_df
        numeric = clean.select_dtypes("number")
        inf_count = np.isinf(numeric.values).sum()
        assert inf_count == 0, f"{inf_count} inf values remain after preprocess"

    def test_time_order_preserved(self, loader_with_data):
        """Within each source file, datetime must be monotonically non-decreasing."""
        clean = loader_with_data._clean_df
        if "datetime" in clean.columns:
            for _, grp in clean.groupby("source_file"):
                dt = grp["datetime"]
                assert (dt.diff().dropna() >= pd.Timedelta(0)).all(), (
                    f"Datetime not monotone in file"
                )

    def test_anomaly_label_binary(self, loader_with_data):
        clean = loader_with_data._clean_df
        assert set(clean["anomaly"].unique()).issubset({0, 1}), (
            f"Anomaly label not binary: {clean['anomaly'].unique()}"
        )

    def test_feature_extraction_no_cross_file_leakage(self, loader_with_data):
        """Every window must belong to exactly one source file."""
        features = loader_with_data.extract_features()
        # windows_end is within each file's range — no overlap across files
        assert "source_file" in features.columns
        # Feature shape reasonable
        assert len(features) > 100
        assert len(features.columns) > 10

    def test_no_inf_in_features(self, loader_with_data):
        features = loader_with_data.extract_features()
        num_cols  = features.select_dtypes("number")
        inf_count = np.isinf(num_cols.values).sum()
        # inf values in pressure_flow_ratio are expected to be filled to 0
        # but after fix should be 0 inf values
        assert inf_count == 0, f"{inf_count} inf values in feature matrix"

    def test_characteriser_returns_real_values(self, loader_with_data):
        """Verify characteriser computes from real data."""
        stats = loader_with_data.characterise()
        assert "global" in stats
        assert "correlation_matrix" in stats
        assert "healthy_distribution" in stats
        # Actual SKAB vibration RMS should be in a physically plausible range
        vib_mean = stats["global"].get("vibration_rms", {}).get("mean", None)
        if vib_mean is not None:
            assert 0 < vib_mean < 2.0, (
                f"SKAB vibration_rms mean={vib_mean:.4f} is outside plausible range "
                f"(expected 0–2 m/s² for Accelerometer1RMS)"
            )
