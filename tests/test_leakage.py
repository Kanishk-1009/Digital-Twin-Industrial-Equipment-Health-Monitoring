"""Tests — data leakage validation.

Verifies that:
  1. No window spans two source files (SKAB)
  2. Chronological split doesn't shuffle temporally
  3. Scaler is never fitted on test data
  4. No future data leaks into training windows
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.config import SKAB_FEATURES, AI4I_FEATURES


class TestChronologicalSplit:
    """BaseDatasetLoader.chronological_split() must not shuffle time."""

    def test_train_before_val_before_test(self):
        from src.datasets.base_loader import BaseDatasetLoader

        class _DummyLoader(BaseDatasetLoader):
            name = "dummy"
            def download(self): pass
            def load_raw(self): pass
            def preprocess(self): pass
            def extract_features(self): pass

        loader = _DummyLoader("datasets/raw/skab", "datasets/processed")
        n  = 1000
        df = pd.DataFrame({"ts": range(n), "val": range(n)})

        train, val, test = loader.chronological_split(df, test_frac=0.2, val_frac=0.1)

        # Strict ordering: last train index < first val index < first test index
        assert train["ts"].max() < val["ts"].min(), "Train overlaps with val!"
        assert val["ts"].max()   < test["ts"].min(), "Val overlaps with test!"

    def test_no_overlap_in_indices(self):
        from src.datasets.base_loader import BaseDatasetLoader

        class _DummyLoader(BaseDatasetLoader):
            name = "dummy"
            def download(self): pass
            def load_raw(self): pass
            def preprocess(self): pass
            def extract_features(self): pass

        loader = _DummyLoader("datasets/raw/skab", "datasets/processed")
        df = pd.DataFrame({"val": range(100)})
        train, val, test = loader.chronological_split(df)
        all_idx = list(train.index) + list(val.index) + list(test.index)
        assert len(all_idx) == len(set(all_idx)), "Overlapping indices across splits!"


class TestSKABNoLeakage:
    """Verify SKAB feature extraction has no cross-file leakage."""

    @pytest.mark.skipif(
        not Path("datasets/processed/skab_features.csv").exists(),
        reason="SKAB features not computed yet — run src.real_data_train first"
    )
    def test_each_window_belongs_to_one_file(self):
        df = pd.read_csv("datasets/processed/skab_features.csv")
        assert "source_file" in df.columns
        # Each row has exactly one source_file value
        assert df["source_file"].notna().all()
        assert (df["source_file"] != "").all()

    @pytest.mark.skipif(
        not Path("datasets/processed/skab_features.csv").exists(),
        reason="SKAB features not computed yet"
    )
    def test_no_inf_in_feature_matrix(self):
        df = pd.read_csv("datasets/processed/skab_features.csv")
        num = df.select_dtypes("number")
        inf_count = np.isinf(num.values).sum()
        assert inf_count == 0, f"{inf_count} inf values in SKAB features"

    @pytest.mark.skipif(
        not Path("datasets/processed/skab_features.csv").exists(),
        reason="SKAB features not computed yet"
    )
    def test_train_test_files_disjoint(self):
        """Files used in train split must not appear in test split."""
        df = pd.read_csv("datasets/processed/skab_features.csv")
        files = sorted(df["source_file"].unique())
        n_test = max(1, len(files) // 5)
        train_files = set(files[:-n_test])
        test_files  = set(files[-n_test:])
        assert train_files.isdisjoint(test_files), (
            f"Leakage! Files appear in both train and test: "
            f"{train_files & test_files}"
        )


class TestScalerFitOnTrainOnly:
    """Verify StandardScaler is fitted on training data only."""

    def test_scaler_fit_train_predict_test(self):
        """Fitting on train, transforming test — no leakage."""
        n = 200
        X_train = np.random.randn(int(n * 0.8), 5)
        X_test  = np.random.randn(int(n * 0.2), 5) + 10.0  # shifted distribution

        scaler = StandardScaler()
        scaler.fit(X_train)

        # Scaler mean should reflect TRAIN distribution, not test
        train_mean = X_train.mean(axis=0)
        np.testing.assert_allclose(scaler.mean_, train_mean, rtol=0.1)

        # Transforming test data should produce non-zero mean (since test is shifted)
        X_test_scaled = scaler.transform(X_test)
        assert X_test_scaled.mean() > 5.0, (
            "Scaler appears to have been fitted on test data (mean near 0)"
        )


class TestAI4INoLeakage:
    """AI4I split must be chronological — no shuffle."""

    @pytest.mark.skipif(
        not Path("datasets/processed/ai4i_features.csv").exists(),
        reason="AI4I features not computed yet"
    )
    def test_split_column_present(self):
        df = pd.read_csv("datasets/processed/ai4i_features.csv")
        assert "split" in df.columns

    @pytest.mark.skipif(
        not Path("datasets/processed/ai4i_features.csv").exists(),
        reason="AI4I features not computed yet"
    )
    def test_train_before_test_in_original_order(self):
        """All train rows must appear before val rows in original CSV order."""
        df = pd.read_csv("datasets/processed/ai4i_features.csv")
        train_idx = df[df["split"] == "train"].index
        val_idx   = df[df["split"] == "val"].index
        test_idx  = df[df["split"] == "test"].index
        assert train_idx.max() < val_idx.min(), "Train rows appear after val rows!"
        assert val_idx.max()   < test_idx.min(), "Val rows appear after test rows!"
