"""Tests — analytics.py inference compatibility.

These tests verify the COMPLETE chain:
  twin_models.joblib → window_features() → ML inference

They replicate EXACTLY what analytics.py does at runtime so that
any breaking change in the model bundle or feature engineering is
caught before it reaches the live MQTT pipeline.

analytics.py loads:
    M = joblib.load(MODEL_PATH)
    scaler, iso, clf = M["scaler"], M["iso"], M["clf"]
    s_good, s_bad    = M["s_good"], M["s_bad"]

and uses:
    x = window_features(buf).reshape(1, -1)
    score = float(iso.decision_function(scaler.transform(x))[0])
    health = clip((score - s_bad) / (s_good - s_bad), 0, 1) * 100
    proba = clf.predict_proba(x)[0]
    fault = clf.classes_[proba.argmax()]
"""

import warnings
import numpy as np
import pytest
import joblib
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.config import MODEL_PATH, SIGNALS, WINDOW, FAIL_THRESHOLD
from src.features import window_features, FEATURE_NAMES

MODEL_EXISTS = Path(MODEL_PATH).exists()


@pytest.mark.skipif(not MODEL_EXISTS, reason="twin_models.joblib not found — run src.train first")
class TestModelBundleKeys:
    """Verify twin_models.joblib has exactly the keys analytics.py expects."""

    @pytest.fixture(scope="class")
    def M(self):
        return joblib.load(MODEL_PATH)

    def test_required_keys_present(self, M):
        for key in ("scaler", "iso", "clf", "s_good", "s_bad"):
            assert key in M, f"Required key '{key}' missing from twin_models.joblib!"

    def test_scaler_type(self, M):
        from sklearn.preprocessing import StandardScaler
        assert isinstance(M["scaler"], StandardScaler)

    def test_iso_type(self, M):
        from sklearn.ensemble import IsolationForest
        assert isinstance(M["iso"], IsolationForest)

    def test_clf_type(self, M):
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(M["clf"], RandomForestClassifier)

    def test_s_good_is_float(self, M):
        assert isinstance(M["s_good"], float), f"s_good type: {type(M['s_good'])}"

    def test_s_bad_is_float(self, M):
        assert isinstance(M["s_bad"], float), f"s_bad type: {type(M['s_bad'])}"

    def test_s_good_greater_than_s_bad(self, M):
        """Health index formula requires s_good > s_bad to work correctly."""
        assert M["s_good"] > M["s_bad"], (
            f"s_good={M['s_good']:.4f} must be > s_bad={M['s_bad']:.4f} "
            "for the health index formula to produce valid [0, 100] output"
        )

    def test_clf_classes_match_faults(self, M):
        from src.config import FAULTS
        model_classes = set(M["clf"].classes_)
        expected      = set(FAULTS)
        assert model_classes == expected, (
            f"Classifier classes {model_classes} != expected FAULTS {expected}"
        )


@pytest.mark.skipif(not MODEL_EXISTS, reason="twin_models.joblib not found")
class TestInferencePipeline:
    """Replicate the analytics.py inference loop exactly."""

    @pytest.fixture(scope="class")
    def M(self):
        return joblib.load(MODEL_PATH)

    def _simulate_window(self, seed: int = 0) -> list:
        rng = np.random.default_rng(seed)
        # Simulate healthy readings close to Digital Twin nominal values
        buf = []
        for _ in range(WINDOW):
            buf.append([
                rng.normal(1480, 5),    # rpm
                rng.normal(14.0, 0.3),  # current_a
                rng.normal(55.0, 1.0),  # flow_m3h
                rng.normal(4.5, 0.1),   # pressure_bar
                rng.normal(1.5, 0.1),   # vibration_mms
                rng.normal(57.0, 0.3),  # temp_c
            ])
        return buf

    def test_window_features_shape(self):
        buf = self._simulate_window()
        x   = window_features(buf).reshape(1, -1)
        assert x.shape == (1, 14)

    def test_health_index_range_healthy(self, M):
        """Health index for a healthy window must be in [0, 100]."""
        scaler, iso = M["scaler"], M["iso"]
        s_good, s_bad = M["s_good"], M["s_bad"]

        buf   = self._simulate_window(seed=42)
        x     = window_features(buf).reshape(1, -1)
        score = float(iso.decision_function(scaler.transform(x))[0])
        hi    = float(np.clip((score - s_bad) / (s_good - s_bad), 0, 1) * 100)

        assert 0.0 <= hi <= 100.0, f"Health index {hi:.1f} is outside [0, 100]"

    def test_fault_prediction_returns_valid_class(self, M):
        from src.config import FAULTS
        clf = M["clf"]
        buf = self._simulate_window(seed=7)
        x   = window_features(buf).reshape(1, -1)
        proba = clf.predict_proba(x)[0]
        fault = str(clf.classes_[proba.argmax()])
        assert fault in FAULTS, f"Predicted fault '{fault}' not in FAULTS list"

    def test_confidence_is_probability(self, M):
        clf = M["clf"]
        buf = self._simulate_window()
        x   = window_features(buf).reshape(1, -1)
        proba = clf.predict_proba(x)[0]
        conf  = float(proba.max())
        assert 0.0 <= conf <= 1.0

    def test_rul_formula_no_crash(self):
        """RUL estimation formula from analytics.py must not crash."""
        from collections import deque
        hist = deque(maxlen=90)
        # Fill with declining health values
        for v in np.linspace(100, 50, 60):
            hist.append(v)

        y     = np.array(hist)
        slope = np.polyfit(np.arange(len(y)), y, 1)[0]
        if slope < -0.05:
            rul = float(max(0.0, (y[-10:].mean() - FAIL_THRESHOLD) / -slope))
        else:
            rul = -1.0
        assert rul > 0.0, f"Expected positive RUL for declining trend, got {rul}"

    def test_calibration_metadata_preserved(self, M):
        """After real_data_train, calibration_metadata key should be present."""
        if "calibration_metadata" in M:
            meta = M["calibration_metadata"]
            assert "skab_s_good" in meta
            assert "blend_alpha_real" in meta
            assert meta["blend_alpha_real"] == 0.30
        # If not present (model trained without real_data_train), that's OK
        # The test just checks it's consistent if present

    def test_feature_names_match_scaler(self, M):
        """Scaler feature count must match window_features output."""
        scaler = M["scaler"]
        n_features_scaler = scaler.n_features_in_
        assert n_features_scaler == len(FEATURE_NAMES) == 14, (
            f"Scaler expects {n_features_scaler} features but "
            f"window_features produces {len(FEATURE_NAMES)}"
        )
