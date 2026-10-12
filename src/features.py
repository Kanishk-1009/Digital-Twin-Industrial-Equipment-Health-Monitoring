"""Re-exports from feature_engineering for backward compatibility.

analytics.py and generate_dataset.py import window_features and
FEATURE_NAMES from here. The actual implementation lives in
feature_engineering.py.
"""
from src.feature_engineering import window_features, FEATURE_NAMES  # noqa: F401
