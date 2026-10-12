"""Abstract base class for all dataset loaders.

Every dataset loader must subclass BaseDatasetLoader and implement:
  - download()        -- fetch raw files into raw_dir
  - load_raw()        -- read raw files into a DataFrame (no modifications yet)
  - preprocess()      -- clean, validate, convert; returns processed DataFrame
  - extract_features()-- compute feature vectors from processed data

All loaders preserve time ordering and never shuffle temporal data.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseDatasetLoader(ABC):
    """Abstract base for all real-world dataset loaders."""

    name: str = "base"
    description: str = ""

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir       = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._raw_df:   Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None

    # ── Abstract interface ──────────────────────────────────────────────────

    @abstractmethod
    def download(self) -> None:
        """Download raw dataset files into self.raw_dir."""

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Read raw files and return an unmodified DataFrame."""

    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """
        Clean and validate the raw data.

        Required behaviour:
          - Preserve time ordering
          - Handle missing values (document strategy)
          - Remove infinite / invalid values
          - Convert timestamps
          - Standardise units where possible
          - Log every transformation applied

        Returns cleaned DataFrame.
        """

    @abstractmethod
    def extract_features(self) -> pd.DataFrame:
        """
        Compute dataset-specific feature vectors.

        Must not use future information (no forward-looking windows).
        Must document the window size and any rolling parameters.

        Returns feature DataFrame with columns documented in the loader.
        """

    # ── Concrete helpers ────────────────────────────────────────────────────

    def validate_raw(self, df: pd.DataFrame) -> dict:
        """
        Compute data quality report for a raw DataFrame.

        Returns a dict with counts/fractions of quality issues.
        This is purely diagnostic; it does NOT modify df.
        """
        report = {
            "rows":           len(df),
            "cols":           len(df.columns),
            "missing_total":  int(df.isna().sum().sum()),
            "missing_frac":   round(float(df.isna().mean().mean()), 6),
            "duplicated_rows": int(df.duplicated().sum()),
            "inf_values":     int((df.select_dtypes("number") == float("inf")).sum().sum()
                                  + (df.select_dtypes("number") == float("-inf")).sum().sum()),
        }
        # per-column missing
        missing_per_col = df.isna().sum()
        report["missing_per_column"] = missing_per_col[missing_per_col > 0].to_dict()
        logger.info("[%s] Data quality report: %s", self.name, report)
        return report

    def chronological_split(
        self,
        df: pd.DataFrame,
        test_frac: float = 0.2,
        val_frac: float  = 0.1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split a time-ordered DataFrame into train / val / test WITHOUT shuffling.

        Uses a strict chronological cutoff so no future data leaks into training.

        Returns (train, val, test).
        """
        n = len(df)
        train_end = int(n * (1 - test_frac - val_frac))
        val_end   = int(n * (1 - test_frac))
        train = df.iloc[:train_end].copy()
        val   = df.iloc[train_end:val_end].copy()
        test  = df.iloc[val_end:].copy()
        logger.info(
            "[%s] Chronological split — train=%d  val=%d  test=%d",
            self.name, len(train), len(val), len(test),
        )
        return train, val, test

    def run(self, force_reprocess: bool = False) -> pd.DataFrame:
        """
        Convenience method: download → load_raw → preprocess → extract_features.

        Caches processed output; pass force_reprocess=True to recompute.
        """
        out_path = self.processed_dir / f"{self.name}_features.csv"
        if out_path.exists() and not force_reprocess:
            logger.info("[%s] Loading cached features from %s", self.name, out_path)
            return pd.read_csv(out_path)

        if not any(self.raw_dir.iterdir()) if self.raw_dir.exists() else True:
            logger.info("[%s] Raw directory empty, downloading...", self.name)
            self.download()

        self._raw_df   = self.load_raw()
        self._clean_df = self.preprocess()
        features       = self.extract_features()
        features.to_csv(out_path, index=False)
        logger.info("[%s] Features saved → %s  shape=%s", self.name, out_path, features.shape)
        return features
