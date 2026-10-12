"""AI4I 2020 Predictive Maintenance dataset loader.

Dataset overview
----------------
AI4I 2020 is a SYNTHETIC dataset designed to represent predictive-maintenance
scenarios.  It is NOT real measured industrial data.

This loader treats it as a secondary classification benchmark only.
No claim is made that AI4I represents real sensor measurements.

Available columns (actual UCI CSV):
  UDI              : unique identifier (row index)
  Product ID       : product/run identifier
  Type             : quality class (L, M, H)
  Air temperature [K]    : air temperature in Kelvin
  Process temperature [K]: process temperature in Kelvin
  Rotational speed [rpm]  : rotational speed
  Torque [Nm]             : torque
  Tool wear [min]         : accumulated tool wear in minutes
  Machine failure         : 0/1 overall failure flag
  TWF                     : Tool Wear Failure
  HDF                     : Heat Dissipation Failure
  PWF                     : Power Failure
  OSF                     : Overstrain Failure
  RNF                     : Random Failure

Fault label mapping to Digital Twin categories
----------------------------------------------
AI4I label  → Normalized label      → DT category        Mapping confidence
-----------   -------------------    ---------------      ------------------
TWF (tool)  → bearing_wear (proxy) → bearing_wear        Medium (proxy)
HDF (heat)  → cooling_failure      → cooling_failure     Medium
PWF (power) → electrical_fault     → (not in DT)         Low
OSF (overst)→ imbalance (proxy)    → imbalance           Low
RNF (random)→ random_failure       → (not in DT)         Very Low
Healthy (no failure) → healthy     → healthy             High

IMPORTANT: These mappings are proxies, not direct equivalents.
The system explicitly reports which classes are directly observed vs proxy-mapped.

Download
--------
UCI Machine Learning Repository (public, no registration):
  https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import AI4I_RAW_DIR, PROCESSED_ROOT
from src.datasets.base_loader import BaseDatasetLoader
from src.feature_engineering import DatasetCharacteriser

logger = logging.getLogger(__name__)

AI4I_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
)

# Actual column names in the AI4I CSV (some may have minor variations)
AI4I_NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
AI4I_FAILURE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
AI4I_LABEL_COL   = "Machine failure"
AI4I_TYPE_COL    = "Type"

# Rename to standardised names for the common feature space
AI4I_RENAME_MAP = {
    "Air temperature [K]":    "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]":            "torque_nm",
    "Tool wear [min]":        "tool_wear_min",
    "Machine failure":        "failure",
}

# Feature columns after rename
AI4I_FEATURE_COLS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

# Derived temperature delta (process_temp - air_temp)
# Available in AI4I, not available in Digital Twin directly
AI4I_DERIVED_FEATURES = ["temp_delta_k", "power_proxy"]

# Explicit fault label → normalized label mapping
FAULT_LABEL_MAP = {
    "TWF": ("bearing_wear",    "Medium  (tool wear ≈ bearing wear proxy)"),
    "HDF": ("cooling_failure", "Medium  (heat dissipation → cooling failure analog)"),
    "PWF": ("electrical_fault","Low     (power failure — not modelled in DT)"),
    "OSF": ("imbalance",       "Low     (overstrain ≈ mechanical imbalance proxy)"),
    "RNF": ("random_failure",  "Very Low (random — no physical mapping)"),
}


class AI4ILoader(BaseDatasetLoader):
    """Loader for AI4I 2020 Predictive Maintenance dataset.

    Used as a secondary classification benchmark.
    Clearly labelled as synthetic data throughout.
    """

    name        = "ai4i"
    description = (
        "AI4I 2020 — SYNTHETIC predictive-maintenance dataset from UCI. "
        "Used as classification benchmark only. Not real industrial measurements."
    )

    def __init__(
        self,
        raw_dir:       str = AI4I_RAW_DIR,
        processed_dir: str = PROCESSED_ROOT,
    ):
        super().__init__(raw_dir, processed_dir)

    # ── Download ────────────────────────────────────────────────────────────

    def download(self) -> None:
        """Download AI4I CSV from UCI repository."""
        dest = self.raw_dir / "ai4i2020.csv"
        if dest.exists():
            logger.info("[AI4I] Raw file already present. Skipping download.")
            return

        logger.info("[AI4I] Downloading from UCI: %s", AI4I_URL)
        resp = requests.get(AI4I_URL, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.info("[AI4I] Saved → %s  (%d bytes)", dest, len(resp.content))

    # ── Load raw ────────────────────────────────────────────────────────────

    def load_raw(self) -> pd.DataFrame:
        """Read the AI4I CSV file."""
        candidates = list(self.raw_dir.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV in {self.raw_dir}. Run download() first.")
        fp  = candidates[0]
        raw = pd.read_csv(fp)
        logger.info("[AI4I] load_raw: %s  shape=%s", fp.name, raw.shape)
        return raw

    # ── Preprocess ──────────────────────────────────────────────────────────

    def preprocess(self) -> pd.DataFrame:
        """
        Clean and validate the AI4I DataFrame.

        Transformations (all logged):
          1. Validate expected columns are present.
          2. Drop duplicate rows.
          3. Check for missing values (report but do not infer).
          4. Convert temperatures K → °C for physical interpretability.
          5. Encode Type (L/M/H) as ordinal 0/1/2.
          6. Rename columns to standardised names.
          7. Create derived features: temp_delta, power_proxy.
          8. Create multi-class fault label (priority order: TWF>HDF>PWF>OSF>RNF>healthy).

        No temporal rolling is applied — AI4I rows are independent operation
        cycles, not a continuous time series.
        """
        raw = self._raw_df if self._raw_df is not None else self.load_raw()
        df  = raw.copy()

        # 1. Validate columns
        expected = AI4I_NUMERIC_COLS + AI4I_FAILURE_COLS + [AI4I_LABEL_COL]
        missing_cols = [c for c in expected if c not in df.columns]
        if missing_cols:
            logger.warning("[AI4I] Missing expected columns: %s", missing_cols)

        # 2. Duplicates
        n_dup = df.duplicated().sum()
        if n_dup:
            df = df.drop_duplicates().reset_index(drop=True)
            logger.info("[AI4I] Removed %d duplicate rows.", n_dup)

        # 3. Missing values
        miss = df[AI4I_NUMERIC_COLS].isna().sum()
        if miss.any():
            logger.warning("[AI4I] Missing values per column:\n%s", miss[miss > 0])

        # 4. Convert temperatures K → C
        for kcol, ccol in [
            ("Air temperature [K]",     "Air temperature [C]"),
            ("Process temperature [K]", "Process temperature [C]"),
        ]:
            if kcol in df.columns:
                df[ccol] = df[kcol] - 273.15
                logger.info("[AI4I] Converted %s → %s", kcol, ccol)

        # 5. Encode Type
        type_map = {"L": 0, "M": 1, "H": 2}
        if AI4I_TYPE_COL in df.columns:
            df["product_type_ord"] = df[AI4I_TYPE_COL].map(type_map).fillna(0).astype(int)

        # 6. Rename numeric columns to standard names
        df = df.rename(columns=AI4I_RENAME_MAP)

        # 7. Derived features (computed from actual column values, not hardcoded)
        if "process_temp_k" in df.columns and "air_temp_k" in df.columns:
            df["temp_delta_k"] = df["process_temp_k"] - df["air_temp_k"]
            logger.info("[AI4I] Created temp_delta_k feature.")
        if "rotational_speed_rpm" in df.columns and "torque_nm" in df.columns:
            # Power ∝ RPM × Torque (not in watts — proportional proxy)
            df["power_proxy"] = df["rotational_speed_rpm"] * df["torque_nm"]
            logger.info("[AI4I] Created power_proxy feature.")

        # 8. Multi-class fault label (priority order matches physical severity)
        fault_priority = ["TWF", "HDF", "OSF", "PWF", "RNF"]
        def _assign_fault(row):
            for f in fault_priority:
                if f in row and row[f] == 1:
                    return FAULT_LABEL_MAP[f][0]
            return "healthy"
        df["fault_label"] = df.apply(_assign_fault, axis=1)
        logger.info("[AI4I] Fault label distribution:\n%s", df["fault_label"].value_counts())

        self._clean_df = df
        quality = self.validate_raw(df)
        logger.info("[AI4I] Preprocess complete. Shape=%s  Quality=%s", df.shape, quality)
        return df

    # ── Feature extraction ──────────────────────────────────────────────────

    def extract_features(self) -> pd.DataFrame:
        """
        Extract feature matrix from AI4I.

        Since AI4I rows are independent per-cycle records (not a time series),
        no rolling window is applied (AI4I_WINDOW = 1).

        Features:
          - air_temp_k, process_temp_k, rotational_speed_rpm, torque_nm, tool_wear_min
          - temp_delta_k, power_proxy   (derived, computed from actual values)
          - product_type_ord

        Labels:
          - failure (0/1)
          - fault_label (multi-class: healthy / bearing_wear / cooling_failure / ...)

        Splitting:
          Chronological — first 70% train, next 10% val, last 20% test.
          (Rows are in UDI order, which is the original recording order.)
        """
        df = self._clean_df if self._clean_df is not None else self.preprocess()

        feat_cols = (
            AI4I_FEATURE_COLS
            + ["temp_delta_k", "power_proxy", "product_type_ord"]
        )
        feat_cols = [c for c in feat_cols if c in df.columns]

        out = df[feat_cols + ["failure", "fault_label"]].copy()

        # Add train/val/test split column (chronological — no shuffle)
        n = len(out)
        split = np.full(n, "train", dtype=object)
        split[int(n * 0.7):int(n * 0.8)] = "val"
        split[int(n * 0.8):]              = "test"
        out["split"] = split

        logger.info(
            "[AI4I] Features extracted: shape=%s  failure_rate=%.4f",
            out.shape, float(out["failure"].mean()),
        )
        logger.info("[AI4I] Fault label distribution:\n%s", out["fault_label"].value_counts())
        return out

    # ── Characterise ────────────────────────────────────────────────────────

    def characterise(self) -> dict:
        """Compute DatasetCharacteriser statistics on the cleaned AI4I data."""
        df        = self._clean_df if self._clean_df is not None else self.preprocess()
        available = [c for c in AI4I_FEATURE_COLS + ["temp_delta_k", "power_proxy"]
                     if c in df.columns]
        ch = DatasetCharacteriser(
            df            = df,
            signal_cols   = available,
            label_col     = "failure",
            window        = 1,
            dataset_name  = "ai4i",
        )
        return ch.compute()

    # ── Fault mapping report ─────────────────────────────────────────────────

    @staticmethod
    def fault_mapping_report() -> str:
        """Return a plain-text summary of the AI4I -> DT fault label mapping."""
        lines = [
            "AI4I 2020 -> Digital Twin Fault Mapping",
            "=" * 50,
            "DATASET TYPE: SYNTHETIC (not real industrial measurements)",
            "",
            f"{'AI4I Label':<12} {'Normalized Label':<20} {'DT Category':<18} Confidence",
            "-" * 80,
        ]
        for ai4i_lbl, (norm_lbl, confidence) in FAULT_LABEL_MAP.items():
            dt_cat = norm_lbl if norm_lbl in ("bearing_wear", "cooling_failure", "imbalance") else "(not in DT)"
            lines.append(f"{ai4i_lbl:<12} {norm_lbl:<20} {dt_cat:<18} {confidence}")
        lines += [
            "",
            "Note: Only TWF->bearing_wear and HDF->cooling_failure have medium-confidence",
            "      mappings to Digital Twin fault categories. All others are low-confidence",
            "      proxies and should not be used as ground truth for DT fault classes.",
        ]
        return "\n".join(lines)

