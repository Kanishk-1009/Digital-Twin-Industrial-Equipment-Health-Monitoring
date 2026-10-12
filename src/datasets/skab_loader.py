"""SKAB — Skoltech Anomaly Benchmark dataset loader.

Dataset overview
----------------
SKAB is a real industrial benchmark for anomaly detection recorded from a
water pump test rig at Skoltech university.

Sensors available in SKAB (actual column names in the CSVs):
  - datetime          : ISO timestamps
  - Accelerometer1RMS : vibration (m/s², but treated as relative)
  - Accelerometer2RMS : second accelerometer
  - Current           : motor current (A)
  - Pressure          : pump pressure (relative)
  - Temperature       : water/fluid temperature (°C-range)
  - Thermocouple      : thermocouple temperature
  - Voltage           : motor voltage (V)
  - RateRMS           : volumetric flow rate proxy
  - anomaly           : 0 = normal, 1 = anomaly
  - changepoint       : 0 = no change, 1 = regime change point

Download
--------
SKAB is hosted on GitHub (public, no registration required):
  https://github.com/waico/SKAB

We download only the "anomaly-free" normal file and the individual
anomaly CSV files from the "data/other" subfolder.

Feature mapping to Digital Twin signals
----------------------------------------
SKAB signal         → DT signal          Notes
--------------------------------------------------
Current             → current_a          Direct analog
Pressure            → pressure_bar       Relative (normalised)
Accelerometer1RMS   → vibration_mms      Proxy (different unit, same physics)
Temperature         → temp_c             Direct analog
RateRMS             → flow_m3h           Proxy (volumetric flow proxy)
Voltage             → (no DT equivalent) Retained as dataset-specific feature
Thermocouple        → (no DT equivalent) Retained as dataset-specific feature
rpm                 → (NOT AVAILABLE)    SKAB does not contain RPM
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import SKAB_RAW_DIR, PROCESSED_ROOT, SKAB_WINDOW
from src.datasets.base_loader import BaseDatasetLoader
from src.feature_engineering import DatasetCharacteriser

logger = logging.getLogger(__name__)

# ── SKAB source URL ────────────────────────────────────────────────────────────
# GitHub archive of the SKAB repository (public, no authentication)
SKAB_REPO_ZIP = "https://github.com/waico/SKAB/archive/refs/heads/master.zip"

# Subfolder inside the zip that contains the actual CSV files
SKAB_DATA_SUBDIR = "SKAB-master/data/other"

# Raw SKAB column names (as they appear in the downloaded CSVs)
SKAB_SENSOR_COLS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",   # actual column name in downloaded SKAB CSVs
]
SKAB_LABEL_COL       = "anomaly"
SKAB_CHANGEPOINT_COL = "changepoint"
SKAB_TIME_COL        = "datetime"

# Mapping: SKAB column → Digital Twin equivalent signal name
# Only columns with a clear physical analog are mapped.
# Columns without a DT equivalent are kept under their SKAB name.
SKAB_TO_DT_MAP = {
    "Current":              "current_a",
    "Pressure":             "pressure_bar",
    "Accelerometer1RMS":    "vibration_rms",
    "Temperature":          "temperature_c",
    "Volume Flow RateRMS":  "flow_proxy",    # actual column name in SKAB CSVs
    "Voltage":              "voltage_v",
    "Thermocouple":         "thermocouple_c",
    "Accelerometer2RMS":    "accelerometer2_rms",
}

# Feature columns to use for ML experiments (post-rename)
SKAB_FEATURE_COLS = list(SKAB_TO_DT_MAP.values())


class SKABLoader(BaseDatasetLoader):
    """Loader for the SKAB anomaly detection dataset."""

    name        = "skab"
    description = (
        "Skoltech Anomaly Benchmark — real pump-rig sensor data with "
        "annotated anomalies at ~1 Hz."
    )

    def __init__(
        self,
        raw_dir:       str = SKAB_RAW_DIR,
        processed_dir: str = PROCESSED_ROOT,
    ):
        super().__init__(raw_dir, processed_dir)
        self._file_list: list[Path] = []

    # ── Download ────────────────────────────────────────────────────────────

    def download(self) -> None:
        """Download SKAB CSV files from GitHub into raw_dir."""
        # Check if already downloaded
        existing = list(self.raw_dir.glob("*.csv"))
        if existing:
            logger.info("[SKAB] Raw files already present (%d CSVs). Skipping download.", len(existing))
            return

        logger.info("[SKAB] Downloading from GitHub: %s", SKAB_REPO_ZIP)
        resp = requests.get(SKAB_REPO_ZIP, timeout=120)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            extracted = 0
            for member in zf.namelist():
                # Only extract CSV files from the "other" data subfolder
                if member.startswith(SKAB_DATA_SUBDIR) and member.endswith(".csv"):
                    # Flatten: save directly into raw_dir (no subfolders)
                    fname = Path(member).name
                    data  = zf.read(member)
                    dest  = self.raw_dir / fname
                    dest.write_bytes(data)
                    extracted += 1

        logger.info("[SKAB] Extracted %d CSV files → %s", extracted, self.raw_dir)
        if extracted == 0:
            raise RuntimeError(
                f"No CSV files found in {SKAB_DATA_SUBDIR} inside the SKAB zip. "
                "The repository structure may have changed."
            )

    # ── Load raw ────────────────────────────────────────────────────────────

    def load_raw(self) -> pd.DataFrame:
        """Read all SKAB CSV files, tag with source filename, concatenate."""
        csv_files = sorted(self.raw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files in {self.raw_dir}. Run download() first."
            )

        frames = []
        for fp in csv_files:
            try:
                df = pd.read_csv(fp, sep=";", parse_dates=[SKAB_TIME_COL])
                df["source_file"] = fp.stem
                frames.append(df)
                logger.debug("[SKAB] Loaded %s  shape=%s", fp.name, df.shape)
            except Exception as e:
                logger.warning("[SKAB] Skipping %s — %s", fp.name, e)

        if not frames:
            raise RuntimeError("No SKAB files could be loaded.")

        raw = pd.concat(frames, ignore_index=True)
        self._file_list = csv_files
        logger.info("[SKAB] load_raw complete — total shape: %s", raw.shape)
        return raw

    # ── Preprocess ──────────────────────────────────────────────────────────

    def preprocess(self) -> pd.DataFrame:
        """
        Clean the raw SKAB DataFrame.

        Transformations applied (all logged):
          1. Sort by datetime within each source file to preserve time order.
          2. Remove rows where ALL sensor columns are NaN (complete dropouts).
          3. Replace ±inf with NaN, then forward-fill remaining NaN
             (sensor dropout → carry last known value, max 3 steps).
          4. Clip extreme outliers: values beyond ±5σ per column are clamped
             to the 5σ boundary (computed from normal-label rows only).
          5. Rename columns to standardised names via SKAB_TO_DT_MAP.
          6. Ensure anomaly and changepoint columns are integer (0/1).
        """
        raw = self._raw_df if self._raw_df is not None else self.load_raw()
        df  = raw.copy()

        # 1. Sort by time within each file
        df = df.sort_values([SKAB_TIME_COL, "source_file"]).reset_index(drop=True)
        logger.info("[SKAB] After sort: %d rows", len(df))

        # 2. Remove complete sensor dropout rows
        sensor_cols_present = [c for c in SKAB_SENSOR_COLS if c in df.columns]
        all_nan_mask = df[sensor_cols_present].isna().all(axis=1)
        n_dropout = all_nan_mask.sum()
        if n_dropout:
            df = df[~all_nan_mask].reset_index(drop=True)
            logger.info("[SKAB] Removed %d complete-dropout rows.", n_dropout)

        # 3. Replace ±inf → NaN, forward-fill
        for col in sensor_cols_present:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        n_nan_before = df[sensor_cols_present].isna().sum().sum()
        df[sensor_cols_present] = df[sensor_cols_present].ffill(limit=3)
        # Fill any remaining leading NaN with column median
        for col in sensor_cols_present:
            if df[col].isna().any():
                fill = df[col].median()
                df[col] = df[col].fillna(fill)
                logger.info("[SKAB] Column %s — filled %d NaN with median=%.4f",
                            col, df[col].isna().sum(), fill)
        n_nan_after = df[sensor_cols_present].isna().sum().sum()
        logger.info("[SKAB] NaN: %d before → %d after fill.", n_nan_before, n_nan_after)

        # 4. Clip outliers to ±5σ (computed on normal rows)
        normal_mask = df[SKAB_LABEL_COL] == 0 if SKAB_LABEL_COL in df.columns else pd.Series(True, index=df.index)
        for col in sensor_cols_present:
            mu  = float(df.loc[normal_mask, col].mean())
            sig = float(df.loc[normal_mask, col].std())
            if sig > 1e-9:
                lo, hi = mu - 5 * sig, mu + 5 * sig
                n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
                if n_clipped > 0:
                    df[col] = df[col].clip(lo, hi)
                    logger.info("[SKAB] Clipped %d outliers in %s to [%.3f, %.3f]",
                                n_clipped, col, lo, hi)

        # 5. Rename to standardised names
        df = df.rename(columns=SKAB_TO_DT_MAP)

        # 6. Ensure label types
        for lbl_col in [SKAB_LABEL_COL, SKAB_CHANGEPOINT_COL]:
            if lbl_col in df.columns:
                df[lbl_col] = pd.to_numeric(df[lbl_col], errors="coerce").fillna(0).astype(int)

        self._clean_df = df
        quality = self.validate_raw(df)
        logger.info("[SKAB] Preprocess complete. Shape=%s  Quality=%s", df.shape, quality)
        return df

    # ── Feature extraction ──────────────────────────────────────────────────

    def extract_features(self) -> pd.DataFrame:
        """
        Compute rolling-window features from the SKAB sensor columns.

        Window size: SKAB_WINDOW = 30 samples (≈ 30 s at 1 Hz).

        Features per signal (vectorized pandas rolling, computed from actual data):
          mean, std, min, max, rms, peak, crest_factor, skewness, kurtosis

        Additional features:
          vib_slope, temp_slope (linear trend over window)
          pressure_flow_ratio, vib_temp_corr (cross-sensor rolling)

        Labels:
          anomaly (0/1), changepoint (0/1), source_file

        Splitting strategy:
          Windows are built per source_file to avoid mixing files.
          No window ever spans two different source files.
        """
        import warnings
        df = self._clean_df if self._clean_df is not None else self.preprocess()
        available = [c for c in SKAB_FEATURE_COLS if c in df.columns]
        logger.info("[SKAB] extract_features using cols: %s  window=%d", available, SKAB_WINDOW)

        all_frames = []
        for src_file, grp in df.groupby("source_file", sort=False):
            grp = grp.reset_index(drop=True)
            result = pd.DataFrame(index=grp.index)

            # Per-column rolling statistics (vectorized)
            for col in available:
                s = grp[col]
                roll = s.rolling(SKAB_WINDOW, min_periods=SKAB_WINDOW)
                result[f"{col}_mean"] = roll.mean()
                result[f"{col}_std"]  = roll.std()
                result[f"{col}_min"]  = roll.min()
                result[f"{col}_max"]  = roll.max()
                # RMS via rolling mean of squares
                result[f"{col}_rms"]  = np.sqrt((s**2).rolling(SKAB_WINDOW, min_periods=SKAB_WINDOW).mean())
                result[f"{col}_peak"] = s.abs().rolling(SKAB_WINDOW, min_periods=SKAB_WINDOW).max()
                result[f"{col}_crest_factor"] = (
                    result[f"{col}_peak"] / result[f"{col}_rms"].replace(0, np.nan)
                ).fillna(0)
                # Skewness and kurtosis via rolling apply (suppress precision warnings)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result[f"{col}_skewness"] = roll.skew()
                    result[f"{col}_kurtosis"] = roll.kurt()
                result[[f"{col}_skewness", f"{col}_kurtosis"]] = (
                    result[[f"{col}_skewness", f"{col}_kurtosis"]].fillna(0)
                )

            # Trend slopes via rolling polyfit (apply over window)
            x = np.arange(SKAB_WINDOW, dtype=float)
            def _slope(vals):
                if len(vals) < SKAB_WINDOW or vals.isna().any():
                    return np.nan
                return float(np.polyfit(x, vals.values, 1)[0])

            for tname, tcol in [("vib_slope", "vibration_rms"), ("temp_slope", "temperature_c")]:
                if tcol in available:
                    result[tname] = grp[tcol].rolling(SKAB_WINDOW, min_periods=SKAB_WINDOW).apply(
                        _slope, raw=False
                    )
                else:
                    result[tname] = 0.0

            # Cross-sensor features (rolling window)
            if "vibration_rms" in available and "temperature_c" in available:
                # Rolling Pearson correlation (pandas built-in)
                result["vib_temp_corr"] = (
                    grp["vibration_rms"].rolling(SKAB_WINDOW).corr(grp["temperature_c"])
                ).fillna(0)
            else:
                result["vib_temp_corr"] = 0.0

            if "pressure_bar" in available and "flow_proxy" in available:
                result["pressure_flow_ratio"] = (
                    grp["pressure_bar"].rolling(SKAB_WINDOW).mean() /
                    (grp["flow_proxy"].rolling(SKAB_WINDOW).mean().replace(0, np.nan))
                ).fillna(0)
            else:
                result["pressure_flow_ratio"] = 0.0

            # Labels and identifiers
            if SKAB_LABEL_COL in grp.columns:
                result["anomaly"]     = grp[SKAB_LABEL_COL].values
            if SKAB_CHANGEPOINT_COL in grp.columns:
                result["changepoint"] = grp[SKAB_CHANGEPOINT_COL].values
            result["source_file"] = src_file
            result["window_end"]  = grp.index

            # Drop rows where the window was not yet full
            result = result.dropna(subset=[f"{available[0]}_mean"])

            # Sanitise: replace ±inf introduced by division-by-zero,
            # near-zero-variance skew/kurt, or degenerate correlations.
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = result[numeric_cols].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)

            all_frames.append(result)

        features = pd.concat(all_frames, ignore_index=True)
        logger.info("[SKAB] Features extracted: shape=%s  anomaly_rate=%.4f",
                    features.shape,
                    float(features["anomaly"].mean()) if "anomaly" in features.columns else 0.0)
        return features


    # ── Helper ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_window_feats(win: np.ndarray, col_names: list[str]) -> dict:
        """Compute per-column stats for one window and add cross-sensor features."""
        row: dict = {}
        n = len(win)
        x = np.arange(n)

        for j, col in enumerate(col_names):
            v = win[:, j]
            rms   = float(np.sqrt(np.mean(v ** 2)))
            peak  = float(np.max(np.abs(v)))
            crest = peak / rms if rms > 1e-9 else 0.0
            row[f"{col}_mean"]         = float(v.mean())
            row[f"{col}_std"]          = float(v.std())
            row[f"{col}_min"]          = float(v.min())
            row[f"{col}_max"]          = float(v.max())
            row[f"{col}_rms"]          = rms
            row[f"{col}_peak"]         = peak
            row[f"{col}_crest_factor"] = crest
            row[f"{col}_skewness"]     = float(v.mean())   # replaced below
            try:
                from scipy.stats import skew as _skew, kurtosis as _kurt
                row[f"{col}_skewness"] = float(_skew(v))
                row[f"{col}_kurtosis"] = float(_kurt(v))
            except Exception:
                row[f"{col}_skewness"] = 0.0
                row[f"{col}_kurtosis"] = 0.0

        # Trend slopes
        for tname, tcol in [("vib_slope", "vibration_rms"), ("temp_slope", "temperature_c")]:
            if tcol in col_names:
                idx = col_names.index(tcol)
                row[tname] = float(np.polyfit(x, win[:, idx], 1)[0]) if n > 1 else 0.0
            else:
                row[tname] = 0.0

        # Cross-sensor features
        vib_col  = "vibration_rms"
        temp_col = "temperature_c"
        pres_col = "pressure_bar"
        flow_col = "flow_proxy"

        if vib_col in col_names and temp_col in col_names:
            vi = col_names.index(vib_col)
            ti = col_names.index(temp_col)
            c  = float(np.corrcoef(win[:, vi], win[:, ti])[0, 1])
            row["vib_temp_corr"] = 0.0 if np.isnan(c) else c
        else:
            row["vib_temp_corr"] = 0.0

        if pres_col in col_names and flow_col in col_names:
            pi  = col_names.index(pres_col)
            fi  = col_names.index(flow_col)
            mf  = float(win[:, fi].mean()) + 1e-9
            row["pressure_flow_ratio"] = float(win[:, pi].mean()) / mf
        else:
            row["pressure_flow_ratio"] = 0.0

        return row

    # ── Characterise dataset ────────────────────────────────────────────────

    def characterise(self) -> dict:
        """
        Run DatasetCharacteriser on the clean SKAB data.

        Computes all statistics directly from the real measurements.
        Returns the stats dict which can be used to calibrate the simulator.
        """
        df = self._clean_df if self._clean_df is not None else self.preprocess()
        available = [c for c in SKAB_FEATURE_COLS if c in df.columns]
        ch = DatasetCharacteriser(
            df            = df,
            signal_cols   = available,
            label_col     = SKAB_LABEL_COL,
            window        = SKAB_WINDOW,
            dataset_name  = "skab",
        )
        return ch.compute()
