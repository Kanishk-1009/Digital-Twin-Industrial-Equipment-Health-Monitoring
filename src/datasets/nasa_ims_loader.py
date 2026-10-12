"""NASA IMS Bearing Dataset loader — OPTIONAL.

This dataset requires manual download (~6.7 GB).
Registration is required at:
  https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

When data is present, set NASA_IMS_AVAILABLE = True in config.py.

Dataset overview
----------------
NASA IMS contains run-to-failure vibration data from 4 bearings
on a rotating shaft, recorded at 20 kHz, in 3 separate test runs.

Signals:
  - ch1..ch4  : vibration channels (1–4 bearings)
  - timestamp  : inferred from filename

Use for:
  - degradation analysis
  - RUL methodology validation
  - bearing wear characterisation

RUL ground truth:
  End-of-life is defined by the last recorded file in each test run.
  Surrogate RUL = (final_file_index - current_file_index) × interval_minutes.
  This is an approximation, not a certified industrial RUL label.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import NASA_IMS_RAW_DIR, PROCESSED_ROOT, NASA_IMS_WINDOW, NASA_IMS_AVAILABLE
from src.datasets.base_loader import BaseDatasetLoader

logger = logging.getLogger(__name__)

NASA_IMS_CHANNELS = ["ch1", "ch2", "ch3", "ch4"]


class NASAIMSLoader(BaseDatasetLoader):
    """Optional loader for NASA IMS Bearing Dataset.

    Will raise DatasetUnavailableError if NASA_IMS_AVAILABLE = False
    or if raw files are not found.
    """

    name        = "nasa_ims"
    description = (
        "NASA IMS Bearing Dataset — run-to-failure vibration data. "
        "Optional: requires manual download (~6.7 GB)."
    )

    class DatasetUnavailableError(RuntimeError):
        pass

    def __init__(
        self,
        raw_dir:       str = NASA_IMS_RAW_DIR,
        processed_dir: str = PROCESSED_ROOT,
    ):
        super().__init__(raw_dir, processed_dir)
        if not NASA_IMS_AVAILABLE:
            logger.info(
                "[NASA-IMS] NASA_IMS_AVAILABLE=False. "
                "Set to True in config.py after downloading the dataset from "
                "https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository"
            )

    def _check_available(self):
        if not NASA_IMS_AVAILABLE:
            raise NASAIMSLoader.DatasetUnavailableError(
                "NASA IMS dataset not available. "
                "Set NASA_IMS_AVAILABLE=True in config.py and download the data from "
                "https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository"
            )
        files = list(self.raw_dir.glob("*.txt")) + list(self.raw_dir.glob("*.csv"))
        if not files:
            raise NASAIMSLoader.DatasetUnavailableError(
                f"No data files found in {self.raw_dir}. "
                "Download the dataset and place .txt files (bearing data) in this directory."
            )

    def download(self) -> None:
        """NASA IMS requires manual download — provides instructions."""
        msg = (
            "\n"
            "NASA IMS dataset requires manual download (registration required):\n"
            "  URL : https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository\n"
            "  File: IMS Bearing Dataset (Bearing.zip, ~6.7 GB)\n"
            f"  Dest: {self.raw_dir}\n"
            "After downloading:\n"
            "  1. Extract the .txt files for Test Set 1 (or 2/3) into datasets/raw/nasa_ims/\n"
            "  2. Set NASA_IMS_AVAILABLE = True in src/config.py\n"
        )
        logger.info(msg)
        print(msg)

    def load_raw(self) -> pd.DataFrame:
        self._check_available()
        files = sorted(self.raw_dir.glob("*.txt"))
        frames = []
        for i, fp in enumerate(files):
            try:
                df = pd.read_csv(fp, sep="\t", header=None,
                                 names=NASA_IMS_CHANNELS)
                df["file_index"] = i
                df["filename"]   = fp.stem
                frames.append(df)
            except Exception as e:
                logger.warning("[NASA-IMS] Skipping %s — %s", fp.name, e)
        if not frames:
            raise RuntimeError(f"No readable files in {self.raw_dir}")
        raw = pd.concat(frames, ignore_index=True)
        logger.info("[NASA-IMS] load_raw: %d files, %d rows total", len(files), len(raw))
        return raw

    def preprocess(self) -> pd.DataFrame:
        self._check_available()
        raw = self._raw_df if self._raw_df is not None else self.load_raw()
        df  = raw.copy()
        # Replace inf
        for col in NASA_IMS_CHANNELS:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # Forward-fill short dropouts
        df[NASA_IMS_CHANNELS] = df[NASA_IMS_CHANNELS].ffill(limit=5)
        df[NASA_IMS_CHANNELS] = df[NASA_IMS_CHANNELS].fillna(0.0)
        self._clean_df = df
        logger.info("[NASA-IMS] preprocess complete: shape=%s", df.shape)
        return df

    def extract_features(self) -> pd.DataFrame:
        """
        Extract per-file (per ~1-second segment) vibration features.

        Window = NASA_IMS_WINDOW (2048 samples ≈ 0.1 s at 20 kHz).

        Features per channel: rms, peak, crest_factor, kurtosis, skewness
        RUL: surrogate = (total_files - file_index) × 1 (arbitrary time units).
             Explicitly labelled as surrogate RUL, not ground truth.
        """
        self._check_available()
        df = self._clean_df if self._clean_df is not None else self.preprocess()

        from scipy.stats import skew as _skew, kurtosis as _kurt

        rows = []
        for file_idx, grp in df.groupby("file_index"):
            for ch in NASA_IMS_CHANNELS:
                vals = grp[ch].values
                # Process in windows
                for start in range(0, len(vals) - NASA_IMS_WINDOW + 1, NASA_IMS_WINDOW):
                    win = vals[start:start + NASA_IMS_WINDOW]
                    rms   = float(np.sqrt(np.mean(win**2)))
                    peak  = float(np.max(np.abs(win)))
                    crest = peak / rms if rms > 1e-9 else 0.0
                    rows.append({
                        "file_index": int(file_idx),
                        "channel":    ch,
                        "rms":        rms,
                        "peak":       peak,
                        "crest_factor": crest,
                        "kurtosis":   float(_kurt(win)),
                        "skewness":   float(_skew(win)),
                        "std":        float(win.std()),
                        "mean":       float(win.mean()),
                    })

        feat = pd.DataFrame(rows)
        # Surrogate RUL — explicitly documented as approximation
        max_file = int(feat["file_index"].max())
        feat["surrogate_rul_files"] = max_file - feat["file_index"]
        feat["rul_note"] = (
            "surrogate_rul: (max_file_index - current_file_index). "
            "NOT a certified industrial RUL ground truth."
        )
        logger.info(
            "[NASA-IMS] Features extracted: shape=%s  "
            "Note: RUL is surrogate approximation only.", feat.shape
        )
        return feat
