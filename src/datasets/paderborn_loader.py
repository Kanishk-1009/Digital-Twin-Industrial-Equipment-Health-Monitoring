"""Paderborn University Bearing Dataset loader — OPTIONAL.

This dataset requires manual download (~32 GB) and free registration at:
  https://mb.uni-paderborn.de/en/kat/main-research/kat-datacenter/bearing-datacenter/data-sets-and-download

Use for:
  - bearing fault classification (inner/outer race, ball faults)
  - vibration-based feature engineering validation
  - bearing wear characterisation

When data is present, set PADERBORN_AVAILABLE = True in config.py.
"""

from __future__ import annotations

import logging

from src.config import PADERBORN_RAW_DIR, PROCESSED_ROOT, PADERBORN_AVAILABLE
from src.datasets.base_loader import BaseDatasetLoader

logger = logging.getLogger(__name__)


class PaderbornLoader(BaseDatasetLoader):
    """Optional loader for Paderborn University Bearing Dataset.

    Will raise DatasetUnavailableError if PADERBORN_AVAILABLE = False.
    """

    name        = "paderborn"
    description = (
        "Paderborn Bearing Dataset — real vibration data with labelled bearing faults. "
        "Optional: requires manual download (~32 GB)."
    )

    class DatasetUnavailableError(RuntimeError):
        pass

    def __init__(
        self,
        raw_dir:       str = PADERBORN_RAW_DIR,
        processed_dir: str = PROCESSED_ROOT,
    ):
        super().__init__(raw_dir, processed_dir)

    def _check_available(self):
        if not PADERBORN_AVAILABLE:
            raise PaderbornLoader.DatasetUnavailableError(
                "Paderborn dataset not available. "
                "Set PADERBORN_AVAILABLE=True in config.py and download from "
                "https://mb.uni-paderborn.de/en/kat/main-research/kat-datacenter/"
                "bearing-datacenter/data-sets-and-download\n"
                "Note: Dataset is ~32 GB. Free registration required."
            )

    def download(self) -> None:
        msg = (
            "\n"
            "Paderborn Bearing Dataset requires manual download (~32 GB):\n"
            "  URL : https://mb.uni-paderborn.de/en/kat/main-research/kat-datacenter/"
            "bearing-datacenter/data-sets-and-download\n"
            f"  Dest: {self.raw_dir}\n"
            "After downloading:\n"
            "  1. Extract .mat files into datasets/raw/paderborn/\n"
            "  2. Set PADERBORN_AVAILABLE = True in src/config.py\n"
        )
        logger.info(msg)
        print(msg)

    def load_raw(self):
        self._check_available()
        raise NotImplementedError(
            "Implement after placing .mat files in datasets/raw/paderborn/. "
            "Requires scipy.io.loadmat for MATLAB format files."
        )

    def preprocess(self):
        self._check_available()
        raise NotImplementedError("Implement after load_raw().")

    def extract_features(self):
        self._check_available()
        raise NotImplementedError("Implement after preprocess().")
