"""Feature availability registry.

Documents which features are available in each dataset.
This is the canonical source of truth for the feature availability matrix.

Do NOT fabricate features. If a dataset does not contain a signal,
mark it as unavailable and do not create synthetic proxy values.
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Feature availability matrix ────────────────────────────────────────────────
# Keys: feature name in common feature space
# Values: dict of dataset → availability status
#   "yes"      : directly measured
#   "proxy"    : physically related but different sensor / unit
#   "derived"  : computed from available signals
#   "no"       : not available in this dataset
#   "optional" : available in some sub-experiments of this dataset

FEATURE_AVAILABILITY: dict[str, dict[str, str]] = {
    "rpm / rotational_speed": {
        "digital_twin": "yes",
        "skab":         "no",          # SKAB pump rig does not record RPM
        "ai4i":         "yes",         # "Rotational speed [rpm]"
        "nasa_ims":     "no",          # only vibration channels recorded
        "paderborn":    "proxy",       # encoder channel available in some runs
    },
    "current_a": {
        "digital_twin": "yes",
        "skab":         "yes",         # "Current" column (Amperes)
        "ai4i":         "no",
        "nasa_ims":     "no",
        "paderborn":    "no",
    },
    "voltage_v": {
        "digital_twin": "no",
        "skab":         "yes",         # "Voltage" column
        "ai4i":         "no",
        "nasa_ims":     "no",
        "paderborn":    "no",
    },
    "pressure_bar": {
        "digital_twin": "yes",
        "skab":         "proxy",       # "Pressure" — relative, not in bar
        "ai4i":         "no",
        "nasa_ims":     "no",
        "paderborn":    "no",
    },
    "flow_m3h": {
        "digital_twin": "yes",
        "skab":         "proxy",       # "Volume Flow RateRMS" — volumetric flow proxy
        "ai4i":         "no",
        "nasa_ims":     "no",
        "paderborn":    "no",
    },
    "vibration_rms": {
        "digital_twin": "yes",         # "vibration_mms" (mm/s)
        "skab":         "yes",         # "Accelerometer1RMS" (m/s²)
        "ai4i":         "no",
        "nasa_ims":     "yes",         # raw vibration at 20 kHz → RMS computed
        "paderborn":    "yes",         # raw vibration → RMS computed
    },
    "temperature_c": {
        "digital_twin": "yes",         # "temp_c" — bearing temperature
        "skab":         "yes",         # "Temperature" and "Thermocouple"
        "ai4i":         "proxy",       # "Air/Process temperature [K]" → converted to C
        "nasa_ims":     "no",
        "paderborn":    "no",
    },
    "torque_nm": {
        "digital_twin": "no",
        "skab":         "no",
        "ai4i":         "yes",         # "Torque [Nm]"
        "nasa_ims":     "no",
        "paderborn":    "no",
    },
    "tool_wear_min": {
        "digital_twin": "no",
        "skab":         "no",
        "ai4i":         "yes",         # "Tool wear [min]"
        "nasa_ims":     "no",
        "paderborn":    "no",
    },
    "anomaly_label": {
        "digital_twin": "yes",         # truth["fault"] != "healthy"
        "skab":         "yes",         # "anomaly" binary column
        "ai4i":         "yes",         # "Machine failure" binary column
        "nasa_ims":     "no",          # no binary label; end-of-life inferred from run end
        "paderborn":    "yes",         # labelled fault conditions
    },
    "fault_class": {
        "digital_twin": "yes",         # 5 simulated fault classes
        "skab":         "no",          # only binary anomaly label; no fault type
        "ai4i":         "proxy",       # TWF/HDF/PWF/OSF/RNF — synthetic proxy labels
        "nasa_ims":     "no",          # no multi-class fault labels
        "paderborn":    "yes",         # labelled inner/outer race, ball faults
    },
    "rul_ground_truth": {
        "digital_twin": "yes",         # simulated severity → RUL extrapolation
        "skab":         "no",          # anomaly detection only; no run-to-failure
        "ai4i":         "no",          # independent cycles; no temporal RUL
        "nasa_ims":     "proxy",       # surrogate RUL: file_index distance to run end
        "paderborn":    "no",          # fault conditions but no continuous RUL
    },
    "vibration_kurtosis": {
        "digital_twin": "derived",     # computable from simulated vib signal
        "skab":         "derived",     # computable from Accelerometer1RMS window
        "ai4i":         "no",
        "nasa_ims":     "derived",     # computed from 20 kHz raw signal window
        "paderborn":    "derived",
    },
    "power_proxy": {
        "digital_twin": "derived",     # current × voltage (not direct)
        "skab":         "derived",     # Current × Voltage columns
        "ai4i":         "derived",     # RPM × Torque
        "nasa_ims":     "no",
        "paderborn":    "no",
    },
}


# ── Dataset metadata ───────────────────────────────────────────────────────────

DATASET_METADATA: dict[str, dict] = {
    "skab": {
        "full_name":       "Skoltech Anomaly Benchmark",
        "type":            "real_industrial",
        "asset_type":      "water pump test rig",
        "sampling_rate_hz": 1,
        "primary_use":     "anomaly detection",
        "label_type":      "binary (anomaly/normal) + changepoint",
        "size_approx":     "~5 MB",
        "availability":    "public (no registration)",
        "source_url":      "https://github.com/waico/SKAB",
        "window_size":     30,
        "note":            "No fault type labels. Only binary anomaly flag.",
    },
    "ai4i": {
        "full_name":       "AI4I 2020 Predictive Maintenance Dataset",
        "type":            "synthetic",
        "asset_type":      "CNC machining process (synthetic)",
        "sampling_rate_hz": "N/A (per-cycle aggregated records)",
        "primary_use":     "fault classification benchmark",
        "label_type":      "multi-class (TWF/HDF/PWF/OSF/RNF/healthy)",
        "size_approx":     "~1 MB",
        "availability":    "public (no registration)",
        "source_url":      "https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset",
        "window_size":     1,
        "note":            "SYNTHETIC data. Not real sensor measurements. Use as benchmark only.",
    },
    "nasa_ims": {
        "full_name":       "NASA IMS Bearing Dataset",
        "type":            "real_industrial",
        "asset_type":      "rolling element bearings (run-to-failure test rig)",
        "sampling_rate_hz": 20000,
        "primary_use":     "degradation analysis, surrogate RUL validation",
        "label_type":      "none (run-to-failure; end-of-life inferred)",
        "size_approx":     "~6.7 GB",
        "availability":    "free with registration",
        "source_url":      "https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository",
        "window_size":     2048,
        "optional":        True,
        "rul_note":        "RUL is a surrogate approximation (file_index distance to run end). Not certified ground truth.",
    },
    "paderborn": {
        "full_name":       "Paderborn University Bearing Dataset (KAt-DataCenter)",
        "type":            "real_industrial",
        "asset_type":      "rolling element bearings (controlled test rig)",
        "sampling_rate_hz": 64000,
        "primary_use":     "bearing fault classification validation",
        "label_type":      "multi-class (healthy / inner race / outer race / ball)",
        "size_approx":     "~32 GB",
        "availability":    "free with registration",
        "source_url":      "https://mb.uni-paderborn.de/en/kat/main-research/kat-datacenter/bearing-datacenter/data-sets-and-download",
        "window_size":     "dataset-specific (see paper)",
        "optional":        True,
    },
    "digital_twin": {
        "full_name":       "PumpTwin Digital Twin Simulation",
        "type":            "synthetic_simulation",
        "asset_type":      "motor-pump (simulated)",
        "sampling_rate_hz": 1,
        "primary_use":     "live MQTT telemetry, real-time inference, Grafana demo",
        "label_type":      "full ground truth (fault, severity, health, load)",
        "size_approx":     "unlimited (live generated)",
        "availability":    "internal (this project)",
        "window_size":     30,
        "note":            "Physics-inspired simulation. NOT real sensor data.",
    },
}


def save_metadata(output_dir: str = "datasets/metadata") -> None:
    """Save feature availability matrix and dataset metadata to JSON files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Feature availability matrix
    avail_path = out / "feature_availability_matrix.json"
    with open(avail_path, "w") as f:
        json.dump(FEATURE_AVAILABILITY, f, indent=2)
    print(f"Saved -> {avail_path}")

    # Dataset metadata
    meta_path = out / "dataset_schema.json"
    with open(meta_path, "w") as f:
        json.dump(DATASET_METADATA, f, indent=2)
    print(f"Saved -> {meta_path}")

    # Feature mapping (DT signal <-> real dataset columns)
    mapping = {
        "digital_twin_to_real": {
            "rpm":           {"skab": None, "ai4i": "Rotational speed [rpm]",
                              "nasa_ims": None, "paderborn": "encoder_ch (some runs)"},
            "current_a":     {"skab": "Current", "ai4i": None,
                              "nasa_ims": None, "paderborn": None},
            "flow_m3h":      {"skab": "RateRMS (proxy)", "ai4i": None,
                              "nasa_ims": None, "paderborn": None},
            "pressure_bar":  {"skab": "Pressure (relative)", "ai4i": None,
                              "nasa_ims": None, "paderborn": None},
            "vibration_mms": {"skab": "Accelerometer1RMS (m/s²)", "ai4i": None,
                              "nasa_ims": "ch1..ch4 (g)", "paderborn": "vibration (g)"},
            "temp_c":        {"skab": "Temperature + Thermocouple",
                              "ai4i": "Air/Process temperature [K] (converted)",
                              "nasa_ims": None, "paderborn": None},
        },
        "notes": {
            "units_differ":   "SKAB accelerometer is m/s², DT vibration is mm/s — both are relative proxies for vibration severity.",
            "rpm_skab":       "SKAB does NOT record RPM. rpm feature is unavailable for SKAB experiments.",
            "ai4i_synthetic": "AI4I temperatures are synthetic. Do not use for real sensor calibration.",
        }
    }
    fmap_path = out / "feature_mapping.json"
    with open(fmap_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved -> {fmap_path}")

    return str(out)


if __name__ == "__main__":
    save_metadata()
