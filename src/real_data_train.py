"""Real-world dataset training pipeline.

Runs dataset-specific experiments using real industrial data,
then calibrates the main twin_models.joblib using insights derived
from those datasets.

Usage
-----
    python -m src.real_data_train [--experiment {skab,ai4i,all}] [--force-reprocess]

Experiments
-----------
A  SKAB — anomaly detection (Isolation Forest trained on real normal data)
D  AI4I — fault classification benchmark (Random Forest on synthetic benchmark)

For each experiment this pipeline:
  1. Loads and preprocesses the dataset
  2. Characterises it (computes actual distributions/correlations)
  3. Performs chronological split (NO random shuffle of time windows)
  4. Trains / evaluates models on real data
  5. Reports real metrics (nothing fabricated)
  6. Saves per-experiment results to results/<dataset>/

After all experiments:
  7. Uses SKAB-derived healthy distribution stats to recalibrate
     the anomaly detector calibration anchors (s_good / s_bad) in
     twin_models.joblib — computed from actual data.
  8. Saves updated twin_models.joblib (same keys as before).

IMPORTANT
---------
analytics.py is NOT modified. It continues to load twin_models.joblib
with keys: scaler, iso, clf, s_good, s_bad
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_PATH, RESULTS_ROOT, SKAB_FEATURES, AI4I_FEATURES
from src.datasets.skab_loader import SKABLoader, SKAB_FEATURE_COLS, SKAB_LABEL_COL
from src.datasets.ai4i_loader import AI4ILoader, AI4I_FEATURE_COLS, FAULT_LABEL_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Result directory helpers ────────────────────────────────────────────────────

def results_dir(name: str) -> Path:
    d = Path(RESULTS_ROOT) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info("Saved → %s", path)


# ── Plotting helpers ────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, labels: list, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Plot saved → %s", path)


def plot_anomaly_timeline(df: pd.DataFrame, score_col: str, label_col: str,
                          title: str, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax1.plot(df[score_col].values, color="steelblue", lw=0.8, label="Anomaly score")
    ax1.set_ylabel("Score (lower = more anomalous)")
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    ax2.fill_between(range(len(df)), df[label_col].values,
                     color="tomato", alpha=0.6, label="True anomaly")
    ax2.set_ylabel("Anomaly label (0/1)"); ax2.set_xlabel("Sample index")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Plot saved → %s", path)


def plot_feature_importance(importances: np.ndarray, names: list,
                            title: str, path: Path, top_n: int = 20) -> None:
    idx   = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([names[i] for i in idx[::-1]], importances[idx[::-1]], color="teal")
    ax.set_xlabel("Importance"); ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sensor_distributions(df: pd.DataFrame, feature_cols: list,
                               label_col: str, path: Path) -> None:
    cols = [c for c in feature_cols if c in df.columns][:6]
    if not cols:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        normal = df[df[label_col] == 0][col].dropna()
        faulty = df[df[label_col] == 1][col].dropna()
        axes[i].hist(normal.values, bins=50, alpha=0.6, color="steelblue", label="Normal", density=True)
        axes[i].hist(faulty.values, bins=50, alpha=0.6, color="tomato",    label="Faulty", density=True)
        axes[i].set_title(col); axes[i].legend(fontsize=8)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Sensor Distributions: Normal vs Anomaly")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── Experiment A — SKAB Anomaly Detection ─────────────────────────────────────

def experiment_skab(force_reprocess: bool = False) -> dict:
    """
    Train and evaluate Isolation Forest on real SKAB pump data.

    Training set: all normal (anomaly=0) windows from train files.
    Test set:     held-out file(s) from test portion (chronological split).
    No random shuffling of windows from the same source file.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT A — SKAB Anomaly Detection")
    logger.info("=" * 60)

    rd = results_dir("skab")
    t0 = time.time()

    # ── Load / process ────────────────────────────────────────────────────
    loader = SKABLoader()
    if not Path(SKAB_FEATURES).exists() or force_reprocess:
        loader.download()
        loader._raw_df   = loader.load_raw()
        loader._clean_df = loader.preprocess()
        feat_df          = loader.extract_features()
        feat_df.to_csv(SKAB_FEATURES, index=False)
    else:
        feat_df = pd.read_csv(SKAB_FEATURES)
        loader._raw_df   = loader.load_raw()
        loader._clean_df = loader.preprocess()

    logger.info("SKAB features loaded: shape=%s", feat_df.shape)

    # ── Characterise (compute actual stats from real data) ────────────────
    stats = loader.characterise()
    save_json(stats, rd / "dataset_characterisation.json")

    # Compute sensor distributions plot from actual data
    raw_cols = [c for c in [
        "current_a_mean", "pressure_bar_mean", "vibration_rms_mean",
        "temperature_c_mean", "flow_proxy_mean", "voltage_v_mean",
    ] if c in feat_df.columns]
    plot_sensor_distributions(feat_df, raw_cols, SKAB_LABEL_COL,
                              rd / "sensor_distributions.png")

    # Correlation matrix from actual data
    num_cols = [c for c in feat_df.columns if feat_df[c].dtype in (float, "float64") and c not in ["anomaly", "changepoint", "window_end"]][:15]
    if num_cols:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = feat_df[num_cols].corr()
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, annot=False)
        ax.set_title("SKAB Feature Correlation Matrix")
        plt.tight_layout()
        fig.savefig(rd / "correlation_matrix.png", dpi=120)
        plt.close(fig)
        logger.info("Plot saved → %s", rd / "correlation_matrix.png")

    # ── Chronological split by source_file ───────────────────────────────
    # Sort files chronologically (alphabetical ≈ temporal in SKAB)
    files = sorted(feat_df["source_file"].unique())
    n_test_files = max(1, len(files) // 5)
    train_files  = files[:-n_test_files]
    test_files   = files[-n_test_files:]
    logger.info("SKAB split — train files: %d  test files: %d", len(train_files), len(test_files))

    tr = feat_df[feat_df["source_file"].isin(train_files)].copy()
    te = feat_df[feat_df["source_file"].isin(test_files)].copy()

    feat_cols = [c for c in feat_df.columns
                 if c not in {"anomaly", "changepoint", "source_file", "window_end"}
                 and feat_df[c].dtype in (float, "float64")]

    # ── Train Isolation Forest on REAL normal (anomaly=0) data ───────────
    train_normal = tr[tr[SKAB_LABEL_COL] == 0][feat_cols].copy()
    # Sanitize: replace inf/−inf → NaN, then drop remaining NaN rows
    train_normal = train_normal.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info("Training IF on %d real normal SKAB windows (after inf/NaN removal)...", len(train_normal))

    scaler_skab = StandardScaler().fit(train_normal)
    X_train_n   = scaler_skab.transform(train_normal)

    iso_skab = IsolationForest(n_estimators=200, random_state=0, contamination="auto")
    iso_skab.fit(X_train_n)
    logger.info("Isolation Forest trained on real SKAB normal data.")

    # ── Compute calibration anchors FROM ACTUAL DATA ──────────────────────
    # Sanitize test features before scoring
    te_feats = te[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(te[feat_cols].median())
    scores = iso_skab.decision_function(scaler_skab.transform(te_feats))
    labels = te[SKAB_LABEL_COL].values

    normal_scores  = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    s_good_skab = float(np.median(normal_scores))  if len(normal_scores)  > 0 else 0.0
    s_bad_skab  = float(np.median(anomaly_scores)) if len(anomaly_scores) > 0 else -0.1

    logger.info("SKAB calibration — s_good=%.4f  s_bad=%.4f", s_good_skab, s_bad_skab)

    # ── Evaluate ──────────────────────────────────────────────────────────
    # Convert IF score to binary prediction: score < threshold → anomaly
    threshold = s_good_skab  # label as anomaly if score below normal median
    preds = (scores < threshold).astype(int)

    # Compute health index from real SKAB anomaly scores
    health_idx = np.clip((scores - s_bad_skab) / (s_good_skab - s_bad_skab + 1e-9), 0, 1) * 100

    # Metrics — only reported from actual data
    precision = float(precision_score(labels, preds, zero_division=0))
    recall    = float(recall_score(labels, preds, zero_division=0))
    f1        = float(f1_score(labels, preds, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(labels, -scores))  # lower score = more anomalous
        pr_auc  = float(average_precision_score(labels, -scores))
    except Exception:
        roc_auc = pr_auc = None

    # False positive / negative rates
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "experiment":       "A — SKAB Anomaly Detection",
        "dataset":          "SKAB (real industrial pump data)",
        "dataset_type":     "real_industrial",
        "n_train_normal":   len(train_normal),
        "n_test":           len(te),
        "n_test_anomaly":   int(labels.sum()),
        "n_test_normal":    int((labels == 0).sum()),
        "anomaly_rate_test": float(labels.mean()),
        "train_files":      train_files,
        "test_files":       test_files,
        "n_features":       len(feat_cols),
        "model":            "Isolation Forest (n_estimators=200)",
        "scaler":           "StandardScaler",
        "calibration": {
            "s_good": s_good_skab,
            "s_bad":  s_bad_skab,
            "method": "median of IF scores on real normal/anomaly test windows",
        },
        "threshold":        threshold,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "roc_auc":          roc_auc,
        "pr_auc":           pr_auc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "training_time_s":  round(time.time() - t0, 2),
        "health_index_normal_mean":  float(health_idx[labels == 0].mean()),
        "health_index_anomaly_mean": float(health_idx[labels == 1].mean()),
        "note": (
            "All metrics computed from actual SKAB data. "
            "Calibration anchors derived from real sensor measurements."
        ),
    }
    save_json(metrics, rd / "metrics.json")

    # Plots from actual data
    score_df = pd.DataFrame({"score": scores, SKAB_LABEL_COL: labels})
    plot_anomaly_timeline(score_df, "score", SKAB_LABEL_COL,
                          "SKAB: Isolation Forest Anomaly Score vs True Labels",
                          rd / "anomaly_score_timeline.png")

    health_df = pd.DataFrame({"health_index": health_idx, SKAB_LABEL_COL: labels})
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(health_idx, color="steelblue", lw=0.8, label="Health Index")
    ax.fill_between(range(len(labels)), 0, 100 * labels,
                    color="tomato", alpha=0.3, label="True anomaly")
    ax.axhline(50, color="orange", ls="--", lw=1, label="50% threshold")
    ax.set_ylabel("Health Index (0-100)"); ax.set_xlabel("Sample index")
    ax.set_title("SKAB: Health Index vs True Anomaly Labels")
    ax.legend(); plt.tight_layout()
    fig.savefig(rd / "health_index_validation.png", dpi=120)
    plt.close(fig)
    logger.info("Plot saved → %s", rd / "health_index_validation.png")

    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, ["Normal", "Anomaly"],
                          "SKAB: Anomaly Detection Confusion Matrix",
                          rd / "confusion_matrix.png")

    logger.info("Experiment A complete — F1=%.3f  ROC-AUC=%s  PR-AUC=%s",
                f1, roc_auc, pr_auc)

    return {
        "metrics":      metrics,
        "s_good_skab":  s_good_skab,
        "s_bad_skab":   s_bad_skab,
        "scaler_skab":  scaler_skab,
        "iso_skab":     iso_skab,
        "feat_cols":    feat_cols,
    }


# ── Experiment D — AI4I Fault Classification ───────────────────────────────────

def experiment_ai4i(force_reprocess: bool = False) -> dict:
    """
    Train and evaluate Random Forest classifier on AI4I dataset.

    IMPORTANT: AI4I is synthetic data. Results are reported as a
    classification benchmark, NOT as evidence of real sensor performance.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT D — AI4I Fault Classification (SYNTHETIC BENCHMARK)")
    logger.info("=" * 60)

    rd = results_dir("ai4i")
    t0 = time.time()

    # ── Load / process ────────────────────────────────────────────────────
    loader = AI4ILoader()
    if not Path(AI4I_FEATURES).exists() or force_reprocess:
        loader.download()
        loader._raw_df   = loader.load_raw()
        loader._clean_df = loader.preprocess()
        feat_df          = loader.extract_features()
        feat_df.to_csv(AI4I_FEATURES, index=False)
    else:
        feat_df = pd.read_csv(AI4I_FEATURES)
        loader._raw_df   = loader.load_raw()
        loader._clean_df = loader.preprocess()

    logger.info("AI4I features loaded: shape=%s", feat_df.shape)

    # Print fault mapping report (explicitly documents proxy nature)
    try:
        print("\n" + AI4ILoader.fault_mapping_report() + "\n")
    except UnicodeEncodeError:
        pass  # suppress on Windows terminals with restricted encoding

    # Characterise
    stats = loader.characterise()
    save_json(stats, rd / "dataset_characterisation.json")

    # ── Chronological split (no shuffle — preserve operation order) ────────
    tr = feat_df[feat_df["split"] == "train"].copy()
    va = feat_df[feat_df["split"] == "val"].copy()
    te = feat_df[feat_df["split"] == "test"].copy()
    logger.info("AI4I split — train=%d  val=%d  test=%d", len(tr), len(va), len(te))

    feat_cols = [c for c in feat_df.columns
                 if c not in {"failure", "fault_label", "split"}]
    feat_cols = [c for c in feat_cols if feat_df[c].dtype in (float, int, "float64", "int64")]

    # ── Binary classification (healthy / failed) ──────────────────────────
    clf_binary = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1)
    clf_binary.fit(tr[feat_cols].fillna(0), tr["failure"])
    bin_preds = clf_binary.predict(te[feat_cols].fillna(0))
    bin_proba = clf_binary.predict_proba(te[feat_cols].fillna(0))[:, 1]

    bin_f1  = float(f1_score(te["failure"], bin_preds, zero_division=0))
    bin_auc = float(roc_auc_score(te["failure"], bin_proba)) if te["failure"].nunique() > 1 else None
    bin_pr  = float(average_precision_score(te["failure"], bin_proba)) if te["failure"].nunique() > 1 else None
    bin_cm  = confusion_matrix(te["failure"], bin_preds)

    # ── Multi-class fault classification ──────────────────────────────────
    clf_multi = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1)
    clf_multi.fit(tr[feat_cols].fillna(0), tr["fault_label"])
    multi_preds = clf_multi.predict(te[feat_cols].fillna(0))

    multi_report = classification_report(te["fault_label"], multi_preds, output_dict=True, zero_division=0)
    multi_cm     = confusion_matrix(te["fault_label"], multi_preds, labels=clf_multi.classes_)

    # ── Feature importance (from actual RF model) ─────────────────────────
    fi_series = pd.Series(clf_multi.feature_importances_, index=feat_cols).sort_values(ascending=False)
    fi_series.to_csv(rd / "feature_importance.csv", header=["importance"])
    plot_feature_importance(
        clf_multi.feature_importances_, feat_cols,
        "AI4I Multi-class RF Feature Importance", rd / "feature_importance.png",
    )

    # Confusion matrices
    plot_confusion_matrix(bin_cm, ["Healthy", "Failed"],
                          "AI4I: Binary Failure Detection", rd / "confusion_matrix_binary.png")
    plot_confusion_matrix(multi_cm, list(clf_multi.classes_),
                          "AI4I: Multi-class Fault Classification", rd / "confusion_matrix_multi.png")

    metrics = {
        "experiment":     "D — AI4I Fault Classification",
        "dataset":        "AI4I 2020 Predictive Maintenance",
        "dataset_type":   "SYNTHETIC — not real sensor measurements",
        "n_train":        len(tr),
        "n_val":          len(va),
        "n_test":         len(te),
        "n_features":     len(feat_cols),
        "feature_names":  feat_cols,
        "failure_rate_train": float(tr["failure"].mean()),
        "failure_rate_test":  float(te["failure"].mean()),
        "binary_classification": {
            "model":  "Random Forest (n_estimators=150)",
            "f1":     bin_f1,
            "roc_auc": bin_auc,
            "pr_auc":  bin_pr,
            "confusion_matrix": bin_cm.tolist(),
        },
        "multi_class_classification": {
            "model":           "Random Forest (n_estimators=150)",
            "classes":         list(clf_multi.classes_),
            "report":          multi_report,
            "confusion_matrix": multi_cm.tolist(),
        },
        "fault_label_mapping_note": (
            "Labels are PROXY mappings from AI4I synthetic fault types to DT categories. "
            "TWF→bearing_wear (Medium), HDF→cooling_failure (Medium), "
            "PWF/OSF/RNF have low-confidence mappings. "
            "Do NOT use these as ground truth for physical fault classes."
        ),
        "top_features": fi_series.head(10).to_dict(),
        "training_time_s": round(time.time() - t0, 2),
        "note": (
            "All metrics computed from actual AI4I data. "
            "AI4I is synthetic — results reflect benchmark performance, "
            "not real sensor measurement accuracy."
        ),
    }
    save_json(metrics, rd / "metrics.json")

    logger.info("Experiment D complete — Binary F1=%.3f  Multi-class macro-F1=%.3f",
                bin_f1, multi_report.get("macro avg", {}).get("f1-score", 0))
    return {"metrics": metrics, "clf_binary": clf_binary, "clf_multi": clf_multi, "feat_cols": feat_cols}


# ── Model calibration using real data ──────────────────────────────────────────

def calibrate_main_model(skab_results: dict) -> None:
    """
    Update twin_models.joblib calibration using SKAB-derived statistics.

    What changes:
      - s_good and s_bad are updated to values computed from real SKAB data
        (instead of synthetic data only)
      - All other keys (scaler, iso, clf) remain unchanged so analytics.py
        continues to work without any modification

    What does NOT change:
      - The Isolation Forest model itself (still trained on synthetic data
        because SKAB does not cover all DT signals: RPM, flow_m3h are missing)
      - The Random Forest classifier
      - The StandardScaler fitted on synthetic data
      - Any MQTT topic, InfluxDB schema, or Grafana dashboard

    Rationale:
      The SKAB calibration anchors provide a real-world reference point for
      where the anomaly score boundary sits on actual industrial pump data.
      The Digital Twin's IF score range is rescaled using these empirically
      derived anchors rather than purely synthetic ones.
    """
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        logger.warning(
            "twin_models.joblib not found at %s. "
            "Run 'python -m src.train' first to create the base model, "
            "then re-run real_data_train to apply calibration.",
            model_path,
        )
        return

    M = joblib.load(model_path)

    # Current synthetic calibration
    old_s_good = M["s_good"]
    old_s_bad  = M["s_bad"]

    # SKAB-derived calibration (computed from actual measurements)
    new_s_good = skab_results["s_good_skab"]
    new_s_bad  = skab_results["s_bad_skab"]

    # Blend: weighted average (70% synthetic, 30% real)
    # Rationale: SKAB covers only a subset of DT signals (no RPM, no flow_m3h).
    # A pure SKAB calibration would be incorrect for the full DT feature space.
    # The blend preserves synthetic calibration for unmatched signals
    # while pulling the boundary toward real industrial reference points.
    alpha      = 0.30  # weight given to real SKAB calibration
    blend_good = (1 - alpha) * old_s_good + alpha * new_s_good
    blend_bad  = (1 - alpha) * old_s_bad  + alpha * new_s_bad

    logger.info(
        "Calibration update:"
        "\n  s_good: %.4f (synthetic) + %.4f (SKAB) → %.4f (blended, α=%.2f)"
        "\n  s_bad:  %.4f (synthetic) + %.4f (SKAB) → %.4f (blended, α=%.2f)",
        old_s_good, new_s_good, blend_good, alpha,
        old_s_bad,  new_s_bad,  blend_bad,  alpha,
    )

    # Update only calibration anchors — all other keys preserved
    M["s_good"] = blend_good
    M["s_bad"]  = blend_bad
    M["calibration_metadata"] = {
        "synthetic_s_good":   old_s_good,
        "synthetic_s_bad":    old_s_bad,
        "skab_s_good":        new_s_good,
        "skab_s_bad":         new_s_bad,
        "blend_alpha_real":   alpha,
        "final_s_good":       blend_good,
        "final_s_bad":        blend_bad,
        "method": (
            "Blended calibration: (1-alpha)*synthetic + alpha*skab. "
            "alpha=0.30 because SKAB covers only a subset of Digital Twin signals "
            "(missing: rpm, flow_m3h). "
            "SKAB signals mapped: Current→current_a, Pressure→pressure_bar, "
            "Accelerometer1RMS→vibration_rms, Temperature→temp_c, RateRMS→flow_proxy."
        ),
    }

    joblib.dump(M, model_path)
    logger.info("Updated twin_models.joblib with real-data-calibrated anchors → %s", model_path)

    # Save calibration record
    rd = results_dir(".")
    save_json(M["calibration_metadata"], Path(RESULTS_ROOT) / "calibration_record.json")


# ── Main entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-world dataset training experiments for Digital Twin"
    )
    parser.add_argument(
        "--experiment", choices=["skab", "ai4i", "all"], default="all",
        help="Which experiment(s) to run (default: all)"
    )
    parser.add_argument(
        "--force-reprocess", action="store_true",
        help="Re-download and reprocess even if cached files exist"
    )
    parser.add_argument(
        "--no-calibrate", action="store_true",
        help="Skip updating twin_models.joblib calibration after experiments"
    )
    args = parser.parse_args()

    Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
    skab_results = None

    if args.experiment in ("skab", "all"):
        skab_results = experiment_skab(force_reprocess=args.force_reprocess)

    if args.experiment in ("ai4i", "all"):
        experiment_ai4i(force_reprocess=args.force_reprocess)

    if not args.no_calibrate and skab_results is not None:
        calibrate_main_model(skab_results)

    logger.info("=" * 60)
    logger.info("All experiments complete. Results saved to: %s/", RESULTS_ROOT)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
