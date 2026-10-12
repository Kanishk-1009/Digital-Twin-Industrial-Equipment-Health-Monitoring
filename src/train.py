"""Train and persist the digital twin ML models.

Two models are trained:
  1. Isolation Forest — anomaly detector on healthy windows only,
     with calibrated score anchors for the [0, 100] health index.
  2. Random Forest — multi-class fault classifier on all 5 classes.

Episode-based test split (episode % 5 == 0) to prevent data leakage.

Run with:  python -m src.train
Outputs:   data/train.csv, models/twin_models.joblib
"""

import joblib
import numpy as np
from pathlib import Path

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from src.config import MODEL_PATH
from src.features import FEATURE_NAMES
from src.generate_dataset import build_dataset

Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# build dataset
print("Building dataset (this may take ~30 s)...")
df = build_dataset()
df.to_csv("data/train.csv", index=False)
print(f"Dataset: {df.shape[0]} windows, label distribution:")
print(df["label"].value_counts())

# episode-based split: hold out episodes where ID % 5 == 0
test_mask = df["episode"] % 5 == 0
tr, te = df[~test_mask], df[test_mask]
print(f"\nTrain size: {len(tr)} | Test size: {len(te)}")

# --- anomaly detector ---
healthy = tr[tr["episode_type"] == "healthy"]
scaler  = StandardScaler().fit(healthy[FEATURE_NAMES])

iso = IsolationForest(n_estimators=200, random_state=0, contamination="auto")
iso.fit(scaler.transform(healthy[FEATURE_NAMES]))

score = lambda d: iso.decision_function(scaler.transform(d[FEATURE_NAMES]))

s_good = float(np.median(score(healthy)))
s_bad  = float(np.median(score(tr[tr["severity"] >= 0.6])))
print(f"\nCalibration anchors:  s_good={s_good:.3f}   s_bad={s_bad:.3f}")

# evaluate health index on held-out set
health_te = np.clip((score(te) - s_bad) / (s_good - s_bad), 0, 1) * 100
true_health_te = 100 * (1 - te["severity"].values)
corr = round(float(np.corrcoef(health_te, true_health_te)[0, 1]), 3)
print(f"Health index correlation with ground truth on test set: {corr}")

# --- fault classifier ---
clf = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1)
clf.fit(tr[FEATURE_NAMES], tr["label"])

pred = clf.predict(te[FEATURE_NAMES])
print("\nClassification report:")
print(classification_report(te["label"], pred))
print("Confusion matrix:")
print(confusion_matrix(te["label"], pred, labels=clf.classes_))
print("Classes:", list(clf.classes_))

# save
joblib.dump(
    dict(scaler=scaler, iso=iso, clf=clf, s_good=s_good, s_bad=s_bad),
    MODEL_PATH,
)
print(f"\nModels saved -> {MODEL_PATH}")
