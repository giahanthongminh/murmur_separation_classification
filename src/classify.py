# classify.py
# Classify murmur timing (Holosystolic / Early-systolic / Mid-systolic)
# Compares 3 feature sources: original WAV, separated murmur, TSV systole only
# Uses patient-level 80/20 split (repeated 10x) to avoid data leakage

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from pathlib import Path

DATA_ROOT = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"


def load(path):
    """Load feature CSV → return X (MFCC), y (labels), groups (patient IDs)."""
    df = pd.read_csv(path)
    X = df[[col for col in df.columns if col.startswith("mfcc_")]].values
    y = df["label"].values
    groups = df["patient_id"].values
    return X, y, groups


def evaluate(model, X, y, groups, n_splits=10):
    """
    Patient-level 80/20 split repeated n_splits times.
    Splits by patient ID so all recordings from one patient
    stay in the same fold — prevents data leakage.
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    accs, f1s = [], []
    all_y_test, all_y_pred = [], []

    for train_idx, test_idx in gss.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred, average="weighted"))
        all_y_test.extend(y[test_idx])
        all_y_pred.extend(y_pred)

    print(f"  Accuracy:    {np.mean(accs)*100:.2f}%")
    print(f"  Weighted F1: {np.mean(f1s)*100:.2f}%")
    print(classification_report(all_y_test, all_y_pred))
    print(confusion_matrix(all_y_test, all_y_pred))


# Load all three feature sets
X_orig, y_orig, g_orig = load(DATA_ROOT / "features_original.csv")
X_sep,  y_sep,  g_sep  = load(DATA_ROOT / "features.csv")
X_tsv,  y_tsv,  g_tsv  = load(DATA_ROOT / "features_tsv.csv")

models = {
    "SVM": SVC(kernel="rbf", random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"),
}

for name, model in models.items():
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print("\n--- Original WAV ---")
    evaluate(model, X_orig, y_orig, g_orig)
    print("\n--- Separated Murmur ---")
    evaluate(model, X_sep, y_sep, g_sep)
    print("\n--- TSV Systole Only ---")
    evaluate(model, X_tsv, y_tsv, g_tsv)