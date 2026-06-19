# classify_v2.py
# SVM + Random Forest on full feature set (MFCC + PSD + Amplitude + Energy + Spectral).
# Compares Original WAV vs Separated Murmur on CirCor systole segments.
#
# Fixes classify.py bug: loads ALL feature columns, not just mfcc_* columns.
# Uses patient-level GroupShuffleSplit (no data leakage) + SMOTE + GridSearchCV.
#
# Usage:
#   python classify_v2.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[INFO] imbalanced-learn not installed — running without SMOTE")

DATA_ROOT = Path("/Users/danggiahan/physionet.org/files/circor-heart-sound/1.0.3")

N_SPLITS   = 10
TEST_SIZE  = 0.2
RANDOM_STATE = 42

SVM_GRID = {"C": [1, 10, 100], "gamma": ["scale", "auto"]}
RF_GRID  = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}


def load(path):
    df = pd.read_csv(path).dropna()
    meta_cols = {"patient_id", "location", "label"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].values.astype(float)
    y = df["label"].values
    groups = df["patient_id"].values

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    return X, y, groups, feat_cols


def evaluate(model_name, X, y, groups, tune_hyperparams=True):
    gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)

    accs, f1s = [], []
    all_y_test, all_y_pred = [], []
    best_params = {}
    tuned = False

    for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        if HAS_SMOTE:
            try:
                counts = np.bincount(
                    np.unique(y_train, return_inverse=True)[1])
                k = min(5, int(np.min(counts)) - 1)
                if k >= 1:
                    X_train, y_train = SMOTE(
                        random_state=RANDOM_STATE, k_neighbors=k
                    ).fit_resample(X_train, y_train)
            except Exception:
                pass

        if tune_hyperparams and not tuned:
            if model_name == "SVM":
                gs = GridSearchCV(
                    SVC(kernel="rbf", class_weight="balanced"),
                    SVM_GRID, cv=3, scoring="f1_weighted", n_jobs=-1
                )
            else:
                gs = GridSearchCV(
                    RandomForestClassifier(class_weight="balanced",
                                           random_state=RANDOM_STATE),
                    RF_GRID, cv=3, scoring="f1_weighted", n_jobs=-1
                )
            gs.fit(X_train, y_train)
            best_params = gs.best_params_
            print(f"    Best {model_name} params (fold 0): {best_params}")
            tuned = True

        if model_name == "SVM":
            model = SVC(kernel="rbf", class_weight="balanced",
                        random_state=RANDOM_STATE, **best_params)
        else:
            model = RandomForestClassifier(class_weight="balanced",
                                           random_state=RANDOM_STATE,
                                           **best_params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    print(f"  Accuracy:    {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  Weighted F1: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    print(classification_report(all_y_test, all_y_pred))
    print(confusion_matrix(all_y_test, all_y_pred))

    return {"accuracy": np.mean(accs)*100, "f1": np.mean(f1s)*100,
            "acc_std": np.std(accs)*100,   "f1_std": np.std(f1s)*100}


def print_summary(results):
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print(f"{'':30s} {'Original WAV':>15s} {'Separated Murmur':>18s}")
    for model_name in ["SVM", "Random Forest"]:
        print(f"\n{model_name}")
        for metric in ["accuracy", "f1"]:
            o = results["Original WAV"][model_name][metric]
            s = results["Separated Murmur"][model_name][metric]
            delta = s - o
            sign = "+" if delta >= 0 else ""
            print(f"  {metric.capitalize():18s}  {o:>12.2f}%  {s:>14.2f}%  "
                  f"(Δ {sign}{delta:.2f}%)")


def main():
    orig_csv = DATA_ROOT / "features_v2_original.csv"
    sep_csv  = DATA_ROOT / "features_v2_separated.csv"
    # Note: labels come from training_data.csv (not labels.csv) in dataset v1.0.3

    missing = [p for p in [orig_csv, sep_csv] if not p.exists()]
    if missing:
        print("Missing feature CSVs — run extract_features_v2.py first:")
        for p in missing:
            print(f"  {p}")
        return

    results = {}

    for condition, csv_path in [("Original WAV", orig_csv),
                                  ("Separated Murmur", sep_csv)]:
        X, y, groups, feat_cols = load(csv_path)
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"  Samples: {len(X)}, Features: {len(feat_cols)}")
        print(f"  Label distribution: {pd.Series(y).value_counts().to_dict()}")

        results[condition] = {}
        for model_name in ["SVM", "Random Forest"]:
            print(f"\n  -- {model_name} --")
            results[condition][model_name] = evaluate(model_name, X, y, groups)

    print_summary(results)


if __name__ == "__main__":
    main()
