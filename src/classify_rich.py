# classify_rich.py
# SVM + RF with GridSearchCV and SMOTE
# Compares rich features: separated murmur vs original WAV

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from config import FEATURE_OUTPUT_DIR
from src.data_validation import validate_dataset

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("imbalanced-learn not installed — running without SMOTE")

def load(path):
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in ["patient_id", "label"]]
    X = df[feat_cols].values
    y = df["label"].values
    groups = df["patient_id"].values
    # Replace NaN with column mean
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    return X, y, groups


def evaluate(name, X, y, groups, n_splits=10):
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    accs, f1s = [], []
    all_y_test, all_y_pred = [], []

    first_fold = True
    best_svm_params = {"C": 10, "gamma": "scale"}
    best_rf_params  = {"n_estimators": 200, "max_depth": None}

    for train_idx, test_idx in gss.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # SMOTE on training fold only
        if HAS_SMOTE:
            try:
                min_samples = min(np.bincount(
                    np.unique(y_train, return_inverse=True)[1]))
                k = min(5, min_samples - 1)
                if k >= 1:
                    sm = SMOTE(random_state=42, k_neighbors=k)
                    X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception:
                pass

        # GridSearch on first fold only (for speed)
        if first_fold and name == "SVM":
            grid = GridSearchCV(
                SVC(kernel="rbf", class_weight="balanced"),
                {"C": [1, 10, 100], "gamma": ["scale", "auto"]},
                cv=3, scoring="f1_weighted", n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_svm_params = grid.best_params_
            print(f"  Best SVM params: {best_svm_params}")

        if first_fold and name == "Random Forest":
            grid = GridSearchCV(
                RandomForestClassifier(class_weight="balanced", random_state=42),
                {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
                cv=3, scoring="f1_weighted", n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_rf_params = grid.best_params_
            print(f"  Best RF params: {best_rf_params}")

        first_fold = False

        # Train with best params
        if name == "SVM":
            model = SVC(kernel="rbf", class_weight="balanced", **best_svm_params)
        else:
            model = RandomForestClassifier(
                class_weight="balanced", random_state=42, **best_rf_params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    print(f"  Accuracy:    {np.mean(accs)*100:.2f}%")
    print(f"  Weighted F1: {np.mean(f1s)*100:.2f}%")
    print(classification_report(all_y_test, all_y_pred))
    print(confusion_matrix(all_y_test, all_y_pred))


validate_dataset()
X_sep,  y_sep,  g_sep  = load(FEATURE_OUTPUT_DIR / "features_rich.csv")
X_orig, y_orig, g_orig = load(FEATURE_OUTPUT_DIR / "features_rich_original.csv")

for model_name in ["SVM", "Random Forest"]:
    print(f"\n{'='*50}\n{model_name}\n{'='*50}")
    print("\n--- Separated Murmur (Rich Features) ---")
    evaluate(model_name, X_sep, y_sep, g_sep)
    print("\n--- Original WAV (Rich Features) ---")
    evaluate(model_name, X_orig, y_orig, g_orig)
