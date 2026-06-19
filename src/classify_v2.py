# classify_v2.py
# SVM + RF + GradientBoosting + XGBoost + MLP on full feature set.
# Compares Original WAV vs Separated Murmur on CirCor systole segments.
#
# Key improvements over v1:
# - Drop Late-systolic (only 4 samples — scientifically unusable)
# - Feature selection: keep top-N features by RF importance → reduce 242 → 80
# - Add GradientBoosting, XGBoost, MLP classifiers
# - SMOTE per fold, GridSearchCV on fold 0 only (speed)
#
# Usage:
#   python classify_v2.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[INFO] imbalanced-learn not installed — running without SMOTE")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] xgboost not installed — skipping XGBoost")

DATA_ROOT    = Path("/Users/danggiahan/physionet.org/files/circor-heart-sound/1.0.3")
N_SPLITS     = 10
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_FEATURES   = 80   # keep top-N features after selection

# Drop this class — 4 samples total, impossible to learn
DROP_CLASSES = {"Late-systolic"}


def load(path):
    df = pd.read_csv(path).dropna()

    # Drop Late-systolic
    df = df[~df["label"].isin(DROP_CLASSES)]

    meta_cols = {"patient_id", "location", "label"}
    feat_cols  = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].values.astype(float)
    y = df["label"].values
    groups = df["patient_id"].values

    X = SimpleImputer(strategy="mean").fit_transform(X)
    return X, y, groups, feat_cols


def select_features(X_train, y_train, n=N_FEATURES):
    """Select top-n features by Random Forest importance."""
    selector = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    selector.fit(X_train, y_train)
    importances = selector.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n]
    return top_idx


def apply_smote(X_train, y_train):
    if not HAS_SMOTE:
        return X_train, y_train
    try:
        counts = np.bincount(np.unique(y_train, return_inverse=True)[1])
        k = min(5, int(np.min(counts)) - 1)
        if k >= 1:
            X_train, y_train = SMOTE(
                random_state=RANDOM_STATE, k_neighbors=k
            ).fit_resample(X_train, y_train)
    except Exception:
        pass
    return X_train, y_train


def make_model(name, best_params):
    if name == "SVM":
        return SVC(kernel="rbf", class_weight="balanced",
                   random_state=RANDOM_STATE, probability=True,
                   **best_params)
    if name == "Random Forest":
        return RandomForestClassifier(class_weight="balanced",
                                      random_state=RANDOM_STATE,
                                      n_jobs=-1, **best_params)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE,
                                          **best_params)
    if name == "XGBoost":
        return XGBClassifier(use_label_encoder=False,
                             eval_metric="mlogloss",
                             random_state=RANDOM_STATE,
                             n_jobs=-1, **best_params)
    if name == "MLP":
        return MLPClassifier(random_state=RANDOM_STATE,
                             max_iter=500, **best_params)


GRIDS = {
    "SVM": (
        SVC(kernel="rbf", class_weight="balanced", probability=True),
        {"C": [1, 10, 100], "gamma": ["scale", "auto"]}
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=-1),
        {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [3, 5],
         "learning_rate": [0.05, 0.1]}
    ),
    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                      random_state=RANDOM_STATE, n_jobs=-1)
        if HAS_XGB else None,
        {"n_estimators": [100, 200], "max_depth": [3, 6],
         "learning_rate": [0.05, 0.1]}
    ),
    "MLP": (
        MLPClassifier(random_state=RANDOM_STATE, max_iter=500),
        {"hidden_layer_sizes": [(128, 64), (256, 128, 64)],
         "alpha": [0.0001, 0.001],
         "learning_rate_init": [0.001, 0.0005]}
    ),
}


def evaluate(model_name, X, y, groups):
    if model_name == "XGBoost" and not HAS_XGB:
        return None

    # Encode labels to integers for XGBoost
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)

    accs, f1s = [], []
    all_y_test, all_y_pred = [], []
    best_params = {}
    tuned = False

    for fold, (train_idx, test_idx) in enumerate(gss.split(X, y_enc, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        # Feature selection on training fold
        top_idx = select_features(X_train, y_train, n=N_FEATURES)
        X_train = X_train[:, top_idx]
        X_test  = X_test[:, top_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        X_train, y_train = apply_smote(X_train, y_train)

        # Tune on fold 0 only
        if not tuned:
            base_model, param_grid = GRIDS[model_name]
            if base_model is None:
                return None
            gs = GridSearchCV(base_model, param_grid,
                              cv=3, scoring="f1_weighted", n_jobs=-1)
            gs.fit(X_train, y_train)
            best_params = gs.best_params_
            print(f"    Best params (fold 0): {best_params}")
            tuned = True

        model = make_model(model_name, best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        all_y_test.extend(le.inverse_transform(y_test))
        all_y_pred.extend(le.inverse_transform(y_pred))

    print(f"  Accuracy:    {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  Weighted F1: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    print(classification_report(all_y_test, all_y_pred))
    print(confusion_matrix(all_y_test, all_y_pred,
                           labels=sorted(set(all_y_test))))

    return {"accuracy": np.mean(accs)*100, "f1": np.mean(f1s)*100,
            "acc_std": np.std(accs)*100,   "f1_std": np.std(f1s)*100}


def print_summary(results):
    models = [m for m in ["SVM", "Random Forest", "Gradient Boosting",
                           "XGBoost", "MLP"]
              if any(results[c].get(m) is not None for c in results)]

    print("\n" + "=" * 75)
    print("SUMMARY TABLE  (Late-systolic dropped, top-80 features, SMOTE)")
    print(f"{'':25s} {'Original WAV':>15s} {'Separated Murmur':>18s} {'Δ':>8s}")
    print("-" * 75)

    for model_name in models:
        for metric in ["accuracy", "f1"]:
            o_res = results["Original WAV"].get(model_name)
            s_res = results["Separated Murmur"].get(model_name)
            if o_res is None or s_res is None:
                continue
            o, s = o_res[metric], s_res[metric]
            delta = s - o
            sign = "+" if delta >= 0 else ""
            label = f"{model_name} {metric.capitalize()}"
            print(f"  {label:23s}  {o:>12.2f}%  {s:>14.2f}%  "
                  f"({sign}{delta:.2f}%)")
        print()


def main():
    orig_csv = DATA_ROOT / "features_v2_original.csv"
    sep_csv  = DATA_ROOT / "features_v2_separated.csv"

    missing = [p for p in [orig_csv, sep_csv] if not p.exists()]
    if missing:
        print("Missing feature CSVs — run extract_features_v2.py first:")
        for p in missing:
            print(f"  {p}")
        return

    model_names = ["SVM", "Random Forest", "Gradient Boosting", "MLP"]
    if HAS_XGB:
        model_names.append("XGBoost")

    results = {}
    for condition, csv_path in [("Original WAV", orig_csv),
                                  ("Separated Murmur", sep_csv)]:
        X, y, groups, feat_cols = load(csv_path)
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"  Samples: {len(X)}, Features: {len(feat_cols)}"
              f" → selecting top {N_FEATURES}")
        print(f"  Label distribution: {pd.Series(y).value_counts().to_dict()}")

        results[condition] = {}
        for model_name in model_names:
            print(f"\n  -- {model_name} --")
            res = evaluate(model_name, X, y, groups)
            if res is not None:
                results[condition][model_name] = res

    print_summary(results)


if __name__ == "__main__":
    main()
