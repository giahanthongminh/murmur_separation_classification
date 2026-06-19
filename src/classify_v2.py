# classify_v2.py
# 5 models: SVM, Random Forest, Gradient Boosting, XGBoost+LightGBM Ensemble,
#           PyTorch MLP with Focal Loss
# Primary metric: F1 macro (fair for imbalanced multi-class)
# Sampling: undersample majority → SMOTE minority (combined strategy)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("[INFO] imbalanced-learn not installed — running without resampling")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] xgboost not installed")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[INFO] lightgbm not installed")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[INFO] torch not installed — skipping MLP with Focal Loss")

DATA_ROOT    = Path("/Users/danggiahan/physionet.org/files/circor-heart-sound/1.0.3")
N_SPLITS     = 10
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_FEATURES   = 80
DROP_CLASSES = {"Late-systolic"}


# ── Focal Loss MLP (PyTorch) ─────────────────────────────────────────────────

if HAS_TORCH:
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, weight=None):
            super().__init__()
            self.gamma = gamma
            self.weight = weight

        def forward(self, logits, targets):
            ce = nn.functional.cross_entropy(logits, targets,
                                              weight=self.weight, reduction="none")
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()


    class MLP(nn.Module):
        def __init__(self, n_in, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.net(x)


    class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
        """Sklearn-compatible wrapper for PyTorch MLP with Focal Loss."""

        def __init__(self, n_epochs=100, batch_size=32, lr=0.001,
                     gamma=2.0, random_state=42):
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.lr = lr
            self.gamma = gamma
            self.random_state = random_state

        def fit(self, X, y):
            torch.manual_seed(self.random_state)
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            self.le_ = LabelEncoder().fit(y)
            y_enc = self.le_.transform(y)

            counts = np.bincount(y_enc)
            weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32)
            weights = weights / weights.sum() * n_classes

            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y_enc, dtype=torch.long)
            loader = DataLoader(TensorDataset(X_t, y_t),
                                batch_size=self.batch_size, shuffle=True)

            self.model_ = MLP(X.shape[1], n_classes)
            criterion = FocalLoss(gamma=self.gamma, weight=weights)
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr,
                                   weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.n_epochs)

            self.model_.train()
            for _ in range(self.n_epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    criterion(self.model_(xb), yb).backward()
                    optimizer.step()
                scheduler.step()
            return self

        def predict(self, X):
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(torch.tensor(X, dtype=torch.float32))
                preds = logits.argmax(dim=1).numpy()
            return self.le_.inverse_transform(preds)

        def predict_proba(self, X):
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(torch.tensor(X, dtype=torch.float32))
                return torch.softmax(logits, dim=1).numpy()


# ── XGBoost + LightGBM soft-voting ensemble ──────────────────────────────────

class XGBLGBEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        y_enc = self.le_.transform(y)

        self.xgb_ = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=self.random_state, n_jobs=-1
        )
        self.lgb_ = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=self.random_state,
            n_jobs=-1, verbose=-1
        )
        self.xgb_.fit(X, y_enc)
        self.lgb_.fit(X, y_enc)
        return self

    def predict_proba(self, X):
        p_xgb = self.xgb_.predict_proba(X)
        p_lgb = self.lgb_.predict_proba(X)
        return (p_xgb + p_lgb) / 2

    def predict(self, X):
        return self.le_.inverse_transform(self.predict_proba(X).argmax(axis=1))


# ── Data loading ─────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path).dropna()
    df = df[~df["label"].isin(DROP_CLASSES)]
    meta_cols = {"patient_id", "location", "label"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = SimpleImputer(strategy="mean").fit_transform(
        df[feat_cols].values.astype(float))
    return X, df["label"].values, df["patient_id"].values, feat_cols


# ── Resampling: undersample majority → SMOTE minority ────────────────────────

def resample(X_train, y_train):
    if not HAS_IMBLEARN:
        return X_train, y_train
    try:
        counts = pd.Series(y_train).value_counts()
        minority_n = counts.min()
        # Undersample: cap majority at 5× minority
        under_strategy = {
            cls: min(cnt, minority_n * 5)
            for cls, cnt in counts.items()
        }
        under = RandomUnderSampler(sampling_strategy=under_strategy,
                                   random_state=RANDOM_STATE)
        X_u, y_u = under.fit_resample(X_train, y_train)

        # SMOTE: bring minority up to ~50% of majority
        counts_u = pd.Series(y_u).value_counts()
        majority_n = counts_u.max()
        over_strategy = {
            cls: max(cnt, majority_n // 2)
            for cls, cnt in counts_u.items()
            if cnt < majority_n
        }
        k = min(5, counts_u.min() - 1)
        if k >= 1 and over_strategy:
            smote = SMOTE(sampling_strategy=over_strategy,
                          k_neighbors=k, random_state=RANDOM_STATE)
            X_u, y_u = smote.fit_resample(X_u, y_u)
        return X_u, y_u
    except Exception:
        return X_train, y_train


# ── Feature selection ─────────────────────────────────────────────────────────

def select_features(X_train, y_train, n=N_FEATURES):
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    return np.argsort(rf.feature_importances_)[::-1][:n]


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(model_name, model_fn, X, y, groups):
    gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)
    accs, f1s_macro, f1s_weighted = [], [], []
    all_y_test, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        top_idx = select_features(X_train, y_train)
        X_train, X_test = X_train[:, top_idx], X_test[:, top_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        X_train, y_train = resample(X_train, y_train)

        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s_macro.append(f1_score(y_test, y_pred, average="macro",
                                   zero_division=0))
        f1s_weighted.append(f1_score(y_test, y_pred, average="weighted",
                                      zero_division=0))
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    print(f"  Accuracy:    {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  F1 macro:    {np.mean(f1s_macro)*100:.2f}% ± {np.std(f1s_macro)*100:.2f}%")
    print(f"  F1 weighted: {np.mean(f1s_weighted)*100:.2f}% ± {np.std(f1s_weighted)*100:.2f}%")
    print(classification_report(all_y_test, all_y_pred, zero_division=0))
    labels = sorted(set(all_y_test))
    print(confusion_matrix(all_y_test, all_y_pred, labels=labels))

    return {
        "accuracy": np.mean(accs) * 100,
        "f1_macro": np.mean(f1s_macro) * 100,
        "f1_weighted": np.mean(f1s_weighted) * 100,
    }


# ── Model registry ────────────────────────────────────────────────────────────

def get_models():
    models = {
        "SVM": lambda: SVC(kernel="rbf", class_weight="balanced",
                           C=10, gamma="scale",
                           random_state=RANDOM_STATE, probability=True),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE),
    }
    if HAS_XGB and HAS_LGB:
        models["XGB+LGB Ensemble"] = lambda: XGBLGBEnsemble(
            random_state=RANDOM_STATE)
    elif HAS_XGB:
        models["XGBoost"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1)
    if HAS_TORCH:
        models["MLP + Focal Loss"] = lambda: TorchMLPClassifier(
            n_epochs=150, lr=0.001, gamma=2.0,
            random_state=RANDOM_STATE)
    return models


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results):
    model_names = list(next(iter(results.values())).keys())
    print("\n" + "=" * 80)
    print("SUMMARY  (Late-systolic dropped | top-80 features | undersample+SMOTE)")
    print(f"  Primary metric: F1 macro\n")
    print(f"{'Model':25s} {'Metric':12s} {'Original':>10s} {'Separated':>10s} {'Δ':>8s}")
    print("-" * 80)
    for name in model_names:
        for metric in ["f1_macro", "accuracy"]:
            o = results["Original WAV"].get(name, {}).get(metric)
            s = results["Separated Murmur"].get(name, {}).get(metric)
            if o is None or s is None:
                continue
            delta = s - o
            sign = "+" if delta >= 0 else ""
            label = f"{name} {metric}"
            print(f"  {label:33s} {o:>10.2f}%  {s:>10.2f}%  ({sign}{delta:.2f}%)")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    orig_csv = DATA_ROOT / "features_v2_original.csv"
    sep_csv  = DATA_ROOT / "features_v2_separated.csv"

    missing = [p for p in [orig_csv, sep_csv] if not p.exists()]
    if missing:
        print("Missing CSVs — run extract_features_v2.py first:")
        for p in missing:
            print(f"  {p}")
        return

    models = get_models()
    results = {}

    for condition, csv_path in [("Original WAV", orig_csv),
                                  ("Separated Murmur", sep_csv)]:
        X, y, groups, feat_cols = load(csv_path)
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"  Samples: {len(X)}, Features: {len(feat_cols)} → top {N_FEATURES}")
        print(f"  Labels: {pd.Series(y).value_counts().to_dict()}")

        results[condition] = {}
        for name, model_fn in models.items():
            print(f"\n  -- {name} --")
            results[condition][name] = evaluate(name, model_fn, X, y, groups)

    print_summary(results)


if __name__ == "__main__":
    main()
