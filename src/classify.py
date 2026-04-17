# classify.py
# Compare classification on original WAV vs separated murmur vs TSV systole only

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score

features_original = "/Users/danggiahan/Documents/heart_sounds/features_original.csv"
features_separated = "/Users/danggiahan/Documents/heart_sounds/features.csv"
features_tsv = "/Users/danggiahan/Documents/heart_sounds/features_tsv.csv"

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def load(path):
    df = pd.read_csv(path)
    X = df[[col for col in df.columns if col.startswith("mfcc_")]].values
    y = df["label"].values
    return X, y

def evaluate(model, X, y):
    accs, f1s, sens, specs = [], [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, pos_label="Holosystolic"))
        sens.append(recall_score(y_test, y_pred, pos_label="Holosystolic"))
        specs.append(recall_score(y_test, y_pred, pos_label="Non-holosystolic"))
    return {
        "Accuracy":    round(np.mean(accs) * 100, 2),
        "F1 Score":    round(np.mean(f1s) * 100, 2),
        "Sensitivity": round(np.mean(sens) * 100, 2),
        "Specificity": round(np.mean(specs) * 100, 2),
    }

X_orig, y_orig = load(features_original)
X_sep, y_sep = load(features_separated)
X_tsv, y_tsv = load(features_tsv)

models = {
    "SVM": SVC(kernel="rbf", random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
}

for name, model in models.items():
    print(f"\n{name}")
    print(f"{'':20} {'Accuracy':>10} {'F1 Score':>10} {'Sensitivity':>12} {'Specificity':>12}")
    print("-" * 70)

    r1 = evaluate(model, X_orig, y_orig)
    print(f"{'Original WAV':<20} {r1['Accuracy']:>9}% {r1['F1 Score']:>9}% {r1['Sensitivity']:>11}% {r1['Specificity']:>11}%")

    r2 = evaluate(model, X_sep, y_sep)
    print(f"{'Separated Murmur':<20} {r2['Accuracy']:>9}% {r2['F1 Score']:>9}% {r2['Sensitivity']:>11}% {r2['Specificity']:>11}%")

    r3 = evaluate(model, X_tsv, y_tsv)
    print(f"{'TSV Systole Only':<20} {r3['Accuracy']:>9}% {r3['F1 Score']:>9}% {r3['Sensitivity']:>11}% {r3['Specificity']:>11}%")