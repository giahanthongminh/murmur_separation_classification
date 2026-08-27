"""Run the fixed Task 6 classical feature baselines with patient-level CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import METADATA_PATH, OUTPUT_ROOT
from src.task6.common import (
    FEATURE_GROUPS,
    TASK6_VERSION,
    aggregate_cycle_probabilities,
    classification_metrics,
    eligible_patient_labels,
    git_value,
    make_outer_folds,
    patient_bootstrap_intervals,
    sha256_file,
    validate_model_inputs,
    verify_manifest_artifacts,
)


TOOL_VERSION: Final = f"{TASK6_VERSION}-baseline"
DEFAULT_OUTPUT_ROOT: Final = OUTPUT_ROOT / "task6_evaluation"


def _verified_sources(task5_run: Path, retention_run: Path) -> tuple[Path, dict[str, Any]]:
    task5_run, retention_run = Path(task5_run).resolve(), Path(retention_run).resolve()
    task5_manifest = json.loads((task5_run / "run_manifest.json").read_text(encoding="utf-8"))
    retention_manifest = json.loads((retention_run / "run_manifest.json").read_text(encoding="utf-8"))
    feature_path = task5_run / "task5_feature_table.csv"
    expected = task5_manifest.get("artifact_sha256", {}).get(feature_path.name)
    if task5_manifest.get("status") != "complete" or not expected or sha256_file(feature_path) != expected:
        raise ValueError("Task 5 Phase 2A feature source is not verified")
    if task5_manifest.get("metadata_sha256") != sha256_file(METADATA_PATH):
        raise ValueError("current metadata hash differs from the Task 5 source")
    if retention_manifest.get("status") != "complete":
        raise ValueError("Task 5 retention source is not complete")
    groups_path = retention_run / "retained_feature_groups.json"
    expected_groups = retention_manifest.get("artifact_sha256", {}).get(groups_path.name)
    if not expected_groups or sha256_file(groups_path) != expected_groups:
        raise ValueError("Task 5 retention group hash mismatch")
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    if tuple(groups.get("core", [])) != FEATURE_GROUPS["core"] or tuple(groups.get("exploratory", [])) != FEATURE_GROUPS["exploratory"]:
        raise ValueError("frozen Task 5 retention groups differ from Task 6 registry")
    if retention_manifest.get("source_hashes", {}).get("phase2a_feature_table") != sha256_file(feature_path):
        raise ValueError("retention source is not the supplied Task 5 feature table")
    return feature_path, {"task5": task5_manifest, "retention": retention_manifest}


def build_patient_feature_table(feature_table: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Give recordings equal weight, then aggregate recording medians per patient."""

    features = tuple(FEATURE_GROUPS["core_plus_exploratory"])
    required = {
        "patient_id", "recording_id", "murmur_phase", "patient_murmur_label",
        "clinical_outcome",
        *features, *[f"{feature}_valid" for feature in features],
    }
    missing = required - set(feature_table.columns)
    if missing:
        raise ValueError(f"Task 5 table is missing baseline columns: {sorted(missing)}")
    eligible = set(labels["patient_id"])
    selected = feature_table.loc[
        feature_table["patient_id"].astype(str).isin(eligible)
        & feature_table["patient_murmur_label"].eq("Present")
        & feature_table["murmur_phase"].eq("systole")
    ].copy()
    selected["patient_id"] = selected["patient_id"].astype(str)
    source_labels = selected[["patient_id", "clinical_outcome"]].drop_duplicates()
    if source_labels["patient_id"].duplicated().any():
        raise ValueError("Task 5 contains inconsistent Outcome labels within patient")
    label_check = labels.merge(
        source_labels, on=["patient_id", "clinical_outcome"], how="outer", indicator=True
    )
    if not label_check["_merge"].eq("both").all():
        raise ValueError("Task 5 and current metadata Outcome labels disagree")
    recording_rows: list[dict[str, Any]] = []
    for (patient_id, recording_id), group in selected.groupby(["patient_id", "recording_id"], sort=True):
        row: dict[str, Any] = {"patient_id": patient_id, "recording_id": recording_id}
        for feature in features:
            valid = group[f"{feature}_valid"].fillna(False).astype(bool)
            values = pd.to_numeric(group.loc[valid, feature], errors="coerce").dropna()
            row[feature] = float(values.median()) if len(values) else np.nan
        recording_rows.append(row)
    recording = pd.DataFrame(recording_rows)
    patient = recording.groupby("patient_id", sort=True)[list(features)].median().reset_index()
    patient = labels.merge(patient, on="patient_id", how="left", validate="one_to_one")
    if patient[list(features)].isna().all(axis=1).any():
        missing_patients = patient.loc[patient[list(features)].isna().all(axis=1), "patient_id"].tolist()
        raise ValueError(f"eligible patients have no valid retained features: {missing_patients}")
    return patient


def fit_feature_pipeline(
    train: pd.DataFrame, test: pd.DataFrame, features: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, Pipeline, dict[str, Any]]:
    """Fit imputation and scaling only on the supplied outer-training patients."""

    validate_model_inputs(features)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    train_values = pipeline.fit_transform(train.loc[:, features])
    test_values = pipeline.transform(test.loc[:, features])
    audit = {
        "training_patient_ids": sorted(train["patient_id"].astype(str)),
        "test_patient_ids": sorted(test["patient_id"].astype(str)),
        "features": list(features),
        "imputer_medians": pipeline.named_steps["imputer"].statistics_.tolist(),
        "scaler_means_after_imputation": pipeline.named_steps["scaler"].mean_.tolist(),
        "scaler_scales": pipeline.named_steps["scaler"].scale_.tolist(),
    }
    return train_values, test_values, pipeline, audit


def _model_predictions(
    patient_features: pd.DataFrame, folds: pd.DataFrame, features: tuple[str, ...], model_name: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    preprocessing: list[dict[str, Any]] = []
    table = patient_features.merge(folds[["patient_id", "outer_fold"]], on="patient_id", validate="one_to_one")
    for outer_fold in sorted(table["outer_fold"].unique()):
        train = table.loc[table["outer_fold"].ne(outer_fold)].copy()
        test = table.loc[table["outer_fold"].eq(outer_fold)].copy()
        y_train = train["clinical_outcome"].map({"Normal": 0, "Abnormal": 1}).to_numpy()
        x_train, x_test, _, audit = fit_feature_pipeline(train, test, features)
        if model_name == "logistic_regression":
            model = LogisticRegression(
                C=1.0, class_weight="balanced", solver="liblinear",
                max_iter=1000, random_state=20260827,
            )
        elif model_name == "dummy_prior_majority":
            model = DummyClassifier(strategy="prior")
        else:
            raise ValueError(f"unknown fixed baseline: {model_name}")
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        fold_rows = test[["patient_id", "clinical_outcome", "outer_fold"]].copy()
        fold_rows["probability_abnormal"] = probability
        fold_rows["model"] = model_name
        rows.append(fold_rows)
        preprocessing.append({"outer_fold": int(outer_fold), "model": model_name, **audit})
    return pd.concat(rows, ignore_index=True).sort_values("patient_id"), preprocessing


def run_baselines(
    *, task5_run: Path, retention_run: Path, run_name: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT, n_splits: int = 5,
    bootstrap_replicates: int = 2000,
) -> Path:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one path component")
    destination = Path(output_root).resolve() / run_name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Task 6 evaluation: {destination}")
    destination.mkdir(parents=True)
    feature_path, source_manifests = _verified_sources(task5_run, retention_run)
    labels = eligible_patient_labels(pd.read_csv(METADATA_PATH, dtype={"Patient ID": str}))
    usecols = [
        "patient_id", "recording_id", "murmur_phase", "patient_murmur_label",
        "clinical_outcome",
        *FEATURE_GROUPS["core_plus_exploratory"],
        *[f"{feature}_valid" for feature in FEATURE_GROUPS["core_plus_exploratory"]],
    ]
    feature_table = pd.read_csv(feature_path, usecols=usecols, dtype={"patient_id": str, "recording_id": str})
    patient_features = build_patient_feature_table(feature_table, labels)
    folds = make_outer_folds(labels, n_splits=n_splits)
    folds.to_csv(destination / "fold_assignments.csv", index=False)
    patient_features.to_csv(destination / "patient_feature_table_audit_only.csv", index=False)

    all_predictions: list[pd.DataFrame] = []
    preprocessing: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    ci_rows: list[pd.DataFrame] = []
    for group_name, features in FEATURE_GROUPS.items():
        for model_name in ("logistic_regression", "dummy_prior_majority") if group_name == "core_plus_exploratory" else ("logistic_regression",):
            raw, audits = _model_predictions(patient_features, folds, features, model_name)
            patient_predictions = aggregate_cycle_probabilities(raw)
            patient_predictions["outer_fold"] = patient_predictions["patient_id"].map(folds.set_index("patient_id")["outer_fold"])
            patient_predictions["feature_group"] = group_name
            patient_predictions["model"] = model_name
            all_predictions.append(patient_predictions)
            preprocessing.extend({"feature_group": group_name, **item} for item in audits)
            metrics = classification_metrics(patient_predictions)
            summary_rows.append({"feature_group": group_name, "model": model_name, **{k: v for k, v in metrics.items() if k != "confusion_matrix_normal_abnormal"}, "confusion_matrix_normal_abnormal": json.dumps(metrics["confusion_matrix_normal_abnormal"])})
            for outer_fold, fold_predictions in patient_predictions.groupby("outer_fold", sort=True):
                fold_metrics = classification_metrics(fold_predictions)
                fold_metric_rows.append({
                    "feature_group": group_name, "model": model_name,
                    "outer_fold": int(outer_fold),
                    **{k: v for k, v in fold_metrics.items() if k != "confusion_matrix_normal_abnormal"},
                    "confusion_matrix_normal_abnormal": json.dumps(fold_metrics["confusion_matrix_normal_abnormal"]),
                })
            ci = patient_bootstrap_intervals(patient_predictions, replicates=bootstrap_replicates)
            ci.insert(0, "model", model_name)
            ci.insert(0, "feature_group", group_name)
            ci_rows.append(ci)

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(summary_rows)
    intervals = pd.concat(ci_rows, ignore_index=True)
    predictions.to_csv(destination / "baseline_patient_predictions.csv", index=False)
    metrics.to_csv(destination / "baseline_metrics.csv", index=False)
    pd.DataFrame(fold_metric_rows).to_csv(destination / "baseline_fold_metrics.csv", index=False)
    intervals.to_csv(destination / "baseline_bootstrap_confidence_intervals.csv", index=False)
    (destination / "preprocessing_by_fold.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    design = {
        "target": "patient-level clinical Outcome: Normal versus Abnormal",
        "cohort": "patient-level Murmur=Present only",
        "feature_groups": {name: list(values) for name, values in FEATURE_GROUPS.items()},
        "fixed_classifier": "class-weighted logistic regression, C=1, liblinear",
        "dummy": "training-fold Outcome prior; 0.5 threshold yields the training majority class",
        "aggregation": "cycle median within recording, then recording median within patient",
        "outer_validation": f"{n_splits}-fold patient-level stratified cross-validation",
        "feature_selection": "frozen before Task 6; no test-fold feature selection",
        "limitations": ["small sample", "class imbalance", "no independent test set", "estimated separated candidates are not clean-source ground truth"],
    }
    (destination / "evaluation_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    artifact_names = (
        "fold_assignments.csv", "patient_feature_table_audit_only.csv",
        "baseline_patient_predictions.csv", "baseline_metrics.csv",
        "baseline_fold_metrics.csv",
        "baseline_bootstrap_confidence_intervals.csv", "preprocessing_by_fold.json",
        "evaluation_design.json",
    )
    manifest = {
        "tool_version": TOOL_VERSION, "status": "complete", "run_name": run_name,
        "eligible_patient_count": int(len(labels)),
        "outcome_counts": {str(k): int(v) for k, v in labels["clinical_outcome"].value_counts().sort_index().items()},
        "source_hashes": {
            "metadata": sha256_file(METADATA_PATH),
            "task5_feature_table": sha256_file(feature_path),
            "task5_run_manifest": sha256_file(Path(task5_run) / "run_manifest.json"),
            "retention_run_manifest": sha256_file(Path(retention_run) / "run_manifest.json"),
        },
        "source_git_commits": {
            "task5": source_manifests["task5"].get("git_commit"),
            "retention": source_manifests["retention"].get("git_commit"),
        },
        "artifact_sha256": {name: sha256_file(destination / name) for name in artifact_names},
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"),
        "python_version": platform.python_version(), "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "scope_statement": "Task 6 clinical Outcome classification among Murmur Present patients using frozen separated-murmur features.",
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_manifest_artifacts(destination, manifest)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task5-run", type=Path, required=True)
    parser.add_argument("--retention-run", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    destination = run_baselines(
        task5_run=args.task5_run, retention_run=args.retention_run,
        run_name=args.run_name, output_root=args.output_root,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(f"Task 6 baseline evaluation: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
