"""Shared, leakage-safe Task 6 data and evaluation utilities."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Final, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


TASK6_VERSION: Final = "task6-v1.0.0"
OUTCOME_ORDER: Final[tuple[str, str]] = ("Normal", "Abnormal")
CORE_FEATURES: Final[tuple[str, ...]] = (
    "tier_a_relative_murmur_duration",
    "tier_a_temporal_midpoint_normalized",
    "tier_a_murmur_rms_relative_s1_s2_db",
)
EXPLORATORY_FEATURES: Final[tuple[str, ...]] = (
    "tier_a_envelope_peak_position_normalized",
    "tier_a_envelope_rise_slope_robust",
    "tier_a_envelope_fullness",
    "tier_a_psd_dominant_frequency_hz",
    "tier_a_psd_bandwidth_95_hz",
    "tier_a_psd_entropy_normalized",
)
FEATURE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "core": CORE_FEATURES,
    "exploratory": EXPLORATORY_FEATURES,
    "core_plus_exploratory": CORE_FEATURES + EXPLORATORY_FEATURES,
}
PROHIBITED_INPUT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "Outcome", "clinical_outcome", "patient_id", "Patient ID",
        "recording_id", "location", "murmur_locations", "Murmur locations",
        "most_audible_location", "Most audible location",
        "candidate_quality_status", "phase_selection_used_fallback",
        "candidate_status", "config_hash", "reconstruction_error",
        "s1_leakage_ratio", "s2_leakage_ratio",
        "outside_murmur_energy_ratio", "murmur_region_energy_retention",
    }
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def validate_model_inputs(columns: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(columns)
    prohibited = sorted(set(selected) & PROHIBITED_INPUT_COLUMNS)
    if prohibited:
        raise ValueError(f"prohibited model input columns: {prohibited}")
    eligible = set(CORE_FEATURES + EXPLORATORY_FEATURES)
    unknown = sorted(set(selected) - eligible)
    if unknown:
        raise ValueError(f"features are not frozen Task 6 inputs: {unknown}")
    return selected


def eligible_patient_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return exactly one valid Outcome row per patient-level Murmur Present case."""

    required = {"Patient ID", "Murmur", "Outcome"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing label columns: {sorted(missing)}")
    table = metadata.loc[metadata["Murmur"].eq("Present"), ["Patient ID", "Outcome"]].copy()
    table = table.rename(columns={"Patient ID": "patient_id", "Outcome": "clinical_outcome"})
    table["patient_id"] = table["patient_id"].astype(str)
    if table["patient_id"].duplicated().any():
        raise ValueError("Murmur Present metadata must have one row per patient")
    invalid = sorted(set(table["clinical_outcome"].dropna()) - set(OUTCOME_ORDER))
    if invalid or table["clinical_outcome"].isna().any():
        raise ValueError(f"invalid clinical Outcome labels: {invalid}")
    return table.sort_values("patient_id", kind="mergesort").reset_index(drop=True)


def make_outer_folds(labels: pd.DataFrame, *, n_splits: int = 5, seed: int = 20260827) -> pd.DataFrame:
    """Create deterministic patient-only outer folds with class stratification."""

    if labels["patient_id"].duplicated().any():
        raise ValueError("fold labels must contain one row per patient")
    counts = labels["clinical_outcome"].value_counts()
    if len(counts) != 2 or counts.min() < n_splits:
        raise ValueError("both Outcome classes must support the requested fold count")
    ordered = labels.sort_values("patient_id", kind="mergesort").reset_index(drop=True)
    fold = np.full(len(ordered), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_index, (_, test_index) in enumerate(
        splitter.split(ordered["patient_id"], ordered["clinical_outcome"])
    ):
        fold[test_index] = fold_index
    if np.any(fold < 0):
        raise AssertionError("every eligible patient must receive one outer fold")
    return ordered.assign(outer_fold=fold)


def make_inner_validation(
    outer_training: pd.DataFrame, *, outer_fold: int, fraction: float = 0.20, seed: int = 20260827
) -> set[str]:
    """Select a deterministic stratified validation subset from outer-training patients."""

    ordered = outer_training.sort_values("patient_id", kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(seed + 1009 * (outer_fold + 1))
    selected: set[str] = set()
    for _, group in ordered.groupby("clinical_outcome", sort=True):
        count = max(1, int(round(len(group) * fraction)))
        indices = rng.permutation(len(group))[:count]
        selected.update(group.iloc[indices]["patient_id"].astype(str))
    return selected


def assert_patient_disjoint(*patient_sets: Iterable[str]) -> None:
    normalized = [set(map(str, values)) for values in patient_sets]
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            overlap = normalized[left] & normalized[right]
            if overlap:
                raise ValueError(f"patient leakage across partitions: {sorted(overlap)[:5]}")


def aggregate_cycle_probabilities(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average cycle probabilities per patient before assigning an Outcome."""

    required = {"patient_id", "clinical_outcome", "probability_abnormal"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    label_counts = predictions.groupby("patient_id")["clinical_outcome"].nunique()
    if (label_counts != 1).any():
        raise ValueError("each patient must have one consistent clinical Outcome")
    result = predictions.groupby("patient_id", sort=True, as_index=False).agg(
        clinical_outcome=("clinical_outcome", "first"),
        probability_abnormal=("probability_abnormal", "mean"),
        contributing_prediction_count=("probability_abnormal", "size"),
    )
    result["predicted_outcome"] = np.where(
        result["probability_abnormal"].to_numpy() >= .5, "Abnormal", "Normal"
    )
    return result


def classification_metrics(patient_predictions: pd.DataFrame) -> dict[str, Any]:
    truth = patient_predictions["clinical_outcome"].map({"Normal": 0, "Abnormal": 1}).to_numpy()
    probability = patient_predictions["probability_abnormal"].to_numpy(float)
    predicted = (probability >= .5).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "patient_count": int(len(truth)),
        "normal_count": int((truth == 0).sum()),
        "abnormal_count": int((truth == 1).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(truth, probability)),
        "sensitivity_abnormal": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity_normal": float(tn / (tn + fp)) if tn + fp else np.nan,
        "confusion_matrix_normal_abnormal": matrix.tolist(),
    }


def patient_bootstrap_intervals(
    predictions: pd.DataFrame, *, replicates: int = 2000, seed: int = 20260827
) -> pd.DataFrame:
    """Percentile CIs from patient-level stratified bootstrap samples."""

    rng = np.random.default_rng(seed)
    groups = [group.reset_index(drop=True) for _, group in predictions.groupby("clinical_outcome", sort=True)]
    metric_names = (
        "balanced_accuracy", "macro_f1", "auroc",
        "sensitivity_abnormal", "specificity_normal",
    )
    estimates: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(replicates):
        sampled = pd.concat(
            [group.iloc[rng.integers(0, len(group), size=len(group))] for group in groups],
            ignore_index=True,
        )
        values = classification_metrics(sampled)
        for name in metric_names:
            estimates[name].append(float(values[name]))
    point = classification_metrics(predictions)
    return pd.DataFrame(
        [
            {
                "metric": name,
                "estimate": point[name],
                "ci_95_low": float(np.quantile(estimates[name], .025)),
                "ci_95_high": float(np.quantile(estimates[name], .975)),
                "bootstrap_replicates": replicates,
            }
            for name in metric_names
        ]
    )


def verify_manifest_artifacts(directory: Path, manifest: dict[str, Any]) -> None:
    for filename, expected in manifest.get("artifact_sha256", {}).items():
        path = Path(directory) / filename
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"artifact hash mismatch: {path}")

