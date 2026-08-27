"""Task 5 Phase 1D separation/candidate-quality sensitivity audit.

This evaluation layer consumes the frozen Phase 1B candidate manifest and the
Phase 1C boundary-change table.  It never reruns or modifies separation,
component selection, fallback, activity detection, boundaries, or Tier A
feature extraction.  Descriptor associations are deliberately excluded: they
belong to the later construct-validation experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Final, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata

from config import OUTPUT_ROOT
from src.separation.tier_a_features import (
    FEATURE_SCHEMA_VERSION,
    TIER_A_FEATURE_COLUMNS,
    TIER_A_FEATURE_UNITS,
)


AUDIT_TOOL_VERSION: Final = "task5-phase1d-candidate-quality-v1.0.0"
DEFAULT_PHASE1B_RUN: Final = (
    OUTPUT_ROOT / "task5_candidate_audit" / "task5_phase1b_real_candidates_v2"
)
DEFAULT_PHASE1C_RUN: Final = (
    OUTPUT_ROOT
    / "task5_boundary_stability"
    / "task5_phase1c_boundary_stability_v2"
)
DEFAULT_OUTPUT_ROOT: Final = OUTPUT_ROOT / "task5_candidate_quality"

IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "selection_rank",
    "candidate_id",
    "patient_id",
    "recording_id",
    "location",
    "cycle_index",
    "murmur_phase",
)

# Exact repository column names are retained, as required by Experiment 3.
CONTINUOUS_QA_COLUMNS: Final[tuple[str, ...]] = (
    "reconstruction_error",
    "normal_residual_correlation",
    "s1_leakage_ratio",
    "s2_leakage_ratio",
    "murmur_region_energy_retention",
    "outside_murmur_energy_ratio",
    "systole_candidate_energy_ratio",
    "diastole_candidate_energy_ratio",
    "noise_energy_ratio",
    "phase_selected_component_count",
    "duration_seconds",
)

CATEGORICAL_QA_COLUMNS: Final[tuple[str, ...]] = (
    "candidate_quality_status",
    "separation_method",
    "selected_method",
    "phase_selection_used_fallback",
    "activity_detection_method",
    "boundary_stability_status",
    "murmur_phase",
)

FAVORABLE_QA_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("outside_murmur_energy_ratio", "low"),
    ("murmur_region_energy_retention", "high"),
    ("maximum_s1_s2_leakage_ratio", "low"),
    ("noise_energy_ratio", "low"),
)

EXCLUDED_DESCRIPTOR_PREFIXES: Final[tuple[str, ...]] = (
    "expert_",
    "systole_timing",
    "systole_shape",
    "systole_pitch",
    "systole_grading",
    "systole_quality",
    "diastole_timing",
    "diastole_shape",
    "diastole_pitch",
    "diastole_grading",
    "diastole_quality",
)


@dataclass(frozen=True)
class CandidateQualityConfig:
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 20260827
    confidence_level: float = 0.95
    minimum_correlation_patients: int = 4

    def __post_init__(self) -> None:
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")
        if self.minimum_correlation_patients < 3:
            raise ValueError("minimum_correlation_patients must be at least three")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _seed_for(config: CandidateQualityConfig, key: Iterable[Any]) -> int:
    encoded = "|".join(str(item) for item in key).encode("utf-8")
    offset = int.from_bytes(sha256(encoded).digest()[:4], "little")
    return (config.bootstrap_seed + offset) % (2**32)


def _percentile_interval(
    values: np.ndarray, config: CandidateQualityConfig
) -> tuple[float, float, int, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    valid_count = int(len(finite))
    failure_rate = 1.0 - valid_count / config.bootstrap_replicates
    if not valid_count:
        return np.nan, np.nan, valid_count, failure_rate
    alpha = (1.0 - config.confidence_level) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high), valid_count, float(failure_rate)


def _inference_status(
    estimate: float, independent_patients: int, failure_rate: float
) -> str:
    if not np.isfinite(estimate):
        return "not_estimable"
    if independent_patients < 8:
        return "descriptive_too_few_independent_patients"
    if failure_rate > 0.05:
        return "unstable_bootstrap_over_5pct_failures"
    return "estimable"


def _validate_unique_candidates(table: pd.DataFrame, label: str) -> None:
    required = {"candidate_id", "patient_id", "murmur_phase"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if table["candidate_id"].duplicated().any():
        raise ValueError(f"{label} has duplicate candidate identifiers")


def _sample_patient_clusters(
    patient_ids: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> list[np.ndarray] | np.ndarray:
    """Return row indices for a patient-cluster bootstrap.

    The common one-row-per-patient case is returned as a matrix for vectorized
    calculations. Repeated-patient inputs retain every row from each sampled
    patient and therefore return a list whose row counts may vary.
    """

    clusters = np.unique(patient_ids.astype(str))
    rows_by_cluster = [np.flatnonzero(patient_ids.astype(str) == cluster) for cluster in clusters]
    draws = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    if all(len(rows) == 1 for rows in rows_by_cluster):
        singleton_rows = np.asarray([rows[0] for rows in rows_by_cluster], dtype=int)
        return singleton_rows[draws]
    return [np.concatenate([rows_by_cluster[index] for index in draw]) for draw in draws]


def _required_phase1b_columns() -> set[str]:
    columns = set(IDENTIFIER_COLUMNS) | set(CONTINUOUS_QA_COLUMNS) | set(
        CATEGORICAL_QA_COLUMNS
    )
    for feature in TIER_A_FEATURE_COLUMNS:
        columns.update((feature, f"{feature}_valid", f"{feature}_invalid_reason"))
    return columns


def load_frozen_sources(
    source_phase1b: Path = DEFAULT_PHASE1B_RUN,
    source_phase1c: Path = DEFAULT_PHASE1C_RUN,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and cross-check the exact frozen Phase 1B/1C sources."""

    phase1b_path = Path(source_phase1b) / "candidate_manifest.csv"
    phase1c_path = Path(source_phase1c) / "feature_changes_long.csv"
    phase1c_manifest_path = Path(source_phase1c) / "run_manifest.json"
    for path in (phase1b_path, phase1c_path, phase1c_manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"required frozen source not found: {path}")

    phase1b = pd.read_csv(
        phase1b_path, dtype={"patient_id": str, "recording_id": str}
    )
    if len(phase1b) != 30:
        raise ValueError(f"Phase 1D requires exactly 30 frozen candidates; found {len(phase1b)}")
    missing = _required_phase1b_columns() - set(phase1b.columns)
    if missing:
        raise ValueError(f"Phase 1B manifest is missing columns: {sorted(missing)}")
    _validate_unique_candidates(phase1b, "Phase 1B manifest")

    phase1c = pd.read_csv(
        phase1c_path, dtype={"patient_id": str, "recording_id": str}
    )
    required_changes = {
        "candidate_id",
        "patient_id",
        "murmur_phase",
        "candidate_quality_status",
        "feature",
        "magnitude_fraction",
        "family",
        "baseline_value",
        "baseline_valid",
        "absolute_change",
        "relative_change",
        "validity_transition",
        "clipped",
    }
    missing_changes = required_changes - set(phase1c.columns)
    if missing_changes:
        raise ValueError(f"Phase 1C table is missing columns: {sorted(missing_changes)}")
    if set(phase1c["candidate_id"]) != set(phase1b["candidate_id"]):
        raise ValueError("Phase 1B and Phase 1C candidate identifiers differ")
    if set(phase1c["feature"]) != set(TIER_A_FEATURE_COLUMNS):
        raise ValueError("Phase 1C feature registry differs from frozen Tier A schema")

    phase1c_manifest = json.loads(phase1c_manifest_path.read_text(encoding="utf-8"))
    actual_hash = _sha256_file(phase1b_path)
    expected_hash = phase1c_manifest.get("source_candidate_manifest_sha256")
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("Phase 1C was not generated from this Phase 1B manifest")

    source_info = {
        "phase1b_manifest": str(phase1b_path.resolve()),
        "phase1b_manifest_sha256": actual_hash,
        "phase1c_feature_changes": str(phase1c_path.resolve()),
        "phase1c_feature_changes_sha256": _sha256_file(phase1c_path),
        "phase1c_run_manifest": str(phase1c_manifest_path.resolve()),
        "phase1c_run_manifest_sha256": _sha256_file(phase1c_manifest_path),
    }
    return phase1b, phase1c, source_info


def assign_high_quality_stratum(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze a phase-relative QA-quantile stratum without descriptor access.

    A candidate is high quality when it is accepted and meets at least three of
    four favorable within-phase median conditions: low outside energy, high
    retention, low maximum S1/S2 leakage, and low noise energy.
    """

    table = candidates.copy()
    table["maximum_s1_s2_leakage_ratio"] = table[
        ["s1_leakage_ratio", "s2_leakage_ratio"]
    ].max(axis=1)
    threshold_rows: list[dict[str, Any]] = []
    score = pd.Series(0, index=table.index, dtype=int)
    for phase in sorted(table["murmur_phase"].dropna().astype(str).unique()):
        phase_mask = table["murmur_phase"].astype(str).eq(phase)
        for column, direction in FAVORABLE_QA_COLUMNS:
            numeric = pd.to_numeric(table.loc[phase_mask, column], errors="coerce")
            threshold = float(numeric.median()) if numeric.notna().any() else np.nan
            threshold_rows.append(
                {
                    "murmur_phase": phase,
                    "qa_variable": column,
                    "quantile": 0.50,
                    "direction": direction,
                    "threshold": threshold,
                    "candidate_count": int(phase_mask.sum()),
                    "nonmissing_count": int(numeric.notna().sum()),
                }
            )
            values = pd.to_numeric(table[column], errors="coerce")
            favorable = values.le(threshold) if direction == "low" else values.ge(threshold)
            score.loc[phase_mask] += favorable.loc[phase_mask].fillna(False).astype(int)
    table["high_quality_favorable_condition_count"] = score
    table["high_quality_stratum"] = (
        table["candidate_quality_status"].astype(str).eq("accepted") & score.ge(3)
    )
    table["high_quality_rule_version"] = "accepted-and-3-of-4-within-phase-medians-v1"
    return table, pd.DataFrame(threshold_rows)


def prepare_analysis_candidates(phase1b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the descriptor-free Phase 1D candidate table."""

    columns = list(dict.fromkeys(
        [*IDENTIFIER_COLUMNS, *CONTINUOUS_QA_COLUMNS, *CATEGORICAL_QA_COLUMNS]
        + [
            item
            for feature in TIER_A_FEATURE_COLUMNS
            for item in (feature, f"{feature}_valid", f"{feature}_invalid_reason")
        ]
    ))
    table = phase1b.loc[:, columns].copy()
    forbidden = [
        column
        for column in table.columns
        if column.startswith(EXCLUDED_DESCRIPTOR_PREFIXES)
    ]
    if forbidden:
        raise AssertionError(f"descriptor columns leaked into Phase 1D: {forbidden}")
    table, thresholds = assign_high_quality_stratum(table)
    return table.sort_values("selection_rank", kind="mergesort"), thresholds


def summarize_ten_percent_boundary_changes(
    phase1c: pd.DataFrame,
) -> pd.DataFrame:
    """Reduce the eight 10% families to per-candidate feature outcomes."""

    ten = phase1c.loc[np.isclose(phase1c["magnitude_fraction"], 0.10)].copy()
    counts = ten.groupby(["candidate_id", "feature"], sort=False).size()
    if not counts.eq(8).all():
        raise ValueError("every candidate-feature must have exactly eight 10% perturbations")
    baseline = ten.drop_duplicates(["candidate_id", "feature"])
    feature_iqr = (
        baseline.loc[baseline["baseline_valid"].astype(bool)]
        .groupby("feature", sort=False)["baseline_value"]
        .quantile(0.75)
        .sub(
            baseline.loc[baseline["baseline_valid"].astype(bool)]
            .groupby("feature", sort=False)["baseline_value"]
            .quantile(0.25)
        )
    )
    rows: list[dict[str, Any]] = []
    for (candidate_id, feature), group in ten.groupby(
        ["candidate_id", "feature"], sort=True
    ):
        changes = pd.to_numeric(group["absolute_change"], errors="coerce")
        relative = pd.to_numeric(group["relative_change"], errors="coerce")
        valid_mask = group["validity_transition"].eq("valid_to_valid") & changes.notna()
        primary_mask = valid_mask & ~group["clipped"].astype(bool)
        valid_changes = changes.loc[primary_mask]
        valid_relative = relative.loc[primary_mask].dropna()
        all_effective_changes = changes.loc[valid_mask]
        iqr = float(feature_iqr.get(feature, np.nan))
        median_change = float(valid_changes.median()) if len(valid_changes) else np.nan
        first = group.sort_values("family", kind="mergesort").iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "patient_id": first["patient_id"],
                "murmur_phase": first["murmur_phase"],
                "candidate_quality_status": first["candidate_quality_status"],
                "feature": feature,
                "feature_unit": TIER_A_FEATURE_UNITS[str(feature)],
                "baseline_value": first["baseline_value"],
                "baseline_valid": bool(first["baseline_valid"]),
                "boundary_change_10pct_family_count": int(len(group)),
                "boundary_change_10pct_valid_count": int(len(valid_changes)),
                "boundary_change_10pct_primary_unclipped_valid_count": int(len(valid_changes)),
                "boundary_change_10pct_median_absolute": median_change,
                "boundary_change_10pct_max_absolute": (
                    float(valid_changes.max()) if len(valid_changes) else np.nan
                ),
                "boundary_change_10pct_median_relative": (
                    float(valid_relative.median()) if len(valid_relative) else np.nan
                ),
                "boundary_change_10pct_baseline_iqr": iqr,
                "boundary_change_10pct_median_iqr_standardized": (
                    median_change / iqr
                    if np.isfinite(median_change) and np.isfinite(iqr) and iqr > 0
                    else np.nan
                ),
                "boundary_change_10pct_valid_to_invalid_count": int(
                    group["validity_transition"].eq("valid_to_invalid").sum()
                ),
                "boundary_change_10pct_clipped_count": int(
                    group["clipped"].astype(bool).sum()
                ),
                "boundary_change_10pct_all_effective_valid_count": int(
                    len(all_effective_changes)
                ),
                "boundary_change_10pct_all_effective_median_absolute": (
                    float(all_effective_changes.median())
                    if len(all_effective_changes)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_feature_status(
    candidates: pd.DataFrame, boundary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions = (("phase_status", ["murmur_phase", "candidate_quality_status"]),)
    for _, group_columns in dimensions:
        for keys, group in candidates.groupby(group_columns, sort=True, dropna=False):
            phase, status = keys
            ids = set(group["candidate_id"])
            for feature in TIER_A_FEATURE_COLUMNS:
                valid = group[f"{feature}_valid"].astype(bool)
                values = pd.to_numeric(group.loc[valid, feature], errors="coerce").dropna()
                stability = boundary.loc[
                    boundary["candidate_id"].isin(ids) & boundary["feature"].eq(feature)
                ]
                changes = pd.to_numeric(
                    stability["boundary_change_10pct_median_absolute"], errors="coerce"
                ).dropna()
                rows.append(
                    {
                        "murmur_phase": phase,
                        "candidate_quality_status": status,
                        "feature": feature,
                        "feature_unit": TIER_A_FEATURE_UNITS[feature],
                        "candidate_count": int(len(group)),
                        "feature_valid_count": int(len(values)),
                        "feature_coverage": float(len(values) / len(group)),
                        "feature_median": float(values.median()) if len(values) else np.nan,
                        "feature_q25": float(values.quantile(0.25)) if len(values) else np.nan,
                        "feature_q75": float(values.quantile(0.75)) if len(values) else np.nan,
                        "boundary_outcome_valid_count": int(len(changes)),
                        "boundary_change_10pct_median_absolute": (
                            float(changes.median()) if len(changes) else np.nan
                        ),
                        "boundary_valid_to_invalid_count": int(
                            stability["boundary_change_10pct_valid_to_invalid_count"].sum()
                        ),
                        "boundary_clipped_family_count": int(
                            stability["boundary_change_10pct_clipped_count"].sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _cliffs_delta(accepted: np.ndarray, fallback: np.ndarray) -> float:
    if not len(accepted) or not len(fallback):
        return np.nan
    differences = accepted[:, None] - fallback[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def accepted_fallback_contrast(
    table: pd.DataFrame,
    value_column: str,
    *,
    phase: str,
    feature: str,
    outcome: str,
    config: CandidateQualityConfig,
) -> dict[str, Any]:
    """Patient-cluster bootstrap accepted-minus-fallback median and Cliff delta."""

    subset = table.loc[
        table["murmur_phase"].astype(str).eq(phase),
        ["patient_id", "candidate_quality_status", value_column],
    ].copy()
    subset[value_column] = pd.to_numeric(subset[value_column], errors="coerce")
    subset.dropna(subset=[value_column], inplace=True)
    accepted = subset.loc[
        subset["candidate_quality_status"].eq("accepted"), value_column
    ].to_numpy(float)
    fallback = subset.loc[
        subset["candidate_quality_status"].eq("fallback"), value_column
    ].to_numpy(float)
    median_difference = (
        float(np.median(accepted) - np.median(fallback))
        if len(accepted) and len(fallback)
        else np.nan
    )
    delta = _cliffs_delta(accepted, fallback)
    bootstrap_median = np.full(config.bootstrap_replicates, np.nan)
    bootstrap_delta = np.full(config.bootstrap_replicates, np.nan)
    if len(subset):
        rng = np.random.default_rng(
            _seed_for(config, ("status", phase, feature, outcome))
        )
        indices = _sample_patient_clusters(
            subset["patient_id"].to_numpy(str),
            replicates=config.bootstrap_replicates,
            rng=rng,
        )
        values = subset[value_column].to_numpy(float)
        statuses = subset["candidate_quality_status"].to_numpy(str)
        for index, sampled in enumerate(indices):
            sampled_values = values[sampled]
            sampled_status = statuses[sampled]
            a = sampled_values[sampled_status == "accepted"]
            f = sampled_values[sampled_status == "fallback"]
            if len(a) and len(f):
                bootstrap_median[index] = np.median(a) - np.median(f)
                bootstrap_delta[index] = _cliffs_delta(a, f)
    med_low, med_high, med_valid, med_failure = _percentile_interval(
        bootstrap_median, config
    )
    delta_low, delta_high, delta_valid, delta_failure = _percentile_interval(
        bootstrap_delta, config
    )
    independent = int(subset["patient_id"].nunique())
    return {
        "murmur_phase": phase,
        "feature": feature,
        "outcome": outcome,
        "accepted_count": int(len(accepted)),
        "fallback_count": int(len(fallback)),
        "independent_patient_count": independent,
        "accepted_median": float(np.median(accepted)) if len(accepted) else np.nan,
        "fallback_median": float(np.median(fallback)) if len(fallback) else np.nan,
        "accepted_minus_fallback_median_difference": median_difference,
        "median_difference_ci_low": med_low,
        "median_difference_ci_high": med_high,
        "median_difference_bootstrap_valid": med_valid,
        "median_difference_bootstrap_failure_rate": med_failure,
        "cliffs_delta_accepted_over_fallback": delta,
        "cliffs_delta_ci_low": delta_low,
        "cliffs_delta_ci_high": delta_high,
        "cliffs_delta_bootstrap_valid": delta_valid,
        "cliffs_delta_bootstrap_failure_rate": delta_failure,
        "inference_status": _inference_status(
            median_difference, independent, max(med_failure, delta_failure)
        ),
    }


def status_contrasts(
    candidates: pd.DataFrame,
    boundary: pd.DataFrame,
    config: CandidateQualityConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    phases = sorted(candidates["murmur_phase"].dropna().astype(str).unique())
    for phase in phases:
        for feature in TIER_A_FEATURE_COLUMNS:
            feature_table = candidates.loc[
                candidates[f"{feature}_valid"].astype(bool),
                ["patient_id", "murmur_phase", "candidate_quality_status", feature],
            ].rename(columns={feature: "value"})
            rows.append(
                accepted_fallback_contrast(
                    feature_table,
                    "value",
                    phase=phase,
                    feature=feature,
                    outcome="nominal_feature_value",
                    config=config,
                )
            )
            stability = boundary.loc[
                boundary["feature"].eq(feature),
                [
                    "patient_id",
                    "murmur_phase",
                    "candidate_quality_status",
                    "boundary_change_10pct_median_absolute",
                ],
            ].rename(columns={"boundary_change_10pct_median_absolute": "value"})
            rows.append(
                accepted_fallback_contrast(
                    stability,
                    "value",
                    phase=phase,
                    feature=feature,
                    outcome="boundary_change_10pct_median_absolute",
                    config=config,
                )
            )
    return pd.DataFrame(rows)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    return float(np.sum(x_centered * y_centered) / denominator) if denominator else np.nan


def spearman_with_cluster_interval(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    key: tuple[Any, ...],
    config: CandidateQualityConfig,
) -> dict[str, Any]:
    subset = table[["patient_id", x_column, y_column]].copy()
    subset[x_column] = pd.to_numeric(subset[x_column], errors="coerce")
    subset[y_column] = pd.to_numeric(subset[y_column], errors="coerce")
    subset.dropna(inplace=True)
    independent = int(subset["patient_id"].nunique())
    x = subset[x_column].to_numpy(float)
    y = subset[y_column].to_numpy(float)
    estimate = (
        _spearman(x, y)
        if independent >= config.minimum_correlation_patients
        else np.nan
    )
    boot = np.full(config.bootstrap_replicates, np.nan)
    if np.isfinite(estimate):
        rng = np.random.default_rng(_seed_for(config, ("spearman", *key)))
        indices = _sample_patient_clusters(
            subset["patient_id"].to_numpy(str),
            replicates=config.bootstrap_replicates,
            rng=rng,
        )
        if isinstance(indices, np.ndarray):
            xb = x[indices]
            yb = y[indices]
            xr = rankdata(xb, axis=1)
            yr = rankdata(yb, axis=1)
            xr -= xr.mean(axis=1, keepdims=True)
            yr -= yr.mean(axis=1, keepdims=True)
            denominator = np.sqrt(np.sum(xr**2, axis=1) * np.sum(yr**2, axis=1))
            valid = denominator > 0
            boot[valid] = np.sum(xr[valid] * yr[valid], axis=1) / denominator[valid]
        else:
            for index, sampled in enumerate(indices):
                boot[index] = _spearman(x[sampled], y[sampled])
    low, high, valid_count, failure_rate = _percentile_interval(boot, config)
    return {
        "overlap_count": int(len(subset)),
        "independent_patient_count": independent,
        "spearman_rho": estimate,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_valid": valid_count,
        "bootstrap_failure_rate": failure_rate,
        "inference_status": _inference_status(estimate, independent, failure_rate),
    }


def qa_associations(
    candidates: pd.DataFrame,
    boundary: pd.DataFrame,
    config: CandidateQualityConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_qa = candidates[["candidate_id", "patient_id", *CONTINUOUS_QA_COLUMNS]]
    phases = sorted(candidates["murmur_phase"].dropna().astype(str).unique())
    for phase in phases:
        for feature in TIER_A_FEATURE_COLUMNS:
            nominal = candidates.loc[
                candidates["murmur_phase"].astype(str).eq(phase)
                & candidates[f"{feature}_valid"].astype(bool),
                ["candidate_id", "patient_id", *CONTINUOUS_QA_COLUMNS, feature],
            ].rename(columns={feature: "outcome_value"})
            stability = boundary.loc[
                boundary["murmur_phase"].astype(str).eq(phase)
                & boundary["feature"].eq(feature),
                ["candidate_id", "patient_id", "boundary_change_10pct_median_absolute"],
            ].merge(candidate_qa, on=["candidate_id", "patient_id"], how="left")
            stability.rename(
                columns={"boundary_change_10pct_median_absolute": "outcome_value"},
                inplace=True,
            )
            for outcome, table in (
                ("nominal_feature_value", nominal),
                ("boundary_change_10pct_median_absolute", stability),
            ):
                for qa in CONTINUOUS_QA_COLUMNS:
                    result = spearman_with_cluster_interval(
                        table,
                        qa,
                        "outcome_value",
                        key=(phase, feature, outcome, qa),
                        config=config,
                    )
                    rows.append(
                        {
                            "murmur_phase": phase,
                            "feature": feature,
                            "outcome": outcome,
                            "qa_variable": qa,
                            **result,
                        }
                    )
    return pd.DataFrame(rows)


def _quartile_labels(values: pd.Series) -> tuple[pd.Series, list[float]]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    output = pd.Series(pd.NA, index=values.index, dtype="object")
    if not len(valid):
        return output, []
    try:
        bins = pd.qcut(valid, 4, labels=False, duplicates="drop")
    except ValueError:
        return output, []
    output.loc[valid.index] = [f"Q{int(value) + 1}" for value in bins]
    cut_points = sorted(set(float(value) for value in valid.quantile([0, .25, .5, .75, 1])))
    return output, cut_points


def qa_quartile_summary(
    candidates: pd.DataFrame, boundary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_qa = candidates[["candidate_id", *CONTINUOUS_QA_COLUMNS]]
    for phase in sorted(candidates["murmur_phase"].dropna().astype(str).unique()):
        for qa in CONTINUOUS_QA_COLUMNS:
            for feature in TIER_A_FEATURE_COLUMNS:
                nominal = candidates.loc[
                    candidates["murmur_phase"].astype(str).eq(phase)
                    & candidates[f"{feature}_valid"].astype(bool),
                    ["candidate_id", qa, feature],
                ].rename(columns={feature: "outcome_value"})
                stability = boundary.loc[
                    boundary["murmur_phase"].astype(str).eq(phase)
                    & boundary["feature"].eq(feature),
                    ["candidate_id", "boundary_change_10pct_median_absolute"],
                ].merge(candidate_qa, on="candidate_id", how="left")
                stability.rename(
                    columns={"boundary_change_10pct_median_absolute": "outcome_value"},
                    inplace=True,
                )
                for outcome, table in (
                    ("nominal_feature_value", nominal),
                    ("boundary_change_10pct_median_absolute", stability),
                ):
                    quartiles, cut_points = _quartile_labels(table[qa])
                    work = table.assign(qa_quartile=quartiles)
                    for quartile, group in work.dropna(
                        subset=["qa_quartile", "outcome_value"]
                    ).groupby("qa_quartile", sort=True):
                        values = pd.to_numeric(group["outcome_value"], errors="coerce").dropna()
                        rows.append(
                            {
                                "murmur_phase": phase,
                                "qa_variable": qa,
                                "qa_quartile": quartile,
                                "qa_cut_points": ";".join(f"{v:.17g}" for v in cut_points),
                                "feature": feature,
                                "outcome": outcome,
                                "candidate_count": int(len(values)),
                                "qa_min": float(pd.to_numeric(group[qa], errors="coerce").min()),
                                "qa_max": float(pd.to_numeric(group[qa], errors="coerce").max()),
                                "outcome_median": float(values.median()),
                                "outcome_q25": float(values.quantile(0.25)),
                                "outcome_q75": float(values.quantile(0.75)),
                            }
                        )
    return pd.DataFrame(rows)


def secondary_single_qa_models(candidates: pd.DataFrame) -> pd.DataFrame:
    """OLS sensitivity models with robust outcome scaling and cluster SEs."""

    rows: list[dict[str, Any]] = []
    phase_dummies = pd.get_dummies(candidates["murmur_phase"], prefix="phase", dtype=float)
    location_dummies = pd.get_dummies(candidates["location"], prefix="location", dtype=float)
    adjustment = pd.concat(
        [
            pd.to_numeric(candidates["duration_seconds"], errors="coerce").rename("duration_seconds"),
            phase_dummies.iloc[:, 1:],
            location_dummies.iloc[:, 1:],
        ],
        axis=1,
    )
    for feature in TIER_A_FEATURE_COLUMNS:
        raw_y = pd.to_numeric(candidates[feature], errors="coerce")
        valid = candidates[f"{feature}_valid"].astype(bool) & raw_y.notna()
        median = float(raw_y.loc[valid].median()) if valid.any() else np.nan
        iqr = (
            float(raw_y.loc[valid].quantile(.75) - raw_y.loc[valid].quantile(.25))
            if valid.any()
            else np.nan
        )
        for qa in CONTINUOUS_QA_COLUMNS:
            model_adjustment = adjustment.drop(
                columns=["duration_seconds"] if qa == "duration_seconds" else [],
                errors="ignore",
            )
            base = pd.concat(
                [
                    candidates[["patient_id"]],
                    pd.to_numeric(candidates[qa], errors="coerce").rename(qa),
                    model_adjustment,
                ],
                axis=1,
            )
            mask = valid & base.notna().all(axis=1)
            n = int(mask.sum())
            clusters = int(candidates.loc[mask, "patient_id"].nunique())
            result: dict[str, Any] = {
                "feature": feature,
                "qa_variable": qa,
                "candidate_count": n,
                "independent_patient_count": clusters,
                "outcome_median": median,
                "outcome_iqr": iqr,
                "qa_coefficient_per_raw_unit": np.nan,
                "cluster_standard_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "design_rank": 0,
                "parameter_count": 0,
                "model_status": "not_estimable",
            }
            if not np.isfinite(iqr) or iqr <= 0 or n < 5:
                result["model_status"] = "zero_outcome_iqr" if iqr == 0 else "insufficient_rows"
                rows.append(result)
                continue
            work = base.loc[mask]
            covariates = work.drop(columns="patient_id").astype(float)
            # Drop constant adjustment columns, but retain the QA variable so a
            # degenerate QA gets an explicit non-estimable status.
            keep = [qa] + [
                column for column in covariates.columns
                if column != qa and covariates[column].nunique() > 1
            ]
            covariates = covariates.loc[:, list(dict.fromkeys(keep))]
            X = np.column_stack([np.ones(n), covariates.to_numpy(float)])
            y = ((raw_y.loc[mask] - median) / iqr).to_numpy(float)
            rank = int(np.linalg.matrix_rank(X))
            k = int(X.shape[1])
            result.update({"design_rank": rank, "parameter_count": k})
            if rank < k or n <= k or covariates[qa].nunique() < 2 or clusters < 2:
                result["model_status"] = "rank_deficient_or_insufficient_dof"
                rows.append(result)
                continue
            bread = np.linalg.inv(X.T @ X)
            beta = bread @ X.T @ y
            residual = y - X @ beta
            meat = np.zeros((k, k), dtype=float)
            patient_values = work["patient_id"].to_numpy(str)
            for patient in np.unique(patient_values):
                group_mask = patient_values == patient
                score = X[group_mask].T @ residual[group_mask]
                meat += np.outer(score, score)
            correction = (clusters / (clusters - 1)) * ((n - 1) / (n - k))
            covariance = correction * bread @ meat @ bread
            qa_index = 1 + list(covariates.columns).index(qa)
            coefficient = float(beta[qa_index])
            variance = float(covariance[qa_index, qa_index])
            standard_error = np.sqrt(variance) if variance >= 0 else np.nan
            result.update(
                {
                    "qa_coefficient_per_raw_unit": coefficient,
                    "cluster_standard_error": standard_error,
                    "ci_low": coefficient - 1.96 * standard_error,
                    "ci_high": coefficient + 1.96 * standard_error,
                    "model_status": "estimable",
                }
            )
            rows.append(result)
    return pd.DataFrame(rows)


def categorical_qa_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phase in sorted(candidates["murmur_phase"].dropna().astype(str).unique()):
        subset = candidates.loc[candidates["murmur_phase"].astype(str).eq(phase)]
        for column in CATEGORICAL_QA_COLUMNS:
            values = subset[column].fillna("missing").astype(str)
            for level, count in values.value_counts(sort=False, dropna=False).sort_index().items():
                rows.append(
                    {
                        "murmur_phase": phase,
                        "qa_variable": column,
                        "level": level,
                        "candidate_count": int(count),
                        "phase_fraction": float(count / len(subset)),
                    }
                )
    return pd.DataFrame(rows)


def data_dictionary() -> pd.DataFrame:
    definitions = {
        "candidate_quality_status": "Selection-path status: accepted or fallback; not a composite quality score.",
        "reconstruction_error": "Relative reconstruction error from the separation audit; expected near numerical tolerance.",
        "normal_residual_correlation": "Correlation between estimated normal and residual components.",
        "s1_leakage_ratio": "Candidate leakage ratio in the S1 interval.",
        "s2_leakage_ratio": "Candidate leakage ratio in the S2 interval.",
        "murmur_region_energy_retention": "Candidate energy retained in the target murmur region.",
        "outside_murmur_energy_ratio": "Candidate energy outside the nominal murmur activity interval.",
        "systole_candidate_energy_ratio": "Fraction of candidate energy in systole.",
        "diastole_candidate_energy_ratio": "Fraction of candidate energy in diastole.",
        "noise_energy_ratio": "Noise/artifact component energy ratio.",
        "separation_method": "Requested separation method recorded by the pipeline.",
        "selected_method": "Method selected by the separation implementation.",
        "phase_selected_component_count": "Count of components selected for the target phase.",
        "phase_selection_used_fallback": "Whether phase-aware selection used its fallback path.",
        "activity_detection_method": "Nominal activity-boundary detection method; not rerun here.",
        "boundary_stability_status": "Legacy fixed-jitter boundary status retained as QA context.",
        "duration_seconds": "Nominal detected candidate duration in seconds.",
        "murmur_phase": "Target cardiac phase: systole or diastole.",
        "boundary_change_10pct_median_absolute": "Primary median absolute Tier A change across valid, unclipped 10% Phase 1C boundary families.",
        "high_quality_stratum": "Accepted and meeting at least 3 of 4 favorable within-phase median QA conditions.",
    }
    rows = []
    for column in (*CATEGORICAL_QA_COLUMNS, *CONTINUOUS_QA_COLUMNS):
        rows.append(
            {
                "column": column,
                "role": "categorical_qa" if column in CATEGORICAL_QA_COLUMNS else "continuous_qa",
                "definition": definitions[column],
            }
        )
    for feature in TIER_A_FEATURE_COLUMNS:
        rows.append(
            {
                "column": feature,
                "role": "frozen_tier_a_feature",
                "definition": f"Frozen Tier A feature; unit={TIER_A_FEATURE_UNITS[feature]}.",
            }
        )
    for column in ("boundary_change_10pct_median_absolute", "high_quality_stratum"):
        rows.append({"column": column, "role": "derived_phase1d", "definition": definitions[column]})
    return pd.DataFrame(rows).drop_duplicates("column", keep="first")


def _write_quartile_plots(
    candidates: pd.DataFrame, boundary: pd.DataFrame, destination: Path
) -> int:
    destination.mkdir()
    candidate_qa = candidates[["candidate_id", "murmur_phase", *CONTINUOUS_QA_COLUMNS]]
    plot_count = 0
    for qa in CONTINUOUS_QA_COLUMNS:
        for outcome in ("nominal", "boundary_10pct"):
            figure, axes = plt.subplots(7, 2, figsize=(14, 24), constrained_layout=True)
            for axis, feature in zip(axes.flat, TIER_A_FEATURE_COLUMNS, strict=True):
                if outcome == "nominal":
                    table = candidates.loc[
                        candidates[f"{feature}_valid"].astype(bool),
                        ["murmur_phase", qa, feature],
                    ].rename(columns={feature: "value"})
                else:
                    table = boundary.loc[
                        boundary["feature"].eq(feature),
                        ["candidate_id", "boundary_change_10pct_median_absolute"],
                    ].merge(candidate_qa, on="candidate_id", how="left")
                    table.rename(
                        columns={"boundary_change_10pct_median_absolute": "value"},
                        inplace=True,
                    )
                table[qa] = pd.to_numeric(table[qa], errors="coerce")
                table["value"] = pd.to_numeric(table["value"], errors="coerce")
                table.dropna(subset=[qa, "value"], inplace=True)
                for phase, marker in (("systole", "o"), ("diastole", "^")):
                    phase_table = table.loc[table["murmur_phase"].eq(phase)]
                    quartiles, cuts = _quartile_labels(phase_table[qa])
                    colors = quartiles.map({"Q1": "#3b4cc0", "Q2": "#78b4e4", "Q3": "#f6b48f", "Q4": "#b40426"})
                    axis.scatter(
                        phase_table[qa], phase_table["value"], c=colors.fillna("0.5"),
                        marker=marker, s=24, alpha=.8, label=phase if feature == TIER_A_FEATURE_COLUMNS[0] else None,
                    )
                    for cut in cuts[1:-1]:
                        axis.axvline(cut, color="0.75", linewidth=.5, alpha=.5)
                axis.set_title(feature.replace("tier_a_", ""), fontsize=8)
                axis.set_xlabel(qa, fontsize=7)
                axis.set_ylabel("feature" if outcome == "nominal" else "median |10% change|", fontsize=7)
                axis.tick_params(labelsize=6)
            figure.suptitle(
                f"Phase 1D: {outcome.replace('_', ' ')} across {qa} quartiles\n"
                "raw QA on x-axis; circle=systole, triangle=diastole; colors Q1 to Q4",
                fontsize=12,
            )
            figure.savefig(destination / f"{qa}__{outcome}.png", dpi=130)
            plt.close(figure)
            plot_count += 1
    return plot_count


def run_candidate_quality_audit(
    *,
    run_name: str,
    source_phase1b: Path = DEFAULT_PHASE1B_RUN,
    source_phase1c: Path = DEFAULT_PHASE1C_RUN,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config: CandidateQualityConfig = CandidateQualityConfig(),
) -> Path:
    """Run Phase 1D once, refusing to overwrite any prior artifact directory."""

    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one non-empty path component")
    destination = Path(output_root) / run_name
    if destination.exists():
        raise FileExistsError(f"Phase 1D output already exists and will not be overwritten: {destination}")

    phase1b, phase1c, source_info = load_frozen_sources(source_phase1b, source_phase1c)
    candidates, thresholds = prepare_analysis_candidates(phase1b)
    boundary = summarize_ten_percent_boundary_changes(phase1c)
    destination.mkdir(parents=True)

    feature_status = summarize_feature_status(candidates, boundary)
    contrasts = status_contrasts(candidates, boundary, config)
    associations = qa_associations(candidates, boundary, config)
    quartiles = qa_quartile_summary(candidates, boundary)
    models = secondary_single_qa_models(candidates)
    categorical = categorical_qa_summary(candidates)
    dictionary = data_dictionary()

    candidates.to_csv(destination / "analysis_candidate_table.csv", index=False)
    boundary.to_csv(destination / "boundary_change_10pct_by_candidate.csv", index=False)
    feature_status.to_csv(destination / "feature_status_summary.csv", index=False)
    contrasts.to_csv(destination / "accepted_fallback_contrasts.csv", index=False)
    associations.to_csv(destination / "qa_spearman_associations.csv", index=False)
    quartiles.to_csv(destination / "qa_quartile_summary.csv", index=False)
    models.to_csv(destination / "secondary_single_qa_models.csv", index=False)
    categorical.to_csv(destination / "categorical_qa_summary.csv", index=False)
    thresholds.to_csv(destination / "high_quality_thresholds.csv", index=False)
    dictionary.to_csv(destination / "data_dictionary.csv", index=False)
    plot_count = _write_quartile_plots(candidates, boundary, destination / "qa_quartile_plots")

    scope = {
        "included": [
            "accepted-versus-fallback distributions and coverage by phase",
            "accepted-minus-fallback median differences and Cliff delta with patient-cluster percentile intervals",
            "within-phase QA associations with nominal features and median absolute 10% boundary changes",
            "raw-QA quartile summaries and plots",
            "secondary one-QA-at-a-time adjusted sensitivity models",
            "a descriptor-blind prespecified high-quality stratum",
        ],
        "excluded": [
            "construct or descriptor association analysis",
            "classification, feature selection, disease, causal, or diagnostic inference",
            "rerunning or changing separation, selection, fallback, activity detection, boundaries, or Tier A formulas",
        ],
        "high_quality_rule": "Within each phase, accepted and at least 3 of 4: outside energy <= median, retention >= median, maximum S1/S2 leakage <= median, noise ratio <= median.",
        "boundary_outcome": "Primary per-candidate-feature median absolute change across valid, unclipped 10% Phase 1C families; all-effective medians plus clipped and invalid-family counts are retained as secondary diagnostics.",
    }
    (destination / "analysis_scope.json").write_text(json.dumps(scope, indent=2), encoding="utf-8")

    manifest = {
        "audit_tool_version": AUDIT_TOOL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "run_name": run_name,
        **source_info,
        "config": asdict(config),
        "candidate_count": int(len(candidates)),
        "patient_count": int(candidates["patient_id"].nunique()),
        "systolic_candidate_count": int(candidates["murmur_phase"].eq("systole").sum()),
        "diastolic_candidate_count": int(candidates["murmur_phase"].eq("diastole").sum()),
        "accepted_candidate_count": int(candidates["candidate_quality_status"].eq("accepted").sum()),
        "fallback_candidate_count": int(candidates["candidate_quality_status"].eq("fallback").sum()),
        "high_quality_candidate_count": int(candidates["high_quality_stratum"].sum()),
        "tier_a_feature_count": len(TIER_A_FEATURE_COLUMNS),
        "continuous_qa_variable_count": len(CONTINUOUS_QA_COLUMNS),
        "boundary_10pct_candidate_feature_count": int(len(boundary)),
        "quartile_plot_count": plot_count,
        "descriptor_columns_exported": 0,
        "scope_statement": "Separation/candidate-quality sensitivity only; no construct validation or classification.",
        "processing_statement": "Read only frozen Phase 1B/1C tables; did not rerun or change the signal pipeline.",
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase1b-run", type=Path, default=DEFAULT_PHASE1B_RUN)
    parser.add_argument("--phase1c-run", type=Path, default=DEFAULT_PHASE1C_RUN)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    args = parser.parse_args(argv)
    config = CandidateQualityConfig(
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    destination = run_candidate_quality_audit(
        run_name=args.run_name,
        source_phase1b=args.phase1b_run,
        source_phase1c=args.phase1c_run,
        config=config,
    )
    print(f"Task 5 Phase 1D candidate-quality audit written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
