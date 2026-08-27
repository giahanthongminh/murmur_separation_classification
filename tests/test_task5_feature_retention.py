"""Tests for the frozen Task 5 feature-retention decision."""

from __future__ import annotations

import pandas as pd

from src.evaluation.task5_feature_retention import (
    DECISIONS,
    RetentionConfig,
    build_decision_log,
    cluster_coverage_intervals,
)
from src.separation.tier_a_features import TIER_A_FEATURE_COLUMNS


def test_decision_registry_covers_exactly_tier_a_and_has_expected_categories() -> None:
    assert set(DECISIONS) == set(TIER_A_FEATURE_COLUMNS)
    categories = {value[0] for value in DECISIONS.values()}
    assert categories == {
        "retain_core", "retain_exploratory", "drop_unstable",
        "drop_redundant", "drop_invalid_interpretation", "drop_poor_coverage",
    }
    assert sum(value[0] == "retain_core" for value in DECISIONS.values()) == 3


def test_cluster_coverage_uses_patient_resampling_and_valid_flags(tmp_path) -> None:
    rows = []
    for patient in range(1, 9):
        for phase in ("systole", "diastole"):
            for status in ("accepted", "fallback"):
                row = {
                    "patient_id": f"p{patient}", "murmur_phase": phase,
                    "candidate_quality_status": status,
                }
                for feature in TIER_A_FEATURE_COLUMNS:
                    row[f"{feature}_valid"] = not (feature == TIER_A_FEATURE_COLUMNS[0] and patient == 1)
                rows.append(row)
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = cluster_coverage_intervals(
        path, RetentionConfig(bootstrap_replicates=30, bootstrap_seed=4)
    )
    complete = result.loc[
        result["murmur_phase"].eq("systole")
        & result["candidate_quality_status"].eq("all")
        & result["feature"].eq(TIER_A_FEATURE_COLUMNS[1])
    ].iloc[0]
    incomplete = result.loc[
        result["murmur_phase"].eq("systole")
        & result["candidate_quality_status"].eq("all")
        & result["feature"].eq(TIER_A_FEATURE_COLUMNS[0])
    ].iloc[0]
    assert complete["coverage"] == 1.0
    assert complete["cluster_ci_low"] == 1.0
    assert incomplete["coverage"] == .875


def test_decision_log_records_redundant_pair_and_never_uses_outcome() -> None:
    coverage = pd.DataFrame([
        {
            "murmur_phase": "systole", "candidate_quality_status": "all",
            "feature": feature, "coverage": 1.0,
            "cluster_ci_low": .99, "cluster_ci_high": 1.0,
        }
        for feature in TIER_A_FEATURE_COLUMNS
    ])
    boundary = pd.DataFrame([
        {"feature": feature, "median_iqr_standardized_absolute_change": .1, "valid_to_invalid_count": 0}
        for feature in TIER_A_FEATURE_COLUMNS
    ])
    quality = pd.DataFrame([
        {
            "murmur_phase": "systole", "outcome": "nominal_feature_value",
            "inference_status": "estimable", "feature": feature,
            "spearman_rho": .2,
        }
        for feature in TIER_A_FEATURE_COLUMNS
    ])
    contrasts = pd.DataFrame([
        {
            "feature": feature, "stratum": "all_valid", "comparison": "planned",
            "effect": .1, "holm_p_value": .5,
        }
        for feature in TIER_A_FEATURE_COLUMNS
    ])
    dominant, median = TIER_A_FEATURE_COLUMNS[7], TIER_A_FEATURE_COLUMNS[8]
    instability = pd.DataFrame([{
        "absolute_rho_threshold": .90, "feature_a": dominant, "feature_b": median,
        "same_cluster_fraction": .875, "same_cluster_context_count": 21,
        "context_count": 24,
    }])
    result = build_decision_log(coverage, boundary, quality, contrasts, instability)
    assert "outcome" not in result.columns
    assert result.loc[result["feature"].eq(median), "decision"].item() == "drop_redundant"
    assert result.loc[result["feature"].eq(dominant), "redundancy_cluster_0_90"].notna().item()
