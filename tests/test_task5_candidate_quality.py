"""Tests for Task 5 Phase 1D candidate-quality sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.task5_candidate_quality import (
    CATEGORICAL_QA_COLUMNS,
    CONTINUOUS_QA_COLUMNS,
    CandidateQualityConfig,
    accepted_fallback_contrast,
    assign_high_quality_stratum,
    prepare_analysis_candidates,
    secondary_single_qa_models,
    spearman_with_cluster_interval,
    summarize_ten_percent_boundary_changes,
)
from src.separation.tier_a_features import TIER_A_FEATURE_COLUMNS


def _candidate(number: int, *, phase: str = "systole", status: str = "accepted") -> dict[str, object]:
    row: dict[str, object] = {
        "selection_rank": number,
        "candidate_id": f"p{number}:r{number}:cycle0:{phase}",
        "patient_id": f"p{number}",
        "recording_id": f"r{number}",
        "location": ("AV", "MV", "PV")[number % 3],
        "cycle_index": 0,
        "murmur_phase": phase,
        "candidate_quality_status": status,
        "separation_method": "zcr",
        "selected_method": "zcr",
        "phase_selection_used_fallback": status == "fallback",
        "activity_detection_method": "adaptive_envelope",
        "boundary_stability_status": "stable",
        "reconstruction_error": number * 1e-15,
        "normal_residual_correlation": number / 100,
        "s1_leakage_ratio": number / 100,
        "s2_leakage_ratio": number / 200,
        "murmur_region_energy_retention": 1 - number / 100,
        "outside_murmur_energy_ratio": number / 100,
        "systole_candidate_energy_ratio": .6,
        "diastole_candidate_energy_ratio": .4,
        "noise_energy_ratio": number / 300,
        "phase_selected_component_count": number,
        "duration_seconds": .05 + number / 1000,
        "expert_shape_label": "must not be exported",
    }
    for index, feature in enumerate(TIER_A_FEATURE_COLUMNS):
        row[feature] = float(number + index)
        row[f"{feature}_valid"] = True
        row[f"{feature}_invalid_reason"] = None
    return row


def test_analysis_table_is_descriptor_blind_and_high_quality_rule_is_deterministic() -> None:
    source = pd.DataFrame(
        [_candidate(number, status="fallback" if number in {4, 8} else "accepted") for number in range(1, 9)]
    )
    first, thresholds = prepare_analysis_candidates(source)
    second, second_thresholds = prepare_analysis_candidates(
        source.sample(frac=1, random_state=42)
    )
    assert "expert_shape_label" not in first.columns
    assert first.sort_values("candidate_id")["high_quality_stratum"].tolist() == second.sort_values(
        "candidate_id"
    )["high_quality_stratum"].tolist()
    pd.testing.assert_frame_equal(thresholds, second_thresholds)
    assert set(thresholds["qa_variable"]) == {
        "outside_murmur_energy_ratio",
        "murmur_region_energy_retention",
        "maximum_s1_s2_leakage_ratio",
        "noise_energy_ratio",
    }


def test_high_quality_requires_accepted_and_three_favorable_conditions() -> None:
    candidates = pd.DataFrame(
        [_candidate(number, status="fallback" if number == 1 else "accepted") for number in range(1, 7)]
    )
    assigned, _ = assign_high_quality_stratum(candidates)
    fallback = assigned.loc[assigned["candidate_quality_status"].eq("fallback")].iloc[0]
    assert not bool(fallback["high_quality_stratum"])
    assert assigned.loc[assigned["high_quality_stratum"], "high_quality_favorable_condition_count"].ge(3).all()


def test_ten_percent_boundary_outcome_uses_exactly_eight_families() -> None:
    feature = TIER_A_FEATURE_COLUMNS[0]
    rows = []
    for family_index in range(8):
        rows.append(
            {
                "candidate_id": "p1:r1:cycle0:systole",
                "patient_id": "p1",
                "murmur_phase": "systole",
                "candidate_quality_status": "accepted",
                "feature": feature,
                "magnitude_fraction": .10,
                "family": f"family_{family_index}",
                "baseline_value": .5,
                "baseline_valid": True,
                "absolute_change": float(family_index + 1),
                "relative_change": float(family_index + 1) / .5,
                "validity_transition": "valid_to_invalid" if family_index == 7 else "valid_to_valid",
                "clipped": family_index == 6,
            }
        )
        rows.append({**rows[-1], "magnitude_fraction": .05, "absolute_change": 999.0})
    summary = summarize_ten_percent_boundary_changes(pd.DataFrame(rows))
    assert len(summary) == 1
    result = summary.iloc[0]
    assert result["boundary_change_10pct_family_count"] == 8
    assert result["boundary_change_10pct_median_absolute"] == pytest.approx(3.5)
    assert result["boundary_change_10pct_max_absolute"] == pytest.approx(6.0)
    assert result["boundary_change_10pct_primary_unclipped_valid_count"] == 6
    assert result["boundary_change_10pct_all_effective_median_absolute"] == pytest.approx(4.0)
    assert result["boundary_change_10pct_valid_to_invalid_count"] == 1
    assert result["boundary_change_10pct_clipped_count"] == 1


def test_status_contrast_sign_and_patient_cluster_interval_are_reproducible() -> None:
    table = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(8)],
            "murmur_phase": ["systole"] * 8,
            "candidate_quality_status": ["accepted"] * 4 + ["fallback"] * 4,
            "value": [8.0, 9.0, 10.0, 11.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    config = CandidateQualityConfig(bootstrap_replicates=200, bootstrap_seed=7)
    first = accepted_fallback_contrast(
        table,
        "value",
        phase="systole",
        feature="fixture",
        outcome="nominal_feature_value",
        config=config,
    )
    second = accepted_fallback_contrast(
        table,
        "value",
        phase="systole",
        feature="fixture",
        outcome="nominal_feature_value",
        config=config,
    )
    assert first == second
    assert first["accepted_minus_fallback_median_difference"] == pytest.approx(7.0)
    assert first["cliffs_delta_accepted_over_fallback"] == pytest.approx(1.0)
    assert first["median_difference_ci_low"] > 0


def test_spearman_cluster_interval_and_secondary_models_are_well_formed() -> None:
    correlation_table = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(10)],
            "qa": np.arange(10, dtype=float),
            "outcome": np.arange(10, dtype=float) ** 2,
        }
    )
    config = CandidateQualityConfig(bootstrap_replicates=100, bootstrap_seed=11)
    result = spearman_with_cluster_interval(
        correlation_table,
        "qa",
        "outcome",
        key=("fixture",),
        config=config,
    )
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["independent_patient_count"] == 10
    assert result["ci_low"] == pytest.approx(1.0)

    candidates = pd.DataFrame([_candidate(number) for number in range(1, 16)])
    models = secondary_single_qa_models(candidates)
    assert len(models) == len(TIER_A_FEATURE_COLUMNS) * len(CONTINUOUS_QA_COLUMNS)
    assert set(models["model_status"]) <= {
        "estimable",
        "zero_outcome_iqr",
        "insufficient_rows",
        "rank_deficient_or_insufficient_dof",
    }
    assert set(CATEGORICAL_QA_COLUMNS).issubset(candidates.columns)
