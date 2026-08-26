"""Tests for the deterministic Task 5 Phase 1C boundary audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.task5_boundary_stability import (
    PERTURBATION_FAMILIES,
    PERTURBATION_MAGNITUDES,
    audit_candidate,
    calculate_feature_change,
    generate_boundary_variants,
    perturbation_matrix,
    samples_to_milliseconds,
    summarize_feature_stability,
)
from src.separation.tier_a_features import TIER_A_FEATURE_COLUMNS


def _variants(
    *, onset: int = 10, offset: int = 90, phase_start: int = 0, phase_end: int = 100
):
    return generate_boundary_variants(
        phase_start_sample=phase_start,
        phase_end_sample=phase_end,
        baseline_onset_sample=onset,
        baseline_offset_sample=offset,
    )


def test_frozen_matrix_and_variant_generation_are_deterministic() -> None:
    first = perturbation_matrix()
    second = perturbation_matrix()
    assert first == second
    assert len(first) == len(PERTURBATION_FAMILIES) * len(PERTURBATION_MAGNITUDES) == 24
    assert [row["variant_order"] for row in first] == list(range(1, 25))
    assert [row["perturbation_id"] for row in first[:3]] == [
        "shift_earlier_05pct",
        "shift_earlier_10pct",
        "shift_earlier_20pct",
    ]

    variants = _variants()
    assert [variant.perturbation_id for variant in variants] == [
        row["perturbation_id"] for row in first
    ]
    assert [variant.delta_samples for variant in variants[:3]] == [5, 10, 20]
    assert variants == _variants()


def test_sample_millisecond_conversion_is_exact() -> None:
    assert samples_to_milliseconds(20, 4000) == pytest.approx(5.0)
    assert samples_to_milliseconds(-40, 4000) == pytest.approx(-10.0)
    with pytest.raises(ValueError, match="sample_rate"):
        samples_to_milliseconds(1, 0)


def test_phase_boundary_clipping_records_intended_and_effective_changes() -> None:
    variants = {variant.perturbation_id: variant for variant in _variants()}
    earlier = variants["shift_earlier_20pct"]
    assert earlier.intended_onset_sample == -10
    assert earlier.actual_onset_sample == 0
    assert earlier.intended_onset_displacement_samples == -20
    assert earlier.actual_onset_displacement_samples == -10
    assert earlier.clipped_onset and earlier.clipped
    assert earlier.actual_offset_sample == 70
    assert earlier.variant_valid

    later = variants["offset_later_20pct"]
    assert later.intended_offset_sample == 110
    assert later.actual_offset_sample == 100
    assert later.clipped_offset and later.clipped
    assert later.actual_offset_sample <= 100


def test_reversed_empty_and_invalid_baselines_are_not_repaired() -> None:
    reversed_variant = {
        variant.perturbation_id: variant for variant in _variants(onset=45, offset=55)
    }["contract_20pct"]
    assert reversed_variant.actual_onset_sample == 65
    assert reversed_variant.actual_offset_sample == 35
    assert not reversed_variant.variant_valid
    assert reversed_variant.variant_invalid_reason == "reversed_interval"

    empty_variant = {
        variant.perturbation_id: variant for variant in _variants(onset=45, offset=55)
    }["contract_05pct"]
    assert empty_variant.actual_onset_sample == empty_variant.actual_offset_sample == 50
    assert empty_variant.variant_invalid_reason == "empty_interval"

    with pytest.raises(ValueError, match="baseline interval"):
        _variants(onset=50, offset=50)
    with pytest.raises(ValueError, match="phase interval"):
        _variants(onset=0, offset=1, phase_start=0, phase_end=0)


def test_absolute_relative_changes_and_validity_transitions() -> None:
    changed = calculate_feature_change(4.0, True, 1.0, True)
    assert changed["signed_change"] == pytest.approx(-3.0)
    assert changed["absolute_change"] == pytest.approx(3.0)
    assert changed["relative_change"] == pytest.approx(0.75)
    assert changed["validity_transition"] == "valid_to_valid"

    zero = calculate_feature_change(0.0, True, 2.0, True)
    assert np.isnan(zero["relative_change"])
    assert not zero["relative_change_meaningful"]

    assert calculate_feature_change(1.0, True, np.nan, False)[
        "validity_transition"
    ] == "valid_to_invalid"
    assert calculate_feature_change(np.nan, False, 1.0, True)[
        "validity_transition"
    ] == "invalid_to_valid"
    assert calculate_feature_change(np.nan, False, np.nan, False)[
        "validity_transition"
    ] == "invalid_to_invalid"


def _summary_rows() -> pd.DataFrame:
    records = []
    for feature_index, feature in enumerate(TIER_A_FEATURE_COLUMNS):
        for candidate_index, candidate in enumerate(("a", "b", "c"), start=1):
            for variant_order, change in enumerate((1.0, 2.0, 9.0), start=1):
                records.append(
                    {
                        "candidate_id": candidate,
                        "feature": feature,
                        "baseline_value": float(feature_index + candidate_index),
                        "absolute_change": change + feature_index,
                        "validity_transition": (
                            "valid_to_invalid"
                            if candidate == "c" and variant_order == 3
                            else "valid_to_valid"
                        ),
                        "clipped": variant_order == 3,
                    }
                )
    return pd.DataFrame(records)


def test_summary_statistics_are_robust_ordered_and_reproducible() -> None:
    rows = _summary_rows()
    first = summarize_feature_stability(rows)
    second = summarize_feature_stability(rows.sample(frac=1, random_state=91))
    pd.testing.assert_frame_equal(first, second)
    assert first["feature"].tolist() == list(TIER_A_FEATURE_COLUMNS)
    row = first.iloc[0]
    assert row["median_absolute_change"] == pytest.approx(2.0)
    assert row["absolute_change_q90"] == pytest.approx(9.0)
    assert row["absolute_change_iqr"] == pytest.approx(8.0)
    assert row["validity_transition_count"] == 1
    assert row["candidate_worst_absolute_change"] == pytest.approx(9.0)
    assert row["candidate_worst_id"] == "a"


def test_identifier_and_baseline_invalid_reason_are_preserved(tmp_path: Path) -> None:
    package = tmp_path / "candidates" / "candidate_1"
    package.mkdir(parents=True)
    signal = np.zeros(40, dtype=float)
    signal[3:7] = np.asarray([1.0, -1.0, 0.5, -0.5])
    time = np.arange(40) / 4000
    original = np.sin(2 * np.pi * 100 * time)
    np.save(package / "murmur_candidate.npy", signal)
    np.save(package / "original.npy", original)
    row = pd.Series(
        {
            "selection_rank": 1,
            "candidate_id": "p1:p1_AV:cycle0:systole",
            "patient_id": "p1",
            "recording_id": "p1_AV",
            "location": "AV",
            "cycle_index": 0,
            "murmur_phase": "systole",
            "candidate_quality_status": "fallback",
            "expert_shape_label": "Plateau",
            "systole_relative_start_sample": 0,
            "systole_relative_end_sample": 20,
            "onset_sample": 3,
            "offset_sample": 7,
            "s1_relative_start_sample": 20,
            "s1_relative_end_sample": 30,
            "s2_relative_start_sample": 30,
            "s2_relative_end_sample": 40,
            "sample_rate": 4000,
            "artifact_directory": "candidates/candidate_1",
        }
    )
    feature_rows, variant_rows, _ = audit_candidate(row, source_run=tmp_path)
    features = pd.DataFrame(feature_rows)
    variants = pd.DataFrame(variant_rows)
    assert len(features) == 24 * 14
    assert len(variants) == 24
    assert features["candidate_id"].unique().tolist() == [
        "p1:p1_AV:cycle0:systole"
    ]
    assert features["patient_id"].unique().tolist() == ["p1"]
    assert features["recording_id"].unique().tolist() == ["p1_AV"]
    assert features["location"].unique().tolist() == ["AV"]
    assert features["cycle_index"].unique().tolist() == [0]
    assert features["murmur_phase"].unique().tolist() == ["systole"]
    peak = TIER_A_FEATURE_COLUMNS[3]
    peak_rows = features.loc[features["feature"].eq(peak)]
    assert set(peak_rows["baseline_invalid_reason"]) == {
        "insufficient_envelope_support"
    }
    assert variants.sort_values("variant_order")["perturbation_id"].tolist() == [
        row["perturbation_id"] for row in perturbation_matrix()
    ]
