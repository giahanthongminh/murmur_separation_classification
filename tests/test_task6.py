"""Focused leakage, scope, aggregation, and provenance tests for Task 6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.task6.baseline import fit_feature_pipeline
from src.task6.common import (
    CORE_FEATURES,
    aggregate_cycle_probabilities,
    assert_patient_disjoint,
    eligible_patient_labels,
    make_outer_folds,
    sha256_file,
    validate_model_inputs,
    verify_manifest_artifacts,
)
from src.task6.prepare_cache import select_cache_candidates


def _metadata() -> pd.DataFrame:
    rows = []
    for index in range(20):
        rows.append({
            "Patient ID": f"p{index:02d}",
            "Murmur": "Present" if index < 18 else "Absent",
            "Outcome": "Normal" if index % 3 == 0 else "Abnormal",
        })
    return pd.DataFrame(rows)


def test_label_filter_is_patient_level_murmur_present_and_valid_outcome_only() -> None:
    labels = eligible_patient_labels(_metadata())
    assert len(labels) == 18
    assert set(labels["clinical_outcome"]) == {"Normal", "Abnormal"}
    assert labels["patient_id"].is_unique

    invalid = _metadata()
    invalid.loc[0, "Outcome"] = "Unknown"
    with pytest.raises(ValueError, match="invalid clinical Outcome"):
        eligible_patient_labels(invalid)


def test_patient_folds_are_deterministic_stratified_and_disjoint() -> None:
    labels = eligible_patient_labels(_metadata())
    first = make_outer_folds(labels, n_splits=3, seed=7)
    second = make_outer_folds(labels.sample(frac=1, random_state=2), n_splits=3, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["patient_id"].is_unique
    assert set(first["outer_fold"]) == {0, 1, 2}
    for fold in range(3):
        train = first.loc[first["outer_fold"].ne(fold), "patient_id"]
        test = first.loc[first["outer_fold"].eq(fold), "patient_id"]
        assert_patient_disjoint(train, test)
    with pytest.raises(ValueError, match="patient leakage"):
        assert_patient_disjoint(["p1", "p2"], ["p2"])


def test_preprocessing_is_fitted_only_on_training_patients() -> None:
    feature = CORE_FEATURES[0]
    train = pd.DataFrame({
        "patient_id": ["train1", "train2", "train3"],
        feature: [1.0, np.nan, 3.0],
    })
    test = pd.DataFrame({"patient_id": ["test"], feature: [1_000_000.0]})
    x_train, x_test, _, audit = fit_feature_pipeline(train, test, (feature,))
    assert audit["imputer_medians"] == [2.0]
    assert audit["training_patient_ids"] == ["train1", "train2", "train3"]
    assert audit["test_patient_ids"] == ["test"]
    assert x_train.mean() == pytest.approx(0.0)
    assert x_test.item() > 1000  # test outlier did not alter training mean/scale


def test_cycle_probabilities_are_aggregated_before_patient_metrics() -> None:
    cycles = pd.DataFrame({
        "patient_id": ["a", "a", "a", "b"],
        "clinical_outcome": ["Normal", "Normal", "Normal", "Abnormal"],
        "probability_abnormal": [.1, .3, .8, .9],
    })
    patients = aggregate_cycle_probabilities(cycles).set_index("patient_id")
    assert len(patients) == 2
    assert patients.loc["a", "probability_abnormal"] == pytest.approx(.4)
    assert patients.loc["a", "contributing_prediction_count"] == 3
    assert patients.loc["a", "predicted_outcome"] == "Normal"


def test_manifest_artifact_hashes_detect_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("frozen", encoding="utf-8")
    manifest = {"artifact_sha256": {artifact.name: sha256_file(artifact)}}
    verify_manifest_artifacts(tmp_path, manifest)
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest_artifacts(tmp_path, manifest)


def test_prohibited_columns_can_never_be_model_inputs() -> None:
    assert validate_model_inputs(CORE_FEATURES) == CORE_FEATURES
    for prohibited in ("clinical_outcome", "patient_id", "recording_id", "location", "candidate_quality_status"):
        with pytest.raises(ValueError, match="prohibited"):
            validate_model_inputs((CORE_FEATURES[0], prohibited))


def test_cache_scope_keeps_all_qa_statuses_but_only_eligible_systolic_rows() -> None:
    labels = pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "clinical_outcome": ["Normal", "Abnormal"],
    })
    rows = []
    for patient, outcome, phase, status in (
        ("p1", "Normal", "systole", "accepted"),
        ("p1", "Normal", "systole", "fallback"),
        ("p1", "Normal", "diastole", "accepted"),
        ("p2", "Abnormal", "systole", "fallback"),
        ("p3", "Abnormal", "systole", "accepted"),
    ):
        number = len(rows)
        rows.append({
            "candidate_id": f"c{number}", "patient_id": patient,
            "recording_id": f"r{number}", "location": "AV", "cycle_index": 0,
            "murmur_phase": phase, "patient_murmur_label": "Present",
            "clinical_outcome": outcome, "candidate_quality_status": status,
            "phase_selection_used_fallback": status == "fallback", "config_hash": "frozen",
        })
    selected = select_cache_candidates(pd.DataFrame(rows), labels)
    assert set(selected["patient_id"]) == {"p1", "p2"}
    assert set(selected["murmur_phase"]) == {"systole"}
    assert set(selected["candidate_quality_status"]) == {"accepted", "fallback"}
    assert len(selected) == 3

