"""Tests for Task 5 Phase 2B construct validation."""

from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd

from src.evaluation.task5_construct_validation import (
    ConstructValidationConfig,
    DURATION,
    MIDPOINT,
    _holm_adjust,
    build_primary_patient_table,
    prepare_candidate_strata,
    run_construct_validation,
    run_planned_contrasts,
)
from src.separation.tier_a_features import FEATURE_SCHEMA_VERSION, TIER_A_FEATURE_COLUMNS


def _hash(path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _candidate(
    patient: int,
    *,
    recording_suffix: str = "AV",
    cycle: int = 0,
    location: str = "AV",
    status: str = "accepted",
    timing: str = "Holosystolic",
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": f"p{patient}:p{patient}_{recording_suffix}:cycle{cycle}:systole",
        "patient_id": f"p{patient}",
        "recording_id": f"p{patient}_{recording_suffix}",
        "location": location,
        "cycle_index": cycle,
        "murmur_phase": "systole",
        "patient_murmur_label": "Present",
        "candidate_quality_status": status,
        "s1_leakage_ratio": patient / 100,
        "s2_leakage_ratio": patient / 200,
        "murmur_region_energy_retention": 1 - patient / 100,
        "outside_murmur_energy_ratio": patient / 100,
        "noise_energy_ratio": patient / 300,
        "expert_timing_label": timing,
        "expert_shape_label": ("Plateau", "Diamond", "Decrescendo")[patient % 3],
        "expert_pitch_label": ("Low", "Medium", "High")[patient % 3],
        "expert_grading_label": ("I/VI", "II/VI", "III/VI")[patient % 3],
        "expert_quality_label": "Harsh" if patient % 2 else "Blowing",
    }
    for index, feature in enumerate(TIER_A_FEATURE_COLUMNS):
        row[feature] = float(patient + index + cycle)
        row[f"{feature}_valid"] = True
        row[f"{feature}_invalid_reason"] = None
    return row


def _metadata(patients: range, *, include_wrong_site: bool = False) -> pd.DataFrame:
    rows = [
        {"Patient ID": f"p{patient}", "Murmur": "Present", "Most audible location": "AV"}
        for patient in patients
    ]
    if include_wrong_site:
        rows[0]["Most audible location"] = "MV"
    return pd.DataFrame(rows)


def test_holm_adjustment_is_monotone_in_rank_and_preserves_missing() -> None:
    values = pd.Series([.01, .04, .03, np.nan])
    adjusted = _holm_adjust(values)
    assert adjusted.iloc[:3].tolist() == [0.03, 0.06, 0.06]
    assert np.isnan(adjusted.iloc[3])


def test_primary_table_uses_most_audible_site_and_median_of_recording_medians() -> None:
    rows = [
        _candidate(1, cycle=0),
        _candidate(1, cycle=1),
        _candidate(1, recording_suffix="AV_2", cycle=0),
        _candidate(2, location="MV", recording_suffix="MV"),
    ]
    rows[0][DURATION] = 1.0
    rows[1][DURATION] = 9.0
    rows[2][DURATION] = 20.0
    candidates, _ = prepare_candidate_strata(pd.DataFrame(rows))
    primary, _ = build_primary_patient_table(candidates, _metadata(range(1, 3)))
    value = primary.loc[
        primary["patient_id"].eq("p1")
        & primary["stratum"].eq("all_valid")
        & primary["feature"].eq(DURATION),
        "value",
    ].item()
    # Recording medians are 5 and 20; the patient value is their median, 12.5.
    assert value == 12.5
    assert "p2" not in set(primary["patient_id"])


def test_high_quality_definition_is_descriptor_blind_and_requires_accepted() -> None:
    source = pd.DataFrame([
        _candidate(i, status="fallback" if i == 1 else "accepted", timing="Holosystolic" if i < 4 else "Early-systolic")
        for i in range(1, 7)
    ])
    first, thresholds = prepare_candidate_strata(source)
    changed = source.copy()
    changed["expert_timing_label"] = "Mid-systolic"
    second, second_thresholds = prepare_candidate_strata(changed)
    assert first["high_quality"].tolist() == second["high_quality"].tolist()
    pd.testing.assert_frame_equal(thresholds, second_thresholds)
    assert not bool(first.loc[first["candidate_quality_status"].eq("fallback"), "high_quality"].iloc[0])


def test_planned_timing_contrasts_have_expected_sign_and_are_reproducible() -> None:
    rows = []
    for patient in range(1, 17):
        timing = "Holosystolic" if patient <= 8 else ("Early-systolic" if patient <= 12 else "Mid-systolic")
        row = _candidate(patient, timing=timing)
        row[DURATION] = 0.9 if timing == "Holosystolic" else 0.4
        row[MIDPOINT] = 0.2 if timing == "Early-systolic" else (0.7 if timing == "Mid-systolic" else 0.5)
        rows.append(row)
    candidates, _ = prepare_candidate_strata(pd.DataFrame(rows))
    primary, _ = build_primary_patient_table(candidates, _metadata(range(1, 17)))
    config = ConstructValidationConfig(bootstrap_replicates=30, bootstrap_seed=4)
    first, _ = run_planned_contrasts(primary, config)
    second, _ = run_planned_contrasts(primary, config)
    timing = first.loc[first["construct_family"].eq("timing") & first["stratum"].eq("all_valid")]
    assert timing.loc[timing["feature"].eq(DURATION), "estimate"].item() > 0
    assert timing.loc[timing["feature"].eq(MIDPOINT), "estimate"].item() > 0
    pd.testing.assert_frame_equal(first, second)


def test_runner_writes_hashed_artifacts_and_excludes_outcome(tmp_path) -> None:
    source = tmp_path / "phase2a"
    source.mkdir()
    table = pd.DataFrame([_candidate(i) for i in range(1, 9)])
    table["clinical_outcome"] = "must_not_be_used"
    feature_path = source / "task5_feature_table.csv"
    table.to_csv(feature_path, index=False)
    metadata_path = tmp_path / "training_data.csv"
    metadata = _metadata(range(1, 9))
    metadata["Outcome"] = "Abnormal"
    metadata.to_csv(metadata_path, index=False)
    manifest = {
        "status": "complete",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "git_commit": "source-commit",
        "metadata_sha256": _hash(metadata_path),
        "artifact_sha256": {"task5_feature_table.csv": _hash(feature_path)},
    }
    (source / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    destination = run_construct_validation(
        source_run=source,
        run_name="construct-test",
        output_root=tmp_path / "outputs",
        metadata_path=metadata_path,
        config=ConstructValidationConfig(bootstrap_replicates=10),
    )
    result_manifest = json.loads((destination / "run_manifest.json").read_text())
    assert result_manifest["outcome_used"] is False
    assert result_manifest["source_git_commit"] == "source-commit"
    primary = pd.read_csv(destination / "primary_patient_feature_long.csv")
    assert "clinical_outcome" not in primary.columns
    for filename, expected in result_manifest["artifact_sha256"].items():
        assert _hash(destination / filename) == expected
