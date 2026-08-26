from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.task5_candidate_audit import (
    AUDIT_MANIFEST_REQUIRED_COLUMNS,
    build_candidate_manifest,
    prepare_candidate_table,
    select_audit_candidates,
    select_screening_recordings,
)
from src.separation.tier_a_features import (
    TIER_A_DIAGNOSTIC_COLUMNS,
    TIER_A_FEATURE_COLUMNS,
)


def _candidate(
    number: int,
    *,
    phase: str = "systole",
    status: str = "accepted",
    shape: str = "Plateau",
    timing: str = "Early-systolic",
    pitch: str = "Low",
    grade: str = "I/VI",
    samples: int = 180,
    leakage: float = 0.1,
    retention: float = 0.5,
    invalid_feature: str | None = None,
    invalid_reason: str = "insufficient_ridge_frames",
) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": f"p{number:02d}",
        "recording_id": f"p{number:02d}_AV",
        "location": "AV",
        "cycle_index": 0,
        "murmur_phase": phase,
        "candidate_quality_status": status,
        "expert_timing_label": timing,
        "expert_shape_label": shape,
        "expert_pitch_label": pitch,
        "expert_grading_label": grade,
        "s1_leakage_ratio": leakage,
        "s2_leakage_ratio": leakage / 2,
        "murmur_region_energy_retention": retention,
        "feature_schema_version": "task5-tier-a-v1.0.0",
        "tier_a_candidate_sample_count": samples,
    }
    for column in TIER_A_DIAGNOSTIC_COLUMNS:
        row.setdefault(column, 1)
    for index, feature in enumerate(TIER_A_FEATURE_COLUMNS):
        row[feature] = float(index + number)
        row[f"{feature}_valid"] = feature != invalid_feature
        row[f"{feature}_invalid_reason"] = (
            invalid_reason if feature == invalid_feature else None
        )
    return row


def test_final_selection_is_reproducible_under_input_shuffle() -> None:
    rows = [
        _candidate(
            number,
            phase="diastole" if number <= 5 else "systole",
            status="fallback" if number % 5 == 0 else "accepted",
            shape=("Plateau", "Diamond", "Crescendo", "Decrescendo")[number % 4],
            leakage=number / 100,
            retention=1 - number / 100,
        )
        for number in range(1, 36)
    ]
    first, first_shortfalls = select_audit_candidates(rows)
    shuffled = pd.DataFrame(rows).sample(frac=1, random_state=18).reset_index(drop=True)
    second, second_shortfalls = select_audit_candidates(shuffled)

    assert first["candidate_id"].tolist() == second["candidate_id"].tolist()
    assert first["selection_reasons"].tolist() == second["selection_reasons"].tolist()
    pd.testing.assert_frame_equal(first_shortfalls, second_shortfalls)


def test_stratum_priority_qa_extremes_and_shortfalls_are_reported() -> None:
    rows = [
        _candidate(1, phase="diastole", leakage=0.10, retention=0.8),
        _candidate(2, phase="systole", status="fallback", leakage=0.20, retention=0.7),
        _candidate(3, phase="systole", leakage=9.00, retention=0.01),
        _candidate(4, phase="systole", leakage=8.00, retention=0.02),
    ]
    selected, shortfalls = select_audit_candidates(rows, target_count=4)

    assert selected.iloc[0]["murmur_phase"] == "diastole"
    assert "status:fallback" in selected.iloc[1]["selection_reasons"]
    high = selected.loc[selected["selection_reasons"].str.contains("high_leakage"), "candidate_id"]
    assert set(high) == {
        "p02:p02_AV:cycle0:systole",
        "p03:p03_AV:cycle0:systole",
        "p04:p04_AV:cycle0:systole",
    }
    fallback = shortfalls[shortfalls["stratum"].eq("status:fallback")].iloc[0]
    assert fallback["requested"] == 5
    assert fallback["eligible"] == 1
    assert fallback["shortfall"] == 4
    assert fallback["shortfall_reason"] == "insufficient_eligible"


def test_manifest_has_frozen_schema_and_stable_order() -> None:
    selected, _ = select_audit_candidates(
        [_candidate(2), _candidate(1, phase="diastole")], target_count=2
    )
    manifest = build_candidate_manifest(selected)

    assert list(manifest.columns[: len(AUDIT_MANIFEST_REQUIRED_COLUMNS)]) == list(
        AUDIT_MANIFEST_REQUIRED_COLUMNS
    )
    assert manifest["selection_rank"].tolist() == [1, 2]
    assert manifest.iloc[0]["patient_id"] == "p01"


def test_identifiers_and_invalid_reasons_are_preserved_verbatim() -> None:
    invalid = TIER_A_FEATURE_COLUMNS[-1]
    row = _candidate(
        7,
        phase="diastole",
        invalid_feature=invalid,
        invalid_reason="insufficient_ridge_frames",
    )
    table = prepare_candidate_table([row])
    manifest = build_candidate_manifest(table.assign(selection_rank=1, selection_reasons="invalid_feature"))

    assert tuple(manifest.loc[0, list(("patient_id", "recording_id", "location", "cycle_index", "murmur_phase"))]) == (
        "p07",
        "p07_AV",
        "AV",
        0,
        "diastole",
    )
    assert manifest.loc[0, f"{invalid}_invalid_reason"] == "insufficient_ridge_frames"
    assert f"{invalid}:insufficient_ridge_frames" in manifest.loc[0, "tier_a_invalid_reasons"]


def test_metadata_screening_is_stable_and_reports_unfillable_strata() -> None:
    recordings = [
        {
            "patient_id": "2",
            "recording_id": "2_MV",
            "location_murmur_label": "Present",
            "systole_shape_label": "Plateau",
            "systole_timing_label": "Early-systolic",
            "systole_pitch_label": "Low",
            "systole_grading_label": "I/VI",
            "diastole_timing_label": None,
        },
        {
            "patient_id": "1",
            "recording_id": "1_PV",
            "location_murmur_label": "Present",
            "systole_shape_label": "Diamond",
            "systole_timing_label": "Holosystolic",
            "systole_pitch_label": "High",
            "systole_grading_label": "III/VI",
            "diastole_timing_label": None,
        },
        {
            "patient_id": "1",
            "recording_id": "1_AV",
            "location_murmur_label": "Present",
            "systole_shape_label": "Diamond",
            "systole_timing_label": "Holosystolic",
            "systole_pitch_label": "High",
            "systole_grading_label": "III/VI",
            "diastole_timing_label": None,
        },
    ]
    first, shortfalls = select_screening_recordings(recordings, target_count=2)
    second, _ = select_screening_recordings(list(reversed(recordings)), target_count=2)

    assert [row["recording_id"] for row in first] == [row["recording_id"] for row in second]
    assert {row["recording_id"] for row in first} == {"1_AV", "2_MV"}
    diastole = shortfalls[shortfalls["stratum"].eq("phase:diastole")].iloc[0]
    assert diastole["eligible"] == 0
    assert diastole["shortfall"] == 5
    assert diastole["shortfall_reason"] == "insufficient_eligible"
