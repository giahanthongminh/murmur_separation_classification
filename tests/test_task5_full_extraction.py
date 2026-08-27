"""Tests for the versioned Task 5 Phase 2A extraction audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.task5_full_extraction import (
    KNOWN_UNUSABLE_ANNOTATIONS,
    FullExtractionConfig,
    aggregate_features_long,
    build_dataset_inventory,
    build_denominator_flow,
    build_feature_table,
    summarize_feature_coverage,
    summarize_invalid_reasons,
    validate_preregistered_snapshot,
)
from src.separation.tier_a_features import (
    TIER_A_EXPORT_COLUMNS,
    TIER_A_FEATURE_COLUMNS,
)


def _validation() -> dict[str, object]:
    invalid = [
        {"file": filename, "detail": "known fixture failure"}
        for filename in KNOWN_UNUSABLE_ANNOTATIONS
    ]
    return {
        "wav_recordings": 3163,
        "tsv_annotations": 3163,
        "patients": 942,
        "wav_missing_tsv": [],
        "tsv_missing_wav": [],
        "sample_rate_counts": {"4000": 3163},
        "invalid_annotations": invalid,
        "errors": [{"code": "invalid_annotations", "detail": str(invalid)}],
    }


def _row(number: int, *, phase: str = "systole", status: str = "accepted") -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": f"p{number // 2}",
        "recording_id": f"r{number // 2}",
        "location": "AV",
        "cycle_index": number % 2,
        "murmur_phase": phase,
        "candidate_quality_status": status,
        "sample_rate": 4000,
        "config_hash": "fixture",
        "onset_sample": 2,
        "offset_sample": 8,
        "duration_seconds": .1,
    }
    for column in TIER_A_EXPORT_COLUMNS:
        row.setdefault(column, 1)
    for index, feature in enumerate(TIER_A_FEATURE_COLUMNS):
        row[feature] = float(number + index)
        row[f"{feature}_valid"] = True
        row[f"{feature}_invalid_reason"] = None
    return row


def test_preregistered_snapshot_accepts_only_exact_known_exception_set() -> None:
    excluded = validate_preregistered_snapshot(_validation())
    assert excluded == sorted(Path(name).stem for name in KNOWN_UNUSABLE_ANNOTATIONS)

    changed = _validation()
    changed["invalid_annotations"] = list(changed["invalid_annotations"])[:-1]
    with pytest.raises(ValueError, match="annotation set changed"):
        validate_preregistered_snapshot(changed)

    changed = _validation()
    changed["wav_missing_tsv"] = ["missing"]
    with pytest.raises(ValueError, match="exact WAV/TSV"):
        validate_preregistered_snapshot(changed)


def test_inventory_reports_exact_pairs_and_sizes(tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(b"wav")
    (tmp_path / "a.tsv").write_bytes(b"tsv-data")
    (tmp_path / "b.wav").write_bytes(b"orphan")
    inventory = build_dataset_inventory(tmp_path).set_index("recording_id")
    assert bool(inventory.loc["a", "exact_pair"])
    assert inventory.loc["a", "wav_size_bytes"] == 3
    assert inventory.loc["a", "tsv_size_bytes"] == 8
    assert not bool(inventory.loc["b", "exact_pair"])


def test_feature_table_has_unique_stable_candidate_ids() -> None:
    summary = pd.DataFrame([_row(2), _row(1), _row(3, phase="diastole")])
    table = build_feature_table(summary)
    assert table["candidate_id"].is_unique
    assert table["candidate_id"].tolist() == [
        "p0:r0:cycle1:systole",
        "p1:r1:cycle0:systole",
        "p1:r1:cycle1:diastole",
    ]
    duplicate = pd.concat([summary, summary.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="identifiers"):
        build_feature_table(duplicate)


def test_denominator_coverage_and_invalid_reasons_preserve_all_attempts() -> None:
    rows = [_row(0), _row(1, status="fallback"), _row(2, phase="diastole")]
    feature = TIER_A_FEATURE_COLUMNS[-1]
    rows[1][f"{feature}_valid"] = False
    rows[1][f"{feature}_invalid_reason"] = "insufficient_ridge_frames"
    table = build_feature_table(pd.DataFrame(rows))
    skipped = pd.DataFrame([{"recording_id": "bad", "reason": "fixture"}])
    flow = build_denominator_flow(
        table, skipped, _validation(), selected_recordings=3156
    )
    selected = flow.loc[
        flow["stage"].eq("metadata_matched_valid_recordings_selected"), "count"
    ].item()
    assert selected == 3156
    assert flow.loc[flow["stage"].eq("candidate_rows") & flow["phase"].eq("all"), "count"].item() == 3

    coverage = summarize_feature_coverage(table)
    overall = coverage.loc[
        coverage["murmur_phase"].eq("all") & coverage["feature"].eq(feature)
    ].iloc[0]
    assert overall["attempted_count"] == 3
    assert overall["valid_count"] == 2
    assert overall["coverage"] == pytest.approx(2 / 3)

    reasons = summarize_invalid_reasons(table)
    assert reasons.loc[reasons["feature"].eq(feature), "invalid_reason"].tolist() == [
        "insufficient_ridge_frames"
    ]


def test_recording_and_patient_location_aggregates_use_cycle_medians() -> None:
    table = build_feature_table(pd.DataFrame([_row(0), _row(1), _row(2)]))
    feature = TIER_A_FEATURE_COLUMNS[0]
    recording = aggregate_features_long(table, level="recording")
    first = recording.loc[
        recording["recording_id"].eq("r0") & recording["feature"].eq(feature)
    ].iloc[0]
    assert first["attempted_cycle_count"] == 2
    assert first["valid_cycle_count"] == 2
    assert first["median"] == pytest.approx(.5)

    patient = aggregate_features_long(table, level="patient_location")
    assert set(patient["aggregation_level"]) == {"patient_location"}


def test_full_extraction_configuration_rejects_method_drift() -> None:
    with pytest.raises(ValueError, match="method"):
        FullExtractionConfig(method="auto")
    assert FullExtractionConfig().cycles_per_recording == 0
