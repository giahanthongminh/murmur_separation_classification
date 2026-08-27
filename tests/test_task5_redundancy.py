"""Tests for Task 5 Phase 2C redundancy analysis."""

from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd

import src.evaluation.task5_redundancy as redundancy
from src.separation.tier_a_features import FEATURE_SCHEMA_VERSION, TIER_A_FEATURE_COLUMNS


FEATURES = TIER_A_FEATURE_COLUMNS[:3]


def _hash(path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _candidate(patient: int, cycle: int = 0) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": f"p{patient}",
        "recording_id": f"p{patient}_AV",
        "location": "AV",
        "murmur_phase": "systole",
        "candidate_quality_status": "accepted" if patient % 3 else "fallback",
    }
    values = (float(patient + cycle), float(2 * (patient + cycle)), float(20 - patient + cycle))
    for feature, value in zip(FEATURES, values, strict=True):
        row[feature] = value
        row[f"{feature}_valid"] = True
    return row


def _primary() -> pd.DataFrame:
    rows = []
    for patient in range(1, 13):
        for stratum in ("all_valid", "accepted_only", "high_quality"):
            for feature, value in zip(FEATURES, (patient, 2 * patient, 20 - patient), strict=True):
                rows.append({
                    "patient_id": f"p{patient}", "location": "AV",
                    "murmur_phase": "systole", "stratum": stratum,
                    "feature": feature, "value": float(value),
                })
    return pd.DataFrame(rows)


def test_recording_aggregation_uses_cycle_medians(monkeypatch) -> None:
    monkeypatch.setattr(redundancy, "TIER_A_FEATURE_COLUMNS", FEATURES)
    table = pd.DataFrame([_candidate(1, 0), _candidate(1, 2), _candidate(2, 0)])
    result = redundancy._aggregate_wide(table, "recording")
    assert result.loc[result["patient_id"].eq("p1"), FEATURES[0]].item() == 2.0


def test_cluster_bootstrap_and_complete_linkage_find_exact_redundancy(monkeypatch) -> None:
    monkeypatch.setattr(redundancy, "TIER_A_FEATURE_COLUMNS", FEATURES)
    frame = pd.DataFrame([_candidate(patient) for patient in range(1, 13)])
    contexts = [(
        {"aggregation_level": "cycle", "murmur_phase": "systole", "analysis_stratum": "all"},
        frame,
    )]
    config = redundancy.RedundancyConfig(bootstrap_replicates=40, bootstrap_seed=3)
    correlations = redundancy.pairwise_correlations(contexts, config)
    exact = correlations.loc[
        correlations["feature_a"].eq(FEATURES[0]) & correlations["feature_b"].eq(FEATURES[1])
    ].iloc[0]
    assert exact["spearman_rho"] == 1.0
    assert exact["ci_low"] > .99
    clusters, _ = redundancy.correlation_clusters(correlations, config)
    at_95 = clusters.loc[clusters["absolute_rho_threshold"].eq(.95)]
    labels = at_95.set_index("feature")["cluster_id"]
    assert labels[FEATURES[0]] == labels[FEATURES[1]]


def test_context_builder_includes_all_levels_and_primary_strata(monkeypatch) -> None:
    monkeypatch.setattr(redundancy, "TIER_A_FEATURE_COLUMNS", FEATURES)
    candidates = pd.DataFrame([_candidate(patient) for patient in range(1, 13)])
    contexts = redundancy.build_analysis_contexts(candidates, _primary())
    assert len(contexts) == 24
    identities = [identity for identity, _ in contexts]
    assert {item["aggregation_level"] for item in identities} == {
        "cycle", "recording", "patient_location", "primary_patient"
    }


def test_runner_writes_hashed_outputs_without_outcome(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(redundancy, "TIER_A_FEATURE_COLUMNS", FEATURES)
    phase2a, phase2b = tmp_path / "phase2a", tmp_path / "phase2b"
    phase2a.mkdir(); phase2b.mkdir()
    table = pd.DataFrame([_candidate(patient) for patient in range(1, 13)])
    feature_path = phase2a / "task5_feature_table.csv"
    table.to_csv(feature_path, index=False)
    (phase2a / "run_manifest.json").write_text(json.dumps({
        "status": "complete", "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "artifact_sha256": {feature_path.name: _hash(feature_path)},
    }))
    primary_path = phase2b / "primary_patient_feature_long.csv"
    _primary().to_csv(primary_path, index=False)
    (phase2b / "run_manifest.json").write_text(json.dumps({
        "status": "complete", "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_feature_table_sha256": _hash(feature_path),
        "artifact_sha256": {primary_path.name: _hash(primary_path)},
    }))
    destination = redundancy.run_redundancy_analysis(
        phase2a_run=phase2a, phase2b_run=phase2b,
        run_name="redundancy-test", output_root=tmp_path / "outputs",
        config=redundancy.RedundancyConfig(bootstrap_replicates=10),
    )
    manifest = json.loads((destination / "run_manifest.json").read_text())
    assert manifest["outcome_used"] is False
    assert manifest["prediction_pruning_permitted"] is False
    for filename, expected in manifest["artifact_sha256"].items():
        assert _hash(destination / filename) == expected
