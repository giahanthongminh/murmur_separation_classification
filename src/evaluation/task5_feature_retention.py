"""Freeze the Task 5 Tier A retention decision before Task 6.

The runner verifies Phase 1C/1D and Phase 2A/2B/2C artifacts, adds
patient-cluster coverage intervals, and writes the outcome-blind decision log.
It does not train, tune, prune for, or otherwise inspect a classifier.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Final

import numpy as np
import pandas as pd

from config import OUTPUT_ROOT
from src.separation.tier_a_features import FEATURE_SCHEMA_VERSION, TIER_A_FEATURE_COLUMNS


TOOL_VERSION: Final = "task5-feature-retention-v1.0.0"
DEFAULT_OUTPUT_ROOT: Final = OUTPUT_ROOT / "task5_feature_retention"

DECISIONS: Final[dict[str, tuple[str, str, bool]]] = {
    "tier_a_relative_murmur_duration": (
        "retain_core", "Direct timing construct; strong consistent holosystolic effect; expected boundary-defined sensitivity is reportable.", False),
    "tier_a_temporal_midpoint_normalized": (
        "retain_core", "Direct timing construct with high-quality Early-to-Mid support; boundary-defined sensitivity is scientifically manageable.", True),
    "tier_a_murmur_rms_relative_s1_s2_db": (
        "retain_core", "Reference-normalized amplitude has strong consistent ordered grade support despite substantial QA association.", False),
    "tier_a_envelope_peak_position_normalized": (
        "retain_exploratory", "Correct and well covered, but boundary sensitive and shape support is weak.", True),
    "tier_a_envelope_rise_slope_robust": (
        "retain_exploratory", "Interpretable morphology measure with sub-core coverage and no stable shape construct support.", True),
    "tier_a_envelope_decay_slope_robust": (
        "drop_unstable", "Highest standardized boundary sensitivity, frequent invalidation, limited coverage, and shape signal disappears in high-quality candidates.", False),
    "tier_a_envelope_fullness": (
        "retain_exploratory", "Correct and highly covered, but shape support is weak and candidate-quality dependence remains unresolved.", True),
    "tier_a_psd_dominant_frequency_hz": (
        "retain_exploratory", "Representative of the dominant/median-frequency redundancy cluster; pitch direction is reversed and requires contamination warning.", True),
    "tier_a_psd_median_frequency_hz": (
        "drop_redundant", "Stable high-correlation cluster with dominant frequency; retained only as an audit alternative.", False),
    "tier_a_psd_bandwidth_95_hz": (
        "retain_exploratory", "Correct, stable, and covered, but Harsh/Blowing support vanishes in high-quality candidates and QA dependence is strong.", True),
    "tier_a_psd_energy_fraction_above_200_hz": (
        "drop_invalid_interpretation", "Near-saturated values, reversed pitch direction, and substantial QA dependence make the intended pitch interpretation unsafe.", False),
    "tier_a_psd_entropy_normalized": (
        "retain_exploratory", "Correct and covered; quality contrast is not robust to the high-quality stratum and QA dependence is strong.", True),
    "tier_a_ridge_slope_robust_hz_per_normalized_time": (
        "drop_poor_coverage", "Approximately 53% candidate coverage, substantial invalidation, and no matching confirmatory descriptor.", False),
    "tier_a_ridge_variability_mad_hz": (
        "drop_poor_coverage", "Approximately 53% candidate coverage, many zero summaries, and no matching confirmatory descriptor.", False),
}


@dataclass(frozen=True)
class RetentionConfig:
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 20260827
    confidence_level: float = .95


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _verified_artifact(run: Path, filename: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(run) / "run_manifest.json"
    artifact_path = Path(run) / filename
    if not manifest_path.exists() or not artifact_path.exists():
        raise FileNotFoundError(f"missing frozen source: {artifact_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {None, "complete"}:
        raise ValueError(f"source run is not complete: {run}")
    expected = manifest.get("artifact_sha256", {}).get(filename)
    if expected and expected != _sha256_file(artifact_path):
        raise ValueError(f"artifact hash mismatch: {artifact_path}")
    return artifact_path, manifest


def cluster_coverage_intervals(
    feature_path: Path, config: RetentionConfig
) -> pd.DataFrame:
    columns = [
        "patient_id", "murmur_phase", "candidate_quality_status",
        *[f"{feature}_valid" for feature in TIER_A_FEATURE_COLUMNS],
    ]
    table = pd.read_csv(feature_path, usecols=columns, dtype={"patient_id": str})
    rows: list[dict[str, Any]] = []
    for phase in ("systole", "diastole"):
        phase_table = table.loc[table["murmur_phase"].eq(phase)]
        for status in ("all", "accepted", "fallback"):
            selected = phase_table if status == "all" else phase_table.loc[
                phase_table["candidate_quality_status"].eq(status)
            ]
            patients = np.unique(selected["patient_id"].astype(str))
            rng = np.random.default_rng(config.bootstrap_seed + sum(map(ord, phase + status)))
            draws = rng.multinomial(
                len(patients), np.full(len(patients), 1 / len(patients)),
                size=config.bootstrap_replicates,
            )
            patient_index = np.searchsorted(patients, selected["patient_id"].astype(str))
            attempted = np.bincount(patient_index, minlength=len(patients)).astype(float)
            bootstrap_attempted = draws @ attempted
            for feature in TIER_A_FEATURE_COLUMNS:
                valid_mask = selected[f"{feature}_valid"].fillna(False).astype(bool).to_numpy(float)
                valid = np.bincount(patient_index, weights=valid_mask, minlength=len(patients))
                estimates = np.divide(
                    draws @ valid, bootstrap_attempted,
                    out=np.full(config.bootstrap_replicates, np.nan), where=bootstrap_attempted > 0,
                )
                finite = estimates[np.isfinite(estimates)]
                alpha = (1 - config.confidence_level) / 2
                low, high = np.quantile(finite, [alpha, 1 - alpha])
                rows.append({
                    "murmur_phase": phase, "candidate_quality_status": status,
                    "feature": feature, "attempted_count": int(len(selected)),
                    "valid_count": int(valid_mask.sum()),
                    "coverage": float(valid_mask.mean()),
                    "cluster_ci_low": float(low), "cluster_ci_high": float(high),
                    "independent_patient_count": len(patients),
                })
    return pd.DataFrame(rows)


def build_decision_log(
    coverage: pd.DataFrame,
    boundary: pd.DataFrame,
    quality: pd.DataFrame,
    contrasts: pd.DataFrame,
    instability: pd.DataFrame,
) -> pd.DataFrame:
    if set(DECISIONS) != set(TIER_A_FEATURE_COLUMNS):
        raise AssertionError("retention registry must cover exactly the frozen Tier A features")
    primary_coverage = coverage.loc[
        coverage["murmur_phase"].eq("systole")
        & coverage["candidate_quality_status"].eq("all")
    ].set_index("feature")
    boundary_index = boundary.set_index("feature")
    qa = quality.loc[
        quality["murmur_phase"].eq("systole")
        & quality["outcome"].eq("nominal_feature_value")
        & quality["inference_status"].eq("estimable")
    ].copy()
    qa["absolute_rho"] = qa["spearman_rho"].abs()
    qa_max = qa.groupby("feature")["absolute_rho"].max()
    construct_rows: dict[str, str] = {}
    for feature, group in contrasts.groupby("feature"):
        construct_rows[feature] = "; ".join(
            f"{row.stratum}:{row.comparison}:effect={row.effect:.3g}:holm_p={row.holm_p_value:.3g}"
            for row in group.itertuples()
        )
    redundant = instability.loc[
        np.isclose(instability["absolute_rho_threshold"], .90)
        & instability["same_cluster_fraction"].ge(.5)
    ]
    cluster_map: dict[str, str] = {}
    for row in redundant.itertuples():
        label = f"{row.feature_a}|{row.feature_b} ({row.same_cluster_context_count}/{row.context_count} contexts)"
        cluster_map[row.feature_a] = cluster_map[row.feature_b] = label
    rows = []
    for feature in TIER_A_FEATURE_COLUMNS:
        decision, rationale, threshold_dependent = DECISIONS[feature]
        cov = primary_coverage.loc[feature]
        stability = boundary_index.loc[feature]
        rows.append({
            "feature": feature,
            "correctness_gate": "pass_implementation_tests",
            "systolic_candidate_coverage": cov["coverage"],
            "coverage_cluster_ci_low": cov["cluster_ci_low"],
            "coverage_cluster_ci_high": cov["cluster_ci_high"],
            "boundary_median_iqr_standardized_change": stability["median_iqr_standardized_absolute_change"],
            "boundary_valid_to_invalid_count": int(stability["valid_to_invalid_count"]),
            "maximum_absolute_qa_spearman_phase1d": float(qa_max.get(feature, np.nan)),
            "construct_summary": construct_rows.get(feature, "no_matching_confirmatory_descriptor"),
            "redundancy_cluster_0_90": cluster_map.get(feature),
            "decision": decision,
            "task6_eligible": decision in {"retain_core", "retain_exploratory"},
            "threshold_dependent": threshold_dependent,
            "rationale": rationale,
        })
    return pd.DataFrame(rows)


def run_feature_retention(
    *, phase2a_run: Path, phase1c_run: Path, phase1d_run: Path,
    phase2b_run: Path, phase2c_run: Path, run_name: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config: RetentionConfig = RetentionConfig(),
) -> Path:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one path component")
    destination = Path(output_root) / run_name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {destination}")
    feature_path, extraction_manifest = _verified_artifact(phase2a_run, "task5_feature_table.csv")
    boundary_path, _ = _verified_artifact(phase1c_run, "feature_stability_summary.csv")
    quality_path, _ = _verified_artifact(phase1d_run, "qa_spearman_associations.csv")
    contrast_path, construct_manifest = _verified_artifact(phase2b_run, "construct_contrasts.csv")
    instability_path, redundancy_manifest = _verified_artifact(phase2c_run, "cluster_instability.csv")
    if extraction_manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Phase 2A feature schema mismatch")
    if construct_manifest.get("source_feature_table_sha256") != _sha256_file(feature_path):
        raise ValueError("Phase 2B source does not match Phase 2A")
    if redundancy_manifest.get("phase2a_feature_table_sha256") != _sha256_file(feature_path):
        raise ValueError("Phase 2C source does not match Phase 2A")
    coverage = cluster_coverage_intervals(feature_path, config)
    decisions = build_decision_log(
        coverage, pd.read_csv(boundary_path), pd.read_csv(quality_path),
        pd.read_csv(contrast_path), pd.read_csv(instability_path),
    )
    groups = {
        "core": decisions.loc[decisions["decision"].eq("retain_core"), "feature"].tolist(),
        "exploratory": decisions.loc[decisions["decision"].eq("retain_exploratory"), "feature"].tolist(),
        "excluded": decisions.loc[~decisions["task6_eligible"], "feature"].tolist(),
        "task6_rule": "Any preprocessing, imputation, scaling, and data-driven pruning must be fitted within patient-grouped training folds.",
    }
    destination.mkdir(parents=True)
    coverage.to_csv(destination / "coverage_cluster_intervals.csv", index=False)
    decisions.to_csv(destination / "feature_retention_decision_log.csv", index=False)
    (destination / "retained_feature_groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    artifacts = {
        name: _sha256_file(destination / name) for name in (
            "coverage_cluster_intervals.csv", "feature_retention_decision_log.csv", "retained_feature_groups.json"
        )
    }
    manifest = {
        "tool_version": TOOL_VERSION, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "status": "complete", "run_name": run_name, "config": asdict(config),
        "source_hashes": {
            "phase2a_feature_table": _sha256_file(feature_path),
            "phase1c_boundary": _sha256_file(boundary_path),
            "phase1d_quality": _sha256_file(quality_path),
            "phase2b_construct": _sha256_file(contrast_path),
            "phase2c_redundancy": _sha256_file(instability_path),
        },
        "decision_counts": decisions["decision"].value_counts().sort_index().to_dict(),
        "artifact_sha256": artifacts, "outcome_used": False,
        "classification_started": False,
        "scope_statement": "Frozen Task 5 retention decision only; no Task 6 performance was viewed.",
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
        "python_version": platform.python_version(), "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2a-run", type=Path, required=True)
    parser.add_argument("--phase1c-run", type=Path, required=True)
    parser.add_argument("--phase1d-run", type=Path, required=True)
    parser.add_argument("--phase2b-run", type=Path, required=True)
    parser.add_argument("--phase2c-run", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    destination = run_feature_retention(
        phase2a_run=args.phase2a_run, phase1c_run=args.phase1c_run,
        phase1d_run=args.phase1d_run, phase2b_run=args.phase2b_run,
        phase2c_run=args.phase2c_run, run_name=args.run_name,
        output_root=args.output_root,
    )
    print(f"Task 5 retention artifacts: {destination}")


if __name__ == "__main__":
    main()
