"""Versioned Task 5 Phase 2A full-dataset extraction and denominator audit.

The runner freezes the CirCor v1.0.3 inventory and validation report, excludes
only the six preregistered unusable annotations, invokes the unchanged
separation/feature pipeline, and writes a prospective Tier A table with
coverage, invalid-reason, recording, and patient-location flows.  It does not
perform construct validation, redundancy analysis, or classification.
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

from config import (
    AUDIO_DIR,
    DATASET_ROOT,
    DEFAULT_SEPARATION_CONFIG,
    EXPECTED_PATIENTS,
    EXPECTED_RECORDINGS,
    METADATA_PATH,
    OUTPUT_ROOT,
    SeparationConfig,
    validate_input_output_isolation,
)
from src.data_validation import validate_dataset
from src.separation.audit import run_audit
from src.separation.tier_a_features import (
    FEATURE_SCHEMA_VERSION,
    TIER_A_EXPORT_COLUMNS,
    TIER_A_FEATURE_COLUMNS,
    TIER_A_FEATURE_UNITS,
)


TOOL_VERSION: Final = "task5-phase2a-full-extraction-v1.0.0"
DEFAULT_OUTPUT_ROOT: Final = OUTPUT_ROOT / "task5_full_extraction"
KNOWN_UNUSABLE_ANNOTATIONS: Final[tuple[str, ...]] = (
    "50150_MV.tsv",
    "50690_MV_2.tsv",
    "50690_TV.tsv",
    "50782_MV_1.tsv",
    "84851_PV.tsv",
    "84930_AV.tsv",
)

IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "candidate_id",
    "patient_id",
    "recording_id",
    "location",
    "cycle_index",
    "murmur_phase",
)

CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    "murmur_label",
    "location_murmur_label",
    "patient_murmur_label",
    "clinical_outcome",
    "expert_timing_label",
    "expert_shape_label",
    "expert_pitch_label",
    "expert_grading_label",
    "expert_quality_label",
    "sample_rate",
    "config_hash",
    "requested_method",
    "separation_method",
    "candidate_quality_status",
    "activity_detection_method",
    "phase_selection_used_fallback",
    "phase_selected_component_count",
    "reconstruction_error",
    "normal_residual_correlation",
    "s1_leakage_ratio",
    "s2_leakage_ratio",
    "murmur_region_energy_retention",
    "outside_murmur_energy_ratio",
    "systole_candidate_energy_ratio",
    "diastole_candidate_energy_ratio",
    "noise_energy_ratio",
    "onset_sample",
    "offset_sample",
    "duration_samples",
    "duration_seconds",
)


@dataclass(frozen=True)
class FullExtractionConfig:
    method: str = "zcr"
    cycles_per_recording: int = 0
    target_phase: str = "auto"
    output_profile: str = "summary"
    recording_limit: int = 0

    def __post_init__(self) -> None:
        if self.method not in {"zcr", "kurtosis"}:
            raise ValueError("Phase 2A method must be zcr or kurtosis")
        if self.cycles_per_recording < 0:
            raise ValueError("cycles_per_recording must be non-negative")
        if self.target_phase not in {"auto", "systole", "diastole"}:
            raise ValueError("invalid target_phase")
        if self.output_profile not in {"summary", "accepted", "full"}:
            raise ValueError("invalid output_profile")
        if self.recording_limit < 0:
            raise ValueError("recording_limit must be non-negative")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode("utf-8")
    return sha256(payload).hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_dataset_inventory(audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    """Return a deterministic size-level inventory of exact WAV/TSV pairs."""

    wav = {path.stem: path for path in Path(audio_dir).glob("*.wav")}
    tsv = {path.stem: path for path in Path(audio_dir).glob("*.tsv")}
    rows = []
    for recording_id in sorted(set(wav) | set(tsv)):
        wav_path = wav.get(recording_id)
        tsv_path = tsv.get(recording_id)
        rows.append(
            {
                "recording_id": recording_id,
                "wav_file": wav_path.name if wav_path else None,
                "wav_size_bytes": wav_path.stat().st_size if wav_path else None,
                "tsv_file": tsv_path.name if tsv_path else None,
                "tsv_size_bytes": tsv_path.stat().st_size if tsv_path else None,
                "exact_pair": wav_path is not None and tsv_path is not None,
            }
        )
    return pd.DataFrame(rows)


def validate_preregistered_snapshot(report: dict[str, Any]) -> list[str]:
    """Accept only the exact known v1.0.3 validation exception set."""

    checks = {
        "wav_recordings": EXPECTED_RECORDINGS,
        "tsv_annotations": EXPECTED_RECORDINGS,
        "patients": EXPECTED_PATIENTS,
    }
    for key, expected in checks.items():
        if int(report.get(key, -1)) != expected:
            raise ValueError(f"dataset {key}={report.get(key)!r}; expected {expected}")
    if report.get("wav_missing_tsv") or report.get("tsv_missing_wav"):
        raise ValueError("dataset does not contain exact WAV/TSV pairing")
    if report.get("sample_rate_counts") != {"4000": EXPECTED_RECORDINGS}:
        raise ValueError(f"unexpected sample-rate inventory: {report.get('sample_rate_counts')}")
    error_codes = [str(item.get("code")) for item in report.get("errors", [])]
    if error_codes != ["invalid_annotations"]:
        raise ValueError(f"unexpected dataset validation errors: {error_codes}")
    observed = sorted(
        str(item["file"]) for item in report.get("invalid_annotations", [])
    )
    expected = sorted(KNOWN_UNUSABLE_ANNOTATIONS)
    if observed != expected:
        raise ValueError(
            f"unusable annotation set changed: observed={observed}, expected={expected}"
        )
    return [Path(name).stem for name in expected]


def _candidate_id(row: pd.Series) -> str:
    return (
        f"{row['patient_id']}:{row['recording_id']}:"
        f"cycle{int(row['cycle_index'])}:{row['murmur_phase']}"
    )


def build_feature_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Freeze the prospective Phase 2A candidate table and stable ordering."""

    required = {
        "patient_id",
        "recording_id",
        "location",
        "cycle_index",
        "murmur_phase",
        "candidate_quality_status",
        "onset_sample",
        "offset_sample",
        *TIER_A_EXPORT_COLUMNS,
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"runner summary is missing Phase 2A columns: {sorted(missing)}")
    table = summary.copy()
    table["candidate_id"] = [_candidate_id(row) for _, row in table.iterrows()]
    if table["candidate_id"].duplicated().any():
        raise ValueError("Phase 2A candidate identifiers must be unique")
    columns = [
        column
        for column in (*IDENTIFIER_COLUMNS, *CONTEXT_COLUMNS, *TIER_A_EXPORT_COLUMNS)
        if column in table.columns
    ]
    return table.loc[:, list(dict.fromkeys(columns))].sort_values(
        ["patient_id", "recording_id", "cycle_index", "murmur_phase"],
        kind="mergesort",
    )


def build_denominator_flow(
    table: pd.DataFrame,
    skipped: pd.DataFrame,
    validation: dict[str, Any],
    *,
    selected_recordings: int,
) -> pd.DataFrame:
    """Build explicit dataset-to-feature denominators without hidden filtering."""

    activity_valid = (
        pd.to_numeric(table.get("offset_sample"), errors="coerce")
        > pd.to_numeric(table.get("onset_sample"), errors="coerce")
    )
    rows: list[dict[str, Any]] = [
        {"stage": "dataset_patients", "phase": "all", "status": "all", "feature": "all", "count": int(validation["patients"])},
        {"stage": "exact_wav_tsv_pairs", "phase": "all", "status": "all", "feature": "all", "count": int(validation["wav_recordings"])},
        {"stage": "preregistered_unusable_annotations", "phase": "all", "status": "excluded", "feature": "all", "count": len(KNOWN_UNUSABLE_ANNOTATIONS)},
        {"stage": "metadata_matched_valid_recordings_selected", "phase": "all", "status": "all", "feature": "all", "count": int(selected_recordings)},
        {"stage": "recordings_with_candidate_rows", "phase": "all", "status": "all", "feature": "all", "count": int(table["recording_id"].nunique())},
        {"stage": "patients_with_candidate_rows", "phase": "all", "status": "all", "feature": "all", "count": int(table["patient_id"].nunique())},
        {"stage": "valid_complete_cycles", "phase": "all", "status": "all", "feature": "all", "count": int(table[["recording_id", "cycle_index"]].drop_duplicates().shape[0])},
        {"stage": "candidate_rows", "phase": "all", "status": "all", "feature": "all", "count": int(len(table))},
        {"stage": "valid_activity_interval", "phase": "all", "status": "all", "feature": "all", "count": int(activity_valid.sum())},
        {"stage": "skipped_segments_or_recordings", "phase": "all", "status": "skipped", "feature": "all", "count": int(len(skipped))},
    ]
    for (phase, status), group in table.groupby(
        ["murmur_phase", "candidate_quality_status"], sort=True, dropna=False
    ):
        rows.append(
            {"stage": "candidate_rows", "phase": phase, "status": status, "feature": "all", "count": int(len(group))}
        )
    for phase in ("all", *sorted(table["murmur_phase"].dropna().astype(str).unique())):
        phase_table = table if phase == "all" else table.loc[table["murmur_phase"].eq(phase)]
        for feature in TIER_A_FEATURE_COLUMNS:
            valid = phase_table[f"{feature}_valid"].fillna(False).astype(bool)
            rows.append(
                {"stage": "feature_valid", "phase": phase, "status": "all", "feature": feature, "count": int(valid.sum())}
            )
    return pd.DataFrame(rows)


def summarize_feature_coverage(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("all", "all", table)]
    for phase, phase_table in table.groupby("murmur_phase", sort=True):
        groups.append((str(phase), "all", phase_table))
        for status, status_table in phase_table.groupby("candidate_quality_status", sort=True):
            groups.append((str(phase), str(status), status_table))
    for phase, status, group in groups:
        for feature in TIER_A_FEATURE_COLUMNS:
            valid = group[f"{feature}_valid"].fillna(False).astype(bool)
            values = pd.to_numeric(group.loc[valid, feature], errors="coerce").dropna()
            rows.append(
                {
                    "murmur_phase": phase,
                    "candidate_quality_status": status,
                    "feature": feature,
                    "unit": TIER_A_FEATURE_UNITS[feature],
                    "attempted_count": int(len(group)),
                    "valid_count": int(len(values)),
                    "coverage": float(len(values) / len(group)) if len(group) else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "q25": float(values.quantile(.25)) if len(values) else np.nan,
                    "q75": float(values.quantile(.75)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_invalid_reasons(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in TIER_A_FEATURE_COLUMNS:
        invalid = table.loc[~table[f"{feature}_valid"].fillna(False).astype(bool)].copy()
        reasons = invalid[f"{feature}_invalid_reason"].fillna("unspecified").astype(str)
        for (phase, status, reason), count in (
            invalid.assign(_reason=reasons)
            .groupby(["murmur_phase", "candidate_quality_status", "_reason"], dropna=False)
            .size()
            .items()
        ):
            rows.append(
                {
                    "murmur_phase": phase,
                    "candidate_quality_status": status,
                    "feature": feature,
                    "invalid_reason": reason,
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def aggregate_features_long(
    table: pd.DataFrame, *, level: str
) -> pd.DataFrame:
    """Aggregate cycles using medians while retaining attempts and dispersion."""

    if level == "recording":
        group_columns = ["patient_id", "recording_id", "location", "murmur_phase"]
    elif level == "patient_location":
        group_columns = ["patient_id", "location", "murmur_phase"]
    else:
        raise ValueError("level must be recording or patient_location")
    rows: list[dict[str, Any]] = []
    for keys, group in table.groupby(group_columns, sort=True, dropna=False):
        identity = dict(zip(group_columns, keys, strict=True))
        for feature in TIER_A_FEATURE_COLUMNS:
            valid = group[f"{feature}_valid"].fillna(False).astype(bool)
            values = pd.to_numeric(group.loc[valid, feature], errors="coerce").dropna()
            rows.append(
                {
                    **identity,
                    "aggregation_level": level,
                    "feature": feature,
                    "unit": TIER_A_FEATURE_UNITS[feature],
                    "attempted_cycle_count": int(len(group)),
                    "valid_cycle_count": int(len(values)),
                    "valid_fraction": float(len(values) / len(group)),
                    "median": float(values.median()) if len(values) else np.nan,
                    "q25": float(values.quantile(.25)) if len(values) else np.nan,
                    "q75": float(values.quantile(.75)) if len(values) else np.nan,
                    "iqr": float(values.quantile(.75) - values.quantile(.25)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_phase2a_artifacts(
    summary: pd.DataFrame,
    skipped: pd.DataFrame,
    validation: dict[str, Any],
    destination: Path,
    *,
    runner_manifest: dict[str, Any],
    inventory_hash: str,
    extraction_config: FullExtractionConfig,
    separation_config: SeparationConfig,
) -> dict[str, Any]:
    table = build_feature_table(summary)
    denominator = build_denominator_flow(
        table,
        skipped,
        validation,
        selected_recordings=int(runner_manifest["selected_recordings"]),
    )
    artifacts = {
        "task5_feature_table.csv": table,
        "denominator_flow.csv": denominator,
        "feature_coverage.csv": summarize_feature_coverage(table),
        "feature_invalid_reasons.csv": summarize_invalid_reasons(table),
        "recording_feature_aggregates.csv": aggregate_features_long(table, level="recording"),
        "patient_location_feature_aggregates.csv": aggregate_features_long(table, level="patient_location"),
    }
    hashes: dict[str, str] = {}
    for filename, frame in artifacts.items():
        path = destination / filename
        frame.to_csv(path, index=False)
        hashes[filename] = _sha256_file(path)
    manifest = {
        "tool_version": TOOL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "run_name": destination.name,
        "run_scope": "pilot" if extraction_config.recording_limit else "full_dataset",
        "status": "complete",
        "extraction_config": asdict(extraction_config),
        "separation_config": separation_config.to_dict(),
        "separation_config_hash": separation_config.config_hash,
        "dataset_root": str(DATASET_ROOT),
        "metadata_sha256": _sha256_file(METADATA_PATH),
        "dataset_inventory_sha256": inventory_hash,
        "dataset_validation_sha256": _sha256_file(destination / "dataset_validation.json"),
        "known_unusable_annotations": list(KNOWN_UNUSABLE_ANNOTATIONS),
        "candidate_count": int(len(table)),
        "patient_count": int(table["patient_id"].nunique()),
        "recording_count": int(table["recording_id"].nunique()),
        "complete_cycle_count": int(table[["recording_id", "cycle_index"]].drop_duplicates().shape[0]),
        "phase_counts": {str(k): int(v) for k, v in table["murmur_phase"].value_counts().sort_index().items()},
        "status_counts": {str(k): int(v) for k, v in table["candidate_quality_status"].value_counts().sort_index().items()},
        "skipped_count": int(len(skipped)),
        "artifact_sha256": hashes,
        "scope_statement": "Prospective Tier A extraction and denominator/coverage audit only; no construct validation, redundancy selection, or classification.",
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _write_preflight(
    destination: Path,
    extraction_config: FullExtractionConfig,
    separation_config: SeparationConfig,
) -> tuple[dict[str, Any], list[str], str]:
    validation_path = destination / "dataset_validation.json"
    validation = validate_dataset(
        report_path=validation_path,
        strict=False,
        inspect_audio=True,
    )
    excluded = validate_preregistered_snapshot(validation)
    inventory = build_dataset_inventory()
    inventory_path = destination / "dataset_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    inventory_hash = _sha256_file(inventory_path)
    schema = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "tier_a_features": [
            {"column": feature, "unit": TIER_A_FEATURE_UNITS[feature]}
            for feature in TIER_A_FEATURE_COLUMNS
        ],
        "validity_columns": [f"{feature}_valid" for feature in TIER_A_FEATURE_COLUMNS],
        "invalid_reason_columns": [f"{feature}_invalid_reason" for feature in TIER_A_FEATURE_COLUMNS],
        "sample_index_convention": "zero-based starts and exclusive ends",
    }
    (destination / "frozen_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    preflight = {
        "tool_version": TOOL_VERSION,
        "status": "preflight_complete",
        "extraction_config": asdict(extraction_config),
        "separation_config": separation_config.to_dict(),
        "separation_config_hash": separation_config.config_hash,
        "metadata_sha256": _sha256_file(METADATA_PATH),
        "dataset_inventory_sha256": inventory_hash,
        "dataset_validation_sha256": _sha256_file(validation_path),
        "excluded_recording_ids": excluded,
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
    }
    (destination / "preflight_manifest.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    return validation, excluded, inventory_hash


def run_full_extraction(
    *,
    run_name: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    extraction_config: FullExtractionConfig = FullExtractionConfig(),
    separation_config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
    resume: bool = False,
    preflight_only: bool = False,
) -> Path:
    """Run or safely resume one isolated Phase 2A extraction."""

    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one non-empty path component")
    validate_input_output_isolation(DATASET_ROOT, output_root)
    destination = Path(output_root) / run_name
    if destination.exists() and not resume:
        raise FileExistsError(f"Phase 2A run exists and will not be overwritten: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    preflight_path = destination / "preflight_manifest.json"
    if resume and preflight_path.exists():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("extraction_config") != asdict(extraction_config):
            raise ValueError("resume extraction configuration differs from preflight")
        if preflight.get("separation_config_hash") != separation_config.config_hash:
            raise ValueError("resume separation configuration differs from preflight")
        validation = json.loads((destination / "dataset_validation.json").read_text(encoding="utf-8"))
        excluded = validate_preregistered_snapshot(validation)
        inventory_hash = _sha256_file(destination / "dataset_inventory.csv")
        if inventory_hash != preflight.get("dataset_inventory_sha256"):
            raise ValueError("dataset inventory changed since preflight")
        if _sha256_frame(build_dataset_inventory()) != inventory_hash:
            raise ValueError("current WAV/TSV inventory differs from preflight")
        if _sha256_file(METADATA_PATH) != preflight.get("metadata_sha256"):
            raise ValueError("metadata changed since preflight")
    else:
        validation, excluded, inventory_hash = _write_preflight(
            destination, extraction_config, separation_config
        )
    if preflight_only:
        return destination

    reports = destination / "runner_reports"
    packages = destination / "packages"
    summary = run_audit(
        limit=extraction_config.recording_limit,
        cycles_per_recording=extraction_config.cycles_per_recording,
        config=separation_config,
        method=extraction_config.method,
        validate_first=False,
        run_name=run_name,
        all_recordings=True,
        output_profile=extraction_config.output_profile,
        resume=resume,
        target_phase=extraction_config.target_phase,
        excluded_recording_ids=excluded,
        report_output_dir=reports,
        separation_output_dir=packages,
    )
    suffix = f"_{run_name}"
    skipped_path = reports / f"separation_skipped{suffix}.csv"
    skipped = pd.read_csv(
        skipped_path, dtype={"patient_id": str, "recording_id": str}
    )
    runner_manifest = json.loads(
        (reports / f"separation_manifest{suffix}.json").read_text(encoding="utf-8")
    )
    write_phase2a_artifacts(
        summary,
        skipped,
        validation,
        destination,
        runner_manifest=runner_manifest,
        inventory_hash=inventory_hash,
        extraction_config=extraction_config,
        separation_config=separation_config,
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--recording-limit", type=int, default=0)
    parser.add_argument("--cycles-per-recording", type=int, default=0)
    parser.add_argument("--method", choices=["zcr", "kurtosis"], default="zcr")
    args = parser.parse_args(argv)
    extraction_config = FullExtractionConfig(
        method=args.method,
        cycles_per_recording=args.cycles_per_recording,
        recording_limit=args.recording_limit,
    )
    destination = run_full_extraction(
        run_name=args.run_name,
        extraction_config=extraction_config,
        resume=args.resume,
        preflight_only=args.preflight_only,
    )
    print(f"Task 5 Phase 2A output: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
