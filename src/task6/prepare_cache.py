"""Build a resumable Task 6 cache of systolic separated-candidate waveforms.

The generator is restricted to patients with patient-level Murmur=Present and
Outcome in {Normal, Abnormal}. It reproduces the frozen Task 5 separation for
the exact Phase 2A systolic candidate rows and saves only the systolic portion
of each candidate. No Outcome-dependent signal processing or QA filtering is
performed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
from typing import Any, Final

import numpy as np
import pandas as pd

from config import AUDIO_DIR, METADATA_PATH, OUTPUT_ROOT, SeparationConfig
from src.separation.audit import _load_wav, build_cardiac_cycle_context
from src.separation.core import separate_signal
from src.task6.common import (
    TASK6_VERSION,
    eligible_patient_labels,
    git_value,
    sha256_file,
    sha256_json,
    verify_manifest_artifacts,
)


TOOL_VERSION: Final = f"{TASK6_VERSION}-cache"
DEFAULT_OUTPUT_ROOT: Final = OUTPUT_ROOT / "task6_cache"


@dataclass(frozen=True)
class CacheConfig:
    target_phase: str = "systole"
    signal_representation: str = "raw_systolic_candidate_float32"
    sample_index_convention: str = "zero_based_start_exclusive_end"
    checkpoint_interval_recordings: int = 1

    def __post_init__(self) -> None:
        if self.target_phase != "systole":
            raise ValueError("Task 6 primary cache is frozen to systole")
        if self.checkpoint_interval_recordings < 1:
            raise ValueError("checkpoint interval must be positive")


def _load_task5_source(run: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    run = Path(run).expanduser().resolve()
    manifest_path = run / "run_manifest.json"
    feature_path = run / "task5_feature_table.csv"
    if not manifest_path.exists() or not feature_path.exists():
        raise FileNotFoundError(f"Task 5 Phase 2A artifacts are incomplete: {run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Task 5 Phase 2A source is not complete")
    expected = manifest.get("artifact_sha256", {}).get("task5_feature_table.csv")
    if not expected or sha256_file(feature_path) != expected:
        raise ValueError("Task 5 feature table hash mismatch")
    required = [
        "candidate_id", "patient_id", "recording_id", "location", "cycle_index",
        "murmur_phase", "patient_murmur_label", "clinical_outcome",
        "candidate_quality_status", "phase_selection_used_fallback", "config_hash",
    ]
    table = pd.read_csv(feature_path, usecols=required, dtype={"patient_id": str, "recording_id": str})
    return table, manifest


def select_cache_candidates(table: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Apply only the preregistered patient label and systolic scope filters."""

    eligible = set(labels["patient_id"])
    selected = table.loc[
        table["patient_id"].isin(eligible)
        & table["patient_murmur_label"].eq("Present")
        & table["murmur_phase"].eq("systole")
    ].copy()
    if set(selected["patient_id"]) != eligible:
        missing = sorted(eligible - set(selected["patient_id"]))
        raise ValueError(f"eligible patients missing systolic candidates: {missing}")
    merged = selected.merge(labels, on="patient_id", how="left", suffixes=("_task5", ""), validate="many_to_one")
    if not merged["clinical_outcome_task5"].eq(merged["clinical_outcome"]).all():
        raise ValueError("Task 5 and current metadata Outcome labels disagree")
    if merged["candidate_id"].duplicated().any():
        raise ValueError("Task 6 candidate identifiers must be unique")
    return merged.sort_values(["recording_id", "cycle_index"], kind="mergesort").reset_index(drop=True)


def _write_checkpoint(index: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    index.sort_values(["recording_id", "cycle_index"], kind="mergesort").to_csv(temporary, index=False)
    temporary.replace(path)


def run_cache_generation(
    *, task5_run: Path, run_name: str, output_root: Path = DEFAULT_OUTPUT_ROOT,
    resume: bool = False, config: CacheConfig = CacheConfig(), recording_limit: int = 0,
) -> Path:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one path component")
    if recording_limit < 0:
        raise ValueError("recording_limit must be non-negative")
    task5_run = Path(task5_run).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve() / run_name
    if destination.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite Task 6 cache: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    signal_root = destination / "signals"
    signal_root.mkdir(exist_ok=True)

    task5_table, task5_manifest = _load_task5_source(task5_run)
    metadata = pd.read_csv(METADATA_PATH, dtype={"Patient ID": str})
    labels = eligible_patient_labels(metadata)
    candidates = select_cache_candidates(task5_table, labels)
    separation_config = SeparationConfig(**task5_manifest["separation_config"])
    method = str(task5_manifest["extraction_config"]["method"])
    if candidates["config_hash"].astype(str).nunique() != 1 or candidates["config_hash"].astype(str).iloc[0] != separation_config.config_hash:
        raise ValueError("Task 5 candidate configuration does not match its manifest")
    recording_ids = sorted(candidates["recording_id"].unique())
    if recording_limit:
        recording_ids = recording_ids[:recording_limit]
        candidates = candidates.loc[candidates["recording_id"].isin(recording_ids)].copy()

    identity = {
        "tool_version": TOOL_VERSION,
        "cache_config": asdict(config),
        "recording_limit": recording_limit,
        "separation_method": method,
        "separation_config": separation_config.to_dict(),
        "separation_config_hash": separation_config.config_hash,
        "metadata_sha256": sha256_file(METADATA_PATH),
        "task5_run_manifest_sha256": sha256_file(Path(task5_run) / "run_manifest.json"),
        "task5_feature_table_sha256": sha256_file(Path(task5_run) / "task5_feature_table.csv"),
        "eligible_patient_count": int(labels["patient_id"].nunique()),
        "eligible_outcome_counts": {str(k): int(v) for k, v in labels["clinical_outcome"].value_counts().sort_index().items()},
        "candidate_count_requested": int(len(candidates)),
    }
    preflight_path = destination / "preflight_manifest.json"
    if resume and preflight_path.exists():
        previous = json.loads(preflight_path.read_text(encoding="utf-8"))
        if previous.get("identity_hash") != sha256_json(identity):
            raise ValueError("resume configuration or Task 5 source differs from preflight")
    else:
        preflight = {**identity, "identity_hash": sha256_json(identity), "git_commit": git_value("rev-parse", "HEAD")}
        preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    checkpoint_path = destination / "candidate_index_checkpoint.csv"
    if resume and checkpoint_path.exists():
        completed_frame = pd.read_csv(checkpoint_path, dtype={"patient_id": str, "recording_id": str})
        for row in completed_frame.itertuples(index=False):
            signal_path = destination / str(row.signal_path)
            if not signal_path.exists() or sha256_file(signal_path) != str(row.signal_sha256):
                raise ValueError(f"resume checkpoint signal hash mismatch: {signal_path}")
    else:
        completed_frame = pd.DataFrame()
    completed = set(completed_frame.get("candidate_id", pd.Series(dtype=str)).astype(str))
    rows = completed_frame.to_dict(orient="records")
    for recording_number, recording_id in enumerate(recording_ids, start=1):
        recording_candidates = candidates.loc[candidates["recording_id"].eq(recording_id)]
        if set(recording_candidates["candidate_id"].astype(str)).issubset(completed):
            continue
        wav_path = AUDIO_DIR / f"{recording_id}.wav"
        tsv_path = AUDIO_DIR / f"{recording_id}.tsv"
        wav_hash, tsv_hash = sha256_file(wav_path), sha256_file(tsv_path)
        signal, sample_rate = _load_wav(wav_path, separation_config.sample_rate)
        annotations = pd.read_csv(tsv_path, sep="\t", header=None, names=["start", "end", "state"])
        systole_positions = np.flatnonzero(annotations["state"].to_numpy() == 2)
        for source in recording_candidates.itertuples(index=False):
            if str(source.candidate_id) in completed:
                continue
            cycle_index = int(source.cycle_index)
            if not 0 <= cycle_index < len(systole_positions):
                raise ValueError(f"Task 5 cycle index unavailable: {source.candidate_id}")
            cycle = build_cardiac_cycle_context(signal, annotations, int(systole_positions[cycle_index]), sample_rate)
            result = separate_signal(
                cycle.signal, config=separation_config, method=method,
                phase_masks=cycle.phase_masks, target_phase="systole",
            )
            observed_status = str(result.metrics["candidate_quality_status"])
            if observed_status != str(source.candidate_quality_status):
                raise ValueError(f"candidate status drift for {source.candidate_id}: {observed_status} != {source.candidate_quality_status}")
            systole_start, systole_end = cycle.phase_bounds["systole"]
            candidate_signal = np.asarray(result.murmur_candidate[systole_start:systole_end], dtype=np.float32)
            if not len(candidate_signal) or not np.all(np.isfinite(candidate_signal)):
                raise ValueError(f"invalid separated signal: {source.candidate_id}")
            patient_directory = signal_root / str(source.patient_id)
            patient_directory.mkdir(exist_ok=True)
            relative_path = Path("signals") / str(source.patient_id) / f"{recording_id}_cycle_{cycle_index}_systole.npy"
            output_path = destination / relative_path
            np.save(output_path, candidate_signal)
            rows.append({
                "candidate_id": source.candidate_id,
                "patient_id": str(source.patient_id),
                "recording_id": recording_id,
                "location": source.location,
                "cycle_index": cycle_index,
                "murmur_phase": "systole",
                "clinical_outcome": source.clinical_outcome,
                "candidate_quality_status": observed_status,
                "phase_selection_used_fallback": bool(result.metrics.get("phase_selection_used_fallback", False)),
                "sample_rate": sample_rate,
                "context_start_sample": cycle.context_start_sample,
                "context_end_sample": cycle.context_end_sample,
                "systole_relative_start_sample": systole_start,
                "systole_relative_end_sample": systole_end,
                "signal_sample_count": len(candidate_signal),
                "signal_path": relative_path.as_posix(),
                "signal_sha256": sha256_file(output_path),
                "source_wav_sha256": wav_hash,
                "source_tsv_sha256": tsv_hash,
                "separation_method": result.selected_method,
                "separation_config_hash": result.config_hash,
            })
            completed.add(str(source.candidate_id))
        if recording_number % config.checkpoint_interval_recordings == 0 or recording_number == len(recording_ids):
            _write_checkpoint(pd.DataFrame(rows), checkpoint_path)
            print(f"Task 6 cache: {recording_number}/{len(recording_ids)} recordings; {len(rows)}/{len(candidates)} candidates")

    index = pd.DataFrame(rows).sort_values(["patient_id", "recording_id", "cycle_index"], kind="mergesort")
    if len(index) != len(candidates) or set(index["candidate_id"]) != set(candidates["candidate_id"]):
        raise ValueError("Task 6 cache is incomplete")
    index_path = destination / "candidate_index.csv"
    index.to_csv(index_path, index=False)
    # Recreate this compact table from the authoritative completed index so resume
    # does not need to retain a separate source-hash checkpoint.
    sources = index[["recording_id", "source_wav_sha256", "source_tsv_sha256"]].drop_duplicates().sort_values("recording_id")
    sources.to_csv(destination / "source_file_hashes.csv", index=False)
    labels.to_csv(destination / "eligible_patients.csv", index=False)
    preprocessing = {
        "representation": config.signal_representation,
        "saved_region": "systole annotation interval from separated full-cycle candidate",
        "saved_dtype": "float32",
        "cache_normalization": "none",
        "model_length_handling": "right zero-pad or deterministic center crop; fixed in training configuration",
        "model_normalization": "mean and standard deviation fitted on outer-training patients only",
        "outcome_dependent_processing": False,
        "candidate_quality_filtering": False,
    }
    (destination / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    artifact_names = ("candidate_index.csv", "source_file_hashes.csv", "eligible_patients.csv", "preprocessing.json", "preflight_manifest.json")
    manifest = {
        **identity,
        "status": "complete",
        "generated_candidate_count": int(len(index)),
        "generated_patient_count": int(index["patient_id"].nunique()),
        "generated_recording_count": int(index["recording_id"].nunique()),
        "candidate_status_counts": {str(k): int(v) for k, v in index["candidate_quality_status"].value_counts().sort_index().items()},
        "artifact_sha256": {name: sha256_file(destination / name) for name in artifact_names},
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scope_statement": "Task 6 cache only: clinical Outcome classification among Murmur Present patients using separated systolic candidate information.",
    }
    manifest_path = destination / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_manifest_artifacts(destination, manifest)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task5-run", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recording-limit", type=int, default=0, help="Pilot only; zero means all eligible recordings")
    parser.add_argument("--checkpoint-interval-recordings", type=int, default=1)
    args = parser.parse_args(argv)
    destination = run_cache_generation(
        task5_run=args.task5_run, run_name=args.run_name, output_root=args.output_root,
        resume=args.resume, recording_limit=args.recording_limit,
        config=CacheConfig(checkpoint_interval_recordings=args.checkpoint_interval_recordings),
    )
    print(f"Task 6 cache: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
