"""Validate the canonical public CirCor training set before experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import wavfile

from config import (
    ANNOTATION_BOUNDARY_TOLERANCE_SECONDS,
    AUDIO_DIR,
    DATASET_ROOT,
    EXPECTED_PATIENTS,
    EXPECTED_RECORDINGS,
    METADATA_PATH,
    REPORT_OUTPUT_DIR,
    ensure_output_directories,
    validate_input_output_isolation,
)


VALID_CARDIAC_STATES = {0, 1, 2, 3, 4}
REQUIRED_CARDIAC_STATES = {1, 2, 3, 4}


class DatasetValidationError(RuntimeError):
    """Raised after a complete report has recorded dataset validation errors."""


@dataclass(frozen=True)
class DatasetPaths:
    root: Path = DATASET_ROOT
    audio_dir: Path = AUDIO_DIR
    metadata_path: Path = METADATA_PATH


def _read_annotation(path: Path) -> pd.DataFrame:
    table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["start", "end", "state"],
        dtype={"start": float, "end": float, "state": float},
    )
    if table.shape[1] != 3 or table.empty:
        raise ValueError("annotation must contain non-empty start/end/state rows")
    return table


def _metadata_references(metadata: pd.DataFrame) -> set[str]:
    references: set[str] = set()
    for _, row in metadata.iterrows():
        patient_id = str(row["Patient ID"]).strip()
        locations = str(row.get("Locations", "")).strip()
        if not locations or locations.lower() == "nan":
            continue
        for location in locations.split("+"):
            if location:
                references.add(f"{patient_id}_{location}")
    return references


def _metadata_reference_for_recording(recording_id: str) -> str:
    """Collapse CirCor repeat suffixes such as ``49748_AV_1`` to ``49748_AV``."""

    parts = recording_id.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return recording_id


def validate_dataset(
    paths: DatasetPaths | None = None,
    *,
    expected_recordings: int = EXPECTED_RECORDINGS,
    expected_patients: int = EXPECTED_PATIENTS,
    report_path: Path | None = None,
    strict: bool = True,
    inspect_audio: bool = True,
) -> dict[str, Any]:
    """Validate paths, counts, pairs, files, annotations, and metadata links.

    The report is written even when errors are found. In strict mode a
    :class:`DatasetValidationError` is raised only after the full audit.
    """

    paths = paths or DatasetPaths()
    root = paths.root.expanduser().resolve()
    audio_dir = paths.audio_dir.expanduser().resolve()
    metadata_path = paths.metadata_path.expanduser().resolve()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def add_error(code: str, detail: object) -> None:
        errors.append({"code": code, "detail": str(detail)})

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "audio_directory": str(audio_dir),
        "metadata_path": str(metadata_path),
        "expected_recordings": expected_recordings,
        "expected_patients": expected_patients,
        "errors": errors,
        "warnings": warnings,
    }

    for label, path in (
        ("dataset_root_missing", root),
        ("audio_directory_missing", audio_dir),
        ("metadata_missing", metadata_path),
    ):
        if not path.exists():
            add_error(label, path)

    wav_files = sorted(audio_dir.glob("*.wav")) if audio_dir.is_dir() else []
    tsv_files = sorted(audio_dir.glob("*.tsv")) if audio_dir.is_dir() else []
    report["wav_recordings"] = len(wav_files)
    report["tsv_annotations"] = len(tsv_files)

    wav_ids = [path.stem for path in wav_files]
    tsv_ids = [path.stem for path in tsv_files]
    duplicate_recording_ids = sorted(
        {recording_id for recording_id in wav_ids if wav_ids.count(recording_id) > 1}
    )
    report["duplicate_recording_ids"] = duplicate_recording_ids
    if duplicate_recording_ids:
        add_error("duplicate_recording_ids", duplicate_recording_ids[:20])
    if len({path.resolve() for path in wav_files}) != len(wav_files):
        add_error("duplicate_wav_paths", "resolved WAV paths are not unique")
    if len(wav_files) != expected_recordings:
        add_error("unexpected_recording_count", f"found {len(wav_files)}")

    missing_tsv = sorted(set(wav_ids) - set(tsv_ids))
    orphan_tsv = sorted(set(tsv_ids) - set(wav_ids))
    report["wav_missing_tsv"] = missing_tsv
    report["tsv_missing_wav"] = orphan_tsv
    if missing_tsv:
        add_error("wav_missing_tsv", missing_tsv[:20])
    if orphan_tsv:
        add_error("tsv_missing_wav", orphan_tsv[:20])

    metadata: pd.DataFrame | None = None
    if metadata_path.is_file():
        try:
            metadata = pd.read_csv(metadata_path, dtype={"Patient ID": str})
            report["patients"] = len(metadata)
            if "Patient ID" not in metadata.columns or "Locations" not in metadata.columns:
                add_error("invalid_metadata_columns", list(metadata.columns))
            elif metadata["Patient ID"].duplicated().any():
                duplicate_patients = metadata.loc[
                    metadata["Patient ID"].duplicated(keep=False), "Patient ID"
                ].tolist()
                add_error("duplicate_patient_ids", duplicate_patients[:20])
            if len(metadata) != expected_patients:
                add_error("unexpected_patient_count", f"found {len(metadata)}")
        except Exception as exc:  # report malformed CSV rather than aborting early
            add_error("unreadable_metadata", repr(exc))
    else:
        report["patients"] = 0

    unreadable_wav: list[str] = []
    empty_signals: list[str] = []
    invalid_sample_rates: list[str] = []
    sample_rate_counts: dict[str, int] = {}
    invalid_annotations: list[dict[str, str]] = []
    durations: dict[str, float] = {}

    if inspect_audio:
        for path in wav_files:
            try:
                sample_rate, signal = wavfile.read(path, mmap=True)
                rate_key = str(int(sample_rate))
                sample_rate_counts[rate_key] = sample_rate_counts.get(rate_key, 0) + 1
                if sample_rate != 4000:
                    invalid_sample_rates.append(f"{path.name}: {sample_rate} Hz")
                if signal.size == 0:
                    empty_signals.append(path.name)
                durations[path.stem] = signal.shape[0] / float(sample_rate)
            except Exception as exc:
                unreadable_wav.append(f"{path.name}: {exc}")
    else:
        warnings.append(
            {"code": "audio_inspection_skipped", "detail": "WAV headers not inspected"}
        )

    for path in tsv_files:
        try:
            table = _read_annotation(path)
            values = table[["start", "end", "state"]].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError("contains NaN or infinite values")
            if not np.allclose(table["state"], np.round(table["state"])):
                raise ValueError("contains non-integer cardiac states")
            invalid_duration = (table["end"] < table["start"]) | (
                (table["end"] == table["start"]) & (table["state"] != 0)
            )
            if (table["start"] < 0).any() or invalid_duration.any():
                raise ValueError("contains negative, zero-length, or reversed intervals")
            states = set(table["state"].astype(int).tolist())
            if not states.issubset(VALID_CARDIAC_STATES):
                raise ValueError(f"invalid cardiac states {sorted(states)}")
            missing_states = REQUIRED_CARDIAC_STATES - states
            if missing_states:
                raise ValueError(f"missing cardiac states {sorted(missing_states)}")
            duration = durations.get(path.stem)
            if duration is not None and float(table["end"].max()) > duration + 1e-3:
                raise ValueError(
                    f"annotation ends at {table['end'].max():.6f}s beyond {duration:.6f}s"
                )
            starts = table["start"].to_numpy()
            ends = table["end"].to_numpy()
            # Public annotations contain sub-millisecond boundary rounding.
            if len(table) > 1 and np.any(
                starts[1:]
                < ends[:-1] - ANNOTATION_BOUNDARY_TOLERANCE_SECONDS
            ):
                raise ValueError("annotation intervals overlap or are out of order")
        except Exception as exc:
            invalid_annotations.append({"file": path.name, "detail": str(exc)})

    report.update(
        {
            "unreadable_wav_files": unreadable_wav,
            "empty_signals": empty_signals,
            "invalid_sample_rates": invalid_sample_rates,
            "sample_rate_counts": sample_rate_counts,
            "invalid_annotations": invalid_annotations,
        }
    )
    for code, values in (
        ("unreadable_wav_files", unreadable_wav),
        ("empty_signals", empty_signals),
        ("invalid_sample_rates", invalid_sample_rates),
        ("invalid_annotations", invalid_annotations),
    ):
        if values:
            add_error(code, values[:20])

    if metadata is not None and {"Patient ID", "Locations"}.issubset(metadata.columns):
        references = _metadata_references(metadata)
        recording_references = {
            _metadata_reference_for_recording(recording_id) for recording_id in wav_ids
        }
        missing_references = sorted(references - recording_references)
        unreferenced_recordings = sorted(
            recording_id
            for recording_id in wav_ids
            if _metadata_reference_for_recording(recording_id) not in references
        )
        report["metadata_recording_references"] = len(references)
        report["metadata_references_missing_wav"] = missing_references
        report["wav_not_referenced_by_metadata"] = unreferenced_recordings
        if missing_references:
            add_error("metadata_references_missing_wav", missing_references[:20])
        if unreferenced_recordings:
            warnings.append(
                {
                    "code": "wav_not_referenced_by_metadata",
                    "detail": str(unreferenced_recordings[:20]),
                }
            )

    report["valid"] = not errors
    destination = report_path
    if destination is None and root == DATASET_ROOT.resolve():
        validate_input_output_isolation(root, REPORT_OUTPUT_DIR)
        ensure_output_directories()
        destination = REPORT_OUTPUT_DIR / "dataset_validation.json"
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(destination)

    if strict and errors:
        raise DatasetValidationError(
            f"CirCor validation failed with {len(errors)} error group(s); "
            f"see {destination or 'returned report'}"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--skip-audio-inspection", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args(argv)
    root = args.dataset_root.expanduser().resolve()
    report = validate_dataset(
        DatasetPaths(root, root / "training_data", root / "training_data.csv"),
        strict=False,
        inspect_audio=not args.skip_audio_inspection,
    )
    print(f"Dataset root: {report['dataset_root']}")
    print(f"Audio directory: {report['audio_directory']}")
    print(f"WAV recordings: {report['wav_recordings']}")
    print(f"TSV annotations: {report['tsv_annotations']}")
    print(f"Patients: {report['patients']}")
    if not report["wav_missing_tsv"]:
        print("All WAV recordings have corresponding TSV files.")
    print(f"Validation report: {report.get('report_path', 'not written')}")
    if report["errors"] and not args.no_strict:
        print(f"Validation failed with {len(report['errors'])} error group(s):")
        for error in report["errors"]:
            print(f"- {error['code']}: {error['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
