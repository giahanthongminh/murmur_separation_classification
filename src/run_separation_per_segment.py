"""Batch per-systolic-segment separation for downstream legacy experiments.

For diagnostic packages and representative-case review, prefer
``python -m src.separation.audit``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly

from config import (
    AUDIO_DIR,
    DEFAULT_SEPARATION_CONFIG,
    LABELS_PATH,
    SEPARATION_OUTPUT_DIR,
    SeparationConfig,
    ensure_output_directories,
)
from src.data_validation import validate_dataset
from src.separation.core import SeparationResult, compare_separation_methods


MIN_SEG_SAMPLES = 200


def get_systolic_segments(tsv_path: Path, sample_rate: int):
    annotations = pd.read_csv(
        tsv_path, sep="\t", header=None, names=["start", "end", "state"]
    )
    systoles = annotations[annotations["state"] == 2]
    return [
        (
            float(row["start"]),
            float(row["end"]),
            int(round(float(row["start"]) * sample_rate)),
            int(round(float(row["end"]) * sample_rate)),
        )
        for _, row in systoles.iterrows()
    ]


def _load_signal(path: Path, target_rate: int) -> tuple[np.ndarray, int]:
    sample_rate, signal = wavfile.read(path)
    values = np.asarray(signal)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        values = values.astype(float) / max(abs(info.min), info.max)
    else:
        values = values.astype(float)
    if sample_rate != target_rate:
        common = int(np.gcd(sample_rate, target_rate))
        values = resample_poly(values, target_rate // common, sample_rate // common)
        sample_rate = target_rate
    return values, sample_rate


def separate_segment(
    segment: np.ndarray,
    config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
) -> SeparationResult | None:
    if len(segment) < MIN_SEG_SAMPLES:
        return None
    result, _ = compare_separation_methods(segment, config=config)
    return result


def process_file(
    wav_path: Path,
    config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
    *,
    overwrite: bool = False,
) -> str | None:
    recording_id = wav_path.stem
    destination = SEPARATION_OUTPUT_DIR / recording_id
    metadata_path = destination / "segments_meta.json"
    if metadata_path.exists() and not overwrite:
        return recording_id
    tsv_path = AUDIO_DIR / f"{recording_id}.tsv"
    if not tsv_path.exists():
        return None
    signal, sample_rate = _load_signal(wav_path, config.sample_rate)
    segments = get_systolic_segments(tsv_path, sample_rate)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for cycle_index, (start, end, start_sample, end_sample) in enumerate(segments):
        result = separate_segment(signal[start_sample:end_sample], config)
        if result is None:
            continue
        for name, values in (
            ("original", result.original),
            ("normal_estimate", result.normal_estimate),
            ("murmur_candidate", result.murmur_candidate),
            ("noise_candidate", result.noise_candidate),
        ):
            np.save(destination / f"seg_{cycle_index}_{name}.npy", values.astype(np.float32))
        row = {
            "patient_id": recording_id.split("_", 1)[0],
            "recording_id": recording_id,
            "location": recording_id.rsplit("_", 1)[-1],
            "cycle_index": cycle_index,
            "phase": "systole",
            "start_seconds": start,
            "end_seconds": end,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "sample_rate": sample_rate,
            "selected_method": result.selected_method,
            "config_hash": result.config_hash,
            **result.metrics,
        }
        rows.append(row)
    if not rows:
        return None
    metadata_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (destination / "run_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    return recording_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--use-dwt", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-dataset-validation", action="store_true")
    args = parser.parse_args(argv)
    ensure_output_directories()
    if not args.skip_dataset_validation:
        validate_dataset()
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Derived labels not found at {LABELS_PATH}; run python -m src.prepare_labels"
        )
    config = replace(DEFAULT_SEPARATION_CONFIG, use_dwt=args.use_dwt)
    labels = pd.read_csv(LABELS_PATH, dtype={"Patient ID": str})
    patient_ids = set(labels["Patient ID"])
    recordings = sorted(
        path
        for path in AUDIO_DIR.glob("*.wav")
        if path.stem.split("_", 1)[0] in patient_ids
    )
    if args.limit is not None:
        recordings = recordings[: args.limit]
    completed = sum(
        process_file(path, config, overwrite=args.overwrite) is not None
        for path in recordings
    )
    print(f"Processed recordings: {completed}/{len(recordings)}")
    print(f"Output: {SEPARATION_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
