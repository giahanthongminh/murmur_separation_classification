"""Generate auditable real-CirCor murmur-isolation diagnostic packages."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import hilbert, resample_poly, spectrogram

from config import (
    AUDIO_DIR,
    DEFAULT_SEPARATION_CONFIG,
    METADATA_PATH,
    REPORT_OUTPUT_DIR,
    SEPARATION_OUTPUT_DIR,
    SeparationConfig,
    ensure_output_directories,
)
from src.data_validation import validate_dataset
from src.separation.core import (
    SeparationResult,
    compare_separation_methods,
    separate_signal,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _load_wav(path: Path, target_rate: int) -> tuple[np.ndarray, int]:
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


def _representative_recordings(metadata: pd.DataFrame, limit: int) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = {
        "Absent": [],
        "Early-systolic": [],
        "Holosystolic": [],
        "Other timing": [],
    }
    for _, row in metadata.iterrows():
        patient_id = str(row["Patient ID"])
        murmur = str(row.get("Murmur", ""))
        timing = str(row.get("Systolic murmur timing", ""))
        if murmur == "Absent":
            group = "Absent"
        elif timing == "Early-systolic":
            group = "Early-systolic"
        elif timing == "Holosystolic":
            group = "Holosystolic"
        else:
            group = "Other timing"
        locations = str(row.get("Locations", "")).split("+")
        preferred = str(row.get("Most audible location", ""))
        location = preferred if preferred in locations else locations[0]
        recording_id = f"{patient_id}_{location}"
        if (AUDIO_DIR / f"{recording_id}.wav").exists():
            groups[group].append(
                {
                    "patient_id": patient_id,
                    "recording_id": recording_id,
                    "location": location,
                    "murmur_label": murmur,
                    "timing_label": timing,
                    "audit_group": group,
                }
            )
    selected: list[dict[str, str]] = []
    while len(selected) < limit and any(groups.values()):
        for group in groups:
            if groups[group] and len(selected) < limit:
                selected.append(groups[group].pop(0))
    return selected


def _write_diagnostic_plot(
    result: SeparationResult,
    destination: Path,
    sample_rate: int,
    metadata: dict[str, Any],
) -> None:
    time = np.arange(len(result.original)) / sample_rate
    figure, axes = plt.subplots(5, 2, figsize=(15, 18))
    axes = axes.ravel()
    axes[0].plot(time, result.original, linewidth=0.7)
    axes[0].set_title("Original systolic segment")
    axes[1].plot(time, result.original, linewidth=0.7)
    axes[1].axvspan(time[0], time[-1], alpha=0.18, color="tab:blue", label="systole")
    axes[1].legend(loc="upper right")
    axes[1].set_title("TSV cardiac phase")
    axes[2].plot(time, result.normal_estimate, linewidth=0.7)
    axes[2].set_title("Normal-heart estimate")
    axes[3].plot(time, result.murmur_candidate, linewidth=0.7)
    axes[3].set_title("Murmur candidate")
    axes[4].plot(time, result.noise_candidate, linewidth=0.7)
    axes[4].set_title("Noise/artifact candidate")
    axes[5].plot(time, np.abs(hilbert(result.murmur_candidate)), linewidth=0.7)
    axes[5].set_title("Murmur-candidate amplitude envelope")
    axes[6].psd(result.murmur_candidate, Fs=sample_rate, NFFT=min(256, len(time)))
    axes[6].set_title("Murmur-candidate PSD")
    frequencies, times, power = spectrogram(
        result.murmur_candidate,
        fs=sample_rate,
        nperseg=min(128, len(result.murmur_candidate)),
    )
    axes[7].pcolormesh(times, frequencies, 10 * np.log10(power + 1e-12), shading="auto")
    axes[7].set_ylim(0, min(1000, sample_rate / 2))
    axes[7].set_title("Murmur-candidate spectrogram")
    assignments = result.assignments
    axes[8].bar(
        [row["component_index"] for row in result.component_features],
        [row["relative_energy"] for row in result.component_features],
        color=[
            "tab:green"
            if row["assignment"] == "normal"
            else "tab:red"
            if row["assignment"] == "murmur_candidate"
            else "tab:gray"
            for row in result.component_features
        ],
    )
    axes[8].set_title("SSA component energy and assignment")
    axes[8].set_xlabel("component")
    axes[9].axis("off")
    metric_lines = [
        f"{key}: {value:.5g}" if isinstance(value, float) else f"{key}: {value}"
        for key, value in result.metrics.items()
        if key
        in {
            "reconstruction_error",
            "normal_residual_correlation",
            "murmur_region_energy_retention",
            "noise_energy_ratio",
            "onset_normalized",
            "offset_normalized",
            "selection_score",
        }
    ]
    def preview(indexes: list[int]) -> str:
        suffix = "..." if len(indexes) > 12 else ""
        return f"{indexes[:12]}{suffix}"

    axes[9].text(
        0,
        1,
        "\n".join(
            [
                metadata["recording_id"],
                f"method: {result.selected_method}",
                f"normal components: {preview(assignments['normal'])}",
                f"murmur components: {preview(assignments['murmur_candidate'])}",
                f"noise components: {preview(assignments['noise_artifact'])}",
                *metric_lines,
            ]
        ),
        va="top",
        family="monospace",
        fontsize=9,
    )
    for axis in axes[:6]:
        axis.set_xlabel("seconds")
    axes[6].set_xlabel("frequency (Hz)")
    axes[7].set_xlabel("seconds")
    figure.tight_layout(pad=1.2)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def _save_segment_package(
    result: SeparationResult,
    directory: Path,
    metadata: dict[str, Any],
    config: SeparationConfig,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, values in (
        ("original.npy", result.original),
        ("normal_estimate.npy", result.normal_estimate),
        ("murmur_candidate.npy", result.murmur_candidate),
        ("noise_candidate.npy", result.noise_candidate),
    ):
        np.save(directory / filename, values.astype(np.float32))
        wavfile.write(
            directory / filename.replace(".npy", ".wav"),
            config.sample_rate,
            values.astype(np.float32),
        )
    pd.DataFrame(result.component_features).to_csv(
        directory / "component_features.csv", index=False
    )
    selection = {
        "selected_method": result.selected_method,
        "assignments": result.assignments,
        "config_hash": result.config_hash,
        "configuration": config.to_dict(),
    }
    (directory / "selected_components.json").write_text(
        json.dumps(selection, indent=2, default=_json_value), encoding="utf-8"
    )
    metrics = {**metadata, **result.metrics}
    (directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_value), encoding="utf-8"
    )
    _write_diagnostic_plot(
        result, directory / "diagnostic_plot.png", config.sample_rate, metadata
    )
    return metrics


def run_audit(
    *,
    limit: int = 20,
    cycles_per_recording: int = 1,
    config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
    method: str = "auto",
    validate_first: bool = True,
) -> pd.DataFrame:
    ensure_output_directories()
    if validate_first:
        validate_dataset()
    metadata_table = pd.read_csv(METADATA_PATH, dtype={"Patient ID": str})
    recordings = _representative_recordings(metadata_table, limit)
    summary_rows: list[dict[str, Any]] = []
    for recording in recordings:
        recording_id = recording["recording_id"]
        signal, sample_rate = _load_wav(
            AUDIO_DIR / f"{recording_id}.wav", config.sample_rate
        )
        annotations = pd.read_csv(
            AUDIO_DIR / f"{recording_id}.tsv",
            sep="\t",
            header=None,
            names=["start", "end", "state"],
        )
        systoles = annotations[annotations["state"] == 2].head(cycles_per_recording)
        for cycle_index, (_, annotation) in enumerate(systoles.iterrows()):
            start_sample = int(round(float(annotation["start"]) * sample_rate))
            end_sample = int(round(float(annotation["end"]) * sample_rate))
            segment = signal[start_sample:end_sample]
            if len(segment) < 8:
                continue
            if method == "auto":
                result, _ = compare_separation_methods(segment, config=config)
            else:
                result = separate_signal(segment, config=config, method=method)
            segment_metadata: dict[str, Any] = {
                **recording,
                "cycle_index": cycle_index,
                "phase": "systole",
                "start_sample": start_sample,
                "end_sample": end_sample,
                "sample_rate": sample_rate,
            }
            directory = (
                SEPARATION_OUTPUT_DIR
                / recording_id
                / f"cycle_{cycle_index}_systole"
            )
            summary_rows.append(
                _save_segment_package(result, directory, segment_metadata, config)
            )
    summary = pd.DataFrame(summary_rows)
    destination = REPORT_OUTPUT_DIR / "separation_summary.csv"
    summary.to_csv(destination, index=False)
    (REPORT_OUTPUT_DIR / "separation_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--cycles-per-recording", type=int, default=1)
    parser.add_argument("--method", choices=["auto", "zcr", "kurtosis"], default="auto")
    parser.add_argument("--energy-threshold", type=float, default=0.99)
    parser.add_argument("--use-dwt", action="store_true")
    parser.add_argument("--skip-dataset-validation", action="store_true")
    args = parser.parse_args(argv)
    config = replace(
        DEFAULT_SEPARATION_CONFIG,
        explained_energy_threshold=args.energy_threshold,
        use_dwt=args.use_dwt,
    )
    summary = run_audit(
        limit=args.limit,
        cycles_per_recording=args.cycles_per_recording,
        config=config,
        method=args.method,
        validate_first=not args.skip_dataset_validation,
    )
    print(f"Processed segments: {len(summary)}")
    print(f"Summary: {REPORT_OUTPUT_DIR / 'separation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
