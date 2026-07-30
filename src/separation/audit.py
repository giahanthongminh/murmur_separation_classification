"""Generate auditable real-CirCor murmur-isolation diagnostic packages."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
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
    ANNOTATION_BOUNDARY_TOLERANCE_SECONDS,
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
from src.separation.metrics import real_proxy_metrics


@dataclass(frozen=True)
class CardiacCycleContext:
    """One contiguous S1-systole-S2-diastole cycle and its phase masks."""

    signal: np.ndarray
    phase_masks: dict[str, np.ndarray]
    phase_bounds: dict[str, tuple[int, int]]
    context_start_sample: int
    context_end_sample: int


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


def _annotation_sample_bounds(
    annotation: pd.Series, sample_rate: int, signal_length: int
) -> tuple[int, int]:
    start = int(round(float(annotation["start"]) * sample_rate))
    end = int(round(float(annotation["end"]) * sample_rate))
    return max(0, min(start, signal_length)), max(0, min(end, signal_length))


def build_cardiac_cycle_context(
    signal: np.ndarray,
    annotations: pd.DataFrame,
    systole_position: int,
    sample_rate: int,
) -> CardiacCycleContext:
    """Extract a contiguous S1-systole-S2-diastole context around one systole."""

    table = annotations.reset_index(drop=True)
    required_positions = {
        "s1": systole_position - 1,
        "systole": systole_position,
        "s2": systole_position + 1,
        "diastole": systole_position + 2,
    }
    expected_states = {"s1": 1, "systole": 2, "s2": 3, "diastole": 4}
    for phase, position in required_positions.items():
        if not 0 <= position < len(table):
            raise ValueError(f"incomplete cardiac-cycle context: missing {phase}")
        state = int(table.iloc[position]["state"])
        if state != expected_states[phase]:
            raise ValueError(
                f"invalid cardiac-cycle sequence at {phase}: expected "
                f"{expected_states[phase]}, found {state}"
            )

    absolute_bounds: dict[str, tuple[int, int]] = {
        phase: _annotation_sample_bounds(table.iloc[position], sample_rate, len(signal))
        for phase, position in required_positions.items()
    }
    ordered_phases = ("s1", "systole", "s2", "diastole")
    for left_phase, right_phase in zip(ordered_phases, ordered_phases[1:]):
        left_annotation = table.iloc[required_positions[left_phase]]
        right_annotation = table.iloc[required_positions[right_phase]]
        left_end_seconds = float(left_annotation["end"])
        right_start_seconds = float(right_annotation["start"])
        boundary_delta = right_start_seconds - left_end_seconds
        if abs(boundary_delta) > ANNOTATION_BOUNDARY_TOLERANCE_SECONDS:
            relation = "gap" if boundary_delta > 0 else "overlap"
            raise ValueError(
                f"cardiac phase boundary {left_phase}->{right_phase} has "
                f"{abs(boundary_delta):.6f}s {relation}, exceeding "
                f"{ANNOTATION_BOUNDARY_TOLERANCE_SECONDS:.6f}s tolerance"
            )
        shared_boundary = int(
            round(
                (left_end_seconds + right_start_seconds)
                * 0.5
                * sample_rate
            )
        )
        shared_boundary = max(0, min(shared_boundary, len(signal)))
        left_start, _ = absolute_bounds[left_phase]
        _, right_end = absolute_bounds[right_phase]
        absolute_bounds[left_phase] = (left_start, shared_boundary)
        absolute_bounds[right_phase] = (shared_boundary, right_end)
    context_start = absolute_bounds["s1"][0]
    context_end = absolute_bounds["diastole"][1]
    if context_end <= context_start:
        raise ValueError("cardiac-cycle context is empty or reversed")
    context = np.asarray(signal[context_start:context_end], dtype=float)
    phase_bounds: dict[str, tuple[int, int]] = {}
    phase_masks: dict[str, np.ndarray] = {}
    for phase, (absolute_start, absolute_end) in absolute_bounds.items():
        start = max(0, absolute_start - context_start)
        end = min(len(context), absolute_end - context_start)
        if end <= start:
            raise ValueError(f"cardiac phase '{phase}' is empty after clipping")
        mask = np.zeros(len(context), dtype=bool)
        mask[start:end] = True
        phase_bounds[phase] = (start, end)
        phase_masks[phase] = mask
    return CardiacCycleContext(
        signal=context,
        phase_masks=phase_masks,
        phase_bounds=phase_bounds,
        context_start_sample=context_start,
        context_end_sample=context_end,
    )


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
    axes[0].set_title("Original complete cardiac-cycle context")
    axes[1].plot(time, result.original, linewidth=0.7)
    phase_colors = {
        "s1": "tab:green",
        "systole": "tab:blue",
        "s2": "tab:orange",
        "diastole": "tab:purple",
    }
    for phase, color in phase_colors.items():
        start = metadata[f"{phase}_relative_start_sample"] / sample_rate
        end = metadata[f"{phase}_relative_end_sample"] / sample_rate
        axes[1].axvspan(start, end, alpha=0.18, color=color, label=phase)
    axes[1].legend(loc="upper right")
    axes[1].set_title("TSV cardiac phases")
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
            "s1_leakage_ratio",
            "s2_leakage_ratio",
            "outside_murmur_energy_ratio",
            "noise_energy_ratio",
            "onset_normalized",
            "offset_normalized",
            "selection_score",
            "phase_selected_component_count",
            "phase_rejected_component_count",
            "phase_selection_used_fallback",
            "candidate_quality_status",
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


def classify_audit_outcome(murmur_label: str, quality_status: str) -> str:
    """Interpret selector confidence against the recording-level murmur label."""

    accepted = quality_status == "accepted"
    if murmur_label == "Present":
        return "present_candidate_accepted" if accepted else "present_candidate_missed"
    if murmur_label == "Absent":
        return "absent_candidate_flagged" if accepted else "absent_negative_control_clear"
    return "unscored_unknown_label"


def summarize_audit_quality(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accepted and low-confidence candidates without mixing them."""

    columns = [
        "candidate_quality_status",
        "murmur_label",
        "audit_outcome",
        "segment_count",
        "s1_leakage_ratio",
        "s2_leakage_ratio",
        "outside_murmur_energy_ratio",
        "murmur_region_energy_retention",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    required = {"candidate_quality_status", "murmur_label"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"audit summary is missing quality columns: {sorted(missing)}")
    metrics = [
        "s1_leakage_ratio",
        "s2_leakage_ratio",
        "outside_murmur_energy_ratio",
        "murmur_region_energy_retention",
    ]
    table = summary.copy()
    if "audit_outcome" not in table:
        table["audit_outcome"] = [
            classify_audit_outcome(str(label), str(status))
            for label, status in zip(
                table["murmur_label"], table["candidate_quality_status"]
            )
        ]
    grouped = table.groupby(
        ["candidate_quality_status", "murmur_label", "audit_outcome"],
        dropna=False,
        sort=True,
    )
    counts = grouped.size().rename("segment_count")
    means = grouped[metrics].mean()
    return pd.concat([counts, means], axis=1).reset_index()[columns]


def run_audit(
    *,
    limit: int = 20,
    cycles_per_recording: int = 1,
    config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
    method: str = "auto",
    validate_first: bool = True,
    run_name: str | None = None,
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
        systole_positions = np.flatnonzero(annotations["state"].to_numpy() == 2)[
            :cycles_per_recording
        ]
        for cycle_index, systole_position in enumerate(systole_positions):
            try:
                cycle = build_cardiac_cycle_context(
                    signal, annotations, int(systole_position), sample_rate
                )
            except ValueError as exc:
                print(f"Skipping {recording_id} cycle {cycle_index}: {exc}")
                continue
            if method == "auto":
                result, _ = compare_separation_methods(
                    cycle.signal, config=config, phase_masks=cycle.phase_masks
                )
            else:
                result = separate_signal(
                    cycle.signal,
                    config=config,
                    method=method,
                    phase_masks=cycle.phase_masks,
                )
            result.metrics.update(
                real_proxy_metrics(
                    result.original,
                    result.normal_estimate,
                    result.murmur_candidate,
                    result.noise_candidate,
                    sample_rate,
                    threshold_mad=config.onset_threshold_mad,
                    minimum_duration_ms=config.minimum_interval_duration_ms,
                    merge_gap_ms=config.gap_merging_duration_ms,
                    phase_masks=cycle.phase_masks,
                )
            )
            segment_metadata: dict[str, Any] = {
                **recording,
                "cycle_index": cycle_index,
                "phase": "complete_cycle",
                "context_start_sample": cycle.context_start_sample,
                "context_end_sample": cycle.context_end_sample,
                "sample_rate": sample_rate,
            }
            for phase, (relative_start, relative_end) in cycle.phase_bounds.items():
                segment_metadata[f"{phase}_relative_start_sample"] = relative_start
                segment_metadata[f"{phase}_relative_end_sample"] = relative_end
                segment_metadata[f"{phase}_absolute_start_sample"] = (
                    cycle.context_start_sample + relative_start
                )
                segment_metadata[f"{phase}_absolute_end_sample"] = (
                    cycle.context_start_sample + relative_end
                )
            segment_metadata["audit_outcome"] = classify_audit_outcome(
                str(recording["murmur_label"]),
                str(result.metrics["candidate_quality_status"]),
            )
            output_root = (
                SEPARATION_OUTPUT_DIR / run_name
                if run_name
                else SEPARATION_OUTPUT_DIR
            )
            directory = (
                output_root
                / recording_id
                / f"cycle_{cycle_index}_context"
            )
            summary_rows.append(
                _save_segment_package(result, directory, segment_metadata, config)
            )
    summary = pd.DataFrame(summary_rows)
    suffix = f"_{run_name}" if run_name else ""
    destination = REPORT_OUTPUT_DIR / f"separation_summary{suffix}.csv"
    summary.to_csv(destination, index=False)
    accepted = summary[
        summary.get("candidate_quality_status", pd.Series(dtype=str)).eq("accepted")
    ]
    accepted.to_csv(
        REPORT_OUTPUT_DIR / f"separation_summary{suffix}_accepted.csv", index=False
    )
    present_accepted = accepted[accepted.get("murmur_label", "").eq("Present")]
    present_accepted.to_csv(
        REPORT_OUTPUT_DIR
        / f"separation_summary{suffix}_present_accepted.csv",
        index=False,
    )
    summarize_audit_quality(summary).to_csv(
        REPORT_OUTPUT_DIR / f"separation_quality{suffix}.csv", index=False
    )
    (REPORT_OUTPUT_DIR / f"separation_config{suffix}.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--cycles-per-recording", type=int, default=1)
    parser.add_argument("--method", choices=["auto", "zcr", "kurtosis"], default="auto")
    parser.add_argument("--energy-threshold", type=float, default=0.99)
    parser.add_argument(
        "--ssa-window-length",
        type=int,
        default=DEFAULT_SEPARATION_CONFIG.ssa_window_length,
    )
    parser.add_argument("--use-dwt", action="store_true")
    parser.add_argument("--disable-phase-aware-selection", action="store_true")
    parser.add_argument(
        "--minimum-systole-focus",
        type=float,
        default=DEFAULT_SEPARATION_CONFIG.minimum_systole_focus,
    )
    parser.add_argument(
        "--minimum-systole-to-s1-s2-ratio",
        type=float,
        default=DEFAULT_SEPARATION_CONFIG.minimum_systole_to_s1_s2_ratio,
    )
    parser.add_argument("--disable-phase-selection-fallback", action="store_true")
    parser.add_argument("--skip-dataset-validation", action="store_true")
    parser.add_argument(
        "--run-name",
        help="Optional label used to isolate outputs and comparison reports",
    )
    args = parser.parse_args(argv)
    config = replace(
        DEFAULT_SEPARATION_CONFIG,
        explained_energy_threshold=args.energy_threshold,
        ssa_window_length=args.ssa_window_length,
        use_dwt=args.use_dwt,
        phase_aware_component_selection=not args.disable_phase_aware_selection,
        minimum_systole_focus=args.minimum_systole_focus,
        minimum_systole_to_s1_s2_ratio=args.minimum_systole_to_s1_s2_ratio,
        phase_selection_fallback=not args.disable_phase_selection_fallback,
    )
    summary = run_audit(
        limit=args.limit,
        cycles_per_recording=args.cycles_per_recording,
        config=config,
        method=args.method,
        validate_first=not args.skip_dataset_validation,
        run_name=args.run_name,
    )
    print(f"Processed segments: {len(summary)}")
    if "candidate_quality_status" in summary:
        counts = summary["candidate_quality_status"].value_counts().to_dict()
        print(f"Candidate quality: {counts}")
    suffix = f"_{args.run_name}" if args.run_name else ""
    print(f"Summary: {REPORT_OUTPUT_DIR / f'separation_summary{suffix}.csv'}")
    print(
        "Accepted summary: "
        f"{REPORT_OUTPUT_DIR / f'separation_summary{suffix}_accepted.csv'}"
    )
    print(
        "Present accepted summary: "
        f"{REPORT_OUTPUT_DIR / f'separation_summary{suffix}_present_accepted.csv'}"
    )
    print(f"Quality report: {REPORT_OUTPUT_DIR / f'separation_quality{suffix}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
