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
from src.separation.metrics import (
    dominant_frequency_trajectory,
    real_proxy_metrics,
    smooth_amplitude_envelope,
)


OBSERVATION_METRICS = [
    "onset_normalized",
    "offset_normalized",
    "duration_ratio",
    "temporal_energy_centroid",
    "peak_position",
    "murmur_onset_systole_seconds",
    "murmur_offset_systole_seconds",
    "murmur_duration_seconds",
    "murmur_onset_cycle_seconds",
    "murmur_offset_cycle_seconds",
    "murmur_onset_recording_seconds",
    "murmur_offset_recording_seconds",
    "amplitude_peak_abs",
    "amplitude_rms",
    "amplitude_mean_abs",
    "amplitude_envelope_peak",
    "amplitude_envelope_mean",
    "amplitude_crest_factor",
    "envelope_shape",
    "envelope_time_to_peak_ratio",
    "envelope_rise_time_seconds",
    "envelope_decay_time_seconds",
    "envelope_rising_slope_normalized",
    "envelope_falling_slope_normalized",
    "envelope_variation_coefficient",
    "envelope_area_normalized",
    "envelope_symmetry",
    "envelope_prominent_peak_count",
    "active_burst_count",
    "active_time_ratio",
    "longest_burst_ratio",
    "residual_dominant_frequency",
    "residual_spectral_centroid",
    "residual_bandwidth",
    "residual_spectral_entropy",
    "psd_peak_frequency_hz",
    "psd_peak_power",
    "psd_frequency_resolution_hz",
    "psd_0_100_hz_ratio",
    "psd_100_200_hz_ratio",
    "psd_200_400_hz_ratio",
    "psd_400_800_hz_ratio",
    "psd_800_1000_hz_ratio",
    "psd_above_1000_hz_ratio",
    "psd_morphology",
    "psd_primary_peak_frequency_hz",
    "psd_prominent_peak_count",
    "psd_primary_peak_width_hz",
    "psd_primary_peak_prominence_ratio",
    "psd_secondary_peak_frequency_hz",
    "psd_secondary_to_primary_ratio",
    "psd_peak_separation_hz",
    "psd_primary_q_factor",
    "psd_energy_concentration",
    "time_frequency_peak_hz",
    "time_frequency_peak_seconds",
    "time_frequency_peak_cycle_seconds",
    "time_frequency_peak_recording_seconds",
    "time_frequency_frame_count",
    "time_frequency_frequency_resolution_hz",
    "time_frequency_window_seconds",
    "time_frequency_entropy",
    "time_frequency_spectral_flux",
    "time_frequency_ridge_direction",
    "time_frequency_ridge_start_hz",
    "time_frequency_ridge_end_hz",
    "time_frequency_ridge_slope_hz_per_second",
    "time_frequency_ridge_variability_hz",
    "time_frequency_ridge_continuity",
]


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


def _metadata_locations(value: Any) -> set[str]:
    """Parse CirCor's plus-separated location fields without treating NaN as text."""

    if pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split("+") if item.strip()}


def _all_recordings(
    metadata: pd.DataFrame,
    limit: int,
    *,
    audio_dir: Path = AUDIO_DIR,
) -> list[dict[str, str]]:
    """Enumerate every exact WAV/TSV pair with conservative location-aware labels."""

    patient_rows = {
        str(row["Patient ID"]): row for _, row in metadata.iterrows()
    }
    selected: list[dict[str, str]] = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        recording_id = wav_path.stem
        if not (audio_dir / f"{recording_id}.tsv").exists():
            continue
        parts = recording_id.split("_")
        if len(parts) < 2 or parts[0] not in patient_rows:
            continue
        patient_id = parts[0]
        location = parts[1]
        row = patient_rows[patient_id]
        patient_label = str(row.get("Murmur", "Unknown"))
        murmur_locations = _metadata_locations(row.get("Murmur locations"))
        if patient_label == "Absent":
            murmur_label = "Absent"
        elif patient_label == "Present" and location in murmur_locations:
            murmur_label = "Present"
        else:
            # A patient-level Present label does not prove that every auscultation
            # location contains murmur. Keep those recordings out of scored groups.
            murmur_label = "Unknown"
        timing = str(row.get("Systolic murmur timing", ""))
        selected.append(
            {
                "patient_id": patient_id,
                "recording_id": recording_id,
                "location": location,
                "murmur_label": murmur_label,
                "patient_murmur_label": patient_label,
                "timing_label": timing,
                "audit_group": "All recordings",
            }
        )
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def _absolute_timing_metrics(
    metrics: dict[str, Any],
    cycle: CardiacCycleContext,
    sample_rate: int,
) -> dict[str, Any]:
    """Convert systole-relative detector output into cycle and recording seconds."""

    onset = metrics.get("onset_sample")
    offset = metrics.get("offset_sample")
    quality = str(metrics.get("candidate_quality_status", "fallback"))
    detection = str(metrics.get("activity_detection_method", "silent_or_invalid"))
    if onset is None or offset is None:
        return {
            "murmur_onset_systole_seconds": None,
            "murmur_offset_systole_seconds": None,
            "murmur_duration_seconds": None,
            "murmur_onset_cycle_seconds": None,
            "murmur_offset_cycle_seconds": None,
            "murmur_onset_recording_seconds": None,
            "murmur_offset_recording_seconds": None,
            "time_frequency_peak_cycle_seconds": None,
            "time_frequency_peak_recording_seconds": None,
            "timing_quality_status": "unavailable",
        }
    systole_start = cycle.phase_bounds["systole"][0]
    onset = int(onset)
    offset = int(offset)
    time_frequency_peak = float(metrics.get("time_frequency_peak_seconds", 0.0))
    if quality != "accepted":
        timing_status = "low_confidence_candidate"
    elif detection == "adaptive_envelope":
        timing_status = "accepted_adaptive"
    else:
        timing_status = "accepted_energy_fallback"
    return {
        "murmur_onset_systole_seconds": onset / sample_rate,
        "murmur_offset_systole_seconds": offset / sample_rate,
        "murmur_duration_seconds": (offset - onset) / sample_rate,
        "murmur_onset_cycle_seconds": (systole_start + onset) / sample_rate,
        "murmur_offset_cycle_seconds": (systole_start + offset) / sample_rate,
        "murmur_onset_recording_seconds": (
            cycle.context_start_sample + systole_start + onset
        )
        / sample_rate,
        "murmur_offset_recording_seconds": (
            cycle.context_start_sample + systole_start + offset
        )
        / sample_rate,
        "time_frequency_peak_cycle_seconds": (
            (systole_start + onset) / sample_rate + time_frequency_peak
        ),
        "time_frequency_peak_recording_seconds": (
            (cycle.context_start_sample + systole_start + onset) / sample_rate
            + time_frequency_peak
        ),
        "timing_quality_status": timing_status,
    }


def _should_save_package(output_profile: str, quality_status: str) -> bool:
    if output_profile == "full":
        return True
    if output_profile == "accepted":
        return quality_status == "accepted"
    if output_profile == "summary":
        return False
    raise ValueError(f"unknown output profile: {output_profile}")


def summarize_murmur_observations(summary: pd.DataFrame) -> pd.DataFrame:
    """Describe accepted Present candidates without promoting them to ground truth."""

    base_columns = [
        "timing_label",
        "activity_detection_method",
        "segment_count",
    ]
    if summary.empty:
        return pd.DataFrame(columns=base_columns)
    required = {"murmur_label", "candidate_quality_status"}
    if not required.issubset(summary.columns):
        return pd.DataFrame(columns=base_columns)
    selected = summary[
        summary["murmur_label"].eq("Present")
        & summary["candidate_quality_status"].eq("accepted")
    ].copy()
    available_metrics = [
        name
        for name in OBSERVATION_METRICS
        if name in selected and pd.api.types.is_numeric_dtype(selected[name])
    ]
    if selected.empty or not available_metrics:
        return pd.DataFrame(columns=base_columns)
    group_columns = ["timing_label", "activity_detection_method"]
    grouped = selected.groupby(group_columns, dropna=False, sort=True)
    result = grouped.size().rename("segment_count").to_frame()
    aggregates = grouped[available_metrics].agg(["mean", "median"])
    aggregates.columns = [
        f"{metric}_{statistic}" for metric, statistic in aggregates.columns
    ]
    result = result.join(aggregates)
    return result.reset_index()


def summarize_murmur_morphology_categories(summary: pd.DataFrame) -> pd.DataFrame:
    """Count rule-based morphology labels among accepted Present candidates."""

    columns = ["timing_label", "dimension", "category", "segment_count", "ratio"]
    required = {"murmur_label", "candidate_quality_status", "timing_label"}
    if summary.empty or not required.issubset(summary.columns):
        return pd.DataFrame(columns=columns)
    selected = summary[
        summary["murmur_label"].eq("Present")
        & summary["candidate_quality_status"].eq("accepted")
    ]
    dimensions = [
        name
        for name in (
            "envelope_shape",
            "psd_morphology",
            "time_frequency_ridge_direction",
        )
        if name in selected
    ]
    rows: list[dict[str, Any]] = []
    for timing_label, group in selected.groupby("timing_label", dropna=False):
        for dimension in dimensions:
            counts = group[dimension].fillna("missing").astype(str).value_counts()
            total = int(counts.sum())
            for category, count in counts.items():
                rows.append(
                    {
                        "timing_label": timing_label,
                        "dimension": dimension,
                        "category": category,
                        "segment_count": int(count),
                        "ratio": float(count / total) if total else 0.0,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _write_checkpoint(rows: list[dict[str, Any]], destination: Path) -> None:
    """Atomically persist completed segments so a long run can safely resume."""

    if not rows:
        return
    temporary = destination.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(destination)


def _write_diagnostic_plot(
    result: SeparationResult,
    destination: Path,
    sample_rate: int,
    metadata: dict[str, Any],
) -> None:
    time = np.arange(len(result.original)) / sample_rate
    onset_seconds = result.metrics.get("murmur_onset_cycle_seconds")
    offset_seconds = result.metrics.get("murmur_offset_cycle_seconds")
    if onset_seconds is not None and offset_seconds is not None:
        observation_start = max(0, int(round(float(onset_seconds) * sample_rate)))
        observation_end = min(
            len(result.murmur_candidate),
            int(round(float(offset_seconds) * sample_rate)),
        )
    else:
        observation_start = 0
        observation_end = len(result.murmur_candidate)
    if observation_end <= observation_start:
        observation_start = 0
        observation_end = len(result.murmur_candidate)
    observation_candidate = result.murmur_candidate[
        observation_start:observation_end
    ]
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
    axes[5].plot(
        time,
        np.abs(hilbert(result.murmur_candidate)),
        linewidth=0.6,
        alpha=0.45,
        label="raw Hilbert envelope",
    )
    observation_time = (
        np.arange(len(observation_candidate)) + observation_start
    ) / sample_rate
    smooth_envelope = smooth_amplitude_envelope(observation_candidate, sample_rate)
    axes[5].plot(
        observation_time,
        smooth_envelope,
        color="tab:red",
        linewidth=1.5,
        label="smoothed detected interval",
    )
    if len(smooth_envelope):
        peak_index = int(np.argmax(smooth_envelope))
        axes[5].scatter(
            observation_time[peak_index],
            smooth_envelope[peak_index],
            color="black",
            s=18,
            zorder=3,
            label="envelope peak",
        )
    axes[5].set_title(
        "Amplitude envelope — "
        f"{result.metrics.get('envelope_shape', 'not characterized')}"
    )
    axes[5].legend(loc="upper right", fontsize=8)
    if onset_seconds is not None and offset_seconds is not None:
        for axis in (axes[3], axes[5]):
            axis.axvline(float(onset_seconds), color="tab:red", linestyle="--")
            axis.axvline(float(offset_seconds), color="tab:red", linestyle="--")
    axes[6].psd(
        observation_candidate,
        Fs=sample_rate,
        NFFT=min(512, len(observation_candidate)),
    )
    primary_frequency = float(
        result.metrics.get("psd_primary_peak_frequency_hz", 0.0) or 0.0
    )
    secondary_frequency = float(
        result.metrics.get("psd_secondary_peak_frequency_hz", 0.0) or 0.0
    )
    if primary_frequency > 0:
        axes[6].axvline(
            primary_frequency,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"primary {primary_frequency:.0f} Hz",
        )
    if secondary_frequency > 0:
        axes[6].axvline(
            secondary_frequency,
            color="tab:orange",
            linestyle=":",
            linewidth=1.0,
            label=f"secondary {secondary_frequency:.0f} Hz",
        )
    axes[6].set_title(
        "Detected interval PSD — "
        f"{result.metrics.get('psd_morphology', 'not characterized')}"
    )
    if primary_frequency > 0:
        axes[6].legend(loc="upper right", fontsize=8)
    frequencies, times, power = spectrogram(
        observation_candidate,
        fs=sample_rate,
        nperseg=min(128, len(observation_candidate)),
    )
    times = times + observation_start / sample_rate
    power_db = 10 * np.log10(power + 1e-12)
    if power.shape[1] == 1:
        half_width = max(
            len(observation_candidate) / sample_rate / 2,
            1 / sample_rate,
        )
        axes[7].imshow(
            power_db,
            origin="lower",
            aspect="auto",
            extent=(
                times[0] - half_width,
                times[0] + half_width,
                frequencies[0],
                frequencies[-1],
            ),
        )
    else:
        axes[7].pcolormesh(times, frequencies, power_db, shading="auto")
    ridge_times, ridge = dominant_frequency_trajectory(
        observation_candidate, sample_rate
    )
    if len(ridge_times):
        axes[7].plot(
            ridge_times + observation_start / sample_rate,
            ridge,
            color="white",
            linewidth=1.2,
            marker=".",
            markersize=2,
            label="dominant-frequency trajectory",
        )
        axes[7].legend(loc="upper right", fontsize=8)
    axes[7].set_ylim(0, min(1000, sample_rate / 2))
    axes[7].set_title(
        "Detected interval spectrogram — frequency "
        f"{result.metrics.get('time_frequency_ridge_direction', 'not characterized')}"
    )
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
            "timing_quality_status",
            "murmur_onset_recording_seconds",
            "murmur_offset_recording_seconds",
            "envelope_shape",
            "envelope_time_to_peak_ratio",
            "active_burst_count",
            "active_time_ratio",
            "psd_morphology",
            "psd_primary_peak_frequency_hz",
            "psd_primary_peak_width_hz",
            "psd_prominent_peak_count",
            "time_frequency_ridge_direction",
            "time_frequency_ridge_slope_hz_per_second",
            "time_frequency_ridge_variability_hz",
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
    all_recordings: bool = False,
    output_profile: str = "full",
    resume: bool = False,
) -> pd.DataFrame:
    if limit < 0:
        raise ValueError("limit must be non-negative; use 0 for no limit")
    if cycles_per_recording < 0:
        raise ValueError(
            "cycles_per_recording must be non-negative; use 0 for all cycles"
        )
    if output_profile not in {"full", "accepted", "summary"}:
        raise ValueError("output_profile must be full, accepted, or summary")
    if resume and not run_name:
        raise ValueError("resume requires a run_name to identify its checkpoint")
    ensure_output_directories()
    if validate_first:
        validate_dataset()
    metadata_table = pd.read_csv(METADATA_PATH, dtype={"Patient ID": str})
    recordings = (
        _all_recordings(metadata_table, limit)
        if all_recordings
        else _representative_recordings(metadata_table, limit)
    )
    suffix = f"_{run_name}" if run_name else ""
    checkpoint_path = REPORT_OUTPUT_DIR / f"separation_checkpoint{suffix}.csv"
    summary_rows: list[dict[str, Any]] = []
    if resume and checkpoint_path.exists():
        checkpoint = pd.read_csv(
            checkpoint_path,
            dtype={
                "recording_id": str,
                "config_hash": str,
                "requested_method": str,
                "recording_scope": str,
                "output_profile": str,
            },
        )
        if "config_hash" in checkpoint and not checkpoint.empty:
            hashes = set(checkpoint["config_hash"].dropna().astype(str))
            if hashes != {config.config_hash}:
                raise ValueError(
                    "checkpoint configuration does not match the requested run"
                )
        expected_identity = {
            "requested_method": method,
            "recording_scope": "all" if all_recordings else "representative",
            "output_profile": output_profile,
        }
        if not checkpoint.empty:
            missing_identity = set(expected_identity) - set(checkpoint.columns)
            if missing_identity:
                raise ValueError(
                    "checkpoint is missing run identity fields; choose a new run_name"
                )
            for column, expected in expected_identity.items():
                observed = set(checkpoint[column].dropna().astype(str))
                if observed != {str(expected)}:
                    raise ValueError(
                        f"checkpoint {column} does not match the requested run"
                    )
        summary_rows = checkpoint.to_dict(orient="records")
    completed = {
        (str(row["recording_id"]), int(row["cycle_index"]))
        for row in summary_rows
        if "recording_id" in row and "cycle_index" in row
    }
    skipped_rows: list[dict[str, Any]] = []
    output_root = (
        SEPARATION_OUTPUT_DIR / run_name if run_name else SEPARATION_OUTPUT_DIR
    )
    for recording_number, recording in enumerate(recordings, start=1):
        recording_id = recording["recording_id"]
        try:
            signal, sample_rate = _load_wav(
                AUDIO_DIR / f"{recording_id}.wav", config.sample_rate
            )
            annotations = pd.read_csv(
                AUDIO_DIR / f"{recording_id}.tsv",
                sep="\t",
                header=None,
                names=["start", "end", "state"],
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            message = f"recording load failed: {exc}"
            print(f"Skipping {recording_id}: {message}")
            skipped_rows.append({**recording, "cycle_index": None, "reason": message})
            continue
        systole_positions = np.flatnonzero(annotations["state"].to_numpy() == 2)
        if cycles_per_recording > 0:
            systole_positions = systole_positions[:cycles_per_recording]
        for cycle_index, systole_position in enumerate(systole_positions):
            if (recording_id, cycle_index) in completed:
                continue
            try:
                cycle = build_cardiac_cycle_context(
                    signal, annotations, int(systole_position), sample_rate
                )
            except ValueError as exc:
                print(f"Skipping {recording_id} cycle {cycle_index}: {exc}")
                skipped_rows.append(
                    {**recording, "cycle_index": cycle_index, "reason": str(exc)}
                )
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
            result.metrics.update(
                _absolute_timing_metrics(result.metrics, cycle, sample_rate)
            )
            segment_metadata: dict[str, Any] = {
                **recording,
                "cycle_index": cycle_index,
                "phase": "complete_cycle",
                "context_start_sample": cycle.context_start_sample,
                "context_end_sample": cycle.context_end_sample,
                "sample_rate": sample_rate,
                "config_hash": config.config_hash,
                "requested_method": method,
                "separation_method": result.selected_method,
                "recording_scope": "all" if all_recordings else "representative",
                "output_profile": output_profile,
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
            directory = (
                output_root
                / recording_id
                / f"cycle_{cycle_index}_context"
            )
            quality_status = str(result.metrics["candidate_quality_status"])
            save_package = _should_save_package(output_profile, quality_status)
            if save_package:
                metrics = _save_segment_package(
                    result, directory, segment_metadata, config
                )
            else:
                metrics = {**segment_metadata, **result.metrics}
            metrics["package_saved"] = save_package
            summary_rows.append(metrics)
            completed.add((recording_id, cycle_index))
        _write_checkpoint(summary_rows, checkpoint_path)
        if recording_number % 25 == 0 or recording_number == len(recordings):
            print(
                f"Completed recordings: {recording_number}/{len(recordings)}; "
                f"segments: {len(summary_rows)}"
            )
    summary = pd.DataFrame(summary_rows)
    destination = REPORT_OUTPUT_DIR / f"separation_summary{suffix}.csv"
    summary.to_csv(destination, index=False)
    if "candidate_quality_status" in summary:
        accepted = summary[summary["candidate_quality_status"].eq("accepted")]
    else:
        accepted = summary.copy()
    accepted.to_csv(
        REPORT_OUTPUT_DIR / f"separation_summary{suffix}_accepted.csv", index=False
    )
    if "murmur_label" in accepted:
        present_accepted = accepted[accepted["murmur_label"].eq("Present")]
    else:
        present_accepted = accepted.copy()
    present_accepted.to_csv(
        REPORT_OUTPUT_DIR
        / f"separation_summary{suffix}_present_accepted.csv",
        index=False,
    )
    summarize_audit_quality(summary).to_csv(
        REPORT_OUTPUT_DIR / f"separation_quality{suffix}.csv", index=False
    )
    observation_columns = [
        "patient_id",
        "recording_id",
        "location",
        "cycle_index",
        "timing_label",
        "candidate_quality_status",
        "timing_quality_status",
        "activity_detection_method",
        *OBSERVATION_METRICS,
    ]
    observation_columns = [
        column for column in observation_columns if column in present_accepted
    ]
    present_accepted[observation_columns].to_csv(
        REPORT_OUTPUT_DIR / f"murmur_observations{suffix}.csv", index=False
    )
    summarize_murmur_observations(summary).to_csv(
        REPORT_OUTPUT_DIR / f"murmur_observation_summary{suffix}.csv", index=False
    )
    summarize_murmur_morphology_categories(summary).to_csv(
        REPORT_OUTPUT_DIR / f"murmur_morphology_summary{suffix}.csv", index=False
    )
    skipped = pd.DataFrame(skipped_rows)
    if skipped.empty:
        skipped = pd.DataFrame(
            columns=["patient_id", "recording_id", "cycle_index", "reason"]
        )
    skipped.to_csv(
        REPORT_OUTPUT_DIR / f"separation_skipped{suffix}.csv", index=False
    )
    (REPORT_OUTPUT_DIR / f"separation_config{suffix}.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    manifest = {
        "run_name": run_name,
        "all_recordings": all_recordings,
        "recording_limit": limit,
        "cycles_per_recording": cycles_per_recording,
        "method": method,
        "output_profile": output_profile,
        "resume": resume,
        "selected_recordings": len(recordings),
        "processed_segments": len(summary),
        "skipped_segments_or_recordings": len(skipped_rows),
        "config_hash": config.config_hash,
        "candidate_interpretation": (
            "Estimated murmur candidate; not clean-source ground truth"
        ),
    }
    (REPORT_OUTPUT_DIR / f"separation_manifest{suffix}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--cycles-per-recording", type=int, default=1)
    parser.add_argument(
        "--all-recordings",
        action="store_true",
        help="Enumerate exact WAV/TSV pairs instead of one recording per patient",
    )
    parser.add_argument(
        "--output-profile",
        choices=["full", "accepted", "summary"],
        default="full",
        help="Choose which per-cycle artifact packages are written",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed recording/cycle pairs from the run checkpoint",
    )
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
        all_recordings=args.all_recordings,
        output_profile=args.output_profile,
        resume=args.resume,
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
    print(
        "Murmur observations: "
        f"{REPORT_OUTPUT_DIR / f'murmur_observations{suffix}.csv'}"
    )
    print(
        "Observation summary: "
        f"{REPORT_OUTPUT_DIR / f'murmur_observation_summary{suffix}.csv'}"
    )
    print(
        "Morphology summary: "
        f"{REPORT_OUTPUT_DIR / f'murmur_morphology_summary{suffix}.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
