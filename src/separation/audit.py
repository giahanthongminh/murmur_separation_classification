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
from scipy.signal import hilbert, resample_poly, spectrogram, welch

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
    wavelet_scalogram,
)
from src.separation.tier_a_features import TIER_A_EXPORT_COLUMNS


OBSERVATION_METRICS = [
    "onset_normalized",
    "offset_normalized",
    "duration_ratio",
    "temporal_energy_centroid",
    "peak_position",
    "murmur_onset_target_phase_seconds",
    "murmur_offset_target_phase_seconds",
    "murmur_onset_systole_seconds",
    "murmur_offset_systole_seconds",
    "murmur_duration_seconds",
    "murmur_duration_target_phase_percent",
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
    "s1_reference_peak_abs",
    "s2_reference_peak_abs",
    "s1_s2_reference_peak_abs",
    "murmur_peak_relative_to_s1_s2_ratio",
    "murmur_peak_relative_to_s1_s2_percent",
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
    "psd_above_200_hz_ratio",
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
    "psd_low_frequency_limit_95_hz",
    "psd_high_frequency_limit_95_hz",
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
    "wavelet_status",
    "wavelet_peak_frequency_hz",
    "wavelet_peak_seconds",
    "wavelet_frequency_centroid_hz",
    "wavelet_frequency_spread_hz",
    "wavelet_entropy",
    "wavelet_peak_scale_energy_ratio",
    "boundary_jitter_ms",
    "boundary_variant_count",
    "boundary_amplitude_rms_relative_range",
    "boundary_psd_peak_frequency_range_hz",
    "boundary_envelope_time_to_peak_ratio_range",
    "boundary_psd_width_relative_range",
    "boundary_envelope_shape_agreement_ratio",
    "boundary_stability_status",
    *TIER_A_EXPORT_COLUMNS,
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


def _representative_recordings(metadata: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
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
        locations = _recording_locations(row)
        if not locations:
            continue
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
                    "location_murmur_label": murmur,
                    "patient_murmur_label": murmur,
                    "clinical_outcome": _metadata_label(row.get("Outcome")),
                    "timing_label": timing,
                    "audit_group": group,
                    **_expert_phase_fields(row),
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


def _recording_locations(row: pd.Series) -> list[str]:
    """Read recording locations from either released CirCor CSV schema."""

    for column in ("Recording locations:", "Locations"):
        locations = sorted(_metadata_locations(row.get(column)))
        if locations:
            return locations
    return []


def _metadata_label(value: Any) -> str | None:
    """Return a usable CirCor annotation label instead of textual NaN."""

    if pd.isna(value):
        return None
    label = str(value).strip()
    return None if not label or label.lower() == "nan" else label


def _interpretation_group(
    location_murmur_label: str | None,
    clinical_outcome: str | None,
) -> str:
    """Create cautious report groups from CirCor's two patient-level labels."""

    if location_murmur_label != "Present":
        return "not_scored"
    if clinical_outcome == "Normal":
        return "present_normal_outcome_innocent_proxy"
    if clinical_outcome == "Abnormal":
        return "present_abnormal_outcome_pathological_proxy"
    return "present_unknown_outcome"


def _expert_phase_fields(row: pd.Series) -> dict[str, str | None]:
    fields: dict[str, str | None] = {}
    for phase, prefix in (("systole", "Systolic"), ("diastole", "Diastolic")):
        for name in ("timing", "shape", "pitch", "grading", "quality"):
            fields[f"{phase}_{name}_label"] = _metadata_label(
                row.get(f"{prefix} murmur {name}")
            )
    return fields


def _target_phases_for_recording(
    recording: dict[str, Any], requested_phase: str
) -> list[str]:
    if requested_phase in {"systole", "diastole"}:
        return [requested_phase]
    if requested_phase != "auto":
        raise ValueError("target_phase must be auto, systole, or diastole")
    if recording.get("location_murmur_label") == "Present":
        phases = [
            phase
            for phase in ("systole", "diastole")
            if recording.get(f"{phase}_timing_label") is not None
        ]
        return phases or ["systole"]
    return ["systole", "diastole"]


def _target_phase_metadata(
    recording: dict[str, Any], target_phase: str
) -> dict[str, Any]:
    location_label = str(recording.get("location_murmur_label", "Unknown"))
    expert_timing = recording.get(f"{target_phase}_timing_label")
    if location_label == "Absent":
        phase_label = "Absent"
    elif location_label == "Present" and expert_timing is not None:
        phase_label = "Present"
    else:
        phase_label = "Unknown"
    clinical_outcome = recording.get("clinical_outcome")
    return {
        "murmur_phase": target_phase,
        "murmur_label": phase_label,
        "clinical_outcome": clinical_outcome,
        "interpretation_group": _interpretation_group(
            phase_label, clinical_outcome
        ),
        "timing_label": expert_timing,
        "expert_timing_label": expert_timing,
        "expert_shape_label": recording.get(f"{target_phase}_shape_label"),
        "expert_pitch_label": recording.get(f"{target_phase}_pitch_label"),
        "expert_grading_label": recording.get(f"{target_phase}_grading_label"),
        "expert_quality_label": recording.get(f"{target_phase}_quality_label"),
    }


def _all_recordings(
    metadata: pd.DataFrame,
    limit: int,
    *,
    audio_dir: Path = AUDIO_DIR,
) -> list[dict[str, Any]]:
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
                "location_murmur_label": murmur_label,
                "patient_murmur_label": patient_label,
                "clinical_outcome": _metadata_label(row.get("Outcome")),
                "timing_label": timing,
                "audit_group": "All recordings",
                **_expert_phase_fields(row),
            }
        )
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def _absolute_timing_metrics(
    metrics: dict[str, Any],
    cycle: CardiacCycleContext,
    sample_rate: int,
    target_phase: str = "systole",
) -> dict[str, Any]:
    """Convert target-phase detector output into cycle and recording seconds."""

    if target_phase not in {"systole", "diastole"}:
        raise ValueError("target_phase must be 'systole' or 'diastole'")

    onset = metrics.get("onset_sample")
    offset = metrics.get("offset_sample")
    quality = str(metrics.get("candidate_quality_status", "fallback"))
    detection = str(metrics.get("activity_detection_method", "silent_or_invalid"))
    if onset is None or offset is None:
        return {
            "murmur_onset_target_phase_seconds": None,
            "murmur_offset_target_phase_seconds": None,
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
    target_start = cycle.phase_bounds[target_phase][0]
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
        "murmur_onset_target_phase_seconds": onset / sample_rate,
        "murmur_offset_target_phase_seconds": offset / sample_rate,
        "murmur_onset_systole_seconds": (
            onset / sample_rate if target_phase == "systole" else None
        ),
        "murmur_offset_systole_seconds": (
            offset / sample_rate if target_phase == "systole" else None
        ),
        "murmur_duration_seconds": (offset - onset) / sample_rate,
        "murmur_onset_cycle_seconds": (target_start + onset) / sample_rate,
        "murmur_offset_cycle_seconds": (target_start + offset) / sample_rate,
        "murmur_onset_recording_seconds": (
            cycle.context_start_sample + target_start + onset
        )
        / sample_rate,
        "murmur_offset_recording_seconds": (
            cycle.context_start_sample + target_start + offset
        )
        / sample_rate,
        "time_frequency_peak_cycle_seconds": (
            (target_start + onset) / sample_rate + time_frequency_peak
        ),
        "time_frequency_peak_recording_seconds": (
            (cycle.context_start_sample + target_start + onset) / sample_rate
            + time_frequency_peak
        ),
        "timing_quality_status": timing_status,
    }


def _predicted_timing_label(metrics: dict[str, Any], target_phase: str) -> str | None:
    onset = metrics.get("onset_normalized")
    offset = metrics.get("offset_normalized")
    if onset is None or offset is None:
        return None
    onset = float(onset)
    offset = float(offset)
    duration = offset - onset
    phase_name = "systolic" if target_phase == "systole" else "diastolic"
    if duration >= 0.75 and onset <= 0.15 and offset >= 0.85:
        return "Holosystolic" if target_phase == "systole" else "Holodiastolic"
    midpoint = 0.5 * (onset + offset)
    position = "Early" if midpoint < 1 / 3 else "Mid" if midpoint < 2 / 3 else "Late"
    return f"{position}-{phase_name}"


def _expert_agreement_metrics(
    metrics: dict[str, Any], metadata: dict[str, Any], target_phase: str
) -> dict[str, Any]:
    predicted_timing = _predicted_timing_label(metrics, target_phase)
    expert_timing = metadata.get("expert_timing_label")
    envelope_shape = str(metrics.get("envelope_shape", "insufficient"))
    shape_mapping = {
        "constant": "Plateau",
        "crescendo": "Crescendo",
        "decrescendo": "Decrescendo",
        "crescendo_decrescendo": "Diamond",
    }
    predicted_shape = shape_mapping.get(envelope_shape)
    expert_shape = metadata.get("expert_shape_label")
    return {
        "predicted_timing_label": predicted_timing,
        "predicted_shape_label": predicted_shape,
        "timing_label_agreement": (
            None
            if predicted_timing is None or expert_timing is None
            else predicted_timing == expert_timing
        ),
        "shape_label_agreement": (
            None
            if predicted_shape is None or expert_shape is None
            else predicted_shape == expert_shape
        ),
    }


def summarize_expert_agreement(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize timing/shape agreement and frequency by expert pitch label."""

    columns = [
        "murmur_phase",
        "comparison",
        "expert_category",
        "segment_count",
        "scorable_count",
        "agreement_count",
        "agreement_ratio",
        "mean_primary_frequency_hz",
        "median_primary_frequency_hz",
    ]
    required = {"murmur_label", "candidate_quality_status", "murmur_phase"}
    if summary.empty or not required.issubset(summary.columns):
        return pd.DataFrame(columns=columns)
    selected = summary[
        summary["murmur_label"].eq("Present")
        & summary["candidate_quality_status"].eq("accepted")
    ]
    rows: list[dict[str, Any]] = []
    for phase, group in selected.groupby("murmur_phase", dropna=False):
        for comparison, expert_column, agreement_column in (
            ("timing", "expert_timing_label", "timing_label_agreement"),
            ("shape", "expert_shape_label", "shape_label_agreement"),
        ):
            if expert_column not in group or agreement_column not in group:
                continue
            scorable = group[group[agreement_column].notna()]
            rows.append(
                {
                    "murmur_phase": phase,
                    "comparison": comparison,
                    "expert_category": "all",
                    "segment_count": int(len(group)),
                    "scorable_count": int(len(scorable)),
                    "agreement_count": int(scorable[agreement_column].astype(bool).sum()),
                    "agreement_ratio": (
                        float(scorable[agreement_column].astype(bool).mean())
                        if len(scorable)
                        else None
                    ),
                    "mean_primary_frequency_hz": None,
                    "median_primary_frequency_hz": None,
                }
            )
        if {
            "expert_pitch_label",
            "psd_primary_peak_frequency_hz",
        }.issubset(group.columns):
            for pitch, pitch_group in group.dropna(
                subset=["expert_pitch_label"]
            ).groupby("expert_pitch_label"):
                frequencies = pd.to_numeric(
                    pitch_group["psd_primary_peak_frequency_hz"], errors="coerce"
                ).dropna()
                rows.append(
                    {
                        "murmur_phase": phase,
                        "comparison": "expert_pitch_frequency",
                        "expert_category": pitch,
                        "segment_count": int(len(pitch_group)),
                        "scorable_count": int(len(frequencies)),
                        "agreement_count": None,
                        "agreement_ratio": None,
                        "mean_primary_frequency_hz": (
                            float(frequencies.mean()) if len(frequencies) else None
                        ),
                        "median_primary_frequency_hz": (
                            float(frequencies.median()) if len(frequencies) else None
                        ),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


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
        "murmur_phase",
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
    group_columns = [
        column
        for column in ("murmur_phase", "timing_label", "activity_detection_method")
        if column in selected
    ]
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

    columns = [
        "murmur_phase",
        "timing_label",
        "dimension",
        "category",
        "segment_count",
        "ratio",
    ]
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
    group_columns = [
        column for column in ("murmur_phase", "timing_label") if column in selected
    ]
    for group_key, group in selected.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(group_columns, group_key))
        for dimension in dimensions:
            counts = group[dimension].fillna("missing").astype(str).value_counts()
            total = int(counts.sum())
            for category, count in counts.items():
                rows.append(
                    {
                        "murmur_phase": group_values.get("murmur_phase"),
                        "timing_label": group_values.get("timing_label"),
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
    """Write one compact, paper-aligned observation figure for either phase."""

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
    phase = str(metadata.get("murmur_phase", "target phase"))
    figure, axes = plt.subplots(4, 2, figsize=(15, 16))
    axes = axes.ravel()
    figure.suptitle(
        f"{metadata['recording_id']} - {phase.title()} murmur observation",
        fontsize=15,
    )

    phase_colors = {
        "s1": "tab:green",
        "systole": "tab:blue",
        "s2": "tab:orange",
        "diastole": "tab:purple",
    }
    axes[0].plot(time, result.original, linewidth=0.7, color="tab:blue")
    for phase, color in phase_colors.items():
        start = metadata[f"{phase}_relative_start_sample"] / sample_rate
        end = metadata[f"{phase}_relative_end_sample"] / sample_rate
        axes[0].axvspan(start, end, alpha=0.16, color=color, label=phase)
    if onset_seconds is not None and offset_seconds is not None:
        axes[0].axvline(float(onset_seconds), color="tab:red", linestyle="--")
        axes[0].axvline(float(offset_seconds), color="tab:red", linestyle="--")
    axes[0].legend(loc="upper right", fontsize=8, ncol=4)
    axes[0].set_title("Original phonocardiogram with cardiac phases")

    axes[1].plot(
        time,
        result.original,
        linewidth=0.6,
        color="0.65",
        alpha=0.65,
        label="original",
    )
    axes[1].plot(
        time,
        result.normal_estimate,
        linewidth=0.8,
        color="tab:blue",
        label="normal-heart estimate",
    )
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("Separation check: original vs normal-heart estimate")

    axes[2].plot(
        time, result.murmur_candidate, linewidth=0.65, label="murmur candidate"
    )
    observation_time = (
        np.arange(len(observation_candidate)) + observation_start
    ) / sample_rate
    smooth_envelope = smooth_amplitude_envelope(observation_candidate, sample_rate)
    axes[2].plot(
        observation_time,
        smooth_envelope,
        color="tab:red",
        linewidth=1.5,
        label="smoothed envelope",
    )
    if len(smooth_envelope):
        peak_index = int(np.argmax(smooth_envelope))
        axes[2].scatter(
            observation_time[peak_index],
            smooth_envelope[peak_index],
            color="black",
            s=18,
            zorder=3,
            label="envelope peak",
        )
    axes[2].set_title(
        f"{metadata.get('murmur_phase', 'Target-phase').title()} murmur and "
        "amplitude envelope - "
        f"{result.metrics.get('envelope_shape', 'not characterized')}"
    )
    axes[2].legend(loc="upper right", fontsize=8)
    if onset_seconds is not None and offset_seconds is not None:
        axes[2].axvline(float(onset_seconds), color="tab:red", linestyle="--")
        axes[2].axvline(float(offset_seconds), color="tab:red", linestyle="--")

    amplitude_values = [
        float(result.metrics.get("s1_reference_peak_abs", 0.0) or 0.0),
        float(result.metrics.get("s2_reference_peak_abs", 0.0) or 0.0),
        float(result.metrics.get("amplitude_peak_abs", 0.0) or 0.0),
    ]
    bars = axes[3].bar(
        ["S1 original", "S2 original", "Murmur candidate"],
        amplitude_values,
        color=["tab:green", "tab:orange", "tab:red"],
    )
    for bar, value in zip(bars, amplitude_values):
        axes[3].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    relative_percent = result.metrics.get(
        "murmur_peak_relative_to_s1_s2_percent"
    )
    relative_text = (
        "unavailable"
        if relative_percent is None
        else f"{float(relative_percent):.1f}% of mean S1/S2 peak"
    )
    axes[3].set_title(f"Paper-aligned relative amplitude - {relative_text}")
    axes[3].set_ylabel("relative digital amplitude")

    if len(observation_candidate) > 1:
        psd_frequencies, psd_power = welch(
            observation_candidate,
            fs=sample_rate,
            nperseg=min(512, len(observation_candidate)),
            detrend="constant",
        )
        axes[4].plot(
            psd_frequencies,
            10 * np.log10(psd_power + 1e-12),
            linewidth=1.0,
        )
    else:
        axes[4].text(0.5, 0.5, "insufficient interval", ha="center", va="center")
    primary_frequency = float(
        result.metrics.get("psd_primary_peak_frequency_hz", 0.0) or 0.0
    )
    secondary_frequency = float(
        result.metrics.get("psd_secondary_peak_frequency_hz", 0.0) or 0.0
    )
    if primary_frequency > 0:
        axes[4].axvline(
            primary_frequency,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"primary {primary_frequency:.0f} Hz",
        )
    if secondary_frequency > 0:
        axes[4].axvline(
            secondary_frequency,
            color="tab:orange",
            linestyle=":",
            linewidth=1.0,
            label=f"secondary {secondary_frequency:.0f} Hz",
        )
    low_frequency = float(
        result.metrics.get("psd_low_frequency_limit_95_hz", 0.0) or 0.0
    )
    high_frequency = float(
        result.metrics.get("psd_high_frequency_limit_95_hz", 0.0) or 0.0
    )
    if high_frequency > low_frequency > 0:
        axes[4].axvspan(
            low_frequency,
            high_frequency,
            color="tab:green",
            alpha=0.12,
            label=f"95% energy: {low_frequency:.0f}-{high_frequency:.0f} Hz",
        )
    axes[4].set_xlim(0, min(1000, sample_rate / 2))
    axes[4].set_title(
        "Murmur PSD - "
        f"{result.metrics.get('psd_morphology', 'not characterized')}"
    )
    if primary_frequency > 0:
        axes[4].legend(loc="upper right", fontsize=8)
    axes[4].set_ylabel("power spectral density (dB/Hz)")

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
        axes[5].imshow(
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
        axes[5].pcolormesh(times, frequencies, power_db, shading="auto")
    ridge_times, ridge = dominant_frequency_trajectory(
        observation_candidate, sample_rate
    )
    if len(ridge_times):
        axes[5].plot(
            ridge_times + observation_start / sample_rate,
            ridge,
            color="white",
            linewidth=1.2,
            marker=".",
            markersize=2,
            label="dominant-frequency trajectory",
        )
        axes[5].legend(loc="upper right", fontsize=8)
    axes[5].set_ylim(0, min(1000, sample_rate / 2))
    axes[5].set_title(
        "Murmur spectrogram - frequency "
        f"{result.metrics.get('time_frequency_ridge_direction', 'not characterized')}"
    )
    wavelet_times, wavelet_frequencies, wavelet_power, wavelet_status = (
        wavelet_scalogram(observation_candidate, sample_rate)
    )
    if wavelet_power.size:
        axes[6].pcolormesh(
            wavelet_times + observation_start / sample_rate,
            wavelet_frequencies,
            10 * np.log10(wavelet_power + 1e-12),
            shading="auto",
        )
        axes[6].set_ylim(20, min(1000, sample_rate / 2))
        axes[6].set_title("Murmur Morlet wavelet scalogram")
    else:
        axes[6].text(0.5, 0.5, wavelet_status, ha="center", va="center")
        axes[6].set_title("Wavelet scalogram unavailable")

    def format_metric(name: str, digits: int = 3) -> str:
        value = result.metrics.get(name)
        if value is None:
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}g}"
        return str(value)

    axes[7].axis("off")
    axes[7].text(
        0,
        1,
        "\n".join(
            [
                "Observation summary",
                f"clinical outcome: {metadata.get('clinical_outcome')}",
                f"interpretation group: {metadata.get('interpretation_group')}",
                f"method / quality: {result.selected_method} / "
                f"{format_metric('candidate_quality_status')}",
                f"phase / timing quality: {metadata.get('murmur_phase')} / "
                f"{format_metric('timing_quality_status')}",
                f"onset-offset (recording s): "
                f"{format_metric('murmur_onset_recording_seconds', 5)} - "
                f"{format_metric('murmur_offset_recording_seconds', 5)}",
                f"duration (% target phase): "
                f"{format_metric('murmur_duration_target_phase_percent', 4)}",
                f"amplitude (% mean S1/S2 peak): "
                f"{format_metric('murmur_peak_relative_to_s1_s2_percent', 4)}",
                f"envelope: {format_metric('envelope_shape')}",
                f"PSD: {format_metric('psd_morphology')}; primary "
                f"{format_metric('psd_primary_peak_frequency_hz', 4)} Hz",
                f"95% PSD range: {format_metric('psd_low_frequency_limit_95_hz', 4)} - "
                f"{format_metric('psd_high_frequency_limit_95_hz', 4)} Hz",
                f"frequency trajectory: "
                f"{format_metric('time_frequency_ridge_direction')}",
                f"boundary stability: "
                f"{format_metric('boundary_stability_status')}",
                "",
                "CirCor expert reference",
                f"timing: {metadata.get('expert_timing_label')} -> "
                f"{format_metric('predicted_timing_label')}",
                f"shape: {metadata.get('expert_shape_label')} -> "
                f"{format_metric('predicted_shape_label')}",
                f"pitch / grade / quality: {metadata.get('expert_pitch_label')} / "
                f"{metadata.get('expert_grading_label')} / "
                f"{metadata.get('expert_quality_label')}",
                "",
                "Separation audit",
                f"outside-target energy: "
                f"{format_metric('outside_murmur_energy_ratio', 4)}",
                f"target retention: "
                f"{format_metric('murmur_region_energy_retention', 4)}",
                f"S1 / S2 leakage: {format_metric('s1_leakage_ratio', 4)} / "
                f"{format_metric('s2_leakage_ratio', 4)}",
                "Expert labels are semantic references,",
                "not clean-source waveform ground truth.",
                "Normal/abnormal outcome groups are report proxies,",
                "not definitive innocent/pathological diagnoses.",
            ]
        ),
        va="top",
        family="monospace",
        fontsize=9.5,
    )

    for axis in axes[:3]:
        axis.set_xlabel("seconds")
    axes[4].set_xlabel("frequency (Hz)")
    axes[5].set_xlabel("seconds")
    axes[5].set_ylabel("frequency (Hz)")
    axes[6].set_xlabel("seconds")
    axes[6].set_ylabel("frequency (Hz)")
    figure.tight_layout(rect=(0, 0, 1, 0.97), pad=1.2)
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
        "murmur_phase",
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
    group_columns = [
        column
        for column in (
            "murmur_phase",
            "candidate_quality_status",
            "murmur_label",
            "audit_outcome",
        )
        if column in table
    ]
    grouped = table.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )
    counts = grouped.size().rename("segment_count")
    means = grouped[metrics].mean()
    result = pd.concat([counts, means], axis=1).reset_index()
    return result[[column for column in columns if column in result]]


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
    target_phase: str = "auto",
    recording_ids: list[str] | None = None,
) -> pd.DataFrame:
    if limit < 0:
        raise ValueError("limit must be non-negative; use 0 for no limit")
    if cycles_per_recording < 0:
        raise ValueError(
            "cycles_per_recording must be non-negative; use 0 for all cycles"
        )
    if output_profile not in {"full", "accepted", "summary"}:
        raise ValueError("output_profile must be full, accepted, or summary")
    if target_phase not in {"auto", "systole", "diastole"}:
        raise ValueError("target_phase must be auto, systole, or diastole")
    if resume and not run_name:
        raise ValueError("resume requires a run_name to identify its checkpoint")
    ensure_output_directories()
    if validate_first:
        validate_dataset()
    metadata_table = pd.read_csv(METADATA_PATH, dtype={"Patient ID": str})
    recording_filter = sorted(set(recording_ids or []))
    if recording_filter:
        available = {
            row["recording_id"]: row for row in _all_recordings(metadata_table, 0)
        }
        missing = [name for name in recording_filter if name not in available]
        if missing:
            raise ValueError(f"recording IDs are unavailable or unpaired: {missing}")
        recordings = [available[name] for name in recording_filter]
    else:
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
                "target_phase_request": str,
                "recording_filter": str,
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
            "target_phase_request": target_phase,
            "recording_filter": ",".join(recording_filter),
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
        (
            str(row["recording_id"]),
            int(row["cycle_index"]),
            str(row.get("murmur_phase", "systole")),
        )
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
            for murmur_phase in _target_phases_for_recording(
                recording, target_phase
            ):
                identity = (recording_id, cycle_index, murmur_phase)
                if identity in completed:
                    continue
                phase_metadata = _target_phase_metadata(recording, murmur_phase)
                if method == "auto":
                    result, _ = compare_separation_methods(
                        cycle.signal,
                        config=config,
                        phase_masks=cycle.phase_masks,
                        target_phase=murmur_phase,
                    )
                else:
                    result = separate_signal(
                        cycle.signal,
                        config=config,
                        method=method,
                        phase_masks=cycle.phase_masks,
                        target_phase=murmur_phase,
                    )
                result.metrics.update(
                    _absolute_timing_metrics(
                        result.metrics, cycle, sample_rate, murmur_phase
                    )
                )
                result.metrics.update(
                    _expert_agreement_metrics(
                        result.metrics, phase_metadata, murmur_phase
                    )
                )
                segment_metadata: dict[str, Any] = {
                    **recording,
                    **phase_metadata,
                    "cycle_index": cycle_index,
                    "phase": "complete_cycle",
                    "context_start_sample": cycle.context_start_sample,
                    "context_end_sample": cycle.context_end_sample,
                    "sample_rate": sample_rate,
                    "config_hash": config.config_hash,
                    "requested_method": method,
                    "target_phase_request": target_phase,
                    "recording_filter": ",".join(recording_filter),
                    "separation_method": result.selected_method,
                    "recording_scope": (
                        "all" if all_recordings else "representative"
                    ),
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
                    str(phase_metadata["murmur_label"]),
                    str(result.metrics["candidate_quality_status"]),
                )
                directory = (
                    output_root
                    / recording_id
                    / f"cycle_{cycle_index}_{murmur_phase}_context"
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
                completed.add(identity)
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
        "murmur_phase",
        "clinical_outcome",
        "interpretation_group",
        "timing_label",
        "expert_timing_label",
        "expert_shape_label",
        "expert_pitch_label",
        "expert_grading_label",
        "expert_quality_label",
        "predicted_timing_label",
        "predicted_shape_label",
        "timing_label_agreement",
        "shape_label_agreement",
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
    expert_columns = [
        column
        for column in observation_columns
        if column in present_accepted
    ]
    present_accepted[expert_columns].to_csv(
        REPORT_OUTPUT_DIR / f"murmur_expert_comparison{suffix}.csv", index=False
    )
    summarize_expert_agreement(summary).to_csv(
        REPORT_OUTPUT_DIR / f"murmur_expert_agreement{suffix}.csv", index=False
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
        "target_phase": target_phase,
        "recording_filter": recording_filter,
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
    parser.add_argument(
        "--target-phase",
        choices=["auto", "systole", "diastole"],
        default="auto",
        help=(
            "Choose the murmur phase; auto uses expert phase metadata for Present "
            "recordings and both phases for controls"
        ),
    )
    parser.add_argument(
        "--recording-id",
        action="append",
        default=[],
        help=(
            "Process one exact recording ID; repeat this option for a focused pilot"
        ),
    )
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
        target_phase=args.target_phase,
        recording_ids=args.recording_id,
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
    print(
        "Expert agreement: "
        f"{REPORT_OUTPUT_DIR / f'murmur_expert_agreement{suffix}.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
