"""Deterministic small real-candidate audit for the frozen Task 5 Tier A schema.

This module selects a metadata-stratified screening set before numerical feature
inspection, runs one cycle per recording, and then chooses a fixed-size audit
set using only identifiers, expert descriptors, candidate status, support, and
separation QA.  Tier A feature values are never selection criteria.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.signal import spectrogram, welch

from config import (
    AUDIO_DIR,
    DEFAULT_SEPARATION_CONFIG,
    METADATA_PATH,
    OUTPUT_ROOT,
    SeparationConfig,
    validate_input_output_isolation,
    DATASET_ROOT,
)
from src.separation.audit import (
    _absolute_timing_metrics,
    _all_recordings,
    _expert_agreement_metrics,
    _load_wav,
    _save_segment_package,
    _target_phase_metadata,
    _target_phases_for_recording,
    build_cardiac_cycle_context,
    classify_audit_outcome,
)
from src.separation.core import SeparationResult, separate_signal
from src.separation.tier_a_features import (
    DEFAULT_TIER_A_FEATURE_CONFIG,
    FEATURE_SCHEMA_VERSION,
    TIER_A_EXPORT_COLUMNS,
    TIER_A_FEATURE_COLUMNS,
    TIER_A_FEATURE_UNITS,
    smoothed_hilbert_envelope,
)


AUDIT_TOOL_VERSION: Final = "task5-phase1b-audit-v1.0.0"
DEFAULT_AUDIT_ROOT: Final = OUTPUT_ROOT / "task5_candidate_audit"

# Screening is metadata-only.  These quotas deliberately exceed the final
# audit size through overlap; they are priorities, not mutually exclusive bins.
SCREENING_STRATA: Final[tuple[tuple[str, int], ...]] = (
    ("phase:diastole", 5),
    ("shape:Crescendo", 3),
    ("shape:Plateau", 5),
    ("shape:Diamond", 5),
    ("shape:Decrescendo", 5),
    ("timing:Early-systolic", 4),
    ("timing:Mid-systolic", 4),
    ("timing:Holosystolic", 4),
    ("timing:Late-systolic", 2),
    ("pitch:Low", 3),
    ("pitch:Medium", 3),
    ("pitch:High", 3),
    ("grade:I/VI", 3),
    ("grade:II/VI", 3),
    ("grade:III/VI", 3),
)

# Final selection uses no Tier A numerical value.  Short support and invalid
# flags are allowed because the audit explicitly requires insufficient-support
# cases.  Continuous ranks use separation QA only.
FINAL_STRATA: Final[tuple[tuple[str, int], ...]] = (
    ("phase:diastole", 5),
    ("status:fallback", 5),
    ("shape:Crescendo", 1),
    ("shape:Plateau", 2),
    ("shape:Diamond", 2),
    ("shape:Decrescendo", 2),
    ("timing:Early-systolic", 1),
    ("timing:Mid-systolic", 1),
    ("timing:Holosystolic", 1),
    ("timing:Late-systolic", 1),
    ("pitch:Low", 1),
    ("pitch:Medium", 1),
    ("pitch:High", 1),
    ("grade:I/VI", 1),
    ("grade:II/VI", 1),
    ("grade:III/VI", 1),
    ("short_support", 3),
    ("invalid_feature", 3),
    ("high_leakage", 3),
    ("low_retention", 3),
    ("status:accepted", 10),
)

IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "patient_id",
    "recording_id",
    "location",
    "cycle_index",
    "murmur_phase",
)

AUDIT_MANIFEST_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "selection_rank",
    "selection_reasons",
    *IDENTIFIER_COLUMNS,
    "candidate_quality_status",
    "expert_timing_label",
    "expert_shape_label",
    "expert_pitch_label",
    "expert_grading_label",
    "tier_a_any_invalid",
    "tier_a_invalid_reasons",
    "s1_leakage_ratio",
    "s2_leakage_ratio",
    "murmur_region_energy_retention",
    "artifact_directory",
    *TIER_A_EXPORT_COLUMNS,
)

PANEL_FEATURE_LABELS: Final[dict[str, str]] = {
    TIER_A_FEATURE_COLUMNS[0]: "duration_fraction",
    TIER_A_FEATURE_COLUMNS[1]: "midpoint_normalized",
    TIER_A_FEATURE_COLUMNS[2]: "rms_vs_s1_s2_db",
    TIER_A_FEATURE_COLUMNS[3]: "envelope_peak_position",
    TIER_A_FEATURE_COLUMNS[4]: "envelope_rise_slope",
    TIER_A_FEATURE_COLUMNS[5]: "envelope_decay_slope",
    TIER_A_FEATURE_COLUMNS[6]: "envelope_fullness",
    TIER_A_FEATURE_COLUMNS[7]: "psd_dominant_hz",
    TIER_A_FEATURE_COLUMNS[8]: "psd_median_hz",
    TIER_A_FEATURE_COLUMNS[9]: "psd_bandwidth_95_hz",
    TIER_A_FEATURE_COLUMNS[10]: "psd_fraction_above_200",
    TIER_A_FEATURE_COLUMNS[11]: "psd_entropy",
    TIER_A_FEATURE_COLUMNS[12]: "ridge_slope_hz_norm_time",
    TIER_A_FEATURE_COLUMNS[13]: "ridge_variability_mad_hz",
}


@dataclass(frozen=True)
class CandidateAuditConfig:
    target_count: int = 30
    screening_recording_count: int = 40
    cycles_per_recording: int = 1
    method: str = "zcr"
    short_candidate_samples: int = 128

    def __post_init__(self) -> None:
        if self.target_count < 1:
            raise ValueError("target_count must be positive")
        if self.screening_recording_count < self.target_count:
            raise ValueError("screening_recording_count must be at least target_count")
        if self.cycles_per_recording < 1:
            raise ValueError("cycles_per_recording must be positive")
        if self.method not in {"zcr", "kurtosis"}:
            raise ValueError("method must be zcr or kurtosis")
        if self.short_candidate_samples < 1:
            raise ValueError("short_candidate_samples must be positive")


@dataclass(frozen=True)
class AuditCandidate:
    row: dict[str, Any]
    result: SeparationResult


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _identity_key(row: pd.Series | dict[str, Any]) -> tuple[str, str, int, str]:
    get = row.get
    return (
        _clean_text(get("patient_id")),
        _clean_text(get("recording_id")),
        int(get("cycle_index", 0) or 0),
        _clean_text(get("murmur_phase")),
    )


def _metadata_matches(recording: dict[str, Any], stratum: str) -> bool:
    kind, value = stratum.split(":", 1)
    if kind == "phase":
        return value == "diastole" and bool(recording.get("diastole_timing_label"))
    column = {
        "shape": "systole_shape_label",
        "timing": "systole_timing_label",
        "pitch": "systole_pitch_label",
        "grade": "systole_grading_label",
    }.get(kind)
    return column is not None and _clean_text(recording.get(column)) == value


def select_screening_recordings(
    recordings: list[dict[str, Any]],
    *,
    target_count: int = 40,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Select metadata strata deterministically, at most one recording per patient."""

    eligible = [
        dict(row)
        for row in recordings
        if _clean_text(row.get("location_murmur_label")) == "Present"
    ]
    eligible.sort(key=lambda row: (_clean_text(row.get("patient_id")), _clean_text(row.get("recording_id"))))

    # Pick the lexicographically first murmur-positive location for each patient.
    patient_rows: dict[str, dict[str, Any]] = {}
    for row in eligible:
        patient_rows.setdefault(_clean_text(row.get("patient_id")), row)
    pool = list(patient_rows.values())

    selected: list[dict[str, Any]] = []
    selected_patients: set[str] = set()
    reasons: dict[str, list[str]] = {}
    shortfall_rows: list[dict[str, Any]] = []
    for stratum, requested in SCREENING_STRATA:
        matching = [row for row in pool if _metadata_matches(row, stratum)]
        already = [row for row in matching if _clean_text(row["patient_id"]) in selected_patients]
        remaining = max(0, requested - len(already))
        for row in matching:
            patient = _clean_text(row["patient_id"])
            if remaining == 0 or len(selected) >= target_count:
                break
            if patient in selected_patients:
                continue
            selected.append(row)
            selected_patients.add(patient)
            remaining -= 1
        covered = sum(
            _metadata_matches(row, stratum) for row in selected
        )
        for row in selected:
            if _metadata_matches(row, stratum):
                reasons.setdefault(_clean_text(row["patient_id"]), []).append(stratum)
        shortfall = max(0, requested - covered)
        shortfall_rows.append(
            {
                "selection_stage": "metadata_screening",
                "stratum": stratum,
                "requested": requested,
                "eligible": len(matching),
                "selected": covered,
                "shortfall": shortfall,
                "shortfall_reason": (
                    "none"
                    if shortfall == 0
                    else "insufficient_eligible"
                    if len(matching) < requested
                    else "screening_capacity"
                ),
            }
        )

    for row in pool:
        if len(selected) >= target_count:
            break
        patient = _clean_text(row["patient_id"])
        if patient in selected_patients:
            continue
        selected.append(row)
        selected_patients.add(patient)
        reasons.setdefault(patient, []).append("stable_metadata_fill")

    output: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        copied = dict(row)
        copied["screening_rank"] = rank
        copied["screening_reasons"] = ";".join(reasons.get(_clean_text(row["patient_id"]), ["stable_metadata_fill"]))
        output.append(copied)
    return output, pd.DataFrame(shortfall_rows)


def _invalid_feature_summary(row: pd.Series) -> tuple[bool, str]:
    reasons: list[str] = []
    for feature in TIER_A_FEATURE_COLUMNS:
        if not bool(row.get(f"{feature}_valid", False)):
            reason = _clean_text(row.get(f"{feature}_invalid_reason")) or "unspecified"
            reasons.append(f"{feature}:{reason}")
    return bool(reasons), ";".join(reasons)


def prepare_candidate_table(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Normalize derived support/QA fields without inspecting Tier A values."""

    table = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if table.empty:
        return table
    missing = set(IDENTIFIER_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"candidate table is missing identifiers: {sorted(missing)}")
    table = table.copy()
    table["candidate_id"] = [
        f"{_clean_text(row.patient_id)}:{_clean_text(row.recording_id)}:"
        f"cycle{int(row.cycle_index)}:{_clean_text(row.murmur_phase)}"
        for row in table.itertuples(index=False)
    ]
    if table["candidate_id"].duplicated().any():
        raise ValueError("candidate identifiers must be unique")
    summaries = [_invalid_feature_summary(row) for _, row in table.iterrows()]
    table["tier_a_any_invalid"] = [item[0] for item in summaries]
    table["tier_a_invalid_reasons"] = [item[1] for item in summaries]
    leakage = pd.concat(
        [
            pd.to_numeric(table.get("s1_leakage_ratio"), errors="coerce"),
            pd.to_numeric(table.get("s2_leakage_ratio"), errors="coerce"),
        ],
        axis=1,
    )
    table["maximum_s1_s2_leakage_ratio"] = leakage.max(axis=1, skipna=True)
    table.sort_values(
        ["patient_id", "recording_id", "cycle_index", "murmur_phase"],
        kind="mergesort",
        inplace=True,
    )
    table.reset_index(drop=True, inplace=True)
    return table


def _final_matches(table: pd.DataFrame, stratum: str, short_samples: int) -> pd.Series:
    if stratum == "short_support":
        return pd.to_numeric(table["tier_a_candidate_sample_count"], errors="coerce").lt(short_samples)
    if stratum == "invalid_feature":
        return table["tier_a_any_invalid"].astype(bool)
    if stratum in {"high_leakage", "low_retention"}:
        return pd.Series(True, index=table.index)
    kind, value = stratum.split(":", 1)
    column = {
        "phase": "murmur_phase",
        "status": "candidate_quality_status",
        "shape": "expert_shape_label",
        "timing": "expert_timing_label",
        "pitch": "expert_pitch_label",
        "grade": "expert_grading_label",
    }[kind]
    return table[column].fillna("").astype(str).eq(value)


def _eligible_indexes(
    table: pd.DataFrame,
    stratum: str,
    requested: int,
    short_samples: int,
) -> list[int]:
    """Return the prespecified eligible set, including fixed QA extremes."""

    if stratum == "high_leakage":
        available = table.index[
            pd.to_numeric(
                table["maximum_s1_s2_leakage_ratio"], errors="coerce"
            ).notna()
        ].tolist()
        return _stratum_order(table, available, stratum)[:requested]
    if stratum == "low_retention":
        available = table.index[
            pd.to_numeric(
                table["murmur_region_energy_retention"], errors="coerce"
            ).notna()
        ].tolist()
        return _stratum_order(table, available, stratum)[:requested]
    return table.index[_final_matches(table, stratum, short_samples)].tolist()


def _stratum_order(table: pd.DataFrame, indexes: list[int], stratum: str) -> list[int]:
    subset = table.loc[indexes].copy()
    if stratum == "high_leakage":
        subset["_rank"] = pd.to_numeric(subset["maximum_s1_s2_leakage_ratio"], errors="coerce").fillna(-np.inf)
        subset.sort_values(["_rank", "candidate_id"], ascending=[False, True], kind="mergesort", inplace=True)
    elif stratum == "low_retention":
        subset["_rank"] = pd.to_numeric(subset["murmur_region_energy_retention"], errors="coerce").fillna(np.inf)
        subset.sort_values(["_rank", "candidate_id"], ascending=[True, True], kind="mergesort", inplace=True)
    elif stratum == "short_support":
        subset["_rank"] = pd.to_numeric(subset["tier_a_candidate_sample_count"], errors="coerce").fillna(np.inf)
        subset.sort_values(["_rank", "candidate_id"], ascending=[True, True], kind="mergesort", inplace=True)
    else:
        subset.sort_values("candidate_id", kind="mergesort", inplace=True)
    return subset.index.tolist()


def select_audit_candidates(
    rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    target_count: int = 30,
    short_candidate_samples: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply frozen final strata and stable identifier tie-breaks."""

    table = prepare_candidate_table(rows)
    if table.empty:
        return table, pd.DataFrame()
    selected: list[int] = []
    selected_set: set[int] = set()
    reasons: dict[int, list[str]] = {}

    for stratum, requested in FINAL_STRATA:
        eligible = _eligible_indexes(
            table, stratum, requested, short_candidate_samples
        )
        already = [index for index in eligible if index in selected_set]
        remaining = max(0, requested - len(already))
        for index in _stratum_order(
            table, [item for item in eligible if item not in selected_set], stratum
        ):
            if remaining == 0 or len(selected) >= target_count:
                break
            selected.append(index)
            selected_set.add(index)
            remaining -= 1
        for index in eligible:
            if index in selected_set:
                reasons.setdefault(index, []).append(stratum)

    for index in table.index:
        if len(selected) >= target_count:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
            reasons.setdefault(index, []).append("stable_candidate_fill")

    selected_table = table.loc[selected].copy()
    selected_table["selection_rank"] = np.arange(1, len(selected_table) + 1)
    selected_table["selection_reasons"] = [
        ";".join(reasons.get(index, ["stable_candidate_fill"])) for index in selected
    ]
    selected_table.reset_index(drop=True, inplace=True)

    shortfalls: list[dict[str, Any]] = []
    selected_ids = set(selected_table["candidate_id"].astype(str))
    for stratum, requested in FINAL_STRATA:
        eligible_indexes = _eligible_indexes(
            table, stratum, requested, short_candidate_samples
        )
        eligible_ids = set(table.loc[eligible_indexes, "candidate_id"].astype(str))
        eligible_count = len(eligible_ids)
        selected_count = len(eligible_ids & selected_ids)
        shortfall = max(0, requested - selected_count)
        shortfalls.append(
            {
                "selection_stage": "final_candidate_selection",
                "stratum": stratum,
                "requested": requested,
                "eligible": eligible_count,
                "selected": selected_count,
                "shortfall": shortfall,
                "shortfall_reason": (
                    "none"
                    if shortfall == 0
                    else "insufficient_eligible"
                    if eligible_count < requested
                    else "target_capacity"
                ),
            }
        )
    return selected_table, pd.DataFrame(shortfalls)


def build_candidate_manifest(
    selected: pd.DataFrame,
    artifact_directories: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Return the stable export schema while preserving invalid reasons verbatim."""

    table = selected.copy()
    paths = artifact_directories or {}
    table["artifact_directory"] = [str(paths.get(str(candidate_id), "")) for candidate_id in table["candidate_id"]]
    for column in AUDIT_MANIFEST_REQUIRED_COLUMNS:
        if column not in table:
            table[column] = None
    leading = list(AUDIT_MANIFEST_REQUIRED_COLUMNS)
    trailing = [column for column in table.columns if column not in leading]
    return table[leading + trailing]


def _phase_metadata(
    recording: dict[str, Any],
    cycle_index: int,
    cycle: Any,
    phase: str,
    sample_rate: int,
    method: str,
    result: SeparationResult,
) -> dict[str, Any]:
    phase_fields = _target_phase_metadata(recording, phase)
    metadata: dict[str, Any] = {
        **recording,
        **phase_fields,
        "cycle_index": cycle_index,
        "phase": "complete_cycle",
        "context_start_sample": cycle.context_start_sample,
        "context_end_sample": cycle.context_end_sample,
        "sample_rate": sample_rate,
        "config_hash": result.config_hash,
        "requested_method": method,
        "target_phase_request": "auto",
        "separation_method": result.selected_method,
        "recording_scope": "task5_metadata_screening",
        "output_profile": "selected_only",
    }
    for name, (start, end) in cycle.phase_bounds.items():
        metadata[f"{name}_relative_start_sample"] = start
        metadata[f"{name}_relative_end_sample"] = end
        metadata[f"{name}_absolute_start_sample"] = cycle.context_start_sample + start
        metadata[f"{name}_absolute_end_sample"] = cycle.context_start_sample + end
    metadata["audit_outcome"] = classify_audit_outcome(
        str(phase_fields["murmur_label"]),
        str(result.metrics["candidate_quality_status"]),
    )
    return metadata


def extract_screening_candidates(
    recordings: list[dict[str, Any]],
    *,
    config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
    audit_config: CandidateAuditConfig = CandidateAuditConfig(),
    audio_dir: Path = AUDIO_DIR,
) -> tuple[list[AuditCandidate], pd.DataFrame]:
    """Run only the fixed screening recordings and keep candidate arrays in memory."""

    candidates: list[AuditCandidate] = []
    skipped: list[dict[str, Any]] = []
    for recording_number, recording in enumerate(recordings, start=1):
        recording_id = str(recording["recording_id"])
        try:
            signal, sample_rate = _load_wav(audio_dir / f"{recording_id}.wav", config.sample_rate)
            annotations = pd.read_csv(
                audio_dir / f"{recording_id}.tsv",
                sep="\t",
                header=None,
                names=["start", "end", "state"],
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            skipped.append({**recording, "cycle_index": None, "murmur_phase": None, "reason": f"recording load failed: {exc}"})
            continue
        positions = np.flatnonzero(annotations["state"].to_numpy() == 2)[: audit_config.cycles_per_recording]
        for cycle_index, position in enumerate(positions):
            try:
                cycle = build_cardiac_cycle_context(signal, annotations, int(position), sample_rate)
            except ValueError as exc:
                skipped.append({**recording, "cycle_index": cycle_index, "murmur_phase": None, "reason": str(exc)})
                continue
            for phase in _target_phases_for_recording(recording, "auto"):
                result = separate_signal(
                    cycle.signal,
                    config=config,
                    method=audit_config.method,
                    phase_masks=cycle.phase_masks,
                    target_phase=phase,
                )
                phase_fields = _target_phase_metadata(recording, phase)
                result.metrics.update(_absolute_timing_metrics(result.metrics, cycle, sample_rate, phase))
                result.metrics.update(_expert_agreement_metrics(result.metrics, phase_fields, phase))
                metadata = _phase_metadata(recording, cycle_index, cycle, phase, sample_rate, audit_config.method, result)
                candidates.append(AuditCandidate(row={**metadata, **result.metrics}, result=result))
        if recording_number % 10 == 0 or recording_number == len(recordings):
            print(f"Screened recordings: {recording_number}/{len(recordings)}; candidates: {len(candidates)}")
    candidates.sort(key=lambda item: _identity_key(item.row))
    return candidates, pd.DataFrame(skipped)


def _candidate_interval(candidate: AuditCandidate) -> tuple[int, int, int, int, np.ndarray]:
    row = candidate.row
    phase = str(row["murmur_phase"])
    phase_start = int(row[f"{phase}_relative_start_sample"])
    phase_end = int(row[f"{phase}_relative_end_sample"])
    onset = row.get("onset_sample")
    offset = row.get("offset_sample")
    onset_in_phase = 0 if onset is None else int(onset)
    offset_in_phase = phase_end - phase_start if offset is None else int(offset)
    start = max(phase_start, min(phase_start + onset_in_phase, phase_end))
    end = max(start, min(phase_start + offset_in_phase, phase_end))
    values = np.asarray(candidate.result.murmur_candidate[start:end], dtype=float)
    return phase_start, phase_end, start, end, values


def _harmonized_psd(values: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < 1:
        return np.asarray([]), np.asarray([])
    nperseg = min(512, len(values))
    noverlap = nperseg // 2 if nperseg > 1 else 0
    frequencies, power = welch(
        values - float(np.mean(values)),
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )
    cfg = DEFAULT_TIER_A_FEATURE_CONFIG
    band = (frequencies >= cfg.spectral_low_hz) & (frequencies <= cfg.spectral_high_hz)
    return frequencies[band], power[band]


def _harmonized_ridge(
    values: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(values) < 1:
        empty = np.asarray([])
        return empty, empty, np.empty((0, 0)), empty
    nperseg = min(128, len(values))
    noverlap = nperseg // 2 if nperseg > 1 else 0
    frequencies, times, power = spectrogram(
        values - float(np.mean(values)),
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        mode="psd",
    )
    cfg = DEFAULT_TIER_A_FEATURE_CONFIG
    band = (frequencies >= cfg.spectral_low_hz) & (frequencies <= cfg.spectral_high_hz)
    band_frequencies = frequencies[band]
    band_power = power[band]
    if band_power.size == 0:
        ridge = np.asarray([])
    else:
        ridge = band_frequencies[np.argmax(band_power, axis=0)]
    return band_frequencies, times, band_power, ridge


def write_tier_a_panel(
    candidate: AuditCandidate,
    destination: Path,
    *,
    sample_rate: int,
) -> None:
    """Write waveform/envelope/25--800 Hz PSD/ridge and numerical checks."""

    row = candidate.row
    result = candidate.result
    phase_start, phase_end, start, end, values = _candidate_interval(candidate)
    time = np.arange(len(result.original)) / sample_rate
    envelope, _, _ = smoothed_hilbert_envelope(
        values,
        sample_rate,
        window_ms=DEFAULT_TIER_A_FEATURE_CONFIG.envelope_window_ms,
        polynomial_order=DEFAULT_TIER_A_FEATURE_CONFIG.envelope_polynomial_order,
        remove_mean=True,
    )
    frequencies, power = _harmonized_psd(values, sample_rate)
    ridge_frequencies, ridge_times, ridge_power, ridge = _harmonized_ridge(values, sample_rate)

    figure, axes = plt.subplots(3, 2, figsize=(15, 13))
    figure.suptitle(
        f"Task 5 Tier A audit: {row['recording_id']} cycle {row['cycle_index']} {row['murmur_phase']}",
        fontsize=14,
    )
    colors = {"s1": "tab:green", "systole": "tab:blue", "s2": "tab:orange", "diastole": "tab:purple"}
    axes[0, 0].plot(time, result.original, color="0.25", linewidth=0.7)
    for name, color in colors.items():
        left = int(row[f"{name}_relative_start_sample"]) / sample_rate
        right = int(row[f"{name}_relative_end_sample"]) / sample_rate
        axes[0, 0].axvspan(left, right, color=color, alpha=0.13, label=name)
    axes[0, 0].axvline(start / sample_rate, color="tab:red", linestyle="--", label="candidate onset")
    axes[0, 0].axvline(end / sample_rate, color="tab:red", linestyle=":", label="candidate offset")
    axes[0, 0].set_title("Original waveform: cardiac phases and candidate boundaries")
    axes[0, 0].set_xlabel("cycle time (s)")
    axes[0, 0].legend(fontsize=7, ncol=3)

    phase_values = result.murmur_candidate[phase_start:phase_end]
    phase_time = np.arange(len(phase_values)) / sample_rate
    axes[0, 1].plot(phase_time, phase_values, linewidth=0.7, color="tab:blue", label="phase murmur candidate")
    if len(values):
        candidate_time = (np.arange(len(values)) + start - phase_start) / sample_rate
        axes[0, 1].plot(candidate_time, envelope, linewidth=1.4, color="tab:red", label="smoothed Hilbert envelope")
        axes[0, 1].axvspan((start - phase_start) / sample_rate, (end - phase_start) / sample_rate, color="tab:red", alpha=0.08)
    axes[0, 1].set_title("Target-phase candidate and frozen envelope path")
    axes[0, 1].set_xlabel("target-phase time (s)")
    axes[0, 1].legend(fontsize=8)

    if len(frequencies):
        axes[1, 0].plot(frequencies, 10 * np.log10(power + 1e-24), linewidth=1.0)
        dominant = row.get("tier_a_psd_dominant_frequency_hz")
        if bool(row.get("tier_a_psd_dominant_frequency_hz_valid", False)) and pd.notna(dominant):
            axes[1, 0].axvline(float(dominant), color="tab:red", linestyle="--", label=f"dominant {float(dominant):.1f} Hz")
            axes[1, 0].legend(fontsize=8)
    else:
        axes[1, 0].text(0.5, 0.5, "insufficient PSD support", ha="center", va="center", transform=axes[1, 0].transAxes)
    axes[1, 0].set_xlim(25, 800)
    axes[1, 0].set_title("Harmonized Welch PSD (25–800 Hz)")
    axes[1, 0].set_xlabel("frequency (Hz)")
    axes[1, 0].set_ylabel("PSD (dB/Hz)")

    if ridge_power.size:
        mesh = axes[1, 1].pcolormesh(ridge_times, ridge_frequencies, 10 * np.log10(ridge_power + 1e-24), shading="auto", cmap="magma")
        axes[1, 1].plot(ridge_times, ridge, color="cyan", linewidth=1.2, label="dominant-frequency ridge")
        figure.colorbar(mesh, ax=axes[1, 1], label="PSD (dB/Hz)")
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "insufficient ridge support", ha="center", va="center", transform=axes[1, 1].transAxes)
    axes[1, 1].set_ylim(25, 800)
    axes[1, 1].set_title("Harmonized dominant-frequency ridge")
    axes[1, 1].set_xlabel("candidate time (s)")
    axes[1, 1].set_ylabel("frequency (Hz)")

    amplitude_values = [
        float(row.get("tier_a_s1_reference_rms", 0.0) or 0.0),
        float(row.get("tier_a_s2_reference_rms", 0.0) or 0.0),
        float(row.get("tier_a_candidate_rms", 0.0) or 0.0),
    ]
    bars = axes[2, 0].bar(["S1 original", "S2 original", "candidate"], amplitude_values, color=["tab:green", "tab:orange", "tab:red"])
    for bar, value in zip(bars, amplitude_values):
        axes[2, 0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4g}", ha="center", va="bottom", fontsize=8)
    axes[2, 0].set_title("RMS support references")

    feature_lines: list[str] = []
    for number, feature in enumerate(TIER_A_FEATURE_COLUMNS, start=1):
        label = PANEL_FEATURE_LABELS[feature]
        valid = bool(row.get(f"{feature}_valid", False))
        if valid:
            value = row.get(feature)
            text = f"{float(value):.5g}" if pd.notna(value) else "NaN"
            feature_lines.append(f"A{number:02d} {label}: {text}")
        else:
            feature_lines.append(
                f"A{number:02d} {label}: INVALID "
                f"({row.get(f'{feature}_invalid_reason')})"
            )
    leakage_values = [
        value
        for value in (
            row.get("s1_leakage_ratio"),
            row.get("s2_leakage_ratio"),
        )
        if value is not None and pd.notna(value)
    ]
    maximum_leakage = max(float(value) for value in leakage_values) if leakage_values else None
    qa_lines = [
        "",
        f"status={row.get('candidate_quality_status')}; method={row.get('separation_method')}",
        f"samples={row.get('tier_a_candidate_sample_count')}; max_s1_s2_leakage={maximum_leakage}",
        f"retention={row.get('murmur_region_energy_retention')}; reconstruction={row.get('reconstruction_error')}",
    ]
    axes[2, 1].axis("off")
    axes[2, 1].text(0.0, 1.0, "\n".join(feature_lines + qa_lines), va="top", family="monospace", fontsize=7.3)
    axes[2, 1].set_title("All 14 Tier A values, validity, support and separation QA")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    completed = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _count_table(selected: pd.DataFrame, short_samples: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions = {
        "phase": "murmur_phase",
        "status": "candidate_quality_status",
        "shape": "expert_shape_label",
        "timing": "expert_timing_label",
        "pitch": "expert_pitch_label",
        "grade": "expert_grading_label",
    }
    for dimension, column in dimensions.items():
        counts = selected[column].fillna("missing").astype(str).value_counts(sort=False)
        for value in sorted(counts.index):
            rows.append({"dimension": dimension, "value": value, "count": int(counts[value])})
    rows.extend(
        [
            {"dimension": "support", "value": f"short_lt_{short_samples}_samples", "count": int(pd.to_numeric(selected["tier_a_candidate_sample_count"], errors="coerce").lt(short_samples).sum())},
            {"dimension": "validity", "value": "any_invalid_feature", "count": int(selected["tier_a_any_invalid"].astype(bool).sum())},
        ]
    )
    return pd.DataFrame(rows)


def run_candidate_audit(
    *,
    run_name: str,
    audit_config: CandidateAuditConfig = CandidateAuditConfig(),
    separation_config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
    metadata_path: Path = METADATA_PATH,
    audio_dir: Path = AUDIO_DIR,
    output_root: Path = DEFAULT_AUDIT_ROOT,
) -> Path:
    """Run a versioned small audit and return the newly created directory."""

    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one non-empty path component")
    validate_input_output_isolation(DATASET_ROOT, OUTPUT_ROOT)
    destination = output_root / run_name
    if destination.exists():
        raise FileExistsError(f"audit run already exists and will not be overwritten: {destination}")

    metadata = pd.read_csv(metadata_path, dtype={"Patient ID": str})
    all_recordings = _all_recordings(metadata, 0, audio_dir=audio_dir)
    screening, screening_shortfalls = select_screening_recordings(
        all_recordings, target_count=audit_config.screening_recording_count
    )
    candidates, skipped = extract_screening_candidates(
        screening,
        config=separation_config,
        audit_config=audit_config,
        audio_dir=audio_dir,
    )
    pool_table = prepare_candidate_table([candidate.row for candidate in candidates])
    selected, final_shortfalls = select_audit_candidates(
        pool_table,
        target_count=audit_config.target_count,
        short_candidate_samples=audit_config.short_candidate_samples,
    )
    if len(selected) < audit_config.target_count:
        print(f"Final audit shortfall: selected {len(selected)} of {audit_config.target_count}")

    destination.mkdir(parents=True)
    candidate_root = destination / "candidates"
    candidate_root.mkdir()
    by_id = {str(item.row["patient_id"]) + ":" + str(item.row["recording_id"]) + ":cycle" + str(int(item.row["cycle_index"])) + ":" + str(item.row["murmur_phase"]): item for item in candidates}
    artifact_paths: dict[str, Path] = {}
    for row in selected.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        candidate = by_id[candidate_id]
        safe_id = candidate_id.replace(":", "_")
        directory = candidate_root / f"{int(row.selection_rank):02d}_{safe_id}"
        directory.mkdir()
        _save_segment_package(candidate.result, directory, candidate.row, separation_config)
        write_tier_a_panel(candidate, directory / "tier_a_audit_panel.png", sample_rate=separation_config.sample_rate)
        tier_payload = {
            "candidate_id": candidate_id,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "features": {
                feature: {
                    "value": candidate.row.get(feature),
                    "unit": TIER_A_FEATURE_UNITS[feature],
                    "valid": candidate.row.get(f"{feature}_valid"),
                    "invalid_reason": candidate.row.get(f"{feature}_invalid_reason"),
                }
                for feature in TIER_A_FEATURE_COLUMNS
            },
            "support_diagnostics": {column: candidate.row.get(column) for column in TIER_A_EXPORT_COLUMNS if column not in TIER_A_FEATURE_COLUMNS and not column.endswith("_valid") and not column.endswith("_invalid_reason")},
            "separation_qa": {name: candidate.row.get(name) for name in ("candidate_quality_status", "activity_detection_method", "s1_leakage_ratio", "s2_leakage_ratio", "outside_murmur_energy_ratio", "murmur_region_energy_retention", "reconstruction_error", "normal_residual_correlation", "phase_selection_used_fallback")},
        }
        (directory / "tier_a_values.json").write_text(json.dumps(tier_payload, indent=2, default=_json_default), encoding="utf-8")
        artifact_paths[candidate_id] = directory.relative_to(destination)

    manifest = build_candidate_manifest(selected, artifact_paths)
    manifest.to_csv(destination / "candidate_manifest.csv", index=False)
    pool_table.to_csv(destination / "screening_candidate_pool.csv", index=False)
    pd.DataFrame(screening).to_csv(destination / "screening_recordings.csv", index=False)
    shortfalls = pd.concat([screening_shortfalls, final_shortfalls], ignore_index=True)
    shortfalls.to_csv(destination / "selection_shortfalls.csv", index=False)
    skipped.to_csv(destination / "skipped.csv", index=False)
    counts = _count_table(selected, audit_config.short_candidate_samples)
    counts.to_csv(destination / "selected_counts.csv", index=False)
    review = manifest[list(IDENTIFIER_COLUMNS) + ["candidate_id", "artifact_directory"]].copy()
    for column in ("phase_boundaries_ok", "candidate_boundaries_ok", "envelope_ok", "psd_ok", "ridge_ok", "numeric_values_ok", "notes"):
        review[column] = ""
    review.to_csv(destination / "manual_review_checklist.csv", index=False)

    rule = {
        "screening_stage": "Location-level Present recordings only; lexicographically first murmur-positive recording per patient; fixed SCREENING_STRATA order and quotas; stable patient/recording ID tie-break; stable metadata fill to screening_recording_count.",
        "final_stage": "Fixed FINAL_STRATA order and quotas. Existing selected candidates count toward later overlapping quotas. Tie-break is candidate_id except short support (ascending sample count), high leakage (descending max S1/S2 leakage), and low retention (ascending target-phase retention), then candidate_id. Stable candidate fill to target_count.",
        "feature_value_blinding": "No numerical Tier A feature value is a selection criterion. Descriptor metadata, candidate status, support length, feature validity flags/reasons, and separation QA are allowed by the prespecified audit objective.",
        "short_definition": f"tier_a_candidate_sample_count < {audit_config.short_candidate_samples}; this is shorter than the frozen maximum 128-sample ridge window.",
        "shortfall_policy": "Report the frozen stratum shortfall; do not hand-pick a replacement.",
        "screening_strata": list(SCREENING_STRATA),
        "final_strata": list(FINAL_STRATA),
    }
    (destination / "selection_rule.json").write_text(json.dumps(rule, indent=2), encoding="utf-8")

    run_manifest = {
        "audit_tool_version": AUDIT_TOOL_VERSION,
        "run_name": run_name,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": _sha256_file(metadata_path),
        "audio_dir": str(audio_dir.resolve()),
        "audit_config": asdict(audit_config),
        "separation_config": separation_config.to_dict(),
        "separation_config_hash": separation_config.config_hash,
        "selected_recordings": len(screening),
        "screened_candidates": len(pool_table),
        "selected_candidates": len(manifest),
        "skipped_items": len(skipped),
        "manifest_sha256": _sha256_file(destination / "candidate_manifest.csv"),
        "candidate_interpretation": "Estimated real murmur candidate for numerical/waveform consistency audit; not clean-source ground truth and not evidence of clinical, construct, classification, or boundary validity.",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }
    (destination / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--target-count", type=int, default=30)
    parser.add_argument("--screening-recordings", type=int, default=40)
    parser.add_argument("--cycles-per-recording", type=int, default=1)
    parser.add_argument("--method", choices=["zcr", "kurtosis"], default="zcr")
    args = parser.parse_args(argv)
    audit_config = CandidateAuditConfig(
        target_count=args.target_count,
        screening_recording_count=args.screening_recordings,
        cycles_per_recording=args.cycles_per_recording,
        method=args.method,
    )
    destination = run_candidate_audit(run_name=args.run_name, audit_config=audit_config)
    print(f"Task 5 candidate audit written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
