"""Deterministic Task 5 Phase 1C boundary-stability audit.

The audit consumes an existing Phase 1B candidate manifest and its saved signal
packages.  It never reruns separation, component selection, or activity-boundary
detection.  Only the Tier A feature-analysis interval is changed.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from config import OUTPUT_ROOT
from src.separation.tier_a_features import (
    FEATURE_SCHEMA_VERSION,
    TIER_A_FEATURE_COLUMNS,
    TIER_A_FEATURE_UNITS,
    extract_tier_a_features,
)


AUDIT_TOOL_VERSION: Final = "task5-phase1c-boundary-stability-v1.0.0"
DEFAULT_PHASE1B_RUN: Final = (
    OUTPUT_ROOT
    / "task5_candidate_audit"
    / "task5_phase1b_real_candidates_v2"
)
DEFAULT_OUTPUT_ROOT: Final = OUTPUT_ROOT / "task5_boundary_stability"
PERTURBATION_MAGNITUDES: Final[tuple[float, ...]] = (0.05, 0.10, 0.20)

# Frozen in protocol order. Values are multipliers of delta_m for onset/offset.
PERTURBATION_FAMILIES: Final[tuple[tuple[str, int, int, str], ...]] = (
    ("shift_earlier", -1, -1, "same intended duration, earlier content"),
    ("shift_later", 1, 1, "same intended duration, later content"),
    ("expand", -1, 1, "add both edges"),
    ("contract", 1, -1, "remove both edges"),
    ("onset_earlier", -1, 0, "add leading content only"),
    ("onset_later", 1, 0, "remove leading content only"),
    ("offset_earlier", 0, -1, "remove trailing content only"),
    ("offset_later", 0, 1, "add trailing content only"),
)

IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "selection_rank",
    "candidate_id",
    "patient_id",
    "recording_id",
    "location",
    "cycle_index",
    "murmur_phase",
)

PRESERVED_COLUMNS: Final[tuple[str, ...]] = (
    *IDENTIFIER_COLUMNS,
    "selection_reasons",
    "candidate_quality_status",
    "expert_timing_label",
    "expert_shape_label",
    "expert_pitch_label",
    "expert_grading_label",
    "expert_quality_label",
    "systole_timing_label",
    "systole_shape_label",
    "systole_pitch_label",
    "systole_grading_label",
    "systole_quality_label",
    "diastole_timing_label",
    "diastole_shape_label",
    "diastole_pitch_label",
    "diastole_grading_label",
    "diastole_quality_label",
    "separation_method",
    "phase_selection_used_fallback",
    "s1_leakage_ratio",
    "s2_leakage_ratio",
    "outside_murmur_energy_ratio",
    "murmur_region_energy_retention",
    "reconstruction_error",
    "normal_residual_correlation",
)

DESCRIPTOR_GROUP_COLUMNS: Final[tuple[str, ...]] = (
    "expert_timing_label",
    "expert_shape_label",
    "expert_pitch_label",
    "expert_grading_label",
    "expert_quality_label",
)


@dataclass(frozen=True)
class BoundaryVariant:
    """One requested and effective perturbation of a candidate interval."""

    variant_order: int
    perturbation_id: str
    family: str
    magnitude_fraction: float
    phase_sample_count: int
    delta_samples: int
    baseline_onset_sample: int
    baseline_offset_sample: int
    intended_onset_sample: int
    intended_offset_sample: int
    actual_onset_sample: int
    actual_offset_sample: int
    intended_onset_displacement_samples: int
    intended_offset_displacement_samples: int
    actual_onset_displacement_samples: int
    actual_offset_displacement_samples: int
    clipped_onset: bool
    clipped_offset: bool
    clipped: bool
    variant_valid: bool
    variant_invalid_reason: str | None


def perturbation_matrix() -> list[dict[str, Any]]:
    """Return the exact frozen 8 x 3 protocol matrix in stable order."""

    rows: list[dict[str, Any]] = []
    order = 0
    for family, onset_direction, offset_direction, interpretation in PERTURBATION_FAMILIES:
        for magnitude in PERTURBATION_MAGNITUDES:
            order += 1
            rows.append(
                {
                    "variant_order": order,
                    "perturbation_id": f"{family}_{int(round(100 * magnitude)):02d}pct",
                    "family": family,
                    "magnitude_fraction": magnitude,
                    "onset_delta_multiplier": onset_direction,
                    "offset_delta_multiplier": offset_direction,
                    "delta_definition": "round(magnitude_fraction * phase_sample_count)",
                    "interpretation": interpretation,
                }
            )
    return rows


def generate_boundary_variants(
    *,
    phase_start_sample: int,
    phase_end_sample: int,
    baseline_onset_sample: int,
    baseline_offset_sample: int,
) -> list[BoundaryVariant]:
    """Generate, clip, and classify all frozen perturbations deterministically."""

    phase_length = phase_end_sample - phase_start_sample
    if phase_length <= 0:
        raise ValueError("phase interval must have positive length")
    if not (
        phase_start_sample <= baseline_onset_sample < baseline_offset_sample <= phase_end_sample
    ):
        raise ValueError("baseline interval must be nonempty and contained in phase")

    variants: list[BoundaryVariant] = []
    for matrix_row in perturbation_matrix():
        delta = int(round(float(matrix_row["magnitude_fraction"]) * phase_length))
        intended_onset_delta = int(matrix_row["onset_delta_multiplier"]) * delta
        intended_offset_delta = int(matrix_row["offset_delta_multiplier"]) * delta
        intended_onset = baseline_onset_sample + intended_onset_delta
        intended_offset = baseline_offset_sample + intended_offset_delta
        actual_onset = min(max(intended_onset, phase_start_sample), phase_end_sample)
        actual_offset = min(max(intended_offset, phase_start_sample), phase_end_sample)
        clipped_onset = actual_onset != intended_onset
        clipped_offset = actual_offset != intended_offset
        if actual_offset < actual_onset:
            invalid_reason = "reversed_interval"
        elif actual_offset == actual_onset:
            invalid_reason = "empty_interval"
        else:
            invalid_reason = None
        variants.append(
            BoundaryVariant(
                variant_order=int(matrix_row["variant_order"]),
                perturbation_id=str(matrix_row["perturbation_id"]),
                family=str(matrix_row["family"]),
                magnitude_fraction=float(matrix_row["magnitude_fraction"]),
                phase_sample_count=phase_length,
                delta_samples=delta,
                baseline_onset_sample=baseline_onset_sample,
                baseline_offset_sample=baseline_offset_sample,
                intended_onset_sample=intended_onset,
                intended_offset_sample=intended_offset,
                actual_onset_sample=actual_onset,
                actual_offset_sample=actual_offset,
                intended_onset_displacement_samples=intended_onset_delta,
                intended_offset_displacement_samples=intended_offset_delta,
                actual_onset_displacement_samples=actual_onset - baseline_onset_sample,
                actual_offset_displacement_samples=actual_offset - baseline_offset_sample,
                clipped_onset=clipped_onset,
                clipped_offset=clipped_offset,
                clipped=clipped_onset or clipped_offset,
                variant_valid=invalid_reason is None,
                variant_invalid_reason=invalid_reason,
            )
        )
    return variants


def samples_to_milliseconds(samples: int | float, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return 1000.0 * float(samples) / sample_rate


def calculate_feature_change(
    baseline_value: Any,
    baseline_valid: bool,
    perturbed_value: Any,
    perturbed_valid: bool,
    *,
    relative_epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Calculate changes and an explicit validity transition."""

    if baseline_valid and perturbed_valid:
        baseline = float(baseline_value)
        perturbed = float(perturbed_value)
        signed = perturbed - baseline
        absolute = abs(signed)
        relative = absolute / abs(baseline) if abs(baseline) > relative_epsilon else np.nan
        transition = "valid_to_valid"
    else:
        signed = absolute = relative = np.nan
        transition = (
            "valid_to_invalid"
            if baseline_valid
            else "invalid_to_valid"
            if perturbed_valid
            else "invalid_to_invalid"
        )
    return {
        "signed_change": signed,
        "absolute_change": absolute,
        "relative_change": relative,
        "relative_change_meaningful": bool(np.isfinite(relative)),
        "validity_transition": transition,
    }


def _clean_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value) if not pd.isna(value) else False


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _candidate_id(row: pd.Series) -> str:
    existing = row.get("candidate_id")
    if pd.notna(existing) and str(existing).strip():
        return str(existing)
    return (
        f"{row['patient_id']}:{row['recording_id']}:"
        f"cycle{int(row['cycle_index'])}:{row['murmur_phase']}"
    )


def _phase_and_baseline(row: pd.Series) -> tuple[int, int, int, int]:
    phase = str(row["murmur_phase"])
    phase_start = int(row[f"{phase}_relative_start_sample"])
    phase_end = int(row[f"{phase}_relative_end_sample"])
    onset = phase_start + int(row["onset_sample"])
    offset = phase_start + int(row["offset_sample"])
    if not phase_start <= onset < offset <= phase_end:
        raise ValueError(f"invalid baseline interval for {_candidate_id(row)}")
    return phase_start, phase_end, onset, offset


def _references(original: np.ndarray, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    s1 = original[int(row["s1_relative_start_sample"]) : int(row["s1_relative_end_sample"])]
    s2 = original[int(row["s2_relative_start_sample"]) : int(row["s2_relative_end_sample"])]
    return np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)


def _preserved(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in PRESERVED_COLUMNS:
        if column in row.index and column not in result:
            result[column] = row[column]
    result["candidate_id"] = _candidate_id(row)
    return result


def _extract_interval(
    signal: np.ndarray,
    sample_rate: int,
    phase_start: int,
    phase_end: int,
    onset: int,
    offset: int,
    s1: np.ndarray,
    s2: np.ndarray,
) -> dict[str, object]:
    return extract_tier_a_features(
        np.asarray(signal[onset:offset], dtype=float),
        sample_rate,
        phase_start_sample=phase_start,
        phase_end_sample=phase_end,
        onset_sample=onset,
        offset_sample=offset,
        s1_reference=s1,
        s2_reference=s2,
    )


def audit_candidate(
    row: pd.Series,
    *,
    source_run: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Audit one frozen candidate and return feature rows, variant rows, plot data."""

    artifact = source_run / str(row["artifact_directory"])
    signal = np.asarray(np.load(artifact / "murmur_candidate.npy"), dtype=float)
    original = np.asarray(np.load(artifact / "original.npy"), dtype=float)
    sample_rate = int(row["sample_rate"])
    phase_start, phase_end, onset, offset = _phase_and_baseline(row)
    s1, s2 = _references(original, row)
    recomputed_baseline = _extract_interval(
        signal, sample_rate, phase_start, phase_end, onset, offset, s1, s2
    )
    # The Phase 1B manifest is the frozen, authoritative baseline. Saved package
    # arrays are float32 and can differ from the in-memory Phase 1B extraction at
    # sub-micro-unit scale, especially for robust envelope slopes.
    baseline = dict(recomputed_baseline)
    for feature in TIER_A_FEATURE_COLUMNS:
        if feature in row.index:
            baseline[feature] = row[feature]
        if f"{feature}_valid" in row.index:
            baseline[f"{feature}_valid"] = _clean_bool(row[f"{feature}_valid"])
        if f"{feature}_invalid_reason" in row.index:
            baseline[f"{feature}_invalid_reason"] = row[f"{feature}_invalid_reason"]
    variants = generate_boundary_variants(
        phase_start_sample=phase_start,
        phase_end_sample=phase_end,
        baseline_onset_sample=onset,
        baseline_offset_sample=offset,
    )
    identity = _preserved(row)
    support_group = "short" if offset - onset < 128 else "normal"
    feature_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    results_by_id: dict[str, dict[str, object]] = {}
    for variant in variants:
        meta = {**identity, **asdict(variant), "sample_rate": sample_rate, "support_group": support_group}
        for displacement in (
            "intended_onset_displacement",
            "intended_offset_displacement",
            "actual_onset_displacement",
            "actual_offset_displacement",
        ):
            meta[f"{displacement}_ms"] = samples_to_milliseconds(
                meta[f"{displacement}_samples"], sample_rate
            )
        if variant.variant_valid:
            perturbed = _extract_interval(
                signal,
                sample_rate,
                phase_start,
                phase_end,
                variant.actual_onset_sample,
                variant.actual_offset_sample,
                s1,
                s2,
            )
        else:
            perturbed = {}
        results_by_id[variant.perturbation_id] = perturbed
        variant_rows.append(meta)
        for feature in TIER_A_FEATURE_COLUMNS:
            baseline_valid = _clean_bool(baseline[f"{feature}_valid"])
            perturbed_valid = (
                _clean_bool(perturbed.get(f"{feature}_valid", False))
                if variant.variant_valid
                else False
            )
            baseline_value = baseline[feature]
            perturbed_value = perturbed.get(feature, np.nan)
            change = calculate_feature_change(
                baseline_value,
                baseline_valid,
                perturbed_value,
                perturbed_valid,
            )
            invalid_reason = (
                perturbed.get(f"{feature}_invalid_reason")
                if variant.variant_valid
                else variant.variant_invalid_reason
            )
            feature_rows.append(
                {
                    **meta,
                    "feature": feature,
                    "feature_unit": TIER_A_FEATURE_UNITS[feature],
                    "baseline_value": baseline_value,
                    "baseline_valid": baseline_valid,
                    "baseline_invalid_reason": baseline[f"{feature}_invalid_reason"],
                    "perturbed_value": perturbed_value,
                    "perturbed_valid": perturbed_valid,
                    "perturbed_invalid_reason": invalid_reason,
                    "invalid_reason": invalid_reason,
                    **change,
                }
            )
    plot_data = {
        "signal": signal,
        "sample_rate": sample_rate,
        "phase_start": phase_start,
        "phase_end": phase_end,
        "baseline_onset": onset,
        "baseline_offset": offset,
        "baseline": baseline,
        "recomputed_baseline": recomputed_baseline,
        "results": results_by_id,
    }
    return feature_rows, variant_rows, plot_data


def _quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(q, interpolation="linear")) if len(numeric) else np.nan


def _summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    changes = pd.to_numeric(group["absolute_change"], errors="coerce")
    valid_changes = changes.dropna()
    candidate_worst = (
        group.assign(_change=changes)
        .groupby("candidate_id", sort=True, dropna=False)["_change"]
        .max()
        .dropna()
    )
    if len(candidate_worst):
        worst_value = float(candidate_worst.max())
        worst_candidate = str(candidate_worst[candidate_worst.eq(worst_value)].index[0])
    else:
        worst_value = np.nan
        worst_candidate = None
    transitions = group["validity_transition"].value_counts()
    return {
        "candidate_count": int(group["candidate_id"].nunique()),
        "variant_feature_row_count": int(len(group)),
        "valid_change_count": int(len(valid_changes)),
        "median_absolute_change": _quantile(changes, 0.50),
        "absolute_change_q25": _quantile(changes, 0.25),
        "absolute_change_q75": _quantile(changes, 0.75),
        "absolute_change_iqr": _quantile(changes, 0.75) - _quantile(changes, 0.25),
        "absolute_change_q90": _quantile(changes, 0.90),
        "absolute_change_q95": _quantile(changes, 0.95),
        "absolute_change_q99": _quantile(changes, 0.99),
        "validity_transition_count": int(
            transitions.get("valid_to_invalid", 0) + transitions.get("invalid_to_valid", 0)
        ),
        "valid_to_invalid_count": int(transitions.get("valid_to_invalid", 0)),
        "invalid_to_valid_count": int(transitions.get("invalid_to_valid", 0)),
        "clipped_row_count": int(group["clipped"].astype(bool).sum()),
        "candidate_worst_absolute_change": worst_value,
        "candidate_worst_id": worst_candidate,
    }


def summarize_feature_stability(feature_rows: pd.DataFrame) -> pd.DataFrame:
    """Return stable, reproducible robust statistics in registry order."""

    baseline_iqr = (
        feature_rows.drop_duplicates(["candidate_id", "feature"])
        .groupby("feature", sort=False)["baseline_value"]
        .agg(lambda values: _quantile(pd.Series(values), 0.75) - _quantile(pd.Series(values), 0.25))
    )
    rows: list[dict[str, Any]] = []
    for feature in TIER_A_FEATURE_COLUMNS:
        group = feature_rows.loc[feature_rows["feature"].eq(feature)]
        summary = _summarize_group(group)
        iqr = float(baseline_iqr.get(feature, np.nan))
        summary["baseline_iqr"] = iqr
        summary["median_iqr_standardized_absolute_change"] = (
            summary["median_absolute_change"] / iqr
            if np.isfinite(iqr) and iqr > 0
            else np.nan
        )
        rows.append({"feature": feature, "feature_unit": TIER_A_FEATURE_UNITS[feature], **summary})
    return pd.DataFrame(rows)


def summarize_candidate_worst_cases(feature_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, feature), group in feature_rows.groupby(
        ["candidate_id", "feature"], sort=True, dropna=False
    ):
        numeric = pd.to_numeric(group["absolute_change"], errors="coerce")
        if numeric.notna().any():
            maximum = float(numeric.max())
            selected = group.loc[numeric.eq(maximum)].sort_values("variant_order", kind="mergesort").iloc[0]
            perturbation_id = selected["perturbation_id"]
        else:
            maximum = np.nan
            perturbation_id = None
        rows.append(
            {
                "candidate_id": candidate_id,
                "feature": feature,
                "worst_absolute_change": maximum,
                "worst_perturbation_id": perturbation_id,
                "validity_transition_count": int(
                    group["validity_transition"].isin(["valid_to_invalid", "invalid_to_valid"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_perturbation_stability(feature_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize each feature separately by family and phase-relative magnitude."""

    rows: list[dict[str, Any]] = []
    for feature in TIER_A_FEATURE_COLUMNS:
        feature_group = feature_rows.loc[feature_rows["feature"].eq(feature)]
        for family, _, _, _ in PERTURBATION_FAMILIES:
            for magnitude in PERTURBATION_MAGNITUDES:
                group = feature_group.loc[
                    feature_group["family"].eq(family)
                    & np.isclose(feature_group["magnitude_fraction"], magnitude)
                ]
                rows.append(
                    {
                        "feature": feature,
                        "family": family,
                        "magnitude_fraction": magnitude,
                        **_summarize_group(group),
                    }
                )
    return pd.DataFrame(rows)


def summarize_grouped_stability(
    feature_rows: pd.DataFrame,
    *,
    minimum_group_candidates: int = 2,
) -> pd.DataFrame:
    """Summarize status, phase, support, and sufficiently populated descriptors."""

    dimensions = (
        ("status", "candidate_quality_status"),
        ("phase", "murmur_phase"),
        ("support", "support_group"),
        *((f"descriptor:{column}", column) for column in DESCRIPTOR_GROUP_COLUMNS),
    )
    rows: list[dict[str, Any]] = []
    for dimension, column in dimensions:
        if column not in feature_rows:
            continue
        values = feature_rows[column].fillna("missing").astype(str)
        for group_value in sorted(values.unique()):
            subset = feature_rows.loc[values.eq(group_value)]
            if subset["candidate_id"].nunique() < minimum_group_candidates:
                continue
            for feature in TIER_A_FEATURE_COLUMNS:
                group = subset.loc[subset["feature"].eq(feature)]
                rows.append(
                    {
                        "group_dimension": dimension,
                        "group_value": group_value,
                        "feature": feature,
                        **_summarize_group(group),
                    }
                )
    return pd.DataFrame(rows)


def _representative_rows(feature_rows: pd.DataFrame) -> dict[str, pd.Series | None]:
    valid = feature_rows.loc[
        feature_rows["validity_transition"].eq("valid_to_valid")
        & ~feature_rows["clipped"].astype(bool)
    ].copy()
    valid["relative_sort"] = pd.to_numeric(valid["relative_change"], errors="coerce")
    valid["absolute_sort"] = pd.to_numeric(valid["absolute_change"], errors="coerce")
    stable = (
        valid.sort_values(
            ["relative_sort", "absolute_sort", "candidate_id", "variant_order"],
            kind="mergesort",
            na_position="last",
        ).iloc[0]
        if len(valid)
        else None
    )
    unstable = (
        valid.sort_values(
            ["relative_sort", "absolute_sort", "candidate_id", "variant_order"],
            ascending=[False, False, True, True],
            kind="mergesort",
            na_position="last",
        ).iloc[0]
        if len(valid)
        else None
    )
    clipped = feature_rows.loc[feature_rows["clipped"].astype(bool)].sort_values(
        ["candidate_id", "variant_order", "feature"], kind="mergesort"
    )
    transition = feature_rows.loc[
        feature_rows["validity_transition"].isin(["valid_to_invalid", "invalid_to_valid"])
    ].sort_values(["candidate_id", "variant_order", "feature"], kind="mergesort")
    return {
        "stable": stable,
        "unstable": unstable,
        "clipped": clipped.iloc[0] if len(clipped) else None,
        "invalid_transition": transition.iloc[0] if len(transition) else None,
    }


def _diagnostic_panel(
    selected: pd.Series,
    plot_data: dict[str, Any],
    destination: Path,
    label: str,
) -> None:
    signal = plot_data["signal"]
    sample_rate = plot_data["sample_rate"]
    phase_start = plot_data["phase_start"]
    phase_end = plot_data["phase_end"]
    baseline_onset = plot_data["baseline_onset"]
    baseline_offset = plot_data["baseline_offset"]
    actual_onset = int(selected["actual_onset_sample"])
    actual_offset = int(selected["actual_offset_sample"])
    phase_signal = signal[phase_start:phase_end]
    time = np.arange(len(phase_signal)) / sample_rate * 1000.0
    baseline = plot_data["baseline"]
    perturbed = plot_data["results"][str(selected["perturbation_id"])]

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(time, phase_signal, color="0.25", linewidth=0.8)
    axes[0].axvspan(
        (baseline_onset - phase_start) / sample_rate * 1000.0,
        (baseline_offset - phase_start) / sample_rate * 1000.0,
        color="tab:blue",
        alpha=0.16,
        label="baseline interval",
    )
    axes[0].axvspan(
        (actual_onset - phase_start) / sample_rate * 1000.0,
        (actual_offset - phase_start) / sample_rate * 1000.0,
        color="tab:orange",
        alpha=0.20,
        label="effective perturbed interval",
    )
    axes[0].set_title(
        f"{label}: {selected['candidate_id']} — {selected['perturbation_id']}"
    )
    axes[0].set_xlabel("target-phase time (ms)")
    axes[0].set_ylabel("separated candidate amplitude")
    axes[0].legend(fontsize=8)

    changes: list[float] = []
    colors: list[str] = []
    labels: list[str] = []
    for index, feature in enumerate(TIER_A_FEATURE_COLUMNS, start=1):
        base_valid = _clean_bool(baseline[f"{feature}_valid"])
        changed_valid = _clean_bool(perturbed.get(f"{feature}_valid", False))
        change = calculate_feature_change(
            baseline[feature], base_valid, perturbed.get(feature, np.nan), changed_valid
        )
        changes.append(float(change["relative_change"]) if np.isfinite(change["relative_change"]) else 0.0)
        colors.append("tab:red" if change["validity_transition"] != "valid_to_valid" else "tab:green")
        labels.append(f"A{index:02d}")
    axes[1].bar(labels, changes, color=colors)
    axes[1].set_ylabel("absolute relative change (0 when undefined)")
    axes[1].set_title(
        f"feature={selected['feature']}; transition={selected['validity_transition']}; "
        f"clipped={bool(selected['clipped'])}; invalid_reason={selected['invalid_reason']}"
    )
    axes[1].tick_params(axis="x", labelrotation=45)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def _write_heatmap(summary: pd.DataFrame, destination: Path) -> None:
    values = summary.set_index("feature")[
        ["median_absolute_change", "absolute_change_q90", "absolute_change_q95", "validity_transition_count"]
    ].to_numpy(dtype=float)
    scaled = values.copy()
    for column in range(scaled.shape[1]):
        maximum = np.nanmax(scaled[:, column])
        if np.isfinite(maximum) and maximum > 0:
            scaled[:, column] /= maximum
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = axis.imshow(scaled, aspect="auto", cmap="magma", vmin=0, vmax=1)
    axis.set_yticks(range(len(summary)), [f"A{i:02d}" for i in range(1, len(summary) + 1)])
    axis.set_xticks(
        range(4), ["median |change|", "q90 |change|", "q95 |change|", "validity transitions"], rotation=25, ha="right"
    )
    axis.set_title("Task 5 Phase 1C robust stability overview (column-normalized)")
    figure.colorbar(image, ax=axis, label="within-column normalized value")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def run_boundary_stability_audit(
    *,
    run_name: str,
    source_run: Path = DEFAULT_PHASE1B_RUN,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Run the 24-condition audit on exactly one frozen Phase 1B manifest."""

    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one non-empty path component")
    source_run = Path(source_run)
    manifest_path = source_run / "candidate_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase 1B candidate manifest not found: {manifest_path}")
    destination = Path(output_root) / run_name
    if destination.exists():
        raise FileExistsError(f"boundary audit already exists and will not be overwritten: {destination}")
    destination.mkdir(parents=True)

    # Freeze the matrix artifact before any feature values are loaded or analyzed.
    matrix_payload = {
        "audit_tool_version": AUDIT_TOOL_VERSION,
        "protocol_design": "24 perturbations: 8 directions x 5%, 10%, 20% of target-phase samples",
        "rounding": "Python round to nearest integer sample (ties to even)",
        "clipping": "clip each requested boundary independently to the inclusive/exclusive target-phase endpoints; never swap or force support",
        "matrix": perturbation_matrix(),
    }
    matrix_path = destination / "perturbation_matrix.json"
    matrix_path.write_text(json.dumps(matrix_payload, indent=2), encoding="utf-8")

    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str, "recording_id": str})
    if len(manifest) != 30:
        raise ValueError(f"Phase 1C requires exactly 30 frozen candidates; found {len(manifest)}")
    manifest["candidate_id"] = [_candidate_id(row) for _, row in manifest.iterrows()]
    if manifest["candidate_id"].duplicated().any():
        raise ValueError("candidate manifest has duplicate candidate identifiers")
    manifest.sort_values("selection_rank", kind="mergesort", inplace=True)

    feature_records: list[dict[str, Any]] = []
    variant_records: list[dict[str, Any]] = []
    plot_data: dict[str, dict[str, Any]] = {}
    for _, row in manifest.iterrows():
        features, variants, candidate_plot_data = audit_candidate(row, source_run=source_run)
        feature_records.extend(features)
        variant_records.extend(variants)
        plot_data[_candidate_id(row)] = candidate_plot_data

    feature_rows = pd.DataFrame(feature_records).sort_values(
        ["selection_rank", "variant_order", "feature"], kind="mergesort"
    )
    variant_rows = pd.DataFrame(variant_records).sort_values(
        ["selection_rank", "variant_order"], kind="mergesort"
    )
    summary = summarize_feature_stability(feature_rows)
    grouped = summarize_grouped_stability(feature_rows)
    perturbation_summary = summarize_perturbation_stability(feature_rows)
    worst = summarize_candidate_worst_cases(feature_rows)

    feature_rows.to_csv(destination / "feature_changes_long.csv", index=False)
    variant_rows.to_csv(destination / "variant_manifest.csv", index=False)
    summary.to_csv(destination / "feature_stability_summary.csv", index=False)
    grouped.to_csv(destination / "grouped_stability_summary.csv", index=False)
    perturbation_summary.to_csv(
        destination / "perturbation_stability_summary.csv", index=False
    )
    worst.to_csv(destination / "candidate_worst_cases.csv", index=False)

    invalid_counts = (
        variant_rows.assign(
            variant_invalid_reason=variant_rows["variant_invalid_reason"].fillna("none")
        )
        .groupby(["clipped", "variant_valid", "variant_invalid_reason"], dropna=False, sort=True)
        .size()
        .reset_index(name="variant_count")
    )
    transition_counts = (
        feature_rows.groupby(
            ["validity_transition", "perturbed_invalid_reason"], dropna=False, sort=True
        )
        .size()
        .reset_index(name="feature_row_count")
    )
    invalid_counts.to_csv(destination / "variant_status_counts.csv", index=False)
    transition_counts.to_csv(destination / "validity_transition_counts.csv", index=False)

    diagnostic_dir = destination / "diagnostics"
    diagnostic_dir.mkdir()
    representatives = _representative_rows(feature_rows)
    diagnostic_index: list[dict[str, Any]] = []
    for label, selected in representatives.items():
        if selected is None:
            diagnostic_index.append({"diagnostic_type": label, "available": False, "reason": "no matching example"})
            continue
        filename = f"{label}_example.png"
        _diagnostic_panel(
            selected,
            plot_data[str(selected["candidate_id"])],
            diagnostic_dir / filename,
            label.replace("_", " ").title(),
        )
        diagnostic_index.append(
            {
                "diagnostic_type": label,
                "available": True,
                "candidate_id": selected["candidate_id"],
                "perturbation_id": selected["perturbation_id"],
                "feature": selected["feature"],
                "validity_transition": selected["validity_transition"],
                "clipped": selected["clipped"],
                "invalid_reason": selected["invalid_reason"],
                "artifact": str(Path("diagnostics") / filename),
            }
        )
    _write_heatmap(summary, diagnostic_dir / "stability_overview_heatmap.png")
    diagnostic_index.append(
        {"diagnostic_type": "overview_heatmap", "available": True, "artifact": "diagnostics/stability_overview_heatmap.png"}
    )
    pd.DataFrame(diagnostic_index).to_csv(destination / "diagnostic_index.csv", index=False)

    source_run_manifest = source_run / "run_manifest.json"
    run_manifest = {
        "audit_tool_version": AUDIT_TOOL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "run_name": run_name,
        "source_phase1b_run": str(source_run.resolve()),
        "source_candidate_manifest": str(manifest_path.resolve()),
        "source_candidate_manifest_sha256": _sha256_file(manifest_path),
        "source_run_manifest_sha256": _sha256_file(source_run_manifest) if source_run_manifest.exists() else None,
        "perturbation_matrix_sha256": _sha256_file(matrix_path),
        "candidate_count": int(manifest["candidate_id"].nunique()),
        "variant_count": int(len(variant_rows)),
        "feature_row_count": int(len(feature_rows)),
        "clipped_variant_count": int(variant_rows["clipped"].astype(bool).sum()),
        "invalid_variant_count": int((~variant_rows["variant_valid"].astype(bool)).sum()),
        "validity_transition_count": int(
            feature_rows["validity_transition"].isin(["valid_to_invalid", "invalid_to_valid"]).sum()
        ),
        "scope_statement": "Boundary stability only; no clinical validity, construct validity, classification usefulness, or redundancy inference.",
        "processing_statement": "Loaded only the 30 Phase 1B selected packages; did not rerun separation, component assignment, candidate selection, fallback logic, or onset/offset detection.",
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, default=_json_value), encoding="utf-8"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_PHASE1B_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    destination = run_boundary_stability_audit(
        run_name=args.run_name,
        source_run=args.source_run,
        output_root=args.output_root,
    )
    print(f"Task 5 boundary-stability audit written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
