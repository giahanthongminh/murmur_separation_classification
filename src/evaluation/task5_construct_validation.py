"""Task 5 Phase 2B waveform-aligned CirCor construct validation.

This analysis consumes a completed Phase 2A feature table.  It does not rerun
separation or feature extraction and never uses clinical Outcome.  The primary
unit is a patient at the annotated most-audible location, with cycle medians
first computed per recording and repeated recordings reduced by the median of
recording medians.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Final, Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.stats import kruskal, mannwhitneyu, spearmanr

from config import METADATA_PATH, OUTPUT_ROOT
from src.evaluation.task5_candidate_quality import assign_high_quality_stratum
from src.separation.tier_a_features import (
    FEATURE_SCHEMA_VERSION,
    TIER_A_FEATURE_COLUMNS,
    TIER_A_FEATURE_UNITS,
)


TOOL_VERSION: Final = "task5-phase2b-construct-validation-v1.0.0"
DEFAULT_OUTPUT_ROOT: Final = OUTPUT_ROOT / "task5_construct_validation"

DURATION: Final = "tier_a_relative_murmur_duration"
MIDPOINT: Final = "tier_a_temporal_midpoint_normalized"
RMS_DB: Final = "tier_a_murmur_rms_relative_s1_s2_db"
PEAK_POSITION: Final = "tier_a_envelope_peak_position_normalized"
RISE_SLOPE: Final = "tier_a_envelope_rise_slope_robust"
DECAY_SLOPE: Final = "tier_a_envelope_decay_slope_robust"
FULLNESS: Final = "tier_a_envelope_fullness"
DOMINANT_FREQUENCY: Final = "tier_a_psd_dominant_frequency_hz"
MEDIAN_FREQUENCY: Final = "tier_a_psd_median_frequency_hz"
BANDWIDTH: Final = "tier_a_psd_bandwidth_95_hz"
HIGH_FREQUENCY_FRACTION: Final = "tier_a_psd_energy_fraction_above_200_hz"
ENTROPY: Final = "tier_a_psd_entropy_normalized"

STRATA: Final[tuple[str, ...]] = ("all_valid", "accepted_only", "high_quality")
DESCRIPTOR_COLUMNS: Final[tuple[str, ...]] = (
    "expert_timing_label",
    "expert_shape_label",
    "expert_pitch_label",
    "expert_grading_label",
    "expert_quality_label",
)


@dataclass(frozen=True)
class ConstructValidationConfig:
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 20260827
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")


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


def _seed(config: ConstructValidationConfig, key: Iterable[Any]) -> int:
    offset = int.from_bytes(
        sha256("|".join(map(str, key)).encode("utf-8")).digest()[:4], "little"
    )
    return (config.bootstrap_seed + offset) % (2**32)


def _clean_label(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    return None if not value or value.lower() == "nan" else value


def _holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Holm family-wise adjusted p-values, preserving missing entries."""

    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    finite = pd.to_numeric(pvalues, errors="coerce").dropna().sort_values()
    if finite.empty:
        return result
    adjusted = np.maximum.accumulate(
        np.minimum(1.0, finite.to_numpy() * np.arange(len(finite), 0, -1))
    )
    result.loc[finite.index] = adjusted
    return result


def _cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return np.nan
    differences = left[:, None] - right[None, :]
    return float((np.count_nonzero(differences > 0) - np.count_nonzero(differences < 0)) / differences.size)


def _percentile_interval(
    values: list[float], config: ConstructValidationConfig
) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    failure_rate = 1.0 - len(finite) / config.bootstrap_replicates
    if not len(finite):
        return np.nan, np.nan, float(failure_rate)
    alpha = (1.0 - config.confidence_level) / 2.0
    low, high = np.quantile(finite, [alpha, 1 - alpha])
    return float(low), float(high), float(failure_rate)


def _bootstrap_rows(
    frame: pd.DataFrame,
    statistic: Any,
    *,
    config: ConstructValidationConfig,
    key: Iterable[Any],
) -> tuple[float, float, float]:
    """Patient bootstrap for the one-row-per-patient primary table."""

    if frame.empty:
        return np.nan, np.nan, 1.0
    rng = np.random.default_rng(_seed(config, key))
    values: list[float] = []
    for _ in range(config.bootstrap_replicates):
        sampled = frame.iloc[rng.integers(0, len(frame), len(frame))]
        try:
            values.append(float(statistic(sampled)))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            values.append(np.nan)
    return _percentile_interval(values, config)


def _validate_source(source_run: Path, metadata_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_path = source_run / "task5_feature_table.csv"
    manifest_path = source_run / "run_manifest.json"
    for path in (feature_path, manifest_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"required Phase 2B source not found: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Phase 2A source run is not complete")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Phase 2A feature schema differs from the frozen Tier A schema")
    expected_table_hash = manifest.get("artifact_sha256", {}).get("task5_feature_table.csv")
    if expected_table_hash != _sha256_file(feature_path):
        raise ValueError("Phase 2A feature-table hash does not match its manifest")
    if manifest.get("metadata_sha256") != _sha256_file(metadata_path):
        raise ValueError("CirCor metadata differs from the Phase 2A snapshot")
    table = pd.read_csv(feature_path, dtype={"patient_id": str, "recording_id": str})
    required = {
        "candidate_id", "patient_id", "recording_id", "location", "murmur_phase",
        "patient_murmur_label", "candidate_quality_status", *DESCRIPTOR_COLUMNS,
        "s1_leakage_ratio", "s2_leakage_ratio", "murmur_region_energy_retention",
        "outside_murmur_energy_ratio", "noise_energy_ratio",
    }
    for feature in TIER_A_FEATURE_COLUMNS:
        required.update((feature, f"{feature}_valid"))
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Phase 2A feature table is missing columns: {sorted(missing)}")
    if table["candidate_id"].duplicated().any():
        raise ValueError("Phase 2A feature table has duplicate candidate identifiers")
    return table, manifest


def prepare_candidate_strata(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the descriptor-blind frozen Phase 1D high-quality definition."""

    assigned, thresholds = assign_high_quality_stratum(table)
    assigned["all_valid"] = True
    assigned["accepted_only"] = assigned["candidate_quality_status"].astype(str).eq("accepted")
    assigned["high_quality"] = assigned["high_quality_stratum"].fillna(False).astype(bool)
    return assigned, thresholds


def _metadata_primary_locations(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"Patient ID", "Murmur", "Most audible location"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing construct columns: {sorted(missing)}")
    selected = metadata.loc[metadata["Murmur"].astype(str).eq("Present")].copy()
    selected["patient_id"] = selected["Patient ID"].astype(str)
    selected["most_audible_location"] = selected["Most audible location"].map(_clean_label)
    if selected["patient_id"].duplicated().any():
        raise ValueError("metadata must contain one row per patient")
    return selected[["patient_id", "most_audible_location"]]


def build_primary_patient_table(
    candidates: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one feature value per patient/phase/stratum at the primary site."""

    locations = _metadata_primary_locations(metadata)
    present = candidates.merge(locations, on="patient_id", how="inner", validate="many_to_one")
    present = present.loc[present["location"].astype(str).eq(present["most_audible_location"].astype(str))].copy()
    rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    for phase in ("systole", "diastole"):
        phase_rows = present.loc[present["murmur_phase"].astype(str).eq(phase)]
        for stratum in STRATA:
            eligible = phase_rows.loc[phase_rows[stratum].fillna(False).astype(bool)]
            flow_rows.append({
                "phase": phase, "stratum": stratum, "feature": "all",
                "candidate_count": int(len(eligible)),
                "recording_count": int(eligible["recording_id"].nunique()),
                "patient_count": int(eligible["patient_id"].nunique()),
            })
            for feature in TIER_A_FEATURE_COLUMNS:
                valid = eligible[f"{feature}_valid"].fillna(False).astype(bool)
                feature_rows = eligible.loc[valid].copy()
                feature_rows[feature] = pd.to_numeric(feature_rows[feature], errors="coerce")
                feature_rows = feature_rows.dropna(subset=[feature])
                flow_rows.append({
                    "phase": phase, "stratum": stratum, "feature": feature,
                    "candidate_count": int(len(feature_rows)),
                    "recording_count": int(feature_rows["recording_id"].nunique()),
                    "patient_count": int(feature_rows["patient_id"].nunique()),
                })
                recording_keys = ["patient_id", "recording_id", "location"]
                recording = feature_rows.groupby(recording_keys, sort=True, dropna=False).agg(
                    value=(feature, "median"), valid_cycle_count=(feature, "size")
                ).reset_index()
                for patient_id, group in recording.groupby("patient_id", sort=True):
                    source = feature_rows.loc[feature_rows["patient_id"].eq(patient_id)].iloc[0]
                    rows.append({
                        "patient_id": patient_id,
                        "location": source["location"],
                        "murmur_phase": phase,
                        "stratum": stratum,
                        "feature": feature,
                        "unit": TIER_A_FEATURE_UNITS[feature],
                        "value": float(group["value"].median()),
                        "recording_count": int(len(group)),
                        "valid_cycle_count": int(group["valid_cycle_count"].sum()),
                        **{column: _clean_label(source.get(column)) for column in DESCRIPTOR_COLUMNS},
                    })
    primary = pd.DataFrame(rows)
    if not primary.empty and primary.duplicated(["patient_id", "murmur_phase", "stratum", "feature"]).any():
        raise AssertionError("primary construct table must have one row per patient-feature-stratum")
    return primary, pd.DataFrame(flow_rows)


def _summaries(frame: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, values in frame.groupby(label, sort=True):
        numeric = pd.to_numeric(values["value"], errors="coerce").dropna()
        rows.append({
            "group": group, "patient_count": int(len(numeric)),
            "median": float(numeric.median()) if len(numeric) else np.nan,
            "q25": float(numeric.quantile(.25)) if len(numeric) else np.nan,
            "q75": float(numeric.quantile(.75)) if len(numeric) else np.nan,
            "iqr": float(numeric.quantile(.75) - numeric.quantile(.25)) if len(numeric) else np.nan,
        })
    return rows


def _pairwise_result(
    frame: pd.DataFrame, label: str, left: str, right: str,
    *, config: ConstructValidationConfig, key: tuple[Any, ...],
) -> dict[str, Any]:
    usable = frame.loc[frame[label].isin([left, right])].copy()
    left_values = usable.loc[usable[label].eq(left), "value"].to_numpy(float)
    right_values = usable.loc[usable[label].eq(right), "value"].to_numpy(float)
    median_difference = float(np.median(left_values) - np.median(right_values)) if len(left_values) and len(right_values) else np.nan
    delta = _cliffs_delta(left_values, right_values)
    median_ci = _bootstrap_rows(
        usable,
        lambda sample: sample.loc[sample[label].eq(left), "value"].median() - sample.loc[sample[label].eq(right), "value"].median(),
        config=config, key=(*key, "median_difference"),
    )
    delta_ci = _bootstrap_rows(
        usable,
        lambda sample: _cliffs_delta(
            sample.loc[sample[label].eq(left), "value"].to_numpy(float),
            sample.loc[sample[label].eq(right), "value"].to_numpy(float),
        ),
        config=config, key=(*key, "cliffs_delta"),
    )
    pvalue = np.nan
    if len(left_values) and len(right_values):
        pvalue = float(mannwhitneyu(left_values, right_values, alternative="two-sided").pvalue)
    return {
        "comparison": f"{left}_minus_{right}", "group_a": left, "group_b": right,
        "group_a_n": len(left_values), "group_b_n": len(right_values),
        "estimate_name": "median_difference", "estimate": median_difference,
        "estimate_ci_low": median_ci[0], "estimate_ci_high": median_ci[1],
        "effect_name": "cliffs_delta", "effect": delta,
        "effect_ci_low": delta_ci[0], "effect_ci_high": delta_ci[1],
        "bootstrap_failure_rate": max(median_ci[2], delta_ci[2]), "raw_p_value": pvalue,
        "eligible_patient_count": int(len(frame)),
        "descriptor_nonmissing_patient_count": int(frame[label].notna().sum()),
        "descriptor_missing_patient_count": int(frame[label].isna().sum()),
        "location_distribution": json.dumps(
            frame.loc[frame[label].isin([left, right]), "location"].value_counts().sort_index().to_dict()
        ),
    }


def _ordered_result(
    frame: pd.DataFrame, label: str, order: dict[str, int],
    *, config: ConstructValidationConfig, key: tuple[Any, ...],
) -> dict[str, Any]:
    usable = frame.loc[frame[label].isin(order)].copy()
    usable["_order"] = usable[label].map(order).astype(float)
    estimate = float(spearmanr(usable["_order"], usable["value"]).statistic) if usable[label].nunique() > 1 else np.nan
    interval = _bootstrap_rows(
        usable,
        lambda sample: spearmanr(sample["_order"], sample["value"]).statistic if sample[label].nunique() > 1 else np.nan,
        config=config, key=(*key, "ordered_spearman"),
    )
    pvalue = float(spearmanr(usable["_order"], usable["value"]).pvalue) if usable[label].nunique() > 1 else np.nan
    counts = usable[label].value_counts().sort_index().to_dict()
    return {
        "comparison": "ordered_trend", "group_a": " < ".join(order), "group_b": None,
        "group_a_n": len(usable), "group_b_n": np.nan,
        "estimate_name": "spearman_rho", "estimate": estimate,
        "estimate_ci_low": interval[0], "estimate_ci_high": interval[1],
        "effect_name": "ordered_spearman", "effect": estimate,
        "effect_ci_low": interval[0], "effect_ci_high": interval[1],
        "bootstrap_failure_rate": interval[2], "raw_p_value": pvalue,
        "ordered_group_counts": json.dumps(counts, sort_keys=True),
        "eligible_patient_count": int(len(frame)),
        "descriptor_nonmissing_patient_count": int(frame[label].notna().sum()),
        "descriptor_missing_patient_count": int(frame[label].isna().sum()),
        "location_distribution": json.dumps(
            usable["location"].value_counts().sort_index().to_dict()
        ),
        "secondary_test": "Jonckheere-Terpstra unavailable in the pinned SciPy; ordered Spearman is primary",
    }


def _omnibus_result(
    frame: pd.DataFrame,
    label: str,
    groups: tuple[str, ...],
    *,
    config: ConstructValidationConfig,
    key: tuple[Any, ...],
) -> dict[str, Any]:
    usable = frame.loc[frame[label].isin(groups)]
    arrays = [usable.loc[usable[label].eq(group), "value"].to_numpy(float) for group in groups]
    if any(not len(values) for values in arrays):
        statistic = pvalue = effect = np.nan
    else:
        test = kruskal(*arrays)
        statistic, pvalue = float(test.statistic), float(test.pvalue)
        effect = float(max(0.0, (statistic - len(groups) + 1) / (len(usable) - len(groups)))) if len(usable) > len(groups) else np.nan
    interval = _bootstrap_rows(
        usable,
        lambda sample: max(
            0.0,
            (
                kruskal(
                    *[
                        sample.loc[sample[label].eq(group), "value"].to_numpy(float)
                        for group in groups
                    ]
                ).statistic
                - len(groups)
                + 1
            )
            / (len(sample) - len(groups)),
        )
        if len(sample) > len(groups) and all(sample[label].eq(group).any() for group in groups)
        else np.nan,
        config=config,
        key=(*key, "epsilon_squared"),
    )
    return {
        "comparison": "kruskal_wallis", "group_a": " | ".join(groups), "group_b": None,
        "group_a_n": len(usable), "group_b_n": np.nan,
        "estimate_name": "kruskal_h", "estimate": statistic,
        "estimate_ci_low": np.nan, "estimate_ci_high": np.nan,
        "effect_name": "epsilon_squared", "effect": effect,
        "effect_ci_low": interval[0], "effect_ci_high": interval[1],
        "bootstrap_failure_rate": interval[2], "raw_p_value": pvalue,
        "eligible_patient_count": int(len(frame)),
        "descriptor_nonmissing_patient_count": int(frame[label].notna().sum()),
        "descriptor_missing_patient_count": int(frame[label].isna().sum()),
        "location_distribution": json.dumps(
            usable["location"].value_counts().sort_index().to_dict()
        ),
    }


def run_planned_contrasts(
    primary: pd.DataFrame, config: ConstructValidationConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run only the preregistered systolic construct families."""

    rows: list[dict[str, Any]] = []
    descriptive: list[dict[str, Any]] = []
    pair_specs = [
        ("timing", DURATION, "expert_timing_label", "Holosystolic", "non_holosystolic"),
        ("timing", MIDPOINT, "expert_timing_label", "Mid-systolic", "Early-systolic"),
        ("quality", ENTROPY, "expert_quality_label", "Harsh", "Blowing"),
        ("quality", BANDWIDTH, "expert_quality_label", "Harsh", "Blowing"),
    ]
    shape_groups = ("Plateau", "Diamond", "Decrescendo")
    shape_features = (PEAK_POSITION, RISE_SLOPE, DECAY_SLOPE, FULLNESS)
    ordered_specs = [
        ("pitch", feature, "expert_pitch_label", {"Low": 1, "Medium": 2, "High": 3})
        for feature in (DOMINANT_FREQUENCY, MEDIAN_FREQUENCY, HIGH_FREQUENCY_FRACTION)
    ] + [("grade", RMS_DB, "expert_grading_label", {"I/VI": 1, "II/VI": 2, "III/VI": 3})]

    for stratum in STRATA:
        for family, feature, label, left, right in pair_specs:
            frame = primary.loc[(primary["murmur_phase"].eq("systole")) & (primary["stratum"].eq(stratum)) & (primary["feature"].eq(feature))].copy()
            if right == "non_holosystolic":
                frame[label] = np.where(frame[label].eq("Holosystolic"), "Holosystolic", np.where(frame[label].notna(), "non_holosystolic", None))
            key = (family, stratum, feature, left, right)
            result = _pairwise_result(frame, label, left, right, config=config, key=key)
            rows.append({"construct_family": family, "stratum": stratum, "feature": feature, "descriptor": label, **result})
            descriptive.extend({"construct_family": family, "stratum": stratum, "feature": feature, "descriptor": label, **item} for item in _summaries(frame.loc[frame[label].isin([left, right])], label))

        for feature in shape_features:
            label = "expert_shape_label"
            frame = primary.loc[(primary["murmur_phase"].eq("systole")) & (primary["stratum"].eq(stratum)) & (primary["feature"].eq(feature))].copy()
            rows.append({
                "construct_family": "shape",
                "stratum": stratum,
                "feature": feature,
                "descriptor": label,
                **_omnibus_result(
                    frame,
                    label,
                    shape_groups,
                    config=config,
                    key=("shape", stratum, feature, "omnibus"),
                ),
            })
            for left, right in (("Diamond", "Plateau"), ("Decrescendo", "Plateau"), ("Diamond", "Decrescendo")):
                result = _pairwise_result(frame, label, left, right, config=config, key=("shape", stratum, feature, left, right))
                rows.append({"construct_family": "shape", "stratum": stratum, "feature": feature, "descriptor": label, **result})
            descriptive.extend({"construct_family": "shape", "stratum": stratum, "feature": feature, "descriptor": label, **item} for item in _summaries(frame.loc[frame[label].isin(shape_groups)], label))

        for family, feature, label, order in ordered_specs:
            frame = primary.loc[(primary["murmur_phase"].eq("systole")) & (primary["stratum"].eq(stratum)) & (primary["feature"].eq(feature))].copy()
            result = _ordered_result(frame, label, order, config=config, key=(family, stratum, feature))
            rows.append({"construct_family": family, "stratum": stratum, "feature": feature, "descriptor": label, **result})
            descriptive.extend({"construct_family": family, "stratum": stratum, "feature": feature, "descriptor": label, **item} for item in _summaries(frame.loc[frame[label].isin(order)], label))

    contrasts = pd.DataFrame(rows)
    contrasts["holm_p_value"] = contrasts.groupby(
        ["construct_family", "stratum"], sort=False, group_keys=False
    )["raw_p_value"].transform(_holm_adjust)
    contrasts["inference_status"] = np.select(
        [contrasts["estimate"].isna(), contrasts["group_a_n"].fillna(0).lt(8), contrasts["bootstrap_failure_rate"].fillna(0).gt(.05)],
        ["not_estimable", "descriptive_too_few_patients", "unstable_bootstrap_over_5pct_failures"],
        default="estimable",
    )
    return contrasts, pd.DataFrame(descriptive)


def build_diastolic_profiles(primary: pd.DataFrame) -> pd.DataFrame:
    diastolic = primary.loc[primary["murmur_phase"].eq("diastole")].copy()
    return diastolic.sort_values(["stratum", "patient_id", "feature"], kind="mergesort")


def run_construct_validation(
    *, source_run: Path, run_name: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    metadata_path: Path = METADATA_PATH,
    config: ConstructValidationConfig = ConstructValidationConfig(),
) -> Path:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one path component")
    destination = Path(output_root) / run_name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {destination}")
    table, source_manifest = _validate_source(Path(source_run), Path(metadata_path))
    metadata = pd.read_csv(metadata_path, dtype={"Patient ID": str})
    candidates, thresholds = prepare_candidate_strata(table)
    primary, flow = build_primary_patient_table(candidates, metadata)
    contrasts, descriptives = run_planned_contrasts(primary, config)
    diastolic = build_diastolic_profiles(primary)

    destination.mkdir(parents=True)
    artifacts = {
        "high_quality_thresholds.csv": thresholds,
        "eligibility_flow.csv": flow,
        "primary_patient_feature_long.csv": primary,
        "group_descriptives.csv": descriptives,
        "construct_contrasts.csv": contrasts,
        "diastolic_profiles.csv": diastolic,
    }
    hashes: dict[str, str] = {}
    for filename, frame in artifacts.items():
        path = destination / filename
        frame.to_csv(path, index=False)
        hashes[filename] = _sha256_file(path)
    manifest = {
        "tool_version": TOOL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "status": "complete",
        "run_name": run_name,
        "config": asdict(config),
        "source_run": str(Path(source_run).resolve()),
        "source_run_manifest_sha256": _sha256_file(Path(source_run) / "run_manifest.json"),
        "source_feature_table_sha256": _sha256_file(Path(source_run) / "task5_feature_table.csv"),
        "source_git_commit": source_manifest.get("git_commit"),
        "metadata_sha256": _sha256_file(Path(metadata_path)),
        "candidate_count": int(len(table)),
        "primary_patient_count": int(primary["patient_id"].nunique()) if len(primary) else 0,
        "contrast_count": int(len(contrasts)),
        "artifact_sha256": hashes,
        "high_quality_rule": "accepted-and-3-of-4-within-phase-medians-v1",
        "multiplicity": "Holm within construct family and analysis stratum",
        "outcome_used": False,
        "diastolic_inference": "descriptive_only",
        "scope_statement": "Task 5 construct validation only; no redundancy selection or classification.",
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    destination = run_construct_validation(
        source_run=args.source_run,
        run_name=args.run_name,
        output_root=args.output_root,
        metadata_path=args.metadata_path,
        config=ConstructValidationConfig(
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        ),
    )
    print(f"Task 5 Phase 2B artifacts: {destination}")


if __name__ == "__main__":
    main()
