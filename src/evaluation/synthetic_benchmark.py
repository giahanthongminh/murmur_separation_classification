"""Synthetic ground-truth benchmark for murmur-isolation methods."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
from itertools import product
import json

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from config import (
    DEFAULT_SEPARATION_CONFIG,
    REPORT_OUTPUT_DIR,
    SYNTHETIC_OUTPUT_DIR,
    SeparationConfig,
    ensure_output_directories,
)
from src.separation.core import separate_signal
from src.separation.metrics import (
    EPSILON,
    detect_activity_interval,
    energy,
    safe_correlation,
    sdr,
    si_sdr,
    spectral_distance,
)


MURMUR_SHAPES = (
    "early_systolic",
    "mid_systolic",
    "late_systolic",
    "holosystolic",
    "crescendo",
    "decrescendo",
    "crescendo_decrescendo",
    "plateau",
)
TARGET_PHASES = ("systole", "diastole")
PHASE_ORDER = ("s1", "systole", "s2", "diastole")
PHASE_FRACTIONS = (0.15, 0.35, 0.15, 0.35)
PHASE_FOCUS_GRID = (0.06, 0.08, 0.12, 0.16)
PHASE_RATIO_GRID = (0.10, 0.15, 0.20, 0.30)
SSA_WINDOW_MS_GRID = (20.0, 25.0, 35.0)


def _smooth_gate(time: np.ndarray, start: float, end: float, ramp: float = 0.025) -> np.ndarray:
    rise = np.clip((time - start) / ramp, 0, 1)
    fall = np.clip((end - time) / ramp, 0, 1)
    return np.minimum(rise, fall)


def _shape_for_target_phase(shape: str, target_phase: str) -> str:
    if target_phase not in TARGET_PHASES:
        raise ValueError(f"target_phase must be one of {TARGET_PHASES}")
    if shape.endswith("_systolic"):
        return shape.removesuffix("_systolic") + f"_{target_phase[:-1]}ic"
    if shape.endswith("_diastolic"):
        return shape.removesuffix("_diastolic") + f"_{target_phase[:-1]}ic"
    if shape == "holosystolic" or shape == "holodiastolic":
        return "holosystolic" if target_phase == "systole" else "holodiastolic"
    return shape


def _murmur_envelope(shape: str, time: np.ndarray) -> tuple[np.ndarray, float, float]:
    duration = float(time[-1] + (time[1] - time[0]))
    windows = {
        "early_systolic": (0.08 * duration, 0.42 * duration),
        "early_diastolic": (0.08 * duration, 0.42 * duration),
        "mid_systolic": (0.30 * duration, 0.70 * duration),
        "mid_diastolic": (0.30 * duration, 0.70 * duration),
        "late_systolic": (0.58 * duration, 0.92 * duration),
        "late_diastolic": (0.58 * duration, 0.92 * duration),
        "holosystolic": (0.06 * duration, 0.94 * duration),
        "holodiastolic": (0.06 * duration, 0.94 * duration),
        "crescendo": (0.08 * duration, 0.92 * duration),
        "decrescendo": (0.08 * duration, 0.92 * duration),
        "crescendo_decrescendo": (0.08 * duration, 0.92 * duration),
        "plateau": (0.12 * duration, 0.88 * duration),
    }
    start, end = windows[shape]
    gate = _smooth_gate(time, start, end)
    position = np.clip((time - start) / (end - start), 0, 1)
    if shape == "crescendo":
        gate *= position
    elif shape == "decrescendo":
        gate *= 1 - position
    elif shape == "crescendo_decrescendo":
        gate *= np.sin(np.pi * position)
    return gate, start, end


def _synthetic_phase_masks(samples: int) -> dict[str, np.ndarray]:
    boundaries = [0]
    cumulative = 0.0
    for fraction in PHASE_FRACTIONS[:-1]:
        cumulative += fraction
        boundaries.append(int(round(cumulative * samples)))
    boundaries.append(samples)
    masks: dict[str, np.ndarray] = {}
    for phase, start, end in zip(PHASE_ORDER, boundaries, boundaries[1:]):
        mask = np.zeros(samples, dtype=bool)
        mask[start:end] = True
        masks[phase] = mask
    return masks


def make_synthetic_mixture(
    shape: str,
    *,
    sample_rate: int,
    duration: float,
    murmur_to_heart_db: float,
    snr_db: float,
    seed: int,
    target_phase: str = "systole",
) -> dict[str, object]:
    if target_phase not in TARGET_PHASES:
        raise ValueError(f"target_phase must be one of {TARGET_PHASES}")
    shape = _shape_for_target_phase(shape, target_phase)
    rng = np.random.default_rng(seed)
    samples = int(round(sample_rate * duration))
    time = np.arange(samples) / sample_rate
    phase_masks = _synthetic_phase_masks(samples)
    s1_times = time[phase_masks["s1"]]
    target_times = time[phase_masks[target_phase]]
    s2_times = time[phase_masks["s2"]]
    s1_center = float(np.mean(s1_times))
    s2_center = float(np.mean(s2_times))
    # Full-cycle S1/S2 transients plus low-frequency cardiac structure.
    normal = (
        np.exp(-0.5 * ((time - s1_center) / 0.018) ** 2)
        * np.sin(2 * np.pi * 55 * time)
        + 0.75
        * np.exp(-0.5 * ((time - s2_center) / 0.015) ** 2)
        * np.sin(2 * np.pi * 72 * time)
        + 0.05 * np.sin(2 * np.pi * 25 * time)
    )
    local_target_time = target_times - target_times[0]
    if shape == "absent":
        local_envelope = np.zeros(len(local_target_time))
        onset = None
        offset = None
    else:
        local_envelope, onset, offset = _murmur_envelope(
            shape, local_target_time
        )
    envelope = np.zeros(samples)
    envelope[phase_masks[target_phase]] = local_envelope
    carrier = (
        np.sin(2 * np.pi * 180 * time + rng.uniform(0, 2 * np.pi))
        + 0.55 * np.sin(2 * np.pi * 260 * time + rng.uniform(0, 2 * np.pi))
    )
    colored = np.convolve(rng.normal(size=samples), np.ones(5) / 5, mode="same")
    murmur = envelope * (0.55 * carrier + 0.45 * colored)
    if shape != "absent":
        murmur *= np.sqrt(
            energy(normal)
            * 10 ** (murmur_to_heart_db / 10)
            / (energy(murmur) + EPSILON)
        )
    clean = normal + murmur
    noise = rng.normal(size=samples)
    noise *= np.sqrt(energy(clean) / (10 ** (snr_db / 10) * (energy(noise) + EPSILON)))
    return {
        "true_normal": normal,
        "true_murmur": murmur,
        "noise": noise,
        "mixture": clean + noise,
        "onset_seconds": onset,
        "offset_seconds": offset,
        "murmur_present": shape != "absent",
        "target_phase": target_phase,
        "shape": shape,
        "phase_masks": phase_masks,
    }


def _projection_leakage(source: np.ndarray, estimate: np.ndarray) -> float:
    scale = float(np.dot(estimate, source) / (np.dot(source, source) + EPSILON))
    return energy(scale * source) / (energy(estimate) + EPSILON)


def _evaluate(
    sample: dict[str, object],
    result,
    sample_rate: int,
) -> dict[str, float | None]:
    true_normal = np.asarray(sample["true_normal"])
    true_murmur = np.asarray(sample["true_murmur"])
    estimate_normal = result.normal_estimate
    estimate_murmur = result.murmur_candidate
    phase_masks = sample["phase_masks"]
    if not isinstance(phase_masks, dict):
        raise TypeError("synthetic phase_masks must be a dictionary")
    target_phase = str(sample.get("target_phase", "systole"))
    target_mask = np.asarray(phase_masks[target_phase], dtype=bool)
    target_estimate = estimate_murmur[target_mask]
    activity = detect_activity_interval(
        target_estimate,
        sample_rate,
        threshold_mad=1.5,
        minimum_duration_ms=15,
        merge_gap_ms=15,
    )
    onset = activity["onset_normalized"]
    offset = activity["offset_normalized"]
    duration = int(target_mask.sum()) / sample_rate
    murmur_present = bool(sample["murmur_present"])
    onset_error = (
        None
        if onset is None or not murmur_present
        else 1000 * abs(float(onset) * duration - float(sample["onset_seconds"]))
    )
    offset_error = (
        None
        if offset is None or not murmur_present
        else 1000 * abs(float(offset) * duration - float(sample["offset_seconds"]))
    )
    true_envelope = np.abs(hilbert(true_murmur[target_mask]))
    estimated_envelope = np.abs(hilbert(target_estimate))
    candidate_energy_ratio = energy(estimate_murmur) / (
        energy(np.asarray(sample["mixture"])) + EPSILON
    )
    return {
        "murmur_present": murmur_present,
        "target_phase": target_phase,
        "si_sdr_murmur": (
            si_sdr(true_murmur, estimate_murmur) if murmur_present else None
        ),
        "si_sdr_normal": si_sdr(true_normal, estimate_normal),
        "sdr_murmur": sdr(true_murmur, estimate_murmur) if murmur_present else None,
        "sdr_normal": sdr(true_normal, estimate_normal),
        "snr_improvement": (
            sdr(true_murmur, estimate_murmur)
            - sdr(true_murmur, np.asarray(sample["mixture"]))
            if murmur_present
            else None
        ),
        "murmur_correlation": (
            safe_correlation(true_murmur, estimate_murmur)
            if murmur_present
            else None
        ),
        "normal_correlation": safe_correlation(true_normal, estimate_normal),
        "murmur_spectral_distance": (
            spectral_distance(true_murmur, estimate_murmur)
            if murmur_present
            else None
        ),
        "normal_spectral_distance": spectral_distance(true_normal, estimate_normal),
        "envelope_error": (
            float(
                np.linalg.norm(true_envelope - estimated_envelope)
                / (np.linalg.norm(true_envelope) + EPSILON)
            )
            if murmur_present
            else None
        ),
        "onset_error_ms": onset_error,
        "offset_error_ms": offset_error,
        "normal_leakage_into_murmur": _projection_leakage(
            true_normal, estimate_murmur
        ),
        "murmur_leakage_into_normal": (
            _projection_leakage(true_murmur, estimate_normal)
            if murmur_present
            else None
        ),
        "estimated_murmur_energy_ratio": candidate_energy_ratio,
        "s1_leakage_ratio": result.metrics["s1_leakage_ratio"],
        "s2_leakage_ratio": result.metrics["s2_leakage_ratio"],
        "outside_murmur_energy_ratio": result.metrics[
            "outside_murmur_energy_ratio"
        ],
        "murmur_region_energy_retention": result.metrics[
            "murmur_region_energy_retention"
        ],
        "candidate_quality_status": result.metrics["candidate_quality_status"],
        "phase_selection_used_fallback": result.metrics[
            "phase_selection_used_fallback"
        ],
        "reconstruction_error": result.metrics["reconstruction_error"],
    }


def run_benchmark(
    *,
    shapes: tuple[str, ...] = MURMUR_SHAPES,
    murmur_to_heart_db: tuple[float, ...] = (-6.0, 0.0),
    snr_db: tuple[float, ...] = (10.0, 20.0),
    seeds: tuple[int, ...] = (42, 43),
    sample_rate: int = 1000,
    duration: float = 0.8,
    target_phase: str = "systole",
    base_config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
) -> pd.DataFrame:
    if target_phase not in TARGET_PHASES:
        raise ValueError(f"target_phase must be one of {TARGET_PHASES}")
    ensure_output_directories()
    config = replace(
        base_config,
        sample_rate=sample_rate,
        ssa_window_length=max(
            2,
            int(
                round(
                    base_config.ssa_window_length
                    / base_config.sample_rate
                    * sample_rate
                )
            ),
        ),
    )
    methods: list[tuple[str, str, bool, str, str]] = [
        ("cssa_zcr", "zcr", False, "db4", "soft"),
        ("cssa_kurtosis", "kurtosis", False, "db4", "soft"),
    ]
    if importlib.util.find_spec("pywt") is not None:
        methods.extend(
            [
                ("cssa_zcr_dwt_db4_soft", "zcr", True, "db4", "soft"),
                ("cssa_zcr_dwt_db4_hard", "zcr", True, "db4", "hard"),
                ("cssa_zcr_dwt_sym4_soft", "zcr", True, "sym4", "soft"),
                ("cssa_kurtosis_dwt_db4_soft", "kurtosis", True, "db4", "soft"),
            ]
        )
    rows: list[dict[str, object]] = []
    for shape, ratio, noise_level, seed in product(
        shapes, murmur_to_heart_db, snr_db, seeds
    ):
        phase_shape = _shape_for_target_phase(shape, target_phase)
        sample = make_synthetic_mixture(
            phase_shape,
            sample_rate=sample_rate,
            duration=duration,
            murmur_to_heart_db=ratio,
            snr_db=noise_level,
            seed=seed,
            target_phase=target_phase,
        )
        sample_id = (
            f"{target_phase}_{phase_shape}_mhr_{ratio:+g}_snr_{noise_level:g}_seed_{seed}"
            .replace("+", "plus")
            .replace("-", "minus")
            .replace(".", "p")
        )
        sample_directory = SYNTHETIC_OUTPUT_DIR / sample_id
        sample_directory.mkdir(parents=True, exist_ok=True)
        for name in ("true_normal", "true_murmur", "noise", "mixture"):
            np.save(
                sample_directory / f"{name}.npy",
                np.asarray(sample[name], dtype=np.float32),
            )
        phase_masks = sample["phase_masks"]
        if not isinstance(phase_masks, dict):
            raise TypeError("synthetic phase_masks must be a dictionary")
        for phase, mask in phase_masks.items():
            np.save(sample_directory / f"{phase}_mask.npy", np.asarray(mask))
        for method_name, method, use_dwt, wavelet, threshold_method in methods:
            method_config = replace(
                config,
                use_dwt=use_dwt,
                dwt_wavelet=wavelet,
                dwt_threshold_method=threshold_method,
                random_seed=seed,
            )
            result = separate_signal(
                np.asarray(sample["mixture"]),
                config=method_config,
                method=method,
                phase_masks=phase_masks,
                target_phase=target_phase,
            )
            metrics = _evaluate(sample, result, sample_rate)
            row = {
                "shape": phase_shape,
                "target_phase": target_phase,
                "murmur_to_heart_db": ratio,
                "snr_db": noise_level,
                "seed": seed,
                "method": method_name,
                "config_hash": method_config.config_hash,
                **metrics,
            }
            rows.append(row)
            method_directory = sample_directory / method_name
            method_directory.mkdir(parents=True, exist_ok=True)
            for name, values in (
                ("estimated_normal", result.normal_estimate),
                ("estimated_murmur", result.murmur_candidate),
                ("estimated_noise", result.noise_candidate),
            ):
                np.save(method_directory / f"{name}.npy", values.astype(np.float32))
            (method_directory / "metrics.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT_OUTPUT_DIR / "synthetic_benchmark.csv", index=False)
    return frame


def _select_tuning_configuration(summary: pd.DataFrame) -> dict[str, object]:
    """Select a near-best SI-SDR configuration using leakage tie-breakers."""

    if summary.empty or summary["mean_si_sdr_murmur"].isna().all():
        raise ValueError("phase tuning produced no finite present-murmur SI-SDR")
    best_si_sdr = float(summary["mean_si_sdr_murmur"].max())
    near_best = summary[
        summary["mean_si_sdr_murmur"] >= best_si_sdr - 0.5
    ].copy()
    near_best["negative_retention"] = -near_best[
        "mean_murmur_region_energy_retention"
    ]
    selected = near_best.sort_values(
        [
            "present_fallback_rate",
            "absent_candidate_energy_ratio",
            "mean_normal_leakage_into_murmur",
            "mean_outside_murmur_energy_ratio",
            "negative_retention",
        ],
        ascending=True,
    ).iloc[0]
    return {
        "minimum_systole_focus": float(selected["minimum_systole_focus"]),
        "minimum_systole_to_s1_s2_ratio": float(
            selected["minimum_systole_to_s1_s2_ratio"]
        ),
        "ssa_window_ms": float(selected["ssa_window_ms"]),
        "mean_si_sdr_murmur": float(selected["mean_si_sdr_murmur"]),
        "present_fallback_rate": float(selected["present_fallback_rate"]),
        "absent_candidate_energy_ratio": float(
            selected["absent_candidate_energy_ratio"]
        ),
        "selection_rule": (
            "within 0.5 dB of best mean present-murmur SI-SDR, then minimize "
            "present fallback rate, absent candidate energy, normal leakage, "
            "and outside-systole energy before maximizing retention"
        ),
    }


def run_phase_threshold_tuning(
    *,
    shapes: tuple[str, ...] = MURMUR_SHAPES,
    murmur_to_heart_db: tuple[float, ...] = (-6.0, 0.0),
    snr_db: tuple[float, ...] = (10.0, 20.0),
    seeds: tuple[int, ...] = (42, 43),
    focus_thresholds: tuple[float, ...] = PHASE_FOCUS_GRID,
    ratio_thresholds: tuple[float, ...] = PHASE_RATIO_GRID,
    ssa_window_ms: tuple[float, ...] = SSA_WINDOW_MS_GRID,
    sample_rate: int = 1000,
    duration: float = 0.8,
    target_phase: str = "systole",
    base_config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Tune phase selection on full-cycle mixtures with known source stems."""

    if target_phase not in TARGET_PHASES:
        raise ValueError(f"target_phase must be one of {TARGET_PHASES}")
    ensure_output_directories()
    samples: list[tuple[str, float, float, int, dict[str, object]]] = []
    for shape, ratio, noise_level, seed in product(
        shapes, murmur_to_heart_db, snr_db, seeds
    ):
        phase_shape = _shape_for_target_phase(shape, target_phase)
        samples.append(
            (
                phase_shape,
                ratio,
                noise_level,
                seed,
                make_synthetic_mixture(
                    phase_shape,
                    sample_rate=sample_rate,
                    duration=duration,
                    murmur_to_heart_db=ratio,
                    snr_db=noise_level,
                    seed=seed,
                    target_phase=target_phase,
                ),
            )
        )
    for noise_level, seed in product(snr_db, seeds):
        samples.append(
            (
                "absent",
                0.0,
                noise_level,
                seed,
                make_synthetic_mixture(
                    "absent",
                    sample_rate=sample_rate,
                    duration=duration,
                    murmur_to_heart_db=0.0,
                    snr_db=noise_level,
                    seed=seed,
                    target_phase=target_phase,
                ),
            )
        )

    rows: list[dict[str, object]] = []
    for focus, ratio_threshold, window_ms in product(
        focus_thresholds, ratio_thresholds, ssa_window_ms
    ):
        window_samples = max(2, int(round(window_ms * sample_rate / 1000)))
        for shape, ratio, noise_level, seed, sample in samples:
            method_config = replace(
                base_config,
                sample_rate=sample_rate,
                ssa_window_length=window_samples,
                minimum_systole_focus=focus,
                minimum_systole_to_s1_s2_ratio=ratio_threshold,
                phase_aware_component_selection=True,
                phase_selection_fallback=True,
                use_dwt=False,
                random_seed=seed,
            )
            phase_masks = sample["phase_masks"]
            if not isinstance(phase_masks, dict):
                raise TypeError("synthetic phase_masks must be a dictionary")
            result = separate_signal(
                np.asarray(sample["mixture"]),
                config=method_config,
                method="zcr",
                phase_masks=phase_masks,
                target_phase=target_phase,
            )
            rows.append(
                {
                    "shape": shape,
                    "target_phase": target_phase,
                    "murmur_to_heart_db": ratio,
                    "snr_db": noise_level,
                    "seed": seed,
                    "minimum_systole_focus": focus,
                    "minimum_systole_to_s1_s2_ratio": ratio_threshold,
                    "ssa_window_ms": window_ms,
                    "config_hash": method_config.config_hash,
                    **_evaluate(sample, result, sample_rate),
                }
            )
    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    group_columns = [
        "minimum_systole_focus",
        "minimum_systole_to_s1_s2_ratio",
        "ssa_window_ms",
    ]
    for keys, group in frame.groupby(group_columns, sort=True):
        present = group[group["murmur_present"]]
        absent = group[~group["murmur_present"]]
        summary_rows.append(
            {
                **dict(zip(group_columns, keys)),
                "present_case_count": len(present),
                "absent_case_count": len(absent),
                "mean_si_sdr_murmur": present["si_sdr_murmur"].mean(),
                "median_si_sdr_murmur": present["si_sdr_murmur"].median(),
                "present_fallback_rate": present[
                    "phase_selection_used_fallback"
                ].mean(),
                "absent_candidate_energy_ratio": absent[
                    "estimated_murmur_energy_ratio"
                ].mean(),
                "mean_normal_leakage_into_murmur": present[
                    "normal_leakage_into_murmur"
                ].mean(),
                "mean_outside_murmur_energy_ratio": present[
                    "outside_murmur_energy_ratio"
                ].mean(),
                "mean_murmur_region_energy_retention": present[
                    "murmur_region_energy_retention"
                ].mean(),
                "mean_onset_error_ms": present["onset_error_ms"].mean(),
                "mean_offset_error_ms": present["offset_error_ms"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    selected = _select_tuning_configuration(summary)
    selected["target_phase"] = target_phase
    selected["ssa_window_length_at_production_rate"] = int(
        round(float(selected["ssa_window_ms"]) * base_config.sample_rate / 1000)
    )
    suffix = "" if target_phase == "systole" else f"_{target_phase}"
    frame.to_csv(
        REPORT_OUTPUT_DIR / f"phase_threshold_tuning{suffix}.csv", index=False
    )
    summary.to_csv(
        REPORT_OUTPUT_DIR / f"phase_threshold_tuning_summary{suffix}.csv",
        index=False,
    )
    (
        REPORT_OUTPUT_DIR / f"phase_threshold_tuning_selected{suffix}.json"
    ).write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    return frame, summary, selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run two shapes and one condition")
    parser.add_argument(
        "--tune-phase-thresholds",
        action="store_true",
        help="Tune full-cycle phase thresholds using present and absent ground truth",
    )
    parser.add_argument(
        "--target-phase",
        choices=TARGET_PHASES,
        default="systole",
        help="Place and score the synthetic murmur in systole or diastole",
    )
    args = parser.parse_args(argv)
    options = {}
    if args.quick:
        options = {
            "shapes": ("early_systolic", "holosystolic"),
            "murmur_to_heart_db": (-3.0,),
            "snr_db": (20.0,),
            "seeds": (42,),
            "base_config": replace(
                DEFAULT_SEPARATION_CONFIG,
                kurtosis_population_size=10,
                kurtosis_generations=8,
            ),
            "target_phase": args.target_phase,
        }
    else:
        options["target_phase"] = args.target_phase
    if args.tune_phase_thresholds:
        tuning_options = {}
        if args.quick:
            tuning_options = {
                "shapes": ("early_systolic", "holosystolic"),
                "murmur_to_heart_db": (-3.0,),
                "snr_db": (20.0,),
                "seeds": (42, 43),
                "target_phase": args.target_phase,
            }
        else:
            tuning_options["target_phase"] = args.target_phase
        frame, summary, selected = run_phase_threshold_tuning(**tuning_options)
        print(f"Tuning rows: {len(frame)}")
        print(f"Configurations: {len(summary)}")
        print(f"Selected: {json.dumps(selected, sort_keys=True)}")
        suffix = "" if args.target_phase == "systole" else f"_{args.target_phase}"
        print(
            f"Report: {REPORT_OUTPUT_DIR / f'phase_threshold_tuning{suffix}.csv'}"
        )
        return 0
    frame = run_benchmark(**options)
    print(f"Benchmark rows: {len(frame)}")
    print(f"Report: {REPORT_OUTPUT_DIR / 'synthetic_benchmark.csv'}")
    print(frame.groupby("method")["si_sdr_murmur"].mean().sort_values(ascending=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
