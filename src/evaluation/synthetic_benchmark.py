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


def _smooth_gate(time: np.ndarray, start: float, end: float, ramp: float = 0.025) -> np.ndarray:
    rise = np.clip((time - start) / ramp, 0, 1)
    fall = np.clip((end - time) / ramp, 0, 1)
    return np.minimum(rise, fall)


def _murmur_envelope(shape: str, time: np.ndarray) -> tuple[np.ndarray, float, float]:
    duration = float(time[-1] + (time[1] - time[0]))
    windows = {
        "early_systolic": (0.08 * duration, 0.42 * duration),
        "mid_systolic": (0.30 * duration, 0.70 * duration),
        "late_systolic": (0.58 * duration, 0.92 * duration),
        "holosystolic": (0.06 * duration, 0.94 * duration),
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


def make_synthetic_mixture(
    shape: str,
    *,
    sample_rate: int,
    duration: float,
    murmur_to_heart_db: float,
    snr_db: float,
    seed: int,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    samples = int(round(sample_rate * duration))
    time = np.arange(samples) / sample_rate
    # S1/S2-like boundary transients plus low-frequency cardiac structure.
    normal = (
        np.exp(-0.5 * ((time - 0.035) / 0.018) ** 2)
        * np.sin(2 * np.pi * 55 * time)
        + 0.75
        * np.exp(-0.5 * ((time - (duration - 0.045)) / 0.015) ** 2)
        * np.sin(2 * np.pi * 72 * time)
        + 0.05 * np.sin(2 * np.pi * 25 * time)
    )
    envelope, onset, offset = _murmur_envelope(shape, time)
    carrier = (
        np.sin(2 * np.pi * 180 * time + rng.uniform(0, 2 * np.pi))
        + 0.55 * np.sin(2 * np.pi * 260 * time + rng.uniform(0, 2 * np.pi))
    )
    colored = np.convolve(rng.normal(size=samples), np.ones(5) / 5, mode="same")
    murmur = envelope * (0.55 * carrier + 0.45 * colored)
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
    }


def _projection_leakage(source: np.ndarray, estimate: np.ndarray) -> float:
    scale = float(np.dot(estimate, source) / (np.dot(source, source) + EPSILON))
    return energy(scale * source) / (energy(estimate) + EPSILON)


def _evaluate(
    sample: dict[str, np.ndarray | float],
    result,
    sample_rate: int,
) -> dict[str, float | None]:
    true_normal = np.asarray(sample["true_normal"])
    true_murmur = np.asarray(sample["true_murmur"])
    estimate_normal = result.normal_estimate
    estimate_murmur = result.murmur_candidate
    activity = detect_activity_interval(
        estimate_murmur,
        sample_rate,
        threshold_mad=1.5,
        minimum_duration_ms=15,
        merge_gap_ms=15,
    )
    onset = activity["onset_normalized"]
    offset = activity["offset_normalized"]
    duration = len(true_murmur) / sample_rate
    onset_error = (
        None
        if onset is None
        else 1000 * abs(float(onset) * duration - float(sample["onset_seconds"]))
    )
    offset_error = (
        None
        if offset is None
        else 1000 * abs(float(offset) * duration - float(sample["offset_seconds"]))
    )
    true_envelope = np.abs(hilbert(true_murmur))
    estimated_envelope = np.abs(hilbert(estimate_murmur))
    return {
        "si_sdr_murmur": si_sdr(true_murmur, estimate_murmur),
        "si_sdr_normal": si_sdr(true_normal, estimate_normal),
        "sdr_murmur": sdr(true_murmur, estimate_murmur),
        "sdr_normal": sdr(true_normal, estimate_normal),
        "snr_improvement": sdr(true_murmur, estimate_murmur)
        - sdr(true_murmur, np.asarray(sample["mixture"])),
        "murmur_correlation": safe_correlation(true_murmur, estimate_murmur),
        "normal_correlation": safe_correlation(true_normal, estimate_normal),
        "murmur_spectral_distance": spectral_distance(true_murmur, estimate_murmur),
        "normal_spectral_distance": spectral_distance(true_normal, estimate_normal),
        "envelope_error": float(
            np.linalg.norm(true_envelope - estimated_envelope)
            / (np.linalg.norm(true_envelope) + EPSILON)
        ),
        "onset_error_ms": onset_error,
        "offset_error_ms": offset_error,
        "normal_leakage_into_murmur": _projection_leakage(
            true_normal, estimate_murmur
        ),
        "murmur_leakage_into_normal": _projection_leakage(
            true_murmur, estimate_normal
        ),
        "reconstruction_error": result.metrics["reconstruction_error"],
    }


def run_benchmark(
    *,
    shapes: tuple[str, ...] = MURMUR_SHAPES,
    murmur_to_heart_db: tuple[float, ...] = (-6.0, 0.0),
    snr_db: tuple[float, ...] = (10.0, 20.0),
    seeds: tuple[int, ...] = (42, 43),
    sample_rate: int = 1000,
    duration: float = 0.6,
    base_config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
) -> pd.DataFrame:
    ensure_output_directories()
    config = replace(
        base_config,
        sample_rate=sample_rate,
        ssa_window_length=min(base_config.ssa_window_length, 64),
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
        sample = make_synthetic_mixture(
            shape,
            sample_rate=sample_rate,
            duration=duration,
            murmur_to_heart_db=ratio,
            snr_db=noise_level,
            seed=seed,
        )
        sample_id = (
            f"{shape}_mhr_{ratio:+g}_snr_{noise_level:g}_seed_{seed}"
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
        for method_name, method, use_dwt, wavelet, threshold_method in methods:
            method_config = replace(
                config,
                use_dwt=use_dwt,
                dwt_wavelet=wavelet,
                dwt_threshold_method=threshold_method,
                random_seed=seed,
            )
            result = separate_signal(
                np.asarray(sample["mixture"]), config=method_config, method=method
            )
            metrics = _evaluate(sample, result, sample_rate)
            row = {
                "shape": shape,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run two shapes and one condition")
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
        }
    frame = run_benchmark(**options)
    print(f"Benchmark rows: {len(frame)}")
    print(f"Report: {REPORT_OUTPUT_DIR / 'synthetic_benchmark.csv'}")
    print(frame.groupby("method")["si_sdr_murmur"].mean().sort_values(ascending=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
