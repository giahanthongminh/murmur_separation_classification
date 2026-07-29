"""Configurable CSSA separation with explicit three-way component assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import DEFAULT_SEPARATION_CONFIG, SeparationConfig
from src.ssa import SSAResult, ssa_decompose_audited
from .metrics import EPSILON, energy, real_proxy_metrics, safe_correlation, spectral_features


@dataclass(frozen=True)
class SeparationResult:
    original: np.ndarray
    normal_estimate: np.ndarray
    murmur_candidate: np.ndarray
    noise_candidate: np.ndarray
    component_features: list[dict[str, Any]]
    assignments: dict[str, list[int]]
    metrics: dict[str, Any]
    selected_method: str
    config_hash: str


def zero_crossing_rate(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=float)
    if len(values) < 2:
        return 0.0
    signs = np.signbit(values)
    return float(np.mean(signs[1:] != signs[:-1]))


def kurtosis_score(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=float)
    centered = values - values.mean()
    deviation = float(centered.std())
    if deviation <= EPSILON:
        return 0.0
    return float(np.mean((centered / deviation) ** 4) - 3.0)


def _component_table(
    ssa: SSAResult, sample_rate: int
) -> list[dict[str, Any]]:
    total_energy = sum(energy(component) for component in ssa.components) + EPSILON
    rows: list[dict[str, Any]] = []
    nyquist = sample_rate / 2.0
    for index, component in enumerate(ssa.components):
        spectral = spectral_features(component, sample_rate)
        spectrum = np.abs(np.fft.rfft(component)) ** 2
        frequencies = np.fft.rfftfreq(len(component), 1.0 / sample_rate)
        low = float(spectrum[frequencies < 100].sum())
        mid = float(spectrum[(frequencies >= 100) & (frequencies < 600)].sum())
        high = float(spectrum[frequencies >= 600].sum())
        spectral_total = low + mid + high + EPSILON
        rms = float(np.sqrt(np.mean(component**2)))
        rows.append(
            {
                "component_index": index,
                "active_by_energy": index < ssa.active_component_count,
                "singular_value": float(ssa.singular_values[index]),
                "relative_energy": energy(component) / total_energy,
                "zcr": zero_crossing_rate(component),
                "kurtosis": kurtosis_score(component),
                "spectral_centroid": spectral["spectral_centroid"],
                "spectral_bandwidth": spectral["spectral_bandwidth"],
                "dominant_frequency": spectral["dominant_frequency"],
                "low_band_ratio": low / spectral_total,
                "mid_band_ratio": mid / spectral_total,
                "high_band_ratio": high / spectral_total,
                "impulsiveness": float(np.max(np.abs(component)) / (rms + EPSILON)),
                "nyquist_ratio": spectral["spectral_centroid"] / max(nyquist, 1.0),
            }
        )
    return rows


def _noise_indexes(
    rows: list[dict[str, Any]], config: SeparationConfig
) -> list[int]:
    return [
        int(row["component_index"])
        for row in rows
        if not row["active_by_energy"]
        or row["relative_energy"] < config.noise_energy_floor
        or row["high_band_ratio"] >= config.high_frequency_noise_ratio
    ]


def _zcr_normal_indexes(
    rows: list[dict[str, Any]], eligible: list[int], config: SeparationConfig
) -> list[int]:
    if not eligible:
        return []
    zcr_values = np.array([rows[index]["zcr"] for index in eligible], dtype=float)
    threshold = (
        config.zcr_threshold
        if config.zcr_threshold_strategy == "fixed"
        else float(np.percentile(zcr_values, config.zcr_percentile))
    )
    return [index for index in eligible if rows[index]["zcr"] <= threshold]


def _kurtosis_normal_indexes(
    components: np.ndarray,
    rows: list[dict[str, Any]],
    eligible: list[int],
    config: SeparationConfig,
) -> list[int]:
    if not eligible:
        return []
    rng = np.random.default_rng(config.random_seed)
    dimension = len(eligible)

    def fitness(mask: np.ndarray) -> float:
        if not mask.any():
            return -np.inf
        indexes = [eligible[i] for i in np.flatnonzero(mask)]
        combined = components[indexes].sum(axis=0)
        retained = sum(float(rows[index]["relative_energy"]) for index in indexes)
        impulse = float(np.mean([rows[index]["impulsiveness"] for index in indexes]))
        return (
            kurtosis_score(combined)
            + config.kurtosis_energy_weight * np.log1p(100 * retained)
            - config.kurtosis_impulse_penalty * max(0.0, impulse - 6.0)
        )

    population = rng.integers(
        0, 2, size=(config.kurtosis_population_size, dimension), dtype=np.int8
    )
    population[0, 0] = 1
    best = population[0].copy()
    best_score = fitness(best)
    for _ in range(config.kurtosis_generations):
        scores = np.array([fitness(mask) for mask in population])
        generation_best = int(np.argmax(scores))
        if scores[generation_best] > best_score:
            best = population[generation_best].copy()
            best_score = float(scores[generation_best])
        next_population = [best.copy()]
        while len(next_population) < config.kurtosis_population_size:
            candidates = rng.integers(0, len(population), size=4)
            first = population[
                candidates[0] if scores[candidates[0]] > scores[candidates[1]] else candidates[1]
            ]
            second = population[
                candidates[2] if scores[candidates[2]] > scores[candidates[3]] else candidates[3]
            ]
            crossover = rng.random(dimension) < 0.5
            child = np.where(crossover, first, second).astype(np.int8)
            mutation = rng.random(dimension) < config.kurtosis_mutation_rate
            child[mutation] = 1 - child[mutation]
            if not child.any():
                child[int(rng.integers(0, dimension))] = 1
            next_population.append(child)
        population = np.asarray(next_population, dtype=np.int8)
    return [eligible[index] for index in np.flatnonzero(best)]


def _sum_components(components: np.ndarray, indexes: list[int], length: int) -> np.ndarray:
    return (
        components[indexes].sum(axis=0)
        if indexes
        else np.zeros(length, dtype=float)
    )


def separate_signal(
    signal: np.ndarray,
    *,
    config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
    method: str = "zcr",
) -> SeparationResult:
    """Separate one cardiac-phase segment into normal, murmur, and noise candidates."""

    values = np.asarray(signal, dtype=float)
    window = min(config.ssa_window_length, max(2, len(values) // 4))
    ssa = ssa_decompose_audited(
        values,
        window,
        energy_threshold=config.explained_energy_threshold,
        maximum_components=config.maximum_ssa_components,
        reconstruction_tolerance=config.reconstruction_tolerance,
    )
    rows = _component_table(ssa, config.sample_rate)
    noise_indexes = _noise_indexes(rows, config)
    eligible = [
        index
        for index in range(ssa.active_component_count)
        if index not in set(noise_indexes)
    ]
    if method == "zcr":
        normal_indexes = _zcr_normal_indexes(rows, eligible, config)
    elif method == "kurtosis":
        normal_indexes = _kurtosis_normal_indexes(
            ssa.components, rows, eligible, config
        )
    else:
        raise ValueError("method must be 'zcr' or 'kurtosis'")
    murmur_indexes = [index for index in eligible if index not in set(normal_indexes)]
    normal = _sum_components(ssa.components, normal_indexes, len(values))
    murmur = _sum_components(ssa.components, murmur_indexes, len(values))
    noise = _sum_components(ssa.components, noise_indexes, len(values))

    if config.use_dwt:
        from src.dwt_refine import dwt_refine

        refined = dwt_refine(
            normal,
            wavelet=config.dwt_wavelet,
            level=config.dwt_level,
            threshold_method=config.dwt_threshold_method,
        )
        murmur = murmur + (normal - refined)
        normal = refined

    assignments = {
        "normal": normal_indexes,
        "murmur_candidate": murmur_indexes,
        "noise_artifact": noise_indexes,
    }
    assignment_by_index = {
        index: label for label, indexes in assignments.items() for index in indexes
    }
    for row in rows:
        row["assignment"] = assignment_by_index[int(row["component_index"])]

    metrics = real_proxy_metrics(
        values,
        normal,
        murmur,
        noise,
        config.sample_rate,
        threshold_mad=config.onset_threshold_mad,
        minimum_duration_ms=config.minimum_interval_duration_ms,
        merge_gap_ms=config.gap_merging_duration_ms,
    )
    metrics.update(
        {
            "ssa_reconstruction_error": ssa.reconstruction_error,
            "ssa_active_component_count": ssa.active_component_count,
            "ssa_total_component_count": len(ssa.components),
            "ssa_explained_energy": ssa.explained_energy,
            "selected_method": method + ("+dwt" if config.use_dwt else ""),
            "config_hash": config.config_hash,
        }
    )
    return SeparationResult(
        original=values,
        normal_estimate=normal,
        murmur_candidate=murmur,
        noise_candidate=noise,
        component_features=rows,
        assignments=assignments,
        metrics=metrics,
        selected_method=str(metrics["selected_method"]),
        config_hash=config.config_hash,
    )


def _selection_score(result: SeparationResult) -> float:
    metrics = result.metrics
    normal_ratio = energy(result.normal_estimate) / (energy(result.original) + EPSILON)
    murmur_ratio = energy(result.murmur_candidate) / (energy(result.original) + EPSILON)
    normal_penalty = max(0.0, 0.10 - normal_ratio) + max(0.0, normal_ratio - 0.95)
    murmur_penalty = max(0.0, 0.01 - murmur_ratio) + max(0.0, murmur_ratio - 0.80)
    return float(
        metrics["reconstruction_error"] * 1e6
        + 0.15 * abs(metrics["normal_residual_correlation"])
        + normal_penalty
        + murmur_penalty
        + 0.05 * metrics["noise_energy_ratio"]
    )


def compare_separation_methods(
    signal: np.ndarray,
    *,
    config: SeparationConfig = DEFAULT_SEPARATION_CONFIG,
) -> tuple[SeparationResult, dict[str, SeparationResult]]:
    """Choose between configured methods with a multi-metric plausibility score."""

    candidates = {
        method: separate_signal(signal, config=config, method=method)
        for method in ("zcr", "kurtosis")
    }
    for result in candidates.values():
        result.metrics["selection_score"] = _selection_score(result)
    best = min(candidates.values(), key=lambda item: item.metrics["selection_score"])
    return best, candidates
