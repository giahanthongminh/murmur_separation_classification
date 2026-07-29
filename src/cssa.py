"""Compatibility API for the auditable CSSA implementation.

New experiments should use :mod:`src.separation.core` and call the output a
``murmur_candidate`` until benchmark evidence supports a stronger claim.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import DEFAULT_SEPARATION_CONFIG, SeparationConfig
from src.separation.core import (
    compare_separation_methods,
    kurtosis_score,
    separate_signal,
    zero_crossing_rate,
)
from src.separation.metrics import safe_correlation


def _legacy_config(
    L: int,
    *,
    zcr_threshold: float | None = None,
    pop_size: int | None = None,
    n_generations: int | None = None,
    mutation_rate: float | None = None,
    random_state: int | None = None,
) -> SeparationConfig:
    updates: dict[str, object] = {"ssa_window_length": L}
    if zcr_threshold is not None:
        updates.update(
            {"zcr_threshold_strategy": "fixed", "zcr_threshold": zcr_threshold}
        )
    if pop_size is not None:
        updates["kurtosis_population_size"] = pop_size
    if n_generations is not None:
        updates["kurtosis_generations"] = n_generations
    if mutation_rate is not None:
        updates["kurtosis_mutation_rate"] = mutation_rate
    if random_state is not None:
        updates["random_seed"] = random_state
    return replace(DEFAULT_SEPARATION_CONFIG, **updates)


def cssa_zcr(signal: np.ndarray, L: int, zcr_threshold: float | None = None):
    config = _legacy_config(L, zcr_threshold=zcr_threshold)
    result = separate_signal(signal, config=config, method="zcr")
    selected = result.assignments["normal"]
    zcr_values = [row["zcr"] for row in result.component_features]
    return (
        result.normal_estimate,
        result.murmur_candidate + result.noise_candidate,
        selected,
        zcr_values,
    )


def cssa_kurtosis(
    signal: np.ndarray,
    L: int,
    pop_size: int = 30,
    n_generations: int = 40,
    mutation_rate: float = 0.05,
    random_state: int = 42,
):
    config = _legacy_config(
        L,
        pop_size=pop_size,
        n_generations=n_generations,
        mutation_rate=mutation_rate,
        random_state=random_state,
    )
    result = separate_signal(signal, config=config, method="kurtosis")
    selected = np.asarray(result.assignments["normal"], dtype=int)
    kurtosis_values = [row["kurtosis"] for row in result.component_features]
    return (
        result.normal_estimate,
        result.murmur_candidate + result.noise_candidate,
        selected,
        kurtosis_values,
    )


def correlation_score(left: np.ndarray, right: np.ndarray) -> float:
    """Backward-compatible diagnostic; no longer the sole selector."""

    return safe_correlation(left, right)


def compare_cssa_methods(
    signal: np.ndarray,
    L: int,
    zcr_threshold: float | None = None,
    **_: object,
) -> dict[str, object]:
    """Backward-compatible comparison backed by multi-metric selection."""

    config = _legacy_config(L, zcr_threshold=zcr_threshold)
    best, candidates = compare_separation_methods(signal, config=config)
    zcr = candidates["zcr"]
    kurtosis = candidates["kurtosis"]
    return {
        "best_method": best.selected_method,
        "best_normal": best.normal_estimate,
        "best_murmur": best.murmur_candidate + best.noise_candidate,
        "best_murmur_candidate": best.murmur_candidate,
        "best_noise_candidate": best.noise_candidate,
        "metrics": best.metrics,
        "corr_zcr": zcr.metrics["normal_residual_correlation"],
        "corr_kurt": kurtosis.metrics["normal_residual_correlation"],
        "score_zcr": zcr.metrics["selection_score"],
        "score_kurt": kurtosis.metrics["selection_score"],
        "selected_zcr": zcr.assignments["normal"],
        "selected_kurt": kurtosis.assignments["normal"],
    }
