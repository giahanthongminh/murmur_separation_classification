from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from config import SeparationConfig, validate_input_output_isolation
from src.separation.core import separate_signal
from src.ssa import select_component_count, ssa_decompose_audited


def _signal() -> np.ndarray:
    time = np.arange(320) / 400
    return np.sin(2 * np.pi * 12 * time) + 0.2 * np.sin(2 * np.pi * 95 * time)


def test_ssa_reconstructs_all_components() -> None:
    signal = _signal()
    result = ssa_decompose_audited(signal, 40, energy_threshold=0.95)
    assert result.reconstruction_error < 1e-10
    np.testing.assert_allclose(result.components.sum(axis=0), signal, atol=1e-10)


@pytest.mark.parametrize("threshold", [0.95, 0.975, 0.99, 0.995])
def test_energy_component_selection(threshold: float) -> None:
    singular_values = np.array([4.0, 2.0, 1.0, 0.2])
    count = select_component_count(singular_values, threshold)
    cumulative = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    assert cumulative[count - 1] >= threshold


def test_input_output_overlap_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    with pytest.raises(ValueError):
        validate_input_output_isolation(dataset, dataset / "outputs")
    validate_input_output_isolation(dataset, tmp_path / "project" / "outputs")


def test_fixed_seed_is_deterministic() -> None:
    config = SeparationConfig(
        sample_rate=400,
        ssa_window_length=40,
        kurtosis_population_size=8,
        kurtosis_generations=5,
        random_seed=7,
    )
    first = separate_signal(_signal(), config=config, method="kurtosis")
    second = separate_signal(_signal(), config=replace(config), method="kurtosis")
    assert first.assignments == second.assignments
    np.testing.assert_array_equal(first.murmur_candidate, second.murmur_candidate)


def test_three_way_outputs_reconstruct_input() -> None:
    result = separate_signal(
        _signal(),
        config=SeparationConfig(sample_rate=400, ssa_window_length=40),
        method="zcr",
    )
    reconstructed = (
        result.normal_estimate + result.murmur_candidate + result.noise_candidate
    )
    np.testing.assert_allclose(reconstructed, result.original, atol=1e-10)
    assigned = sum(result.assignments.values(), [])
    assert sorted(assigned) == list(range(len(result.component_features)))
