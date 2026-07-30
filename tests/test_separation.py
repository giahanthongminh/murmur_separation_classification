from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import SeparationConfig, validate_input_output_isolation
from src.separation.core import (
    _phase_aware_murmur_indexes,
    _phase_energy_features,
    separate_signal,
)
from src.separation.audit import build_cardiac_cycle_context
from src.separation.metrics import detect_activity_interval, real_proxy_metrics
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


def test_phase_selection_rejects_s1_s2_dominated_components() -> None:
    rows = [
        {
            "systole_focus_score": 0.70,
            "systole_to_s1_s2_ratio": 2.0,
            "relative_energy": 0.10,
        },
        {
            "systole_focus_score": 0.08,
            "systole_to_s1_s2_ratio": 0.05,
            "relative_energy": 0.20,
        },
        {
            "systole_focus_score": 0.20,
            "systole_to_s1_s2_ratio": 0.10,
            "relative_energy": 0.15,
        },
    ]
    config = SeparationConfig(phase_selection_fallback=False)
    selected, rejected, used_fallback = _phase_aware_murmur_indexes(
        rows, [0, 1, 2], config
    )
    assert selected == [0]
    assert rejected == [1, 2]
    assert not used_fallback


def test_phase_energy_features_use_density_not_phase_duration() -> None:
    lengths = {"s1": 10, "systole": 20, "s2": 10, "diastole": 40}
    amplitudes = {"s1": 2.0, "systole": 4.0, "s2": 1.0, "diastole": 0.5}
    component_parts = []
    masks: dict[str, np.ndarray] = {}
    offset = 0
    total_length = sum(lengths.values())
    for phase in ("s1", "systole", "s2", "diastole"):
        component_parts.append(np.full(lengths[phase], amplitudes[phase]))
        mask = np.zeros(total_length, dtype=bool)
        mask[offset : offset + lengths[phase]] = True
        masks[phase] = mask
        offset += lengths[phase]
    features = _phase_energy_features(np.concatenate(component_parts), masks)
    expected_density_total = 2.0**2 + 4.0**2 + 1.0**2 + 0.5**2
    assert features["systole_focus_score"] == pytest.approx(
        4.0**2 / expected_density_total
    )
    assert features["systole_to_s1_s2_ratio"] == pytest.approx(
        4.0**2 / (2.0**2 + 1.0**2)
    )


def test_phase_selection_fallback_retains_best_systolic_component() -> None:
    rows = [
        {
            "systole_focus_score": 0.02,
            "systole_to_s1_s2_ratio": 0.01,
            "relative_energy": 0.30,
        },
        {
            "systole_focus_score": 0.08,
            "systole_to_s1_s2_ratio": 0.10,
            "relative_energy": 0.05,
        },
    ]
    selected, rejected, used_fallback = _phase_aware_murmur_indexes(
        rows, [0, 1], SeparationConfig()
    )
    assert selected == [1]
    assert rejected == [0]
    assert used_fallback


def test_phase_aware_outputs_preserve_full_cycle_reconstruction() -> None:
    signal = _signal()
    phase_masks: dict[str, np.ndarray] = {}
    phase_length = len(signal) // 4
    for index, phase in enumerate(("s1", "systole", "s2", "diastole")):
        mask = np.zeros(len(signal), dtype=bool)
        mask[index * phase_length : (index + 1) * phase_length] = True
        phase_masks[phase] = mask
    result = separate_signal(
        signal,
        config=SeparationConfig(sample_rate=400, ssa_window_length=40),
        method="zcr",
        phase_masks=phase_masks,
    )
    reconstructed = (
        result.normal_estimate + result.murmur_candidate + result.noise_candidate
    )
    np.testing.assert_allclose(reconstructed, signal, atol=1e-10)
    assert result.metrics["phase_aware_selection_applied"]
    assert result.metrics["onset_normalized"] is not None
    assert result.metrics["offset_normalized"] is not None
    assert all(
        "systole_focus_score" in row
        and "systole_to_s1_s2_ratio" in row
        and "base_assignment" in row
        for row in result.component_features
    )


@pytest.mark.parametrize("duration_ms", [70, 100, 200])
def test_valid_short_candidates_have_finite_timing(duration_ms: int) -> None:
    """Valid 70--200 ms candidates must not export all-NaN timing fields."""

    sample_rate = 4000
    samples = int(sample_rate * duration_ms / 1000)
    time = np.arange(samples) / sample_rate
    envelope = np.sin(np.pi * np.arange(samples) / max(1, samples - 1)) ** 2
    candidate = envelope * np.sin(2 * np.pi * 180 * time)
    timing = detect_activity_interval(candidate, sample_rate)
    timing_values = [
        timing["onset_normalized"],
        timing["offset_normalized"],
        timing["duration_ratio"],
        timing["temporal_energy_centroid"],
        timing["peak_position"],
    ]
    assert all(value is not None and np.isfinite(value) for value in timing_values)
    assert 0 <= timing["onset_normalized"] < timing["offset_normalized"] <= 1


def test_full_cycle_proxy_metrics_compute_phase_leakage() -> None:
    samples_per_phase = 100
    length = 4 * samples_per_phase
    original = np.ones(length)
    candidate = np.concatenate(
        [
            np.full(samples_per_phase, 0.5),
            np.full(samples_per_phase, 0.25),
            np.full(samples_per_phase, 0.2),
            np.full(samples_per_phase, 0.1),
        ]
    )
    masks = {}
    for index, phase in enumerate(("s1", "systole", "s2", "diastole")):
        mask = np.zeros(length, dtype=bool)
        mask[index * samples_per_phase : (index + 1) * samples_per_phase] = True
        masks[phase] = mask
    metrics = real_proxy_metrics(
        original,
        original - candidate,
        candidate,
        np.zeros(length),
        4000,
        phase_masks=masks,
    )
    assert metrics["s1_leakage_ratio"] == pytest.approx(0.25)
    assert metrics["s2_leakage_ratio"] == pytest.approx(0.04)
    assert metrics["murmur_region_energy_retention"] == pytest.approx(0.0625)
    expected_outside = (0.5**2 + 0.2**2 + 0.1**2) / (
        0.5**2 + 0.25**2 + 0.2**2 + 0.1**2
    )
    assert metrics["outside_murmur_energy_ratio"] == pytest.approx(expected_outside)
    assert metrics["onset_normalized"] is not None
    assert metrics["offset_normalized"] is not None


def test_cycle_context_contains_all_four_phases() -> None:
    sample_rate = 1000
    signal = np.arange(600, dtype=float)
    annotations = pd.DataFrame(
        [
            (0.0, 0.1, 0),
            (0.1, 0.2, 1),
            (0.2, 0.3, 2),
            (0.3, 0.4, 3),
            (0.4, 0.6, 4),
        ],
        columns=["start", "end", "state"],
    )
    context = build_cardiac_cycle_context(signal, annotations, 2, sample_rate)
    assert context.context_start_sample == 100
    assert context.context_end_sample == 600
    assert len(context.signal) == 500
    assert context.phase_bounds == {
        "s1": (0, 100),
        "systole": (100, 200),
        "s2": (200, 300),
        "diastole": (300, 500),
    }
    assert all(mask.any() for mask in context.phase_masks.values())


def test_cycle_context_normalizes_submillisecond_boundary_overlap() -> None:
    sample_rate = 4000
    signal = np.arange(2400, dtype=float)
    annotations = pd.DataFrame(
        [
            (0.0, 0.1, 1),
            (0.1, 0.2, 2),
            (0.2, 0.4, 3),
            (0.3996, 0.6, 4),
        ],
        columns=["start", "end", "state"],
    )
    context = build_cardiac_cycle_context(signal, annotations, 1, sample_rate)
    assert context.phase_bounds["s2"][1] == context.phase_bounds["diastole"][0]
    coverage = np.sum(np.stack(list(context.phase_masks.values())), axis=0)
    assert np.all(coverage == 1)


def test_cycle_context_rejects_boundary_overlap_beyond_tolerance() -> None:
    sample_rate = 4000
    signal = np.arange(2400, dtype=float)
    annotations = pd.DataFrame(
        [
            (0.0, 0.1, 1),
            (0.1, 0.2, 2),
            (0.2, 0.4, 3),
            (0.398, 0.6, 4),
        ],
        columns=["start", "end", "state"],
    )
    with pytest.raises(ValueError, match="exceeding .* tolerance"):
        build_cardiac_cycle_context(signal, annotations, 1, sample_rate)
