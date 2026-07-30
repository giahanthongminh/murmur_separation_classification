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
from src.separation.audit import (
    CardiacCycleContext,
    _absolute_timing_metrics,
    _all_recordings,
    _should_save_package,
    build_cardiac_cycle_context,
    classify_audit_outcome,
    summarize_audit_quality,
    summarize_murmur_observations,
)
from src.separation.metrics import (
    detect_activity_interval,
    murmur_observation_features,
    real_proxy_metrics,
)
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
    assert result.metrics["candidate_quality_status"] in {"accepted", "fallback"}
    assert result.metrics["onset_normalized"] is not None
    assert result.metrics["offset_normalized"] is not None
    assert all(
        "systole_focus_score" in row
        and "systole_to_s1_s2_ratio" in row
        and "base_assignment" in row
        for row in result.component_features
    )


def test_quality_summary_keeps_fallback_separate() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_quality_status": "accepted",
                "murmur_label": "Present",
                "s1_leakage_ratio": 0.1,
                "s2_leakage_ratio": 0.2,
                "outside_murmur_energy_ratio": 0.3,
                "murmur_region_energy_retention": 0.4,
            },
            {
                "candidate_quality_status": "fallback",
                "murmur_label": "Present",
                "s1_leakage_ratio": 0.9,
                "s2_leakage_ratio": 0.8,
                "outside_murmur_energy_ratio": 0.7,
                "murmur_region_energy_retention": 0.01,
            },
        ]
    )
    quality = summarize_audit_quality(frame).set_index("candidate_quality_status")
    assert quality.loc["accepted", "segment_count"] == 1
    assert quality.loc["accepted", "s1_leakage_ratio"] == pytest.approx(0.1)
    assert quality.loc["fallback", "s1_leakage_ratio"] == pytest.approx(0.9)
    assert quality.loc["accepted", "audit_outcome"] == "present_candidate_accepted"
    assert quality.loc["fallback", "audit_outcome"] == "present_candidate_missed"


@pytest.mark.parametrize(
    ("label", "status", "expected"),
    [
        ("Present", "accepted", "present_candidate_accepted"),
        ("Present", "fallback", "present_candidate_missed"),
        ("Absent", "accepted", "absent_candidate_flagged"),
        ("Absent", "fallback", "absent_negative_control_clear"),
        ("Unknown", "accepted", "unscored_unknown_label"),
    ],
)
def test_audit_outcome_is_label_aware(
    label: str, status: str, expected: str
) -> None:
    assert classify_audit_outcome(label, status) == expected


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
    assert timing["onset_sample"] is not None
    assert timing["offset_sample"] is not None
    assert timing["onset_seconds"] == pytest.approx(
        timing["onset_sample"] / sample_rate
    )
    assert timing["offset_seconds"] == pytest.approx(
        timing["offset_sample"] / sample_rate
    )


def test_murmur_observation_features_capture_tone_and_amplitude() -> None:
    sample_rate = 4000
    time = np.arange(int(0.2 * sample_rate)) / sample_rate
    signal = 0.4 * np.sin(2 * np.pi * 180 * time)
    features = murmur_observation_features(signal, sample_rate)
    assert features["amplitude_peak_abs"] == pytest.approx(0.4, rel=0.02)
    assert features["amplitude_rms"] == pytest.approx(0.4 / np.sqrt(2), rel=0.03)
    assert features["psd_peak_frequency_hz"] == pytest.approx(180, abs=10)
    assert features["psd_100_200_hz_ratio"] > 0.9
    assert features["time_frequency_peak_hz"] == pytest.approx(180, abs=20)
    assert features["time_frequency_frame_count"] > 1
    assert features["time_frequency_frequency_resolution_hz"] > 0
    assert np.isfinite(features["time_frequency_entropy"])
    assert np.isfinite(features["time_frequency_spectral_flux"])


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


def test_absolute_timing_is_exported_in_recording_seconds() -> None:
    sample_rate = 1000
    length = 500
    bounds = {
        "s1": (0, 100),
        "systole": (100, 200),
        "s2": (200, 300),
        "diastole": (300, 500),
    }
    masks = {}
    for phase, (start, end) in bounds.items():
        mask = np.zeros(length, dtype=bool)
        mask[start:end] = True
        masks[phase] = mask
    cycle = CardiacCycleContext(
        signal=np.zeros(length),
        phase_masks=masks,
        phase_bounds=bounds,
        context_start_sample=2000,
        context_end_sample=2500,
    )
    timing = _absolute_timing_metrics(
        {
            "onset_sample": 20,
            "offset_sample": 80,
            "candidate_quality_status": "accepted",
            "activity_detection_method": "adaptive_envelope",
        },
        cycle,
        sample_rate,
    )
    assert timing["murmur_onset_systole_seconds"] == pytest.approx(0.02)
    assert timing["murmur_offset_systole_seconds"] == pytest.approx(0.08)
    assert timing["murmur_onset_cycle_seconds"] == pytest.approx(0.12)
    assert timing["murmur_offset_cycle_seconds"] == pytest.approx(0.18)
    assert timing["murmur_onset_recording_seconds"] == pytest.approx(2.12)
    assert timing["murmur_offset_recording_seconds"] == pytest.approx(2.18)
    assert timing["timing_quality_status"] == "accepted_adaptive"


def test_all_recordings_uses_exact_pairs_and_location_labels(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        [
            {
                "Patient ID": "111",
                "Murmur": "Present",
                "Murmur locations": "AV+PV",
                "Systolic murmur timing": "Early-systolic",
            },
            {
                "Patient ID": "222",
                "Murmur": "Absent",
                "Murmur locations": np.nan,
                "Systolic murmur timing": np.nan,
            },
        ]
    )
    for recording_id in ("111_AV", "111_MV_1", "222_PV"):
        (tmp_path / f"{recording_id}.wav").touch()
        (tmp_path / f"{recording_id}.tsv").touch()
    (tmp_path / "111_PV.wav").touch()
    (tmp_path / "999_AV.wav").touch()
    (tmp_path / "999_AV.tsv").touch()

    rows = _all_recordings(metadata, 0, audio_dir=tmp_path)
    labels = {row["recording_id"]: row["murmur_label"] for row in rows}
    assert labels == {
        "111_AV": "Present",
        "111_MV_1": "Unknown",
        "222_PV": "Absent",
    }
    assert next(row for row in rows if row["recording_id"] == "111_MV_1")[
        "location"
    ] == "MV"


@pytest.mark.parametrize(
    ("profile", "status", "expected"),
    [
        ("full", "fallback", True),
        ("accepted", "accepted", True),
        ("accepted", "fallback", False),
        ("summary", "accepted", False),
    ],
)
def test_output_profile_controls_heavy_packages(
    profile: str, status: str, expected: bool
) -> None:
    assert _should_save_package(profile, status) is expected


def test_observation_summary_only_uses_accepted_present_candidates() -> None:
    frame = pd.DataFrame(
        [
            {
                "murmur_label": "Present",
                "candidate_quality_status": "accepted",
                "timing_label": "Early-systolic",
                "activity_detection_method": "adaptive_envelope",
                "amplitude_rms": 0.2,
                "psd_peak_frequency_hz": 180.0,
            },
            {
                "murmur_label": "Present",
                "candidate_quality_status": "fallback",
                "timing_label": "Early-systolic",
                "activity_detection_method": "adaptive_envelope",
                "amplitude_rms": 9.0,
                "psd_peak_frequency_hz": 900.0,
            },
            {
                "murmur_label": "Absent",
                "candidate_quality_status": "accepted",
                "timing_label": "nan",
                "activity_detection_method": "adaptive_envelope",
                "amplitude_rms": 8.0,
                "psd_peak_frequency_hz": 800.0,
            },
        ]
    )
    observation = summarize_murmur_observations(frame)
    assert len(observation) == 1
    assert observation.loc[0, "segment_count"] == 1
    assert observation.loc[0, "amplitude_rms_mean"] == pytest.approx(0.2)
    assert observation.loc[0, "psd_peak_frequency_hz_mean"] == pytest.approx(180)
