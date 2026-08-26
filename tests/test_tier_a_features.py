"""Correctness and numerical-behavior tests for Task 5 Tier A features."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert, savgol_filter, welch

from src.separation.audit import OBSERVATION_METRICS
from src.separation.metrics import real_proxy_metrics
from src.separation.tier_a_features import (
    FEATURE_SCHEMA_VERSION,
    LEGACY_TO_TIER_A_COLUMN_MAP,
    TIER_A_25_1000_HZ_SENSITIVITY_CONFIG,
    TIER_A_EXPORT_COLUMNS,
    TIER_A_FEATURE_COLUMNS,
    TierAFeatureConfig,
    extract_tier_a_features,
)


FS = 4000


def _tone(
    frequency: float = 180.0,
    *,
    duration: float = 0.4,
    amplitude: float = 0.6,
    envelope: np.ndarray | None = None,
) -> np.ndarray:
    time = np.arange(int(round(FS * duration))) / FS
    values = amplitude * np.sin(2 * np.pi * frequency * time)
    return values if envelope is None else values * envelope


def _references(amplitude: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(800) / FS
    return (
        amplitude * np.sin(2 * np.pi * 90 * time),
        0.75 * amplitude * np.sin(2 * np.pi * 110 * time),
    )


def _extract(
    candidate: np.ndarray,
    *,
    onset: int = 100,
    phase_padding: int = 200,
    references: tuple[np.ndarray, np.ndarray] | None = None,
    config: TierAFeatureConfig = TierAFeatureConfig(),
    sample_rate: int = FS,
) -> dict[str, object]:
    s1, s2 = references if references is not None else _references()
    return extract_tier_a_features(
        candidate,
        sample_rate,
        phase_start_sample=0,
        phase_end_sample=len(candidate) + phase_padding,
        onset_sample=onset,
        offset_sample=onset + len(candidate),
        s1_reference=s1,
        s2_reference=s2,
        config=config,
    )


def _assert_valid(result: dict[str, object], name: str) -> float:
    assert result[f"{name}_valid"] is True
    assert result[f"{name}_invalid_reason"] is None
    return float(result[name])


def test_registry_and_export_schema_are_complete() -> None:
    assert len(TIER_A_FEATURE_COLUMNS) == 14
    assert len(set(TIER_A_FEATURE_COLUMNS)) == 14
    assert set(TIER_A_EXPORT_COLUMNS).issubset(OBSERVATION_METRICS)
    assert set(LEGACY_TO_TIER_A_COLUMN_MAP.values()).issubset(TIER_A_FEATURE_COLUMNS)
    result = _extract(_tone())
    assert result["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    for feature in TIER_A_FEATURE_COLUMNS:
        assert {feature, f"{feature}_valid", f"{feature}_invalid_reason"}.issubset(
            result
        )
    assert set(TIER_A_EXPORT_COLUMNS).issubset(result)


def test_exclusive_index_timing_matches_independent_arithmetic() -> None:
    candidate = _tone(duration=0.1)
    result = extract_tier_a_features(
        candidate,
        FS,
        phase_start_sample=1000,
        phase_end_sample=2000,
        onset_sample=1200,
        offset_sample=1600,
        s1_reference=_references()[0],
        s2_reference=_references()[1],
    )
    assert _assert_valid(result, TIER_A_FEATURE_COLUMNS[0]) == pytest.approx(0.4)
    assert _assert_valid(result, TIER_A_FEATURE_COLUMNS[1]) == pytest.approx(0.4)
    assert result["tier_a_candidate_sample_count"] == 400
    assert result["tier_a_candidate_start_sample_in_phase"] == 200
    assert result["tier_a_candidate_end_sample_in_phase_exclusive"] == 600


@pytest.mark.parametrize(
    ("phase_start", "phase_end", "onset", "offset", "reason"),
    [
        (10, 10, 10, 20, "invalid_phase"),
        (0, 100, 50, 50, "invalid_interval"),
        (0, 100, -1, 50, "invalid_interval"),
        (0, 100, 10, 101, "invalid_interval"),
    ],
)
def test_invalid_phase_and_intervals_are_nan_with_reasons(
    phase_start: int,
    phase_end: int,
    onset: int,
    offset: int,
    reason: str,
) -> None:
    candidate = np.ones(max(0, offset - onset))
    result = extract_tier_a_features(
        candidate,
        FS,
        phase_start_sample=phase_start,
        phase_end_sample=phase_end,
        onset_sample=onset,
        offset_sample=offset,
        s1_reference=_references()[0],
        s2_reference=_references()[1],
    )
    for feature in TIER_A_FEATURE_COLUMNS[:2]:
        assert np.isnan(result[feature])
        assert result[f"{feature}_valid"] is False
        assert result[f"{feature}_invalid_reason"] == reason


def test_empty_silent_nonfinite_and_short_candidates_are_distinguished() -> None:
    empty = extract_tier_a_features(
        np.asarray([]),
        FS,
        phase_start_sample=0,
        phase_end_sample=10,
        onset_sample=2,
        offset_sample=3,
        s1_reference=_references()[0],
        s2_reference=_references()[1],
    )
    assert empty[f"{TIER_A_FEATURE_COLUMNS[2]}_invalid_reason"] == "empty_candidate"

    silent = _extract(np.zeros(20))
    assert silent[f"{TIER_A_FEATURE_COLUMNS[2]}_invalid_reason"] == "silent_candidate"
    assert np.isnan(silent[TIER_A_FEATURE_COLUMNS[2]])

    nonfinite = _tone(duration=0.05)
    nonfinite[4] = np.nan
    invalid = _extract(nonfinite)
    assert invalid[f"{TIER_A_FEATURE_COLUMNS[2]}_invalid_reason"] == "nonfinite_input"

    short = _extract(np.asarray([1.0, -1.0, 1.0, -1.0]))
    assert short[f"{TIER_A_FEATURE_COLUMNS[3]}_invalid_reason"] == (
        "insufficient_envelope_support"
    )
    assert short[f"{TIER_A_FEATURE_COLUMNS[12]}_invalid_reason"] == (
        "insufficient_ridge_frames"
    )


@pytest.mark.parametrize(
    "references",
    [
        (np.zeros(100), np.zeros(100)),
        (np.ones(100), np.ones(100)),
        (np.asarray([]), np.ones(100)),
    ],
)
def test_silent_or_dc_s1_s2_references_are_invalid(
    references: tuple[np.ndarray, np.ndarray],
) -> None:
    result = _extract(_tone(), references=references)
    name = TIER_A_FEATURE_COLUMNS[2]
    assert np.isnan(result[name])
    assert result[f"{name}_invalid_reason"] == "silent_reference"


def test_reference_normalized_rms_matches_independent_formula() -> None:
    candidate = _tone(amplitude=0.4)
    s1, s2 = _references(amplitude=0.9)
    result = _extract(candidate, references=(s1, s2))

    def centered_rms(values: np.ndarray) -> float:
        centered = values - np.mean(values)
        return float(np.sqrt(np.mean(centered**2)))

    reference = np.sqrt((centered_rms(s1) ** 2 + centered_rms(s2) ** 2) / 2)
    expected = 20 * np.log10(centered_rms(candidate) / reference)
    assert _assert_valid(result, TIER_A_FEATURE_COLUMNS[2]) == pytest.approx(
        expected, abs=1e-12
    )


def test_global_and_candidate_only_gain_transformations() -> None:
    candidate = _tone()
    references = _references()
    baseline = _extract(candidate, references=references)
    global_gain = _extract(
        3.0 * candidate, references=(3.0 * references[0], 3.0 * references[1])
    )
    candidate_gain = _extract(2.0 * candidate, references=references)
    amplitude = TIER_A_FEATURE_COLUMNS[2]
    assert global_gain[amplitude] == pytest.approx(baseline[amplitude], abs=1e-12)
    assert candidate_gain[amplitude] - baseline[amplitude] == pytest.approx(
        20 * np.log10(2.0), abs=1e-12
    )
    for feature in TIER_A_FEATURE_COLUMNS[3:]:
        if baseline[f"{feature}_valid"]:
            assert global_gain[feature] == pytest.approx(baseline[feature], abs=1e-9)
            assert candidate_gain[feature] == pytest.approx(baseline[feature], abs=1e-9)


def test_polarity_and_dc_transformations_are_invariant() -> None:
    position = np.linspace(0.0, 1.0, int(FS * 0.45))
    candidate = _tone(220.0, duration=0.45, envelope=0.3 + 0.7 * position)
    baseline = _extract(candidate)
    for transformed in (-candidate, candidate + 17.25):
        changed = _extract(transformed)
        for feature in TIER_A_FEATURE_COLUMNS[2:]:
            assert changed[f"{feature}_valid"] == baseline[f"{feature}_valid"]
            if baseline[f"{feature}_valid"]:
                assert changed[feature] == pytest.approx(baseline[feature], abs=1e-8)


def test_candidate_translation_changes_only_phase_midpoint() -> None:
    candidate = _tone()
    first = _extract(candidate, onset=100, phase_padding=400)
    second = _extract(candidate, onset=220, phase_padding=400)
    duration_name, midpoint_name = TIER_A_FEATURE_COLUMNS[:2]
    assert first[duration_name] == second[duration_name]
    assert second[midpoint_name] - first[midpoint_name] == pytest.approx(
        120 / (len(candidate) + 400)
    )
    for feature in TIER_A_FEATURE_COLUMNS[2:]:
        if first[f"{feature}_valid"]:
            assert second[feature] == pytest.approx(first[feature], abs=1e-12)


def test_envelope_reference_calculation_and_shape_ordering() -> None:
    n = 1600
    position = np.linspace(0.0, 1.0, n)
    plateau = 0.1 + 0.9 * np.minimum(1.0, np.minimum(position / 0.15, (1 - position) / 0.15))
    triangle = 0.1 + 0.9 * np.maximum(0.0, 1.0 - np.abs(position - 0.3) / 0.3)
    diamond = 0.1 + 0.9 * (1.0 - np.abs(2 * position - 1.0))
    crescendo = 0.1 + 0.9 * position
    decrescendo = crescendo[::-1]
    results = {
        "plateau": _extract(_tone(envelope=plateau)),
        "triangle": _extract(_tone(envelope=triangle)),
        "diamond": _extract(_tone(envelope=diamond)),
        "crescendo": _extract(_tone(envelope=crescendo)),
        "decrescendo": _extract(_tone(envelope=decrescendo)),
    }
    fullness = TIER_A_FEATURE_COLUMNS[6]
    peak_position = TIER_A_FEATURE_COLUMNS[3]
    rise = TIER_A_FEATURE_COLUMNS[4]
    decay = TIER_A_FEATURE_COLUMNS[5]
    assert results["plateau"][fullness] > results["diamond"][fullness]
    assert results["triangle"][peak_position] == pytest.approx(0.3, abs=0.03)
    assert results["crescendo"][peak_position] > 0.8
    assert results["decrescendo"][peak_position] < 0.2
    assert results["diamond"][rise] > 0
    assert results["diamond"][decay] < 0

    centered = _tone(envelope=diamond) - np.mean(_tone(envelope=diamond))
    raw = np.abs(hilbert(centered))
    requested = int(round(FS * 0.012)) | 1
    reference_envelope = np.maximum(
        savgol_filter(raw, requested, 2, mode="interp"), 0.0
    )
    expected_fullness = np.mean(reference_envelope / np.max(reference_envelope))
    assert results["diamond"][fullness] == pytest.approx(expected_fullness, abs=1e-12)

    peak_index = int(np.argmax(reference_envelope))
    normalized = reference_envelope / np.max(reference_envelope)
    normalized_time = np.linspace(0.0, 1.0, n)
    pairwise_slopes = [
        (normalized[right] - normalized[left])
        / (normalized_time[right] - normalized_time[left])
        for left in range(peak_index)
        for right in range(left + 1, peak_index + 1)
    ]
    assert results["diamond"][rise] == pytest.approx(np.median(pairwise_slopes))


def test_valid_features_have_protocol_ranges_and_units() -> None:
    n = 2048
    position = np.linspace(0.0, 1.0, n)
    candidate = _tone(260.0, duration=n / FS, envelope=0.2 + 0.8 * position)
    result = _extract(candidate)
    for feature in TIER_A_FEATURE_COLUMNS:
        if result[f"{feature}_valid"]:
            assert np.isfinite(result[feature])
    assert 0 < result[TIER_A_FEATURE_COLUMNS[0]] <= 1
    for feature in (
        TIER_A_FEATURE_COLUMNS[1],
        TIER_A_FEATURE_COLUMNS[3],
        TIER_A_FEATURE_COLUMNS[6],
        TIER_A_FEATURE_COLUMNS[10],
        TIER_A_FEATURE_COLUMNS[11],
    ):
        assert 0 <= result[feature] <= 1
    for feature in TIER_A_FEATURE_COLUMNS[7:9]:
        assert 25 <= result[feature] <= 800
    assert 0 <= result[TIER_A_FEATURE_COLUMNS[9]] <= 775
    assert result[TIER_A_FEATURE_COLUMNS[13]] >= 0


def test_time_reversal_has_expected_envelope_psd_and_ridge_behavior() -> None:
    # 2048 gives a complete, reversal-symmetric set of 512-point Welch frames.
    n = 2048
    time = np.arange(n) / FS
    candidate = (0.15 + 0.85 * time / time[-1]) * np.sin(
        2 * np.pi * (100 * time + 0.5 * 800 * time**2)
    )
    forward = _extract(candidate)
    reverse = _extract(candidate[::-1])
    for feature in (
        TIER_A_FEATURE_COLUMNS[0],
        TIER_A_FEATURE_COLUMNS[1],
        TIER_A_FEATURE_COLUMNS[2],
        TIER_A_FEATURE_COLUMNS[6],
        *TIER_A_FEATURE_COLUMNS[7:10],
        TIER_A_FEATURE_COLUMNS[13],
    ):
        assert reverse[feature] == pytest.approx(forward[feature], abs=1e-8)
    # Periodic Hann windows can leave sub-bin numerical differences on reversal.
    for feature in TIER_A_FEATURE_COLUMNS[10:12]:
        assert reverse[feature] == pytest.approx(forward[feature], abs=5e-4)
    assert reverse[TIER_A_FEATURE_COLUMNS[3]] == pytest.approx(
        1.0 - forward[TIER_A_FEATURE_COLUMNS[3]], abs=2 / n
    )
    assert reverse[TIER_A_FEATURE_COLUMNS[12]] == pytest.approx(
        -forward[TIER_A_FEATURE_COLUMNS[12]], abs=1e-8
    )
    assert reverse[TIER_A_FEATURE_COLUMNS[4]] == pytest.approx(
        -forward[TIER_A_FEATURE_COLUMNS[5]], abs=1e-8
    )
    assert reverse[TIER_A_FEATURE_COLUMNS[5]] == pytest.approx(
        -forward[TIER_A_FEATURE_COLUMNS[4]], abs=1e-8
    )


def test_welch_psd_features_match_independent_discrete_reference() -> None:
    time = np.arange(2048) / FS
    candidate = 0.8 * np.sin(2 * np.pi * 180 * time)
    candidate += 0.4 * np.sin(2 * np.pi * 420 * time)
    result = _extract(candidate)
    frequencies, power = welch(
        candidate - np.mean(candidate),
        fs=FS,
        window="hann",
        nperseg=512,
        noverlap=256,
        nfft=512,
        detrend="constant",
        scaling="density",
    )
    band = (frequencies >= 25) & (frequencies <= 800)
    frequencies = frequencies[band]
    power = power[band]
    probabilities = power / power.sum()
    cumulative = np.cumsum(probabilities)

    def q(value: float) -> float:
        return float(frequencies[np.searchsorted(cumulative, value)])

    assert result[TIER_A_FEATURE_COLUMNS[7]] == pytest.approx(
        frequencies[np.argmax(power)]
    )
    assert result[TIER_A_FEATURE_COLUMNS[8]] == pytest.approx(q(0.5))
    assert result[TIER_A_FEATURE_COLUMNS[9]] == pytest.approx(q(0.975) - q(0.025))
    assert result[TIER_A_FEATURE_COLUMNS[10]] == pytest.approx(
        power[frequencies > 200].sum() / power.sum()
    )
    positive = probabilities > 0
    expected_entropy = -np.sum(probabilities[positive] * np.log(probabilities[positive]))
    expected_entropy /= np.log(len(probabilities))
    assert result[TIER_A_FEATURE_COLUMNS[11]] == pytest.approx(expected_entropy)
    assert result["tier_a_psd_frequency_resolution_hz"] == pytest.approx(FS / 512)
    assert result["tier_a_psd_segment_count"] == 7


def test_sinusoid_and_two_tone_psd_units_and_ranges() -> None:
    sinusoid = _extract(_tone(250.0))
    assert sinusoid[TIER_A_FEATURE_COLUMNS[7]] == pytest.approx(250, abs=FS / 512)
    time = np.arange(2048) / FS
    two_tone = _extract(
        0.8 * np.sin(2 * np.pi * 180 * time)
        + 0.4 * np.sin(2 * np.pi * 420 * time)
    )
    assert two_tone[TIER_A_FEATURE_COLUMNS[7]] == pytest.approx(180, abs=FS / 512)
    assert two_tone[TIER_A_FEATURE_COLUMNS[9]] > 200
    assert 0 <= two_tone[TIER_A_FEATURE_COLUMNS[10]] <= 1
    assert 0 <= two_tone[TIER_A_FEATURE_COLUMNS[11]] <= 1


def test_frequency_shift_away_from_edges_moves_location_features() -> None:
    low = _extract(_tone(180.0))
    high = _extract(_tone(280.0))
    resolution = float(low["tier_a_psd_frequency_resolution_hz"])
    for feature in TIER_A_FEATURE_COLUMNS[7:9]:
        assert high[feature] - low[feature] == pytest.approx(100, abs=2 * resolution)


def test_linear_chirps_have_opposite_robust_normalized_time_slopes() -> None:
    duration = 0.5
    time = np.arange(int(FS * duration)) / FS

    def chirp_signal(start: float, end: float) -> np.ndarray:
        rate = (end - start) / duration
        phase = 2 * np.pi * (start * time + 0.5 * rate * time**2)
        return 0.7 * np.sin(phase)

    increasing = _extract(chirp_signal(100, 500))
    decreasing = _extract(chirp_signal(500, 100))
    slope = TIER_A_FEATURE_COLUMNS[12]
    variability = TIER_A_FEATURE_COLUMNS[13]
    assert increasing[slope] == pytest.approx(400, abs=70)
    assert decreasing[slope] == pytest.approx(-400, abs=70)
    assert increasing[slope] == pytest.approx(-decreasing[slope], abs=35)
    assert increasing[variability] <= 2 * increasing["tier_a_ridge_frequency_resolution_hz"]


def test_insufficient_ridge_frames_are_nan_not_zero() -> None:
    candidate = _tone(duration=0.02)
    result = _extract(candidate)
    for feature in TIER_A_FEATURE_COLUMNS[12:14]:
        assert np.isnan(result[feature])
        assert result[f"{feature}_invalid_reason"] == "insufficient_ridge_frames"


def test_primary_and_25_1000_sensitivity_bands_are_configurable() -> None:
    candidate = _tone(900.0)
    primary = _extract(candidate)
    sensitivity = _extract(candidate, config=TIER_A_25_1000_HZ_SENSITIVITY_CONFIG)
    assert primary["tier_a_psd_band_high_hz"] == 800
    assert sensitivity["tier_a_psd_band_high_hz"] == 1000
    assert sensitivity[TIER_A_FEATURE_COLUMNS[7]] == pytest.approx(900, abs=FS / 512)


def test_band_truncation_is_explicit() -> None:
    sample_rate = 1000
    time = np.arange(500) / sample_rate
    candidate = np.sin(2 * np.pi * 180 * time)
    result = _extract(candidate, sample_rate=sample_rate)
    for feature in TIER_A_FEATURE_COLUMNS[7:14]:
        assert np.isnan(result[feature])
        assert result[f"{feature}_invalid_reason"] == "band_edge_truncated"


def test_repeated_execution_is_bitwise_deterministic() -> None:
    candidate = _tone(230.0, duration=0.47)
    first = _extract(candidate)
    second = _extract(candidate.copy())
    assert first.keys() == second.keys()
    for key in first:
        left, right = first[key], second[key]
        if isinstance(left, float) and np.isnan(left):
            assert isinstance(right, float) and np.isnan(right)
        else:
            assert left == right


def test_real_proxy_integration_uses_detected_interval_and_original_references() -> None:
    phase_length = 800
    original = np.concatenate(
        [
            _tone(90, duration=0.2, amplitude=0.8),
            _tone(180, duration=0.2, amplitude=0.4),
            _tone(110, duration=0.2, amplitude=0.6),
            _tone(140, duration=0.2, amplitude=0.2),
        ]
    )
    candidate = np.zeros_like(original)
    candidate[phase_length : 2 * phase_length] = original[
        phase_length : 2 * phase_length
    ]
    masks: dict[str, np.ndarray] = {}
    for index, phase in enumerate(("s1", "systole", "s2", "diastole")):
        mask = np.zeros_like(original, dtype=bool)
        mask[index * phase_length : (index + 1) * phase_length] = True
        masks[phase] = mask
    metrics = real_proxy_metrics(
        original,
        original - candidate,
        candidate,
        np.zeros_like(original),
        FS,
        phase_masks=masks,
        target_phase="systole",
    )
    assert metrics["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert metrics[TIER_A_FEATURE_COLUMNS[0]] == pytest.approx(metrics["duration_ratio"])
    assert metrics["tier_a_s1_reference_rms"] == pytest.approx(0.8 / np.sqrt(2))
    assert metrics["tier_a_s2_reference_rms"] == pytest.approx(0.6 / np.sqrt(2))
    np.testing.assert_allclose(
        original,
        (original - candidate) + candidate + np.zeros_like(original),
        atol=0,
    )
