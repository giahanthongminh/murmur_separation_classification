"""Versioned Tier A murmur-candidate features from the Task 5 protocol.

The functions in this module characterize an already detected candidate interval.
They do not detect boundaries, select components, or alter separated signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.signal import hilbert, savgol_filter, spectrogram, welch


FEATURE_SCHEMA_VERSION: Final = "task5-tier-a-v1.0.0"

TIER_A_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "tier_a_relative_murmur_duration",
    "tier_a_temporal_midpoint_normalized",
    "tier_a_murmur_rms_relative_s1_s2_db",
    "tier_a_envelope_peak_position_normalized",
    "tier_a_envelope_rise_slope_robust",
    "tier_a_envelope_decay_slope_robust",
    "tier_a_envelope_fullness",
    "tier_a_psd_dominant_frequency_hz",
    "tier_a_psd_median_frequency_hz",
    "tier_a_psd_bandwidth_95_hz",
    "tier_a_psd_energy_fraction_above_200_hz",
    "tier_a_psd_entropy_normalized",
    "tier_a_ridge_slope_robust_hz_per_normalized_time",
    "tier_a_ridge_variability_mad_hz",
)

TIER_A_FEATURE_UNITS: Final[dict[str, str]] = {
    TIER_A_FEATURE_COLUMNS[0]: "dimensionless_phase_fraction",
    TIER_A_FEATURE_COLUMNS[1]: "dimensionless_phase_position",
    TIER_A_FEATURE_COLUMNS[2]: "dB",
    TIER_A_FEATURE_COLUMNS[3]: "dimensionless_candidate_position",
    TIER_A_FEATURE_COLUMNS[4]: "normalized_envelope_per_normalized_candidate_time",
    TIER_A_FEATURE_COLUMNS[5]: "normalized_envelope_per_normalized_candidate_time",
    TIER_A_FEATURE_COLUMNS[6]: "dimensionless",
    TIER_A_FEATURE_COLUMNS[7]: "Hz",
    TIER_A_FEATURE_COLUMNS[8]: "Hz",
    TIER_A_FEATURE_COLUMNS[9]: "Hz",
    TIER_A_FEATURE_COLUMNS[10]: "dimensionless",
    TIER_A_FEATURE_COLUMNS[11]: "dimensionless",
    TIER_A_FEATURE_COLUMNS[12]: "Hz_per_normalized_candidate_time",
    TIER_A_FEATURE_COLUMNS[13]: "Hz",
}

# These are definition changes, not aliases. Existing columns remain exported.
LEGACY_TO_TIER_A_COLUMN_MAP: Final[dict[str, str]] = {
    "duration_ratio": TIER_A_FEATURE_COLUMNS[0],
    "murmur_peak_relative_to_s1_s2_ratio": TIER_A_FEATURE_COLUMNS[2],
    "envelope_time_to_peak_ratio": TIER_A_FEATURE_COLUMNS[3],
    "envelope_rising_slope_normalized": TIER_A_FEATURE_COLUMNS[4],
    "envelope_falling_slope_normalized": TIER_A_FEATURE_COLUMNS[5],
    "envelope_area_normalized": TIER_A_FEATURE_COLUMNS[6],
    "psd_primary_peak_frequency_hz": TIER_A_FEATURE_COLUMNS[7],
    "psd_above_200_hz_ratio": TIER_A_FEATURE_COLUMNS[10],
    "residual_spectral_entropy": TIER_A_FEATURE_COLUMNS[11],
    "time_frequency_ridge_slope_hz_per_second": TIER_A_FEATURE_COLUMNS[12],
    "time_frequency_ridge_variability_hz": TIER_A_FEATURE_COLUMNS[13],
}

TIER_A_DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "feature_schema_version",
    "tier_a_candidate_sample_count",
    "tier_a_phase_sample_count",
    "tier_a_phase_start_sample",
    "tier_a_phase_end_sample_exclusive",
    "tier_a_candidate_start_sample_in_phase",
    "tier_a_candidate_end_sample_in_phase_exclusive",
    "tier_a_candidate_rms",
    "tier_a_s1_reference_rms",
    "tier_a_s2_reference_rms",
    "tier_a_s1_s2_reference_rms",
    "tier_a_envelope_requested_window_samples",
    "tier_a_envelope_effective_window_samples",
    "tier_a_envelope_short_window_adjusted",
    "tier_a_envelope_peak_index",
    "tier_a_envelope_peak_tie_width_samples",
    "tier_a_psd_band_low_hz",
    "tier_a_psd_band_high_hz",
    "tier_a_psd_nperseg",
    "tier_a_psd_noverlap",
    "tier_a_psd_nfft",
    "tier_a_psd_segment_count",
    "tier_a_psd_frequency_resolution_hz",
    "tier_a_psd_inband_bin_count",
    "tier_a_psd_usable_low_hz",
    "tier_a_psd_usable_high_hz",
    "tier_a_psd_inband_energy",
    "tier_a_psd_peak_tie_count",
    "tier_a_psd_peak_at_band_edge",
    "tier_a_psd_low_frequency_limit_95_hz",
    "tier_a_psd_high_frequency_limit_95_hz",
    "tier_a_ridge_band_low_hz",
    "tier_a_ridge_band_high_hz",
    "tier_a_ridge_nperseg",
    "tier_a_ridge_noverlap",
    "tier_a_ridge_frame_count",
    "tier_a_ridge_frequency_resolution_hz",
    "tier_a_ridge_valid_frame_count",
    "tier_a_ridge_valid_frame_fraction",
    "tier_a_ridge_edge_frame_count",
    "tier_a_ridge_edge_frame_fraction",
    "tier_a_ridge_weak_three_frame_support",
)

TIER_A_EXPORT_COLUMNS: Final[tuple[str, ...]] = (
    *TIER_A_DIAGNOSTIC_COLUMNS,
    *(
        column
        for feature in TIER_A_FEATURE_COLUMNS
        for column in (feature, f"{feature}_valid", f"{feature}_invalid_reason")
    ),
)


@dataclass(frozen=True)
class TierAFeatureConfig:
    """Frozen numerical choices for Task 5 candidate characterization."""

    spectral_low_hz: float = 25.0
    spectral_high_hz: float = 800.0
    envelope_window_ms: float = 12.0
    envelope_polynomial_order: int = 2
    ridge_minimum_frames: int = 3
    ridge_minimum_valid_fraction: float = 0.80
    amplitude_floor: float = 1e-12
    psd_energy_floor: float = 1e-24

    def __post_init__(self) -> None:
        if not 0 <= self.spectral_low_hz < self.spectral_high_hz:
            raise ValueError("spectral band must have increasing non-negative limits")
        if self.envelope_window_ms <= 0:
            raise ValueError("envelope_window_ms must be positive")
        if self.envelope_polynomial_order < 0:
            raise ValueError("envelope_polynomial_order must be non-negative")
        if self.ridge_minimum_frames < 3:
            raise ValueError("ridge_minimum_frames must be at least three")
        if not 0 < self.ridge_minimum_valid_fraction <= 1:
            raise ValueError("ridge_minimum_valid_fraction must be in (0, 1]")


DEFAULT_TIER_A_FEATURE_CONFIG: Final = TierAFeatureConfig()
TIER_A_25_1000_HZ_SENSITIVITY_CONFIG: Final = TierAFeatureConfig(
    spectral_high_hz=1000.0
)


def _feature_status_columns() -> dict[str, object]:
    result: dict[str, object] = {}
    for name in TIER_A_FEATURE_COLUMNS:
        result[name] = np.nan
        result[f"{name}_valid"] = False
        result[f"{name}_invalid_reason"] = None
    return result


def _set_feature(
    result: dict[str, object],
    name: str,
    value: float | None,
    reason: str | None = None,
) -> None:
    valid = reason is None and value is not None and np.isfinite(value)
    result[name] = float(value) if valid else np.nan
    result[f"{name}_valid"] = bool(valid)
    result[f"{name}_invalid_reason"] = None if valid else (reason or "computation_error")


def _invalidate(result: dict[str, object], names: tuple[str, ...], reason: str) -> None:
    for name in names:
        _set_feature(result, name, None, reason)


def _as_vector(values: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(values, dtype=float)
    return vector if vector.ndim == 1 else None


def _center(values: np.ndarray) -> np.ndarray:
    return values - float(np.mean(values))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _median_pairwise_slope(x: np.ndarray, y: np.ndarray) -> float:
    slopes: list[float] = []
    for left in range(len(x) - 1):
        delta_x = x[left + 1 :] - x[left]
        valid = delta_x != 0
        if np.any(valid):
            slopes.extend(((y[left + 1 :] - y[left])[valid] / delta_x[valid]).tolist())
    if not slopes:
        raise ValueError("robust slope has no distinct time pairs")
    return float(np.median(np.asarray(slopes, dtype=float)))


def smoothed_hilbert_envelope(
    signal: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float = 12.0,
    polynomial_order: int = 2,
    remove_mean: bool = False,
) -> tuple[np.ndarray, int, bool]:
    """Return the common Hilbert/Savitzky--Golay envelope and support metadata."""

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        return np.asarray([], dtype=float), 0, False
    if remove_mean:
        values = _center(values)
    envelope = np.abs(hilbert(values)) if len(values) > 2 else np.abs(values)
    minimum_window = polynomial_order + 2
    if minimum_window % 2 == 0:
        minimum_window += 1
    if len(values) < minimum_window:
        return envelope.astype(float), 0, False
    requested = max(minimum_window, int(round(sample_rate * window_ms / 1000.0)) | 1)
    largest = len(values) if len(values) % 2 else len(values) - 1
    effective = min(requested, largest)
    adjusted = effective < requested
    smoothed = savgol_filter(
        envelope,
        effective,
        polynomial_order,
        mode="interp",
    )
    return np.maximum(smoothed, 0.0).astype(float), int(effective), bool(adjusted)


def _timing_features(
    result: dict[str, object],
    phase_start: int,
    phase_end: int,
    onset: int | None,
    offset: int | None,
) -> str | None:
    timing_names = TIER_A_FEATURE_COLUMNS[:2]
    phase_length = phase_end - phase_start
    if phase_length <= 0:
        _invalidate(result, timing_names, "invalid_phase")
        return "invalid_phase"
    if (
        onset is None
        or offset is None
        or onset < phase_start
        or offset > phase_end
        or offset <= onset
    ):
        _invalidate(result, timing_names, "invalid_interval")
        return "invalid_interval"
    _set_feature(
        result,
        timing_names[0],
        (offset - onset) / phase_length,
    )
    _set_feature(
        result,
        timing_names[1],
        ((onset + offset) / 2.0 - phase_start) / phase_length,
    )
    return None


def _amplitude_features(
    result: dict[str, object],
    candidate: np.ndarray,
    s1_reference: np.ndarray | None,
    s2_reference: np.ndarray | None,
    config: TierAFeatureConfig,
) -> None:
    name = TIER_A_FEATURE_COLUMNS[2]
    candidate_rms = _rms(_center(candidate))
    result["tier_a_candidate_rms"] = candidate_rms
    if s1_reference is None or s2_reference is None:
        _set_feature(result, name, None, "silent_reference")
        return
    s1 = _as_vector(s1_reference)
    s2 = _as_vector(s2_reference)
    if s1 is None or s2 is None or s1.size == 0 or s2.size == 0:
        _set_feature(result, name, None, "silent_reference")
        return
    if not np.all(np.isfinite(s1)) or not np.all(np.isfinite(s2)):
        _set_feature(result, name, None, "nonfinite_input")
        return
    s1_rms = _rms(_center(s1))
    s2_rms = _rms(_center(s2))
    reference_rms = float(np.sqrt((s1_rms**2 + s2_rms**2) / 2.0))
    result["tier_a_s1_reference_rms"] = s1_rms
    result["tier_a_s2_reference_rms"] = s2_rms
    result["tier_a_s1_s2_reference_rms"] = reference_rms
    if candidate_rms <= config.amplitude_floor:
        _set_feature(result, name, None, "silent_candidate")
    elif reference_rms <= config.amplitude_floor:
        _set_feature(result, name, None, "silent_reference")
    else:
        _set_feature(result, name, 20.0 * np.log10(candidate_rms / reference_rms))


def _envelope_features(
    result: dict[str, object],
    candidate: np.ndarray,
    sample_rate: int,
    config: TierAFeatureConfig,
) -> None:
    peak_name, rise_name, decay_name, fullness_name = TIER_A_FEATURE_COLUMNS[3:7]
    if candidate.size < 5:
        _invalidate(
            result,
            (peak_name, rise_name, decay_name, fullness_name),
            "insufficient_envelope_support",
        )
        return
    try:
        envelope, window, adjusted = smoothed_hilbert_envelope(
            candidate,
            sample_rate,
            window_ms=config.envelope_window_ms,
            polynomial_order=config.envelope_polynomial_order,
            remove_mean=True,
        )
        result["tier_a_envelope_requested_window_samples"] = max(
            config.envelope_polynomial_order + 3,
            int(round(sample_rate * config.envelope_window_ms / 1000.0)) | 1,
        )
        result["tier_a_envelope_effective_window_samples"] = window
        result["tier_a_envelope_short_window_adjusted"] = adjusted
        maximum = float(np.max(envelope, initial=0.0))
        if not np.all(np.isfinite(envelope)):
            _invalidate(
                result,
                (peak_name, rise_name, decay_name, fullness_name),
                "nonfinite_input",
            )
            return
        if maximum <= config.amplitude_floor:
            _invalidate(
                result,
                (peak_name, rise_name, decay_name, fullness_name),
                "silent_candidate",
            )
            return
        peak_index = int(np.argmax(envelope))
        normalized = envelope / maximum
        normalized_time = np.linspace(0.0, 1.0, len(envelope))
        tied = np.isclose(envelope, maximum, rtol=1e-12, atol=config.amplitude_floor)
        tie_end = peak_index
        while tie_end + 1 < len(tied) and tied[tie_end + 1]:
            tie_end += 1
        result["tier_a_envelope_peak_index"] = peak_index
        result["tier_a_envelope_peak_tie_width_samples"] = tie_end - peak_index + 1
        _set_feature(result, peak_name, peak_index / (len(envelope) - 1))
        _set_feature(result, fullness_name, float(np.mean(normalized)))
        if peak_index + 1 < 3:
            _set_feature(result, rise_name, None, "insufficient_envelope_support")
        else:
            _set_feature(
                result,
                rise_name,
                _median_pairwise_slope(
                    normalized_time[: peak_index + 1], normalized[: peak_index + 1]
                ),
            )
        if len(envelope) - peak_index < 3:
            _set_feature(result, decay_name, None, "insufficient_envelope_support")
        else:
            _set_feature(
                result,
                decay_name,
                _median_pairwise_slope(
                    normalized_time[peak_index:], normalized[peak_index:]
                ),
            )
    except (FloatingPointError, ValueError, ZeroDivisionError):
        _invalidate(
            result,
            (peak_name, rise_name, decay_name, fullness_name),
            "computation_error",
        )


def _psd_features(
    result: dict[str, object],
    candidate: np.ndarray,
    sample_rate: int,
    config: TierAFeatureConfig,
) -> None:
    names = TIER_A_FEATURE_COLUMNS[7:12]
    if sample_rate <= 0 or sample_rate / 2.0 < config.spectral_high_hz:
        _invalidate(result, names, "band_edge_truncated")
        return
    try:
        nperseg = min(512, len(candidate))
        noverlap = nperseg // 2 if nperseg > 1 else 0
        frequencies, power = welch(
            _center(candidate),
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nperseg,
            detrend="constant",
            return_onesided=True,
            scaling="density",
        )
        band = (frequencies >= config.spectral_low_hz) & (
            frequencies <= config.spectral_high_hz
        )
        band_frequencies = frequencies[band]
        band_power = power[band]
        hop = nperseg - noverlap
        segment_count = 1 + max(0, (len(candidate) - nperseg) // max(1, hop))
        resolution = float(sample_rate / nperseg)
        result.update(
            {
                "tier_a_psd_band_low_hz": config.spectral_low_hz,
                "tier_a_psd_band_high_hz": config.spectral_high_hz,
                "tier_a_psd_nperseg": nperseg,
                "tier_a_psd_noverlap": noverlap,
                "tier_a_psd_nfft": nperseg,
                "tier_a_psd_segment_count": segment_count,
                "tier_a_psd_frequency_resolution_hz": resolution,
                "tier_a_psd_inband_bin_count": int(len(band_frequencies)),
                "tier_a_psd_usable_low_hz": (
                    float(band_frequencies[0]) if len(band_frequencies) else np.nan
                ),
                "tier_a_psd_usable_high_hz": (
                    float(band_frequencies[-1]) if len(band_frequencies) else np.nan
                ),
            }
        )
        total = float(np.sum(band_power))
        result["tier_a_psd_inband_energy"] = total
        if (
            band_power.size == 0
            or not np.all(np.isfinite(band_power))
            or total <= config.psd_energy_floor
        ):
            _invalidate(result, names, "unusable_psd")
            return
        primary = int(np.argmax(band_power))
        peak = float(band_power[primary])
        peak_ties = np.isclose(band_power, peak, rtol=1e-12, atol=config.psd_energy_floor)
        result["tier_a_psd_peak_tie_count"] = int(np.count_nonzero(peak_ties))
        result["tier_a_psd_peak_at_band_edge"] = bool(
            primary == 0 or primary == len(band_power) - 1
        )
        cumulative = np.cumsum(band_power) / total

        def quantile(q: float) -> float:
            index = min(len(band_frequencies) - 1, int(np.searchsorted(cumulative, q)))
            return float(band_frequencies[index])

        low_95 = quantile(0.025)
        high_95 = quantile(0.975)
        result["tier_a_psd_low_frequency_limit_95_hz"] = low_95
        result["tier_a_psd_high_frequency_limit_95_hz"] = high_95
        _set_feature(result, names[0], float(band_frequencies[primary]))
        _set_feature(result, names[1], quantile(0.5))
        _set_feature(result, names[2], high_95 - low_95)
        above = band_frequencies > 200.0
        if not np.any(above):
            _set_feature(result, names[3], None, "unusable_psd")
        else:
            _set_feature(result, names[3], float(np.sum(band_power[above]) / total))
        if len(band_power) < 2:
            _set_feature(result, names[4], None, "unusable_psd")
        else:
            probabilities = band_power / total
            positive = probabilities > 0
            entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
            _set_feature(result, names[4], entropy / np.log(len(probabilities)))
    except (FloatingPointError, ValueError, ZeroDivisionError):
        _invalidate(result, names, "computation_error")


def _ridge_features(
    result: dict[str, object],
    candidate: np.ndarray,
    sample_rate: int,
    config: TierAFeatureConfig,
) -> None:
    names = TIER_A_FEATURE_COLUMNS[12:14]
    if sample_rate <= 0 or sample_rate / 2.0 < config.spectral_high_hz:
        _invalidate(result, names, "band_edge_truncated")
        return
    try:
        nperseg = min(128, len(candidate))
        noverlap = nperseg // 2 if nperseg > 1 else 0
        frequencies, times, power = spectrogram(
            _center(candidate),
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nperseg,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            mode="psd",
        )
        band = (frequencies >= config.spectral_low_hz) & (
            frequencies <= config.spectral_high_hz
        )
        band_frequencies = frequencies[band]
        band_power = power[band]
        frame_count = int(power.shape[1]) if power.ndim == 2 else 0
        result.update(
            {
                "tier_a_ridge_band_low_hz": config.spectral_low_hz,
                "tier_a_ridge_band_high_hz": config.spectral_high_hz,
                "tier_a_ridge_nperseg": nperseg,
                "tier_a_ridge_noverlap": noverlap,
                "tier_a_ridge_frame_count": frame_count,
                "tier_a_ridge_frequency_resolution_hz": sample_rate / nperseg,
            }
        )
        if band_power.size == 0 or frame_count == 0:
            _invalidate(result, names, "insufficient_ridge_frames")
            return
        frame_energy = np.sum(band_power, axis=0)
        valid_frames = np.all(np.isfinite(band_power), axis=0) & (
            frame_energy > config.psd_energy_floor
        )
        valid_count = int(np.count_nonzero(valid_frames))
        valid_fraction = valid_count / frame_count
        result["tier_a_ridge_valid_frame_count"] = valid_count
        result["tier_a_ridge_valid_frame_fraction"] = valid_fraction
        if (
            valid_count < config.ridge_minimum_frames
            or valid_fraction < config.ridge_minimum_valid_fraction
        ):
            _invalidate(result, names, "insufficient_ridge_frames")
            return
        valid_power = band_power[:, valid_frames]
        ridge_indexes = np.argmax(valid_power, axis=0)
        ridge = band_frequencies[ridge_indexes]
        valid_times = times[valid_frames]
        normalized_time = valid_times / (len(candidate) / sample_rate)
        slope = _median_pairwise_slope(normalized_time, ridge)
        intercept = float(np.median(ridge - slope * normalized_time))
        residuals = ridge - (slope * normalized_time + intercept)
        variability = 1.4826 * float(
            np.median(np.abs(residuals - np.median(residuals)))
        )
        edge = (ridge_indexes == 0) | (ridge_indexes == len(band_frequencies) - 1)
        result["tier_a_ridge_edge_frame_count"] = int(np.count_nonzero(edge))
        result["tier_a_ridge_edge_frame_fraction"] = float(np.mean(edge))
        result["tier_a_ridge_weak_three_frame_support"] = valid_count == 3
        _set_feature(result, names[0], slope)
        _set_feature(result, names[1], variability)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        _invalidate(result, names, "computation_error")


def extract_tier_a_features(
    candidate: np.ndarray,
    sample_rate: int,
    *,
    phase_start_sample: int,
    phase_end_sample: int,
    onset_sample: int | None,
    offset_sample: int | None,
    s1_reference: np.ndarray | None,
    s2_reference: np.ndarray | None,
    config: TierAFeatureConfig = DEFAULT_TIER_A_FEATURE_CONFIG,
) -> dict[str, object]:
    """Compute all Tier A outputs and explicit feature-level validity metadata."""

    result: dict[str, object] = {
        column: np.nan for column in TIER_A_DIAGNOSTIC_COLUMNS
    }
    result.update(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "tier_a_candidate_sample_count": int(np.asarray(candidate).size),
            "tier_a_phase_sample_count": int(phase_end_sample - phase_start_sample),
            "tier_a_phase_start_sample": phase_start_sample,
            "tier_a_phase_end_sample_exclusive": phase_end_sample,
            "tier_a_candidate_start_sample_in_phase": (
                None if onset_sample is None else onset_sample - phase_start_sample
            ),
            "tier_a_candidate_end_sample_in_phase_exclusive": (
                None if offset_sample is None else offset_sample - phase_start_sample
            ),
            **_feature_status_columns(),
        }
    )
    interval_reason = _timing_features(
        result,
        phase_start_sample,
        phase_end_sample,
        onset_sample,
        offset_sample,
    )
    values = _as_vector(candidate)
    signal_names = TIER_A_FEATURE_COLUMNS[2:]
    if values is None:
        _invalidate(result, signal_names, "nonfinite_input")
        return result
    if interval_reason is not None or offset_sample is None or onset_sample is None:
        _invalidate(result, signal_names, interval_reason or "invalid_interval")
        return result
    if values.size == 0:
        _invalidate(result, signal_names, "empty_candidate")
        return result
    if len(values) != offset_sample - onset_sample:
        _invalidate(result, signal_names, "invalid_interval")
        return result
    if sample_rate <= 0 or not np.all(np.isfinite(values)):
        _invalidate(result, signal_names, "nonfinite_input")
        return result
    centered = _center(values)
    if _rms(centered) <= config.amplitude_floor:
        _invalidate(result, signal_names, "silent_candidate")
        return result

    _amplitude_features(result, values, s1_reference, s2_reference, config)
    _envelope_features(result, values, sample_rate, config)
    _psd_features(result, values, sample_rate, config)
    _ridge_features(result, values, sample_rate, config)
    return result
