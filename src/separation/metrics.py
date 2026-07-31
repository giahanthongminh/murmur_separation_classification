"""Proxy and ground-truth metrics for murmur-isolation experiments."""

from __future__ import annotations

import numpy as np
from scipy.signal import (
    find_peaks,
    hilbert,
    peak_widths,
    savgol_filter,
    spectrogram,
    welch,
)


EPSILON = 1e-12


def smooth_amplitude_envelope(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return a deterministic short-window envelope for morphology analysis."""

    values = np.asarray(signal, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    envelope = np.abs(hilbert(values)) if len(values) > 2 else np.abs(values)
    if len(values) < 5:
        return envelope
    window = min(
        len(values) if len(values) % 2 else len(values) - 1,
        max(5, int(round(sample_rate * 0.012)) | 1),
    )
    if window < 5:
        return envelope
    return np.maximum(savgol_filter(envelope, window, 2, mode="interp"), 0.0)


def dominant_frequency_trajectory(
    signal: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return spectrogram frame times and the dominant-frequency ridge."""

    values = np.asarray(signal, dtype=float)
    if values.size < 4 or energy(values) <= EPSILON:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    nperseg = min(128, len(values))
    frequencies, times, power = spectrogram(
        values,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
    )
    band = (frequencies >= 20.0) & (frequencies <= min(1000.0, sample_rate / 2))
    if not band.any() or power.shape[1] == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    band_frequencies = frequencies[band]
    ridge = band_frequencies[np.argmax(power[band], axis=0)]
    return times.astype(float), ridge.astype(float)


def wavelet_scalogram(
    signal: np.ndarray,
    sample_rate: int,
    *,
    frequency_bins: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return a Morlet scalogram, with a graceful optional-dependency fallback."""

    values = np.asarray(signal, dtype=float)
    if values.size < 8 or energy(values) <= EPSILON:
        empty = np.asarray([], dtype=float)
        return empty, empty, np.empty((0, 0), dtype=float), "insufficient"
    try:
        import pywt
    except ImportError:
        empty = np.asarray([], dtype=float)
        return empty, empty, np.empty((0, 0), dtype=float), "pywavelets_unavailable"
    maximum_frequency = min(1000.0, sample_rate / 2.0)
    if maximum_frequency <= 20.0:
        empty = np.asarray([], dtype=float)
        return empty, empty, np.empty((0, 0), dtype=float), "insufficient"
    requested_frequencies = np.geomspace(20.0, maximum_frequency, frequency_bins)
    central_frequency = float(pywt.central_frequency("morl"))
    scales = central_frequency * sample_rate / requested_frequencies
    coefficients, frequencies = pywt.cwt(
        values,
        scales,
        "morl",
        sampling_period=1.0 / sample_rate,
    )
    power = np.abs(coefficients) ** 2
    order = np.argsort(frequencies)
    times = np.arange(len(values), dtype=float) / sample_rate
    return times, frequencies[order].astype(float), power[order], "available"


def boundary_robustness_metrics(
    phase_candidate: np.ndarray,
    sample_rate: int,
    onset_sample: int | None,
    offset_sample: int | None,
    *,
    jitter_ms: float = 10.0,
) -> dict[str, float | str]:
    """Measure feature sensitivity to small onset/offset perturbations."""

    values = np.asarray(phase_candidate, dtype=float)
    if onset_sample is None or offset_sample is None or values.size < 8:
        return {
            "boundary_jitter_ms": float(jitter_ms),
            "boundary_variant_count": 0.0,
            "boundary_amplitude_rms_relative_range": 0.0,
            "boundary_psd_peak_frequency_range_hz": 0.0,
            "boundary_envelope_time_to_peak_ratio_range": 0.0,
            "boundary_psd_width_relative_range": 0.0,
            "boundary_envelope_shape_agreement_ratio": 0.0,
            "boundary_stability_status": "insufficient",
        }
    onset = int(onset_sample)
    offset = int(offset_sample)
    jitter = max(1, int(round(jitter_ms * sample_rate / 1000.0)))
    proposed = [
        (onset, offset),
        (onset - jitter, offset),
        (onset + jitter, offset),
        (onset, offset - jitter),
        (onset, offset + jitter),
        (onset - jitter, offset + jitter),
        (onset + jitter, offset - jitter),
    ]
    bounds: list[tuple[int, int]] = []
    for start, end in proposed:
        start = max(0, start)
        end = min(len(values), end)
        if end - start >= 8 and (start, end) not in bounds:
            bounds.append((start, end))
    measurements: list[tuple[float, float, float, float, float]] = []
    envelope_shapes: list[str] = []
    for start, end in bounds:
        segment = values[start:end]
        rms = float(np.sqrt(np.mean(segment**2)))
        envelope_features = _envelope_morphology(
            smooth_amplitude_envelope(segment, sample_rate), sample_rate
        )
        frequencies, psd = welch(
            segment,
            fs=sample_rate,
            nperseg=min(512, len(segment)),
            detrend="constant",
        )
        psd_features = _psd_morphology(frequencies, psd)
        envelope_shapes.append(str(envelope_features["envelope_shape"]))
        measurements.append(
            (
                rms,
                float(psd_features["psd_primary_peak_frequency_hz"]),
                float(envelope_features["envelope_time_to_peak_ratio"]),
                float(psd_features["psd_primary_peak_width_hz"]),
                float(sample_rate / min(512, len(segment))),
            )
        )
    if not measurements:
        return boundary_robustness_metrics(
            np.asarray([], dtype=float), sample_rate, None, None, jitter_ms=jitter_ms
        )
    table = np.asarray(measurements, dtype=float)
    nominal = table[0]
    rms_relative_range = float(np.ptp(table[:, 0]) / (nominal[0] + EPSILON))
    frequency_range = float(np.ptp(table[:, 1]))
    time_to_peak_range = (
        0.0
        if envelope_shapes and all(shape == "constant" for shape in envelope_shapes)
        else float(np.ptp(table[:, 2]))
    )
    width_relative_range = float(np.ptp(table[:, 3]) / (nominal[3] + EPSILON))
    shape_agreement = float(
        np.mean(np.asarray(envelope_shapes) == envelope_shapes[0])
    )
    frequency_tolerance = max(50.0, 2.0 * nominal[4])
    status = (
        "robust"
        if rms_relative_range <= 0.25
        and frequency_range <= frequency_tolerance
        and time_to_peak_range <= 0.25
        and width_relative_range <= 0.50
        and shape_agreement >= 0.70
        else "sensitive"
    )
    return {
        "boundary_jitter_ms": float(jitter_ms),
        "boundary_variant_count": float(len(measurements)),
        "boundary_amplitude_rms_relative_range": rms_relative_range,
        "boundary_psd_peak_frequency_range_hz": frequency_range,
        "boundary_envelope_time_to_peak_ratio_range": time_to_peak_range,
        "boundary_psd_width_relative_range": width_relative_range,
        "boundary_envelope_shape_agreement_ratio": shape_agreement,
        "boundary_stability_status": status,
    }


def _contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.pad(np.asarray(mask, dtype=np.int8), (1, 1)))
    return [
        (int(start), int(end))
        for start, end in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        )
    ]


def _envelope_morphology(
    envelope: np.ndarray, sample_rate: int
) -> dict[str, float | str]:
    n_samples = len(envelope)
    if n_samples < 5 or float(np.max(envelope, initial=0.0)) <= EPSILON:
        return {
            "envelope_shape": "insufficient",
            "envelope_time_to_peak_ratio": 0.0,
            "envelope_rise_time_seconds": 0.0,
            "envelope_decay_time_seconds": 0.0,
            "envelope_rising_slope_normalized": 0.0,
            "envelope_falling_slope_normalized": 0.0,
            "envelope_variation_coefficient": 0.0,
            "envelope_area_normalized": 0.0,
            "envelope_symmetry": 0.0,
            "envelope_prominent_peak_count": 0.0,
            "active_burst_count": 0.0,
            "active_time_ratio": 0.0,
            "longest_burst_ratio": 0.0,
        }

    edge = min(max(1, int(round(0.04 * n_samples))), max(1, n_samples // 4))
    stop = max(edge + 1, n_samples - edge)
    interior = envelope[edge:stop]
    peak_index = edge + int(np.argmax(interior))
    peak = float(envelope[peak_index])
    normalized = envelope / (peak + EPSILON)
    x = np.linspace(0.0, 1.0, n_samples)
    time_to_peak = peak_index / max(1, n_samples - 1)

    def fitted_slope(start: int, end: int) -> float:
        if end - start < 3:
            return 0.0
        return float(np.polyfit(x[start:end], normalized[start:end], 1)[0])

    rising_slope = fitted_slope(0, peak_index + 1)
    falling_slope = fitted_slope(peak_index, n_samples)
    segment = max(1, n_samples // 5)
    start_level = float(np.mean(normalized[:segment]))
    middle_start = max(0, n_samples // 2 - segment // 2)
    middle_level = float(np.mean(normalized[middle_start : middle_start + segment]))
    end_level = float(np.mean(normalized[-segment:]))
    coefficient_variation = float(
        np.std(envelope) / (float(np.mean(envelope)) + EPSILON)
    )
    robust_range = float(np.percentile(normalized, 90) - np.percentile(normalized, 10))

    distance = max(1, int(round(0.08 * n_samples)))
    peaks, _ = find_peaks(normalized, prominence=0.12, distance=distance)
    prominent_peak_count = len(peaks)

    if coefficient_variation < 0.12 or robust_range < 0.20:
        shape = "constant"
    elif end_level - start_level > 0.25 and end_level >= 0.90 * middle_level:
        shape = "crescendo"
    elif start_level - end_level > 0.25 and start_level >= 0.90 * middle_level:
        shape = "decrescendo"
    elif (
        0.18 <= time_to_peak <= 0.82
        and middle_level > 1.20 * start_level
        and middle_level > 1.20 * end_level
    ):
        shape = "crescendo_decrescendo"
    elif prominent_peak_count >= 2:
        shape = "multi_peak"
    else:
        shape = "irregular"

    baseline = float(np.percentile(envelope, 20))
    mad = float(np.median(np.abs(envelope - baseline)))
    threshold = max(
        baseline + 2.0 * 1.4826 * mad,
        baseline + 0.20 * (float(np.max(envelope)) - baseline),
    )
    active = envelope >= threshold
    minimum = max(1, min(int(round(0.005 * sample_rate)), n_samples // 10))
    intervals = [
        interval
        for interval in _contiguous_intervals(active)
        if interval[1] - interval[0] >= minimum
    ]
    active_samples = sum(end - start for start, end in intervals)
    longest = max((end - start for start, end in intervals), default=0)
    return {
        "envelope_shape": shape,
        "envelope_time_to_peak_ratio": float(time_to_peak),
        "envelope_rise_time_seconds": float(peak_index / sample_rate),
        "envelope_decay_time_seconds": float((n_samples - 1 - peak_index) / sample_rate),
        "envelope_rising_slope_normalized": rising_slope,
        "envelope_falling_slope_normalized": falling_slope,
        "envelope_variation_coefficient": coefficient_variation,
        "envelope_area_normalized": float(np.mean(normalized)),
        "envelope_symmetry": float(max(0.0, 1.0 - abs(2.0 * time_to_peak - 1.0))),
        "envelope_prominent_peak_count": float(prominent_peak_count),
        "active_burst_count": float(len(intervals)),
        "active_time_ratio": float(active_samples / n_samples),
        "longest_burst_ratio": float(longest / n_samples),
    }


def _psd_morphology(
    frequencies: np.ndarray, psd: np.ndarray
) -> dict[str, float | str]:
    resolution = float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.0
    band = (frequencies >= 20.0) & (frequencies <= 1000.0)
    band_frequencies = frequencies[band]
    band_psd = psd[band]
    if band_psd.size < 3 or float(np.max(band_psd, initial=0.0)) <= EPSILON:
        return {
            "psd_morphology": "insufficient",
            "psd_primary_peak_frequency_hz": 0.0,
            "psd_prominent_peak_count": 0.0,
            "psd_primary_peak_width_hz": 0.0,
            "psd_primary_peak_prominence_ratio": 0.0,
            "psd_secondary_peak_frequency_hz": 0.0,
            "psd_secondary_to_primary_ratio": 0.0,
            "psd_peak_separation_hz": 0.0,
            "psd_primary_q_factor": 0.0,
            "psd_energy_concentration": 0.0,
        }
    normalized = band_psd / (float(np.max(band_psd)) + EPSILON)
    distance = max(1, int(round(40.0 / max(resolution, EPSILON))))
    peaks, properties = find_peaks(normalized, prominence=0.10, distance=distance)
    detected_peak = peaks.size > 0
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(normalized))])
        prominences = np.asarray([0.0])
    else:
        prominences = np.asarray(properties["prominences"], dtype=float)
    order = np.argsort(normalized[peaks])[::-1]
    peaks = peaks[order]
    prominences = prominences[order]
    primary = int(peaks[0])
    primary_frequency = float(band_frequencies[primary])
    if detected_peak:
        widths, _, left, right = peak_widths(normalized, [primary], rel_height=0.5)
        width_hz = float(max(resolution, widths[0] * resolution))
        left_index = max(0, int(np.floor(left[0])))
        right_index = min(len(band_psd), int(np.ceil(right[0])) + 1)
    else:
        width_hz = resolution
        left_index = primary
        right_index = primary + 1
    concentration = float(
        np.sum(band_psd[left_index:right_index]) / (np.sum(band_psd) + EPSILON)
    )
    secondary_frequency = 0.0
    secondary_ratio = 0.0
    separation = 0.0
    if len(peaks) > 1:
        secondary = int(peaks[1])
        secondary_frequency = float(band_frequencies[secondary])
        secondary_ratio = float(normalized[secondary] / (normalized[primary] + EPSILON))
        separation = abs(secondary_frequency - primary_frequency)
    q_factor = primary_frequency / (width_hz + EPSILON)
    peak_count = int(np.sum(normalized[peaks] >= 0.15))
    primary_prominence = float(prominences[0]) if len(prominences) else 0.0
    if primary_prominence < 0.10:
        morphology = "no_clear_peak"
    elif peak_count >= 3:
        morphology = "multi_peak"
    elif peak_count == 2 and secondary_ratio >= 0.25:
        morphology = "double_peak"
    elif q_factor >= 4.0:
        morphology = "narrow_single_peak"
    else:
        morphology = "broad_single_peak"
    return {
        "psd_morphology": morphology,
        "psd_primary_peak_frequency_hz": primary_frequency,
        "psd_prominent_peak_count": float(peak_count),
        "psd_primary_peak_width_hz": width_hz,
        "psd_primary_peak_prominence_ratio": primary_prominence,
        "psd_secondary_peak_frequency_hz": secondary_frequency,
        "psd_secondary_to_primary_ratio": secondary_ratio,
        "psd_peak_separation_hz": float(separation),
        "psd_primary_q_factor": float(q_factor),
        "psd_energy_concentration": concentration,
    }


def _ridge_morphology(
    times: np.ndarray, ridge: np.ndarray, frequency_resolution: float
) -> dict[str, float | str]:
    if len(times) < 3 or len(ridge) != len(times):
        return {
            "time_frequency_ridge_direction": "insufficient",
            "time_frequency_ridge_start_hz": 0.0,
            "time_frequency_ridge_end_hz": 0.0,
            "time_frequency_ridge_slope_hz_per_second": 0.0,
            "time_frequency_ridge_variability_hz": 0.0,
            "time_frequency_ridge_continuity": 0.0,
        }
    slope, intercept = np.polyfit(times, ridge, 1)
    fitted = slope * times + intercept
    edge_frames = min(2, len(ridge))
    start = float(np.median(ridge[:edge_frames]))
    end = float(np.median(ridge[-edge_frames:]))
    total_change = float(slope * (times[-1] - times[0]))
    threshold = max(2.0 * frequency_resolution, 50.0)
    if total_change > threshold:
        direction = "rising"
    elif total_change < -threshold:
        direction = "falling"
    else:
        direction = "stable"
    jumps = np.abs(np.diff(ridge))
    continuity = float(np.mean(jumps <= max(2.0 * frequency_resolution, 100.0)))
    return {
        "time_frequency_ridge_direction": direction,
        "time_frequency_ridge_start_hz": start,
        "time_frequency_ridge_end_hz": end,
        "time_frequency_ridge_slope_hz_per_second": float(slope),
        "time_frequency_ridge_variability_hz": float(np.std(ridge - fitted)),
        "time_frequency_ridge_continuity": continuity,
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.shape != y.shape or x.size == 0 or np.std(x) <= EPSILON or np.std(y) <= EPSILON:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def energy(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=float)
    return float(np.dot(values, values))


def spectral_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    values = np.asarray(signal, dtype=float)
    if values.size == 0 or energy(values) <= EPSILON:
        return {
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_entropy": 0.0,
            "dominant_frequency": 0.0,
        }
    power = np.abs(np.fft.rfft(values)) ** 2
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sample_rate)
    normalized = power / (power.sum() + EPSILON)
    centroid = float(np.sum(frequencies * normalized))
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * normalized)))
    entropy = float(-np.sum(normalized * np.log2(normalized + EPSILON)))
    max_entropy = np.log2(max(2, len(normalized)))
    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_entropy": entropy / max_entropy,
        "dominant_frequency": float(frequencies[int(np.argmax(power))]),
    }


def murmur_observation_features(
    signal: np.ndarray, sample_rate: int
) -> dict[str, float | str]:
    """Return amplitude, morphology, PSD, and time-frequency descriptors."""

    values = np.asarray(signal, dtype=float)
    empty = (
        values.size == 0
        or not np.all(np.isfinite(values))
        or energy(values) <= EPSILON
    )
    if empty:
        return {
            "amplitude_peak_abs": 0.0,
            "amplitude_rms": 0.0,
            "amplitude_mean_abs": 0.0,
            "amplitude_envelope_peak": 0.0,
            "amplitude_envelope_mean": 0.0,
            "amplitude_crest_factor": 0.0,
            "psd_peak_frequency_hz": 0.0,
            "psd_peak_power": 0.0,
            "psd_frequency_resolution_hz": 0.0,
            "psd_0_100_hz_ratio": 0.0,
            "psd_100_200_hz_ratio": 0.0,
            "psd_200_400_hz_ratio": 0.0,
            "psd_400_800_hz_ratio": 0.0,
            "psd_800_1000_hz_ratio": 0.0,
            "psd_above_1000_hz_ratio": 0.0,
            "time_frequency_peak_hz": 0.0,
            "time_frequency_peak_seconds": 0.0,
            "time_frequency_frame_count": 0.0,
            "time_frequency_frequency_resolution_hz": 0.0,
            "time_frequency_window_seconds": 0.0,
            "time_frequency_entropy": 0.0,
            "time_frequency_spectral_flux": 0.0,
            "wavelet_status": "insufficient",
            "wavelet_peak_frequency_hz": 0.0,
            "wavelet_peak_seconds": 0.0,
            "wavelet_frequency_centroid_hz": 0.0,
            "wavelet_frequency_spread_hz": 0.0,
            "wavelet_entropy": 0.0,
            "wavelet_peak_scale_energy_ratio": 0.0,
            **_envelope_morphology(np.asarray([], dtype=float), sample_rate),
            **_psd_morphology(np.asarray([], dtype=float), np.asarray([], dtype=float)),
            **_ridge_morphology(
                np.asarray([], dtype=float), np.asarray([], dtype=float), 0.0
            ),
        }

    envelope = np.abs(hilbert(values)) if len(values) > 2 else np.abs(values)
    smooth_envelope = smooth_amplitude_envelope(values, sample_rate)
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(values**2)))

    psd_frequencies, psd = welch(
        values,
        fs=sample_rate,
        nperseg=min(512, len(values)),
        detrend="constant",
    )
    psd_total = float(np.sum(psd)) + EPSILON

    def band_ratio(low: float, high: float | None) -> float:
        mask = psd_frequencies >= low
        if high is not None:
            mask &= psd_frequencies < high
        return float(np.sum(psd[mask]) / psd_total)

    nperseg = min(128, len(values))
    tf_frequencies, tf_times, tf_power = spectrogram(
        values,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
    )
    peak_frequency = 0.0
    peak_seconds = 0.0
    tf_entropy = 0.0
    spectral_flux = 0.0
    if tf_power.size:
        peak_index = np.unravel_index(int(np.argmax(tf_power)), tf_power.shape)
        peak_frequency = float(tf_frequencies[peak_index[0]])
        peak_seconds = float(tf_times[peak_index[1]])
        normalized_tf = tf_power / (float(np.sum(tf_power)) + EPSILON)
        entropy = -float(np.sum(normalized_tf * np.log2(normalized_tf + EPSILON)))
        tf_entropy = entropy / np.log2(max(2, normalized_tf.size))
        if tf_power.shape[1] > 1:
            frame_power = tf_power / (np.sum(tf_power, axis=0, keepdims=True) + EPSILON)
            spectral_flux = float(
                np.mean(np.sqrt(np.sum(np.diff(frame_power, axis=1) ** 2, axis=0)))
            )
    ridge_times, ridge = dominant_frequency_trajectory(values, sample_rate)
    frequency_resolution = float(sample_rate / nperseg)
    wavelet_times, wavelet_frequencies, wavelet_power, wavelet_status = (
        wavelet_scalogram(values, sample_rate)
    )
    wavelet_peak_frequency = 0.0
    wavelet_peak_seconds = 0.0
    wavelet_centroid = 0.0
    wavelet_spread = 0.0
    wavelet_entropy = 0.0
    wavelet_concentration = 0.0
    if wavelet_power.size:
        wavelet_peak = np.unravel_index(
            int(np.argmax(wavelet_power)), wavelet_power.shape
        )
        wavelet_peak_frequency = float(wavelet_frequencies[wavelet_peak[0]])
        wavelet_peak_seconds = float(wavelet_times[wavelet_peak[1]])
        frequency_energy = np.sum(wavelet_power, axis=1)
        normalized_frequency_energy = frequency_energy / (
            float(np.sum(frequency_energy)) + EPSILON
        )
        wavelet_centroid = float(
            np.sum(wavelet_frequencies * normalized_frequency_energy)
        )
        wavelet_spread = float(
            np.sqrt(
                np.sum(
                    (wavelet_frequencies - wavelet_centroid) ** 2
                    * normalized_frequency_energy
                )
            )
        )
        normalized_wavelet = wavelet_power / (
            float(np.sum(wavelet_power)) + EPSILON
        )
        entropy = -float(
            np.sum(normalized_wavelet * np.log2(normalized_wavelet + EPSILON))
        )
        wavelet_entropy = entropy / np.log2(max(2, normalized_wavelet.size))
        wavelet_concentration = float(np.max(normalized_frequency_energy))

    return {
        "amplitude_peak_abs": peak,
        "amplitude_rms": rms,
        "amplitude_mean_abs": float(np.mean(np.abs(values))),
        "amplitude_envelope_peak": float(np.max(envelope)),
        "amplitude_envelope_mean": float(np.mean(envelope)),
        "amplitude_crest_factor": peak / (rms + EPSILON),
        "psd_peak_frequency_hz": float(psd_frequencies[int(np.argmax(psd))]),
        "psd_peak_power": float(np.max(psd)),
        "psd_frequency_resolution_hz": float(sample_rate / min(512, len(values))),
        "psd_0_100_hz_ratio": band_ratio(0, 100),
        "psd_100_200_hz_ratio": band_ratio(100, 200),
        "psd_200_400_hz_ratio": band_ratio(200, 400),
        "psd_400_800_hz_ratio": band_ratio(400, 800),
        "psd_800_1000_hz_ratio": band_ratio(800, 1000),
        "psd_above_1000_hz_ratio": band_ratio(1000, None),
        "time_frequency_peak_hz": peak_frequency,
        "time_frequency_peak_seconds": peak_seconds,
        "time_frequency_frame_count": float(tf_power.shape[1]),
        "time_frequency_frequency_resolution_hz": float(sample_rate / nperseg),
        "time_frequency_window_seconds": float(nperseg / sample_rate),
        "time_frequency_entropy": float(tf_entropy),
        "time_frequency_spectral_flux": spectral_flux,
        "wavelet_status": wavelet_status,
        "wavelet_peak_frequency_hz": wavelet_peak_frequency,
        "wavelet_peak_seconds": wavelet_peak_seconds,
        "wavelet_frequency_centroid_hz": wavelet_centroid,
        "wavelet_frequency_spread_hz": wavelet_spread,
        "wavelet_entropy": wavelet_entropy,
        "wavelet_peak_scale_energy_ratio": wavelet_concentration,
        **_envelope_morphology(smooth_envelope, sample_rate),
        **_psd_morphology(psd_frequencies, psd),
        **_ridge_morphology(ridge_times, ridge, frequency_resolution),
    }


def detect_activity_interval(
    signal: np.ndarray,
    sample_rate: int,
    *,
    threshold_mad: float = 3.0,
    minimum_duration_ms: float = 30.0,
    merge_gap_ms: float = 20.0,
) -> dict[str, float | str | None]:
    """Detect activity inside a short cardiac phase with an energy fallback."""

    values = np.asarray(signal, dtype=float)
    if values.size == 0:
        return _empty_activity()
    if not np.all(np.isfinite(values)) or energy(values) <= EPSILON:
        return _empty_activity()
    envelope = np.abs(hilbert(values)) if len(values) > 2 else np.abs(values)
    # Use 5 ms smoothing, capped at 10% of the phase, so 70--200 ms
    # candidates are not flattened by the previous fixed 20 ms window.
    smoothing = min(
        len(values),
        max(1, min(int(round(sample_rate * 0.005)), len(values) // 10)),
    )
    kernel = np.ones(smoothing, dtype=float) / smoothing
    smooth = np.convolve(envelope, kernel, mode="same")
    baseline = float(np.percentile(smooth, 20))
    mad = float(np.median(np.abs(smooth - baseline)))
    raw_threshold = baseline + threshold_mad * 1.4826 * mad
    peak_limited_threshold = baseline + 0.60 * (float(np.max(smooth)) - baseline)
    threshold = min(raw_threshold, peak_limited_threshold)
    active = smooth >= threshold

    merge_gap = min(
        max(0, int(round(merge_gap_ms * sample_rate / 1000.0))),
        max(1, len(values) // 10),
    )
    if merge_gap and active.any():
        indexes = np.flatnonzero(active)
        for left, right in zip(indexes[:-1], indexes[1:]):
            if right - left - 1 <= merge_gap:
                active[left : right + 1] = True

    minimum = min(
        max(1, int(round(minimum_duration_ms * sample_rate / 1000.0))),
        max(2, int(round(0.15 * len(values)))),
    )
    valid_intervals: list[tuple[int, int]] = []
    changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for start, end in zip(starts, ends):
        if end - start >= minimum:
            valid_intervals.append((int(start), int(end)))
    phase_energy = values**2
    if valid_intervals:
        start = valid_intervals[0][0]
        end = valid_intervals[-1][1]
        detection_method = "adaptive_envelope"
    else:
        # A non-silent short candidate should still receive auditable timing.
        # The 5--95% cumulative-energy interval is stable and bounded even when
        # no envelope run survives duration filtering.
        cumulative = np.cumsum(phase_energy)
        cumulative /= cumulative[-1]
        start = int(np.searchsorted(cumulative, 0.05))
        end = min(len(values), int(np.searchsorted(cumulative, 0.95)) + 1)
        detection_method = "energy_quantile_fallback"
    positions = np.arange(len(values), dtype=float) / max(1, len(values) - 1)
    total = float(phase_energy.sum())
    return {
        "onset_sample": int(start),
        "offset_sample": int(end),
        "duration_samples": int(end - start),
        "onset_seconds": start / sample_rate,
        "offset_seconds": end / sample_rate,
        "duration_seconds": (end - start) / sample_rate,
        "onset_normalized": start / len(values),
        "offset_normalized": end / len(values),
        "duration_ratio": (end - start) / len(values),
        "temporal_energy_centroid": float(
            np.sum(positions * phase_energy) / (total + EPSILON)
        ),
        "peak_position": float(np.argmax(smooth) / max(1, len(values) - 1)),
        "activity_detection_method": detection_method,
        "activity_threshold": float(threshold),
    }


def _empty_activity() -> dict[str, float | str | None]:
    return {
        "onset_sample": None,
        "offset_sample": None,
        "duration_samples": None,
        "onset_seconds": None,
        "offset_seconds": None,
        "duration_seconds": None,
        "onset_normalized": None,
        "offset_normalized": None,
        "duration_ratio": None,
        "temporal_energy_centroid": None,
        "peak_position": None,
        "activity_detection_method": "silent_or_invalid",
        "activity_threshold": None,
    }


def _validated_phase_mask(
    phase_masks: dict[str, np.ndarray], name: str, length: int
) -> np.ndarray:
    if name not in phase_masks:
        raise ValueError(f"phase_masks is missing '{name}'")
    mask = np.asarray(phase_masks[name], dtype=bool)
    if mask.shape != (length,) or not mask.any():
        raise ValueError(f"phase mask '{name}' must select samples from the context")
    return mask


def real_proxy_metrics(
    original: np.ndarray,
    normal: np.ndarray,
    murmur_candidate: np.ndarray,
    noise_candidate: np.ndarray,
    sample_rate: int,
    *,
    threshold_mad: float = 3.0,
    minimum_duration_ms: float = 30.0,
    merge_gap_ms: float = 20.0,
    phase_masks: dict[str, np.ndarray] | None = None,
    target_phase: str = "systole",
) -> dict[str, float | str | None]:
    if target_phase not in {"systole", "diastole"}:
        raise ValueError("target_phase must be 'systole' or 'diastole'")
    original_energy = energy(original)
    reconstructed = normal + murmur_candidate + noise_candidate
    reconstruction_error = float(
        np.linalg.norm(np.asarray(original) - reconstructed)
        / (np.linalg.norm(original) + EPSILON)
    )
    if phase_masks is None:
        timing_candidate = np.asarray(murmur_candidate)
        murmur_region_energy_retention = energy(murmur_candidate) / (
            original_energy + EPSILON
        )
        s1_leakage_ratio = None
        s2_leakage_ratio = None
        outside_murmur_energy_ratio = None
        systole_candidate_energy_ratio = None
        diastole_candidate_energy_ratio = None
    else:
        length = len(np.asarray(original))
        s1_mask = _validated_phase_mask(phase_masks, "s1", length)
        systole_mask = _validated_phase_mask(phase_masks, "systole", length)
        s2_mask = _validated_phase_mask(phase_masks, "s2", length)
        diastole_mask = _validated_phase_mask(phase_masks, "diastole", length)
        target_mask = systole_mask if target_phase == "systole" else diastole_mask
        candidate_values = np.asarray(murmur_candidate)
        original_values = np.asarray(original)
        timing_candidate = candidate_values[target_mask]
        s1_leakage_ratio = energy(candidate_values[s1_mask]) / (
            energy(original_values[s1_mask]) + EPSILON
        )
        s2_leakage_ratio = energy(candidate_values[s2_mask]) / (
            energy(original_values[s2_mask]) + EPSILON
        )
        systole_candidate_energy_ratio = energy(candidate_values[systole_mask]) / (
            energy(original_values[systole_mask]) + EPSILON
        )
        diastole_candidate_energy_ratio = energy(candidate_values[diastole_mask]) / (
            energy(original_values[diastole_mask]) + EPSILON
        )
        murmur_region_energy_retention = energy(candidate_values[target_mask]) / (
            energy(original_values[target_mask]) + EPSILON
        )
        outside_murmur_energy_ratio = energy(candidate_values[~target_mask]) / (
            energy(candidate_values) + EPSILON
        )
    timing = detect_activity_interval(
        timing_candidate,
        sample_rate,
        threshold_mad=threshold_mad,
        minimum_duration_ms=minimum_duration_ms,
        merge_gap_ms=merge_gap_ms,
    )
    spectral = spectral_features(timing_candidate, sample_rate)
    onset = timing["onset_sample"]
    offset = timing["offset_sample"]
    if onset is not None and offset is not None and int(offset) > int(onset):
        observation_candidate = timing_candidate[int(onset) : int(offset)]
    else:
        observation_candidate = timing_candidate
    observation = murmur_observation_features(observation_candidate, sample_rate)
    robustness = boundary_robustness_metrics(
        timing_candidate,
        sample_rate,
        None if onset is None else int(onset),
        None if offset is None else int(offset),
    )
    return {
        "reconstruction_error": reconstruction_error,
        "normal_residual_correlation": safe_correlation(normal, murmur_candidate),
        "s1_leakage_ratio": s1_leakage_ratio,
        "s2_leakage_ratio": s2_leakage_ratio,
        "murmur_region_energy_retention": murmur_region_energy_retention,
        "outside_murmur_energy_ratio": outside_murmur_energy_ratio,
        "systole_candidate_energy_ratio": systole_candidate_energy_ratio,
        "diastole_candidate_energy_ratio": diastole_candidate_energy_ratio,
        "murmur_phase": target_phase,
        "noise_energy_ratio": energy(noise_candidate) / (original_energy + EPSILON),
        "residual_spectral_centroid": spectral["spectral_centroid"],
        "residual_bandwidth": spectral["spectral_bandwidth"],
        "residual_spectral_entropy": spectral["spectral_entropy"],
        "residual_dominant_frequency": spectral["dominant_frequency"],
        **observation,
        **robustness,
        **timing,
    }


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    scale = float(np.dot(est, ref) / (np.dot(ref, ref) + EPSILON))
    target = scale * ref
    distortion = est - target
    return float(10 * np.log10((energy(target) + EPSILON) / (energy(distortion) + EPSILON)))


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(
        10
        * np.log10(
            (energy(reference) + EPSILON)
            / (energy(np.asarray(reference) - np.asarray(estimate)) + EPSILON)
        )
    )


def spectral_distance(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.log1p(np.abs(np.fft.rfft(reference)))
    est = np.log1p(np.abs(np.fft.rfft(estimate)))
    return float(np.linalg.norm(ref - est) / (np.linalg.norm(ref) + EPSILON))
