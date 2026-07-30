"""Proxy and ground-truth metrics for murmur-isolation experiments."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


EPSILON = 1e-12


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
) -> dict[str, float | str | None]:
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
    else:
        length = len(np.asarray(original))
        s1_mask = _validated_phase_mask(phase_masks, "s1", length)
        systole_mask = _validated_phase_mask(phase_masks, "systole", length)
        s2_mask = _validated_phase_mask(phase_masks, "s2", length)
        candidate_values = np.asarray(murmur_candidate)
        original_values = np.asarray(original)
        timing_candidate = candidate_values[systole_mask]
        s1_leakage_ratio = energy(candidate_values[s1_mask]) / (
            energy(original_values[s1_mask]) + EPSILON
        )
        s2_leakage_ratio = energy(candidate_values[s2_mask]) / (
            energy(original_values[s2_mask]) + EPSILON
        )
        murmur_region_energy_retention = energy(candidate_values[systole_mask]) / (
            energy(original_values[systole_mask]) + EPSILON
        )
        outside_murmur_energy_ratio = energy(candidate_values[~systole_mask]) / (
            energy(candidate_values) + EPSILON
        )
    spectral = spectral_features(timing_candidate, sample_rate)
    timing = detect_activity_interval(
        timing_candidate,
        sample_rate,
        threshold_mad=threshold_mad,
        minimum_duration_ms=minimum_duration_ms,
        merge_gap_ms=merge_gap_ms,
    )
    return {
        "reconstruction_error": reconstruction_error,
        "normal_residual_correlation": safe_correlation(normal, murmur_candidate),
        "s1_leakage_ratio": s1_leakage_ratio,
        "s2_leakage_ratio": s2_leakage_ratio,
        "murmur_region_energy_retention": murmur_region_energy_retention,
        "outside_murmur_energy_ratio": outside_murmur_energy_ratio,
        "noise_energy_ratio": energy(noise_candidate) / (original_energy + EPSILON),
        "residual_spectral_centroid": spectral["spectral_centroid"],
        "residual_bandwidth": spectral["spectral_bandwidth"],
        "residual_spectral_entropy": spectral["spectral_entropy"],
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
