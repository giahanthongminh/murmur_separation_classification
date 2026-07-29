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
) -> dict[str, float | None]:
    """Detect candidate activity within one already-restricted cardiac phase."""

    values = np.asarray(signal, dtype=float)
    if values.size == 0:
        return _empty_activity()
    envelope = np.abs(hilbert(values))
    smoothing = max(1, int(round(sample_rate * 0.02)))
    kernel = np.ones(smoothing, dtype=float) / smoothing
    smooth = np.convolve(envelope, kernel, mode="same")
    baseline = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - baseline)))
    threshold = baseline + threshold_mad * 1.4826 * mad
    active = smooth > threshold

    merge_gap = max(0, int(round(merge_gap_ms * sample_rate / 1000.0)))
    if merge_gap and active.any():
        indexes = np.flatnonzero(active)
        for left, right in zip(indexes[:-1], indexes[1:]):
            if right - left - 1 <= merge_gap:
                active[left : right + 1] = True

    minimum = max(1, int(round(minimum_duration_ms * sample_rate / 1000.0)))
    valid_intervals: list[tuple[int, int]] = []
    changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for start, end in zip(starts, ends):
        if end - start >= minimum:
            valid_intervals.append((int(start), int(end)))
    if not valid_intervals:
        return _empty_activity()

    start = valid_intervals[0][0]
    end = valid_intervals[-1][1]
    phase_energy = values**2
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
    }


def _empty_activity() -> dict[str, None]:
    return {
        "onset_normalized": None,
        "offset_normalized": None,
        "duration_ratio": None,
        "temporal_energy_centroid": None,
        "peak_position": None,
    }


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
) -> dict[str, float | None]:
    original_energy = energy(original)
    reconstructed = normal + murmur_candidate + noise_candidate
    reconstruction_error = float(
        np.linalg.norm(np.asarray(original) - reconstructed)
        / (np.linalg.norm(original) + EPSILON)
    )
    spectral = spectral_features(murmur_candidate, sample_rate)
    timing = detect_activity_interval(
        murmur_candidate,
        sample_rate,
        threshold_mad=threshold_mad,
        minimum_duration_ms=minimum_duration_ms,
        merge_gap_ms=merge_gap_ms,
    )
    return {
        "reconstruction_error": reconstruction_error,
        "normal_residual_correlation": safe_correlation(normal, murmur_candidate),
        # A systolic-only segment contains neither full S1 nor full S2. These are
        # explicitly unavailable rather than silently reported as zero.
        "s1_leakage_ratio": None,
        "s2_leakage_ratio": None,
        "murmur_region_energy_retention": energy(murmur_candidate)
        / (original_energy + EPSILON),
        "outside_murmur_energy_ratio": None,
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
